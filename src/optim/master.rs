//! Explicit training storage: keep sub-quantum updates outside the forward tensor.
use crate::autograd::{StoragePreference, Tensor, TensorStorageView};
use crate::device::Device;
use crate::ops::cuda;
use crate::precision::{DType, ParameterQuantization};

#[derive(Default)]
pub(super) struct MasterWeights {
    entries: Option<Vec<Option<MasterWeight>>>,
    pub(super) dynamic_i8: bool,
}

struct MasterWeight {
    tensor: Tensor,
    dtype: DType,
    scale: Option<f32>,
}

fn copy_cuda_f32(source: &cuda::CudaBuffer) -> cuda::CudaBuffer {
    let copy = cuda::alloc_f32(source.len()).expect("allocate optimizer parameter copy");
    cuda::copy_f32_offset(&copy, 0, source, 0, source.len())
        .expect("copy optimizer parameter on CUDA");
    copy
}

impl MasterWeights {
    pub(super) fn enable(&mut self, len: usize) {
        assert!(self.entries.is_none(), "F32 master weights already enabled");
        self.entries = Some((0..len).map(|_| None).collect());
    }

    pub(super) fn prepare(&mut self, params: &[Tensor]) -> Option<Vec<Tensor>> {
        let entries = self.entries.as_mut()?;
        Some(
            params
                .iter()
                .zip(entries.iter_mut())
                .map(|(param, entry)| {
                    if let Some(master) = entry {
                        assert_eq!(
                            param.dtype(),
                            master.dtype,
                            "master parameter dtype changed"
                        );
                        assert_eq!(
                            param.device(),
                            master.tensor.device(),
                            "master parameter device changed"
                        );
                        assert_eq!(
                            param.shape_vec(),
                            master.tensor.shape_vec(),
                            "master parameter shape changed"
                        );
                        assert_eq!(
                            param.quantization_scale(),
                            master.scale,
                            "master parameter scale changed"
                        );
                    }
                    if param.dtype() == DType::F32 || !param.has_pending_grad() {
                        return param.clone();
                    }
                    let master = entry.get_or_insert_with(|| {
                        let tensor = if param.is_cuda() {
                            let source = param.cloned_cuda_f32_buffer().expect("CUDA parameter");
                            Tensor::from_cuda_f32_buffer_no_host(
                                &param.shape_vec(),
                                copy_cuda_f32(&source),
                                Device::Cuda,
                            )
                        } else {
                            let data = param.with_storage_view_preferring(
                                StoragePreference::Native,
                                |view| match view {
                                    TensorStorageView::F32(v) => v.to_owned(),
                                    TensorStorageView::F16(v) => v.mapv(|x| x.to_f32()),
                                    TensorStorageView::BF16(v) => v.mapv(|x| x.to_f32()),
                                },
                            );
                            Tensor::from_array_no_grad(data)
                        };
                        MasterWeight {
                            tensor,
                            dtype: param.dtype(),
                            scale: param.quantization_scale(),
                        }
                    });
                    // Share gradients without downloading CUDA buffers or touching the graph.
                    let source = param.0.borrow();
                    let mut target = master.tensor.0.borrow_mut();
                    target.grad = source.grad.clone();
                    target.cuda_f32_grad = source.cuda_f32_grad.clone();
                    drop(target);
                    master.tensor.clone()
                })
                .collect(),
        )
    }

    pub(super) fn publish(&mut self, params: &[Tensor]) {
        let Some(entries) = &mut self.entries else {
            return;
        };
        for (param, entry) in params.iter().zip(entries) {
            let Some(master) = entry else { continue };
            if !param.has_pending_grad() {
                continue;
            }
            if param.is_cuda() {
                let source = master.tensor.cloned_cuda_f32_buffer().expect("CUDA master");
                if self.dynamic_i8 && master.dtype == DType::I8 {
                    let (native, scale) = cuda::quantize_f32_to_i8_dynamic_no_host(&source)
                        .expect("calibrate I8 master scale on CUDA");
                    let rounded = copy_cuda_f32(&source);
                    cuda::quantize_f32_storage_no_host(&rounded, DType::I8, Some(scale))
                        .expect("round CUDA I8 forward compute copy");
                    param.replace_cuda_i8_and_f32_buffers_no_host_sync(native, rounded, scale);
                } else {
                    // Quantize a distinct buffer: rounding the master would lose accumulated updates.
                    let published = copy_cuda_f32(&source);
                    cuda::quantize_f32_storage_no_host(&published, master.dtype, master.scale)
                        .expect("quantize master for CUDA forward storage");
                    param.replace_cuda_f32_buffer_no_host_sync(published);
                }
            } else {
                let grad = param.0.borrow().grad.clone();
                let values = master.tensor.data();
                if let Some(scale) = master.scale {
                    let quantization = if self.dynamic_i8 {
                        ParameterQuantization::Int8
                    } else {
                        ParameterQuantization::Int8.with_scale(scale)
                    };
                    param.set_array_f32_with_quantization(values, quantization);
                } else {
                    param.set_array_f32_with_dtype(values, master.dtype);
                }
                // CPU optimizer steps retain gradients until zero_grad(), as on the native path.
                param.0.borrow_mut().grad = grad;
            }
            master.scale = param.quantization_scale();
            master.tensor.zero_grad();
        }
    }
}

#[cfg(test)]
#[path = "../../tests/unit/optim/master/tests.rs"]
mod tests;
