use crate::autograd::{
    StoragePreference, Tensor, TensorData, TensorStorageOwned, TensorStorageView, is_no_grad,
};
use crate::device::{Device, assert_native_device_support, is_strict_device_execution};
use crate::ops::cuda;
use crate::precision::{DType, default_activation_dtype};
use half::{bf16, f16};
use ndarray::{Array, ArrayD, Ix2, Zip, s};
use std::cell::RefCell;
use std::rc::Rc;

pub struct RotaryEmbedding {
    dim: usize,
    max_seq_len: usize,
    // 缓存预计算的 cos/sin
    // Shape: [1, 1, Max_Seq, Dim]
    cos_cache: Tensor,
    sin_cache: Tensor,
}

fn cuda_rope_storage_buffer(tensor: &Tensor) -> Option<(DType, cuda::CudaBuffer, Option<f32>)> {
    tensor.cloned_cuda_native_lowp_buffer().or_else(|| {
        if tensor.dtype() == DType::F32 && tensor.is_cuda() {
            tensor
                .cloned_cuda_f32_buffer()
                .map(|buffer| (DType::F32, buffer, None))
        } else {
            None
        }
    })
}

impl RotaryEmbedding {
    pub fn new(dim: usize, max_seq_len: usize, theta: f32) -> Self {
        Self::new_with_dtype(dim, max_seq_len, theta, default_activation_dtype())
    }

    pub fn new_with_dtype(dim: usize, max_seq_len: usize, theta: f32, dtype: DType) -> Self {
        let cache_dtype = if dtype == DType::I8 {
            DType::F32
        } else {
            dtype
        };
        assert!(
            cache_dtype.is_float(),
            "RotaryEmbedding cache currently only supports floating runtime dtypes, got {:?}",
            dtype
        );
        let (cos, sin) = Self::precompute_freqs_cis(dim, max_seq_len, theta);
        let cos_cache = Tensor::from_array_no_grad(cos);
        let sin_cache = Tensor::from_array_no_grad(sin);
        cos_cache.cast_inplace(cache_dtype);
        sin_cache.cast_inplace(cache_dtype);

        Self {
            dim,
            max_seq_len,
            cos_cache,
            sin_cache,
        }
    }

    #[inline]
    pub fn cache_dtype(&self) -> DType {
        self.cos_cache.dtype()
    }

    fn precompute_freqs_cis(
        dim: usize,
        max_seq_len: usize,
        theta: f32,
    ) -> (ArrayD<f32>, ArrayD<f32>) {
        let half_d = dim / 2;
        let mut cos_arr = Array::zeros((1, 1, max_seq_len, dim));
        let mut sin_arr = Array::zeros((1, 1, max_seq_len, dim));

        for i in 0..max_seq_len {
            let pos = i as f32;
            for j in 0..half_d {
                let freq = 1.0 / theta.powf((j as f32 * 2.0) / dim as f32);
                let val = pos * freq;

                let c = val.cos();
                let s = val.sin();

                cos_arr[[0, 0, i, j]] = c;
                cos_arr[[0, 0, i, j + half_d]] = c;
                sin_arr[[0, 0, i, j]] = s;
                sin_arr[[0, 0, i, j + half_d]] = s;
            }
        }
        (cos_arr.into_dyn(), sin_arr.into_dyn())
    }

    pub fn to_device(&self, device: Device) {
        self.cos_cache.to_device_inplace(device);
        self.sin_cache.to_device_inplace(device);
    }

    pub fn to_cpu(&self) {
        self.to_device(Device::Cpu);
    }

    pub fn to_cuda(&self) {
        self.to_device(Device::Cuda);
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn decode_rope_q_append_kv_cuda(
        &self,
        q_src: &cuda::CudaBuffer,
        k_src: &cuda::CudaBuffer,
        v_src: &cuda::CudaBuffer,
        k_cache: &cuda::CudaBuffer,
        v_cache: &cuda::CudaBuffer,
        batch_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        dst_seq_len: usize,
        offset: usize,
    ) -> Result<cuda::CudaBuffer, String> {
        self.to_device(Device::Cuda);
        if offset >= self.max_seq_len {
            return Err(format!(
                "CUDA decode RoPE offset out of bounds: offset {} >= max_seq_len {}",
                offset, self.max_seq_len
            ));
        }
        self.cos_cache.with_cuda_f32_buffer(|cos_buf| {
            self.sin_cache.with_cuda_f32_buffer(|sin_buf| {
                cuda::decode_rope_q_append_kv_f32_buffer(
                    q_src,
                    k_src,
                    v_src,
                    cos_buf,
                    sin_buf,
                    k_cache,
                    v_cache,
                    batch_size,
                    num_heads,
                    num_kv_heads,
                    self.dim,
                    dst_seq_len,
                    offset,
                    self.max_seq_len,
                )
            })
        })
    }

    pub fn forward(&self, x: &Tensor, offset: usize) -> Tensor {
        let output_device = x.device();
        let build_graph = !is_no_grad() && x.requires_grad();
        let cuda_native_supported = output_device == Device::Cuda;
        assert_native_device_support(output_device, "rope", cuda_native_supported);
        self.to_device(output_device);

        if !build_graph {
            let shape = x.shape_vec();
            assert_eq!(shape.len(), 4, "RoPE expects input [B,H,S,D]");
            let (b, h, seq_len, d) = (shape[0], shape[1], shape[2], shape[3]);
            assert_eq!(d, self.dim, "RoPE dimension mismatch");
            let end = offset + seq_len;
            if end > self.max_seq_len {
                panic!(
                    "RoPE index out of range: offset {} + len {} > max {}",
                    offset, seq_len, self.max_seq_len
                );
            }

            if output_device == Device::Cuda && !x.is_empty() {
                if let Some((dtype, x_buf, scale)) = x.cloned_cuda_native_lowp_buffer()
                    && let (
                        Some((cos_dtype, cos_buf, cos_scale)),
                        Some((sin_dtype, sin_buf, sin_scale)),
                    ) = (
                        cuda_rope_storage_buffer(&self.cos_cache),
                        cuda_rope_storage_buffer(&self.sin_cache),
                    )
                    && cos_dtype == sin_dtype
                {
                    if x.dtype() == DType::I8 {
                        match cuda::rope_typed_i8_dynamic_buffer(
                            &x_buf,
                            dtype,
                            scale,
                            &cos_buf,
                            &sin_buf,
                            cos_dtype,
                            cos_scale,
                            sin_scale,
                            b,
                            h,
                            seq_len,
                            d,
                            offset,
                            self.max_seq_len,
                        ) {
                            Ok((i8_buffer, scale)) => {
                                return Tensor::from_cuda_native_lowp_buffer_no_host_with_dtype(
                                    &shape,
                                    i8_buffer,
                                    output_device,
                                    DType::I8,
                                    Some(scale),
                                );
                            }
                            Err(err) => {
                                assert!(
                                    !is_strict_device_execution(),
                                    "CUDA RoPE fused I8 output quantization failed while strict device execution is enabled: {err}"
                                );
                            }
                        }
                    }
                    let cuda_out = cuda::rope_typed_buffer(
                        &x_buf,
                        dtype,
                        scale,
                        &cos_buf,
                        &sin_buf,
                        cos_dtype,
                        cos_scale,
                        sin_scale,
                        b,
                        h,
                        seq_len,
                        d,
                        offset,
                        self.max_seq_len,
                    );
                    if let Ok(buffer) = cuda_out {
                        if matches!(x.dtype(), DType::F16 | DType::BF16) {
                            match cuda::f32_to_lowp_storage_no_host(&buffer, x.dtype()) {
                                Ok(lowp_buffer) => {
                                    return Tensor::from_cuda_native_lowp_buffer_no_host_with_dtype(
                                        &shape,
                                        lowp_buffer,
                                        output_device,
                                        x.dtype(),
                                        None,
                                    );
                                }
                                Err(err) => {
                                    assert!(
                                        !is_strict_device_execution(),
                                        "CUDA RoPE {:?} output conversion failed while strict device execution is enabled: {err}",
                                        x.dtype()
                                    );
                                }
                            }
                        }
                        return Tensor::from_cuda_f32_buffer_no_host_with_dtype(
                            &shape,
                            buffer,
                            output_device,
                            x.dtype(),
                        );
                    }
                }

                if x.dtype().is_float() {
                    let cuda_out = x.with_cuda_f32_buffer(|x_buf| {
                        self.cos_cache.with_cuda_f32_buffer(|cos_buf| {
                            self.sin_cache.with_cuda_f32_buffer(|sin_buf| {
                                cuda::rope_f32_buffer(
                                    x_buf,
                                    cos_buf,
                                    sin_buf,
                                    b,
                                    h,
                                    seq_len,
                                    d,
                                    offset,
                                    self.max_seq_len,
                                )
                            })
                        })
                    });
                    if let Ok(buffer) = cuda_out {
                        if matches!(x.dtype(), DType::F16 | DType::BF16) {
                            match cuda::f32_to_lowp_storage_no_host(&buffer, x.dtype()) {
                                Ok(lowp_buffer) => {
                                    return Tensor::from_cuda_native_lowp_buffer_no_host_with_dtype(
                                        &shape,
                                        lowp_buffer,
                                        output_device,
                                        x.dtype(),
                                        None,
                                    );
                                }
                                Err(err) => {
                                    assert!(
                                        !is_strict_device_execution(),
                                        "CUDA RoPE {:?} output conversion failed while strict device execution is enabled: {err}",
                                        x.dtype()
                                    );
                                }
                            }
                        }
                        return Tensor::from_cuda_f32_buffer_no_host_with_dtype(
                            &shape,
                            buffer,
                            output_device,
                            x.dtype(),
                        );
                    }
                } else {
                    let cuda_i8_out = if x.dtype() == DType::I8 {
                        x.with_cuda_f32_buffer(|x_buf| {
                            self.cos_cache.with_cuda_f32_buffer(|cos_buf| {
                                self.sin_cache.with_cuda_f32_buffer(|sin_buf| {
                                    cuda::rope_typed_i8_dynamic_buffer(
                                        x_buf,
                                        DType::F32,
                                        None,
                                        cos_buf,
                                        sin_buf,
                                        DType::F32,
                                        None,
                                        None,
                                        b,
                                        h,
                                        seq_len,
                                        d,
                                        offset,
                                        self.max_seq_len,
                                    )
                                })
                            })
                        })
                    } else {
                        Err("CUDA RoPE fused I8 fallback is only valid for I8 output".to_string())
                    };
                    if let Ok((i8_buffer, scale)) = cuda_i8_out {
                        return Tensor::from_cuda_native_lowp_buffer_no_host_with_dtype(
                            &shape,
                            i8_buffer,
                            output_device,
                            DType::I8,
                            Some(scale),
                        );
                    }

                    let cuda_out = x.with_cuda_f32_buffer(|x_buf| {
                        self.cos_cache.with_cuda_f32_buffer(|cos_buf| {
                            self.sin_cache.with_cuda_f32_buffer(|sin_buf| {
                                cuda::rope_f32_buffer(
                                    x_buf,
                                    cos_buf,
                                    sin_buf,
                                    b,
                                    h,
                                    seq_len,
                                    d,
                                    offset,
                                    self.max_seq_len,
                                )
                            })
                        })
                    });
                    if let Ok(buffer) = cuda_out {
                        if x.dtype() == DType::I8 {
                            match cuda::quantize_f32_to_i8_dynamic_no_host(&buffer) {
                                Ok((i8_buffer, scale)) => {
                                    return Tensor::from_cuda_native_lowp_buffer_no_host_with_dtype(
                                        &shape,
                                        i8_buffer,
                                        output_device,
                                        DType::I8,
                                        Some(scale),
                                    );
                                }
                                Err(err) => {
                                    assert!(
                                        !is_strict_device_execution(),
                                        "CUDA RoPE I8 output quantization failed while strict device execution is enabled: {err}"
                                    );
                                }
                            }
                        }
                        if matches!(x.dtype(), DType::F16 | DType::BF16) {
                            match cuda::f32_to_lowp_storage_no_host(&buffer, x.dtype()) {
                                Ok(lowp_buffer) => {
                                    return Tensor::from_cuda_native_lowp_buffer_no_host_with_dtype(
                                        &shape,
                                        lowp_buffer,
                                        output_device,
                                        x.dtype(),
                                        None,
                                    );
                                }
                                Err(err) => {
                                    assert!(
                                        !is_strict_device_execution(),
                                        "CUDA RoPE {:?} output conversion failed while strict device execution is enabled: {err}",
                                        x.dtype()
                                    );
                                }
                            }
                        }
                        return Tensor::from_cuda_f32_buffer_no_host_with_dtype(
                            &shape,
                            buffer,
                            output_device,
                            x.dtype(),
                        );
                    }
                }
            }

            if x.dtype() == DType::I8 {
                return match x.native_storage_owned() {
                    TensorStorageOwned::I8(x_data, x_scale) => self
                        .cos_cache
                        .with_storage_view_preferring(StoragePreference::F32Compute, |cos_view| {
                            self.sin_cache.with_storage_view_preferring(
                                StoragePreference::F32Compute,
                                |sin_view| {
                                    let cos_4d = match cos_view {
                                        TensorStorageView::F32(view) => view
                                            .into_dimensionality::<ndarray::Ix4>()
                                            .expect("RoPE Cache dimensionality mismatch"),
                                        TensorStorageView::F16(_) => unreachable!(
                                            "f32 compute preference should expose f32 view"
                                        ),
                                        TensorStorageView::BF16(_) => unreachable!(
                                            "f32 compute preference should expose f32 view"
                                        ),
                                    };
                                    let sin_4d = match sin_view {
                                        TensorStorageView::F32(view) => view
                                            .into_dimensionality::<ndarray::Ix4>()
                                            .expect("RoPE Cache dimensionality mismatch"),
                                        TensorStorageView::F16(_) => unreachable!(
                                            "f32 compute preference should expose f32 view"
                                        ),
                                        TensorStorageView::BF16(_) => unreachable!(
                                            "f32 compute preference should expose f32 view"
                                        ),
                                    };
                                    let cos_slice_2d = cos_4d
                                        .slice(s![0, 0, offset..end, ..])
                                        .into_dimensionality::<Ix2>()
                                        .expect("RoPE Cache dimensionality mismatch");
                                    let sin_slice_2d = sin_4d
                                        .slice(s![0, 0, offset..end, ..])
                                        .into_dimensionality::<Ix2>()
                                        .expect("RoPE Cache dimensionality mismatch");

                                    let x_view = x_data
                                        .view()
                                        .into_dimensionality::<ndarray::Ix4>()
                                        .unwrap();
                                    let mut out = Array::zeros(x_data.raw_dim());
                                    let mut out_view = out
                                        .view_mut()
                                        .into_dimensionality::<ndarray::Ix4>()
                                        .unwrap();

                                    Zip::from(out_view.outer_iter_mut())
                                        .and(x_view.outer_iter())
                                        .par_for_each(|mut out_b, x_b| {
                                            Zip::from(out_b.outer_iter_mut())
                                                .and(x_b.outer_iter())
                                                .for_each(|mut out_h, x_h| {
                                                    let half = d / 2;
                                                    for ss in 0..seq_len {
                                                        for j in 0..half {
                                                            let x1 = x_h[[ss, j]] as f32 * x_scale;
                                                            let x2 = x_h[[ss, j + half]] as f32
                                                                * x_scale;
                                                            let c = cos_slice_2d[[ss, j]];
                                                            let s_val = sin_slice_2d[[ss, j]];
                                                            out_h[[ss, j]] = x1 * c - x2 * s_val;
                                                            out_h[[ss, j + half]] =
                                                                x2 * c + x1 * s_val;
                                                        }
                                                    }
                                                });
                                        });

                                    let out = Tensor::from_f32_data_no_grad_with_device_dtype(
                                        out.into_dyn(),
                                        DType::F32,
                                        output_device,
                                    );
                                    out.cast_inplace(DType::I8);
                                    out
                                },
                            )
                        }),
                    TensorStorageOwned::F32(_)
                    | TensorStorageOwned::F16(_)
                    | TensorStorageOwned::BF16(_) => unreachable!("checked i8 input above"),
                };
            }

            return x.with_storage_view_preferring(StoragePreference::Native, |x_view| {
                let shape = match &x_view {
                    TensorStorageView::F32(x_view) => x_view.shape().to_vec(),
                    TensorStorageView::F16(x_view) => x_view.shape().to_vec(),
                    TensorStorageView::BF16(x_view) => x_view.shape().to_vec(),
                };
                self.cos_cache.with_storage_view_preferring(
                    StoragePreference::F32Compute,
                    |cos_view| {
                        self.sin_cache.with_storage_view_preferring(
                            StoragePreference::F32Compute,
                            |sin_view| {
                                let cos_4d = match cos_view {
                                    TensorStorageView::F32(view) => view
                                        .into_dimensionality::<ndarray::Ix4>()
                                        .expect("RoPE Cache dimensionality mismatch"),
                                    TensorStorageView::F16(_) => {
                                        unreachable!(
                                            "f32 compute preference should expose f32 view"
                                        )
                                    }
                                    TensorStorageView::BF16(_) => {
                                        unreachable!(
                                            "f32 compute preference should expose f32 view"
                                        )
                                    }
                                };
                                let sin_4d = match sin_view {
                                    TensorStorageView::F32(view) => view
                                        .into_dimensionality::<ndarray::Ix4>()
                                        .expect("RoPE Cache dimensionality mismatch"),
                                    TensorStorageView::F16(_) => {
                                        unreachable!(
                                            "f32 compute preference should expose f32 view"
                                        )
                                    }
                                    TensorStorageView::BF16(_) => {
                                        unreachable!(
                                            "f32 compute preference should expose f32 view"
                                        )
                                    }
                                };
                                let cos_slice_2d = cos_4d
                                    .slice(s![0, 0, offset..end, ..])
                                    .into_dimensionality::<Ix2>()
                                    .expect("RoPE Cache dimensionality mismatch");
                                let sin_slice_2d = sin_4d
                                    .slice(s![0, 0, offset..end, ..])
                                    .into_dimensionality::<Ix2>()
                                    .expect("RoPE Cache dimensionality mismatch");

                                match x_view {
                                    TensorStorageView::F32(x_view) => {
                                        let mut out = Array::zeros(x_view.raw_dim());
                                        let x_view =
                                            x_view.into_dimensionality::<ndarray::Ix4>().unwrap();
                                        let mut out_view = out
                                            .view_mut()
                                            .into_dimensionality::<ndarray::Ix4>()
                                            .unwrap();

                                        Zip::from(out_view.outer_iter_mut())
                                            .and(x_view.outer_iter())
                                            .par_for_each(|mut out_b, x_b| {
                                                Zip::from(out_b.outer_iter_mut())
                                                    .and(x_b.outer_iter())
                                                    .for_each(|mut out_h, x_h| {
                                                        let half = d / 2;
                                                        for ss in 0..seq_len {
                                                            for j in 0..half {
                                                                let x1 = x_h[[ss, j]];
                                                                let x2 = x_h[[ss, j + half]];
                                                                let c = cos_slice_2d[[ss, j]];
                                                                let s_val = sin_slice_2d[[ss, j]];
                                                                out_h[[ss, j]] =
                                                                    x1 * c - x2 * s_val;
                                                                out_h[[ss, j + half]] =
                                                                    x2 * c + x1 * s_val;
                                                            }
                                                        }
                                                    });
                                            });

                                        Tensor::from_f32_data_no_grad_with_device_dtype(
                                            out.into_dyn(),
                                            DType::F32,
                                            output_device,
                                        )
                                    }
                                    TensorStorageView::F16(x_view) => {
                                        let mut out = ndarray::ArrayD::<f16>::from_elem(
                                            ndarray::IxDyn(&shape),
                                            f16::from_bits(0),
                                        );
                                        let x_view =
                                            x_view.into_dimensionality::<ndarray::Ix4>().unwrap();
                                        let mut out_view = out
                                            .view_mut()
                                            .into_dimensionality::<ndarray::Ix4>()
                                            .unwrap();

                                        Zip::from(out_view.outer_iter_mut())
                                            .and(x_view.outer_iter())
                                            .par_for_each(|mut out_b, x_b| {
                                                Zip::from(out_b.outer_iter_mut())
                                                    .and(x_b.outer_iter())
                                                    .for_each(|mut out_h, x_h| {
                                                        let half = d / 2;
                                                        for ss in 0..seq_len {
                                                            for j in 0..half {
                                                                let x1 = x_h[[ss, j]].to_f32();
                                                                let x2 =
                                                                    x_h[[ss, j + half]].to_f32();
                                                                let c = cos_slice_2d[[ss, j]];
                                                                let s_val = sin_slice_2d[[ss, j]];
                                                                out_h[[ss, j]] = f16::from_f32(
                                                                    x1 * c - x2 * s_val,
                                                                );
                                                                out_h[[ss, j + half]] =
                                                                    f16::from_f32(
                                                                        x2 * c + x1 * s_val,
                                                                    );
                                                            }
                                                        }
                                                    });
                                            });

                                        Tensor::from_shared_f16_no_grad_with_device(
                                            out.into_shared(),
                                            output_device,
                                        )
                                    }
                                    TensorStorageView::BF16(x_view) => {
                                        let mut out = ndarray::ArrayD::<bf16>::from_elem(
                                            ndarray::IxDyn(&shape),
                                            bf16::from_bits(0),
                                        );
                                        let x_view =
                                            x_view.into_dimensionality::<ndarray::Ix4>().unwrap();
                                        let mut out_view = out
                                            .view_mut()
                                            .into_dimensionality::<ndarray::Ix4>()
                                            .unwrap();

                                        Zip::from(out_view.outer_iter_mut())
                                            .and(x_view.outer_iter())
                                            .par_for_each(|mut out_b, x_b| {
                                                Zip::from(out_b.outer_iter_mut())
                                                    .and(x_b.outer_iter())
                                                    .for_each(|mut out_h, x_h| {
                                                        let half = d / 2;
                                                        for ss in 0..seq_len {
                                                            for j in 0..half {
                                                                let x1 = x_h[[ss, j]].to_f32();
                                                                let x2 =
                                                                    x_h[[ss, j + half]].to_f32();
                                                                let c = cos_slice_2d[[ss, j]];
                                                                let s_val = sin_slice_2d[[ss, j]];
                                                                out_h[[ss, j]] = bf16::from_f32(
                                                                    x1 * c - x2 * s_val,
                                                                );
                                                                out_h[[ss, j + half]] =
                                                                    bf16::from_f32(
                                                                        x2 * c + x1 * s_val,
                                                                    );
                                                            }
                                                        }
                                                    });
                                            });

                                        Tensor::from_shared_bf16_no_grad_with_device(
                                            out.into_shared(),
                                            output_device,
                                        )
                                    }
                                }
                            },
                        )
                    },
                )
            });
        }

        if output_device == Device::Cuda && !x.is_empty() {
            let shape = x.shape_vec();
            assert_eq!(shape.len(), 4, "RoPE expects input [B,H,S,D]");
            let (b, h, seq_len, d) = (shape[0], shape[1], shape[2], shape[3]);
            assert_eq!(d, self.dim, "RoPE dimension mismatch");
            let end = offset + seq_len;
            if end > self.max_seq_len {
                panic!(
                    "RoPE index out of range: offset {} + len {} > max {}",
                    offset, seq_len, self.max_seq_len
                );
            }

            let cuda_out = if let (
                Some((x_dtype, x_buf, x_scale)),
                Some((cos_dtype, cos_buf, cos_scale)),
                Some((sin_dtype, sin_buf, sin_scale)),
            ) = (
                x.cloned_cuda_native_lowp_buffer(),
                cuda_rope_storage_buffer(&self.cos_cache),
                cuda_rope_storage_buffer(&self.sin_cache),
            ) {
                if cos_dtype == sin_dtype {
                    cuda::rope_typed_buffer(
                        &x_buf,
                        x_dtype,
                        x_scale,
                        &cos_buf,
                        &sin_buf,
                        cos_dtype,
                        cos_scale,
                        sin_scale,
                        b,
                        h,
                        seq_len,
                        d,
                        offset,
                        self.max_seq_len,
                    )
                } else {
                    x.with_cuda_f32_buffer(|x_buf| {
                        self.cos_cache.with_cuda_f32_buffer(|cos_buf| {
                            self.sin_cache.with_cuda_f32_buffer(|sin_buf| {
                                cuda::rope_f32_buffer(
                                    x_buf,
                                    cos_buf,
                                    sin_buf,
                                    b,
                                    h,
                                    seq_len,
                                    d,
                                    offset,
                                    self.max_seq_len,
                                )
                            })
                        })
                    })
                }
            } else {
                x.with_cuda_f32_buffer(|x_buf| {
                    self.cos_cache.with_cuda_f32_buffer(|cos_buf| {
                        self.sin_cache.with_cuda_f32_buffer(|sin_buf| {
                            cuda::rope_f32_buffer(
                                x_buf,
                                cos_buf,
                                sin_buf,
                                b,
                                h,
                                seq_len,
                                d,
                                offset,
                                self.max_seq_len,
                            )
                        })
                    })
                })
            };
            if let Ok(buffer) = cuda_out {
                let x_clone = x.clone();
                let cos_cache = self.cos_cache.clone();
                let sin_cache = self.sin_cache.clone();
                let max_seq_len = self.max_seq_len;
                let output_self =
                    Rc::new(RefCell::new(std::rc::Weak::<RefCell<TensorData>>::new()));
                let output_self_for_backward = output_self.clone();

                let tensor = Tensor(Rc::new(RefCell::new(TensorData {
                    data: Array::zeros(ndarray::IxDyn(&shape)).into_shared(),
                    f16_data: None,
                    bf16_data: None,
                    i8_data: None,
                    cuda_f32_data: Some(buffer),
                    cuda_f16_data: None,
                    cuda_bf16_data: None,
                    cuda_i8_data: None,
                    i8_scale: None,
                    has_f32_data: false,
                    storage_dtype: crate::precision::DType::F32,
                    cache_dirty: false,
                    is_parameter: false,
                    grad: None,
                    cuda_f32_grad: None,
                    parents: vec![x.clone()],
                    backward_op: Some(std::rc::Rc::new(move |grad| {
                        let cuda_grad = if let Some(grad_buf) = output_self_for_backward
                            .borrow()
                            .upgrade()
                            .map(Tensor)
                            .and_then(|output| output.cloned_cuda_f32_grad())
                            .filter(|grad_buf| grad_buf.len() == grad.len())
                        {
                            if is_strict_device_execution() {
                                match cos_cache.with_cuda_f32_buffer(|cos_buf| {
                                    sin_cache.with_cuda_f32_buffer(|sin_buf| {
                                        cuda::rope_backward_f32_buffer(
                                            &grad_buf,
                                            cos_buf,
                                            sin_buf,
                                            b,
                                            h,
                                            seq_len,
                                            d,
                                            offset,
                                            max_seq_len,
                                        )
                                    })
                                }) {
                                    Ok(grad_buffer) => {
                                        x_clone.add_backward_cuda_grad_buffer_only(grad_buffer);
                                        return;
                                    }
                                    Err(err) => {
                                        panic!("CUDA RoPE backward failed: {err}");
                                    }
                                }
                            }
                            cos_cache.with_cuda_f32_buffer(|cos_buf| {
                                sin_cache.with_cuda_f32_buffer(|sin_buf| {
                                    cuda::rope_backward_f32(
                                        &grad_buf,
                                        cos_buf,
                                        sin_buf,
                                        b,
                                        h,
                                        seq_len,
                                        d,
                                        offset,
                                        max_seq_len,
                                    )
                                })
                            })
                        } else {
                            let grad_host = grad.iter().copied().collect::<Vec<_>>();
                            if is_strict_device_execution() {
                                match cuda::upload_f32(&grad_host).and_then(|grad_buf| {
                                    cos_cache.with_cuda_f32_buffer(|cos_buf| {
                                        sin_cache.with_cuda_f32_buffer(|sin_buf| {
                                            cuda::rope_backward_f32_buffer(
                                                &grad_buf,
                                                cos_buf,
                                                sin_buf,
                                                b,
                                                h,
                                                seq_len,
                                                d,
                                                offset,
                                                max_seq_len,
                                            )
                                        })
                                    })
                                }) {
                                    Ok(grad_buffer) => {
                                        x_clone.add_backward_cuda_grad_buffer_only(grad_buffer);
                                        return;
                                    }
                                    Err(err) => {
                                        panic!("CUDA RoPE backward failed: {err}");
                                    }
                                }
                            }
                            cuda::upload_f32(&grad_host).and_then(|grad_buf| {
                                cos_cache.with_cuda_f32_buffer(|cos_buf| {
                                    sin_cache.with_cuda_f32_buffer(|sin_buf| {
                                        cuda::rope_backward_f32(
                                            &grad_buf,
                                            cos_buf,
                                            sin_buf,
                                            b,
                                            h,
                                            seq_len,
                                            d,
                                            offset,
                                            max_seq_len,
                                        )
                                    })
                                })
                            })
                        };
                        match cuda_grad {
                            Ok((grad_buffer, grad_host)) => {
                                let d_x = Array::from_shape_vec(ndarray::IxDyn(&shape), grad_host)
                                    .expect("CUDA RoPE grad shape build failed")
                                    .into_dyn();
                                x_clone.add_backward_grad_with_cuda_buffer(d_x, Some(grad_buffer));
                            }
                            Err(err) => {
                                panic!("CUDA RoPE backward failed: {err}");
                            }
                        }
                    })),
                    requires_grad: true,
                    device: output_device,
                })));
                *output_self.borrow_mut() = Rc::downgrade(&tensor.0);
                return tensor;
            }
        }

        let x_data = x.data_ref();
        let shape = x_data.shape();
        assert_eq!(shape.len(), 4, "RoPE expects input [B,H,S,D]");
        let (b, h, seq_len, d) = (shape[0], shape[1], shape[2], shape[3]);
        assert_eq!(d, self.dim, "RoPE dimension mismatch");

        let end = offset + seq_len;
        if end > self.max_seq_len {
            panic!(
                "RoPE index out of range: offset {} + len {} > max {}",
                offset, seq_len, self.max_seq_len
            );
        }

        self.cos_cache
            .with_storage_view_preferring(StoragePreference::F32Compute, |cos_view| {
                self.sin_cache.with_storage_view_preferring(
                    StoragePreference::F32Compute,
                    |sin_view| {
                        let cos_4d = match cos_view {
                            TensorStorageView::F32(view) => view
                                .into_dimensionality::<ndarray::Ix4>()
                                .expect("K cache update expects [B,H,S,D]"),
                            TensorStorageView::F16(_) => {
                                unreachable!("f32 compute preference should expose f32 view")
                            }
                            TensorStorageView::BF16(_) => {
                                unreachable!("f32 compute preference should expose f32 view")
                            }
                        };

                        let sin_4d = match sin_view {
                            TensorStorageView::F32(view) => view
                                .into_dimensionality::<ndarray::Ix4>()
                                .expect("K cache update expects [B,H,S,D]"),
                            TensorStorageView::F16(_) => {
                                unreachable!("f32 compute preference should expose f32 view")
                            }
                            TensorStorageView::BF16(_) => {
                                unreachable!("f32 compute preference should expose f32 view")
                            }
                        };
                        let cos_slice_2d = cos_4d
                            .slice(s![0, 0, offset..end, ..])
                            .into_dimensionality::<Ix2>()
                            .expect("RoPE Cache dimensionality mismatch");
                        let sin_slice_2d = sin_4d
                            .slice(s![0, 0, offset..end, ..])
                            .into_dimensionality::<Ix2>()
                            .expect("RoPE Cache dimensionality mismatch");

                        let mut out = Array::zeros(x_data.dim());

                        let x_view = x_data.view().into_dimensionality::<ndarray::Ix4>().unwrap();
                        let mut out_view = out
                            .view_mut()
                            .into_dimensionality::<ndarray::Ix4>()
                            .unwrap();

                        Zip::from(out_view.outer_iter_mut())
                            .and(x_view.outer_iter())
                            .par_for_each(|mut out_b, x_b| {
                                Zip::from(out_b.outer_iter_mut())
                                    .and(x_b.outer_iter())
                                    .for_each(|mut out_h, x_h| {
                                        let half = d / 2;
                                        for ss in 0..seq_len {
                                            for j in 0..half {
                                                let x1 = x_h[[ss, j]];
                                                let x2 = x_h[[ss, j + half]];
                                                let c = cos_slice_2d[[ss, j]];
                                                let s_val = sin_slice_2d[[ss, j]];
                                                out_h[[ss, j]] = x1 * c - x2 * s_val;
                                                out_h[[ss, j + half]] = x2 * c + x1 * s_val;
                                            }
                                        }
                                    });
                            });

                        let x_clone = x.clone();
                        let cos_backward = cos_slice_2d.to_owned();
                        let sin_backward = sin_slice_2d.to_owned();

                        Tensor(Rc::new(RefCell::new(TensorData {
                            data: out.into_dyn().into_shared(),
                            f16_data: None,
                            bf16_data: None,
                            i8_data: None,
                            cuda_f32_data: None,
                            cuda_f16_data: None,
                            cuda_bf16_data: None,
                            cuda_i8_data: None,
                            i8_scale: None,
                            has_f32_data: true,
                            storage_dtype: crate::precision::DType::F32,
                            cache_dirty: false,
                            is_parameter: false,
                            grad: None,
                            cuda_f32_grad: None,
                            parents: vec![x.clone()],
                            backward_op: Some(std::rc::Rc::new(move |grad| {
                                let grad_view =
                                    grad.view().into_dimensionality::<ndarray::Ix4>().unwrap();
                                let mut d_x = Array::zeros((b, h, seq_len, d));

                                Zip::from(d_x.outer_iter_mut())
                                    .and(grad_view.outer_iter())
                                    .par_for_each(|mut dx_b, g_b| {
                                        Zip::from(dx_b.outer_iter_mut())
                                            .and(g_b.outer_iter())
                                            .for_each(|mut dx_h, g_h| {
                                                let half = d / 2;
                                                for ss in 0..seq_len {
                                                    for j in 0..half {
                                                        let g1 = g_h[[ss, j]];
                                                        let g2 = g_h[[ss, j + half]];

                                                        let c = cos_backward[[ss, j]];
                                                        let s_val = sin_backward[[ss, j]];

                                                        dx_h[[ss, j]] = g1 * c + g2 * s_val;
                                                        dx_h[[ss, j + half]] = g2 * c - g1 * s_val;
                                                    }
                                                }
                                            });
                                    });

                                x_clone.add_backward_grad(d_x.into_dyn());
                            })),
                            requires_grad: true,
                            device: output_device,
                        })))
                    },
                )
            })
    }

    // Apply RoPE for a single token at absolute position `pos`.
    //
    // Decode (S=1) hot-path helper to avoid allocating intermediate q_rot/k_rot tensors.
    // `src` and `dst` must both have length == `self.dim`.
    #[inline]
    pub fn rope_1token_copy(&self, src: &[f32], dst: &mut [f32], pos: usize) {
        assert_eq!(src.len(), self.dim, "RoPE src len mismatch");
        assert_eq!(dst.len(), self.dim, "RoPE dst len mismatch");
        if pos >= self.max_seq_len {
            panic!(
                "RoPE index out of range: pos {} >= max {}",
                pos, self.max_seq_len
            );
        }

        self.cos_cache
            .with_storage_view_preferring(StoragePreference::F32Compute, |cos_view| {
                self.sin_cache.with_storage_view_preferring(
                    StoragePreference::F32Compute,
                    |sin_view| {
                        let cos_row = match cos_view {
                            TensorStorageView::F32(view) => {
                                let cache4 = view
                                    .into_dimensionality::<ndarray::Ix4>()
                                    .expect("RoPE Cache dimensionality mismatch");
                                cache4.slice(s![0, 0, pos, ..]).to_owned().into_raw_vec()
                            }
                            TensorStorageView::F16(_) => {
                                unreachable!("f32 compute preference should expose f32 view")
                            }
                            TensorStorageView::BF16(_) => {
                                unreachable!("f32 compute preference should expose f32 view")
                            }
                        };
                        let sin_row = match sin_view {
                            TensorStorageView::F32(view) => {
                                let cache4 = view
                                    .into_dimensionality::<ndarray::Ix4>()
                                    .expect("RoPE Cache dimensionality mismatch");
                                cache4.slice(s![0, 0, pos, ..]).to_owned().into_raw_vec()
                            }
                            TensorStorageView::F16(_) => {
                                unreachable!("f32 compute preference should expose f32 view")
                            }
                            TensorStorageView::BF16(_) => {
                                unreachable!("f32 compute preference should expose f32 view")
                            }
                        };

                        let half = self.dim / 2;
                        for j in 0..half {
                            let x1 = src[j];
                            let x2 = src[j + half];
                            let c = cos_row[j];
                            let s_val = sin_row[j];
                            dst[j] = x1 * c - x2 * s_val;
                            dst[j + half] = x2 * c + x1 * s_val;
                        }
                    },
                )
            });
    }

    // Get (cos, sin) row at position `pos` as owned Vecs.
    // This is useful to pass into rayon-parallel decode kernels without capturing Tensor/Rc.
    pub fn cos_sin_row_vec(&self, pos: usize) -> (Vec<f32>, Vec<f32>) {
        if pos >= self.max_seq_len {
            panic!(
                "RoPE index out of range: pos {} >= max {}",
                pos, self.max_seq_len
            );
        }
        self.cos_cache
            .with_storage_view_preferring(StoragePreference::F32Compute, |cos_view| {
                self.sin_cache.with_storage_view_preferring(
                    StoragePreference::F32Compute,
                    |sin_view| {
                        let cos_row = match cos_view {
                            TensorStorageView::F32(view) => {
                                let cache4 = view
                                    .into_dimensionality::<ndarray::Ix4>()
                                    .expect("RoPE Cache dimensionality mismatch");
                                cache4.slice(s![0, 0, pos, ..]).to_owned().into_raw_vec()
                            }
                            TensorStorageView::F16(_) => {
                                unreachable!("f32 compute preference should expose f32 view")
                            }
                            TensorStorageView::BF16(_) => {
                                unreachable!("f32 compute preference should expose f32 view")
                            }
                        };
                        let sin_row = match sin_view {
                            TensorStorageView::F32(view) => {
                                let cache4 = view
                                    .into_dimensionality::<ndarray::Ix4>()
                                    .expect("RoPE Cache dimensionality mismatch");
                                cache4.slice(s![0, 0, pos, ..]).to_owned().into_raw_vec()
                            }
                            TensorStorageView::F16(_) => {
                                unreachable!("f32 compute preference should expose f32 view")
                            }
                            TensorStorageView::BF16(_) => {
                                unreachable!("f32 compute preference should expose f32 view")
                            }
                        };

                        (cos_row, sin_row)
                    },
                )
            })
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/layers/attention/encoding/tests.rs"]
mod tests;
