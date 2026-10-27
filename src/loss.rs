use crate::autograd::{
    StoragePreference, Tensor, TensorData, TensorStorageView, assert_native_device_support,
    is_no_grad, is_strict_device_execution,
};
use crate::ops::cuda;
use crate::precision::DType;
use ndarray::{Array2, Zip, arr0};
use rayon::prelude::*;
use std::cell::RefCell;
use std::rc::Rc; // 引入并行迭代

fn cuda_loss_storage(tensor: &Tensor) -> Option<(cuda::CudaBuffer, DType, Option<f32>)> {
    tensor
        .cloned_cuda_native_lowp_buffer()
        .map(|(dtype, buffer, scale)| (buffer, dtype, scale))
        .or_else(|| {
            if tensor.dtype() == DType::F32 && tensor.is_cuda() {
                tensor
                    .cloned_cuda_f32_buffer()
                    .map(|buffer| (buffer, DType::F32, None))
            } else {
                None
            }
        })
}

// --- MSE Loss ---
pub struct MSELoss;
impl MSELoss {
    pub fn apply(output: &Tensor, target: &Tensor) -> Tensor {
        let output_device = crate::autograd::assert_same_device(output, target, "mse_loss");
        assert_eq!(
            output.shape_vec(),
            target.shape_vec(),
            "mse_loss expects output and target to have the same shape"
        );
        assert!(!output.is_empty(), "mse_loss expects at least one element");
        let build_graph = !is_no_grad() && (output.requires_grad() || target.requires_grad());
        let cuda_native_supported = output_device == crate::autograd::Device::Cuda;
        assert_native_device_support(output_device, "mse_loss", cuda_native_supported);

        if output_device == crate::autograd::Device::Cuda {
            let cuda_forward = (|| {
                let (out_buf, out_dtype, out_scale) = cuda_loss_storage(output)
                    .ok_or_else(|| "CUDA mse_loss output missing resident buffer".to_string())?;
                let (tar_buf, tar_dtype, tar_scale) = cuda_loss_storage(target)
                    .ok_or_else(|| "CUDA mse_loss target missing resident buffer".to_string())?;
                let (diff_buf, loss_buf, loss_host) = cuda::mse_forward_typed(
                    &out_buf, out_dtype, out_scale, &tar_buf, tar_dtype, tar_scale,
                )?;
                let loss_val = loss_host
                    .first()
                    .copied()
                    .ok_or_else(|| "CUDA mse_loss returned empty host scalar".to_string())?;
                Ok::<_, String>((diff_buf, loss_buf, loss_val))
            })();
            let cuda_forward = cuda_forward.map_err(|err| {
                assert!(
                    !is_strict_device_execution(),
                    "mse_loss CUDA forward failed in strict device execution: {err}"
                );
                err
            });
            if let Ok((diff_buf, loss_buf, loss_val)) = cuda_forward {
                if !build_graph {
                    return Tensor::from_f32_data_no_grad_with_device_dtype_and_cuda_buffer(
                        arr0(loss_val).into_dyn(),
                        DType::F32,
                        output_device,
                        Some(loss_buf),
                    );
                }

                let output_clone = output.clone();
                let target_clone = target.clone();
                return Tensor(Rc::new(RefCell::new(TensorData {
                    data: arr0(loss_val).into_dyn().into_shared(),
                    f16_data: None,
                    bf16_data: None,
                    i8_data: None,
                    cuda_f32_data: Some(loss_buf),
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
                    parents: vec![output.clone(), target.clone()],
                    backward_op: Some(std::rc::Rc::new(move |grad_output| {
                        let grad_val = *grad_output.first().unwrap();
                        let factor = 2.0 / diff_buf.len() as f32 * grad_val;
                        if is_strict_device_execution() {
                            let (grad_output_buffer, grad_target_buffer) =
                                cuda::mse_backward_f32_buffers(&diff_buf, factor).unwrap_or_else(
                                    |err| panic!("CUDA mse_loss backward failed: {}", err),
                                );
                            if output_clone.requires_grad() {
                                output_clone.add_backward_cuda_grad_buffer_only(grad_output_buffer);
                            }
                            if target_clone.requires_grad() {
                                target_clone.add_backward_cuda_grad_buffer_only(grad_target_buffer);
                            }
                            return;
                        }
                        let (
                            (grad_output_buffer, grad_output_host),
                            (grad_target_buffer, grad_target_host),
                        ) = cuda::mse_backward_f32(&diff_buf, factor)
                            .unwrap_or_else(|err| panic!("CUDA mse_loss backward failed: {}", err));
                        let grad_out = ndarray::Array::from_shape_vec(
                            output_clone.shape_vec(),
                            grad_output_host,
                        )
                        .expect("CUDA mse_loss output grad shape build failed")
                        .into_dyn();
                        let grad_target = ndarray::Array::from_shape_vec(
                            target_clone.shape_vec(),
                            grad_target_host,
                        )
                        .expect("CUDA mse_loss target grad shape build failed")
                        .into_dyn();
                        if output_clone.requires_grad() {
                            output_clone.add_backward_grad_with_cuda_buffer(
                                grad_out,
                                Some(grad_output_buffer),
                            );
                        }
                        if target_clone.requires_grad() {
                            target_clone.add_backward_grad_with_cuda_buffer(
                                grad_target,
                                Some(grad_target_buffer),
                            );
                        }
                    })),
                    requires_grad: true,
                    device: output_device,
                })));
            }
        }

        let loss_val =
            output.with_storage_view_preferring(StoragePreference::F32Compute, |out_view| {
                target.with_storage_view_preferring(StoragePreference::F32Compute, |tar_view| {
                    let out_ref = match out_view {
                        TensorStorageView::F32(view) => view,
                        TensorStorageView::F16(_) => {
                            unreachable!("f32 compute preference should expose f32 view")
                        }
                        TensorStorageView::BF16(_) => {
                            unreachable!("f32 compute preference should expose f32 view")
                        }
                    };
                    let tar_ref = match tar_view {
                        TensorStorageView::F32(view) => view,
                        TensorStorageView::F16(_) => {
                            unreachable!("f32 compute preference should expose f32 view")
                        }
                        TensorStorageView::BF16(_) => {
                            unreachable!("f32 compute preference should expose f32 view")
                        }
                    };
                    let n = out_ref.len() as f32;
                    let sum_sq: f32 = Zip::from(&out_ref)
                        .and(&tar_ref)
                        .par_map_collect(|&o, &t| (o - t).powi(2))
                        .sum();
                    sum_sq / n
                })
            });

        if !build_graph {
            return Tensor::from_f32_data_no_grad_with_device_dtype(
                arr0(loss_val).into_dyn(),
                DType::F32,
                output_device,
            );
        }

        let output_clone = output.clone();
        let target_clone = target.clone();

        Tensor(Rc::new(RefCell::new(TensorData {
            data: arr0(loss_val).into_dyn().into_shared(),
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
            parents: vec![output.clone(), target.clone()],
            backward_op: Some(std::rc::Rc::new(move |grad_output| {
                let grad_val = grad_output.first().unwrap();
                let grad = output_clone.with_storage_view_preferring(
                    StoragePreference::F32Compute,
                    |out| {
                        target_clone.with_storage_view_preferring(
                            StoragePreference::F32Compute,
                            |tar| {
                                let TensorStorageView::F32(out) = out else {
                                    unreachable!("F32 compute view")
                                };
                                let TensorStorageView::F32(tar) = tar else {
                                    unreachable!("F32 compute view")
                                };
                                let factor = 2.0 / out.len() as f32 * grad_val;
                                Zip::from(out)
                                    .and(tar)
                                    .par_map_collect(|&o, &t| (o - t) * factor)
                            },
                        )
                    },
                );
                if output_clone.requires_grad() {
                    output_clone.add_backward_grad(grad.clone());
                }
                if target_clone.requires_grad() {
                    target_clone.add_backward_grad(grad.mapv(|v| -v));
                }
            })),
            requires_grad: true,
            device: output_device,
        })))
    }
}

// --- Cross Entropy Loss ---
// 针对 Batch 进行行级并行优化
/// Mean dense-target cross entropy: `-sum(target * log_softmax(logits)) / batch`.
/// Targets are used as supplied, without normalization; gradients are supported
/// for both logits and floating/quantized target storage through F32 autograd.
pub struct CrossEntropyLoss;

#[cfg(test)]
#[path = "../tests/unit/loss/dense_targets.rs"]
mod dense_targets_tests;

impl CrossEntropyLoss {
    pub fn apply(input_logits: &Tensor, target_onehot: &Tensor) -> Tensor {
        let output_device =
            crate::autograd::assert_same_device(input_logits, target_onehot, "cross_entropy");
        let build_graph =
            !is_no_grad() && (input_logits.requires_grad() || target_onehot.requires_grad());
        let cuda_native_supported = output_device == crate::autograd::Device::Cuda;
        assert_native_device_support(output_device, "cross_entropy", cuda_native_supported);
        let shape = input_logits.shape_vec();
        assert_eq!(
            shape.len(),
            2,
            "cross_entropy currently expects [B, C] logits"
        );
        assert_eq!(
            target_onehot.shape_vec(),
            shape,
            "cross_entropy expects logits and target to have the same shape"
        );
        let batch_size = shape[0];
        let dim = shape[1];
        assert!(
            batch_size > 0,
            "cross_entropy batch size must be greater than zero"
        );
        assert!(
            dim > 0,
            "cross_entropy class dimension must be greater than zero"
        );
        if output_device == crate::autograd::Device::Cuda {
            let cuda_forward = (|| {
                let (logits_buf, logits_dtype, logits_scale) = cuda_loss_storage(input_logits)
                    .ok_or_else(|| {
                        "CUDA cross_entropy logits missing resident buffer".to_string()
                    })?;
                let (target_buf, target_dtype, target_scale) = cuda_loss_storage(target_onehot)
                    .ok_or_else(|| {
                        "CUDA cross_entropy target missing resident buffer".to_string()
                    })?;
                let (softmax_buf, loss_buf, loss_val) = cuda::cross_entropy_logits_typed(
                    &logits_buf,
                    logits_dtype,
                    logits_scale,
                    &target_buf,
                    target_dtype,
                    target_scale,
                    batch_size,
                )?;
                Ok::<_, String>((softmax_buf, loss_buf, loss_val))
            })();
            let cuda_forward = cuda_forward.map_err(|err| {
                assert!(
                    !is_strict_device_execution(),
                    "cross_entropy CUDA forward failed in strict device execution: {err}"
                );
                err
            });
            if let Ok((softmax_buf, loss_buf, loss_val)) = cuda_forward {
                if !build_graph {
                    return Tensor::from_f32_data_no_grad_with_device_dtype_and_cuda_buffer(
                        arr0(loss_val).into_dyn(),
                        DType::F32,
                        output_device,
                        Some(loss_buf),
                    );
                }

                let input_clone = input_logits.clone();
                let target_clone = target_onehot.clone();
                let softmax_buffer = softmax_buf.clone();

                return Tensor(Rc::new(RefCell::new(TensorData {
                    data: arr0(loss_val).into_dyn().into_shared(),
                    f16_data: None,
                    bf16_data: None,
                    i8_data: None,
                    cuda_f32_data: Some(loss_buf),
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
                    parents: vec![input_logits.clone(), target_onehot.clone()],
                    backward_op: Some(std::rc::Rc::new(move |grad_output| {
                        let grad_val = *grad_output.first().unwrap();
                        let factor = grad_val / batch_size as f32;
                        let (xb, xd, xs) = cuda_loss_storage(&input_clone)
                            .expect("CUDA cross_entropy backward logits missing resident buffer");
                        let (tb, td, ts) = cuda_loss_storage(&target_clone)
                            .expect("CUDA cross_entropy backward target missing resident buffer");
                        let (dx, dt) = cuda::cross_entropy_logits_backward_typed(
                            &xb,
                            xd,
                            xs,
                            &tb,
                            td,
                            ts,
                            &softmax_buffer,
                            batch_size,
                            factor,
                            input_clone.requires_grad(),
                            target_clone.requires_grad(),
                        )
                        .unwrap_or_else(|err| panic!("CUDA cross_entropy backward failed: {err}"));
                        for (tensor, buffer) in [(&input_clone, dx), (&target_clone, dt)] {
                            if let Some(buffer) = buffer {
                                if is_strict_device_execution() {
                                    tensor.add_backward_cuda_grad_buffer_only(buffer);
                                } else {
                                    let host = cuda::download_f32(&buffer)
                                        .expect("CUDA cross_entropy gradient download failed");
                                    let grad =
                                        ndarray::Array::from_shape_vec(tensor.shape_vec(), host)
                                            .expect("CUDA cross_entropy grad shape build failed");
                                    tensor.add_backward_grad_with_cuda_buffer(grad, Some(buffer));
                                }
                            }
                        }
                    })),
                    requires_grad: true,
                    device: output_device,
                })));
            }
        }
        // Forward
        let (loss_val, logits_gradient, normalizers) = input_logits.with_storage_view_preferring(
            StoragePreference::F32Compute,
            |logits_view| {
                target_onehot.with_storage_view_preferring(
                    StoragePreference::F32Compute,
                    |targets_view| {
                        let logits_ref = match logits_view {
                            TensorStorageView::F32(view) => view,
                            TensorStorageView::F16(_) => {
                                unreachable!("f32 compute preference should expose f32 view")
                            }
                            TensorStorageView::BF16(_) => {
                                unreachable!("f32 compute preference should expose f32 view")
                            }
                        };
                        let targets_ref = match targets_view {
                            TensorStorageView::F32(view) => view,
                            TensorStorageView::F16(_) => {
                                unreachable!("f32 compute preference should expose f32 view")
                            }
                            TensorStorageView::BF16(_) => {
                                unreachable!("f32 compute preference should expose f32 view")
                            }
                        };

                        let batch_size = logits_ref.shape()[0];
                        let dim = logits_ref.shape()[1];

                        let logits_2d = logits_ref
                            .view()
                            .into_dimensionality::<ndarray::Ix2>()
                            .unwrap();
                        let targets_2d = targets_ref
                            .view()
                            .into_dimensionality::<ndarray::Ix2>()
                            .unwrap();

                        let mut softmax_out_flat = Array2::<f32>::zeros((batch_size, dim));
                        let mut normalizers = Array2::<f32>::zeros((batch_size, 2));
                        let total_loss: f32 = Zip::from(softmax_out_flat.outer_iter_mut())
                            .and(logits_2d.outer_iter())
                            .and(targets_2d.outer_iter())
                            .and(normalizers.outer_iter_mut())
                            .into_par_iter()
                            .map(|(mut sm_row, l_row, t_row, mut norm)| {
                                let max_val = l_row.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                                let mut sum_exp = 0.0f32;

                                for (s_val, &l_val) in sm_row.iter_mut().zip(l_row.iter()) {
                                    let e = (l_val - max_val).exp();
                                    *s_val = e;
                                    sum_exp += e;
                                }

                                let inv_sum = 1.0 / sum_exp;
                                let log_sum = sum_exp.ln();
                                norm[0] = max_val;
                                norm[1] = log_sum;
                                let target_mass = t_row.sum();
                                let mut row_loss = 0.0;

                                for ((s_val, &t_val), &l_val) in
                                    sm_row.iter_mut().zip(t_row.iter()).zip(l_row.iter())
                                {
                                    *s_val = (*s_val * inv_sum) * target_mass - t_val;
                                    if t_val != 0.0 {
                                        row_loss += t_val * ((max_val - l_val) + log_sum);
                                    }
                                }
                                row_loss
                            })
                            .sum();

                        (
                            total_loss / batch_size as f32,
                            softmax_out_flat.into_dyn(),
                            normalizers,
                        )
                    },
                )
            },
        );

        if !build_graph {
            return Tensor::from_f32_data_no_grad_with_device_dtype(
                arr0(loss_val).into_dyn(),
                DType::F32,
                output_device,
            );
        }

        let input_clone = input_logits.clone();
        let target_clone = target_onehot.clone();

        Tensor(Rc::new(RefCell::new(TensorData {
            data: arr0(loss_val).into_dyn().into_shared(),
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
            parents: vec![input_logits.clone(), target_onehot.clone()],
            backward_op: Some(std::rc::Rc::new(move |grad_output| {
                let grad_val = grad_output.first().unwrap();
                let factor = grad_val / batch_size as f32;
                if input_clone.requires_grad() {
                    input_clone.add_backward_grad(logits_gradient.mapv(|v| v * factor));
                }
                if target_clone.requires_grad() {
                    let grad = input_clone.with_storage_view_preferring(
                        StoragePreference::F32Compute,
                        |view| {
                            let TensorStorageView::F32(view) = view else {
                                unreachable!("F32 compute view")
                            };
                            let logits = view.into_dimensionality::<ndarray::Ix2>().unwrap();
                            Array2::from_shape_fn((batch_size, dim), |(row, col)| {
                                ((normalizers[[row, 0]] - logits[[row, col]])
                                    + normalizers[[row, 1]])
                                    * factor
                            })
                            .into_dyn()
                        },
                    );
                    target_clone.add_backward_grad(grad);
                }
            })),
            requires_grad: true,
            device: output_device,
        })))
    }
}

#[cfg(test)]
#[path = "../tests/unit/loss/tests.rs"]
mod tests;
