use crate::autograd::{
    StoragePreference, Tensor, TensorData, TensorStorageOwned, TensorStorageView,
    assert_native_device_support, is_no_grad, is_strict_device_execution,
};
use crate::module::Module;
use crate::ops::cuda;
use crate::precision::DType;
use half::{bf16, f16};
use ndarray::{Array1, Array2, Zip};
use rayon::prelude::*;
use std::cell::RefCell;
use std::rc::Rc;

type RmsNormBackwardTypedResult =
    Result<((cuda::CudaBuffer, Vec<f32>), (cuda::CudaBuffer, Vec<f32>)), String>;

pub struct RMSNorm {
    pub weight: Tensor,
    pub eps: f32,
}

impl RMSNorm {
    pub fn new(dim: usize, eps: f32) -> Self {
        let data = Array1::ones(dim).into_dyn();
        Self {
            weight: Tensor::parameter(data),
            eps,
        }
    }

    pub fn new_with_dtype(dim: usize, eps: f32, dtype: DType) -> Self {
        let data = Array1::ones(dim).into_dyn();
        Self {
            weight: Tensor::parameter_with_dtype(data, dtype),
            eps,
        }
    }
}

fn cuda_rms_norm_storage_buffer(tensor: &Tensor) -> Option<(DType, cuda::CudaBuffer, Option<f32>)> {
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

fn try_cuda_rms_norm_typed(
    input: &Tensor,
    weight: &Tensor,
    rows: usize,
    dim: usize,
    eps: f32,
) -> Option<Result<(cuda::CudaBuffer, Vec<f32>), String>> {
    if input.dtype() == DType::F32 && weight.dtype() == DType::F32 {
        return None;
    }
    let (input_dtype, input_buffer, input_scale) = cuda_rms_norm_storage_buffer(input)?;
    let (weight_dtype, weight_buffer, weight_scale) = cuda_rms_norm_storage_buffer(weight)?;
    Some(cuda::rms_norm_typed(
        &input_buffer,
        input_dtype,
        input_scale,
        &weight_buffer,
        weight_dtype,
        weight_scale,
        rows,
        dim,
        eps,
    ))
}

fn try_cuda_rms_norm_typed_buffer(
    input: &Tensor,
    weight: &Tensor,
    rows: usize,
    dim: usize,
    eps: f32,
) -> Option<Result<cuda::CudaBuffer, String>> {
    if input.dtype() == DType::F32 && weight.dtype() == DType::F32 {
        return None;
    }
    let (input_dtype, input_buffer, input_scale) = cuda_rms_norm_storage_buffer(input)?;
    let (weight_dtype, weight_buffer, weight_scale) = cuda_rms_norm_storage_buffer(weight)?;
    Some(cuda::rms_norm_typed_buffer(
        &input_buffer,
        input_dtype,
        input_scale,
        &weight_buffer,
        weight_dtype,
        weight_scale,
        rows,
        dim,
        eps,
    ))
}

fn try_cuda_rms_norm_i8_typed_output_buffer(
    input: &Tensor,
    weight: &Tensor,
    rows: usize,
    dim: usize,
    eps: f32,
) -> Option<Result<(cuda::CudaBuffer, f32), String>> {
    if input.dtype() != DType::I8 {
        return None;
    }
    let (input_dtype, input_buffer, input_scale) = cuda_rms_norm_storage_buffer(input)?;
    let (weight_dtype, weight_buffer, weight_scale) = cuda_rms_norm_storage_buffer(weight)?;
    Some(cuda::rms_norm_i8_typed_output_buffer(
        &input_buffer,
        input_dtype,
        input_scale,
        &weight_buffer,
        weight_dtype,
        weight_scale,
        rows,
        dim,
        eps,
    ))
}

fn try_cuda_rms_norm_backward_typed(
    input: &Tensor,
    weight: &Tensor,
    grad: &cuda::CudaBuffer,
    rows: usize,
    dim: usize,
    eps: f32,
) -> Option<RmsNormBackwardTypedResult> {
    if input.dtype() == DType::F32 && weight.dtype() == DType::F32 {
        return None;
    }
    let (input_dtype, input_buffer, input_scale) = cuda_rms_norm_storage_buffer(input)?;
    let (weight_dtype, weight_buffer, weight_scale) = cuda_rms_norm_storage_buffer(weight)?;
    Some(cuda::rms_norm_backward_typed(
        &input_buffer,
        input_dtype,
        input_scale,
        &weight_buffer,
        weight_dtype,
        weight_scale,
        grad,
        rows,
        dim,
        eps,
    ))
}

fn try_cuda_rms_norm_backward_typed_buffers(
    input: &Tensor,
    weight: &Tensor,
    grad: &cuda::CudaBuffer,
    rows: usize,
    dim: usize,
    eps: f32,
) -> Option<Result<(cuda::CudaBuffer, cuda::CudaBuffer), String>> {
    if input.dtype() == DType::F32 && weight.dtype() == DType::F32 {
        return None;
    }
    let (input_dtype, input_buffer, input_scale) = cuda_rms_norm_storage_buffer(input)?;
    let (weight_dtype, weight_buffer, weight_scale) = cuda_rms_norm_storage_buffer(weight)?;
    Some(cuda::rms_norm_backward_typed_buffers(
        &input_buffer,
        input_dtype,
        input_scale,
        &weight_buffer,
        weight_dtype,
        weight_scale,
        grad,
        rows,
        dim,
        eps,
    ))
}

impl Module for RMSNorm {
    fn forward(&self, input: Tensor) -> Tensor {
        let output_device = input.device();
        let build_graph = !is_no_grad() && (input.requires_grad() || self.weight.requires_grad());
        assert_native_device_support(
            output_device,
            "rms_norm",
            output_device == crate::autograd::Device::Cuda,
        );
        assert_eq!(
            self.weight.device(),
            output_device,
            "rms_norm expects input and weight on the same device"
        );
        let input_shape = input.shape_vec();
        assert!(!input_shape.is_empty(), "RMSNorm expects at least 1D input");
        let last_dim = *input_shape.last().expect("checked non-empty shape");
        assert!(
            last_dim > 0,
            "RMSNorm last dimension must be greater than zero"
        );
        assert_eq!(
            self.weight.len(),
            last_dim,
            "RMSNorm weight length must match input last dimension"
        );

        if !build_graph {
            if output_device == crate::autograd::Device::Cuda && !input.is_empty() {
                let shape = input.shape_vec();
                let dim = shape[shape.len() - 1];
                let rows = input.len() / dim;
                if input.dtype() == DType::I8 {
                    match try_cuda_rms_norm_i8_typed_output_buffer(
                        &input,
                        &self.weight,
                        rows,
                        dim,
                        self.eps,
                    ) {
                        Some(Ok((i8_buffer, scale))) => {
                            return Tensor::from_cuda_native_lowp_buffer_no_host_with_dtype(
                                &shape,
                                i8_buffer,
                                output_device,
                                DType::I8,
                                Some(scale),
                            );
                        }
                        Some(Err(err)) => {
                            assert!(
                                !is_strict_device_execution(),
                                "CUDA RMSNorm I8 typed-output failed while strict device execution is enabled: {err}"
                            );
                        }
                        None => {}
                    }
                }
                let cuda_out =
                    try_cuda_rms_norm_typed_buffer(&input, &self.weight, rows, dim, self.eps)
                        .unwrap_or_else(|| {
                            input.with_cuda_f32_buffer(|input_buf| {
                                self.weight.with_cuda_f32_buffer(|weight_buf| {
                                    cuda::rms_norm_f32_buffer(
                                        input_buf, weight_buf, rows, dim, self.eps,
                                    )
                                })
                            })
                        });
                if let Ok(buffer) = cuda_out {
                    if input.dtype() == DType::I8 {
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
                                    "CUDA RMSNorm I8 output quantization failed while strict device execution is enabled: {err}"
                                );
                            }
                        }
                    }
                    if matches!(input.dtype(), DType::F16 | DType::BF16) {
                        match cuda::f32_to_lowp_storage_no_host(&buffer, input.dtype()) {
                            Ok(lowp_buffer) => {
                                return Tensor::from_cuda_native_lowp_buffer_no_host_with_dtype(
                                    &shape,
                                    lowp_buffer,
                                    output_device,
                                    input.dtype(),
                                    None,
                                );
                            }
                            Err(err) => {
                                assert!(
                                    !is_strict_device_execution(),
                                    "CUDA RMSNorm low precision output conversion failed while strict device execution is enabled: {err}"
                                );
                            }
                        }
                    }
                    return Tensor::from_cuda_f32_buffer_no_host_with_dtype(
                        &shape,
                        buffer,
                        output_device,
                        input.dtype(),
                    );
                }
            }

            if input.dtype() == DType::I8 {
                return match input.native_storage_owned() {
                    TensorStorageOwned::I8(input_data, input_scale) => {
                        self.weight.with_storage_view_preferring(
                            StoragePreference::F32Compute,
                            |weight_view| {
                                let shape = input_data.shape().to_vec();
                                let dim = shape[shape.len() - 1];
                                let rows = shape.iter().product::<usize>() / dim;
                                let eps = self.eps;
                                let x_cow = input_data.as_standard_layout();
                                let x_2d = x_cow.view().into_shape((rows, dim)).unwrap();
                                let mut output_flat = Array2::<f32>::zeros((rows, dim));

                                match weight_view {
                                    TensorStorageView::F32(weight_view) => {
                                        let weight_view = weight_view.as_standard_layout();
                                        let w_1d = weight_view
                                            .into_dimensionality::<ndarray::Ix1>()
                                            .expect("RMSNorm weight must be 1D");
                                        let w_slice = w_1d
                                            .as_slice()
                                            .expect("RMSNorm weight should be contiguous");

                                        Zip::from(output_flat.outer_iter_mut())
                                            .and(x_2d.outer_iter())
                                            .par_for_each(|mut out_row, x_row| {
                                                let sum_sq = x_row.fold(0.0f32, |acc, &val| {
                                                    let v = val as f32 * input_scale;
                                                    acc + v * v
                                                });
                                                let rms = (sum_sq / dim as f32 + eps).sqrt();
                                                let inv_rms = 1.0 / rms;

                                                for (o, (&xi, &wi)) in out_row
                                                    .iter_mut()
                                                    .zip(x_row.iter().zip(w_slice))
                                                {
                                                    *o = (xi as f32 * input_scale) * inv_rms * wi;
                                                }
                                            });
                                    }
                                    TensorStorageView::F16(weight_view) => {
                                        let weight_view = weight_view.as_standard_layout();
                                        let w_1d = weight_view
                                            .into_dimensionality::<ndarray::Ix1>()
                                            .expect("RMSNorm weight must be 1D");
                                        Zip::from(output_flat.outer_iter_mut())
                                            .and(x_2d.outer_iter())
                                            .par_for_each(|mut out_row, x_row| {
                                                let sum_sq = x_row.fold(0.0f32, |acc, &val| {
                                                    let v = val as f32 * input_scale;
                                                    acc + v * v
                                                });
                                                let rms = (sum_sq / dim as f32 + eps).sqrt();
                                                let inv_rms = 1.0 / rms;

                                                for (o, (&xi, &wi)) in out_row
                                                    .iter_mut()
                                                    .zip(x_row.iter().zip(w_1d.iter()))
                                                {
                                                    *o = (xi as f32 * input_scale)
                                                        * inv_rms
                                                        * wi.to_f32();
                                                }
                                            });
                                    }
                                    TensorStorageView::BF16(weight_view) => {
                                        let weight_view = weight_view.as_standard_layout();
                                        let w_1d = weight_view
                                            .into_dimensionality::<ndarray::Ix1>()
                                            .expect("RMSNorm weight must be 1D");
                                        Zip::from(output_flat.outer_iter_mut())
                                            .and(x_2d.outer_iter())
                                            .par_for_each(|mut out_row, x_row| {
                                                let sum_sq = x_row.fold(0.0f32, |acc, &val| {
                                                    let v = val as f32 * input_scale;
                                                    acc + v * v
                                                });
                                                let rms = (sum_sq / dim as f32 + eps).sqrt();
                                                let inv_rms = 1.0 / rms;

                                                for (o, (&xi, &wi)) in out_row
                                                    .iter_mut()
                                                    .zip(x_row.iter().zip(w_1d.iter()))
                                                {
                                                    *o = (xi as f32 * input_scale)
                                                        * inv_rms
                                                        * wi.to_f32();
                                                }
                                            });
                                    }
                                }

                                let out = Tensor::from_f32_data_no_grad_with_device_dtype(
                                    output_flat.into_shape(shape).unwrap().into_dyn(),
                                    DType::F32,
                                    output_device,
                                );
                                out.cast_inplace(DType::I8);
                                out
                            },
                        )
                    }
                    TensorStorageOwned::F32(_)
                    | TensorStorageOwned::F16(_)
                    | TensorStorageOwned::BF16(_) => unreachable!("checked i8 input above"),
                };
            }

            return input.with_storage_view_preferring(StoragePreference::Native, |input_view| {
                self.weight.with_storage_view(|weight_view| {
                    let shape = match &input_view {
                        TensorStorageView::F32(input_view) => input_view.shape().to_vec(),
                        TensorStorageView::F16(input_view) => input_view.shape().to_vec(),
                        TensorStorageView::BF16(input_view) => input_view.shape().to_vec(),
                    };
                    let dim = shape[shape.len() - 1];
                    let rows = shape.iter().product::<usize>() / dim;
                    let eps = self.eps;

                    match input_view {
                        TensorStorageView::F32(input_view) => {
                            let x_cow = input_view.as_standard_layout();
                            let x_2d = x_cow.view().into_shape((rows, dim)).unwrap();
                            let mut output_flat = Array2::<f32>::zeros((rows, dim));

                            match weight_view {
                                TensorStorageView::F32(weight_view) => {
                                    let weight_view = weight_view.as_standard_layout();
                                    let w_1d = weight_view
                                        .into_dimensionality::<ndarray::Ix1>()
                                        .expect("RMSNorm weight must be 1D");
                                    let w_slice = w_1d
                                        .as_slice()
                                        .expect("RMSNorm weight should be contiguous");

                                    Zip::from(output_flat.outer_iter_mut())
                                        .and(x_2d.outer_iter())
                                        .par_for_each(|mut out_row, x_row| {
                                            let sum_sq =
                                                x_row.fold(0.0f32, |acc, &val| acc + val * val);
                                            let rms = (sum_sq / dim as f32 + eps).sqrt();
                                            let inv_rms = 1.0 / rms;

                                            for (o, (&xi, &wi)) in
                                                out_row.iter_mut().zip(x_row.iter().zip(w_slice))
                                            {
                                                *o = xi * inv_rms * wi;
                                            }
                                        });
                                }
                                TensorStorageView::F16(weight_view) => {
                                    let weight_view = weight_view.as_standard_layout();
                                    let w_1d = weight_view
                                        .into_dimensionality::<ndarray::Ix1>()
                                        .expect("RMSNorm weight must be 1D");

                                    Zip::from(output_flat.outer_iter_mut())
                                        .and(x_2d.outer_iter())
                                        .par_for_each(|mut out_row, x_row| {
                                            let sum_sq =
                                                x_row.fold(0.0f32, |acc, &val| acc + val * val);
                                            let rms = (sum_sq / dim as f32 + eps).sqrt();
                                            let inv_rms = 1.0 / rms;

                                            for (o, (&xi, &wi)) in out_row
                                                .iter_mut()
                                                .zip(x_row.iter().zip(w_1d.iter()))
                                            {
                                                *o = xi * inv_rms * wi.to_f32();
                                            }
                                        });
                                }
                                TensorStorageView::BF16(weight_view) => {
                                    let weight_view = weight_view.as_standard_layout();
                                    let w_1d = weight_view
                                        .into_dimensionality::<ndarray::Ix1>()
                                        .expect("RMSNorm weight must be 1D");

                                    Zip::from(output_flat.outer_iter_mut())
                                        .and(x_2d.outer_iter())
                                        .par_for_each(|mut out_row, x_row| {
                                            let sum_sq =
                                                x_row.fold(0.0f32, |acc, &val| acc + val * val);
                                            let rms = (sum_sq / dim as f32 + eps).sqrt();
                                            let inv_rms = 1.0 / rms;

                                            for (o, (&xi, &wi)) in out_row
                                                .iter_mut()
                                                .zip(x_row.iter().zip(w_1d.iter()))
                                            {
                                                *o = xi * inv_rms * wi.to_f32();
                                            }
                                        });
                                }
                            }

                            Tensor::from_f32_data_no_grad_with_device_dtype(
                                output_flat.into_shape(shape).unwrap().into_dyn(),
                                input.dtype(),
                                output_device,
                            )
                        }
                        TensorStorageView::F16(input_view) => {
                            let x_cow = input_view.as_standard_layout();
                            let x_2d = x_cow.view().into_shape((rows, dim)).unwrap();
                            let mut output_flat =
                                Array2::<f16>::from_elem((rows, dim), f16::from_bits(0));

                            match weight_view {
                                TensorStorageView::F32(weight_view) => {
                                    let weight_view = weight_view.as_standard_layout();
                                    let w_1d = weight_view
                                        .into_dimensionality::<ndarray::Ix1>()
                                        .expect("RMSNorm weight must be 1D");
                                    let w_slice = w_1d
                                        .as_slice()
                                        .expect("RMSNorm weight should be contiguous");

                                    Zip::from(output_flat.outer_iter_mut())
                                        .and(x_2d.outer_iter())
                                        .par_for_each(|mut out_row, x_row| {
                                            let sum_sq = x_row.fold(0.0f32, |acc, &val| {
                                                let v = val.to_f32();
                                                acc + v * v
                                            });
                                            let rms = (sum_sq / dim as f32 + eps).sqrt();
                                            let inv_rms = 1.0 / rms;

                                            for (o, (&xi, &wi)) in
                                                out_row.iter_mut().zip(x_row.iter().zip(w_slice))
                                            {
                                                *o = f16::from_f32(xi.to_f32() * inv_rms * wi);
                                            }
                                        });
                                }
                                TensorStorageView::F16(weight_view) => {
                                    let weight_view = weight_view.as_standard_layout();
                                    let w_1d = weight_view
                                        .into_dimensionality::<ndarray::Ix1>()
                                        .expect("RMSNorm weight must be 1D");

                                    Zip::from(output_flat.outer_iter_mut())
                                        .and(x_2d.outer_iter())
                                        .par_for_each(|mut out_row, x_row| {
                                            let sum_sq = x_row.fold(0.0f32, |acc, &val| {
                                                let v = val.to_f32();
                                                acc + v * v
                                            });
                                            let rms = (sum_sq / dim as f32 + eps).sqrt();
                                            let inv_rms = 1.0 / rms;

                                            for (o, (&xi, &wi)) in out_row
                                                .iter_mut()
                                                .zip(x_row.iter().zip(w_1d.iter()))
                                            {
                                                *o = f16::from_f32(
                                                    xi.to_f32() * inv_rms * wi.to_f32(),
                                                );
                                            }
                                        });
                                }
                                TensorStorageView::BF16(weight_view) => {
                                    let weight_view = weight_view.as_standard_layout();
                                    let w_1d = weight_view
                                        .into_dimensionality::<ndarray::Ix1>()
                                        .expect("RMSNorm weight must be 1D");

                                    Zip::from(output_flat.outer_iter_mut())
                                        .and(x_2d.outer_iter())
                                        .par_for_each(|mut out_row, x_row| {
                                            let sum_sq = x_row.fold(0.0f32, |acc, &val| {
                                                let v = val.to_f32();
                                                acc + v * v
                                            });
                                            let rms = (sum_sq / dim as f32 + eps).sqrt();
                                            let inv_rms = 1.0 / rms;

                                            for (o, (&xi, &wi)) in out_row
                                                .iter_mut()
                                                .zip(x_row.iter().zip(w_1d.iter()))
                                            {
                                                *o = f16::from_f32(
                                                    xi.to_f32() * inv_rms * wi.to_f32(),
                                                );
                                            }
                                        });
                                }
                            }

                            Tensor::from_shared_f16_no_grad_with_device(
                                output_flat
                                    .into_shape(shape)
                                    .unwrap()
                                    .into_dyn()
                                    .into_shared(),
                                output_device,
                            )
                        }
                        TensorStorageView::BF16(input_view) => {
                            let x_cow = input_view.as_standard_layout();
                            let x_2d = x_cow.view().into_shape((rows, dim)).unwrap();
                            let mut output_flat =
                                Array2::<bf16>::from_elem((rows, dim), bf16::from_bits(0));

                            match weight_view {
                                TensorStorageView::F32(weight_view) => {
                                    let weight_view = weight_view.as_standard_layout();
                                    let w_1d = weight_view
                                        .into_dimensionality::<ndarray::Ix1>()
                                        .expect("RMSNorm weight must be 1D");
                                    let w_slice = w_1d
                                        .as_slice()
                                        .expect("RMSNorm weight should be contiguous");

                                    Zip::from(output_flat.outer_iter_mut())
                                        .and(x_2d.outer_iter())
                                        .par_for_each(|mut out_row, x_row| {
                                            let sum_sq = x_row.fold(0.0f32, |acc, &val| {
                                                let v = val.to_f32();
                                                acc + v * v
                                            });
                                            let rms = (sum_sq / dim as f32 + eps).sqrt();
                                            let inv_rms = 1.0 / rms;

                                            for (o, (&xi, &wi)) in
                                                out_row.iter_mut().zip(x_row.iter().zip(w_slice))
                                            {
                                                *o = bf16::from_f32(xi.to_f32() * inv_rms * wi);
                                            }
                                        });
                                }
                                TensorStorageView::F16(weight_view) => {
                                    let weight_view = weight_view.as_standard_layout();
                                    let w_1d = weight_view
                                        .into_dimensionality::<ndarray::Ix1>()
                                        .expect("RMSNorm weight must be 1D");

                                    Zip::from(output_flat.outer_iter_mut())
                                        .and(x_2d.outer_iter())
                                        .par_for_each(|mut out_row, x_row| {
                                            let sum_sq = x_row.fold(0.0f32, |acc, &val| {
                                                let v = val.to_f32();
                                                acc + v * v
                                            });
                                            let rms = (sum_sq / dim as f32 + eps).sqrt();
                                            let inv_rms = 1.0 / rms;

                                            for (o, (&xi, &wi)) in out_row
                                                .iter_mut()
                                                .zip(x_row.iter().zip(w_1d.iter()))
                                            {
                                                *o = bf16::from_f32(
                                                    xi.to_f32() * inv_rms * wi.to_f32(),
                                                );
                                            }
                                        });
                                }
                                TensorStorageView::BF16(weight_view) => {
                                    let weight_view = weight_view.as_standard_layout();
                                    let w_1d = weight_view
                                        .into_dimensionality::<ndarray::Ix1>()
                                        .expect("RMSNorm weight must be 1D");

                                    Zip::from(output_flat.outer_iter_mut())
                                        .and(x_2d.outer_iter())
                                        .par_for_each(|mut out_row, x_row| {
                                            let sum_sq = x_row.fold(0.0f32, |acc, &val| {
                                                let v = val.to_f32();
                                                acc + v * v
                                            });
                                            let rms = (sum_sq / dim as f32 + eps).sqrt();
                                            let inv_rms = 1.0 / rms;

                                            for (o, (&xi, &wi)) in out_row
                                                .iter_mut()
                                                .zip(x_row.iter().zip(w_1d.iter()))
                                            {
                                                *o = bf16::from_f32(
                                                    xi.to_f32() * inv_rms * wi.to_f32(),
                                                );
                                            }
                                        });
                                }
                            }

                            Tensor::from_shared_bf16_no_grad_with_device(
                                output_flat
                                    .into_shape(shape)
                                    .unwrap()
                                    .into_dyn()
                                    .into_shared(),
                                output_device,
                            )
                        }
                    }
                })
            });
        }

        if output_device == crate::autograd::Device::Cuda && !input.is_empty() {
            let shape = input.shape_vec();
            let dim = shape[shape.len() - 1];
            let rows = input.len() / dim;
            let cuda_out = try_cuda_rms_norm_typed(&input, &self.weight, rows, dim, self.eps)
                .unwrap_or_else(|| {
                    input.with_cuda_f32_buffer(|input_buf| {
                        self.weight.with_cuda_f32_buffer(|weight_buf| {
                            cuda::rms_norm_f32(input_buf, weight_buf, rows, dim, self.eps)
                        })
                    })
                });
            match cuda_out {
                Ok((buffer, output_host)) => {
                    let output_data =
                        ndarray::Array::from_shape_vec(ndarray::IxDyn(&shape), output_host)
                            .expect("CUDA RMSNorm training output shape build failed")
                            .into_dyn();
                    let input_clone = input.clone();
                    let weight_clone = self.weight.clone();
                    let eps = self.eps;
                    let output_self =
                        Rc::new(RefCell::new(std::rc::Weak::<RefCell<TensorData>>::new()));
                    let output_self_for_backward = output_self.clone();
                    let tensor = Tensor(Rc::new(RefCell::new(TensorData {
                        data: output_data.into_shared(),
                        f16_data: None,
                        bf16_data: None,
                        i8_data: None,
                        cuda_f32_data: Some(buffer),
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
                        parents: vec![input.clone(), self.weight.clone()],
                        backward_op: Some(std::rc::Rc::new(move |grad_output| {
                            let upstream_cuda_grad = output_self_for_backward
                                .borrow()
                                .upgrade()
                                .map(Tensor)
                                .and_then(|output| output.cloned_cuda_f32_grad())
                                .filter(|grad_buf| grad_buf.len() == grad_output.len());
                            let cuda_grad = if let Some(grad_buf) = upstream_cuda_grad {
                                if is_strict_device_execution() {
                                    let grad_result = try_cuda_rms_norm_backward_typed_buffers(
                                        &input_clone,
                                        &weight_clone,
                                        &grad_buf,
                                        rows,
                                        dim,
                                        eps,
                                    )
                                    .unwrap_or_else(|| {
                                        input_clone.with_cuda_f32_buffer(|input_buf| {
                                            weight_clone.with_cuda_f32_buffer(|weight_buf| {
                                                cuda::rms_norm_backward_f32_buffers(
                                                    input_buf, weight_buf, &grad_buf, rows, dim,
                                                    eps,
                                                )
                                            })
                                        })
                                    });
                                    match grad_result {
                                        Ok((grad_input_buffer, grad_weight_buffer)) => {
                                            input_clone.add_backward_cuda_grad_buffer_only(
                                                grad_input_buffer,
                                            );
                                            weight_clone.add_backward_cuda_grad_buffer_only(
                                                grad_weight_buffer,
                                            );
                                            return;
                                        }
                                        Err(err) => {
                                            panic!("CUDA RMSNorm backward failed: {err}");
                                        }
                                    }
                                }
                                try_cuda_rms_norm_backward_typed(
                                    &input_clone,
                                    &weight_clone,
                                    &grad_buf,
                                    rows,
                                    dim,
                                    eps,
                                )
                                .unwrap_or_else(|| {
                                    input_clone.with_cuda_f32_buffer(|input_buf| {
                                        weight_clone.with_cuda_f32_buffer(|weight_buf| {
                                            cuda::rms_norm_backward_f32(
                                                input_buf, weight_buf, &grad_buf, rows, dim, eps,
                                            )
                                        })
                                    })
                                })
                            } else {
                                let grad_host = grad_output.iter().copied().collect::<Vec<_>>();
                                if is_strict_device_execution() {
                                    match cuda::upload_f32(&grad_host).and_then(|grad_buf| {
                                        try_cuda_rms_norm_backward_typed_buffers(
                                            &input_clone,
                                            &weight_clone,
                                            &grad_buf,
                                            rows,
                                            dim,
                                            eps,
                                        )
                                        .unwrap_or_else(
                                            || {
                                                input_clone.with_cuda_f32_buffer(|input_buf| {
                                                    weight_clone.with_cuda_f32_buffer(
                                                        |weight_buf| {
                                                            cuda::rms_norm_backward_f32_buffers(
                                                                input_buf, weight_buf, &grad_buf,
                                                                rows, dim, eps,
                                                            )
                                                        },
                                                    )
                                                })
                                            },
                                        )
                                    }) {
                                        Ok((grad_input_buffer, grad_weight_buffer)) => {
                                            input_clone.add_backward_cuda_grad_buffer_only(
                                                grad_input_buffer,
                                            );
                                            weight_clone.add_backward_cuda_grad_buffer_only(
                                                grad_weight_buffer,
                                            );
                                            return;
                                        }
                                        Err(err) => {
                                            panic!("CUDA RMSNorm backward failed: {err}");
                                        }
                                    }
                                }
                                cuda::upload_f32(&grad_host).and_then(|grad_buf| {
                                    try_cuda_rms_norm_backward_typed(
                                        &input_clone,
                                        &weight_clone,
                                        &grad_buf,
                                        rows,
                                        dim,
                                        eps,
                                    )
                                    .unwrap_or_else(|| {
                                        input_clone.with_cuda_f32_buffer(|input_buf| {
                                            weight_clone.with_cuda_f32_buffer(|weight_buf| {
                                                cuda::rms_norm_backward_f32(
                                                    input_buf, weight_buf, &grad_buf, rows, dim,
                                                    eps,
                                                )
                                            })
                                        })
                                    })
                                })
                            };
                            match cuda_grad {
                                Ok((
                                    (grad_input_buffer, grad_input_host),
                                    (grad_weight_buffer, grad_weight_host),
                                )) => {
                                    let d_input = ndarray::Array::from_shape_vec(
                                        ndarray::IxDyn(&shape),
                                        grad_input_host,
                                    )
                                    .expect("CUDA RMSNorm input grad shape build failed")
                                    .into_dyn();
                                    let d_weight = Array1::from_vec(grad_weight_host).into_dyn();
                                    input_clone.add_backward_grad_with_cuda_buffer(
                                        d_input,
                                        Some(grad_input_buffer),
                                    );
                                    weight_clone.add_backward_grad_with_cuda_buffer(
                                        d_weight,
                                        Some(grad_weight_buffer),
                                    );
                                }
                                Err(err) => {
                                    panic!("CUDA RMSNorm backward failed: {err}");
                                }
                            }
                        })),
                        requires_grad: true,
                        device: output_device,
                    })));
                    *output_self.borrow_mut() = Rc::downgrade(&tensor.0);
                    return tensor;
                }
                Err(err) => {
                    assert!(
                        !is_strict_device_execution(),
                        "rms_norm CUDA forward failed while strict device execution is enabled: {err}"
                    );
                }
            }
        }

        let (output_data, rows, dim, shape) = self.weight.with_storage_view(|weight_view| {
            input.with_storage_view_preferring(StoragePreference::F32Compute, |input_view| {
                let TensorStorageView::F32(x) = input_view else {
                    unreachable!("CPU RMSNorm requested an F32 compute view")
                };

                let shape = x.shape().to_vec();
                let dim = shape[shape.len() - 1];
                let rows = x.len() / dim;

                let x_cow = x.as_standard_layout();
                let x_2d = x_cow.view().into_shape((rows, dim)).unwrap();
                let mut output_flat = Array2::<f32>::zeros((rows, dim));
                let eps = self.eps;

                match weight_view {
                    TensorStorageView::F32(weight_view) => {
                        let weight_view = weight_view.as_standard_layout();
                        let w_1d = weight_view
                            .into_dimensionality::<ndarray::Ix1>()
                            .expect("RMSNorm weight must be 1D");
                        let w_slice = w_1d
                            .as_slice()
                            .expect("RMSNorm weight should be contiguous");

                        Zip::from(output_flat.outer_iter_mut())
                            .and(x_2d.outer_iter())
                            .par_for_each(|mut out_row, x_row| {
                                let sum_sq = x_row.fold(0.0f32, |acc, &val| acc + val * val);
                                let rms = (sum_sq / dim as f32 + eps).sqrt();
                                let inv_rms = 1.0 / rms;

                                for (o, (&xi, &wi)) in
                                    out_row.iter_mut().zip(x_row.iter().zip(w_slice))
                                {
                                    *o = xi * inv_rms * wi;
                                }
                            });
                    }
                    TensorStorageView::F16(weight_view) => {
                        let weight_view = weight_view.as_standard_layout();
                        let w_1d = weight_view
                            .into_dimensionality::<ndarray::Ix1>()
                            .expect("RMSNorm weight must be 1D");

                        Zip::from(output_flat.outer_iter_mut())
                            .and(x_2d.outer_iter())
                            .par_for_each(|mut out_row, x_row| {
                                let sum_sq = x_row.fold(0.0f32, |acc, &val| acc + val * val);
                                let rms = (sum_sq / dim as f32 + eps).sqrt();
                                let inv_rms = 1.0 / rms;

                                for (o, (&xi, &wi)) in
                                    out_row.iter_mut().zip(x_row.iter().zip(w_1d.iter()))
                                {
                                    *o = xi * inv_rms * wi.to_f32();
                                }
                            });
                    }
                    TensorStorageView::BF16(weight_view) => {
                        let weight_view = weight_view.as_standard_layout();
                        let w_1d = weight_view
                            .into_dimensionality::<ndarray::Ix1>()
                            .expect("RMSNorm weight must be 1D");

                        Zip::from(output_flat.outer_iter_mut())
                            .and(x_2d.outer_iter())
                            .par_for_each(|mut out_row, x_row| {
                                let sum_sq = x_row.fold(0.0f32, |acc, &val| acc + val * val);
                                let rms = (sum_sq / dim as f32 + eps).sqrt();
                                let inv_rms = 1.0 / rms;

                                for (o, (&xi, &wi)) in
                                    out_row.iter_mut().zip(x_row.iter().zip(w_1d.iter()))
                                {
                                    *o = xi * inv_rms * wi.to_f32();
                                }
                            });
                    }
                }

                let out_d = output_flat.into_shape(shape.clone()).unwrap().into_dyn();
                (out_d, rows, dim, shape)
            })
        });

        let input_clone = input.clone();
        let weight_clone = self.weight.clone();
        let eps = self.eps;

        Tensor(Rc::new(RefCell::new(TensorData {
            data: output_data.into_shared(),
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
            parents: vec![input.clone(), self.weight.clone()],
            backward_op: Some(std::rc::Rc::new(move |grad_output| {
                let (d_input, d_weight) = input_clone.with_storage_view_preferring(
                    StoragePreference::F32Compute,
                    |input_view| {
                        let TensorStorageView::F32(x_ref) = input_view else {
                            unreachable!("CPU RMSNorm backward requested an F32 compute view")
                        };
                        let x_cow = x_ref.as_standard_layout();
                        let x_2d = x_cow.view().into_shape((rows, dim)).unwrap();

                        weight_clone.with_storage_view_preferring(
                            StoragePreference::F32Compute,
                            |weight_view| {
                                let TensorStorageView::F32(w_ref) = weight_view else {
                                    unreachable!(
                                        "CPU RMSNorm backward requested an F32 weight view"
                                    )
                                };
                                let w_cow = w_ref.as_standard_layout();
                                let w_slice = w_cow.as_slice().unwrap();

                                let g_cow = grad_output.as_standard_layout();
                                let g_2d = g_cow.view().into_shape((rows, dim)).unwrap();

                                let mut d_input_flat = Array2::<f32>::zeros((rows, dim));
                                Zip::from(d_input_flat.outer_iter_mut())
                                    .and(x_2d.outer_iter())
                                    .and(g_2d.outer_iter())
                                    .par_for_each(|mut dx_row, x_row, g_row| {
                                        let sum_sq =
                                            x_row.fold(0.0f32, |acc, &val| acc + val * val);
                                        let inv_rms = 1.0 / (sum_sq / dim as f32 + eps).sqrt();
                                        let inv_dim = 1.0 / dim as f32;

                                        let mut dot = 0.0f32;
                                        for (&gi, (&wi, &xi)) in
                                            g_row.iter().zip(w_slice.iter().zip(x_row.iter()))
                                        {
                                            dot += (gi * wi) * (xi * inv_rms);
                                        }

                                        let mean_dot = dot * inv_dim;

                                        for (dxi, (&gi, (&wi, &xi))) in dx_row
                                            .iter_mut()
                                            .zip(g_row.iter().zip(w_slice.iter().zip(x_row.iter())))
                                        {
                                            let term1 = gi * wi;
                                            let x_norm = xi * inv_rms;
                                            *dxi = inv_rms * (term1 - x_norm * mean_dot);
                                        }
                                    });

                                let dw_accum = (0..rows)
                                    .into_par_iter()
                                    .fold(
                                        || Array1::<f32>::zeros(dim),
                                        |mut acc, r| {
                                            let x_row = x_2d.row(r);
                                            let g_row = g_2d.row(r);

                                            let sum_sq = x_row.fold(0.0f32, |a, &v| a + v * v);
                                            let inv_rms = 1.0 / (sum_sq / dim as f32 + eps).sqrt();

                                            for (a, (&gi, &xi)) in
                                                acc.iter_mut().zip(g_row.iter().zip(x_row.iter()))
                                            {
                                                *a += gi * xi * inv_rms;
                                            }
                                            acc
                                        },
                                    )
                                    .reduce(
                                        || Array1::<f32>::zeros(dim),
                                        |mut a, b| {
                                            a += &b;
                                            a
                                        },
                                    );

                                (
                                    d_input_flat.into_shape(shape.clone()).unwrap().into_dyn(),
                                    dw_accum.into_dyn(),
                                )
                            },
                        )
                    },
                );

                input_clone.add_backward_grad(d_input);
                weight_clone.add_backward_grad(d_weight);
            })),
            requires_grad: true,
            device: output_device,
        })))
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.weight.clone()]
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/layers/norm/rms_norm/tests.rs"]
mod tests;
