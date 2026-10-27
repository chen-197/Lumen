// src/ops/convolution.rs
use crate::autograd::{
    StoragePreference, Tensor, TensorData, TensorStorageView, assert_native_device_support,
    is_no_grad, is_strict_device_execution,
};
use crate::ops::cuda;
use crate::precision::DType;
use ndarray::linalg::general_mat_mul;
use ndarray::{
    Array, ArrayBase, ArrayD, ArrayView2, ArrayView3, ArrayViewD, ArrayViewMut2, Axis, Data, IxDyn,
    Zip, s,
};
use std::cell::RefCell;
use std::rc::Rc;

thread_local! {
    // conv2d forward: per-thread im2col buffer (K_dim * Out_pixels)
    static IM2COL_BUF: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    // conv2d forward: per-thread output GEMM buffer (OutC * Out_pixels)
    static OUT_BUF: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };

    // backward: d_col buffer (K_dim * Out_pixels)
    static DCOL_BUF: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    // backward: dW buffer (OutC * K_dim)
    static DW_BUF: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

fn cuda_conv_storage(tensor: &Tensor) -> Option<(cuda::CudaBuffer, DType, Option<f32>)> {
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

fn cpu_conv_data(tensor: &Tensor) -> ArrayD<f32> {
    tensor.with_storage_view_preferring(StoragePreference::F32Compute, |view| match view {
        TensorStorageView::F32(view) => view.as_standard_layout().to_owned(),
        _ => unreachable!("conv2d F32 compute view"),
    })
}

fn checked_padded_dim(input: usize, padding: usize, axis: &str) -> usize {
    padding
        .checked_mul(2)
        .and_then(|padding| input.checked_add(padding))
        .unwrap_or_else(|| panic!("conv2d padded {axis} overflow"))
}

fn checked_dim_product(factors: &[usize], label: &str) -> usize {
    factors
        .iter()
        .try_fold(1usize, |product, &factor| product.checked_mul(factor))
        .unwrap_or_else(|| panic!("conv2d {label} size overflow"))
}

fn conv_output_dim(
    input: usize,
    kernel: usize,
    padding: usize,
    stride: usize,
    axis: &str,
) -> usize {
    assert!(stride > 0, "conv2d stride must be greater than zero");
    let padded = checked_padded_dim(input, padding, axis);
    assert!(
        padded >= kernel,
        "conv2d kernel is larger than the padded input"
    );
    ((padded - kernel) / stride)
        .checked_add(1)
        .unwrap_or_else(|| panic!("conv2d output {axis} overflow"))
}

//填充
fn padding_array<S>(input: &ArrayBase<S, IxDyn>, pad: (usize, usize)) -> ArrayD<f32>
where
    S: Data<Elem = f32>,
{
    let (pad_h, pad_w) = pad;
    if pad_h == 0 && pad_w == 0 {
        // 需要返回 owned ArrayD
        return input.to_owned();
    }

    let input_view = input.view().into_dimensionality::<ndarray::Ix4>().unwrap();
    let (b, c, h, w) = input_view.dim();
    let padded_h = checked_padded_dim(h, pad_h, "height");
    let padded_w = checked_padded_dim(w, pad_w, "width");
    let mut padded = Array::zeros((b, c, padded_h, padded_w));
    padded
        .slice_mut(s![.., .., pad_h..pad_h + h, pad_w..pad_w + w])
        .assign(&input_view);
    padded.into_dyn()
}

// Input: [Cin, H, W] -> Output: [Cin*KH*KW, Hout*Wout]
fn im2col_2d_fast_into(
    input: &ArrayView3<f32>,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    out_dim: (usize, usize),
    mut col: ArrayViewMut2<'_, f32>,
) {
    let (cin, _, _) = input.dim();
    let (kh, kw) = kernel_size;
    let (sh, sw) = stride;
    let (hout, wout) = out_dim;

    let col_height = cin * kh * kw;
    let col_width = hout * wout;
    debug_assert_eq!(col.dim(), (col_height, col_width));

    let input_ptr = input.as_ptr();
    let strides = input.strides();
    let s_c = strides[0];
    let s_h = strides[1];
    let s_w = strides[2];

    let mut col_idx = 0;
    for y in 0..hout {
        let h_offset_base = (y * sh) as isize * s_h;
        for x in 0..wout {
            let w_offset_base = (x * sw) as isize * s_w;
            let mut row_idx = 0;
            for ic in 0..cin {
                let c_offset = ic as isize * s_c;
                for ky in 0..kh {
                    let h_offset = h_offset_base + ky as isize * s_h;
                    for kx in 0..kw {
                        let w_offset = w_offset_base + kx as isize * s_w;
                        unsafe {
                            let val = *input_ptr.offset(c_offset + h_offset + w_offset);
                            *col.uget_mut((row_idx, col_idx)) = val;
                        }
                        row_idx += 1;
                    }
                }
            }
            col_idx += 1;
        }
    }
}

// View 版本：避免为了 col2im 再分配一个 Array2。
fn col2im_2d_fast_view(
    col: &ArrayView2<f32>,
    input_shape: (usize, usize, usize),
    kernel_size: (usize, usize),
    stride: (usize, usize),
    out_dim: (usize, usize),
) -> Array<f32, ndarray::Ix3> {
    let (cin, h_in, w_in) = input_shape;
    let (kh, kw) = kernel_size;
    let (sh, sw) = stride;
    let (hout, wout) = out_dim;

    let mut img = Array::<f32, ndarray::Ix3>::zeros((cin, h_in, w_in));

    let img_ptr = img.as_mut_ptr();
    let img_strides = img.strides();
    let s_c = img_strides[0];
    let s_h = img_strides[1];
    let s_w = img_strides[2];

    let mut col_idx = 0;
    for y in 0..hout {
        let h_base = (y * sh) as isize * s_h;
        for x in 0..wout {
            let w_base = (x * sw) as isize * s_w;

            let mut row_idx = 0;
            for ic in 0..cin {
                let c_offset = ic as isize * s_c;
                for ky in 0..kh {
                    let h_offset = h_base + ky as isize * s_h;
                    for kx in 0..kw {
                        let w_offset = w_base + kx as isize * s_w;
                        unsafe {
                            let val = *col.uget((row_idx, col_idx));
                            *img_ptr.offset(c_offset + h_offset + w_offset) += val;
                        }
                        row_idx += 1;
                    }
                }
            }
            col_idx += 1;
        }
    }
    img
}

pub fn conv2d(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: (usize, usize),
    padding: (usize, usize),
) -> Tensor {
    let output_device = crate::autograd::assert_same_device(input, weight, "conv2d");
    let build_graph = !is_no_grad()
        && (input.requires_grad()
            || weight.requires_grad()
            || bias.is_some_and(|bias| bias.requires_grad()));
    let cuda_native_supported = output_device == crate::autograd::Device::Cuda;
    assert_native_device_support(output_device, "conv2d", cuda_native_supported);
    if let Some(bias) = bias {
        assert_eq!(
            bias.device(),
            output_device,
            "conv2d expects bias on the same device"
        );
    }

    let x_shape = input.shape_vec();
    let w_shape = weight.shape_vec();
    assert_eq!(x_shape.len(), 4, "conv2d expects 4D input [N, C, H, W]");
    assert_eq!(w_shape.len(), 4, "conv2d expects 4D weight [O, I, KH, KW]");
    let (batch_size, in_channels, in_h, in_w) = (x_shape[0], x_shape[1], x_shape[2], x_shape[3]);
    let (out_channels, weight_in_channels, k_h, k_w) =
        (w_shape[0], w_shape[1], w_shape[2], w_shape[3]);
    assert!(
        in_channels > 0 && out_channels > 0,
        "conv2d input and output channels must be greater than zero"
    );
    assert!(
        k_h > 0 && k_w > 0,
        "conv2d kernel dimensions must be greater than zero"
    );
    assert_eq!(
        weight_in_channels, in_channels,
        "conv2d input/weight channel mismatch"
    );
    let (pad_h, pad_w) = padding;
    let (stride_h, stride_w) = stride;
    let out_h = conv_output_dim(in_h, k_h, pad_h, stride_h, "height");
    let out_w = conv_output_dim(in_w, k_w, pad_w, stride_w, "width");
    let padded_h = checked_padded_dim(in_h, pad_h, "height");
    let padded_w = checked_padded_dim(in_w, pad_w, "width");
    checked_dim_product(
        &[batch_size, in_channels, padded_h, padded_w],
        "padded input",
    );
    checked_dim_product(&[batch_size, out_channels, out_h, out_w], "output");
    checked_dim_product(&[in_channels, k_h, k_w], "kernel");
    checked_dim_product(&[out_h, out_w], "output pixels");

    if output_device == crate::autograd::Device::Cuda && !input.is_empty() && build_graph {
        let bias_buffer = bias.and_then(|tensor| tensor.cloned_cuda_f32_buffer());
        let cuda_out = input.with_cuda_f32_buffer(|input_buf| {
            weight.with_cuda_f32_buffer(|weight_buf| {
                cuda::conv2d_f32_buffer(
                    input_buf,
                    weight_buf,
                    bias_buffer.as_ref(),
                    batch_size,
                    in_channels,
                    in_h,
                    in_w,
                    out_channels,
                    k_h,
                    k_w,
                    pad_h,
                    pad_w,
                    stride_h,
                    stride_w,
                )
            })
        });
        if let Ok((buffer, cuda_out_h, cuda_out_w)) = cuda_out {
            let out_shape = vec![batch_size, out_channels, cuda_out_h, cuda_out_w];
            let input_clone = input.clone();
            let weight_clone = weight.clone();
            let bias_clone = bias.cloned();
            let output_self = Rc::new(RefCell::new(std::rc::Weak::<RefCell<TensorData>>::new()));
            let output_self_for_backward = output_self.clone();
            let tensor = Tensor(Rc::new(RefCell::new(TensorData {
                data: ArrayD::<f32>::zeros(IxDyn(&out_shape)).into_shared(),
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
                parents: if let Some(b) = &bias_clone {
                    vec![input.clone(), weight.clone(), b.clone()]
                } else {
                    vec![input.clone(), weight.clone()]
                },
                backward_op: Some(std::rc::Rc::new(move |grad_output| {
                    let upstream_cuda_grad = output_self_for_backward
                        .borrow()
                        .upgrade()
                        .map(Tensor)
                        .and_then(|output| output.cloned_cuda_f32_grad())
                        .filter(|grad_buffer| grad_buffer.len() == grad_output.len());
                    let input_buffer = input_clone
                        .cloned_cuda_f32_buffer()
                        .expect("CUDA conv2d input missing resident buffer");
                    let weight_buffer = weight_clone
                        .cloned_cuda_f32_buffer()
                        .expect("CUDA conv2d weight missing resident buffer");
                    let grad_buffer = if let Some(grad_buffer) = upstream_cuda_grad {
                        grad_buffer
                    } else {
                        let grad_host = grad_output.iter().copied().collect::<Vec<_>>();
                        cuda::upload_f32(&grad_host).expect("CUDA conv2d grad upload failed")
                    };
                    if is_strict_device_execution() {
                        let (grad_input_buffer, grad_weight_buffer, grad_bias_buffer) =
                            cuda::conv2d_backward_f32_buffers(
                                &input_buffer,
                                &weight_buffer,
                                &grad_buffer,
                                bias_clone.is_some(),
                                batch_size,
                                in_channels,
                                in_h,
                                in_w,
                                out_channels,
                                k_h,
                                k_w,
                                pad_h,
                                pad_w,
                                stride_h,
                                stride_w,
                            )
                            .expect("CUDA conv2d backward failed");
                        input_clone.add_backward_cuda_grad_buffer_only(grad_input_buffer);
                        weight_clone.add_backward_cuda_grad_buffer_only(grad_weight_buffer);
                        if let (Some(bias), Some(grad_bias_buffer)) =
                            (bias_clone.as_ref(), grad_bias_buffer)
                        {
                            bias.add_backward_cuda_grad_buffer_only(grad_bias_buffer);
                        }
                        return;
                    }
                    let (grad_input_buffer, grad_input, grad_weight_buffer, grad_weight, grad_bias) =
                        cuda::conv2d_backward_f32(
                            &input_buffer,
                            &weight_buffer,
                            &grad_buffer,
                            bias_clone.is_some(),
                            batch_size,
                            in_channels,
                            in_h,
                            in_w,
                            out_channels,
                            k_h,
                            k_w,
                            pad_h,
                            pad_w,
                            stride_h,
                            stride_w,
                        )
                        .expect("CUDA conv2d backward failed");
                    let grad_input =
                        Array::from_shape_vec((batch_size, in_channels, in_h, in_w), grad_input)
                            .expect("CUDA conv2d input grad shape build failed")
                            .into_dyn();
                    input_clone
                        .add_backward_grad_with_cuda_buffer(grad_input, Some(grad_input_buffer));
                    let grad_weight =
                        Array::from_shape_vec((out_channels, in_channels, k_h, k_w), grad_weight)
                            .expect("CUDA conv2d weight grad shape build failed")
                            .into_dyn();
                    weight_clone
                        .add_backward_grad_with_cuda_buffer(grad_weight, Some(grad_weight_buffer));
                    if let (Some(bias), Some((grad_bias_buffer, grad_bias))) =
                        (bias_clone.as_ref(), grad_bias)
                    {
                        let grad_bias = Array::from_shape_vec(IxDyn(&[out_channels]), grad_bias)
                            .expect("CUDA conv2d bias grad shape build failed")
                            .into_dyn();
                        bias.add_backward_grad_with_cuda_buffer(grad_bias, Some(grad_bias_buffer));
                    }
                })),
                requires_grad: true,
                device: output_device,
            })));
            *output_self.borrow_mut() = Rc::downgrade(&tensor.0);
            return tensor;
        }
    }

    if output_device == crate::autograd::Device::Cuda && !input.is_empty() && !build_graph {
        let use_typed_conv = input.dtype() != DType::F32
            || weight.dtype() != DType::F32
            || bias.is_some_and(|b| b.dtype() != DType::F32);
        if use_typed_conv
            && let (
                Some((input_buffer, input_dtype, input_scale)),
                Some((weight_buffer, weight_dtype, weight_scale)),
            ) = (cuda_conv_storage(input), cuda_conv_storage(weight))
        {
            let bias_storage = match bias {
                Some(tensor) => match cuda_conv_storage(tensor) {
                    Some((buffer, dtype, scale)) => Some((buffer, dtype, scale)),
                    None => {
                        assert!(
                            !is_strict_device_execution(),
                            "CUDA typed conv2d missing resident bias buffer in strict device execution mode"
                        );
                        None
                    }
                },
                None => None,
            };
            if bias.is_none() || bias_storage.is_some() {
                let cuda_out = cuda::conv2d_typed(
                    &input_buffer,
                    input_dtype,
                    input_scale,
                    &weight_buffer,
                    weight_dtype,
                    weight_scale,
                    bias_storage
                        .as_ref()
                        .map(|(buffer, dtype, scale)| (buffer, *dtype, *scale)),
                    batch_size,
                    in_channels,
                    in_h,
                    in_w,
                    out_channels,
                    k_h,
                    k_w,
                    pad_h,
                    pad_w,
                    stride_h,
                    stride_w,
                );
                match cuda_out {
                    Ok((buffer, out, cuda_out_h, cuda_out_w)) => {
                        let out = Array::from_shape_vec(
                            (batch_size, out_channels, cuda_out_h, cuda_out_w),
                            out,
                        )
                        .expect("CUDA typed conv2d output shape build failed")
                        .into_dyn();
                        return Tensor::from_f32_data_no_grad_with_device_dtype_and_cuda_buffer(
                            out,
                            DType::F32,
                            output_device,
                            Some(buffer),
                        );
                    }
                    Err(err) => {
                        assert!(
                            !is_strict_device_execution(),
                            "CUDA typed conv2d failed in strict device execution mode: {err}"
                        );
                    }
                }
            }
        }

        let bias_buffer = bias.and_then(|tensor| tensor.cloned_cuda_f32_buffer());
        let cuda_out = input.with_cuda_f32_buffer(|input_buf| {
            weight.with_cuda_f32_buffer(|weight_buf| {
                cuda::conv2d_f32(
                    input_buf,
                    weight_buf,
                    bias_buffer.as_ref(),
                    batch_size,
                    in_channels,
                    in_h,
                    in_w,
                    out_channels,
                    k_h,
                    k_w,
                    pad_h,
                    pad_w,
                    stride_h,
                    stride_w,
                )
            })
        });
        if let Ok((buffer, out, cuda_out_h, cuda_out_w)) = cuda_out {
            let out =
                Array::from_shape_vec((batch_size, out_channels, cuda_out_h, cuda_out_w), out)
                    .expect("CUDA conv2d output shape build failed")
                    .into_dyn();
            return Tensor::from_f32_data_no_grad_with_device_dtype_and_cuda_buffer(
                out,
                DType::F32,
                output_device,
                Some(buffer),
            );
        }
    }

    let (x_data, w_data, b_data) = {
        let x = cpu_conv_data(input);
        let w = cpu_conv_data(weight);
        let b = bias.map(cpu_conv_data);
        (x, w, b)
    };

    // --- Forward Pass ---
    let x_padded = padding_array(&x_data, padding);
    let x_padded_view = x_padded
        .view()
        .into_dimensionality::<ndarray::Ix4>()
        .unwrap();

    let mut output = Array::zeros((batch_size, out_channels, out_h, out_w));

    // Weight: [OutC, InC * KH * KW]
    let w_col = w_data
        .to_shape((out_channels, in_channels * k_h * k_w))
        .unwrap();
    let w_col = w_col.as_standard_layout();

    Zip::from(output.outer_iter_mut())
        .and(x_padded_view.outer_iter())
        .par_for_each(|mut out_sample, x_sample| {
            let k_dim = in_channels * k_h * k_w;
            let out_pixels = out_h * out_w;

            IM2COL_BUF.with(|cb| {
                OUT_BUF.with(|ob| {
                    let mut col_buf = cb.borrow_mut();
                    let mut out_buf = ob.borrow_mut();

                    if col_buf.len() != k_dim * out_pixels {
                        col_buf.resize(k_dim * out_pixels, 0.0);
                    }
                    if out_buf.len() != out_channels * out_pixels {
                        out_buf.resize(out_channels * out_pixels, 0.0);
                    }

                    let mut col_view =
                        ArrayViewMut2::from_shape((k_dim, out_pixels), &mut col_buf[..])
                            .expect("im2col buffer shape mismatch");
                    let mut out_view =
                        ArrayViewMut2::from_shape((out_channels, out_pixels), &mut out_buf[..])
                            .expect("out buffer shape mismatch");

                    // im2col into preallocated buffer
                    im2col_2d_fast_into(
                        &x_sample,
                        (k_h, k_w),
                        (stride_h, stride_w),
                        (out_h, out_w),
                        col_view.view_mut(),
                    );

                    // GEMM into preallocated out buffer: out_view = w_col @ col_view
                    out_view.fill(0.0);
                    general_mat_mul(1.0, &w_col, &col_view, 0.0, &mut out_view);

                    // reshape view: [OutC, Out_pixels] -> [OutC, OutH, OutW]
                    let out_reshaped = out_view
                        .into_shape((out_channels, out_h, out_w))
                        .expect("out reshape failed");
                    out_sample.assign(&out_reshaped);

                    if let Some(ref bb) = b_data {
                        let bb_view = bb.view().into_dimensionality::<ndarray::Ix1>().unwrap();
                        for o_c in 0..out_channels {
                            out_sample
                                .slice_mut(s![o_c, .., ..])
                                .mapv_inplace(|v| v + bb_view[o_c]);
                        }
                    }
                })
            });
        });

    let output_dyn = output.into_dyn();

    if !build_graph {
        return Tensor::from_f32_data_no_grad_with_device_dtype(
            output_dyn,
            DType::F32,
            output_device,
        );
    }

    let input_clone = input.clone();
    let weight_clone = weight.clone();
    let bias_clone = bias.cloned();

    Tensor(Rc::new(RefCell::new(TensorData {
        data: output_dyn.into_shared(),
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
        parents: if let Some(b) = &bias_clone {
            vec![input.clone(), weight.clone(), b.clone()]
        } else {
            vec![input.clone(), weight.clone()]
        },
        backward_op: Some(std::rc::Rc::new(move |grad_output| {
            run_backward_conv2d_gemm(
                grad_output,
                &input_clone,
                &weight_clone,
                bias_clone.as_ref(),
                padding,
                stride,
                (in_channels, out_channels, k_h, k_w, out_h, out_w),
            );
        })),
        requires_grad: true,
        device: output_device,
    })))
}

fn run_backward_conv2d_gemm(
    grad_output: &ArrayViewD<'_, f32>,
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    padding: (usize, usize),
    stride: (usize, usize),
    shapes: (usize, usize, usize, usize, usize, usize),
) {
    let (dx, dw, db) = input.with_storage_view_preferring(StoragePreference::F32Compute, |x| {
        weight.with_storage_view_preferring(StoragePreference::F32Compute, |w| match (x, w) {
            (TensorStorageView::F32(x), TensorStorageView::F32(w)) => {
                backward_conv2d_views(grad_output, x, w, padding, stride, shapes, bias.is_some())
            }
            _ => unreachable!("conv2d backward F32 compute views"),
        })
    });
    // Drop all input read guards before writing gradients, including aliases.
    input.add_backward_grad(dx);
    if let Some(dw) = dw {
        weight.add_backward_grad(dw);
    }
    if let (Some(bias), Some(db)) = (bias, db) {
        bias.add_backward_grad(db);
    }
}

type ConvBackwardGrads = (ArrayD<f32>, Option<ArrayD<f32>>, Option<ArrayD<f32>>);

fn backward_conv2d_views(
    grad_output: &ArrayViewD<'_, f32>,
    x_dat: ArrayViewD<'_, f32>,
    w_dat: ArrayViewD<'_, f32>,
    padding: (usize, usize),
    stride: (usize, usize),
    shapes: (usize, usize, usize, usize, usize, usize),
    has_bias: bool,
) -> ConvBackwardGrads {
    let (in_c, out_c, kh, kw, out_h, out_w) = shapes;
    let (pad_h, pad_w) = padding;
    let (sh, sw) = stride;

    // 准备视图
    let grad_standard = grad_output.as_standard_layout();
    let grad_out_view = grad_standard
        .view()
        .into_dimensionality::<ndarray::Ix4>()
        .unwrap();
    let x_pad_view = padding_array(&x_dat, padding);
    let x_pad_4d = x_pad_view
        .view()
        .into_dimensionality::<ndarray::Ix4>()
        .unwrap();
    let _batch_size = x_dat.shape()[0];

    // 公式: dX_col = W^T * dY
    // W: [OutC, InC*KH*KW] -> W^T: [InC*KH*KW, OutC]
    // dY: [OutC, OutH*OutW]
    // dX_col: [InC*KH*KW, OutH*OutW]

    let weight_standard = w_dat.as_standard_layout();
    let w_col = weight_standard
        .view()
        .into_shape((out_c, in_c * kh * kw))
        .unwrap();
    let w_col_t = w_col.t(); // [K_dim, OutC]

    let mut grad_input_padded = Array::zeros(x_pad_4d.dim());
    let mut grad_input_view = grad_input_padded
        .view_mut()
        .into_dimensionality::<ndarray::Ix4>()
        .unwrap();

    // 并行计算 dX
    Zip::from(grad_input_view.outer_iter_mut())
        .and(grad_out_view.outer_iter())
        .par_for_each(|mut g_in_sample, g_out_sample| {
            // g_out_sample: [OutC, OutH, OutW] -> Reshape -> [OutC, OutPixels]
            let g_out_col = g_out_sample.to_shape((out_c, out_h * out_w)).unwrap();

            // GEMM: dCol = W^T * dY（复用 per-thread buffer，避免每步分配 Array2）
            let k_dim = in_c * kh * kw;
            let out_pixels = out_h * out_w;

            DCOL_BUF.with(|db| {
                let mut dcol_buf = db.borrow_mut();
                if dcol_buf.len() != k_dim * out_pixels {
                    dcol_buf.resize(k_dim * out_pixels, 0.0);
                }
                let mut d_col_view =
                    ArrayViewMut2::from_shape((k_dim, out_pixels), &mut dcol_buf[..])
                        .expect("dcol buffer shape mismatch");
                d_col_view.fill(0.0);
                general_mat_mul(1.0, &w_col_t, &g_out_col, 0.0, &mut d_col_view);

                // Col2Im: dCol -> dX_padded (view 版本避免分配)
                let d_im = col2im_2d_fast_view(
                    &d_col_view.view(),
                    (in_c, x_pad_4d.shape()[2], x_pad_4d.shape()[3]),
                    (kh, kw),
                    (sh, sw),
                    (out_h, out_w),
                );

                g_in_sample.assign(&d_im);
            });
        });

    // 去除 padding
    let grad_input = grad_input_padded
        .slice(s![
            ..,
            ..,
            pad_h..pad_h + x_dat.shape()[2],
            pad_w..pad_w + x_dat.shape()[3]
        ])
        .to_owned()
        .into_dyn();

    // 公式: dW = dY * X_col^T
    // dY: [OutC, OutPixels]
    // X_col: [K_dim, OutPixels] -> X_col^T: [OutPixels, K_dim]
    // dW: [OutC, K_dim]

    let grad_weight_sum = Zip::from(grad_out_view.outer_iter())
        .and(x_pad_4d.outer_iter())
        .par_map_collect(|g_out_sample, x_sample| {
            let k_dim = in_c * kh * kw;
            let out_pixels = out_h * out_w;
            let g_out_col = g_out_sample.to_shape((out_c, out_pixels)).unwrap();

            IM2COL_BUF.with(|cb| {
                DW_BUF.with(|wb| {
                    let mut col_buf = cb.borrow_mut();
                    let mut dw_buf = wb.borrow_mut();

                    if col_buf.len() != k_dim * out_pixels {
                        col_buf.resize(k_dim * out_pixels, 0.0);
                    }
                    if dw_buf.len() != out_c * k_dim {
                        dw_buf.resize(out_c * k_dim, 0.0);
                    }

                    let mut col_view =
                        ArrayViewMut2::from_shape((k_dim, out_pixels), &mut col_buf[..])
                            .expect("im2col buffer shape mismatch (bwd)");
                    im2col_2d_fast_into(
                        &x_sample,
                        (kh, kw),
                        (sh, sw),
                        (out_h, out_w),
                        col_view.view_mut(),
                    );

                    let mut dw_view = ArrayViewMut2::from_shape((out_c, k_dim), &mut dw_buf[..])
                        .expect("dw buffer shape mismatch");
                    dw_view.fill(0.0);
                    // dW_sample = dY @ X_col^T
                    general_mat_mul(1.0, &g_out_col, &col_view.t(), 0.0, &mut dw_view);
                    dw_view.to_owned()
                })
            })
        });

    // 累加所有样本的梯度 (Reduce)
    // grad_weight_sum 是 Vec<Array2>
    let grad_weight = if !grad_weight_sum.is_empty() {
        let mut final_grad_w = grad_weight_sum[0].clone();
        for i in 1..grad_weight_sum.len() {
            final_grad_w += &grad_weight_sum[i];
        }
        // Reshape 回 [OutC, InC, KH, KW]
        let final_grad_w_reshaped = final_grad_w.into_shape(w_dat.shape()).unwrap().into_dyn();
        Some(final_grad_w_reshaped)
    } else {
        None
    };

    // --- 3. Grad Bias ---
    let grad_bias = if has_bias {
        let grad_bias = grad_out_view
            .sum_axis(Axis(0))
            .sum_axis(Axis(1))
            .sum_axis(Axis(1));
        Some(grad_bias.into_dyn())
    } else {
        None
    };
    (grad_input, grad_weight, grad_bias)
}

pub fn max_pool2d(input: &Tensor, kernel_size: (usize, usize), stride: (usize, usize)) -> Tensor {
    let output_device = input.device();
    let build_graph = !is_no_grad() && input.requires_grad();
    assert_native_device_support(
        output_device,
        "max_pool2d",
        output_device == crate::autograd::Device::Cuda,
    );

    let shape = input.shape_vec();
    assert_eq!(shape.len(), 4, "max_pool2d expects 4D input [N, C, H, W]");
    let (b, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
    let (kh, kw) = kernel_size;
    let (sh, sw) = stride;
    assert!(
        kh > 0 && kw > 0,
        "max_pool2d kernel must be greater than zero"
    );
    assert!(
        sh > 0 && sw > 0,
        "max_pool2d stride must be greater than zero"
    );
    assert!(h >= kh && w >= kw, "max_pool2d kernel is larger than input");
    let out_h = (h - kh) / sh + 1;
    let out_w = (w - kw) / sw + 1;

    if output_device == crate::autograd::Device::Cuda && !input.is_empty() {
        let cuda_out =
            if let Some((dtype, input_buf, scale)) = input.cloned_cuda_native_lowp_buffer() {
                cuda::max_pool2d_typed(&input_buf, dtype, scale, b, c, h, w, kh, kw, sh, sw)
            } else {
                input.with_cuda_f32_buffer(|input_buf| {
                    cuda::max_pool2d_f32(input_buf, b, c, h, w, kh, kw, sh, sw)
                })
            };
        if let Ok((buffer, out_host, cuda_out_h, cuda_out_w)) = cuda_out {
            let out = Array::from_shape_vec((b, c, cuda_out_h, cuda_out_w), out_host)
                .expect("CUDA max_pool2d output shape build failed")
                .into_dyn();
            if !build_graph {
                return Tensor::from_f32_data_no_grad_with_device_dtype_and_cuda_buffer(
                    out,
                    DType::F32,
                    output_device,
                    Some(buffer),
                );
            }

            let input_clone = input.clone();
            let output_self = Rc::new(RefCell::new(std::rc::Weak::<RefCell<TensorData>>::new()));
            let output_self_for_backward = output_self.clone();
            let tensor = Tensor(Rc::new(RefCell::new(TensorData {
                data: out.into_shared(),
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
                parents: vec![input.clone()],
                backward_op: Some(std::rc::Rc::new(move |grad_output| {
                    let grad_output_buffer = output_self_for_backward
                        .borrow()
                        .upgrade()
                        .map(Tensor)
                        .and_then(|output| output.cloned_cuda_f32_grad())
                        .filter(|grad_buffer| grad_buffer.len() == grad_output.len())
                        .unwrap_or_else(|| {
                            let grad_host = grad_output.iter().copied().collect::<Vec<_>>();
                            cuda::upload_f32(&grad_host)
                                .expect("CUDA max_pool2d grad upload failed")
                        });
                    let typed_grad_input = input_clone.cloned_cuda_native_lowp_buffer().and_then(
                        |(dtype, input_buffer, scale)| {
                            cuda::max_pool2d_backward_typed_buffer(
                                &input_buffer,
                                dtype,
                                scale,
                                &grad_output_buffer,
                                b,
                                c,
                                h,
                                w,
                                kh,
                                kw,
                                sh,
                                sw,
                            )
                            .ok()
                        },
                    );
                    if is_strict_device_execution() {
                        let grad_input_buffer = typed_grad_input.unwrap_or_else(|| {
                            let input_buffer = input_clone
                                .cloned_cuda_f32_buffer()
                                .expect("CUDA max_pool2d input missing resident buffer");
                            cuda::max_pool2d_backward_f32_buffer(
                                &input_buffer,
                                &grad_output_buffer,
                                b,
                                c,
                                h,
                                w,
                                kh,
                                kw,
                                sh,
                                sw,
                            )
                            .expect("CUDA max_pool2d backward failed")
                        });
                        input_clone.add_backward_cuda_grad_buffer_only(grad_input_buffer);
                        return;
                    }
                    let grad_input_buffer = if let Some(buffer) = typed_grad_input {
                        buffer
                    } else {
                        let input_buffer = input_clone
                            .cloned_cuda_f32_buffer()
                            .expect("CUDA max_pool2d input missing resident buffer");
                        cuda::max_pool2d_backward_f32_buffer(
                            &input_buffer,
                            &grad_output_buffer,
                            b,
                            c,
                            h,
                            w,
                            kh,
                            kw,
                            sh,
                            sw,
                        )
                        .expect("CUDA max_pool2d backward failed")
                    };
                    let grad_input = cuda::download_f32(&grad_input_buffer)
                        .expect("CUDA max_pool2d grad download failed");
                    let grad_input = Array::from_shape_vec((b, c, h, w), grad_input)
                        .expect("CUDA max_pool2d input grad shape build failed")
                        .into_dyn();
                    input_clone
                        .add_backward_grad_with_cuda_buffer(grad_input, Some(grad_input_buffer));
                })),
                requires_grad: true,
                device: output_device,
            })));
            *output_self.borrow_mut() = Rc::downgrade(&tensor.0);
            return tensor;
        }
    }

    // Avoid cloning the full tensor data; we only need a read-only view.
    let x_data_ref = input.data_ref();
    let mut output = Array::zeros((b, c, out_h, out_w)).into_dyn();
    let mut argmax = Array::zeros((b, c, out_h, out_w));
    let x_view = x_data_ref
        .view()
        .into_dimensionality::<ndarray::Ix4>()
        .unwrap();
    let mut out_view = output
        .view_mut()
        .into_dimensionality::<ndarray::Ix4>()
        .unwrap();
    let mut argmax_view = argmax
        .view_mut()
        .into_dimensionality::<ndarray::Ix4>()
        .unwrap();

    Zip::from(out_view.outer_iter_mut())
        .and(x_view.outer_iter())
        .and(argmax_view.outer_iter_mut())
        .par_for_each(|mut out_sample, x_sample, mut argmax_sample| {
            Zip::from(out_sample.outer_iter_mut())
                .and(x_sample.outer_iter())
                .and(argmax_sample.outer_iter_mut())
                .for_each(|mut out_plane, x_plane, mut argmax_plane| {
                    for y in 0..out_h {
                        for x in 0..out_w {
                            let h_start = y * sh;
                            let w_start = x * sw;
                            let window =
                                x_plane.slice(s![h_start..h_start + kh, w_start..w_start + kw]);
                            let mut max_val = f32::MIN;
                            let mut max_idx = (0, 0);
                            for ky in 0..kh {
                                for kx in 0..kw {
                                    let v = window[[ky, kx]];
                                    if v > max_val {
                                        max_val = v;
                                        max_idx = (ky, kx);
                                    }
                                }
                            }
                            out_plane[[y, x]] = max_val;
                            if build_graph {
                                argmax_plane[[y, x]] =
                                    (h_start + max_idx.0) * w + (w_start + max_idx.1);
                            }
                        }
                    }
                });
        });

    if !build_graph {
        return Tensor::from_f32_data_no_grad_with_device_dtype(output, DType::F32, output_device);
    }

    let input_clone = input.clone();
    Tensor(Rc::new(RefCell::new(TensorData {
        data: output.into_shared(),
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
        parents: vec![input.clone()],
        backward_op: Some(std::rc::Rc::new(move |grad_output| {
            let grad_view = grad_output
                .view()
                .into_dimensionality::<ndarray::Ix4>()
                .unwrap();
            let mut grad_input = Array::zeros((b, c, h, w));
            let mut grad_input_view = grad_input
                .view_mut()
                .into_dimensionality::<ndarray::Ix4>()
                .unwrap();
            Zip::from(grad_input_view.outer_iter_mut())
                .and(grad_view.outer_iter())
                .and(argmax.view().outer_iter())
                .par_for_each(|mut g_in_sample, g_out_sample, argmax_sample| {
                    Zip::from(g_in_sample.outer_iter_mut())
                        .and(g_out_sample.outer_iter())
                        .and(argmax_sample.outer_iter())
                        .for_each(|mut g_in_plane, g_out_plane, argmax_plane| {
                            for y in 0..out_h {
                                for x in 0..out_w {
                                    let g = g_out_plane[[y, x]];
                                    let flat_idx = argmax_plane[[y, x]];
                                    let yy = flat_idx / w;
                                    let xx = flat_idx % w;
                                    g_in_plane[[yy, xx]] += g;
                                }
                            }
                        });
                });
            input_clone.add_backward_grad(grad_input.into_dyn());
        })),
        requires_grad: true,
        device: output_device,
    })))
}

#[cfg(test)]
#[path = "../../tests/unit/ops/convolution/tests.rs"]
mod tests;
