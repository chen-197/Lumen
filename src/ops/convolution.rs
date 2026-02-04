// src/ops/convolution.rs
use crate::autograd::{Tensor, TensorData};
use ndarray::{Array, Array2, ArrayD, ArrayView3, Axis, Zip, s};
use std::cell::RefCell;
use std::rc::Rc;

//填充
fn padding_array(input: &ArrayD<f32>, pad: (usize, usize)) -> ArrayD<f32> {
    let (pad_h, pad_w) = pad;
    if pad_h == 0 && pad_w == 0 {
        return input.clone();
    }

    let input_view = input.view().into_dimensionality::<ndarray::Ix4>().unwrap();
    let (b, c, h, w) = input_view.dim();
    let mut padded = Array::zeros((b, c, h + 2 * pad_h, w + 2 * pad_w));
    padded
        .slice_mut(s![.., .., pad_h..pad_h + h, pad_w..pad_w + w])
        .assign(&input_view);
    padded.into_dyn()
}

// Input: [Cin, H, W] -> Output: [Cin*KH*KW, Hout*Wout]
fn im2col_2d_fast(
    input: &ArrayView3<f32>,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    out_dim: (usize, usize),
) -> Array2<f32> {
    let (cin, _, _) = input.dim();
    let (kh, kw) = kernel_size;
    let (sh, sw) = stride;
    let (hout, wout) = out_dim;

    let col_height = cin * kh * kw;
    let col_width = hout * wout;
    let mut col = Array2::<f32>::zeros((col_height, col_width));

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
    col
}

// Input: [Cin*KH*KW, Hout*Wout] -> Accumulate to: [Cin, H, W]
// 这是一个累加过程 (+=)，因为多个卷积窗口可能重叠在同一个输入像素上
fn col2im_2d_fast(
    col: &Array2<f32>,
    input_shape: (usize, usize, usize), // (Cin, H, W)
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

    // 我们遍历 col 的每一列 (对应一个输出像素)
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
                            // img += val
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
    let (x_data, w_data, b_data) = {
        let x = input.0.borrow();
        let w = weight.0.borrow();
        let b = bias.map(|t| t.0.borrow().data.clone());
        (x.data.clone(), w.data.clone(), b)
    };

    let x_view_4d = x_data.view().into_dimensionality::<ndarray::Ix4>().unwrap();
    let w_view_4d = w_data.view().into_dimensionality::<ndarray::Ix4>().unwrap();

    let (batch_size, in_channels, in_h, in_w) = x_view_4d.dim();
    let (out_channels, _, k_h, k_w) = w_view_4d.dim();
    let (pad_h, pad_w) = padding;
    let (stride_h, stride_w) = stride;

    let out_h = (in_h + 2 * pad_h - k_h) / stride_h + 1;
    let out_w = (in_w + 2 * pad_w - k_w) / stride_w + 1;

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
            // im2col: [K_dim, Out_pixels]
            let im2col_matrix =
                im2col_2d_fast(&x_sample, (k_h, k_w), (stride_h, stride_w), (out_h, out_w));

            // GEMM: [OutC, K_dim] @ [K_dim, Out_pixels] -> [OutC, Out_pixels]
            let out_matrix = w_col.dot(&im2col_matrix);

            let out_reshaped = out_matrix.into_shape((out_channels, out_h, out_w)).unwrap();
            out_sample.assign(&out_reshaped);

            if let Some(ref bb) = b_data {
                let bb_view = bb.view().into_dimensionality::<ndarray::Ix1>().unwrap();
                for o_c in 0..out_channels {
                    out_sample
                        .slice_mut(s![o_c, .., ..])
                        .mapv_inplace(|v| v + bb_view[o_c]);
                }
            }
        });

    let output_dyn = output.into_dyn();

    let input_clone = input.clone();
    let weight_clone = weight.clone();
    let bias_clone = bias.map(|t| t.clone());

    Tensor(Rc::new(RefCell::new(TensorData {
        data: output_dyn,
        grad: None,
        parents: if let Some(b) = &bias_clone {
            vec![input.clone(), weight.clone(), b.clone()]
        } else {
            vec![input.clone(), weight.clone()]
        },
        backward_op: Some(Box::new(move |grad_output| {
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
    })))
}

fn run_backward_conv2d_gemm(
    grad_output: &ArrayD<f32>,
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    padding: (usize, usize),
    stride: (usize, usize),
    shapes: (usize, usize, usize, usize, usize, usize),
) {
    let (in_c, out_c, kh, kw, out_h, out_w) = shapes;
    let (pad_h, pad_w) = padding;
    let (sh, sw) = stride;

    let (x_dat, w_dat) = {
        let xx = input.0.borrow();
        let ww = weight.0.borrow();
        (xx.data.clone(), ww.data.clone())
    };

    // 准备视图
    let grad_out_view = grad_output
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

    let w_col = w_dat.view().into_shape((out_c, in_c * kh * kw)).unwrap();
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

            // GEMM: dCol = W^T * dY
            let d_col = w_col_t.dot(&g_out_col);

            // Col2Im: dCol -> dX_padded
            let d_im = col2im_2d_fast(
                &d_col,
                (in_c, x_pad_4d.shape()[2], x_pad_4d.shape()[3]),
                (kh, kw),
                (sh, sw),
                (out_h, out_w),
            );

            g_in_sample.assign(&d_im);
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
    input.add_grad(grad_input);

    // 公式: dW = dY * X_col^T
    // dY: [OutC, OutPixels]
    // X_col: [K_dim, OutPixels] -> X_col^T: [OutPixels, K_dim]
    // dW: [OutC, K_dim]

    let grad_weight_sum = Zip::from(grad_out_view.outer_iter())
        .and(x_pad_4d.outer_iter())
        .par_map_collect(|g_out_sample, x_sample| {
            let im2col_matrix = im2col_2d_fast(&x_sample, (kh, kw), (sh, sw), (out_h, out_w));
            let im2col_t = im2col_matrix.t();

            let g_out_col = g_out_sample.to_shape((out_c, out_h * out_w)).unwrap();

            // GEMM: dW_sample = dY * Im2Col^T
            g_out_col.dot(&im2col_t)
        });

    // 累加所有样本的梯度 (Reduce)
    // grad_weight_sum 是 Vec<Array2>
    if !grad_weight_sum.is_empty() {
        let mut final_grad_w = grad_weight_sum[0].clone();
        for i in 1..grad_weight_sum.len() {
            final_grad_w = final_grad_w + &grad_weight_sum[i];
        }
        // Reshape 回 [OutC, InC, KH, KW]
        let final_grad_w_reshaped = final_grad_w.into_shape(w_dat.shape()).unwrap().into_dyn();
        weight.add_grad(final_grad_w_reshaped);
    }

    // --- 3. Grad Bias ---
    if let Some(bc) = bias {
        let grad_bias = grad_out_view
            .sum_axis(Axis(0))
            .sum_axis(Axis(1))
            .sum_axis(Axis(1));
        bc.add_grad(grad_bias.into_dyn());
    }
}

pub fn max_pool2d(input: &Tensor, kernel_size: (usize, usize), stride: (usize, usize)) -> Tensor {
    let x_data = input.data();
    let shape = x_data.shape();
    let (b, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
    let (kh, kw) = kernel_size;
    let (sh, sw) = stride;
    let out_h = (h - kh) / sh + 1;
    let out_w = (w - kw) / sw + 1;
    let mut output = Array::zeros((b, c, out_h, out_w)).into_dyn();
    let mut mask = Array::zeros((b, c, h, w));
    let x_view = x_data.view().into_dimensionality::<ndarray::Ix4>().unwrap();
    let mut out_view = output
        .view_mut()
        .into_dimensionality::<ndarray::Ix4>()
        .unwrap();
    let mut mask_view = mask
        .view_mut()
        .into_dimensionality::<ndarray::Ix4>()
        .unwrap();

    Zip::from(out_view.outer_iter_mut())
        .and(x_view.outer_iter())
        .and(mask_view.outer_iter_mut())
        .par_for_each(|mut out_sample, x_sample, mut mask_sample| {
            Zip::from(out_sample.outer_iter_mut())
                .and(x_sample.outer_iter())
                .and(mask_sample.outer_iter_mut())
                .for_each(|mut out_plane, x_plane, mut mask_plane| {
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
                            mask_plane[[h_start + max_idx.0, w_start + max_idx.1]] = 1.0;
                        }
                    }
                });
        });

    let input_clone = input.clone();
    Tensor(Rc::new(RefCell::new(TensorData {
        data: output,
        grad: None,
        parents: vec![input.clone()],
        backward_op: Some(Box::new(move |grad_output| {
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
                .and(mask.view().outer_iter())
                .par_for_each(|mut g_in_sample, g_out_sample, mask_sample| {
                    Zip::from(g_in_sample.outer_iter_mut())
                        .and(g_out_sample.outer_iter())
                        .and(mask_sample.outer_iter())
                        .for_each(|mut g_in_plane, g_out_plane, mask_plane| {
                            for y in 0..out_h {
                                for x in 0..out_w {
                                    let g = g_out_plane[[y, x]];
                                    let h_start = y * sh;
                                    let w_start = x * sw;
                                    for ky in 0..kh {
                                        for kx in 0..kw {
                                            if mask_plane[[h_start + ky, w_start + kx]] > 0.0 {
                                                g_in_plane[[h_start + ky, w_start + kx]] += g;
                                            }
                                        }
                                    }
                                }
                            }
                        });
                });
            input_clone.add_grad(grad_input.into_dyn());
        })),
        requires_grad: true,
    })))
}
