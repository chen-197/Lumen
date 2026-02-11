use crate::autograd::{Tensor, TensorData};
use ndarray::{Array, Axis, Zip};
use std::cell::RefCell;
use std::rc::Rc;

pub fn fused_softmax(input: &Tensor, scale: f32, is_causal: bool) -> Tensor {
    let (output, output_data) = {
        let x = input.data_ref();
        let shape = x.shape(); // [B, H, Q, K]
        if shape.len() != 4 {
            panic!("Fused Softmax expects 4D input [B, H, Q, K]");
        }

        let q_len = shape[2];
        let k_len = shape[3];

        let mut out = Array::zeros(x.dim());

        // 使用 4D View 进行并行遍历
        let x_view = x.view().into_dimensionality::<ndarray::Ix4>().unwrap();
        let mut out_view = out
            .view_mut()
            .into_dimensionality::<ndarray::Ix4>()
            .unwrap();

        // Rayon 并行化: Batch * Head
        Zip::from(out_view.outer_iter_mut())
            .and(x_view.outer_iter())
            .par_for_each(|mut out_b, x_b| {
                Zip::from(out_b.outer_iter_mut())
                    .and(x_b.outer_iter())
                    .for_each(|mut out_h, x_h| {
                        for i in 0..q_len {
                            let row_in = x_h.slice(ndarray::s![i, ..]);
                            let mut row_out = out_h.slice_mut(ndarray::s![i, ..]);

                            let mut max_val = f32::NEG_INFINITY;
                            for j in 0..k_len {
                                let is_masked = if is_causal && q_len > 1 { j > i } else { false };

                                if !is_masked {
                                    let val = row_in[j] * scale;
                                    if val > max_val {
                                        max_val = val;
                                    }
                                }
                            }

                            // 2. Exp & Sum
                            let mut sum_exp = 0.0;
                            for j in 0..k_len {
                                let is_masked = if is_causal && q_len > 1 { j > i } else { false };

                                if is_masked {
                                    row_out[j] = 0.0;
                                } else {
                                    // exp(x - max) 防止溢出
                                    let val = (row_in[j] * scale - max_val).exp();
                                    row_out[j] = val;
                                    sum_exp += val;
                                }
                            }

                            // 3. Normalize
                            let inv_sum = 1.0 / (sum_exp + 1e-10);
                            for j in 0..k_len {
                                row_out[j] *= inv_sum;
                            }
                        }
                    });
            });

        (out.clone().into_dyn(), out)
    };

    let input_clone = input.clone();

    // Backward: Softmax 的梯度推导
    // dL/dx_i = scale * sum_j ( dL/dy_j * dy_j/dx_i )
    // Softmax derivative: dy_j/dx_i = y_i * (delta_ij - y_j)
    // Combined: dX = scale * ( Y * (Grad - sum(Y * Grad, axis=-1)) )
    Tensor(Rc::new(RefCell::new(TensorData {
        data: output.into_shared(),
        grad: None,
        parents: vec![input.clone()],
        backward_op: Some(Box::new(move |grad| {
            let y = &output_data; // Softmax output

            // 1. dot = Y * Grad (Element-wise)
            let y_grad = y * grad;

            // 2. sum_dot = sum(Y * Grad) along last axis (K_Len)
            // insert_axis 保持维度 [B, H, Q, 1] 以便广播
            let sum_y_grad = y_grad.sum_axis(Axis(3)).insert_axis(Axis(3));

            // 3. dX = scale * Y * (Grad - sum_dot)
            let dx = y * (grad - &sum_y_grad) * scale;

            input_clone.add_grad(dx);
        })),
        requires_grad: true,
    })))
}
