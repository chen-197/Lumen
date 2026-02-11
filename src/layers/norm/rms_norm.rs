use crate::autograd::{Tensor, TensorData};
use crate::module::Module;
use ndarray::{Array1, Array2, Zip};
use rayon::prelude::*;
use std::cell::RefCell;
use std::rc::Rc;

pub struct RMSNorm {
    pub weight: Tensor, // Gamma 参数
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
}

impl Module for RMSNorm {
    fn forward(&self, input: Tensor) -> Tensor {
        let (output_data, rows, dim, shape) = {
            let input_ref = input.data_ref();
            let x = &*input_ref;

            let shape = x.shape().to_vec(); // clone shape to return it out of block
            let dim = shape[shape.len() - 1];
            let rows = x.len() / dim;

            let x_cow = x.as_standard_layout();
            let x_2d = x_cow.view().into_shape((rows, dim)).unwrap();

            let w_ref = self.weight.data_ref();
            let w_1d = w_ref
                .as_slice()
                .expect("RMSNorm weight should be contiguous");

            let mut output_flat = Array2::<f32>::zeros((rows, dim));
            let eps = self.eps;

            // Forward 并行计算 (Fused Kernel)
            Zip::from(output_flat.outer_iter_mut())
                .and(x_2d.outer_iter())
                .par_for_each(|mut out_row, x_row| {
                    // SumSq
                    let sum_sq = x_row.fold(0.0f32, |acc, &val| acc + val * val);
                    // RMS & InvRMS
                    let rms = (sum_sq / dim as f32 + eps).sqrt();
                    let inv_rms = 1.0 / rms;

                    // Normalize & Scale
                    for (o, (&xi, &wi)) in out_row.iter_mut().zip(x_row.iter().zip(w_1d)) {
                        *o = xi * inv_rms * wi;
                    }
                });

            let out_d = output_flat.into_shape(shape.clone()).unwrap().into_dyn();
            (out_d, rows, dim, shape)
        };

        let input_clone = input.clone();
        let weight_clone = self.weight.clone();
        let eps = self.eps; // copy scalar

        Tensor(Rc::new(RefCell::new(TensorData {
            data: output_data.into_shared(),
            grad: None,
            parents: vec![input.clone(), self.weight.clone()],
            backward_op: Some(Box::new(move |grad_output| {
                // Backward 逻辑
                let x_ref = input_clone.data_ref();
                let x_cow = x_ref.as_standard_layout();
                let x_2d = x_cow.view().into_shape((rows, dim)).unwrap();

                let w_ref = weight_clone.data_ref();
                let w_slice = w_ref.as_slice().unwrap();

                let g_cow = grad_output.as_standard_layout();
                let g_2d = g_cow.view().into_shape((rows, dim)).unwrap();

                //计算输入梯度 dL/dx
                let mut d_input_flat = Array2::<f32>::zeros((rows, dim));

                //计算 dInput (并行)
                Zip::from(d_input_flat.outer_iter_mut())
                    .and(x_2d.outer_iter())
                    .and(g_2d.outer_iter())
                    .par_for_each(|mut dx_row, x_row, g_row| {
                        // Recompute RMS
                        let sum_sq = x_row.fold(0.0f32, |acc, &val| acc + val * val);
                        let inv_rms = 1.0 / (sum_sq / dim as f32 + eps).sqrt();
                        let inv_dim = 1.0 / dim as f32;

                        // dot = sum(g * w * x_norm)
                        let mut dot = 0.0f32;
                        for (&gi, (&wi, &xi)) in g_row.iter().zip(w_slice.iter().zip(x_row.iter()))
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

                input_clone.add_grad(d_input_flat.into_shape(shape.clone()).unwrap().into_dyn());

                //计算权重梯度 dL/dw (并行归约)
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

                weight_clone.add_grad(dw_accum.into_dyn());
            })),
            requires_grad: true,
        })))
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.weight.clone()]
    }
}
