use crate::autograd::{Tensor, TensorData};
use ndarray::{arr0, Array2, Zip};
use std::cell::RefCell;
use std::rc::Rc;
use rayon::prelude::*; // 引入并行迭代

// --- MSE Loss ---
pub struct MSELoss;
impl MSELoss {
    pub fn apply(output: &Tensor, target: &Tensor) -> Tensor {
        let loss_val = {
            let out_ref = output.data_ref();
            let tar_ref = target.data_ref();
            // Parallel Mean Squared Error
            // 避免创建 huge diff array，直接 reduce
            let n = out_ref.len() as f32;
            let sum_sq: f32 = Zip::from(&*out_ref)
                .and(&*tar_ref)
                .par_map_collect(|&o, &t| (o - t).powi(2)) // Map
                .sum(); // Reduce
            
            sum_sq / n
        };

        let output_clone = output.clone();
        let target_clone = target.clone();

        Tensor(Rc::new(RefCell::new(TensorData {
            data: arr0(loss_val).into_dyn().into_shared(),
            grad: None,
            parents: vec![output.clone(), target.clone()],
            backward_op: Some(Box::new(move |grad_output| {
                let grad_val = grad_output.first().unwrap();
                let out_d = output_clone.data_ref();
                let tar_d = target_clone.data_ref();
                let n = out_d.len() as f32;
                let factor = 2.0 / n * grad_val;

                // dL/dx = 2/N * (x - y) * g
                // 直接并行生成梯度，不创建 diff 中间量
                let grad = Zip::from(&*out_d)
                    .and(&*tar_d)
                    .par_map_collect(|&o, &t| (o - t) * factor);

                output_clone.add_grad(grad.clone());
                target_clone.add_grad(grad.mapv(|x| -x));
            })),
            requires_grad: true
        })))
    }
}

// --- Cross Entropy Loss ---
// 针对 Batch 进行行级并行优化
pub struct CrossEntropyLoss;

impl CrossEntropyLoss {
    pub fn apply(input_logits: &Tensor, target_onehot: &Tensor) -> Tensor {
        // Forward
        let (loss_val, softmax_output) = {
            let logits_ref = input_logits.data_ref();
            let targets_ref = target_onehot.data_ref();
            
            let batch_size = logits_ref.shape()[0];
            let dim = logits_ref.shape()[1];

            // 1. Reshape to 2D [Batch, Classes] (View)
            let logits_2d = logits_ref.view().into_shape((batch_size, dim)).unwrap();
            let targets_2d = targets_ref.view().into_shape((batch_size, dim)).unwrap();

            // 2. Prepare Output container
            // 分配内存存 Softmax 结果供 Backward 使用
            let mut softmax_out_flat = Array2::<f32>::zeros((batch_size, dim));
            
            // 3. Parallel Compute: Softmax + CrossEntropy Sum
            // Zip 遍历每一行
            let total_loss: f32 = Zip::from(softmax_out_flat.outer_iter_mut())
                .and(logits_2d.outer_iter())
                .and(targets_2d.outer_iter())
                .into_par_iter() // Rayon 并行
                .map(|(mut sm_row, l_row, t_row)| {
                    // --- Row-wise Softmax ---
                    let max_val = l_row.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                    let mut sum_exp = 0.0f32;
                    
                    // Exp & Sum
                    for (s_val, &l_val) in sm_row.iter_mut().zip(l_row.iter()) {
                        let e = (l_val - max_val).exp();
                        *s_val = e;
                        sum_exp += e;
                    }
                    
                    // Div & Local Loss Calculation
                    let inv_sum = 1.0 / sum_exp;
                    let epsilon = 1e-9;
                    let mut row_loss = 0.0;
                    
                    for (s_val, &t_val) in sm_row.iter_mut().zip(t_row.iter()) {
                        *s_val *= inv_sum; // Final Prob
                        // Loss: -t * log(p)
                        if t_val > 0.0 { // Optimization: target 通常是 sparse 的 (One-hot)
                             row_loss -= t_val * (*s_val + epsilon).ln();
                        }
                    }
                    row_loss
                })
                .sum(); // Parallel Reduction of loss

            (total_loss / batch_size as f32, softmax_out_flat.into_dyn())
        };

        let input_clone = input_logits.clone();
        let target_clone = target_onehot.clone();
        let softmax_cache = softmax_output;

        Tensor(Rc::new(RefCell::new(TensorData {
            data: arr0(loss_val).into_dyn().into_shared(),
            grad: None,
            parents: vec![input_logits.clone(), target_onehot.clone()],
            backward_op: Some(Box::new(move |grad_output| {
                let grad_val = grad_output.first().unwrap();
                let targets_ref = target_clone.data_ref();
                let batch_size = targets_ref.shape()[0] as f32;
                let factor = grad_val / batch_size;

                // Backward: (Prob - Target) / N * grad
                // Zip 并行计算
                let grad = Zip::from(&softmax_cache)
                    .and(&*targets_ref)
                    .par_map_collect(|&p, &t| (p - t) * factor);

                input_clone.add_grad(grad);
            })),
            requires_grad: true
        })))
    }
}