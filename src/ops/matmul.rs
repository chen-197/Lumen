use crate::autograd::{Tensor, TensorData, is_no_grad};
use ndarray::{Array2, Array4, Ix2, Ix4, Zip};
use ndarray::linalg::general_mat_mul;
use rayon::prelude::*; // 用于并行优化
use std::cell::RefCell;
use std::rc::Rc;

// A[..., K] @ B^T, where B is [N(out), K(in)]
// output: [..., N]
pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor {
    let build_graph = !is_no_grad() && (a.requires_grad() || b.requires_grad());

    // 只读取 shape/len，不 clone 整个 data
    let (a_shape, b_shape, a_len) = {
        let ad = a.0.borrow();
        let bd = b.0.borrow();
        (
            ad.data.shape().to_vec(),
            bd.data.shape().to_vec(),
            ad.data.len(),
        )
    };

    if b_shape.len() != 2 {
        panic!("MatMul RHS must be 2D, got {:?}", b_shape);
    }

    // b: [N, K]
    let k_dim_a = a_shape[a_shape.len() - 1];
    let n_dim = b_shape[0];
    let k_dim_b = b_shape[1];

    if k_dim_a != k_dim_b {
        panic!(
            "MatMul shape mismatch: a {:?} (K={}) vs b {:?} (K={})",
            a_shape, k_dim_a, b_shape, k_dim_b
        );
    }

    let m_dim = a_len / k_dim_a;

    let res_2d = if m_dim == 1 {
        let ad = a.0.borrow();
        let bd = b.0.borrow();
        
        // A: [K] (视为向量)
        let a_vec = ad.data.as_slice().expect("A data should be contiguous for optimal perf");
        
        // B: [N, K]
        let b_view = bd.data.view().into_dimensionality::<Ix2>().unwrap();
        
        // 预分配结果 [1, N]
        let mut res = Array2::<f32>::zeros((1, n_dim));
        let res_slice = res.as_slice_mut().unwrap();

        // Rayon 并行遍历输出维度 N
        // 将计算任务均匀分配给所有 CPU 核心
        res_slice.par_iter_mut().enumerate().for_each(|(i, out_val)| {
            let row_b = b_view.row(i);
            
            // 手动点积 (编译器会自动 SIMD 优化)
            let mut sum = 0.0f32;
            for (&w, &x) in row_b.iter().zip(a_vec.iter()) {
                sum += w * x;
            }
            *out_val = sum;
        });
        
        res
    } else {
        // M > 1 (Prefill 阶段): 使用通用的 BLAS 加速 (general_mat_mul)
        // 此时计算量足够大，BLAS 效率通常高于手写并行
        let ad = a.0.borrow();
        let bd = b.0.borrow();

        let b_2d = bd.data.view().into_dimensionality::<Ix2>().unwrap(); // [N,K]
        let mut res = Array2::<f32>::zeros((m_dim, n_dim));

        // 尝试 View，失败则 Copy (处理非连续内存)
        if let Ok(a_2d_view) = ad.data.view().into_shape((m_dim, k_dim_a)) {
            general_mat_mul(1.0, &a_2d_view, &b_2d.t(), 0.0, &mut res);
        } else {
            let a_2d_owned = ad
                .data
                .to_owned()
                .into_shape((m_dim, k_dim_a))
                .expect("Reshape A failed");
            general_mat_mul(1.0, &a_2d_owned, &b_2d.t(), 0.0, &mut res);
        }

        res
    };

    // 恢复输出形状: [..., N]
    let mut out_shape = a_shape.clone();
    let last_idx = out_shape.len() - 1;
    out_shape[last_idx] = n_dim;

    let result = res_2d.into_shape(out_shape).unwrap().into_dyn();

    if !build_graph {
        return Tensor::from_array_no_grad(result);
    }

    let a_clone = a.clone();
    let b_clone = b.clone();

    Tensor(Rc::new(RefCell::new(TensorData {
        data: result.into_shared(),
        grad: None,
        parents: vec![a_clone.clone(), b_clone.clone()],
        requires_grad: true,
        backward_op: Some(Box::new(move |grad: &ndarray::ArrayViewD<f32>| {
            // grad: [..., N] -> [M,N]
            let g_len = grad.len();
            let g_m = g_len / n_dim;

            let grad_2d = grad
                .view()
                .into_shape((g_m, n_dim))
                .expect("Grad reshape failed: non-contiguous gradient?");

            // Backward 必须 clone 数据以避免 RefCell 借用冲突 (仅训练时触发)
            let (a_data, b_data) = {
                let ad = a_clone.0.borrow();
                let bd = b_clone.0.borrow();
                (ad.data.clone(), bd.data.clone())
            };

            // A -> [M,K]
            let a_2d_view = a_data.view().into_shape((m_dim, k_dim_a));
            let a_2d_owned;
            let a_2d = match a_2d_view {
                Ok(v) => v,
                Err(_) => {
                    a_2d_owned = a_data.to_owned().into_shape((m_dim, k_dim_a)).unwrap();
                    a_2d_owned.view()
                }
            };

            // B -> [N,K]
            let b_2d = b_data.view().into_dimensionality::<Ix2>().unwrap();

            // dA = dY @ B  -> [M,K]
            let mut da_2d = Array2::<f32>::zeros((m_dim, k_dim_a));
            general_mat_mul(1.0, &grad_2d, &b_2d, 0.0, &mut da_2d);
            a_clone.add_grad(da_2d.into_shape(a_data.shape()).unwrap().into_dyn());

            // dB = dY^T @ A -> [N,K]
            let mut db_2d = Array2::<f32>::zeros((n_dim, k_dim_a));
            general_mat_mul(1.0, &grad_2d.t(), &a_2d, 0.0, &mut db_2d);
            b_clone.add_grad(db_2d.into_dyn());
        })),
    })))
}

// lhs: [B, H, M, K]
// rhs: [B, H, K, N]
// out: [B, H, M, N]
pub fn batch_matmul(lhs: &Tensor, rhs: &Tensor) -> Tensor {
    let build_graph = !is_no_grad() && (lhs.requires_grad() || rhs.requires_grad());

    let lhs_ref = lhs.data_ref();
    let rhs_ref = rhs.data_ref();

    let lhs_view = lhs_ref.view().into_dimensionality::<Ix4>().unwrap();
    let rhs_view = rhs_ref.view().into_dimensionality::<Ix4>().unwrap();

    let (b, h, m, k) = lhs_view.dim();
    let (b2, h2, k2, n) = rhs_view.dim();

    assert_eq!(b, b2, "batch dim mismatch");
    assert_eq!(h, h2, "head dim mismatch");
    assert_eq!(k, k2, "k dim mismatch");

    let mut output = Array4::<f32>::zeros((b, h, m, n));

    // 使用串行 Zip 避免 RefCell 跨线程问题 (内部 BLAS 会处理并行)
    Zip::from(output.outer_iter_mut())
        .and(lhs_view.outer_iter())
        .and(rhs_view.outer_iter())
        .for_each(|mut out_batch, lhs_batch, rhs_batch| {
            Zip::from(out_batch.outer_iter_mut())
                .and(lhs_batch.outer_iter())
                .and(rhs_batch.outer_iter())
                .for_each(|mut out_mat, lhs_mat, rhs_mat| {
                    general_mat_mul(1.0, &lhs_mat, &rhs_mat, 0.0, &mut out_mat);
                });
        });

    let output_dyn = output.into_dyn();

    if !build_graph {
        return Tensor::from_array_no_grad(output_dyn);
    }

    let lhs_clone = lhs.clone();
    let rhs_clone = rhs.clone();

    Tensor(Rc::new(RefCell::new(TensorData {
        data: output_dyn.into_shared(),
        grad: None,
        parents: vec![lhs_clone.clone(), rhs_clone.clone()],
        backward_op: Some(Box::new(move |grad: &ndarray::ArrayViewD<f32>| {
            let grad_view = grad.view().into_dimensionality::<Ix4>().unwrap();
            let l_data = lhs_clone.0.borrow().data.clone();
            let r_data = rhs_clone.0.borrow().data.clone();

            let l_view_4d = l_data.view().into_dimensionality::<Ix4>().unwrap();
            let r_view_4d = r_data.view().into_dimensionality::<Ix4>().unwrap();

            let mut d_lhs = Array4::<f32>::zeros((b, h, m, k));
            Zip::from(d_lhs.outer_iter_mut())
                .and(grad_view.outer_iter())
                .and(r_view_4d.outer_iter())
                .for_each(|mut d_l_b, g_b, r_b| {
                    Zip::from(d_l_b.outer_iter_mut())
                        .and(g_b.outer_iter())
                        .and(r_b.outer_iter())
                        .for_each(|mut d_l_mat, g_mat, r_mat| {
                            general_mat_mul(1.0, &g_mat, &r_mat.t(), 0.0, &mut d_l_mat);
                        });
                });
            lhs_clone.add_grad(d_lhs.into_dyn());

            let mut d_rhs = Array4::<f32>::zeros((b, h, k, n));
            Zip::from(d_rhs.outer_iter_mut())
                .and(l_view_4d.outer_iter())
                .and(grad_view.outer_iter())
                .for_each(|mut d_r_b, l_b, g_b| {
                    Zip::from(d_r_b.outer_iter_mut())
                        .and(l_b.outer_iter())
                        .and(g_b.outer_iter())
                        .for_each(|mut d_r_mat, l_mat, g_mat| {
                            general_mat_mul(1.0, &l_mat.t(), &g_mat, 0.0, &mut d_r_mat);
                        });
                });
            rhs_clone.add_grad(d_rhs.into_dyn());
        })),
        requires_grad: true,
    })))
}