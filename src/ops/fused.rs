use crate::autograd::{is_no_grad, Tensor, TensorData};
use crate::ops::matmul::{
    dual_matvec_rowmajor_parallel,
    dual_matvec_rowmajor_parallel_bf16,
    dual_matvec_silu_mul_rowmajor_parallel,
    dual_matvec_silu_mul_rowmajor_parallel_bf16,
    matvec_rowmajor_parallel,
    matvec_rowmajor_parallel_bf16,
};
use ndarray::{Array, Array2, Array3, Axis, Ix2, Zip};
use rayon::prelude::*;
use std::cell::RefCell;
use std::rc::Rc;

pub fn fused_softmax(input: &Tensor, scale: f32, is_causal: bool) -> Tensor {
    let (output, output_data) = {
        let x = input.data_ref();
        let shape = x.shape();
        if shape.len() != 4 {
            panic!("Fused Softmax expects 4D input [B, H, Q, K]");
        }

        let q_len = shape[2];
        let k_len = shape[3];

        let mut out = Array::zeros(x.dim());
        let x_view = x.view().into_dimensionality::<ndarray::Ix4>().unwrap();
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

                            let mut sum_exp = 0.0;
                            for j in 0..k_len {
                                let is_masked = if is_causal && q_len > 1 { j > i } else { false };
                                if is_masked {
                                    row_out[j] = 0.0;
                                } else {
                                    let val = (row_in[j] * scale - max_val).exp();
                                    row_out[j] = val;
                                    sum_exp += val;
                                }
                            }

                            let inv_sum = 1.0 / (sum_exp + 1e-10);
                            for j in 0..k_len {
                                row_out[j] *= inv_sum;
                            }
                        }
                    });
            });

        (out.clone().into_dyn(), out)
    };
    let build_graph = !is_no_grad() && input.requires_grad();

    if !build_graph {
        return Tensor::from_array_no_grad(output);
    }

    let input_clone = input.clone();
    Tensor(Rc::new(RefCell::new(TensorData {
        data: output.into_shared(),
        grad: None,
        parents: vec![input.clone()],
        backward_op: Some(std::rc::Rc::new(move |grad| {
            let y = &output_data;
            let y_grad = y * grad;
            let sum_y_grad = y_grad.sum_axis(Axis(3)).insert_axis(Axis(3));
            let dx = y * (grad - &sum_y_grad) * scale;
            input_clone.add_grad(dx);
        })),
        requires_grad: true,
    })))
}

pub fn fused_gate_up_silu_infer(input: &Tensor, gate_weight: &Tensor, up_weight: &Tensor) -> Tensor {
    assert!(is_no_grad(), "fused_gate_up_silu_infer is inference-only");

    let x_data = input.data_arc();
    let gate_data = gate_weight.data_arc();
    let up_data = up_weight.data_arc();

    let x_shape = x_data.shape().to_vec();
    let k_dim = *x_shape.last().expect("input must have last dim");
    let m_dim = x_data.len() / k_dim;

    let gate_2d = gate_data
        .view()
        .into_dimensionality::<Ix2>()
        .expect("gate weight must be 2D [N, K]");
    let up_2d = up_data
        .view()
        .into_dimensionality::<Ix2>()
        .expect("up weight must be 2D [N, K]");

    let n_dim = gate_2d.nrows();
    assert_eq!(gate_2d.ncols(), k_dim, "gate weight K mismatch");
    assert_eq!(up_2d.nrows(), n_dim, "gate/up out dim mismatch");
    assert_eq!(up_2d.ncols(), k_dim, "up weight K mismatch");

    let x_2d = x_data
        .view()
        .into_shape((m_dim, k_dim))
        .expect("input reshape failed");

    let mut out = Array2::<f32>::zeros((m_dim, n_dim));

    let gate_owned;
    let gate_slice: &[f32] = if let Some(s) = gate_2d.as_slice() {
        s
    } else {
        gate_owned = gate_2d.iter().copied().collect::<Vec<f32>>();
        gate_owned.as_slice()
    };
    let up_owned;
    let up_slice: &[f32] = if let Some(s) = up_2d.as_slice() {
        s
    } else {
        up_owned = up_2d.iter().copied().collect::<Vec<f32>>();
        up_owned.as_slice()
    };

    Zip::from(out.outer_iter_mut())
        .and(x_2d.outer_iter())
        .par_for_each(|mut out_row, x_row| {
            let x_owned;
            let x_slice: &[f32] = if let Some(s) = x_row.as_slice() {
                s
            } else {
                x_owned = x_row.iter().copied().collect::<Vec<f32>>();
                x_owned.as_slice()
            };
            let out_slice = out_row.as_slice_mut().expect("output row should be contiguous");
            dual_matvec_silu_mul_rowmajor_parallel(x_slice, gate_slice, up_slice, n_dim, k_dim, out_slice);
        });

    let mut out_shape = x_shape;
    let last = out_shape.len() - 1;
    out_shape[last] = n_dim;
    Tensor::from_array_no_grad(out.into_shape(out_shape).unwrap().into_dyn())
}





pub fn fused_gate_up_silu_infer_into(
    input: &Tensor,
    gate_weight: &Tensor,
    up_weight: &Tensor,
    out: &mut [f32],
) {
    assert!(is_no_grad(), "fused_gate_up_silu_infer_into is inference-only");

    let x_data = input.data_arc();
    let gate_data = gate_weight.data_arc();
    let up_data = up_weight.data_arc();

    let x_shape = x_data.shape().to_vec();
    let k_dim = *x_shape.last().expect("input must have last dim");
    let m_dim = x_data.len() / k_dim;
    assert_eq!(m_dim, 1, "fused_gate_up_silu_infer_into currently expects single-token decode input");

    let gate_2d = gate_data
        .view()
        .into_dimensionality::<Ix2>()
        .expect("gate weight must be 2D [N, K]");
    let up_2d = up_data
        .view()
        .into_dimensionality::<Ix2>()
        .expect("up weight must be 2D [N, K]");

    let n_dim = gate_2d.nrows();
    assert_eq!(out.len(), n_dim, "output size mismatch");
    assert_eq!(gate_2d.ncols(), k_dim, "gate weight K mismatch");
    assert_eq!(up_2d.nrows(), n_dim, "gate/up out dim mismatch");
    assert_eq!(up_2d.ncols(), k_dim, "up weight K mismatch");

    let x_2d = x_data
        .view()
        .into_shape((m_dim, k_dim))
        .expect("input reshape failed");
    let x_row = x_2d.row(0);

    let x_owned;
    let x_slice: &[f32] = if let Some(s) = x_row.as_slice() {
        s
    } else {
        x_owned = x_row.iter().copied().collect::<Vec<f32>>();
        x_owned.as_slice()
    };

    let gate_owned;
    let gate_slice: &[f32] = if let Some(s) = gate_2d.as_slice() {
        s
    } else {
        gate_owned = gate_2d.iter().copied().collect::<Vec<f32>>();
        gate_owned.as_slice()
    };
    let up_owned;
    let up_slice: &[f32] = if let Some(s) = up_2d.as_slice() {
        s
    } else {
        up_owned = up_2d.iter().copied().collect::<Vec<f32>>();
        up_owned.as_slice()
    };

    dual_matvec_silu_mul_rowmajor_parallel(x_slice, gate_slice, up_slice, n_dim, k_dim, out);
}

pub fn fused_qkv_decode_infer_into(
    input: &Tensor,
    q_weight: &Tensor,
    k_weight: &Tensor,
    v_weight: &Tensor,
    q_out: &mut [f32],
    k_out: &mut [f32],
    v_out: &mut [f32],
) {
    assert!(is_no_grad(), "fused_qkv_decode_infer_into is inference-only");

    let x_data = input.data_arc();
    let q_data = q_weight.data_arc();
    let k_data = k_weight.data_arc();
    let v_data = v_weight.data_arc();

    let x_shape = x_data.shape().to_vec();
    assert_eq!(x_shape.len(), 3, "decode input must be [B, S, K]");
    let (b, s, k_dim) = (x_shape[0], x_shape[1], x_shape[2]);
    assert_eq!(b, 1, "fused_qkv_decode_infer_into currently expects batch size 1");
    assert_eq!(s, 1, "fused_qkv_decode_infer_into only supports S=1 decode");

    let q_2d = q_data
        .view()
        .into_dimensionality::<Ix2>()
        .expect("Q weight must be 2D [Nq, K]");
    let k_2d = k_data
        .view()
        .into_dimensionality::<Ix2>()
        .expect("K weight must be 2D [Nk, K]");
    let v_2d = v_data
        .view()
        .into_dimensionality::<Ix2>()
        .expect("V weight must be 2D [Nv, K]");

    let q_n = q_2d.nrows();
    let k_n = k_2d.nrows();
    let v_n = v_2d.nrows();

    assert_eq!(q_out.len(), q_n, "Q output size mismatch");
    assert_eq!(k_out.len(), k_n, "K output size mismatch");
    assert_eq!(v_out.len(), v_n, "V output size mismatch");
    assert_eq!(q_2d.ncols(), k_dim, "Q weight K mismatch");
    assert_eq!(k_2d.ncols(), k_dim, "K weight K mismatch");
    assert_eq!(v_2d.ncols(), k_dim, "V weight K mismatch");
    assert_eq!(v_n, k_n, "K/V dim mismatch");

    let x_2d = x_data
        .view()
        .into_shape((b, k_dim))
        .expect("decode input reshape failed");
    let x_row = x_2d.row(0);
    let x_owned;
    let x_slice: &[f32] = if let Some(slc) = x_row.as_slice() {
        slc
    } else {
        x_owned = x_row.iter().copied().collect::<Vec<f32>>();
        x_owned.as_slice()
    };

    let q_owned;
    let q_slice: &[f32] = if let Some(slc) = q_2d.as_slice() {
        slc
    } else {
        q_owned = q_2d.iter().copied().collect::<Vec<f32>>();
        q_owned.as_slice()
    };
    let k_owned;
    let k_slice: &[f32] = if let Some(slc) = k_2d.as_slice() {
        slc
    } else {
        k_owned = k_2d.iter().copied().collect::<Vec<f32>>();
        k_owned.as_slice()
    };
    let v_owned;
    let v_slice: &[f32] = if let Some(slc) = v_2d.as_slice() {
        slc
    } else {
        v_owned = v_2d.iter().copied().collect::<Vec<f32>>();
        v_owned.as_slice()
    };

    matvec_rowmajor_parallel(x_slice, q_slice, q_n, k_dim, q_out);
    dual_matvec_rowmajor_parallel(x_slice, k_slice, v_slice, k_n, k_dim, k_out, v_out);
}
pub fn fused_qkv_decode_infer(
    input: &Tensor,
    q_weight: &Tensor,
    k_weight: &Tensor,
    v_weight: &Tensor,
    n_head: usize,
    n_kv_head: usize,
) -> (Array3<f32>, Array3<f32>, Array3<f32>) {
    assert!(is_no_grad(), "fused_qkv_decode_infer is inference-only");

    let x_data = input.data_arc();
    let q_data = q_weight.data_arc();
    let k_data = k_weight.data_arc();
    let v_data = v_weight.data_arc();

    let x_shape = x_data.shape().to_vec();
    assert_eq!(x_shape.len(), 3, "decode input must be [B, S, K]");
    let (b, s, k_dim) = (x_shape[0], x_shape[1], x_shape[2]);
    assert_eq!(s, 1, "fused_qkv_decode_infer only supports S=1 decode");

    let q_2d = q_data
        .view()
        .into_dimensionality::<Ix2>()
        .expect("Q weight must be 2D [Nq, K]");
    let k_2d = k_data
        .view()
        .into_dimensionality::<Ix2>()
        .expect("K weight must be 2D [Nk, K]");
    let v_2d = v_data
        .view()
        .into_dimensionality::<Ix2>()
        .expect("V weight must be 2D [Nv, K]");

    let q_n = q_2d.nrows();
    let k_n = k_2d.nrows();
    let v_n = v_2d.nrows();

    assert_eq!(q_2d.ncols(), k_dim, "Q weight K mismatch");
    assert_eq!(k_2d.ncols(), k_dim, "K weight K mismatch");
    assert_eq!(v_2d.ncols(), k_dim, "V weight K mismatch");
    assert_eq!(q_n % n_head, 0, "Q dim must be divisible by n_head");
    assert_eq!(k_n % n_kv_head, 0, "K dim must be divisible by n_kv_head");
    assert_eq!(v_n, k_n, "K/V dim mismatch");

    let d = q_n / n_head;
    assert_eq!(k_n / n_kv_head, d, "Q/K head dim mismatch");

    let x_2d = x_data
        .view()
        .into_shape((b, k_dim))
        .expect("decode input reshape failed");

    let mut q_out = Array2::<f32>::zeros((b, q_n));
    let mut k_out = Array2::<f32>::zeros((b, k_n));
    let mut v_out = Array2::<f32>::zeros((b, v_n));

    let q_owned;
    let q_slice: &[f32] = if let Some(s) = q_2d.as_slice() {
        s
    } else {
        q_owned = q_2d.iter().copied().collect::<Vec<f32>>();
        q_owned.as_slice()
    };
    let k_owned;
    let k_slice: &[f32] = if let Some(s) = k_2d.as_slice() {
        s
    } else {
        k_owned = k_2d.iter().copied().collect::<Vec<f32>>();
        k_owned.as_slice()
    };
    let v_owned;
    let v_slice: &[f32] = if let Some(s) = v_2d.as_slice() {
        s
    } else {
        v_owned = v_2d.iter().copied().collect::<Vec<f32>>();
        v_owned.as_slice()
    };

    for bb in 0..b {
        let x_row = x_2d.row(bb);
        let x_owned;
        let x_slice: &[f32] = if let Some(s) = x_row.as_slice() {
            s
        } else {
            x_owned = x_row.iter().copied().collect::<Vec<f32>>();
            x_owned.as_slice()
        };

        {
            let mut q_row_view = q_out.slice_mut(ndarray::s![bb, ..]);
            let q_row = q_row_view
                .as_slice_mut()
                .expect("Q output row not contiguous");
            matvec_rowmajor_parallel(x_slice, q_slice, q_n, k_dim, q_row);
        }

        {
            let mut k_row_view = k_out.slice_mut(ndarray::s![bb, ..]);
            let k_row = k_row_view
                .as_slice_mut()
                .expect("K output row not contiguous");
            let mut v_row_view = v_out.slice_mut(ndarray::s![bb, ..]);
            let v_row = v_row_view
                .as_slice_mut()
                .expect("V output row not contiguous");
            dual_matvec_rowmajor_parallel(x_slice, k_slice, v_slice, k_n, k_dim, k_row, v_row);
        }
    }

    (
        q_out
            .into_shape((b, n_head, d))
            .expect("Q output reshape failed"),
        k_out
            .into_shape((b, n_kv_head, d))
            .expect("K output reshape failed"),
        v_out
            .into_shape((b, n_kv_head, d))
            .expect("V output reshape failed"),
    )
}


// 临时：推理压缩权重路径，后续由全链路 dtype 替代。
pub fn fused_gate_up_silu_infer_tmp_f32_into(
    input: &Tensor,
    gate_weight: &[f32],
    up_weight: &[f32],
    gate_n: usize,
    k_dim: usize,
    out: &mut [f32],
) {
    assert!(is_no_grad(), "fused_gate_up_silu_infer_tmp_f32_into is inference-only");

    let x_data = input.data_arc();
    let x_shape = x_data.shape().to_vec();
    let x_k_dim = *x_shape.last().expect("input must have last dim");
    let m_dim = x_data.len() / x_k_dim;
    assert_eq!(m_dim, 1, "fused_gate_up_silu_infer_tmp_f32_into currently expects single-token decode input");
    assert_eq!(x_k_dim, k_dim, "input width mismatch");
    assert_eq!(out.len(), gate_n, "output size mismatch");

    let x_2d = x_data.view().into_shape((m_dim, k_dim)).expect("input reshape failed");
    let x_row = x_2d.row(0);
    let x_owned;
    let x_slice: &[f32] = if let Some(s) = x_row.as_slice() {
        s
    } else {
        x_owned = x_row.iter().copied().collect::<Vec<f32>>();
        x_owned.as_slice()
    };

    dual_matvec_silu_mul_rowmajor_parallel(x_slice, gate_weight, up_weight, gate_n, k_dim, out);
}

// 临时：推理压缩权重路径，后续由全链路 dtype 替代。
pub fn fused_gate_up_silu_infer_tmp_bf16_into(
    input: &Tensor,
    gate_weight: &[half::bf16],
    up_weight: &[half::bf16],
    gate_n: usize,
    k_dim: usize,
    out: &mut [f32],
) {
    assert!(is_no_grad(), "fused_gate_up_silu_infer_tmp_bf16_into is inference-only");

    let x_data = input.data_arc();
    let x_shape = x_data.shape().to_vec();
    let x_k_dim = *x_shape.last().expect("input must have last dim");
    let m_dim = x_data.len() / x_k_dim;
    assert_eq!(m_dim, 1, "fused_gate_up_silu_infer_tmp_bf16_into currently expects single-token decode input");
    assert_eq!(x_k_dim, k_dim, "input width mismatch");
    assert_eq!(out.len(), gate_n, "output size mismatch");

    let x_2d = x_data.view().into_shape((m_dim, k_dim)).expect("input reshape failed");
    let x_row = x_2d.row(0);
    let x_owned;
    let x_slice: &[f32] = if let Some(s) = x_row.as_slice() {
        s
    } else {
        x_owned = x_row.iter().copied().collect::<Vec<f32>>();
        x_owned.as_slice()
    };

    dual_matvec_silu_mul_rowmajor_parallel_bf16(x_slice, gate_weight, up_weight, gate_n, k_dim, out);
}

// 临时：推理压缩权重路径，后续由全链路 dtype 替代。
pub fn fused_qkv_decode_infer_tmp_f32_into(
    input: &Tensor,
    q_weight: &[f32],
    k_weight: &[f32],
    v_weight: &[f32],
    q_n: usize,
    k_n: usize,
    k_dim: usize,
    q_out: &mut [f32],
    k_out: &mut [f32],
    v_out: &mut [f32],
) {
    assert!(is_no_grad(), "fused_qkv_decode_infer_tmp_f32_into is inference-only");

    let x_data = input.data_arc();
    let x_shape = x_data.shape().to_vec();
    assert_eq!(x_shape.len(), 3, "decode input must be [B, S, K]");
    let (b, s, x_k_dim) = (x_shape[0], x_shape[1], x_shape[2]);
    assert_eq!(b, 1, "fused_qkv_decode_infer_tmp_f32_into currently expects batch size 1");
    assert_eq!(s, 1, "fused_qkv_decode_infer_tmp_f32_into only supports S=1 decode");
    assert_eq!(x_k_dim, k_dim, "input width mismatch");
    assert_eq!(q_out.len(), q_n, "Q output size mismatch");
    assert_eq!(k_out.len(), k_n, "K output size mismatch");
    assert_eq!(v_out.len(), k_n, "V output size mismatch");

    let x_2d = x_data.view().into_shape((b, k_dim)).expect("decode input reshape failed");
    let x_row = x_2d.row(0);
    let x_owned;
    let x_slice: &[f32] = if let Some(slc) = x_row.as_slice() {
        slc
    } else {
        x_owned = x_row.iter().copied().collect::<Vec<f32>>();
        x_owned.as_slice()
    };

    matvec_rowmajor_parallel(x_slice, q_weight, q_n, k_dim, q_out);
    dual_matvec_rowmajor_parallel(x_slice, k_weight, v_weight, k_n, k_dim, k_out, v_out);
}

// 临时：推理压缩权重路径，后续由全链路 dtype 替代。
pub fn fused_qkv_decode_infer_tmp_bf16_into(
    input: &Tensor,
    q_weight: &[half::bf16],
    k_weight: &[half::bf16],
    v_weight: &[half::bf16],
    q_n: usize,
    k_n: usize,
    k_dim: usize,
    q_out: &mut [f32],
    k_out: &mut [f32],
    v_out: &mut [f32],
) {
    assert!(is_no_grad(), "fused_qkv_decode_infer_tmp_bf16_into is inference-only");

    let x_data = input.data_arc();
    let x_shape = x_data.shape().to_vec();
    assert_eq!(x_shape.len(), 3, "decode input must be [B, S, K]");
    let (b, s, x_k_dim) = (x_shape[0], x_shape[1], x_shape[2]);
    assert_eq!(b, 1, "fused_qkv_decode_infer_tmp_bf16_into currently expects batch size 1");
    assert_eq!(s, 1, "fused_qkv_decode_infer_tmp_bf16_into only supports S=1 decode");
    assert_eq!(x_k_dim, k_dim, "input width mismatch");
    assert_eq!(q_out.len(), q_n, "Q output size mismatch");
    assert_eq!(k_out.len(), k_n, "K output size mismatch");
    assert_eq!(v_out.len(), k_n, "V output size mismatch");

    let x_2d = x_data.view().into_shape((b, k_dim)).expect("decode input reshape failed");
    let x_row = x_2d.row(0);
    let x_owned;
    let x_slice: &[f32] = if let Some(slc) = x_row.as_slice() {
        slc
    } else {
        x_owned = x_row.iter().copied().collect::<Vec<f32>>();
        x_owned.as_slice()
    };

    matvec_rowmajor_parallel_bf16(x_slice, q_weight, q_n, k_dim, q_out);
    dual_matvec_rowmajor_parallel_bf16(x_slice, k_weight, v_weight, k_n, k_dim, k_out, v_out);
}

