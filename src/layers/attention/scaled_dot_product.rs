//! Attention over already projected Q/K/V, independent of positional encoding.
use crate::autograd::{Tensor, assert_same_device, is_no_grad};
use crate::ops::{fused::fused_softmax, matmul::batch_matmul};
use ndarray::{ArrayD, IxDyn};
use std::cell::RefCell;
use std::rc::Rc;

/// Masks broadcast from the right to `[B,H,Q,K]`.
pub enum AttentionMask<'a> {
    /// `true` permits attending to a key. `[B,1,1,K]` is a padding mask.
    Keep(&'a ArrayD<bool>),
    /// Added after score scaling. Use finite values or `-inf`; supports gradients.
    Additive(&'a Tensor),
}

#[derive(Clone, Copy, Debug, Default)]
pub struct AttentionOptions {
    /// Defaults to `1 / sqrt(query_head_dim)`.
    pub scale: Option<f32>,
    pub causal: bool,
    /// Query `i` can see keys `j <= query_offset + i` when causal.
    /// The default is upper-left alignment, including rectangular attention.
    pub query_offset: usize,
}

/// Differentiable CPU/CUDA attention. Q=`[B,Hq,Q,D]`, K=`[B,Hkv,K,D]`,
/// V=`[B,Hkv,K,Dv]`; supports MHA/GQA/MQA and cross-attention.
///
/// No projection, positional encoding, dropout, or cache mutation is performed.
/// Fully masked rows produce zero context and zero score gradients. F32 CPU
/// inference tiles query rows, bounds score scratch and views grouped K/V heads.
/// Other precision/device/gradient routes retain the composed implementation.
/// Operations tracked by autograd produce F32 outputs. Operations on entirely
/// frozen inputs follow inference dtype rules: same-dtype lowp matmuls preserve
/// their dtype; masked arithmetic promotes scores/probabilities to F32. Choose
/// F32 Q/K/V for an all-F32 path, including partially frozen computations.
pub fn scaled_dot_product_attention(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    mask: Option<AttentionMask<'_>>,
    options: AttentionOptions,
) -> Tensor {
    let device = assert_same_device(query, key, "attention Q/K");
    assert_same_device(query, value, "attention Q/V");
    let q = query.shape_vec();
    let k = key.shape_vec();
    let v = value.shape_vec();
    assert!(
        q.len() == 4 && k.len() == 4 && v.len() == 4,
        "attention expects 4D Q/K/V"
    );
    assert!(
        q[1..].iter().all(|&d| d > 0) && k[1..].iter().all(|&d| d > 0) && v[3] > 0,
        "attention head, sequence and feature dimensions must be positive"
    );
    assert_eq!(q[0], k[0], "attention batch mismatch");
    assert_eq!(k[..3], v[..3], "attention K/V shape mismatch");
    assert_eq!(q[3], k[3], "attention Q/K feature mismatch");
    assert_eq!(
        q[1] % k[1],
        0,
        "attention Q heads must be divisible by KV heads"
    );
    assert!(
        query.dtype().is_float() && key.dtype().is_float() && value.dtype().is_float(),
        "attention expects floating Q/K/V"
    );
    let scale = options.scale.unwrap_or((q[3] as f32).sqrt().recip());
    assert!(scale.is_finite(), "attention scale must be finite");
    options
        .query_offset
        .checked_add(q[2] - 1)
        .expect("attention query offset overflow");
    let target = [q[0], q[1], q[2], k[2]];
    if let Some(mask) = &mask {
        match mask {
            AttentionMask::Keep(keep) => check_mask_shape(keep.shape(), &target),
            AttentionMask::Additive(additive) => {
                check_mask_shape(&additive.shape_vec(), &target);
                assert_same_device(query, additive, "attention mask");
                assert!(
                    additive.dtype().is_float(),
                    "attention additive mask must be floating point"
                );
            }
        }
    }
    // Empty CUDA tensors intentionally own no device allocation. No numerical
    // kernel should request a resident buffer for this valid empty-batch case.
    if q[0] == 0 {
        let mut parents = vec![query.clone(), key.clone(), value.clone()];
        if let Some(AttentionMask::Additive(bias)) = &mask {
            parents.push((*bias).clone());
        }
        let build_graph = !is_no_grad() && parents.iter().any(Tensor::requires_grad);
        let shape = [0, q[1], q[2], v[3]];
        let data = ArrayD::zeros(IxDyn(&shape));
        if !build_graph {
            let preserves_lowp = mask.is_none()
                && (!options.causal || (q[2] == k[2] && options.query_offset == 0))
                && query.dtype() == key.dtype()
                && key.dtype() == value.dtype();
            let dtype = if preserves_lowp {
                query.dtype()
            } else {
                crate::precision::DType::F32
            };
            return Tensor::from_f32_data_no_grad_with_device_dtype(data, dtype, device);
        }
        let backward_parents = parents.clone();
        let backward = Rc::new(move |_: &ndarray::ArrayViewD<f32>| {
            for parent in &backward_parents {
                parent.add_backward_grad(ArrayD::zeros(IxDyn(&parent.shape_vec())));
            }
        });
        let mut data = Tensor::op_data(data.into_shared(), parents, Some(backward), true);
        data.device = device;
        return Tensor(Rc::new(RefCell::new(data)));
    }
    if let Some(output) = super::cpu_infer::try_attention(query, key, value, mask.as_ref(), options)
    {
        return output;
    }
    if mask.is_none() && (!options.causal || (q[2] == k[2] && options.query_offset == 0)) {
        return unmasked_attention(query, key, value, scale, options.causal);
    }
    let n_rep = q[1] / k[1];
    let key = if n_rep == 1 {
        key.clone()
    } else {
        super::self_attention::repeat_kv(key.clone(), n_rep)
    };
    let value = if n_rep == 1 {
        value.clone()
    } else {
        super::self_attention::repeat_kv(value.clone(), n_rep)
    };
    let scores = batch_matmul(query, &key.transpose(2, 3));
    let probabilities = {
        let scalar =
            Tensor::from_array_no_grad(ArrayD::from_elem(IxDyn(&[]), scale)).to_device(device);
        let mut masked = &scores * &scalar;
        if let Some(mask) = mask {
            match mask {
                AttentionMask::Keep(keep) => {
                    let additive = keep.mapv(|yes| if yes { 0.0 } else { f32::NEG_INFINITY });
                    masked = masked + Tensor::from_array_no_grad(additive).to_device(device);
                }
                AttentionMask::Additive(additive) => {
                    masked = masked + additive.clone();
                }
            }
        }
        if options.causal {
            let causal = ArrayD::from_shape_fn(IxDyn(&[q[2], k[2]]), |i| {
                if i[1] <= options.query_offset + i[0] {
                    0.0
                } else {
                    f32::NEG_INFINITY
                }
            });
            masked = masked + Tensor::from_array_no_grad(causal).to_device(device);
        }
        masked_softmax(&masked)
    };
    batch_matmul(&probabilities, &value)
}

// Shared established path, also used by SelfAttention's quantized/frozen
// projections. The public masked API deliberately accepts floating Q/K/V only;
// that contract must not narrow the older layer's I8 execution support.
pub(crate) fn unmasked_attention(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    scale: f32,
    causal: bool,
) -> Tensor {
    let n_rep = query.shape_vec()[1] / key.shape_vec()[1];
    let key = if n_rep == 1 {
        key.clone()
    } else {
        super::self_attention::repeat_kv(key.clone(), n_rep)
    };
    let value = if n_rep == 1 {
        value.clone()
    } else {
        super::self_attention::repeat_kv(value.clone(), n_rep)
    };
    let scores = batch_matmul(query, &key.transpose(2, 3));
    let probabilities = fused_softmax(&scores, scale, causal);
    batch_matmul(&probabilities, &value)
}

fn check_mask_shape(shape: &[usize], target: &[usize; 4]) {
    assert!(
        shape.len() <= 4
            && shape
                .iter()
                .rev()
                .zip(target.iter().rev())
                .all(|(&s, &t)| s == 1 || s == t),
        "attention mask must broadcast to [B,H,Q,K]"
    );
}

fn masked_softmax(input: &Tensor) -> Tensor {
    let shape = input.shape_vec();
    let keys = shape[3];
    let build_graph = !is_no_grad() && input.requires_grad();
    #[cfg(feature = "cuda")]
    if input.is_cuda() && !input.is_empty() {
        use crate::ops::cuda;
        let buffer = input
            .with_cuda_f32_buffer(|x| cuda::generic::masked_softmax(x, keys))
            .expect("attention CUDA masked softmax");
        let output = Tensor::from_cuda_f32_buffer_no_host(&shape, buffer.clone(), input.device());
        if !build_graph {
            return output;
        }
        let weak = Rc::downgrade(&output.0);
        let parent = input.clone();
        let mut data = output.0.borrow_mut();
        data.requires_grad = true;
        data.parents = vec![input.clone()];
        data.backward_op = Some(Rc::new(move |grad| {
            let upstream = weak
                .upgrade()
                .map(Tensor)
                .and_then(|t| t.cloned_cuda_f32_grad())
                .unwrap_or_else(|| {
                    cuda::upload_f32(&grad.iter().copied().collect::<Vec<_>>())
                        .expect("attention CUDA upstream")
                });
            let dx = cuda::fused_softmax_backward_f32_buffer(
                &buffer,
                &upstream,
                shape[0] * shape[1],
                shape[2],
                keys,
                1.0,
            )
            .expect("attention CUDA softmax backward");
            parent.add_backward_cuda_grad_buffer_only(dx);
        }));
        drop(data);
        return output;
    }
    let x = input.data_ref().iter().copied().collect::<Vec<_>>();
    let mut y = vec![0.0; x.len()];
    crate::cpu::for_each_chunk_mut(&mut y, keys, keys.saturating_mul(4), |(row, dst)| {
        let src = &x[row * keys..(row + 1) * keys];
        let max = src.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        if max == f32::NEG_INFINITY {
            return;
        }
        let mut sum = 0.0;
        for (out, &v) in dst.iter_mut().zip(src) {
            *out = (v - max).exp();
            sum += *out;
        }
        for out in dst {
            *out /= sum;
        }
    });
    let output = ArrayD::from_shape_vec(IxDyn(&shape), y.clone()).unwrap();
    if !build_graph {
        return Tensor::from_f32_data_no_grad_with_device_dtype(
            output,
            crate::precision::DType::F32,
            input.device(),
        );
    }
    let parent = input.clone();
    let backward = Rc::new(move |grad: &ndarray::ArrayViewD<f32>| {
        let g = grad.iter().copied().collect::<Vec<_>>();
        let mut dx = vec![0.0; g.len()];
        for row in 0..g.len() / keys {
            let start = row * keys;
            let dot = (0..keys).map(|j| g[start + j] * y[start + j]).sum::<f32>();
            for j in 0..keys {
                dx[start + j] = y[start + j] * (g[start + j] - dot);
            }
        }
        parent.add_backward_grad(ArrayD::from_shape_vec(IxDyn(&shape), dx).unwrap());
    });
    let mut data = Tensor::op_data(
        output.into_shared(),
        vec![input.clone()],
        Some(backward),
        true,
    );
    data.device = input.device();
    Tensor(Rc::new(RefCell::new(data)))
}
