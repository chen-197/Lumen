use super::*;

#[test]
fn tiled_attention_matches_scalar_for_batches_tails_masks_and_strided_inputs() {
    use crate::cpu::CpuKernelContext;
    use std::num::NonZeroUsize;
    use std::sync::Arc;
    let pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(3)
            .build()
            .unwrap(),
    );
    for (b, h, hk, s, l, past) in [
        (2, 4, 2, 129, 143, 14),
        (1, 1, 1, 193, 207, 14),
        (2, 1, 1, 128, 128, 0),
        (1, 3, 1, 127, 135, 8),
        (3, 2, 1, 1, 31, 18),
    ] {
        let d = 6;
        let data = |shape, seed| {
            Array4::from_shape_fn(shape, |(bb, hh, ss, dd)| {
                (((bb * 7 + hh * 13 + ss * 17 + dd * 19 + seed) % 67) as f32 - 33.) / 128.
            })
        };
        let q = data((b, h, s, d * 2), 3);
        let k = data((b, hk, l, d * 2), 11);
        let v = data((b, hk, l, d * 2), 23);
        let q = q.slice(ndarray::s![.., .., .., ..;2]);
        let k = k.slice(ndarray::s![.., .., .., ..;2]);
        let v = v.slice(ndarray::s![.., .., .., ..;2]);
        for causal in [true, false] {
            let mut expected = Array4::<f32>::zeros((b, s, h, d));
            for bb in 0..b {
                for hh in 0..h {
                    for row in 0..s {
                        let end = if causal { l.min(past + row + 1) } else { l };
                        let scores: Vec<f64> = (0..end)
                            .map(|j| {
                                (0..d)
                                    .map(|i| {
                                        q[[bb, hh, row, i]] as f64
                                            * k[[bb, hh / (h / hk), j, i]] as f64
                                    })
                                    .sum::<f64>()
                                    * 0.5
                            })
                            .collect();
                        let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                        let weights: Vec<f64> = scores.iter().map(|x| (x - max).exp()).collect();
                        let sum: f64 = weights.iter().sum::<f64>() + 1e-9;
                        for i in 0..d {
                            expected[[bb, row, hh, i]] = weights
                                .iter()
                                .enumerate()
                                .map(|(j, w)| w * v[[bb, hh / (h / hk), j, i]] as f64)
                                .sum::<f64>()
                                as f32
                                / sum as f32;
                        }
                    }
                }
            }
            for limit in [1, 3] {
                let context = CpuKernelContext::with_pool(pool.clone())
                    .with_task_limit(NonZeroUsize::new(limit).unwrap());
                let actual = context.scope(|| {
                    gqa_attention_no_repeat_bshd_view(&q, &k, &v, 0.5, causal, h / hk, past)
                });
                assert_eq!(actual.shape(), expected.shape());
                for (a, e) in actual.iter().zip(expected.iter()) {
                    assert!(
                        (a - e).abs() < 2e-6,
                        "{a} vs {e}, S={s}, causal={causal}, limit={limit}"
                    );
                }
            }
        }
    }
}

#[test]
fn prefill_and_cached_decode_match_across_attention_task_budgets() {
    use crate::cpu::CpuKernelContext;
    use std::num::NonZeroUsize;
    use std::sync::Arc;
    let pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(3)
            .build()
            .unwrap(),
    );
    let attention = SelfAttention::new_with_dtype(24, 3, 1, 160, 10000., true, DType::F32);
    let x = make_tensor(
        &[2, 129, 24],
        (0..2 * 129 * 24).map(|i| (i % 31) as f32 / 128.).collect(),
        DType::F32,
    );
    let next = make_tensor(&[2, 1, 24], vec![0.125; 48], DType::F32);
    let run = |limit| {
        CpuKernelContext::with_pool(pool.clone())
            .with_task_limit(NonZeroUsize::new(limit).unwrap())
            .scope(|| {
                no_grad(|| {
                    let (prefill, cache) = attention.forward(x.clone(), None);
                    let (decode, _) = attention.forward(next.clone(), cache);
                    (prefill.data_ref().to_owned(), decode.data_ref().to_owned())
                })
            })
    };
    let a = run(1);
    let b = run(3);
    for (left, right) in [(&a.0, &b.0), (&a.1, &b.1)] {
        for (x, y) in left.iter().zip(right.iter()) {
            assert!((x - y).abs() < 1e-5, "{x} vs {y}");
        }
    }
}

#[test]
fn active_cache_views_pack_each_batch_and_head_without_unused_capacity() {
    for dtype in [DType::F32, DType::F16, DType::BF16] {
        let shape = [2, 3, 9, 4];
        let values = (0..shape.iter().product())
            .map(|i| i as f32 * 0.125)
            .collect::<Vec<_>>();
        let k = make_tensor(&shape, values.clone(), dtype);
        let v = make_tensor(&shape, values.iter().map(|x| -x).collect(), dtype);
        for active_len in [0, 3, 9, 1] {
            with_cache_f32_views(&k, &v, active_len, |k4, v4| {
                assert_eq!(k4.shape(), &[2, 3, active_len, 4]);
                assert_eq!(v4.shape(), k4.shape());
                for bb in 0..2 {
                    for hh in 0..3 {
                        for pos in 0..active_len {
                            for d in 0..4 {
                                let expected = values[((bb * 3 + hh) * 9 + pos) * 4 + d];
                                assert_eq!(k4[[bb, hh, pos, d]], expected);
                                assert_eq!(v4[[bb, hh, pos, d]], -expected);
                            }
                        }
                    }
                }
            });
        }
    }
}
use crate::autograd::no_grad;
#[cfg(feature = "cuda")]
use crate::device::set_strict_device_execution;
use crate::precision::{DType, PrecisionConfig, set_default_runtime_dtype, with_precision_config};
use ndarray::{Array, IxDyn};

fn make_tensor(shape: &[usize], data: Vec<f32>, dtype: DType) -> Tensor {
    let t = Tensor::from_array_no_grad(
        Array::from_shape_vec(IxDyn(shape), data)
            .expect("test tensor shape mismatch")
            .into_dyn(),
    );
    t.cast_inplace(dtype);
    t
}

#[cfg(feature = "cuda")]
fn make_grad_tensor(shape: &[usize], data: Vec<f32>) -> Tensor {
    Tensor::from_data_with_grad_flag(
        Array::from_shape_vec(IxDyn(shape), data)
            .expect("test tensor shape mismatch")
            .into_dyn(),
        true,
    )
}

#[test]
#[should_panic(expected = "n_head must be > 0")]
fn self_attention_rejects_zero_query_heads() {
    let _ = SelfAttention::new(16, 0, 1, 8, 10000.0, true);
}

#[test]
#[should_panic(expected = "n_kv_head must be > 0")]
fn self_attention_rejects_zero_kv_heads() {
    let _ = SelfAttention::new(16, 4, 0, 8, 10000.0, true);
}

#[test]
#[should_panic(expected = "n_head must be divisible by n_kv_head")]
fn self_attention_rejects_non_divisible_gqa_ratio() {
    let _ = SelfAttention::new(12, 6, 4, 8, 10000.0, true);
}

#[test]
#[should_panic(expected = "attention input must be [B,S,H]")]
fn self_attention_rejects_non_3d_input() {
    let attn = SelfAttention::new(8, 2, 1, 8, 10000.0, true);
    let input = make_tensor(&[1, 8], vec![0.0; 8], DType::F32);
    no_grad(|| {
        let _ = attn.forward(input, None);
    });
}

#[test]
fn no_grad_forward_keeps_bf16_input_storage() {
    let attn = SelfAttention::new(8, 2, 1, 8, 10000.0, true);
    let input = make_tensor(
        &[1, 2, 8],
        (0..16).map(|i| i as f32 * 0.1 - 0.5).collect(),
        DType::BF16,
    );

    no_grad(|| {
        let _ = attn.forward(input.clone(), None);
    });

    assert_eq!(input.dtype(), DType::BF16);
    input.with_storage_view(|view| match view {
        crate::autograd::TensorStorageView::BF16(_) => {}
        crate::autograd::TensorStorageView::F16(_) => {
            panic!("shape inspection should not materialize bf16 attention input")
        }
        crate::autograd::TensorStorageView::F32(_) => {
            panic!("shape inspection should not materialize bf16 attention input")
        }
    });
}

#[test]
fn kv_cache_creation_follows_runtime_dtype() {
    with_precision_config(
        PrecisionConfig {
            parameter_dtype: DType::F32,
            runtime_dtype: DType::BF16,
            allow_parameter_dtype_copies: false,
        },
        || {
            let cache = KVCacheInner::new(1, 1, 4, 4);
            assert_eq!(cache.dtype, DType::BF16);
            assert!(cache.follows_global_dtype);
            cache
                .k
                .with_storage_view_preferring(StoragePreference::Native, |view| match view {
                    TensorStorageView::BF16(view) => assert_eq!(view.shape(), &[1, 1, 4, 4]),
                    TensorStorageView::F16(_) => {
                        panic!("kv cache should follow bf16 runtime dtype")
                    }
                    TensorStorageView::F32(_) => {
                        panic!("kv cache should follow bf16 runtime dtype")
                    }
                });
        },
    );
}

#[test]
fn kv_cache_manual_cast_disables_global_following() {
    let mut cache = KVCacheInner::new(1, 1, 4, 4);
    assert!(cache.follows_global_dtype);
    cache.cast_inplace(DType::BF16);
    assert_eq!(cache.dtype, DType::BF16);
    assert!(!cache.follows_global_dtype);
    cache
        .v
        .with_storage_view_preferring(StoragePreference::Native, |view| match view {
            TensorStorageView::BF16(view) => assert_eq!(view.shape(), &[1, 1, 4, 4]),
            TensorStorageView::F16(_) => {
                panic!("manual cast should switch cache to explicit bf16 storage")
            }
            TensorStorageView::F32(_) => {
                panic!("manual cast should switch cache to explicit bf16 storage")
            }
        });
}

#[test]
fn kv_cache_explicit_dtype_overrides_global_default() {
    with_precision_config(
        PrecisionConfig {
            parameter_dtype: DType::F32,
            runtime_dtype: DType::BF16,
            allow_parameter_dtype_copies: false,
        },
        || {
            let cache = KVCacheInner::new_with_dtype(1, 1, 4, 4, DType::F32);
            assert_eq!(cache.dtype, DType::F32);
            assert!(!cache.follows_global_dtype);
            cache
                .k
                .with_storage_view_preferring(StoragePreference::Native, |view| match view {
                    TensorStorageView::F32(view) => assert_eq!(view.shape(), &[1, 1, 4, 4]),
                    TensorStorageView::F16(_) => {
                        panic!("explicit kv cache dtype should override global bf16 runtime dtype")
                    }
                    TensorStorageView::BF16(_) => {
                        panic!("explicit kv cache dtype should override global bf16 runtime dtype")
                    }
                });
        },
    );
}

#[test]
fn self_attention_allows_i8_activation_dtype_with_float_kv_cache() {
    let attn = SelfAttention::new_with_runtime_dtypes(
        8,
        2,
        1,
        8,
        10000.0,
        true,
        DType::I8,
        DType::I8,
        DType::BF16,
    );
    assert_eq!(attn.activation_dtype(), DType::I8);
    assert_eq!(attn.kv_cache_dtype(), DType::BF16);
}

#[test]
#[should_panic(expected = "KV cache batch size must be > 0")]
fn kv_cache_rejects_zero_batch() {
    let _ = KVCacheInner::new(0, 1, 4, 4);
}

#[test]
#[should_panic(expected = "SelfAttention KV cache shape mismatch")]
fn self_attention_rejects_cache_shape_mismatch() {
    let attn = SelfAttention::new(8, 2, 1, 8, 10000.0, true);
    let input = make_tensor(&[1, 1, 8], vec![0.0; 8], DType::F32);
    let cache: KVCache = Rc::new(RefCell::new(KVCacheInner::new(2, 1, 8, 4)));
    no_grad(|| {
        let _ = attn.forward(input, Some(cache));
    });
}

#[test]
#[should_panic(expected = "SelfAttention KV cache current length out of bounds")]
fn self_attention_rejects_cache_len_overflow() {
    let attn = SelfAttention::new(8, 2, 1, 8, 10000.0, true);
    let input = make_tensor(&[1, 1, 8], vec![0.0; 8], DType::F32);
    let mut cache = KVCacheInner::new(1, 1, 8, 4);
    cache.len = 9;
    let cache: KVCache = Rc::new(RefCell::new(cache));
    no_grad(|| {
        let _ = attn.forward(input, Some(cache));
    });
}

#[test]
fn self_attention_explicit_dtype_overrides_global_default() {
    with_precision_config(
        PrecisionConfig {
            parameter_dtype: DType::BF16,
            runtime_dtype: DType::F32,
            allow_parameter_dtype_copies: false,
        },
        || {
            let attn = SelfAttention::new_with_dtype(8, 2, 1, 8, 10000.0, true, DType::F32);
            for weight in [
                &attn.w_q.weight,
                &attn.w_k.weight,
                &attn.w_v.weight,
                &attn.w_o.weight,
            ] {
                assert_eq!(weight.dtype(), DType::F32);
            }
            assert_eq!(attn.rope.cache_dtype(), DType::F32);
        },
    );
}

#[test]
fn self_attention_default_construction_splits_parameter_and_runtime_defaults() {
    with_precision_config(
        PrecisionConfig {
            parameter_dtype: DType::F32,
            runtime_dtype: DType::BF16,
            allow_parameter_dtype_copies: false,
        },
        || {
            let attn = SelfAttention::new(8, 2, 1, 8, 10000.0, true);
            for weight in [
                &attn.w_q.weight,
                &attn.w_k.weight,
                &attn.w_v.weight,
                &attn.w_o.weight,
            ] {
                assert_eq!(weight.dtype(), DType::F32);
            }
            assert_eq!(attn.rope.cache_dtype(), DType::BF16);
        },
    );
}

#[test]
fn self_attention_default_construction_captures_runtime_dtype_for_future_cache_creation() {
    with_precision_config(
        PrecisionConfig {
            parameter_dtype: DType::F32,
            runtime_dtype: DType::BF16,
            allow_parameter_dtype_copies: false,
        },
        || {
            let attn = SelfAttention::new(8, 2, 1, 8, 10000.0, true);
            set_default_runtime_dtype(DType::F32);
            let input = make_tensor(&[1, 1, 8], vec![0.0; 8], DType::F32);
            let (_out, cache) = no_grad(|| attn.forward(input, None));
            let cache = cache.expect("no_grad attention should create cache");
            assert_eq!(cache.borrow().dtype, DType::BF16);
        },
    );
}

#[test]
fn repeat_kv_no_grad_preserves_bf16_input_dtype() {
    let input = make_tensor(
        &[1, 2, 2, 2],
        vec![1.0, 2.0, -1.0, -2.0, 3.0, 4.0, -3.0, -4.0],
        DType::BF16,
    );
    let out = no_grad(|| repeat_kv(input, 2));
    assert_eq!(out.shape_vec(), vec![1, 4, 2, 2]);
    assert_eq!(out.dtype(), DType::BF16);
}

#[test]
fn repeat_kv_no_grad_preserves_i8_input_dtype() {
    let input = make_tensor(&[1, 2, 1, 2], vec![1.0, -2.0, 3.0, -4.0], DType::I8);
    let out = no_grad(|| repeat_kv(input, 3));
    assert_eq!(out.shape_vec(), vec![1, 6, 1, 2]);
    assert_eq!(out.dtype(), DType::I8);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_repeat_kv_matches_cpu_reference_in_strict_mode() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let input = make_tensor(
        &[1, 2, 2, 3],
        vec![
            1.0, 2.0, 3.0, -1.0, -2.0, -3.0, 4.0, 5.0, 6.0, -4.0, -5.0, -6.0,
        ],
        DType::BF16,
    )
    .to_cuda();

    assert!(input.0.borrow().cuda_f32_data.is_none());
    assert!(input.0.borrow().cuda_bf16_data.is_some());

    set_strict_device_execution(true);
    let cuda_out = no_grad(|| repeat_kv(input.clone(), 2));
    set_strict_device_execution(false);

    let cpu_out = no_grad(|| repeat_kv(input.to_cpu(), 2));
    assert!(cuda_out.is_cuda());
    assert_eq!(cuda_out.dtype(), DType::BF16);
    assert!(cuda_out.0.borrow().cuda_f32_data.is_none());
    assert!(cuda_out.0.borrow().cuda_bf16_data.is_some());
    assert_eq!(cuda_out.shape_vec(), cpu_out.shape_vec());
    for (got, expect) in cuda_out.data_ref().iter().zip(cpu_out.data_ref().iter()) {
        assert!((got - expect).abs() < 2e-2, "got {got}, expect {expect}");
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_empty_repeat_kv_no_grad_stays_on_cuda_in_strict_mode() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let input = make_tensor(&[1, 2, 0, 3], vec![], DType::BF16).to_cuda();

    crate::ops::cuda::set_enabled(true);
    set_strict_device_execution(true);
    let out = no_grad(|| repeat_kv(input, 2));
    set_strict_device_execution(false);
    crate::ops::cuda::set_enabled(false);

    assert!(out.is_cuda());
    assert_eq!(out.dtype(), DType::BF16);
    assert_eq!(out.shape_vec(), vec![1, 4, 0, 3]);
    assert_eq!(out.len(), 0);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_kv_cache_prefix_preserves_bf16_native_storage_in_strict_mode() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let cache = make_tensor(
        &[1, 2, 4, 2],
        vec![
            1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, -1.0, -1.5, -2.0, -2.5, -3.0, -3.5, -4.0, -4.5,
        ],
        DType::BF16,
    )
    .to_cuda();
    assert!(cache.0.borrow().cuda_f32_data.is_none());
    assert!(cache.0.borrow().cuda_bf16_data.is_some());

    set_strict_device_execution(true);
    let cuda_prefix = cache_prefix_tensor(&cache, 2);
    set_strict_device_execution(false);

    let cpu_prefix = cache_prefix_tensor(&cache.to_cpu(), 2);
    assert!(cuda_prefix.is_cuda());
    assert_eq!(cuda_prefix.dtype(), DType::BF16);
    assert_eq!(cuda_prefix.shape_vec(), vec![1, 2, 2, 2]);
    assert!(cuda_prefix.0.borrow().cuda_f32_data.is_none());
    assert!(cuda_prefix.0.borrow().cuda_bf16_data.is_some());
    for (got, expect) in cuda_prefix
        .data_ref()
        .iter()
        .zip(cpu_prefix.data_ref().iter())
    {
        assert!((got - expect).abs() < 2e-2, "got {got}, expect {expect}");
    }
}

#[test]
fn repeat_kv_backward_accumulates_repeated_heads_on_cpu() {
    let input = Tensor::from_data_with_grad_flag(
        Array::from_shape_vec(
            IxDyn(&[1, 2, 2, 2]),
            vec![1.0, -1.0, 2.0, -2.0, 0.5, 1.5, -0.5, -1.5],
        )
        .expect("input shape mismatch")
        .into_dyn(),
        true,
    );
    let coeff = Tensor::from_data_with_grad_flag(
        Array::from_shape_vec(
            IxDyn(&[1, 4, 2, 2]),
            (0..16).map(|v| v as f32 * 0.25 - 1.0).collect(),
        )
        .expect("coeff shape mismatch")
        .into_dyn(),
        false,
    );

    let out = repeat_kv(input.clone(), 2);
    let loss = crate::ops::arithmetic::sum(&(&out * &coeff));
    loss.backward();

    let grad = input.grad().expect("repeat_kv input grad");
    let vals = grad.iter().copied().collect::<Vec<_>>();
    let coeff_vals = coeff.data_ref().iter().copied().collect::<Vec<_>>();
    let mut expected = vec![0.0f32; 8];
    for hk in 0..2 {
        for rep in 0..2 {
            for ss in 0..2 {
                for dd in 0..2 {
                    let src = ((hk * 2 + rep) * 2 + ss) * 2 + dd;
                    let dst = (hk * 2 + ss) * 2 + dd;
                    expected[dst] += coeff_vals[src];
                }
            }
        }
    }
    assert_eq!(vals, expected);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_repeat_kv_backward_matches_cpu_reference_in_strict_mode() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let input_cpu = Tensor::from_data_with_grad_flag(
        Array::from_shape_vec(
            IxDyn(&[1, 2, 2, 2]),
            vec![1.0, -1.0, 2.0, -2.0, 0.5, 1.5, -0.5, -1.5],
        )
        .expect("input shape mismatch")
        .into_dyn(),
        true,
    );
    let coeff_cpu = Tensor::from_data_with_grad_flag(
        Array::from_shape_vec(
            IxDyn(&[1, 4, 2, 2]),
            (0..16).map(|v| v as f32 * 0.25 - 1.0).collect(),
        )
        .expect("coeff shape mismatch")
        .into_dyn(),
        false,
    );
    let input_cuda = input_cpu.to_cuda();
    let coeff_cuda = coeff_cpu.to_cuda();

    crate::ops::cuda::set_enabled(true);
    set_strict_device_execution(true);
    let cuda_out = repeat_kv(input_cuda.clone(), 2);
    let cuda_loss = crate::ops::arithmetic::sum(&(&cuda_out * &coeff_cuda));
    cuda_loss.backward();
    assert!(input_cuda.cloned_cuda_f32_grad().is_some());
    assert!(!input_cuda.has_host_grad());
    set_strict_device_execution(false);
    crate::ops::cuda::set_enabled(false);

    let cpu_out = repeat_kv(input_cpu.clone(), 2);
    let cpu_loss = crate::ops::arithmetic::sum(&(&cpu_out * &coeff_cpu));
    cpu_loss.backward();

    let cuda_grad = input_cuda.grad().expect("cuda repeat_kv grad");
    let cpu_grad = input_cpu.grad().expect("cpu repeat_kv grad");
    for (got, expect) in cuda_grad.iter().zip(cpu_grad.iter()) {
        assert!((got - expect).abs() < 1e-5, "got {got}, expect {expect}");
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_repeat_kv_preserves_i8_dtype_in_strict_mode() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let input = make_tensor(&[1, 2, 1, 2], vec![1.0, -2.0, 3.0, -4.0], DType::I8).to_cuda();

    assert!(input.0.borrow().cuda_f32_data.is_none());
    assert!(input.0.borrow().cuda_i8_data.is_some());

    set_strict_device_execution(true);
    let out = no_grad(|| repeat_kv(input, 2));
    set_strict_device_execution(false);

    assert!(out.is_cuda());
    assert_eq!(out.dtype(), DType::I8);
    assert!(out.0.borrow().cuda_f32_data.is_none());
    assert!(out.0.borrow().cuda_i8_data.is_some());
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_self_attention_training_backward_matches_cpu_reference_in_strict_mode() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let attn = SelfAttention::new_with_dtype(8, 2, 1, 8, 10000.0, true, DType::F32);
    let attn_ref = SelfAttention::new_with_dtype(8, 2, 1, 8, 10000.0, true, DType::F32);

    for ((dst, src), shape, scale) in [
        (
            (&attn.w_q.weight, &attn_ref.w_q.weight),
            vec![8, 8],
            0.03125f32,
        ),
        (
            (&attn.w_k.weight, &attn_ref.w_k.weight),
            vec![4, 8],
            0.0275f32,
        ),
        (
            (&attn.w_v.weight, &attn_ref.w_v.weight),
            vec![4, 8],
            -0.021f32,
        ),
        (
            (&attn.w_o.weight, &attn_ref.w_o.weight),
            vec![8, 8],
            0.01875f32,
        ),
    ] {
        let data = (0..shape.iter().product::<usize>())
            .map(|i| (i as f32 * scale) - 0.25)
            .collect::<Vec<_>>();
        let arr = Array::from_shape_vec(IxDyn(&shape), data).expect("weight shape");
        dst.set_array_f32_with_dtype(arr.clone(), DType::F32);
        src.set_array_f32_with_dtype(arr, DType::F32);
    }

    attn.to_cuda();
    let input_cpu = make_grad_tensor(
        &[1, 3, 8],
        (0..24).map(|i| (i as f32 * 0.0625) - 0.4).collect(),
    );
    let coeff_cpu = Tensor::from_data_with_grad_flag(
        Array::from_shape_vec(
            IxDyn(&[1, 3, 8]),
            (0..24).map(|i| (i as f32 * 0.0375) - 0.2).collect(),
        )
        .expect("coeff shape mismatch")
        .into_dyn(),
        false,
    );
    let input_cuda = input_cpu.to_cuda();
    let coeff_cuda = coeff_cpu.to_cuda();

    crate::ops::cuda::set_enabled(true);
    set_strict_device_execution(true);
    let (cuda_out, cuda_cache) = attn.forward(input_cuda.clone(), None);
    assert!(cuda_out.is_cuda());
    assert!(cuda_cache.is_none());
    let cuda_loss = crate::ops::arithmetic::sum(&(&cuda_out * &coeff_cuda));
    cuda_loss.backward();
    assert!(input_cuda.cloned_cuda_f32_grad().is_some());
    assert!(!input_cuda.has_host_grad());
    for param in attn.parameters() {
        assert!(param.cloned_cuda_f32_grad().is_some());
        assert!(!param.has_host_grad());
    }
    set_strict_device_execution(false);
    crate::ops::cuda::set_enabled(false);

    let (cpu_out, cpu_cache) = attn_ref.forward(input_cpu.clone(), None);
    assert!(cpu_cache.is_none());
    let cpu_loss = crate::ops::arithmetic::sum(&(&cpu_out * &coeff_cpu));
    cpu_loss.backward();

    for (got, expect) in cuda_out.data_ref().iter().zip(cpu_out.data_ref().iter()) {
        assert!((got - expect).abs() < 1e-4, "got {got}, expect {expect}");
    }
    let cuda_input_grad = input_cuda.grad().expect("cuda attention input grad");
    let cpu_input_grad = input_cpu.grad().expect("cpu attention input grad");
    for (got, expect) in cuda_input_grad.iter().zip(cpu_input_grad.iter()) {
        assert!((got - expect).abs() < 1e-3, "got {got}, expect {expect}");
    }
    for (cuda_param, cpu_param) in attn.parameters().iter().zip(attn_ref.parameters().iter()) {
        let cuda_grad = cuda_param.grad().expect("cuda attention parameter grad");
        let cpu_grad = cpu_param.grad().expect("cpu attention parameter grad");
        for (got, expect) in cuda_grad.iter().zip(cpu_grad.iter()) {
            assert!((got - expect).abs() < 1e-3, "got {got}, expect {expect}");
        }
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_self_attention_decode_output_projection_stays_on_cuda_and_matches_cpu() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let attn = SelfAttention::new_with_dtype(8, 2, 1, 8, 10000.0, true, DType::BF16);
    let attn_ref = SelfAttention::new_with_dtype(8, 2, 1, 8, 10000.0, true, DType::BF16);

    for ((dst, src), shape, scale) in [
        (
            (&attn.w_q.weight, &attn_ref.w_q.weight),
            vec![8, 8],
            0.03125f32,
        ),
        (
            (&attn.w_k.weight, &attn_ref.w_k.weight),
            vec![4, 8],
            0.0275f32,
        ),
        (
            (&attn.w_v.weight, &attn_ref.w_v.weight),
            vec![4, 8],
            -0.021f32,
        ),
        (
            (&attn.w_o.weight, &attn_ref.w_o.weight),
            vec![8, 8],
            0.01875f32,
        ),
    ] {
        let data = (0..shape.iter().product::<usize>())
            .map(|i| (i as f32 * scale) - 0.25)
            .collect::<Vec<_>>();
        let arr = Array::from_shape_vec(IxDyn(&shape), data).expect("weight shape");
        dst.set_array_f32_with_dtype(arr.clone(), DType::BF16);
        src.set_array_f32_with_dtype(arr, DType::BF16);
    }

    attn.to_cuda();
    let input = make_tensor(
        &[1, 1, 8],
        (0..8).map(|i| (i as f32 * 0.125) - 0.25).collect(),
        DType::BF16,
    )
    .to_cuda();

    set_strict_device_execution(true);
    let (cuda_out, _) = no_grad(|| attn.forward(input.clone(), None));
    set_strict_device_execution(false);

    let (cpu_out, _) = no_grad(|| attn_ref.forward(input.to_cpu(), None));
    assert!(cuda_out.is_cuda());
    assert_eq!(cuda_out.shape_vec(), cpu_out.shape_vec());
    for (got, expect) in cuda_out.data_ref().iter().zip(cpu_out.data_ref().iter()) {
        assert!((got - expect).abs() < 3e-2, "got {got}, expect {expect}");
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_self_attention_decode_second_step_matches_cpu_reference() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let attn = SelfAttention::new_with_dtype(8, 2, 1, 8, 10000.0, true, DType::BF16);
    let attn_ref = SelfAttention::new_with_dtype(8, 2, 1, 8, 10000.0, true, DType::BF16);

    for ((dst, src), shape, scale) in [
        (
            (&attn.w_q.weight, &attn_ref.w_q.weight),
            vec![8, 8],
            0.03125f32,
        ),
        (
            (&attn.w_k.weight, &attn_ref.w_k.weight),
            vec![4, 8],
            0.0275f32,
        ),
        (
            (&attn.w_v.weight, &attn_ref.w_v.weight),
            vec![4, 8],
            -0.021f32,
        ),
        (
            (&attn.w_o.weight, &attn_ref.w_o.weight),
            vec![8, 8],
            0.01875f32,
        ),
    ] {
        let data = (0..shape.iter().product::<usize>())
            .map(|i| (i as f32 * scale) - 0.25)
            .collect::<Vec<_>>();
        let arr = Array::from_shape_vec(IxDyn(&shape), data).expect("weight shape");
        dst.set_array_f32_with_dtype(arr.clone(), DType::BF16);
        src.set_array_f32_with_dtype(arr, DType::BF16);
    }

    attn.to_cuda();
    let cuda_cache: KVCache = Rc::new(RefCell::new(KVCacheInner::new_with_dtype(
        1,
        1,
        8,
        4,
        DType::BF16,
    )));
    cuda_cache.borrow_mut().to_device_inplace(Device::Cuda);
    let cpu_cache: KVCache = Rc::new(RefCell::new(KVCacheInner::new_with_dtype(
        1,
        1,
        8,
        4,
        DType::BF16,
    )));

    let step1 = make_tensor(
        &[1, 1, 8],
        (0..8).map(|i| (i as f32 * 0.125) - 0.25).collect(),
        DType::BF16,
    );
    let step2 = make_tensor(
        &[1, 1, 8],
        (0..8).map(|i| (i as f32 * 0.09375) + 0.1).collect(),
        DType::BF16,
    );

    set_strict_device_execution(true);
    let (_cuda_out1, cuda_cache) = no_grad(|| attn.forward(step1.to_cuda(), Some(cuda_cache)));
    let cuda_cache = cuda_cache.expect("cuda cache after step1");
    let (cuda_out2, cuda_cache) = no_grad(|| attn.forward(step2.to_cuda(), Some(cuda_cache)));
    let cuda_cache = cuda_cache.expect("cuda cache after step2");
    set_strict_device_execution(false);

    let (_cpu_out1, cpu_cache) = no_grad(|| attn_ref.forward(step1, Some(cpu_cache)));
    let cpu_cache = cpu_cache.expect("cpu cache after step1");
    let (cpu_out2, cpu_cache) = no_grad(|| attn_ref.forward(step2, Some(cpu_cache)));
    let cpu_cache = cpu_cache.expect("cpu cache after step2");

    assert!(cuda_out2.is_cuda());
    assert_eq!(cuda_cache.borrow().len, 2);
    assert_eq!(cpu_cache.borrow().len, 2);
    assert_eq!(cuda_out2.shape_vec(), cpu_out2.shape_vec());
    for (got, expect) in cuda_out2.data_ref().iter().zip(cpu_out2.data_ref().iter()) {
        assert!((got - expect).abs() < 5e-2, "got {got}, expect {expect}");
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_self_attention_prefill_matches_cpu_reference_in_strict_mode() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let attn = SelfAttention::new_with_dtype(8, 2, 1, 8, 10000.0, true, DType::BF16);
    let attn_ref = SelfAttention::new_with_dtype(8, 2, 1, 8, 10000.0, true, DType::BF16);

    for ((dst, src), shape, scale) in [
        (
            (&attn.w_q.weight, &attn_ref.w_q.weight),
            vec![8, 8],
            0.03125f32,
        ),
        (
            (&attn.w_k.weight, &attn_ref.w_k.weight),
            vec![4, 8],
            0.0275f32,
        ),
        (
            (&attn.w_v.weight, &attn_ref.w_v.weight),
            vec![4, 8],
            -0.021f32,
        ),
        (
            (&attn.w_o.weight, &attn_ref.w_o.weight),
            vec![8, 8],
            0.01875f32,
        ),
    ] {
        let data = (0..shape.iter().product::<usize>())
            .map(|i| (i as f32 * scale) - 0.25)
            .collect::<Vec<_>>();
        let arr = Array::from_shape_vec(IxDyn(&shape), data).expect("weight shape");
        dst.set_array_f32_with_dtype(arr.clone(), DType::BF16);
        src.set_array_f32_with_dtype(arr, DType::BF16);
    }

    attn.to_cuda();
    let input = make_tensor(
        &[1, 3, 8],
        (0..24).map(|i| (i as f32 * 0.0625) - 0.4).collect(),
        DType::BF16,
    );

    set_strict_device_execution(true);
    let (cuda_out, cuda_cache) = no_grad(|| attn.forward(input.to_cuda(), None));
    set_strict_device_execution(false);

    let (cpu_out, cpu_cache) = no_grad(|| attn_ref.forward(input, None));
    let cuda_cache = cuda_cache.expect("cuda cache after prefill");
    let cpu_cache = cpu_cache.expect("cpu cache after prefill");

    assert!(cuda_out.is_cuda());
    assert_eq!(cuda_out.shape_vec(), cpu_out.shape_vec());
    assert_eq!(cuda_cache.borrow().len, 3);
    assert_eq!(cpu_cache.borrow().len, 3);
    for (got, expect) in cuda_out.data_ref().iter().zip(cpu_out.data_ref().iter()) {
        assert!((got - expect).abs() < 6e-2, "got {got}, expect {expect}");
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_self_attention_multi_token_continuation_matches_cpu_reference_in_strict_mode() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let attn = SelfAttention::new_with_dtype(8, 2, 1, 8, 10000.0, true, DType::BF16);
    let attn_ref = SelfAttention::new_with_dtype(8, 2, 1, 8, 10000.0, true, DType::BF16);

    for ((dst, src), shape, scale) in [
        (
            (&attn.w_q.weight, &attn_ref.w_q.weight),
            vec![8, 8],
            0.03125f32,
        ),
        (
            (&attn.w_k.weight, &attn_ref.w_k.weight),
            vec![4, 8],
            0.0275f32,
        ),
        (
            (&attn.w_v.weight, &attn_ref.w_v.weight),
            vec![4, 8],
            -0.021f32,
        ),
        (
            (&attn.w_o.weight, &attn_ref.w_o.weight),
            vec![8, 8],
            0.01875f32,
        ),
    ] {
        let data = (0..shape.iter().product::<usize>())
            .map(|i| (i as f32 * scale) - 0.25)
            .collect::<Vec<_>>();
        let arr = Array::from_shape_vec(IxDyn(&shape), data).expect("weight shape");
        dst.set_array_f32_with_dtype(arr.clone(), DType::BF16);
        src.set_array_f32_with_dtype(arr, DType::BF16);
    }

    attn.to_cuda();
    let prefill = make_tensor(
        &[1, 2, 8],
        (0..16).map(|i| (i as f32 * 0.0625) - 0.4).collect(),
        DType::BF16,
    );
    let continuation = make_tensor(
        &[1, 2, 8],
        (0..16).map(|i| (i as f32 * 0.0575) + 0.15).collect(),
        DType::BF16,
    );

    set_strict_device_execution(true);
    let (_cuda_prefill_out, cuda_cache) = no_grad(|| attn.forward(prefill.to_cuda(), None));
    let cuda_cache = cuda_cache.expect("cuda cache after prefill");
    let (cuda_out, cuda_cache) = no_grad(|| attn.forward(continuation.to_cuda(), Some(cuda_cache)));
    let cuda_cache = cuda_cache.expect("cuda cache after continuation");
    set_strict_device_execution(false);

    let (_cpu_prefill_out, cpu_cache) = no_grad(|| attn_ref.forward(prefill, None));
    let cpu_cache = cpu_cache.expect("cpu cache after prefill");
    let (cpu_out, cpu_cache) = no_grad(|| attn_ref.forward(continuation, Some(cpu_cache)));
    let cpu_cache = cpu_cache.expect("cpu cache after continuation");

    assert!(cuda_out.is_cuda());
    assert_eq!(cuda_cache.borrow().len, 4);
    assert_eq!(cpu_cache.borrow().len, 4);
    assert_eq!(cuda_out.shape_vec(), cpu_out.shape_vec());
    for (got, expect) in cuda_out.data_ref().iter().zip(cpu_out.data_ref().iter()) {
        assert!((got - expect).abs() < 7e-2, "got {got}, expect {expect}");
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_kv_cache_writes_preserve_resident_buffers_across_decode_steps() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let attn = SelfAttention::new_with_dtype(8, 2, 1, 8, 10000.0, true, DType::BF16);
    attn.to_cuda();
    let cache: KVCache = Rc::new(RefCell::new(KVCacheInner::new_with_dtype(
        1,
        1,
        8,
        4,
        DType::BF16,
    )));
    cache.borrow().k.to_cuda_inplace();
    cache.borrow().v.to_cuda_inplace();

    let step1 = make_tensor(
        &[1, 1, 8],
        (0..8).map(|i| (i as f32 * 0.125) - 0.25).collect(),
        DType::BF16,
    )
    .to_cuda();
    let step2 = make_tensor(
        &[1, 1, 8],
        (0..8).map(|i| (i as f32 * 0.09375) + 0.1).collect(),
        DType::BF16,
    )
    .to_cuda();

    set_strict_device_execution(true);
    let (_out1, cache) = no_grad(|| attn.forward(step1, Some(cache)));
    let cache = cache.expect("decode step 1 should return KV cache");
    assert!(cache.borrow().k.cloned_cuda_f32_buffer().is_some());
    assert!(cache.borrow().v.cloned_cuda_f32_buffer().is_some());
    let (_out2, cache) = no_grad(|| attn.forward(step2, Some(cache)));
    let cache = cache.expect("decode step 2 should return KV cache");
    set_strict_device_execution(false);

    assert_eq!(cache.borrow().len, 2);
    assert!(cache.borrow().k.cloned_cuda_f32_buffer().is_some());
    assert!(cache.borrow().v.cloned_cuda_f32_buffer().is_some());
}
