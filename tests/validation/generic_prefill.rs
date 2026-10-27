//! Forward/gradient references for query/row tiling and caller task budgets.
use crate::accuracy_gate::{check, ordered, tensor, values};
use crate::autograd::{Device, Tensor, no_grad};
use crate::cpu::CpuKernelContext;
use crate::layers::{AttentionMask, AttentionOptions, scaled_dot_product_attention};
use crate::ops::matmul::{batch_matmul, matmul};
use crate::precision::DType;
use ndarray::{ArrayD, IxDyn};
use serde_json::json;
use std::{num::NonZeroUsize, sync::Arc};

fn data(len: usize, seed: usize) -> Vec<f32> {
    (0..len)
        .map(|i| (((i * 17 + seed) % 251) as f32 - 125.0) * 0.00371)
        .collect()
}
fn make(shape: &[usize], dtype: DType, layout: usize, seed: usize) -> Tensor {
    tensor(
        shape,
        &data(shape.iter().product(), seed),
        dtype,
        layout,
        Device::Cpu,
    )
}
fn snapshot(t: &Tensor) -> serde_json::Value {
    json!({"shape":t.shape_vec(),"values":values(t),"grad":t.grad().map(|g|g.iter().copied().collect::<Vec<_>>())})
}
fn trace(value: serde_json::Value) {
    if std::env::var_os("LUMEN_OPERATOR_AUDIT").is_some() {
        println!("PREFILL_CPU_AUDIT={value}");
    }
}
fn contexts() -> Vec<CpuKernelContext> {
    let pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(3)
            .build()
            .unwrap(),
    );
    (1..=3)
        .map(|n| {
            CpuKernelContext::with_pool(pool.clone()).with_task_limit(NonZeroUsize::new(n).unwrap())
        })
        .collect()
}

#[test]
fn row_fallback_all_dtype_pairs_layouts_and_budgets_match_f64() {
    use crate::autograd::TensorStorageOwned;
    let contexts = contexts();
    let types = [DType::F32, DType::F16, DType::BF16, DType::I8];
    for xd in types {
        for wd in types {
            for layout in 0..3 {
                for (case, (m, k, n)) in [(1, 65, 67), (9, 37, 19), (33, 65, 67)]
                    .into_iter()
                    .enumerate()
                {
                    let x = make(&[m, k], xd, layout, 3);
                    let w = make(&[n, k], wd, layout, 7);
                    let xv = values(&x);
                    let wv = values(&w);
                    let dy = data(m * n, 11);
                    let expected = (0..m * n)
                        .map(|i| {
                            (0..k)
                                .map(|j| xv[i / n * k + j] as f64 * wv[i % n * k + j] as f64)
                                .sum::<f64>()
                        })
                        .collect::<Vec<_>>();
                    let dx = (0..m * k)
                        .map(|i| {
                            (0..n)
                                .map(|j| dy[i / k * n + j] as f64 * wv[j * k + i % k] as f64)
                                .sum::<f64>()
                        })
                        .collect::<Vec<_>>();
                    let dw = (0..n * k)
                        .map(|i| {
                            (0..m)
                                .map(|j| dy[j * n + i / k] as f64 * xv[j * k + i % k] as f64)
                                .sum::<f64>()
                        })
                        .collect::<Vec<_>>();
                    for (budget, context) in contexts.iter().enumerate() {
                        x.zero_grad();
                        w.zero_grad();
                        let y = context.scope(|| matmul(&x, &w));
                        check("row fallback forward", &values(&y), &expected, 8e-6, 8e-6);
                        context.scope(|| y.backward_with_grad(ordered(&[m, n], &dy, layout)));
                        check(
                            "row fallback input VJP",
                            &x.grad().unwrap().iter().copied().collect::<Vec<_>>(),
                            &dx,
                            8e-6,
                            8e-6,
                        );
                        check(
                            "row fallback weight VJP",
                            &w.grad().unwrap().iter().copied().collect::<Vec<_>>(),
                            &dw,
                            8e-6,
                            8e-6,
                        );
                        let inference = context.scope(|| no_grad(|| matmul(&x, &w)));
                        let scale = match inference.native_storage_owned() {
                            TensorStorageOwned::I8(_, scale) => Some(scale),
                            _ => None,
                        };
                        let rounded = expected
                            .iter()
                            .map(|&value| match inference.dtype() {
                                DType::F16 => half::f16::from_f32(value as f32).to_f32() as f64,
                                DType::BF16 => half::bf16::from_f32(value as f32).to_f32() as f64,
                                DType::I8 => value,
                                DType::F32 => value,
                            })
                            .collect::<Vec<_>>();
                        let (atol, rtol) = match inference.dtype() {
                            DType::F16 => (2e-5, 1.0 / 1024.),
                            DType::BF16 => (2e-5, 1.0 / 128.),
                            DType::I8 => (scale.unwrap() as f64 * 0.5001 + 8e-6, 8e-6),
                            DType::F32 => (8e-6, 8e-6),
                        };
                        check(
                            "row fallback native inference",
                            &values(&inference),
                            &rounded,
                            atol,
                            rtol,
                        );
                        if budget == 2 && std::env::var_os("LUMEN_OPERATOR_AUDIT").is_some() {
                            println!(
                                "ROW_MATMUL_AUDIT={}",
                                json!({"case":case,"input_dtype":format!("{xd:?}"),"weight_dtype":format!("{wd:?}"),"layout":layout,"x":snapshot(&x),"w":snapshot(&w),"y":values(&y),"dy":dy,"infer":values(&inference),"infer_dtype":format!("{:?}",inference.dtype()),"infer_scale":scale})
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn matrix_row_tiles_keep_layouts_precision_and_gradients() {
    let contexts = contexts();
    for dtype in [DType::F32, DType::F16, DType::BF16] {
        for layout in 0..3 {
            for batched in [false, true] {
                let (xs, ws) = if dtype == DType::F32 && batched {
                    (vec![1, 1, 257, 129], vec![1, 1, 129, 257])
                } else if dtype == DType::F32 {
                    (vec![129, 513], vec![129, 513])
                } else if batched {
                    (vec![1, 1, 129, 33], vec![1, 1, 33, 65])
                } else {
                    (vec![129, 65], vec![33, 65])
                };
                let x = make(&xs, dtype, layout, 3);
                let w = make(&ws, dtype, layout, 7);
                let mut expected: Option<Vec<f32>> = None;
                let mut expected_gradients: Option<[Vec<f32>; 2]> = None;
                for (budget, context) in contexts.iter().enumerate() {
                    x.zero_grad();
                    w.zero_grad();
                    let y = context.scope(|| {
                        if batched {
                            batch_matmul(&x, &w)
                        } else {
                            matmul(&x, &w)
                        }
                    });
                    let actual = values(&y);
                    if let Some(reference) = &expected {
                        check(
                            "matrix task budget",
                            &actual,
                            &reference.iter().map(|&v| v as f64).collect::<Vec<_>>(),
                            5e-6,
                            5e-6,
                        );
                    } else {
                        expected = Some(actual.clone());
                    }
                    let dy = ordered(&y.shape_vec(), &data(y.len(), 11), layout);
                    context.scope(|| y.backward_with_grad(dy.clone()));
                    let gradients =
                        [&x, &w].map(|t| t.grad().unwrap().iter().copied().collect::<Vec<_>>());
                    if let Some(reference) = &expected_gradients {
                        for (got, reference) in gradients.iter().zip(reference) {
                            check(
                                "matrix VJP task budget",
                                got,
                                &reference.iter().map(|&v| v as f64).collect::<Vec<_>>(),
                                5e-6,
                                5e-6,
                            );
                        }
                    } else {
                        expected_gradients = Some(gradients);
                    }
                    assert!(
                        x.grad()
                            .unwrap()
                            .iter()
                            .chain(w.grad().unwrap().iter())
                            .all(|v| v.is_finite())
                    );
                    let infer = context.scope(|| {
                        no_grad(|| {
                            if batched {
                                batch_matmul(&x, &w)
                            } else {
                                matmul(&x, &w)
                            }
                        })
                    });
                    if budget == 2 {
                        trace(
                            json!({"op":if batched{"batch_matmul"}else{"matmul"},"dtype":format!("{dtype:?}"),"layout":layout,"x":snapshot(&x),"w":snapshot(&w),"y":actual,"dy":dy.iter().copied().collect::<Vec<_>>(),"inference_y":values(&infer)}),
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn f32_strided_prefill_weights_match_full_f64_without_repacking() {
    for (m, k, n) in [(32, 513, 517), (17, 513, 2049)] {
        for layout in 0..3 {
            let x = make(&[m, k], DType::F32, layout, 3);
            let w = make(&[n, k], DType::F32, layout, 7);
            let xv = values(&x);
            let wv = values(&w);
            let expected = (0..m * n)
                .map(|i| {
                    let (row, col) = (i / n, i % n);
                    (0..k)
                        .map(|j| xv[row * k + j] as f64 * wv[col * k + j] as f64)
                        .sum::<f64>()
                })
                .collect::<Vec<_>>();
            for context in contexts() {
                let y = context.scope(|| no_grad(|| matmul(&x, &w)));
                check(
                    "F32 strided prefill GEMM",
                    &values(&y),
                    &expected,
                    8e-6,
                    8e-6,
                );
            }
        }
    }
}

#[test]
fn batch_head_tiles_and_zero_work_match_scalar() {
    for (b, h, m, k, n) in [
        (2, 3, 129, 65, 193),
        (1, 2, 33, 7, 65),
        (0, 2, 3, 4, 5),
        (2, 1, 0, 3, 4),
        (1, 1, 3, 0, 4),
        (1, 1, 3, 4, 0),
    ] {
        for layout in 0..3 {
            let x = make(&[b, h, m, k], DType::F32, layout, 3);
            let w = make(&[b, h, k, n], DType::F32, layout, 7);
            let xv = values(&x);
            let wv = values(&w);
            let expected = (0..b * h * m * n)
                .map(|i| {
                    let head = i / (m * n);
                    let row = i / n % m;
                    let col = i % n;
                    (0..k)
                        .map(|j| {
                            xv[(head * m + row) * k + j] as f64
                                * wv[(head * k + j) * n + col] as f64
                        })
                        .sum::<f64>()
                })
                .collect::<Vec<_>>();
            for context in contexts() {
                let y = context.scope(|| no_grad(|| batch_matmul(&x, &w)));
                check("batch head/empty tiles", &values(&y), &expected, 5e-6, 5e-6);
            }
        }
    }
}

#[test]
fn tiled_inference_attention_masks_offsets_and_value_widths() {
    for (case, (b, h, hk, q, k, d, dv, kind, causal, offset)) in [
        (1, 1, 1, 129, 97, 7, 5, 0, false, 0),
        (2, 6, 2, 65, 71, 5, 3, 1, false, 0),
        (1, 3, 1, 131, 129, 7, 11, 1, true, 19),
        (1, 2, 1, 129, 193, 9, 5, 2, true, 33),
        (1, 3, 1, 1, 1025, 13, 7, 2, true, 700),
    ]
    .into_iter()
    .enumerate()
    {
        for layout in 0..3 {
            let query = make(&[b, h, q, d], DType::F32, layout, 3);
            let key = make(&[b, hk, k, d], DType::F32, layout, 7);
            let value = make(&[b, hk, k, dv], DType::F32, layout, 11);
            let keep = ArrayD::from_shape_fn(IxDyn(&[b, 1, q, k]), |i| {
                i[3] % 7 != 2 && i[2] % 65 != 64 && !(b > 1 && i[0] == b - 1)
            });
            let bias_values = (0..q * k)
                .map(|i| {
                    if i / k == 64 || i % k % 7 == 2 {
                        f32::NEG_INFINITY
                    } else {
                        data(1, i % 251)[0]
                    }
                })
                .collect::<Vec<_>>();
            let bias = tensor(&[q, k], &bias_values, DType::F32, layout, Device::Cpu);
            let options = AttentionOptions {
                scale: Some(0.37),
                causal,
                query_offset: offset,
            };
            let invoke = || {
                no_grad(|| {
                    scaled_dot_product_attention(
                        &query,
                        &key,
                        &value,
                        match kind {
                            1 => Some(AttentionMask::Keep(&keep)),
                            2 => Some(AttentionMask::Additive(&bias)),
                            _ => None,
                        },
                        options,
                    )
                })
            };
            let mut reference: Option<Vec<f32>> = None;
            for (budget, context) in contexts().into_iter().enumerate() {
                let y = context.scope(invoke);
                assert!(!y.requires_grad());
                let actual = values(&y);
                if let Some(reference) = &reference {
                    check(
                        "attention task budget",
                        &actual,
                        &reference.iter().map(|&v| v as f64).collect::<Vec<_>>(),
                        4e-6,
                        4e-6,
                    );
                } else {
                    reference = Some(actual.clone());
                }
                if budget == 2 {
                    trace(
                        json!({"op":"attention","dtype":"F32","case":case,"layout":layout,"q":snapshot(&query),"k":snapshot(&key),"v":snapshot(&value),"keep":if kind==1{Some(json!({"shape":keep.shape(),"values":keep.iter().copied().collect::<Vec<_>>()}))}else{None},"bias":if kind==2{Some(snapshot(&bias))}else{None},"scale":0.37f32,"causal":causal,"offset":offset,"output_shape":y.shape_vec(),"y":actual}),
                    );
                }
            }
            assert!(
                query.grad().is_none()
                    && key.grad().is_none()
                    && value.grad().is_none()
                    && bias.grad().is_none()
            );
        }
    }
}

#[test]
fn tiled_attention_oversize_key_row_matches_f64() {
    let k = 65537;
    let query = make(&[1, 1, 1, 3], DType::F32, 0, 3);
    let key = make(&[1, 1, k, 3], DType::F32, 0, 7);
    let value = make(&[1, 1, k, 2], DType::F32, 0, 11);
    let q = values(&query);
    let keys = values(&key);
    let v = values(&value);
    let scores = (0..k)
        .map(|j| {
            (0..3)
                .map(|d| q[d] as f64 * keys[j * 3 + d] as f64)
                .sum::<f64>()
                * (3f32).sqrt().recip() as f64
        })
        .collect::<Vec<_>>();
    let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let sum = scores.iter().map(|s| (s - maximum).exp()).sum::<f64>();
    let expected = (0..2)
        .map(|d| {
            scores
                .iter()
                .enumerate()
                .map(|(j, s)| (s - maximum).exp() / sum * v[j * 2 + d] as f64)
                .sum::<f64>()
        })
        .collect::<Vec<_>>();
    let y = no_grad(|| {
        scaled_dot_product_attention(&query, &key, &value, None, AttentionOptions::default())
    });
    check("oversize key row", &values(&y), &expected, 5e-6, 5e-6);
    let y = no_grad(|| {
        scaled_dot_product_attention(
            &query,
            &key,
            &value,
            None,
            AttentionOptions {
                causal: true,
                query_offset: usize::MAX,
                ..Default::default()
            },
        )
    });
    check(
        "saturating causal query offset",
        &values(&y),
        &expected,
        5e-6,
        5e-6,
    );
}
