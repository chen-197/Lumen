use super::*;

#[cfg(feature = "cuda")]
#[test]
fn fused_lowp_i8_prefill_respects_activation_quantization_policy() {
    use crate::precision::{
        InferenceActivationQuantization as Policy, with_inference_activation_quantization,
    };
    if !is_available() {
        return;
    }
    let _enabled = set_enabled_scoped(true);
    for (m, n, k) in [(8usize, 16usize, 16usize), (8, 128, 1024), (128, 128, 128)] {
        let g: Vec<i8> = (0..n * k).map(|i| ((i * 7) % 19) as i8 - 9).collect();
        let u: Vec<i8> = (0..n * k).map(|i| ((i * 11) % 17) as i8 - 8).collect();
        let gate = upload_i8_storage(&g).unwrap();
        let up = upload_i8_storage(&u).unwrap();
        for dtype in [crate::precision::DType::F16, crate::precision::DType::BF16] {
            let raw: Vec<f32> = (0..m * k)
                .map(|i| ((i % 23) as f32 - 11.0) * 0.091 + 0.003)
                .collect();
            let bits: Vec<u16> = raw
                .iter()
                .map(|&v| match dtype {
                    crate::precision::DType::F16 => half::f16::from_f32(v).to_bits(),
                    _ => half::bf16::from_f32(v).to_bits(),
                })
                .collect();
            let x: Vec<f32> = bits
                .iter()
                .map(|&v| match dtype {
                    crate::precision::DType::F16 => half::f16::from_bits(v).to_f32(),
                    _ => half::bf16::from_bits(v).to_f32(),
                })
                .collect();
            let input = upload_u16_storage(&bits).unwrap();
            let mut results = Vec::new();
            for policy in [Policy::Disabled, Policy::AllowDynamicI8] {
                let (silu, qkv) = with_inference_activation_quantization(policy, || {
                    let silu = fused_gate_up_silu_typed_buffer(
                        &input,
                        dtype,
                        None,
                        &gate,
                        crate::precision::DType::I8,
                        Some(0.021),
                        &up,
                        Some(0.017),
                        m,
                        n,
                        k,
                    )
                    .unwrap();
                    let qkv = fused_qkv_typed_buffer(
                        &input,
                        dtype,
                        None,
                        &gate,
                        &up,
                        &gate,
                        crate::precision::DType::I8,
                        Some(0.021),
                        Some(0.017),
                        Some(0.009),
                        m,
                        n,
                        n,
                        k,
                    )
                    .unwrap();
                    (
                        download_f32(&silu).unwrap(),
                        [
                            download_f32(&qkv.0).unwrap(),
                            download_f32(&qkv.1).unwrap(),
                            download_f32(&qkv.2).unwrap(),
                        ],
                    )
                });
                for row in 0..m {
                    let values = &x[row * k..(row + 1) * k];
                    let scale = values.iter().map(|v| v.abs()).fold(0.0f32, f32::max) / 127.0;
                    let quantized: Vec<i32> = values
                        .iter()
                        .map(|&v| (v / scale).round_ties_even().clamp(-127.0, 127.0) as i32)
                        .collect();
                    for col in 0..n {
                        let projection = |weight: &[i8], ws: f32| -> f64 {
                            if policy == Policy::AllowDynamicI8 {
                                let dot: i32 = quantized
                                    .iter()
                                    .zip(&weight[col * k..(col + 1) * k])
                                    .map(|(&a, &b)| a * i32::from(b))
                                    .sum();
                                f64::from(dot as f32 * scale * ws)
                            } else {
                                values
                                    .iter()
                                    .zip(&weight[col * k..(col + 1) * k])
                                    .map(|(&a, &b)| f64::from(a) * f64::from(f32::from(b) * ws))
                                    .sum()
                            }
                        };
                        let gr = projection(&g, 0.021);
                        let ur = projection(&u, 0.017);
                        let expected = gr / (1.0 + (-gr).exp()) * ur;
                        assert!(
                            (f64::from(silu[row * n + col]) - expected).abs()
                                <= 3e-5 + 3e-5 * expected.abs(),
                            "{dtype:?} {policy:?} gate/up"
                        );
                        for (got, expected) in qkv.iter().zip([gr, ur, projection(&g, 0.009)]) {
                            assert!(
                                (f64::from(got[row * n + col]) - expected).abs()
                                    <= 3e-5 + 3e-5 * expected.abs(),
                                "{dtype:?} {policy:?} qkv"
                            );
                        }
                    }
                }
                results.push(silu);
            }
            assert!(
                results[0]
                    .iter()
                    .zip(&results[1])
                    .any(|(a, b)| (a - b).abs() > 1e-5)
            );
        }
        assert!(!crate::precision::allow_dynamic_i8_activations());
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_gate_up_f32_i8_matvec_matches_f64_at_thresholds_and_tails() {
    use crate::precision::DType;
    if !is_available() {
        return;
    }
    let mut shapes = Vec::new();
    for n in [1, 3, 127, 128, 129, 5632] {
        for k in [
            1, 7, 31, 32, 33, 127, 128, 129, 255, 256, 257, 1025, 2048, 5632,
        ] {
            shapes.push((n, k));
        }
    }
    shapes.push((65537, 128));
    let mut cases = 0;
    let mut values = 0;
    let mut max_absolute_error = 0.0f64;
    for (n, k) in shapes {
        for revision in 0..2 {
            let (gs, us) = if revision == 0 {
                (0.013f32, 0.017f32)
            } else {
                (0.023, 0.009)
            };
            let mut state = 1234567u32 + revision;
            let mut next = || {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                state
            };
            let x: Vec<_> = (0..k)
                .map(|_| ((next() >> 16) as i32 - 32768) as f32 / 131072.0)
                .collect();
            let g: Vec<_> = (0..n * k)
                .map(|_| ((next() >> 24) as i32 - 128) as i8)
                .collect();
            let u: Vec<_> = (0..n * k)
                .map(|_| ((next() >> 24) as i32 - 128) as i8)
                .collect();
            let input = upload_f32(&x).unwrap();
            let gate = upload_i8_storage(&g).unwrap();
            let up = upload_i8_storage(&u).unwrap();
            let output = fused_gate_up_silu_typed_buffer(
                &input,
                DType::F32,
                None,
                &gate,
                DType::I8,
                Some(gs),
                &up,
                Some(us),
                1,
                n,
                k,
            )
            .unwrap();
            let got = download_f32(&output).unwrap();
            for col in 0..n {
                let mut gate_ref = 0.0f64;
                let mut up_ref = 0.0f64;
                for d in 0..k {
                    gate_ref += f64::from(x[d]) * f64::from(f32::from(g[col * k + d]) * gs);
                    up_ref += f64::from(x[d]) * f64::from(f32::from(u[col * k + d]) * us);
                }
                let expected = gate_ref / (1.0 + (-gate_ref).exp()) * up_ref;
                let error = (f64::from(got[col]) - expected).abs();
                max_absolute_error = max_absolute_error.max(error);
                assert!(
                    got[col].is_finite() && error <= 3e-4 + 5e-5 * expected.abs(),
                    "gate/up n={n} k={k} revision={revision} col={col}: {} vs {expected}, error={error}",
                    got[col]
                );
            }
            cases += 1;
            values += n;
        }
    }
    println!(
        "gate_up_f64 {}",
        serde_json::json!({"cases":cases,"values":values,"max_absolute_error":max_absolute_error})
    );
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_gate_up_f32_i8_matvec_preserves_tiny_inputs_and_nonfinite_signals() {
    use crate::precision::DType;
    if !is_available() {
        return;
    }
    for k in [128, 129, 2048, 2049] {
        let n = 129;
        let mut x = vec![0.0f32; k];
        x[0] = 1000.0;
        x[1] = 1e-4;
        x[2] = -7e-5;
        let mut g = vec![0i8; n * k];
        let mut u = vec![0i8; n * k];
        for col in 0..n {
            g[col * k + 1] = 79;
            g[col * k + 2] = -101;
            u[col * k + 1] = -81;
            u[col * k + 2] = 103;
        }
        let gate = upload_i8_storage(&g).unwrap();
        let up = upload_i8_storage(&u).unwrap();
        let input = upload_f32(&x).unwrap();
        let output = fused_gate_up_silu_typed_buffer(
            &input,
            DType::F32,
            None,
            &gate,
            DType::I8,
            Some(0.013),
            &up,
            Some(0.017),
            1,
            n,
            k,
        )
        .unwrap();
        let gr = f64::from(x[1]) * f64::from(79.0f32 * 0.013)
            + f64::from(x[2]) * f64::from(-101.0f32 * 0.013);
        let ur = f64::from(x[1]) * f64::from(-81.0f32 * 0.017)
            + f64::from(x[2]) * f64::from(103.0f32 * 0.017);
        let expected = gr / (1.0 + (-gr).exp()) * ur;
        assert!(expected.abs() > 1e-9);
        for value in download_f32(&output).unwrap() {
            assert!((f64::from(value) - expected).abs() < 1e-11);
        }
        for nonfinite in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            x.fill(0.01);
            x[1] = nonfinite;
            let input = upload_f32(&x).unwrap();
            let output = fused_gate_up_silu_typed_buffer(
                &input,
                DType::F32,
                None,
                &gate,
                DType::I8,
                Some(0.013),
                &up,
                Some(0.017),
                1,
                n,
                k,
            )
            .unwrap();
            assert!(
                download_f32(&output)
                    .unwrap()
                    .iter()
                    .all(|v| !v.is_finite())
            );
        }
    }
}

#[cfg(feature = "cuda")]
fn attention_f64_reference(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    batch: usize,
    heads: usize,
    kv_heads: usize,
    qs: usize,
    active: usize,
    cache: usize,
    dim: usize,
    past: usize,
    scale: f32,
    causal: bool,
) -> Vec<f64> {
    let mut output = vec![0.0; batch * qs * heads * dim];
    for b in 0..batch {
        for h in 0..heads {
            let base = (b * kv_heads + h / (heads / kv_heads)) * cache * dim;
            for sq in 0..qs {
                let visible = if causal {
                    active.min(past + sq + 1)
                } else {
                    active
                };
                let q_base = ((b * heads + h) * qs + sq) * dim;
                let scores: Vec<_> = (0..visible)
                    .map(|pos| {
                        (0..dim)
                            .map(|d| f64::from(q[q_base + d]) * f64::from(k[base + pos * dim + d]))
                            .sum::<f64>()
                            * f64::from(scale)
                    })
                    .collect();
                let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let weights: Vec<_> = scores.iter().map(|s| (s - maximum).exp()).collect();
                let denominator = weights.iter().sum::<f64>() + 1e-9;
                for d in 0..dim {
                    output[((b * qs + sq) * heads + h) * dim + d] = weights
                        .iter()
                        .enumerate()
                        .map(|(pos, &weight)| weight * f64::from(v[base + pos * dim + d]))
                        .sum::<f64>()
                        / denominator;
                }
            }
        }
    }
    output
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_online_attention_matches_f64_across_layouts_masks_gqa_and_tails() {
    if !is_available() {
        return;
    }
    let mut cases = 0;
    let mut max_error = 0.0f64;
    let mut state = 178523u32;
    let mut random = || {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        ((state >> 16) as i32 - 32768) as f32 / 16384.0
    };
    for dim in [1, 7, 31, 32, 33, 63, 64, 65, 96, 127, 128, 129, 257] {
        let mut shapes = vec![
            (1, 1, 1, 1, 0, 1, 5),
            (2, 6, 2, 3, 5, 9, 13),
            (1, 4, 1, 5, 0, 5, 9),
            (2, 4, 4, 3, 31, 35, 39),
            (1, 8, 2, 3, 175, 178, 193),
            (1, 3, 1, 3, 512, 515, 529),
        ];
        if dim == 64 {
            shapes.extend([
                (1, 32, 4, 46, 0, 46, 64),
                (1, 32, 4, 178, 0, 178, 193),
                (1, 3, 1, 1, 4096, 4097, 4113),
            ]);
        }
        for (batch, heads, kv_heads, qs, past, active, cache) in shapes {
            let q: Vec<_> = (0..batch * heads * qs * dim).map(|_| random()).collect();
            let mut k = vec![f32::NAN; batch * kv_heads * cache * dim];
            let mut v = k.clone();
            for head in 0..batch * kv_heads {
                for pos in 0..active {
                    for d in 0..dim {
                        k[(head * cache + pos) * dim + d] = random();
                        v[(head * cache + pos) * dim + d] = random();
                    }
                }
            }
            let qb = upload_f32(&q).unwrap();
            let kb = upload_f32(&k).unwrap();
            let vb = upload_f32(&v).unwrap();
            let decode_q: Vec<_> = (0..batch * heads)
                .flat_map(|h| {
                    q[(h * qs + qs - 1) * dim..(h * qs + qs) * dim]
                        .iter()
                        .copied()
                })
                .collect();
            let dqb = upload_f32(&decode_q).unwrap();
            for factor in [1.0f32, -0.25, 0.0] {
                let scale = factor / (dim as f32).sqrt();
                for causal in [false, true] {
                    let out = prefill_attention_f32_buffer(
                        &qb,
                        &kb,
                        &vb,
                        batch,
                        heads,
                        kv_heads,
                        qs,
                        active,
                        cache,
                        dim,
                        heads / kv_heads,
                        past,
                        scale,
                        causal,
                    )
                    .unwrap();
                    let got = download_f32(&out).unwrap();
                    let reference = attention_f64_reference(
                        &q, &k, &v, batch, heads, kv_heads, qs, active, cache, dim, past, scale,
                        causal,
                    );
                    for (i, (&actual, expected)) in got.iter().zip(reference).enumerate() {
                        let error = (f64::from(actual) - expected).abs();
                        max_error = max_error.max(error);
                        assert!(
                            error <= 2e-5 + 3e-5 * expected.abs(),
                            "prefill D={dim} B={batch} H={heads} KV={kv_heads} Q={qs} active={active} causal={causal} scale={scale} index={i}: {actual} vs {expected}"
                        );
                    }
                    cases += 1;
                }
                let out = decode_attention_f32_buffer(
                    &dqb,
                    &kb,
                    &vb,
                    batch,
                    heads,
                    kv_heads,
                    active,
                    cache,
                    dim,
                    heads / kv_heads,
                    scale,
                )
                .unwrap();
                let got = download_f32(&out).unwrap();
                let reference = attention_f64_reference(
                    &decode_q, &k, &v, batch, heads, kv_heads, 1, active, cache, dim, 0, scale,
                    false,
                );
                for (i, (&actual, expected)) in got.iter().zip(reference).enumerate() {
                    let error = (f64::from(actual) - expected).abs();
                    max_error = max_error.max(error);
                    assert!(
                        error <= 2e-5 + 3e-5 * expected.abs(),
                        "decode D={dim} B={batch} H={heads} KV={kv_heads} active={active} scale={scale} index={i}: {actual} vs {expected}"
                    );
                }
                cases += 1;
            }
        }
    }
    println!("online_attention_f64 cases={cases} max_absolute_error={max_error:e}");
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_online_attention_preserves_nonfinite_signals_and_ignores_masked_cache() {
    if !is_available() {
        return;
    }
    let dim = 64;
    let active = 33;
    let cache = 40;
    let mut q = vec![0.0f32; dim];
    q[0] = 1.0;
    for inf_positions in [vec![0], vec![0, 1], vec![0, 8]] {
        let mut k = vec![0.0; cache * dim];
        let mut v: Vec<_> = (0..cache * dim).map(|i| (i % 101) as f32 / 101.0).collect();
        for &pos in &inf_positions {
            k[pos * dim] = f32::INFINITY;
        }
        k[active * dim..].fill(f32::NAN);
        v[active * dim..].fill(f32::NAN);
        let qb = upload_f32(&q).unwrap();
        let kb = upload_f32(&k).unwrap();
        let vb = upload_f32(&v).unwrap();
        let decode =
            decode_attention_f32_buffer(&qb, &kb, &vb, 1, 1, 1, active, cache, dim, 1, 1.0)
                .unwrap();
        let prefill = prefill_attention_f32_buffer(
            &qb,
            &kb,
            &vb,
            1,
            1,
            1,
            1,
            active,
            cache,
            dim,
            1,
            active - 1,
            1.0,
            true,
        )
        .unwrap();
        for out in [decode, prefill] {
            let got = download_f32(&out).unwrap();
            if inf_positions.len() == 1 {
                assert_eq!(got, v[..dim]);
            } else {
                assert!(got.iter().all(|x| x.is_nan()));
            }
        }
    }
    let mut k = vec![0.0; cache * dim];
    let v = vec![0.5; cache * dim];
    k[2 * dim] = f32::NAN;
    let qb = upload_f32(&[q.clone(), q].concat()).unwrap();
    let kb = upload_f32(&k).unwrap();
    let vb = upload_f32(&v).unwrap();
    let masked = prefill_attention_f32_buffer(
        &qb, &kb, &vb, 1, 1, 1, 2, active, cache, dim, 1, 0, 1.0, true,
    )
    .unwrap();
    assert!(
        download_f32(&masked)
            .unwrap()
            .iter()
            .all(|&x| (x - 0.5).abs() < 1e-6)
    );
    let unmasked = prefill_attention_f32_buffer(
        &qb, &kb, &vb, 1, 1, 1, 2, active, cache, dim, 1, 0, 1.0, false,
    )
    .unwrap();
    assert!(download_f32(&unmasked).unwrap().iter().all(|x| x.is_nan()));

    let mut q = vec![0.0; dim];
    q[0] = 1.0;
    let qb = upload_f32(&q).unwrap();
    let mut k = vec![0.0; cache * dim];
    let mut v: Vec<_> = (0..cache * dim).map(|i| (i % 113) as f32 / 113.0).collect();
    for pos in 0..active {
        k[pos * dim] = -10000.0;
    }
    k[9 * dim] = 10000.0;
    k[18 * dim] = 9999.0;
    k[active * dim..].fill(f32::NAN);
    v[active * dim..].fill(f32::NAN);
    let kb = upload_f32(&k).unwrap();
    let vb = upload_f32(&v).unwrap();
    let reference =
        attention_f64_reference(&q, &k, &v, 1, 1, 1, 1, active, cache, dim, 0, 1.0, false);
    let out =
        decode_attention_f32_buffer(&qb, &kb, &vb, 1, 1, 1, active, cache, dim, 1, 1.0).unwrap();
    for (&actual, expected) in download_f32(&out).unwrap().iter().zip(reference) {
        assert!((f64::from(actual) - expected).abs() < 1e-6);
    }
    for pos in 0..active {
        k[pos * dim] = f32::NEG_INFINITY;
    }
    let kb = upload_f32(&k).unwrap();
    let out =
        decode_attention_f32_buffer(&qb, &kb, &vb, 1, 1, 1, active, cache, dim, 1, 1.0).unwrap();
    assert!(download_f32(&out).unwrap().iter().all(|&x| x == 0.0));
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_online_attention_rejects_inconsistent_gqa_head_counts() {
    if !is_available() {
        return;
    }
    for (heads, kv_heads, repeats) in [(6, 2, 1), (5, 2, 2), (4, 1, 2)] {
        let q = upload_f32(&vec![0.0; heads * 64]).unwrap();
        let kv = upload_f32(&vec![0.0; kv_heads * 8 * 64]).unwrap();
        let decode =
            decode_attention_f32_buffer(&q, &kv, &kv, 1, heads, kv_heads, 3, 8, 64, repeats, 1.0)
                .err()
                .expect("invalid decode GQA must fail before launching");
        let prefill = prefill_attention_f32_buffer(
            &q, &kv, &kv, 1, heads, kv_heads, 1, 3, 8, 64, repeats, 2, 1.0, true,
        )
        .err()
        .expect("invalid prefill GQA must fail before launching");
        assert!(decode.contains("num_heads == num_kv_heads * n_rep"));
        assert!(prefill.contains("num_heads == num_kv_heads * n_rep"));
    }
}

#[cfg(feature = "cuda")]
fn rms_norm_test_storage(
    values: &[f32],
    dtype: crate::precision::DType,
    scale: f32,
) -> (CudaBuffer, Vec<f32>, Option<f32>) {
    use crate::precision::DType;
    match dtype {
        DType::F32 => (upload_f32(values).unwrap(), values.to_vec(), None),
        DType::F16 | DType::BF16 => {
            let (bits, decoded): (Vec<_>, Vec<_>) = values
                .iter()
                .map(|&v| {
                    if dtype == DType::F16 {
                        let v = half::f16::from_f32(v);
                        (v.to_bits(), v.to_f32())
                    } else {
                        let v = half::bf16::from_f32(v);
                        (v.to_bits(), v.to_f32())
                    }
                })
                .unzip();
            (upload_u16_storage(&bits).unwrap(), decoded, None)
        }
        DType::I8 => {
            let bytes: Vec<_> = values
                .iter()
                .map(|v| (v / scale).round().clamp(-127.0, 127.0) as i8)
                .collect();
            let decoded = bytes.iter().map(|&v| f32::from(v) * scale).collect();
            (upload_i8_storage(&bytes).unwrap(), decoded, Some(scale))
        }
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_rms_norm_cooperative_matches_f64_for_types_tails_and_grid_stride() {
    use crate::precision::DType;
    if !is_available() {
        return;
    }
    let types = [DType::F32, DType::F16, DType::BF16, DType::I8];
    let mut cases = 0;
    let mut max_error = 0.0f64;
    for &(rows, dim) in &[
        (1, 1),
        (3, 31),
        (1, 32),
        (3, 33),
        (5, 127),
        (3, 128),
        (5, 129),
        (3, 513),
        (1, 2048),
        (46, 2048),
        (3, 4099),
        (65536, 33),
    ] {
        let values: Vec<_> = (0..rows * dim)
            .map(|i| {
                if i / dim % 7 == 1 {
                    0.0
                } else {
                    ((i * 37 % 257) as f32 - 128.0) / 41.0
                }
            })
            .collect();
        let weights: Vec<_> = (0..dim)
            .map(|i| ((i * 13 % 113) as f32 - 56.0) / 37.0)
            .collect();
        for &input_type in &types {
            for &weight_type in &types {
                // One type pair suffices to exercise the capped-grid row loop.
                if rows > 65535 && (input_type != DType::BF16 || weight_type != DType::I8) {
                    continue;
                }
                let (input, x, xs) = rms_norm_test_storage(&values, input_type, 0.03125);
                let (weight, w, ws) = rms_norm_test_storage(&weights, weight_type, 0.015625);
                let output = rms_norm_typed_buffer(
                    &input,
                    input_type,
                    xs,
                    &weight,
                    weight_type,
                    ws,
                    rows,
                    dim,
                    1e-5,
                )
                .unwrap();
                let got = download_f32(&output).unwrap();
                for row in 0..rows {
                    let inv = (x[row * dim..(row + 1) * dim]
                        .iter()
                        .map(|&v| f64::from(v).powi(2))
                        .sum::<f64>()
                        / dim as f64
                        + f64::from(1e-5f32))
                    .sqrt()
                    .recip();
                    for j in 0..dim {
                        let reference = f64::from(x[row * dim + j]) * inv * f64::from(w[j]);
                        let error = (f64::from(got[row * dim + j]) - reference).abs();
                        max_error = max_error.max(error);
                        assert!(
                            error <= 2e-6 + 3e-6 * reference.abs(),
                            "{rows}x{dim} {input_type:?}/{weight_type:?}, row {row} col {j}: {} vs {reference}",
                            got[row * dim + j]
                        );
                    }
                }
                cases += 1;
            }
        }
    }
    println!("cooperative RMSNorm: {cases} cases, max absolute F64 error {max_error:e}");
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_rms_norm_cooperative_preserves_nonfinite_rows() {
    use crate::precision::DType;
    if !is_available() {
        return;
    }
    for dtype in [DType::F32, DType::F16, DType::BF16] {
        let mut values = vec![1.0; 3 * 513];
        values[512] = f32::NAN;
        values[513 + 256] = f32::INFINITY;
        let (input, _, scale) = rms_norm_test_storage(&values, dtype, 1.0);
        let weight = upload_i8_storage(&vec![64; 513]).unwrap();
        let out = rms_norm_typed_buffer(
            &input,
            dtype,
            scale,
            &weight,
            DType::I8,
            Some(0.015625),
            3,
            513,
            1e-5,
        )
        .unwrap();
        let got = download_f32(&out).unwrap();
        assert!(got[..513].iter().all(|x| x.is_nan()));
        assert!(got[513 + 256].is_nan());
        assert!(
            got[2 * 513..]
                .iter()
                .all(|x| x.is_finite() && (*x - 1.0).abs() < 1e-5)
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_rms_norm_cooperative_squared_loss_gradients_match_f64() {
    use crate::precision::DType;
    if !is_available() {
        return;
    }
    let rows = 3;
    let dim = 2049;
    let values: Vec<_> = (0..rows * dim)
        .map(|i| ((i * 17 % 211) as f32 - 105.0) / 43.0)
        .collect();
    let weights: Vec<_> = (0..dim)
        .map(|i| ((i * 11 % 127) as f32 - 63.0) / 63.0)
        .collect();
    let (input, x, xs) = rms_norm_test_storage(&values, DType::BF16, 1.0);
    let (weight, w, ws) = rms_norm_test_storage(&weights, DType::I8, 0.015625);
    let output = rms_norm_typed_buffer(
        &input,
        DType::BF16,
        xs,
        &weight,
        DType::I8,
        ws,
        rows,
        dim,
        1e-5,
    )
    .unwrap();
    let y = download_f32(&output).unwrap();
    // d(0.5 * sum(y^2))/dy = y couples the changed forward to backward.
    let ((_, dx), (_, dw)) = rms_norm_backward_typed(
        &input,
        DType::BF16,
        xs,
        &weight,
        DType::I8,
        ws,
        &upload_f32(&y).unwrap(),
        rows,
        dim,
        1e-5,
    )
    .unwrap();
    let mut ref_dw = vec![0.0f64; dim];
    for row in 0..rows {
        let xr = &x[row * dim..(row + 1) * dim];
        let inv = (xr.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>() / dim as f64
            + f64::from(1e-5f32))
        .sqrt()
        .recip();
        let dot = xr
            .iter()
            .zip(&w)
            .map(|(&x, &w)| (f64::from(x) * inv * f64::from(w)).powi(2))
            .sum::<f64>()
            / dim as f64;
        for j in 0..dim {
            let xn = f64::from(xr[j]) * inv;
            let reference = inv * (xn * f64::from(w[j]).powi(2) - xn * dot);
            assert!((f64::from(dx[row * dim + j]) - reference).abs() < 2e-5);
            ref_dw[j] += xn * xn * f64::from(w[j]);
        }
    }
    for (&got, reference) in dw.iter().zip(ref_dw) {
        assert!((f64::from(got) - reference).abs() < 4e-5);
    }
}

#[test]
fn cuda_enabled_flag_is_thread_local() {
    set_enabled(false);

    let (tx, rx) = std::sync::mpsc::channel();
    let handle = std::thread::spawn(move || {
        set_enabled(true);
        tx.send(is_enabled_requested())
            .expect("send thread-local CUDA enabled state");
        std::thread::sleep(std::time::Duration::from_millis(25));
        tx.send(is_enabled_requested())
            .expect("send thread-local CUDA enabled state");
        set_enabled(false);
    });

    assert!(
        rx.recv().expect("receive spawned thread state"),
        "spawned thread should observe its own CUDA enabled flag"
    );
    assert!(
        !is_enabled_requested(),
        "main thread should not inherit spawned thread's CUDA enabled flag"
    );
    assert!(
        rx.recv().expect("receive spawned thread state"),
        "spawned thread should keep CUDA enabled until it resets"
    );
    assert!(
        !is_enabled_requested(),
        "main thread should still keep its own CUDA disabled flag"
    );

    handle.join().expect("join CUDA enabled thread");
    set_enabled(false);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_shape_helpers_reject_overflow_and_zero_stride() {
    assert!(imp::checked_len("test length", &[usize::MAX, 2]).is_err());
    assert!(imp::conv_output_dim(4, 0, 3, 0, "test conv").is_err());
    assert!(imp::conv_output_dim(4, usize::MAX, 3, 1, "test conv").is_err());
    assert!(!imp::range_fits(usize::MAX, 1, usize::MAX));
    assert!(imp::range_fits(3, 2, 5));
    assert!(imp::ensure_finite("test scale", f32::NAN).is_err());
    assert!(imp::ensure_positive_finite("test epsilon", 0.0).is_err());
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_release_cached_memory_allows_cublas_recreation() {
    if !is_available() {
        return;
    }

    let run_matmul = || {
        let lhs = upload_f32(&[1.0, 2.0, 3.0, 4.0]).expect("upload matmul lhs");
        let rhs = upload_f32(&[5.0, 6.0, 7.0, 8.0]).expect("upload matmul rhs");
        let output = matmul_f32_no_host(&lhs, &rhs, 2, 2, 2).expect("run CUDA matmul");
        download_f32(&output).expect("download CUDA matmul output")
    };

    assert_eq!(run_matmul(), vec![17.0, 23.0, 39.0, 53.0]);
    release_cached_memory().expect("release CUDA cached memory");
    assert_eq!(run_matmul(), vec![17.0, 23.0, 39.0, 53.0]);
    release_cached_memory().expect("release recreated CUDA cached memory");
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_sum_lastdim_rejects_dimension_overflow_before_ffi() {
    if !is_available() {
        return;
    }

    let input = upload_f32(&[1.0]).expect("upload sum_lastdim overflow input");
    let error = sum_lastdim_f32_buffer(&input, usize::MAX, 2)
        .err()
        .expect("sum_lastdim dimension overflow must fail");
    assert!(error.contains("overflow"), "unexpected error: {error}");
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_matmul_wrappers_reject_dimension_overflow_before_ffi() {
    if !is_available() {
        return;
    }

    let input = upload_f32(&[1.0]).expect("upload matmul overflow input");
    for error in [
        matmul_f32_no_host(&input, &input, usize::MAX, 1, 2)
            .err()
            .expect("matmul dimension overflow must fail"),
        batch_matmul_f32_no_host(&input, &input, usize::MAX, 2, 1, 1)
            .err()
            .expect("batch_matmul dimension overflow must fail"),
        matmul_backward_f32_no_host(&input, &input, &input, usize::MAX, 2, 1)
            .err()
            .expect("matmul backward dimension overflow must fail"),
        batch_matmul_backward_f32_no_host(&input, &input, &input, usize::MAX, 2, 1, 1)
            .err()
            .expect("batch_matmul backward dimension overflow must fail"),
        softmax_lastdim_f32_no_host(&input, usize::MAX, 2)
            .err()
            .expect("softmax dimension overflow must fail"),
    ] {
        assert!(error.contains("overflow"), "unexpected error: {error}");
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_i8_matmul_paths_reject_combined_scale_overflow() {
    if !is_available() {
        return;
    }

    fn require_error<T>(result: Result<T, String>, message: &str) -> String {
        match result {
            Ok(_) => panic!("{message}"),
            Err(error) => error,
        }
    }

    let input = upload_i8_storage(&[1]).expect("upload i8 scale overflow input");
    for error in [
        require_error(
            matvec_argmax_i8_i8(&input, f32::MAX, &input, 2.0, 1, 1, 1),
            "I8xI8 argmax combined scale overflow must fail",
        ),
        require_error(
            matmul_i8_buffer_no_host(&input, f32::MAX, &input, 2.0, 1, 1, 1),
            "I8xI8 matmul combined scale overflow must fail",
        ),
        require_error(
            matmul_i8_typed_output_buffer_no_host(&input, f32::MAX, &input, 2.0, 1, 1, 1),
            "typed I8xI8 matmul combined scale overflow must fail",
        ),
        require_error(
            batch_matmul_i8_buffer_no_host(&input, f32::MAX, &input, 2.0, 1, 1, 1, 1),
            "I8xI8 batch_matmul combined scale overflow must fail",
        ),
        require_error(
            batch_matmul_i8_typed_output_buffer_no_host(&input, f32::MAX, &input, 2.0, 1, 1, 1, 1),
            "typed I8xI8 batch_matmul combined scale overflow must fail",
        ),
    ] {
        assert!(
            error.contains("combined scale overflow"),
            "unexpected error: {error}"
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_dynamic_i8_quantize_matches_reference_and_handles_zero_absmax() {
    if !is_available() {
        return;
    }

    for values in [
        (0..4099)
            .map(|i| ((i * 37 % 1009) as f32 - 504.0) / 31.0)
            .collect::<Vec<_>>(),
        vec![0.0; 4099],
        vec![f32::from_bits(1); 4099],
    ] {
        let input = upload_f32(&values).expect("upload dynamic i8 quantize input");
        let (output, scale) =
            quantize_f32_to_i8_dynamic_no_host(&input).expect("dynamic i8 quantize");
        let got = download_i8_storage(&output).expect("download dynamic i8 quantize output");

        let max_abs = values.iter().copied().map(f32::abs).fold(0.0f32, f32::max);
        let expected_scale = if max_abs > 0.0 {
            (max_abs / 127.0).max(f32::MIN_POSITIVE)
        } else {
            1.0
        };
        let expected = values
            .iter()
            .map(|value| (value / expected_scale).round().clamp(-127.0, 127.0) as i8)
            .collect::<Vec<_>>();

        assert!((scale - expected_scale).abs() <= 1e-6);
        assert_eq!(got, expected);
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_i8_typed_binary_zero_absmax_uses_unit_scale() {
    if !is_available() {
        return;
    }

    let len = 4099;
    let lhs = upload_i8_storage(&vec![0; len]).expect("upload zero i8 binary lhs");
    let rhs = upload_i8_storage(&vec![0; len]).expect("upload zero i8 binary rhs");
    let (output, scale) =
        binary_i8_typed_output_buffer_no_host(&lhs, 0.03125, &rhs, 0.046875, BinaryOp::Mul)
            .expect("zero typed-output i8 binary");
    let got = download_i8_storage(&output).expect("download zero typed-output i8 binary");

    assert_eq!(scale, 1.0);
    assert_eq!(got, vec![0; len]);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_i8_typed_binary_overflow_keeps_scale_finite() {
    if !is_available() {
        return;
    }

    let lhs = upload_i8_storage(&[1, -1]).expect("upload overflow i8 binary lhs");
    let rhs = upload_i8_storage(&[1, 1]).expect("upload overflow i8 binary rhs");
    let (output, scale) =
        binary_i8_typed_output_buffer_no_host(&lhs, f32::MAX, &rhs, f32::MAX, BinaryOp::Mul)
            .expect("overflow typed-output i8 binary");
    let got = download_i8_storage(&output).expect("download overflow typed-output i8 binary");

    assert_eq!(scale, 1.0);
    assert_eq!(got, vec![127, -127]);
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "performance sanity test; run with --ignored --nocapture"]
fn cuda_dynamic_i8_quantize_perf_smoke() {
    if !is_available() {
        return;
    }

    let len = 1 << 20;
    let values = (0..len)
        .map(|i| ((i * 37 % 1009) as f32 - 504.0) / 31.0)
        .collect::<Vec<_>>();
    let input = upload_f32(&values).expect("upload dynamic i8 quantize perf input");

    let mut samples = Vec::with_capacity(9);
    let _ = quantize_f32_to_i8_dynamic_no_host(&input).expect("warm up dynamic i8 quantize");
    synchronize().expect("sync dynamic i8 quantize warmup");
    for _ in 0..9 {
        let start = std::time::Instant::now();
        let _ = quantize_f32_to_i8_dynamic_no_host(&input)
            .expect("dynamic i8 quantize performance sample");
        synchronize().expect("sync dynamic i8 quantize performance sample");
        samples.push(start.elapsed().as_secs_f64() * 1.0e6);
    }
    samples.sort_by(f64::total_cmp);
    println!(
        "cuda dynamic i8 quantize len={len}: median={:.1}us",
        samples[samples.len() / 2]
    );
}
#[test]
fn cuda_quantization_boundary_preserves_rounding_and_tiny_scales() {
    if !is_available() {
        return;
    }
    for scale in [1.0, f32::from_bits(0x10000)] {
        let values = [
            0.0,
            scale,
            -scale,
            0.5 * scale,
            -0.5 * scale,
            1.5 * scale,
            -1.5 * scale,
            200.0 * scale,
        ];
        let input = upload_f32(&values).unwrap();
        let output = f32_to_i8_storage_no_host(&input, scale).unwrap();
        let expected = values.map(|v| (v as f64 / scale as f64).round().clamp(-127.0, 127.0) as i8);
        assert_eq!(download_i8_storage(&output).unwrap(), expected);
        quantize_f32_storage_no_host(&input, crate::precision::DType::I8, Some(scale)).unwrap();
        let expected = expected.map(|q| q as f32 * scale);
        assert_eq!(download_f32(&input).unwrap(), expected);
    }
    let values = [127.0, 0.5, -0.5, 1.5, -1.5];
    let input = upload_f32(&values).unwrap();
    let (output, scale) = quantize_f32_to_i8_dynamic_no_host(&input).unwrap();
    assert_eq!(scale, 1.0);
    assert_eq!(
        download_i8_storage(&output).unwrap(),
        vec![127, 1, -1, 2, -2]
    );
}
