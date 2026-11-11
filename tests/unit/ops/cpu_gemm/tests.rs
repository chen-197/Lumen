use super::*;

#[test]
fn standard_rhs_packing_preserves_bits_strides_and_overwrites_padding() {
    use ndarray::{Array2, s};
    for (k, n) in [(257, 257), (129, 67), (33, 17), (8, 9), (1, 1)] {
        let weights = Array2::from_shape_fn((k, n), |(kk, col)| {
            bf16::from_bits(((kk * n + col) as u16).wrapping_mul(197))
        });
        let strided = weights.t().as_standard_layout().to_owned().reversed_axes();
        let row = weights.slice(s![0..1, ..]);
        for view in [
            weights.view(),
            strided.view(),
            weights.slice(s![..;-1, ..;-1]),
            row.broadcast((k, n)).unwrap(),
        ] {
            for lanes in [8, 16] {
                let mut packed = vec![bf16::from_bits(u16::MAX); k * lanes];
                for first in (0..n.div_ceil(CHANNELS) * CHANNELS).step_by(lanes) {
                    pack_standard_panel(view, first, lanes, &mut packed);
                    for kk in 0..k {
                        for c in 0..lanes {
                            let expected = if first + c < n {
                                view[(kk, first + c)].to_bits()
                            } else {
                                0
                            };
                            assert_eq!(packed[kk * lanes + c].to_bits(), expected);
                        }
                    }
                }
            }
            let mut pairs = vec![u32::MAX; k.div_ceil(2) * CHANNELS];
            for first in (0..n).step_by(CHANNELS) {
                pack_bf16_standard(view, first, &mut pairs);
                for pair in 0..k.div_ceil(2) {
                    for c in 0..CHANNELS {
                        for half in 0..2 {
                            let expected = if first + c < n && pair * 2 + half < k {
                                view[(pair * 2 + half, first + c)].to_bits()
                            } else {
                                0
                            };
                            assert_eq!(
                                (pairs[pair * CHANNELS + c] >> (half * 16)) as u16,
                                expected
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn half_panel_packing_preserves_all_bits_and_reused_tails() {
    fn check<T: Copy + Default>(from_bits: fn(u16) -> T, to_bits: fn(T) -> u16) {
        let mut packed = Vec::new();
        let cases = std::iter::once((257usize, 257usize)).chain(
            [513usize, 129, 33, 32, 31, 17, 16, 15, 9, 8, 7, 2, 1]
                .into_iter()
                .flat_map(|k| [25usize, 17, 16, 15, 9, 8, 7, 1].map(|n| (k, n))),
        );
        for (k, n) in cases {
            // The 257x257 case covers every 16-bit representation, including
            // signed zero, subnormals, infinities and all NaN payloads.
            let weights: Vec<T> = (0..n * k)
                .map(|i| from_bits((i as u16).wrapping_mul(197)))
                .collect();
            for lanes in [16, 8] {
                packed.resize(k * lanes, from_bits(u16::MAX));
                for first in (0..n.div_ceil(CHANNELS) * CHANNELS).step_by(lanes) {
                    pack_native_panel(&weights, k, n, first, lanes, &mut packed);
                    for kk in 0..k {
                        for c in 0..lanes {
                            let expected = if first + c < n {
                                to_bits(weights[(first + c) * k + kk])
                            } else {
                                to_bits(T::default())
                            };
                            assert_eq!(
                                to_bits(packed[kk * lanes + c]),
                                expected,
                                "K={k} N={n} first={first} lanes={lanes} k={kk} channel={c}"
                            );
                        }
                    }
                }
            }
        }
    }
    check(f16::from_bits, f16::to_bits);
    check(bf16::from_bits, bf16::to_bits);
}

#[test]
fn bf16_weight_panels_preserve_f32_inputs_on_avx2_and_avx512() {
    for wide in [false, true] {
        let supported = if wide {
            std::is_x86_feature_detected!("avx512f")
        } else {
            std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
        };
        if !supported {
            continue;
        }
        let mut scratch = Vec::new();
        for k in [31usize, 513, 5633] {
            for m in 1..=9 {
                for n in [1, 17, 65] {
                    // These activations are deliberately not BF16-rounded.
                    let input: Vec<f32> = (0..m * k)
                        .map(|i| ((i * 7 % 19) as f32 - 9.) / 31.)
                        .collect();
                    let weights: Vec<bf16> = (0..n * k)
                        .map(|i| bf16::from_f32(((i * 13 % 23) as f32 - 11.) / 37.))
                        .collect();
                    for first in (0..n).step_by(CHANNELS) {
                        let mut output = vec![f32::NAN; m * CHANNELS];
                        run::<bf16, true, true>(
                            &input,
                            &weights,
                            1.,
                            k,
                            n,
                            first,
                            &mut output,
                            &mut scratch,
                            wide,
                        );
                        for token in 0..m {
                            for channel in 0..CHANNELS {
                                let row = first + channel;
                                let expected: f64 = if row < n {
                                    input[token * k..(token + 1) * k]
                                        .iter()
                                        .zip(&weights[row * k..(row + 1) * k])
                                        .map(|(x, w)| *x as f64 * w.to_f32() as f64)
                                        .sum()
                                } else {
                                    0.
                                };
                                let actual = output[token * CHANNELS + channel] as f64;
                                assert!(
                                    (actual - expected).abs() < 1e-4,
                                    "wide={wide} F32xBF16 M={m} N={n} K={k}: {actual} vs {expected}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn bf16_panels_match_f64_with_odd_k_row_channel_tails_and_reused_scratch() {
    if !crate::arch::x86_avx512_bf16_kernel_runtime_available() {
        return;
    }
    for k in [1usize, 31, 129, 513, 5633] {
        for m in 1..=9 {
            let input: Vec<bf16> = (0..m * k)
                .map(|i| bf16::from_f32(((i * 7 % 19) as f32 - 9.) / 31.))
                .collect();
            let mut pairs = vec![0u32; m * k.div_ceil(2)];
            for (i, value) in input.iter().enumerate() {
                pairs[(i / k) * k.div_ceil(2) + (i % k) / 2] |=
                    (value.to_bits() as u32) << ((i % k % 2) * 16);
            }
            for n in [1, 17, 65] {
                let weights: Vec<bf16> = (0..n * k)
                    .map(|i| bf16::from_f32(((i * 13 % 23) as f32 - 11.) / 37.))
                    .collect();
                for first in (0..n).step_by(CHANNELS) {
                    let mut output = vec![f32::NAN; m * CHANNELS];
                    bf16_panel(&pairs, &weights, k, n, first, &mut output);
                    for token in 0..m {
                        for channel in 0..CHANNELS {
                            let row = first + channel;
                            let expected: f64 = if row < n {
                                input[token * k..(token + 1) * k]
                                    .iter()
                                    .zip(&weights[row * k..(row + 1) * k])
                                    .map(|(x, w)| x.to_f32() as f64 * w.to_f32() as f64)
                                    .sum()
                            } else {
                                0.
                            };
                            let actual = output[token * CHANNELS + channel] as f64;
                            assert!(
                                (actual - expected).abs() < 1e-4,
                                "BF16 M={m} N={n} K={k} row={row}: {actual} vs {expected}"
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn native_panels_match_f64_with_tails_and_reused_scratch() {
    let mut half_scratch = Vec::new();
    let mut int_scratch = Vec::new();
    for wide in [false, true] {
        let supported = if wide {
            std::is_x86_feature_detected!("avx512f")
        } else {
            std::is_x86_feature_detected!("avx2")
                && std::is_x86_feature_detected!("fma")
                && std::is_x86_feature_detected!("f16c")
        };
        if !supported {
            continue;
        }
        for k in [31, 513, 5632] {
            for m in 1..=9 {
                for n in [1, 17, 65] {
                    let input: Vec<f32> = (0..m * k)
                        .map(|i| ((i * 7 % 19) as f32 - 9.) / 31.)
                        .collect();
                    let half: Vec<f16> = (0..n * k)
                        .map(|i| f16::from_f32(((i * 13 % 23) as f32 - 11.) / 37.))
                        .collect();
                    let quant: Vec<i8> = (0..n * k)
                        .map(|i| (i * 17 % 255) as i32 - 127)
                        .map(|x| x as i8)
                        .collect();
                    for first in (0..n).step_by(CHANNELS) {
                        let mut half_output = vec![f32::NAN; m * CHANNELS];
                        let mut int_output = half_output.clone();
                        run::<f16, true, false>(
                            &input,
                            &half,
                            1.,
                            k,
                            n,
                            first,
                            &mut half_output,
                            &mut half_scratch,
                            wide,
                        );
                        run::<i8, false, false>(
                            &input,
                            &quant,
                            0.0037,
                            k,
                            n,
                            first,
                            &mut int_output,
                            &mut int_scratch,
                            wide,
                        );
                        for token in 0..m {
                            for channel in 0..CHANNELS {
                                let channel_idx = first + channel;
                                let x = &input[token * k..(token + 1) * k];
                                let (expected_half, expected_int) = if channel_idx < n {
                                    let start = channel_idx * k;
                                    (
                                        x.iter()
                                            .zip(&half[start..start + k])
                                            .map(|(&a, &b)| a as f64 * b.to_f32() as f64)
                                            .sum::<f64>(),
                                        x.iter()
                                            .zip(&quant[start..start + k])
                                            .map(|(&a, &b)| a as f64 * b as f64)
                                            .sum::<f64>()
                                            * 0.0037f32 as f64,
                                    )
                                } else {
                                    (0., 0.)
                                };
                                for (actual, expected) in [
                                    (half_output[token * CHANNELS + channel], expected_half),
                                    (int_output[token * CHANNELS + channel], expected_int),
                                ] {
                                    assert!(
                                        (actual as f64 - expected).abs() < 1e-4,
                                        "wide={wide} m={m} n={n} k={k}: {actual} vs {expected}"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn bf16_panel_packing_preserves_bits_and_overwrites_reused_tails() {
    let patterns = [
        0u16, 0x8000, 1, 0x007f, 0x0080, 0x3f80, 0xbf80, 0x7f80, 0xff80, 0x7fc1, 0x7f81, 0xffff,
    ];
    let mut packed = Vec::new();
    for k in [5633usize, 129, 128, 33, 32, 31, 17, 16, 15, 9, 8, 7, 2, 1] {
        for n in [17, 16, 15, 7, 4, 3, 1] {
            let weights: Vec<_> = (0..n * k)
                .map(|i| bf16::from_bits(patterns[(i * 7 + i / k) % patterns.len()]))
                .collect();
            packed.resize(k.div_ceil(2) * CHANNELS, u32::MAX);
            for first in (0..n).step_by(CHANNELS) {
                pack_bf16_panel(&weights, k, n, first, &mut packed);
                for pair in 0..k.div_ceil(2) {
                    for c in 0..CHANNELS {
                        let expected = if first + c < n {
                            let offset = (first + c) * k + pair * 2;
                            let low = weights[offset].to_bits() as u32;
                            let high = if pair * 2 + 1 < k {
                                weights[offset + 1].to_bits() as u32
                            } else {
                                0
                            };
                            low | (high << 16)
                        } else {
                            0
                        };
                        assert_eq!(
                            packed[pair * CHANNELS + c],
                            expected,
                            "K={k} N={n} first={first} pair={pair} channel={c}"
                        );
                    }
                }
            }
        }
    }
}
