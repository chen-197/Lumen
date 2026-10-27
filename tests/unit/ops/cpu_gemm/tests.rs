use super::*;

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
                        run::<f16, true>(
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
                        run::<i8, false>(
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
