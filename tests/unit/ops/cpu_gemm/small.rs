use super::*;

#[test]
fn direct_small_rows_match_f64_on_avx2_and_avx512_with_tails_and_offsets() {
    fn check<T: Weight>(weights: Vec<T>, input: &[f32], n: usize, scale: f32, wide: bool) {
        for (first, count) in [(0, n), (1, n - 1), (7, 17), (9, 1)] {
            let mut output = vec![f32::NAN; count];
            compute_row(input, &weights, n, first, scale, &mut output, wide);
            for (col, &got) in output.iter().enumerate() {
                let expected = input
                    .iter()
                    .enumerate()
                    .map(|(k, &a)| a as f64 * weights[k * n + first + col].f32() as f64)
                    .sum::<f64>()
                    * scale as f64;
                assert!(
                    got.is_finite()
                        && (got as f64 - expected).abs() <= 2e-5 + 2e-5 * expected.abs(),
                    "wide={wide} first={first} col={col}: {got} != {expected}"
                );
            }
        }
    }
    for wide in [false, true] {
        if !std::is_x86_feature_detected!("avx2")
            || !std::is_x86_feature_detected!("fma")
            || (wide && !std::is_x86_feature_detected!("avx512f"))
        {
            continue;
        }
        for k in [1, 7, 8, 127, 128, 129, 513] {
            let n = 145;
            let input = (0..k)
                .map(|i| ((i * 37 % 127) as f32 - 63.) * 0.00317)
                .collect::<Vec<_>>();
            let weights = (0..k * n)
                .map(|i| ((i * 19 % 131) as f32 - 65.) * 0.00371)
                .collect::<Vec<_>>();
            check(weights.clone(), &input, n, 1., wide);
            check(
                weights.iter().copied().map(bf16::from_f32).collect(),
                &input,
                n,
                1.,
                wide,
            );
            if wide || std::is_x86_feature_detected!("f16c") {
                check(
                    weights.iter().copied().map(f16::from_f32).collect(),
                    &input,
                    n,
                    1.,
                    wide,
                );
            }
            check(
                weights.iter().map(|v| (v * 97.).round() as i8).collect(),
                &input,
                n,
                0.01031,
                wide,
            );
        }
    }
}
