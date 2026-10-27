use super::max_abs_diff;

#[test]
fn max_abs_diff_rejects_non_finite_values() {
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(max_abs_diff(&[value], &[value]).is_infinite());
    }
}
