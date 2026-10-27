use super::diff_stats;

#[test]
fn diff_stats_rejects_non_finite_values() {
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let (abs, rel, rmse) = diff_stats(&[value], &[value]);
        assert!(abs.is_infinite());
        assert!(rel.is_infinite());
        assert!(rmse.is_infinite());
    }
}
