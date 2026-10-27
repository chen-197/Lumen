use super::*;

#[test]
fn replay_ids_reject_invalid_or_ambiguous_inputs() {
    assert_eq!(
        parse_ids("[0, 2, 31999]", 3, 32000).unwrap(),
        vec![0, 2, 31999]
    );
    for (text, count) in [
        ("[]", 0),
        ("[1]", 2),
        ("[32000]", 1),
        ("[-1]", 1),
        ("[1.5]", 1),
        ("[\"1\"]", 1),
    ] {
        assert!(parse_ids(text, count, 32000).is_err());
    }
}

#[test]
fn logit_capture_records_nonfinite_values_before_serialization() {
    let mut stats = RunStats::default();
    capture(&mut stats, &[1.0, f32::NAN, f32::INFINITY], false);
    assert!(stats.logits.is_empty());
    capture(&mut stats, &[1.0, f32::NAN, f32::INFINITY], true);
    assert_eq!(stats.nonfinite_logits, 2);
    assert!(stats.logits[0][1].is_nan());
}
