use super::*;
#[test]
fn raw_sample_preserves_decode_steps_and_tokens() {
    let stats = RunStats {
        decode_steps: vec![Duration::from_micros(2), Duration::from_micros(3)],
        generated_token_ids: vec![8, 9],
        ..Default::default()
    };
    let raw = sample(&stats, true, 4, false, Value::Null);
    assert_eq!(raw["decode_steps_us"], json!([2.0, 3.0]));
    assert_eq!(raw["token_ids"], json!([8, 9]));
    assert_eq!(raw["warmup"], true);
}
