use super::*;

#[test]
fn sample_top_p_replays_the_same_draw() {
    let logits = [0.0, 1.0, 2.0, 3.0];
    let (first_token, first_draw) = sample_top_p(&logits, 0.8, 0.95, 1.0, &[], Some(0.72));
    let (second_token, second_draw) = sample_top_p(&logits, 0.8, 0.95, 1.0, &[], Some(0.72));

    assert_eq!(first_token, second_token);
    assert_eq!(first_draw, Some(0.72));
    assert_eq!(second_draw, Some(0.72));
}

#[test]
fn sample_top_p_greedy_does_not_consume_draws() {
    let logits = [0.0, 4.0, 2.0];
    let (token, draw) = sample_top_p(&logits, 0.0, 0.9, 1.0, &[], Some(0.5));

    assert_eq!(token, 1);
    assert_eq!(draw, None);
}

#[test]
fn sample_top_p_empty_logits_has_no_draw() {
    let (token, draw) = sample_top_p(&[], 0.8, 0.9, 1.0, &[], Some(0.5));

    assert_eq!(token, 0);
    assert_eq!(draw, None);
}
