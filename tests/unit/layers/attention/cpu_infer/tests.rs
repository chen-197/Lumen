use super::*;
#[test]
fn score_scratch_is_bounded_and_reentrant() {
    with_scores(SCORE_VALUES, |values| {
        values.fill(3.0);
        with_scores(17, |nested| {
            nested.fill(7.0);
        });
        assert!(values.iter().all(|&v| v == 3.0));
    });
    assert!(retained_score_capacity() <= SCORE_VALUES);
    with_scores(SCORE_VALUES + 1, |values| {
        assert_eq!(values.len(), SCORE_VALUES + 1);
    });
    assert!(retained_score_capacity() <= SCORE_VALUES);
}
