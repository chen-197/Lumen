use super::*;
fn result(case: Case, costs: [f64; 2], matching: bool) -> ResultCase {
    ResultCase {
        case,
        samples: (0..2)
            .flat_map(|round| {
                (0..3).map(move |_| Sample {
                    round,
                    warmup: false,
                    prefill_us: costs[round],
                    decode_us: costs[round],
                    total_us: costs[round],
                    generated_tokens: 1,
                    token_ids: vec![7],
                    tokens_match: matching,
                })
            })
            .collect(),
    }
}
#[test]
fn candidates_follow_pool_capacity_and_reject_invalid_values() {
    assert_eq!(
        budgets("auto", 10)
            .unwrap()
            .iter()
            .map(|v| v.get())
            .collect::<Vec<_>>(),
        [1, 2, 4, 8, 10]
    );
    assert_eq!(
        budgets("8,2,2", 8)
            .unwrap()
            .iter()
            .map(|v| v.get())
            .collect::<Vec<_>>(),
        [2, 8]
    );
    for spec in ["0", "-1", "", "1,,2", "11"] {
        assert!(budgets(spec, 10).is_err());
    }
}
#[test]
fn recommendations_reject_token_mismatch_noise_and_one_round_wins() {
    let mut candidate = Case::baseline();
    candidate.prefill_tasks = NonZeroUsize::new(4);
    let results = vec![
        result(Case::baseline(), [100., 100.], true),
        result(candidate.clone(), [80., 80.], false),
        result(candidate.clone(), [90., 110.], true),
        result(candidate.clone(), [98., 98.], true),
        result(candidate, [90., 91.], true),
    ];
    assert_eq!(best(&results, Metric::Prefill), 4);
    assert_eq!(best(&results[..4], Metric::Prefill), 0);
    assert_eq!(best(&results, Metric::Decode), 0);
}

#[test]
fn combined_validation_rejects_stage_regressions_and_worse_total_latency() {
    let mut results = vec![
        result(Case::baseline(), [100., 100.], true),
        result(Case::baseline(), [90., 90.], true),
    ];
    assert!(validated_pair(&results));
    for sample in &mut results[1].samples {
        sample.prefill_us = 110.;
    }
    assert!(!validated_pair(&results));
    for sample in &mut results[1].samples {
        sample.prefill_us = 80.;
        sample.total_us = 105.;
    }
    assert!(!validated_pair(&results));
}

#[test]
fn combined_validation_uses_two_round_phase_mean_but_requires_both_total_wins() {
    let mut results = vec![
        result(Case::baseline(), [100., 100.], true),
        result(Case::baseline(), [90., 90.], true),
    ];
    for sample in &mut results[1].samples {
        sample.prefill_us = if sample.round == 0 { 98. } else { 104. };
    }
    assert!(validated_pair(&results));
    for sample in &mut results[1].samples {
        if sample.round == 1 {
            sample.total_us = 101.;
        }
    }
    assert!(!validated_pair(&results));
}
