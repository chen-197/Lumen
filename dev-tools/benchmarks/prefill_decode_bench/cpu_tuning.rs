//! Opt-in, in-process task-budget sweeps. All cases use the same loaded model
//! and Rayon pool. Recommendations are advisory and never change defaults.
use super::*;
use serde::Serialize;
use serde_json::json;

const MIN_GAIN: f64 = 0.03;

#[derive(Clone, Serialize)]
struct Case {
    label: String,
    prefill_tasks: Option<NonZeroUsize>,
    decode_tasks: Option<NonZeroUsize>,
}

impl Case {
    fn baseline() -> Self {
        Self {
            label: "default".into(),
            prefill_tasks: None,
            decode_tasks: None,
        }
    }
}

#[derive(Serialize)]
struct Sample {
    round: usize,
    warmup: bool,
    prefill_us: f64,
    decode_us: f64,
    total_us: f64,
    generated_tokens: usize,
    token_ids: Vec<usize>,
    tokens_match: bool,
}

#[derive(Serialize)]
struct ResultCase {
    case: Case,
    samples: Vec<Sample>,
}

#[derive(Clone, Copy)]
enum Metric {
    Prefill,
    Decode,
    Total,
}

pub(super) fn validate(args: &Args) -> Result<(), String> {
    if args.cpu_tune_tasks.is_none() {
        return if args.cpu_tune_output.is_some() {
            Err("--cpu-tune-output requires --cpu-tune-tasks".into())
        } else {
            Ok(())
        };
    }
    if args.device != Device::Cpu || args.mode != DecodeMode::Greedy {
        return Err("CPU task tuning requires --device cpu --mode greedy".into());
    }
    if args.warmup == 0 || args.runs < 2 || args.max_gen == 0 {
        return Err("CPU task tuning requires warmup >= 1, runs >= 2 and max-gen >= 1".into());
    }
    if args.stop_on_eos
        || args.stop_on_chat_marker
        || args.cpu_prefill_tasks.is_some()
        || args.cpu_decode_tasks.is_some()
    {
        return Err(
            "CPU task tuning requires fixed-length generation and no explicit phase task limits"
                .into(),
        );
    }
    Ok(())
}

pub(super) fn validate_pool(args: &Args) -> Result<(), String> {
    if let Some(spec) = args.cpu_tune_tasks.as_deref() {
        budgets(spec, rayon::current_num_threads())?;
        let output = Path::new(
            args.cpu_tune_output
                .as_deref()
                .unwrap_or("cpu-task-tuning.json"),
        );
        if let Some(parent) = output.parent().filter(|p| !p.as_os_str().is_empty())
            && !parent.is_dir()
        {
            return Err(format!(
                "tuning report directory does not exist: {}",
                parent.display()
            ));
        }
        if let Ok(output) = output.canonicalize() {
            for input in [&args.weights, &args.tokenizer] {
                if Path::new(input)
                    .canonicalize()
                    .is_ok_and(|path| path == output)
                {
                    return Err("tuning report must not overwrite weights or tokenizer".into());
                }
            }
        }
    }
    Ok(())
}

fn budgets(spec: &str, workers: usize) -> Result<Vec<NonZeroUsize>, String> {
    let mut values = Vec::new();
    if spec == "auto" {
        let mut value = 1;
        while value < workers {
            values.push(value);
            value = value.checked_mul(2).unwrap_or(workers);
        }
        values.push(workers);
    } else {
        for part in spec.split(',') {
            let value = part.trim().parse::<NonZeroUsize>().map_err(
                |_| "--cpu-tune-tasks expects auto or comma-separated positive integers",
            )?;
            if value.get() > workers {
                return Err(format!(
                    "task budget {value} exceeds pool capacity {workers}"
                ));
            }
            values.push(value.get());
        }
    }
    values.sort_unstable();
    values.dedup();
    values
        .into_iter()
        .map(|v| NonZeroUsize::new(v).ok_or_else(|| "empty pool".into()))
        .collect()
}

fn median(mut values: Vec<f64>) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    Some(if values.len().is_multiple_of(2) {
        (values[middle - 1] + values[middle]) * 0.5
    } else {
        values[middle]
    })
}

fn round_cost(result: &ResultCase, round: usize, metric: Metric) -> Option<f64> {
    if result.samples.iter().any(|s| !s.tokens_match) {
        return None;
    }
    let values: Vec<_> = result
        .samples
        .iter()
        .filter(|s| s.round == round && !s.warmup)
        .map(|s| match metric {
            Metric::Prefill => s.prefill_us,
            Metric::Decode => s.decode_us / s.generated_tokens as f64,
            Metric::Total => s.total_us,
        })
        .collect();
    if values.iter().any(|v| !v.is_finite() || *v <= 0.) {
        return None;
    }
    median(values)
}

// Require a win in both sweep directions and >=3% on the mean of round medians.
fn improvement(base: &ResultCase, candidate: &ResultCase, metric: Metric) -> Option<f64> {
    let mut a = 0.;
    let mut b = 0.;
    for round in 0..2 {
        let before = round_cost(base, round, metric)?;
        let after = round_cost(candidate, round, metric)?;
        if after >= before {
            return None;
        }
        a += before;
        b += after;
    }
    let gain = 1. - b / a;
    (gain >= MIN_GAIN).then_some(gain)
}

fn best(results: &[ResultCase], metric: Metric) -> usize {
    let mut best = 0;
    let mut gain = 0.;
    for (index, candidate) in results.iter().enumerate().skip(1) {
        let relevant = match metric {
            Metric::Prefill => {
                candidate.case.prefill_tasks.is_some() && candidate.case.decode_tasks.is_none()
            }
            Metric::Decode => {
                candidate.case.decode_tasks.is_some() && candidate.case.prefill_tasks.is_none()
            }
            Metric::Total => true,
        };
        if relevant
            && let Some(value) = improvement(&results[0], candidate, metric)
            && value > gain
        {
            gain = value;
            best = index;
        }
    }
    best
}

fn validated_pair(results: &[ResultCase]) -> bool {
    let phases_ok = [Metric::Prefill, Metric::Decode].iter().all(|&metric| {
        // Use the same mean of round medians as phase selection. An unchanged
        // phase can fluctuate in one batch; total latency must still win in
        // both directions below.
        let costs = (0..2).try_fold((0., 0.), |(before, after), round| {
            Some((
                before + round_cost(&results[0], round, metric)?,
                after + round_cost(&results[1], round, metric)?,
            ))
        });
        costs.is_some_and(|(before, after)| after <= before * (1. + MIN_GAIN))
    });
    phases_ok && improvement(&results[0], &results[1], Metric::Total).is_some()
}

struct Experiment<'a> {
    model: &'a LlamaModel,
    tokenizer: &'a LlamaTokenizer,
    config: &'a LlamaConfig,
    stop_ids: &'a [usize],
    prompt: &'a [usize],
    args: &'a Args,
    expected: Option<Vec<usize>>,
    report: serde_json::Value,
    path: &'a Path,
}

impl Experiment<'_> {
    fn save(&self) -> Result<(), Box<dyn std::error::Error>> {
        std::fs::write(self.path, serde_json::to_vec_pretty(&self.report)?)?;
        Ok(())
    }

    fn sweep(
        &mut self,
        cases: Vec<Case>,
        key: &str,
    ) -> Result<Vec<ResultCase>, Box<dyn std::error::Error>> {
        let mut results: Vec<_> = cases
            .into_iter()
            .map(|case| ResultCase {
                case,
                samples: Vec::new(),
            })
            .collect();
        for round in 0..2 {
            let mut indices: Vec<_> = (0..results.len()).collect();
            if round == 1 {
                indices.reverse();
            }
            for index in indices {
                let result = &mut results[index];
                let mut args = self.args.clone();
                args.cpu_prefill_tasks = result.case.prefill_tasks;
                args.cpu_decode_tasks = result.case.decode_tasks;
                args.show_token_ids = true;
                for iteration in 0..args.warmup + args.runs {
                    let stats = run_once(
                        self.model,
                        self.tokenizer,
                        self.config,
                        self.stop_ids,
                        self.prompt,
                        &args,
                    );
                    let expected = self
                        .expected
                        .get_or_insert_with(|| stats.generated_token_ids.clone());
                    let tokens_match = stats.generated_tokens == args.max_gen
                        && stats.generated_token_ids == *expected;
                    result.samples.push(Sample {
                        round,
                        warmup: iteration < args.warmup,
                        prefill_us: stats.prefill_forward.as_secs_f64() * 1e6,
                        decode_us: stats.decode_forward.as_secs_f64() * 1e6,
                        total_us: stats.total.as_secs_f64() * 1e6,
                        generated_tokens: stats.generated_tokens,
                        token_ids: stats.generated_token_ids,
                        tokens_match,
                    });
                }
                println!(
                    "cpu_tuning {key} round={} case={} prefill_us={:.0} decode_us_per_token={:.0} tokens_match={}",
                    round + 1,
                    result.case.label,
                    round_cost(result, round, Metric::Prefill).unwrap_or(f64::NAN),
                    round_cost(result, round, Metric::Decode).unwrap_or(f64::NAN),
                    result.samples.iter().all(|s| s.tokens_match)
                );
                self.report[key] = json!(&results);
                self.save()?;
            }
        }
        Ok(results)
    }
}

pub(super) fn run(
    model: &LlamaModel,
    tokenizer: &LlamaTokenizer,
    config: &LlamaConfig,
    stop_ids: &[usize],
    prompt: &[usize],
    args: &Args,
) -> Result<(), Box<dyn std::error::Error>> {
    let workers = rayon::current_num_threads();
    let values = budgets(args.cpu_tune_tasks.as_deref().unwrap(), workers)?;
    let mut cases = vec![Case::baseline()];
    for limit in values {
        cases.push(Case {
            label: format!("prefill-{limit}"),
            prefill_tasks: Some(limit),
            decode_tasks: None,
        });
        cases.push(Case {
            label: format!("decode-{limit}"),
            prefill_tasks: None,
            decode_tasks: Some(limit),
        });
    }
    let path = Path::new(
        args.cpu_tune_output
            .as_deref()
            .unwrap_or("cpu-task-tuning.json"),
    );
    let mut experiment = Experiment {
        model,
        tokenizer,
        config,
        stop_ids,
        prompt,
        args,
        expected: None,
        path,
        report: json!({
            "schema_version": 1, "selection_policy_version":2, "status": "running", "mode":"greedy", "scope": "integrated_cpu_projection_kernels_only",
            "pool_threads": workers, "arch": std::env::consts::ARCH, "os": std::env::consts::OS,
            "processor_identifier": std::env::var("PROCESSOR_IDENTIFIER").ok(),
            "available_parallelism": std::thread::available_parallelism().map(|n| n.get()).ok(),
            "backend": {"float": active_float_backend_name(), "int8": active_int8_backend_name()},
            "weights": args.weights, "tokenizer": args.tokenizer,
            "dtype": {"parameter": format!("{:?}", args.parameter_dtype), "runtime":format!("{:?}",args.runtime_dtype), "activation":format!("{:?}",args.activation_dtype), "kv_cache":format!("{:?}",args.kv_cache_dtype)},
            "allow_parameter_copies":args.allow_parameter_copies, "stream_weights":args.stream_weights,
            "prompt_token_ids": prompt, "max_gen": args.max_gen, "runs_per_round": args.runs, "warmup_per_round":args.warmup,
            "min_gain": MIN_GAIN, "round_order": "forward_then_reverse", "recommendation_applied": false,
        }),
    };
    experiment.save()?;
    let results = experiment.sweep(cases, "sweep")?;
    if round_cost(&results[0], 0, Metric::Total).is_none()
        || round_cost(&results[0], 1, Metric::Total).is_none()
    {
        experiment.report["status"] = json!("invalid_baseline");
        experiment.save()?;
        return Err("baseline did not produce stable full-length token sequences".into());
    }
    let prefill = best(&results, Metric::Prefill);
    let decode = best(&results, Metric::Decode);
    let proposal = Case {
        label: "combined".into(),
        prefill_tasks: results[prefill].case.prefill_tasks,
        decode_tasks: results[decode].case.decode_tasks,
    };
    experiment.report["phase_suggestion"] = json!(&proposal);
    let mut recommended = Case::baseline();
    let mut reason = "no_stable_phase_gain";
    if proposal.prefill_tasks.is_some() || proposal.decode_tasks.is_some() {
        let validation =
            experiment.sweep(vec![Case::baseline(), proposal.clone()], "validation")?;
        reason = "combined_validation_failed";
        if validated_pair(&validation) {
            recommended = proposal;
            reason = "combined_validation_passed";
        }
    }
    let changed = recommended.prefill_tasks.is_some() || recommended.decode_tasks.is_some();
    let mut cli = format!("--cpu-threads {workers}");
    if let Some(limit) = recommended.prefill_tasks {
        cli.push_str(&format!(" --cpu-prefill-tasks {limit}"));
    }
    if let Some(limit) = recommended.decode_tasks {
        cli.push_str(&format!(" --cpu-decode-tasks {limit}"));
    }
    experiment.report["status"] = json!(if changed { "validated" } else { "keep_default" });
    experiment.report["recommended"] = json!(recommended);
    experiment.report["recommended_cli"] = json!(&cli);
    experiment.report["selection_reason"] = json!(reason);
    experiment.save()?;
    println!("cpu_tuning selection: {reason}");
    println!(
        "cpu_tuning recommendation: {cli}\nreport: {}\nRecommendation applies only to this measured workload; defaults were not changed.",
        path.display()
    );
    Ok(())
}

#[cfg(test)]
#[path = "../../../tests/unit/dev-tools/benchmarks/prefill_decode_bench/cpu_tuning/tests.rs"]
mod tests;
