//! Raw, opt-in samples. Instrumented/plain pairs share model, pool and allocator.
use super::*;
use serde_json::{Value, json};

pub(super) fn validate(args: &Args) -> Result<(), String> {
    if args.profile_cpu {
        if !cfg!(feature = "cpu-profiling") {
            return Err("--profile-cpu requires the cpu-profiling build feature".into());
        }
        if args.samples_output.is_none()
            || args.device != Device::Cpu
            || args.mode != DecodeMode::Greedy
        {
            return Err(
                "--profile-cpu requires --samples-output, --device cpu and --mode greedy".into(),
            );
        }
    }
    if args.samples_output.is_some() && args.cpu_tune_tasks.is_some() {
        return Err("--samples-output and --cpu-tune-tasks use separate report formats".into());
    }
    Ok(())
}

fn sample(stats: &RunStats, warmup: bool, index: usize, profiled: bool, regions: Value) -> Value {
    let us = |d: Duration| d.as_secs_f64() * 1e6;
    json!({
        "warmup": warmup, "index": index, "profiled": profiled,
        "prompt_tokens": stats.prompt_tokens, "generated_tokens": stats.generated_tokens,
        "token_ids": stats.generated_token_ids, "sample_draws": stats.sample_draws,
        "generated_text": stats.generated_text,
        "display_text": stats.display_text,
        "pending_utf8_token_ids": stats.pending_utf8_token_ids,
        "total_us": us(stats.total), "prefill_us": us(stats.prefill_forward),
        "decode_us": us(stats.decode_forward),
        "prefill_logits_extract_us": us(stats.prefill_logits_extract),
        "decode_logits_extract_us": us(stats.decode_logits_extract),
        "prefill_input_build_us": us(stats.prefill_input_build),
        "decode_input_build_us": us(stats.decode_input_build),
        "sampling_us": us(stats.sampling),
        "tokenizer_decode_us": us(stats.tokenizer_decode),
        "overhead_us": us(stats.overhead()),
        "decode_steps_us": stats.decode_steps.iter().copied().map(us).collect::<Vec<_>>(),
        "regions": regions,
        "predicted_token_ids": stats.predicted_token_ids,
        "logits": stats.logits,
        "nonfinite_logits": stats.nonfinite_logits,
    })
}

pub(super) fn run(
    model: &LlamaModel,
    tokenizer: &LlamaTokenizer,
    config: &LlamaConfig,
    stop_ids: &[usize],
    prompt_tokens: &[usize],
    args: &Args,
) -> Result<Vec<RunStats>, Box<dyn std::error::Error>> {
    let path = args.samples_output.as_ref().unwrap();
    // Fail before expensive inference if the output cannot be created.
    let file = std::fs::File::create(path)?;
    let mut records = Vec::new();
    let mut runs = Vec::new();
    let mut reference = None;
    for (warmup, count) in [(true, args.warmup), (false, args.runs)] {
        for index in 0..count {
            let order: &[bool] = if !args.profile_cpu {
                &[false]
            } else if index % 2 == 0 {
                &[false, true]
            } else {
                &[true, false]
            };
            for &profiled in order {
                let execute = || run_once(model, tokenizer, config, stop_ids, prompt_tokens, args);
                #[cfg(feature = "cpu-profiling")]
                let (stats, regions) = if profiled {
                    let (stats, regions) = lumen::cpu::profiling::capture(execute);
                    (stats, serde_json::to_value(regions)?)
                } else {
                    (execute(), Value::Null)
                };
                #[cfg(not(feature = "cpu-profiling"))]
                let (stats, regions) = (execute(), Value::Null);
                let matches = if args.mode == DecodeMode::Greedy && args.replay_token_ids.is_none()
                {
                    let ids = reference.get_or_insert_with(|| stats.generated_token_ids.clone());
                    Some(*ids == stats.generated_token_ids)
                } else {
                    None
                };
                let mut record = sample(&stats, warmup, index, profiled, regions);
                record["run_type"] = json!(if args.replay_token_ids.is_some() {
                    "fixed_token_replay"
                } else {
                    "free_generation"
                });
                record["logits_captured"] = json!(args.capture_logits);
                if args.replay_token_ids.is_some() {
                    record["replay_tokens_match"] =
                        json!(args.replay_token_ids.as_ref() == Some(&stats.generated_token_ids));
                }
                record["tokens_match"] = json!(matches);
                records.push(record);
                // Preserve a mismatch sample before failing, including all its token IDs.
                if matches == Some(false) {
                    serde_json::to_writer_pretty(
                        &file,
                        &json!({"samples": records, "error": "greedy token mismatch"}),
                    )?;
                    return Err("greedy token mismatch across raw samples".into());
                }
                if !warmup && !profiled {
                    runs.push(stats);
                }
            }
        }
    }
    serde_json::to_writer_pretty(
        file,
        &json!({
            "schema_version": 1, "allocator": "mimalloc", "command": env::args().collect::<Vec<_>>(),
            "args": format!("{args:?}"), "cpu_threads": rayon::current_num_threads(),
            "cpu_profiling_compiled": cfg!(feature = "cpu-profiling"),
            "float_backend": active_float_backend_name(), "int8_backend": active_int8_backend_name(),
            "note": "Caller wall times include dispatch and worker completion; regions do not split memory, dequantization or SIMD costs. Ordinary summaries use only unprofiled timed samples. Each generated token has a decode forward, including the final token.",
            "samples": records,
            "prompt_token_ids": prompt_tokens,
            "replay_token_ids": args.replay_token_ids,
            "diagnostic_only": args.capture_logits,
            "activation_quantization": format!("{:?}", args.activation_quantization),
        }),
    )?;
    println!("raw_samples: {path}");
    Ok(runs)
}

#[cfg(test)]
#[path = "../../../tests/unit/dev-tools/benchmarks/prefill_decode_bench/samples/tests.rs"]
mod tests;
