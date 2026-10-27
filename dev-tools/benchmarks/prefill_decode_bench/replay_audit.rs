//! Fixed inputs for timing and opt-in, untimed full-logit diagnostics.
use super::*;

pub(super) fn parse_ids(text: &str, count: usize, vocab: usize) -> Result<Vec<usize>, String> {
    let ids: Vec<usize> = serde_json::from_str(text).map_err(|e| {
        format!("replay token IDs must be a JSON array of nonnegative integers: {e}")
    })?;
    if ids.is_empty() || ids.len() != count {
        return Err("replay token count must equal --max-gen and be nonzero".into());
    }
    if ids.iter().any(|&id| id >= vocab) {
        return Err("replay token ID is outside the model vocabulary".into());
    }
    Ok(ids)
}

pub(super) fn validate(args: &Args) -> Result<(), String> {
    if args.replay_token_ids.is_some() {
        if args.stop_on_eos || args.stop_on_chat_marker || args.cpu_tune_tasks.is_some() {
            return Err(
                "fixed token replay requires stopping and CPU tuning to be disabled".into(),
            );
        }
        if args.compare_cpu {
            return Err("--compare-cpu compares free generation; use separate replay reports for each device".into());
        }
        if args.samples_output.is_none() {
            return Err(
                "fixed token replay requires --samples-output to label forced tokens".into(),
            );
        }
        if args.mode != DecodeMode::Greedy && !args.capture_logits {
            return Err("fixed token performance replay requires --mode greedy".into());
        }
    }
    if args.capture_logits
        && (args.mode != DecodeMode::Sample
            || args.samples_output.is_none()
            || args.runs != 1
            || args.warmup != 0
            || args.max_gen > 32
            || args.profile_cpu
            || args.cpu_tune_tasks.is_some())
    {
        return Err("--capture-logits requires sample mode, --samples-output, runs=1, warmup=0, max-gen<=32, and no profiling/tuning; this diagnostic is not a performance measurement".into());
    }
    Ok(())
}

pub(super) fn capture(stats: &mut RunStats, logits: &[f32], enabled: bool) {
    if enabled {
        stats.nonfinite_logits += logits.iter().filter(|v| !v.is_finite()).count();
        stats.logits.push(logits.to_vec());
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/dev-tools/benchmarks/prefill_decode_bench/replay_audit/tests.rs"]
mod tests;
