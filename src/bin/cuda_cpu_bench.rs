use mimalloc::MiMalloc;

mod cuda_cpu_bench_path;

use lumen::autograd::{Device, Tensor, no_grad, set_strict_device_execution_scoped};
use lumen::layers::activation::{ReLU, SiLU, Softmax};
use lumen::layers::{Conv2D, Embedding, MaxPool2D, RMSNorm, RotaryEmbedding, SelfAttention};
use lumen::loss::{CrossEntropyLoss, MSELoss};
use lumen::models::{LlamaConfig, LlamaModel};
use lumen::module::Module;
use lumen::ops::arithmetic::sum;
use lumen::ops::fused::{
    fused_gate_up_silu_infer, fused_qkv_decode_infer_tensors, fused_qkv_prefill_infer_tensors,
    fused_softmax,
};
use lumen::ops::matmul::{batch_matmul, matmul};
use lumen::optim::{Adam, Optimizer, SGD};
use lumen::precision::DType;

use ndarray::{Array, IxDyn};
use std::env;
use std::hint::black_box;
use std::time::{Duration, Instant};

const OPTIMIZER_BATCHED_PARAM_COUNT: usize = 32;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Suite {
    All,
    Ops,
    Nn,
    Backward,
    Path,
}

#[derive(Debug, Clone, Copy)]
enum SizeProfile {
    Small,
    Medium,
    Large,
}

#[derive(Debug, Clone)]
struct Args {
    runs: usize,
    warmup: usize,
    suite: Suite,
    size: SizeProfile,
    dtype: DType,
    case_filter: Option<String>,
    check: bool,
    weights: Option<String>,
    tokenizer: Option<String>,
    prompt: String,
    system: String,
    max_gen: usize,
    max_seq_len: usize,
    path_device: Device,
    allow_parameter_copies: bool,
    stream_weights: bool,
    show_output: bool,
}

#[derive(Debug, Clone, Copy)]
struct ShapeConfig {
    matmul_m: usize,
    matmul_n: usize,
    matmul_k: usize,
    elem_len: usize,
    softmax_outer: usize,
    softmax_last: usize,
    conv_batch: usize,
    conv_in: usize,
    conv_out: usize,
    conv_hw: usize,
    attention_batch: usize,
    attention_seq: usize,
    attention_hidden: usize,
    attention_heads: usize,
    attention_kv_heads: usize,
}

#[derive(Debug, Clone)]
struct BenchResult {
    name: &'static str,
    cpu: Duration,
    cuda: Option<Duration>,
}

#[derive(Clone, Copy)]
struct BenchDef {
    name: &'static str,
    run: fn(&Args, ShapeConfig) -> BenchResult,
}

#[derive(Debug, Clone)]
struct CheckMetric {
    label: &'static str,
    abs: f32,
    rel: f32,
    rmse: f32,
}

#[derive(Debug, Clone)]
struct CheckResult {
    name: &'static str,
    metrics: Vec<CheckMetric>,
}

#[derive(Clone, Copy)]
struct CheckDef {
    name: &'static str,
    run: fn(&Args, ShapeConfig) -> CheckResult,
}

struct BenchPlan<'a> {
    checks: &'a [CheckDef],
    benches: &'a [BenchDef],
}

impl BenchPlan<'_> {
    fn run(&self, args: &Args, cfg: ShapeConfig) {
        self.run_checks(args, cfg);
        self.run_benches(args, cfg);
    }

    fn run_checks(&self, args: &Args, cfg: ShapeConfig) {
        if !args.check {
            return;
        }
        if !self.checks.iter().any(|check| should_run(args, check.name)) {
            return;
        }
        if !lumen::ops::cuda::is_available() {
            println!("check: skipped because CUDA is unavailable.");
            return;
        }
        for check in self.checks {
            if should_run(args, check.name) {
                if let Some(reason) = skip_reason_for_dtype(args, check.name) {
                    print_dtype_skip("check", check.name, args.dtype, reason);
                    continue;
                }
                let result = (check.run)(args, cfg);
                print_check_result(&result);
            }
        }
    }

    fn run_benches(&self, args: &Args, cfg: ShapeConfig) {
        for bench in self.benches {
            if should_run(args, bench.name) {
                if let Some(reason) = skip_reason_for_dtype(args, bench.name) {
                    print_dtype_skip("bench", bench.name, args.dtype, reason);
                    continue;
                }
                let result = (bench.run)(args, cfg);
                print_result(&result);
            }
        }
    }
}

fn usage(program: &str) {
    eprintln!(
        "Usage:\n  {program} [options]\n\nOptions:\n  --runs N       Timed runs per case (default: 10)\n  --warmup N     Warmup runs per case (default: 3)\n  --suite NAME   all/ops/nn/backward/path (default: all)\n  --size NAME    small/medium/large (default: medium)\n  --dtype DTYPE  f32/f16/bf16/i8 (default: f32)\n  --case TEXT    Run only cases whose names contain TEXT\n  --check        Run CPU/CUDA correctness and path checks\n  --weights PATH     Optional TinyLlama safetensors for inference path check\n  --tokenizer PATH   Optional tokenizer.json for inference path check\n  --prompt TEXT      Inference path prompt\n  --system TEXT      Inference path system prompt\n  --max-gen N        Generated tokens for inference path check (default: 32)\n  --max-seq-len N    KV cache length for real-model path check (default: 256)\n  --path-device DEV  cpu/cuda for real-model inference path check (default: cpu)\n  --allow-parameter-copies\n  --stream-weights\n  --show-output      Print generated text from inference path check\n\nExamples:\n  cargo run --release --features \"dev-tools cuda\" --bin cuda_cpu_bench -- --suite all --size medium --dtype bf16\n  cargo run --release --features \"dev-tools cuda\" --bin cuda_cpu_bench -- --suite backward --case optimizer --size small --dtype bf16 --check\n  cargo run --release --features \"dev-tools cuda\" --bin cuda_cpu_bench -- --suite path --check --weights model.safetensors --tokenizer tokenizer.json --show-output\n"
    );
}

fn parse_dtype(value: &str) -> Result<DType, String> {
    match value {
        "f32" => Ok(DType::F32),
        "f16" => Ok(DType::F16),
        "bf16" => Ok(DType::BF16),
        "i8" => Ok(DType::I8),
        other => Err(format!(
            "未知 dtype: {other}; cuda_cpu_bench 目前支持 f32/f16/bf16/i8"
        )),
    }
}

fn parse_args() -> Result<Args, String> {
    let argv = env::args().collect::<Vec<_>>();
    let program = argv
        .first()
        .cloned()
        .unwrap_or_else(|| "cuda_cpu_bench".to_string());
    let mut runs = 10usize;
    let mut warmup = 3usize;
    let mut suite = Suite::All;
    let mut size = SizeProfile::Medium;
    let mut dtype = DType::F32;
    let mut case_filter = None;
    let mut check = false;
    let mut weights = None;
    let mut tokenizer = None;
    let mut prompt = "Explain Transformer KV cache in one concise sentence.".to_string();
    let mut system = "You are a helpful AI assistant.".to_string();
    let mut max_gen = 32usize;
    let mut max_seq_len = 256usize;
    let mut path_device = Device::Cpu;
    let mut allow_parameter_copies = false;
    let mut stream_weights = false;
    let mut show_output = false;

    let mut i = 1usize;
    while i < argv.len() {
        match argv[i].as_str() {
            "-h" | "--help" => {
                usage(&program);
                std::process::exit(0);
            }
            "--runs" => {
                i += 1;
                runs = argv
                    .get(i)
                    .ok_or("--runs 缺少数字")?
                    .parse::<usize>()
                    .map_err(|_| "--runs 需要 usize")?;
                if runs == 0 {
                    return Err("--runs 必须 >= 1".to_string());
                }
            }
            "--warmup" => {
                i += 1;
                warmup = argv
                    .get(i)
                    .ok_or("--warmup 缺少数字")?
                    .parse::<usize>()
                    .map_err(|_| "--warmup 需要 usize")?;
            }
            "--suite" => {
                i += 1;
                suite = match argv.get(i).ok_or("--suite 缺少名称")?.as_str() {
                    "all" => Suite::All,
                    "ops" => Suite::Ops,
                    "nn" => Suite::Nn,
                    "backward" => Suite::Backward,
                    "path" => Suite::Path,
                    other => return Err(format!("未知 suite: {other}")),
                };
            }
            "--size" => {
                i += 1;
                size = match argv.get(i).ok_or("--size 缺少名称")?.as_str() {
                    "small" => SizeProfile::Small,
                    "medium" => SizeProfile::Medium,
                    "large" => SizeProfile::Large,
                    other => return Err(format!("未知 size: {other}")),
                };
            }
            "--dtype" => {
                i += 1;
                dtype = parse_dtype(argv.get(i).ok_or("--dtype 缺少名称")?.as_str())?;
            }
            "--case" => {
                i += 1;
                case_filter = Some(argv.get(i).ok_or("--case 缺少过滤文本")?.clone());
            }
            "--check" => check = true,
            "--weights" => {
                i += 1;
                weights = Some(argv.get(i).ok_or("--weights 缺少路径")?.clone());
            }
            "--tokenizer" => {
                i += 1;
                tokenizer = Some(argv.get(i).ok_or("--tokenizer 缺少路径")?.clone());
            }
            "--prompt" => {
                i += 1;
                prompt = argv.get(i).ok_or("--prompt 缺少文本")?.clone();
            }
            "--system" => {
                i += 1;
                system = argv.get(i).ok_or("--system 缺少文本")?.clone();
            }
            "--max-gen" => {
                i += 1;
                max_gen = argv
                    .get(i)
                    .ok_or("--max-gen 缺少数字")?
                    .parse::<usize>()
                    .map_err(|_| "--max-gen 需要 usize")?;
                if max_gen == 0 {
                    return Err("--max-gen 必须 >= 1".to_string());
                }
            }
            "--max-seq-len" => {
                i += 1;
                max_seq_len = argv
                    .get(i)
                    .ok_or("--max-seq-len 缺少数字")?
                    .parse::<usize>()
                    .map_err(|_| "--max-seq-len 需要 usize")?;
            }
            "--path-device" => {
                i += 1;
                path_device = match argv.get(i).ok_or("--path-device 缺少设备")?.as_str() {
                    "cpu" => Device::Cpu,
                    "cuda" | "gpu" => Device::Cuda,
                    other => return Err(format!("未知 path device: {other}")),
                };
            }
            "--allow-parameter-copies" => allow_parameter_copies = true,
            "--stream-weights" => stream_weights = true,
            "--show-output" => show_output = true,
            other => return Err(format!("未知参数: {other}")),
        }
        i += 1;
    }

    Ok(Args {
        runs,
        warmup,
        suite,
        size,
        dtype,
        case_filter,
        check,
        weights,
        tokenizer,
        prompt,
        system,
        max_gen,
        max_seq_len,
        path_device,
        allow_parameter_copies,
        stream_weights,
        show_output,
    })
}

fn shape_config(size: SizeProfile) -> ShapeConfig {
    match size {
        SizeProfile::Small => ShapeConfig {
            matmul_m: 256,
            matmul_n: 256,
            matmul_k: 256,
            elem_len: 1 << 18,
            softmax_outer: 512,
            softmax_last: 256,
            conv_batch: 2,
            conv_in: 8,
            conv_out: 16,
            conv_hw: 24,
            attention_batch: 1,
            attention_seq: 16,
            attention_hidden: 64,
            attention_heads: 4,
            attention_kv_heads: 2,
        },
        SizeProfile::Medium => ShapeConfig {
            matmul_m: 512,
            matmul_n: 512,
            matmul_k: 512,
            elem_len: 1 << 20,
            softmax_outer: 2048,
            softmax_last: 512,
            conv_batch: 4,
            conv_in: 16,
            conv_out: 32,
            conv_hw: 32,
            attention_batch: 2,
            attention_seq: 32,
            attention_hidden: 128,
            attention_heads: 8,
            attention_kv_heads: 4,
        },
        SizeProfile::Large => ShapeConfig {
            matmul_m: 1024,
            matmul_n: 1024,
            matmul_k: 1024,
            elem_len: 1 << 22,
            softmax_outer: 4096,
            softmax_last: 1024,
            conv_batch: 8,
            conv_in: 32,
            conv_out: 64,
            conv_hw: 48,
            attention_batch: 2,
            attention_seq: 64,
            attention_hidden: 256,
            attention_heads: 8,
            attention_kv_heads: 4,
        },
    }
}

fn sample_data(len: usize, scale: f32) -> Vec<f32> {
    (0..len)
        .map(|i| (((i * 17 + 11) % 97) as f32 - 48.0) * scale)
        .collect()
}

fn one_hot_data(rows: usize, cols: usize) -> Vec<f32> {
    let mut data = vec![0.0; rows * cols];
    for row in 0..rows {
        data[row * cols + (row * 13 + 7) % cols] = 1.0;
    }
    data
}

fn token_id_data(batch: usize, seq: usize, vocab_size: usize) -> Vec<f32> {
    (0..batch * seq)
        .map(|i| ((i * 7 + 3) % vocab_size) as f32)
        .collect()
}

fn llama_config(cfg: ShapeConfig) -> LlamaConfig {
    LlamaConfig {
        vocab_size: (cfg.attention_hidden * 2).max(64),
        hidden_size: cfg.attention_hidden,
        intermediate_size: cfg.attention_hidden * 4,
        num_hidden_layers: 1,
        num_attention_heads: cfg.attention_heads,
        num_key_value_heads: cfg.attention_kv_heads,
        rms_norm_eps: 1e-5,
        max_seq_len: cfg.attention_seq + 1,
        rope_theta: 10000.0,
    }
}

fn array_from_vec(shape: &[usize], data: Vec<f32>) -> ndarray::ArrayD<f32> {
    Array::from_shape_vec(IxDyn(shape), data)
        .expect("bench tensor shape mismatch")
        .into_dyn()
}

fn tensor_no_grad(shape: &[usize], data: Vec<f32>, dtype: DType) -> Tensor {
    no_grad(|| Tensor::new_with_dtype(array_from_vec(shape, data), dtype))
}

fn tensor_grad(shape: &[usize], data: Vec<f32>, dtype: DType) -> Tensor {
    Tensor::new_with_dtype(array_from_vec(shape, data), dtype)
}

fn tensor_const(shape: &[usize], data: Vec<f32>, dtype: DType) -> Tensor {
    tensor_no_grad(shape, data, dtype)
}

fn token_tensor(shape: &[usize], data: Vec<f32>) -> Tensor {
    Tensor::from_array_no_grad(array_from_vec(shape, data))
}

fn median_duration(mut values: Vec<Duration>) -> Duration {
    values.sort_unstable();
    values[values.len() / 2]
}

fn measure<F>(args: &Args, mut f: F) -> Duration
where
    F: FnMut(),
{
    for _ in 0..args.warmup {
        f();
    }
    let mut values = Vec::with_capacity(args.runs);
    for _ in 0..args.runs {
        let start = Instant::now();
        f();
        values.push(start.elapsed());
    }
    median_duration(values)
}

fn measure_cuda<F>(args: &Args, f: F) -> Option<Duration>
where
    F: FnMut(),
{
    if !lumen::ops::cuda::is_available() {
        return None;
    }
    let cuda_enabled_guard = lumen::ops::cuda::set_enabled_scoped(true);
    let strict_device_execution_guard = set_strict_device_execution_scoped(true);
    let mut f = f;
    for _ in 0..args.warmup {
        f();
        lumen::ops::cuda::synchronize()
            .unwrap_or_else(|err| panic!("CUDA bench warmup sync failed: {err}"));
    }
    let mut values = Vec::with_capacity(args.runs);
    for _ in 0..args.runs {
        lumen::ops::cuda::synchronize()
            .unwrap_or_else(|err| panic!("CUDA bench pre-run sync failed: {err}"));
        let start = Instant::now();
        f();
        lumen::ops::cuda::synchronize()
            .unwrap_or_else(|err| panic!("CUDA bench timed sync failed: {err}"));
        values.push(start.elapsed());
    }
    let elapsed = median_duration(values);
    drop(strict_device_execution_guard);
    drop(cuda_enabled_guard);
    Some(elapsed)
}

fn measure_with_setup<S, F>(args: &Args, mut setup: S, mut f: F) -> Duration
where
    S: FnMut(),
    F: FnMut(),
{
    for _ in 0..args.warmup {
        setup();
        f();
    }
    let mut values = Vec::with_capacity(args.runs);
    for _ in 0..args.runs {
        setup();
        let start = Instant::now();
        f();
        values.push(start.elapsed());
    }
    median_duration(values)
}

fn measure_cuda_with_setup<S, F>(args: &Args, setup: S, f: F) -> Option<Duration>
where
    S: FnMut(),
    F: FnMut(),
{
    if !lumen::ops::cuda::is_available() {
        return None;
    }
    let cuda_enabled_guard = lumen::ops::cuda::set_enabled_scoped(true);
    let strict_device_execution_guard = set_strict_device_execution_scoped(true);
    let mut setup = setup;
    let mut f = f;
    for _ in 0..args.warmup {
        setup();
        lumen::ops::cuda::synchronize()
            .unwrap_or_else(|err| panic!("CUDA bench setup sync failed: {err}"));
        f();
        lumen::ops::cuda::synchronize()
            .unwrap_or_else(|err| panic!("CUDA bench warmup sync failed: {err}"));
    }
    let mut values = Vec::with_capacity(args.runs);
    for _ in 0..args.runs {
        setup();
        lumen::ops::cuda::synchronize()
            .unwrap_or_else(|err| panic!("CUDA bench setup sync failed: {err}"));
        let start = Instant::now();
        f();
        lumen::ops::cuda::synchronize()
            .unwrap_or_else(|err| panic!("CUDA bench timed sync failed: {err}"));
        values.push(start.elapsed());
    }
    let elapsed = median_duration(values);
    drop(strict_device_execution_guard);
    drop(cuda_enabled_guard);
    Some(elapsed)
}

fn zero_all(tensors: &[&Tensor]) {
    for tensor in tensors {
        tensor.zero_grad();
    }
}

fn zero_params(module: &impl Module) {
    for param in module.parameters() {
        param.zero_grad();
    }
}

fn zero_param_list(params: &[Tensor]) {
    for param in params {
        param.zero_grad();
    }
}

fn copy_parameters(src: &impl Module, dst: &impl Module) {
    let src_params = src.parameters();
    let dst_params = dst.parameters();
    assert_eq!(
        src_params.len(),
        dst_params.len(),
        "parameter count mismatch while preparing CUDA check"
    );
    for (src_param, dst_param) in src_params.iter().zip(dst_params.iter()) {
        let (shape, dtype, raw) = src_param.export_raw();
        dst_param
            .import_raw(shape, dtype, raw)
            .expect("parameter copy for CUDA check failed");
    }
}

fn dtype_check_tolerance(dtype: DType) -> (f32, f32) {
    match dtype {
        DType::F32 => (1e-2, 1e-2),
        DType::F16 | DType::BF16 => (3e-1, 8e-2),
        DType::I8 => (1.0, 2e-1),
    }
}

fn diff_stats(lhs: &[f32], rhs: &[f32]) -> (f32, f32, f32) {
    assert_eq!(lhs.len(), rhs.len(), "check vector length mismatch");
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    let mut sq_sum = 0.0f64;
    for (&a, &b) in lhs.iter().zip(rhs.iter()) {
        if !a.is_finite() || !b.is_finite() {
            return (f32::INFINITY, f32::INFINITY, f32::INFINITY);
        }
        let abs = (a - b).abs();
        let denom = a.abs().max(b.abs()).max(1e-6);
        max_abs = max_abs.max(abs);
        max_rel = max_rel.max(abs / denom);
        sq_sum += (abs as f64) * (abs as f64);
    }
    let rmse = if lhs.is_empty() {
        0.0
    } else {
        (sq_sum / lhs.len() as f64).sqrt() as f32
    };
    (max_abs, max_rel, rmse)
}

#[cfg(test)]
mod accuracy_tests {
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
}

fn assert_close_vec(
    label: &str,
    lhs: &[f32],
    rhs: &[f32],
    abs_tol: f32,
    rel_tol: f32,
) -> (f32, f32) {
    let (max_abs, max_rel, _rmse) = diff_stats(lhs, rhs);
    assert!(
        max_abs <= abs_tol || max_rel <= rel_tol,
        "{label} CPU/CUDA mismatch: max_abs={max_abs:.6e} max_rel={max_rel:.6e} abs_tol={abs_tol:.6e} rel_tol={rel_tol:.6e}"
    );
    (max_abs, max_rel)
}

fn check_metric(
    label: &'static str,
    lhs: &[f32],
    rhs: &[f32],
    abs_tol: f32,
    rel_tol: f32,
) -> CheckMetric {
    let (abs, rel, rmse) = diff_stats(lhs, rhs);
    assert!(
        abs <= abs_tol || rel <= rel_tol,
        "{label} CPU/CUDA mismatch: max_abs={abs:.6e} max_rel={rel:.6e} rmse={rmse:.6e} abs_tol={abs_tol:.6e} rel_tol={rel_tol:.6e}"
    );
    CheckMetric {
        label,
        abs,
        rel,
        rmse,
    }
}

fn collect_parameter_grads(module: &impl Module) -> Vec<Vec<f32>> {
    module
        .parameters()
        .into_iter()
        .map(|param| {
            param
                .grad()
                .expect("training check expected parameter grad")
                .iter()
                .copied()
                .collect()
        })
        .collect()
}

fn collect_parameter_data(module: &impl Module) -> Vec<Vec<f32>> {
    module
        .parameters()
        .into_iter()
        .map(|param| param.data_ref().iter().copied().collect())
        .collect()
}

fn tensor_data_vec(tensor: &Tensor) -> Vec<f32> {
    tensor.data_ref().iter().copied().collect()
}

fn scalar_value(tensor: &Tensor) -> f32 {
    tensor.data_ref().first().copied().unwrap_or_default()
}

fn tensor_grad_vec(tensor: &Tensor, label: &str) -> Vec<f32> {
    tensor
        .grad()
        .unwrap_or_else(|| panic!("{label} check expected tensor grad"))
        .iter()
        .copied()
        .collect()
}

fn llama_train_loss(
    model: &LlamaModel,
    input: Tensor,
    targets: &Tensor,
    vocab_size: usize,
) -> Tensor {
    let logits = model.forward_train(input);
    CrossEntropyLoss::apply(&logits.reshape(vec![-1, vocab_size as i32]), targets)
}

fn assert_cuda_model_f32_grads_resident(model: &impl Module, name: &str) {
    let params = model.parameters();
    assert!(!params.is_empty(), "{name} expected trainable parameters");
    for (idx, param) in params.iter().enumerate() {
        assert!(
            param.dev_has_cuda_f32_grad(),
            "{name} expected parameter {idx} to have a CUDA f32 gradient before host materialization"
        );
        assert!(
            !param.dev_has_host_grad(),
            "{name} parameter {idx} unexpectedly materialized a host gradient"
        );
    }
}

fn check_llama_train(args: &Args, cfg: ShapeConfig, step: bool, name: &'static str) -> CheckResult {
    assert!(
        lumen::ops::cuda::is_available(),
        "{name} check requires CUDA"
    );
    let llama_cfg = llama_config(cfg);
    let rows = cfg.attention_batch * cfg.attention_seq;
    let input = token_tensor(
        &[cfg.attention_batch, cfg.attention_seq],
        token_id_data(cfg.attention_batch, cfg.attention_seq, llama_cfg.vocab_size),
    );
    let targets = tensor_const(
        &[rows, llama_cfg.vocab_size],
        one_hot_data(rows, llama_cfg.vocab_size),
        args.dtype,
    );
    let cpu_model = LlamaModel::new_with_dtype(llama_cfg.clone(), args.dtype);
    let cuda_model = LlamaModel::new_with_dtype(llama_cfg.clone(), args.dtype);
    copy_parameters(&cpu_model, &cuda_model);

    zero_params(&cpu_model);
    let cpu_loss = llama_train_loss(&cpu_model, input.clone(), &targets, llama_cfg.vocab_size);
    let cpu_loss_value = scalar_value(&cpu_loss);
    cpu_loss.backward();
    let cpu_grads = collect_parameter_grads(&cpu_model);
    let mut cpu_step_model_data = Vec::new();
    if step {
        let mut cpu_opt =
            SGD::new_with_dtype(cpu_model.parameters(), 0.001, DType::F32).with_momentum(0.5);
        cpu_opt.step();
        cpu_step_model_data = collect_parameter_data(&cpu_model);
    }

    let cuda_enabled_guard = lumen::ops::cuda::set_enabled_scoped(true);
    let strict_device_execution_guard = set_strict_device_execution_scoped(true);
    cuda_model.to_cuda();
    let input_cuda = input.to_cuda();
    let targets_cuda = targets.to_cuda();
    zero_params(&cuda_model);
    let cuda_loss = llama_train_loss(&cuda_model, input_cuda, &targets_cuda, llama_cfg.vocab_size);
    let cuda_loss_value = scalar_value(&cuda_loss);
    cuda_loss.backward();
    assert_cuda_model_f32_grads_resident(&cuda_model, name);
    let cuda_grads = collect_parameter_grads(&cuda_model);
    let mut cuda_step_model_data = Vec::new();
    if step {
        let cuda_params = cuda_model.parameters();
        let mut cuda_opt =
            SGD::new_with_dtype(cuda_params.clone(), 0.001, DType::F32).with_momentum(0.5);
        cuda_opt.step();
        assert_eq!(
            cuda_opt.dev_velocity_count(),
            cuda_params.len(),
            "{name} expected one f32 momentum velocity per CUDA parameter"
        );
        assert!(
            cuda_opt.dev_all_velocities_are_f32_cuda_resident(),
            "{name} expected all momentum velocities to be f32 CUDA-resident"
        );
        cuda_step_model_data = collect_parameter_data(&cuda_model);
    }
    drop(strict_device_execution_guard);
    drop(cuda_enabled_guard);

    let (abs_tol, rel_tol) = dtype_check_tolerance(args.dtype);
    let loss_metric = check_metric(
        "loss",
        &[cpu_loss_value],
        &[cuda_loss_value],
        abs_tol,
        rel_tol,
    );
    let mut grad_max_abs = 0.0f32;
    let mut grad_max_rel = 0.0f32;
    assert_eq!(
        cpu_grads.len(),
        cuda_grads.len(),
        "training check grad parameter count mismatch"
    );
    for (idx, (cpu_grad, cuda_grad)) in cpu_grads.iter().zip(cuda_grads.iter()).enumerate() {
        let (abs, rel) = assert_close_vec(
            &format!("llama.train.grad[{idx}]"),
            cpu_grad,
            cuda_grad,
            abs_tol,
            rel_tol,
        );
        grad_max_abs = grad_max_abs.max(abs);
        grad_max_rel = grad_max_rel.max(rel);
    }

    let mut param_abs_rel = None;
    if step {
        let mut param_max_abs = 0.0f32;
        let mut param_max_rel = 0.0f32;
        assert_eq!(
            cpu_step_model_data.len(),
            cuda_step_model_data.len(),
            "training check parameter count mismatch after step"
        );
        for (idx, (cpu_param, cuda_param)) in cpu_step_model_data
            .iter()
            .zip(cuda_step_model_data.iter())
            .enumerate()
        {
            let (abs, rel) = assert_close_vec(
                &format!("llama.train.step.param[{idx}]"),
                cpu_param,
                cuda_param,
                abs_tol,
                rel_tol,
            );
            param_max_abs = param_max_abs.max(abs);
            param_max_rel = param_max_rel.max(rel);
        }
        param_abs_rel = Some((param_max_abs, param_max_rel));
    }

    let mut metrics = vec![
        loss_metric,
        CheckMetric {
            label: "grad",
            abs: grad_max_abs,
            rel: grad_max_rel,
            rmse: 0.0,
        },
    ];
    if let Some((abs, rel)) = param_abs_rel {
        metrics.push(CheckMetric {
            label: "param",
            abs,
            rel,
            rmse: 0.0,
        });
    }

    CheckResult { name, metrics }
}

fn check_llama_train_backward(args: &Args, cfg: ShapeConfig) -> CheckResult {
    check_llama_train(args, cfg, false, "llama.train.backward")
}

fn check_llama_train_step(args: &Args, cfg: ShapeConfig) -> CheckResult {
    check_llama_train(args, cfg, true, "llama.train.step")
}

fn check_matmul_backward(args: &Args, cfg: ShapeConfig) -> CheckResult {
    let (abs_tol, rel_tol) = dtype_check_tolerance(args.dtype);
    let a_data = sample_data(cfg.matmul_m * cfg.matmul_k, 0.01);
    let b_data = sample_data(cfg.matmul_k * cfg.matmul_n, -0.007);
    let coeff_data = sample_data(cfg.matmul_m * cfg.matmul_n, 0.003);

    let a_cpu = tensor_grad(&[cfg.matmul_m, cfg.matmul_k], a_data.clone(), args.dtype);
    let b_cpu = tensor_grad(&[cfg.matmul_n, cfg.matmul_k], b_data.clone(), args.dtype);
    let coeff_cpu = tensor_const(
        &[cfg.matmul_m, cfg.matmul_n],
        coeff_data.clone(),
        args.dtype,
    );
    let cpu_out = matmul(&a_cpu, &b_cpu);
    let cpu_loss = sum(&(&cpu_out * &coeff_cpu));
    let cpu_loss_value = scalar_value(&cpu_loss);
    cpu_loss.backward();
    let cpu_a_grad = tensor_grad_vec(&a_cpu, "matmul lhs");
    let cpu_b_grad = tensor_grad_vec(&b_cpu, "matmul rhs");

    let cuda_enabled_guard = lumen::ops::cuda::set_enabled_scoped(true);
    let strict_device_execution_guard = set_strict_device_execution_scoped(true);
    let a_cuda = tensor_grad(&[cfg.matmul_m, cfg.matmul_k], a_data, args.dtype).to_cuda();
    let b_cuda = tensor_grad(&[cfg.matmul_n, cfg.matmul_k], b_data, args.dtype).to_cuda();
    let coeff_cuda = tensor_const(&[cfg.matmul_m, cfg.matmul_n], coeff_data, args.dtype).to_cuda();
    let cuda_out = matmul(&a_cuda, &b_cuda);
    let cuda_loss = sum(&(&cuda_out * &coeff_cuda));
    let cuda_loss_value = scalar_value(&cuda_loss);
    cuda_loss.backward();
    let cuda_a_grad = tensor_grad_vec(&a_cuda, "matmul CUDA lhs");
    let cuda_b_grad = tensor_grad_vec(&b_cuda, "matmul CUDA rhs");
    drop(strict_device_execution_guard);
    drop(cuda_enabled_guard);

    let loss = check_metric(
        "loss",
        &[cpu_loss_value],
        &[cuda_loss_value],
        abs_tol,
        rel_tol,
    );
    let (a_abs, a_rel) = assert_close_vec(
        "matmul.backward.lhs_grad",
        &cpu_a_grad,
        &cuda_a_grad,
        abs_tol,
        rel_tol,
    );
    let (b_abs, b_rel) = assert_close_vec(
        "matmul.backward.rhs_grad",
        &cpu_b_grad,
        &cuda_b_grad,
        abs_tol,
        rel_tol,
    );
    CheckResult {
        name: "matmul.backward",
        metrics: vec![
            loss,
            CheckMetric {
                label: "grad",
                abs: a_abs.max(b_abs),
                rel: a_rel.max(b_rel),
                rmse: 0.0,
            },
        ],
    }
}

fn check_matmul_accuracy_case(
    case_name: &str,
    dtype_case_name: &str,
    scale_case_name: &str,
    m: usize,
    k: usize,
    n: usize,
    a_scale: f32,
    b_scale: f32,
    coeff_scale: f32,
    a_dtype: DType,
    b_dtype: DType,
    abs_tol: f32,
    rel_tol: f32,
    fwd: &mut AccuracyAgg,
    loss: &mut AccuracyAgg,
    lhs_grad: &mut AccuracyAgg,
    rhs_grad: &mut AccuracyAgg,
) {
    let a_data = sample_data(m * k, a_scale);
    let b_data = sample_data(k * n, b_scale);
    let coeff_data = sample_data(m * n, coeff_scale);

    let a_cpu = tensor_grad(&[m, k], a_data.clone(), a_dtype);
    let b_cpu = tensor_grad(&[n, k], b_data.clone(), b_dtype);
    let coeff_cpu = tensor_const(&[m, n], coeff_data.clone(), DType::F32);
    let cpu_out = matmul(&a_cpu, &b_cpu);
    let cpu_out_data = tensor_data_vec(&cpu_out);
    let cpu_loss = sum(&(&cpu_out * &coeff_cpu));
    let cpu_loss_value = scalar_value(&cpu_loss);
    cpu_loss.backward();
    let cpu_a_grad = tensor_grad_vec(&a_cpu, "matmul accuracy CPU lhs");
    let cpu_b_grad = tensor_grad_vec(&b_cpu, "matmul accuracy CPU rhs");

    let cuda_enabled_guard = lumen::ops::cuda::set_enabled_scoped(true);
    let strict_device_execution_guard = set_strict_device_execution_scoped(true);
    let a_cuda = tensor_grad(&[m, k], a_data, a_dtype).to_cuda();
    let b_cuda = tensor_grad(&[n, k], b_data, b_dtype).to_cuda();
    let coeff_cuda = tensor_const(&[m, n], coeff_data, DType::F32).to_cuda();
    let cuda_out = matmul(&a_cuda, &b_cuda);
    let cuda_out_data = tensor_data_vec(&cuda_out);
    let cuda_loss = sum(&(&cuda_out * &coeff_cuda));
    let cuda_loss_value = scalar_value(&cuda_loss);
    cuda_loss.backward();
    let cuda_a_grad = tensor_grad_vec(&a_cuda, "matmul accuracy CUDA lhs");
    let cuda_b_grad = tensor_grad_vec(&b_cuda, "matmul accuracy CUDA rhs");
    drop(strict_device_execution_guard);
    drop(cuda_enabled_guard);

    let prefix = format!("{case_name}.{dtype_case_name}.{scale_case_name}");
    fwd.observe(
        &format!("{prefix}.forward"),
        &cpu_out_data,
        &cuda_out_data,
        abs_tol,
        rel_tol,
    );
    loss.observe(
        &format!("{prefix}.loss"),
        &[cpu_loss_value],
        &[cuda_loss_value],
        abs_tol,
        rel_tol,
    );
    lhs_grad.observe(
        &format!("{prefix}.lhs_grad"),
        &cpu_a_grad,
        &cuda_a_grad,
        abs_tol,
        rel_tol,
    );
    rhs_grad.observe(
        &format!("{prefix}.rhs_grad"),
        &cpu_b_grad,
        &cuda_b_grad,
        abs_tol,
        rel_tol,
    );
}

fn check_matmul_matrix_accuracy(args: &Args, _cfg: ShapeConfig) -> CheckResult {
    let (abs_tol, rel_tol) = dtype_check_tolerance(args.dtype);
    let mut fwd = AccuracyAgg::default();
    let mut loss = AccuracyAgg::default();
    let mut lhs_grad = AccuracyAgg::default();
    let mut rhs_grad = AccuracyAgg::default();
    let cases = [
        ("square", 8usize, 9usize, 7usize),
        ("skinny_n", 6usize, 11usize, 1usize),
        ("skinny_m", 1usize, 13usize, 5usize),
        ("tall_k", 5usize, 17usize, 4usize),
    ];
    let scale_cases = [
        ("tiny", 0.002, -0.003, 0.004),
        ("normal", 0.011, -0.007, 0.005),
        ("wide", 0.071, -0.049, 0.013),
    ];
    let dtype_cases = if args.dtype == DType::F32 {
        vec![("f32_f32", DType::F32, DType::F32)]
    } else {
        vec![
            ("lowp_f32", args.dtype, DType::F32),
            ("f32_lowp", DType::F32, args.dtype),
            ("lowp_lowp", args.dtype, args.dtype),
        ]
    };

    for (case_name, m, k, n) in cases {
        for (dtype_case_name, a_dtype, b_dtype) in dtype_cases.iter().copied() {
            for (scale_case_name, a_scale, b_scale, coeff_scale) in scale_cases {
                check_matmul_accuracy_case(
                    case_name,
                    dtype_case_name,
                    scale_case_name,
                    m,
                    k,
                    n,
                    a_scale,
                    b_scale,
                    coeff_scale,
                    a_dtype,
                    b_dtype,
                    abs_tol,
                    rel_tol,
                    &mut fwd,
                    &mut loss,
                    &mut lhs_grad,
                    &mut rhs_grad,
                );
            }
        }
    }

    CheckResult {
        name: "matmul.matrix.accuracy",
        metrics: vec![
            CheckMetric {
                label: "forward",
                abs: fwd.abs,
                rel: fwd.rel,
                rmse: fwd.rmse,
            },
            CheckMetric {
                label: "loss",
                abs: loss.abs,
                rel: loss.rel,
                rmse: loss.rmse,
            },
            CheckMetric {
                label: "lhs_grad",
                abs: lhs_grad.abs,
                rel: lhs_grad.rel,
                rmse: lhs_grad.rmse,
            },
            CheckMetric {
                label: "rhs_grad",
                abs: rhs_grad.abs,
                rel: rhs_grad.rel,
                rmse: rhs_grad.rmse,
            },
        ],
    }
}

fn check_binary_forward(args: &Args, cfg: ShapeConfig) -> CheckResult {
    let a_data = sample_data(cfg.elem_len, 0.01);
    let b_data = sample_data(cfg.elem_len, -0.02);
    let cpu_a = tensor_no_grad(&[cfg.elem_len], a_data.clone(), args.dtype);
    let cpu_b = tensor_no_grad(&[cfg.elem_len], b_data.clone(), args.dtype);
    let cpu_out = no_grad(|| &cpu_a * &cpu_b);

    let cuda_enabled_guard = lumen::ops::cuda::set_enabled_scoped(true);
    let strict_device_execution_guard = set_strict_device_execution_scoped(true);
    let cuda_a = tensor_no_grad(&[cfg.elem_len], a_data, args.dtype).to_cuda();
    let cuda_b = tensor_no_grad(&[cfg.elem_len], b_data, args.dtype).to_cuda();
    let cuda_out = no_grad(|| &cuda_a * &cuda_b);
    drop(strict_device_execution_guard);
    drop(cuda_enabled_guard);

    let (abs_tol, rel_tol) = dtype_check_tolerance(args.dtype);
    CheckResult {
        name: "binary.same_shape.forward",
        metrics: vec![check_metric(
            "binary.same_shape.forward",
            &tensor_data_vec(&cpu_out),
            &tensor_data_vec(&cuda_out),
            abs_tol,
            rel_tol,
        )],
    }
}

fn check_binary_row_broadcast_forward(args: &Args, cfg: ShapeConfig) -> CheckResult {
    let last_dim = cfg.softmax_last;
    let rows = (cfg.elem_len / last_dim).max(1);
    let matrix_len = rows * last_dim;
    let matrix_data = sample_data(matrix_len, 0.01);
    let row_data = sample_data(last_dim, -0.02);
    let cpu_matrix = tensor_no_grad(&[rows, last_dim], matrix_data.clone(), args.dtype);
    let cpu_row = tensor_no_grad(&[last_dim], row_data.clone(), args.dtype);
    let cpu_out = no_grad(|| &cpu_matrix * &cpu_row);

    let cuda_enabled_guard = lumen::ops::cuda::set_enabled_scoped(true);
    let strict_device_execution_guard = set_strict_device_execution_scoped(true);
    let cuda_matrix = tensor_no_grad(&[rows, last_dim], matrix_data, args.dtype).to_cuda();
    let cuda_row = tensor_no_grad(&[last_dim], row_data, args.dtype).to_cuda();
    let cuda_out = no_grad(|| &cuda_matrix * &cuda_row);
    drop(strict_device_execution_guard);
    drop(cuda_enabled_guard);

    let (abs_tol, rel_tol) = dtype_check_tolerance(args.dtype);
    CheckResult {
        name: "binary.row_broadcast.forward",
        metrics: vec![check_metric(
            "binary.row_broadcast.forward",
            &tensor_data_vec(&cpu_out),
            &tensor_data_vec(&cuda_out),
            abs_tol,
            rel_tol,
        )],
    }
}

fn check_binary_special_broadcast_forward(args: &Args, _cfg: ShapeConfig) -> CheckResult {
    let (abs_tol, rel_tol) = dtype_check_tolerance(args.dtype);
    let mut metrics = Vec::new();
    let cases: [(&'static str, &[usize], &[usize]); 4] = [
        ("binary.row_scalar.forward", &[4, 5], &[4, 1]),
        ("binary.b1d_1h1.forward", &[2, 1, 3], &[1, 4, 1]),
        ("binary.b1d_1hd.forward", &[2, 1, 3], &[1, 4, 3]),
        (
            "binary.general_broadcast.forward",
            &[2, 1, 1, 3],
            &[1, 4, 2, 3],
        ),
    ];

    for (label, lhs_shape, rhs_shape) in cases {
        let lhs_len = lhs_shape.iter().product::<usize>();
        let rhs_len = rhs_shape.iter().product::<usize>();
        let lhs_data = sample_data(lhs_len, 0.017);
        let rhs_data = sample_data(rhs_len, -0.011);
        let cpu_lhs = tensor_no_grad(lhs_shape, lhs_data.clone(), args.dtype);
        let cpu_rhs = tensor_no_grad(rhs_shape, rhs_data.clone(), args.dtype);
        let cpu_out = no_grad(|| &cpu_lhs * &cpu_rhs);

        let cuda_enabled_guard = lumen::ops::cuda::set_enabled_scoped(true);
        let strict_device_execution_guard = set_strict_device_execution_scoped(true);
        let cuda_lhs = tensor_no_grad(lhs_shape, lhs_data, args.dtype).to_cuda();
        let cuda_rhs = tensor_no_grad(rhs_shape, rhs_data, args.dtype).to_cuda();
        let cuda_out = no_grad(|| &cuda_lhs * &cuda_rhs);
        drop(strict_device_execution_guard);
        drop(cuda_enabled_guard);

        metrics.push(check_metric(
            label,
            &tensor_data_vec(&cpu_out),
            &tensor_data_vec(&cuda_out),
            abs_tol,
            rel_tol,
        ));
    }

    CheckResult {
        name: "binary.special_broadcast.forward",
        metrics,
    }
}

fn check_cross_entropy_backward(args: &Args, cfg: ShapeConfig) -> CheckResult {
    let (abs_tol, rel_tol) = dtype_check_tolerance(args.dtype);
    let shape = [cfg.softmax_outer, cfg.softmax_last];
    let logits_data = sample_data(cfg.softmax_outer * cfg.softmax_last, 0.01);
    let target_data = one_hot_data(cfg.softmax_outer, cfg.softmax_last);

    let logits_cpu = tensor_grad(&shape, logits_data.clone(), args.dtype);
    let targets_cpu = tensor_const(&shape, target_data.clone(), args.dtype);
    let cpu_loss = CrossEntropyLoss::apply(&logits_cpu, &targets_cpu);
    let cpu_loss_value = scalar_value(&cpu_loss);
    cpu_loss.backward();
    let cpu_grad = tensor_grad_vec(&logits_cpu, "cross entropy logits");

    let cuda_enabled_guard = lumen::ops::cuda::set_enabled_scoped(true);
    let strict_device_execution_guard = set_strict_device_execution_scoped(true);
    let logits_cuda = tensor_grad(&shape, logits_data, args.dtype).to_cuda();
    let targets_cuda = tensor_const(&shape, target_data, args.dtype).to_cuda();
    let cuda_loss = CrossEntropyLoss::apply(&logits_cuda, &targets_cuda);
    let cuda_loss_value = scalar_value(&cuda_loss);
    cuda_loss.backward();
    let cuda_grad = tensor_grad_vec(&logits_cuda, "cross entropy CUDA logits");
    drop(strict_device_execution_guard);
    drop(cuda_enabled_guard);

    CheckResult {
        name: "cross_entropy.backward",
        metrics: vec![
            check_metric(
                "loss",
                &[cpu_loss_value],
                &[cuda_loss_value],
                abs_tol,
                rel_tol,
            ),
            check_metric("grad", &cpu_grad, &cuda_grad, abs_tol, rel_tol),
        ],
    }
}

fn check_elementwise_backward(args: &Args, cfg: ShapeConfig) -> CheckResult {
    let (abs_tol, rel_tol) = dtype_check_tolerance(args.dtype);
    let shape = [cfg.elem_len];
    let a_data = sample_data(cfg.elem_len, 0.01);
    let b_data = sample_data(cfg.elem_len, -0.02);

    let a_cpu = tensor_grad(&shape, a_data.clone(), args.dtype);
    let b_cpu = tensor_grad(&shape, b_data.clone(), args.dtype);
    let cpu_out = &(&a_cpu * &b_cpu) + &a_cpu;
    let cpu_loss = sum(&cpu_out);
    let cpu_loss_value = scalar_value(&cpu_loss);
    cpu_loss.backward();
    let cpu_a_grad = tensor_grad_vec(&a_cpu, "elementwise lhs");
    let cpu_b_grad = tensor_grad_vec(&b_cpu, "elementwise rhs");

    let cuda_enabled_guard = lumen::ops::cuda::set_enabled_scoped(true);
    let strict_device_execution_guard = set_strict_device_execution_scoped(true);
    let a_cuda = tensor_grad(&shape, a_data, args.dtype).to_cuda();
    let b_cuda = tensor_grad(&shape, b_data, args.dtype).to_cuda();
    let cuda_out = &(&a_cuda * &b_cuda) + &a_cuda;
    let cuda_loss = sum(&cuda_out);
    let cuda_loss_value = scalar_value(&cuda_loss);
    cuda_loss.backward();
    let cuda_a_grad = tensor_grad_vec(&a_cuda, "elementwise CUDA lhs");
    let cuda_b_grad = tensor_grad_vec(&b_cuda, "elementwise CUDA rhs");
    drop(strict_device_execution_guard);
    drop(cuda_enabled_guard);

    let (a_abs, a_rel) = assert_close_vec(
        "elementwise.backward.lhs_grad",
        &cpu_a_grad,
        &cuda_a_grad,
        abs_tol,
        rel_tol,
    );
    let (b_abs, b_rel) = assert_close_vec(
        "elementwise.backward.rhs_grad",
        &cpu_b_grad,
        &cuda_b_grad,
        abs_tol,
        rel_tol,
    );
    CheckResult {
        name: "elementwise.mul_add.backward",
        metrics: vec![
            check_metric(
                "loss",
                &[cpu_loss_value],
                &[cuda_loss_value],
                abs_tol,
                rel_tol,
            ),
            CheckMetric {
                label: "grad",
                abs: a_abs.max(b_abs),
                rel: a_rel.max(b_rel),
                rmse: 0.0,
            },
        ],
    }
}

#[derive(Clone, Copy)]
enum AccuracyBinaryOp {
    Add,
    Sub,
    Mul,
}

impl AccuracyBinaryOp {
    fn name(self) -> &'static str {
        match self {
            AccuracyBinaryOp::Add => "add",
            AccuracyBinaryOp::Sub => "sub",
            AccuracyBinaryOp::Mul => "mul",
        }
    }
}

fn apply_accuracy_binary(lhs: &Tensor, rhs: &Tensor, op: AccuracyBinaryOp) -> Tensor {
    match op {
        AccuracyBinaryOp::Add => lhs + rhs,
        AccuracyBinaryOp::Sub => lhs - rhs,
        AccuracyBinaryOp::Mul => lhs * rhs,
    }
}

#[derive(Default)]
struct AccuracyAgg {
    abs: f32,
    rel: f32,
    rmse: f32,
}

impl AccuracyAgg {
    fn observe(&mut self, label: &str, lhs: &[f32], rhs: &[f32], abs_tol: f32, rel_tol: f32) {
        let (abs, rel, rmse) = diff_stats(lhs, rhs);
        assert!(
            abs <= abs_tol || rel <= rel_tol,
            "{label} CPU/CUDA mismatch: max_abs={abs:.6e} max_rel={rel:.6e} rmse={rmse:.6e} abs_tol={abs_tol:.6e} rel_tol={rel_tol:.6e}"
        );
        self.abs = self.abs.max(abs);
        self.rel = self.rel.max(rel);
        self.rmse = self.rmse.max(rmse);
    }
}

fn check_broadcast_accuracy_case(
    case_name: &str,
    dtype_case_name: &str,
    scale_case_name: &str,
    lhs_shape: &[usize],
    rhs_shape: &[usize],
    lhs_scale: f32,
    rhs_scale: f32,
    lhs_dtype: DType,
    rhs_dtype: DType,
    op: AccuracyBinaryOp,
    abs_tol: f32,
    rel_tol: f32,
    fwd: &mut AccuracyAgg,
    lhs_grad: &mut AccuracyAgg,
    rhs_grad: &mut AccuracyAgg,
) {
    let lhs_len = lhs_shape.iter().product::<usize>();
    let rhs_len = rhs_shape.iter().product::<usize>();
    let lhs_data = sample_data(lhs_len, lhs_scale);
    let rhs_data = sample_data(rhs_len, rhs_scale);

    let lhs_cpu = tensor_grad(lhs_shape, lhs_data.clone(), lhs_dtype);
    let rhs_cpu = tensor_grad(rhs_shape, rhs_data.clone(), rhs_dtype);
    let cpu_out = apply_accuracy_binary(&lhs_cpu, &rhs_cpu, op);
    let cpu_out_data = tensor_data_vec(&cpu_out);
    let cpu_loss = sum(&cpu_out);
    cpu_loss.backward();
    let cpu_lhs_grad = tensor_grad_vec(&lhs_cpu, "broadcast accuracy CPU lhs");
    let cpu_rhs_grad = tensor_grad_vec(&rhs_cpu, "broadcast accuracy CPU rhs");

    let cuda_enabled_guard = lumen::ops::cuda::set_enabled_scoped(true);
    let strict_device_execution_guard = set_strict_device_execution_scoped(true);
    let lhs_cuda = tensor_grad(lhs_shape, lhs_data, lhs_dtype).to_cuda();
    let rhs_cuda = tensor_grad(rhs_shape, rhs_data, rhs_dtype).to_cuda();
    let cuda_out = apply_accuracy_binary(&lhs_cuda, &rhs_cuda, op);
    let cuda_out_data = tensor_data_vec(&cuda_out);
    let cuda_loss = sum(&cuda_out);
    cuda_loss.backward();
    let cuda_lhs_grad = tensor_grad_vec(&lhs_cuda, "broadcast accuracy CUDA lhs");
    let cuda_rhs_grad = tensor_grad_vec(&rhs_cuda, "broadcast accuracy CUDA rhs");
    drop(strict_device_execution_guard);
    drop(cuda_enabled_guard);

    let label = format!(
        "{case_name}.{dtype_case_name}.{scale_case_name}.{}.forward",
        op.name()
    );
    fwd.observe(&label, &cpu_out_data, &cuda_out_data, abs_tol, rel_tol);
    let label = format!(
        "{case_name}.{dtype_case_name}.{scale_case_name}.{}.lhs_grad",
        op.name()
    );
    lhs_grad.observe(&label, &cpu_lhs_grad, &cuda_lhs_grad, abs_tol, rel_tol);
    let label = format!(
        "{case_name}.{dtype_case_name}.{scale_case_name}.{}.rhs_grad",
        op.name()
    );
    rhs_grad.observe(&label, &cpu_rhs_grad, &cuda_rhs_grad, abs_tol, rel_tol);
}

fn check_elementwise_broadcast_matrix_accuracy(args: &Args, _cfg: ShapeConfig) -> CheckResult {
    let (abs_tol, rel_tol) = dtype_check_tolerance(args.dtype);
    let mut fwd = AccuracyAgg::default();
    let mut lhs_grad = AccuracyAgg::default();
    let mut rhs_grad = AccuracyAgg::default();
    let cases: [(&str, &[usize], &[usize]); 7] = [
        ("same", &[4, 5], &[4, 5]),
        ("row_vector", &[4, 5], &[5]),
        ("row_scalar", &[4, 5], &[4, 1]),
        ("scalar", &[4, 5], &[1]),
        ("b1d_1h1", &[2, 1, 3], &[1, 4, 1]),
        ("b1d_1hd", &[2, 1, 3], &[1, 4, 3]),
        ("general", &[2, 1, 1, 3], &[1, 4, 2, 3]),
    ];
    let scale_cases = [
        ("tiny", 0.003, -0.002),
        ("normal", 0.017, -0.011),
        ("wide", 0.19, -0.13),
    ];
    let dtype_cases = if args.dtype == DType::F32 {
        vec![("f32_f32", DType::F32, DType::F32)]
    } else {
        vec![
            ("lowp_f32", args.dtype, DType::F32),
            ("f32_lowp", DType::F32, args.dtype),
            ("lowp_lowp", args.dtype, args.dtype),
        ]
    };
    for (case_name, lhs_shape, rhs_shape) in cases {
        for (dtype_case_name, lhs_dtype, rhs_dtype) in dtype_cases.iter().copied() {
            for (scale_case_name, lhs_scale, rhs_scale) in scale_cases {
                for op in [
                    AccuracyBinaryOp::Add,
                    AccuracyBinaryOp::Sub,
                    AccuracyBinaryOp::Mul,
                ] {
                    check_broadcast_accuracy_case(
                        case_name,
                        dtype_case_name,
                        scale_case_name,
                        lhs_shape,
                        rhs_shape,
                        lhs_scale,
                        rhs_scale,
                        lhs_dtype,
                        rhs_dtype,
                        op,
                        abs_tol,
                        rel_tol,
                        &mut fwd,
                        &mut lhs_grad,
                        &mut rhs_grad,
                    );
                }
            }
        }
    }
    CheckResult {
        name: "elementwise.broadcast_matrix.accuracy",
        metrics: vec![
            CheckMetric {
                label: "forward",
                abs: fwd.abs,
                rel: fwd.rel,
                rmse: fwd.rmse,
            },
            CheckMetric {
                label: "lhs_grad",
                abs: lhs_grad.abs,
                rel: lhs_grad.rel,
                rmse: lhs_grad.rmse,
            },
            CheckMetric {
                label: "rhs_grad",
                abs: rhs_grad.abs,
                rel: rhs_grad.rel,
                rmse: rhs_grad.rmse,
            },
        ],
    }
}

fn check_unary_relu_forward(args: &Args, cfg: ShapeConfig) -> CheckResult {
    let input_data = sample_data(cfg.elem_len, 0.012);
    let relu = ReLU::new();

    let cpu_input = tensor_no_grad(&[cfg.elem_len], input_data.clone(), args.dtype);
    let cpu_out = no_grad(|| relu.forward(cpu_input));

    let cuda_enabled_guard = lumen::ops::cuda::set_enabled_scoped(true);
    let strict_device_execution_guard = set_strict_device_execution_scoped(true);
    let cuda_input = tensor_no_grad(&[cfg.elem_len], input_data, args.dtype).to_cuda();
    let cuda_out = no_grad(|| relu.forward(cuda_input));
    drop(strict_device_execution_guard);
    drop(cuda_enabled_guard);

    assert!(
        !cuda_out.dev_has_host_f32_data(),
        "CUDA ReLU check should keep forward output resident until accuracy materialization"
    );
    if args.dtype == DType::I8 {
        assert!(
            cuda_out.dev_has_cuda_i8_data(),
            "CUDA I8 ReLU should produce an I8 resident output buffer"
        );
    }

    let (abs_tol, rel_tol) = dtype_check_tolerance(args.dtype);
    CheckResult {
        name: "unary.relu.forward",
        metrics: vec![check_metric(
            "relu.forward",
            &tensor_data_vec(&cpu_out),
            &tensor_data_vec(&cuda_out),
            abs_tol,
            rel_tol,
        )],
    }
}

fn check_unary_silu_backward(args: &Args, cfg: ShapeConfig) -> CheckResult {
    let (abs_tol, rel_tol) = dtype_check_tolerance(args.dtype);
    let shape = [cfg.elem_len];
    let input_data = sample_data(cfg.elem_len, 0.012);
    let coeff_data = sample_data(cfg.elem_len, -0.007);
    let silu = SiLU::new();

    let cpu_input = tensor_grad(&shape, input_data.clone(), args.dtype);
    let coeff_cpu = tensor_no_grad(&shape, coeff_data.clone(), DType::F32);
    let cpu_out = silu.forward(cpu_input.clone());
    let cpu_loss = sum(&(&cpu_out * &coeff_cpu));
    cpu_loss.backward();
    let cpu_loss_value = scalar_value(&cpu_loss);
    let cpu_grad = cpu_input
        .grad()
        .expect("CPU SiLU backward should populate input grad")
        .iter()
        .copied()
        .collect::<Vec<_>>();

    let cuda_enabled_guard = lumen::ops::cuda::set_enabled_scoped(true);
    let strict_device_execution_guard = set_strict_device_execution_scoped(true);
    let cuda_input = tensor_grad(&shape, input_data, args.dtype).to_cuda();
    let coeff_cuda = tensor_no_grad(&shape, coeff_data, DType::F32).to_cuda();
    let cuda_out = silu.forward(cuda_input.clone());
    let cuda_loss = sum(&(&cuda_out * &coeff_cuda));
    cuda_loss.backward();
    let cuda_loss_value = scalar_value(&cuda_loss);
    let cuda_grad = cuda_input
        .grad()
        .expect("CUDA SiLU backward should populate input grad")
        .iter()
        .copied()
        .collect::<Vec<_>>();
    drop(strict_device_execution_guard);
    drop(cuda_enabled_guard);

    CheckResult {
        name: "unary.silu.backward",
        metrics: vec![
            check_metric(
                "loss",
                &[cpu_loss_value],
                &[cuda_loss_value],
                abs_tol,
                rel_tol,
            ),
            check_metric("grad", &cpu_grad, &cuda_grad, abs_tol, rel_tol),
        ],
    }
}

fn check_softmax_backward(args: &Args, cfg: ShapeConfig) -> CheckResult {
    let (abs_tol, rel_tol) = dtype_check_tolerance(args.dtype);
    let shape = [cfg.softmax_outer, cfg.softmax_last];
    let input_data = sample_data(cfg.softmax_outer * cfg.softmax_last, 0.01);
    let coeff_data = sample_data(cfg.softmax_outer * cfg.softmax_last, -0.004);
    let softmax = Softmax::new(1);

    let input_cpu = tensor_grad(&shape, input_data.clone(), args.dtype);
    let coeff_cpu = tensor_const(&shape, coeff_data.clone(), args.dtype);
    let cpu_out = softmax.forward(input_cpu.clone());
    let cpu_loss = sum(&(&cpu_out * &coeff_cpu));
    let cpu_loss_value = scalar_value(&cpu_loss);
    cpu_loss.backward();
    let cpu_grad = tensor_grad_vec(&input_cpu, "softmax input");

    let cuda_enabled_guard = lumen::ops::cuda::set_enabled_scoped(true);
    let strict_device_execution_guard = set_strict_device_execution_scoped(true);
    let input_cuda = tensor_grad(&shape, input_data, args.dtype).to_cuda();
    let coeff_cuda = tensor_const(&shape, coeff_data, args.dtype).to_cuda();
    let cuda_out = softmax.forward(input_cuda.clone());
    let cuda_loss = sum(&(&cuda_out * &coeff_cuda));
    let cuda_loss_value = scalar_value(&cuda_loss);
    cuda_loss.backward();
    let cuda_grad = tensor_grad_vec(&input_cuda, "softmax CUDA input");
    drop(strict_device_execution_guard);
    drop(cuda_enabled_guard);

    CheckResult {
        name: "softmax.backward",
        metrics: vec![
            check_metric(
                "loss",
                &[cpu_loss_value],
                &[cuda_loss_value],
                abs_tol,
                rel_tol,
            ),
            check_metric("grad", &cpu_grad, &cuda_grad, abs_tol, rel_tol),
        ],
    }
}

fn check_softmax_forward(args: &Args, cfg: ShapeConfig) -> CheckResult {
    let (abs_tol, rel_tol) = dtype_check_tolerance(args.dtype);
    let shape = [cfg.softmax_outer, cfg.softmax_last];
    let input_data = sample_data(cfg.softmax_outer * cfg.softmax_last, 0.01);
    let softmax = Softmax::new(1);

    let input_cpu = tensor_no_grad(&shape, input_data.clone(), args.dtype);
    let cpu_out = no_grad(|| softmax.forward(input_cpu));
    let cpu_values = tensor_data_vec(&cpu_out);

    let cuda_enabled_guard = lumen::ops::cuda::set_enabled_scoped(true);
    let strict_device_execution_guard = set_strict_device_execution_scoped(true);
    let input_cuda = tensor_no_grad(&shape, input_data, args.dtype).to_cuda();
    let cuda_out = no_grad(|| softmax.forward(input_cuda));
    let cuda_values = tensor_data_vec(&cuda_out);
    drop(strict_device_execution_guard);
    drop(cuda_enabled_guard);

    CheckResult {
        name: "softmax.forward",
        metrics: vec![check_metric(
            "forward",
            &cpu_values,
            &cuda_values,
            abs_tol,
            rel_tol,
        )],
    }
}

fn fused_softmax_shape(cfg: ShapeConfig) -> (usize, usize, usize, usize) {
    let q_len = cfg.softmax_last;
    let batch_heads = (cfg.softmax_outer / q_len).max(1);
    (1, batch_heads, q_len, q_len)
}

fn check_fused_softmax_backward(args: &Args, cfg: ShapeConfig) -> CheckResult {
    let (abs_tol, rel_tol) = dtype_check_tolerance(args.dtype);
    let (batch, heads, q_len, k_len) = fused_softmax_shape(cfg);
    let shape = [batch, heads, q_len, k_len];
    let len = batch * heads * q_len * k_len;
    let input_data = sample_data(len, 0.01);
    let coeff_data = sample_data(len, -0.004);
    let scale = 0.75f32;

    let input_cpu = tensor_grad(&shape, input_data.clone(), args.dtype);
    let coeff_cpu = tensor_const(&shape, coeff_data.clone(), args.dtype);
    let cpu_out = fused_softmax(&input_cpu, scale, true);
    let cpu_loss = sum(&(&cpu_out * &coeff_cpu));
    let cpu_loss_value = scalar_value(&cpu_loss);
    cpu_loss.backward();
    let cpu_grad = tensor_grad_vec(&input_cpu, "fused softmax input");

    let cuda_enabled_guard = lumen::ops::cuda::set_enabled_scoped(true);
    let strict_device_execution_guard = set_strict_device_execution_scoped(true);
    let input_cuda = tensor_grad(&shape, input_data, args.dtype).to_cuda();
    let coeff_cuda = tensor_const(&shape, coeff_data, args.dtype).to_cuda();
    let cuda_out = fused_softmax(&input_cuda, scale, true);
    let cuda_loss = sum(&(&cuda_out * &coeff_cuda));
    let cuda_loss_value = scalar_value(&cuda_loss);
    cuda_loss.backward();
    let cuda_grad = tensor_grad_vec(&input_cuda, "fused softmax CUDA input");
    drop(strict_device_execution_guard);
    drop(cuda_enabled_guard);

    CheckResult {
        name: "fused_softmax.backward",
        metrics: vec![
            check_metric(
                "loss",
                &[cpu_loss_value],
                &[cuda_loss_value],
                abs_tol,
                rel_tol,
            ),
            check_metric("grad", &cpu_grad, &cuda_grad, abs_tol, rel_tol),
        ],
    }
}

fn check_fused_softmax_forward(args: &Args, cfg: ShapeConfig) -> CheckResult {
    let (abs_tol, rel_tol) = dtype_check_tolerance(args.dtype);
    let (batch, heads, q_len, k_len) = fused_softmax_shape(cfg);
    let shape = [batch, heads, q_len, k_len];
    let len = batch * heads * q_len * k_len;
    let input_data = sample_data(len, 0.01);
    let scale = 0.75f32;

    let input_cpu = tensor_no_grad(&shape, input_data.clone(), args.dtype);
    let cpu_out = no_grad(|| fused_softmax(&input_cpu, scale, true));
    let cpu_values = tensor_data_vec(&cpu_out);

    let cuda_enabled_guard = lumen::ops::cuda::set_enabled_scoped(true);
    let strict_device_execution_guard = set_strict_device_execution_scoped(true);
    let input_cuda = tensor_no_grad(&shape, input_data, args.dtype).to_cuda();
    let cuda_out = no_grad(|| fused_softmax(&input_cuda, scale, true));
    let cuda_values = tensor_data_vec(&cuda_out);
    drop(strict_device_execution_guard);
    drop(cuda_enabled_guard);

    CheckResult {
        name: "fused_softmax.forward",
        metrics: vec![check_metric(
            "forward",
            &cpu_values,
            &cuda_values,
            abs_tol,
            rel_tol,
        )],
    }
}

fn check_embedding_forward(args: &Args, cfg: ShapeConfig) -> CheckResult {
    let (abs_tol, rel_tol) = dtype_check_tolerance(args.dtype);
    let vocab_size = cfg.softmax_last.max(64);
    let embed_dim = cfg.attention_hidden;
    let token_count = cfg.softmax_outer;
    let weight_data = sample_data(vocab_size * embed_dim, 0.01);
    let index_data = token_id_data(1, token_count, vocab_size);

    let emb_cpu = Embedding::new_with_dtype(vocab_size, embed_dim, args.dtype);
    emb_cpu.weight.set_array_f32_with_dtype(
        array_from_vec(&[vocab_size, embed_dim], weight_data.clone()),
        args.dtype,
    );
    let indices_cpu = token_tensor(&[token_count], index_data.clone());
    let cpu_out = no_grad(|| emb_cpu.forward(&indices_cpu));
    let cpu_values = tensor_data_vec(&cpu_out);

    let cuda_enabled_guard = lumen::ops::cuda::set_enabled_scoped(true);
    let strict_device_execution_guard = set_strict_device_execution_scoped(true);
    let emb_cuda = Embedding::new_with_dtype(vocab_size, embed_dim, args.dtype);
    emb_cuda.weight.set_array_f32_with_dtype(
        array_from_vec(&[vocab_size, embed_dim], weight_data),
        args.dtype,
    );
    emb_cuda.weight.to_cuda_inplace();
    let indices_cuda = token_tensor(&[token_count], index_data).to_cuda();
    let cuda_out = no_grad(|| emb_cuda.forward(&indices_cuda));
    let cuda_values = tensor_data_vec(&cuda_out);
    drop(strict_device_execution_guard);
    drop(cuda_enabled_guard);

    CheckResult {
        name: "embedding.forward",
        metrics: vec![check_metric(
            "forward",
            &cpu_values,
            &cuda_values,
            abs_tol,
            rel_tol,
        )],
    }
}

fn check_rms_norm_forward(args: &Args, cfg: ShapeConfig) -> CheckResult {
    let (abs_tol, rel_tol) = dtype_check_tolerance(args.dtype);
    let rows = cfg.softmax_outer;
    let dim = cfg.attention_hidden;
    let input_data = sample_data(rows * dim, 0.01);
    let weight_data = sample_data(dim, 0.003)
        .into_iter()
        .map(|value| value + 1.0)
        .collect::<Vec<_>>();

    let norm_cpu = RMSNorm::new_with_dtype(dim, 1e-5, args.dtype);
    norm_cpu
        .weight
        .set_array_f32_with_dtype(array_from_vec(&[dim], weight_data.clone()), args.dtype);
    let input_cpu = tensor_no_grad(&[rows, dim], input_data.clone(), args.dtype);
    let cpu_out = no_grad(|| norm_cpu.forward(input_cpu));
    let cpu_values = tensor_data_vec(&cpu_out);

    let cuda_enabled_guard = lumen::ops::cuda::set_enabled_scoped(true);
    let strict_device_execution_guard = set_strict_device_execution_scoped(true);
    let norm_cuda = RMSNorm::new_with_dtype(dim, 1e-5, args.dtype);
    norm_cuda
        .weight
        .set_array_f32_with_dtype(array_from_vec(&[dim], weight_data), args.dtype);
    norm_cuda.weight.to_cuda_inplace();
    let input_cuda = tensor_no_grad(&[rows, dim], input_data, args.dtype).to_cuda();
    let cuda_out = no_grad(|| norm_cuda.forward(input_cuda));
    let cuda_values = tensor_data_vec(&cuda_out);
    drop(strict_device_execution_guard);
    drop(cuda_enabled_guard);

    CheckResult {
        name: "rms_norm.forward",
        metrics: vec![check_metric(
            "forward",
            &cpu_values,
            &cuda_values,
            abs_tol,
            rel_tol,
        )],
    }
}

fn check_rope_forward(args: &Args, cfg: ShapeConfig) -> CheckResult {
    let (abs_tol, rel_tol) = dtype_check_tolerance(args.dtype);
    let batch = cfg.attention_batch;
    let heads = cfg.attention_heads;
    let seq = cfg.attention_seq;
    let dim = cfg.attention_hidden / cfg.attention_heads;
    let shape = [batch, heads, seq, dim];
    let input_data = sample_data(batch * heads * seq * dim, 0.01);
    let offset = 1usize;
    let max_seq = seq + offset + 1;

    let rope_cpu = RotaryEmbedding::new_with_dtype(dim, max_seq, 10000.0, args.dtype);
    let input_cpu = tensor_no_grad(&shape, input_data.clone(), args.dtype);
    let cpu_out = no_grad(|| rope_cpu.forward(&input_cpu, offset));
    let cpu_values = tensor_data_vec(&cpu_out);

    let cuda_enabled_guard = lumen::ops::cuda::set_enabled_scoped(true);
    let strict_device_execution_guard = set_strict_device_execution_scoped(true);
    let rope_cuda = RotaryEmbedding::new_with_dtype(dim, max_seq, 10000.0, args.dtype);
    rope_cuda.to_cuda();
    let input_cuda = tensor_no_grad(&shape, input_data, args.dtype).to_cuda();
    let cuda_out = no_grad(|| rope_cuda.forward(&input_cuda, offset));
    let cuda_values = tensor_data_vec(&cuda_out);
    drop(strict_device_execution_guard);
    drop(cuda_enabled_guard);

    CheckResult {
        name: "rope.forward",
        metrics: vec![check_metric(
            "forward",
            &cpu_values,
            &cuda_values,
            abs_tol,
            rel_tol,
        )],
    }
}

fn check_optimizer_step(
    args: &Args,
    cfg: ShapeConfig,
    adam: bool,
    state_dtype: DType,
    name: &'static str,
) -> CheckResult {
    let (abs_tol, rel_tol) = dtype_check_tolerance(args.dtype);
    let shape = [cfg.elem_len];
    let param_data = sample_data(cfg.elem_len, 0.01);
    let grad_data = sample_data(cfg.elem_len, -0.002);

    let param_cpu = tensor_grad(&shape, param_data.clone(), args.dtype);
    param_cpu.add_grad(grad_array(&shape, &grad_data));
    if adam {
        let mut opt = Adam::new_with_dtype(vec![param_cpu.clone()], 0.001, state_dtype);
        opt.step();
    } else {
        let mut opt = SGD::new(vec![param_cpu.clone()], 0.001);
        opt.step();
    }
    let cpu_param = tensor_data_vec(&param_cpu);

    let cuda_enabled_guard = lumen::ops::cuda::set_enabled_scoped(true);
    let strict_device_execution_guard = set_strict_device_execution_scoped(true);
    let param_cuda = tensor_grad(&shape, param_data, args.dtype).to_cuda();
    param_cuda.add_grad(grad_array(&shape, &grad_data));
    if adam {
        let mut opt = Adam::new_with_dtype(vec![param_cuda.clone()], 0.001, state_dtype);
        opt.step();
    } else {
        let mut opt = SGD::new(vec![param_cuda.clone()], 0.001);
        opt.step();
    }
    let cuda_param = tensor_data_vec(&param_cuda);
    drop(strict_device_execution_guard);
    drop(cuda_enabled_guard);

    CheckResult {
        name,
        metrics: vec![check_metric(
            "param",
            &cpu_param,
            &cuda_param,
            abs_tol,
            rel_tol,
        )],
    }
}

fn optimizer_batched_len(cfg: ShapeConfig) -> usize {
    (cfg.elem_len / 128).clamp(1024, 1 << 14)
}

fn check_optimizer_batched_step(
    args: &Args,
    cfg: ShapeConfig,
    adam: bool,
    name: &'static str,
) -> CheckResult {
    let (abs_tol, rel_tol) = dtype_check_tolerance(args.dtype);
    let len = optimizer_batched_len(cfg);
    let shape = [len];
    let mut cpu_params = Vec::with_capacity(OPTIMIZER_BATCHED_PARAM_COUNT);
    let mut cuda_params = Vec::with_capacity(OPTIMIZER_BATCHED_PARAM_COUNT);
    let mut expected = Vec::with_capacity(OPTIMIZER_BATCHED_PARAM_COUNT * len);
    let mut actual = Vec::with_capacity(OPTIMIZER_BATCHED_PARAM_COUNT * len);

    for idx in 0..OPTIMIZER_BATCHED_PARAM_COUNT {
        let param_data = sample_data(len, 0.01 + idx as f32 * 0.0003);
        let grad_data = sample_data(len, -0.002 - idx as f32 * 0.0001);
        let param = tensor_grad(&shape, param_data, args.dtype);
        param.add_grad(grad_array(&shape, &grad_data));
        cpu_params.push(param);
    }
    if adam {
        let mut opt = Adam::new_with_dtype(cpu_params.clone(), 0.001, DType::F32);
        opt.step();
    } else {
        let mut opt = SGD::new(cpu_params.clone(), 0.001);
        opt.step();
    }
    for param in &cpu_params {
        expected.extend(tensor_data_vec(param));
    }

    let cuda_enabled_guard = lumen::ops::cuda::set_enabled_scoped(true);
    let strict_device_execution_guard = set_strict_device_execution_scoped(true);
    for idx in 0..OPTIMIZER_BATCHED_PARAM_COUNT {
        let param_data = sample_data(len, 0.01 + idx as f32 * 0.0003);
        let grad_data = sample_data(len, -0.002 - idx as f32 * 0.0001);
        let param = tensor_grad(&shape, param_data, args.dtype).to_cuda();
        param.add_grad(grad_array(&shape, &grad_data));
        cuda_params.push(param);
    }
    if adam {
        let mut opt = Adam::new_with_dtype(cuda_params.clone(), 0.001, DType::F32);
        opt.step();
    } else {
        let mut opt = SGD::new(cuda_params.clone(), 0.001);
        opt.step();
    }
    for param in &cuda_params {
        actual.extend(tensor_data_vec(param));
    }
    drop(strict_device_execution_guard);
    drop(cuda_enabled_guard);

    CheckResult {
        name,
        metrics: vec![check_metric("param", &expected, &actual, abs_tol, rel_tol)],
    }
}

fn check_sgd_step(args: &Args, cfg: ShapeConfig) -> CheckResult {
    check_optimizer_step(args, cfg, false, args.dtype, "optimizer.sgd.step")
}

fn check_adam_step(args: &Args, cfg: ShapeConfig) -> CheckResult {
    check_optimizer_step(args, cfg, true, args.dtype, "optimizer.adam.step")
}

fn check_adam_f32_state_step(args: &Args, cfg: ShapeConfig) -> CheckResult {
    check_optimizer_step(args, cfg, true, DType::F32, "optimizer.adam_f32_state.step")
}

fn check_sgd_batched_step(args: &Args, cfg: ShapeConfig) -> CheckResult {
    check_optimizer_batched_step(args, cfg, false, "optimizer.sgd_batched.step")
}

fn check_adam_f32_state_batched_step(args: &Args, cfg: ShapeConfig) -> CheckResult {
    check_optimizer_batched_step(args, cfg, true, "optimizer.adam_f32_state_batched.step")
}

fn fused_qkv_prefill_cpu_reference(
    input: &Tensor,
    q_weight: &Tensor,
    k_weight: &Tensor,
    v_weight: &Tensor,
    batch: usize,
    seq: usize,
    hidden: usize,
    heads: usize,
    kv_heads: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let head_dim = hidden / heads;
    let kv_hidden = kv_heads * head_dim;
    let x = input.data_ref().iter().copied().collect::<Vec<_>>();
    let q_w = q_weight.data_ref().iter().copied().collect::<Vec<_>>();
    let k_w = k_weight.data_ref().iter().copied().collect::<Vec<_>>();
    let v_w = v_weight.data_ref().iter().copied().collect::<Vec<_>>();
    let mut q_out = vec![0.0f32; batch * heads * seq * head_dim];
    let mut k_out = vec![0.0f32; batch * kv_heads * seq * head_dim];
    let mut v_out = vec![0.0f32; batch * kv_heads * seq * head_dim];

    for bb in 0..batch {
        for ss in 0..seq {
            let x_base = (bb * seq + ss) * hidden;
            for out_idx in 0..hidden {
                let mut sum = 0.0f32;
                for kk in 0..hidden {
                    sum += x[x_base + kk] * q_w[out_idx * hidden + kk];
                }
                let head = out_idx / head_dim;
                let dim = out_idx % head_dim;
                q_out[((bb * heads + head) * seq + ss) * head_dim + dim] = sum;
            }
            for out_idx in 0..kv_hidden {
                let mut k_sum = 0.0f32;
                let mut v_sum = 0.0f32;
                for kk in 0..hidden {
                    let x_val = x[x_base + kk];
                    k_sum += x_val * k_w[out_idx * hidden + kk];
                    v_sum += x_val * v_w[out_idx * hidden + kk];
                }
                let head = out_idx / head_dim;
                let dim = out_idx % head_dim;
                let offset = ((bb * kv_heads + head) * seq + ss) * head_dim + dim;
                k_out[offset] = k_sum;
                v_out[offset] = v_sum;
            }
        }
    }

    (q_out, k_out, v_out)
}

fn bench_matmul_forward(args: &Args, cfg: ShapeConfig) -> BenchResult {
    let a = tensor_no_grad(
        &[cfg.matmul_m, cfg.matmul_k],
        sample_data(cfg.matmul_m * cfg.matmul_k, 0.01),
        args.dtype,
    );
    let b = tensor_no_grad(
        &[cfg.matmul_n, cfg.matmul_k],
        sample_data(cfg.matmul_k * cfg.matmul_n, -0.007),
        args.dtype,
    );
    let cpu = measure(args, || {
        let out = no_grad(|| matmul(&a, &b));
        black_box(out.shape_vec());
    });
    let cuda = if lumen::ops::cuda::is_available() {
        let a_cuda = a.to_cuda();
        let b_cuda = b.to_cuda();
        measure_cuda(args, || {
            let out = no_grad(|| matmul(&a_cuda, &b_cuda));
            black_box(out.shape_vec());
        })
    } else {
        None
    };
    BenchResult {
        name: "matmul.forward",
        cpu,
        cuda,
    }
}

fn bench_batch_matmul_forward(args: &Args, cfg: ShapeConfig) -> BenchResult {
    let batch = cfg.attention_batch;
    let heads = cfg.attention_heads;
    let seq = cfg.attention_seq;
    let head_dim = cfg.attention_hidden / cfg.attention_heads;
    let lhs_len = batch * heads * seq * head_dim;
    let rhs_len = batch * heads * head_dim * seq;
    let lhs = tensor_no_grad(
        &[batch, heads, seq, head_dim],
        sample_data(lhs_len, 0.01),
        args.dtype,
    );
    let rhs = tensor_no_grad(
        &[batch, heads, head_dim, seq],
        sample_data(rhs_len, -0.007),
        args.dtype,
    );
    let cpu = measure(args, || {
        let out = no_grad(|| batch_matmul(&lhs, &rhs));
        black_box(out.shape_vec());
    });
    let cuda = if lumen::ops::cuda::is_available() {
        let lhs_cuda = lhs.to_cuda();
        let rhs_cuda = rhs.to_cuda();
        measure_cuda(args, || {
            let out = no_grad(|| batch_matmul(&lhs_cuda, &rhs_cuda));
            black_box(out.shape_vec());
        })
    } else {
        None
    };
    BenchResult {
        name: "batch_matmul.forward",
        cpu,
        cuda,
    }
}

fn bench_matmul_backward(args: &Args, cfg: ShapeConfig) -> BenchResult {
    let a = tensor_grad(
        &[cfg.matmul_m, cfg.matmul_k],
        sample_data(cfg.matmul_m * cfg.matmul_k, 0.01),
        args.dtype,
    );
    let b = tensor_grad(
        &[cfg.matmul_n, cfg.matmul_k],
        sample_data(cfg.matmul_k * cfg.matmul_n, -0.007),
        args.dtype,
    );
    let coeff = tensor_const(
        &[cfg.matmul_m, cfg.matmul_n],
        sample_data(cfg.matmul_m * cfg.matmul_n, 0.003),
        args.dtype,
    );
    let cpu = measure(args, || {
        zero_all(&[&a, &b]);
        let out = matmul(&a, &b);
        let loss = sum(&(&out * &coeff));
        loss.backward();
        black_box(a.grad().is_some());
    });
    let cuda = if lumen::ops::cuda::is_available() {
        let a_cuda = a.to_cuda();
        let b_cuda = b.to_cuda();
        let coeff_cuda = coeff.to_cuda();
        measure_cuda(args, || {
            zero_all(&[&a_cuda, &b_cuda]);
            let out = matmul(&a_cuda, &b_cuda);
            let loss = sum(&(&out * &coeff_cuda));
            loss.backward();
            black_box(a_cuda.requires_grad());
        })
    } else {
        None
    };
    BenchResult {
        name: "matmul.backward",
        cpu,
        cuda,
    }
}

fn bench_elementwise_forward(args: &Args, cfg: ShapeConfig) -> BenchResult {
    let a = tensor_no_grad(&[cfg.elem_len], sample_data(cfg.elem_len, 0.01), args.dtype);
    let b = tensor_no_grad(
        &[cfg.elem_len],
        sample_data(cfg.elem_len, -0.02),
        args.dtype,
    );
    let cpu = measure(args, || {
        let out = no_grad(|| &(&a * &b) + &a);
        black_box(out.shape_vec());
    });
    let cuda = if lumen::ops::cuda::is_available() {
        let a_cuda = a.to_cuda();
        let b_cuda = b.to_cuda();
        measure_cuda(args, || {
            let out = no_grad(|| &(&a_cuda * &b_cuda) + &a_cuda);
            black_box(out.shape_vec());
        })
    } else {
        None
    };
    BenchResult {
        name: "elementwise.mul_add.forward",
        cpu,
        cuda,
    }
}

fn bench_binary_forward(args: &Args, cfg: ShapeConfig) -> BenchResult {
    let a = tensor_no_grad(&[cfg.elem_len], sample_data(cfg.elem_len, 0.01), args.dtype);
    let b = tensor_no_grad(
        &[cfg.elem_len],
        sample_data(cfg.elem_len, -0.02),
        args.dtype,
    );
    let cpu = measure(args, || {
        let out = no_grad(|| &a * &b);
        black_box(out.shape_vec());
    });
    let cuda = if lumen::ops::cuda::is_available() {
        let a_cuda = a.to_cuda();
        let b_cuda = b.to_cuda();
        measure_cuda(args, || {
            let out = no_grad(|| &a_cuda * &b_cuda);
            black_box(out.shape_vec());
        })
    } else {
        None
    };
    BenchResult {
        name: "binary.same_shape.forward",
        cpu,
        cuda,
    }
}

fn bench_binary_row_broadcast_forward(args: &Args, cfg: ShapeConfig) -> BenchResult {
    let last_dim = cfg.softmax_last;
    let rows = (cfg.elem_len / last_dim).max(1);
    let matrix_len = rows * last_dim;
    let matrix = tensor_no_grad(&[rows, last_dim], sample_data(matrix_len, 0.01), args.dtype);
    let row = tensor_no_grad(&[last_dim], sample_data(last_dim, -0.02), args.dtype);
    let cpu = measure(args, || {
        let out = no_grad(|| &matrix * &row);
        black_box(out.shape_vec());
    });
    let cuda = if lumen::ops::cuda::is_available() {
        let matrix_cuda = matrix.to_cuda();
        let row_cuda = row.to_cuda();
        measure_cuda(args, || {
            let out = no_grad(|| &matrix_cuda * &row_cuda);
            black_box(out.shape_vec());
        })
    } else {
        None
    };
    BenchResult {
        name: "binary.row_broadcast.forward",
        cpu,
        cuda,
    }
}

fn bench_binary_row_scalar_forward(args: &Args, cfg: ShapeConfig) -> BenchResult {
    let last_dim = cfg.softmax_last;
    let rows = (cfg.elem_len / last_dim).max(1);
    let matrix_len = rows * last_dim;
    let matrix = tensor_no_grad(&[rows, last_dim], sample_data(matrix_len, 0.01), args.dtype);
    let scalar = tensor_no_grad(&[rows, 1], sample_data(rows, -0.02), args.dtype);
    let cpu = measure(args, || {
        let out = no_grad(|| &matrix * &scalar);
        black_box(out.shape_vec());
    });
    let cuda = if lumen::ops::cuda::is_available() {
        let matrix_cuda = matrix.to_cuda();
        let scalar_cuda = scalar.to_cuda();
        measure_cuda(args, || {
            let out = no_grad(|| &matrix_cuda * &scalar_cuda);
            black_box(out.shape_vec());
        })
    } else {
        None
    };
    BenchResult {
        name: "binary.row_scalar.forward",
        cpu,
        cuda,
    }
}

fn bench_binary_b1d_1h1_forward(args: &Args, cfg: ShapeConfig) -> BenchResult {
    let heads = cfg.attention_heads;
    let dim = cfg.softmax_last;
    let batch = (cfg.elem_len / (heads * dim)).max(1);
    let lhs_len = batch * dim;
    let rhs_len = heads;
    let lhs = tensor_no_grad(&[batch, 1, dim], sample_data(lhs_len, 0.01), args.dtype);
    let rhs = tensor_no_grad(&[1, heads, 1], sample_data(rhs_len, -0.02), args.dtype);
    let cpu = measure(args, || {
        let out = no_grad(|| &lhs * &rhs);
        black_box(out.shape_vec());
    });
    let cuda = if lumen::ops::cuda::is_available() {
        let lhs_cuda = lhs.to_cuda();
        let rhs_cuda = rhs.to_cuda();
        measure_cuda(args, || {
            let out = no_grad(|| &lhs_cuda * &rhs_cuda);
            black_box(out.shape_vec());
        })
    } else {
        None
    };
    BenchResult {
        name: "binary.b1d_1h1.forward",
        cpu,
        cuda,
    }
}

fn bench_binary_b1d_1hd_forward(args: &Args, cfg: ShapeConfig) -> BenchResult {
    let heads = cfg.attention_heads;
    let dim = cfg.softmax_last;
    let batch = (cfg.elem_len / (heads * dim)).max(1);
    let lhs_len = batch * dim;
    let rhs_len = heads * dim;
    let lhs = tensor_no_grad(&[batch, 1, dim], sample_data(lhs_len, 0.01), args.dtype);
    let rhs = tensor_no_grad(&[1, heads, dim], sample_data(rhs_len, -0.02), args.dtype);
    let cpu = measure(args, || {
        let out = no_grad(|| &lhs * &rhs);
        black_box(out.shape_vec());
    });
    let cuda = if lumen::ops::cuda::is_available() {
        let lhs_cuda = lhs.to_cuda();
        let rhs_cuda = rhs.to_cuda();
        measure_cuda(args, || {
            let out = no_grad(|| &lhs_cuda * &rhs_cuda);
            black_box(out.shape_vec());
        })
    } else {
        None
    };
    BenchResult {
        name: "binary.b1d_1hd.forward",
        cpu,
        cuda,
    }
}

fn bench_binary_general_broadcast_forward(args: &Args, cfg: ShapeConfig) -> BenchResult {
    let batch = cfg.attention_batch.max(1);
    let heads = cfg.attention_heads.max(1);
    let dim = cfg.softmax_last.max(1);
    let mid = (cfg.elem_len / (batch * heads * dim)).max(1);
    let lhs_len = batch * dim;
    let rhs_len = heads * mid * dim;
    let lhs = tensor_no_grad(&[batch, 1, 1, dim], sample_data(lhs_len, 0.01), args.dtype);
    let rhs = tensor_no_grad(
        &[1, heads, mid, dim],
        sample_data(rhs_len, -0.02),
        args.dtype,
    );
    let cpu = measure(args, || {
        let out = no_grad(|| &lhs * &rhs);
        black_box(out.shape_vec());
    });
    let cuda = if lumen::ops::cuda::is_available() {
        let lhs_cuda = lhs.to_cuda();
        let rhs_cuda = rhs.to_cuda();
        measure_cuda(args, || {
            let out = no_grad(|| &lhs_cuda * &rhs_cuda);
            black_box(out.shape_vec());
        })
    } else {
        None
    };
    BenchResult {
        name: "binary.general_broadcast.forward",
        cpu,
        cuda,
    }
}

fn bench_elementwise_backward(args: &Args, cfg: ShapeConfig) -> BenchResult {
    let a = tensor_grad(&[cfg.elem_len], sample_data(cfg.elem_len, 0.01), args.dtype);
    let b = tensor_grad(
        &[cfg.elem_len],
        sample_data(cfg.elem_len, -0.02),
        args.dtype,
    );
    let cpu = measure(args, || {
        zero_all(&[&a, &b]);
        let out = &(&a * &b) + &a;
        let loss = sum(&out);
        loss.backward();
        black_box(a.grad().is_some());
    });
    let cuda = if lumen::ops::cuda::is_available() {
        let a_cuda = a.to_cuda();
        let b_cuda = b.to_cuda();
        measure_cuda(args, || {
            zero_all(&[&a_cuda, &b_cuda]);
            let out = &(&a_cuda * &b_cuda) + &a_cuda;
            let loss = sum(&out);
            loss.backward();
            black_box(a_cuda.requires_grad());
        })
    } else {
        None
    };
    BenchResult {
        name: "elementwise.mul_add.backward",
        cpu,
        cuda,
    }
}

fn bench_elementwise_mixed_mul_forward(args: &Args, cfg: ShapeConfig) -> BenchResult {
    let a = tensor_no_grad(&[cfg.elem_len], sample_data(cfg.elem_len, 0.01), args.dtype);
    let mask = tensor_no_grad(
        &[cfg.elem_len],
        sample_data(cfg.elem_len, -0.02),
        DType::F32,
    );
    let cpu = measure(args, || {
        let out = no_grad(|| &a * &mask);
        black_box(out.shape_vec());
    });
    let cuda = if lumen::ops::cuda::is_available() {
        let a_cuda = a.to_cuda();
        let mask_cuda = mask.to_cuda();
        measure_cuda(args, || {
            let out = no_grad(|| &a_cuda * &mask_cuda);
            black_box(out.shape_vec());
        })
    } else {
        None
    };
    BenchResult {
        name: "elementwise.mixed_mul.forward",
        cpu,
        cuda,
    }
}

fn bench_elementwise_mixed_broadcast_1hd_mul_forward(args: &Args, cfg: ShapeConfig) -> BenchResult {
    let lhs_shape = [cfg.attention_batch, 1, cfg.softmax_last];
    let rhs_shape = [1, cfg.attention_heads, cfg.softmax_last];
    let lhs_len = lhs_shape.iter().product::<usize>();
    let rhs_len = rhs_shape.iter().product::<usize>();
    let lhs = tensor_no_grad(&lhs_shape, sample_data(lhs_len, 0.01), args.dtype);
    let rhs = tensor_no_grad(&rhs_shape, sample_data(rhs_len, -0.02), DType::F32);
    let cpu = measure(args, || {
        let out = no_grad(|| &lhs * &rhs);
        black_box(out.shape_vec());
    });
    let cuda = if lumen::ops::cuda::is_available() {
        let lhs_cuda = lhs.to_cuda();
        let rhs_cuda = rhs.to_cuda();
        measure_cuda(args, || {
            let out = no_grad(|| &lhs_cuda * &rhs_cuda);
            black_box(out.shape_vec());
        })
    } else {
        None
    };
    BenchResult {
        name: "elementwise.mixed_broadcast_1hd_mul.forward",
        cpu,
        cuda,
    }
}

fn bench_elementwise_mixed_row_scalar_mul_forward(args: &Args, cfg: ShapeConfig) -> BenchResult {
    let lhs_shape = [cfg.attention_batch, cfg.attention_heads, cfg.softmax_last];
    let rhs_shape = [cfg.attention_batch, cfg.attention_heads, 1];
    let lhs_len = lhs_shape.iter().product::<usize>();
    let rhs_len = rhs_shape.iter().product::<usize>();
    let lhs = tensor_no_grad(&lhs_shape, sample_data(lhs_len, 0.01), args.dtype);
    let rhs = tensor_no_grad(&rhs_shape, sample_data(rhs_len, -0.02), DType::F32);
    let cpu = measure(args, || {
        let out = no_grad(|| &lhs * &rhs);
        black_box(out.shape_vec());
    });
    let cuda = if lumen::ops::cuda::is_available() {
        let lhs_cuda = lhs.to_cuda();
        let rhs_cuda = rhs.to_cuda();
        measure_cuda(args, || {
            let out = no_grad(|| &lhs_cuda * &rhs_cuda);
            black_box(out.shape_vec());
        })
    } else {
        None
    };
    BenchResult {
        name: "elementwise.mixed_row_scalar_mul.forward",
        cpu,
        cuda,
    }
}

fn bench_elementwise_mixed_mul_backward(args: &Args, cfg: ShapeConfig) -> BenchResult {
    let a = tensor_grad(&[cfg.elem_len], sample_data(cfg.elem_len, 0.01), args.dtype);
    let mask = tensor_const(
        &[cfg.elem_len],
        sample_data(cfg.elem_len, -0.02),
        DType::F32,
    );
    let cpu = measure(args, || {
        a.zero_grad();
        let out = &a * &mask;
        let loss = sum(&out);
        loss.backward();
        black_box(a.grad().is_some());
    });
    let cuda = if lumen::ops::cuda::is_available() {
        let a_cuda = a.to_cuda();
        let mask_cuda = mask.to_cuda();
        measure_cuda(args, || {
            a_cuda.zero_grad();
            let out = &a_cuda * &mask_cuda;
            let loss = sum(&out);
            loss.backward();
            black_box(a_cuda.requires_grad());
        })
    } else {
        None
    };
    BenchResult {
        name: "elementwise.mixed_mul.backward",
        cpu,
        cuda,
    }
}

fn bench_elementwise_mixed_row_scalar_mul_backward(args: &Args, cfg: ShapeConfig) -> BenchResult {
    let lhs_shape = [cfg.attention_batch, cfg.attention_heads, cfg.softmax_last];
    let rhs_shape = [cfg.attention_batch, cfg.attention_heads, 1];
    let lhs_len = lhs_shape.iter().product::<usize>();
    let rhs_len = rhs_shape.iter().product::<usize>();
    let lhs = tensor_grad(&lhs_shape, sample_data(lhs_len, 0.01), args.dtype);
    let rhs = tensor_grad(&rhs_shape, sample_data(rhs_len, -0.02), DType::F32);
    let cpu = measure(args, || {
        zero_all(&[&lhs, &rhs]);
        let out = &lhs * &rhs;
        let loss = sum(&out);
        loss.backward();
        black_box(lhs.grad().is_some());
    });
    let cuda = if lumen::ops::cuda::is_available() {
        let lhs_cuda = lhs.to_cuda();
        let rhs_cuda = rhs.to_cuda();
        measure_cuda(args, || {
            zero_all(&[&lhs_cuda, &rhs_cuda]);
            let out = &lhs_cuda * &rhs_cuda;
            let loss = sum(&out);
            loss.backward();
            black_box(lhs_cuda.requires_grad());
        })
    } else {
        None
    };
    BenchResult {
        name: "elementwise.mixed_row_scalar_mul.backward",
        cpu,
        cuda,
    }
}

fn bench_elementwise_mixed_row_mul_forward(args: &Args, cfg: ShapeConfig) -> BenchResult {
    let shape = [cfg.softmax_outer, cfg.softmax_last];
    let matrix = tensor_no_grad(
        &shape,
        sample_data(cfg.softmax_outer * cfg.softmax_last, 0.01),
        args.dtype,
    );
    let row = tensor_no_grad(
        &[cfg.softmax_last],
        sample_data(cfg.softmax_last, -0.02),
        DType::F32,
    );
    let cpu = measure(args, || {
        let out = no_grad(|| &matrix * &row);
        black_box(out.shape_vec());
    });
    let cuda = if lumen::ops::cuda::is_available() {
        let matrix_cuda = matrix.to_cuda();
        let row_cuda = row.to_cuda();
        measure_cuda(args, || {
            let out = no_grad(|| &matrix_cuda * &row_cuda);
            black_box(out.shape_vec());
        })
    } else {
        None
    };
    BenchResult {
        name: "elementwise.mixed_row_mul.forward",
        cpu,
        cuda,
    }
}

fn bench_elementwise_mixed_row_mul_backward(args: &Args, cfg: ShapeConfig) -> BenchResult {
    let shape = [cfg.softmax_outer, cfg.softmax_last];
    let matrix = tensor_grad(
        &shape,
        sample_data(cfg.softmax_outer * cfg.softmax_last, 0.01),
        args.dtype,
    );
    let row = tensor_grad(
        &[cfg.softmax_last],
        sample_data(cfg.softmax_last, -0.02),
        DType::F32,
    );
    let cpu = measure(args, || {
        zero_all(&[&matrix, &row]);
        let out = &matrix * &row;
        let loss = sum(&out);
        loss.backward();
        black_box(matrix.grad().is_some());
    });
    let cuda = if lumen::ops::cuda::is_available() {
        let matrix_cuda = matrix.to_cuda();
        let row_cuda = row.to_cuda();
        measure_cuda(args, || {
            zero_all(&[&matrix_cuda, &row_cuda]);
            let out = &matrix_cuda * &row_cuda;
            let loss = sum(&out);
            loss.backward();
            black_box(matrix_cuda.requires_grad());
        })
    } else {
        None
    };
    BenchResult {
        name: "elementwise.mixed_row_mul.backward",
        cpu,
        cuda,
    }
}

fn bench_elementwise_mixed_row_sub_backward(args: &Args, cfg: ShapeConfig) -> BenchResult {
    let shape = [cfg.softmax_outer, cfg.softmax_last];
    let matrix = tensor_grad(
        &shape,
        sample_data(cfg.softmax_outer * cfg.softmax_last, 0.01),
        args.dtype,
    );
    let row = tensor_grad(
        &[cfg.softmax_last],
        sample_data(cfg.softmax_last, -0.02),
        DType::F32,
    );
    let cpu = measure(args, || {
        zero_all(&[&matrix, &row]);
        let out = &matrix - &row;
        let loss = sum(&out);
        loss.backward();
        black_box(matrix.grad().is_some());
    });
    let cuda = if lumen::ops::cuda::is_available() {
        let matrix_cuda = matrix.to_cuda();
        let row_cuda = row.to_cuda();
        measure_cuda(args, || {
            zero_all(&[&matrix_cuda, &row_cuda]);
            let out = &matrix_cuda - &row_cuda;
            let loss = sum(&out);
            loss.backward();
            black_box(matrix_cuda.requires_grad());
        })
    } else {
        None
    };
    BenchResult {
        name: "elementwise.mixed_row_sub.backward",
        cpu,
        cuda,
    }
}

fn bench_elementwise_mixed_broadcast_sub_backward(args: &Args, cfg: ShapeConfig) -> BenchResult {
    let lhs_shape = [cfg.attention_batch, 1, cfg.softmax_last];
    let rhs_shape = [1, cfg.attention_heads, 1];
    let lhs_len = lhs_shape.iter().product::<usize>();
    let rhs_len = rhs_shape.iter().product::<usize>();
    let lhs = tensor_grad(&lhs_shape, sample_data(lhs_len, 0.01), args.dtype);
    let rhs = tensor_grad(&rhs_shape, sample_data(rhs_len, -0.02), DType::F32);
    let cpu = measure(args, || {
        zero_all(&[&lhs, &rhs]);
        let out = &lhs - &rhs;
        let loss = sum(&out);
        loss.backward();
        black_box(lhs.grad().is_some());
    });
    let cuda = if lumen::ops::cuda::is_available() {
        let lhs_cuda = lhs.to_cuda();
        let rhs_cuda = rhs.to_cuda();
        measure_cuda(args, || {
            zero_all(&[&lhs_cuda, &rhs_cuda]);
            let out = &lhs_cuda - &rhs_cuda;
            let loss = sum(&out);
            loss.backward();
            black_box(lhs_cuda.requires_grad());
        })
    } else {
        None
    };
    BenchResult {
        name: "elementwise.mixed_broadcast_sub.backward",
        cpu,
        cuda,
    }
}

fn bench_elementwise_mixed_broadcast_1hd_sub_backward(
    args: &Args,
    cfg: ShapeConfig,
) -> BenchResult {
    let lhs_shape = [cfg.attention_batch, 1, cfg.softmax_last];
    let rhs_shape = [1, cfg.attention_heads, cfg.softmax_last];
    let lhs_len = lhs_shape.iter().product::<usize>();
    let rhs_len = rhs_shape.iter().product::<usize>();
    let lhs = tensor_grad(&lhs_shape, sample_data(lhs_len, 0.01), args.dtype);
    let rhs = tensor_grad(&rhs_shape, sample_data(rhs_len, -0.02), DType::F32);
    let cpu = measure(args, || {
        zero_all(&[&lhs, &rhs]);
        let out = &lhs - &rhs;
        let loss = sum(&out);
        loss.backward();
        black_box(lhs.grad().is_some());
    });
    let cuda = if lumen::ops::cuda::is_available() {
        let lhs_cuda = lhs.to_cuda();
        let rhs_cuda = rhs.to_cuda();
        measure_cuda(args, || {
            zero_all(&[&lhs_cuda, &rhs_cuda]);
            let out = &lhs_cuda - &rhs_cuda;
            let loss = sum(&out);
            loss.backward();
            black_box(lhs_cuda.requires_grad());
        })
    } else {
        None
    };
    BenchResult {
        name: "elementwise.mixed_broadcast_1hd_sub.backward",
        cpu,
        cuda,
    }
}

fn bench_elementwise_mixed_broadcast_mul_backward(args: &Args, cfg: ShapeConfig) -> BenchResult {
    let lhs_shape = [cfg.attention_batch, 1, cfg.softmax_last];
    let rhs_shape = [1, cfg.attention_heads, 1];
    let lhs_len = lhs_shape.iter().product::<usize>();
    let rhs_len = rhs_shape.iter().product::<usize>();
    let lhs = tensor_grad(&lhs_shape, sample_data(lhs_len, 0.01), args.dtype);
    let rhs = tensor_grad(&rhs_shape, sample_data(rhs_len, -0.02), DType::F32);
    let cpu = measure(args, || {
        zero_all(&[&lhs, &rhs]);
        let out = &lhs * &rhs;
        let loss = sum(&out);
        loss.backward();
        black_box(lhs.grad().is_some());
    });
    let cuda = if lumen::ops::cuda::is_available() {
        let lhs_cuda = lhs.to_cuda();
        let rhs_cuda = rhs.to_cuda();
        measure_cuda(args, || {
            zero_all(&[&lhs_cuda, &rhs_cuda]);
            let out = &lhs_cuda * &rhs_cuda;
            let loss = sum(&out);
            loss.backward();
            black_box(lhs_cuda.requires_grad());
        })
    } else {
        None
    };
    BenchResult {
        name: "elementwise.mixed_broadcast_mul.backward",
        cpu,
        cuda,
    }
}

fn bench_elementwise_mixed_broadcast_1hd_mul_backward(
    args: &Args,
    cfg: ShapeConfig,
) -> BenchResult {
    let lhs_shape = [cfg.attention_batch, 1, cfg.softmax_last];
    let rhs_shape = [1, cfg.attention_heads, cfg.softmax_last];
    let lhs_len = lhs_shape.iter().product::<usize>();
    let rhs_len = rhs_shape.iter().product::<usize>();
    let lhs = tensor_grad(&lhs_shape, sample_data(lhs_len, 0.01), args.dtype);
    let rhs = tensor_grad(&rhs_shape, sample_data(rhs_len, -0.02), DType::F32);
    let cpu = measure(args, || {
        zero_all(&[&lhs, &rhs]);
        let out = &lhs * &rhs;
        let loss = sum(&out);
        loss.backward();
        black_box(lhs.grad().is_some());
    });
    let cuda = if lumen::ops::cuda::is_available() {
        let lhs_cuda = lhs.to_cuda();
        let rhs_cuda = rhs.to_cuda();
        measure_cuda(args, || {
            zero_all(&[&lhs_cuda, &rhs_cuda]);
            let out = &lhs_cuda * &rhs_cuda;
            let loss = sum(&out);
            loss.backward();
            black_box(lhs_cuda.requires_grad());
        })
    } else {
        None
    };
    BenchResult {
        name: "elementwise.mixed_broadcast_1hd_mul.backward",
        cpu,
        cuda,
    }
}

fn bench_elementwise_mixed_scalar_sub_backward(args: &Args, cfg: ShapeConfig) -> BenchResult {
    let shape = [cfg.softmax_outer, cfg.softmax_last];
    let lhs = tensor_grad(
        &shape,
        sample_data(cfg.softmax_outer * cfg.softmax_last, 0.01),
        args.dtype,
    );
    let rhs = tensor_grad(&[1], vec![0.5], DType::F32);
    let cpu = measure(args, || {
        zero_all(&[&lhs, &rhs]);
        let out = &lhs - &rhs;
        let loss = sum(&out);
        loss.backward();
        black_box(lhs.grad().is_some());
    });
    let cuda = if lumen::ops::cuda::is_available() {
        let lhs_cuda = lhs.to_cuda();
        let rhs_cuda = rhs.to_cuda();
        measure_cuda(args, || {
            zero_all(&[&lhs_cuda, &rhs_cuda]);
            let out = &lhs_cuda - &rhs_cuda;
            let loss = sum(&out);
            loss.backward();
            black_box(lhs_cuda.requires_grad());
        })
    } else {
        None
    };
    BenchResult {
        name: "elementwise.mixed_scalar_sub.backward",
        cpu,
        cuda,
    }
}

fn bench_elementwise_mixed_scalar_mul_backward(args: &Args, cfg: ShapeConfig) -> BenchResult {
    let shape = [cfg.softmax_outer, cfg.softmax_last];
    let lhs = tensor_grad(
        &shape,
        sample_data(cfg.softmax_outer * cfg.softmax_last, 0.01),
        args.dtype,
    );
    let rhs = tensor_grad(&[1], vec![0.5], DType::F32);
    let cpu = measure(args, || {
        zero_all(&[&lhs, &rhs]);
        let out = &lhs * &rhs;
        let loss = sum(&out);
        loss.backward();
        black_box(lhs.grad().is_some());
    });
    let cuda = if lumen::ops::cuda::is_available() {
        let lhs_cuda = lhs.to_cuda();
        let rhs_cuda = rhs.to_cuda();
        measure_cuda(args, || {
            zero_all(&[&lhs_cuda, &rhs_cuda]);
            let out = &lhs_cuda * &rhs_cuda;
            let loss = sum(&out);
            loss.backward();
            black_box(lhs_cuda.requires_grad());
        })
    } else {
        None
    };
    BenchResult {
        name: "elementwise.mixed_scalar_mul.backward",
        cpu,
        cuda,
    }
}

fn bench_unary_silu_forward(args: &Args, cfg: ShapeConfig) -> BenchResult {
    let input = tensor_no_grad(
        &[cfg.elem_len],
        sample_data(cfg.elem_len, 0.012),
        args.dtype,
    );
    let silu = SiLU::new();
    let cpu = measure(args, || {
        let out = no_grad(|| silu.forward(input.clone()));
        black_box(out.shape_vec());
    });
    let cuda = if lumen::ops::cuda::is_available() {
        let input_cuda = input.to_cuda();
        measure_cuda(args, || {
            let out = no_grad(|| silu.forward(input_cuda.clone()));
            black_box(out.shape_vec());
        })
    } else {
        None
    };
    BenchResult {
        name: "unary.silu.forward",
        cpu,
        cuda,
    }
}

fn bench_unary_relu_forward(args: &Args, cfg: ShapeConfig) -> BenchResult {
    let input = tensor_no_grad(
        &[cfg.elem_len],
        sample_data(cfg.elem_len, 0.012),
        args.dtype,
    );
    let relu = ReLU::new();
    let cpu = measure(args, || {
        let out = no_grad(|| relu.forward(input.clone()));
        black_box(out.shape_vec());
    });
    let cuda = if lumen::ops::cuda::is_available() {
        let input_cuda = input.to_cuda();
        measure_cuda(args, || {
            let out = no_grad(|| relu.forward(input_cuda.clone()));
            black_box(out.shape_vec());
        })
    } else {
        None
    };
    BenchResult {
        name: "unary.relu.forward",
        cpu,
        cuda,
    }
}

fn bench_unary_silu_backward(args: &Args, cfg: ShapeConfig) -> BenchResult {
    let input = tensor_grad(
        &[cfg.elem_len],
        sample_data(cfg.elem_len, 0.012),
        args.dtype,
    );
    let coeff = tensor_no_grad(
        &[cfg.elem_len],
        sample_data(cfg.elem_len, -0.007),
        DType::F32,
    );
    let silu = SiLU::new();
    let cpu = measure(args, || {
        input.zero_grad();
        let out = silu.forward(input.clone());
        let loss = sum(&(&out * &coeff));
        loss.backward();
        black_box(input.grad().is_some());
    });
    let cuda = if lumen::ops::cuda::is_available() {
        let input_cuda = input.to_cuda();
        let coeff_cuda = coeff.to_cuda();
        measure_cuda(args, || {
            input_cuda.zero_grad();
            let out = silu.forward(input_cuda.clone());
            let loss = sum(&(&out * &coeff_cuda));
            loss.backward();
            black_box(input_cuda.requires_grad());
        })
    } else {
        None
    };
    BenchResult {
        name: "unary.silu.backward",
        cpu,
        cuda,
    }
}

fn bench_fused_gateup_forward(args: &Args, cfg: ShapeConfig) -> BenchResult {
    let hidden = cfg.attention_hidden;
    let inter = hidden * 4;
    let rows = cfg.attention_batch * cfg.attention_seq;
    let input = tensor_no_grad(
        &[rows, 1, hidden],
        sample_data(rows * hidden, 0.01),
        args.dtype,
    );
    let gate = tensor_no_grad(
        &[inter, hidden],
        sample_data(inter * hidden, 0.007),
        args.dtype,
    );
    let up = tensor_no_grad(
        &[inter, hidden],
        sample_data(inter * hidden, -0.005),
        args.dtype,
    );

    let cpu = measure(args, || {
        let out = no_grad(|| fused_gate_up_silu_infer(&input, &gate, &up));
        black_box(out.shape_vec());
    });
    let cuda = if lumen::ops::cuda::is_available() {
        let input_cuda = input.to_cuda();
        let gate_cuda = gate.to_cuda();
        let up_cuda = up.to_cuda();
        measure_cuda(args, || {
            let out = no_grad(|| fused_gate_up_silu_infer(&input_cuda, &gate_cuda, &up_cuda));
            black_box(out.shape_vec());
        })
    } else {
        None
    };
    BenchResult {
        name: "fused_gateup.forward",
        cpu,
        cuda,
    }
}

fn bench_fused_qkv_decode(args: &Args, cfg: ShapeConfig) -> BenchResult {
    let hidden = cfg.attention_hidden;
    let heads = cfg.attention_heads;
    let kv_heads = cfg.attention_kv_heads;
    let head_dim = hidden / heads;
    let kv_hidden = kv_heads * head_dim;
    let input = tensor_no_grad(
        &[cfg.attention_batch, 1, hidden],
        sample_data(cfg.attention_batch * hidden, 0.01),
        args.dtype,
    );
    let q = tensor_no_grad(
        &[hidden, hidden],
        sample_data(hidden * hidden, 0.007),
        args.dtype,
    );
    let k = tensor_no_grad(
        &[kv_hidden, hidden],
        sample_data(kv_hidden * hidden, -0.005),
        args.dtype,
    );
    let v = tensor_no_grad(
        &[kv_hidden, hidden],
        sample_data(kv_hidden * hidden, 0.003),
        args.dtype,
    );

    let cpu = measure(args, || {
        let (q_out, k_out, v_out) =
            no_grad(|| fused_qkv_decode_infer_tensors(&input, &q, &k, &v, heads, kv_heads));
        black_box((q_out.shape_vec(), k_out.shape_vec(), v_out.shape_vec()));
    });
    let cuda = if lumen::ops::cuda::is_available() {
        let input_cuda = input.to_cuda();
        let q_cuda = q.to_cuda();
        let k_cuda = k.to_cuda();
        let v_cuda = v.to_cuda();
        measure_cuda(args, || {
            let (q_out, k_out, v_out) = no_grad(|| {
                fused_qkv_decode_infer_tensors(
                    &input_cuda,
                    &q_cuda,
                    &k_cuda,
                    &v_cuda,
                    heads,
                    kv_heads,
                )
            });
            black_box((q_out.shape_vec(), k_out.shape_vec(), v_out.shape_vec()));
        })
    } else {
        None
    };
    BenchResult {
        name: "fused_qkv.decode",
        cpu,
        cuda,
    }
}

fn bench_fused_qkv_prefill(args: &Args, cfg: ShapeConfig) -> BenchResult {
    let batch = cfg.attention_batch;
    let seq = cfg.attention_seq;
    let hidden = cfg.attention_hidden;
    let heads = cfg.attention_heads;
    let kv_heads = cfg.attention_kv_heads;
    let head_dim = hidden / heads;
    let kv_hidden = kv_heads * head_dim;
    let input = tensor_no_grad(
        &[batch, seq, hidden],
        sample_data(batch * seq * hidden, 0.01),
        args.dtype,
    );
    let q = tensor_no_grad(
        &[hidden, hidden],
        sample_data(hidden * hidden, 0.007),
        args.dtype,
    );
    let k = tensor_no_grad(
        &[kv_hidden, hidden],
        sample_data(kv_hidden * hidden, -0.005),
        args.dtype,
    );
    let v = tensor_no_grad(
        &[kv_hidden, hidden],
        sample_data(kv_hidden * hidden, 0.003),
        args.dtype,
    );

    let cpu = measure(args, || {
        let out = fused_qkv_prefill_cpu_reference(
            &input, &q, &k, &v, batch, seq, hidden, heads, kv_heads,
        );
        black_box((out.0.len(), out.1.len(), out.2.len()));
    });
    let cuda = if lumen::ops::cuda::is_available() {
        let input_cuda = input.to_cuda();
        let q_cuda = q.to_cuda();
        let k_cuda = k.to_cuda();
        let v_cuda = v.to_cuda();
        measure_cuda(args, || {
            let (q_out, k_out, v_out) = no_grad(|| {
                fused_qkv_prefill_infer_tensors(
                    &input_cuda,
                    &q_cuda,
                    &k_cuda,
                    &v_cuda,
                    heads,
                    kv_heads,
                )
                .expect("CUDA fused QKV prefill should run")
            });
            black_box((q_out.shape_vec(), k_out.shape_vec(), v_out.shape_vec()));
        })
    } else {
        None
    };
    BenchResult {
        name: "fused_qkv.prefill",
        cpu,
        cuda,
    }
}

fn bench_softmax_forward(args: &Args, cfg: ShapeConfig) -> BenchResult {
    let shape = [cfg.softmax_outer, cfg.softmax_last];
    let input = tensor_no_grad(
        &shape,
        sample_data(cfg.softmax_outer * cfg.softmax_last, 0.01),
        args.dtype,
    );
    let softmax = Softmax::new(1);

    let cpu = measure(args, || {
        let out = no_grad(|| softmax.forward(input.clone()));
        black_box(out.shape_vec());
    });
    let cuda = if lumen::ops::cuda::is_available() {
        let input_cuda = input.to_cuda();
        measure_cuda(args, || {
            let out = no_grad(|| softmax.forward(input_cuda.clone()));
            black_box(out.shape_vec());
        })
    } else {
        None
    };
    BenchResult {
        name: "softmax.forward",
        cpu,
        cuda,
    }
}

fn bench_softmax_backward(args: &Args, cfg: ShapeConfig) -> BenchResult {
    let shape = [cfg.softmax_outer, cfg.softmax_last];
    let input = tensor_grad(
        &shape,
        sample_data(cfg.softmax_outer * cfg.softmax_last, 0.01),
        args.dtype,
    );
    let coeff = tensor_const(
        &shape,
        sample_data(cfg.softmax_outer * cfg.softmax_last, -0.004),
        args.dtype,
    );
    let softmax = Softmax::new(1);

    let cpu = measure(args, || {
        input.zero_grad();
        let out = softmax.forward(input.clone());
        let loss = sum(&(&out * &coeff));
        loss.backward();
        black_box(input.grad().is_some());
    });
    let cuda = if lumen::ops::cuda::is_available() {
        let input_cuda = input.to_cuda();
        let coeff_cuda = coeff.to_cuda();
        measure_cuda(args, || {
            input_cuda.zero_grad();
            let out = softmax.forward(input_cuda.clone());
            let loss = sum(&(&out * &coeff_cuda));
            loss.backward();
            black_box(input_cuda.requires_grad());
        })
    } else {
        None
    };
    BenchResult {
        name: "softmax.backward",
        cpu,
        cuda,
    }
}

fn bench_fused_softmax_forward(args: &Args, cfg: ShapeConfig) -> BenchResult {
    let (batch, heads, q_len, k_len) = fused_softmax_shape(cfg);
    let shape = [batch, heads, q_len, k_len];
    let len = batch * heads * q_len * k_len;
    let input = tensor_no_grad(&shape, sample_data(len, 0.01), args.dtype);
    let scale = 0.75f32;

    let cpu = measure(args, || {
        let out = no_grad(|| fused_softmax(&input, scale, true));
        black_box(out.shape_vec());
    });
    let cuda = if lumen::ops::cuda::is_available() {
        let input_cuda = input.to_cuda();
        measure_cuda(args, || {
            let out = no_grad(|| fused_softmax(&input_cuda, scale, true));
            black_box(out.shape_vec());
        })
    } else {
        None
    };
    BenchResult {
        name: "fused_softmax.forward",
        cpu,
        cuda,
    }
}

fn bench_fused_softmax_backward(args: &Args, cfg: ShapeConfig) -> BenchResult {
    let (batch, heads, q_len, k_len) = fused_softmax_shape(cfg);
    let shape = [batch, heads, q_len, k_len];
    let len = batch * heads * q_len * k_len;
    let input = tensor_grad(&shape, sample_data(len, 0.01), args.dtype);
    let coeff = tensor_const(&shape, sample_data(len, -0.004), args.dtype);
    let scale = 0.75f32;

    let cpu = measure(args, || {
        input.zero_grad();
        let out = fused_softmax(&input, scale, true);
        let loss = sum(&(&out * &coeff));
        loss.backward();
        black_box(input.grad().is_some());
    });
    let cuda = if lumen::ops::cuda::is_available() {
        let input_cuda = input.to_cuda();
        let coeff_cuda = coeff.to_cuda();
        measure_cuda(args, || {
            input_cuda.zero_grad();
            let out = fused_softmax(&input_cuda, scale, true);
            let loss = sum(&(&out * &coeff_cuda));
            loss.backward();
            black_box(input_cuda.requires_grad());
        })
    } else {
        None
    };
    BenchResult {
        name: "fused_softmax.backward",
        cpu,
        cuda,
    }
}

fn bench_embedding_forward(args: &Args, cfg: ShapeConfig) -> BenchResult {
    let vocab_size = cfg.softmax_last.max(64);
    let embed_dim = cfg.attention_hidden;
    let token_count = cfg.softmax_outer;
    let weight_data = sample_data(vocab_size * embed_dim, 0.01);
    let index_data = token_id_data(1, token_count, vocab_size);

    let emb = Embedding::new_with_dtype(vocab_size, embed_dim, args.dtype);
    emb.weight.set_array_f32_with_dtype(
        array_from_vec(&[vocab_size, embed_dim], weight_data.clone()),
        args.dtype,
    );
    let indices = token_tensor(&[token_count], index_data.clone());

    let cpu = measure(args, || {
        let out = no_grad(|| emb.forward(&indices));
        black_box(out.shape_vec());
    });
    let cuda = if lumen::ops::cuda::is_available() {
        let emb_cuda = Embedding::new_with_dtype(vocab_size, embed_dim, args.dtype);
        emb_cuda.weight.set_array_f32_with_dtype(
            array_from_vec(&[vocab_size, embed_dim], weight_data),
            args.dtype,
        );
        emb_cuda.weight.to_cuda_inplace();
        let indices_cuda = token_tensor(&[token_count], index_data).to_cuda();
        measure_cuda(args, || {
            let out = no_grad(|| emb_cuda.forward(&indices_cuda));
            black_box(out.shape_vec());
        })
    } else {
        None
    };
    BenchResult {
        name: "embedding.forward",
        cpu,
        cuda,
    }
}

fn bench_rms_norm_forward(args: &Args, cfg: ShapeConfig) -> BenchResult {
    let rows = cfg.softmax_outer;
    let dim = cfg.attention_hidden;
    let input_data = sample_data(rows * dim, 0.01);
    let weight_data = sample_data(dim, 0.003)
        .into_iter()
        .map(|value| value + 1.0)
        .collect::<Vec<_>>();

    let norm = RMSNorm::new_with_dtype(dim, 1e-5, args.dtype);
    norm.weight
        .set_array_f32_with_dtype(array_from_vec(&[dim], weight_data.clone()), args.dtype);
    let input = tensor_no_grad(&[rows, dim], input_data.clone(), args.dtype);

    let cpu = measure(args, || {
        let out = no_grad(|| norm.forward(input.clone()));
        black_box(out.shape_vec());
    });
    let cuda = if lumen::ops::cuda::is_available() {
        let norm_cuda = RMSNorm::new_with_dtype(dim, 1e-5, args.dtype);
        norm_cuda
            .weight
            .set_array_f32_with_dtype(array_from_vec(&[dim], weight_data), args.dtype);
        norm_cuda.weight.to_cuda_inplace();
        let input_cuda = tensor_no_grad(&[rows, dim], input_data, args.dtype).to_cuda();
        measure_cuda(args, || {
            let out = no_grad(|| norm_cuda.forward(input_cuda.clone()));
            black_box(out.shape_vec());
        })
    } else {
        None
    };
    BenchResult {
        name: "rms_norm.forward",
        cpu,
        cuda,
    }
}

fn bench_rope_forward(args: &Args, cfg: ShapeConfig) -> BenchResult {
    let batch = cfg.attention_batch;
    let heads = cfg.attention_heads;
    let seq = cfg.attention_seq;
    let dim = cfg.attention_hidden / cfg.attention_heads;
    let shape = [batch, heads, seq, dim];
    let input_data = sample_data(batch * heads * seq * dim, 0.01);
    let offset = 1usize;
    let max_seq = seq + offset + 1;

    let rope = RotaryEmbedding::new_with_dtype(dim, max_seq, 10000.0, args.dtype);
    let input = tensor_no_grad(&shape, input_data.clone(), args.dtype);

    let cpu = measure(args, || {
        let out = no_grad(|| rope.forward(&input, offset));
        black_box(out.shape_vec());
    });
    let cuda = if lumen::ops::cuda::is_available() {
        let rope_cuda = RotaryEmbedding::new_with_dtype(dim, max_seq, 10000.0, args.dtype);
        rope_cuda.to_cuda();
        let input_cuda = tensor_no_grad(&shape, input_data, args.dtype).to_cuda();
        measure_cuda(args, || {
            let out = no_grad(|| rope_cuda.forward(&input_cuda, offset));
            black_box(out.shape_vec());
        })
    } else {
        None
    };
    BenchResult {
        name: "rope.forward",
        cpu,
        cuda,
    }
}

fn bench_cross_entropy_forward(args: &Args, cfg: ShapeConfig) -> BenchResult {
    let shape = [cfg.softmax_outer, cfg.softmax_last];
    let logits = tensor_no_grad(
        &shape,
        sample_data(cfg.softmax_outer * cfg.softmax_last, 0.01),
        args.dtype,
    );
    let targets = tensor_const(
        &shape,
        one_hot_data(cfg.softmax_outer, cfg.softmax_last),
        args.dtype,
    );

    let cpu = measure(args, || {
        let loss = no_grad(|| CrossEntropyLoss::apply(&logits, &targets));
        black_box(scalar_value(&loss));
    });
    let cuda = if lumen::ops::cuda::is_available() {
        let logits_cuda = logits.to_cuda();
        let targets_cuda = targets.to_cuda();
        measure_cuda(args, || {
            let loss = no_grad(|| CrossEntropyLoss::apply(&logits_cuda, &targets_cuda));
            black_box(loss.shape_vec());
        })
    } else {
        None
    };
    BenchResult {
        name: "cross_entropy.forward",
        cpu,
        cuda,
    }
}

fn bench_cross_entropy_backward(args: &Args, cfg: ShapeConfig) -> BenchResult {
    let shape = [cfg.softmax_outer, cfg.softmax_last];
    let logits = tensor_grad(
        &shape,
        sample_data(cfg.softmax_outer * cfg.softmax_last, 0.01),
        args.dtype,
    );
    let targets = tensor_const(
        &shape,
        one_hot_data(cfg.softmax_outer, cfg.softmax_last),
        args.dtype,
    );

    let cpu = measure(args, || {
        logits.zero_grad();
        let loss = CrossEntropyLoss::apply(&logits, &targets);
        loss.backward();
        black_box(logits.grad().is_some());
    });
    let cuda = if lumen::ops::cuda::is_available() {
        let logits_cuda = logits.to_cuda();
        let targets_cuda = targets.to_cuda();
        measure_cuda(args, || {
            logits_cuda.zero_grad();
            let loss = CrossEntropyLoss::apply(&logits_cuda, &targets_cuda);
            loss.backward();
            black_box(logits_cuda.requires_grad());
        })
    } else {
        None
    };
    BenchResult {
        name: "cross_entropy.backward",
        cpu,
        cuda,
    }
}

fn bench_mse_loss_forward(args: &Args, cfg: ShapeConfig) -> BenchResult {
    let shape = [cfg.elem_len];
    let output = tensor_no_grad(&shape, sample_data(cfg.elem_len, 0.01), args.dtype);
    let target = tensor_const(&shape, sample_data(cfg.elem_len, -0.007), args.dtype);

    let cpu = measure(args, || {
        let loss = no_grad(|| MSELoss::apply(&output, &target));
        black_box(scalar_value(&loss));
    });
    let cuda = if lumen::ops::cuda::is_available() {
        let output_cuda = output.to_cuda();
        let target_cuda = target.to_cuda();
        measure_cuda(args, || {
            let loss = no_grad(|| MSELoss::apply(&output_cuda, &target_cuda));
            black_box(loss.shape_vec());
        })
    } else {
        None
    };
    BenchResult {
        name: "mse_loss.forward",
        cpu,
        cuda,
    }
}

fn bench_mse_loss_backward(args: &Args, cfg: ShapeConfig) -> BenchResult {
    let shape = [cfg.elem_len];
    let output = tensor_grad(&shape, sample_data(cfg.elem_len, 0.01), args.dtype);
    let target = tensor_const(&shape, sample_data(cfg.elem_len, -0.007), args.dtype);

    let cpu = measure(args, || {
        output.zero_grad();
        let loss = MSELoss::apply(&output, &target);
        loss.backward();
        black_box(output.grad().is_some());
    });
    let cuda = if lumen::ops::cuda::is_available() {
        let output_cuda = output.to_cuda();
        let target_cuda = target.to_cuda();
        measure_cuda(args, || {
            output_cuda.zero_grad();
            let loss = MSELoss::apply(&output_cuda, &target_cuda);
            loss.backward();
            black_box(output_cuda.requires_grad());
        })
    } else {
        None
    };
    BenchResult {
        name: "mse_loss.backward",
        cpu,
        cuda,
    }
}

fn grad_array(shape: &[usize], data: &[f32]) -> ndarray::ArrayD<f32> {
    Array::from_shape_vec(IxDyn(shape), data.to_vec())
        .expect("bench grad shape mismatch")
        .into_dyn()
}

fn bench_sgd_step(args: &Args, cfg: ShapeConfig) -> BenchResult {
    let shape = [cfg.elem_len];
    let param = tensor_grad(&shape, sample_data(cfg.elem_len, 0.01), args.dtype);
    let grad = sample_data(cfg.elem_len, -0.002);
    let grad_template = grad_array(&shape, &grad);
    let mut opt = SGD::new(vec![param.clone()], 0.001);

    let cpu = measure_with_setup(
        args,
        || {
            param.zero_grad();
            param.add_grad(grad_template.clone());
        },
        || {
            opt.step();
            black_box(param.shape_vec());
        },
    );
    let cuda = if lumen::ops::cuda::is_available() {
        let param_cuda = tensor_grad(&shape, sample_data(cfg.elem_len, 0.01), args.dtype).to_cuda();
        let cuda_grad_template = grad_template.clone();
        let mut opt_cuda = SGD::new(vec![param_cuda.clone()], 0.001);
        measure_cuda_with_setup(
            args,
            || {
                param_cuda.zero_grad();
                param_cuda.add_grad(cuda_grad_template.clone());
            },
            || {
                opt_cuda.step();
                black_box(param_cuda.shape_vec());
            },
        )
    } else {
        None
    };
    BenchResult {
        name: "optimizer.sgd.step",
        cpu,
        cuda,
    }
}

fn bench_adam_step(args: &Args, cfg: ShapeConfig) -> BenchResult {
    bench_adam_step_with_state_dtype(args, cfg, args.dtype, "optimizer.adam.step")
}

fn bench_adam_f32_state_step(args: &Args, cfg: ShapeConfig) -> BenchResult {
    bench_adam_step_with_state_dtype(args, cfg, DType::F32, "optimizer.adam_f32_state.step")
}

fn bench_adam_step_with_state_dtype(
    args: &Args,
    cfg: ShapeConfig,
    state_dtype: DType,
    name: &'static str,
) -> BenchResult {
    let shape = [cfg.elem_len];
    let param = tensor_grad(&shape, sample_data(cfg.elem_len, 0.01), args.dtype);
    let grad = sample_data(cfg.elem_len, -0.002);
    let grad_template = grad_array(&shape, &grad);
    let mut opt = Adam::new_with_dtype(vec![param.clone()], 0.001, state_dtype);

    let cpu = measure_with_setup(
        args,
        || {
            param.zero_grad();
            param.add_grad(grad_template.clone());
        },
        || {
            opt.step();
            black_box(param.shape_vec());
        },
    );
    let cuda = if lumen::ops::cuda::is_available() {
        let param_cuda = tensor_grad(&shape, sample_data(cfg.elem_len, 0.01), args.dtype).to_cuda();
        let cuda_grad_template = grad_template.clone();
        let mut opt_cuda = Adam::new_with_dtype(vec![param_cuda.clone()], 0.001, state_dtype);
        measure_cuda_with_setup(
            args,
            || {
                param_cuda.zero_grad();
                param_cuda.add_grad(cuda_grad_template.clone());
            },
            || {
                opt_cuda.step();
                black_box(param_cuda.shape_vec());
            },
        )
    } else {
        None
    };
    BenchResult { name, cpu, cuda }
}

fn bench_sgd_batched_step(args: &Args, cfg: ShapeConfig) -> BenchResult {
    bench_optimizer_batched_step(args, cfg, false, "optimizer.sgd_batched.step")
}

fn bench_adam_f32_state_batched_step(args: &Args, cfg: ShapeConfig) -> BenchResult {
    bench_optimizer_batched_step(args, cfg, true, "optimizer.adam_f32_state_batched.step")
}

fn bench_optimizer_batched_step(
    args: &Args,
    cfg: ShapeConfig,
    adam: bool,
    name: &'static str,
) -> BenchResult {
    let len = optimizer_batched_len(cfg);
    let shape = [len];
    let mut params = Vec::with_capacity(OPTIMIZER_BATCHED_PARAM_COUNT);
    let mut grad_templates = Vec::with_capacity(OPTIMIZER_BATCHED_PARAM_COUNT);
    for idx in 0..OPTIMIZER_BATCHED_PARAM_COUNT {
        let param = tensor_grad(
            &shape,
            sample_data(len, 0.01 + idx as f32 * 0.0003),
            args.dtype,
        );
        let grad = sample_data(len, -0.002 - idx as f32 * 0.0001);
        grad_templates.push(grad_array(&shape, &grad));
        params.push(param);
    }
    let cpu = if adam {
        let mut opt = Adam::new_with_dtype(params.clone(), 0.001, DType::F32);
        measure_with_setup(
            args,
            || {
                for (param, grad) in params.iter().zip(grad_templates.iter()) {
                    param.zero_grad();
                    param.add_grad(grad.clone());
                }
            },
            || {
                opt.step();
                black_box(params.len());
            },
        )
    } else {
        let mut opt = SGD::new(params.clone(), 0.001);
        measure_with_setup(
            args,
            || {
                for (param, grad) in params.iter().zip(grad_templates.iter()) {
                    param.zero_grad();
                    param.add_grad(grad.clone());
                }
            },
            || {
                opt.step();
                black_box(params.len());
            },
        )
    };

    let cuda = if lumen::ops::cuda::is_available() {
        let mut cuda_params = Vec::with_capacity(OPTIMIZER_BATCHED_PARAM_COUNT);
        let mut cuda_grad_templates = Vec::with_capacity(OPTIMIZER_BATCHED_PARAM_COUNT);
        for idx in 0..OPTIMIZER_BATCHED_PARAM_COUNT {
            let param = tensor_grad(
                &shape,
                sample_data(len, 0.01 + idx as f32 * 0.0003),
                args.dtype,
            )
            .to_cuda();
            let grad = sample_data(len, -0.002 - idx as f32 * 0.0001);
            cuda_grad_templates.push(grad_array(&shape, &grad));
            cuda_params.push(param);
        }
        if adam {
            let mut opt = Adam::new_with_dtype(cuda_params.clone(), 0.001, DType::F32);
            measure_cuda_with_setup(
                args,
                || {
                    for (param, grad) in cuda_params.iter().zip(cuda_grad_templates.iter()) {
                        param.zero_grad();
                        param.add_grad(grad.clone());
                    }
                },
                || {
                    opt.step();
                    black_box(cuda_params.len());
                },
            )
        } else {
            let mut opt = SGD::new(cuda_params.clone(), 0.001);
            measure_cuda_with_setup(
                args,
                || {
                    for (param, grad) in cuda_params.iter().zip(cuda_grad_templates.iter()) {
                        param.zero_grad();
                        param.add_grad(grad.clone());
                    }
                },
                || {
                    opt.step();
                    black_box(cuda_params.len());
                },
            )
        }
    } else {
        None
    };
    BenchResult { name, cpu, cuda }
}

fn bench_conv2d_forward(args: &Args, cfg: ShapeConfig) -> BenchResult {
    let input = tensor_no_grad(
        &[cfg.conv_batch, cfg.conv_in, cfg.conv_hw, cfg.conv_hw],
        sample_data(
            cfg.conv_batch * cfg.conv_in * cfg.conv_hw * cfg.conv_hw,
            0.01,
        ),
        args.dtype,
    );
    let conv = Conv2D::new_with_dtype(cfg.conv_in, cfg.conv_out, 3, 1, 1, args.dtype);

    let cpu = measure(args, || {
        let out = no_grad(|| conv.forward(input.clone()));
        black_box(out.shape_vec());
    });
    let cuda = if lumen::ops::cuda::is_available() {
        let conv_cuda = Conv2D::new_with_dtype(cfg.conv_in, cfg.conv_out, 3, 1, 1, args.dtype);
        conv_cuda.to_cuda();
        let input_cuda = input.to_cuda();
        measure_cuda(args, || {
            let out = no_grad(|| conv_cuda.forward(input_cuda.clone()));
            black_box(out.shape_vec());
        })
    } else {
        None
    };
    BenchResult {
        name: "conv2d.forward",
        cpu,
        cuda,
    }
}

fn bench_conv2d_backward(args: &Args, cfg: ShapeConfig) -> BenchResult {
    let input = tensor_grad(
        &[cfg.conv_batch, cfg.conv_in, cfg.conv_hw, cfg.conv_hw],
        sample_data(
            cfg.conv_batch * cfg.conv_in * cfg.conv_hw * cfg.conv_hw,
            0.01,
        ),
        args.dtype,
    );
    let coeff = tensor_const(
        &[cfg.conv_batch, cfg.conv_out, cfg.conv_hw, cfg.conv_hw],
        sample_data(
            cfg.conv_batch * cfg.conv_out * cfg.conv_hw * cfg.conv_hw,
            0.002,
        ),
        args.dtype,
    );
    let conv = Conv2D::new_with_dtype(cfg.conv_in, cfg.conv_out, 3, 1, 1, args.dtype);

    let cpu = measure(args, || {
        input.zero_grad();
        zero_params(&conv);
        let out = conv.forward(input.clone());
        let loss = sum(&(&out * &coeff));
        loss.backward();
        black_box(input.grad().is_some());
    });
    let cuda = if lumen::ops::cuda::is_available() {
        let conv_cuda = Conv2D::new_with_dtype(cfg.conv_in, cfg.conv_out, 3, 1, 1, args.dtype);
        conv_cuda.to_cuda();
        let input_cuda = input.to_cuda();
        let coeff_cuda = coeff.to_cuda();
        measure_cuda(args, || {
            input_cuda.zero_grad();
            zero_params(&conv_cuda);
            let out = conv_cuda.forward(input_cuda.clone());
            let loss = sum(&(&out * &coeff_cuda));
            loss.backward();
            black_box(input_cuda.requires_grad());
        })
    } else {
        None
    };
    BenchResult {
        name: "conv2d.backward",
        cpu,
        cuda,
    }
}

fn bench_max_pool_forward(args: &Args, cfg: ShapeConfig) -> BenchResult {
    let input = tensor_no_grad(
        &[cfg.conv_batch, cfg.conv_in, cfg.conv_hw, cfg.conv_hw],
        sample_data(
            cfg.conv_batch * cfg.conv_in * cfg.conv_hw * cfg.conv_hw,
            0.01,
        ),
        args.dtype,
    );
    let pool = MaxPool2D::new(2, 2);

    let cpu = measure(args, || {
        let out = no_grad(|| pool.forward(input.clone()));
        black_box(out.shape_vec());
    });
    let cuda = if lumen::ops::cuda::is_available() {
        let input_cuda = input.to_cuda();
        measure_cuda(args, || {
            let out = no_grad(|| pool.forward(input_cuda.clone()));
            black_box(out.shape_vec());
        })
    } else {
        None
    };
    BenchResult {
        name: "max_pool2d.forward",
        cpu,
        cuda,
    }
}

fn bench_max_pool_backward(args: &Args, cfg: ShapeConfig) -> BenchResult {
    let out_hw = cfg.conv_hw / 2;
    let input = tensor_grad(
        &[cfg.conv_batch, cfg.conv_in, cfg.conv_hw, cfg.conv_hw],
        sample_data(
            cfg.conv_batch * cfg.conv_in * cfg.conv_hw * cfg.conv_hw,
            0.01,
        ),
        args.dtype,
    );
    let coeff = tensor_const(
        &[cfg.conv_batch, cfg.conv_in, out_hw, out_hw],
        sample_data(cfg.conv_batch * cfg.conv_in * out_hw * out_hw, 0.002),
        args.dtype,
    );
    let pool = MaxPool2D::new(2, 2);

    let cpu = measure(args, || {
        input.zero_grad();
        let out = pool.forward(input.clone());
        let loss = sum(&(&out * &coeff));
        loss.backward();
        black_box(input.grad().is_some());
    });
    let cuda = if lumen::ops::cuda::is_available() {
        let input_cuda = input.to_cuda();
        let coeff_cuda = coeff.to_cuda();
        measure_cuda(args, || {
            input_cuda.zero_grad();
            let out = pool.forward(input_cuda.clone());
            let loss = sum(&(&out * &coeff_cuda));
            loss.backward();
            black_box(input_cuda.requires_grad());
        })
    } else {
        None
    };
    BenchResult {
        name: "max_pool2d.backward",
        cpu,
        cuda,
    }
}

fn bench_attention_forward(args: &Args, cfg: ShapeConfig) -> BenchResult {
    let input = tensor_no_grad(
        &[cfg.attention_batch, cfg.attention_seq, cfg.attention_hidden],
        sample_data(
            cfg.attention_batch * cfg.attention_seq * cfg.attention_hidden,
            0.01,
        ),
        args.dtype,
    );
    let attn = SelfAttention::new_with_dtype(
        cfg.attention_hidden,
        cfg.attention_heads,
        cfg.attention_kv_heads,
        cfg.attention_seq,
        10000.0,
        true,
        args.dtype,
    );

    let cpu = measure(args, || {
        let (out, _) = no_grad(|| attn.forward(input.clone(), None));
        black_box(out.shape_vec());
    });
    let cuda = if lumen::ops::cuda::is_available() {
        let attn_cuda = SelfAttention::new_with_dtype(
            cfg.attention_hidden,
            cfg.attention_heads,
            cfg.attention_kv_heads,
            cfg.attention_seq,
            10000.0,
            true,
            args.dtype,
        );
        attn_cuda.to_cuda();
        let input_cuda = input.to_cuda();
        measure_cuda(args, || {
            let (out, _) = no_grad(|| attn_cuda.forward(input_cuda.clone(), None));
            black_box(out.shape_vec());
        })
    } else {
        None
    };
    BenchResult {
        name: "self_attention.forward",
        cpu,
        cuda,
    }
}

fn attention_bias_dims(cfg: ShapeConfig) -> (usize, usize) {
    let head_dim = cfg.attention_hidden / cfg.attention_heads;
    let kv_dim = cfg.attention_kv_heads * head_dim;
    (cfg.attention_hidden, kv_dim)
}

fn attach_attention_biases(attn: &mut SelfAttention, cfg: ShapeConfig, dtype: DType) {
    let (q_dim, kv_dim) = attention_bias_dims(cfg);
    attn.w_q.bias = Some(tensor_grad(&[q_dim], sample_data(q_dim, 0.001), dtype));
    attn.w_k.bias = Some(tensor_grad(&[kv_dim], sample_data(kv_dim, -0.0015), dtype));
    attn.w_v.bias = Some(tensor_grad(&[kv_dim], sample_data(kv_dim, 0.002), dtype));
    attn.w_o.bias = Some(tensor_grad(&[q_dim], sample_data(q_dim, -0.0005), dtype));
}

fn bench_attention_backward(args: &Args, cfg: ShapeConfig) -> BenchResult {
    let input = tensor_grad(
        &[cfg.attention_batch, cfg.attention_seq, cfg.attention_hidden],
        sample_data(
            cfg.attention_batch * cfg.attention_seq * cfg.attention_hidden,
            0.01,
        ),
        args.dtype,
    );
    let coeff = tensor_const(
        &[cfg.attention_batch, cfg.attention_seq, cfg.attention_hidden],
        sample_data(
            cfg.attention_batch * cfg.attention_seq * cfg.attention_hidden,
            0.002,
        ),
        args.dtype,
    );
    let attn = SelfAttention::new_with_dtype(
        cfg.attention_hidden,
        cfg.attention_heads,
        cfg.attention_kv_heads,
        cfg.attention_seq,
        10000.0,
        true,
        args.dtype,
    );
    let attn_params = attn.parameters();

    let cpu = measure(args, || {
        input.zero_grad();
        zero_param_list(&attn_params);
        let (out, _) = attn.forward(input.clone(), None);
        let loss = sum(&(&out * &coeff));
        loss.backward();
        black_box(input.grad().is_some());
    });
    let cuda = if lumen::ops::cuda::is_available() {
        let attn_cuda = SelfAttention::new_with_dtype(
            cfg.attention_hidden,
            cfg.attention_heads,
            cfg.attention_kv_heads,
            cfg.attention_seq,
            10000.0,
            true,
            args.dtype,
        );
        attn_cuda.to_cuda();
        let attn_cuda_params = attn_cuda.parameters();
        let input_cuda = input.to_cuda();
        let coeff_cuda = coeff.to_cuda();
        measure_cuda(args, || {
            input_cuda.zero_grad();
            zero_param_list(&attn_cuda_params);
            let (out, _) = attn_cuda.forward(input_cuda.clone(), None);
            let loss = sum(&(&out * &coeff_cuda));
            loss.backward();
            black_box(input_cuda.requires_grad());
        })
    } else {
        None
    };
    BenchResult {
        name: "self_attention.backward",
        cpu,
        cuda,
    }
}

fn bench_attention_bias_backward(args: &Args, cfg: ShapeConfig) -> BenchResult {
    let input = tensor_grad(
        &[cfg.attention_batch, cfg.attention_seq, cfg.attention_hidden],
        sample_data(
            cfg.attention_batch * cfg.attention_seq * cfg.attention_hidden,
            0.01,
        ),
        args.dtype,
    );
    let coeff = tensor_const(
        &[cfg.attention_batch, cfg.attention_seq, cfg.attention_hidden],
        sample_data(
            cfg.attention_batch * cfg.attention_seq * cfg.attention_hidden,
            0.002,
        ),
        args.dtype,
    );
    let mut attn = SelfAttention::new_with_dtype(
        cfg.attention_hidden,
        cfg.attention_heads,
        cfg.attention_kv_heads,
        cfg.attention_seq,
        10000.0,
        true,
        args.dtype,
    );
    attach_attention_biases(&mut attn, cfg, args.dtype);
    let attn_params = attn.parameters();
    let attn_param_count = attn_params.len();

    let cpu = measure(args, || {
        input.zero_grad();
        zero_param_list(&attn_params);
        let (out, _) = attn.forward(input.clone(), None);
        let loss = sum(&(&out * &coeff));
        loss.backward();
        black_box(attn_param_count);
    });
    let cuda = if lumen::ops::cuda::is_available() {
        let mut attn_cuda = SelfAttention::new_with_dtype(
            cfg.attention_hidden,
            cfg.attention_heads,
            cfg.attention_kv_heads,
            cfg.attention_seq,
            10000.0,
            true,
            args.dtype,
        );
        attach_attention_biases(&mut attn_cuda, cfg, args.dtype);
        attn_cuda.to_cuda();
        let attn_cuda_params = attn_cuda.parameters();
        let attn_cuda_param_count = attn_cuda_params.len();
        let input_cuda = input.to_cuda();
        let coeff_cuda = coeff.to_cuda();
        measure_cuda(args, || {
            input_cuda.zero_grad();
            zero_param_list(&attn_cuda_params);
            let (out, _) = attn_cuda.forward(input_cuda.clone(), None);
            let loss = sum(&(&out * &coeff_cuda));
            loss.backward();
            black_box(attn_cuda_param_count);
        })
    } else {
        None
    };
    BenchResult {
        name: "self_attention_bias.backward",
        cpu,
        cuda,
    }
}

fn bench_llama_infer_last_logits(args: &Args, cfg: ShapeConfig) -> BenchResult {
    let llama_cfg = llama_config(cfg);
    let input = token_tensor(
        &[cfg.attention_batch, cfg.attention_seq],
        token_id_data(cfg.attention_batch, cfg.attention_seq, llama_cfg.vocab_size),
    );
    let model = LlamaModel::new_with_dtype(llama_cfg.clone(), args.dtype);
    let mut caches = model.init_kv_caches(cfg.attention_batch);

    let cpu = measure(args, || {
        model.reset_kv_caches(&mut caches);
        let out = no_grad(|| model.forward_last_logits(input.clone(), &mut caches, 0));
        black_box(out.shape_vec());
    });
    let cuda = if lumen::ops::cuda::is_available() {
        let model_cuda = LlamaModel::new_with_dtype(llama_cfg, args.dtype);
        model_cuda.to_cuda();
        let mut caches_cuda = model_cuda.init_kv_caches(cfg.attention_batch);
        let input_cuda = input.to_cuda();
        measure_cuda(args, || {
            model_cuda.reset_kv_caches(&mut caches_cuda);
            let out =
                no_grad(|| model_cuda.forward_last_logits(input_cuda.clone(), &mut caches_cuda, 0));
            black_box(out.shape_vec());
        })
    } else {
        None
    };
    BenchResult {
        name: "llama.infer_last_logits",
        cpu,
        cuda,
    }
}

fn bench_llama_prefill_decode(args: &Args, cfg: ShapeConfig) -> BenchResult {
    let llama_cfg = llama_config(cfg);
    let prefill_input = token_tensor(
        &[cfg.attention_batch, cfg.attention_seq],
        token_id_data(cfg.attention_batch, cfg.attention_seq, llama_cfg.vocab_size),
    );
    let decode_input = token_tensor(
        &[cfg.attention_batch, 1],
        token_id_data(cfg.attention_batch, 1, llama_cfg.vocab_size),
    );
    let model = LlamaModel::new_with_dtype(llama_cfg.clone(), args.dtype);
    let mut caches = model.init_kv_caches(cfg.attention_batch);

    let cpu = measure(args, || {
        model.reset_kv_caches(&mut caches);
        no_grad(|| {
            let prefill = model.forward_last_logits(prefill_input.clone(), &mut caches, 0);
            let decode =
                model.forward_last_logits(decode_input.clone(), &mut caches, cfg.attention_seq);
            black_box((prefill.shape_vec(), decode.shape_vec()));
        });
    });
    let cuda = if lumen::ops::cuda::is_available() {
        let model_cuda = LlamaModel::new_with_dtype(llama_cfg, args.dtype);
        model_cuda.to_cuda();
        let mut caches_cuda = model_cuda.init_kv_caches(cfg.attention_batch);
        let prefill_cuda = prefill_input.to_cuda();
        let decode_cuda = decode_input.to_cuda();
        measure_cuda(args, || {
            model_cuda.reset_kv_caches(&mut caches_cuda);
            no_grad(|| {
                let prefill =
                    model_cuda.forward_last_logits(prefill_cuda.clone(), &mut caches_cuda, 0);
                let decode = model_cuda.forward_last_logits(
                    decode_cuda.clone(),
                    &mut caches_cuda,
                    cfg.attention_seq,
                );
                black_box((prefill.shape_vec(), decode.shape_vec()));
            });
        })
    } else {
        None
    };
    BenchResult {
        name: "llama.prefill_decode",
        cpu,
        cuda,
    }
}

fn bench_llama_train_backward(args: &Args, cfg: ShapeConfig) -> BenchResult {
    let llama_cfg = llama_config(cfg);
    let rows = cfg.attention_batch * cfg.attention_seq;
    let input = token_tensor(
        &[cfg.attention_batch, cfg.attention_seq],
        token_id_data(cfg.attention_batch, cfg.attention_seq, llama_cfg.vocab_size),
    );
    let targets = tensor_const(
        &[rows, llama_cfg.vocab_size],
        one_hot_data(rows, llama_cfg.vocab_size),
        args.dtype,
    );
    let model = LlamaModel::new_with_dtype(llama_cfg.clone(), args.dtype);
    let model_params = model.parameters();
    let model_param_count = model_params.len();

    let cpu = measure(args, || {
        zero_param_list(&model_params);
        let logits = model.forward_train(input.clone());
        let loss = CrossEntropyLoss::apply(
            &logits.reshape(vec![-1, llama_cfg.vocab_size as i32]),
            &targets,
        );
        loss.backward();
        black_box(model_param_count);
    });
    let cuda = if lumen::ops::cuda::is_available() {
        let model_cuda = LlamaModel::new_with_dtype(llama_cfg.clone(), args.dtype);
        model_cuda.to_cuda();
        let model_cuda_params = model_cuda.parameters();
        let model_cuda_param_count = model_cuda_params.len();
        let input_cuda = input.to_cuda();
        let targets_cuda = targets.to_cuda();
        measure_cuda(args, || {
            zero_param_list(&model_cuda_params);
            let logits = model_cuda.forward_train(input_cuda.clone());
            let loss = CrossEntropyLoss::apply(
                &logits.reshape(vec![-1, llama_cfg.vocab_size as i32]),
                &targets_cuda,
            );
            loss.backward();
            black_box(model_cuda_param_count);
        })
    } else {
        None
    };
    BenchResult {
        name: "llama.train.backward",
        cpu,
        cuda,
    }
}

fn bench_llama_train_step(args: &Args, cfg: ShapeConfig) -> BenchResult {
    let llama_cfg = llama_config(cfg);
    let rows = cfg.attention_batch * cfg.attention_seq;
    let input = token_tensor(
        &[cfg.attention_batch, cfg.attention_seq],
        token_id_data(cfg.attention_batch, cfg.attention_seq, llama_cfg.vocab_size),
    );
    let targets = tensor_const(
        &[rows, llama_cfg.vocab_size],
        one_hot_data(rows, llama_cfg.vocab_size),
        args.dtype,
    );
    let model = LlamaModel::new_with_dtype(llama_cfg.clone(), args.dtype);
    let model_params = model.parameters();
    let model_param_count = model_params.len();
    let mut opt = SGD::new_with_dtype(model_params.clone(), 0.001, DType::F32).with_momentum(0.5);

    let cpu = measure(args, || {
        zero_param_list(&model_params);
        let loss = llama_train_loss(&model, input.clone(), &targets, llama_cfg.vocab_size);
        loss.backward();
        opt.step();
        black_box(model_param_count);
    });
    let cuda = if lumen::ops::cuda::is_available() {
        let model_cuda = LlamaModel::new_with_dtype(llama_cfg.clone(), args.dtype);
        model_cuda.to_cuda();
        let model_cuda_params = model_cuda.parameters();
        let model_cuda_param_count = model_cuda_params.len();
        let input_cuda = input.to_cuda();
        let targets_cuda = targets.to_cuda();
        let mut opt_cuda =
            SGD::new_with_dtype(model_cuda_params.clone(), 0.001, DType::F32).with_momentum(0.5);
        measure_cuda(args, || {
            zero_param_list(&model_cuda_params);
            let loss = llama_train_loss(
                &model_cuda,
                input_cuda.clone(),
                &targets_cuda,
                llama_cfg.vocab_size,
            );
            loss.backward();
            opt_cuda.step();
            black_box(model_cuda_param_count);
        })
    } else {
        None
    };
    BenchResult {
        name: "llama.train.step",
        cpu,
        cuda,
    }
}

fn should_run(args: &Args, name: &str) -> bool {
    let suite_match = match args.suite {
        Suite::All => true,
        Suite::Ops => {
            name.starts_with("matmul")
                || name.starts_with("batch_matmul")
                || name.starts_with("elementwise")
                || name.starts_with("binary")
                || name.starts_with("unary")
                || name.starts_with("fused_")
                || name.starts_with("softmax")
                || name.starts_with("embedding")
                || name.starts_with("rms_norm")
                || name.starts_with("rope")
                || name.starts_with("cross_entropy")
                || name.starts_with("mse_loss")
        }
        Suite::Nn => {
            name.starts_with("conv2d")
                || name.starts_with("max_pool")
                || name.starts_with("embedding")
                || name.starts_with("rms_norm")
                || name.starts_with("rope")
                || name.starts_with("self_attention")
                || name.starts_with("llama")
        }
        Suite::Backward => {
            name.ends_with(".backward")
                || name.starts_with("optimizer")
                || name == "llama.train.step"
        }
        Suite::Path => false,
    };
    suite_match
        && args
            .case_filter
            .as_ref()
            .is_none_or(|filter| name.contains(filter))
}

fn skip_reason_for_dtype(args: &Args, name: &str) -> Option<&'static str> {
    if args.dtype != DType::I8 {
        return None;
    }
    if name == "optimizer.adam.step" {
        return Some(
            "Adam optimizer state intentionally supports floating dtypes only; use optimizer.adam_f32_state.step for I8 parameters with F32 optimizer state",
        );
    }
    if name.starts_with("self_attention") {
        return Some(
            "SelfAttention KV cache currently supports floating dtypes only; use path.train for I8 parameter training checks",
        );
    }
    if name.starts_with("llama") {
        return Some(
            "Llama bench/check constructors require floating runtime and KV-cache dtypes; use path.train for I8 parameter training checks",
        );
    }
    None
}

fn print_dtype_skip(kind: &str, name: &str, dtype: DType, reason: &str) {
    println!("{kind} {name:<26} skipped dtype={dtype:?} note={reason}");
}

fn ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1e3
}

fn print_result(result: &BenchResult) {
    match result.cuda {
        Some(cuda) => {
            let speedup = result.cpu.as_secs_f64() / cuda.as_secs_f64();
            println!(
                "{:<32} cpu={:>9.3} ms  cuda={:>9.3} ms  speedup={:>7.2}x",
                result.name,
                ms(result.cpu),
                ms(cuda),
                speedup
            );
        }
        None => {
            println!(
                "{:<32} cpu={:>9.3} ms  cuda=   skipped  speedup=    n/a",
                result.name,
                ms(result.cpu)
            );
        }
    }
}

fn print_check_result(result: &CheckResult) {
    print!("check {:<26} ok", result.name);
    for metric in &result.metrics {
        print!(
            " {}_abs={:.3e} {}_rel={:.3e} {}_rmse={:.3e}",
            metric.label, metric.abs, metric.label, metric.rel, metric.label, metric.rmse
        );
    }
    println!();
}

fn main() {
    let args = parse_args().unwrap_or_else(|err| {
        eprintln!("error: {err}");
        std::process::exit(2);
    });
    let cfg = shape_config(args.size);
    println!(
        "cuda/cpu bench: suite={:?} size={:?} dtype={:?} runs={} warmup={} check={} cuda_available={}",
        args.suite,
        args.size,
        args.dtype,
        args.runs,
        args.warmup,
        args.check,
        lumen::ops::cuda::is_available()
    );
    if !lumen::ops::cuda::is_available() {
        println!(
            "note: CUDA unavailable or binary was not built with --features cuda; CUDA columns will be skipped."
        );
    }

    let benches = [
        BenchDef {
            name: "matmul.forward",
            run: bench_matmul_forward,
        },
        BenchDef {
            name: "batch_matmul.forward",
            run: bench_batch_matmul_forward,
        },
        BenchDef {
            name: "matmul.backward",
            run: bench_matmul_backward,
        },
        BenchDef {
            name: "elementwise.mul_add.forward",
            run: bench_elementwise_forward,
        },
        BenchDef {
            name: "binary.same_shape.forward",
            run: bench_binary_forward,
        },
        BenchDef {
            name: "binary.row_broadcast.forward",
            run: bench_binary_row_broadcast_forward,
        },
        BenchDef {
            name: "binary.row_scalar.forward",
            run: bench_binary_row_scalar_forward,
        },
        BenchDef {
            name: "binary.b1d_1h1.forward",
            run: bench_binary_b1d_1h1_forward,
        },
        BenchDef {
            name: "binary.b1d_1hd.forward",
            run: bench_binary_b1d_1hd_forward,
        },
        BenchDef {
            name: "binary.general_broadcast.forward",
            run: bench_binary_general_broadcast_forward,
        },
        BenchDef {
            name: "elementwise.mul_add.backward",
            run: bench_elementwise_backward,
        },
        BenchDef {
            name: "elementwise.mixed_mul.forward",
            run: bench_elementwise_mixed_mul_forward,
        },
        BenchDef {
            name: "elementwise.mixed_broadcast_1hd_mul.forward",
            run: bench_elementwise_mixed_broadcast_1hd_mul_forward,
        },
        BenchDef {
            name: "elementwise.mixed_row_scalar_mul.forward",
            run: bench_elementwise_mixed_row_scalar_mul_forward,
        },
        BenchDef {
            name: "elementwise.mixed_mul.backward",
            run: bench_elementwise_mixed_mul_backward,
        },
        BenchDef {
            name: "elementwise.mixed_row_mul.forward",
            run: bench_elementwise_mixed_row_mul_forward,
        },
        BenchDef {
            name: "elementwise.mixed_row_mul.backward",
            run: bench_elementwise_mixed_row_mul_backward,
        },
        BenchDef {
            name: "elementwise.mixed_row_scalar_mul.backward",
            run: bench_elementwise_mixed_row_scalar_mul_backward,
        },
        BenchDef {
            name: "elementwise.mixed_row_sub.backward",
            run: bench_elementwise_mixed_row_sub_backward,
        },
        BenchDef {
            name: "elementwise.mixed_broadcast_sub.backward",
            run: bench_elementwise_mixed_broadcast_sub_backward,
        },
        BenchDef {
            name: "elementwise.mixed_broadcast_1hd_sub.backward",
            run: bench_elementwise_mixed_broadcast_1hd_sub_backward,
        },
        BenchDef {
            name: "elementwise.mixed_broadcast_mul.backward",
            run: bench_elementwise_mixed_broadcast_mul_backward,
        },
        BenchDef {
            name: "elementwise.mixed_broadcast_1hd_mul.backward",
            run: bench_elementwise_mixed_broadcast_1hd_mul_backward,
        },
        BenchDef {
            name: "elementwise.mixed_scalar_sub.backward",
            run: bench_elementwise_mixed_scalar_sub_backward,
        },
        BenchDef {
            name: "elementwise.mixed_scalar_mul.backward",
            run: bench_elementwise_mixed_scalar_mul_backward,
        },
        BenchDef {
            name: "unary.silu.forward",
            run: bench_unary_silu_forward,
        },
        BenchDef {
            name: "unary.relu.forward",
            run: bench_unary_relu_forward,
        },
        BenchDef {
            name: "unary.silu.backward",
            run: bench_unary_silu_backward,
        },
        BenchDef {
            name: "fused_gateup.forward",
            run: bench_fused_gateup_forward,
        },
        BenchDef {
            name: "fused_qkv.decode",
            run: bench_fused_qkv_decode,
        },
        BenchDef {
            name: "fused_qkv.prefill",
            run: bench_fused_qkv_prefill,
        },
        BenchDef {
            name: "softmax.forward",
            run: bench_softmax_forward,
        },
        BenchDef {
            name: "softmax.backward",
            run: bench_softmax_backward,
        },
        BenchDef {
            name: "fused_softmax.forward",
            run: bench_fused_softmax_forward,
        },
        BenchDef {
            name: "fused_softmax.backward",
            run: bench_fused_softmax_backward,
        },
        BenchDef {
            name: "embedding.forward",
            run: bench_embedding_forward,
        },
        BenchDef {
            name: "rms_norm.forward",
            run: bench_rms_norm_forward,
        },
        BenchDef {
            name: "rope.forward",
            run: bench_rope_forward,
        },
        BenchDef {
            name: "cross_entropy.forward",
            run: bench_cross_entropy_forward,
        },
        BenchDef {
            name: "cross_entropy.backward",
            run: bench_cross_entropy_backward,
        },
        BenchDef {
            name: "mse_loss.forward",
            run: bench_mse_loss_forward,
        },
        BenchDef {
            name: "mse_loss.backward",
            run: bench_mse_loss_backward,
        },
        BenchDef {
            name: "optimizer.sgd.step",
            run: bench_sgd_step,
        },
        BenchDef {
            name: "optimizer.adam.step",
            run: bench_adam_step,
        },
        BenchDef {
            name: "optimizer.adam_f32_state.step",
            run: bench_adam_f32_state_step,
        },
        BenchDef {
            name: "optimizer.sgd_batched.step",
            run: bench_sgd_batched_step,
        },
        BenchDef {
            name: "optimizer.adam_f32_state_batched.step",
            run: bench_adam_f32_state_batched_step,
        },
        BenchDef {
            name: "conv2d.forward",
            run: bench_conv2d_forward,
        },
        BenchDef {
            name: "conv2d.backward",
            run: bench_conv2d_backward,
        },
        BenchDef {
            name: "max_pool2d.forward",
            run: bench_max_pool_forward,
        },
        BenchDef {
            name: "max_pool2d.backward",
            run: bench_max_pool_backward,
        },
        BenchDef {
            name: "self_attention.forward",
            run: bench_attention_forward,
        },
        BenchDef {
            name: "self_attention.backward",
            run: bench_attention_backward,
        },
        BenchDef {
            name: "self_attention_bias.backward",
            run: bench_attention_bias_backward,
        },
        BenchDef {
            name: "llama.infer_last_logits",
            run: bench_llama_infer_last_logits,
        },
        BenchDef {
            name: "llama.prefill_decode",
            run: bench_llama_prefill_decode,
        },
        BenchDef {
            name: "llama.train.backward",
            run: bench_llama_train_backward,
        },
        BenchDef {
            name: "llama.train.step",
            run: bench_llama_train_step,
        },
    ];

    let checks = [
        CheckDef {
            name: "matmul.backward",
            run: check_matmul_backward,
        },
        CheckDef {
            name: "matmul.matrix.accuracy",
            run: check_matmul_matrix_accuracy,
        },
        CheckDef {
            name: "elementwise.mul_add.backward",
            run: check_elementwise_backward,
        },
        CheckDef {
            name: "binary.same_shape.forward",
            run: check_binary_forward,
        },
        CheckDef {
            name: "binary.row_broadcast.forward",
            run: check_binary_row_broadcast_forward,
        },
        CheckDef {
            name: "binary.special_broadcast.forward",
            run: check_binary_special_broadcast_forward,
        },
        CheckDef {
            name: "elementwise.broadcast_matrix.accuracy",
            run: check_elementwise_broadcast_matrix_accuracy,
        },
        CheckDef {
            name: "unary.relu.forward",
            run: check_unary_relu_forward,
        },
        CheckDef {
            name: "unary.silu.backward",
            run: check_unary_silu_backward,
        },
        CheckDef {
            name: "softmax.backward",
            run: check_softmax_backward,
        },
        CheckDef {
            name: "softmax.forward",
            run: check_softmax_forward,
        },
        CheckDef {
            name: "fused_softmax.forward",
            run: check_fused_softmax_forward,
        },
        CheckDef {
            name: "fused_softmax.backward",
            run: check_fused_softmax_backward,
        },
        CheckDef {
            name: "embedding.forward",
            run: check_embedding_forward,
        },
        CheckDef {
            name: "rms_norm.forward",
            run: check_rms_norm_forward,
        },
        CheckDef {
            name: "rope.forward",
            run: check_rope_forward,
        },
        CheckDef {
            name: "cross_entropy.backward",
            run: check_cross_entropy_backward,
        },
        CheckDef {
            name: "optimizer.sgd.step",
            run: check_sgd_step,
        },
        CheckDef {
            name: "optimizer.adam.step",
            run: check_adam_step,
        },
        CheckDef {
            name: "optimizer.adam_f32_state.step",
            run: check_adam_f32_state_step,
        },
        CheckDef {
            name: "optimizer.sgd_batched.step",
            run: check_sgd_batched_step,
        },
        CheckDef {
            name: "optimizer.adam_f32_state_batched.step",
            run: check_adam_f32_state_batched_step,
        },
        CheckDef {
            name: "llama.train.backward",
            run: check_llama_train_backward,
        },
        CheckDef {
            name: "llama.train.step",
            run: check_llama_train_step,
        },
    ];

    BenchPlan {
        checks: &checks,
        benches: &benches,
    }
    .run(&args, cfg);
    cuda_cpu_bench_path::run_path_checks(&args);
}
