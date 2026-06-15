use super::{Args, Suite, scalar_value};
use lumen::autograd::{Device, Tensor, no_grad, set_strict_device_execution_scoped};
use lumen::init::{ParameterInitMode, with_parameter_init_mode};
use lumen::loader::{ModelLoader, WeightLoadOptions};
use lumen::loss::MSELoss;
use lumen::models::{LlamaConfig, LlamaModel};
use lumen::module::Module;
use lumen::optim::{Optimizer, SGD};
use lumen::precision::{
    DType, ParameterQuantization, PrecisionConfig, with_precision_config,
    with_runtime_component_dtypes,
};
use lumen::tokenizer::LlamaTokenizer;
use ndarray::{Array, Ix3, IxDyn, s};
use std::path::Path;
use std::time::{Duration, Instant};

fn real_llama_config(max_seq_len: usize) -> LlamaConfig {
    LlamaConfig {
        vocab_size: 32000,
        hidden_size: 2048,
        intermediate_size: 5632,
        num_hidden_layers: 22,
        num_attention_heads: 32,
        num_key_value_heads: 4,
        rms_norm_eps: 1e-5,
        max_seq_len,
        rope_theta: 10000.0,
    }
}

fn build_first_turn_prompt(system: &str, user: &str) -> String {
    format!(
        "<|system|>\n{}\n</s>\n<|user|>\n{}\n</s>\n<|assistant|>\n",
        system, user
    )
}

fn token_ids_tensor(ids: &[usize], device: Device) -> Tensor {
    Tensor::from_array_no_grad(
        Array::from_shape_vec((1, ids.len()), ids.iter().map(|&id| id as f32).collect())
            .expect("token id tensor shape mismatch")
            .into_dyn(),
    )
    .to_device(device)
}

fn generated_stop_ids(tokenizer: &LlamaTokenizer) -> Vec<usize> {
    let mut stop_ids = Vec::new();
    for token in ["</s>", "<|system|>", "<|user|>", "<|assistant|>"] {
        if let Some(id) = tokenizer.token_to_id(token) {
            stop_ids.push(id);
        }
    }
    if let Some(id) = tokenizer.eos_id() {
        stop_ids.push(id);
    }
    if let Some(id) = tokenizer.eot_id() {
        stop_ids.push(id);
    }
    stop_ids.sort_unstable();
    stop_ids.dedup();
    stop_ids
}

fn last_step_logits_vec(logits: &Tensor) -> Vec<f32> {
    let logits_ref = logits.data_ref();
    let view3 = logits_ref
        .view()
        .into_dimensionality::<Ix3>()
        .expect("logits must be [B, S, V]");
    let last_t = view3.shape()[1] - 1;
    view3.slice(s![0, last_t, ..]).iter().copied().collect()
}

fn argmax(values: &[f32]) -> usize {
    let mut best_idx = 0usize;
    let mut best_value = f32::NEG_INFINITY;
    for (idx, &value) in values.iter().enumerate() {
        if value > best_value {
            best_value = value;
            best_idx = idx;
        }
    }
    best_idx
}

struct TrainingPathStats {
    losses: Vec<f32>,
    elapsed: Duration,
}

impl TrainingPathStats {
    fn steps(&self) -> usize {
        self.losses.len()
    }

    fn us_per_step(&self) -> f64 {
        let steps = self.steps().max(1) as f64;
        self.elapsed.as_secs_f64() * 1e6 / steps
    }

    fn first(&self) -> f32 {
        self.losses.first().copied().unwrap_or_default()
    }

    fn last(&self) -> f32 {
        self.losses.last().copied().unwrap_or_default()
    }

    fn best(&self) -> f32 {
        self.losses
            .iter()
            .copied()
            .fold(f32::INFINITY, |acc, loss| acc.min(loss))
    }
}

fn simple_training_path_stats(dtype: DType, device: Device) -> TrainingPathStats {
    let _cuda_enabled_guard =
        (device == Device::Cuda).then(|| lumen::ops::cuda::set_enabled_scoped(true));
    let _strict_guard = (device == Device::Cuda).then(|| set_strict_device_execution_scoped(true));

    let x = Tensor::from_array_no_grad(
        Array::from_shape_vec(IxDyn(&[4]), vec![-1.0, 0.0, 1.0, 2.0])
            .unwrap()
            .into_dyn(),
    )
    .to_device(device);
    let y = Tensor::from_array_no_grad(
        Array::from_shape_vec(IxDyn(&[4]), vec![-1.0, 1.0, 3.0, 5.0])
            .unwrap()
            .into_dyn(),
    )
    .to_device(device);
    let make_param = |value: f32| {
        let data = Array::from_shape_vec(IxDyn(&[1]), vec![value])
            .unwrap()
            .into_dyn();
        if dtype == DType::I8 {
            Tensor::parameter_with_quantization(data, ParameterQuantization::Int8.with_scale(0.02))
        } else {
            Tensor::parameter_with_dtype(data, dtype)
        }
        .to_device(device)
    };
    let w = make_param(0.0);
    let b = make_param(0.0);
    let mut opt =
        SGD::new_with_dtype(vec![w.clone(), b.clone()], 0.08, DType::F32).with_momentum(0.5);
    let mut losses = Vec::new();

    if device == Device::Cuda {
        lumen::ops::cuda::synchronize().expect("CUDA sync before path training failed");
    }
    let start = Instant::now();
    for _ in 0..24 {
        let pred = &(&x * &w) + &b;
        let loss = MSELoss::apply(&pred, &y);
        losses.push(scalar_value(&loss));
        loss.backward();
        if device == Device::Cuda {
            assert!(
                w.dev_has_cuda_f32_grad() && b.dev_has_cuda_f32_grad(),
                "path.train.cuda expected CUDA f32 gradients after backward"
            );
            assert!(
                !w.dev_has_host_grad() && !b.dev_has_host_grad(),
                "path.train.cuda should not materialize host gradients during strict CUDA training"
            );
        }
        opt.step();
        if device == Device::Cuda {
            assert_eq!(
                opt.dev_velocity_count(),
                2,
                "path.train.cuda expected one momentum velocity per parameter"
            );
            assert!(
                opt.dev_all_velocities_are_f32_cuda_resident(),
                "path.train.cuda expected all momentum velocities to be f32 CUDA-resident state"
            );
        }
        opt.zero_grad();
    }
    if device == Device::Cuda {
        lumen::ops::cuda::synchronize().expect("CUDA sync after path training failed");
    }

    TrainingPathStats {
        losses,
        elapsed: start.elapsed(),
    }
}

fn count_loss_increases(losses: &[f32]) -> usize {
    losses.windows(2).filter(|pair| pair[1] > pair[0]).count()
}

fn assert_sgd_like_loss_trace(name: &str, losses: &[f32]) {
    assert!(!losses.is_empty(), "{name} produced no losses");
    assert!(
        losses.iter().all(|loss| loss.is_finite()),
        "{name} produced non-finite losses: {losses:?}"
    );
    let first = losses[0];
    let last = *losses.last().expect("checked non-empty");
    let best = losses
        .iter()
        .copied()
        .fold(f32::INFINITY, |acc, loss| acc.min(loss));
    assert!(
        last < first && best < first * 0.05,
        "{name} loss trace does not look SGD-like: first={first:.6} last={last:.6} best={best:.6} losses={losses:?}"
    );
}

fn run_training_path_check(args: &Args) {
    let cpu_stats = simple_training_path_stats(args.dtype, Device::Cpu);
    assert_sgd_like_loss_trace("path.train.cpu", &cpu_stats.losses);
    println!(
        "path.train.cpu             ok first={:.6} last={:.6} best={:.6} increases={} steps={} total_ms={:.3} us_per_step={:.2} note=SGD+momentum trace is not required to be monotonic",
        cpu_stats.first(),
        cpu_stats.last(),
        cpu_stats.best(),
        count_loss_increases(&cpu_stats.losses),
        cpu_stats.steps(),
        cpu_stats.elapsed.as_secs_f64() * 1e3,
        cpu_stats.us_per_step()
    );

    if lumen::ops::cuda::is_available() {
        let cuda_stats = simple_training_path_stats(args.dtype, Device::Cuda);
        assert_sgd_like_loss_trace("path.train.cuda", &cuda_stats.losses);
        println!(
            "path.train.cuda            ok first={:.6} last={:.6} best={:.6} increases={} steps={} total_ms={:.3} us_per_step={:.2} note=lowp params with f32 gradients and f32 momentum state",
            cuda_stats.first(),
            cuda_stats.last(),
            cuda_stats.best(),
            count_loss_increases(&cuda_stats.losses),
            cuda_stats.steps(),
            cuda_stats.elapsed.as_secs_f64() * 1e3,
            cuda_stats.us_per_step()
        );
    } else {
        println!("path.train.cuda            skipped cuda_available=false");
    }
}

fn load_real_llama_model(args: &Args, config: &LlamaConfig) -> LlamaModel {
    let weights = args
        .weights
        .as_ref()
        .expect("weights path should be checked before loading");
    let precision_config = PrecisionConfig {
        parameter_dtype: args.dtype,
        runtime_dtype: args.dtype,
        allow_parameter_dtype_copies: args.allow_parameter_copies,
    };
    let load_options = WeightLoadOptions {
        float_source_quantization: Default::default(),
        stream_from_disk: args.stream_weights,
    };
    let model = with_precision_config(precision_config, || {
        with_runtime_component_dtypes(Some(args.dtype), Some(args.dtype), || {
            with_parameter_init_mode(ParameterInitMode::Placeholder, || {
                LlamaModel::new(config.clone())
            })
        })
    });
    ModelLoader::load_llama_weights_with_options(weights, &model.named_parameters(), load_options)
        .expect("real-model inference path weight load failed");
    model
}

fn run_real_inference_path_check(args: &Args) {
    let Some(weights) = args.weights.as_ref() else {
        println!("path.infer.real            skipped missing --weights/--tokenizer");
        return;
    };
    let Some(tokenizer_path) = args.tokenizer.as_ref() else {
        println!("path.infer.real            skipped missing --weights/--tokenizer");
        return;
    };
    if !Path::new(weights).exists() {
        panic!("path.infer.real weights file does not exist: {weights}");
    }
    if !Path::new(tokenizer_path).exists() {
        panic!("path.infer.real tokenizer file does not exist: {tokenizer_path}");
    }
    if args.path_device == Device::Cuda && !lumen::ops::cuda::is_available() {
        println!("path.infer.real            skipped path_device=cuda cuda_available=false");
        return;
    }
    if args.dtype == DType::I8 {
        println!(
            "path.infer.real            skipped dtype=I8 note=this unified path uses one dtype for parameters/runtime/KV cache, and real Llama runtime currently requires floating dtypes"
        );
        return;
    }

    let tokenizer = LlamaTokenizer::from_file(tokenizer_path)
        .expect("real-model inference path tokenizer load failed");
    let config = real_llama_config(args.max_seq_len);
    assert_eq!(
        tokenizer.vocab_size(),
        config.vocab_size,
        "tokenizer/model vocab mismatch"
    );
    let prompt = build_first_turn_prompt(&args.system, &args.prompt);
    let prompt_tokens = tokenizer
        .encode(&prompt, false)
        .expect("real-model inference path tokenization failed");
    assert!(
        prompt_tokens.len() + args.max_gen + 2 < config.max_seq_len,
        "path.infer.real prompt_tokens={} max_gen={} exceed max_seq_len={}",
        prompt_tokens.len(),
        args.max_gen,
        config.max_seq_len
    );

    let _cuda_enabled_guard =
        (args.path_device == Device::Cuda).then(|| lumen::ops::cuda::set_enabled_scoped(true));
    let _strict_guard =
        (args.path_device == Device::Cuda).then(|| set_strict_device_execution_scoped(true));

    let load_start = Instant::now();
    let model = load_real_llama_model(args, &config);
    let load_elapsed = load_start.elapsed();
    let move_start = Instant::now();
    if args.path_device == Device::Cuda {
        model.to_cuda();
        lumen::ops::cuda::synchronize().expect("CUDA sync after real-model move failed");
    }
    let move_elapsed = move_start.elapsed();

    let stop_ids = generated_stop_ids(&tokenizer);
    let mut caches = model.init_kv_caches(1);
    model.reset_kv_caches(&mut caches);
    let mut generated = Vec::new();

    if args.path_device == Device::Cuda {
        lumen::ops::cuda::synchronize().expect("CUDA sync before real-model inference failed");
    }
    let infer_start = Instant::now();
    no_grad(|| {
        let prefill = token_ids_tensor(&prompt_tokens, args.path_device);
        let logits = model.forward_last_logits(prefill, &mut caches, 0);
        let mut next = argmax(&last_step_logits_vec(&logits));

        for _ in 0..args.max_gen {
            if stop_ids.contains(&next) {
                break;
            }
            generated.push(next);
            let decode = token_ids_tensor(&[next], args.path_device);
            let logits = model.forward_last_logits(decode, &mut caches, 0);
            next = argmax(&last_step_logits_vec(&logits));
            if caches[0].borrow().len + 2 >= config.max_seq_len {
                break;
            }
        }
    });
    if args.path_device == Device::Cuda {
        lumen::ops::cuda::synchronize().expect("CUDA sync after real-model inference failed");
    }
    let infer_elapsed = infer_start.elapsed();

    let text = tokenizer.decode(&generated, true);
    assert_inference_text_quality(&text);
    let tokens_per_second = if infer_elapsed.is_zero() {
        0.0
    } else {
        generated.len() as f64 / infer_elapsed.as_secs_f64()
    };
    println!(
        "path.infer.real            ok device={:?} prompt_tokens={} generated_tokens={} load_ms={:.3} move_ms={:.3} infer_ms={:.3} tok_s={:.2} chars={} replacement={} control={} repeat_run={}",
        args.path_device,
        prompt_tokens.len(),
        generated.len(),
        load_elapsed.as_secs_f64() * 1e3,
        move_elapsed.as_secs_f64() * 1e3,
        infer_elapsed.as_secs_f64() * 1e3,
        tokens_per_second,
        text.chars().count(),
        text.matches('\u{FFFD}').count(),
        text.chars()
            .filter(|ch| ch.is_control() && !ch.is_whitespace())
            .count(),
        max_repeated_char_run(&text)
    );
    if args.show_output {
        println!("path.infer.real.output:\n{}", text.trim());
    }
}

fn max_repeated_char_run(text: &str) -> usize {
    let mut prev = None;
    let mut cur = 0usize;
    let mut best = 0usize;
    for ch in text.chars() {
        if Some(ch) == prev {
            cur += 1;
        } else {
            prev = Some(ch);
            cur = 1;
        }
        best = best.max(cur);
    }
    best
}

fn assert_inference_text_quality(text: &str) {
    let chars = text.chars().collect::<Vec<_>>();
    let visible = chars.iter().filter(|ch| !ch.is_whitespace()).count();
    let replacement = chars.iter().filter(|&&ch| ch == '\u{FFFD}').count();
    let trailing_replacement = chars
        .iter()
        .rev()
        .take_while(|&&ch| ch == '\u{FFFD}')
        .count();
    let body_replacement = replacement.saturating_sub(trailing_replacement.min(4));
    let control = chars
        .iter()
        .filter(|&&ch| ch.is_control() && !ch.is_whitespace())
        .count();
    let bad_ratio = if chars.is_empty() {
        1.0
    } else {
        (body_replacement + control) as f32 / chars.len() as f32
    };

    assert!(
        visible >= 4,
        "inference path generated too little visible text: {text:?}"
    );
    assert!(
        body_replacement <= 1 && bad_ratio <= 0.05,
        "inference path generated likely garbled text: replacement={replacement} trailing_replacement={trailing_replacement} control={control} chars={} text={text:?}",
        chars.len()
    );
    assert!(
        max_repeated_char_run(text) <= 24,
        "inference path generated an excessive repeated-character run: {text:?}"
    );
}

pub(super) fn run_path_checks(args: &Args) {
    if !args.check {
        return;
    }
    let should_run_path = |name: &str| {
        args.case_filter
            .as_ref()
            .is_none_or(|filter| name.contains(filter))
    };
    if matches!(args.suite, Suite::All | Suite::Backward | Suite::Path)
        && should_run_path("path.train")
    {
        run_training_path_check(args);
    }
    if matches!(args.suite, Suite::All | Suite::Nn | Suite::Path)
        && should_run_path("path.infer.real")
    {
        run_real_inference_path_check(args);
    }
}
