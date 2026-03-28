use mimalloc::MiMalloc;

use lumen::autograd::{no_grad, Tensor};
use lumen::infer_mode::set_pure_bf16_infer_loading;
use lumen::loader::ModelLoader;
use lumen::models::{LlamaConfig, LlamaModel};
use lumen::module::Module;
use lumen::tokenizer::LlamaTokenizer;

use ndarray::{s, Array, Array1, Ix3};
use ndarray_rand::RandomExt;
use rand_distr::Uniform;

use std::env;
use std::io::{self, Write};
use std::path::Path;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn model_config() -> LlamaConfig {
    LlamaConfig {
        vocab_size: 32000,
        hidden_size: 2048,
        intermediate_size: 5632,
        num_hidden_layers: 22,
        num_attention_heads: 32,
        num_key_value_heads: 4,
        rms_norm_eps: 1e-5,
        max_seq_len: 2048,
        rope_theta: 10000.0,
    }
}

#[derive(Debug, Clone)]
struct Args {
    weights: String,
    tokenizer: String,
    system: String,
    temperature: f32,
    top_p: f32,
    repetition_penalty: f32,
    recent_window: usize,
    max_gen: usize,
}

fn usage(program: &str) {
    eprintln!(
        "Usage:\n  {program} --weights PATH --tokenizer PATH [options]\n\nOptions:\n  --system TEXT              System prompt\n  --temperature FLOAT        Sampling temperature (default: 0.8)\n  --top-p FLOAT              Top-p nucleus sampling (default: 0.9)\n  --repetition-penalty FLOAT Repetition penalty (default: 1.05)\n  --recent-window N          Recent token window for repetition penalty (default: 96)\n  --max-gen N                Max generated tokens per turn (default: 200)\n\nCommands in chat:\n  /reset   Clear history and KV cache\n  /exit    Quit"
    );
}

fn parse_args() -> Result<Args, String> {
    let argv: Vec<String> = env::args().collect();
    let program = argv.first().cloned().unwrap_or_else(|| "lumen".to_string());

    if argv.len() == 1 {
        usage(&program);
        return Err("缺少参数".to_string());
    }

    let mut weights: Option<String> = None;
    let mut tokenizer: Option<String> = None;
    let mut system = "You are a helpful AI assistant.".to_string();
    let mut temperature = 0.8f32;
    let mut top_p = 0.9f32;
    let mut repetition_penalty = 1.05f32;
    let mut recent_window = 96usize;
    let mut max_gen = 200usize;

    let mut i = 1usize;
    while i < argv.len() {
        match argv[i].as_str() {
            "-h" | "--help" => {
                usage(&program);
                std::process::exit(0);
            }
            "--weights" => {
                i += 1;
                weights = Some(argv.get(i).ok_or("--weights 缺少路径")?.clone());
            }
            "--tokenizer" => {
                i += 1;
                tokenizer = Some(argv.get(i).ok_or("--tokenizer 缺少路径")?.clone());
            }
            "--system" => {
                i += 1;
                system = argv.get(i).ok_or("--system 缺少文本")?.clone();
            }
            "--temperature" => {
                i += 1;
                temperature = argv
                    .get(i)
                    .ok_or("--temperature 缺少数字")?
                    .parse::<f32>()
                    .map_err(|_| "--temperature 需要 f32")?;
            }
            "--top-p" => {
                i += 1;
                top_p = argv
                    .get(i)
                    .ok_or("--top-p 缺少数字")?
                    .parse::<f32>()
                    .map_err(|_| "--top-p 需要 f32")?;
            }
            "--repetition-penalty" => {
                i += 1;
                repetition_penalty = argv
                    .get(i)
                    .ok_or("--repetition-penalty 缺少数字")?
                    .parse::<f32>()
                    .map_err(|_| "--repetition-penalty 需要 f32")?;
            }
            "--recent-window" => {
                i += 1;
                recent_window = argv
                    .get(i)
                    .ok_or("--recent-window 缺少数字")?
                    .parse::<usize>()
                    .map_err(|_| "--recent-window 需要 usize")?;
            }
            "--max-gen" => {
                i += 1;
                max_gen = argv
                    .get(i)
                    .ok_or("--max-gen 缺少数字")?
                    .parse::<usize>()
                    .map_err(|_| "--max-gen 需要 usize")?;
            }
            other => return Err(format!("未知参数: {other}")),
        }
        i += 1;
    }

    if !(0.0..=1.0).contains(&top_p) {
        return Err("--top-p 必须在 [0, 1] 范围内".to_string());
    }
    if temperature < 0.0 {
        return Err("--temperature 不能小于 0".to_string());
    }
    if repetition_penalty < 1.0 {
        return Err("--repetition-penalty 不能小于 1.0".to_string());
    }

    Ok(Args {
        weights: weights.ok_or("必须提供 --weights")?,
        tokenizer: tokenizer.ok_or("必须提供 --tokenizer")?,
        system,
        temperature,
        top_p,
        repetition_penalty,
        recent_window,
        max_gen,
    })
}

fn build_first_turn_prompt(system: &str, user: &str) -> String {
    format!(
        "<|system|>\n{}\n</s>\n<|user|>\n{}\n</s>\n<|assistant|>\n",
        system, user
    )
}

fn build_next_turn_prompt(user: &str) -> String {
    format!("</s>\n<|user|>\n{}\n</s>\n<|assistant|>\n", user)
}

fn lcp_char_boundary(prev: &str, cur: &str) -> usize {
    let pb = prev.as_bytes();
    let cb = cur.as_bytes();
    let mut i = 0usize;
    let n = pb.len().min(cb.len());
    while i < n && pb[i] == cb[i] {
        i += 1;
    }
    while i > 0 && !cur.is_char_boundary(i) {
        i -= 1;
    }
    i
}

fn print_new_suffix(prev_printed: &mut String, cur_full: String) {
    if cur_full.contains('\u{FFFD}') {
        return;
    }
    let cut = lcp_char_boundary(prev_printed, &cur_full);
    if cut < cur_full.len() {
        print!("{}", &cur_full[cut..]);
        let _ = io::stdout().flush();
    }
    *prev_printed = cur_full;
}

#[inline]
fn rand01() -> f32 {
    Array1::<f32>::random(1, Uniform::new(0.0f32, 1.0f32))[0]
}

fn sample_top_p(
    logits: &[f32],
    temperature: f32,
    top_p: f32,
    repetition_penalty: f32,
    recent_tokens: &[usize],
) -> usize {
    let mut adjusted = logits.to_vec();

    if repetition_penalty > 1.0 {
        for &t in recent_tokens {
            if t < adjusted.len() {
                adjusted[t] /= repetition_penalty;
            }
        }
    }

    if temperature <= 1e-5 {
        let mut best_i = 0usize;
        let mut best_v = f32::NEG_INFINITY;
        for (i, &v) in adjusted.iter().enumerate() {
            if v > best_v {
                best_v = v;
                best_i = i;
            }
        }
        return best_i;
    }

    for v in adjusted.iter_mut() {
        *v /= temperature;
    }

    let mut maxv = f32::NEG_INFINITY;
    for &v in &adjusted {
        if v > maxv {
            maxv = v;
        }
    }

    let mut probs: Vec<f32> = adjusted.iter().map(|x| (x - maxv).exp()).collect();
    let sum: f32 = probs.iter().sum();
    let inv = 1.0 / (sum + 1e-9);
    for p in probs.iter_mut() {
        *p *= inv;
    }

    let mut idxs: Vec<usize> = (0..probs.len()).collect();
    idxs.sort_by(|&i, &j| probs[j].partial_cmp(&probs[i]).unwrap());

    let mut cumulative = 0.0f32;
    let mut cut = 0usize;
    let target_p = top_p.clamp(0.0, 1.0).max(1e-6);
    for (rank, &i) in idxs.iter().enumerate() {
        cumulative += probs[i];
        cut = rank + 1;
        if cumulative >= target_p {
            break;
        }
    }
    idxs.truncate(cut.max(1));

    let r = rand01();
    let mut acc = 0.0f32;
    for &i in &idxs {
        acc += probs[i] / cumulative;
        if r <= acc {
            return i;
        }
    }
    *idxs.last().unwrap()
}

fn tensor_from_token_ids(ids: &[usize]) -> Tensor {
    Tensor::from_array_no_grad(
        Array::from_shape_vec((1, ids.len()), ids.iter().map(|&x| x as f32).collect())
            .unwrap()
            .into_dyn(),
    )
}

fn last_step_logits_vec(logits: &Tensor) -> Vec<f32> {
    let logits_ref = logits.data_ref();
    let l3 = logits_ref
        .view()
        .into_dimensionality::<Ix3>()
        .expect("logits must be 3D [B,S,V]");
    let t = l3.shape()[1] - 1;
    l3.slice(s![0, t, ..]).iter().copied().collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args().map_err(|e| format!("参数错误: {e}"))?;
    let config = model_config();

    if !Path::new(&args.tokenizer).exists() {
        return Err(format!("tokenizer 文件不存在: {}", args.tokenizer).into());
    }
    if !Path::new(&args.weights).exists() {
        return Err(format!("weights 文件不存在: {}", args.weights).into());
    }

    println!("🦀 Loading Rusty Llama...");
    println!("⚙️  Pure BF16 infer loading: ON");

    let tokenizer = LlamaTokenizer::from_file(&args.tokenizer)?;
    if tokenizer.vocab_size() != config.vocab_size {
        return Err(format!(
            "tokenizer vocab_size={} 与 model config vocab_size={} 不一致",
            tokenizer.vocab_size(),
            config.vocab_size
        )
        .into());
    }

    set_pure_bf16_infer_loading(true);

    let mut model = LlamaModel::new(config.clone());

    println!("📦 Loading weights from: {}", args.weights);
    ModelLoader::load_llama_weights_direct(&args.weights, &mut model)?;

    model.eval_mode();

    println!("🚀 Direct BF16 infer weights loaded.");
    println!("\n✨ System Ready. Commands: /reset  /exit");

    let mut stop_ids: Vec<usize> = Vec::new();
    for t in ["</s>", "<|system|>", "<|user|>", "<|assistant|>"] {
        if let Some(id) = tokenizer.token_to_id(t) {
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

    let mut all_tokens: Vec<usize> = Vec::new();
    let mut first_turn = true;

    let mut kv_caches = model.init_kv_caches(1);
    model.reset_kv_caches(&mut kv_caches);

    loop {
        print!("\n👤 User: ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let user_msg = input.trim();

        if user_msg.is_empty() {
            continue;
        }
        if user_msg == "/exit" || user_msg == "exit" || user_msg == "quit" {
            break;
        }
        if user_msg == "/reset" || user_msg == "reset" {
            all_tokens.clear();
            model.reset_kv_caches(&mut kv_caches);
            first_turn = true;
            println!("🔄 reset done.");
            continue;
        }

        print!("🤖 Assistant: ");
        io::stdout().flush()?;

        no_grad(|| {
            let turn_prompt = if first_turn {
                build_first_turn_prompt(&args.system, user_msg)
            } else {
                build_next_turn_prompt(user_msg)
            };

            let mut new_tokens = tokenizer.encode(&turn_prompt, false);

            if new_tokens.is_empty() {
                println!();
                first_turn = false;
                return;
            }

            let cur_len = kv_caches[0].borrow().len;
            if cur_len + new_tokens.len() + args.max_gen + 8 >= config.max_seq_len {
                all_tokens.clear();
                model.reset_kv_caches(&mut kv_caches);
                first_turn = true;

                let prompt2 = build_first_turn_prompt(&args.system, user_msg);
                new_tokens = tokenizer.encode(&prompt2, false);
                if new_tokens.is_empty() {
                    println!();
                    first_turn = false;
                    return;
                }
            }

            all_tokens.extend_from_slice(&new_tokens);
            let assistant_start = all_tokens.len();

            let prefill_logits =
                model.forward_last_logits(tensor_from_token_ids(&new_tokens), &mut kv_caches, 0);
            let mut logits_vec = last_step_logits_vec(&prefill_logits);

            let mut prev_gen_text = String::new();

            for _ in 0..args.max_gen {
                let start = all_tokens.len().saturating_sub(args.recent_window);
                let recent = &all_tokens[start..];

                let next_token = sample_top_p(
                    &logits_vec,
                    args.temperature,
                    args.top_p,
                    args.repetition_penalty,
                    recent,
                );

                if stop_ids.contains(&next_token) {
                    break;
                }

                all_tokens.push(next_token);

                let gen_tokens = &all_tokens[assistant_start..];
                let cur_gen_text = tokenizer.decode(gen_tokens, true);
                if cur_gen_text.contains("<|user|>") || cur_gen_text.contains("<|assistant|>") {
                    break;
                }
                print_new_suffix(&mut prev_gen_text, cur_gen_text);

                if args.temperature <= 1e-5 && args.repetition_penalty <= 1.0 {
                    let next = model.forward_last_argmax(
                        tensor_from_token_ids(&[next_token]),
                        &mut kv_caches,
                        0,
                    );
                    logits_vec.fill(f32::NEG_INFINITY);
                    if next < logits_vec.len() {
                        logits_vec[next] = 0.0;
                    }
                } else {
                    let logits2 = model.forward_last_logits(
                        tensor_from_token_ids(&[next_token]),
                        &mut kv_caches,
                        0,
                    );
                    logits_vec = last_step_logits_vec(&logits2);
                }

                if kv_caches[0].borrow().len + 2 >= config.max_seq_len {
                    break;
                }
            }

            println!();
            first_turn = false;
        });
    }

    Ok(())
}
