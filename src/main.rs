use ndarray::{Array, Array1};
use ndarray_rand::RandomExt;
use rand_distr::Uniform;

use lumen::autograd::{no_grad, Tensor};
use lumen::loader::ModelLoader;
use lumen::models::{LlamaConfig, LlamaModel};
use lumen::tokenizer::LlamaTokenizer;

use std::io::{self, Write};
use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

// 构建 Prompt（TinyLlama chat template）
fn build_tinyllama_chat_prompt(system: &str, user: &str) -> String {
    format!(
        "<|system|>\n{}\n</s>\n<|user|>\n{}\n</s>\n<|assistant|>\n",
        system, user
    )
}

// 回退到合法 UTF-8 边界（用于流式输出）
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

// 采样函数：避免使用 rng.gen（在 Rust 2024 里 gen 是保留字）
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
    // 1) Repetition Penalty
    let mut adjusted = logits.to_vec();
    if repetition_penalty > 1.0 {
        for &t in recent_tokens {
            if t < adjusted.len() {
                adjusted[t] /= repetition_penalty;
            }
        }
    }

    // 2) Temperature
    let temp = temperature.max(1e-5);
    for v in adjusted.iter_mut() {
        *v /= temp;
    }

    // 3) Softmax
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

    // 4) Top-P
    let mut idxs: Vec<usize> = (0..probs.len()).collect();
    idxs.sort_by(|&i, &j| probs[j].partial_cmp(&probs[i]).unwrap());

    let mut cumulative = 0.0f32;
    let mut cut = 0usize;
    for (rank, &i) in idxs.iter().enumerate() {
        cumulative += probs[i];
        cut = rank + 1;
        if cumulative >= top_p {
            break;
        }
    }
    idxs.truncate(cut.max(1));

    // 5) Sample
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 初始化配置（TinyLlama-1.1B，需与你的权重一致）
    let config = LlamaConfig {
        vocab_size: 32000,
        hidden_size: 2048,
        intermediate_size: 5632,
        num_hidden_layers: 22,
        num_attention_heads: 32,
        rms_norm_eps: 1e-5,
        max_seq_len: 2048,
        num_key_value_heads: 4, // GQA
        rope_theta: 10000.0,
    };

    println!("🦀 Loading Rusty Llama...");
    let model = LlamaModel::new(config.clone());

    // TODO: 改成你本机路径
    let tokenizer_path = r"C:\Users\chen-\Downloads\tokenizer.json";
    let weight_path = r"C:\Users\chen-\Downloads\model.safetensors";

    let tokenizer = LlamaTokenizer::from_file(tokenizer_path)?;

    if std::path::Path::new(weight_path).exists() {
        println!("📦 Loading weights from: {}", weight_path);
        ModelLoader::load_llama_weights(weight_path, &model.named_parameters())?;
    } else {
        println!("⚠️ Weights not found at {}, output will be random noise.", weight_path);
    }

    println!("\n✨ System Ready. Type 'exit' to quit.");

    // Stop tokens
    let stop_ids: Vec<usize> = [
        tokenizer.token_to_id("</s>"),
        tokenizer.token_to_id("<|system|>"),
        tokenizer.token_to_id("<|user|>"),
        tokenizer.token_to_id("<|assistant|>"),
    ]
    .into_iter()
    .flatten()
    .collect();

    loop {
        print!("\n👤 User: ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let user_msg = input.trim();

        if user_msg == "exit" || user_msg == "quit" {
            break;
        }
        if user_msg.is_empty() {
            continue;
        }

        print!("🤖 Assistant: ");
        io::stdout().flush()?;

        no_grad(|| {
            // 1) 构建 Prompt & Tokenize
            let system = "You are a helpful AI assistant.";
            let chat_prompt = build_tinyllama_chat_prompt(system, user_msg);
            let mut tokens = tokenizer.encode(&chat_prompt, false);
            let prompt_len = tokens.len();

            // 2) 初始化 KV Cache（✅ 新 llama.rs API）
            //   - 不再使用 LlamaKVCache
            //   - 由 model 统一初始化，并在每轮对话前 reset
            let mut kv_caches = model.init_kv_caches(1);
            model.reset_kv_caches(&mut kv_caches);

            let max_gen = 200;
            let mut prev_gen_text = String::new();

            // 采样参数
            let temperature = 0.8;
            let top_p = 0.9;
            let repetition_penalty = 1.05;
            let recent_window = 64usize;

            // 位置指针 pos（新实现里 pos 被忽略，但保留以兼容旧代码）
            let mut pos = 0usize;

            for _ in 0..max_gen {
                // 3) Prefill vs Decoding
                let input_ids: Vec<usize> = if pos == 0 {
                    tokens.clone()
                } else {
                    vec![*tokens.last().unwrap()]
                };

                let input_tensor = Tensor::from_array_no_grad(
                    Array::from_shape_vec(
                        (1, input_ids.len()),
                        input_ids.iter().map(|&x| x as f32).collect(),
                    )
                    .unwrap()
                    .into_dyn(),
                );

                // 4) Forward (传递 &mut caches 和 pos)
                // logits: [Batch=1, Seq, Vocab]
                let logits = model.forward(input_tensor, &mut kv_caches, pos);

                // 更新 pos（虽然新实现忽略 pos，但留着无害）
                pos += input_ids.len();

                // 5) 取最后一步 logits + 采样
                // 这里保持你原来的写法（不会触发 E0716）
                let logits_ref = logits.data_ref();
                let last_step_logits = logits_ref.slice(ndarray::s![0, -1, ..]);
                let logits_vec: Vec<f32> = last_step_logits.iter().copied().collect();

                let start = tokens.len().saturating_sub(recent_window);
                let recent = &tokens[start..];

                let next_token = sample_top_p(
                    &logits_vec,
                    temperature,
                    top_p,
                    repetition_penalty,
                    recent,
                );

                if stop_ids.contains(&next_token) {
                    break;
                }

                tokens.push(next_token);

                // 6) 流式输出
                let gen_tokens = &tokens[prompt_len..];
                let cur_gen_text = tokenizer.decode(gen_tokens, true);

                if cur_gen_text.contains("<|user|>") || cur_gen_text.contains("<|assistant|>") {
                    break;
                }
                print_new_suffix(&mut prev_gen_text, cur_gen_text);
            }

            println!();
        });
    }

    Ok(())
}
