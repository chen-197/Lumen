use mimalloc::MiMalloc;

use lumen::autograd::{no_grad, Tensor};
use lumen::loader::ModelLoader;
use lumen::models::{LlamaConfig, LlamaModel};
use lumen::tokenizer::LlamaTokenizer;

use ndarray::{s, Array, Array1, Ix3};
use ndarray_rand::RandomExt;
use rand_distr::Uniform;

use std::io::{self, Write};

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

// ------------------------------
// Prompt template (TinyLlama-style)
// ------------------------------

fn build_first_turn_prompt(system: &str, user: &str) -> String {
    format!(
        "<|system|>\n{}\n</s>\n<|user|>\n{}\n</s>\n<|assistant|>\n",
        system, user
    )
}

fn build_next_turn_prompt(user: &str) -> String {
    // Previous assistant reply is expected to end at generation stop.
    // We explicitly add a </s> separator before the next user turn.
    format!("</s>\n<|user|>\n{}\n</s>\n<|assistant|>\n", user)
}

// ------------------------------
// Streaming-print helpers (UTF-8 safe)
// ------------------------------

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

// ------------------------------
// Sampling
// ------------------------------

// Avoid rng.gen (Rust 2024 reserved keyword)
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
    let target_p = top_p.clamp(0.0, 1.0).max(1e-6);
    for (rank, &i) in idxs.iter().enumerate() {
        cumulative += probs[i];
        cut = rank + 1;
        if cumulative >= target_p {
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

// ------------------------------
// Tensor helpers
// ------------------------------

fn tensor_from_token_ids(ids: &[usize]) -> Tensor {
    Tensor::from_array_no_grad(
        Array::from_shape_vec(
            (1, ids.len()),
            ids.iter().map(|&x| x as f32).collect(),
        )
        .unwrap()
        .into_dyn(),
    )
}

// logits: [1, S, V] -> take last step [V]
fn last_step_logits_vec(logits: &Tensor) -> Vec<f32> {
    let logits_ref = logits.data_ref();
    let l3 = logits_ref
        .view()
        .into_dimensionality::<Ix3>()
        .expect("logits must be 3D [B,S,V]");
    let t = l3.shape()[1] - 1;
    l3.slice(s![0, t, ..]).iter().copied().collect()
}

// ------------------------------
// Main (chat with history memory)
// ------------------------------

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // NOTE: Ensure this config matches your weights.
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

    println!("\n✨ System Ready. Commands: /reset  /exit");

    // Stop tokens
    let mut stop_ids: Vec<usize> = Vec::new();
    for t in ["</s>", "<|system|>", "<|user|>", "<|assistant|>"] {
        if let Some(id) = tokenizer.token_to_id(t) {
            stop_ids.push(id);
        }
    }
    if let Some(id) = tokenizer.eos_id() {
        stop_ids.push(id);
    }
    stop_ids.sort_unstable();
    stop_ids.dedup();

    // ✅ persistent conversation state (history memory)
    let system = "You are a helpful AI assistant.";
    let mut all_tokens: Vec<usize> = Vec::new(); // full history tokens
    let mut first_turn = true;

    // ✅ new llama.rs API: init caches once; keep them across turns
    let mut kv_caches = model.init_kv_caches(1);
    model.reset_kv_caches(&mut kv_caches);

    // Sampling params
    let temperature = 0.8;
    let top_p = 0.9;
    let repetition_penalty = 1.05;
    let recent_window = 96usize;
    let max_gen = 200usize;

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
            // 1) Build this turn prompt (only the *incremental* chunk)
            let turn_prompt = if first_turn {
                build_first_turn_prompt(system, user_msg)
            } else {
                build_next_turn_prompt(user_msg)
            };

            let new_tokens = tokenizer.encode(&turn_prompt, false);

            // context safety: if overflow is near, auto reset
            let cur_len = kv_caches[0].borrow().len;
            if cur_len + new_tokens.len() + 8 >= config.max_seq_len {
                all_tokens.clear();
                model.reset_kv_caches(&mut kv_caches);
                first_turn = true;

                // rebuild as first turn after reset
                let prompt2 = build_first_turn_prompt(system, user_msg);
                let new_tokens2 = tokenizer.encode(&prompt2, false);
                all_tokens.extend_from_slice(&new_tokens2);

                // prefill
                let logits = model.forward(
                    tensor_from_token_ids(&new_tokens2),
                    &mut kv_caches,
                    0,
                );
                let mut logits_vec = last_step_logits_vec(&logits);

                let assistant_start = all_tokens.len();
                let mut prev_gen_text = String::new();

                for _ in 0..max_gen {
                    let start = all_tokens.len().saturating_sub(recent_window);
                    let recent = &all_tokens[start..];

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

                    all_tokens.push(next_token);

                    let gen_tokens = &all_tokens[assistant_start..];
                    let cur_gen_text = tokenizer.decode(gen_tokens, true);

                    if cur_gen_text.contains("<|user|>")
                        || cur_gen_text.contains("<|assistant|>")
                    {
                        break;
                    }
                    print_new_suffix(&mut prev_gen_text, cur_gen_text);

                    let logits2 = model.forward(
                        tensor_from_token_ids(&[next_token]),
                        &mut kv_caches,
                        0,
                    );
                    logits_vec = last_step_logits_vec(&logits2);
                }

                println!();
                first_turn = false;
                return;
            }

            // 2) Append to global history tokens
            all_tokens.extend_from_slice(&new_tokens);
            let assistant_start = all_tokens.len();

            // 3) Prefill only incremental tokens into KV cache
            let logits =
                model.forward(tensor_from_token_ids(&new_tokens), &mut kv_caches, 0);
            let mut logits_vec = last_step_logits_vec(&logits);

            // 4) Generate
            let mut prev_gen_text = String::new();
            for _ in 0..max_gen {
                let start = all_tokens.len().saturating_sub(recent_window);
                let recent = &all_tokens[start..];

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

                all_tokens.push(next_token);

                let gen_tokens = &all_tokens[assistant_start..];
                let cur_gen_text = tokenizer.decode(gen_tokens, true);

                if cur_gen_text.contains("<|user|>") || cur_gen_text.contains("<|assistant|>") {
                    break;
                }
                print_new_suffix(&mut prev_gen_text, cur_gen_text);

                // decode step: feed only last token
                let logits2 = model.forward(
                    tensor_from_token_ids(&[next_token]),
                    &mut kv_caches,
                    0,
                );
                logits_vec = last_step_logits_vec(&logits2);

                // Hard stop if we hit max length
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
