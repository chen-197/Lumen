use crate::autograd::Tensor;
use crate::layers::{Embedding, KVCache, Linear, RMSNorm, SelfAttention, SiLU};
use crate::layers::attention::self_attention::KVCacheInner;
use crate::module::Module;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

// Llama 配置参数
#[derive(Clone, Debug)]
pub struct LlamaConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize, // 支持 GQA
    pub rms_norm_eps: f32,
    pub max_seq_len: usize,
    pub rope_theta: f32,
}

impl Default for LlamaConfig {
    fn default() -> Self {
        Self {
            vocab_size: 32000,
            hidden_size: 2048,
            intermediate_size: 5632,
            num_hidden_layers: 22,
            num_attention_heads: 32,
            num_key_value_heads: 4, // TinyLlama 1.1B 其实是 32 (MHA)，但 Qwen 是 GQA
            rms_norm_eps: 1e-5,
            max_seq_len: 2048,
            rope_theta: 10000.0,
        }
    }
}

// Llama MLP 层 (SwiGLU)
// 公式: down(act(gate(x)) * up(x))
struct LlamaMLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act: SiLU,
}

impl LlamaMLP {
    fn new(config: &LlamaConfig) -> Self {
        Self {
            // Llama 官方通常没有 bias，使用 new_no_bias
            gate_proj: Linear::new_no_bias(config.hidden_size, config.intermediate_size),
            up_proj: Linear::new_no_bias(config.hidden_size, config.intermediate_size),
            down_proj: Linear::new_no_bias(config.intermediate_size, config.hidden_size),
            act: SiLU::new(),
        }
    }

    fn forward(&self, x: Tensor) -> Tensor {
        let gate = self.gate_proj.forward(x.clone());
        let gate_act = self.act.forward(gate);
        let up = self.up_proj.forward(x);
        let fused = gate_act * up;
        self.down_proj.forward(fused)
    }
}

// NOTE:
// llama.rs 之前自带了一套 LlamaAttention（repeat_kv + 显式 score/prob Tensor + KVCache::get_view 分配）。
// 这里直接复用 self_attention.rs 的实现：
// - eval/no_grad: 支持 KV cache 预分配、decode(S=1) online-softmax 热路径、GQA 不 repeat_kv。
// - train: 走可导的标准路径（fused_softmax + batch_matmul）。
//
// 因此：
// - LlamaDecoderLayer::self_attn 改为 SelfAttention
// - Cache 类型改为 layers::KVCache（Rc<RefCell<KVCacheInner>>）

// Llama Decoder Block
struct LlamaDecoderLayer {
    self_attn: SelfAttention,
    mlp: LlamaMLP,
    input_layernorm: RMSNorm,
    post_attention_layernorm: RMSNorm,
}

impl LlamaDecoderLayer {
    fn new(config: &LlamaConfig) -> Self {
        Self {
            self_attn: SelfAttention::new(
                config.hidden_size,
                config.num_attention_heads,
                config.num_key_value_heads,
                config.max_seq_len,
                config.rope_theta,
                true, // causal
            ),
            mlp: LlamaMLP::new(config),
            input_layernorm: RMSNorm::new(config.hidden_size, config.rms_norm_eps),
            post_attention_layernorm: RMSNorm::new(config.hidden_size, config.rms_norm_eps),
        }
    }

    // 推理路径：传入 cache（Rc<RefCell<_>>），用于增量 decode。
    fn forward_infer(&self, x: Tensor, cache: KVCache) -> (Tensor, KVCache) {
        // Pre-Norm Architecture
        // h = x + Attention(Norm(x))
        let norm_x = self.input_layernorm.forward(x.clone());
        let (attn_out, cache_out) = self.self_attn.forward(norm_x, Some(cache));
        let cache_out = cache_out.expect("SelfAttention should return cache in eval/no_grad path");
        let h = x + attn_out;

        // out = h + MLP(Norm(h))
        let norm_h = self.post_attention_layernorm.forward(h.clone());
        let mlp_out = self.mlp.forward(norm_h);
        (h + mlp_out, cache_out)
    }

    // 训练路径：不允许传 cache（SelfAttention 会 panic）。
    fn forward_train(&self, x: Tensor) -> Tensor {
        let norm_x = self.input_layernorm.forward(x.clone());
        let (attn_out, _cache) = self.self_attn.forward(norm_x, None);
        let h = x + attn_out;
        let norm_h = self.post_attention_layernorm.forward(h.clone());
        let mlp_out = self.mlp.forward(norm_h);
        h + mlp_out
    }
}

pub struct LlamaModel {
    embed_tokens: Embedding,
    layers: Vec<LlamaDecoderLayer>,
    norm: RMSNorm,
    lm_head: Linear,
    pub config: LlamaConfig,
}

impl LlamaModel {
    pub fn new(config: LlamaConfig) -> Self {
        let embed_tokens = Embedding::new(config.vocab_size, config.hidden_size);

        let mut layers = Vec::new();
        for _ in 0..config.num_hidden_layers {
            layers.push(LlamaDecoderLayer::new(&config));
        }

        let norm = RMSNorm::new(config.hidden_size, config.rms_norm_eps);
        let lm_head = Linear::new_no_bias(config.hidden_size, config.vocab_size);

        Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            config,
        }
    }

    /// 为推理/生成初始化每层 KV cache。
    ///
    /// SelfAttention 的 cache 内部会维护 `len`，所以推理阶段不再需要显式传 `pos`。
    pub fn init_kv_caches(&self, batch_size: usize) -> Vec<KVCache> {
        let head_dim = self.config.hidden_size / self.config.num_attention_heads;
        let h_kv = self.config.num_key_value_heads;
        let max_seq = self.config.max_seq_len;

        (0..self.config.num_hidden_layers)
            .map(|_| Rc::new(RefCell::new(KVCacheInner::new(batch_size, h_kv, max_seq, head_dim))))
            .collect()
    }

    /// 重置 cache（在新对话/新样本开始前调用）。
    pub fn reset_kv_caches(&self, caches: &mut [KVCache]) {
        for c in caches {
            c.borrow_mut().reset();
        }
    }

    /// 推理/生成（需要 caches）。
    ///
    /// `pos` 参数为了兼容旧调用方保留，但会被忽略：长度由 cache 内部维护。
    pub fn forward(&self, input_ids: Tensor, caches: &mut Vec<KVCache>, _pos: usize) -> Tensor {
        assert_eq!(
            caches.len(),
            self.layers.len(),
            "KV cache count mismatch: got {}, expected {}",
            caches.len(),
            self.layers.len()
        );

        // Embedding: [B,S] -> [B,S,H]
        let mut x = self.embed_tokens.forward(&input_ids);

        // Decoder Layers
        for (i, layer) in self.layers.iter().enumerate() {
            let cache_in = caches[i].clone();
            let (y, cache_out) = layer.forward_infer(x, cache_in);
            caches[i] = cache_out;
            x = y;
        }

        // Final Norm
        x = self.norm.forward(x);

        // LM Head
        self.lm_head.forward(x)
    }

    /// 训练（不使用 cache，支持 autograd）。
    pub fn forward_train(&self, input_ids: Tensor) -> Tensor {
        let mut x = self.embed_tokens.forward(&input_ids);
        for layer in self.layers.iter() {
            x = layer.forward_train(x);
        }
        x = self.norm.forward(x);
        self.lm_head.forward(x)
    }

    pub fn named_parameters(&self) -> HashMap<String, Tensor> {
        let mut params = HashMap::new();

        // Embedding
        params.insert(
            "model.embed_tokens.weight".to_string(),
            self.embed_tokens.weight.clone(),
        );

        // Layers
        for (i, layer) in self.layers.iter().enumerate() {
            let prefix = format!("model.layers.{}", i);

            // Self Attention
            params.insert(
                format!("{}.self_attn.q_proj.weight", prefix),
                layer.self_attn.w_q.weight.clone(),
            );
            params.insert(
                format!("{}.self_attn.k_proj.weight", prefix),
                layer.self_attn.w_k.weight.clone(),
            );
            params.insert(
                format!("{}.self_attn.v_proj.weight", prefix),
                layer.self_attn.w_v.weight.clone(),
            );
            params.insert(
                format!("{}.self_attn.o_proj.weight", prefix),
                layer.self_attn.w_o.weight.clone(),
            );

            // MLP
            params.insert(
                format!("{}.mlp.gate_proj.weight", prefix),
                layer.mlp.gate_proj.weight.clone(),
            );
            params.insert(
                format!("{}.mlp.up_proj.weight", prefix),
                layer.mlp.up_proj.weight.clone(),
            );
            params.insert(
                format!("{}.mlp.down_proj.weight", prefix),
                layer.mlp.down_proj.weight.clone(),
            );

            // Layernorms
            params.insert(
                format!("{}.input_layernorm.weight", prefix),
                layer.input_layernorm.weight.clone(),
            );
            params.insert(
                format!("{}.post_attention_layernorm.weight", prefix),
                layer.post_attention_layernorm.weight.clone(),
            );
        }

        // Final Norm & Head
        params.insert("model.norm.weight".to_string(), self.norm.weight.clone());
        params.insert("lm_head.weight".to_string(), self.lm_head.weight.clone());

        params
    }
}

impl Module for LlamaModel {
    fn forward(&self, input: Tensor) -> Tensor {
        // 训练/全序列前向：不使用 KV cache（可导）。
        self.forward_train(input)
    }

    fn parameters(&self) -> Vec<Tensor> {
        // 收集所有参数
        let mut params = vec![self.embed_tokens.weight.clone()];
        for layer in &self.layers {
            params.extend(layer.self_attn.w_q.parameters());
            params.extend(layer.self_attn.w_k.parameters());
            params.extend(layer.self_attn.w_v.parameters());
            params.extend(layer.self_attn.w_o.parameters());
            params.extend(layer.mlp.gate_proj.parameters());
            params.extend(layer.mlp.up_proj.parameters());
            params.extend(layer.mlp.down_proj.parameters());
            params.push(layer.input_layernorm.weight.clone());
            params.push(layer.post_attention_layernorm.weight.clone());
        }
        params.push(self.norm.weight.clone());
        params.push(self.lm_head.weight.clone());
        params
    }
}
