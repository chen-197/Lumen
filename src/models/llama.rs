use crate::autograd::Tensor;
use crate::kv_cache::LlamaKVCache;
use crate::layers::{Embedding, Linear, RMSNorm, RotaryEmbedding, SiLU};
use crate::module::Module;
use crate::ops::matmul::batch_matmul;
use std::collections::HashMap;

/// Llama 配置参数
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

/// Llama MLP 层 (SwiGLU)
/// 公式: down(act(gate(x)) * up(x))
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

/// Llama Attention 层 (集成 Static KV Cache)
struct LlamaAttention {
    w_q: Linear,
    w_k: Linear,
    w_v: Linear,
    w_o: Linear,
    rope: RotaryEmbedding,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    hidden_size: usize,
}

impl LlamaAttention {
    fn new(config: &LlamaConfig) -> Self {
        let head_dim = config.hidden_size / config.num_attention_heads;
        Self {
            w_q: Linear::new_no_bias(config.hidden_size, config.num_attention_heads * head_dim),
            w_k: Linear::new_no_bias(config.hidden_size, config.num_key_value_heads * head_dim),
            w_v: Linear::new_no_bias(config.hidden_size, config.num_key_value_heads * head_dim),
            w_o: Linear::new_no_bias(config.num_attention_heads * head_dim, config.hidden_size),
            rope: RotaryEmbedding::new(head_dim, config.max_seq_len, config.rope_theta),
            num_heads: config.num_attention_heads,
            num_kv_heads: config.num_key_value_heads,
            head_dim,
            hidden_size: config.hidden_size,
        }
    }

    /// Forward
    /// x: [Batch, Seq, Hidden]
    /// cache: 预分配的 Static Cache
    /// pos: 当前 x 在序列中的起始位置
    fn forward(&self, x: Tensor, cache: &mut LlamaKVCache, pos: usize) -> Tensor {
        let (b, seq_len, _) = {
            let d = x.data_ref();
            (d.shape()[0], d.shape()[1], d.shape()[2])
        };

        //QKV Proj
        let q = self.w_q.forward(x.clone());
        let k = self.w_k.forward(x.clone());
        let v = self.w_v.forward(x);

        // Reshape Heads
        // [B, Seq, H * D] -> [B, H, Seq, D]
        let q = self.reshape_heads(q, self.num_heads);
        let k = self.reshape_heads(k, self.num_kv_heads);
        let v = self.reshape_heads(v, self.num_kv_heads);

        // RoPE
        let q_rot = self.rope.forward(&q, pos);
        let k_rot = self.rope.forward(&k, pos);

        // KV Cache Update
        cache.update(&k_rot, &v, pos);

        // Retrieve KV (获取完整的 context)
        let total_len = pos + seq_len;
        let (k_all, v_all) = cache.get_view(total_len);

        // GQA Handling (Grouped Query Attention)
        // 如果 KV heads 少于 Q heads，需要重复 KV
        // [B, H_kv, T, D] -> [B, H_q, T, D]
        let k_all = self.repeat_kv(k_all);
        let v_all = self.repeat_kv(v_all);

        // Attention Scores: Q @ K^T
        // Q: [B, H, Seq, D]
        // K: [B, H, Total, D] -> K^T: [B, H, D, Total]
        // Scores: [B, H, Seq, Total]
        let mut scores = batch_matmul(&q_rot, &k_all.transpose(2, 3));

        // Scale
        let scale = (self.head_dim as f32).sqrt().recip();
        scores = scores * Tensor::from_array_no_grad(ndarray::arr0(scale).into_dyn());

        // Causal Masking
        // 只有在 prefill (seq_len > 1) 时才需要 mask
        // 在 decoding (seq_len == 1) 时，Q 只有一个 token，自然关注所有之前的 token
        if seq_len > 1 {
            // 创建一个下三角 Mask
            let mask = self.create_causal_mask(seq_len, total_len, pos);
            scores = scores + mask;
        }

        // Softmax
        // 对最后一个维度 (Total_Len) 做 Softmax
        let probs = crate::layers::activation::Softmax::new(3).forward(scores);

        // Output: Probs @ V
        // [B, H, Seq, Total] @ [B, H, Total, D] -> [B, H, Seq, D]
        let output = batch_matmul(&probs, &v_all);

        // Reshape back
        // [B, H, Seq, D] -> [B, Seq, H, D] -> [B, Seq, Hidden]
        let output = output.permute(vec![0, 2, 1, 3]).reshape(vec![
            b.try_into().unwrap(),
            seq_len.try_into().unwrap(),
            self.hidden_size.try_into().unwrap(),
        ]);

        // Out Proj
        self.w_o.forward(output)
    }

    fn reshape_heads(&self, x: Tensor, n_heads: usize) -> Tensor {
        let d = x.data_ref();
        let shape = d.shape();
        let b = shape[0];
        let seq = shape[1];
        let dim = self.head_dim;

        // [B, Seq, H * D] -> [B, Seq, H, D] -> [B, H, Seq, D]
        x.reshape(vec![
            b.try_into().unwrap(),
            seq.try_into().unwrap(),
            n_heads.try_into().unwrap(),
            dim.try_into().unwrap(),
        ])
        .permute(vec![0, 2, 1, 3])
    }

    fn repeat_kv(&self, x: Tensor) -> Tensor {
        let n_rep = self.num_heads / self.num_kv_heads;
        if n_rep == 1 {
            return x;
        }

        // GQA 展开逻辑
        // x: [B, H_kv, T, D]
        // Need: [B, H_q, T, D] where H_q = H_kv * n_rep

        let d = x.data_ref();
        let shape = d.shape();
        let (b, h_kv, t, d_dim) = (shape[0], shape[1], shape[2], shape[3]);

        // 1. [B, H_kv, 1, T, D]
        let x_expanded = x.reshape(vec![
            b.try_into().unwrap(),
            h_kv.try_into().unwrap(),
            1,
            t.try_into().unwrap(),
            d_dim.try_into().unwrap(),
        ]);

        // 2. Broadcast/Repeat to [B, H_kv, n_rep, T, D]
        // 临时方案：使用 ndarray 的 broadcast + to_owned
        let x_arr = x_expanded.data_ref();
        let target_shape = vec![b, h_kv, n_rep, t, d_dim];

        let broadcasted = x_arr
            .broadcast(target_shape)
            .expect("Broadcast failed")
            .to_owned();

        // 3. Reshape to [B, H_q, T, D]
        let res = broadcasted
            .into_shape((b, h_kv * n_rep, t, d_dim))
            .unwrap()
            .into_dyn();

        Tensor::from_array_no_grad(res)
    }

    fn create_causal_mask(&self, seq_len: usize, total_len: usize, pos: usize) -> Tensor {
        // Mask shape: [1, 1, Seq, Total]
        // 创建全 0 矩阵
        // 将非法位置填为 -inf
        let mut mask = ndarray::Array4::<f32>::zeros((1, 1, seq_len, total_len));
        let min_val = f32::NEG_INFINITY;

        for i in 0..seq_len {
            // row (query idx)
            for j in 0..total_len {
                // col (key idx)
                // 绝对位置比较
                let q_pos = pos + i;
                let k_pos = j;
                if k_pos > q_pos {
                    mask[[0, 0, i, j]] = min_val;
                }
            }
        }
        Tensor::from_array_no_grad(mask.into_dyn())
    }
}

/// Llama Decoder Block
struct LlamaDecoderLayer {
    self_attn: LlamaAttention,
    mlp: LlamaMLP,
    input_layernorm: RMSNorm,
    post_attention_layernorm: RMSNorm,
}

impl LlamaDecoderLayer {
    fn new(config: &LlamaConfig) -> Self {
        Self {
            self_attn: LlamaAttention::new(config),
            mlp: LlamaMLP::new(config),
            input_layernorm: RMSNorm::new(config.hidden_size, config.rms_norm_eps),
            post_attention_layernorm: RMSNorm::new(config.hidden_size, config.rms_norm_eps),
        }
    }

    fn forward(&self, x: Tensor, cache: &mut LlamaKVCache, pos: usize) -> Tensor {
        // Pre-Norm Architecture
        // h = x + Attention(Norm(x))
        let norm_x = self.input_layernorm.forward(x.clone());
        let attn_out = self.self_attn.forward(norm_x, cache, pos);
        let h = x + attn_out;

        // out = h + MLP(Norm(h))
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

    /// caches: 每一层对应的预分配 Cache
    /// pos: 当前输入的起始位置 (Generation Step)
    pub fn forward(&self, input_ids: Tensor, caches: &mut Vec<LlamaKVCache>, pos: usize) -> Tensor {
        // Embedding
        let mut x = self.embed_tokens.forward(&input_ids);

        // Decoder Layers
        for (i, layer) in self.layers.iter().enumerate() {
            let layer_cache = &mut caches[i];
            x = layer.forward(x, layer_cache, pos);
        }

        // Final Norm
        x = self.norm.forward(x);

        // LM Head
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
    fn forward(&self, _input: Tensor) -> Tensor {
        panic!(
            "LlamaModel requires cache and pos arguments. Use forward_with_cache instead or refactor trait."
        );
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
