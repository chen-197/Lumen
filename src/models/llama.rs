use crate::autograd::{is_no_grad, Tensor};
use crate::inference::{InferenceWeightStorage, WeightDType};
use crate::layers::{Embedding, KVCache, Linear, RMSNorm, SelfAttention, SiLU};
use crate::layers::attention::self_attention::KVCacheInner;
use crate::module::Module;
use crate::ops::fused::{fused_gate_up_silu_infer_into, fused_gate_up_silu_infer_tmp_bf16_into, fused_gate_up_silu_infer_tmp_f32_into};
use crate::ops::matmul::{matvec_argmax_rowmajor_parallel, matvec_argmax_rowmajor_parallel_bf16};
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

thread_local! {
    static MLP_INTER_BUF: RefCell<Vec<f32>> = RefCell::new(Vec::new());
}

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

    #[inline]
    fn should_use_fused_gate_up(x: &Tensor) -> bool {
        if !is_no_grad() {
            return false;
        }
        let xr = x.data_ref();
        let shape = xr.shape();
        let k_dim = *shape.last().expect("MLP input must have last dim");
        let rows = xr.len() / k_dim;
        rows == 1
    }

    fn forward(&self, x: Tensor) -> Tensor {
        if Self::should_use_fused_gate_up(&x) {
            let inter_dim = self.down_proj.in_features;
            return MLP_INTER_BUF.with(|buf| {
                let mut buf = buf.borrow_mut();
                if buf.len() < inter_dim {
                    buf.resize(inter_dim, 0.0);
                }
                {
                    let inter = &mut buf[..inter_dim];
                    if let (Some(gw), Some(uw)) = (
                        self.gate_proj.infer_weight.as_ref(),
                        self.up_proj.infer_weight.as_ref(),
                    ) {
                        match (gw, uw) {
                            (
                                InferenceWeightStorage::F32 { data: gw, .. },
                                InferenceWeightStorage::F32 { data: uw, .. },
                            ) => {
                                fused_gate_up_silu_infer_tmp_f32_into(&x, gw.as_slice(), uw.as_slice(), self.down_proj.in_features, self.gate_proj.in_features, inter);
                            }
                            (
                                InferenceWeightStorage::BF16 { data: gw, .. },
                                InferenceWeightStorage::BF16 { data: uw, .. },
                            ) => {
                                let gw = unsafe { std::slice::from_raw_parts(gw.as_ptr() as *const half::bf16, gw.len()) };
                                let uw = unsafe { std::slice::from_raw_parts(uw.as_ptr() as *const half::bf16, uw.len()) };
                                fused_gate_up_silu_infer_tmp_bf16_into(&x, gw, uw, self.down_proj.in_features, self.gate_proj.in_features, inter);
                            }
                            _ => panic!("temporary fused gate/up expects matching weight dtypes"),
                        }
                    } else {
                        fused_gate_up_silu_infer_into(&x, &self.gate_proj.weight, &self.up_proj.weight, inter);
                    }
                }
                self.down_proj.forward_decode_slice_no_bias(&buf[..inter_dim])
            });
        }

        let gate = {
            self.gate_proj.forward(x.clone())
        };
        let gate_act = {
            self.act.forward(gate)
        };
        let up = {
            self.up_proj.forward(x)
        };
        let fused = {
            gate_act * up
        };
        {
            self.down_proj.forward(fused)
        }
    }

    fn prepare_infer_weights_tmp(&mut self, dtype: WeightDType, migrate_primary_weight: bool) {
        self.gate_proj.prepare_infer_weight_tmp(dtype, migrate_primary_weight);
        self.up_proj.prepare_infer_weight_tmp(dtype, migrate_primary_weight);
        self.down_proj.prepare_infer_weight_tmp(dtype, migrate_primary_weight);
    }

    fn has_temporary_infer_migration(&self) -> bool {
        self.gate_proj.has_temporary_infer_migration()
            || self.up_proj.has_temporary_infer_migration()
            || self.down_proj.has_temporary_infer_migration()
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

    fn prepare_infer_weights_tmp(&mut self, dtype: WeightDType, migrate_primary_weight: bool) {
        self.self_attn.w_q.prepare_infer_weight_tmp(dtype, migrate_primary_weight);
        self.self_attn.w_k.prepare_infer_weight_tmp(dtype, migrate_primary_weight);
        self.self_attn.w_v.prepare_infer_weight_tmp(dtype, migrate_primary_weight);
        self.self_attn.w_o.prepare_infer_weight_tmp(dtype, migrate_primary_weight);
        self.mlp.prepare_infer_weights_tmp(dtype, migrate_primary_weight);
    }

    fn has_temporary_infer_migration(&self) -> bool {
        self.self_attn.has_temporary_infer_migration() || self.mlp.has_temporary_infer_migration()
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

    // 临时：推理压缩权重准备接口，后续由全链路 dtype 替代。
    pub fn prepare_infer_weights_tmp(&mut self, dtype: WeightDType, migrate_primary_weight: bool) {
        self.embed_tokens.prepare_infer_weight_tmp(dtype, migrate_primary_weight);
        for layer in self.layers.iter_mut() {
            layer.prepare_infer_weights_tmp(dtype, migrate_primary_weight);
        }
        self.lm_head.prepare_infer_weight_tmp(dtype, migrate_primary_weight);
    }

    pub fn has_temporary_infer_migration(&self) -> bool {
        self.embed_tokens.has_temporary_infer_migration()
            || self.layers.iter().any(|layer| layer.has_temporary_infer_migration())
            || self.lm_head.has_temporary_infer_migration()
    }


    fn forward_hidden_infer(&self, input_ids: Tensor, caches: &mut Vec<KVCache>) -> Tensor {
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

        {
            self.norm.forward(x)
        }
    }



    fn lm_head_argmax_from_last_hidden(&self, hidden: &Tensor) -> usize {
        assert!(is_no_grad(), "forward_last_argmax is inference-only");

        let hidden_ref = hidden.data_ref();
        let hidden_shape = hidden_ref.shape().to_vec();
        assert_eq!(hidden_shape.len(), 3, "hidden states must be [B,S,H]");
        let (b, s, h) = (hidden_shape[0], hidden_shape[1], hidden_shape[2]);
        assert_eq!(b, 1, "lm_head_argmax currently expects batch size 1");
        assert!(s >= 1, "sequence length must be >= 1");

        let hidden_owned;
        let hidden_slice: &[f32] = if s == 1 {
            if let Some(slice) = hidden_ref.as_slice() {
                assert_eq!(slice.len(), h, "last hidden width mismatch");
                slice
            } else {
                hidden_owned = hidden_ref.iter().copied().collect::<Vec<f32>>();
                assert_eq!(hidden_owned.len(), h, "last hidden width mismatch");
                hidden_owned.as_slice()
            }
        } else {
            let hidden3 = hidden_ref
                .view()
                .into_dimensionality::<ndarray::Ix3>()
                .expect("hidden states must be [B,S,H]");
            let last = hidden3.slice(ndarray::s![0, s - 1, ..]);
            hidden_owned = last.iter().copied().collect::<Vec<f32>>();
            assert_eq!(hidden_owned.len(), h, "last hidden width mismatch");
            hidden_owned.as_slice()
        };

        if let Some(infer) = self.lm_head.infer_weight.as_ref() {
            let (vocab, in_features) = infer.rows_cols();
            assert_eq!(in_features, h, "lm_head in_features mismatch");
            return match infer {
                InferenceWeightStorage::F32 { data, .. } => {
                    matvec_argmax_rowmajor_parallel(hidden_slice, data.as_slice(), vocab, in_features)
                }
                InferenceWeightStorage::BF16 { data, .. } => {
                    let weight = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const half::bf16, data.len()) };
                    matvec_argmax_rowmajor_parallel_bf16(hidden_slice, weight, vocab, in_features)
                }
            };
        }

        let weight_arc = self.lm_head.weight.data_arc();
        let weight2 = weight_arc
            .view()
            .into_dimensionality::<ndarray::Ix2>()
            .expect("lm_head weight must be [V,H]");
        let (_vocab, in_features) = weight2.dim();
        assert_eq!(in_features, h, "lm_head in_features mismatch");

        let weight_slice = weight2
            .as_slice()
            .expect("lm_head weight must be contiguous row-major");
        matvec_argmax_rowmajor_parallel(hidden_slice, weight_slice, weight2.nrows(), in_features)
    }

    /// 推理/生成（需要 caches）。
    ///
    /// `pos` 参数为了兼容旧调用方保留，但会被忽略：长度由 cache 内部维护。
    pub fn forward(&self, input_ids: Tensor, caches: &mut Vec<KVCache>, _pos: usize) -> Tensor {
        let x = self.forward_hidden_infer(input_ids, caches);

        {
            self.lm_head.forward(x)
        }
    }

    /// 生成/benchmark 专用：只返回最后一个位置的 logits。
    ///
    /// - prefill(S>1) 时，避免对整段序列都跑 lm_head
    /// - decode(S=1) 时，等价于普通 forward
    pub fn forward_last_logits(&self, input_ids: Tensor, caches: &mut Vec<KVCache>, _pos: usize) -> Tensor {
        let x = self.forward_hidden_infer(input_ids, caches);

        let x_shape = x.data_ref().shape().to_vec();
        assert_eq!(x_shape.len(), 3, "hidden states must be [B,S,H]");
        let (b, s, h) = (x_shape[0], x_shape[1], x_shape[2]);

        let last_hidden = if s == 1 {
            x
        } else {
            let x_ref = x.data_ref();
            let x3 = x_ref
                .view()
                .into_dimensionality::<ndarray::Ix3>()
                .expect("hidden states must be [B,S,H]");
            let last = x3
                .slice(ndarray::s![.., s - 1..s, ..])
                .to_owned()
                .into_dyn();
            Tensor::from_array_no_grad(last)
        };

        debug_assert_eq!(last_hidden.data_ref().shape(), &[b, 1, h]);

        {
            self.lm_head.forward(last_hidden)
        }
    }



    /// 生成/benchmark 热路径：直接返回最后一个位置的 greedy argmax token。
    ///
    /// - decode(S=1) 时，避免物化 [1,1,V] logits Tensor
    /// - prefill(S>1) 时，也只扫描最后一个位置对应的 hidden
    pub fn forward_last_argmax(&self, input_ids: Tensor, caches: &mut Vec<KVCache>, _pos: usize) -> usize {
        let x = self.forward_hidden_infer(input_ids, caches);
        self.lm_head_argmax_from_last_hidden(&x)
    }

    /// 训练（不使用 cache，支持 autograd）。
    pub fn forward_train(&self, input_ids: Tensor) -> Tensor {
        let mut x = self.embed_tokens.forward(&input_ids);
        for layer in self.layers.iter() {
            x = layer.forward_train(x);
        }
        x = {
            self.norm.forward(x)
        };
        {
            self.lm_head.forward(x)
        }
    }

    pub fn load_named_bf16_infer_weight_direct(
        &mut self,
        name: &str,
        shape: Vec<usize>,
        data: Vec<u16>,
    ) -> Result<(), String> {
        match name {
            "model.embed_tokens.weight" => {
                self.embed_tokens.load_infer_weight_bf16_direct(shape, data);
                Ok(())
            }
            "lm_head.weight" => {
                self.lm_head.load_infer_weight_bf16_direct(shape, data);
                Ok(())
            }
            _ => {
                if let Some(rest) = name.strip_prefix("model.layers.") {
                    let dot = rest.find('.').ok_or_else(|| format!("bad layer name: {name}"))?;
                    let layer_idx: usize = rest[..dot]
                        .parse()
                        .map_err(|_| format!("bad layer idx in {name}"))?;
                    let suffix = &rest[dot + 1..];
                    let layer = self
                        .layers
                        .get_mut(layer_idx)
                        .ok_or_else(|| format!("layer idx out of range in {name}"))?;

                    match suffix {
                        "self_attn.q_proj.weight" => {
                            layer.self_attn.w_q.load_infer_weight_bf16_direct(shape, data);
                            Ok(())
                        }
                        "self_attn.k_proj.weight" => {
                            layer.self_attn.w_k.load_infer_weight_bf16_direct(shape, data);
                            Ok(())
                        }
                        "self_attn.v_proj.weight" => {
                            layer.self_attn.w_v.load_infer_weight_bf16_direct(shape, data);
                            Ok(())
                        }
                        "self_attn.o_proj.weight" => {
                            layer.self_attn.w_o.load_infer_weight_bf16_direct(shape, data);
                            Ok(())
                        }
                        "mlp.gate_proj.weight" => {
                            layer.mlp.gate_proj.load_infer_weight_bf16_direct(shape, data);
                            Ok(())
                        }
                        "mlp.up_proj.weight" => {
                            layer.mlp.up_proj.load_infer_weight_bf16_direct(shape, data);
                            Ok(())
                        }
                        "mlp.down_proj.weight" => {
                            layer.mlp.down_proj.load_infer_weight_bf16_direct(shape, data);
                            Ok(())
                        }
                        "input_layernorm.weight" | "post_attention_layernorm.weight" => {
                            Err(format!("norm weight should stay on normal tensor path: {name}"))
                        }
                        _ => Err(format!("unknown layer weight name: {name}")),
                    }
                } else if name == "model.norm.weight" {
                    Err(format!("final norm weight should stay on normal tensor path: {name}"))
                } else {
                    Err(format!("unknown weight name: {name}"))
                }
            }
        }
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

    fn has_temporary_infer_migration(&self) -> bool {
        LlamaModel::has_temporary_infer_migration(self)
    }
}
