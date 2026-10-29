//! Configuration-driven construction for supported dense causal decoders.
use super::llama::{DecoderOptions, LlamaConfig, LlamaModel};
use crate::precision::DType;
use serde::Deserialize;
use std::path::Path;

#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
pub enum TokenIds {
    One(usize),
    Many(Vec<usize>),
}
impl TokenIds {
    pub fn values(&self) -> Vec<usize> {
        match self {
            Self::One(id) => vec![*id],
            Self::Many(ids) => ids.clone(),
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct CausalLMConfig {
    pub model_type: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f32,
    pub max_position_embeddings: usize,
    pub rope_theta: f32,
    #[serde(default)]
    pub head_dim: Option<usize>,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub eos_token_id: Option<TokenIds>,
    #[serde(default)]
    pub bos_token_id: Option<usize>,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub hidden_act: Option<String>,
    #[serde(default)]
    pub use_sliding_window: bool,
    #[serde(default)]
    pub rope_scaling: Option<serde_json::Value>,
    #[serde(default)]
    pub rope_parameters: Option<serde_json::Value>,
    #[serde(default)]
    pub layer_types: Option<Vec<String>>,
    #[serde(default)]
    pub partial_rotary_factor: Option<f32>,
}

impl CausalLMConfig {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, Box<dyn std::error::Error>> {
        let config: Self = serde_json::from_reader(std::fs::File::open(path)?)?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<(), String> {
        if !matches!(self.model_type.as_str(), "llama" | "qwen3") {
            return Err(format!(
                "Unsupported model_type: {}; supported: llama, qwen3 (dense)",
                self.model_type
            ));
        }
        if [
            self.vocab_size,
            self.hidden_size,
            self.intermediate_size,
            self.num_hidden_layers,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.max_position_embeddings,
        ]
        .contains(&0)
        {
            return Err("Model dimensions must be positive".into());
        }
        if !self
            .num_attention_heads
            .is_multiple_of(self.num_key_value_heads)
            || !self.hidden_size.is_multiple_of(self.num_attention_heads)
        {
            return Err("Invalid attention head geometry".into());
        }
        let head_dim = self
            .head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads);
        if head_dim == 0
            || !head_dim.is_multiple_of(2)
            || self.num_attention_heads.checked_mul(head_dim).is_none()
            || self.num_key_value_heads.checked_mul(head_dim).is_none()
        {
            return Err("RoPE head_dim must be positive, even and fit in usize".into());
        }
        if !(self.rope_theta.is_finite()
            && self.rope_theta > 0.0
            && self.rms_norm_eps.is_finite()
            && self.rms_norm_eps > 0.0)
        {
            return Err("RoPE theta and RMSNorm epsilon must be finite and positive".into());
        }
        if self.attention_bias
            || self.hidden_act.as_deref().is_some_and(|act| act != "silu")
            || self.use_sliding_window
            || self.rope_scaling.is_some()
            || self.rope_parameters.is_some()
            || self
                .partial_rotary_factor
                .is_some_and(|factor| factor != 1.0)
            || self.layer_types.as_ref().is_some_and(|types| {
                types.len() != self.num_hidden_layers
                    || types.iter().any(|kind| kind != "full_attention")
            })
        {
            return Err("Unsupported decoder configuration: requires bias-free SiLU, full attention and unscaled full-head RoPE".into());
        }
        if self
            .eos_token_id
            .as_ref()
            .is_some_and(|ids| ids.values().iter().any(|&id| id >= self.vocab_size))
            || self.bos_token_id.is_some_and(|id| id >= self.vocab_size)
        {
            return Err("Special token ID outside model vocabulary".into());
        }
        Ok(())
    }

    pub fn build(
        &self,
        max_seq_len: usize,
        parameter_dtype: DType,
        activation_dtype: DType,
        kv_cache_dtype: DType,
    ) -> Result<LlamaModel, String> {
        if !kv_cache_dtype.is_float() {
            return Err("KV cache dtype must be floating point".into());
        }
        let config = self.decoder_config(max_seq_len)?;
        Ok(LlamaModel::new_with_decoder_options(
            config,
            parameter_dtype,
            activation_dtype,
            kv_cache_dtype,
            DecoderOptions {
                head_dim: self.head_dim,
                qk_norm: self.model_type == "qwen3",
                tie_word_embeddings: self.tie_word_embeddings,
            },
        ))
    }

    pub fn decoder_config(&self, max_seq_len: usize) -> Result<LlamaConfig, String> {
        self.validate()?;
        if max_seq_len == 0 || max_seq_len > self.max_position_embeddings {
            return Err(format!(
                "max_seq_len must be in 1..={}",
                self.max_position_embeddings
            ));
        }
        Ok(LlamaConfig {
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads,
            rms_norm_eps: self.rms_norm_eps,
            max_seq_len,
            rope_theta: self.rope_theta,
        })
    }
}

#[cfg(test)]
#[path = "../../tests/unit/models/pretrained.rs"]
mod tests;
