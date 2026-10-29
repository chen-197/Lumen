//! Dense Qwen3 models, using the shared rotary/SwiGLU decoder implementation.
use super::{CausalLMConfig, LlamaModel};
use crate::{
    autograd::Tensor,
    module::Module,
    precision::{DType, default_activation_dtype, default_kv_cache_dtype, default_parameter_dtype},
};

pub type Qwen3Config = CausalLMConfig;
pub struct Qwen3Model {
    inner: LlamaModel,
}
impl Qwen3Model {
    pub fn new(config: Qwen3Config) -> Result<Self, String> {
        Self::new_with_runtime_dtypes(
            config,
            default_parameter_dtype(),
            default_activation_dtype(),
            default_kv_cache_dtype(),
        )
    }
    pub fn new_with_runtime_dtypes(
        config: Qwen3Config,
        parameter_dtype: DType,
        activation_dtype: DType,
        kv_cache_dtype: DType,
    ) -> Result<Self, String> {
        if config.model_type != "qwen3" {
            return Err("Qwen3Model requires model_type=qwen3".into());
        }
        let inner = config.build(
            config.max_position_embeddings,
            parameter_dtype,
            activation_dtype,
            kv_cache_dtype,
        )?;
        Ok(Self { inner })
    }
}
impl std::ops::Deref for Qwen3Model {
    type Target = LlamaModel;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}
impl Module for Qwen3Model {
    fn forward(&self, input: Tensor) -> Tensor {
        self.inner.forward_train(input)
    }
    fn parameters(&self) -> Vec<Tensor> {
        self.inner.parameters()
    }
}
