//! CUDA buffers, execution policy, and operator interfaces.

mod imp;

#[path = "api/memory.rs"]
mod memory;
#[cfg(test)]
pub(crate) use memory::test_host_downloads;
pub use memory::{
    add_inplace_f32, alloc_f32, alloc_storage, copy_f32_offset, download_f32, download_f32_offset,
    download_i8_storage, download_u16_storage, f32_to_i8_storage_no_host,
    f32_to_lowp_storage_no_host, fill_scalar_f32, fill_scalar_f32_buffer, is_available,
    quantize_f32_storage_no_host, quantize_f32_to_i8_dynamic_no_host, release_cached_memory,
    synchronize, upload_f32, upload_f32_offset, upload_i8_storage, upload_u16_storage,
};
#[path = "api/selection.rs"]
mod selection;
pub use selection::{
    matvec_argmax_bf16_i8, matvec_argmax_f16_i8, matvec_argmax_f32, matvec_argmax_f32_i8,
    matvec_argmax_i8_i8,
};
#[path = "api/attention.rs"]
mod attention;
pub use attention::{
    append_kv_cache_f32, append_kv_cache_pair_f32, decode_attention_f32,
    decode_attention_f32_buffer, decode_rope_q_append_kv_f32_buffer, kv_cache_prefix_f32_buffer,
    kv_cache_prefix_typed_buffer, prefill_attention_f32_buffer, repeat_kv_backward_f32,
    repeat_kv_backward_f32_buffer, repeat_kv_f32, repeat_kv_f32_buffer, repeat_kv_typed_buffer,
    rope_backward_f32, rope_backward_f32_buffer, rope_f32, rope_f32_buffer, rope_typed_buffer,
    rope_typed_i8_dynamic_buffer,
};
#[path = "api/matmul.rs"]
mod matmul;
pub use matmul::{
    batch_matmul_backward_bf16_i8_no_host, batch_matmul_backward_bf16_no_host,
    batch_matmul_backward_f16_i8_no_host, batch_matmul_backward_f16_no_host,
    batch_matmul_backward_f32_i8_no_host, batch_matmul_backward_f32_no_host,
    batch_matmul_backward_i8_bf16_no_host, batch_matmul_backward_i8_f16_no_host,
    batch_matmul_backward_i8_f32_no_host, batch_matmul_backward_i8_no_host,
    batch_matmul_bf16_buffer_no_host, batch_matmul_bf16_i8_buffer_no_host,
    batch_matmul_bf16_typed_output_buffer_no_host, batch_matmul_f16_buffer_no_host,
    batch_matmul_f16_i8_buffer_no_host, batch_matmul_f16_typed_output_buffer_no_host,
    batch_matmul_f32, batch_matmul_f32_i8_buffer_no_host, batch_matmul_f32_no_host,
    batch_matmul_i8_bf16_buffer_no_host, batch_matmul_i8_buffer_no_host,
    batch_matmul_i8_f16_buffer_no_host, batch_matmul_i8_f32_buffer_no_host,
    batch_matmul_i8_typed_output_buffer_no_host, matmul_backward_bf16_i8_no_host,
    matmul_backward_f16_i8_no_host, matmul_backward_f32_i8_no_host, matmul_backward_f32_no_host,
    matmul_backward_i8_bf16_no_host, matmul_backward_i8_f16_no_host,
    matmul_backward_i8_f32_no_host, matmul_bf16_buffer_no_host, matmul_bf16_host_no_host,
    matmul_bf16_i8_buffer_no_host, matmul_bf16_i8_dynamic_buffer_no_host,
    matmul_bf16_typed_output_buffer_no_host, matmul_f16_buffer_no_host, matmul_f16_host_no_host,
    matmul_f16_i8_buffer_no_host, matmul_f16_i8_dynamic_buffer_no_host,
    matmul_f16_typed_output_buffer_no_host, matmul_f32, matmul_f32_i8_buffer_no_host,
    matmul_f32_no_host, matmul_i8_bf16_buffer_no_host, matmul_i8_buffer_no_host,
    matmul_i8_f16_buffer_no_host, matmul_i8_f32_buffer_no_host, matmul_i8_host_no_host,
    matmul_i8_typed_output_buffer_no_host,
};
#[path = "api/elementwise.rs"]
mod elementwise;
pub use elementwise::{
    add_sub_backward_b1d_1h1_f32_buffers, add_sub_backward_b1d_1hd_f32_buffers,
    add_sub_backward_f32_buffers, add_sub_backward_lastdim_f32_buffers,
    add_sub_backward_row_scalar_f32_buffers, add_sub_backward_scalar_f32_buffers,
    add_sub_broadcast_backward_f32_buffers, binary_backward_f32, binary_backward_f32_buffers,
    binary_bf16_buffer_no_host, binary_bf16_host_no_host, binary_bf16_lastdim_buffer_no_host,
    binary_broadcast_backward_f32, binary_broadcast_backward_f32_buffers, binary_broadcast_f32,
    binary_broadcast_f32_buffer, binary_f16_buffer_no_host, binary_f16_host_no_host,
    binary_f16_lastdim_buffer_no_host, binary_f32, binary_f32_buffer, binary_i8_buffer_no_host,
    binary_i8_host_no_host, binary_i8_lastdim_buffer_no_host,
    binary_i8_typed_b1d_1h1_output_buffer_no_host, binary_i8_typed_b1d_1hd_output_buffer_no_host,
    binary_i8_typed_broadcast_output_buffer_no_host, binary_i8_typed_lastdim_output_buffer_no_host,
    binary_i8_typed_output_buffer_no_host, binary_i8_typed_row_scalar_output_buffer_no_host,
    binary_lowp_typed_b1d_1h1_output_buffer_no_host,
    binary_lowp_typed_b1d_1hd_output_buffer_no_host,
    binary_lowp_typed_broadcast_output_buffer_no_host,
    binary_lowp_typed_lastdim_output_buffer_no_host, binary_lowp_typed_output_buffer_no_host,
    binary_lowp_typed_row_scalar_output_buffer_no_host, binary_typed_b1d_1h1_buffer_no_host,
    binary_typed_b1d_1hd_buffer_no_host, binary_typed_broadcast_buffer_no_host,
    binary_typed_buffer_no_host, binary_typed_lastdim_buffer_no_host,
    binary_typed_row_scalar_buffer_no_host, mul_grad_bf16_buffer_no_host,
    mul_grad_bf16_host_no_host, mul_grad_bf16_lastdim_buffer_no_host, mul_grad_f16_buffer_no_host,
    mul_grad_f16_host_no_host, mul_grad_f16_lastdim_buffer_no_host, mul_grad_i8_buffer_no_host,
    mul_grad_i8_host_no_host, mul_grad_i8_lastdim_buffer_no_host,
    mul_grad_typed_b1d_1h1_buffer_no_host, mul_grad_typed_b1d_1hd_buffer_no_host,
    mul_grad_typed_broadcast_buffer_no_host, mul_grad_typed_buffer_no_host,
    mul_grad_typed_lastdim_buffer_no_host, mul_grad_typed_row_scalar_buffer_no_host,
    mul_grad_typed_scalar_buffer_no_host, unary_backward_bf16_buffer, unary_backward_f16_buffer,
    unary_backward_f32, unary_backward_f32_buffer, unary_backward_i8_buffer, unary_bf16_buffer,
    unary_bf16_typed_output_buffer, unary_f16_buffer, unary_f16_typed_output_buffer, unary_f32,
    unary_f32_buffer, unary_i8_buffer, unary_i8_relu_typed_output_buffer,
};
#[path = "api/reduction_loss.rs"]
mod reduction_loss;
pub use reduction_loss::{
    cross_entropy_backward_f32, cross_entropy_backward_f32_buffer,
    cross_entropy_backward_typed_target, cross_entropy_backward_typed_target_buffer,
    cross_entropy_loss_f32, cross_entropy_loss_typed_target, mse_backward_f32,
    mse_backward_f32_buffers, mse_forward_typed, sum_bf16_buffer, sum_f16_buffer, sum_f32,
    sum_i8_buffer, sum_lastdim_f32_buffer,
};
pub(crate) use reduction_loss::{cross_entropy_logits_backward_typed, cross_entropy_logits_typed};
#[path = "api/shape.rs"]
mod shape;
pub use shape::{
    bshd_to_bhsd_add_bias_f32_buffer, cat_backward_slice_f32, cat_backward_slice_f32_buffer,
    cat_f32, cat_f32_buffer, cat_typed_buffer, embedding_backward_f32,
    embedding_backward_f32_buffer, embedding_f32, embedding_f32_buffer, embedding_typed,
    embedding_typed_buffer, embedding_typed_same_dtype_buffer, permute_f32, permute_f32_buffer,
    permute_typed_buffer, slice_lastdim_backward_f32, slice_lastdim_backward_f32_buffer,
    slice_lastdim_f32, slice_lastdim_f32_buffer, slice_lastdim_typed_buffer,
};
#[path = "api/optimizer.rs"]
mod optimizer;
pub(crate) use optimizer::optimizer_lowp_state_update_no_host;
pub use optimizer::{
    adam_update_f32, adam_update_f32_batched_no_host, adam_update_f32_no_host,
    sgd_momentum_update_f32, sgd_momentum_update_f32_batched_no_host,
    sgd_momentum_update_f32_no_host, sgd_update_f32, sgd_update_f32_batched_no_host,
    sgd_update_f32_no_host,
};
#[path = "api/normalization.rs"]
mod normalization;
pub use normalization::{
    fused_softmax_backward_f32, fused_softmax_backward_f32_buffer, fused_softmax_f32,
    fused_softmax_f32_no_host, fused_softmax_f32_with_past, fused_softmax_f32_with_past_no_host,
    rms_norm_backward_f32, rms_norm_backward_f32_buffers, rms_norm_backward_typed,
    rms_norm_backward_typed_buffers, rms_norm_f32, rms_norm_f32_buffer,
    rms_norm_i8_typed_output_buffer, rms_norm_typed, rms_norm_typed_buffer,
    softmax_lastdim_backward_f32, softmax_lastdim_backward_f32_buffer, softmax_lastdim_f32,
    softmax_lastdim_f32_no_host, softmax_lastdim_typed, softmax_lastdim_typed_no_host,
};
#[path = "api/projection.rs"]
mod projection;
pub use projection::{
    fused_gate_up_silu_f32, fused_gate_up_silu_f32_buffer, fused_gate_up_silu_typed,
    fused_gate_up_silu_typed_buffer, fused_gate_up_silu_typed_output_buffer, fused_qkv_f32,
    fused_qkv_f32_buffer, fused_qkv_typed_buffer, fused_qkv_typed_output_buffer,
    silu_mul_f32_buffer_no_host,
};
#[path = "api/convolution.rs"]
mod convolution;
pub use convolution::{
    conv2d_backward_f32, conv2d_backward_f32_buffers, conv2d_f32, conv2d_f32_buffer, conv2d_typed,
    conv2d_typed_buffer, max_pool2d_backward_f32, max_pool2d_backward_f32_buffer,
    max_pool2d_backward_typed_buffer, max_pool2d_f32, max_pool2d_typed,
};

use std::cell::Cell;
use std::sync::Arc;

pub(crate) mod generic;

const CUDA_MATMUL_MIN_WORK: usize = 1 << 18;
const CUDA_BATCH_MATMUL_MIN_WORK: usize = 1 << 18;
const CUDA_ELEMENTWISE_MIN_WORK: usize = 1 << 14;
const CUDA_SOFTMAX_MIN_WORK: usize = 1 << 13;

#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnaryOp {
    Relu = 0,
    Sigmoid = 1,
    Tanh = 2,
    Silu = 3,
    Gelu = 4,
}

#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinaryOp {
    Add = 0,
    Sub = 1,
    Mul = 2,
}

struct CudaBufferInner {
    handle: u64,
    len: usize,
}

#[derive(Clone)]
pub struct CudaBuffer(Arc<CudaBufferInner>);

#[cfg(test)]
pub(crate) struct BufferDropProbe(std::sync::Weak<CudaBufferInner>);

#[cfg(test)]
impl BufferDropProbe {
    pub(crate) fn is_released(&self) -> bool {
        self.0.strong_count() == 0
    }
}

#[derive(Clone, Copy)]
pub(crate) struct LowpOptimizerConfig {
    pub state_dtype: crate::precision::DType,
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub correction1: f32,
    pub correction2: f32,
    pub eps: f32,
}

impl CudaBuffer {
    pub(crate) fn has_unique_storage(&mut self) -> bool {
        Arc::get_mut(&mut self.0).is_some()
    }

    #[cfg(test)]
    pub(crate) fn drop_probe(&self) -> BufferDropProbe {
        BufferDropProbe(Arc::downgrade(&self.0))
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn from_raw(handle: u64, len: usize) -> Self {
        Self(Arc::new(CudaBufferInner { handle, len }))
    }

    pub fn handle(&self) -> u64 {
        self.0.handle
    }

    pub fn len(&self) -> usize {
        self.0.len
    }

    pub fn is_empty(&self) -> bool {
        self.0.len == 0
    }
}

pub type CudaHostBuffer = (CudaBuffer, Vec<f32>);
pub type CudaTwoHostBuffers = (CudaHostBuffer, CudaHostBuffer);
pub type CudaThreeHostBuffers = (CudaHostBuffer, CudaHostBuffer, CudaHostBuffer);
pub type CudaAdamHostState = (Vec<f32>, Vec<f32>, Vec<f32>);
pub type CudaConv2dBackwardHostBuffers = (
    CudaBuffer,
    Vec<f32>,
    CudaBuffer,
    Vec<f32>,
    Option<CudaHostBuffer>,
);

type BroadcastMetadata = (Vec<usize>, Vec<usize>, Vec<usize>, Vec<usize>, Vec<usize>);

impl Drop for CudaBufferInner {
    fn drop(&mut self) {
        if self.handle != 0 {
            imp::free_f32(self.handle, self.len);
        }
    }
}

fn env_enabled() -> bool {
    std::env::var("LUMEN_CUDA")
        .ok()
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

thread_local! {
    static CUDA_ENABLED: Cell<bool> = Cell::new(env_enabled());
}

pub fn set_enabled(enabled: bool) {
    CUDA_ENABLED.with(|flag| flag.set(enabled));
}

fn is_enabled_requested() -> bool {
    CUDA_ENABLED.with(|flag| flag.get())
}

pub fn is_enabled() -> bool {
    is_enabled_requested() && is_available()
}

/// Restores the caller-thread CUDA selection; cannot move between threads.
///
/// ```compile_fail
/// let guard = lumen::ops::cuda::set_enabled_scoped(true);
/// std::thread::spawn(move || drop(guard));
/// ```
pub struct CudaEnabledGuard {
    previous: bool,
    _caller_thread: std::marker::PhantomData<std::rc::Rc<()>>,
}

pub fn set_enabled_scoped(enabled: bool) -> CudaEnabledGuard {
    let previous = is_enabled_requested();
    set_enabled(enabled);
    CudaEnabledGuard {
        previous,
        _caller_thread: std::marker::PhantomData,
    }
}

impl Drop for CudaEnabledGuard {
    fn drop(&mut self) {
        set_enabled(self.previous);
    }
}

pub fn should_accelerate_matmul(m: usize, n: usize, k: usize) -> bool {
    is_enabled()
        && m.checked_mul(n)
            .and_then(|value| value.checked_mul(k))
            .is_some_and(|work| work >= CUDA_MATMUL_MIN_WORK)
}

pub fn should_accelerate_batch_matmul(batch_count: usize, m: usize, n: usize, k: usize) -> bool {
    is_enabled()
        && batch_count
            .checked_mul(m)
            .and_then(|value| value.checked_mul(n))
            .and_then(|value| value.checked_mul(k))
            .is_some_and(|work| work >= CUDA_BATCH_MATMUL_MIN_WORK)
}

pub fn should_accelerate_elementwise(len: usize) -> bool {
    is_enabled() && len >= CUDA_ELEMENTWISE_MIN_WORK
}

pub fn should_accelerate_softmax(outer: usize, last_dim: usize) -> bool {
    is_enabled()
        && outer
            .checked_mul(last_dim)
            .is_some_and(|work| work >= CUDA_SOFTMAX_MIN_WORK)
}

#[cfg(test)]
#[path = "../../../tests/unit/ops/cuda/tests.rs"]
mod tests;

#[cfg(test)]
thread_local! {
    static TEST_HOST_DOWNLOADS: Cell<usize> = const { Cell::new(0) };
}
