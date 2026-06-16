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

#[derive(Clone)]
pub struct CudaBuffer {
    handle: u64,
    len: usize,
}

impl CudaBuffer {
    pub fn handle(&self) -> u64 {
        self.handle
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
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

pub fn set_enabled(_enabled: bool) {}

pub struct CudaEnabledGuard;

pub fn set_enabled_scoped(_enabled: bool) -> CudaEnabledGuard {
    CudaEnabledGuard
}

impl Drop for CudaEnabledGuard {
    fn drop(&mut self) {}
}

pub fn is_enabled() -> bool {
    false
}

pub fn is_available() -> bool {
    false
}

pub fn synchronize() -> Result<(), String> {
    Err("CUDA feature is disabled".to_string())
}

/// No-op counterpart of the CUDA cache-release API for non-CUDA builds.
pub fn release_cached_memory() -> Result<(), String> {
    Ok(())
}

pub fn should_accelerate_matmul(_m: usize, _n: usize, _k: usize) -> bool {
    false
}

pub fn should_accelerate_batch_matmul(
    _batch_count: usize,
    _m: usize,
    _n: usize,
    _k: usize,
) -> bool {
    false
}

pub fn should_accelerate_elementwise(_len: usize) -> bool {
    false
}

pub fn should_accelerate_softmax(_outer: usize, _last_dim: usize) -> bool {
    false
}

pub fn alloc_f32(_len: usize) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn alloc_storage(_len: usize) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn upload_f32(_src: &[f32]) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn upload_u16_storage(_src: &[u16]) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn upload_i8_storage(_src: &[i8]) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn download_f32(_buffer: &CudaBuffer) -> Result<Vec<f32>, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn download_u16_storage(_buffer: &CudaBuffer) -> Result<Vec<u16>, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn download_i8_storage(_buffer: &CudaBuffer) -> Result<Vec<i8>, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn download_f32_offset(
    _buffer: &CudaBuffer,
    _offset: usize,
    _len: usize,
) -> Result<Vec<f32>, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn matvec_argmax_f32(
    _input: &CudaBuffer,
    _weight: &CudaBuffer,
    _batch_size: usize,
    _vocab_size: usize,
    _hidden_size: usize,
) -> Result<Vec<usize>, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn matvec_argmax_bf16_i8(
    _input: &CudaBuffer,
    _weight: &CudaBuffer,
    _weight_scale: f32,
    _batch_size: usize,
    _vocab_size: usize,
    _hidden_size: usize,
) -> Result<Vec<usize>, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn matvec_argmax_f16_i8(
    _input: &CudaBuffer,
    _weight: &CudaBuffer,
    _weight_scale: f32,
    _batch_size: usize,
    _vocab_size: usize,
    _hidden_size: usize,
) -> Result<Vec<usize>, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn matvec_argmax_f32_i8(
    _input: &CudaBuffer,
    _weight: &CudaBuffer,
    _weight_scale: f32,
    _batch_size: usize,
    _vocab_size: usize,
    _hidden_size: usize,
) -> Result<Vec<usize>, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn matvec_argmax_i8_i8(
    _input: &CudaBuffer,
    _input_scale: f32,
    _weight: &CudaBuffer,
    _weight_scale: f32,
    _batch_size: usize,
    _vocab_size: usize,
    _hidden_size: usize,
) -> Result<Vec<usize>, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn upload_f32_offset(_buffer: &CudaBuffer, _offset: usize, _src: &[f32]) -> Result<(), String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn copy_f32_offset(
    _dst: &CudaBuffer,
    _dst_offset: usize,
    _src: &CudaBuffer,
    _src_offset: usize,
    _len: usize,
) -> Result<(), String> {
    Err("CUDA feature is disabled".to_string())
}

#[allow(clippy::too_many_arguments)]
pub fn append_kv_cache_f32(
    _dst: &CudaBuffer,
    _src: &CudaBuffer,
    _batch_size: usize,
    _num_heads: usize,
    _src_seq_len: usize,
    _dst_seq_len: usize,
    _dim: usize,
    _dst_start: usize,
) -> Result<(), String> {
    Err("CUDA feature is disabled".to_string())
}

#[allow(clippy::too_many_arguments)]
pub fn append_kv_cache_pair_f32(
    _k_dst: &CudaBuffer,
    _v_dst: &CudaBuffer,
    _k_src: &CudaBuffer,
    _v_src: &CudaBuffer,
    _batch_size: usize,
    _num_heads: usize,
    _src_seq_len: usize,
    _dst_seq_len: usize,
    _dim: usize,
    _dst_start: usize,
) -> Result<(), String> {
    Err("CUDA feature is disabled".to_string())
}

#[allow(clippy::too_many_arguments)]
pub fn decode_rope_q_append_kv_f32_buffer(
    _q_src: &CudaBuffer,
    _k_src: &CudaBuffer,
    _v_src: &CudaBuffer,
    _cos: &CudaBuffer,
    _sin: &CudaBuffer,
    _k_cache: &CudaBuffer,
    _v_cache: &CudaBuffer,
    _batch_size: usize,
    _num_heads: usize,
    _num_kv_heads: usize,
    _dim: usize,
    _dst_seq_len: usize,
    _offset: usize,
    _cache_seq_len: usize,
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

#[allow(clippy::too_many_arguments)]
pub fn prefill_attention_f32_buffer(
    _q: &CudaBuffer,
    _k: &CudaBuffer,
    _v: &CudaBuffer,
    _batch_size: usize,
    _num_heads: usize,
    _num_kv_heads: usize,
    _q_seq_len: usize,
    _active_seq_len: usize,
    _cache_seq_len: usize,
    _dim: usize,
    _n_rep: usize,
    _past_len: usize,
    _scale: f32,
    _is_causal: bool,
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn kv_cache_prefix_f32_buffer(
    _src: &CudaBuffer,
    _batch_size: usize,
    _num_heads: usize,
    _active_seq_len: usize,
    _src_seq_len: usize,
    _dim: usize,
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

#[allow(clippy::too_many_arguments)]
pub fn kv_cache_prefix_typed_buffer(
    _src: &CudaBuffer,
    _dtype: crate::precision::DType,
    _batch_size: usize,
    _num_heads: usize,
    _active_seq_len: usize,
    _src_seq_len: usize,
    _dim: usize,
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn matmul_f32(
    _a: &CudaBuffer,
    _b: &CudaBuffer,
    _m: usize,
    _n: usize,
    _k: usize,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn matmul_f32_no_host(
    _a: &CudaBuffer,
    _b: &CudaBuffer,
    _m: usize,
    _n: usize,
    _k: usize,
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn matmul_bf16_host_no_host(
    _a: &[u16],
    _b: &[u16],
    _m: usize,
    _n: usize,
    _k: usize,
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn matmul_bf16_buffer_no_host(
    _a: &CudaBuffer,
    _b: &CudaBuffer,
    _m: usize,
    _n: usize,
    _k: usize,
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn matmul_bf16_typed_output_buffer_no_host(
    _a: &CudaBuffer,
    _b: &CudaBuffer,
    _m: usize,
    _n: usize,
    _k: usize,
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn matmul_f16_host_no_host(
    _a: &[u16],
    _b: &[u16],
    _m: usize,
    _n: usize,
    _k: usize,
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn matmul_f16_typed_output_buffer_no_host(
    _a: &CudaBuffer,
    _b: &CudaBuffer,
    _m: usize,
    _n: usize,
    _k: usize,
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn matmul_f16_buffer_no_host(
    _a: &CudaBuffer,
    _b: &CudaBuffer,
    _m: usize,
    _n: usize,
    _k: usize,
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

#[allow(clippy::too_many_arguments)]
pub fn matmul_i8_host_no_host(
    _a: &[i8],
    _a_scale: f32,
    _b: &[i8],
    _b_scale: f32,
    _m: usize,
    _n: usize,
    _k: usize,
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

#[allow(clippy::too_many_arguments)]
pub fn matmul_i8_buffer_no_host(
    _a: &CudaBuffer,
    _a_scale: f32,
    _b: &CudaBuffer,
    _b_scale: f32,
    _m: usize,
    _n: usize,
    _k: usize,
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn matmul_bf16_i8_buffer_no_host(
    _a: &CudaBuffer,
    _b: &CudaBuffer,
    _b_scale: f32,
    _m: usize,
    _n: usize,
    _k: usize,
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn matmul_f16_i8_buffer_no_host(
    _a: &CudaBuffer,
    _b: &CudaBuffer,
    _b_scale: f32,
    _m: usize,
    _n: usize,
    _k: usize,
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn matmul_bf16_i8_dynamic_buffer_no_host(
    _a: &CudaBuffer,
    _b: &CudaBuffer,
    _b_scale: f32,
    _m: usize,
    _n: usize,
    _k: usize,
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn matmul_f16_i8_dynamic_buffer_no_host(
    _a: &CudaBuffer,
    _b: &CudaBuffer,
    _b_scale: f32,
    _m: usize,
    _n: usize,
    _k: usize,
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn matmul_f32_i8_buffer_no_host(
    _a: &CudaBuffer,
    _b: &CudaBuffer,
    _b_scale: f32,
    _m: usize,
    _n: usize,
    _k: usize,
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn matmul_i8_bf16_buffer_no_host(
    _a: &CudaBuffer,
    _a_scale: f32,
    _b: &CudaBuffer,
    _m: usize,
    _n: usize,
    _k: usize,
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn matmul_i8_f16_buffer_no_host(
    _a: &CudaBuffer,
    _a_scale: f32,
    _b: &CudaBuffer,
    _m: usize,
    _n: usize,
    _k: usize,
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn matmul_i8_f32_buffer_no_host(
    _a: &CudaBuffer,
    _a_scale: f32,
    _b: &CudaBuffer,
    _m: usize,
    _n: usize,
    _k: usize,
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn matmul_i8_typed_output_buffer_no_host(
    _a: &CudaBuffer,
    _a_scale: f32,
    _b: &CudaBuffer,
    _b_scale: f32,
    _m: usize,
    _n: usize,
    _k: usize,
) -> Result<(CudaBuffer, f32), String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn batch_matmul_f32(
    _lhs: &CudaBuffer,
    _rhs: &CudaBuffer,
    _batch_count: usize,
    _m: usize,
    _n: usize,
    _k: usize,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn batch_matmul_f32_no_host(
    _lhs: &CudaBuffer,
    _rhs: &CudaBuffer,
    _batch_count: usize,
    _m: usize,
    _n: usize,
    _k: usize,
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn batch_matmul_bf16_buffer_no_host(
    _lhs: &CudaBuffer,
    _rhs: &CudaBuffer,
    _batch_count: usize,
    _m: usize,
    _n: usize,
    _k: usize,
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn batch_matmul_bf16_typed_output_buffer_no_host(
    _lhs: &CudaBuffer,
    _rhs: &CudaBuffer,
    _batch_count: usize,
    _m: usize,
    _n: usize,
    _k: usize,
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn batch_matmul_f16_buffer_no_host(
    _lhs: &CudaBuffer,
    _rhs: &CudaBuffer,
    _batch_count: usize,
    _m: usize,
    _n: usize,
    _k: usize,
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn batch_matmul_f16_typed_output_buffer_no_host(
    _lhs: &CudaBuffer,
    _rhs: &CudaBuffer,
    _batch_count: usize,
    _m: usize,
    _n: usize,
    _k: usize,
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn batch_matmul_bf16_i8_buffer_no_host(
    _lhs: &CudaBuffer,
    _rhs: &CudaBuffer,
    _rhs_scale: f32,
    _batch_count: usize,
    _m: usize,
    _n: usize,
    _k: usize,
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn batch_matmul_f16_i8_buffer_no_host(
    _lhs: &CudaBuffer,
    _rhs: &CudaBuffer,
    _rhs_scale: f32,
    _batch_count: usize,
    _m: usize,
    _n: usize,
    _k: usize,
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn batch_matmul_f32_i8_buffer_no_host(
    _lhs: &CudaBuffer,
    _rhs: &CudaBuffer,
    _rhs_scale: f32,
    _batch_count: usize,
    _m: usize,
    _n: usize,
    _k: usize,
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn batch_matmul_i8_bf16_buffer_no_host(
    _lhs: &CudaBuffer,
    _lhs_scale: f32,
    _rhs: &CudaBuffer,
    _batch_count: usize,
    _m: usize,
    _n: usize,
    _k: usize,
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn batch_matmul_i8_f16_buffer_no_host(
    _lhs: &CudaBuffer,
    _lhs_scale: f32,
    _rhs: &CudaBuffer,
    _batch_count: usize,
    _m: usize,
    _n: usize,
    _k: usize,
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn batch_matmul_i8_f32_buffer_no_host(
    _lhs: &CudaBuffer,
    _lhs_scale: f32,
    _rhs: &CudaBuffer,
    _batch_count: usize,
    _m: usize,
    _n: usize,
    _k: usize,
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

#[allow(clippy::too_many_arguments)]
pub fn batch_matmul_i8_buffer_no_host(
    _lhs: &CudaBuffer,
    _lhs_scale: f32,
    _rhs: &CudaBuffer,
    _rhs_scale: f32,
    _batch_count: usize,
    _m: usize,
    _n: usize,
    _k: usize,
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn batch_matmul_i8_typed_output_buffer_no_host(
    _lhs: &CudaBuffer,
    _lhs_scale: f32,
    _rhs: &CudaBuffer,
    _rhs_scale: f32,
    _batch_count: usize,
    _m: usize,
    _n: usize,
    _k: usize,
) -> Result<(CudaBuffer, f32), String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn matmul_backward_f32_no_host(
    _grad: &CudaBuffer,
    _a: &CudaBuffer,
    _b: &CudaBuffer,
    _m: usize,
    _k: usize,
    _n: usize,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn matmul_backward_bf16_i8_no_host(
    _grad: &CudaBuffer,
    _lhs: &CudaBuffer,
    _rhs: &CudaBuffer,
    _rhs_scale: f32,
    _m: usize,
    _k: usize,
    _n: usize,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn matmul_backward_f16_i8_no_host(
    _grad: &CudaBuffer,
    _lhs: &CudaBuffer,
    _rhs: &CudaBuffer,
    _rhs_scale: f32,
    _m: usize,
    _k: usize,
    _n: usize,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn matmul_backward_f32_i8_no_host(
    _grad: &CudaBuffer,
    _lhs: &CudaBuffer,
    _rhs: &CudaBuffer,
    _rhs_scale: f32,
    _m: usize,
    _k: usize,
    _n: usize,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn matmul_backward_i8_bf16_no_host(
    _grad: &CudaBuffer,
    _lhs: &CudaBuffer,
    _lhs_scale: f32,
    _rhs: &CudaBuffer,
    _m: usize,
    _k: usize,
    _n: usize,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn matmul_backward_i8_f16_no_host(
    _grad: &CudaBuffer,
    _lhs: &CudaBuffer,
    _lhs_scale: f32,
    _rhs: &CudaBuffer,
    _m: usize,
    _k: usize,
    _n: usize,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn matmul_backward_i8_f32_no_host(
    _grad: &CudaBuffer,
    _lhs: &CudaBuffer,
    _lhs_scale: f32,
    _rhs: &CudaBuffer,
    _m: usize,
    _k: usize,
    _n: usize,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn batch_matmul_backward_f32_no_host(
    _grad: &CudaBuffer,
    _lhs: &CudaBuffer,
    _rhs: &CudaBuffer,
    _batch_count: usize,
    _m: usize,
    _k: usize,
    _n: usize,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn batch_matmul_backward_bf16_no_host(
    _grad: &CudaBuffer,
    _lhs: &CudaBuffer,
    _rhs: &CudaBuffer,
    _batch_count: usize,
    _m: usize,
    _k: usize,
    _n: usize,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn batch_matmul_backward_f16_no_host(
    _grad: &CudaBuffer,
    _lhs: &CudaBuffer,
    _rhs: &CudaBuffer,
    _batch_count: usize,
    _m: usize,
    _k: usize,
    _n: usize,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn batch_matmul_backward_bf16_i8_no_host(
    _grad: &CudaBuffer,
    _lhs: &CudaBuffer,
    _rhs: &CudaBuffer,
    _rhs_scale: f32,
    _batch_count: usize,
    _m: usize,
    _k: usize,
    _n: usize,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn batch_matmul_backward_f16_i8_no_host(
    _grad: &CudaBuffer,
    _lhs: &CudaBuffer,
    _rhs: &CudaBuffer,
    _rhs_scale: f32,
    _batch_count: usize,
    _m: usize,
    _k: usize,
    _n: usize,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn batch_matmul_backward_f32_i8_no_host(
    _grad: &CudaBuffer,
    _lhs: &CudaBuffer,
    _rhs: &CudaBuffer,
    _rhs_scale: f32,
    _batch_count: usize,
    _m: usize,
    _k: usize,
    _n: usize,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn batch_matmul_backward_i8_bf16_no_host(
    _grad: &CudaBuffer,
    _lhs: &CudaBuffer,
    _lhs_scale: f32,
    _rhs: &CudaBuffer,
    _batch_count: usize,
    _m: usize,
    _k: usize,
    _n: usize,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn batch_matmul_backward_i8_f16_no_host(
    _grad: &CudaBuffer,
    _lhs: &CudaBuffer,
    _lhs_scale: f32,
    _rhs: &CudaBuffer,
    _batch_count: usize,
    _m: usize,
    _k: usize,
    _n: usize,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn batch_matmul_backward_i8_f32_no_host(
    _grad: &CudaBuffer,
    _lhs: &CudaBuffer,
    _lhs_scale: f32,
    _rhs: &CudaBuffer,
    _batch_count: usize,
    _m: usize,
    _k: usize,
    _n: usize,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    Err("CUDA feature is disabled".to_string())
}

#[allow(clippy::too_many_arguments)]
pub fn batch_matmul_backward_i8_no_host(
    _grad: &CudaBuffer,
    _lhs: &CudaBuffer,
    _lhs_scale: f32,
    _rhs: &CudaBuffer,
    _rhs_scale: f32,
    _batch_count: usize,
    _m: usize,
    _k: usize,
    _n: usize,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn unary_f32(_input: &CudaBuffer, _op: UnaryOp) -> Result<(CudaBuffer, Vec<f32>), String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn unary_f32_buffer(_input: &CudaBuffer, _op: UnaryOp) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn unary_f16_buffer(_input: &CudaBuffer, _op: UnaryOp) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn unary_f16_typed_output_buffer(
    _input: &CudaBuffer,
    _op: UnaryOp,
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn unary_bf16_buffer(_input: &CudaBuffer, _op: UnaryOp) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn unary_bf16_typed_output_buffer(
    _input: &CudaBuffer,
    _op: UnaryOp,
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn unary_i8_buffer(
    _input: &CudaBuffer,
    _scale: f32,
    _op: UnaryOp,
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn unary_i8_relu_typed_output_buffer(_input: &CudaBuffer) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn unary_backward_f32(
    _input: &CudaBuffer,
    _output: &CudaBuffer,
    _grad: &CudaBuffer,
    _op: UnaryOp,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn unary_backward_f32_buffer(
    _input: &CudaBuffer,
    _output: &CudaBuffer,
    _grad: &CudaBuffer,
    _op: UnaryOp,
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn unary_backward_f16_buffer(
    _input: &CudaBuffer,
    _output: &CudaBuffer,
    _grad: &CudaBuffer,
    _op: UnaryOp,
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn unary_backward_bf16_buffer(
    _input: &CudaBuffer,
    _output: &CudaBuffer,
    _grad: &CudaBuffer,
    _op: UnaryOp,
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn unary_backward_i8_buffer(
    _input: &CudaBuffer,
    _scale: f32,
    _output: &CudaBuffer,
    _grad: &CudaBuffer,
    _op: UnaryOp,
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn binary_f32(
    _lhs: &CudaBuffer,
    _rhs: &CudaBuffer,
    _op: BinaryOp,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn binary_f32_buffer(
    _lhs: &CudaBuffer,
    _rhs: &CudaBuffer,
    _op: BinaryOp,
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn binary_typed_buffer_no_host(
    _lhs: &CudaBuffer,
    _lhs_dtype: crate::precision::DType,
    _lhs_scale: Option<f32>,
    _rhs: &CudaBuffer,
    _rhs_dtype: crate::precision::DType,
    _rhs_scale: Option<f32>,
    _op: BinaryOp,
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn binary_lowp_typed_output_buffer_no_host(
    _lhs: &CudaBuffer,
    _rhs: &CudaBuffer,
    _dtype: crate::precision::DType,
    _op: BinaryOp,
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

#[allow(clippy::too_many_arguments)]
pub fn binary_typed_lastdim_buffer_no_host(
    _lhs: &CudaBuffer,
    _lhs_dtype: crate::precision::DType,
    _lhs_scale: Option<f32>,
    _rhs: &CudaBuffer,
    _rhs_dtype: crate::precision::DType,
    _rhs_scale: Option<f32>,
    _out_len: usize,
    _last_dim: usize,
    _vector_on_rhs: bool,
    _op: BinaryOp,
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

#[allow(clippy::too_many_arguments)]
pub fn binary_lowp_typed_lastdim_output_buffer_no_host(
    _lhs: &CudaBuffer,
    _rhs: &CudaBuffer,
    _out_len: usize,
    _last_dim: usize,
    _vector_on_rhs: bool,
    _dtype: crate::precision::DType,
    _op: BinaryOp,
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

#[allow(clippy::too_many_arguments)]
pub fn binary_typed_row_scalar_buffer_no_host(
    _lhs: &CudaBuffer,
    _lhs_dtype: crate::precision::DType,
    _lhs_scale: Option<f32>,
    _rhs: &CudaBuffer,
    _rhs_dtype: crate::precision::DType,
    _rhs_scale: Option<f32>,
    _rows: usize,
    _last_dim: usize,
    _scalar_on_rhs: bool,
    _op: BinaryOp,
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

#[allow(clippy::too_many_arguments)]
pub fn binary_lowp_typed_row_scalar_output_buffer_no_host(
    _lhs: &CudaBuffer,
    _rhs: &CudaBuffer,
    _rows: usize,
    _last_dim: usize,
    _scalar_on_rhs: bool,
    _dtype: crate::precision::DType,
    _op: BinaryOp,
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

#[allow(clippy::too_many_arguments)]
pub fn binary_i8_typed_row_scalar_output_buffer_no_host(
    _lhs: &CudaBuffer,
    _lhs_scale: f32,
    _rhs: &CudaBuffer,
    _rhs_scale: f32,
    _rows: usize,
    _last_dim: usize,
    _scalar_on_rhs: bool,
    _op: BinaryOp,
) -> Result<(CudaBuffer, f32), String> {
    Err("CUDA feature is disabled".to_string())
}

#[allow(clippy::too_many_arguments)]
pub fn binary_typed_broadcast_buffer_no_host(
    _lhs: &CudaBuffer,
    _lhs_dtype: crate::precision::DType,
    _lhs_scale: Option<f32>,
    _rhs: &CudaBuffer,
    _rhs_dtype: crate::precision::DType,
    _rhs_scale: Option<f32>,
    _lhs_shape: &[usize],
    _rhs_shape: &[usize],
    _out_shape: &[usize],
    _op: BinaryOp,
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

#[allow(clippy::too_many_arguments)]
pub fn binary_lowp_typed_broadcast_output_buffer_no_host(
    _lhs: &CudaBuffer,
    _rhs: &CudaBuffer,
    _lhs_shape: &[usize],
    _rhs_shape: &[usize],
    _out_shape: &[usize],
    _dtype: crate::precision::DType,
    _op: BinaryOp,
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

#[allow(clippy::too_many_arguments)]
pub fn binary_i8_typed_broadcast_output_buffer_no_host(
    _lhs: &CudaBuffer,
    _lhs_scale: f32,
    _rhs: &CudaBuffer,
    _rhs_scale: f32,
    _lhs_shape: &[usize],
    _rhs_shape: &[usize],
    _out_shape: &[usize],
    _op: BinaryOp,
) -> Result<(CudaBuffer, f32), String> {
    Err("CUDA feature is disabled".to_string())
}

#[allow(clippy::too_many_arguments)]
pub fn binary_typed_b1d_1h1_buffer_no_host(
    _lhs: &CudaBuffer,
    _lhs_dtype: crate::precision::DType,
    _lhs_scale: Option<f32>,
    _rhs: &CudaBuffer,
    _rhs_dtype: crate::precision::DType,
    _rhs_scale: Option<f32>,
    _batch: usize,
    _heads: usize,
    _dim: usize,
    _b1d_on_lhs: bool,
    _op: BinaryOp,
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

#[allow(clippy::too_many_arguments)]
pub fn binary_lowp_typed_b1d_1h1_output_buffer_no_host(
    _lhs: &CudaBuffer,
    _rhs: &CudaBuffer,
    _batch: usize,
    _heads: usize,
    _dim: usize,
    _b1d_on_lhs: bool,
    _dtype: crate::precision::DType,
    _op: BinaryOp,
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

#[allow(clippy::too_many_arguments)]
pub fn binary_i8_typed_b1d_1h1_output_buffer_no_host(
    _lhs: &CudaBuffer,
    _lhs_scale: f32,
    _rhs: &CudaBuffer,
    _rhs_scale: f32,
    _batch: usize,
    _heads: usize,
    _dim: usize,
    _b1d_on_lhs: bool,
    _op: BinaryOp,
) -> Result<(CudaBuffer, f32), String> {
    Err("CUDA feature is disabled".to_string())
}

#[allow(clippy::too_many_arguments)]
pub fn binary_typed_b1d_1hd_buffer_no_host(
    _lhs: &CudaBuffer,
    _lhs_dtype: crate::precision::DType,
    _lhs_scale: Option<f32>,
    _rhs: &CudaBuffer,
    _rhs_dtype: crate::precision::DType,
    _rhs_scale: Option<f32>,
    _batch: usize,
    _heads: usize,
    _dim: usize,
    _b1d_on_lhs: bool,
    _op: BinaryOp,
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

#[allow(clippy::too_many_arguments)]
pub fn binary_lowp_typed_b1d_1hd_output_buffer_no_host(
    _lhs: &CudaBuffer,
    _rhs: &CudaBuffer,
    _batch: usize,
    _heads: usize,
    _dim: usize,
    _b1d_on_lhs: bool,
    _dtype: crate::precision::DType,
    _op: BinaryOp,
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

#[allow(clippy::too_many_arguments)]
pub fn binary_i8_typed_b1d_1hd_output_buffer_no_host(
    _lhs: &CudaBuffer,
    _lhs_scale: f32,
    _rhs: &CudaBuffer,
    _rhs_scale: f32,
    _batch: usize,
    _heads: usize,
    _dim: usize,
    _b1d_on_lhs: bool,
    _op: BinaryOp,
) -> Result<(CudaBuffer, f32), String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn binary_f16_host_no_host(
    _lhs: &[u16],
    _rhs: &[u16],
    _op: BinaryOp,
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn binary_f16_buffer_no_host(
    _lhs: &CudaBuffer,
    _rhs: &CudaBuffer,
    _op: BinaryOp,
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn binary_f16_lastdim_buffer_no_host(
    _lhs: &CudaBuffer,
    _rhs: &CudaBuffer,
    _out_len: usize,
    _last_dim: usize,
    _vector_on_rhs: bool,
    _op: BinaryOp,
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn binary_bf16_host_no_host(
    _lhs: &[u16],
    _rhs: &[u16],
    _op: BinaryOp,
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn binary_bf16_buffer_no_host(
    _lhs: &CudaBuffer,
    _rhs: &CudaBuffer,
    _op: BinaryOp,
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn binary_bf16_lastdim_buffer_no_host(
    _lhs: &CudaBuffer,
    _rhs: &CudaBuffer,
    _out_len: usize,
    _last_dim: usize,
    _vector_on_rhs: bool,
    _op: BinaryOp,
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn binary_i8_host_no_host(
    _lhs: &[i8],
    _lhs_scale: f32,
    _rhs: &[i8],
    _rhs_scale: f32,
    _op: BinaryOp,
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn binary_i8_buffer_no_host(
    _lhs: &CudaBuffer,
    _lhs_scale: f32,
    _rhs: &CudaBuffer,
    _rhs_scale: f32,
    _op: BinaryOp,
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn binary_i8_typed_output_buffer_no_host(
    _lhs: &CudaBuffer,
    _lhs_scale: f32,
    _rhs: &CudaBuffer,
    _rhs_scale: f32,
    _op: BinaryOp,
) -> Result<(CudaBuffer, f32), String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn binary_i8_lastdim_buffer_no_host(
    _lhs: &CudaBuffer,
    _lhs_scale: f32,
    _rhs: &CudaBuffer,
    _rhs_scale: f32,
    _out_len: usize,
    _last_dim: usize,
    _vector_on_rhs: bool,
    _op: BinaryOp,
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn binary_i8_typed_lastdim_output_buffer_no_host(
    _lhs: &CudaBuffer,
    _lhs_scale: f32,
    _rhs: &CudaBuffer,
    _rhs_scale: f32,
    _out_len: usize,
    _last_dim: usize,
    _vector_on_rhs: bool,
    _op: BinaryOp,
) -> Result<(CudaBuffer, f32), String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn mul_grad_f16_host_no_host(
    _grad: &CudaBuffer,
    _operand: &[u16],
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn mul_grad_f16_buffer_no_host(
    _grad: &CudaBuffer,
    _operand: &CudaBuffer,
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn mul_grad_f16_lastdim_buffer_no_host(
    _grad: &CudaBuffer,
    _lhs: &CudaBuffer,
    _rhs: &CudaBuffer,
    _out_len: usize,
    _last_dim: usize,
    _vector_on_rhs: bool,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn mul_grad_bf16_host_no_host(
    _grad: &CudaBuffer,
    _operand: &[u16],
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn mul_grad_bf16_buffer_no_host(
    _grad: &CudaBuffer,
    _operand: &CudaBuffer,
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn mul_grad_bf16_lastdim_buffer_no_host(
    _grad: &CudaBuffer,
    _lhs: &CudaBuffer,
    _rhs: &CudaBuffer,
    _out_len: usize,
    _last_dim: usize,
    _vector_on_rhs: bool,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn mul_grad_i8_host_no_host(
    _grad: &CudaBuffer,
    _operand: &[i8],
    _scale: f32,
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn mul_grad_i8_buffer_no_host(
    _grad: &CudaBuffer,
    _operand: &CudaBuffer,
    _scale: f32,
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn mul_grad_i8_lastdim_buffer_no_host(
    _grad: &CudaBuffer,
    _lhs: &CudaBuffer,
    _lhs_scale: f32,
    _rhs: &CudaBuffer,
    _rhs_scale: f32,
    _out_len: usize,
    _last_dim: usize,
    _vector_on_rhs: bool,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    Err("CUDA feature is disabled".to_string())
}

#[allow(clippy::too_many_arguments)]
pub fn mul_grad_typed_lastdim_buffer_no_host(
    _grad: &CudaBuffer,
    _lhs: &CudaBuffer,
    _lhs_dtype: crate::precision::DType,
    _lhs_scale: Option<f32>,
    _rhs: &CudaBuffer,
    _rhs_dtype: crate::precision::DType,
    _rhs_scale: Option<f32>,
    _out_len: usize,
    _last_dim: usize,
    _vector_on_rhs: bool,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    Err("CUDA feature is disabled".to_string())
}

#[allow(clippy::too_many_arguments)]
pub fn mul_grad_typed_buffer_no_host(
    _grad: &CudaBuffer,
    _lhs: &CudaBuffer,
    _lhs_dtype: crate::precision::DType,
    _lhs_scale: Option<f32>,
    _rhs: &CudaBuffer,
    _rhs_dtype: crate::precision::DType,
    _rhs_scale: Option<f32>,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    Err("CUDA feature is disabled".to_string())
}

#[allow(clippy::too_many_arguments)]
pub fn mul_grad_typed_row_scalar_buffer_no_host(
    _grad: &CudaBuffer,
    _lhs: &CudaBuffer,
    _lhs_dtype: crate::precision::DType,
    _lhs_scale: Option<f32>,
    _rhs: &CudaBuffer,
    _rhs_dtype: crate::precision::DType,
    _rhs_scale: Option<f32>,
    _rows: usize,
    _last_dim: usize,
    _scalar_on_rhs: bool,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    Err("CUDA feature is disabled".to_string())
}

#[allow(clippy::too_many_arguments)]
pub fn mul_grad_typed_broadcast_buffer_no_host(
    _grad: &CudaBuffer,
    _lhs: &CudaBuffer,
    _lhs_dtype: crate::precision::DType,
    _lhs_scale: Option<f32>,
    _rhs: &CudaBuffer,
    _rhs_dtype: crate::precision::DType,
    _rhs_scale: Option<f32>,
    _lhs_shape: &[usize],
    _rhs_shape: &[usize],
    _out_shape: &[usize],
) -> Result<(CudaBuffer, CudaBuffer), String> {
    Err("CUDA feature is disabled".to_string())
}

#[allow(clippy::too_many_arguments)]
pub fn mul_grad_typed_b1d_1h1_buffer_no_host(
    _grad: &CudaBuffer,
    _lhs: &CudaBuffer,
    _lhs_dtype: crate::precision::DType,
    _lhs_scale: Option<f32>,
    _rhs: &CudaBuffer,
    _rhs_dtype: crate::precision::DType,
    _rhs_scale: Option<f32>,
    _batch: usize,
    _heads: usize,
    _dim: usize,
    _b1d_on_lhs: bool,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    Err("CUDA feature is disabled".to_string())
}

#[allow(clippy::too_many_arguments)]
pub fn mul_grad_typed_b1d_1hd_buffer_no_host(
    _grad: &CudaBuffer,
    _lhs: &CudaBuffer,
    _lhs_dtype: crate::precision::DType,
    _lhs_scale: Option<f32>,
    _rhs: &CudaBuffer,
    _rhs_dtype: crate::precision::DType,
    _rhs_scale: Option<f32>,
    _batch: usize,
    _heads: usize,
    _dim: usize,
    _b1d_on_lhs: bool,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    Err("CUDA feature is disabled".to_string())
}

#[allow(clippy::too_many_arguments)]
pub fn mul_grad_typed_scalar_buffer_no_host(
    _grad: &CudaBuffer,
    _lhs: &CudaBuffer,
    _lhs_dtype: crate::precision::DType,
    _lhs_scale: Option<f32>,
    _rhs: &CudaBuffer,
    _rhs_dtype: crate::precision::DType,
    _rhs_scale: Option<f32>,
    _out_len: usize,
    _scalar_on_rhs: bool,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn binary_backward_f32(
    _lhs: &CudaBuffer,
    _rhs: &CudaBuffer,
    _grad: &CudaBuffer,
    _op: BinaryOp,
) -> Result<CudaTwoHostBuffers, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn binary_backward_f32_buffers(
    _lhs: &CudaBuffer,
    _rhs: &CudaBuffer,
    _grad: &CudaBuffer,
    _op: BinaryOp,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn binary_broadcast_f32(
    _lhs: &CudaBuffer,
    _rhs: &CudaBuffer,
    _lhs_shape: &[usize],
    _rhs_shape: &[usize],
    _out_shape: &[usize],
    _op: BinaryOp,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn binary_broadcast_f32_buffer(
    _lhs: &CudaBuffer,
    _rhs: &CudaBuffer,
    _lhs_shape: &[usize],
    _rhs_shape: &[usize],
    _out_shape: &[usize],
    _op: BinaryOp,
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn binary_broadcast_backward_f32(
    _lhs: &CudaBuffer,
    _rhs: &CudaBuffer,
    _grad: &CudaBuffer,
    _lhs_shape: &[usize],
    _rhs_shape: &[usize],
    _out_shape: &[usize],
    _op: BinaryOp,
) -> Result<CudaTwoHostBuffers, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn binary_broadcast_backward_f32_buffers(
    _lhs: &CudaBuffer,
    _rhs: &CudaBuffer,
    _grad: &CudaBuffer,
    _lhs_shape: &[usize],
    _rhs_shape: &[usize],
    _out_shape: &[usize],
    _op: BinaryOp,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn add_sub_backward_f32_buffers(
    _grad: &CudaBuffer,
    _len: usize,
    _op: BinaryOp,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn add_sub_backward_lastdim_f32_buffers(
    _grad: &CudaBuffer,
    _out_len: usize,
    _last_dim: usize,
    _vector_on_rhs: bool,
    _op: BinaryOp,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn add_sub_backward_scalar_f32_buffers(
    _grad: &CudaBuffer,
    _out_len: usize,
    _scalar_on_rhs: bool,
    _op: BinaryOp,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn add_sub_backward_row_scalar_f32_buffers(
    _grad: &CudaBuffer,
    _rows: usize,
    _last_dim: usize,
    _scalar_on_rhs: bool,
    _op: BinaryOp,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn add_sub_backward_b1d_1h1_f32_buffers(
    _grad: &CudaBuffer,
    _batch: usize,
    _heads: usize,
    _dim: usize,
    _b1d_on_lhs: bool,
    _op: BinaryOp,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn add_sub_backward_b1d_1hd_f32_buffers(
    _grad: &CudaBuffer,
    _batch: usize,
    _heads: usize,
    _dim: usize,
    _b1d_on_lhs: bool,
    _op: BinaryOp,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn add_sub_broadcast_backward_f32_buffers(
    _grad: &CudaBuffer,
    _lhs_shape: &[usize],
    _rhs_shape: &[usize],
    _out_shape: &[usize],
    _op: BinaryOp,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn sum_f32(_input: &CudaBuffer) -> Result<(CudaBuffer, Vec<f32>), String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn sum_f16_buffer(_input: &CudaBuffer) -> Result<(CudaBuffer, Vec<f32>), String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn sum_bf16_buffer(_input: &CudaBuffer) -> Result<(CudaBuffer, Vec<f32>), String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn sum_i8_buffer(_input: &CudaBuffer, _scale: f32) -> Result<(CudaBuffer, Vec<f32>), String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn fill_scalar_f32(_len: usize, _value: f32) -> Result<(CudaBuffer, Vec<f32>), String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn fill_scalar_f32_buffer(_len: usize, _value: f32) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn add_inplace_f32(_dst: &CudaBuffer, _src: &CudaBuffer) -> Result<(), String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn sum_lastdim_f32_buffer(
    _input: &CudaBuffer,
    _rows: usize,
    _last_dim: usize,
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn bshd_to_bhsd_add_bias_f32_buffer(
    _input: &CudaBuffer,
    _bias: &CudaBuffer,
    _batch: usize,
    _seq: usize,
    _heads: usize,
    _dim: usize,
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn mse_forward_typed(
    _output: &CudaBuffer,
    _output_dtype: crate::precision::DType,
    _output_scale: Option<f32>,
    _target: &CudaBuffer,
    _target_dtype: crate::precision::DType,
    _target_scale: Option<f32>,
) -> Result<(CudaBuffer, CudaBuffer, Vec<f32>), String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn mse_backward_f32(_diff: &CudaBuffer, _factor: f32) -> Result<CudaTwoHostBuffers, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn mse_backward_f32_buffers(
    _diff: &CudaBuffer,
    _factor: f32,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn cross_entropy_backward_f32(
    _softmax: &CudaBuffer,
    _target: &CudaBuffer,
    _factor: f32,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn cross_entropy_backward_f32_buffer(
    _softmax: &CudaBuffer,
    _target: &CudaBuffer,
    _factor: f32,
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn cross_entropy_loss_f32(
    _softmax: &CudaBuffer,
    _target: &CudaBuffer,
    _batch_size: usize,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn cross_entropy_backward_typed_target_buffer(
    _softmax: &CudaBuffer,
    _target: &CudaBuffer,
    _target_dtype: crate::precision::DType,
    _target_scale: Option<f32>,
    _factor: f32,
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn cross_entropy_backward_typed_target(
    _softmax: &CudaBuffer,
    _target: &CudaBuffer,
    _target_dtype: crate::precision::DType,
    _target_scale: Option<f32>,
    _factor: f32,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn cross_entropy_loss_typed_target(
    _softmax: &CudaBuffer,
    _target: &CudaBuffer,
    _target_dtype: crate::precision::DType,
    _target_scale: Option<f32>,
    _batch_size: usize,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn sgd_update_f32(
    _param: &CudaBuffer,
    _grad: &CudaBuffer,
    _lr: f32,
) -> Result<Vec<f32>, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn sgd_update_f32_no_host(
    _param: &CudaBuffer,
    _grad: &CudaBuffer,
    _lr: f32,
) -> Result<(), String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn sgd_update_f32_batched_no_host(
    _params: &[CudaBuffer],
    _grads: &[CudaBuffer],
    _lr: f32,
) -> Result<(), String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn quantize_f32_storage_no_host(
    _param: &CudaBuffer,
    _dtype: crate::precision::DType,
    _scale: Option<f32>,
) -> Result<(), String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn quantize_f32_to_i8_dynamic_no_host(
    _input: &CudaBuffer,
) -> Result<(CudaBuffer, f32), String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn f32_to_lowp_storage_no_host(
    _input: &CudaBuffer,
    _dtype: crate::precision::DType,
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn sgd_momentum_update_f32(
    _param: &CudaBuffer,
    _grad: &CudaBuffer,
    _velocity: &CudaBuffer,
    _lr: f32,
    _momentum: f32,
) -> Result<(Vec<f32>, Vec<f32>), String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn sgd_momentum_update_f32_no_host(
    _param: &CudaBuffer,
    _grad: &CudaBuffer,
    _velocity: &CudaBuffer,
    _lr: f32,
    _momentum: f32,
) -> Result<(), String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn sgd_momentum_update_f32_batched_no_host(
    _params: &[CudaBuffer],
    _grads: &[CudaBuffer],
    _velocities: &[CudaBuffer],
    _lr: f32,
    _momentum: f32,
) -> Result<(), String> {
    Err("CUDA feature is disabled".to_string())
}

#[allow(clippy::too_many_arguments)]
pub fn adam_update_f32(
    _param: &CudaBuffer,
    _grad: &CudaBuffer,
    _exp_avg: &CudaBuffer,
    _exp_avg_sq: &CudaBuffer,
    _lr: f32,
    _beta1: f32,
    _beta2: f32,
    _bias_correction1: f32,
    _bias_correction2: f32,
    _eps: f32,
) -> Result<CudaAdamHostState, String> {
    Err("CUDA feature is disabled".to_string())
}

#[allow(clippy::too_many_arguments)]
pub fn adam_update_f32_no_host(
    _param: &CudaBuffer,
    _grad: &CudaBuffer,
    _exp_avg: &CudaBuffer,
    _exp_avg_sq: &CudaBuffer,
    _lr: f32,
    _beta1: f32,
    _beta2: f32,
    _bias_correction1: f32,
    _bias_correction2: f32,
    _eps: f32,
) -> Result<(), String> {
    Err("CUDA feature is disabled".to_string())
}

#[allow(clippy::too_many_arguments)]
pub fn adam_update_f32_batched_no_host(
    _params: &[CudaBuffer],
    _grads: &[CudaBuffer],
    _exp_avgs: &[CudaBuffer],
    _exp_avg_sqs: &[CudaBuffer],
    _lr: f32,
    _beta1: f32,
    _beta2: f32,
    _bias_correction1: f32,
    _bias_correction2: f32,
    _eps: f32,
) -> Result<(), String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn softmax_lastdim_f32(
    _input: &CudaBuffer,
    _outer: usize,
    _last_dim: usize,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn softmax_lastdim_f32_no_host(
    _input: &CudaBuffer,
    _outer: usize,
    _last_dim: usize,
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn softmax_lastdim_typed(
    _input: &CudaBuffer,
    _input_dtype: crate::precision::DType,
    _input_scale: Option<f32>,
    _outer: usize,
    _last_dim: usize,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn softmax_lastdim_typed_no_host(
    _input: &CudaBuffer,
    _input_dtype: crate::precision::DType,
    _input_scale: Option<f32>,
    _outer: usize,
    _last_dim: usize,
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn softmax_lastdim_backward_f32(
    _output: &CudaBuffer,
    _grad: &CudaBuffer,
    _outer: usize,
    _last_dim: usize,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn softmax_lastdim_backward_f32_buffer(
    _output: &CudaBuffer,
    _grad: &CudaBuffer,
    _outer: usize,
    _last_dim: usize,
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn fused_softmax_f32(
    _input: &CudaBuffer,
    _batch_heads: usize,
    _q_len: usize,
    _k_len: usize,
    _scale: f32,
    _is_causal: bool,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn fused_softmax_f32_no_host(
    _input: &CudaBuffer,
    _batch_heads: usize,
    _q_len: usize,
    _k_len: usize,
    _scale: f32,
    _is_causal: bool,
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn fused_softmax_backward_f32(
    _output: &CudaBuffer,
    _grad: &CudaBuffer,
    _batch_heads: usize,
    _q_len: usize,
    _k_len: usize,
    _scale: f32,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn fused_softmax_backward_f32_buffer(
    _output: &CudaBuffer,
    _grad: &CudaBuffer,
    _batch_heads: usize,
    _q_len: usize,
    _k_len: usize,
    _scale: f32,
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn fused_softmax_f32_with_past(
    _input: &CudaBuffer,
    _batch_heads: usize,
    _q_len: usize,
    _k_len: usize,
    _scale: f32,
    _is_causal: bool,
    _past_len: usize,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn fused_softmax_f32_with_past_no_host(
    _input: &CudaBuffer,
    _batch_heads: usize,
    _q_len: usize,
    _k_len: usize,
    _scale: f32,
    _is_causal: bool,
    _past_len: usize,
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn embedding_f32(
    _indices: &CudaBuffer,
    _weight: &CudaBuffer,
    _num_indices: usize,
    _vocab_size: usize,
    _embed_dim: usize,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn embedding_f32_buffer(
    _indices: &CudaBuffer,
    _weight: &CudaBuffer,
    _num_indices: usize,
    _vocab_size: usize,
    _embed_dim: usize,
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn embedding_typed(
    _indices: &CudaBuffer,
    _weight: &CudaBuffer,
    _weight_dtype: crate::precision::DType,
    _weight_scale: Option<f32>,
    _num_indices: usize,
    _vocab_size: usize,
    _embed_dim: usize,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn embedding_typed_buffer(
    _indices: &CudaBuffer,
    _weight: &CudaBuffer,
    _weight_dtype: crate::precision::DType,
    _weight_scale: Option<f32>,
    _num_indices: usize,
    _vocab_size: usize,
    _embed_dim: usize,
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn embedding_typed_same_dtype_buffer(
    _indices: &CudaBuffer,
    _weight: &CudaBuffer,
    _weight_dtype: crate::precision::DType,
    _num_indices: usize,
    _vocab_size: usize,
    _embed_dim: usize,
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn embedding_backward_f32(
    _indices: &CudaBuffer,
    _grad: &CudaBuffer,
    _num_indices: usize,
    _vocab_size: usize,
    _embed_dim: usize,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn embedding_backward_f32_buffer(
    _indices: &CudaBuffer,
    _grad: &CudaBuffer,
    _num_indices: usize,
    _vocab_size: usize,
    _embed_dim: usize,
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn rms_norm_f32(
    _input: &CudaBuffer,
    _weight: &CudaBuffer,
    _rows: usize,
    _dim: usize,
    _eps: f32,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn rms_norm_f32_buffer(
    _input: &CudaBuffer,
    _weight: &CudaBuffer,
    _rows: usize,
    _dim: usize,
    _eps: f32,
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

#[allow(clippy::too_many_arguments)]
pub fn rms_norm_typed(
    _input: &CudaBuffer,
    _input_dtype: crate::precision::DType,
    _input_scale: Option<f32>,
    _weight: &CudaBuffer,
    _weight_dtype: crate::precision::DType,
    _weight_scale: Option<f32>,
    _rows: usize,
    _dim: usize,
    _eps: f32,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    Err("CUDA feature is disabled".to_string())
}

#[allow(clippy::too_many_arguments)]
pub fn rms_norm_typed_buffer(
    _input: &CudaBuffer,
    _input_dtype: crate::precision::DType,
    _input_scale: Option<f32>,
    _weight: &CudaBuffer,
    _weight_dtype: crate::precision::DType,
    _weight_scale: Option<f32>,
    _rows: usize,
    _dim: usize,
    _eps: f32,
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

#[allow(clippy::too_many_arguments)]
pub fn rms_norm_i8_typed_output_buffer(
    _input: &CudaBuffer,
    _input_dtype: crate::precision::DType,
    _input_scale: Option<f32>,
    _weight: &CudaBuffer,
    _weight_dtype: crate::precision::DType,
    _weight_scale: Option<f32>,
    _rows: usize,
    _dim: usize,
    _eps: f32,
) -> Result<(CudaBuffer, f32), String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn rms_norm_backward_f32(
    _input: &CudaBuffer,
    _weight: &CudaBuffer,
    _grad: &CudaBuffer,
    _rows: usize,
    _dim: usize,
    _eps: f32,
) -> Result<CudaTwoHostBuffers, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn rms_norm_backward_f32_buffers(
    _input: &CudaBuffer,
    _weight: &CudaBuffer,
    _grad: &CudaBuffer,
    _rows: usize,
    _dim: usize,
    _eps: f32,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    Err("CUDA feature is disabled".to_string())
}

#[allow(clippy::too_many_arguments)]
pub fn rms_norm_backward_typed(
    _input: &CudaBuffer,
    _input_dtype: crate::precision::DType,
    _input_scale: Option<f32>,
    _weight: &CudaBuffer,
    _weight_dtype: crate::precision::DType,
    _weight_scale: Option<f32>,
    _grad: &CudaBuffer,
    _rows: usize,
    _dim: usize,
    _eps: f32,
) -> Result<CudaTwoHostBuffers, String> {
    Err("CUDA feature is disabled".to_string())
}

#[allow(clippy::too_many_arguments)]
pub fn rms_norm_backward_typed_buffers(
    _input: &CudaBuffer,
    _input_dtype: crate::precision::DType,
    _input_scale: Option<f32>,
    _weight: &CudaBuffer,
    _weight_dtype: crate::precision::DType,
    _weight_scale: Option<f32>,
    _grad: &CudaBuffer,
    _rows: usize,
    _dim: usize,
    _eps: f32,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn permute_f32(
    _input: &CudaBuffer,
    _out_shape: &[usize],
    _axes: &[usize],
) -> Result<(CudaBuffer, Vec<f32>), String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn permute_f32_buffer(
    _input: &CudaBuffer,
    _out_shape: &[usize],
    _axes: &[usize],
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn permute_typed_buffer(
    _input: &CudaBuffer,
    _dtype: crate::precision::DType,
    _out_shape: &[usize],
    _axes: &[usize],
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn slice_lastdim_f32(
    _input: &CudaBuffer,
    _outer: usize,
    _input_last_dim: usize,
    _start: usize,
    _slice_len: usize,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn slice_lastdim_f32_buffer(
    _input: &CudaBuffer,
    _outer: usize,
    _input_last_dim: usize,
    _start: usize,
    _slice_len: usize,
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn slice_lastdim_typed_buffer(
    _input: &CudaBuffer,
    _dtype: crate::precision::DType,
    _outer: usize,
    _input_last_dim: usize,
    _start: usize,
    _slice_len: usize,
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn slice_lastdim_backward_f32(
    _grad: &CudaBuffer,
    _outer: usize,
    _input_last_dim: usize,
    _start: usize,
    _slice_len: usize,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn slice_lastdim_backward_f32_buffer(
    _grad: &CudaBuffer,
    _outer: usize,
    _input_last_dim: usize,
    _start: usize,
    _slice_len: usize,
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn cat_f32(
    _lhs: &CudaBuffer,
    _rhs: &CudaBuffer,
    _out_shape: &[usize],
    _axis: usize,
    _lhs_axis_len: usize,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn cat_f32_buffer(
    _lhs: &CudaBuffer,
    _rhs: &CudaBuffer,
    _out_shape: &[usize],
    _axis: usize,
    _lhs_axis_len: usize,
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn cat_typed_buffer(
    _lhs: &CudaBuffer,
    _rhs: &CudaBuffer,
    _dtype: crate::precision::DType,
    _out_shape: &[usize],
    _axis: usize,
    _lhs_axis_len: usize,
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn cat_backward_slice_f32(
    _grad: &CudaBuffer,
    _input_shape: &[usize],
    _out_shape: &[usize],
    _axis: usize,
    _axis_start: usize,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn cat_backward_slice_f32_buffer(
    _grad: &CudaBuffer,
    _input_shape: &[usize],
    _out_shape: &[usize],
    _axis: usize,
    _axis_start: usize,
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn repeat_kv_f32(
    _input: &CudaBuffer,
    _batch_size: usize,
    _num_kv_heads: usize,
    _seq_len: usize,
    _dim: usize,
    _n_rep: usize,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn repeat_kv_f32_buffer(
    _input: &CudaBuffer,
    _batch_size: usize,
    _num_kv_heads: usize,
    _seq_len: usize,
    _dim: usize,
    _n_rep: usize,
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn repeat_kv_typed_buffer(
    _input: &CudaBuffer,
    _dtype: crate::precision::DType,
    _batch_size: usize,
    _num_kv_heads: usize,
    _seq_len: usize,
    _dim: usize,
    _n_rep: usize,
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn repeat_kv_backward_f32(
    _grad: &CudaBuffer,
    _batch_size: usize,
    _num_kv_heads: usize,
    _seq_len: usize,
    _dim: usize,
    _n_rep: usize,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn repeat_kv_backward_f32_buffer(
    _grad: &CudaBuffer,
    _batch_size: usize,
    _num_kv_heads: usize,
    _seq_len: usize,
    _dim: usize,
    _n_rep: usize,
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

#[allow(clippy::too_many_arguments)]
pub fn decode_attention_f32(
    _q: &CudaBuffer,
    _k: &CudaBuffer,
    _v: &CudaBuffer,
    _batch_size: usize,
    _num_heads: usize,
    _num_kv_heads: usize,
    _active_seq_len: usize,
    _cache_seq_len: usize,
    _dim: usize,
    _n_rep: usize,
    _scale: f32,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    Err("CUDA feature is disabled".to_string())
}

#[allow(clippy::too_many_arguments)]
pub fn decode_attention_f32_buffer(
    _q: &CudaBuffer,
    _k: &CudaBuffer,
    _v: &CudaBuffer,
    _batch_size: usize,
    _num_heads: usize,
    _num_kv_heads: usize,
    _active_seq_len: usize,
    _cache_seq_len: usize,
    _dim: usize,
    _n_rep: usize,
    _scale: f32,
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn fused_gate_up_silu_f32(
    _input: &CudaBuffer,
    _gate: &CudaBuffer,
    _up: &CudaBuffer,
    _rows: usize,
    _n_dim: usize,
    _k_dim: usize,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn fused_gate_up_silu_f32_buffer(
    _input: &CudaBuffer,
    _gate: &CudaBuffer,
    _up: &CudaBuffer,
    _rows: usize,
    _n_dim: usize,
    _k_dim: usize,
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn silu_mul_f32_buffer_no_host(
    _gate: &CudaBuffer,
    _up: &CudaBuffer,
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

#[allow(clippy::too_many_arguments)]
pub fn fused_gate_up_silu_typed(
    _input: &CudaBuffer,
    _input_dtype: crate::precision::DType,
    _input_scale: Option<f32>,
    _gate: &CudaBuffer,
    _weight_dtype: crate::precision::DType,
    _gate_scale: Option<f32>,
    _up: &CudaBuffer,
    _up_scale: Option<f32>,
    _rows: usize,
    _n_dim: usize,
    _k_dim: usize,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    Err("CUDA feature is disabled".to_string())
}

#[allow(clippy::too_many_arguments)]
pub fn fused_gate_up_silu_typed_buffer(
    _input: &CudaBuffer,
    _input_dtype: crate::precision::DType,
    _input_scale: Option<f32>,
    _gate: &CudaBuffer,
    _weight_dtype: crate::precision::DType,
    _gate_scale: Option<f32>,
    _up: &CudaBuffer,
    _up_scale: Option<f32>,
    _rows: usize,
    _n_dim: usize,
    _k_dim: usize,
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

#[allow(clippy::too_many_arguments)]
pub fn fused_gate_up_silu_typed_output_buffer(
    _input: &CudaBuffer,
    _input_dtype: crate::precision::DType,
    _input_scale: Option<f32>,
    _gate: &CudaBuffer,
    _weight_dtype: crate::precision::DType,
    _gate_scale: Option<f32>,
    _up: &CudaBuffer,
    _up_scale: Option<f32>,
    _output_dtype: crate::precision::DType,
    _rows: usize,
    _n_dim: usize,
    _k_dim: usize,
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn fused_qkv_f32(
    _input: &CudaBuffer,
    _q: &CudaBuffer,
    _k: &CudaBuffer,
    _v: &CudaBuffer,
    _rows: usize,
    _q_n: usize,
    _k_n: usize,
    _k_dim: usize,
) -> Result<CudaThreeHostBuffers, String> {
    Err("CUDA feature is disabled".to_string())
}

pub fn fused_qkv_f32_buffer(
    _input: &CudaBuffer,
    _q: &CudaBuffer,
    _k: &CudaBuffer,
    _v: &CudaBuffer,
    _rows: usize,
    _q_n: usize,
    _k_n: usize,
    _k_dim: usize,
) -> Result<(CudaBuffer, CudaBuffer, CudaBuffer), String> {
    Err("CUDA feature is disabled".to_string())
}

#[allow(clippy::too_many_arguments)]
pub fn fused_qkv_typed_buffer(
    _input: &CudaBuffer,
    _input_dtype: crate::precision::DType,
    _input_scale: Option<f32>,
    _q: &CudaBuffer,
    _k: &CudaBuffer,
    _v: &CudaBuffer,
    _weight_dtype: crate::precision::DType,
    _q_scale: Option<f32>,
    _k_scale: Option<f32>,
    _v_scale: Option<f32>,
    _rows: usize,
    _q_n: usize,
    _k_n: usize,
    _k_dim: usize,
) -> Result<(CudaBuffer, CudaBuffer, CudaBuffer), String> {
    Err("CUDA feature is disabled".to_string())
}

#[allow(clippy::too_many_arguments)]
pub fn fused_qkv_typed_output_buffer(
    _input: &CudaBuffer,
    _input_dtype: crate::precision::DType,
    _input_scale: Option<f32>,
    _q: &CudaBuffer,
    _k: &CudaBuffer,
    _v: &CudaBuffer,
    _weight_dtype: crate::precision::DType,
    _q_scale: Option<f32>,
    _k_scale: Option<f32>,
    _v_scale: Option<f32>,
    _output_dtype: crate::precision::DType,
    _rows: usize,
    _q_n: usize,
    _k_n: usize,
    _k_dim: usize,
) -> Result<(CudaBuffer, CudaBuffer, CudaBuffer), String> {
    Err("CUDA feature is disabled".to_string())
}

#[allow(clippy::too_many_arguments)]
pub fn rope_f32(
    _input: &CudaBuffer,
    _cos: &CudaBuffer,
    _sin: &CudaBuffer,
    _batch_size: usize,
    _num_heads: usize,
    _seq_len: usize,
    _dim: usize,
    _offset: usize,
    _cache_seq_len: usize,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    Err("CUDA feature is disabled".to_string())
}

#[allow(clippy::too_many_arguments)]
pub fn rope_f32_buffer(
    _input: &CudaBuffer,
    _cos: &CudaBuffer,
    _sin: &CudaBuffer,
    _batch_size: usize,
    _num_heads: usize,
    _seq_len: usize,
    _dim: usize,
    _offset: usize,
    _cache_seq_len: usize,
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

#[allow(clippy::too_many_arguments)]
pub fn rope_typed_buffer(
    _input: &CudaBuffer,
    _input_dtype: crate::precision::DType,
    _input_scale: Option<f32>,
    _cos: &CudaBuffer,
    _sin: &CudaBuffer,
    _cache_dtype: crate::precision::DType,
    _cos_scale: Option<f32>,
    _sin_scale: Option<f32>,
    _batch_size: usize,
    _num_heads: usize,
    _seq_len: usize,
    _dim: usize,
    _offset: usize,
    _cache_seq_len: usize,
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

#[allow(clippy::too_many_arguments)]
pub fn rope_typed_i8_dynamic_buffer(
    _input: &CudaBuffer,
    _input_dtype: crate::precision::DType,
    _input_scale: Option<f32>,
    _cos: &CudaBuffer,
    _sin: &CudaBuffer,
    _cache_dtype: crate::precision::DType,
    _cos_scale: Option<f32>,
    _sin_scale: Option<f32>,
    _batch_size: usize,
    _num_heads: usize,
    _seq_len: usize,
    _dim: usize,
    _offset: usize,
    _cache_seq_len: usize,
) -> Result<(CudaBuffer, f32), String> {
    Err("CUDA feature is disabled".to_string())
}

#[allow(clippy::too_many_arguments)]
pub fn rope_backward_f32(
    _grad: &CudaBuffer,
    _cos: &CudaBuffer,
    _sin: &CudaBuffer,
    _batch_size: usize,
    _num_heads: usize,
    _seq_len: usize,
    _dim: usize,
    _offset: usize,
    _cache_seq_len: usize,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    Err("CUDA feature is disabled".to_string())
}

#[allow(clippy::too_many_arguments)]
pub fn rope_backward_f32_buffer(
    _grad: &CudaBuffer,
    _cos: &CudaBuffer,
    _sin: &CudaBuffer,
    _batch_size: usize,
    _num_heads: usize,
    _seq_len: usize,
    _dim: usize,
    _offset: usize,
    _cache_seq_len: usize,
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

#[allow(clippy::too_many_arguments)]
pub fn conv2d_f32(
    _input: &CudaBuffer,
    _weight: &CudaBuffer,
    _bias: Option<&CudaBuffer>,
    _batch_size: usize,
    _in_channels: usize,
    _in_h: usize,
    _in_w: usize,
    _out_channels: usize,
    _k_h: usize,
    _k_w: usize,
    _pad_h: usize,
    _pad_w: usize,
    _stride_h: usize,
    _stride_w: usize,
) -> Result<(CudaBuffer, Vec<f32>, usize, usize), String> {
    Err("CUDA feature is disabled".to_string())
}

#[allow(clippy::too_many_arguments)]
pub fn conv2d_f32_buffer(
    _input: &CudaBuffer,
    _weight: &CudaBuffer,
    _bias: Option<&CudaBuffer>,
    _batch_size: usize,
    _in_channels: usize,
    _in_h: usize,
    _in_w: usize,
    _out_channels: usize,
    _k_h: usize,
    _k_w: usize,
    _pad_h: usize,
    _pad_w: usize,
    _stride_h: usize,
    _stride_w: usize,
) -> Result<(CudaBuffer, usize, usize), String> {
    Err("CUDA feature is disabled".to_string())
}

#[allow(clippy::too_many_arguments)]
pub fn conv2d_typed(
    _input: &CudaBuffer,
    _input_dtype: crate::precision::DType,
    _input_scale: Option<f32>,
    _weight: &CudaBuffer,
    _weight_dtype: crate::precision::DType,
    _weight_scale: Option<f32>,
    _bias: Option<(&CudaBuffer, crate::precision::DType, Option<f32>)>,
    _batch_size: usize,
    _in_channels: usize,
    _in_h: usize,
    _in_w: usize,
    _out_channels: usize,
    _k_h: usize,
    _k_w: usize,
    _pad_h: usize,
    _pad_w: usize,
    _stride_h: usize,
    _stride_w: usize,
) -> Result<(CudaBuffer, Vec<f32>, usize, usize), String> {
    Err("CUDA feature is disabled".to_string())
}

#[allow(clippy::too_many_arguments)]
pub fn conv2d_typed_buffer(
    _input: &CudaBuffer,
    _input_dtype: crate::precision::DType,
    _input_scale: Option<f32>,
    _weight: &CudaBuffer,
    _weight_dtype: crate::precision::DType,
    _weight_scale: Option<f32>,
    _bias: Option<(&CudaBuffer, crate::precision::DType, Option<f32>)>,
    _batch_size: usize,
    _in_channels: usize,
    _in_h: usize,
    _in_w: usize,
    _out_channels: usize,
    _k_h: usize,
    _k_w: usize,
    _pad_h: usize,
    _pad_w: usize,
    _stride_h: usize,
    _stride_w: usize,
) -> Result<(CudaBuffer, usize, usize), String> {
    Err("CUDA feature is disabled".to_string())
}

#[allow(clippy::too_many_arguments)]
pub fn conv2d_backward_f32(
    _input: &CudaBuffer,
    _weight: &CudaBuffer,
    _grad_output: &CudaBuffer,
    _compute_bias_grad: bool,
    _batch_size: usize,
    _in_channels: usize,
    _in_h: usize,
    _in_w: usize,
    _out_channels: usize,
    _k_h: usize,
    _k_w: usize,
    _pad_h: usize,
    _pad_w: usize,
    _stride_h: usize,
    _stride_w: usize,
) -> Result<CudaConv2dBackwardHostBuffers, String> {
    Err("CUDA feature is disabled".to_string())
}

#[allow(clippy::too_many_arguments)]
pub fn conv2d_backward_f32_buffers(
    _input: &CudaBuffer,
    _weight: &CudaBuffer,
    _grad_output: &CudaBuffer,
    _compute_bias_grad: bool,
    _batch_size: usize,
    _in_channels: usize,
    _in_h: usize,
    _in_w: usize,
    _out_channels: usize,
    _k_h: usize,
    _k_w: usize,
    _pad_h: usize,
    _pad_w: usize,
    _stride_h: usize,
    _stride_w: usize,
) -> Result<(CudaBuffer, CudaBuffer, Option<CudaBuffer>), String> {
    Err("CUDA feature is disabled".to_string())
}

#[allow(clippy::too_many_arguments)]
pub fn max_pool2d_f32(
    _input: &CudaBuffer,
    _batch_size: usize,
    _channels: usize,
    _in_h: usize,
    _in_w: usize,
    _kernel_h: usize,
    _kernel_w: usize,
    _stride_h: usize,
    _stride_w: usize,
) -> Result<(CudaBuffer, Vec<f32>, usize, usize), String> {
    Err("CUDA feature is disabled".to_string())
}

#[allow(clippy::too_many_arguments)]
pub fn max_pool2d_typed(
    _input: &CudaBuffer,
    _input_dtype: crate::precision::DType,
    _input_scale: Option<f32>,
    _batch_size: usize,
    _channels: usize,
    _in_h: usize,
    _in_w: usize,
    _kernel_h: usize,
    _kernel_w: usize,
    _stride_h: usize,
    _stride_w: usize,
) -> Result<(CudaBuffer, Vec<f32>, usize, usize), String> {
    Err("CUDA feature is disabled".to_string())
}

#[allow(clippy::too_many_arguments)]
pub fn max_pool2d_backward_f32(
    _input: &CudaBuffer,
    _grad_output: &CudaBuffer,
    _batch_size: usize,
    _channels: usize,
    _in_h: usize,
    _in_w: usize,
    _kernel_h: usize,
    _kernel_w: usize,
    _stride_h: usize,
    _stride_w: usize,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    Err("CUDA feature is disabled".to_string())
}

#[allow(clippy::too_many_arguments)]
pub fn max_pool2d_backward_f32_buffer(
    _input: &CudaBuffer,
    _grad_output: &CudaBuffer,
    _batch_size: usize,
    _channels: usize,
    _in_h: usize,
    _in_w: usize,
    _kernel_h: usize,
    _kernel_w: usize,
    _stride_h: usize,
    _stride_w: usize,
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}

#[allow(clippy::too_many_arguments)]
pub fn max_pool2d_backward_typed_buffer(
    _input: &CudaBuffer,
    _input_dtype: crate::precision::DType,
    _input_scale: Option<f32>,
    _grad_output: &CudaBuffer,
    _batch_size: usize,
    _channels: usize,
    _in_h: usize,
    _in_w: usize,
    _kernel_h: usize,
    _kernel_w: usize,
    _stride_h: usize,
    _stride_w: usize,
) -> Result<CudaBuffer, String> {
    Err("CUDA feature is disabled".to_string())
}
