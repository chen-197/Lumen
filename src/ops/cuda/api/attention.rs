use super::*;

#[allow(clippy::too_many_arguments)]
pub fn append_kv_cache_f32(
    dst: &CudaBuffer,
    src: &CudaBuffer,
    batch_size: usize,
    num_heads: usize,
    src_seq_len: usize,
    dst_seq_len: usize,
    dim: usize,
    dst_start: usize,
) -> Result<(), String> {
    imp::append_kv_cache_f32(
        dst,
        src,
        batch_size,
        num_heads,
        src_seq_len,
        dst_seq_len,
        dim,
        dst_start,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn append_kv_cache_pair_f32(
    k_dst: &CudaBuffer,
    v_dst: &CudaBuffer,
    k_src: &CudaBuffer,
    v_src: &CudaBuffer,
    batch_size: usize,
    num_heads: usize,
    src_seq_len: usize,
    dst_seq_len: usize,
    dim: usize,
    dst_start: usize,
) -> Result<(), String> {
    imp::append_kv_cache_pair_f32(
        k_dst,
        v_dst,
        k_src,
        v_src,
        batch_size,
        num_heads,
        src_seq_len,
        dst_seq_len,
        dim,
        dst_start,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn decode_rope_q_append_kv_f32_buffer(
    q_src: &CudaBuffer,
    k_src: &CudaBuffer,
    v_src: &CudaBuffer,
    cos: &CudaBuffer,
    sin: &CudaBuffer,
    k_cache: &CudaBuffer,
    v_cache: &CudaBuffer,
    batch_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    dim: usize,
    dst_seq_len: usize,
    offset: usize,
    cache_seq_len: usize,
) -> Result<CudaBuffer, String> {
    imp::decode_rope_q_append_kv_f32_buffer(
        q_src,
        k_src,
        v_src,
        cos,
        sin,
        k_cache,
        v_cache,
        batch_size,
        num_heads,
        num_kv_heads,
        dim,
        dst_seq_len,
        offset,
        cache_seq_len,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn prefill_attention_f32_buffer(
    q: &CudaBuffer,
    k: &CudaBuffer,
    v: &CudaBuffer,
    batch_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    q_seq_len: usize,
    active_seq_len: usize,
    cache_seq_len: usize,
    dim: usize,
    n_rep: usize,
    past_len: usize,
    scale: f32,
    is_causal: bool,
) -> Result<CudaBuffer, String> {
    imp::prefill_attention_f32_buffer(
        q,
        k,
        v,
        batch_size,
        num_heads,
        num_kv_heads,
        q_seq_len,
        active_seq_len,
        cache_seq_len,
        dim,
        n_rep,
        past_len,
        scale,
        is_causal,
    )
}

pub fn kv_cache_prefix_f32_buffer(
    src: &CudaBuffer,
    batch_size: usize,
    num_heads: usize,
    active_seq_len: usize,
    src_seq_len: usize,
    dim: usize,
) -> Result<CudaBuffer, String> {
    imp::kv_cache_prefix_f32_buffer(src, batch_size, num_heads, active_seq_len, src_seq_len, dim)
}

#[allow(clippy::too_many_arguments)]
pub fn kv_cache_prefix_typed_buffer(
    src: &CudaBuffer,
    dtype: crate::precision::DType,
    batch_size: usize,
    num_heads: usize,
    active_seq_len: usize,
    src_seq_len: usize,
    dim: usize,
) -> Result<CudaBuffer, String> {
    imp::kv_cache_prefix_typed_buffer(
        src,
        dtype,
        batch_size,
        num_heads,
        active_seq_len,
        src_seq_len,
        dim,
    )
}

pub fn repeat_kv_f32(
    input: &CudaBuffer,
    batch_size: usize,
    num_kv_heads: usize,
    seq_len: usize,
    dim: usize,
    n_rep: usize,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    imp::repeat_kv_f32(input, batch_size, num_kv_heads, seq_len, dim, n_rep)
}

pub fn repeat_kv_f32_buffer(
    input: &CudaBuffer,
    batch_size: usize,
    num_kv_heads: usize,
    seq_len: usize,
    dim: usize,
    n_rep: usize,
) -> Result<CudaBuffer, String> {
    imp::repeat_kv_f32_buffer(input, batch_size, num_kv_heads, seq_len, dim, n_rep)
}

pub fn repeat_kv_typed_buffer(
    input: &CudaBuffer,
    dtype: crate::precision::DType,
    batch_size: usize,
    num_kv_heads: usize,
    seq_len: usize,
    dim: usize,
    n_rep: usize,
) -> Result<CudaBuffer, String> {
    imp::repeat_kv_typed_buffer(input, dtype, batch_size, num_kv_heads, seq_len, dim, n_rep)
}

pub fn repeat_kv_backward_f32(
    grad: &CudaBuffer,
    batch_size: usize,
    num_kv_heads: usize,
    seq_len: usize,
    dim: usize,
    n_rep: usize,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    imp::repeat_kv_backward_f32(grad, batch_size, num_kv_heads, seq_len, dim, n_rep)
}

pub fn repeat_kv_backward_f32_buffer(
    grad: &CudaBuffer,
    batch_size: usize,
    num_kv_heads: usize,
    seq_len: usize,
    dim: usize,
    n_rep: usize,
) -> Result<CudaBuffer, String> {
    imp::repeat_kv_backward_f32_buffer(grad, batch_size, num_kv_heads, seq_len, dim, n_rep)
}

#[allow(clippy::too_many_arguments)]
pub fn decode_attention_f32(
    q: &CudaBuffer,
    k: &CudaBuffer,
    v: &CudaBuffer,
    batch_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    active_seq_len: usize,
    cache_seq_len: usize,
    dim: usize,
    n_rep: usize,
    scale: f32,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    imp::decode_attention_f32(
        q,
        k,
        v,
        batch_size,
        num_heads,
        num_kv_heads,
        active_seq_len,
        cache_seq_len,
        dim,
        n_rep,
        scale,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn decode_attention_f32_buffer(
    q: &CudaBuffer,
    k: &CudaBuffer,
    v: &CudaBuffer,
    batch_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    active_seq_len: usize,
    cache_seq_len: usize,
    dim: usize,
    n_rep: usize,
    scale: f32,
) -> Result<CudaBuffer, String> {
    imp::decode_attention_f32_buffer(
        q,
        k,
        v,
        batch_size,
        num_heads,
        num_kv_heads,
        active_seq_len,
        cache_seq_len,
        dim,
        n_rep,
        scale,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn rope_f32(
    input: &CudaBuffer,
    cos: &CudaBuffer,
    sin: &CudaBuffer,
    batch_size: usize,
    num_heads: usize,
    seq_len: usize,
    dim: usize,
    offset: usize,
    cache_seq_len: usize,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    imp::rope_f32(
        input,
        cos,
        sin,
        batch_size,
        num_heads,
        seq_len,
        dim,
        offset,
        cache_seq_len,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn rope_f32_buffer(
    input: &CudaBuffer,
    cos: &CudaBuffer,
    sin: &CudaBuffer,
    batch_size: usize,
    num_heads: usize,
    seq_len: usize,
    dim: usize,
    offset: usize,
    cache_seq_len: usize,
) -> Result<CudaBuffer, String> {
    imp::rope_f32_buffer(
        input,
        cos,
        sin,
        batch_size,
        num_heads,
        seq_len,
        dim,
        offset,
        cache_seq_len,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn rope_typed_buffer(
    input: &CudaBuffer,
    input_dtype: crate::precision::DType,
    input_scale: Option<f32>,
    cos: &CudaBuffer,
    sin: &CudaBuffer,
    cache_dtype: crate::precision::DType,
    cos_scale: Option<f32>,
    sin_scale: Option<f32>,
    batch_size: usize,
    num_heads: usize,
    seq_len: usize,
    dim: usize,
    offset: usize,
    cache_seq_len: usize,
) -> Result<CudaBuffer, String> {
    imp::rope_typed_buffer(
        input,
        input_dtype,
        input_scale,
        cos,
        sin,
        cache_dtype,
        cos_scale,
        sin_scale,
        batch_size,
        num_heads,
        seq_len,
        dim,
        offset,
        cache_seq_len,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn rope_typed_i8_dynamic_buffer(
    input: &CudaBuffer,
    input_dtype: crate::precision::DType,
    input_scale: Option<f32>,
    cos: &CudaBuffer,
    sin: &CudaBuffer,
    cache_dtype: crate::precision::DType,
    cos_scale: Option<f32>,
    sin_scale: Option<f32>,
    batch_size: usize,
    num_heads: usize,
    seq_len: usize,
    dim: usize,
    offset: usize,
    cache_seq_len: usize,
) -> Result<(CudaBuffer, f32), String> {
    imp::rope_typed_i8_dynamic_buffer(
        input,
        input_dtype,
        input_scale,
        cos,
        sin,
        cache_dtype,
        cos_scale,
        sin_scale,
        batch_size,
        num_heads,
        seq_len,
        dim,
        offset,
        cache_seq_len,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn rope_backward_f32(
    grad: &CudaBuffer,
    cos: &CudaBuffer,
    sin: &CudaBuffer,
    batch_size: usize,
    num_heads: usize,
    seq_len: usize,
    dim: usize,
    offset: usize,
    cache_seq_len: usize,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    imp::rope_backward_f32(
        grad,
        cos,
        sin,
        batch_size,
        num_heads,
        seq_len,
        dim,
        offset,
        cache_seq_len,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn rope_backward_f32_buffer(
    grad: &CudaBuffer,
    cos: &CudaBuffer,
    sin: &CudaBuffer,
    batch_size: usize,
    num_heads: usize,
    seq_len: usize,
    dim: usize,
    offset: usize,
    cache_seq_len: usize,
) -> Result<CudaBuffer, String> {
    imp::rope_backward_f32_buffer(
        grad,
        cos,
        sin,
        batch_size,
        num_heads,
        seq_len,
        dim,
        offset,
        cache_seq_len,
    )
}
