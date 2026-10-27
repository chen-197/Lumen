use super::*;

pub fn softmax_lastdim_f32(
    input: &CudaBuffer,
    outer: usize,
    last_dim: usize,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    imp::softmax_lastdim_f32(input, outer, last_dim)
}

pub fn softmax_lastdim_f32_no_host(
    input: &CudaBuffer,
    outer: usize,
    last_dim: usize,
) -> Result<CudaBuffer, String> {
    imp::softmax_lastdim_f32_no_host(input, outer, last_dim)
}

pub fn softmax_lastdim_typed(
    input: &CudaBuffer,
    input_dtype: crate::precision::DType,
    input_scale: Option<f32>,
    outer: usize,
    last_dim: usize,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    imp::softmax_lastdim_typed(input, input_dtype, input_scale, outer, last_dim)
}

pub fn softmax_lastdim_typed_no_host(
    input: &CudaBuffer,
    input_dtype: crate::precision::DType,
    input_scale: Option<f32>,
    outer: usize,
    last_dim: usize,
) -> Result<CudaBuffer, String> {
    imp::softmax_lastdim_typed_no_host(input, input_dtype, input_scale, outer, last_dim)
}

pub fn softmax_lastdim_backward_f32(
    output: &CudaBuffer,
    grad: &CudaBuffer,
    outer: usize,
    last_dim: usize,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    imp::softmax_lastdim_backward_f32(output, grad, outer, last_dim)
}

pub fn softmax_lastdim_backward_f32_buffer(
    output: &CudaBuffer,
    grad: &CudaBuffer,
    outer: usize,
    last_dim: usize,
) -> Result<CudaBuffer, String> {
    imp::softmax_lastdim_backward_f32_buffer(output, grad, outer, last_dim)
}

pub fn fused_softmax_f32(
    input: &CudaBuffer,
    batch_heads: usize,
    q_len: usize,
    k_len: usize,
    scale: f32,
    is_causal: bool,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    imp::fused_softmax_f32(input, batch_heads, q_len, k_len, scale, is_causal)
}

pub fn fused_softmax_f32_no_host(
    input: &CudaBuffer,
    batch_heads: usize,
    q_len: usize,
    k_len: usize,
    scale: f32,
    is_causal: bool,
) -> Result<CudaBuffer, String> {
    imp::fused_softmax_f32_no_host(input, batch_heads, q_len, k_len, scale, is_causal)
}

pub fn fused_softmax_backward_f32(
    output: &CudaBuffer,
    grad: &CudaBuffer,
    batch_heads: usize,
    q_len: usize,
    k_len: usize,
    scale: f32,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    imp::fused_softmax_backward_f32(output, grad, batch_heads, q_len, k_len, scale)
}

pub fn fused_softmax_backward_f32_buffer(
    output: &CudaBuffer,
    grad: &CudaBuffer,
    batch_heads: usize,
    q_len: usize,
    k_len: usize,
    scale: f32,
) -> Result<CudaBuffer, String> {
    imp::fused_softmax_backward_f32_buffer(output, grad, batch_heads, q_len, k_len, scale)
}

pub fn fused_softmax_f32_with_past(
    input: &CudaBuffer,
    batch_heads: usize,
    q_len: usize,
    k_len: usize,
    scale: f32,
    is_causal: bool,
    past_len: usize,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    imp::fused_softmax_f32_with_past(input, batch_heads, q_len, k_len, scale, is_causal, past_len)
}

pub fn fused_softmax_f32_with_past_no_host(
    input: &CudaBuffer,
    batch_heads: usize,
    q_len: usize,
    k_len: usize,
    scale: f32,
    is_causal: bool,
    past_len: usize,
) -> Result<CudaBuffer, String> {
    imp::fused_softmax_f32_with_past_no_host(
        input,
        batch_heads,
        q_len,
        k_len,
        scale,
        is_causal,
        past_len,
    )
}

pub fn rms_norm_f32(
    input: &CudaBuffer,
    weight: &CudaBuffer,
    rows: usize,
    dim: usize,
    eps: f32,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    imp::rms_norm_f32(input, weight, rows, dim, eps)
}

pub fn rms_norm_f32_buffer(
    input: &CudaBuffer,
    weight: &CudaBuffer,
    rows: usize,
    dim: usize,
    eps: f32,
) -> Result<CudaBuffer, String> {
    imp::rms_norm_f32_buffer(input, weight, rows, dim, eps)
}

#[allow(clippy::too_many_arguments)]
pub fn rms_norm_typed(
    input: &CudaBuffer,
    input_dtype: crate::precision::DType,
    input_scale: Option<f32>,
    weight: &CudaBuffer,
    weight_dtype: crate::precision::DType,
    weight_scale: Option<f32>,
    rows: usize,
    dim: usize,
    eps: f32,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    imp::rms_norm_typed(
        input,
        input_dtype,
        input_scale,
        weight,
        weight_dtype,
        weight_scale,
        rows,
        dim,
        eps,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn rms_norm_typed_buffer(
    input: &CudaBuffer,
    input_dtype: crate::precision::DType,
    input_scale: Option<f32>,
    weight: &CudaBuffer,
    weight_dtype: crate::precision::DType,
    weight_scale: Option<f32>,
    rows: usize,
    dim: usize,
    eps: f32,
) -> Result<CudaBuffer, String> {
    imp::rms_norm_typed_buffer(
        input,
        input_dtype,
        input_scale,
        weight,
        weight_dtype,
        weight_scale,
        rows,
        dim,
        eps,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn rms_norm_i8_typed_output_buffer(
    input: &CudaBuffer,
    input_dtype: crate::precision::DType,
    input_scale: Option<f32>,
    weight: &CudaBuffer,
    weight_dtype: crate::precision::DType,
    weight_scale: Option<f32>,
    rows: usize,
    dim: usize,
    eps: f32,
) -> Result<(CudaBuffer, f32), String> {
    imp::rms_norm_i8_typed_output_buffer(
        input,
        input_dtype,
        input_scale,
        weight,
        weight_dtype,
        weight_scale,
        rows,
        dim,
        eps,
    )
}

pub fn rms_norm_backward_f32(
    input: &CudaBuffer,
    weight: &CudaBuffer,
    grad: &CudaBuffer,
    rows: usize,
    dim: usize,
    eps: f32,
) -> Result<CudaTwoHostBuffers, String> {
    imp::rms_norm_backward_f32(input, weight, grad, rows, dim, eps)
}

pub fn rms_norm_backward_f32_buffers(
    input: &CudaBuffer,
    weight: &CudaBuffer,
    grad: &CudaBuffer,
    rows: usize,
    dim: usize,
    eps: f32,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    imp::rms_norm_backward_f32_buffers(input, weight, grad, rows, dim, eps)
}

#[allow(clippy::too_many_arguments)]
pub fn rms_norm_backward_typed(
    input: &CudaBuffer,
    input_dtype: crate::precision::DType,
    input_scale: Option<f32>,
    weight: &CudaBuffer,
    weight_dtype: crate::precision::DType,
    weight_scale: Option<f32>,
    grad: &CudaBuffer,
    rows: usize,
    dim: usize,
    eps: f32,
) -> Result<CudaTwoHostBuffers, String> {
    imp::rms_norm_backward_typed(
        input,
        input_dtype,
        input_scale,
        weight,
        weight_dtype,
        weight_scale,
        grad,
        rows,
        dim,
        eps,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn rms_norm_backward_typed_buffers(
    input: &CudaBuffer,
    input_dtype: crate::precision::DType,
    input_scale: Option<f32>,
    weight: &CudaBuffer,
    weight_dtype: crate::precision::DType,
    weight_scale: Option<f32>,
    grad: &CudaBuffer,
    rows: usize,
    dim: usize,
    eps: f32,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    imp::rms_norm_backward_typed_buffers(
        input,
        input_dtype,
        input_scale,
        weight,
        weight_dtype,
        weight_scale,
        grad,
        rows,
        dim,
        eps,
    )
}
