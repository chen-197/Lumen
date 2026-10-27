use super::*;

pub fn fused_gate_up_silu_f32(
    input: &CudaBuffer,
    gate: &CudaBuffer,
    up: &CudaBuffer,
    rows: usize,
    n_dim: usize,
    k_dim: usize,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    imp::fused_gate_up_silu_f32(input, gate, up, rows, n_dim, k_dim)
}

pub fn fused_gate_up_silu_f32_buffer(
    input: &CudaBuffer,
    gate: &CudaBuffer,
    up: &CudaBuffer,
    rows: usize,
    n_dim: usize,
    k_dim: usize,
) -> Result<CudaBuffer, String> {
    imp::fused_gate_up_silu_f32_buffer(input, gate, up, rows, n_dim, k_dim)
}

pub fn silu_mul_f32_buffer_no_host(
    gate: &CudaBuffer,
    up: &CudaBuffer,
) -> Result<CudaBuffer, String> {
    imp::silu_mul_f32_buffer_no_host(gate, up)
}

#[allow(clippy::too_many_arguments)]
pub fn fused_gate_up_silu_typed(
    input: &CudaBuffer,
    input_dtype: crate::precision::DType,
    input_scale: Option<f32>,
    gate: &CudaBuffer,
    weight_dtype: crate::precision::DType,
    gate_scale: Option<f32>,
    up: &CudaBuffer,
    up_scale: Option<f32>,
    rows: usize,
    n_dim: usize,
    k_dim: usize,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    imp::fused_gate_up_silu_typed(
        input,
        input_dtype,
        input_scale,
        gate,
        weight_dtype,
        gate_scale,
        up,
        up_scale,
        rows,
        n_dim,
        k_dim,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn fused_gate_up_silu_typed_buffer(
    input: &CudaBuffer,
    input_dtype: crate::precision::DType,
    input_scale: Option<f32>,
    gate: &CudaBuffer,
    weight_dtype: crate::precision::DType,
    gate_scale: Option<f32>,
    up: &CudaBuffer,
    up_scale: Option<f32>,
    rows: usize,
    n_dim: usize,
    k_dim: usize,
) -> Result<CudaBuffer, String> {
    imp::fused_gate_up_silu_typed_buffer(
        input,
        input_dtype,
        input_scale,
        gate,
        weight_dtype,
        gate_scale,
        up,
        up_scale,
        rows,
        n_dim,
        k_dim,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn fused_gate_up_silu_typed_output_buffer(
    input: &CudaBuffer,
    input_dtype: crate::precision::DType,
    input_scale: Option<f32>,
    gate: &CudaBuffer,
    weight_dtype: crate::precision::DType,
    gate_scale: Option<f32>,
    up: &CudaBuffer,
    up_scale: Option<f32>,
    output_dtype: crate::precision::DType,
    rows: usize,
    n_dim: usize,
    k_dim: usize,
) -> Result<CudaBuffer, String> {
    imp::fused_gate_up_silu_typed_output_buffer(
        input,
        input_dtype,
        input_scale,
        gate,
        weight_dtype,
        gate_scale,
        up,
        up_scale,
        output_dtype,
        rows,
        n_dim,
        k_dim,
    )
}

pub fn fused_qkv_f32(
    input: &CudaBuffer,
    q: &CudaBuffer,
    k: &CudaBuffer,
    v: &CudaBuffer,
    rows: usize,
    q_n: usize,
    k_n: usize,
    k_dim: usize,
) -> Result<CudaThreeHostBuffers, String> {
    imp::fused_qkv_f32(input, q, k, v, rows, q_n, k_n, k_dim)
}

pub fn fused_qkv_f32_buffer(
    input: &CudaBuffer,
    q: &CudaBuffer,
    k: &CudaBuffer,
    v: &CudaBuffer,
    rows: usize,
    q_n: usize,
    k_n: usize,
    k_dim: usize,
) -> Result<(CudaBuffer, CudaBuffer, CudaBuffer), String> {
    imp::fused_qkv_f32_buffer(input, q, k, v, rows, q_n, k_n, k_dim)
}

#[allow(clippy::too_many_arguments)]
pub fn fused_qkv_typed_buffer(
    input: &CudaBuffer,
    input_dtype: crate::precision::DType,
    input_scale: Option<f32>,
    q: &CudaBuffer,
    k: &CudaBuffer,
    v: &CudaBuffer,
    weight_dtype: crate::precision::DType,
    q_scale: Option<f32>,
    k_scale: Option<f32>,
    v_scale: Option<f32>,
    rows: usize,
    q_n: usize,
    k_n: usize,
    k_dim: usize,
) -> Result<(CudaBuffer, CudaBuffer, CudaBuffer), String> {
    imp::fused_qkv_typed_buffer(
        input,
        input_dtype,
        input_scale,
        q,
        k,
        v,
        weight_dtype,
        q_scale,
        k_scale,
        v_scale,
        rows,
        q_n,
        k_n,
        k_dim,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn fused_qkv_typed_output_buffer(
    input: &CudaBuffer,
    input_dtype: crate::precision::DType,
    input_scale: Option<f32>,
    q: &CudaBuffer,
    k: &CudaBuffer,
    v: &CudaBuffer,
    weight_dtype: crate::precision::DType,
    q_scale: Option<f32>,
    k_scale: Option<f32>,
    v_scale: Option<f32>,
    output_dtype: crate::precision::DType,
    rows: usize,
    q_n: usize,
    k_n: usize,
    k_dim: usize,
) -> Result<(CudaBuffer, CudaBuffer, CudaBuffer), String> {
    imp::fused_qkv_typed_output_buffer(
        input,
        input_dtype,
        input_scale,
        q,
        k,
        v,
        weight_dtype,
        q_scale,
        k_scale,
        v_scale,
        output_dtype,
        rows,
        q_n,
        k_n,
        k_dim,
    )
}
