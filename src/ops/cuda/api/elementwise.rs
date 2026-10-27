use super::*;

pub fn unary_f32(input: &CudaBuffer, op: UnaryOp) -> Result<(CudaBuffer, Vec<f32>), String> {
    imp::unary_f32(input, op)
}

pub fn unary_f32_buffer(input: &CudaBuffer, op: UnaryOp) -> Result<CudaBuffer, String> {
    imp::unary_f32_buffer(input, op)
}

pub fn unary_f16_buffer(input: &CudaBuffer, op: UnaryOp) -> Result<CudaBuffer, String> {
    imp::unary_f16_buffer(input, op)
}

pub fn unary_f16_typed_output_buffer(
    input: &CudaBuffer,
    op: UnaryOp,
) -> Result<CudaBuffer, String> {
    imp::unary_f16_typed_output_buffer(input, op)
}

pub fn unary_bf16_buffer(input: &CudaBuffer, op: UnaryOp) -> Result<CudaBuffer, String> {
    imp::unary_bf16_buffer(input, op)
}

pub fn unary_bf16_typed_output_buffer(
    input: &CudaBuffer,
    op: UnaryOp,
) -> Result<CudaBuffer, String> {
    imp::unary_bf16_typed_output_buffer(input, op)
}

pub fn unary_i8_buffer(input: &CudaBuffer, scale: f32, op: UnaryOp) -> Result<CudaBuffer, String> {
    imp::unary_i8_buffer(input, scale, op)
}

pub fn unary_i8_relu_typed_output_buffer(input: &CudaBuffer) -> Result<CudaBuffer, String> {
    imp::unary_i8_relu_typed_output_buffer(input)
}

pub fn unary_backward_f32(
    input: &CudaBuffer,
    output: &CudaBuffer,
    grad: &CudaBuffer,
    op: UnaryOp,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    imp::unary_backward_f32(input, output, grad, op)
}

pub fn unary_backward_f32_buffer(
    input: &CudaBuffer,
    output: &CudaBuffer,
    grad: &CudaBuffer,
    op: UnaryOp,
) -> Result<CudaBuffer, String> {
    imp::unary_backward_f32_buffer(input, output, grad, op)
}

pub fn unary_backward_f16_buffer(
    input: &CudaBuffer,
    output: &CudaBuffer,
    grad: &CudaBuffer,
    op: UnaryOp,
) -> Result<CudaBuffer, String> {
    imp::unary_backward_f16_buffer(input, output, grad, op)
}

pub fn unary_backward_bf16_buffer(
    input: &CudaBuffer,
    output: &CudaBuffer,
    grad: &CudaBuffer,
    op: UnaryOp,
) -> Result<CudaBuffer, String> {
    imp::unary_backward_bf16_buffer(input, output, grad, op)
}

pub fn unary_backward_i8_buffer(
    input: &CudaBuffer,
    scale: f32,
    output: &CudaBuffer,
    grad: &CudaBuffer,
    op: UnaryOp,
) -> Result<CudaBuffer, String> {
    imp::unary_backward_i8_buffer(input, scale, output, grad, op)
}

pub fn binary_f32(
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    op: BinaryOp,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    imp::binary_f32(lhs, rhs, op)
}

pub fn binary_f32_buffer(
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    op: BinaryOp,
) -> Result<CudaBuffer, String> {
    imp::binary_f32_buffer(lhs, rhs, op)
}

pub fn binary_typed_buffer_no_host(
    lhs: &CudaBuffer,
    lhs_dtype: crate::precision::DType,
    lhs_scale: Option<f32>,
    rhs: &CudaBuffer,
    rhs_dtype: crate::precision::DType,
    rhs_scale: Option<f32>,
    op: BinaryOp,
) -> Result<CudaBuffer, String> {
    imp::binary_typed_buffer_no_host(lhs, lhs_dtype, lhs_scale, rhs, rhs_dtype, rhs_scale, op)
}

pub fn binary_lowp_typed_output_buffer_no_host(
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    dtype: crate::precision::DType,
    op: BinaryOp,
) -> Result<CudaBuffer, String> {
    imp::binary_lowp_typed_output_buffer_no_host(lhs, rhs, dtype, op)
}

#[allow(clippy::too_many_arguments)]
pub fn binary_typed_lastdim_buffer_no_host(
    lhs: &CudaBuffer,
    lhs_dtype: crate::precision::DType,
    lhs_scale: Option<f32>,
    rhs: &CudaBuffer,
    rhs_dtype: crate::precision::DType,
    rhs_scale: Option<f32>,
    out_len: usize,
    last_dim: usize,
    vector_on_rhs: bool,
    op: BinaryOp,
) -> Result<CudaBuffer, String> {
    imp::binary_typed_lastdim_buffer_no_host(
        lhs,
        lhs_dtype,
        lhs_scale,
        rhs,
        rhs_dtype,
        rhs_scale,
        out_len,
        last_dim,
        vector_on_rhs,
        op,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn binary_lowp_typed_lastdim_output_buffer_no_host(
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    out_len: usize,
    last_dim: usize,
    vector_on_rhs: bool,
    dtype: crate::precision::DType,
    op: BinaryOp,
) -> Result<CudaBuffer, String> {
    imp::binary_lowp_typed_lastdim_output_buffer_no_host(
        lhs,
        rhs,
        out_len,
        last_dim,
        vector_on_rhs,
        dtype,
        op,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn binary_typed_row_scalar_buffer_no_host(
    lhs: &CudaBuffer,
    lhs_dtype: crate::precision::DType,
    lhs_scale: Option<f32>,
    rhs: &CudaBuffer,
    rhs_dtype: crate::precision::DType,
    rhs_scale: Option<f32>,
    rows: usize,
    last_dim: usize,
    scalar_on_rhs: bool,
    op: BinaryOp,
) -> Result<CudaBuffer, String> {
    imp::binary_typed_row_scalar_buffer_no_host(
        lhs,
        lhs_dtype,
        lhs_scale,
        rhs,
        rhs_dtype,
        rhs_scale,
        rows,
        last_dim,
        scalar_on_rhs,
        op,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn binary_lowp_typed_row_scalar_output_buffer_no_host(
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    rows: usize,
    last_dim: usize,
    scalar_on_rhs: bool,
    dtype: crate::precision::DType,
    op: BinaryOp,
) -> Result<CudaBuffer, String> {
    imp::binary_lowp_typed_row_scalar_output_buffer_no_host(
        lhs,
        rhs,
        rows,
        last_dim,
        scalar_on_rhs,
        dtype,
        op,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn binary_i8_typed_row_scalar_output_buffer_no_host(
    lhs: &CudaBuffer,
    lhs_scale: f32,
    rhs: &CudaBuffer,
    rhs_scale: f32,
    rows: usize,
    last_dim: usize,
    scalar_on_rhs: bool,
    op: BinaryOp,
) -> Result<(CudaBuffer, f32), String> {
    imp::binary_i8_typed_row_scalar_output_buffer_no_host(
        lhs,
        lhs_scale,
        rhs,
        rhs_scale,
        rows,
        last_dim,
        scalar_on_rhs,
        op,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn binary_typed_broadcast_buffer_no_host(
    lhs: &CudaBuffer,
    lhs_dtype: crate::precision::DType,
    lhs_scale: Option<f32>,
    rhs: &CudaBuffer,
    rhs_dtype: crate::precision::DType,
    rhs_scale: Option<f32>,
    lhs_shape: &[usize],
    rhs_shape: &[usize],
    out_shape: &[usize],
    op: BinaryOp,
) -> Result<CudaBuffer, String> {
    imp::binary_typed_broadcast_buffer_no_host(
        lhs, lhs_dtype, lhs_scale, rhs, rhs_dtype, rhs_scale, lhs_shape, rhs_shape, out_shape, op,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn binary_lowp_typed_broadcast_output_buffer_no_host(
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    lhs_shape: &[usize],
    rhs_shape: &[usize],
    out_shape: &[usize],
    dtype: crate::precision::DType,
    op: BinaryOp,
) -> Result<CudaBuffer, String> {
    imp::binary_lowp_typed_broadcast_output_buffer_no_host(
        lhs, rhs, lhs_shape, rhs_shape, out_shape, dtype, op,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn binary_i8_typed_broadcast_output_buffer_no_host(
    lhs: &CudaBuffer,
    lhs_scale: f32,
    rhs: &CudaBuffer,
    rhs_scale: f32,
    lhs_shape: &[usize],
    rhs_shape: &[usize],
    out_shape: &[usize],
    op: BinaryOp,
) -> Result<(CudaBuffer, f32), String> {
    imp::binary_i8_typed_broadcast_output_buffer_no_host(
        lhs, lhs_scale, rhs, rhs_scale, lhs_shape, rhs_shape, out_shape, op,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn binary_typed_b1d_1h1_buffer_no_host(
    lhs: &CudaBuffer,
    lhs_dtype: crate::precision::DType,
    lhs_scale: Option<f32>,
    rhs: &CudaBuffer,
    rhs_dtype: crate::precision::DType,
    rhs_scale: Option<f32>,
    batch: usize,
    heads: usize,
    dim: usize,
    b1d_on_lhs: bool,
    op: BinaryOp,
) -> Result<CudaBuffer, String> {
    imp::binary_typed_b1d_1h1_buffer_no_host(
        lhs, lhs_dtype, lhs_scale, rhs, rhs_dtype, rhs_scale, batch, heads, dim, b1d_on_lhs, op,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn binary_lowp_typed_b1d_1h1_output_buffer_no_host(
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    batch: usize,
    heads: usize,
    dim: usize,
    b1d_on_lhs: bool,
    dtype: crate::precision::DType,
    op: BinaryOp,
) -> Result<CudaBuffer, String> {
    imp::binary_lowp_typed_b1d_1h1_output_buffer_no_host(
        lhs, rhs, batch, heads, dim, b1d_on_lhs, dtype, op,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn binary_i8_typed_b1d_1h1_output_buffer_no_host(
    lhs: &CudaBuffer,
    lhs_scale: f32,
    rhs: &CudaBuffer,
    rhs_scale: f32,
    batch: usize,
    heads: usize,
    dim: usize,
    b1d_on_lhs: bool,
    op: BinaryOp,
) -> Result<(CudaBuffer, f32), String> {
    imp::binary_i8_typed_b1d_1h1_output_buffer_no_host(
        lhs, lhs_scale, rhs, rhs_scale, batch, heads, dim, b1d_on_lhs, op,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn binary_typed_b1d_1hd_buffer_no_host(
    lhs: &CudaBuffer,
    lhs_dtype: crate::precision::DType,
    lhs_scale: Option<f32>,
    rhs: &CudaBuffer,
    rhs_dtype: crate::precision::DType,
    rhs_scale: Option<f32>,
    batch: usize,
    heads: usize,
    dim: usize,
    b1d_on_lhs: bool,
    op: BinaryOp,
) -> Result<CudaBuffer, String> {
    imp::binary_typed_b1d_1hd_buffer_no_host(
        lhs, lhs_dtype, lhs_scale, rhs, rhs_dtype, rhs_scale, batch, heads, dim, b1d_on_lhs, op,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn binary_lowp_typed_b1d_1hd_output_buffer_no_host(
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    batch: usize,
    heads: usize,
    dim: usize,
    b1d_on_lhs: bool,
    dtype: crate::precision::DType,
    op: BinaryOp,
) -> Result<CudaBuffer, String> {
    imp::binary_lowp_typed_b1d_1hd_output_buffer_no_host(
        lhs, rhs, batch, heads, dim, b1d_on_lhs, dtype, op,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn binary_i8_typed_b1d_1hd_output_buffer_no_host(
    lhs: &CudaBuffer,
    lhs_scale: f32,
    rhs: &CudaBuffer,
    rhs_scale: f32,
    batch: usize,
    heads: usize,
    dim: usize,
    b1d_on_lhs: bool,
    op: BinaryOp,
) -> Result<(CudaBuffer, f32), String> {
    imp::binary_i8_typed_b1d_1hd_output_buffer_no_host(
        lhs, lhs_scale, rhs, rhs_scale, batch, heads, dim, b1d_on_lhs, op,
    )
}

pub fn binary_f16_host_no_host(
    lhs: &[u16],
    rhs: &[u16],
    op: BinaryOp,
) -> Result<CudaBuffer, String> {
    imp::binary_f16_host_no_host(lhs, rhs, op)
}

pub fn binary_f16_buffer_no_host(
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    op: BinaryOp,
) -> Result<CudaBuffer, String> {
    imp::binary_f16_buffer_no_host(lhs, rhs, op)
}

pub fn binary_f16_lastdim_buffer_no_host(
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    out_len: usize,
    last_dim: usize,
    vector_on_rhs: bool,
    op: BinaryOp,
) -> Result<CudaBuffer, String> {
    imp::binary_f16_lastdim_buffer_no_host(lhs, rhs, out_len, last_dim, vector_on_rhs, op)
}

pub fn binary_bf16_host_no_host(
    lhs: &[u16],
    rhs: &[u16],
    op: BinaryOp,
) -> Result<CudaBuffer, String> {
    imp::binary_bf16_host_no_host(lhs, rhs, op)
}

pub fn binary_bf16_buffer_no_host(
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    op: BinaryOp,
) -> Result<CudaBuffer, String> {
    imp::binary_bf16_buffer_no_host(lhs, rhs, op)
}

pub fn binary_bf16_lastdim_buffer_no_host(
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    out_len: usize,
    last_dim: usize,
    vector_on_rhs: bool,
    op: BinaryOp,
) -> Result<CudaBuffer, String> {
    imp::binary_bf16_lastdim_buffer_no_host(lhs, rhs, out_len, last_dim, vector_on_rhs, op)
}

pub fn binary_i8_host_no_host(
    lhs: &[i8],
    lhs_scale: f32,
    rhs: &[i8],
    rhs_scale: f32,
    op: BinaryOp,
) -> Result<CudaBuffer, String> {
    imp::binary_i8_host_no_host(lhs, lhs_scale, rhs, rhs_scale, op)
}

pub fn binary_i8_buffer_no_host(
    lhs: &CudaBuffer,
    lhs_scale: f32,
    rhs: &CudaBuffer,
    rhs_scale: f32,
    op: BinaryOp,
) -> Result<CudaBuffer, String> {
    imp::binary_i8_buffer_no_host(lhs, lhs_scale, rhs, rhs_scale, op)
}

pub fn binary_i8_typed_output_buffer_no_host(
    lhs: &CudaBuffer,
    lhs_scale: f32,
    rhs: &CudaBuffer,
    rhs_scale: f32,
    op: BinaryOp,
) -> Result<(CudaBuffer, f32), String> {
    imp::binary_i8_typed_output_buffer_no_host(lhs, lhs_scale, rhs, rhs_scale, op)
}

pub fn binary_i8_lastdim_buffer_no_host(
    lhs: &CudaBuffer,
    lhs_scale: f32,
    rhs: &CudaBuffer,
    rhs_scale: f32,
    out_len: usize,
    last_dim: usize,
    vector_on_rhs: bool,
    op: BinaryOp,
) -> Result<CudaBuffer, String> {
    imp::binary_i8_lastdim_buffer_no_host(
        lhs,
        lhs_scale,
        rhs,
        rhs_scale,
        out_len,
        last_dim,
        vector_on_rhs,
        op,
    )
}

pub fn binary_i8_typed_lastdim_output_buffer_no_host(
    lhs: &CudaBuffer,
    lhs_scale: f32,
    rhs: &CudaBuffer,
    rhs_scale: f32,
    out_len: usize,
    last_dim: usize,
    vector_on_rhs: bool,
    op: BinaryOp,
) -> Result<(CudaBuffer, f32), String> {
    imp::binary_i8_typed_lastdim_output_buffer_no_host(
        lhs,
        lhs_scale,
        rhs,
        rhs_scale,
        out_len,
        last_dim,
        vector_on_rhs,
        op,
    )
}

pub fn mul_grad_f16_host_no_host(grad: &CudaBuffer, operand: &[u16]) -> Result<CudaBuffer, String> {
    imp::mul_grad_f16_host_no_host(grad, operand)
}

pub fn mul_grad_f16_buffer_no_host(
    grad: &CudaBuffer,
    operand: &CudaBuffer,
) -> Result<CudaBuffer, String> {
    imp::mul_grad_f16_buffer_no_host(grad, operand)
}

pub fn mul_grad_f16_lastdim_buffer_no_host(
    grad: &CudaBuffer,
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    out_len: usize,
    last_dim: usize,
    vector_on_rhs: bool,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    imp::mul_grad_f16_lastdim_buffer_no_host(grad, lhs, rhs, out_len, last_dim, vector_on_rhs)
}

pub fn mul_grad_bf16_host_no_host(
    grad: &CudaBuffer,
    operand: &[u16],
) -> Result<CudaBuffer, String> {
    imp::mul_grad_bf16_host_no_host(grad, operand)
}

pub fn mul_grad_bf16_buffer_no_host(
    grad: &CudaBuffer,
    operand: &CudaBuffer,
) -> Result<CudaBuffer, String> {
    imp::mul_grad_bf16_buffer_no_host(grad, operand)
}

pub fn mul_grad_bf16_lastdim_buffer_no_host(
    grad: &CudaBuffer,
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    out_len: usize,
    last_dim: usize,
    vector_on_rhs: bool,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    imp::mul_grad_bf16_lastdim_buffer_no_host(grad, lhs, rhs, out_len, last_dim, vector_on_rhs)
}

pub fn mul_grad_i8_host_no_host(
    grad: &CudaBuffer,
    operand: &[i8],
    scale: f32,
) -> Result<CudaBuffer, String> {
    imp::mul_grad_i8_host_no_host(grad, operand, scale)
}

pub fn mul_grad_i8_buffer_no_host(
    grad: &CudaBuffer,
    operand: &CudaBuffer,
    scale: f32,
) -> Result<CudaBuffer, String> {
    imp::mul_grad_i8_buffer_no_host(grad, operand, scale)
}

pub fn mul_grad_i8_lastdim_buffer_no_host(
    grad: &CudaBuffer,
    lhs: &CudaBuffer,
    lhs_scale: f32,
    rhs: &CudaBuffer,
    rhs_scale: f32,
    out_len: usize,
    last_dim: usize,
    vector_on_rhs: bool,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    imp::mul_grad_i8_lastdim_buffer_no_host(
        grad,
        lhs,
        lhs_scale,
        rhs,
        rhs_scale,
        out_len,
        last_dim,
        vector_on_rhs,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn mul_grad_typed_lastdim_buffer_no_host(
    grad: &CudaBuffer,
    lhs: &CudaBuffer,
    lhs_dtype: crate::precision::DType,
    lhs_scale: Option<f32>,
    rhs: &CudaBuffer,
    rhs_dtype: crate::precision::DType,
    rhs_scale: Option<f32>,
    out_len: usize,
    last_dim: usize,
    vector_on_rhs: bool,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    imp::mul_grad_typed_lastdim_buffer_no_host(
        grad,
        lhs,
        lhs_dtype,
        lhs_scale,
        rhs,
        rhs_dtype,
        rhs_scale,
        out_len,
        last_dim,
        vector_on_rhs,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn mul_grad_typed_row_scalar_buffer_no_host(
    grad: &CudaBuffer,
    lhs: &CudaBuffer,
    lhs_dtype: crate::precision::DType,
    lhs_scale: Option<f32>,
    rhs: &CudaBuffer,
    rhs_dtype: crate::precision::DType,
    rhs_scale: Option<f32>,
    rows: usize,
    last_dim: usize,
    scalar_on_rhs: bool,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    imp::mul_grad_typed_row_scalar_buffer_no_host(
        grad,
        lhs,
        lhs_dtype,
        lhs_scale,
        rhs,
        rhs_dtype,
        rhs_scale,
        rows,
        last_dim,
        scalar_on_rhs,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn mul_grad_typed_buffer_no_host(
    grad: &CudaBuffer,
    lhs: &CudaBuffer,
    lhs_dtype: crate::precision::DType,
    lhs_scale: Option<f32>,
    rhs: &CudaBuffer,
    rhs_dtype: crate::precision::DType,
    rhs_scale: Option<f32>,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    imp::mul_grad_typed_buffer_no_host(grad, lhs, lhs_dtype, lhs_scale, rhs, rhs_dtype, rhs_scale)
}

#[allow(clippy::too_many_arguments)]
pub fn mul_grad_typed_broadcast_buffer_no_host(
    grad: &CudaBuffer,
    lhs: &CudaBuffer,
    lhs_dtype: crate::precision::DType,
    lhs_scale: Option<f32>,
    rhs: &CudaBuffer,
    rhs_dtype: crate::precision::DType,
    rhs_scale: Option<f32>,
    lhs_shape: &[usize],
    rhs_shape: &[usize],
    out_shape: &[usize],
) -> Result<(CudaBuffer, CudaBuffer), String> {
    imp::mul_grad_typed_broadcast_buffer_no_host(
        grad, lhs, lhs_dtype, lhs_scale, rhs, rhs_dtype, rhs_scale, lhs_shape, rhs_shape, out_shape,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn mul_grad_typed_b1d_1h1_buffer_no_host(
    grad: &CudaBuffer,
    lhs: &CudaBuffer,
    lhs_dtype: crate::precision::DType,
    lhs_scale: Option<f32>,
    rhs: &CudaBuffer,
    rhs_dtype: crate::precision::DType,
    rhs_scale: Option<f32>,
    batch: usize,
    heads: usize,
    dim: usize,
    b1d_on_lhs: bool,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    imp::mul_grad_typed_b1d_1h1_buffer_no_host(
        grad, lhs, lhs_dtype, lhs_scale, rhs, rhs_dtype, rhs_scale, batch, heads, dim, b1d_on_lhs,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn mul_grad_typed_b1d_1hd_buffer_no_host(
    grad: &CudaBuffer,
    lhs: &CudaBuffer,
    lhs_dtype: crate::precision::DType,
    lhs_scale: Option<f32>,
    rhs: &CudaBuffer,
    rhs_dtype: crate::precision::DType,
    rhs_scale: Option<f32>,
    batch: usize,
    heads: usize,
    dim: usize,
    b1d_on_lhs: bool,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    imp::mul_grad_typed_b1d_1hd_buffer_no_host(
        grad, lhs, lhs_dtype, lhs_scale, rhs, rhs_dtype, rhs_scale, batch, heads, dim, b1d_on_lhs,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn mul_grad_typed_scalar_buffer_no_host(
    grad: &CudaBuffer,
    lhs: &CudaBuffer,
    lhs_dtype: crate::precision::DType,
    lhs_scale: Option<f32>,
    rhs: &CudaBuffer,
    rhs_dtype: crate::precision::DType,
    rhs_scale: Option<f32>,
    out_len: usize,
    scalar_on_rhs: bool,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    imp::mul_grad_typed_scalar_buffer_no_host(
        grad,
        lhs,
        lhs_dtype,
        lhs_scale,
        rhs,
        rhs_dtype,
        rhs_scale,
        out_len,
        scalar_on_rhs,
    )
}

pub fn binary_backward_f32(
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    grad: &CudaBuffer,
    op: BinaryOp,
) -> Result<CudaTwoHostBuffers, String> {
    imp::binary_backward_f32(lhs, rhs, grad, op)
}

pub fn binary_backward_f32_buffers(
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    grad: &CudaBuffer,
    op: BinaryOp,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    imp::binary_backward_f32_buffers(lhs, rhs, grad, op)
}

pub fn add_sub_backward_f32_buffers(
    grad: &CudaBuffer,
    len: usize,
    op: BinaryOp,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    imp::add_sub_backward_f32_buffers(grad, len, op)
}

pub fn add_sub_backward_lastdim_f32_buffers(
    grad: &CudaBuffer,
    out_len: usize,
    last_dim: usize,
    vector_on_rhs: bool,
    op: BinaryOp,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    imp::add_sub_backward_lastdim_f32_buffers(grad, out_len, last_dim, vector_on_rhs, op)
}

pub fn add_sub_backward_scalar_f32_buffers(
    grad: &CudaBuffer,
    out_len: usize,
    scalar_on_rhs: bool,
    op: BinaryOp,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    imp::add_sub_backward_scalar_f32_buffers(grad, out_len, scalar_on_rhs, op)
}

pub fn add_sub_backward_row_scalar_f32_buffers(
    grad: &CudaBuffer,
    rows: usize,
    last_dim: usize,
    scalar_on_rhs: bool,
    op: BinaryOp,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    imp::add_sub_backward_row_scalar_f32_buffers(grad, rows, last_dim, scalar_on_rhs, op)
}

pub fn add_sub_backward_b1d_1h1_f32_buffers(
    grad: &CudaBuffer,
    batch: usize,
    heads: usize,
    dim: usize,
    b1d_on_lhs: bool,
    op: BinaryOp,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    imp::add_sub_backward_b1d_1h1_f32_buffers(grad, batch, heads, dim, b1d_on_lhs, op)
}

pub fn add_sub_backward_b1d_1hd_f32_buffers(
    grad: &CudaBuffer,
    batch: usize,
    heads: usize,
    dim: usize,
    b1d_on_lhs: bool,
    op: BinaryOp,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    imp::add_sub_backward_b1d_1hd_f32_buffers(grad, batch, heads, dim, b1d_on_lhs, op)
}

pub fn add_sub_broadcast_backward_f32_buffers(
    grad: &CudaBuffer,
    lhs_shape: &[usize],
    rhs_shape: &[usize],
    out_shape: &[usize],
    op: BinaryOp,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    imp::add_sub_broadcast_backward_f32_buffers(grad, lhs_shape, rhs_shape, out_shape, op)
}

pub fn binary_broadcast_f32(
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    lhs_shape: &[usize],
    rhs_shape: &[usize],
    out_shape: &[usize],
    op: BinaryOp,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    imp::binary_broadcast_f32(lhs, rhs, lhs_shape, rhs_shape, out_shape, op)
}

pub fn binary_broadcast_f32_buffer(
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    lhs_shape: &[usize],
    rhs_shape: &[usize],
    out_shape: &[usize],
    op: BinaryOp,
) -> Result<CudaBuffer, String> {
    imp::binary_broadcast_f32_buffer(lhs, rhs, lhs_shape, rhs_shape, out_shape, op)
}

pub fn binary_broadcast_backward_f32(
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    grad: &CudaBuffer,
    lhs_shape: &[usize],
    rhs_shape: &[usize],
    out_shape: &[usize],
    op: BinaryOp,
) -> Result<CudaTwoHostBuffers, String> {
    imp::binary_broadcast_backward_f32(lhs, rhs, grad, lhs_shape, rhs_shape, out_shape, op)
}

pub fn binary_broadcast_backward_f32_buffers(
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    grad: &CudaBuffer,
    lhs_shape: &[usize],
    rhs_shape: &[usize],
    out_shape: &[usize],
    op: BinaryOp,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    imp::binary_broadcast_backward_f32_buffers(lhs, rhs, grad, lhs_shape, rhs_shape, out_shape, op)
}
