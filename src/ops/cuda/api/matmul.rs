use super::*;

pub fn matmul_f32(
    a: &CudaBuffer,
    b: &CudaBuffer,
    m: usize,
    n: usize,
    k: usize,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    imp::matmul_f32(a, b, m, n, k)
}

pub fn matmul_f32_no_host(
    a: &CudaBuffer,
    b: &CudaBuffer,
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaBuffer, String> {
    imp::matmul_f32_no_host(a, b, m, n, k)
}

pub fn matmul_bf16_host_no_host(
    a: &[u16],
    b: &[u16],
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaBuffer, String> {
    imp::matmul_bf16_host_no_host(a, b, m, n, k)
}

pub fn matmul_bf16_buffer_no_host(
    a: &CudaBuffer,
    b: &CudaBuffer,
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaBuffer, String> {
    imp::matmul_bf16_buffer_no_host(a, b, m, n, k)
}

pub fn matmul_bf16_typed_output_buffer_no_host(
    a: &CudaBuffer,
    b: &CudaBuffer,
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaBuffer, String> {
    imp::matmul_bf16_typed_output_buffer_no_host(a, b, m, n, k)
}

pub fn matmul_f16_host_no_host(
    a: &[u16],
    b: &[u16],
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaBuffer, String> {
    imp::matmul_f16_host_no_host(a, b, m, n, k)
}

pub fn matmul_f16_buffer_no_host(
    a: &CudaBuffer,
    b: &CudaBuffer,
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaBuffer, String> {
    imp::matmul_f16_buffer_no_host(a, b, m, n, k)
}

pub fn matmul_f16_typed_output_buffer_no_host(
    a: &CudaBuffer,
    b: &CudaBuffer,
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaBuffer, String> {
    imp::matmul_f16_typed_output_buffer_no_host(a, b, m, n, k)
}

#[allow(clippy::too_many_arguments)]
pub fn matmul_i8_host_no_host(
    a: &[i8],
    a_scale: f32,
    b: &[i8],
    b_scale: f32,
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaBuffer, String> {
    imp::matmul_i8_host_no_host(a, a_scale, b, b_scale, m, n, k)
}

#[allow(clippy::too_many_arguments)]
pub fn matmul_i8_buffer_no_host(
    a: &CudaBuffer,
    a_scale: f32,
    b: &CudaBuffer,
    b_scale: f32,
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaBuffer, String> {
    imp::matmul_i8_buffer_no_host(a, a_scale, b, b_scale, m, n, k)
}

pub fn matmul_bf16_i8_buffer_no_host(
    a: &CudaBuffer,
    b: &CudaBuffer,
    b_scale: f32,
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaBuffer, String> {
    imp::matmul_bf16_i8_buffer_no_host(a, b, b_scale, m, n, k)
}

pub fn matmul_f16_i8_buffer_no_host(
    a: &CudaBuffer,
    b: &CudaBuffer,
    b_scale: f32,
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaBuffer, String> {
    imp::matmul_f16_i8_buffer_no_host(a, b, b_scale, m, n, k)
}

pub fn matmul_bf16_i8_dynamic_buffer_no_host(
    a: &CudaBuffer,
    b: &CudaBuffer,
    b_scale: f32,
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaBuffer, String> {
    imp::matmul_bf16_i8_dynamic_buffer_no_host(a, b, b_scale, m, n, k)
}

pub fn matmul_f16_i8_dynamic_buffer_no_host(
    a: &CudaBuffer,
    b: &CudaBuffer,
    b_scale: f32,
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaBuffer, String> {
    imp::matmul_f16_i8_dynamic_buffer_no_host(a, b, b_scale, m, n, k)
}

pub fn matmul_f32_i8_buffer_no_host(
    a: &CudaBuffer,
    b: &CudaBuffer,
    b_scale: f32,
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaBuffer, String> {
    imp::matmul_f32_i8_buffer_no_host(a, b, b_scale, m, n, k)
}

pub fn matmul_i8_bf16_buffer_no_host(
    a: &CudaBuffer,
    a_scale: f32,
    b: &CudaBuffer,
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaBuffer, String> {
    imp::matmul_i8_bf16_buffer_no_host(a, a_scale, b, m, n, k)
}

pub fn matmul_i8_f16_buffer_no_host(
    a: &CudaBuffer,
    a_scale: f32,
    b: &CudaBuffer,
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaBuffer, String> {
    imp::matmul_i8_f16_buffer_no_host(a, a_scale, b, m, n, k)
}

pub fn matmul_i8_f32_buffer_no_host(
    a: &CudaBuffer,
    a_scale: f32,
    b: &CudaBuffer,
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaBuffer, String> {
    imp::matmul_i8_f32_buffer_no_host(a, a_scale, b, m, n, k)
}

#[allow(clippy::too_many_arguments)]
pub fn matmul_i8_typed_output_buffer_no_host(
    a: &CudaBuffer,
    a_scale: f32,
    b: &CudaBuffer,
    b_scale: f32,
    m: usize,
    n: usize,
    k: usize,
) -> Result<(CudaBuffer, f32), String> {
    imp::matmul_i8_typed_output_buffer_no_host(a, a_scale, b, b_scale, m, n, k)
}

pub fn batch_matmul_f32(
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    batch_count: usize,
    m: usize,
    n: usize,
    k: usize,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    imp::batch_matmul_f32(lhs, rhs, batch_count, m, n, k)
}

pub fn batch_matmul_f32_no_host(
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    batch_count: usize,
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaBuffer, String> {
    imp::batch_matmul_f32_no_host(lhs, rhs, batch_count, m, n, k)
}

pub fn batch_matmul_bf16_buffer_no_host(
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    batch_count: usize,
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaBuffer, String> {
    imp::batch_matmul_bf16_buffer_no_host(lhs, rhs, batch_count, m, n, k)
}

pub fn batch_matmul_bf16_typed_output_buffer_no_host(
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    batch_count: usize,
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaBuffer, String> {
    imp::batch_matmul_bf16_typed_output_buffer_no_host(lhs, rhs, batch_count, m, n, k)
}

pub fn batch_matmul_f16_buffer_no_host(
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    batch_count: usize,
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaBuffer, String> {
    imp::batch_matmul_f16_buffer_no_host(lhs, rhs, batch_count, m, n, k)
}

pub fn batch_matmul_f16_typed_output_buffer_no_host(
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    batch_count: usize,
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaBuffer, String> {
    imp::batch_matmul_f16_typed_output_buffer_no_host(lhs, rhs, batch_count, m, n, k)
}

pub fn batch_matmul_bf16_i8_buffer_no_host(
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    rhs_scale: f32,
    batch_count: usize,
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaBuffer, String> {
    imp::batch_matmul_bf16_i8_buffer_no_host(lhs, rhs, rhs_scale, batch_count, m, n, k)
}

pub fn batch_matmul_f16_i8_buffer_no_host(
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    rhs_scale: f32,
    batch_count: usize,
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaBuffer, String> {
    imp::batch_matmul_f16_i8_buffer_no_host(lhs, rhs, rhs_scale, batch_count, m, n, k)
}

pub fn batch_matmul_f32_i8_buffer_no_host(
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    rhs_scale: f32,
    batch_count: usize,
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaBuffer, String> {
    imp::batch_matmul_f32_i8_buffer_no_host(lhs, rhs, rhs_scale, batch_count, m, n, k)
}

pub fn batch_matmul_i8_bf16_buffer_no_host(
    lhs: &CudaBuffer,
    lhs_scale: f32,
    rhs: &CudaBuffer,
    batch_count: usize,
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaBuffer, String> {
    imp::batch_matmul_i8_bf16_buffer_no_host(lhs, lhs_scale, rhs, batch_count, m, n, k)
}

pub fn batch_matmul_i8_f16_buffer_no_host(
    lhs: &CudaBuffer,
    lhs_scale: f32,
    rhs: &CudaBuffer,
    batch_count: usize,
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaBuffer, String> {
    imp::batch_matmul_i8_f16_buffer_no_host(lhs, lhs_scale, rhs, batch_count, m, n, k)
}

pub fn batch_matmul_i8_f32_buffer_no_host(
    lhs: &CudaBuffer,
    lhs_scale: f32,
    rhs: &CudaBuffer,
    batch_count: usize,
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaBuffer, String> {
    imp::batch_matmul_i8_f32_buffer_no_host(lhs, lhs_scale, rhs, batch_count, m, n, k)
}

#[allow(clippy::too_many_arguments)]
pub fn batch_matmul_i8_buffer_no_host(
    lhs: &CudaBuffer,
    lhs_scale: f32,
    rhs: &CudaBuffer,
    rhs_scale: f32,
    batch_count: usize,
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaBuffer, String> {
    imp::batch_matmul_i8_buffer_no_host(lhs, lhs_scale, rhs, rhs_scale, batch_count, m, n, k)
}

#[allow(clippy::too_many_arguments)]
pub fn batch_matmul_i8_typed_output_buffer_no_host(
    lhs: &CudaBuffer,
    lhs_scale: f32,
    rhs: &CudaBuffer,
    rhs_scale: f32,
    batch_count: usize,
    m: usize,
    n: usize,
    k: usize,
) -> Result<(CudaBuffer, f32), String> {
    imp::batch_matmul_i8_typed_output_buffer_no_host(
        lhs,
        lhs_scale,
        rhs,
        rhs_scale,
        batch_count,
        m,
        n,
        k,
    )
}

pub fn matmul_backward_f32_no_host(
    grad: &CudaBuffer,
    a: &CudaBuffer,
    b: &CudaBuffer,
    m: usize,
    k: usize,
    n: usize,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    imp::matmul_backward_f32_no_host(grad, a, b, m, k, n)
}

pub fn matmul_backward_bf16_i8_no_host(
    grad: &CudaBuffer,
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    rhs_scale: f32,
    m: usize,
    k: usize,
    n: usize,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    imp::matmul_backward_bf16_i8_no_host(grad, lhs, rhs, rhs_scale, m, k, n)
}

pub fn matmul_backward_f16_i8_no_host(
    grad: &CudaBuffer,
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    rhs_scale: f32,
    m: usize,
    k: usize,
    n: usize,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    imp::matmul_backward_f16_i8_no_host(grad, lhs, rhs, rhs_scale, m, k, n)
}

pub fn matmul_backward_f32_i8_no_host(
    grad: &CudaBuffer,
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    rhs_scale: f32,
    m: usize,
    k: usize,
    n: usize,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    imp::matmul_backward_f32_i8_no_host(grad, lhs, rhs, rhs_scale, m, k, n)
}

pub fn matmul_backward_i8_bf16_no_host(
    grad: &CudaBuffer,
    lhs: &CudaBuffer,
    lhs_scale: f32,
    rhs: &CudaBuffer,
    m: usize,
    k: usize,
    n: usize,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    imp::matmul_backward_i8_bf16_no_host(grad, lhs, lhs_scale, rhs, m, k, n)
}

pub fn matmul_backward_i8_f16_no_host(
    grad: &CudaBuffer,
    lhs: &CudaBuffer,
    lhs_scale: f32,
    rhs: &CudaBuffer,
    m: usize,
    k: usize,
    n: usize,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    imp::matmul_backward_i8_f16_no_host(grad, lhs, lhs_scale, rhs, m, k, n)
}

pub fn matmul_backward_i8_f32_no_host(
    grad: &CudaBuffer,
    lhs: &CudaBuffer,
    lhs_scale: f32,
    rhs: &CudaBuffer,
    m: usize,
    k: usize,
    n: usize,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    imp::matmul_backward_i8_f32_no_host(grad, lhs, lhs_scale, rhs, m, k, n)
}

pub fn batch_matmul_backward_f32_no_host(
    grad: &CudaBuffer,
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    batch_count: usize,
    m: usize,
    k: usize,
    n: usize,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    imp::batch_matmul_backward_f32_no_host(grad, lhs, rhs, batch_count, m, k, n)
}

pub fn batch_matmul_backward_bf16_no_host(
    grad: &CudaBuffer,
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    batch_count: usize,
    m: usize,
    k: usize,
    n: usize,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    imp::batch_matmul_backward_bf16_no_host(grad, lhs, rhs, batch_count, m, k, n)
}

pub fn batch_matmul_backward_f16_no_host(
    grad: &CudaBuffer,
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    batch_count: usize,
    m: usize,
    k: usize,
    n: usize,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    imp::batch_matmul_backward_f16_no_host(grad, lhs, rhs, batch_count, m, k, n)
}

pub fn batch_matmul_backward_bf16_i8_no_host(
    grad: &CudaBuffer,
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    rhs_scale: f32,
    batch_count: usize,
    m: usize,
    k: usize,
    n: usize,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    imp::batch_matmul_backward_bf16_i8_no_host(grad, lhs, rhs, rhs_scale, batch_count, m, k, n)
}

pub fn batch_matmul_backward_f16_i8_no_host(
    grad: &CudaBuffer,
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    rhs_scale: f32,
    batch_count: usize,
    m: usize,
    k: usize,
    n: usize,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    imp::batch_matmul_backward_f16_i8_no_host(grad, lhs, rhs, rhs_scale, batch_count, m, k, n)
}

pub fn batch_matmul_backward_f32_i8_no_host(
    grad: &CudaBuffer,
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    rhs_scale: f32,
    batch_count: usize,
    m: usize,
    k: usize,
    n: usize,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    imp::batch_matmul_backward_f32_i8_no_host(grad, lhs, rhs, rhs_scale, batch_count, m, k, n)
}

pub fn batch_matmul_backward_i8_bf16_no_host(
    grad: &CudaBuffer,
    lhs: &CudaBuffer,
    lhs_scale: f32,
    rhs: &CudaBuffer,
    batch_count: usize,
    m: usize,
    k: usize,
    n: usize,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    imp::batch_matmul_backward_i8_bf16_no_host(grad, lhs, lhs_scale, rhs, batch_count, m, k, n)
}

pub fn batch_matmul_backward_i8_f16_no_host(
    grad: &CudaBuffer,
    lhs: &CudaBuffer,
    lhs_scale: f32,
    rhs: &CudaBuffer,
    batch_count: usize,
    m: usize,
    k: usize,
    n: usize,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    imp::batch_matmul_backward_i8_f16_no_host(grad, lhs, lhs_scale, rhs, batch_count, m, k, n)
}

pub fn batch_matmul_backward_i8_f32_no_host(
    grad: &CudaBuffer,
    lhs: &CudaBuffer,
    lhs_scale: f32,
    rhs: &CudaBuffer,
    batch_count: usize,
    m: usize,
    k: usize,
    n: usize,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    imp::batch_matmul_backward_i8_f32_no_host(grad, lhs, lhs_scale, rhs, batch_count, m, k, n)
}

#[allow(clippy::too_many_arguments)]
pub fn batch_matmul_backward_i8_no_host(
    grad: &CudaBuffer,
    lhs: &CudaBuffer,
    lhs_scale: f32,
    rhs: &CudaBuffer,
    rhs_scale: f32,
    batch_count: usize,
    m: usize,
    k: usize,
    n: usize,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    imp::batch_matmul_backward_i8_no_host(
        grad,
        lhs,
        lhs_scale,
        rhs,
        rhs_scale,
        batch_count,
        m,
        k,
        n,
    )
}
