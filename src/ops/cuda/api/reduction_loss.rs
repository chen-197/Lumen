use super::*;

pub fn sum_f32(input: &CudaBuffer) -> Result<(CudaBuffer, Vec<f32>), String> {
    imp::sum_f32(input)
}

pub fn sum_f16_buffer(input: &CudaBuffer) -> Result<(CudaBuffer, Vec<f32>), String> {
    imp::sum_f16_buffer(input)
}

pub fn sum_bf16_buffer(input: &CudaBuffer) -> Result<(CudaBuffer, Vec<f32>), String> {
    imp::sum_bf16_buffer(input)
}

pub fn sum_i8_buffer(input: &CudaBuffer, scale: f32) -> Result<(CudaBuffer, Vec<f32>), String> {
    imp::sum_i8_buffer(input, scale)
}

pub fn sum_lastdim_f32_buffer(
    input: &CudaBuffer,
    rows: usize,
    last_dim: usize,
) -> Result<CudaBuffer, String> {
    imp::sum_lastdim_f32_buffer(input, rows, last_dim)
}

pub fn mse_forward_typed(
    output: &CudaBuffer,
    output_dtype: crate::precision::DType,
    output_scale: Option<f32>,
    target: &CudaBuffer,
    target_dtype: crate::precision::DType,
    target_scale: Option<f32>,
) -> Result<(CudaBuffer, CudaBuffer, Vec<f32>), String> {
    imp::mse_forward_typed(
        output,
        output_dtype,
        output_scale,
        target,
        target_dtype,
        target_scale,
    )
}

pub fn mse_backward_f32(diff: &CudaBuffer, factor: f32) -> Result<CudaTwoHostBuffers, String> {
    imp::mse_backward_f32(diff, factor)
}

pub fn mse_backward_f32_buffers(
    diff: &CudaBuffer,
    factor: f32,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    imp::mse_backward_f32_buffers(diff, factor)
}

pub fn cross_entropy_backward_f32(
    softmax: &CudaBuffer,
    target: &CudaBuffer,
    factor: f32,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    imp::cross_entropy_backward_f32(softmax, target, factor)
}

pub fn cross_entropy_backward_f32_buffer(
    softmax: &CudaBuffer,
    target: &CudaBuffer,
    factor: f32,
) -> Result<CudaBuffer, String> {
    imp::cross_entropy_backward_f32_buffer(softmax, target, factor)
}

pub fn cross_entropy_loss_f32(
    softmax: &CudaBuffer,
    target: &CudaBuffer,
    batch_size: usize,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    imp::cross_entropy_loss_f32(softmax, target, batch_size)
}

pub fn cross_entropy_backward_typed_target_buffer(
    softmax: &CudaBuffer,
    target: &CudaBuffer,
    target_dtype: crate::precision::DType,
    target_scale: Option<f32>,
    factor: f32,
) -> Result<CudaBuffer, String> {
    imp::cross_entropy_backward_typed_target_buffer(
        softmax,
        target,
        target_dtype,
        target_scale,
        factor,
    )
}

pub fn cross_entropy_backward_typed_target(
    softmax: &CudaBuffer,
    target: &CudaBuffer,
    target_dtype: crate::precision::DType,
    target_scale: Option<f32>,
    factor: f32,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    imp::cross_entropy_backward_typed_target(softmax, target, target_dtype, target_scale, factor)
}

pub fn cross_entropy_loss_typed_target(
    softmax: &CudaBuffer,
    target: &CudaBuffer,
    target_dtype: crate::precision::DType,
    target_scale: Option<f32>,
    batch_size: usize,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    imp::cross_entropy_loss_typed_target(softmax, target, target_dtype, target_scale, batch_size)
}

pub(crate) fn cross_entropy_logits_typed(
    logits: &CudaBuffer,
    logits_dtype: crate::precision::DType,
    logits_scale: Option<f32>,
    target: &CudaBuffer,
    target_dtype: crate::precision::DType,
    target_scale: Option<f32>,
    batch_size: usize,
) -> Result<(CudaBuffer, CudaBuffer, f32), String> {
    imp::cross_entropy_logits_typed(
        logits,
        logits_dtype,
        logits_scale,
        target,
        target_dtype,
        target_scale,
        batch_size,
    )
}

pub(crate) fn cross_entropy_logits_backward_typed(
    logits: &CudaBuffer,
    logits_dtype: crate::precision::DType,
    logits_scale: Option<f32>,
    target: &CudaBuffer,
    target_dtype: crate::precision::DType,
    target_scale: Option<f32>,
    probabilities: &CudaBuffer,
    batch_size: usize,
    factor: f32,
    want_logits: bool,
    want_target: bool,
) -> Result<(Option<CudaBuffer>, Option<CudaBuffer>), String> {
    imp::cross_entropy_logits_backward_typed(
        logits,
        logits_dtype,
        logits_scale,
        target,
        target_dtype,
        target_scale,
        probabilities,
        batch_size,
        factor,
        want_logits,
        want_target,
    )
}
