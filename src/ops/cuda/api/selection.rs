use super::*;

pub fn matvec_argmax_f32(
    input: &CudaBuffer,
    weight: &CudaBuffer,
    batch_size: usize,
    vocab_size: usize,
    hidden_size: usize,
) -> Result<Vec<usize>, String> {
    imp::matvec_argmax_f32(input, weight, batch_size, vocab_size, hidden_size)
}

pub fn matvec_argmax_bf16_i8(
    input: &CudaBuffer,
    weight: &CudaBuffer,
    weight_scale: f32,
    batch_size: usize,
    vocab_size: usize,
    hidden_size: usize,
) -> Result<Vec<usize>, String> {
    imp::matvec_argmax_bf16_i8(
        input,
        weight,
        weight_scale,
        batch_size,
        vocab_size,
        hidden_size,
    )
}

pub fn matvec_argmax_f16_i8(
    input: &CudaBuffer,
    weight: &CudaBuffer,
    weight_scale: f32,
    batch_size: usize,
    vocab_size: usize,
    hidden_size: usize,
) -> Result<Vec<usize>, String> {
    imp::matvec_argmax_f16_i8(
        input,
        weight,
        weight_scale,
        batch_size,
        vocab_size,
        hidden_size,
    )
}

pub fn matvec_argmax_f32_i8(
    input: &CudaBuffer,
    weight: &CudaBuffer,
    weight_scale: f32,
    batch_size: usize,
    vocab_size: usize,
    hidden_size: usize,
) -> Result<Vec<usize>, String> {
    imp::matvec_argmax_f32_i8(
        input,
        weight,
        weight_scale,
        batch_size,
        vocab_size,
        hidden_size,
    )
}

pub fn matvec_argmax_i8_i8(
    input: &CudaBuffer,
    input_scale: f32,
    weight: &CudaBuffer,
    weight_scale: f32,
    batch_size: usize,
    vocab_size: usize,
    hidden_size: usize,
) -> Result<Vec<usize>, String> {
    imp::matvec_argmax_i8_i8(
        input,
        input_scale,
        weight,
        weight_scale,
        batch_size,
        vocab_size,
        hidden_size,
    )
}
