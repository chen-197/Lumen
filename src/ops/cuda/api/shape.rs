use super::*;

pub fn bshd_to_bhsd_add_bias_f32_buffer(
    input: &CudaBuffer,
    bias: &CudaBuffer,
    batch: usize,
    seq: usize,
    heads: usize,
    dim: usize,
) -> Result<CudaBuffer, String> {
    imp::bshd_to_bhsd_add_bias_f32_buffer(input, bias, batch, seq, heads, dim)
}

pub fn embedding_f32(
    indices: &CudaBuffer,
    weight: &CudaBuffer,
    num_indices: usize,
    vocab_size: usize,
    embed_dim: usize,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    imp::embedding_f32(indices, weight, num_indices, vocab_size, embed_dim)
}

pub fn embedding_f32_buffer(
    indices: &CudaBuffer,
    weight: &CudaBuffer,
    num_indices: usize,
    vocab_size: usize,
    embed_dim: usize,
) -> Result<CudaBuffer, String> {
    imp::embedding_f32_buffer(indices, weight, num_indices, vocab_size, embed_dim)
}

pub fn embedding_typed(
    indices: &CudaBuffer,
    weight: &CudaBuffer,
    weight_dtype: crate::precision::DType,
    weight_scale: Option<f32>,
    num_indices: usize,
    vocab_size: usize,
    embed_dim: usize,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    imp::embedding_typed(
        indices,
        weight,
        weight_dtype,
        weight_scale,
        num_indices,
        vocab_size,
        embed_dim,
    )
}

pub fn embedding_typed_buffer(
    indices: &CudaBuffer,
    weight: &CudaBuffer,
    weight_dtype: crate::precision::DType,
    weight_scale: Option<f32>,
    num_indices: usize,
    vocab_size: usize,
    embed_dim: usize,
) -> Result<CudaBuffer, String> {
    imp::embedding_typed_buffer(
        indices,
        weight,
        weight_dtype,
        weight_scale,
        num_indices,
        vocab_size,
        embed_dim,
    )
}

pub fn embedding_typed_same_dtype_buffer(
    indices: &CudaBuffer,
    weight: &CudaBuffer,
    weight_dtype: crate::precision::DType,
    num_indices: usize,
    vocab_size: usize,
    embed_dim: usize,
) -> Result<CudaBuffer, String> {
    imp::embedding_typed_same_dtype_buffer(
        indices,
        weight,
        weight_dtype,
        num_indices,
        vocab_size,
        embed_dim,
    )
}

pub fn embedding_backward_f32(
    indices: &CudaBuffer,
    grad: &CudaBuffer,
    num_indices: usize,
    vocab_size: usize,
    embed_dim: usize,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    imp::embedding_backward_f32(indices, grad, num_indices, vocab_size, embed_dim)
}

pub fn embedding_backward_f32_buffer(
    indices: &CudaBuffer,
    grad: &CudaBuffer,
    num_indices: usize,
    vocab_size: usize,
    embed_dim: usize,
) -> Result<CudaBuffer, String> {
    imp::embedding_backward_f32_buffer(indices, grad, num_indices, vocab_size, embed_dim)
}

pub fn permute_f32(
    input: &CudaBuffer,
    out_shape: &[usize],
    axes: &[usize],
) -> Result<(CudaBuffer, Vec<f32>), String> {
    imp::permute_f32(input, out_shape, axes)
}

pub fn permute_f32_buffer(
    input: &CudaBuffer,
    out_shape: &[usize],
    axes: &[usize],
) -> Result<CudaBuffer, String> {
    imp::permute_f32_buffer(input, out_shape, axes)
}

pub fn permute_typed_buffer(
    input: &CudaBuffer,
    dtype: crate::precision::DType,
    out_shape: &[usize],
    axes: &[usize],
) -> Result<CudaBuffer, String> {
    imp::permute_typed_buffer(input, dtype, out_shape, axes)
}

pub fn slice_lastdim_f32(
    input: &CudaBuffer,
    outer: usize,
    input_last_dim: usize,
    start: usize,
    slice_len: usize,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    imp::slice_lastdim_f32(input, outer, input_last_dim, start, slice_len)
}

pub fn slice_lastdim_f32_buffer(
    input: &CudaBuffer,
    outer: usize,
    input_last_dim: usize,
    start: usize,
    slice_len: usize,
) -> Result<CudaBuffer, String> {
    imp::slice_lastdim_f32_buffer(input, outer, input_last_dim, start, slice_len)
}

pub fn slice_lastdim_typed_buffer(
    input: &CudaBuffer,
    dtype: crate::precision::DType,
    outer: usize,
    input_last_dim: usize,
    start: usize,
    slice_len: usize,
) -> Result<CudaBuffer, String> {
    imp::slice_lastdim_typed_buffer(input, dtype, outer, input_last_dim, start, slice_len)
}

pub fn slice_lastdim_backward_f32(
    grad: &CudaBuffer,
    outer: usize,
    input_last_dim: usize,
    start: usize,
    slice_len: usize,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    imp::slice_lastdim_backward_f32(grad, outer, input_last_dim, start, slice_len)
}

pub fn slice_lastdim_backward_f32_buffer(
    grad: &CudaBuffer,
    outer: usize,
    input_last_dim: usize,
    start: usize,
    slice_len: usize,
) -> Result<CudaBuffer, String> {
    imp::slice_lastdim_backward_f32_buffer(grad, outer, input_last_dim, start, slice_len)
}

pub fn cat_f32(
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    out_shape: &[usize],
    axis: usize,
    lhs_axis_len: usize,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    imp::cat_f32(lhs, rhs, out_shape, axis, lhs_axis_len)
}

pub fn cat_f32_buffer(
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    out_shape: &[usize],
    axis: usize,
    lhs_axis_len: usize,
) -> Result<CudaBuffer, String> {
    imp::cat_f32_buffer(lhs, rhs, out_shape, axis, lhs_axis_len)
}

pub fn cat_typed_buffer(
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    dtype: crate::precision::DType,
    out_shape: &[usize],
    axis: usize,
    lhs_axis_len: usize,
) -> Result<CudaBuffer, String> {
    imp::cat_typed_buffer(lhs, rhs, dtype, out_shape, axis, lhs_axis_len)
}

pub fn cat_backward_slice_f32(
    grad: &CudaBuffer,
    input_shape: &[usize],
    out_shape: &[usize],
    axis: usize,
    axis_start: usize,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    imp::cat_backward_slice_f32(grad, input_shape, out_shape, axis, axis_start)
}

pub fn cat_backward_slice_f32_buffer(
    grad: &CudaBuffer,
    input_shape: &[usize],
    out_shape: &[usize],
    axis: usize,
    axis_start: usize,
) -> Result<CudaBuffer, String> {
    imp::cat_backward_slice_f32_buffer(grad, input_shape, out_shape, axis, axis_start)
}
