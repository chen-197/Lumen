use super::*;

#[allow(clippy::too_many_arguments)]
pub fn conv2d_f32(
    input: &CudaBuffer,
    weight: &CudaBuffer,
    bias: Option<&CudaBuffer>,
    batch_size: usize,
    in_channels: usize,
    in_h: usize,
    in_w: usize,
    out_channels: usize,
    k_h: usize,
    k_w: usize,
    pad_h: usize,
    pad_w: usize,
    stride_h: usize,
    stride_w: usize,
) -> Result<(CudaBuffer, Vec<f32>, usize, usize), String> {
    imp::conv2d_f32(
        input,
        weight,
        bias,
        batch_size,
        in_channels,
        in_h,
        in_w,
        out_channels,
        k_h,
        k_w,
        pad_h,
        pad_w,
        stride_h,
        stride_w,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn conv2d_f32_buffer(
    input: &CudaBuffer,
    weight: &CudaBuffer,
    bias: Option<&CudaBuffer>,
    batch_size: usize,
    in_channels: usize,
    in_h: usize,
    in_w: usize,
    out_channels: usize,
    k_h: usize,
    k_w: usize,
    pad_h: usize,
    pad_w: usize,
    stride_h: usize,
    stride_w: usize,
) -> Result<(CudaBuffer, usize, usize), String> {
    imp::conv2d_f32_buffer(
        input,
        weight,
        bias,
        batch_size,
        in_channels,
        in_h,
        in_w,
        out_channels,
        k_h,
        k_w,
        pad_h,
        pad_w,
        stride_h,
        stride_w,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn conv2d_typed(
    input: &CudaBuffer,
    input_dtype: crate::precision::DType,
    input_scale: Option<f32>,
    weight: &CudaBuffer,
    weight_dtype: crate::precision::DType,
    weight_scale: Option<f32>,
    bias: Option<(&CudaBuffer, crate::precision::DType, Option<f32>)>,
    batch_size: usize,
    in_channels: usize,
    in_h: usize,
    in_w: usize,
    out_channels: usize,
    k_h: usize,
    k_w: usize,
    pad_h: usize,
    pad_w: usize,
    stride_h: usize,
    stride_w: usize,
) -> Result<(CudaBuffer, Vec<f32>, usize, usize), String> {
    imp::conv2d_typed(
        input,
        input_dtype,
        input_scale,
        weight,
        weight_dtype,
        weight_scale,
        bias,
        batch_size,
        in_channels,
        in_h,
        in_w,
        out_channels,
        k_h,
        k_w,
        pad_h,
        pad_w,
        stride_h,
        stride_w,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn conv2d_typed_buffer(
    input: &CudaBuffer,
    input_dtype: crate::precision::DType,
    input_scale: Option<f32>,
    weight: &CudaBuffer,
    weight_dtype: crate::precision::DType,
    weight_scale: Option<f32>,
    bias: Option<(&CudaBuffer, crate::precision::DType, Option<f32>)>,
    batch_size: usize,
    in_channels: usize,
    in_h: usize,
    in_w: usize,
    out_channels: usize,
    k_h: usize,
    k_w: usize,
    pad_h: usize,
    pad_w: usize,
    stride_h: usize,
    stride_w: usize,
) -> Result<(CudaBuffer, usize, usize), String> {
    imp::conv2d_typed_buffer(
        input,
        input_dtype,
        input_scale,
        weight,
        weight_dtype,
        weight_scale,
        bias,
        batch_size,
        in_channels,
        in_h,
        in_w,
        out_channels,
        k_h,
        k_w,
        pad_h,
        pad_w,
        stride_h,
        stride_w,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn conv2d_backward_f32(
    input: &CudaBuffer,
    weight: &CudaBuffer,
    grad_output: &CudaBuffer,
    compute_bias_grad: bool,
    batch_size: usize,
    in_channels: usize,
    in_h: usize,
    in_w: usize,
    out_channels: usize,
    k_h: usize,
    k_w: usize,
    pad_h: usize,
    pad_w: usize,
    stride_h: usize,
    stride_w: usize,
) -> Result<CudaConv2dBackwardHostBuffers, String> {
    imp::conv2d_backward_f32(
        input,
        weight,
        grad_output,
        compute_bias_grad,
        batch_size,
        in_channels,
        in_h,
        in_w,
        out_channels,
        k_h,
        k_w,
        pad_h,
        pad_w,
        stride_h,
        stride_w,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn conv2d_backward_f32_buffers(
    input: &CudaBuffer,
    weight: &CudaBuffer,
    grad_output: &CudaBuffer,
    compute_bias_grad: bool,
    batch_size: usize,
    in_channels: usize,
    in_h: usize,
    in_w: usize,
    out_channels: usize,
    k_h: usize,
    k_w: usize,
    pad_h: usize,
    pad_w: usize,
    stride_h: usize,
    stride_w: usize,
) -> Result<(CudaBuffer, CudaBuffer, Option<CudaBuffer>), String> {
    imp::conv2d_backward_f32_buffers(
        input,
        weight,
        grad_output,
        compute_bias_grad,
        batch_size,
        in_channels,
        in_h,
        in_w,
        out_channels,
        k_h,
        k_w,
        pad_h,
        pad_w,
        stride_h,
        stride_w,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn max_pool2d_f32(
    input: &CudaBuffer,
    batch_size: usize,
    channels: usize,
    in_h: usize,
    in_w: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
) -> Result<(CudaBuffer, Vec<f32>, usize, usize), String> {
    imp::max_pool2d_f32(
        input, batch_size, channels, in_h, in_w, kernel_h, kernel_w, stride_h, stride_w,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn max_pool2d_typed(
    input: &CudaBuffer,
    input_dtype: crate::precision::DType,
    input_scale: Option<f32>,
    batch_size: usize,
    channels: usize,
    in_h: usize,
    in_w: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
) -> Result<(CudaBuffer, Vec<f32>, usize, usize), String> {
    imp::max_pool2d_typed(
        input,
        input_dtype,
        input_scale,
        batch_size,
        channels,
        in_h,
        in_w,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn max_pool2d_backward_f32(
    input: &CudaBuffer,
    grad_output: &CudaBuffer,
    batch_size: usize,
    channels: usize,
    in_h: usize,
    in_w: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    imp::max_pool2d_backward_f32(
        input,
        grad_output,
        batch_size,
        channels,
        in_h,
        in_w,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn max_pool2d_backward_typed_buffer(
    input: &CudaBuffer,
    input_dtype: crate::precision::DType,
    input_scale: Option<f32>,
    grad_output: &CudaBuffer,
    batch_size: usize,
    channels: usize,
    in_h: usize,
    in_w: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
) -> Result<CudaBuffer, String> {
    imp::max_pool2d_backward_typed_buffer(
        input,
        input_dtype,
        input_scale,
        grad_output,
        batch_size,
        channels,
        in_h,
        in_w,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn max_pool2d_backward_f32_buffer(
    input: &CudaBuffer,
    grad_output: &CudaBuffer,
    batch_size: usize,
    channels: usize,
    in_h: usize,
    in_w: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
) -> Result<CudaBuffer, String> {
    imp::max_pool2d_backward_f32_buffer(
        input,
        grad_output,
        batch_size,
        channels,
        in_h,
        in_w,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
    )
}
