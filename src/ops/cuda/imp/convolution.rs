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
    let (out, out_h, out_w) = conv2d_f32_buffer(
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
    )?;
    let host = download_f32(&out)?;
    Ok((out, host, out_h, out_w))
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
    let input_len = checked_len(
        "CUDA conv2d input length",
        &[batch_size, in_channels, in_h, in_w],
    )?;
    let weight_len = checked_len(
        "CUDA conv2d weight length",
        &[out_channels, in_channels, k_h, k_w],
    )?;
    if input.len() != input_len {
        return Err(format!(
            "CUDA conv2d input length mismatch: expected {}, got {}",
            input_len,
            input.len()
        ));
    }
    if weight.len() != weight_len {
        return Err(format!(
            "CUDA conv2d weight length mismatch: expected {}, got {}",
            weight_len,
            weight.len()
        ));
    }
    if let Some(bias) = bias
        && bias.len() != out_channels
    {
        return Err(format!(
            "CUDA conv2d bias length mismatch: expected {}, got {}",
            out_channels,
            bias.len()
        ));
    }
    let out_h = conv_output_dim(in_h, pad_h, k_h, stride_h, "CUDA conv2d height")?;
    let out_w = conv_output_dim(in_w, pad_w, k_w, stride_w, "CUDA conv2d width")?;
    let output_len = checked_len(
        "CUDA conv2d output length",
        &[batch_size, out_channels, out_h, out_w],
    )?;
    let out = alloc_f32(output_len)?;
    let bias_handle = bias.map(|buf| buf.handle()).unwrap_or(0);
    let status = unsafe {
        lumen_cuda_conv2d_f32_device(
            input.handle(),
            weight.handle(),
            bias_handle,
            out.handle(),
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
            out_h,
            out_w,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok((out, out_h, out_w))
}

#[allow(clippy::too_many_arguments)]
pub fn conv2d_typed(
    input: &CudaBuffer,
    input_dtype: DType,
    input_scale: Option<f32>,
    weight: &CudaBuffer,
    weight_dtype: DType,
    weight_scale: Option<f32>,
    bias: Option<(&CudaBuffer, DType, Option<f32>)>,
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
    let (out, out_h, out_w) = conv2d_typed_buffer(
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
    )?;
    let host = download_f32(&out)?;
    Ok((out, host, out_h, out_w))
}

#[allow(clippy::too_many_arguments)]
pub fn conv2d_typed_buffer(
    input: &CudaBuffer,
    input_dtype: DType,
    input_scale: Option<f32>,
    weight: &CudaBuffer,
    weight_dtype: DType,
    weight_scale: Option<f32>,
    bias: Option<(&CudaBuffer, DType, Option<f32>)>,
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
    if stride_h == 0 || stride_w == 0 {
        return Err("CUDA typed conv2d stride must be greater than zero".to_string());
    }
    let input_len = checked_len(
        "CUDA typed conv2d input length",
        &[batch_size, in_channels, in_h, in_w],
    )?;
    let weight_len = checked_len(
        "CUDA typed conv2d weight length",
        &[out_channels, in_channels, k_h, k_w],
    )?;
    if input.len() != input_len {
        return Err(format!(
            "CUDA typed conv2d input length mismatch: expected {}, got {}",
            input_len,
            input.len()
        ));
    }
    if weight.len() != weight_len {
        return Err(format!(
            "CUDA typed conv2d weight length mismatch: expected {}, got {}",
            weight_len,
            weight.len()
        ));
    }
    if let Some((bias, _, _)) = bias
        && bias.len() != out_channels
    {
        return Err(format!(
            "CUDA typed conv2d bias length mismatch: expected {}, got {}",
            out_channels,
            bias.len()
        ));
    }
    if input_dtype == DType::I8
        && !input_scale.is_some_and(|scale| scale.is_finite() && scale > 0.0)
    {
        return Err("CUDA typed conv2d input I8 scale must be finite and > 0".to_string());
    }
    if weight_dtype == DType::I8
        && !weight_scale.is_some_and(|scale| scale.is_finite() && scale > 0.0)
    {
        return Err("CUDA typed conv2d weight I8 scale must be finite and > 0".to_string());
    }
    if let Some((_, dtype, scale)) = bias
        && dtype == DType::I8
        && !scale.is_some_and(|value| value.is_finite() && value > 0.0)
    {
        return Err("CUDA typed conv2d bias I8 scale must be finite and > 0".to_string());
    }
    let out_h = conv_output_dim(in_h, pad_h, k_h, stride_h, "CUDA typed conv2d height")?;
    let out_w = conv_output_dim(in_w, pad_w, k_w, stride_w, "CUDA typed conv2d width")?;
    let output_len = checked_len(
        "CUDA typed conv2d output length",
        &[batch_size, out_channels, out_h, out_w],
    )?;
    let out = alloc_f32(output_len)?;
    let (bias_handle, bias_dtype, bias_scale) = match bias {
        Some((buffer, dtype, scale)) => (buffer.handle(), dtype_tag(dtype)?, scale.unwrap_or(1.0)),
        None => (0, dtype_tag(DType::F32)?, 1.0),
    };
    let status = unsafe {
        lumen_cuda_conv2d_typed_device(
            input.handle(),
            dtype_tag(input_dtype)?,
            input_scale.unwrap_or(1.0),
            weight.handle(),
            dtype_tag(weight_dtype)?,
            weight_scale.unwrap_or(1.0),
            bias_handle,
            bias_dtype,
            bias_scale,
            out.handle(),
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
            out_h,
            out_w,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok((out, out_h, out_w))
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
    let (grad_input, grad_weight, grad_bias) = conv2d_backward_f32_buffers(
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
    )?;
    let grad_input_host = download_f32(&grad_input)?;
    let grad_weight_host = download_f32(&grad_weight)?;
    let grad_bias_host = match grad_bias {
        Some(buffer) => Some((buffer.clone(), download_f32(&buffer)?)),
        None => None,
    };
    Ok((
        grad_input,
        grad_input_host,
        grad_weight,
        grad_weight_host,
        grad_bias_host,
    ))
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
    let out_h = conv_output_dim(in_h, pad_h, k_h, stride_h, "CUDA conv2d backward height")?;
    let out_w = conv_output_dim(in_w, pad_w, k_w, stride_w, "CUDA conv2d backward width")?;
    let input_len = checked_len(
        "CUDA conv2d backward input length",
        &[batch_size, in_channels, in_h, in_w],
    )?;
    let weight_len = checked_len(
        "CUDA conv2d backward weight length",
        &[out_channels, in_channels, k_h, k_w],
    )?;
    let grad_output_len = checked_len(
        "CUDA conv2d backward grad output length",
        &[batch_size, out_channels, out_h, out_w],
    )?;
    if input.len() != input_len {
        return Err(format!(
            "CUDA conv2d backward input length mismatch: expected {}, got {}",
            input_len,
            input.len()
        ));
    }
    if weight.len() != weight_len {
        return Err(format!(
            "CUDA conv2d backward weight length mismatch: expected {}, got {}",
            weight_len,
            weight.len()
        ));
    }
    if grad_output.len() != grad_output_len {
        return Err(format!(
            "CUDA conv2d backward grad output length mismatch: expected {}, got {}",
            grad_output_len,
            grad_output.len()
        ));
    }
    let grad_input = alloc_f32(input_len)?;
    let grad_weight = alloc_f32(weight_len)?;
    let grad_bias = if compute_bias_grad {
        Some(alloc_f32(out_channels)?)
    } else {
        None
    };
    let status = unsafe {
        lumen_cuda_conv2d_backward_f32_device(
            input.handle(),
            weight.handle(),
            grad_output.handle(),
            grad_input.handle(),
            grad_weight.handle(),
            grad_bias.as_ref().map(|buf| buf.handle()).unwrap_or(0),
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
            out_h,
            out_w,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok((grad_input, grad_weight, grad_bias))
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
    if kernel_h == 0 || kernel_w == 0 || stride_h == 0 || stride_w == 0 {
        return Err("CUDA max_pool2d kernel and stride must be greater than zero".to_string());
    }
    if in_h < kernel_h || in_w < kernel_w {
        return Err("CUDA max_pool2d kernel is larger than input".to_string());
    }
    let input_len = batch_size
        .checked_mul(channels)
        .and_then(|v| v.checked_mul(in_h))
        .and_then(|v| v.checked_mul(in_w))
        .ok_or_else(|| "CUDA max_pool2d input length overflow".to_string())?;
    if input.len() != input_len {
        return Err(format!(
            "CUDA max_pool2d input length mismatch: expected {}, got {}",
            input_len,
            input.len()
        ));
    }
    let out_h = (in_h - kernel_h) / stride_h + 1;
    let out_w = (in_w - kernel_w) / stride_w + 1;
    let out_len = batch_size
        .checked_mul(channels)
        .and_then(|v| v.checked_mul(out_h))
        .and_then(|v| v.checked_mul(out_w))
        .ok_or_else(|| "CUDA max_pool2d output length overflow".to_string())?;
    let out = alloc_f32(out_len)?;
    let status = unsafe {
        lumen_cuda_max_pool2d_f32_device(
            input.handle(),
            out.handle(),
            batch_size,
            channels,
            in_h,
            in_w,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            out_h,
            out_w,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    let host = download_f32(&out)?;
    Ok((out, host, out_h, out_w))
}

#[allow(clippy::too_many_arguments)]
pub fn max_pool2d_typed(
    input: &CudaBuffer,
    input_dtype: DType,
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
    if kernel_h == 0 || kernel_w == 0 || stride_h == 0 || stride_w == 0 {
        return Err(
            "CUDA typed max_pool2d kernel and stride must be greater than zero".to_string(),
        );
    }
    if in_h < kernel_h || in_w < kernel_w {
        return Err("CUDA typed max_pool2d kernel is larger than input".to_string());
    }
    let input_len = batch_size
        .checked_mul(channels)
        .and_then(|v| v.checked_mul(in_h))
        .and_then(|v| v.checked_mul(in_w))
        .ok_or_else(|| "CUDA typed max_pool2d input length overflow".to_string())?;
    if input.len() != input_len {
        return Err(format!(
            "CUDA typed max_pool2d input length mismatch: expected {}, got {}",
            input_len,
            input.len()
        ));
    }
    if input_dtype == DType::I8
        && !input_scale.is_some_and(|scale| scale.is_finite() && scale > 0.0)
    {
        return Err("CUDA typed max_pool2d I8 scale must be finite and > 0".to_string());
    }
    let out_h = (in_h - kernel_h) / stride_h + 1;
    let out_w = (in_w - kernel_w) / stride_w + 1;
    let out_len = batch_size
        .checked_mul(channels)
        .and_then(|v| v.checked_mul(out_h))
        .and_then(|v| v.checked_mul(out_w))
        .ok_or_else(|| "CUDA typed max_pool2d output length overflow".to_string())?;
    let out = alloc_f32(out_len)?;
    let status = unsafe {
        lumen_cuda_max_pool2d_typed_device(
            input.handle(),
            dtype_tag(input_dtype)?,
            input_scale.unwrap_or(1.0),
            out.handle(),
            batch_size,
            channels,
            in_h,
            in_w,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            out_h,
            out_w,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    let host = download_f32(&out)?;
    Ok((out, host, out_h, out_w))
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
    let grad_input = max_pool2d_backward_f32_buffer(
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
    )?;
    let host = download_f32(&grad_input)?;
    Ok((grad_input, host))
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
    if kernel_h == 0 || kernel_w == 0 || stride_h == 0 || stride_w == 0 {
        return Err(
            "CUDA max_pool2d backward kernel and stride must be greater than zero".to_string(),
        );
    }
    if in_h < kernel_h || in_w < kernel_w {
        return Err("CUDA max_pool2d backward kernel is larger than input".to_string());
    }
    let out_h = (in_h - kernel_h) / stride_h + 1;
    let out_w = (in_w - kernel_w) / stride_w + 1;
    let input_len = batch_size
        .checked_mul(channels)
        .and_then(|v| v.checked_mul(in_h))
        .and_then(|v| v.checked_mul(in_w))
        .ok_or_else(|| "CUDA max_pool2d backward input length overflow".to_string())?;
    let grad_output_len = batch_size
        .checked_mul(channels)
        .and_then(|v| v.checked_mul(out_h))
        .and_then(|v| v.checked_mul(out_w))
        .ok_or_else(|| "CUDA max_pool2d backward grad output length overflow".to_string())?;
    if input.len() != input_len {
        return Err(format!(
            "CUDA max_pool2d backward input length mismatch: expected {}, got {}",
            input_len,
            input.len()
        ));
    }
    if grad_output.len() != grad_output_len {
        return Err(format!(
            "CUDA max_pool2d backward grad output length mismatch: expected {}, got {}",
            grad_output_len,
            grad_output.len()
        ));
    }
    let grad_input = alloc_f32(input_len)?;
    let status = unsafe {
        lumen_cuda_max_pool2d_backward_f32_device(
            input.handle(),
            grad_output.handle(),
            grad_input.handle(),
            batch_size,
            channels,
            in_h,
            in_w,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            out_h,
            out_w,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(grad_input)
}

#[allow(clippy::too_many_arguments)]
pub fn max_pool2d_backward_typed_buffer(
    input: &CudaBuffer,
    input_dtype: DType,
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
    if kernel_h == 0 || kernel_w == 0 || stride_h == 0 || stride_w == 0 {
        return Err(
            "CUDA typed max_pool2d backward kernel and stride must be greater than zero"
                .to_string(),
        );
    }
    if in_h < kernel_h || in_w < kernel_w {
        return Err("CUDA typed max_pool2d backward kernel is larger than input".to_string());
    }
    if input_dtype == DType::I8
        && !input_scale.is_some_and(|scale| scale.is_finite() && scale > 0.0)
    {
        return Err("CUDA typed max_pool2d backward I8 scale must be finite and > 0".to_string());
    }
    let out_h = (in_h - kernel_h) / stride_h + 1;
    let out_w = (in_w - kernel_w) / stride_w + 1;
    let input_len = batch_size
        .checked_mul(channels)
        .and_then(|v| v.checked_mul(in_h))
        .and_then(|v| v.checked_mul(in_w))
        .ok_or_else(|| "CUDA typed max_pool2d backward input length overflow".to_string())?;
    let grad_output_len = batch_size
        .checked_mul(channels)
        .and_then(|v| v.checked_mul(out_h))
        .and_then(|v| v.checked_mul(out_w))
        .ok_or_else(|| "CUDA typed max_pool2d backward grad output length overflow".to_string())?;
    if input.len() != input_len {
        return Err(format!(
            "CUDA typed max_pool2d backward input length mismatch: expected {}, got {}",
            input_len,
            input.len()
        ));
    }
    if grad_output.len() != grad_output_len {
        return Err(format!(
            "CUDA typed max_pool2d backward grad output length mismatch: expected {}, got {}",
            grad_output_len,
            grad_output.len()
        ));
    }
    let grad_input = alloc_f32(input_len)?;
    let status = unsafe {
        lumen_cuda_max_pool2d_backward_typed_device(
            input.handle(),
            dtype_tag(input_dtype)?,
            input_scale.unwrap_or(1.0),
            grad_output.handle(),
            grad_input.handle(),
            batch_size,
            channels,
            in_h,
            in_w,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            out_h,
            out_w,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(grad_input)
}
