use super::*;

pub fn softmax_lastdim_f32(
    input: &CudaBuffer,
    outer: usize,
    last_dim: usize,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    let expected_len = checked_len("CUDA softmax input length", &[outer, last_dim])?;
    if input.len() != expected_len {
        return Err(format!(
            "CUDA softmax input length mismatch: expected {}, got {}",
            expected_len,
            input.len()
        ));
    }
    let out = alloc_f32(input.len())?;
    let status = unsafe {
        lumen_cuda_softmax_lastdim_f32_device(input.handle(), out.handle(), outer, last_dim)
    };
    if status != 0 {
        return Err(last_error_message());
    }
    let host = download_f32(&out)?;
    Ok((out, host))
}

pub fn softmax_lastdim_f32_no_host(
    input: &CudaBuffer,
    outer: usize,
    last_dim: usize,
) -> Result<CudaBuffer, String> {
    let expected_len = checked_len("CUDA softmax input length", &[outer, last_dim])?;
    if input.len() != expected_len {
        return Err(format!(
            "CUDA softmax input length mismatch: expected {}, got {}",
            expected_len,
            input.len()
        ));
    }
    let out = alloc_f32(input.len())?;
    let status = unsafe {
        lumen_cuda_softmax_lastdim_f32_device(input.handle(), out.handle(), outer, last_dim)
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}

pub fn softmax_lastdim_typed(
    input: &CudaBuffer,
    input_dtype: DType,
    input_scale: Option<f32>,
    outer: usize,
    last_dim: usize,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    let out = softmax_lastdim_typed_no_host(input, input_dtype, input_scale, outer, last_dim)?;
    let host = download_f32(&out)?;
    Ok((out, host))
}

pub fn softmax_lastdim_typed_no_host(
    input: &CudaBuffer,
    input_dtype: DType,
    input_scale: Option<f32>,
    outer: usize,
    last_dim: usize,
) -> Result<CudaBuffer, String> {
    let expected_len = outer
        .checked_mul(last_dim)
        .ok_or_else(|| "CUDA typed softmax input length overflow".to_string())?;
    if input.len() != expected_len {
        return Err(format!(
            "CUDA typed softmax input length mismatch: expected {}, got {}",
            expected_len,
            input.len()
        ));
    }
    let out = alloc_f32(expected_len)?;
    let status = unsafe {
        lumen_cuda_softmax_lastdim_typed_device(
            input.handle(),
            dtype_tag(input_dtype)?,
            input_scale.unwrap_or(1.0),
            out.handle(),
            outer,
            last_dim,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}

pub fn softmax_lastdim_backward_f32(
    output: &CudaBuffer,
    grad: &CudaBuffer,
    outer: usize,
    last_dim: usize,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    let out = softmax_lastdim_backward_f32_buffer(output, grad, outer, last_dim)?;
    let host = download_f32(&out)?;
    Ok((out, host))
}

pub fn softmax_lastdim_backward_f32_buffer(
    output: &CudaBuffer,
    grad: &CudaBuffer,
    outer: usize,
    last_dim: usize,
) -> Result<CudaBuffer, String> {
    let len = outer
        .checked_mul(last_dim)
        .ok_or_else(|| "CUDA softmax backward length overflow".to_string())?;
    if output.len() != len || grad.len() != len {
        return Err(format!(
            "CUDA softmax backward length mismatch: expected {}, output={}, grad={}",
            len,
            output.len(),
            grad.len()
        ));
    }
    let out = alloc_f32(len)?;
    let status = unsafe {
        lumen_cuda_softmax_lastdim_backward_f32_device(
            output.handle(),
            grad.handle(),
            out.handle(),
            outer,
            last_dim,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}

pub fn fused_softmax_f32(
    input: &CudaBuffer,
    batch_heads: usize,
    q_len: usize,
    k_len: usize,
    scale: f32,
    is_causal: bool,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    let out = fused_softmax_f32_no_host(input, batch_heads, q_len, k_len, scale, is_causal)?;
    let host = download_f32(&out)?;
    Ok((out, host))
}

pub fn fused_softmax_f32_no_host(
    input: &CudaBuffer,
    batch_heads: usize,
    q_len: usize,
    k_len: usize,
    scale: f32,
    is_causal: bool,
) -> Result<CudaBuffer, String> {
    ensure_finite("CUDA fused_softmax scale", scale)?;
    let expected_len = batch_heads
        .checked_mul(q_len)
        .and_then(|value| value.checked_mul(k_len))
        .ok_or_else(|| "CUDA fused_softmax input length overflow".to_string())?;
    if input.len() != expected_len {
        return Err(format!(
            "CUDA fused_softmax input length mismatch: expected {}, got {}",
            expected_len,
            input.len()
        ));
    }
    let out = alloc_f32(expected_len)?;
    let status = unsafe {
        lumen_cuda_fused_softmax_f32_device(
            input.handle(),
            out.handle(),
            batch_heads,
            q_len,
            k_len,
            scale,
            if is_causal { 1 } else { 0 },
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}

pub fn fused_softmax_backward_f32(
    output: &CudaBuffer,
    grad: &CudaBuffer,
    batch_heads: usize,
    q_len: usize,
    k_len: usize,
    scale: f32,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    let out = fused_softmax_backward_f32_buffer(output, grad, batch_heads, q_len, k_len, scale)?;
    let host = download_f32(&out)?;
    Ok((out, host))
}

pub fn fused_softmax_backward_f32_buffer(
    output: &CudaBuffer,
    grad: &CudaBuffer,
    batch_heads: usize,
    q_len: usize,
    k_len: usize,
    scale: f32,
) -> Result<CudaBuffer, String> {
    ensure_finite("CUDA fused_softmax backward scale", scale)?;
    let expected_len = batch_heads
        .checked_mul(q_len)
        .and_then(|value| value.checked_mul(k_len))
        .ok_or_else(|| "CUDA fused_softmax backward input length overflow".to_string())?;
    if output.len() != expected_len || grad.len() != expected_len {
        return Err(format!(
            "CUDA fused_softmax backward length mismatch: expected {}, output={}, grad={}",
            expected_len,
            output.len(),
            grad.len()
        ));
    }
    let out = alloc_f32(expected_len)?;
    let status = unsafe {
        lumen_cuda_fused_softmax_backward_f32_device(
            output.handle(),
            grad.handle(),
            out.handle(),
            batch_heads,
            q_len,
            k_len,
            scale,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
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
    let out = fused_softmax_f32_with_past_no_host(
        input,
        batch_heads,
        q_len,
        k_len,
        scale,
        is_causal,
        past_len,
    )?;
    let host = download_f32(&out)?;
    Ok((out, host))
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
    ensure_finite("CUDA fused_softmax_with_past scale", scale)?;
    let expected_len = batch_heads
        .checked_mul(q_len)
        .and_then(|value| value.checked_mul(k_len))
        .ok_or_else(|| "CUDA fused_softmax_with_past input length overflow".to_string())?;
    if input.len() != expected_len {
        return Err(format!(
            "CUDA fused_softmax_with_past input length mismatch: expected {}, got {}",
            expected_len,
            input.len()
        ));
    }
    if batch_heads == 0 || q_len == 0 || k_len == 0 {
        return Err(
            "CUDA fused_softmax_with_past dimensions must be greater than zero".to_string(),
        );
    }
    let causal_window_end = past_len
        .checked_add(q_len)
        .ok_or_else(|| "CUDA fused_softmax_with_past causal window overflow".to_string())?;
    if causal_window_end > k_len {
        return Err(format!(
            "CUDA fused_softmax_with_past causal window out of bounds: past_len({}) + q_len({}) > k_len({})",
            past_len, q_len, k_len
        ));
    }
    let out = alloc_f32(expected_len)?;
    let status = unsafe {
        lumen_cuda_fused_softmax_f32_with_past_device(
            input.handle(),
            out.handle(),
            batch_heads,
            q_len,
            k_len,
            scale,
            if is_causal { 1 } else { 0 },
            past_len,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}

pub fn rms_norm_f32(
    input: &CudaBuffer,
    weight: &CudaBuffer,
    rows: usize,
    dim: usize,
    eps: f32,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    let out = rms_norm_f32_buffer(input, weight, rows, dim, eps)?;
    let host = download_f32(&out)?;
    Ok((out, host))
}

pub fn rms_norm_f32_buffer(
    input: &CudaBuffer,
    weight: &CudaBuffer,
    rows: usize,
    dim: usize,
    eps: f32,
) -> Result<CudaBuffer, String> {
    ensure_positive_finite("CUDA RMSNorm epsilon", eps)?;
    let len = checked_len("CUDA RMSNorm length", &[rows, dim])?;
    if input.len() != len {
        return Err(format!(
            "CUDA RMSNorm input length mismatch: expected {}, got {}",
            len,
            input.len()
        ));
    }
    if weight.len() != dim {
        return Err(format!(
            "CUDA RMSNorm weight length mismatch: expected {}, got {}",
            dim,
            weight.len()
        ));
    }
    let out = alloc_f32(len)?;
    let status = unsafe {
        lumen_cuda_rms_norm_f32_device(
            input.handle(),
            weight.handle(),
            out.handle(),
            rows,
            dim,
            eps,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
pub fn rms_norm_typed(
    input: &CudaBuffer,
    input_dtype: DType,
    input_scale: Option<f32>,
    weight: &CudaBuffer,
    weight_dtype: DType,
    weight_scale: Option<f32>,
    rows: usize,
    dim: usize,
    eps: f32,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    let out = rms_norm_typed_buffer(
        input,
        input_dtype,
        input_scale,
        weight,
        weight_dtype,
        weight_scale,
        rows,
        dim,
        eps,
    )?;
    let host = download_f32(&out)?;
    Ok((out, host))
}

#[allow(clippy::too_many_arguments)]
pub fn rms_norm_typed_buffer(
    input: &CudaBuffer,
    input_dtype: DType,
    input_scale: Option<f32>,
    weight: &CudaBuffer,
    weight_dtype: DType,
    weight_scale: Option<f32>,
    rows: usize,
    dim: usize,
    eps: f32,
) -> Result<CudaBuffer, String> {
    ensure_positive_finite("CUDA typed RMSNorm epsilon", eps)?;
    let len = rows
        .checked_mul(dim)
        .ok_or_else(|| "CUDA typed RMSNorm length overflow".to_string())?;
    if input.len() != len {
        return Err(format!(
            "CUDA typed RMSNorm input length mismatch: expected {}, got {}",
            len,
            input.len()
        ));
    }
    if weight.len() != dim {
        return Err(format!(
            "CUDA typed RMSNorm weight length mismatch: expected {}, got {}",
            dim,
            weight.len()
        ));
    }
    let out = alloc_f32(len)?;
    let status = unsafe {
        lumen_cuda_rms_norm_typed_device(
            input.handle(),
            dtype_tag(input_dtype)?,
            input_scale.unwrap_or(1.0),
            weight.handle(),
            dtype_tag(weight_dtype)?,
            weight_scale.unwrap_or(1.0),
            out.handle(),
            rows,
            dim,
            eps,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
pub fn rms_norm_i8_typed_output_buffer(
    input: &CudaBuffer,
    input_dtype: DType,
    input_scale: Option<f32>,
    weight: &CudaBuffer,
    weight_dtype: DType,
    weight_scale: Option<f32>,
    rows: usize,
    dim: usize,
    eps: f32,
) -> Result<(CudaBuffer, f32), String> {
    ensure_positive_finite("CUDA I8 typed-output RMSNorm epsilon", eps)?;
    let len = rows
        .checked_mul(dim)
        .ok_or_else(|| "CUDA I8 typed-output RMSNorm length overflow".to_string())?;
    if input.len() != len {
        return Err(format!(
            "CUDA I8 typed-output RMSNorm input length mismatch: expected {}, got {}",
            len,
            input.len()
        ));
    }
    if weight.len() != dim {
        return Err(format!(
            "CUDA I8 typed-output RMSNorm weight length mismatch: expected {}, got {}",
            dim,
            weight.len()
        ));
    }
    let out = alloc_storage(len)?;
    let mut out_scale = 1.0f32;
    let status = unsafe {
        lumen_cuda_rms_norm_i8_typed_out_device(
            input.handle(),
            dtype_tag(input_dtype)?,
            input_scale.unwrap_or(1.0),
            weight.handle(),
            dtype_tag(weight_dtype)?,
            weight_scale.unwrap_or(1.0),
            out.handle(),
            rows,
            dim,
            eps,
            &mut out_scale as *mut f32,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok((out, out_scale))
}

pub fn rms_norm_backward_f32(
    input: &CudaBuffer,
    weight: &CudaBuffer,
    grad: &CudaBuffer,
    rows: usize,
    dim: usize,
    eps: f32,
) -> Result<CudaTwoHostBuffers, String> {
    let (grad_input, grad_weight) =
        rms_norm_backward_f32_buffers(input, weight, grad, rows, dim, eps)?;
    let grad_input_host = download_f32(&grad_input)?;
    let grad_weight_host = download_f32(&grad_weight)?;
    Ok((
        (grad_input, grad_input_host),
        (grad_weight, grad_weight_host),
    ))
}

pub fn rms_norm_backward_f32_buffers(
    input: &CudaBuffer,
    weight: &CudaBuffer,
    grad: &CudaBuffer,
    rows: usize,
    dim: usize,
    eps: f32,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    ensure_positive_finite("CUDA RMSNorm backward epsilon", eps)?;
    let len = rows
        .checked_mul(dim)
        .ok_or_else(|| "CUDA RMSNorm backward length overflow".to_string())?;
    if input.len() != len || grad.len() != len {
        return Err(format!(
            "CUDA RMSNorm backward input/grad length mismatch: expected {}, input={}, grad={}",
            len,
            input.len(),
            grad.len()
        ));
    }
    if weight.len() != dim {
        return Err(format!(
            "CUDA RMSNorm backward weight length mismatch: expected {}, got {}",
            dim,
            weight.len()
        ));
    }
    let grad_input = alloc_f32(len)?;
    let grad_weight = alloc_f32(dim)?;
    let status = unsafe {
        lumen_cuda_rms_norm_backward_f32_device(
            input.handle(),
            weight.handle(),
            grad.handle(),
            grad_input.handle(),
            grad_weight.handle(),
            rows,
            dim,
            eps,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok((grad_input, grad_weight))
}

#[allow(clippy::too_many_arguments)]
pub fn rms_norm_backward_typed(
    input: &CudaBuffer,
    input_dtype: DType,
    input_scale: Option<f32>,
    weight: &CudaBuffer,
    weight_dtype: DType,
    weight_scale: Option<f32>,
    grad: &CudaBuffer,
    rows: usize,
    dim: usize,
    eps: f32,
) -> Result<CudaTwoHostBuffers, String> {
    let (grad_input, grad_weight) = rms_norm_backward_typed_buffers(
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
    )?;
    let grad_input_host = download_f32(&grad_input)?;
    let grad_weight_host = download_f32(&grad_weight)?;
    Ok((
        (grad_input, grad_input_host),
        (grad_weight, grad_weight_host),
    ))
}

#[allow(clippy::too_many_arguments)]
pub fn rms_norm_backward_typed_buffers(
    input: &CudaBuffer,
    input_dtype: DType,
    input_scale: Option<f32>,
    weight: &CudaBuffer,
    weight_dtype: DType,
    weight_scale: Option<f32>,
    grad: &CudaBuffer,
    rows: usize,
    dim: usize,
    eps: f32,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    ensure_positive_finite("CUDA typed RMSNorm backward epsilon", eps)?;
    let len = rows
        .checked_mul(dim)
        .ok_or_else(|| "CUDA typed RMSNorm backward length overflow".to_string())?;
    if input.len() != len || grad.len() != len {
        return Err(format!(
            "CUDA typed RMSNorm backward input/grad length mismatch: expected {}, input={}, grad={}",
            len,
            input.len(),
            grad.len()
        ));
    }
    if weight.len() != dim {
        return Err(format!(
            "CUDA typed RMSNorm backward weight length mismatch: expected {}, got {}",
            dim,
            weight.len()
        ));
    }
    let grad_input = alloc_f32(len)?;
    let grad_weight = alloc_f32(dim)?;
    let status = unsafe {
        lumen_cuda_rms_norm_backward_typed_device(
            input.handle(),
            dtype_tag(input_dtype)?,
            input_scale.unwrap_or(1.0),
            weight.handle(),
            dtype_tag(weight_dtype)?,
            weight_scale.unwrap_or(1.0),
            grad.handle(),
            grad_input.handle(),
            grad_weight.handle(),
            rows,
            dim,
            eps,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok((grad_input, grad_weight))
}
