use super::*;

pub fn sum_f32(input: &CudaBuffer) -> Result<(CudaBuffer, Vec<f32>), String> {
    let out = alloc_f32(1)?;
    let status = unsafe { lumen_cuda_sum_f32_device(input.handle(), out.handle(), input.len()) };
    if status != 0 {
        return Err(last_error_message());
    }
    let host = download_f32(&out)?;
    Ok((out, host))
}

pub fn sum_f16_buffer(input: &CudaBuffer) -> Result<(CudaBuffer, Vec<f32>), String> {
    let out = alloc_f32(1)?;
    let status = unsafe { lumen_cuda_sum_f16_device(input.handle(), out.handle(), input.len()) };
    if status != 0 {
        return Err(last_error_message());
    }
    let host = download_f32(&out)?;
    Ok((out, host))
}

pub fn sum_bf16_buffer(input: &CudaBuffer) -> Result<(CudaBuffer, Vec<f32>), String> {
    let out = alloc_f32(1)?;
    let status = unsafe { lumen_cuda_sum_bf16_device(input.handle(), out.handle(), input.len()) };
    if status != 0 {
        return Err(last_error_message());
    }
    let host = download_f32(&out)?;
    Ok((out, host))
}

pub fn sum_i8_buffer(input: &CudaBuffer, scale: f32) -> Result<(CudaBuffer, Vec<f32>), String> {
    let out = alloc_f32(1)?;
    let status =
        unsafe { lumen_cuda_sum_i8_device(input.handle(), scale, out.handle(), input.len()) };
    if status != 0 {
        return Err(last_error_message());
    }
    let host = download_f32(&out)?;
    Ok((out, host))
}

pub fn sum_lastdim_f32_buffer(
    input: &CudaBuffer,
    rows: usize,
    last_dim: usize,
) -> Result<CudaBuffer, String> {
    let expected_len = rows
        .checked_mul(last_dim)
        .ok_or_else(|| "CUDA sum_lastdim length overflow".to_string())?;
    if rows == 0 || last_dim == 0 || input.len() != expected_len {
        return Err(format!(
            "CUDA sum_lastdim length mismatch: input={}, rows={}, last_dim={}",
            input.len(),
            rows,
            last_dim
        ));
    }
    let out = alloc_f32(last_dim)?;
    let status =
        unsafe { lumen_cuda_sum_lastdim_f32_device(input.handle(), out.handle(), rows, last_dim) };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}

pub fn mse_forward_typed(
    output: &CudaBuffer,
    output_dtype: DType,
    output_scale: Option<f32>,
    target: &CudaBuffer,
    target_dtype: DType,
    target_scale: Option<f32>,
) -> Result<(CudaBuffer, CudaBuffer, Vec<f32>), String> {
    if output.is_empty() {
        return Err("CUDA typed MSE expects at least one element".to_string());
    }
    if output.len() != target.len() {
        return Err(format!(
            "CUDA typed MSE length mismatch: output={}, target={}",
            output.len(),
            target.len()
        ));
    }
    let diff = alloc_f32(output.len())?;
    let loss = alloc_f32(1)?;
    let status = unsafe {
        lumen_cuda_mse_forward_typed_device(
            output.handle(),
            dtype_tag(output_dtype)?,
            output_scale.unwrap_or(1.0),
            target.handle(),
            dtype_tag(target_dtype)?,
            target_scale.unwrap_or(1.0),
            diff.handle(),
            loss.handle(),
            output.len(),
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    let host = download_f32(&loss)?;
    Ok((diff, loss, host))
}

pub fn mse_backward_f32(diff: &CudaBuffer, factor: f32) -> Result<CudaTwoHostBuffers, String> {
    let (grad_output, grad_target) = mse_backward_f32_buffers(diff, factor)?;
    let grad_output_host = download_f32(&grad_output)?;
    let grad_target_host = download_f32(&grad_target)?;
    Ok((
        (grad_output, grad_output_host),
        (grad_target, grad_target_host),
    ))
}

pub fn mse_backward_f32_buffers(
    diff: &CudaBuffer,
    factor: f32,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    let grad_output = alloc_f32(diff.len())?;
    let grad_target = alloc_f32(diff.len())?;
    let status = unsafe {
        lumen_cuda_mse_backward_f32_device(
            diff.handle(),
            grad_output.handle(),
            grad_target.handle(),
            diff.len(),
            factor,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok((grad_output, grad_target))
}

pub fn cross_entropy_backward_f32(
    softmax: &CudaBuffer,
    target: &CudaBuffer,
    factor: f32,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    let out = cross_entropy_backward_f32_buffer(softmax, target, factor)?;
    let host = download_f32(&out)?;
    Ok((out, host))
}

pub fn cross_entropy_backward_f32_buffer(
    softmax: &CudaBuffer,
    target: &CudaBuffer,
    factor: f32,
) -> Result<CudaBuffer, String> {
    if softmax.len() != target.len() {
        return Err(format!(
            "CUDA cross_entropy backward length mismatch: softmax={}, target={}",
            softmax.len(),
            target.len()
        ));
    }
    let out = alloc_f32(softmax.len())?;
    let status = unsafe {
        lumen_cuda_cross_entropy_backward_f32_device(
            softmax.handle(),
            target.handle(),
            out.handle(),
            softmax.len(),
            factor,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}

pub fn cross_entropy_loss_f32(
    softmax: &CudaBuffer,
    target: &CudaBuffer,
    batch_size: usize,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    if batch_size == 0 {
        return Err("CUDA cross_entropy loss batch size must be greater than zero".to_string());
    }
    if softmax.len() != target.len() {
        return Err(format!(
            "CUDA cross_entropy loss length mismatch: softmax={}, target={}",
            softmax.len(),
            target.len()
        ));
    }
    let out = alloc_f32(1)?;
    let factor = 1.0 / batch_size as f32;
    let status = unsafe {
        lumen_cuda_cross_entropy_loss_f32_device(
            softmax.handle(),
            target.handle(),
            out.handle(),
            softmax.len(),
            factor,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    let host = download_f32(&out)?;
    Ok((out, host))
}

pub fn cross_entropy_backward_typed_target_buffer(
    softmax: &CudaBuffer,
    target: &CudaBuffer,
    target_dtype: DType,
    target_scale: Option<f32>,
    factor: f32,
) -> Result<CudaBuffer, String> {
    if softmax.len() != target.len() {
        return Err(format!(
            "CUDA typed target cross_entropy backward length mismatch: softmax={}, target={}",
            softmax.len(),
            target.len()
        ));
    }
    let out = alloc_f32(softmax.len())?;
    let status = unsafe {
        lumen_cuda_cross_entropy_backward_typed_target_device(
            softmax.handle(),
            target.handle(),
            dtype_tag(target_dtype)?,
            target_scale.unwrap_or(1.0),
            out.handle(),
            softmax.len(),
            factor,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}

pub fn cross_entropy_backward_typed_target(
    softmax: &CudaBuffer,
    target: &CudaBuffer,
    target_dtype: DType,
    target_scale: Option<f32>,
    factor: f32,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    let out = cross_entropy_backward_typed_target_buffer(
        softmax,
        target,
        target_dtype,
        target_scale,
        factor,
    )?;
    let host = download_f32(&out)?;
    Ok((out, host))
}

pub fn cross_entropy_logits_typed(
    logits: &CudaBuffer,
    logits_dtype: crate::precision::DType,
    logits_scale: Option<f32>,
    target: &CudaBuffer,
    target_dtype: crate::precision::DType,
    target_scale: Option<f32>,
    batch_size: usize,
) -> Result<(CudaBuffer, CudaBuffer, f32), String> {
    if batch_size == 0
        || logits.is_empty()
        || logits.len() != target.len()
        || !logits.len().is_multiple_of(batch_size)
    {
        return Err("invalid CUDA cross entropy logits/target shape".to_string());
    }
    let probabilities = alloc_f32(logits.len())?;
    let loss = alloc_f32(1)?;
    let status = unsafe {
        lumen_cuda_cross_entropy_logits_device(
            logits.handle(),
            dtype_tag(logits_dtype)?,
            logits_scale.unwrap_or(1.0),
            target.handle(),
            dtype_tag(target_dtype)?,
            target_scale.unwrap_or(1.0),
            probabilities.handle(),
            loss.handle(),
            batch_size,
            logits.len() / batch_size,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    let host = download_f32(&loss)?;
    Ok((probabilities, loss, host[0]))
}

pub fn cross_entropy_logits_backward_typed(
    logits: &CudaBuffer,
    logits_dtype: DType,
    logits_scale: Option<f32>,
    target: &CudaBuffer,
    target_dtype: DType,
    target_scale: Option<f32>,
    probabilities: &CudaBuffer,
    batch_size: usize,
    factor: f32,
    want_logits: bool,
    want_target: bool,
) -> Result<(Option<CudaBuffer>, Option<CudaBuffer>), String> {
    if batch_size == 0
        || logits.is_empty()
        || logits.len() != target.len()
        || logits.len() != probabilities.len()
        || !logits.len().is_multiple_of(batch_size)
        || (!want_logits && !want_target)
    {
        return Err("invalid CUDA dense cross entropy backward shape or outputs".to_string());
    }
    let dx = if want_logits {
        Some(alloc_f32(logits.len())?)
    } else {
        None
    };
    let dt = if want_target {
        Some(alloc_f32(logits.len())?)
    } else {
        None
    };
    let status = unsafe {
        lumen_cuda_cross_entropy_logits_backward_device(
            logits.handle(),
            dtype_tag(logits_dtype)?,
            logits_scale.unwrap_or(1.0),
            target.handle(),
            dtype_tag(target_dtype)?,
            target_scale.unwrap_or(1.0),
            probabilities.handle(),
            dx.as_ref().map_or(0, CudaBuffer::handle),
            dt.as_ref().map_or(0, CudaBuffer::handle),
            batch_size,
            logits.len() / batch_size,
            factor,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok((dx, dt))
}

pub fn cross_entropy_loss_typed_target(
    softmax: &CudaBuffer,
    target: &CudaBuffer,
    target_dtype: DType,
    target_scale: Option<f32>,
    batch_size: usize,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    if batch_size == 0 {
        return Err(
            "CUDA typed target cross_entropy loss batch size must be greater than zero".to_string(),
        );
    }
    if softmax.len() != target.len() {
        return Err(format!(
            "CUDA typed target cross_entropy loss length mismatch: softmax={}, target={}",
            softmax.len(),
            target.len()
        ));
    }
    let out = alloc_f32(1)?;
    let factor = 1.0 / batch_size as f32;
    let status = unsafe {
        lumen_cuda_cross_entropy_loss_typed_target_device(
            softmax.handle(),
            target.handle(),
            dtype_tag(target_dtype)?,
            target_scale.unwrap_or(1.0),
            out.handle(),
            softmax.len(),
            factor,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    let host = download_f32(&out)?;
    Ok((out, host))
}
