use super::memory::validate_i8_scale_product;
use super::*;

pub fn matvec_argmax_f32(
    input: &CudaBuffer,
    weight: &CudaBuffer,
    batch_size: usize,
    vocab_size: usize,
    hidden_size: usize,
) -> Result<Vec<usize>, String> {
    if batch_size == 0 || vocab_size == 0 || hidden_size == 0 {
        return Err("CUDA matvec argmax dimensions must be greater than zero".to_string());
    }
    let input_len = batch_size
        .checked_mul(hidden_size)
        .ok_or_else(|| "CUDA matvec argmax input length overflow".to_string())?;
    if input.len() != input_len {
        return Err(format!(
            "CUDA matvec argmax input length mismatch: expected {}, got {}",
            input_len,
            input.len()
        ));
    }
    let weight_len = vocab_size
        .checked_mul(hidden_size)
        .ok_or_else(|| "CUDA matvec argmax weight length overflow".to_string())?;
    if weight.len() != weight_len {
        return Err(format!(
            "CUDA matvec argmax weight length mismatch: expected {}, got {}",
            weight_len,
            weight.len()
        ));
    }

    let mut out = vec![0usize; batch_size];
    let status = unsafe {
        lumen_cuda_matvec_argmax_f32_device(
            input.handle(),
            weight.handle(),
            out.as_mut_ptr(),
            batch_size,
            vocab_size,
            hidden_size,
        )
    };
    if status == 0 {
        Ok(out)
    } else {
        Err(last_error_message())
    }
}

pub(super) fn validate_matvec_argmax_dims(
    input: &CudaBuffer,
    weight: &CudaBuffer,
    batch_size: usize,
    vocab_size: usize,
    hidden_size: usize,
    label: &str,
) -> Result<(), String> {
    if batch_size == 0 || vocab_size == 0 || hidden_size == 0 {
        return Err(format!(
            "CUDA {label} argmax dimensions must be greater than zero"
        ));
    }
    let input_len = batch_size
        .checked_mul(hidden_size)
        .ok_or_else(|| format!("CUDA {label} argmax input length overflow"))?;
    if input.len() != input_len {
        return Err(format!(
            "CUDA {label} argmax input length mismatch: expected {}, got {}",
            input_len,
            input.len()
        ));
    }
    let weight_len = vocab_size
        .checked_mul(hidden_size)
        .ok_or_else(|| format!("CUDA {label} argmax weight length overflow"))?;
    if weight.len() != weight_len {
        return Err(format!(
            "CUDA {label} argmax weight length mismatch: expected {}, got {}",
            weight_len,
            weight.len()
        ));
    }
    Ok(())
}

pub(super) fn validate_matvec_argmax_scale(scale: f32, label: &str) -> Result<(), String> {
    if !scale.is_finite() || scale <= 0.0 {
        return Err(format!("CUDA {label} argmax scale must be finite and > 0"));
    }
    Ok(())
}

pub fn matvec_argmax_bf16_i8(
    input: &CudaBuffer,
    weight: &CudaBuffer,
    weight_scale: f32,
    batch_size: usize,
    vocab_size: usize,
    hidden_size: usize,
) -> Result<Vec<usize>, String> {
    validate_matvec_argmax_dims(
        input,
        weight,
        batch_size,
        vocab_size,
        hidden_size,
        "BF16xI8",
    )?;
    validate_matvec_argmax_scale(weight_scale, "BF16xI8 weight")?;

    let mut out = vec![0usize; batch_size];
    let status = unsafe {
        lumen_cuda_matvec_argmax_bf16_i8_device(
            input.handle(),
            weight.handle(),
            weight_scale,
            out.as_mut_ptr(),
            batch_size,
            vocab_size,
            hidden_size,
        )
    };
    if status == 0 {
        Ok(out)
    } else {
        Err(last_error_message())
    }
}

pub fn matvec_argmax_f16_i8(
    input: &CudaBuffer,
    weight: &CudaBuffer,
    weight_scale: f32,
    batch_size: usize,
    vocab_size: usize,
    hidden_size: usize,
) -> Result<Vec<usize>, String> {
    validate_matvec_argmax_dims(input, weight, batch_size, vocab_size, hidden_size, "F16xI8")?;
    validate_matvec_argmax_scale(weight_scale, "F16xI8 weight")?;

    let mut out = vec![0usize; batch_size];
    let status = unsafe {
        lumen_cuda_matvec_argmax_f16_i8_device(
            input.handle(),
            weight.handle(),
            weight_scale,
            out.as_mut_ptr(),
            batch_size,
            vocab_size,
            hidden_size,
        )
    };
    if status == 0 {
        Ok(out)
    } else {
        Err(last_error_message())
    }
}

pub fn matvec_argmax_f32_i8(
    input: &CudaBuffer,
    weight: &CudaBuffer,
    weight_scale: f32,
    batch_size: usize,
    vocab_size: usize,
    hidden_size: usize,
) -> Result<Vec<usize>, String> {
    validate_matvec_argmax_dims(input, weight, batch_size, vocab_size, hidden_size, "F32xI8")?;
    validate_matvec_argmax_scale(weight_scale, "F32xI8 weight")?;

    let mut out = vec![0usize; batch_size];
    let status = unsafe {
        lumen_cuda_matvec_argmax_f32_i8_device(
            input.handle(),
            weight.handle(),
            weight_scale,
            out.as_mut_ptr(),
            batch_size,
            vocab_size,
            hidden_size,
        )
    };
    if status == 0 {
        Ok(out)
    } else {
        Err(last_error_message())
    }
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
    validate_matvec_argmax_dims(input, weight, batch_size, vocab_size, hidden_size, "I8xI8")?;
    validate_i8_scale_product(input_scale, weight_scale, "I8xI8 argmax")?;

    let mut out = vec![0usize; batch_size];
    let status = unsafe {
        lumen_cuda_matvec_argmax_i8_i8_device(
            input.handle(),
            input_scale,
            weight.handle(),
            weight_scale,
            out.as_mut_ptr(),
            batch_size,
            vocab_size,
            hidden_size,
        )
    };
    if status == 0 {
        Ok(out)
    } else {
        Err(last_error_message())
    }
}
