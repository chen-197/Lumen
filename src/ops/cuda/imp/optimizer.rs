use super::*;

pub fn sgd_update_f32(param: &CudaBuffer, grad: &CudaBuffer, lr: f32) -> Result<Vec<f32>, String> {
    sgd_update_f32_no_host(param, grad, lr)?;
    download_f32(param)
}

pub fn sgd_update_f32_no_host(
    param: &CudaBuffer,
    grad: &CudaBuffer,
    lr: f32,
) -> Result<(), String> {
    if param.len() != grad.len() {
        return Err(format!(
            "CUDA SGD length mismatch: param={}, grad={}",
            param.len(),
            grad.len()
        ));
    }
    let status =
        unsafe { lumen_cuda_sgd_update_f32_device(param.handle(), grad.handle(), param.len(), lr) };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(())
}

pub fn sgd_update_f32_batched_no_host(
    params: &[CudaBuffer],
    grads: &[CudaBuffer],
    lr: f32,
) -> Result<(), String> {
    if params.len() != grads.len() {
        return Err(format!(
            "CUDA batched SGD count mismatch: params={}, grads={}",
            params.len(),
            grads.len()
        ));
    }
    if params.is_empty() {
        return Err("CUDA batched SGD needs at least one tensor".to_string());
    }
    let mut param_handles = Vec::with_capacity(params.len());
    let mut grad_handles = Vec::with_capacity(params.len());
    let mut lens = Vec::with_capacity(params.len());
    for (idx, (param, grad)) in params.iter().zip(grads.iter()).enumerate() {
        if param.len() != grad.len() {
            return Err(format!(
                "CUDA batched SGD length mismatch at {idx}: param={}, grad={}",
                param.len(),
                grad.len()
            ));
        }
        param_handles.push(param.handle());
        grad_handles.push(grad.handle());
        lens.push(param.len());
    }
    let status = unsafe {
        lumen_cuda_sgd_update_f32_batched_device(
            param_handles.as_ptr(),
            grad_handles.as_ptr(),
            lens.as_ptr(),
            params.len(),
            lr,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(())
}

pub(crate) fn optimizer_lowp_state_update_no_host(
    params: &[CudaBuffer],
    grads: &[CudaBuffer],
    first: &[CudaBuffer],
    second: Option<&[CudaBuffer]>,
    config: super::LowpOptimizerConfig,
) -> Result<(), String> {
    use crate::precision::DType;
    if !matches!(config.state_dtype, DType::F16 | DType::BF16) {
        return Err("CUDA lowp optimizer state must be F16 or BF16".into());
    }
    if params.is_empty()
        || params.len() != grads.len()
        || params.len() != first.len()
        || second.is_some_and(|s| s.len() != params.len())
    {
        return Err("CUDA lowp optimizer tensor count mismatch or empty batch".into());
    }
    let mut p = Vec::with_capacity(params.len());
    let mut g = Vec::with_capacity(params.len());
    let mut m = Vec::with_capacity(params.len());
    let mut v = Vec::with_capacity(params.len());
    let mut lens = Vec::with_capacity(params.len());
    for (i, param) in params.iter().enumerate() {
        if param.is_empty()
            || param.len() != grads[i].len()
            || param.len() != first[i].len()
            || second.is_some_and(|s| param.len() != s[i].len())
        {
            return Err(format!("CUDA lowp optimizer length mismatch at {i}"));
        }
        p.push(param.handle());
        g.push(grads[i].handle());
        m.push(first[i].handle());
        if let Some(second) = second {
            v.push(second[i].handle());
        }
        lens.push(param.len());
    }
    let status = unsafe {
        lumen_cuda_optimizer_lowp_state_device(
            p.as_ptr(),
            g.as_ptr(),
            m.as_ptr(),
            if second.is_some() {
                v.as_ptr()
            } else {
                std::ptr::null()
            },
            lens.as_ptr(),
            params.len(),
            dtype_tag(config.state_dtype)?,
            config.lr,
            config.beta1,
            config.beta2,
            config.correction1,
            config.correction2,
            config.eps,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(())
}

pub fn sgd_momentum_update_f32(
    param: &CudaBuffer,
    grad: &CudaBuffer,
    velocity: &CudaBuffer,
    lr: f32,
    momentum: f32,
) -> Result<(Vec<f32>, Vec<f32>), String> {
    sgd_momentum_update_f32_no_host(param, grad, velocity, lr, momentum)?;
    Ok((download_f32(param)?, download_f32(velocity)?))
}

pub fn sgd_momentum_update_f32_no_host(
    param: &CudaBuffer,
    grad: &CudaBuffer,
    velocity: &CudaBuffer,
    lr: f32,
    momentum: f32,
) -> Result<(), String> {
    if param.len() != grad.len() || param.len() != velocity.len() {
        return Err(format!(
            "CUDA SGD momentum length mismatch: param={}, grad={}, velocity={}",
            param.len(),
            grad.len(),
            velocity.len()
        ));
    }
    let status = unsafe {
        lumen_cuda_sgd_momentum_update_f32_device(
            param.handle(),
            grad.handle(),
            velocity.handle(),
            param.len(),
            lr,
            momentum,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(())
}

pub fn sgd_momentum_update_f32_batched_no_host(
    params: &[CudaBuffer],
    grads: &[CudaBuffer],
    velocities: &[CudaBuffer],
    lr: f32,
    momentum: f32,
) -> Result<(), String> {
    if params.len() != grads.len() || params.len() != velocities.len() {
        return Err(format!(
            "CUDA batched SGD momentum count mismatch: params={}, grads={}, velocities={}",
            params.len(),
            grads.len(),
            velocities.len()
        ));
    }
    if params.is_empty() {
        return Err("CUDA batched SGD momentum needs at least one tensor".to_string());
    }
    let mut param_handles = Vec::with_capacity(params.len());
    let mut grad_handles = Vec::with_capacity(params.len());
    let mut velocity_handles = Vec::with_capacity(params.len());
    let mut lens = Vec::with_capacity(params.len());
    for (idx, ((param, grad), velocity)) in params
        .iter()
        .zip(grads.iter())
        .zip(velocities.iter())
        .enumerate()
    {
        if param.len() != grad.len() || param.len() != velocity.len() {
            return Err(format!(
                "CUDA batched SGD momentum length mismatch at {idx}: param={}, grad={}, velocity={}",
                param.len(),
                grad.len(),
                velocity.len()
            ));
        }
        param_handles.push(param.handle());
        grad_handles.push(grad.handle());
        velocity_handles.push(velocity.handle());
        lens.push(param.len());
    }
    let status = unsafe {
        lumen_cuda_sgd_momentum_update_f32_batched_device(
            param_handles.as_ptr(),
            grad_handles.as_ptr(),
            velocity_handles.as_ptr(),
            lens.as_ptr(),
            params.len(),
            lr,
            momentum,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn adam_update_f32(
    param: &CudaBuffer,
    grad: &CudaBuffer,
    exp_avg: &CudaBuffer,
    exp_avg_sq: &CudaBuffer,
    lr: f32,
    beta1: f32,
    beta2: f32,
    bias_correction1: f32,
    bias_correction2: f32,
    eps: f32,
) -> Result<CudaAdamHostState, String> {
    adam_update_f32_no_host(
        param,
        grad,
        exp_avg,
        exp_avg_sq,
        lr,
        beta1,
        beta2,
        bias_correction1,
        bias_correction2,
        eps,
    )?;
    Ok((
        download_f32(param)?,
        download_f32(exp_avg)?,
        download_f32(exp_avg_sq)?,
    ))
}

#[allow(clippy::too_many_arguments)]
pub fn adam_update_f32_no_host(
    param: &CudaBuffer,
    grad: &CudaBuffer,
    exp_avg: &CudaBuffer,
    exp_avg_sq: &CudaBuffer,
    lr: f32,
    beta1: f32,
    beta2: f32,
    bias_correction1: f32,
    bias_correction2: f32,
    eps: f32,
) -> Result<(), String> {
    if param.len() != grad.len() || param.len() != exp_avg.len() || param.len() != exp_avg_sq.len()
    {
        return Err(format!(
            "CUDA Adam length mismatch: param={}, grad={}, exp_avg={}, exp_avg_sq={}",
            param.len(),
            grad.len(),
            exp_avg.len(),
            exp_avg_sq.len()
        ));
    }
    let status = unsafe {
        lumen_cuda_adam_update_f32_device(
            param.handle(),
            grad.handle(),
            exp_avg.handle(),
            exp_avg_sq.handle(),
            param.len(),
            lr,
            beta1,
            beta2,
            bias_correction1,
            bias_correction2,
            eps,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn adam_update_f32_batched_no_host(
    params: &[CudaBuffer],
    grads: &[CudaBuffer],
    exp_avgs: &[CudaBuffer],
    exp_avg_sqs: &[CudaBuffer],
    lr: f32,
    beta1: f32,
    beta2: f32,
    bias_correction1: f32,
    bias_correction2: f32,
    eps: f32,
) -> Result<(), String> {
    if params.len() != grads.len()
        || params.len() != exp_avgs.len()
        || params.len() != exp_avg_sqs.len()
    {
        return Err(format!(
            "CUDA batched Adam count mismatch: params={}, grads={}, exp_avgs={}, exp_avg_sqs={}",
            params.len(),
            grads.len(),
            exp_avgs.len(),
            exp_avg_sqs.len()
        ));
    }
    if params.is_empty() {
        return Err("CUDA batched Adam needs at least one tensor".to_string());
    }
    let mut param_handles = Vec::with_capacity(params.len());
    let mut grad_handles = Vec::with_capacity(params.len());
    let mut exp_avg_handles = Vec::with_capacity(params.len());
    let mut exp_avg_sq_handles = Vec::with_capacity(params.len());
    let mut lens = Vec::with_capacity(params.len());
    for (idx, (((param, grad), exp_avg), exp_avg_sq)) in params
        .iter()
        .zip(grads.iter())
        .zip(exp_avgs.iter())
        .zip(exp_avg_sqs.iter())
        .enumerate()
    {
        if param.len() != grad.len()
            || param.len() != exp_avg.len()
            || param.len() != exp_avg_sq.len()
        {
            return Err(format!(
                "CUDA batched Adam length mismatch at {idx}: param={}, grad={}, exp_avg={}, exp_avg_sq={}",
                param.len(),
                grad.len(),
                exp_avg.len(),
                exp_avg_sq.len()
            ));
        }
        param_handles.push(param.handle());
        grad_handles.push(grad.handle());
        exp_avg_handles.push(exp_avg.handle());
        exp_avg_sq_handles.push(exp_avg_sq.handle());
        lens.push(param.len());
    }
    let status = unsafe {
        lumen_cuda_adam_update_f32_batched_device(
            param_handles.as_ptr(),
            grad_handles.as_ptr(),
            exp_avg_handles.as_ptr(),
            exp_avg_sq_handles.as_ptr(),
            lens.as_ptr(),
            params.len(),
            lr,
            beta1,
            beta2,
            bias_correction1,
            bias_correction2,
            eps,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(())
}
