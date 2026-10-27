use super::*;

pub fn sgd_update_f32(param: &CudaBuffer, grad: &CudaBuffer, lr: f32) -> Result<Vec<f32>, String> {
    imp::sgd_update_f32(param, grad, lr)
}

pub fn sgd_update_f32_no_host(
    param: &CudaBuffer,
    grad: &CudaBuffer,
    lr: f32,
) -> Result<(), String> {
    imp::sgd_update_f32_no_host(param, grad, lr)
}

pub fn sgd_update_f32_batched_no_host(
    params: &[CudaBuffer],
    grads: &[CudaBuffer],
    lr: f32,
) -> Result<(), String> {
    imp::sgd_update_f32_batched_no_host(params, grads, lr)
}

pub(crate) fn optimizer_lowp_state_update_no_host(
    params: &[CudaBuffer],
    grads: &[CudaBuffer],
    first: &[CudaBuffer],
    second: Option<&[CudaBuffer]>,
    config: LowpOptimizerConfig,
) -> Result<(), String> {
    imp::optimizer_lowp_state_update_no_host(params, grads, first, second, config)
}

pub fn sgd_momentum_update_f32(
    param: &CudaBuffer,
    grad: &CudaBuffer,
    velocity: &CudaBuffer,
    lr: f32,
    momentum: f32,
) -> Result<(Vec<f32>, Vec<f32>), String> {
    imp::sgd_momentum_update_f32(param, grad, velocity, lr, momentum)
}

pub fn sgd_momentum_update_f32_no_host(
    param: &CudaBuffer,
    grad: &CudaBuffer,
    velocity: &CudaBuffer,
    lr: f32,
    momentum: f32,
) -> Result<(), String> {
    imp::sgd_momentum_update_f32_no_host(param, grad, velocity, lr, momentum)
}

pub fn sgd_momentum_update_f32_batched_no_host(
    params: &[CudaBuffer],
    grads: &[CudaBuffer],
    velocities: &[CudaBuffer],
    lr: f32,
    momentum: f32,
) -> Result<(), String> {
    imp::sgd_momentum_update_f32_batched_no_host(params, grads, velocities, lr, momentum)
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
    imp::adam_update_f32(
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
    )
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
    imp::adam_update_f32_no_host(
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
    )
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
    imp::adam_update_f32_batched_no_host(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        lr,
        beta1,
        beta2,
        bias_correction1,
        bias_correction2,
        eps,
    )
}
