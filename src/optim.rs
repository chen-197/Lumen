use crate::autograd::Tensor;
use crate::ops::cuda;
use crate::precision::{DType, default_runtime_dtype};
use ndarray::Zip;
use ndarray::prelude::*;

const CUDA_POINTER_BATCH_MIN_PARAMS: usize = 8;
const CUDA_POINTER_BATCH_MIN_ELEMENTS: usize = 1 << 18;

fn should_use_cuda_pointer_batch(param_count: usize, total_elements: usize) -> bool {
    param_count >= CUDA_POINTER_BATCH_MIN_PARAMS
        && total_elements >= CUDA_POINTER_BATCH_MIN_ELEMENTS
}

fn supports_cuda_param_update(param: &Tensor) -> bool {
    match param.dtype() {
        DType::F32 | DType::F16 | DType::BF16 => true,
        DType::I8 => param
            .quantization_scale()
            .map(|scale| scale.is_finite() && scale > 0.0)
            .unwrap_or(false),
    }
}

fn finalize_cuda_param_update(param: &Tensor, param_buf: crate::ops::cuda::CudaBuffer) -> bool {
    if cuda::quantize_f32_storage_no_host(&param_buf, param.dtype(), param.quantization_scale())
        .is_err()
    {
        return false;
    }
    param.replace_cuda_f32_buffer_no_host_sync(param_buf);
    true
}

pub trait Optimizer {
    fn step(&mut self);
    fn zero_grad(&self) {
        for param in self.params() {
            param.zero_grad();
        }
    }
    fn params(&self) -> &Vec<Tensor>;
}

pub struct SGD {
    params: Vec<Tensor>,
    lr: f32,
    momentum: f32,
    state_dtype: DType,
    velocities: Vec<Option<Tensor>>,
}

impl SGD {
    pub fn new(params: Vec<Tensor>, lr: f32) -> Self {
        Self::new_with_dtype(params, lr, default_runtime_dtype())
    }

    pub fn new_with_dtype(params: Vec<Tensor>, lr: f32, state_dtype: DType) -> Self {
        assert!(
            state_dtype.is_float(),
            "Optimizer state currently only supports floating dtypes, got {:?}",
            state_dtype
        );
        let len = params.len();
        SGD {
            params,
            lr,
            momentum: 0.0, // 默认无动量
            state_dtype,
            velocities: vec![None; len],
        }
    }

    #[inline]
    pub fn state_dtype(&self) -> DType {
        self.state_dtype
    }

    pub fn with_momentum(mut self, momentum: f32) -> Self {
        self.momentum = momentum;
        self
    }
}

impl Optimizer for SGD {
    fn params(&self) -> &Vec<Tensor> {
        &self.params
    }

    fn step(&mut self) {
        let cuda_updated = if self.momentum == 0.0 {
            try_cuda_sgd_step_batched(&self.params, self.lr)
        } else if self.state_dtype == DType::F32 {
            for index in 0..self.params.len() {
                if self.params[index].has_pending_grad() && self.velocities[index].is_none() {
                    self.ensure_velocity(index);
                }
            }
            try_cuda_sgd_momentum_step_batched(
                &self.params,
                &self.velocities,
                self.lr,
                self.momentum,
            )
        } else {
            vec![false; self.params.len()]
        };

        for i in 0..self.params.len() {
            if cuda_updated.get(i).copied().unwrap_or(false) {
                continue;
            }
            if self.momentum == 0.0 {
                let param = &self.params[i];
                if try_cuda_sgd_step(param, self.lr) || param.apply_sgd_update_cpu(self.lr) {
                    continue;
                }

                let Some(grad) = param.grad_arc() else {
                    continue;
                };
                let mut data = param.data_mut();
                if let (Some(data_slice), Some(grad_slice)) = (
                    data.as_slice_memory_order_mut(),
                    grad.as_slice_memory_order(),
                ) {
                    for (w, g) in data_slice.iter_mut().zip(grad_slice.iter()) {
                        *w -= self.lr * *g;
                    }
                } else {
                    let lr = self.lr;
                    Zip::from(data.view_mut())
                        .and(grad.view())
                        .for_each(|w, g| {
                            *w -= lr * *g;
                        });
                }
            } else {
                if !self.params[i].has_pending_grad() {
                    continue;
                }
                self.ensure_velocity(i);

                let m = self.momentum;
                let lr = self.lr;
                let param = &self.params[i];
                let v_buf = self.velocities[i].as_ref().unwrap();
                if self.state_dtype == DType::F32 && try_cuda_sgd_momentum_step(param, v_buf, lr, m)
                {
                    continue;
                }
                if self.state_dtype == DType::F32
                    && param.apply_sgd_momentum_update_cpu(v_buf, lr, m)
                {
                    continue;
                }
                let Some(grad) = param.grad_arc() else {
                    continue;
                };
                let mut next_v = v_buf.data();
                let mut data = param.data_mut();
                if let (Some(data_slice), Some(v_slice), Some(grad_slice)) = (
                    data.as_slice_memory_order_mut(),
                    next_v.as_slice_memory_order_mut(),
                    grad.as_slice_memory_order(),
                ) {
                    for ((w, v), g) in data_slice
                        .iter_mut()
                        .zip(v_slice.iter_mut())
                        .zip(grad_slice.iter())
                    {
                        *v = m * *v + *g;
                        *w -= lr * *v;
                    }
                } else {
                    Zip::from(next_v.view_mut())
                        .and(grad.view())
                        .for_each(|v, g| {
                            *v = m * (*v) + *g;
                        });

                    Zip::from(data.view_mut())
                        .and(next_v.view())
                        .for_each(|w, vv| {
                            *w -= lr * *vv;
                        });
                }
                v_buf.set_array_f32_with_dtype(next_v, self.state_dtype);
            }
        }
    }
}

impl SGD {
    fn ensure_velocity(&mut self, index: usize) {
        if self.velocities[index].is_some() {
            return;
        }
        let param = &self.params[index];
        let mut state = Tensor::from_array_no_grad(ArrayD::zeros(IxDyn(&param.shape_vec())));
        state.cast_inplace(self.state_dtype);
        if param.is_cuda() && self.state_dtype == DType::F32 {
            state = state.to_cuda();
        }
        self.velocities[index] = Some(state);
    }

    #[cfg(feature = "dev-tools")]
    pub fn dev_velocity_count(&self) -> usize {
        self.velocities
            .iter()
            .filter(|velocity| velocity.is_some())
            .count()
    }

    #[cfg(feature = "dev-tools")]
    pub fn dev_all_velocities_are_f32_cuda_resident(&self) -> bool {
        self.velocities.iter().all(|velocity| {
            let Some(velocity) = velocity.as_ref() else {
                return false;
            };
            velocity.dtype() == DType::F32
                && velocity.is_cuda()
                && velocity.cloned_cuda_f32_buffer().is_some()
                && !velocity.dev_has_host_f32_data()
        })
    }
}

fn try_cuda_sgd_step_batched(params: &[Tensor], lr: f32) -> Vec<bool> {
    let mut updated = vec![false; params.len()];
    let mut indices = Vec::new();
    let mut param_bufs = Vec::new();
    let mut grad_bufs = Vec::new();

    for (idx, param) in params.iter().enumerate() {
        if !param.is_cuda() || !supports_cuda_param_update(param) {
            continue;
        }
        let Some(param_buf) = param.cloned_cuda_f32_buffer() else {
            continue;
        };
        let Some(grad_buf) = param.cloned_cuda_f32_grad() else {
            continue;
        };
        indices.push(idx);
        param_bufs.push(param_buf);
        grad_bufs.push(grad_buf);
    }

    if indices.len() < 2 {
        return updated;
    }
    let total_elements = param_bufs
        .iter()
        .fold(0usize, |acc, buffer| acc.saturating_add(buffer.len()));
    if !should_use_cuda_pointer_batch(indices.len(), total_elements) {
        for ((slot, param_buf), grad_buf) in indices.iter().copied().zip(param_bufs).zip(grad_bufs)
        {
            if cuda::sgd_update_f32_no_host(&param_buf, &grad_buf, lr).is_ok() {
                updated[slot] = finalize_cuda_param_update(&params[slot], param_buf);
            }
        }
        return updated;
    }
    if cuda::sgd_update_f32_batched_no_host(&param_bufs, &grad_bufs, lr).is_err() {
        return updated;
    }

    for (slot, param_buf) in indices.iter().copied().zip(param_bufs) {
        updated[slot] = finalize_cuda_param_update(&params[slot], param_buf);
    }
    updated
}

fn try_cuda_sgd_momentum_step_batched(
    params: &[Tensor],
    velocities: &[Option<Tensor>],
    lr: f32,
    momentum: f32,
) -> Vec<bool> {
    let mut updated = vec![false; params.len()];
    let mut indices = Vec::new();
    let mut param_bufs = Vec::new();
    let mut grad_bufs = Vec::new();
    let mut velocity_bufs = Vec::new();

    for (idx, param) in params.iter().enumerate() {
        if !param.is_cuda() || !supports_cuda_param_update(param) {
            continue;
        }
        let Some(velocity) = velocities.get(idx).and_then(|v| v.as_ref()) else {
            continue;
        };
        if !velocity.is_cuda() {
            continue;
        }
        let Some(param_buf) = param.cloned_cuda_f32_buffer() else {
            continue;
        };
        let Some(grad_buf) = param.cloned_cuda_f32_grad() else {
            continue;
        };
        let Some(velocity_buf) = velocity.cloned_cuda_f32_buffer() else {
            continue;
        };
        indices.push(idx);
        param_bufs.push(param_buf);
        grad_bufs.push(grad_buf);
        velocity_bufs.push(velocity_buf);
    }

    if indices.len() < 2 {
        return updated;
    }
    let total_elements = param_bufs
        .iter()
        .fold(0usize, |acc, buffer| acc.saturating_add(buffer.len()));
    if !should_use_cuda_pointer_batch(indices.len(), total_elements) {
        for (((slot, param_buf), grad_buf), velocity_buf) in indices
            .iter()
            .copied()
            .zip(param_bufs)
            .zip(grad_bufs)
            .zip(velocity_bufs)
        {
            if cuda::sgd_momentum_update_f32_no_host(
                &param_buf,
                &grad_buf,
                &velocity_buf,
                lr,
                momentum,
            )
            .is_ok()
                && finalize_cuda_param_update(&params[slot], param_buf)
            {
                if let Some(velocity) = velocities[slot].as_ref() {
                    velocity.replace_cuda_f32_buffer_no_host_sync(velocity_buf);
                }
                updated[slot] = true;
            }
        }
        return updated;
    }
    if cuda::sgd_momentum_update_f32_batched_no_host(
        &param_bufs,
        &grad_bufs,
        &velocity_bufs,
        lr,
        momentum,
    )
    .is_err()
    {
        return updated;
    }

    for ((slot, param_buf), velocity_buf) in
        indices.iter().copied().zip(param_bufs).zip(velocity_bufs)
    {
        if finalize_cuda_param_update(&params[slot], param_buf) {
            if let Some(velocity) = velocities[slot].as_ref() {
                velocity.replace_cuda_f32_buffer_no_host_sync(velocity_buf);
            }
            updated[slot] = true;
        }
    }
    updated
}

fn try_cuda_sgd_step(param: &Tensor, lr: f32) -> bool {
    if !param.is_cuda() || !supports_cuda_param_update(param) {
        return false;
    }
    let Some(param_buf) = param.cloned_cuda_f32_buffer() else {
        return false;
    };
    let Some(grad_buf) = param.cloned_cuda_f32_grad() else {
        return false;
    };
    let Ok(()) = cuda::sgd_update_f32_no_host(&param_buf, &grad_buf, lr) else {
        return false;
    };
    finalize_cuda_param_update(param, param_buf)
}

fn try_cuda_sgd_momentum_step(param: &Tensor, velocity: &Tensor, lr: f32, momentum: f32) -> bool {
    if !param.is_cuda() || !supports_cuda_param_update(param) || !velocity.is_cuda() {
        return false;
    }
    let Some(param_buf) = param.cloned_cuda_f32_buffer() else {
        return false;
    };
    let Some(grad_buf) = param.cloned_cuda_f32_grad() else {
        return false;
    };
    let Some(velocity_buf) = velocity.cloned_cuda_f32_buffer() else {
        return false;
    };
    let Ok(()) =
        cuda::sgd_momentum_update_f32_no_host(&param_buf, &grad_buf, &velocity_buf, lr, momentum)
    else {
        return false;
    };
    if !finalize_cuda_param_update(param, param_buf) {
        return false;
    }
    velocity.replace_cuda_f32_buffer_no_host_sync(velocity_buf);
    true
}

pub struct Adam {
    params: Vec<Tensor>,
    lr: f32,
    betas: (f32, f32),
    eps: f32,

    // 状态
    step_count: usize,
    state_dtype: DType,
    exp_avg: Vec<Option<Tensor>>,    // m (一阶矩)
    exp_avg_sq: Vec<Option<Tensor>>, // v (二阶矩)
}

impl Adam {
    pub fn new(params: Vec<Tensor>, lr: f32) -> Self {
        Self::new_with_dtype(params, lr, default_runtime_dtype())
    }

    pub fn new_with_dtype(params: Vec<Tensor>, lr: f32, state_dtype: DType) -> Self {
        assert!(
            state_dtype.is_float(),
            "Optimizer state currently only supports floating dtypes, got {:?}",
            state_dtype
        );
        let len = params.len();
        Adam {
            params,
            lr,
            betas: (0.9, 0.999),
            eps: 1e-8,
            step_count: 0,
            state_dtype,
            exp_avg: vec![None; len],
            exp_avg_sq: vec![None; len],
        }
    }

    #[inline]
    pub fn state_dtype(&self) -> DType {
        self.state_dtype
    }

    fn ensure_state(&mut self, index: usize) {
        if self.exp_avg[index].is_some() {
            return;
        }
        let param = &self.params[index];
        let mut exp_avg = Tensor::from_array_no_grad(ArrayD::zeros(IxDyn(&param.shape_vec())));
        exp_avg.cast_inplace(self.state_dtype);
        let mut exp_avg_sq = Tensor::from_array_no_grad(ArrayD::zeros(IxDyn(&param.shape_vec())));
        exp_avg_sq.cast_inplace(self.state_dtype);
        if param.is_cuda() && self.state_dtype == DType::F32 {
            exp_avg = exp_avg.to_cuda();
            exp_avg_sq = exp_avg_sq.to_cuda();
        }
        self.exp_avg[index] = Some(exp_avg);
        self.exp_avg_sq[index] = Some(exp_avg_sq);
    }
}

impl Optimizer for Adam {
    fn params(&self) -> &Vec<Tensor> {
        &self.params
    }

    fn step(&mut self) {
        self.step_count += 1;
        let (beta1, beta2) = self.betas;

        // 预计算 Bias Correction
        let bias_correction1 = 1.0 - beta1.powi(self.step_count as i32);
        let bias_correction2 = 1.0 - beta2.powi(self.step_count as i32);

        for index in 0..self.params.len() {
            if self.params[index].has_pending_grad() {
                self.ensure_state(index);
            }
        }

        let cuda_updated = if self.state_dtype == DType::F32 {
            try_cuda_adam_step_batched(
                &self.params,
                &self.exp_avg,
                &self.exp_avg_sq,
                self.lr,
                beta1,
                beta2,
                bias_correction1,
                bias_correction2,
                self.eps,
            )
        } else {
            vec![false; self.params.len()]
        };

        for i in 0..self.params.len() {
            if cuda_updated.get(i).copied().unwrap_or(false) {
                continue;
            }
            if !self.params[i].has_pending_grad() {
                continue;
            }
            self.ensure_state(i);

            let lr = self.lr;
            let eps = self.eps;
            let param = &self.params[i];
            let m_buf = self.exp_avg[i].as_ref().unwrap();
            let v_buf = self.exp_avg_sq[i].as_ref().unwrap();

            if self.state_dtype == DType::F32
                && try_cuda_adam_step(
                    param,
                    m_buf,
                    v_buf,
                    lr,
                    beta1,
                    beta2,
                    bias_correction1,
                    bias_correction2,
                    eps,
                )
            {
                continue;
            }
            if self.state_dtype == DType::F32
                && param.apply_adam_update_cpu(
                    m_buf,
                    v_buf,
                    lr,
                    beta1,
                    beta2,
                    bias_correction1,
                    bias_correction2,
                    eps,
                )
            {
                continue;
            }

            let grad = match param.grad_arc() {
                Some(g) => g,
                None => continue,
            };

            let mut m_next = m_buf.data();
            let mut v_next = v_buf.data();
            let mut data = param.data_mut();

            if let (Some(data_slice), Some(m_slice), Some(v_slice), Some(grad_slice)) = (
                data.as_slice_memory_order_mut(),
                m_next.as_slice_memory_order_mut(),
                v_next.as_slice_memory_order_mut(),
                grad.as_slice_memory_order(),
            ) {
                for (((w, m), v), g) in data_slice
                    .iter_mut()
                    .zip(m_slice.iter_mut())
                    .zip(v_slice.iter_mut())
                    .zip(grad_slice.iter())
                {
                    *m = beta1 * (*m) + (1.0 - beta1) * *g;
                    *v = beta2 * (*v) + (1.0 - beta2) * *g * *g;
                    let m_hat = *m / bias_correction1;
                    let v_hat = *v / bias_correction2;
                    *w -= lr * (m_hat / (v_hat.sqrt() + eps));
                }
            } else {
                Zip::from(data.view_mut())
                    .and(m_next.view_mut())
                    .and(v_next.view_mut())
                    .and(grad.view())
                    .for_each(|w, m, v, g| {
                        *m = beta1 * (*m) + (1.0 - beta1) * g;
                        *v = beta2 * (*v) + (1.0 - beta2) * g * g;
                        let m_hat = *m / bias_correction1;
                        let v_hat = *v / bias_correction2;
                        *w -= lr * (m_hat / (v_hat.sqrt() + eps));
                    });
            }
            m_buf.set_array_f32_with_dtype(m_next, self.state_dtype);
            v_buf.set_array_f32_with_dtype(v_next, self.state_dtype);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn try_cuda_adam_step_batched(
    params: &[Tensor],
    exp_avgs: &[Option<Tensor>],
    exp_avg_sqs: &[Option<Tensor>],
    lr: f32,
    beta1: f32,
    beta2: f32,
    bias_correction1: f32,
    bias_correction2: f32,
    eps: f32,
) -> Vec<bool> {
    let mut updated = vec![false; params.len()];
    let mut indices = Vec::new();
    let mut param_bufs = Vec::new();
    let mut grad_bufs = Vec::new();
    let mut exp_avg_bufs = Vec::new();
    let mut exp_avg_sq_bufs = Vec::new();

    for (idx, param) in params.iter().enumerate() {
        if !param.is_cuda() || !supports_cuda_param_update(param) {
            continue;
        }
        let Some(exp_avg) = exp_avgs.get(idx).and_then(|value| value.as_ref()) else {
            continue;
        };
        let Some(exp_avg_sq) = exp_avg_sqs.get(idx).and_then(|value| value.as_ref()) else {
            continue;
        };
        if !exp_avg.is_cuda() || !exp_avg_sq.is_cuda() {
            continue;
        }
        let Some(param_buf) = param.cloned_cuda_f32_buffer() else {
            continue;
        };
        let Some(grad_buf) = param.cloned_cuda_f32_grad() else {
            continue;
        };
        let Some(exp_avg_buf) = exp_avg.cloned_cuda_f32_buffer() else {
            continue;
        };
        let Some(exp_avg_sq_buf) = exp_avg_sq.cloned_cuda_f32_buffer() else {
            continue;
        };
        indices.push(idx);
        param_bufs.push(param_buf);
        grad_bufs.push(grad_buf);
        exp_avg_bufs.push(exp_avg_buf);
        exp_avg_sq_bufs.push(exp_avg_sq_buf);
    }

    if indices.len() < 2 {
        return updated;
    }
    let total_elements = param_bufs
        .iter()
        .fold(0usize, |acc, buffer| acc.saturating_add(buffer.len()));
    if !should_use_cuda_pointer_batch(indices.len(), total_elements) {
        for ((((slot, param_buf), grad_buf), exp_avg_buf), exp_avg_sq_buf) in indices
            .iter()
            .copied()
            .zip(param_bufs)
            .zip(grad_bufs)
            .zip(exp_avg_bufs)
            .zip(exp_avg_sq_bufs)
        {
            if cuda::adam_update_f32_no_host(
                &param_buf,
                &grad_buf,
                &exp_avg_buf,
                &exp_avg_sq_buf,
                lr,
                beta1,
                beta2,
                bias_correction1,
                bias_correction2,
                eps,
            )
            .is_ok()
                && finalize_cuda_param_update(&params[slot], param_buf)
            {
                if let Some(exp_avg) = exp_avgs[slot].as_ref() {
                    exp_avg.replace_cuda_f32_buffer_no_host_sync(exp_avg_buf);
                }
                if let Some(exp_avg_sq) = exp_avg_sqs[slot].as_ref() {
                    exp_avg_sq.replace_cuda_f32_buffer_no_host_sync(exp_avg_sq_buf);
                }
                updated[slot] = true;
            }
        }
        return updated;
    }
    if cuda::adam_update_f32_batched_no_host(
        &param_bufs,
        &grad_bufs,
        &exp_avg_bufs,
        &exp_avg_sq_bufs,
        lr,
        beta1,
        beta2,
        bias_correction1,
        bias_correction2,
        eps,
    )
    .is_err()
    {
        return updated;
    }

    for (((slot, param_buf), exp_avg_buf), exp_avg_sq_buf) in indices
        .iter()
        .copied()
        .zip(param_bufs)
        .zip(exp_avg_bufs)
        .zip(exp_avg_sq_bufs)
    {
        if finalize_cuda_param_update(&params[slot], param_buf) {
            if let Some(exp_avg) = exp_avgs[slot].as_ref() {
                exp_avg.replace_cuda_f32_buffer_no_host_sync(exp_avg_buf);
            }
            if let Some(exp_avg_sq) = exp_avg_sqs[slot].as_ref() {
                exp_avg_sq.replace_cuda_f32_buffer_no_host_sync(exp_avg_sq_buf);
            }
            updated[slot] = true;
        }
    }
    updated
}

#[allow(clippy::too_many_arguments)]
fn try_cuda_adam_step(
    param: &Tensor,
    exp_avg: &Tensor,
    exp_avg_sq: &Tensor,
    lr: f32,
    beta1: f32,
    beta2: f32,
    bias_correction1: f32,
    bias_correction2: f32,
    eps: f32,
) -> bool {
    if !param.is_cuda()
        || !supports_cuda_param_update(param)
        || !exp_avg.is_cuda()
        || !exp_avg_sq.is_cuda()
    {
        return false;
    }
    let Some(param_buf) = param.cloned_cuda_f32_buffer() else {
        return false;
    };
    let Some(grad_buf) = param.cloned_cuda_f32_grad() else {
        return false;
    };
    let Some(exp_avg_buf) = exp_avg.cloned_cuda_f32_buffer() else {
        return false;
    };
    let Some(exp_avg_sq_buf) = exp_avg_sq.cloned_cuda_f32_buffer() else {
        return false;
    };

    let Ok(()) = cuda::adam_update_f32_no_host(
        &param_buf,
        &grad_buf,
        &exp_avg_buf,
        &exp_avg_sq_buf,
        lr,
        beta1,
        beta2,
        bias_correction1,
        bias_correction2,
        eps,
    ) else {
        return false;
    };

    if !finalize_cuda_param_update(param, param_buf) {
        return false;
    }
    exp_avg.replace_cuda_f32_buffer_no_host_sync(exp_avg_buf);
    exp_avg_sq.replace_cuda_f32_buffer_no_host_sync(exp_avg_sq_buf);
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::precision::{PrecisionConfig, set_default_runtime_dtype, with_precision_config};

    #[test]
    fn sgd_default_construction_captures_runtime_dtype_for_future_state() {
        with_precision_config(
            PrecisionConfig {
                parameter_dtype: DType::F32,
                runtime_dtype: DType::BF16,
                allow_parameter_dtype_copies: false,
            },
            || {
                let param = Tensor::parameter(ArrayD::from_elem(IxDyn(&[2]), 1.0));
                let mut opt = SGD::new(vec![param.clone()], 0.1).with_momentum(0.9);
                set_default_runtime_dtype(DType::F32);
                param.add_grad(ArrayD::from_elem(IxDyn(&[2]), 0.5));
                opt.step();

                assert_eq!(opt.state_dtype(), DType::BF16);
                assert_eq!(
                    opt.velocities[0].as_ref().expect("velocity state").dtype(),
                    DType::BF16
                );
            },
        );
    }

    #[test]
    fn sgd_explicit_state_dtype_overrides_global_default() {
        with_precision_config(
            PrecisionConfig {
                parameter_dtype: DType::F32,
                runtime_dtype: DType::BF16,
                allow_parameter_dtype_copies: false,
            },
            || {
                let param = Tensor::parameter(ArrayD::from_elem(IxDyn(&[2]), 1.0));
                let mut opt =
                    SGD::new_with_dtype(vec![param.clone()], 0.1, DType::F32).with_momentum(0.9);
                param.add_grad(ArrayD::from_elem(IxDyn(&[2]), 0.5));
                opt.step();

                assert_eq!(opt.state_dtype(), DType::F32);
                assert_eq!(
                    opt.velocities[0].as_ref().expect("velocity state").dtype(),
                    DType::F32
                );
            },
        );
    }

    #[test]
    fn cpu_sgd_step_updates_low_precision_parameters_without_f32_cache() {
        let initial = vec![1.0, -2.0, 0.5, 3.0];
        let grad_values = vec![0.5, -1.0, 2.0, 0.25];
        let expected = [0.95, -1.9, 0.3, 2.975];

        for dtype in [DType::F16, DType::BF16, DType::I8] {
            let param = Tensor::parameter_with_dtype(
                ArrayD::from_shape_vec(IxDyn(&[4]), initial.clone()).unwrap(),
                dtype,
            );
            assert_eq!(param.dtype(), dtype);
            assert!(
                !param.has_host_f32_data(),
                "{dtype:?} parameter should start without a host f32 cache"
            );

            param.add_grad(ArrayD::from_shape_vec(IxDyn(&[4]), grad_values.clone()).unwrap());
            let mut opt = SGD::new_with_dtype(vec![param.clone()], 0.1, DType::F32);
            opt.step();

            assert_eq!(param.dtype(), dtype);
            assert!(
                !param.has_host_f32_data(),
                "{dtype:?} SGD update should stay on native CPU storage"
            );

            let values = param.data_ref().iter().copied().collect::<Vec<_>>();
            for (got, expect) in values.iter().zip(expected.iter()) {
                assert!(
                    (got - expect).abs() <= 0.03,
                    "{dtype:?} CPU SGD update got {got}, expected approximately {expect}"
                );
            }
        }
    }

    #[test]
    fn cpu_sgd_momentum_step_updates_low_precision_parameters_without_f32_cache() {
        let initial = vec![1.0, -2.0, 0.5, 3.0];
        let grad_values = vec![0.5, -1.0, 2.0, 0.25];
        let expected = [0.95, -1.9, 0.3, 2.975];

        for dtype in [DType::F16, DType::BF16, DType::I8] {
            let param = Tensor::parameter_with_dtype(
                ArrayD::from_shape_vec(IxDyn(&[4]), initial.clone()).unwrap(),
                dtype,
            );
            param.add_grad(ArrayD::from_shape_vec(IxDyn(&[4]), grad_values.clone()).unwrap());

            let mut opt =
                SGD::new_with_dtype(vec![param.clone()], 0.1, DType::F32).with_momentum(0.9);
            opt.step();

            assert_eq!(param.dtype(), dtype);
            assert!(
                !param.has_host_f32_data(),
                "{dtype:?} momentum SGD update should stay on native CPU storage"
            );
            assert_eq!(
                opt.velocities[0].as_ref().expect("velocity").dtype(),
                DType::F32
            );

            let values = param.data_ref().iter().copied().collect::<Vec<_>>();
            for (got, expect) in values.iter().zip(expected.iter()) {
                assert!(
                    (got - expect).abs() <= 0.03,
                    "{dtype:?} CPU momentum SGD update got {got}, expected approximately {expect}"
                );
            }
        }
    }

    #[test]
    fn cpu_adam_step_updates_low_precision_parameters_without_f32_cache() {
        let initial = vec![1.0, -2.0, 0.5, 3.0];
        let grad_values = vec![0.5, -1.0, 2.0, 0.25];
        let expected = [0.9, -1.9, 0.4, 2.9];

        for dtype in [DType::F16, DType::BF16, DType::I8] {
            let param = Tensor::parameter_with_dtype(
                ArrayD::from_shape_vec(IxDyn(&[4]), initial.clone()).unwrap(),
                dtype,
            );
            param.add_grad(ArrayD::from_shape_vec(IxDyn(&[4]), grad_values.clone()).unwrap());

            let mut opt = Adam::new_with_dtype(vec![param.clone()], 0.1, DType::F32);
            opt.step();

            assert_eq!(param.dtype(), dtype);
            assert!(
                !param.has_host_f32_data(),
                "{dtype:?} Adam update should stay on native CPU storage"
            );
            assert_eq!(
                opt.exp_avg[0].as_ref().expect("exp_avg").dtype(),
                DType::F32
            );
            assert_eq!(
                opt.exp_avg_sq[0].as_ref().expect("exp_avg_sq").dtype(),
                DType::F32
            );

            let values = param.data_ref().iter().copied().collect::<Vec<_>>();
            for (got, expect) in values.iter().zip(expected.iter()) {
                assert!(
                    (got - expect).abs() <= 0.04,
                    "{dtype:?} CPU Adam update got {got}, expected approximately {expect}"
                );
            }
        }
    }

    #[test]
    #[ignore = "performance sanity test; run with --ignored --nocapture"]
    fn perf_low_precision_cpu_optimizer_updates_keep_f32_grad_state() {
        let len = 8192usize;
        let iters = 128usize;
        let shape = IxDyn(&[len]);
        let initial = (0..len)
            .map(|i| ((i * 17 + 5) % 257) as f32 / 257.0 - 0.5)
            .collect::<Vec<_>>();
        let grad_values = (0..len)
            .map(|i| ((i * 29 + 11) % 127) as f32 * 1e-4)
            .collect::<Vec<_>>();
        let grad = ArrayD::from_shape_vec(shape.clone(), grad_values).unwrap();

        let bench_sgd = |dtype: DType| {
            let param = Tensor::parameter_with_dtype(
                ArrayD::from_shape_vec(shape.clone(), initial.clone()).unwrap(),
                dtype,
            );
            let mut opt = SGD::new_with_dtype(vec![param.clone()], 1e-4, DType::F32);
            let start = std::time::Instant::now();
            for _ in 0..iters {
                param.zero_grad();
                param.add_grad(grad.clone());
                opt.step();
            }
            assert_eq!(param.dtype(), dtype);
            if dtype != DType::F32 {
                assert!(!param.has_host_f32_data());
            }
            start.elapsed()
        };

        let bench_sgd_momentum = |dtype: DType| {
            let param = Tensor::parameter_with_dtype(
                ArrayD::from_shape_vec(shape.clone(), initial.clone()).unwrap(),
                dtype,
            );
            let mut opt =
                SGD::new_with_dtype(vec![param.clone()], 1e-4, DType::F32).with_momentum(0.9);
            let start = std::time::Instant::now();
            for _ in 0..iters {
                param.zero_grad();
                param.add_grad(grad.clone());
                opt.step();
            }
            assert_eq!(param.dtype(), dtype);
            if dtype != DType::F32 {
                assert!(!param.has_host_f32_data());
            }
            start.elapsed()
        };

        let bench_adam = |dtype: DType| {
            let param = Tensor::parameter_with_dtype(
                ArrayD::from_shape_vec(shape.clone(), initial.clone()).unwrap(),
                dtype,
            );
            let mut opt = Adam::new_with_dtype(vec![param.clone()], 1e-4, DType::F32);
            let start = std::time::Instant::now();
            for _ in 0..iters {
                param.zero_grad();
                param.add_grad(grad.clone());
                opt.step();
            }
            assert_eq!(param.dtype(), dtype);
            if dtype != DType::F32 {
                assert!(!param.has_host_f32_data());
            }
            start.elapsed()
        };

        let sgd_f32 = bench_sgd(DType::F32);
        let sgd_bf16 = bench_sgd(DType::BF16);
        let sgd_f16 = bench_sgd(DType::F16);
        let sgd_i8 = bench_sgd(DType::I8);
        let sgd_momentum_f32 = bench_sgd_momentum(DType::F32);
        let sgd_momentum_bf16 = bench_sgd_momentum(DType::BF16);
        let sgd_momentum_f16 = bench_sgd_momentum(DType::F16);
        let sgd_momentum_i8 = bench_sgd_momentum(DType::I8);
        let adam_f32 = bench_adam(DType::F32);
        let adam_bf16 = bench_adam(DType::BF16);
        let adam_f16 = bench_adam(DType::F16);
        let adam_i8 = bench_adam(DType::I8);

        let us = |duration: std::time::Duration| duration.as_secs_f64() * 1e6 / iters as f64;
        eprintln!(
            "perf_optimizer len={len} iters={iters} sgd_us: f32={:.3} bf16={:.3} f16={:.3} i8={:.3}; sgd_momentum_us: f32={:.3} bf16={:.3} f16={:.3} i8={:.3}; adam_us: f32={:.3} bf16={:.3} f16={:.3} i8={:.3}",
            us(sgd_f32),
            us(sgd_bf16),
            us(sgd_f16),
            us(sgd_i8),
            us(sgd_momentum_f32),
            us(sgd_momentum_bf16),
            us(sgd_momentum_f16),
            us(sgd_momentum_i8),
            us(adam_f32),
            us(adam_bf16),
            us(adam_f16),
            us(adam_i8),
        );
    }

    #[test]
    fn adam_default_construction_captures_runtime_dtype_for_future_state() {
        with_precision_config(
            PrecisionConfig {
                parameter_dtype: DType::F32,
                runtime_dtype: DType::BF16,
                allow_parameter_dtype_copies: false,
            },
            || {
                let param = Tensor::parameter(ArrayD::from_elem(IxDyn(&[2]), 1.0));
                let mut opt = Adam::new(vec![param.clone()], 0.1);
                set_default_runtime_dtype(DType::F32);
                param.add_grad(ArrayD::from_elem(IxDyn(&[2]), 0.25));
                opt.step();

                assert_eq!(opt.state_dtype(), DType::BF16);
                assert_eq!(
                    opt.exp_avg[0].as_ref().expect("exp_avg").dtype(),
                    DType::BF16
                );
                assert_eq!(
                    opt.exp_avg_sq[0].as_ref().expect("exp_avg_sq").dtype(),
                    DType::BF16
                );
            },
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn sgd_step_uses_cuda_grad_and_keeps_parameter_resident() {
        if !crate::ops::cuda::is_available() {
            return;
        }

        crate::ops::cuda::set_enabled(true);
        let param = Tensor::parameter(
            ArrayD::from_shape_vec(IxDyn(&[4]), vec![1.0, -2.0, 0.5, 3.0])
                .expect("parameter shape"),
        )
        .to_cuda();
        let grad =
            ArrayD::from_shape_vec(IxDyn(&[4]), vec![0.5, -1.0, 2.0, 0.25]).expect("grad shape");
        let grad_buffer =
            crate::ops::cuda::upload_f32(grad.as_slice().expect("contiguous grad")).unwrap();
        param.add_grad_with_cuda_buffer(grad, Some(grad_buffer));

        let mut opt = SGD::new(vec![param.clone()], 0.1);
        opt.step();
        crate::ops::cuda::set_enabled(false);

        assert!(param.is_cuda());
        assert!(param.cloned_cuda_f32_buffer().is_some());
        let values = param.data_ref().iter().copied().collect::<Vec<_>>();
        let expected = [0.95, -1.9, 0.3, 2.975];
        for (got, expect) in values.iter().zip(expected.iter()) {
            assert!(
                (got - expect).abs() <= 1e-6,
                "SGD CUDA update got {got}, expected {expect}"
            );
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn sgd_step_accepts_bf16_parameter_on_cuda_fast_path() {
        if !crate::ops::cuda::is_available() {
            return;
        }

        let initial = vec![1.0, -2.0, 0.5, 3.0];
        let grad_values = vec![0.5, -1.0, 2.0, 0.25];

        let cpu_param = Tensor::parameter_with_dtype(
            ArrayD::from_shape_vec(IxDyn(&[4]), initial.clone()).unwrap(),
            DType::BF16,
        );
        cpu_param.add_grad(ArrayD::from_shape_vec(IxDyn(&[4]), grad_values.clone()).unwrap());
        let mut cpu_opt = SGD::new(vec![cpu_param.clone()], 0.1);
        cpu_opt.step();
        let cpu_values = cpu_param.data_ref().iter().copied().collect::<Vec<_>>();

        crate::ops::cuda::set_enabled(true);
        let cuda_param = Tensor::parameter_with_dtype(
            ArrayD::from_shape_vec(IxDyn(&[4]), initial).unwrap(),
            DType::BF16,
        )
        .to_cuda();
        let grad = ArrayD::from_shape_vec(IxDyn(&[4]), grad_values).unwrap();
        let grad_buffer =
            crate::ops::cuda::upload_f32(grad.as_slice().expect("contiguous grad")).unwrap();
        cuda_param.add_grad_with_cuda_buffer(grad, Some(grad_buffer));

        let mut cuda_opt = SGD::new(vec![cuda_param.clone()], 0.1);
        cuda_opt.step();
        crate::ops::cuda::set_enabled(false);

        assert!(cuda_param.is_cuda());
        assert!(cuda_param.cloned_cuda_f32_buffer().is_some());
        assert_eq!(cuda_param.dtype(), DType::BF16);

        let cuda_values = cuda_param.data_ref().iter().copied().collect::<Vec<_>>();
        for (got, expect) in cuda_values.iter().zip(cpu_values.iter()) {
            assert!(
                (got - expect).abs() <= 1e-6,
                "BF16 SGD CUDA update got {got}, expected {expect}"
            );
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn sgd_step_preserves_f16_and_i8_parameters_on_cuda_fast_path() {
        if !crate::ops::cuda::is_available() {
            return;
        }

        let initial = vec![1.0, -2.0, 0.5, 3.0];
        let grad_values = vec![0.5, -1.0, 2.0, 0.25];

        for dtype in [DType::F16, DType::I8] {
            let cpu_param = Tensor::parameter_with_dtype(
                ArrayD::from_shape_vec(IxDyn(&[4]), initial.clone()).unwrap(),
                dtype,
            );
            cpu_param.add_grad(ArrayD::from_shape_vec(IxDyn(&[4]), grad_values.clone()).unwrap());
            let mut cpu_opt = SGD::new_with_dtype(vec![cpu_param.clone()], 0.1, DType::F32);
            cpu_opt.step();
            let cpu_values = cpu_param.data_ref().iter().copied().collect::<Vec<_>>();

            crate::ops::cuda::set_enabled(true);
            let cuda_param = Tensor::parameter_with_dtype(
                ArrayD::from_shape_vec(IxDyn(&[4]), initial.clone()).unwrap(),
                dtype,
            )
            .to_cuda();
            let grad = ArrayD::from_shape_vec(IxDyn(&[4]), grad_values.clone()).unwrap();
            let grad_buffer =
                crate::ops::cuda::upload_f32(grad.as_slice().expect("contiguous grad")).unwrap();
            cuda_param.add_grad_with_cuda_buffer(grad, Some(grad_buffer));

            let mut cuda_opt = SGD::new_with_dtype(vec![cuda_param.clone()], 0.1, DType::F32);
            cuda_opt.step();
            crate::ops::cuda::set_enabled(false);

            assert!(cuda_param.is_cuda());
            assert!(cuda_param.cloned_cuda_f32_buffer().is_some());
            assert_eq!(cuda_param.dtype(), dtype);

            let cuda_values = cuda_param.data_ref().iter().copied().collect::<Vec<_>>();
            let tolerance = if dtype == DType::F16 { 1e-3 } else { 0.04 };
            for (got, expect) in cuda_values.iter().zip(cpu_values.iter()) {
                assert!(
                    (got - expect).abs() <= tolerance,
                    "{dtype:?} SGD CUDA update got {got}, expected {expect}"
                );
            }
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_sgd_step_clears_consumed_grad() {
        if !crate::ops::cuda::is_available() {
            return;
        }

        crate::ops::cuda::set_enabled(true);
        let param = Tensor::parameter(
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 2.0]).expect("parameter shape"),
        )
        .to_cuda();
        let grad = ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.5, -1.0]).expect("grad shape");
        let grad_buffer =
            crate::ops::cuda::upload_f32(grad.as_slice().expect("contiguous grad")).unwrap();
        param.add_grad_with_cuda_buffer(grad, Some(grad_buffer));

        let mut opt = SGD::new(vec![param.clone()], 0.1);
        opt.step();
        let after_first = param.data_ref().iter().copied().collect::<Vec<_>>();
        opt.step();
        let after_second = param.data_ref().iter().copied().collect::<Vec<_>>();
        crate::ops::cuda::set_enabled(false);

        assert_eq!(after_first.len(), after_second.len());
        for (got, expect) in after_second.iter().zip(after_first.iter()) {
            assert!(
                (got - expect).abs() <= 1e-6,
                "CUDA SGD reused a consumed grad: got {got}, expected {expect}"
            );
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn adam_step_uses_cuda_grad_and_keeps_state_resident() {
        if !crate::ops::cuda::is_available() {
            return;
        }

        let initial = vec![1.0, -2.0, 0.5, 3.0];
        let grad_values = vec![0.25, -0.5, 1.0, -0.125];

        let cpu_param =
            Tensor::parameter(ArrayD::from_shape_vec(IxDyn(&[4]), initial.clone()).unwrap());
        cpu_param.add_grad(ArrayD::from_shape_vec(IxDyn(&[4]), grad_values.clone()).unwrap());
        let mut cpu_opt = Adam::new_with_dtype(vec![cpu_param.clone()], 0.1, DType::F32);
        cpu_opt.step();
        let cpu_values = cpu_param.data_ref().iter().copied().collect::<Vec<_>>();

        crate::ops::cuda::set_enabled(true);
        let cuda_param =
            Tensor::parameter(ArrayD::from_shape_vec(IxDyn(&[4]), initial).unwrap()).to_cuda();
        let grad = ArrayD::from_shape_vec(IxDyn(&[4]), grad_values).unwrap();
        let grad_buffer =
            crate::ops::cuda::upload_f32(grad.as_slice().expect("contiguous grad")).unwrap();
        cuda_param.add_grad_with_cuda_buffer(grad, Some(grad_buffer));

        let mut cuda_opt = Adam::new_with_dtype(vec![cuda_param.clone()], 0.1, DType::F32);
        cuda_opt.step();
        crate::ops::cuda::set_enabled(false);

        assert!(cuda_param.is_cuda());
        assert!(cuda_param.cloned_cuda_f32_buffer().is_some());
        assert!(
            cuda_opt.exp_avg[0]
                .as_ref()
                .expect("exp_avg")
                .cloned_cuda_f32_buffer()
                .is_some()
        );
        assert!(
            cuda_opt.exp_avg_sq[0]
                .as_ref()
                .expect("exp_avg_sq")
                .cloned_cuda_f32_buffer()
                .is_some()
        );

        let cuda_values = cuda_param.data_ref().iter().copied().collect::<Vec<_>>();
        for (got, expect) in cuda_values.iter().zip(cpu_values.iter()) {
            assert!(
                (got - expect).abs() <= 1e-6,
                "Adam CUDA update got {got}, expected {expect}"
            );
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn adam_step_accepts_bf16_parameter_with_f32_state_on_cuda_fast_path() {
        if !crate::ops::cuda::is_available() {
            return;
        }

        let initial = vec![1.0, -2.0, 0.5, 3.0];
        let grad_values = vec![0.25, -0.5, 1.0, -0.125];

        let cpu_param = Tensor::parameter_with_dtype(
            ArrayD::from_shape_vec(IxDyn(&[4]), initial.clone()).unwrap(),
            DType::BF16,
        );
        cpu_param.add_grad(ArrayD::from_shape_vec(IxDyn(&[4]), grad_values.clone()).unwrap());
        let mut cpu_opt = Adam::new_with_dtype(vec![cpu_param.clone()], 0.1, DType::F32);
        cpu_opt.step();
        let cpu_values = cpu_param.data_ref().iter().copied().collect::<Vec<_>>();
        let cpu_exp_avg = cpu_opt.exp_avg[0]
            .as_ref()
            .expect("cpu exp_avg")
            .data_ref()
            .iter()
            .copied()
            .collect::<Vec<_>>();
        let cpu_exp_avg_sq = cpu_opt.exp_avg_sq[0]
            .as_ref()
            .expect("cpu exp_avg_sq")
            .data_ref()
            .iter()
            .copied()
            .collect::<Vec<_>>();

        crate::ops::cuda::set_enabled(true);
        let cuda_param = Tensor::parameter_with_dtype(
            ArrayD::from_shape_vec(IxDyn(&[4]), initial).unwrap(),
            DType::BF16,
        )
        .to_cuda();
        let grad = ArrayD::from_shape_vec(IxDyn(&[4]), grad_values).unwrap();
        let grad_buffer =
            crate::ops::cuda::upload_f32(grad.as_slice().expect("contiguous grad")).unwrap();
        cuda_param.add_grad_with_cuda_buffer(grad, Some(grad_buffer));

        let mut cuda_opt = Adam::new_with_dtype(vec![cuda_param.clone()], 0.1, DType::F32);
        cuda_opt.step();
        crate::ops::cuda::set_enabled(false);

        assert!(cuda_param.is_cuda());
        assert!(cuda_param.cloned_cuda_f32_buffer().is_some());
        assert_eq!(cuda_param.dtype(), DType::BF16);
        let cuda_exp_avg_tensor = cuda_opt.exp_avg[0].as_ref().expect("cuda exp_avg");
        let cuda_exp_avg_sq_tensor = cuda_opt.exp_avg_sq[0].as_ref().expect("cuda exp_avg_sq");
        assert!(cuda_exp_avg_tensor.is_cuda());
        assert!(cuda_exp_avg_sq_tensor.is_cuda());
        assert!(cuda_exp_avg_tensor.cloned_cuda_f32_buffer().is_some());
        assert!(cuda_exp_avg_sq_tensor.cloned_cuda_f32_buffer().is_some());

        let cuda_values = cuda_param.data_ref().iter().copied().collect::<Vec<_>>();
        let cuda_exp_avg = cuda_exp_avg_tensor
            .data_ref()
            .iter()
            .copied()
            .collect::<Vec<_>>();
        let cuda_exp_avg_sq = cuda_exp_avg_sq_tensor
            .data_ref()
            .iter()
            .copied()
            .collect::<Vec<_>>();
        for (got, expect) in cuda_values.iter().zip(cpu_values.iter()) {
            assert!(
                (got - expect).abs() <= 1e-6,
                "BF16 Adam CUDA update got {got}, expected {expect}"
            );
        }
        for (got, expect) in cuda_exp_avg.iter().zip(cpu_exp_avg.iter()) {
            assert!(
                (got - expect).abs() <= 1e-6,
                "BF16 Adam CUDA exp_avg got {got}, expected {expect}"
            );
        }
        for (got, expect) in cuda_exp_avg_sq.iter().zip(cpu_exp_avg_sq.iter()) {
            assert!(
                (got - expect).abs() <= 1e-6,
                "BF16 Adam CUDA exp_avg_sq got {got}, expected {expect}"
            );
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn sgd_momentum_step_uses_cuda_grad_and_keeps_velocity_resident() {
        if !crate::ops::cuda::is_available() {
            return;
        }

        let initial = vec![1.0, -2.0, 0.5, 3.0];
        let grad_values = vec![0.5, -1.0, 2.0, 0.25];

        let cpu_param =
            Tensor::parameter(ArrayD::from_shape_vec(IxDyn(&[4]), initial.clone()).unwrap());
        cpu_param.add_grad(ArrayD::from_shape_vec(IxDyn(&[4]), grad_values.clone()).unwrap());
        let mut cpu_opt =
            SGD::new_with_dtype(vec![cpu_param.clone()], 0.1, DType::F32).with_momentum(0.9);
        cpu_opt.step();
        let cpu_values = cpu_param.data_ref().iter().copied().collect::<Vec<_>>();
        let cpu_velocity = cpu_opt.velocities[0]
            .as_ref()
            .expect("cpu velocity")
            .data_ref()
            .iter()
            .copied()
            .collect::<Vec<_>>();

        crate::ops::cuda::set_enabled(true);
        let cuda_param =
            Tensor::parameter(ArrayD::from_shape_vec(IxDyn(&[4]), initial).unwrap()).to_cuda();
        let grad = ArrayD::from_shape_vec(IxDyn(&[4]), grad_values).unwrap();
        let grad_buffer =
            crate::ops::cuda::upload_f32(grad.as_slice().expect("contiguous grad")).unwrap();
        cuda_param.add_grad_with_cuda_buffer(grad, Some(grad_buffer));

        let mut cuda_opt =
            SGD::new_with_dtype(vec![cuda_param.clone()], 0.1, DType::F32).with_momentum(0.9);
        cuda_opt.step();
        crate::ops::cuda::set_enabled(false);

        assert!(cuda_param.is_cuda());
        assert!(cuda_param.cloned_cuda_f32_buffer().is_some());
        let cuda_velocity_tensor = cuda_opt.velocities[0].as_ref().expect("cuda velocity");
        assert!(cuda_velocity_tensor.is_cuda());
        assert!(cuda_velocity_tensor.cloned_cuda_f32_buffer().is_some());

        let cuda_values = cuda_param.data_ref().iter().copied().collect::<Vec<_>>();
        let cuda_velocity = cuda_velocity_tensor
            .data_ref()
            .iter()
            .copied()
            .collect::<Vec<_>>();
        for (got, expect) in cuda_values.iter().zip(cpu_values.iter()) {
            assert!(
                (got - expect).abs() <= 1e-6,
                "SGD momentum CUDA update got {got}, expected {expect}"
            );
        }
        for (got, expect) in cuda_velocity.iter().zip(cpu_velocity.iter()) {
            assert!(
                (got - expect).abs() <= 1e-6,
                "SGD momentum CUDA velocity got {got}, expected {expect}"
            );
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn sgd_momentum_step_accepts_bf16_parameter_on_cuda_fast_path() {
        if !crate::ops::cuda::is_available() {
            return;
        }

        let initial = vec![1.0, -2.0, 0.5, 3.0];
        let grad_values = vec![0.5, -1.0, 2.0, 0.25];

        let cpu_param = Tensor::parameter_with_dtype(
            ArrayD::from_shape_vec(IxDyn(&[4]), initial.clone()).unwrap(),
            DType::BF16,
        );
        cpu_param.add_grad(ArrayD::from_shape_vec(IxDyn(&[4]), grad_values.clone()).unwrap());
        let mut cpu_opt =
            SGD::new_with_dtype(vec![cpu_param.clone()], 0.1, DType::F32).with_momentum(0.9);
        cpu_opt.step();
        let cpu_values = cpu_param.data_ref().iter().copied().collect::<Vec<_>>();
        let cpu_velocity = cpu_opt.velocities[0]
            .as_ref()
            .expect("cpu velocity")
            .data_ref()
            .iter()
            .copied()
            .collect::<Vec<_>>();

        crate::ops::cuda::set_enabled(true);
        let cuda_param = Tensor::parameter_with_dtype(
            ArrayD::from_shape_vec(IxDyn(&[4]), initial).unwrap(),
            DType::BF16,
        )
        .to_cuda();
        let grad = ArrayD::from_shape_vec(IxDyn(&[4]), grad_values).unwrap();
        let grad_buffer =
            crate::ops::cuda::upload_f32(grad.as_slice().expect("contiguous grad")).unwrap();
        cuda_param.add_grad_with_cuda_buffer(grad, Some(grad_buffer));

        let mut cuda_opt =
            SGD::new_with_dtype(vec![cuda_param.clone()], 0.1, DType::F32).with_momentum(0.9);
        cuda_opt.step();
        crate::ops::cuda::set_enabled(false);

        assert!(cuda_param.is_cuda());
        assert!(cuda_param.cloned_cuda_f32_buffer().is_some());
        assert_eq!(cuda_param.dtype(), DType::BF16);
        let cuda_velocity_tensor = cuda_opt.velocities[0].as_ref().expect("cuda velocity");
        assert!(cuda_velocity_tensor.is_cuda());
        assert!(cuda_velocity_tensor.cloned_cuda_f32_buffer().is_some());

        let cuda_values = cuda_param.data_ref().iter().copied().collect::<Vec<_>>();
        let cuda_velocity = cuda_velocity_tensor
            .data_ref()
            .iter()
            .copied()
            .collect::<Vec<_>>();
        for (got, expect) in cuda_values.iter().zip(cpu_values.iter()) {
            assert!(
                (got - expect).abs() <= 1e-6,
                "BF16 SGD momentum CUDA update got {got}, expected {expect}"
            );
        }
        for (got, expect) in cuda_velocity.iter().zip(cpu_velocity.iter()) {
            assert!(
                (got - expect).abs() <= 1e-6,
                "BF16 SGD momentum CUDA velocity got {got}, expected {expect}"
            );
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn sgd_step_batches_multiple_cuda_parameters() {
        if !crate::ops::cuda::is_available() {
            return;
        }

        let p0_initial = vec![1.0, -2.0, 0.5, 3.0];
        let p1_initial = vec![0.25, -0.75, 1.5];
        let g0_values = vec![0.5, -1.0, 2.0, 0.25];
        let g1_values = vec![-0.25, 0.5, 1.25];

        let p0_cpu =
            Tensor::parameter(ArrayD::from_shape_vec(IxDyn(&[4]), p0_initial.clone()).unwrap());
        let p1_cpu =
            Tensor::parameter(ArrayD::from_shape_vec(IxDyn(&[3]), p1_initial.clone()).unwrap());
        p0_cpu.add_grad(ArrayD::from_shape_vec(IxDyn(&[4]), g0_values.clone()).unwrap());
        p1_cpu.add_grad(ArrayD::from_shape_vec(IxDyn(&[3]), g1_values.clone()).unwrap());
        let mut cpu_opt = SGD::new(vec![p0_cpu.clone(), p1_cpu.clone()], 0.1);
        cpu_opt.step();

        crate::ops::cuda::set_enabled(true);
        let p0_cuda =
            Tensor::parameter(ArrayD::from_shape_vec(IxDyn(&[4]), p0_initial).unwrap()).to_cuda();
        let p1_cuda =
            Tensor::parameter(ArrayD::from_shape_vec(IxDyn(&[3]), p1_initial).unwrap()).to_cuda();
        let g0 = ArrayD::from_shape_vec(IxDyn(&[4]), g0_values).unwrap();
        let g1 = ArrayD::from_shape_vec(IxDyn(&[3]), g1_values).unwrap();
        let g0_buf = crate::ops::cuda::upload_f32(g0.as_slice().unwrap()).unwrap();
        let g1_buf = crate::ops::cuda::upload_f32(g1.as_slice().unwrap()).unwrap();
        p0_cuda.add_grad_with_cuda_buffer(g0, Some(g0_buf));
        p1_cuda.add_grad_with_cuda_buffer(g1, Some(g1_buf));
        let mut cuda_opt = SGD::new(vec![p0_cuda.clone(), p1_cuda.clone()], 0.1);
        cuda_opt.step();
        crate::ops::cuda::set_enabled(false);

        for (got, expect) in p0_cuda.data_ref().iter().zip(p0_cpu.data_ref().iter()) {
            assert!((got - expect).abs() <= 1e-6);
        }
        for (got, expect) in p1_cuda.data_ref().iter().zip(p1_cpu.data_ref().iter()) {
            assert!((got - expect).abs() <= 1e-6);
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn adam_step_batches_multiple_cuda_parameters_with_f32_state() {
        if !crate::ops::cuda::is_available() {
            return;
        }

        let p0_initial = vec![1.0, -2.0, 0.5, 3.0];
        let p1_initial = vec![0.25, -0.75, 1.5];
        let g0_values = vec![0.25, -0.5, 1.0, -0.125];
        let g1_values = vec![-0.25, 0.5, 1.25];

        let p0_cpu =
            Tensor::parameter(ArrayD::from_shape_vec(IxDyn(&[4]), p0_initial.clone()).unwrap());
        let p1_cpu =
            Tensor::parameter(ArrayD::from_shape_vec(IxDyn(&[3]), p1_initial.clone()).unwrap());
        p0_cpu.add_grad(ArrayD::from_shape_vec(IxDyn(&[4]), g0_values.clone()).unwrap());
        p1_cpu.add_grad(ArrayD::from_shape_vec(IxDyn(&[3]), g1_values.clone()).unwrap());
        let mut cpu_opt =
            Adam::new_with_dtype(vec![p0_cpu.clone(), p1_cpu.clone()], 0.1, DType::F32);
        cpu_opt.step();

        crate::ops::cuda::set_enabled(true);
        let p0_cuda =
            Tensor::parameter(ArrayD::from_shape_vec(IxDyn(&[4]), p0_initial).unwrap()).to_cuda();
        let p1_cuda =
            Tensor::parameter(ArrayD::from_shape_vec(IxDyn(&[3]), p1_initial).unwrap()).to_cuda();
        let g0 = ArrayD::from_shape_vec(IxDyn(&[4]), g0_values).unwrap();
        let g1 = ArrayD::from_shape_vec(IxDyn(&[3]), g1_values).unwrap();
        let g0_buf = crate::ops::cuda::upload_f32(g0.as_slice().unwrap()).unwrap();
        let g1_buf = crate::ops::cuda::upload_f32(g1.as_slice().unwrap()).unwrap();
        p0_cuda.add_grad_with_cuda_buffer(g0, Some(g0_buf));
        p1_cuda.add_grad_with_cuda_buffer(g1, Some(g1_buf));
        let mut cuda_opt =
            Adam::new_with_dtype(vec![p0_cuda.clone(), p1_cuda.clone()], 0.1, DType::F32);
        cuda_opt.step();
        crate::ops::cuda::set_enabled(false);

        for (got, expect) in p0_cuda.data_ref().iter().zip(p0_cpu.data_ref().iter()) {
            assert!((got - expect).abs() <= 1e-6);
        }
        for (got, expect) in p1_cuda.data_ref().iter().zip(p1_cpu.data_ref().iter()) {
            assert!((got - expect).abs() <= 1e-6);
        }
        assert!(
            cuda_opt.exp_avg[0]
                .as_ref()
                .expect("first exp_avg")
                .cloned_cuda_f32_buffer()
                .is_some()
        );
        assert!(
            cuda_opt.exp_avg[1]
                .as_ref()
                .expect("second exp_avg")
                .cloned_cuda_f32_buffer()
                .is_some()
        );
    }

    #[test]
    #[should_panic(expected = "Optimizer state currently only supports floating dtypes")]
    fn optimizer_state_rejects_integer_dtype() {
        let _ = SGD::new_with_dtype(Vec::new(), 0.1, DType::I8);
    }
}
