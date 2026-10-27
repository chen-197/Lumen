use crate::autograd::{StoragePreference, Tensor, TensorStorageView};
use crate::ops::cuda;
use crate::precision::{DType, ParameterQuantization, default_runtime_dtype};
use ndarray::Zip;
use ndarray::prelude::*;

mod master;
use master::MasterWeights;
mod lowp_cpu;
mod lowp_cuda;

#[cfg(test)]
#[path = "../../tests/unit/optim/conditional.rs"]
mod conditional_tests;
#[cfg(test)]
#[path = "../../tests/unit/optim/layout.rs"]
mod layout_tests;
#[cfg(test)]
#[path = "../../tests/unit/optim/lowp_cpu.rs"]
mod lowp_cpu_tests;

const CUDA_POINTER_BATCH_MIN_PARAMS: usize = 8;
const CUDA_POINTER_BATCH_MIN_ELEMENTS: usize = 1 << 18;

fn with_parameter_update_preserving_dtype(
    param: &Tensor,
    update: impl FnOnce(ArrayViewMutD<'_, f32>),
) {
    let dtype = param.dtype();
    if dtype == DType::F32 {
        update(param.data_mut().view_mut());
        return;
    }
    let scale = param.quantization_scale();
    let grad = param.0.borrow().grad.clone();
    let mut values =
        param.with_storage_view_preferring(StoragePreference::Native, |view| match view {
            TensorStorageView::F32(data) => data.to_owned(),
            TensorStorageView::F16(data) => data.mapv(|x| x.to_f32()),
            TensorStorageView::BF16(data) => data.mapv(|x| x.to_f32()),
        });
    update(values.view_mut());
    if let Some(scale) = scale {
        param
            .set_array_f32_with_quantization(values, ParameterQuantization::Int8.with_scale(scale));
    } else {
        param.set_array_f32_with_dtype(values, dtype);
    }
    // CPU steps retain pending gradients until the caller clears them.
    param.0.borrow_mut().grad = grad;
}

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
    /// Apply an update. Malformed gradient shapes panic before any parameter or
    /// optimizer state changes, including gradients replaced through `grad_mut()`.
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
    master_weights: MasterWeights,
    has_stepped: bool,
    last_cuda_batched_update_count: usize,
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
            master_weights: MasterWeights::default(),
            has_stepped: false,
            last_cuda_batched_update_count: 0,
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

    /// Preserve small updates in lazy F32 master weights for F16/BF16/I8 parameters.
    /// Requires F32 state and adds four bytes per low-precision parameter element.
    /// Call before training; rebuild the optimizer after replacing parameter values,
    /// shapes, dtypes, scales or devices. Forward parameters retain their native dtype.
    /// Without this option, updates are rounded to parameter precision each step.
    pub fn with_fp32_master_weights(mut self) -> Self {
        assert_eq!(
            self.state_dtype,
            DType::F32,
            "master weights require F32 optimizer state"
        );
        assert!(!self.has_stepped, "enable master weights before training");
        self.master_weights.enable(self.params.len());
        self
    }

    /// Keep F32 masters and recalibrate I8 absmax scales after every update.
    /// Use for trainable I8 tensors whose range may grow beyond initialization.
    /// This explicitly replaces fixed I8 scales; CUDA reads back only the scale scalar.
    pub fn with_dynamic_i8_master_weights(self) -> Self {
        let mut opt = self.with_fp32_master_weights();
        opt.master_weights.dynamic_i8 = true;
        opt
    }
}

impl Optimizer for SGD {
    fn params(&self) -> &Vec<Tensor> {
        &self.params
    }

    fn step(&mut self) {
        for (index, param) in self.params.iter().enumerate() {
            param.validate_optimizer_gradient(index);
        }
        self.has_stepped = true;
        if self.momentum != 0.0 {
            for index in 0..self.params.len() {
                if self.params[index].has_pending_grad() && self.velocities[index].is_none() {
                    self.ensure_velocity(index);
                }
            }
        }
        let prepared = self.master_weights.prepare(&self.params);
        let params = prepared.as_deref().unwrap_or(&self.params);
        let (cuda_updated, cuda_batched_update_count) = if self.momentum == 0.0 {
            try_cuda_sgd_step_batched(params, self.lr)
        } else if self.state_dtype == DType::F32 {
            try_cuda_sgd_momentum_step_batched(params, &self.velocities, self.lr, self.momentum)
        } else {
            lowp_cuda::momentum_step(
                params,
                &self.velocities,
                self.lr,
                self.momentum,
                self.state_dtype,
            )
        };
        self.last_cuda_batched_update_count = cuda_batched_update_count;

        for (i, param) in params.iter().enumerate() {
            if cuda_updated.get(i).copied().unwrap_or(false) {
                continue;
            }
            if self.momentum == 0.0 {
                if try_cuda_sgd_step(param, self.lr) || param.apply_sgd_update_cpu(self.lr) {
                    continue;
                }

                let Some(grad) = param.grad_arc() else {
                    continue;
                };
                let mut data = param.data_mut();
                if let (Some(data_slice), Some(grad_slice)) = (data.as_slice_mut(), grad.as_slice())
                {
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
                if !param.has_pending_grad() {
                    continue;
                }
                let m = self.momentum;
                let lr = self.lr;
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
                if lowp_cpu::momentum(param, v_buf, lr, m) {
                    continue;
                }
                let Some(grad) = param.grad_arc() else {
                    continue;
                };
                let mut next_v = v_buf.data();
                with_parameter_update_preserving_dtype(param, |mut data| {
                    if let (Some(data_slice), Some(v_slice), Some(grad_slice)) =
                        (data.as_slice_mut(), next_v.as_slice_mut(), grad.as_slice())
                    {
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
                });
                v_buf.set_array_f32_with_dtype(next_v, self.state_dtype);
            }
        }
        self.master_weights.publish(&self.params);
    }
}

impl SGD {
    fn ensure_velocity(&mut self, index: usize) {
        if self.velocities[index].is_some() {
            return;
        }
        let param = &self.params[index];
        if param.is_cuda() && self.state_dtype != DType::F32 {
            self.velocities[index] = Some(lowp_cuda::zero_state(param, self.state_dtype));
            return;
        }
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

    #[cfg(feature = "dev-tools")]
    pub fn dev_last_cuda_batched_update_count(&self) -> usize {
        self.last_cuda_batched_update_count
    }
}

fn try_cuda_sgd_step_batched(params: &[Tensor], lr: f32) -> (Vec<bool>, usize) {
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
        return (updated, 0);
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
        return (updated, 0);
    }
    if cuda::sgd_update_f32_batched_no_host(&param_bufs, &grad_bufs, lr).is_err() {
        return (updated, 0);
    }

    let batched_update_count = indices.len();
    for (slot, param_buf) in indices.iter().copied().zip(param_bufs) {
        updated[slot] = finalize_cuda_param_update(&params[slot], param_buf);
    }
    (updated, batched_update_count)
}

fn try_cuda_sgd_momentum_step_batched(
    params: &[Tensor],
    velocities: &[Option<Tensor>],
    lr: f32,
    momentum: f32,
) -> (Vec<bool>, usize) {
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
        return (updated, 0);
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
        return (updated, 0);
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
        return (updated, 0);
    }

    let batched_update_count = indices.len();
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
    (updated, batched_update_count)
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
    parameter_steps: Vec<usize>,
    state_dtype: DType,
    exp_avg: Vec<Option<Tensor>>,    // m (一阶矩)
    exp_avg_sq: Vec<Option<Tensor>>, // v (二阶矩)
    master_weights: MasterWeights,
    last_cuda_batched_update_count: usize,
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
            parameter_steps: vec![0; len],
            state_dtype,
            exp_avg: vec![None; len],
            exp_avg_sq: vec![None; len],
            master_weights: MasterWeights::default(),
            last_cuda_batched_update_count: 0,
        }
    }

    #[inline]
    pub fn state_dtype(&self) -> DType {
        self.state_dtype
    }

    /// Preserve sub-quantum updates in lazy F32 master weights for low-precision parameters.
    /// Requires F32 state and four extra bytes per low-precision parameter element.
    /// Enable before the first step. Rebuild the optimizer after externally replacing
    /// parameters or changing their shape, dtype, scale or device. Forward storage
    /// remains F16/BF16/I8; the default native update path does not allocate masters.
    pub fn with_fp32_master_weights(mut self) -> Self {
        assert_eq!(
            self.state_dtype,
            DType::F32,
            "master weights require F32 optimizer state"
        );
        assert_eq!(self.step_count, 0, "enable master weights before training");
        self.master_weights.enable(self.params.len());
        self
    }

    /// Keep F32 masters and recalibrate I8 absmax scales after every update.
    /// This explicitly replaces fixed I8 scales, avoiding clipping to the initial
    /// parameter range. CUDA keeps parameters/gradients on device and reads back
    /// only a scale scalar. F16/BF16 parameters use the ordinary master path.
    pub fn with_dynamic_i8_master_weights(self) -> Self {
        let mut opt = self.with_fp32_master_weights();
        opt.master_weights.dynamic_i8 = true;
        opt
    }

    fn ensure_state(&mut self, index: usize) {
        if self.exp_avg[index].is_some() {
            return;
        }
        let param = &self.params[index];
        if param.is_cuda() && self.state_dtype != DType::F32 {
            self.exp_avg[index] = Some(lowp_cuda::zero_state(param, self.state_dtype));
            self.exp_avg_sq[index] = Some(lowp_cuda::zero_state(param, self.state_dtype));
            return;
        }
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

    #[cfg(feature = "dev-tools")]
    pub fn dev_state_pair_count(&self) -> usize {
        self.exp_avg
            .iter()
            .zip(self.exp_avg_sq.iter())
            .filter(|(exp_avg, exp_avg_sq)| exp_avg.is_some() && exp_avg_sq.is_some())
            .count()
    }

    #[cfg(feature = "dev-tools")]
    pub fn dev_all_states_are_f32_cuda_resident(&self) -> bool {
        self.exp_avg
            .iter()
            .zip(self.exp_avg_sq.iter())
            .all(|(exp_avg, exp_avg_sq)| {
                let (Some(exp_avg), Some(exp_avg_sq)) = (exp_avg.as_ref(), exp_avg_sq.as_ref())
                else {
                    return false;
                };
                exp_avg.dtype() == DType::F32
                    && exp_avg_sq.dtype() == DType::F32
                    && exp_avg.is_cuda()
                    && exp_avg_sq.is_cuda()
                    && exp_avg.cloned_cuda_f32_buffer().is_some()
                    && exp_avg_sq.cloned_cuda_f32_buffer().is_some()
                    && !exp_avg.dev_has_host_f32_data()
                    && !exp_avg_sq.dev_has_host_f32_data()
            })
    }

    #[cfg(feature = "dev-tools")]
    pub fn dev_last_cuda_batched_update_count(&self) -> usize {
        self.last_cuda_batched_update_count
    }
}

impl Optimizer for Adam {
    fn params(&self) -> &Vec<Tensor> {
        &self.params
    }

    fn step(&mut self) {
        for (index, param) in self.params.iter().enumerate() {
            param.validate_optimizer_gradient(index);
        }
        self.step_count += 1;
        let (beta1, beta2) = self.betas;

        for index in 0..self.params.len() {
            if self.params[index].has_pending_grad() {
                self.ensure_state(index);
                self.parameter_steps[index] += 1;
            }
        }

        let prepared = self.master_weights.prepare(&self.params);
        let params = prepared.as_deref().unwrap_or(&self.params);

        let (cuda_updated, cuda_batched_update_count) = if self.state_dtype == DType::F32 {
            try_cuda_adam_step_batched(
                params,
                &self.exp_avg,
                &self.exp_avg_sq,
                self.lr,
                beta1,
                beta2,
                &self.parameter_steps,
                self.eps,
            )
        } else {
            lowp_cuda::adam_step(
                params,
                &self.exp_avg,
                &self.exp_avg_sq,
                self.lr,
                beta1,
                beta2,
                &self.parameter_steps,
                self.eps,
                self.state_dtype,
            )
        };
        self.last_cuda_batched_update_count = cuda_batched_update_count;

        for (i, param) in params.iter().enumerate() {
            if cuda_updated.get(i).copied().unwrap_or(false) {
                continue;
            }
            if !param.has_pending_grad() {
                continue;
            }

            let (bias_correction1, bias_correction2) =
                adam_bias_corrections(beta1, beta2, self.parameter_steps[i]);

            let lr = self.lr;
            let eps = self.eps;
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

            if lowp_cpu::adam(
                param,
                m_buf,
                v_buf,
                lowp_cpu::AdamConfig {
                    lr,
                    beta1,
                    beta2,
                    correction1: bias_correction1,
                    correction2: bias_correction2,
                    eps,
                },
            ) {
                continue;
            }

            let grad = match param.grad_arc() {
                Some(g) => g,
                None => continue,
            };

            let mut m_next = m_buf.data();
            let mut v_next = v_buf.data();
            with_parameter_update_preserving_dtype(param, |mut data| {
                if let (Some(data_slice), Some(m_slice), Some(v_slice), Some(grad_slice)) = (
                    data.as_slice_mut(),
                    m_next.as_slice_mut(),
                    v_next.as_slice_mut(),
                    grad.as_slice(),
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
            });
            m_buf.set_array_f32_with_dtype(m_next, self.state_dtype);
            v_buf.set_array_f32_with_dtype(v_next, self.state_dtype);
        }
        self.master_weights.publish(&self.params);
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
    parameter_steps: &[usize],
    eps: f32,
) -> (Vec<bool>, usize) {
    let active = || {
        params.iter().enumerate().filter_map(|(index, param)| {
            (param.is_cuda() && param.has_pending_grad()).then_some(index)
        })
    };
    let Some(first) = active().next() else {
        return (vec![false; params.len()], 0);
    };
    let step = parameter_steps[first];
    if active().all(|index| parameter_steps[index] == step) {
        // Ordinary dense training retains the existing single-batch path.
        let (bc1, bc2) = adam_bias_corrections(beta1, beta2, step);
        return try_cuda_adam_step_same_age(
            params,
            exp_avgs,
            exp_avg_sqs,
            lr,
            beta1,
            beta2,
            bc1,
            bc2,
            eps,
        );
    }

    // Conditional parameters can have different moment ages. Batch only peers
    // with the same age so a shared kernel never applies the wrong correction.
    let mut groups = std::collections::BTreeMap::<usize, Vec<usize>>::new();
    for index in active() {
        groups
            .entry(parameter_steps[index])
            .or_default()
            .push(index);
    }
    let mut updated = vec![false; params.len()];
    let mut batched = 0;
    for (step, indices) in groups {
        let group_params = indices
            .iter()
            .map(|&i| params[i].clone())
            .collect::<Vec<_>>();
        let group_m = indices
            .iter()
            .map(|&i| exp_avgs[i].clone())
            .collect::<Vec<_>>();
        let group_v = indices
            .iter()
            .map(|&i| exp_avg_sqs[i].clone())
            .collect::<Vec<_>>();
        let (bc1, bc2) = adam_bias_corrections(beta1, beta2, step);
        let (flags, count) = try_cuda_adam_step_same_age(
            &group_params,
            &group_m,
            &group_v,
            lr,
            beta1,
            beta2,
            bc1,
            bc2,
            eps,
        );
        for (&index, flag) in indices.iter().zip(flags) {
            updated[index] = flag;
        }
        batched += count;
    }
    (updated, batched)
}

fn adam_bias_corrections(beta1: f32, beta2: f32, step: usize) -> (f32, f32) {
    let exponent = step.min(i32::MAX as usize) as i32;
    (1.0 - beta1.powi(exponent), 1.0 - beta2.powi(exponent))
}

#[allow(clippy::too_many_arguments)]
fn try_cuda_adam_step_same_age(
    params: &[Tensor],
    exp_avgs: &[Option<Tensor>],
    exp_avg_sqs: &[Option<Tensor>],
    lr: f32,
    beta1: f32,
    beta2: f32,
    bias_correction1: f32,
    bias_correction2: f32,
    eps: f32,
) -> (Vec<bool>, usize) {
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
        return (updated, 0);
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
        return (updated, 0);
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
        return (updated, 0);
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
    (updated, indices.len())
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
#[path = "../../tests/unit/optim/tests.rs"]
mod tests;
