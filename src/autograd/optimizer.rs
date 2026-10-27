use super::*;
use crate::precision::I8Requantizer;

impl Tensor {
    pub(crate) fn validate_optimizer_gradient(&self, parameter_index: usize) {
        let inner = self.0.borrow();
        let shape = Self::logical_shape(&inner);
        if let Some(grad) = &inner.grad {
            assert_eq!(
                shape,
                grad.shape(),
                "Optimizer gradient shape mismatch at parameter {parameter_index}"
            );
        }
        if let Some(buffer) = &inner.cuda_f32_grad {
            assert_eq!(
                buffer.len(),
                Self::checked_shape_len(shape, "Optimizer gradient"),
                "Optimizer CUDA gradient length mismatch at parameter {parameter_index}"
            );
        }
    }

    pub(crate) fn apply_sgd_update_cpu(&self, lr: f32) -> bool {
        if self.device() != Device::Cpu {
            return false;
        }
        let Some(grad) = self.0.borrow().grad.clone() else {
            return false;
        };

        if self.dtype() != DType::F32 {
            self.with_native_storage_view_mut(|view| match view {
                TensorStorageViewMut::F32(_) => unreachable!("checked non-f32 dtype above"),
                TensorStorageViewMut::F16(mut data) => {
                    if let (Some(data_slice), Some(grad_slice)) =
                        (data.as_slice_mut(), grad.as_slice())
                    {
                        if !sgd_update_f16_f32_arch(data_slice, grad_slice, lr) {
                            for (w, g) in data_slice.iter_mut().zip(grad_slice.iter()) {
                                *w = f16::from_f32(w.to_f32() - lr * *g);
                            }
                        }
                    } else {
                        Zip::from(data.view_mut())
                            .and(grad.view())
                            .for_each(|w, g| {
                                *w = f16::from_f32(w.to_f32() - lr * *g);
                            });
                    }
                }
                TensorStorageViewMut::BF16(mut data) => {
                    if let (Some(data_slice), Some(grad_slice)) =
                        (data.as_slice_mut(), grad.as_slice())
                    {
                        if !sgd_update_bf16_f32_arch(data_slice, grad_slice, lr) {
                            for (w, g) in data_slice.iter_mut().zip(grad_slice.iter()) {
                                *w = bf16::from_f32(w.to_f32() - lr * *g);
                            }
                        }
                    } else {
                        Zip::from(data.view_mut())
                            .and(grad.view())
                            .for_each(|w, g| {
                                *w = bf16::from_f32(w.to_f32() - lr * *g);
                            });
                    }
                }
                TensorStorageViewMut::I8(mut data, scale) => {
                    let quantizer = I8Requantizer::new(scale);
                    let quantize = |value| quantizer.quantize(value);
                    if let (Some(data_slice), Some(grad_slice)) =
                        (data.as_slice_mut(), grad.as_slice())
                    {
                        if !sgd_update_i8_f32_arch(data_slice, scale, grad_slice, lr) {
                            for (w, g) in data_slice.iter_mut().zip(grad_slice.iter()) {
                                let updated = (*w as f32) * scale - lr * *g;
                                *w = quantize(updated);
                            }
                        }
                    } else {
                        Zip::from(data.view_mut())
                            .and(grad.view())
                            .for_each(|w, g| {
                                let updated = (*w as f32) * scale - lr * *g;
                                *w = quantize(updated);
                            });
                    }
                }
            });
            return true;
        }

        self.ensure_f32_data(true);
        let mut inner = self.0.borrow_mut();
        let data = &mut inner.data;
        if let (Some(data_slice), Some(grad_slice)) = (data.as_slice_mut(), grad.as_slice()) {
            for (w, g) in data_slice.iter_mut().zip(grad_slice.iter()) {
                *w -= lr * *g;
            }
        } else {
            Zip::from(data.view_mut())
                .and(grad.view())
                .for_each(|w, g| *w -= lr * *g);
        }
        true
    }

    pub(crate) fn apply_sgd_momentum_update_cpu(
        &self,
        velocity: &Tensor,
        lr: f32,
        momentum: f32,
    ) -> bool {
        if self.device() != Device::Cpu
            || velocity.device() != Device::Cpu
            || velocity.dtype() != DType::F32
            || Rc::ptr_eq(&self.0, &velocity.0)
        {
            return false;
        }
        let Some(grad) = self.0.borrow().grad.clone() else {
            return false;
        };

        if self.dtype() != DType::F32 {
            velocity.ensure_f32_data(true);
            let mut velocity_inner = velocity.0.borrow_mut();
            let velocity_data = &mut velocity_inner.data;

            self.with_native_storage_view_mut(|view| match view {
                TensorStorageViewMut::F32(_) => unreachable!("checked non-f32 dtype above"),
                TensorStorageViewMut::F16(mut data) => {
                    if let (Some(data_slice), Some(velocity_slice), Some(grad_slice)) = (
                        data.as_slice_mut(),
                        velocity_data.as_slice_mut(),
                        grad.as_slice(),
                    ) {
                        if !sgd_momentum_update_f16_f32_arch(
                            data_slice,
                            velocity_slice,
                            grad_slice,
                            lr,
                            momentum,
                        ) {
                            for ((w, v), g) in data_slice
                                .iter_mut()
                                .zip(velocity_slice.iter_mut())
                                .zip(grad_slice.iter())
                            {
                                *v = momentum * *v + *g;
                                *w = f16::from_f32(w.to_f32() - lr * *v);
                            }
                        }
                    } else {
                        Zip::from(data.view_mut())
                            .and(velocity_data.view_mut())
                            .and(grad.view())
                            .for_each(|w, v, g| {
                                *v = momentum * *v + *g;
                                *w = f16::from_f32(w.to_f32() - lr * *v);
                            });
                    }
                }
                TensorStorageViewMut::BF16(mut data) => {
                    if let (Some(data_slice), Some(velocity_slice), Some(grad_slice)) = (
                        data.as_slice_mut(),
                        velocity_data.as_slice_mut(),
                        grad.as_slice(),
                    ) {
                        if !sgd_momentum_update_bf16_f32_arch(
                            data_slice,
                            velocity_slice,
                            grad_slice,
                            lr,
                            momentum,
                        ) {
                            for ((w, v), g) in data_slice
                                .iter_mut()
                                .zip(velocity_slice.iter_mut())
                                .zip(grad_slice.iter())
                            {
                                *v = momentum * *v + *g;
                                *w = bf16::from_f32(w.to_f32() - lr * *v);
                            }
                        }
                    } else {
                        Zip::from(data.view_mut())
                            .and(velocity_data.view_mut())
                            .and(grad.view())
                            .for_each(|w, v, g| {
                                *v = momentum * *v + *g;
                                *w = bf16::from_f32(w.to_f32() - lr * *v);
                            });
                    }
                }
                TensorStorageViewMut::I8(mut data, scale) => {
                    let quantizer = I8Requantizer::new(scale);
                    let quantize = |value| quantizer.quantize(value);
                    if let (Some(data_slice), Some(velocity_slice), Some(grad_slice)) = (
                        data.as_slice_mut(),
                        velocity_data.as_slice_mut(),
                        grad.as_slice(),
                    ) {
                        if !sgd_momentum_update_i8_f32_arch(
                            data_slice,
                            scale,
                            velocity_slice,
                            grad_slice,
                            lr,
                            momentum,
                        ) {
                            for ((w, v), g) in data_slice
                                .iter_mut()
                                .zip(velocity_slice.iter_mut())
                                .zip(grad_slice.iter())
                            {
                                *v = momentum * *v + *g;
                                let updated = (*w as f32) * scale - lr * *v;
                                *w = quantize(updated);
                            }
                        }
                    } else {
                        Zip::from(data.view_mut())
                            .and(velocity_data.view_mut())
                            .and(grad.view())
                            .for_each(|w, v, g| {
                                *v = momentum * *v + *g;
                                let updated = (*w as f32) * scale - lr * *v;
                                *w = quantize(updated);
                            });
                    }
                }
            });
            return true;
        }

        self.ensure_f32_data(true);
        velocity.ensure_f32_data(true);
        let mut param_inner = self.0.borrow_mut();
        let mut velocity_inner = velocity.0.borrow_mut();
        let data = &mut param_inner.data;
        let velocity_data = &mut velocity_inner.data;

        if let (Some(data_slice), Some(velocity_slice), Some(grad_slice)) = (
            data.as_slice_mut(),
            velocity_data.as_slice_mut(),
            grad.as_slice(),
        ) {
            for ((w, v), g) in data_slice
                .iter_mut()
                .zip(velocity_slice.iter_mut())
                .zip(grad_slice.iter())
            {
                *v = momentum * *v + *g;
                *w -= lr * *v;
            }
        } else {
            Zip::from(data.view_mut())
                .and(velocity_data.view_mut())
                .and(grad.view())
                .for_each(|w, v, g| {
                    *v = momentum * *v + *g;
                    *w -= lr * *v;
                });
        }
        true
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn apply_adam_update_cpu(
        &self,
        exp_avg: &Tensor,
        exp_avg_sq: &Tensor,
        lr: f32,
        beta1: f32,
        beta2: f32,
        bias_correction1: f32,
        bias_correction2: f32,
        eps: f32,
    ) -> bool {
        if self.device() != Device::Cpu
            || exp_avg.device() != Device::Cpu
            || exp_avg_sq.device() != Device::Cpu
            || exp_avg.dtype() != DType::F32
            || exp_avg_sq.dtype() != DType::F32
            || Rc::ptr_eq(&self.0, &exp_avg.0)
            || Rc::ptr_eq(&self.0, &exp_avg_sq.0)
            || Rc::ptr_eq(&exp_avg.0, &exp_avg_sq.0)
        {
            return false;
        }
        let Some(grad) = self.0.borrow().grad.clone() else {
            return false;
        };

        if self.dtype() != DType::F32 {
            exp_avg.ensure_f32_data(true);
            exp_avg_sq.ensure_f32_data(true);
            let mut exp_avg_inner = exp_avg.0.borrow_mut();
            let mut exp_avg_sq_inner = exp_avg_sq.0.borrow_mut();
            let m_data = &mut exp_avg_inner.data;
            let v_data = &mut exp_avg_sq_inner.data;

            self.with_native_storage_view_mut(|view| match view {
                TensorStorageViewMut::F32(_) => unreachable!("checked non-f32 dtype above"),
                TensorStorageViewMut::F16(mut data) => {
                    if let (Some(data_slice), Some(m_slice), Some(v_slice), Some(grad_slice)) = (
                        data.as_slice_mut(),
                        m_data.as_slice_mut(),
                        v_data.as_slice_mut(),
                        grad.as_slice(),
                    ) {
                        if !adam_update_f16_f32_arch(
                            data_slice,
                            m_slice,
                            v_slice,
                            grad_slice,
                            lr,
                            beta1,
                            beta2,
                            bias_correction1,
                            bias_correction2,
                            eps,
                        ) {
                            for (((w, m), v), g) in data_slice
                                .iter_mut()
                                .zip(m_slice.iter_mut())
                                .zip(v_slice.iter_mut())
                                .zip(grad_slice.iter())
                            {
                                *m = beta1 * *m + (1.0 - beta1) * *g;
                                *v = beta2 * *v + (1.0 - beta2) * *g * *g;
                                let m_hat = *m / bias_correction1;
                                let v_hat = *v / bias_correction2;
                                *w =
                                    f16::from_f32(w.to_f32() - lr * (m_hat / (v_hat.sqrt() + eps)));
                            }
                        }
                    } else {
                        Zip::from(data.view_mut())
                            .and(m_data.view_mut())
                            .and(v_data.view_mut())
                            .and(grad.view())
                            .for_each(|w, m, v, g| {
                                *m = beta1 * *m + (1.0 - beta1) * *g;
                                *v = beta2 * *v + (1.0 - beta2) * *g * *g;
                                let m_hat = *m / bias_correction1;
                                let v_hat = *v / bias_correction2;
                                *w =
                                    f16::from_f32(w.to_f32() - lr * (m_hat / (v_hat.sqrt() + eps)));
                            });
                    }
                }
                TensorStorageViewMut::BF16(mut data) => {
                    if let (Some(data_slice), Some(m_slice), Some(v_slice), Some(grad_slice)) = (
                        data.as_slice_mut(),
                        m_data.as_slice_mut(),
                        v_data.as_slice_mut(),
                        grad.as_slice(),
                    ) {
                        if !adam_update_bf16_f32_arch(
                            data_slice,
                            m_slice,
                            v_slice,
                            grad_slice,
                            lr,
                            beta1,
                            beta2,
                            bias_correction1,
                            bias_correction2,
                            eps,
                        ) {
                            for (((w, m), v), g) in data_slice
                                .iter_mut()
                                .zip(m_slice.iter_mut())
                                .zip(v_slice.iter_mut())
                                .zip(grad_slice.iter())
                            {
                                *m = beta1 * *m + (1.0 - beta1) * *g;
                                *v = beta2 * *v + (1.0 - beta2) * *g * *g;
                                let m_hat = *m / bias_correction1;
                                let v_hat = *v / bias_correction2;
                                *w = bf16::from_f32(
                                    w.to_f32() - lr * (m_hat / (v_hat.sqrt() + eps)),
                                );
                            }
                        }
                    } else {
                        Zip::from(data.view_mut())
                            .and(m_data.view_mut())
                            .and(v_data.view_mut())
                            .and(grad.view())
                            .for_each(|w, m, v, g| {
                                *m = beta1 * *m + (1.0 - beta1) * *g;
                                *v = beta2 * *v + (1.0 - beta2) * *g * *g;
                                let m_hat = *m / bias_correction1;
                                let v_hat = *v / bias_correction2;
                                *w = bf16::from_f32(
                                    w.to_f32() - lr * (m_hat / (v_hat.sqrt() + eps)),
                                );
                            });
                    }
                }
                TensorStorageViewMut::I8(mut data, scale) => {
                    let quantizer = I8Requantizer::new(scale);
                    let quantize = |value| quantizer.quantize(value);
                    if let (Some(data_slice), Some(m_slice), Some(v_slice), Some(grad_slice)) = (
                        data.as_slice_mut(),
                        m_data.as_slice_mut(),
                        v_data.as_slice_mut(),
                        grad.as_slice(),
                    ) {
                        if !adam_update_i8_f32_arch(
                            data_slice,
                            scale,
                            m_slice,
                            v_slice,
                            grad_slice,
                            lr,
                            beta1,
                            beta2,
                            bias_correction1,
                            bias_correction2,
                            eps,
                        ) {
                            for (((w, m), v), g) in data_slice
                                .iter_mut()
                                .zip(m_slice.iter_mut())
                                .zip(v_slice.iter_mut())
                                .zip(grad_slice.iter())
                            {
                                *m = beta1 * *m + (1.0 - beta1) * *g;
                                *v = beta2 * *v + (1.0 - beta2) * *g * *g;
                                let m_hat = *m / bias_correction1;
                                let v_hat = *v / bias_correction2;
                                let updated =
                                    (*w as f32) * scale - lr * (m_hat / (v_hat.sqrt() + eps));
                                *w = quantize(updated);
                            }
                        }
                    } else {
                        Zip::from(data.view_mut())
                            .and(m_data.view_mut())
                            .and(v_data.view_mut())
                            .and(grad.view())
                            .for_each(|w, m, v, g| {
                                *m = beta1 * *m + (1.0 - beta1) * *g;
                                *v = beta2 * *v + (1.0 - beta2) * *g * *g;
                                let m_hat = *m / bias_correction1;
                                let v_hat = *v / bias_correction2;
                                let updated =
                                    (*w as f32) * scale - lr * (m_hat / (v_hat.sqrt() + eps));
                                *w = quantize(updated);
                            });
                    }
                }
            });
            return true;
        }

        self.ensure_f32_data(true);
        exp_avg.ensure_f32_data(true);
        exp_avg_sq.ensure_f32_data(true);
        let mut param_inner = self.0.borrow_mut();
        let mut exp_avg_inner = exp_avg.0.borrow_mut();
        let mut exp_avg_sq_inner = exp_avg_sq.0.borrow_mut();
        let data = &mut param_inner.data;
        let m_data = &mut exp_avg_inner.data;
        let v_data = &mut exp_avg_sq_inner.data;

        if let (Some(data_slice), Some(m_slice), Some(v_slice), Some(grad_slice)) = (
            data.as_slice_mut(),
            m_data.as_slice_mut(),
            v_data.as_slice_mut(),
            grad.as_slice(),
        ) {
            for (((w, m), v), g) in data_slice
                .iter_mut()
                .zip(m_slice.iter_mut())
                .zip(v_slice.iter_mut())
                .zip(grad_slice.iter())
            {
                *m = beta1 * *m + (1.0 - beta1) * *g;
                *v = beta2 * *v + (1.0 - beta2) * *g * *g;
                let m_hat = *m / bias_correction1;
                let v_hat = *v / bias_correction2;
                *w -= lr * (m_hat / (v_hat.sqrt() + eps));
            }
        } else {
            Zip::from(data.view_mut())
                .and(m_data.view_mut())
                .and(v_data.view_mut())
                .and(grad.view())
                .for_each(|w, m, v, g| {
                    *m = beta1 * *m + (1.0 - beta1) * *g;
                    *v = beta2 * *v + (1.0 - beta2) * *g * *g;
                    let m_hat = *m / bias_correction1;
                    let v_hat = *v / bias_correction2;
                    *w -= lr * (m_hat / (v_hat.sqrt() + eps));
                });
        }
        true
    }
}
