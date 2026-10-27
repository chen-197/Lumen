use crate::autograd::Tensor;
use crate::precision::{DType, default_parameter_dtype, default_parameter_quantization};
use ndarray::{Array, ArrayD, IxDyn};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::{Normal, Uniform};
use std::cell::Cell;
use std::marker::PhantomData;
use std::rc::Rc;

pub enum InitType {
    XavierUniform, // For Tanh/Sigmoid (Glorot)
    KaimingNormal, // For ReLU/GELU (He)
    Zeros,         // For Bias
    Ones,          // For LayerNorm/RMSNorm weights
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParameterInitMode {
    Full,
    Placeholder,
}

thread_local! {
    static PARAMETER_INIT_MODE: Cell<ParameterInitMode> = const { Cell::new(ParameterInitMode::Full) };
}

/// Restores this thread's parameter initialization mode on drop.
///
/// ```compile_fail
/// use lumen::init::{parameter_init_guard, ParameterInitMode};
/// let guard = parameter_init_guard(ParameterInitMode::Placeholder);
/// std::thread::spawn(move || drop(guard));
/// ```
pub struct ParameterInitModeGuard {
    previous: ParameterInitMode,
    _caller_thread: PhantomData<Rc<()>>,
}

#[inline]
pub fn parameter_init_mode() -> ParameterInitMode {
    PARAMETER_INIT_MODE.with(|mode| mode.get())
}

#[inline]
pub fn set_parameter_init_mode(mode: ParameterInitMode) {
    PARAMETER_INIT_MODE.with(|cell| cell.set(mode));
}

#[inline]
pub fn parameter_init_guard(mode: ParameterInitMode) -> ParameterInitModeGuard {
    let previous = parameter_init_mode();
    set_parameter_init_mode(mode);
    ParameterInitModeGuard {
        previous,
        _caller_thread: PhantomData,
    }
}

#[inline]
pub fn with_parameter_init_mode<R>(mode: ParameterInitMode, f: impl FnOnce() -> R) -> R {
    let _guard = parameter_init_guard(mode);
    f()
}

impl Drop for ParameterInitModeGuard {
    fn drop(&mut self) {
        set_parameter_init_mode(self.previous);
    }
}

#[inline]
fn build_init_data(shape: &[usize], init_type: InitType) -> ArrayD<f32> {
    let shape_dyn = IxDyn(shape);
    if parameter_init_mode() == ParameterInitMode::Placeholder {
        return ArrayD::zeros(shape_dyn);
    }

    match init_type {
        InitType::Zeros => ArrayD::zeros(shape_dyn.clone()),

        InitType::Ones => ArrayD::ones(shape_dyn.clone()),

        InitType::XavierUniform => {
            let fan_in = shape[0] as f32;
            let fan_out = if shape.len() > 1 { shape[1] } else { shape[0] } as f32;
            let limit = (6.0 / (fan_in + fan_out)).sqrt();
            Array::random(shape_dyn.clone(), Uniform::new(-limit, limit)).into_dyn()
        }

        InitType::KaimingNormal => {
            let fan_in = shape[0] as f32;
            let std = (2.0 / fan_in).sqrt();
            Array::random(shape_dyn, Normal::new(0.0, std).unwrap()).into_dyn()
        }
    }
}

pub fn tensor_init(shape: Vec<usize>, init_type: InitType) -> Tensor {
    if parameter_init_mode() == ParameterInitMode::Placeholder {
        let quantization = default_parameter_quantization();
        if quantization.is_enabled() {
            return Tensor::parameter_placeholder_with_quantization(shape.as_slice(), quantization);
        }
        return Tensor::parameter_placeholder_with_dtype(
            shape.as_slice(),
            default_parameter_dtype(),
        );
    }
    Tensor::parameter(build_init_data(shape.as_slice(), init_type))
}

pub fn tensor_init_with_dtype(shape: Vec<usize>, init_type: InitType, dtype: DType) -> Tensor {
    if parameter_init_mode() == ParameterInitMode::Placeholder {
        let quantization = default_parameter_quantization();
        if quantization.is_enabled() && quantization.storage_dtype() == Some(dtype) {
            return Tensor::parameter_placeholder_with_quantization(shape.as_slice(), quantization);
        }
        return Tensor::parameter_placeholder_with_dtype(shape.as_slice(), dtype);
    }
    Tensor::parameter_with_dtype(build_init_data(shape.as_slice(), init_type), dtype)
}

#[cfg(test)]
#[path = "../tests/unit/init/tests.rs"]
mod tests;
