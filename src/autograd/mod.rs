//! Tensor storage, device state, and dynamic reverse-mode differentiation.

mod backward;
mod checkpoint;
mod creation;
mod device;
mod diagnostics;
mod operations;
mod optimizer;
mod quantization;
mod storage;
mod views;

use crate::ops::fp_kernels::{
    adam_update_bf16_f32_arch, adam_update_f16_f32_arch, sgd_momentum_update_bf16_f32_arch,
    sgd_momentum_update_f16_f32_arch, sgd_update_bf16_f32_arch, sgd_update_f16_f32_arch,
};
use crate::ops::int8_kernels::{
    adam_update_i8_f32_arch, sgd_momentum_update_i8_f32_arch, sgd_update_i8_f32_arch,
};
use crate::precision::{
    DType, ParameterQuantization, allow_parameter_dtype_copies, default_parameter_dtype,
    default_parameter_quantization,
};
use half::{bf16, f16, slice::HalfFloatSliceExt};
use ndarray::Zip;
use ndarray::prelude::*;
pub use ndarray::{ArcArray, IxDyn};
use serde::{Deserialize, Serialize};
use std::cell::{Cell, Ref, RefCell, RefMut};
use std::collections::HashSet;
use std::marker::PhantomData;
use std::rc::Rc;

pub type BackwardOp = Rc<dyn Fn(&ArrayViewD<f32>)>;

#[inline]
fn cuda_gradient_operation<T>(operation: &str, run: impl FnOnce() -> Result<T, String>) -> T {
    #[cfg(test)]
    let result =
        crate::gradient_accumulation_tests::check_cuda_failure(operation).and_then(|()| run());
    #[cfg(not(test))]
    let result = run();
    result.unwrap_or_else(|error| panic!("CUDA gradient failure ({operation}): {error}"))
}

enum BackwardSeed {
    Ones,
    Host(ArrayD<f32>),
    Cuda(crate::ops::cuda::CudaBuffer),
}

thread_local! {
    static INFERENCE_MODE: Cell<bool> = const { Cell::new(false) };
    static NO_GRAD_DEPTH: Cell<usize> = const { Cell::new(0) };
    static F32_COMPUTE_SCRATCH: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    static STRICT_DEVICE_EXECUTION: Cell<bool> = Cell::new(env_strict_device_execution());
}

/// Caller-thread no-grad scope, restored when the guard is dropped.
///
/// ```compile_fail
/// let guard = lumen::autograd::NoGradGuard::enter();
/// std::thread::spawn(move || drop(guard));
/// ```
pub struct NoGradGuard {
    _caller_thread: PhantomData<Rc<()>>,
}

impl NoGradGuard {
    pub fn enter() -> Self {
        NO_GRAD_DEPTH.with(|depth| depth.set(depth.get() + 1));
        Self {
            _caller_thread: PhantomData,
        }
    }
}

impl Drop for NoGradGuard {
    fn drop(&mut self) {
        NO_GRAD_DEPTH.with(|depth| depth.set(depth.get().saturating_sub(1)));
    }
}

// 开/关 全局推理模式（eval_mode/train_mode 可调用它）
pub fn set_inference_mode(on: bool) {
    INFERENCE_MODE.with(|flag| flag.set(on));
}

#[inline]
pub fn is_inference_mode() -> bool {
    INFERENCE_MODE.with(|flag| flag.get())
}

// no_grad 的判定：
// - 在 NoGradGuard 作用域内为 true
// - 或者处于 inference_mode 为 true
#[inline]
pub fn is_no_grad() -> bool {
    NO_GRAD_DEPTH.with(|depth| depth.get() > 0) || is_inference_mode()
}

// 便利封装：no_grad(|| { ... })
pub fn no_grad<R>(f: impl FnOnce() -> R) -> R {
    let _g = NoGradGuard::enter();
    f()
}

fn env_strict_device_execution() -> bool {
    std::env::var("LUMEN_STRICT_DEVICE")
        .ok()
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

pub fn set_strict_device_execution(enabled: bool) {
    STRICT_DEVICE_EXECUTION.with(|flag| flag.set(enabled));
}

pub fn is_strict_device_execution() -> bool {
    STRICT_DEVICE_EXECUTION.with(|flag| flag.get())
}

/// Caller-thread strict-device scope; the guard must stay on this thread.
///
/// ```compile_fail
/// let guard = lumen::autograd::set_strict_device_execution_scoped(true);
/// std::thread::spawn(move || drop(guard));
/// ```
pub struct StrictDeviceExecutionGuard {
    previous: bool,
    _caller_thread: PhantomData<Rc<()>>,
}

pub fn set_strict_device_execution_scoped(enabled: bool) -> StrictDeviceExecutionGuard {
    let previous = is_strict_device_execution();
    set_strict_device_execution(enabled);
    StrictDeviceExecutionGuard {
        previous,
        _caller_thread: PhantomData,
    }
}

impl Drop for StrictDeviceExecutionGuard {
    fn drop(&mut self) {
        set_strict_device_execution(self.previous);
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub enum TensorRawData {
    F32(Vec<f32>),
    F16(Vec<u16>),
    BF16(Vec<u16>),
    I8 { values: Vec<i8>, scale: f32 },
}

pub enum TensorStorageView<'a> {
    F32(ArrayViewD<'a, f32>),
    F16(ArrayViewD<'a, f16>),
    BF16(ArrayViewD<'a, bf16>),
}

pub enum TensorStorageViewMut<'a> {
    F32(ArrayViewMutD<'a, f32>),
    F16(ArrayViewMutD<'a, f16>),
    BF16(ArrayViewMutD<'a, bf16>),
    I8(ArrayViewMutD<'a, i8>, f32),
}

pub(crate) enum TensorStorageOwned {
    F32(ArcArray<f32, IxDyn>),
    F16(ArcArray<f16, IxDyn>),
    BF16(ArcArray<bf16, IxDyn>),
    I8(ArcArray<i8, IxDyn>, f32),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Device {
    Cpu,
    Cuda,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StoragePreference {
    Auto,
    Native,
    F32Compute,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DTypeDispatch {
    PureF32,
    SameF16,
    SameBF16,
    SameI8,
    Mixed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KernelRouteClass {
    GenericMatmul,
    DecodeKernel,
    Argmax,
}

#[inline]
pub fn classify_dtype_dispatch(input_dtype: DType, tensor_dtype: DType) -> DTypeDispatch {
    match (input_dtype, tensor_dtype) {
        (DType::F32, DType::F32) => DTypeDispatch::PureF32,
        (DType::F16, DType::F16) => DTypeDispatch::SameF16,
        (DType::BF16, DType::BF16) => DTypeDispatch::SameBF16,
        (DType::I8, DType::I8) => DTypeDispatch::SameI8,
        _ => DTypeDispatch::Mixed,
    }
}

#[inline]
pub fn preferred_parameter_storage_for_input_dtype(
    input_dtype: DType,
    tensor_dtype: DType,
) -> StoragePreference {
    preferred_parameter_storage_for_route(input_dtype, tensor_dtype, KernelRouteClass::DecodeKernel)
}

#[inline]
pub fn preferred_parameter_storage_for_route(
    input_dtype: DType,
    tensor_dtype: DType,
    route: KernelRouteClass,
) -> StoragePreference {
    match classify_dtype_dispatch(input_dtype, tensor_dtype) {
        DTypeDispatch::PureF32 => StoragePreference::Auto,
        DTypeDispatch::SameF16 => StoragePreference::Auto,
        DTypeDispatch::SameBF16 if matches!(route, KernelRouteClass::GenericMatmul) => {
            StoragePreference::Auto
        }
        DTypeDispatch::SameBF16 | DTypeDispatch::SameI8 => StoragePreference::Native,
        DTypeDispatch::Mixed => {
            #[cfg(all(feature = "arm64-fp-kernels", target_arch = "aarch64"))]
            {
                if matches!(route, KernelRouteClass::GenericMatmul)
                    && input_dtype == DType::F32
                    && matches!(tensor_dtype, DType::BF16 | DType::F16)
                {
                    return StoragePreference::Native;
                }
            }

            let _ = route;
            StoragePreference::Auto
        }
    }
}

pub struct TensorData {
    pub data: ArcArray<f32, IxDyn>,
    pub f16_data: Option<ArcArray<f16, IxDyn>>,
    pub bf16_data: Option<ArcArray<bf16, IxDyn>>,
    pub i8_data: Option<ArcArray<i8, IxDyn>>,
    pub cuda_f32_data: Option<crate::ops::cuda::CudaBuffer>,
    pub cuda_f16_data: Option<crate::ops::cuda::CudaBuffer>,
    pub cuda_bf16_data: Option<crate::ops::cuda::CudaBuffer>,
    pub cuda_i8_data: Option<crate::ops::cuda::CudaBuffer>,
    pub i8_scale: Option<f32>,
    pub has_f32_data: bool,
    pub storage_dtype: DType,
    pub cache_dirty: bool,
    pub is_parameter: bool,
    // 梯度：使用 ArcArray 便于 optimizer 侧 clone 为零拷贝（仅增 refcount）
    pub grad: Option<ArcArray<f32, IxDyn>>,
    pub cuda_f32_grad: Option<crate::ops::cuda::CudaBuffer>,
    pub parents: Vec<Tensor>,
    // backward_op 接收 grad 的 view，避免在反传遍历时额外 to_owned
    pub backward_op: Option<BackwardOp>,
    pub requires_grad: bool,
    pub device: Device,
}

#[derive(Clone)]
pub struct Tensor(pub(crate) Rc<RefCell<TensorData>>);

// 切断梯度流（等价于 t.detach()）
pub fn detach(t: &Tensor) -> Tensor {
    t.detach()
}

pub fn assert_same_device(lhs: &Tensor, rhs: &Tensor, op_name: &str) -> Device {
    let lhs_device = lhs.device();
    let rhs_device = rhs.device();
    assert!(
        lhs_device == rhs_device,
        "{op_name} expects tensors on the same device, got lhs={lhs_device:?}, rhs={rhs_device:?}. Move them with to_cuda()/to_cpu() before combining them."
    );
    lhs_device
}

pub fn assert_native_device_support(device: Device, op_name: &str, cuda_supported: bool) {
    if !is_strict_device_execution() {
        return;
    }

    match device {
        Device::Cpu => {}
        Device::Cuda => {
            assert!(
                cuda_supported,
                "{op_name} does not have a native CUDA implementation yet. This operation would fall back through CPU semantics, which is disallowed while strict device execution is enabled."
            );
        }
    }
}

#[cfg(test)]
#[path = "../../tests/unit/autograd/tests.rs"]
mod tests;
