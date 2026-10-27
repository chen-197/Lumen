use crate::arch;
use half::{bf16, f16};

#[cfg(all(feature = "x86-fp-kernels-nightly", target_arch = "x86"))]
use std::arch::x86::__m512h;
#[cfg(all(feature = "x86-fp-kernels", target_arch = "x86"))]
use std::arch::x86::{__m256, __m256bh, __m256i, __m512};
#[cfg(all(feature = "x86-fp-kernels-nightly", target_arch = "x86_64"))]
use std::arch::x86_64::__m512h;
#[cfg(all(feature = "x86-fp-kernels", target_arch = "x86_64"))]
use std::arch::x86_64::{__m256, __m256bh, __m256i, __m512};

#[inline]
fn dot_len_matches<T>(x: &[f32], row: &[T]) -> bool {
    x.len() == row.len()
}

#[inline]
fn dot2_len_matches<T>(x: &[f32], row0: &[T], row1: &[T]) -> bool {
    x.len() == row0.len() && x.len() == row1.len()
}

#[inline]
fn dot3_len_matches<T>(x: &[f32], row0: &[T], row1: &[T], row2: &[T]) -> bool {
    x.len() == row0.len() && x.len() == row1.len() && x.len() == row2.len()
}

#[inline]
fn dot_len_matches_same<T>(x: &[T], row: &[T]) -> bool {
    x.len() == row.len()
}

#[inline]
fn dot2_len_matches_same<T>(x: &[T], row0: &[T], row1: &[T]) -> bool {
    x.len() == row0.len() && x.len() == row1.len()
}

#[inline]
fn dot3_len_matches_same<T>(x: &[T], row0: &[T], row1: &[T], row2: &[T]) -> bool {
    x.len() == row0.len() && x.len() == row1.len() && x.len() == row2.len()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FloatKernelBackend {
    Portable,
    Arm64Neon,
    X86Avx512,
    X86Avx2,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum F16F16KernelBackend {
    Portable,
    X86Avx512Fp16,
    X86Avx2F16c,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bf16Bf16KernelBackend {
    Portable,
    X86Avx512Bf16,
    X86Avx2F32Acc,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FpElementwiseOp {
    Add,
    Sub,
    Mul,
}

#[inline]
pub fn active_float_backend() -> FloatKernelBackend {
    if arch::arm64_fp_kernel_runtime_available() {
        FloatKernelBackend::Arm64Neon
    } else if arch::x86_avx512_fp_kernel_runtime_available() {
        FloatKernelBackend::X86Avx512
    } else if arch::x86_fp_kernel_runtime_available() {
        FloatKernelBackend::X86Avx2
    } else {
        FloatKernelBackend::Portable
    }
}

#[inline]
pub fn active_float_backend_name() -> &'static str {
    match active_float_backend() {
        FloatKernelBackend::Portable => "portable",
        FloatKernelBackend::Arm64Neon => "arm64-neon",
        FloatKernelBackend::X86Avx512 => "x86-avx512",
        FloatKernelBackend::X86Avx2 => "x86-avx2",
    }
}

#[inline]
pub fn active_f16_f16_backend() -> F16F16KernelBackend {
    if arch::x86_avx512_fp16_kernel_runtime_available() {
        F16F16KernelBackend::X86Avx512Fp16
    } else if arch::x86_f16c_kernel_runtime_available() {
        F16F16KernelBackend::X86Avx2F16c
    } else {
        F16F16KernelBackend::Portable
    }
}

#[inline]
pub fn active_f16_f16_backend_name() -> &'static str {
    match active_f16_f16_backend() {
        F16F16KernelBackend::Portable => "portable",
        F16F16KernelBackend::X86Avx512Fp16 => "x86-avx512fp16-nightly",
        F16F16KernelBackend::X86Avx2F16c => "x86-avx2-f16c",
    }
}

#[inline]
pub fn active_bf16_bf16_backend() -> Bf16Bf16KernelBackend {
    if arch::x86_avx512_bf16_kernel_runtime_available() {
        Bf16Bf16KernelBackend::X86Avx512Bf16
    } else if arch::x86_fp_kernel_runtime_available() {
        Bf16Bf16KernelBackend::X86Avx2F32Acc
    } else {
        Bf16Bf16KernelBackend::Portable
    }
}

#[inline]
pub fn active_bf16_bf16_backend_name() -> &'static str {
    match active_bf16_bf16_backend() {
        Bf16Bf16KernelBackend::Portable => "portable",
        Bf16Bf16KernelBackend::X86Avx512Bf16 => "x86-avx512bf16",
        Bf16Bf16KernelBackend::X86Avx2F32Acc => "x86-avx2-bf16-f32acc",
    }
}

#[inline]
fn elementwise_len_matches<T>(lhs: &[T], rhs: &[T], out: &[T]) -> bool {
    lhs.len() == rhs.len() && lhs.len() == out.len()
}

#[inline]
pub fn add_f16_f16_arch(_lhs: &[f16], _rhs: &[f16], _out: &mut [f16]) -> bool {
    elementwise_f16_f16_arch(_lhs, _rhs, _out, FpElementwiseOp::Add)
}

#[inline]
pub fn sub_f16_f16_arch(_lhs: &[f16], _rhs: &[f16], _out: &mut [f16]) -> bool {
    elementwise_f16_f16_arch(_lhs, _rhs, _out, FpElementwiseOp::Sub)
}

#[inline]
pub fn mul_f16_f16_arch(_lhs: &[f16], _rhs: &[f16], _out: &mut [f16]) -> bool {
    elementwise_f16_f16_arch(_lhs, _rhs, _out, FpElementwiseOp::Mul)
}

#[inline]
pub fn add_f16_f16_to_f32_arch(_lhs: &[f16], _rhs: &[f16], _out: &mut [f32]) -> bool {
    elementwise_f16_f16_to_f32_arch(_lhs, _rhs, _out, FpElementwiseOp::Add)
}

#[inline]
pub fn sub_f16_f16_to_f32_arch(_lhs: &[f16], _rhs: &[f16], _out: &mut [f32]) -> bool {
    elementwise_f16_f16_to_f32_arch(_lhs, _rhs, _out, FpElementwiseOp::Sub)
}

#[inline]
pub fn mul_f16_f16_to_f32_arch(_lhs: &[f16], _rhs: &[f16], _out: &mut [f32]) -> bool {
    elementwise_f16_f16_to_f32_arch(_lhs, _rhs, _out, FpElementwiseOp::Mul)
}

#[inline]
fn elementwise_f16_f16_arch(
    _lhs: &[f16],
    _rhs: &[f16],
    _out: &mut [f16],
    _op: FpElementwiseOp,
) -> bool {
    if !elementwise_len_matches(_lhs, _rhs, _out) {
        return false;
    }

    #[cfg(all(
        feature = "x86-fp-kernels-nightly",
        any(target_arch = "x86_64", target_arch = "x86")
    ))]
    {
        if arch::x86_avx512_fp16_kernel_runtime_available() {
            unsafe { elementwise_f16_f16_x86_avx512(_lhs, _rhs, _out, _op) };
            return true;
        }
    }

    #[cfg(all(
        feature = "x86-fp-kernels",
        any(target_arch = "x86_64", target_arch = "x86")
    ))]
    {
        if arch::x86_f16c_kernel_runtime_available() {
            unsafe { elementwise_f16_f16_x86_avx2(_lhs, _rhs, _out, _op) };
            return true;
        }
    }

    false
}

#[inline]
fn elementwise_f16_f16_to_f32_arch(
    _lhs: &[f16],
    _rhs: &[f16],
    _out: &mut [f32],
    _op: FpElementwiseOp,
) -> bool {
    if _lhs.len() != _rhs.len() || _lhs.len() != _out.len() {
        return false;
    }

    #[cfg(all(
        feature = "x86-fp-kernels",
        any(target_arch = "x86_64", target_arch = "x86")
    ))]
    {
        if arch::x86_avx512_fp_kernel_runtime_available() {
            unsafe { elementwise_f16_f16_to_f32_x86_avx512(_lhs, _rhs, _out, _op) };
            return true;
        }
        if arch::x86_f16c_kernel_runtime_available() {
            unsafe { elementwise_f16_f16_to_f32_x86_avx2(_lhs, _rhs, _out, _op) };
            return true;
        }
    }

    false
}

#[inline]
pub fn add_bf16_bf16_arch(_lhs: &[bf16], _rhs: &[bf16], _out: &mut [bf16]) -> bool {
    elementwise_bf16_bf16_arch(_lhs, _rhs, _out, FpElementwiseOp::Add)
}

#[inline]
pub fn sub_bf16_bf16_arch(_lhs: &[bf16], _rhs: &[bf16], _out: &mut [bf16]) -> bool {
    elementwise_bf16_bf16_arch(_lhs, _rhs, _out, FpElementwiseOp::Sub)
}

#[inline]
pub fn mul_bf16_bf16_arch(_lhs: &[bf16], _rhs: &[bf16], _out: &mut [bf16]) -> bool {
    elementwise_bf16_bf16_arch(_lhs, _rhs, _out, FpElementwiseOp::Mul)
}

#[inline]
pub fn add_bf16_bf16_to_f32_arch(_lhs: &[bf16], _rhs: &[bf16], _out: &mut [f32]) -> bool {
    elementwise_bf16_bf16_to_f32_arch(_lhs, _rhs, _out, FpElementwiseOp::Add)
}

#[inline]
pub fn sub_bf16_bf16_to_f32_arch(_lhs: &[bf16], _rhs: &[bf16], _out: &mut [f32]) -> bool {
    elementwise_bf16_bf16_to_f32_arch(_lhs, _rhs, _out, FpElementwiseOp::Sub)
}

#[inline]
pub fn mul_bf16_bf16_to_f32_arch(_lhs: &[bf16], _rhs: &[bf16], _out: &mut [f32]) -> bool {
    elementwise_bf16_bf16_to_f32_arch(_lhs, _rhs, _out, FpElementwiseOp::Mul)
}

#[inline]
fn elementwise_bf16_bf16_arch(
    _lhs: &[bf16],
    _rhs: &[bf16],
    _out: &mut [bf16],
    _op: FpElementwiseOp,
) -> bool {
    if !elementwise_len_matches(_lhs, _rhs, _out) {
        return false;
    }

    #[cfg(all(
        feature = "x86-fp-kernels",
        any(target_arch = "x86_64", target_arch = "x86")
    ))]
    {
        if arch::x86_fp_kernel_runtime_available() {
            unsafe { elementwise_bf16_bf16_x86_avx2(_lhs, _rhs, _out, _op) };
            return true;
        }
    }

    false
}

#[inline]
fn elementwise_bf16_bf16_to_f32_arch(
    _lhs: &[bf16],
    _rhs: &[bf16],
    _out: &mut [f32],
    _op: FpElementwiseOp,
) -> bool {
    if _lhs.len() != _rhs.len() || _lhs.len() != _out.len() {
        return false;
    }

    #[cfg(all(
        feature = "x86-fp-kernels",
        any(target_arch = "x86_64", target_arch = "x86")
    ))]
    {
        if arch::x86_avx512_bf16_kernel_runtime_available() {
            unsafe { elementwise_bf16_bf16_to_f32_x86_avx512(_lhs, _rhs, _out, _op) };
            return true;
        }
        if arch::x86_fp_kernel_runtime_available() {
            unsafe { elementwise_bf16_bf16_to_f32_x86_avx2(_lhs, _rhs, _out, _op) };
            return true;
        }
    }

    false
}

#[inline]
pub fn sum_f16_arch(_x: &[f16]) -> Option<f32> {
    if _x.is_empty() {
        return Some(0.0);
    }

    #[cfg(all(
        feature = "x86-fp-kernels",
        any(target_arch = "x86_64", target_arch = "x86")
    ))]
    {
        if arch::x86_avx512_fp_kernel_runtime_available() {
            return Some(unsafe { sum_f16_x86_avx512(_x) });
        }
        if arch::x86_f16c_kernel_runtime_available() {
            return Some(unsafe { sum_f16_x86_avx2(_x) });
        }
    }

    None
}

#[inline]
pub fn sum_bf16_arch(_x: &[bf16]) -> Option<f32> {
    if _x.is_empty() {
        return Some(0.0);
    }

    #[cfg(all(
        feature = "x86-fp-kernels",
        any(target_arch = "x86_64", target_arch = "x86")
    ))]
    {
        if arch::x86_avx512_bf16_kernel_runtime_available() {
            return Some(unsafe { sum_bf16_x86_avx512(_x) });
        }
        if arch::x86_fp_kernel_runtime_available() {
            return Some(unsafe { sum_bf16_x86_avx2(_x) });
        }
    }

    None
}

#[inline]
pub fn mul_f32_f16_to_f32_arch(_lhs: &[f32], _rhs: &[f16], _out: &mut [f32]) -> bool {
    if _lhs.len() != _rhs.len() || _lhs.len() != _out.len() {
        return false;
    }

    #[cfg(all(
        feature = "x86-fp-kernels",
        any(target_arch = "x86_64", target_arch = "x86")
    ))]
    {
        if arch::x86_avx512_fp_kernel_runtime_available() {
            unsafe { mul_f32_f16_to_f32_x86_avx512(_lhs, _rhs, _out) };
            return true;
        }
        if arch::x86_f16c_kernel_runtime_available() {
            unsafe { mul_f32_f16_to_f32_x86_avx2(_lhs, _rhs, _out) };
            return true;
        }
    }

    false
}

#[inline]
pub fn add_f32_f16_to_f32_arch(_lhs: &[f32], _rhs: &[f16], _out: &mut [f32]) -> bool {
    f32_f16_to_f32_arch(_lhs, _rhs, false, _out, F32LowpElementwiseOp::Add)
}

#[inline]
pub fn sub_f32_f16_to_f32_arch(
    _lhs: &[f32],
    _rhs: &[f16],
    _lowp_on_lhs: bool,
    _out: &mut [f32],
) -> bool {
    f32_f16_to_f32_arch(_lhs, _rhs, _lowp_on_lhs, _out, F32LowpElementwiseOp::Sub)
}

#[inline]
pub fn mul_f32_bf16_to_f32_arch(_lhs: &[f32], _rhs: &[bf16], _out: &mut [f32]) -> bool {
    if _lhs.len() != _rhs.len() || _lhs.len() != _out.len() {
        return false;
    }

    #[cfg(all(
        feature = "x86-fp-kernels",
        any(target_arch = "x86_64", target_arch = "x86")
    ))]
    {
        if arch::x86_avx512_bf16_kernel_runtime_available() {
            unsafe { mul_f32_bf16_to_f32_x86_avx512(_lhs, _rhs, _out) };
            return true;
        }
        if arch::x86_fp_kernel_runtime_available() {
            unsafe { mul_f32_bf16_to_f32_x86_avx2(_lhs, _rhs, _out) };
            return true;
        }
    }

    false
}

#[derive(Clone, Copy)]
enum F32LowpElementwiseOp {
    Add,
    Sub,
}

#[inline]
fn f32_f16_to_f32_arch(
    _lhs: &[f32],
    _rhs: &[f16],
    _lowp_on_lhs: bool,
    _out: &mut [f32],
    _op: F32LowpElementwiseOp,
) -> bool {
    if _lhs.len() != _rhs.len() || _lhs.len() != _out.len() {
        return false;
    }

    #[cfg(all(
        feature = "x86-fp-kernels",
        any(target_arch = "x86_64", target_arch = "x86")
    ))]
    {
        if arch::x86_avx512_fp_kernel_runtime_available() {
            unsafe { f32_f16_to_f32_x86_avx512(_lhs, _rhs, _lowp_on_lhs, _out, _op) };
            return true;
        }
        if arch::x86_f16c_kernel_runtime_available() {
            unsafe { f32_f16_to_f32_x86_avx2(_lhs, _rhs, _lowp_on_lhs, _out, _op) };
            return true;
        }
    }

    false
}

#[inline]
pub fn add_f32_bf16_to_f32_arch(_lhs: &[f32], _rhs: &[bf16], _out: &mut [f32]) -> bool {
    f32_bf16_to_f32_arch(_lhs, _rhs, false, _out, F32LowpElementwiseOp::Add)
}

#[inline]
pub fn sub_f32_bf16_to_f32_arch(
    _lhs: &[f32],
    _rhs: &[bf16],
    _lowp_on_lhs: bool,
    _out: &mut [f32],
) -> bool {
    f32_bf16_to_f32_arch(_lhs, _rhs, _lowp_on_lhs, _out, F32LowpElementwiseOp::Sub)
}

#[inline]
fn f32_bf16_to_f32_arch(
    _lhs: &[f32],
    _rhs: &[bf16],
    _lowp_on_lhs: bool,
    _out: &mut [f32],
    _op: F32LowpElementwiseOp,
) -> bool {
    if _lhs.len() != _rhs.len() || _lhs.len() != _out.len() {
        return false;
    }

    #[cfg(all(
        feature = "x86-fp-kernels",
        any(target_arch = "x86_64", target_arch = "x86")
    ))]
    {
        if arch::x86_avx512_bf16_kernel_runtime_available() {
            unsafe { f32_bf16_to_f32_x86_avx512(_lhs, _rhs, _lowp_on_lhs, _out, _op) };
            return true;
        }
        if arch::x86_fp_kernel_runtime_available() {
            unsafe { f32_bf16_to_f32_x86_avx2(_lhs, _rhs, _lowp_on_lhs, _out, _op) };
            return true;
        }
    }

    false
}

#[inline]
pub fn mul_f16_scalar_to_f32_arch(_lhs: &[f16], _rhs: f32, _out: &mut [f32]) -> bool {
    if _lhs.len() != _out.len() || !_rhs.is_finite() {
        return false;
    }

    #[cfg(all(
        feature = "x86-fp-kernels",
        any(target_arch = "x86_64", target_arch = "x86")
    ))]
    {
        if arch::x86_avx512_fp_kernel_runtime_available() {
            unsafe { mul_f16_scalar_to_f32_x86_avx512(_lhs, _rhs, _out) };
            return true;
        }
        if arch::x86_f16c_kernel_runtime_available() {
            unsafe { mul_f16_scalar_to_f32_x86_avx2(_lhs, _rhs, _out) };
            return true;
        }
    }

    false
}

#[inline]
pub fn add_f16_scalar_to_f32_arch(_lhs: &[f16], _rhs: f32, _out: &mut [f32]) -> bool {
    f16_scalar_to_f32_arch(_lhs, _rhs, false, _out, F32LowpElementwiseOp::Add)
}

#[inline]
pub fn sub_f16_scalar_to_f32_arch(
    _lhs: &[f16],
    _rhs: f32,
    _lowp_on_lhs: bool,
    _out: &mut [f32],
) -> bool {
    f16_scalar_to_f32_arch(_lhs, _rhs, _lowp_on_lhs, _out, F32LowpElementwiseOp::Sub)
}

#[inline]
fn f16_scalar_to_f32_arch(
    _lhs: &[f16],
    _rhs: f32,
    _lowp_on_lhs: bool,
    _out: &mut [f32],
    _op: F32LowpElementwiseOp,
) -> bool {
    if _lhs.len() != _out.len() || !_rhs.is_finite() {
        return false;
    }

    #[cfg(all(
        feature = "x86-fp-kernels",
        any(target_arch = "x86_64", target_arch = "x86")
    ))]
    {
        if arch::x86_avx512_fp_kernel_runtime_available() {
            unsafe { f16_scalar_to_f32_x86_avx512(_lhs, _rhs, _lowp_on_lhs, _out, _op) };
            return true;
        }
        if arch::x86_f16c_kernel_runtime_available() {
            unsafe { f16_scalar_to_f32_x86_avx2(_lhs, _rhs, _lowp_on_lhs, _out, _op) };
            return true;
        }
    }

    false
}

#[inline]
pub fn mul_bf16_scalar_to_f32_arch(_lhs: &[bf16], _rhs: f32, _out: &mut [f32]) -> bool {
    if _lhs.len() != _out.len() || !_rhs.is_finite() {
        return false;
    }

    #[cfg(all(
        feature = "x86-fp-kernels",
        any(target_arch = "x86_64", target_arch = "x86")
    ))]
    {
        if arch::x86_avx512_bf16_kernel_runtime_available() {
            unsafe { mul_bf16_scalar_to_f32_x86_avx512(_lhs, _rhs, _out) };
            return true;
        }
        if arch::x86_fp_kernel_runtime_available() {
            unsafe { mul_bf16_scalar_to_f32_x86_avx2(_lhs, _rhs, _out) };
            return true;
        }
    }

    false
}

#[inline]
pub fn add_bf16_scalar_to_f32_arch(_lhs: &[bf16], _rhs: f32, _out: &mut [f32]) -> bool {
    bf16_scalar_to_f32_arch(_lhs, _rhs, false, _out, F32LowpElementwiseOp::Add)
}

#[inline]
pub fn sub_bf16_scalar_to_f32_arch(
    _lhs: &[bf16],
    _rhs: f32,
    _lowp_on_lhs: bool,
    _out: &mut [f32],
) -> bool {
    bf16_scalar_to_f32_arch(_lhs, _rhs, _lowp_on_lhs, _out, F32LowpElementwiseOp::Sub)
}

#[inline]
fn bf16_scalar_to_f32_arch(
    _lhs: &[bf16],
    _rhs: f32,
    _lowp_on_lhs: bool,
    _out: &mut [f32],
    _op: F32LowpElementwiseOp,
) -> bool {
    if _lhs.len() != _out.len() || !_rhs.is_finite() {
        return false;
    }

    #[cfg(all(
        feature = "x86-fp-kernels",
        any(target_arch = "x86_64", target_arch = "x86")
    ))]
    {
        if arch::x86_avx512_bf16_kernel_runtime_available() {
            unsafe { bf16_scalar_to_f32_x86_avx512(_lhs, _rhs, _lowp_on_lhs, _out, _op) };
            return true;
        }
        if arch::x86_fp_kernel_runtime_available() {
            unsafe { bf16_scalar_to_f32_x86_avx2(_lhs, _rhs, _lowp_on_lhs, _out, _op) };
            return true;
        }
    }

    false
}

#[inline]
pub fn dot_f32_arch(_x: &[f32], _row: &[f32]) -> Option<f32> {
    if !dot_len_matches(_x, _row) {
        return None;
    }

    match active_float_backend() {
        FloatKernelBackend::Portable => None,
        #[cfg(all(feature = "arm64-fp-kernels", target_arch = "aarch64"))]
        FloatKernelBackend::Arm64Neon => Some(unsafe { dot_f32_arm64_neon(_x, _row) }),
        #[cfg(all(
            feature = "x86-fp-kernels",
            any(target_arch = "x86_64", target_arch = "x86")
        ))]
        FloatKernelBackend::X86Avx512 => Some(unsafe { dot_f32_x86_avx512(_x, _row) }),
        #[cfg(all(
            feature = "x86-fp-kernels",
            any(target_arch = "x86_64", target_arch = "x86")
        ))]
        FloatKernelBackend::X86Avx2 => Some(unsafe { dot_f32_x86_avx2(_x, _row) }),
        #[allow(unreachable_patterns)]
        _ => None,
    }
}

#[inline]
pub fn dot2_f32_arch(_x: &[f32], _row0: &[f32], _row1: &[f32]) -> Option<(f32, f32)> {
    if !dot2_len_matches(_x, _row0, _row1) {
        return None;
    }

    match active_float_backend() {
        FloatKernelBackend::Portable => None,
        #[cfg(all(feature = "arm64-fp-kernels", target_arch = "aarch64"))]
        FloatKernelBackend::Arm64Neon => Some(unsafe { dot2_f32_arm64_neon(_x, _row0, _row1) }),
        #[cfg(all(
            feature = "x86-fp-kernels",
            any(target_arch = "x86_64", target_arch = "x86")
        ))]
        FloatKernelBackend::X86Avx512 => Some(unsafe { dot2_f32_x86_avx512(_x, _row0, _row1) }),
        #[cfg(all(
            feature = "x86-fp-kernels",
            any(target_arch = "x86_64", target_arch = "x86")
        ))]
        FloatKernelBackend::X86Avx2 => Some(unsafe { dot2_f32_x86_avx2(_x, _row0, _row1) }),
        #[allow(unreachable_patterns)]
        _ => None,
    }
}

#[inline]
pub fn dot3_f32_arch(
    _x: &[f32],
    _row0: &[f32],
    _row1: &[f32],
    _row2: &[f32],
) -> Option<(f32, f32, f32)> {
    if !dot3_len_matches(_x, _row0, _row1, _row2) {
        return None;
    }

    match active_float_backend() {
        FloatKernelBackend::Portable => None,
        #[cfg(all(feature = "arm64-fp-kernels", target_arch = "aarch64"))]
        FloatKernelBackend::Arm64Neon => {
            Some(unsafe { dot3_f32_arm64_neon(_x, _row0, _row1, _row2) })
        }
        #[cfg(all(
            feature = "x86-fp-kernels",
            any(target_arch = "x86_64", target_arch = "x86")
        ))]
        FloatKernelBackend::X86Avx512 => {
            Some(unsafe { dot3_f32_x86_avx512(_x, _row0, _row1, _row2) })
        }
        #[cfg(all(
            feature = "x86-fp-kernels",
            any(target_arch = "x86_64", target_arch = "x86")
        ))]
        FloatKernelBackend::X86Avx2 => Some(unsafe { dot3_f32_x86_avx2(_x, _row0, _row1, _row2) }),
        #[allow(unreachable_patterns)]
        _ => None,
    }
}

#[inline]
pub fn dot_f32_f16_arch(_x: &[f32], _row: &[f16]) -> Option<f32> {
    if !dot_len_matches(_x, _row) {
        return None;
    }

    match active_float_backend() {
        FloatKernelBackend::Portable => None,
        #[cfg(all(feature = "arm64-fp-kernels", target_arch = "aarch64"))]
        FloatKernelBackend::Arm64Neon => arch::arm64_fp16_kernel_runtime_available()
            .then(|| unsafe { dot_f32_f16_arm64_neon(_x, _row) }),
        #[cfg(all(
            feature = "x86-fp-kernels",
            any(target_arch = "x86_64", target_arch = "x86")
        ))]
        FloatKernelBackend::X86Avx512 => Some(unsafe { dot_f32_f16_x86_avx512(_x, _row) }),
        #[cfg(all(
            feature = "x86-fp-kernels",
            any(target_arch = "x86_64", target_arch = "x86")
        ))]
        FloatKernelBackend::X86Avx2 => arch::x86_fp16_kernel_runtime_available()
            .then(|| unsafe { dot_f32_f16_x86_avx2(_x, _row) }),
        #[allow(unreachable_patterns)]
        _ => None,
    }
}

#[inline]
pub fn dot2_f32_f16_arch(_x: &[f32], _row0: &[f16], _row1: &[f16]) -> Option<(f32, f32)> {
    if !dot2_len_matches(_x, _row0, _row1) {
        return None;
    }

    match active_float_backend() {
        FloatKernelBackend::Portable => None,
        #[cfg(all(feature = "arm64-fp-kernels", target_arch = "aarch64"))]
        FloatKernelBackend::Arm64Neon => arch::arm64_fp16_kernel_runtime_available()
            .then(|| unsafe { dot2_f32_f16_arm64_neon(_x, _row0, _row1) }),
        #[cfg(all(
            feature = "x86-fp-kernels",
            any(target_arch = "x86_64", target_arch = "x86")
        ))]
        FloatKernelBackend::X86Avx512 => Some(unsafe { dot2_f32_f16_x86_avx512(_x, _row0, _row1) }),
        #[cfg(all(
            feature = "x86-fp-kernels",
            any(target_arch = "x86_64", target_arch = "x86")
        ))]
        FloatKernelBackend::X86Avx2 => arch::x86_fp16_kernel_runtime_available()
            .then(|| unsafe { dot2_f32_f16_x86_avx2(_x, _row0, _row1) }),
        #[allow(unreachable_patterns)]
        _ => None,
    }
}

#[inline]
pub fn dot3_f32_f16_arch(
    _x: &[f32],
    _row0: &[f16],
    _row1: &[f16],
    _row2: &[f16],
) -> Option<(f32, f32, f32)> {
    if !dot3_len_matches(_x, _row0, _row1, _row2) {
        return None;
    }

    match active_float_backend() {
        FloatKernelBackend::Portable => None,
        #[cfg(all(feature = "arm64-fp-kernels", target_arch = "aarch64"))]
        FloatKernelBackend::Arm64Neon => arch::arm64_fp16_kernel_runtime_available()
            .then(|| unsafe { dot3_f32_f16_arm64_neon(_x, _row0, _row1, _row2) }),
        #[cfg(all(
            feature = "x86-fp-kernels",
            any(target_arch = "x86_64", target_arch = "x86")
        ))]
        FloatKernelBackend::X86Avx512 => {
            Some(unsafe { dot3_f32_f16_x86_avx512(_x, _row0, _row1, _row2) })
        }
        #[cfg(all(
            feature = "x86-fp-kernels",
            any(target_arch = "x86_64", target_arch = "x86")
        ))]
        FloatKernelBackend::X86Avx2 => arch::x86_fp16_kernel_runtime_available()
            .then(|| unsafe { dot3_f32_f16_x86_avx2(_x, _row0, _row1, _row2) }),
        #[allow(unreachable_patterns)]
        _ => None,
    }
}

#[inline]
pub fn dot_f32_bf16_arch(_x: &[f32], _row: &[bf16]) -> Option<f32> {
    if !dot_len_matches(_x, _row) {
        return None;
    }

    match active_float_backend() {
        FloatKernelBackend::Portable => None,
        #[cfg(all(feature = "arm64-fp-kernels", target_arch = "aarch64"))]
        FloatKernelBackend::Arm64Neon => Some(unsafe { dot_f32_bf16_arm64_neon(_x, _row) }),
        #[cfg(all(
            feature = "x86-fp-kernels",
            any(target_arch = "x86_64", target_arch = "x86")
        ))]
        FloatKernelBackend::X86Avx512 => Some(unsafe { dot_f32_bf16_x86_avx512(_x, _row) }),
        #[cfg(all(
            feature = "x86-fp-kernels",
            any(target_arch = "x86_64", target_arch = "x86")
        ))]
        FloatKernelBackend::X86Avx2 => Some(unsafe { dot_f32_bf16_x86_avx2(_x, _row) }),
        #[allow(unreachable_patterns)]
        _ => None,
    }
}

#[inline]
pub fn dot2_f32_bf16_arch(_x: &[f32], _row0: &[bf16], _row1: &[bf16]) -> Option<(f32, f32)> {
    if !dot2_len_matches(_x, _row0, _row1) {
        return None;
    }

    match active_float_backend() {
        FloatKernelBackend::Portable => None,
        #[cfg(all(feature = "arm64-fp-kernels", target_arch = "aarch64"))]
        FloatKernelBackend::Arm64Neon => {
            Some(unsafe { dot2_f32_bf16_arm64_neon(_x, _row0, _row1) })
        }
        #[cfg(all(
            feature = "x86-fp-kernels",
            any(target_arch = "x86_64", target_arch = "x86")
        ))]
        FloatKernelBackend::X86Avx512 => {
            Some(unsafe { dot2_f32_bf16_x86_avx512(_x, _row0, _row1) })
        }
        #[cfg(all(
            feature = "x86-fp-kernels",
            any(target_arch = "x86_64", target_arch = "x86")
        ))]
        FloatKernelBackend::X86Avx2 => Some(unsafe { dot2_f32_bf16_x86_avx2(_x, _row0, _row1) }),
        #[allow(unreachable_patterns)]
        _ => None,
    }
}

#[inline]
pub fn dot3_f32_bf16_arch(
    _x: &[f32],
    _row0: &[bf16],
    _row1: &[bf16],
    _row2: &[bf16],
) -> Option<(f32, f32, f32)> {
    if !dot3_len_matches(_x, _row0, _row1, _row2) {
        return None;
    }

    match active_float_backend() {
        FloatKernelBackend::Portable => None,
        #[cfg(all(feature = "arm64-fp-kernels", target_arch = "aarch64"))]
        FloatKernelBackend::Arm64Neon => {
            Some(unsafe { dot3_f32_bf16_arm64_neon(_x, _row0, _row1, _row2) })
        }
        #[cfg(all(
            feature = "x86-fp-kernels",
            any(target_arch = "x86_64", target_arch = "x86")
        ))]
        FloatKernelBackend::X86Avx512 => {
            Some(unsafe { dot3_f32_bf16_x86_avx512(_x, _row0, _row1, _row2) })
        }
        #[cfg(all(
            feature = "x86-fp-kernels",
            any(target_arch = "x86_64", target_arch = "x86")
        ))]
        FloatKernelBackend::X86Avx2 => {
            Some(unsafe { dot3_f32_bf16_x86_avx2(_x, _row0, _row1, _row2) })
        }
        #[allow(unreachable_patterns)]
        _ => None,
    }
}

#[inline]
pub fn dot_bf16_bf16_arch(_x: &[bf16], _row: &[bf16]) -> Option<f32> {
    if !dot_len_matches_same(_x, _row) {
        return None;
    }

    #[cfg(all(
        feature = "x86-fp-kernels",
        any(target_arch = "x86_64", target_arch = "x86")
    ))]
    {
        if arch::x86_avx512_bf16_kernel_runtime_available() {
            return Some(unsafe { dot_bf16_bf16_x86_avx512(_x, _row) });
        }
        if arch::x86_fp_kernel_runtime_available() {
            return Some(unsafe { dot_bf16_bf16_x86_avx2(_x, _row) });
        }
    }

    None
}

#[inline]
pub fn dot2_bf16_bf16_arch(_x: &[bf16], _row0: &[bf16], _row1: &[bf16]) -> Option<(f32, f32)> {
    if !dot2_len_matches_same(_x, _row0, _row1) {
        return None;
    }

    #[cfg(all(
        feature = "x86-fp-kernels",
        any(target_arch = "x86_64", target_arch = "x86")
    ))]
    {
        if arch::x86_avx512_bf16_kernel_runtime_available() {
            return Some(unsafe { dot2_bf16_bf16_x86_avx512(_x, _row0, _row1) });
        }
        if arch::x86_fp_kernel_runtime_available() {
            return Some(unsafe { dot2_bf16_bf16_x86_avx2(_x, _row0, _row1) });
        }
    }

    None
}

#[inline]
pub fn dot3_bf16_bf16_arch(
    _x: &[bf16],
    _row0: &[bf16],
    _row1: &[bf16],
    _row2: &[bf16],
) -> Option<(f32, f32, f32)> {
    if !dot3_len_matches_same(_x, _row0, _row1, _row2) {
        return None;
    }

    #[cfg(all(
        feature = "x86-fp-kernels",
        any(target_arch = "x86_64", target_arch = "x86")
    ))]
    {
        if arch::x86_avx512_bf16_kernel_runtime_available() {
            return Some(unsafe { dot3_bf16_bf16_x86_avx512(_x, _row0, _row1, _row2) });
        }
        if arch::x86_fp_kernel_runtime_available() {
            return Some(unsafe { dot3_bf16_bf16_x86_avx2(_x, _row0, _row1, _row2) });
        }
    }

    None
}

#[inline]
pub fn dot_f16_f16_arch(_x: &[f16], _row: &[f16]) -> Option<f32> {
    if !dot_len_matches_same(_x, _row) {
        return None;
    }

    #[cfg(all(
        feature = "x86-fp-kernels-nightly",
        any(target_arch = "x86_64", target_arch = "x86")
    ))]
    {
        if arch::x86_avx512_fp16_kernel_runtime_available() {
            return Some(unsafe { dot_f16_f16_x86_avx512(_x, _row) });
        }
    }

    #[cfg(all(
        feature = "x86-fp-kernels",
        any(target_arch = "x86_64", target_arch = "x86")
    ))]
    {
        if arch::x86_fp16_kernel_runtime_available() {
            return Some(unsafe { dot_f16_f16_x86_avx2(_x, _row) });
        }
    }

    None
}

#[inline]
pub fn dot2_f16_f16_arch(_x: &[f16], _row0: &[f16], _row1: &[f16]) -> Option<(f32, f32)> {
    if !dot2_len_matches_same(_x, _row0, _row1) {
        return None;
    }

    #[cfg(all(
        feature = "x86-fp-kernels-nightly",
        any(target_arch = "x86_64", target_arch = "x86")
    ))]
    {
        if arch::x86_avx512_fp16_kernel_runtime_available() {
            return Some(unsafe { dot2_f16_f16_x86_avx512(_x, _row0, _row1) });
        }
    }

    #[cfg(all(
        feature = "x86-fp-kernels",
        any(target_arch = "x86_64", target_arch = "x86")
    ))]
    {
        if arch::x86_fp16_kernel_runtime_available() {
            return Some(unsafe { dot2_f16_f16_x86_avx2(_x, _row0, _row1) });
        }
    }

    None
}

#[inline]
pub fn dot3_f16_f16_arch(
    _x: &[f16],
    _row0: &[f16],
    _row1: &[f16],
    _row2: &[f16],
) -> Option<(f32, f32, f32)> {
    if !dot3_len_matches_same(_x, _row0, _row1, _row2) {
        return None;
    }

    #[cfg(all(
        feature = "x86-fp-kernels-nightly",
        any(target_arch = "x86_64", target_arch = "x86")
    ))]
    {
        if arch::x86_avx512_fp16_kernel_runtime_available() {
            return Some(unsafe { dot3_f16_f16_x86_avx512(_x, _row0, _row1, _row2) });
        }
    }

    #[cfg(all(
        feature = "x86-fp-kernels",
        any(target_arch = "x86_64", target_arch = "x86")
    ))]
    {
        if arch::x86_fp16_kernel_runtime_available() {
            return Some(unsafe { dot3_f16_f16_x86_avx2(_x, _row0, _row1, _row2) });
        }
    }

    None
}

#[inline]
pub fn sgd_update_f16_f32_arch(_data: &mut [f16], _grad: &[f32], _lr: f32) -> bool {
    if _data.len() != _grad.len() {
        return false;
    }

    #[cfg(all(
        feature = "x86-fp-kernels",
        any(target_arch = "x86_64", target_arch = "x86")
    ))]
    {
        if arch::x86_avx512_fp_kernel_runtime_available() {
            unsafe { sgd_update_f16_f32_x86_avx512(_data, _grad, _lr) };
            return true;
        }
        if arch::x86_f16c_kernel_runtime_available() {
            unsafe { sgd_update_f16_f32_x86_avx2(_data, _grad, _lr) };
            return true;
        }
    }

    false
}

#[inline]
pub fn sgd_update_bf16_f32_arch(_data: &mut [bf16], _grad: &[f32], _lr: f32) -> bool {
    if _data.len() != _grad.len() {
        return false;
    }

    #[cfg(all(
        feature = "x86-fp-kernels",
        any(target_arch = "x86_64", target_arch = "x86")
    ))]
    {
        if arch::x86_avx512_bf16_kernel_runtime_available() {
            unsafe { sgd_update_bf16_f32_x86_avx512(_data, _grad, _lr) };
            return true;
        }
        if arch::x86_fp_kernel_runtime_available() {
            unsafe { sgd_update_bf16_f32_x86_avx2(_data, _grad, _lr) };
            return true;
        }
    }

    false
}

#[inline]
pub fn sgd_momentum_update_f16_f32_arch(
    _data: &mut [f16],
    _velocity: &mut [f32],
    _grad: &[f32],
    _lr: f32,
    _momentum: f32,
) -> bool {
    if _data.len() != _grad.len() || _data.len() != _velocity.len() {
        return false;
    }

    #[cfg(all(
        feature = "x86-fp-kernels",
        any(target_arch = "x86_64", target_arch = "x86")
    ))]
    {
        if arch::x86_avx512_fp_kernel_runtime_available() {
            unsafe {
                sgd_momentum_update_f16_f32_x86_avx512(_data, _velocity, _grad, _lr, _momentum)
            };
            return true;
        }
        if arch::x86_f16c_kernel_runtime_available() {
            unsafe {
                sgd_momentum_update_f16_f32_x86_avx2(_data, _velocity, _grad, _lr, _momentum)
            };
            return true;
        }
    }

    false
}

#[inline]
pub fn sgd_momentum_update_bf16_f32_arch(
    _data: &mut [bf16],
    _velocity: &mut [f32],
    _grad: &[f32],
    _lr: f32,
    _momentum: f32,
) -> bool {
    if _data.len() != _grad.len() || _data.len() != _velocity.len() {
        return false;
    }

    #[cfg(all(
        feature = "x86-fp-kernels",
        any(target_arch = "x86_64", target_arch = "x86")
    ))]
    {
        if arch::x86_avx512_bf16_kernel_runtime_available() {
            unsafe {
                sgd_momentum_update_bf16_f32_x86_avx512(_data, _velocity, _grad, _lr, _momentum)
            };
            return true;
        }
        if arch::x86_fp_kernel_runtime_available() {
            unsafe {
                sgd_momentum_update_bf16_f32_x86_avx2(_data, _velocity, _grad, _lr, _momentum)
            };
            return true;
        }
    }

    false
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub fn adam_update_f16_f32_arch(
    _data: &mut [f16],
    _exp_avg: &mut [f32],
    _exp_avg_sq: &mut [f32],
    _grad: &[f32],
    _lr: f32,
    _beta1: f32,
    _beta2: f32,
    _bias_correction1: f32,
    _bias_correction2: f32,
    _eps: f32,
) -> bool {
    if _data.len() != _grad.len()
        || _data.len() != _exp_avg.len()
        || _data.len() != _exp_avg_sq.len()
        || !_bias_correction1.is_finite()
        || _bias_correction1 == 0.0
        || !_bias_correction2.is_finite()
        || _bias_correction2 == 0.0
    {
        return false;
    }

    #[cfg(all(
        feature = "x86-fp-kernels",
        any(target_arch = "x86_64", target_arch = "x86")
    ))]
    {
        if arch::x86_avx512_fp_kernel_runtime_available() {
            unsafe {
                adam_update_f16_f32_x86_avx512(
                    _data,
                    _exp_avg,
                    _exp_avg_sq,
                    _grad,
                    _lr,
                    _beta1,
                    _beta2,
                    _bias_correction1,
                    _bias_correction2,
                    _eps,
                )
            };
            return true;
        }
        if arch::x86_f16c_kernel_runtime_available() {
            unsafe {
                adam_update_f16_f32_x86_avx2(
                    _data,
                    _exp_avg,
                    _exp_avg_sq,
                    _grad,
                    _lr,
                    _beta1,
                    _beta2,
                    _bias_correction1,
                    _bias_correction2,
                    _eps,
                )
            };
            return true;
        }
    }

    false
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub fn adam_update_bf16_f32_arch(
    _data: &mut [bf16],
    _exp_avg: &mut [f32],
    _exp_avg_sq: &mut [f32],
    _grad: &[f32],
    _lr: f32,
    _beta1: f32,
    _beta2: f32,
    _bias_correction1: f32,
    _bias_correction2: f32,
    _eps: f32,
) -> bool {
    if _data.len() != _grad.len()
        || _data.len() != _exp_avg.len()
        || _data.len() != _exp_avg_sq.len()
        || !_bias_correction1.is_finite()
        || _bias_correction1 == 0.0
        || !_bias_correction2.is_finite()
        || _bias_correction2 == 0.0
    {
        return false;
    }

    #[cfg(all(
        feature = "x86-fp-kernels",
        any(target_arch = "x86_64", target_arch = "x86")
    ))]
    {
        if arch::x86_avx512_bf16_kernel_runtime_available() {
            unsafe {
                adam_update_bf16_f32_x86_avx512(
                    _data,
                    _exp_avg,
                    _exp_avg_sq,
                    _grad,
                    _lr,
                    _beta1,
                    _beta2,
                    _bias_correction1,
                    _bias_correction2,
                    _eps,
                )
            };
            return true;
        }
        if arch::x86_fp_kernel_runtime_available() {
            unsafe {
                adam_update_bf16_f32_x86_avx2(
                    _data,
                    _exp_avg,
                    _exp_avg_sq,
                    _grad,
                    _lr,
                    _beta1,
                    _beta2,
                    _bias_correction1,
                    _bias_correction2,
                    _eps,
                )
            };
            return true;
        }
    }

    false
}

#[cfg(all(feature = "arm64-fp-kernels", target_arch = "aarch64"))]
#[target_feature(enable = "neon")]
unsafe fn dot_f32_arm64_neon(x: &[f32], row: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    let k_dim = x.len();
    let mut kk = 0usize;
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);

    while kk + 8 <= k_dim {
        let row_lo = unsafe { vld1q_f32(row.as_ptr().add(kk)) };
        let row_hi = unsafe { vld1q_f32(row.as_ptr().add(kk + 4)) };
        let x_lo = unsafe { vld1q_f32(x.as_ptr().add(kk)) };
        let x_hi = unsafe { vld1q_f32(x.as_ptr().add(kk + 4)) };
        acc0 = vfmaq_f32(acc0, row_lo, x_lo);
        acc1 = vfmaq_f32(acc1, row_hi, x_hi);
        kk += 8;
    }

    let mut sum = vaddvq_f32(acc0) + vaddvq_f32(acc1);
    while kk < k_dim {
        sum += row[kk] * x[kk];
        kk += 1;
    }
    sum
}

#[cfg(all(feature = "arm64-fp-kernels", target_arch = "aarch64"))]
#[target_feature(enable = "neon")]
unsafe fn dot2_f32_arm64_neon(x: &[f32], row0: &[f32], row1: &[f32]) -> (f32, f32) {
    use std::arch::aarch64::*;

    let k_dim = x.len();
    let mut kk = 0usize;
    let mut acc00 = vdupq_n_f32(0.0);
    let mut acc01 = vdupq_n_f32(0.0);
    let mut acc10 = vdupq_n_f32(0.0);
    let mut acc11 = vdupq_n_f32(0.0);

    while kk + 8 <= k_dim {
        let x_lo = unsafe { vld1q_f32(x.as_ptr().add(kk)) };
        let x_hi = unsafe { vld1q_f32(x.as_ptr().add(kk + 4)) };
        let row0_lo = unsafe { vld1q_f32(row0.as_ptr().add(kk)) };
        let row0_hi = unsafe { vld1q_f32(row0.as_ptr().add(kk + 4)) };
        let row1_lo = unsafe { vld1q_f32(row1.as_ptr().add(kk)) };
        let row1_hi = unsafe { vld1q_f32(row1.as_ptr().add(kk + 4)) };
        acc00 = vfmaq_f32(acc00, row0_lo, x_lo);
        acc01 = vfmaq_f32(acc01, row0_hi, x_hi);
        acc10 = vfmaq_f32(acc10, row1_lo, x_lo);
        acc11 = vfmaq_f32(acc11, row1_hi, x_hi);
        kk += 8;
    }

    let mut sum0 = vaddvq_f32(acc00) + vaddvq_f32(acc01);
    let mut sum1 = vaddvq_f32(acc10) + vaddvq_f32(acc11);
    while kk < k_dim {
        let xv = x[kk];
        sum0 += row0[kk] * xv;
        sum1 += row1[kk] * xv;
        kk += 1;
    }
    (sum0, sum1)
}

#[cfg(all(feature = "arm64-fp-kernels", target_arch = "aarch64"))]
#[target_feature(enable = "neon")]
unsafe fn dot3_f32_arm64_neon(
    x: &[f32],
    row0: &[f32],
    row1: &[f32],
    row2: &[f32],
) -> (f32, f32, f32) {
    use std::arch::aarch64::*;

    let k_dim = x.len();
    let mut kk = 0usize;
    let mut acc00 = vdupq_n_f32(0.0);
    let mut acc01 = vdupq_n_f32(0.0);
    let mut acc10 = vdupq_n_f32(0.0);
    let mut acc11 = vdupq_n_f32(0.0);
    let mut acc20 = vdupq_n_f32(0.0);
    let mut acc21 = vdupq_n_f32(0.0);

    while kk + 8 <= k_dim {
        let x_lo = unsafe { vld1q_f32(x.as_ptr().add(kk)) };
        let x_hi = unsafe { vld1q_f32(x.as_ptr().add(kk + 4)) };
        let row0_lo = unsafe { vld1q_f32(row0.as_ptr().add(kk)) };
        let row0_hi = unsafe { vld1q_f32(row0.as_ptr().add(kk + 4)) };
        let row1_lo = unsafe { vld1q_f32(row1.as_ptr().add(kk)) };
        let row1_hi = unsafe { vld1q_f32(row1.as_ptr().add(kk + 4)) };
        let row2_lo = unsafe { vld1q_f32(row2.as_ptr().add(kk)) };
        let row2_hi = unsafe { vld1q_f32(row2.as_ptr().add(kk + 4)) };
        acc00 = vfmaq_f32(acc00, row0_lo, x_lo);
        acc01 = vfmaq_f32(acc01, row0_hi, x_hi);
        acc10 = vfmaq_f32(acc10, row1_lo, x_lo);
        acc11 = vfmaq_f32(acc11, row1_hi, x_hi);
        acc20 = vfmaq_f32(acc20, row2_lo, x_lo);
        acc21 = vfmaq_f32(acc21, row2_hi, x_hi);
        kk += 8;
    }

    let mut sum0 = vaddvq_f32(acc00) + vaddvq_f32(acc01);
    let mut sum1 = vaddvq_f32(acc10) + vaddvq_f32(acc11);
    let mut sum2 = vaddvq_f32(acc20) + vaddvq_f32(acc21);
    while kk < k_dim {
        let xv = x[kk];
        sum0 += row0[kk] * xv;
        sum1 += row1[kk] * xv;
        sum2 += row2[kk] * xv;
        kk += 1;
    }
    (sum0, sum1, sum2)
}

#[cfg(all(feature = "arm64-fp-kernels", target_arch = "aarch64"))]
#[target_feature(enable = "neon,fp16")]
unsafe fn dot_f32_f16_arm64_neon(x: &[f32], row: &[f16]) -> f32 {
    use std::arch::aarch64::*;

    let k_dim = x.len();
    let mut kk = 0usize;
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);

    while kk + 8 <= k_dim {
        let row_u16 = unsafe { vld1q_u16(row.as_ptr().add(kk) as *const u16) };
        let row_f16 = vreinterpretq_f16_u16(row_u16);
        let row_lo = vcvt_f32_f16(vget_low_f16(row_f16));
        let row_hi = vcvt_f32_f16(vget_high_f16(row_f16));
        let x_lo = unsafe { vld1q_f32(x.as_ptr().add(kk)) };
        let x_hi = unsafe { vld1q_f32(x.as_ptr().add(kk + 4)) };
        acc0 = vfmaq_f32(acc0, row_lo, x_lo);
        acc1 = vfmaq_f32(acc1, row_hi, x_hi);
        kk += 8;
    }

    let mut sum = vaddvq_f32(acc0) + vaddvq_f32(acc1);
    while kk < k_dim {
        sum += row[kk].to_f32() * x[kk];
        kk += 1;
    }
    sum
}

#[cfg(all(feature = "arm64-fp-kernels", target_arch = "aarch64"))]
#[target_feature(enable = "neon,fp16")]
unsafe fn dot2_f32_f16_arm64_neon(x: &[f32], row0: &[f16], row1: &[f16]) -> (f32, f32) {
    use std::arch::aarch64::*;

    let k_dim = x.len();
    let mut kk = 0usize;
    let mut acc00 = vdupq_n_f32(0.0);
    let mut acc01 = vdupq_n_f32(0.0);
    let mut acc10 = vdupq_n_f32(0.0);
    let mut acc11 = vdupq_n_f32(0.0);

    while kk + 8 <= k_dim {
        let x_lo = unsafe { vld1q_f32(x.as_ptr().add(kk)) };
        let x_hi = unsafe { vld1q_f32(x.as_ptr().add(kk + 4)) };
        let row0_u16 = unsafe { vld1q_u16(row0.as_ptr().add(kk) as *const u16) };
        let row1_u16 = unsafe { vld1q_u16(row1.as_ptr().add(kk) as *const u16) };
        let row0_f16 = vreinterpretq_f16_u16(row0_u16);
        let row1_f16 = vreinterpretq_f16_u16(row1_u16);
        let row0_lo = vcvt_f32_f16(vget_low_f16(row0_f16));
        let row0_hi = vcvt_f32_f16(vget_high_f16(row0_f16));
        let row1_lo = vcvt_f32_f16(vget_low_f16(row1_f16));
        let row1_hi = vcvt_f32_f16(vget_high_f16(row1_f16));
        acc00 = vfmaq_f32(acc00, row0_lo, x_lo);
        acc01 = vfmaq_f32(acc01, row0_hi, x_hi);
        acc10 = vfmaq_f32(acc10, row1_lo, x_lo);
        acc11 = vfmaq_f32(acc11, row1_hi, x_hi);
        kk += 8;
    }

    let mut sum0 = vaddvq_f32(acc00) + vaddvq_f32(acc01);
    let mut sum1 = vaddvq_f32(acc10) + vaddvq_f32(acc11);
    while kk < k_dim {
        let xv = x[kk];
        sum0 += row0[kk].to_f32() * xv;
        sum1 += row1[kk].to_f32() * xv;
        kk += 1;
    }
    (sum0, sum1)
}

#[cfg(all(feature = "arm64-fp-kernels", target_arch = "aarch64"))]
#[target_feature(enable = "neon,fp16")]
unsafe fn dot3_f32_f16_arm64_neon(
    x: &[f32],
    row0: &[f16],
    row1: &[f16],
    row2: &[f16],
) -> (f32, f32, f32) {
    use std::arch::aarch64::*;

    let k_dim = x.len();
    let mut kk = 0usize;
    let mut acc00 = vdupq_n_f32(0.0);
    let mut acc01 = vdupq_n_f32(0.0);
    let mut acc10 = vdupq_n_f32(0.0);
    let mut acc11 = vdupq_n_f32(0.0);
    let mut acc20 = vdupq_n_f32(0.0);
    let mut acc21 = vdupq_n_f32(0.0);

    while kk + 8 <= k_dim {
        let x_lo = unsafe { vld1q_f32(x.as_ptr().add(kk)) };
        let x_hi = unsafe { vld1q_f32(x.as_ptr().add(kk + 4)) };
        let row0_u16 = unsafe { vld1q_u16(row0.as_ptr().add(kk) as *const u16) };
        let row1_u16 = unsafe { vld1q_u16(row1.as_ptr().add(kk) as *const u16) };
        let row2_u16 = unsafe { vld1q_u16(row2.as_ptr().add(kk) as *const u16) };
        let row0_f16 = vreinterpretq_f16_u16(row0_u16);
        let row1_f16 = vreinterpretq_f16_u16(row1_u16);
        let row2_f16 = vreinterpretq_f16_u16(row2_u16);
        let row0_lo = vcvt_f32_f16(vget_low_f16(row0_f16));
        let row0_hi = vcvt_f32_f16(vget_high_f16(row0_f16));
        let row1_lo = vcvt_f32_f16(vget_low_f16(row1_f16));
        let row1_hi = vcvt_f32_f16(vget_high_f16(row1_f16));
        let row2_lo = vcvt_f32_f16(vget_low_f16(row2_f16));
        let row2_hi = vcvt_f32_f16(vget_high_f16(row2_f16));
        acc00 = vfmaq_f32(acc00, row0_lo, x_lo);
        acc01 = vfmaq_f32(acc01, row0_hi, x_hi);
        acc10 = vfmaq_f32(acc10, row1_lo, x_lo);
        acc11 = vfmaq_f32(acc11, row1_hi, x_hi);
        acc20 = vfmaq_f32(acc20, row2_lo, x_lo);
        acc21 = vfmaq_f32(acc21, row2_hi, x_hi);
        kk += 8;
    }

    let mut sum0 = vaddvq_f32(acc00) + vaddvq_f32(acc01);
    let mut sum1 = vaddvq_f32(acc10) + vaddvq_f32(acc11);
    let mut sum2 = vaddvq_f32(acc20) + vaddvq_f32(acc21);
    while kk < k_dim {
        let xv = x[kk];
        sum0 += row0[kk].to_f32() * xv;
        sum1 += row1[kk].to_f32() * xv;
        sum2 += row2[kk].to_f32() * xv;
        kk += 1;
    }
    (sum0, sum1, sum2)
}

#[cfg(all(feature = "arm64-fp-kernels", target_arch = "aarch64"))]
#[target_feature(enable = "neon")]
unsafe fn dot_f32_bf16_arm64_neon(x: &[f32], row: &[bf16]) -> f32 {
    use std::arch::aarch64::*;

    let k_dim = x.len();
    let mut kk = 0usize;
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);

    while kk + 8 <= k_dim {
        let row_lo_u16 = unsafe { vld1_u16(row.as_ptr().add(kk) as *const u16) };
        let row_hi_u16 = unsafe { vld1_u16(row.as_ptr().add(kk + 4) as *const u16) };
        let row_lo_u32 = vshlq_n_u32(vmovl_u16(row_lo_u16), 16);
        let row_hi_u32 = vshlq_n_u32(vmovl_u16(row_hi_u16), 16);
        let row_lo = vreinterpretq_f32_u32(row_lo_u32);
        let row_hi = vreinterpretq_f32_u32(row_hi_u32);
        let x_lo = unsafe { vld1q_f32(x.as_ptr().add(kk)) };
        let x_hi = unsafe { vld1q_f32(x.as_ptr().add(kk + 4)) };
        acc0 = vfmaq_f32(acc0, row_lo, x_lo);
        acc1 = vfmaq_f32(acc1, row_hi, x_hi);
        kk += 8;
    }

    let mut sum = vaddvq_f32(acc0) + vaddvq_f32(acc1);
    while kk < k_dim {
        sum += row[kk].to_f32() * x[kk];
        kk += 1;
    }
    sum
}

#[cfg(all(feature = "arm64-fp-kernels", target_arch = "aarch64"))]
#[target_feature(enable = "neon")]
unsafe fn dot2_f32_bf16_arm64_neon(x: &[f32], row0: &[bf16], row1: &[bf16]) -> (f32, f32) {
    use std::arch::aarch64::*;

    let k_dim = x.len();
    let mut kk = 0usize;
    let mut acc00 = vdupq_n_f32(0.0);
    let mut acc01 = vdupq_n_f32(0.0);
    let mut acc10 = vdupq_n_f32(0.0);
    let mut acc11 = vdupq_n_f32(0.0);

    while kk + 8 <= k_dim {
        let x_lo = unsafe { vld1q_f32(x.as_ptr().add(kk)) };
        let x_hi = unsafe { vld1q_f32(x.as_ptr().add(kk + 4)) };
        let row0_lo_u16 = unsafe { vld1_u16(row0.as_ptr().add(kk) as *const u16) };
        let row0_hi_u16 = unsafe { vld1_u16(row0.as_ptr().add(kk + 4) as *const u16) };
        let row1_lo_u16 = unsafe { vld1_u16(row1.as_ptr().add(kk) as *const u16) };
        let row1_hi_u16 = unsafe { vld1_u16(row1.as_ptr().add(kk + 4) as *const u16) };
        let row0_lo = vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(row0_lo_u16), 16));
        let row0_hi = vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(row0_hi_u16), 16));
        let row1_lo = vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(row1_lo_u16), 16));
        let row1_hi = vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(row1_hi_u16), 16));
        acc00 = vfmaq_f32(acc00, row0_lo, x_lo);
        acc01 = vfmaq_f32(acc01, row0_hi, x_hi);
        acc10 = vfmaq_f32(acc10, row1_lo, x_lo);
        acc11 = vfmaq_f32(acc11, row1_hi, x_hi);
        kk += 8;
    }

    let mut sum0 = vaddvq_f32(acc00) + vaddvq_f32(acc01);
    let mut sum1 = vaddvq_f32(acc10) + vaddvq_f32(acc11);
    while kk < k_dim {
        let xv = x[kk];
        sum0 += row0[kk].to_f32() * xv;
        sum1 += row1[kk].to_f32() * xv;
        kk += 1;
    }
    (sum0, sum1)
}

#[cfg(all(feature = "arm64-fp-kernels", target_arch = "aarch64"))]
#[target_feature(enable = "neon")]
unsafe fn dot3_f32_bf16_arm64_neon(
    x: &[f32],
    row0: &[bf16],
    row1: &[bf16],
    row2: &[bf16],
) -> (f32, f32, f32) {
    use std::arch::aarch64::*;

    let k_dim = x.len();
    let mut kk = 0usize;
    let mut acc00 = vdupq_n_f32(0.0);
    let mut acc01 = vdupq_n_f32(0.0);
    let mut acc10 = vdupq_n_f32(0.0);
    let mut acc11 = vdupq_n_f32(0.0);
    let mut acc20 = vdupq_n_f32(0.0);
    let mut acc21 = vdupq_n_f32(0.0);

    while kk + 8 <= k_dim {
        let x_lo = unsafe { vld1q_f32(x.as_ptr().add(kk)) };
        let x_hi = unsafe { vld1q_f32(x.as_ptr().add(kk + 4)) };
        let row0_lo_u16 = unsafe { vld1_u16(row0.as_ptr().add(kk) as *const u16) };
        let row0_hi_u16 = unsafe { vld1_u16(row0.as_ptr().add(kk + 4) as *const u16) };
        let row1_lo_u16 = unsafe { vld1_u16(row1.as_ptr().add(kk) as *const u16) };
        let row1_hi_u16 = unsafe { vld1_u16(row1.as_ptr().add(kk + 4) as *const u16) };
        let row2_lo_u16 = unsafe { vld1_u16(row2.as_ptr().add(kk) as *const u16) };
        let row2_hi_u16 = unsafe { vld1_u16(row2.as_ptr().add(kk + 4) as *const u16) };
        let row0_lo = vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(row0_lo_u16), 16));
        let row0_hi = vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(row0_hi_u16), 16));
        let row1_lo = vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(row1_lo_u16), 16));
        let row1_hi = vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(row1_hi_u16), 16));
        let row2_lo = vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(row2_lo_u16), 16));
        let row2_hi = vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(row2_hi_u16), 16));
        acc00 = vfmaq_f32(acc00, row0_lo, x_lo);
        acc01 = vfmaq_f32(acc01, row0_hi, x_hi);
        acc10 = vfmaq_f32(acc10, row1_lo, x_lo);
        acc11 = vfmaq_f32(acc11, row1_hi, x_hi);
        acc20 = vfmaq_f32(acc20, row2_lo, x_lo);
        acc21 = vfmaq_f32(acc21, row2_hi, x_hi);
        kk += 8;
    }

    let mut sum0 = vaddvq_f32(acc00) + vaddvq_f32(acc01);
    let mut sum1 = vaddvq_f32(acc10) + vaddvq_f32(acc11);
    let mut sum2 = vaddvq_f32(acc20) + vaddvq_f32(acc21);
    while kk < k_dim {
        let xv = x[kk];
        sum0 += row0[kk].to_f32() * xv;
        sum1 += row1[kk].to_f32() * xv;
        sum2 += row2[kk].to_f32() * xv;
        kk += 1;
    }
    (sum0, sum1, sum2)
}

#[cfg(all(
    feature = "x86-fp-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn reduce_f32x16_x86(v: __m512) -> f32 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let mut buf = [0.0f32; 16];
    unsafe {
        _mm512_storeu_ps(buf.as_mut_ptr(), v);
    }
    buf.iter().sum()
}

#[cfg(all(
    feature = "x86-fp-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn load_f16_as_f32x16_x86(ptr: *const f16) -> __m512 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let raw = unsafe { std::ptr::read_unaligned(ptr as *const __m256i) };
    _mm512_cvtph_ps(raw)
}

#[cfg(all(
    feature = "x86-fp-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn load_bf16_as_f32x16_x86(ptr: *const bf16) -> __m512 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    // BF16 is the upper sixteen bits of F32. Expand weights exactly, keeping
    // the original F32 activation values for the following F32 FMA.
    let raw = unsafe { std::ptr::read_unaligned(ptr.cast::<__m256i>()) };
    _mm512_castsi512_ps(_mm512_slli_epi32::<16>(_mm512_cvtepu16_epi32(raw)))
}

#[cfg(all(
    feature = "x86-fp-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[inline]
#[target_feature(enable = "avx512f,avx512bf16,avx512vl")]
unsafe fn load_bf16_x16_x86(ptr: *const bf16) -> __m256bh {
    let raw = unsafe { std::ptr::read_unaligned(ptr as *const __m256i) };
    unsafe { std::mem::transmute::<__m256i, __m256bh>(raw) }
}

#[cfg(all(
    feature = "x86-fp-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[inline]
#[target_feature(enable = "avx512f,avx512bf16")]
unsafe fn convert_f32_to_bf16x16_x86(v: __m512) -> __m256bh {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    _mm512_cvtneps_pbh(v)
}

#[cfg(all(
    feature = "x86-fp-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[inline]
#[target_feature(enable = "avx512f,avx512bf16,avx512vl")]
unsafe fn store_bf16_x16_x86(ptr: *mut bf16, v: __m256bh) {
    let raw = unsafe { std::mem::transmute::<__m256bh, __m256i>(v) };
    unsafe { std::ptr::write_unaligned(ptr as *mut __m256i, raw) };
}

#[cfg(all(
    feature = "x86-fp-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn store_f16_from_f32x16_x86(ptr: *mut f16, v: __m512) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let raw = _mm512_cvtps_ph::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(v);
    unsafe { std::ptr::write_unaligned(ptr as *mut __m256i, raw) };
}

#[cfg(all(
    feature = "x86-fp-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn apply_f32_lowp_op_x86_avx512(
    lhs: __m512,
    rhs: __m512,
    lowp_on_lhs: bool,
    op: F32LowpElementwiseOp,
) -> __m512 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    match op {
        F32LowpElementwiseOp::Add => _mm512_add_ps(lhs, rhs),
        F32LowpElementwiseOp::Sub if lowp_on_lhs => _mm512_sub_ps(rhs, lhs),
        F32LowpElementwiseOp::Sub => _mm512_sub_ps(lhs, rhs),
    }
}

#[cfg(all(
    feature = "x86-fp-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[target_feature(enable = "avx512f")]
unsafe fn f32_f16_to_f32_x86_avx512(
    lhs: &[f32],
    rhs: &[f16],
    lowp_on_lhs: bool,
    out: &mut [f32],
    op: F32LowpElementwiseOp,
) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let len = lhs.len();
    let mut kk = 0usize;
    while kk + 32 <= len {
        let a0 = unsafe { _mm512_loadu_ps(lhs.as_ptr().add(kk)) };
        let a1 = unsafe { _mm512_loadu_ps(lhs.as_ptr().add(kk + 16)) };
        let b0 = unsafe { load_f16_as_f32x16_x86(rhs.as_ptr().add(kk)) };
        let b1 = unsafe { load_f16_as_f32x16_x86(rhs.as_ptr().add(kk + 16)) };
        let y0 = unsafe { apply_f32_lowp_op_x86_avx512(a0, b0, lowp_on_lhs, op) };
        let y1 = unsafe { apply_f32_lowp_op_x86_avx512(a1, b1, lowp_on_lhs, op) };
        unsafe { _mm512_storeu_ps(out.as_mut_ptr().add(kk), y0) };
        unsafe { _mm512_storeu_ps(out.as_mut_ptr().add(kk + 16), y1) };
        kk += 32;
    }
    while kk + 16 <= len {
        let a = unsafe { _mm512_loadu_ps(lhs.as_ptr().add(kk)) };
        let b = unsafe { load_f16_as_f32x16_x86(rhs.as_ptr().add(kk)) };
        let y = unsafe { apply_f32_lowp_op_x86_avx512(a, b, lowp_on_lhs, op) };
        unsafe { _mm512_storeu_ps(out.as_mut_ptr().add(kk), y) };
        kk += 16;
    }
    while kk < len {
        out[kk] = match op {
            F32LowpElementwiseOp::Add => lhs[kk] + rhs[kk].to_f32(),
            F32LowpElementwiseOp::Sub if lowp_on_lhs => rhs[kk].to_f32() - lhs[kk],
            F32LowpElementwiseOp::Sub => lhs[kk] - rhs[kk].to_f32(),
        };
        kk += 1;
    }
}

#[cfg(all(
    feature = "x86-fp-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[target_feature(enable = "avx512f,avx512bf16,avx512vl")]
unsafe fn f32_bf16_to_f32_x86_avx512(
    lhs: &[f32],
    rhs: &[bf16],
    lowp_on_lhs: bool,
    out: &mut [f32],
    op: F32LowpElementwiseOp,
) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let len = lhs.len();
    let mut kk = 0usize;
    while kk + 32 <= len {
        let a0 = unsafe { _mm512_loadu_ps(lhs.as_ptr().add(kk)) };
        let a1 = unsafe { _mm512_loadu_ps(lhs.as_ptr().add(kk + 16)) };
        let b0 = unsafe { _mm512_cvtpbh_ps(load_bf16_x16_x86(rhs.as_ptr().add(kk))) };
        let b1 = unsafe { _mm512_cvtpbh_ps(load_bf16_x16_x86(rhs.as_ptr().add(kk + 16))) };
        let y0 = unsafe { apply_f32_lowp_op_x86_avx512(a0, b0, lowp_on_lhs, op) };
        let y1 = unsafe { apply_f32_lowp_op_x86_avx512(a1, b1, lowp_on_lhs, op) };
        unsafe { _mm512_storeu_ps(out.as_mut_ptr().add(kk), y0) };
        unsafe { _mm512_storeu_ps(out.as_mut_ptr().add(kk + 16), y1) };
        kk += 32;
    }
    while kk + 16 <= len {
        let a = unsafe { _mm512_loadu_ps(lhs.as_ptr().add(kk)) };
        let b = unsafe { _mm512_cvtpbh_ps(load_bf16_x16_x86(rhs.as_ptr().add(kk))) };
        let y = unsafe { apply_f32_lowp_op_x86_avx512(a, b, lowp_on_lhs, op) };
        unsafe { _mm512_storeu_ps(out.as_mut_ptr().add(kk), y) };
        kk += 16;
    }
    while kk < len {
        out[kk] = match op {
            F32LowpElementwiseOp::Add => lhs[kk] + rhs[kk].to_f32(),
            F32LowpElementwiseOp::Sub if lowp_on_lhs => rhs[kk].to_f32() - lhs[kk],
            F32LowpElementwiseOp::Sub => lhs[kk] - rhs[kk].to_f32(),
        };
        kk += 1;
    }
}

#[cfg(all(
    feature = "x86-fp-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[target_feature(enable = "avx512f")]
unsafe fn mul_f32_f16_to_f32_x86_avx512(lhs: &[f32], rhs: &[f16], out: &mut [f32]) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let len = lhs.len();
    let mut kk = 0usize;
    while kk + 32 <= len {
        let a0 = unsafe { _mm512_loadu_ps(lhs.as_ptr().add(kk)) };
        let a1 = unsafe { _mm512_loadu_ps(lhs.as_ptr().add(kk + 16)) };
        let b0 = unsafe { load_f16_as_f32x16_x86(rhs.as_ptr().add(kk)) };
        let b1 = unsafe { load_f16_as_f32x16_x86(rhs.as_ptr().add(kk + 16)) };
        unsafe { _mm512_storeu_ps(out.as_mut_ptr().add(kk), _mm512_mul_ps(a0, b0)) };
        unsafe { _mm512_storeu_ps(out.as_mut_ptr().add(kk + 16), _mm512_mul_ps(a1, b1)) };
        kk += 32;
    }
    while kk + 16 <= len {
        let a = unsafe { _mm512_loadu_ps(lhs.as_ptr().add(kk)) };
        let b = unsafe { load_f16_as_f32x16_x86(rhs.as_ptr().add(kk)) };
        unsafe { _mm512_storeu_ps(out.as_mut_ptr().add(kk), _mm512_mul_ps(a, b)) };
        kk += 16;
    }
    while kk < len {
        out[kk] = lhs[kk] * rhs[kk].to_f32();
        kk += 1;
    }
}

#[cfg(all(
    feature = "x86-fp-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[target_feature(enable = "avx512f,avx512bf16,avx512vl")]
unsafe fn mul_f32_bf16_to_f32_x86_avx512(lhs: &[f32], rhs: &[bf16], out: &mut [f32]) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let len = lhs.len();
    let mut kk = 0usize;
    while kk + 32 <= len {
        let a0 = unsafe { _mm512_loadu_ps(lhs.as_ptr().add(kk)) };
        let a1 = unsafe { _mm512_loadu_ps(lhs.as_ptr().add(kk + 16)) };
        let b0 = unsafe { _mm512_cvtpbh_ps(load_bf16_x16_x86(rhs.as_ptr().add(kk))) };
        let b1 = unsafe { _mm512_cvtpbh_ps(load_bf16_x16_x86(rhs.as_ptr().add(kk + 16))) };
        unsafe { _mm512_storeu_ps(out.as_mut_ptr().add(kk), _mm512_mul_ps(a0, b0)) };
        unsafe { _mm512_storeu_ps(out.as_mut_ptr().add(kk + 16), _mm512_mul_ps(a1, b1)) };
        kk += 32;
    }
    while kk + 16 <= len {
        let a = unsafe { _mm512_loadu_ps(lhs.as_ptr().add(kk)) };
        let b = unsafe { _mm512_cvtpbh_ps(load_bf16_x16_x86(rhs.as_ptr().add(kk))) };
        unsafe { _mm512_storeu_ps(out.as_mut_ptr().add(kk), _mm512_mul_ps(a, b)) };
        kk += 16;
    }
    while kk < len {
        out[kk] = lhs[kk] * rhs[kk].to_f32();
        kk += 1;
    }
}

#[cfg(all(
    feature = "x86-fp-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[target_feature(enable = "avx512f")]
unsafe fn f16_scalar_to_f32_x86_avx512(
    lhs: &[f16],
    rhs: f32,
    lowp_on_lhs: bool,
    out: &mut [f32],
    op: F32LowpElementwiseOp,
) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let len = lhs.len();
    let rhs_v = _mm512_set1_ps(rhs);
    let mut kk = 0usize;
    while kk + 32 <= len {
        let a0 = unsafe { load_f16_as_f32x16_x86(lhs.as_ptr().add(kk)) };
        let a1 = unsafe { load_f16_as_f32x16_x86(lhs.as_ptr().add(kk + 16)) };
        let y0 = unsafe { apply_f32_lowp_op_x86_avx512(rhs_v, a0, lowp_on_lhs, op) };
        let y1 = unsafe { apply_f32_lowp_op_x86_avx512(rhs_v, a1, lowp_on_lhs, op) };
        unsafe { _mm512_storeu_ps(out.as_mut_ptr().add(kk), y0) };
        unsafe { _mm512_storeu_ps(out.as_mut_ptr().add(kk + 16), y1) };
        kk += 32;
    }
    while kk + 16 <= len {
        let a = unsafe { load_f16_as_f32x16_x86(lhs.as_ptr().add(kk)) };
        let y = unsafe { apply_f32_lowp_op_x86_avx512(rhs_v, a, lowp_on_lhs, op) };
        unsafe { _mm512_storeu_ps(out.as_mut_ptr().add(kk), y) };
        kk += 16;
    }
    while kk < len {
        out[kk] = match op {
            F32LowpElementwiseOp::Add => rhs + lhs[kk].to_f32(),
            F32LowpElementwiseOp::Sub if lowp_on_lhs => lhs[kk].to_f32() - rhs,
            F32LowpElementwiseOp::Sub => rhs - lhs[kk].to_f32(),
        };
        kk += 1;
    }
}

#[cfg(all(
    feature = "x86-fp-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[target_feature(enable = "avx512f,avx512bf16,avx512vl")]
unsafe fn bf16_scalar_to_f32_x86_avx512(
    lhs: &[bf16],
    rhs: f32,
    lowp_on_lhs: bool,
    out: &mut [f32],
    op: F32LowpElementwiseOp,
) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let len = lhs.len();
    let rhs_v = _mm512_set1_ps(rhs);
    let mut kk = 0usize;
    while kk + 32 <= len {
        let a0 = unsafe { _mm512_cvtpbh_ps(load_bf16_x16_x86(lhs.as_ptr().add(kk))) };
        let a1 = unsafe { _mm512_cvtpbh_ps(load_bf16_x16_x86(lhs.as_ptr().add(kk + 16))) };
        let y0 = unsafe { apply_f32_lowp_op_x86_avx512(rhs_v, a0, lowp_on_lhs, op) };
        let y1 = unsafe { apply_f32_lowp_op_x86_avx512(rhs_v, a1, lowp_on_lhs, op) };
        unsafe { _mm512_storeu_ps(out.as_mut_ptr().add(kk), y0) };
        unsafe { _mm512_storeu_ps(out.as_mut_ptr().add(kk + 16), y1) };
        kk += 32;
    }
    while kk + 16 <= len {
        let a = unsafe { _mm512_cvtpbh_ps(load_bf16_x16_x86(lhs.as_ptr().add(kk))) };
        let y = unsafe { apply_f32_lowp_op_x86_avx512(rhs_v, a, lowp_on_lhs, op) };
        unsafe { _mm512_storeu_ps(out.as_mut_ptr().add(kk), y) };
        kk += 16;
    }
    while kk < len {
        out[kk] = match op {
            F32LowpElementwiseOp::Add => rhs + lhs[kk].to_f32(),
            F32LowpElementwiseOp::Sub if lowp_on_lhs => lhs[kk].to_f32() - rhs,
            F32LowpElementwiseOp::Sub => rhs - lhs[kk].to_f32(),
        };
        kk += 1;
    }
}

#[cfg(all(
    feature = "x86-fp-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[target_feature(enable = "avx512f")]
unsafe fn mul_f16_scalar_to_f32_x86_avx512(lhs: &[f16], rhs: f32, out: &mut [f32]) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let len = lhs.len();
    let rhs_v = _mm512_set1_ps(rhs);
    let mut kk = 0usize;
    while kk + 32 <= len {
        let a0 = unsafe { load_f16_as_f32x16_x86(lhs.as_ptr().add(kk)) };
        let a1 = unsafe { load_f16_as_f32x16_x86(lhs.as_ptr().add(kk + 16)) };
        unsafe { _mm512_storeu_ps(out.as_mut_ptr().add(kk), _mm512_mul_ps(a0, rhs_v)) };
        unsafe { _mm512_storeu_ps(out.as_mut_ptr().add(kk + 16), _mm512_mul_ps(a1, rhs_v)) };
        kk += 32;
    }
    while kk + 16 <= len {
        let a = unsafe { load_f16_as_f32x16_x86(lhs.as_ptr().add(kk)) };
        unsafe { _mm512_storeu_ps(out.as_mut_ptr().add(kk), _mm512_mul_ps(a, rhs_v)) };
        kk += 16;
    }
    while kk < len {
        out[kk] = lhs[kk].to_f32() * rhs;
        kk += 1;
    }
}

#[cfg(all(
    feature = "x86-fp-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[target_feature(enable = "avx512f,avx512bf16,avx512vl")]
unsafe fn mul_bf16_scalar_to_f32_x86_avx512(lhs: &[bf16], rhs: f32, out: &mut [f32]) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let len = lhs.len();
    let rhs_v = _mm512_set1_ps(rhs);
    let mut kk = 0usize;
    while kk + 32 <= len {
        let a0 = unsafe { _mm512_cvtpbh_ps(load_bf16_x16_x86(lhs.as_ptr().add(kk))) };
        let a1 = unsafe { _mm512_cvtpbh_ps(load_bf16_x16_x86(lhs.as_ptr().add(kk + 16))) };
        unsafe { _mm512_storeu_ps(out.as_mut_ptr().add(kk), _mm512_mul_ps(a0, rhs_v)) };
        unsafe { _mm512_storeu_ps(out.as_mut_ptr().add(kk + 16), _mm512_mul_ps(a1, rhs_v)) };
        kk += 32;
    }
    while kk + 16 <= len {
        let a = unsafe { _mm512_cvtpbh_ps(load_bf16_x16_x86(lhs.as_ptr().add(kk))) };
        unsafe { _mm512_storeu_ps(out.as_mut_ptr().add(kk), _mm512_mul_ps(a, rhs_v)) };
        kk += 16;
    }
    while kk < len {
        out[kk] = lhs[kk].to_f32() * rhs;
        kk += 1;
    }
}

#[cfg(all(
    feature = "x86-fp-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[target_feature(enable = "avx512f")]
unsafe fn elementwise_f16_f16_to_f32_x86_avx512(
    lhs: &[f16],
    rhs: &[f16],
    out: &mut [f32],
    op: FpElementwiseOp,
) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let len = lhs.len();
    let mut kk = 0usize;
    while kk + 32 <= len {
        let a0 = unsafe { load_f16_as_f32x16_x86(lhs.as_ptr().add(kk)) };
        let a1 = unsafe { load_f16_as_f32x16_x86(lhs.as_ptr().add(kk + 16)) };
        let b0 = unsafe { load_f16_as_f32x16_x86(rhs.as_ptr().add(kk)) };
        let b1 = unsafe { load_f16_as_f32x16_x86(rhs.as_ptr().add(kk + 16)) };
        let y0 = match op {
            FpElementwiseOp::Add => _mm512_add_ps(a0, b0),
            FpElementwiseOp::Sub => _mm512_sub_ps(a0, b0),
            FpElementwiseOp::Mul => _mm512_mul_ps(a0, b0),
        };
        let y1 = match op {
            FpElementwiseOp::Add => _mm512_add_ps(a1, b1),
            FpElementwiseOp::Sub => _mm512_sub_ps(a1, b1),
            FpElementwiseOp::Mul => _mm512_mul_ps(a1, b1),
        };
        unsafe { _mm512_storeu_ps(out.as_mut_ptr().add(kk), y0) };
        unsafe { _mm512_storeu_ps(out.as_mut_ptr().add(kk + 16), y1) };
        kk += 32;
    }
    while kk + 16 <= len {
        let a = unsafe { load_f16_as_f32x16_x86(lhs.as_ptr().add(kk)) };
        let b = unsafe { load_f16_as_f32x16_x86(rhs.as_ptr().add(kk)) };
        let y = match op {
            FpElementwiseOp::Add => _mm512_add_ps(a, b),
            FpElementwiseOp::Sub => _mm512_sub_ps(a, b),
            FpElementwiseOp::Mul => _mm512_mul_ps(a, b),
        };
        unsafe { _mm512_storeu_ps(out.as_mut_ptr().add(kk), y) };
        kk += 16;
    }
    while kk < len {
        out[kk] = match op {
            FpElementwiseOp::Add => lhs[kk].to_f32() + rhs[kk].to_f32(),
            FpElementwiseOp::Sub => lhs[kk].to_f32() - rhs[kk].to_f32(),
            FpElementwiseOp::Mul => lhs[kk].to_f32() * rhs[kk].to_f32(),
        };
        kk += 1;
    }
}

#[cfg(all(
    feature = "x86-fp-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[target_feature(enable = "avx512f,avx512bf16,avx512vl")]
unsafe fn elementwise_bf16_bf16_to_f32_x86_avx512(
    lhs: &[bf16],
    rhs: &[bf16],
    out: &mut [f32],
    op: FpElementwiseOp,
) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let len = lhs.len();
    let mut kk = 0usize;
    while kk + 32 <= len {
        let a0 = unsafe { _mm512_cvtpbh_ps(load_bf16_x16_x86(lhs.as_ptr().add(kk))) };
        let a1 = unsafe { _mm512_cvtpbh_ps(load_bf16_x16_x86(lhs.as_ptr().add(kk + 16))) };
        let b0 = unsafe { _mm512_cvtpbh_ps(load_bf16_x16_x86(rhs.as_ptr().add(kk))) };
        let b1 = unsafe { _mm512_cvtpbh_ps(load_bf16_x16_x86(rhs.as_ptr().add(kk + 16))) };
        let y0 = match op {
            FpElementwiseOp::Add => _mm512_add_ps(a0, b0),
            FpElementwiseOp::Sub => _mm512_sub_ps(a0, b0),
            FpElementwiseOp::Mul => _mm512_mul_ps(a0, b0),
        };
        let y1 = match op {
            FpElementwiseOp::Add => _mm512_add_ps(a1, b1),
            FpElementwiseOp::Sub => _mm512_sub_ps(a1, b1),
            FpElementwiseOp::Mul => _mm512_mul_ps(a1, b1),
        };
        unsafe { _mm512_storeu_ps(out.as_mut_ptr().add(kk), y0) };
        unsafe { _mm512_storeu_ps(out.as_mut_ptr().add(kk + 16), y1) };
        kk += 32;
    }
    while kk + 16 <= len {
        let a = unsafe { _mm512_cvtpbh_ps(load_bf16_x16_x86(lhs.as_ptr().add(kk))) };
        let b = unsafe { _mm512_cvtpbh_ps(load_bf16_x16_x86(rhs.as_ptr().add(kk))) };
        let y = match op {
            FpElementwiseOp::Add => _mm512_add_ps(a, b),
            FpElementwiseOp::Sub => _mm512_sub_ps(a, b),
            FpElementwiseOp::Mul => _mm512_mul_ps(a, b),
        };
        unsafe { _mm512_storeu_ps(out.as_mut_ptr().add(kk), y) };
        kk += 16;
    }
    while kk < len {
        out[kk] = match op {
            FpElementwiseOp::Add => lhs[kk].to_f32() + rhs[kk].to_f32(),
            FpElementwiseOp::Sub => lhs[kk].to_f32() - rhs[kk].to_f32(),
            FpElementwiseOp::Mul => lhs[kk].to_f32() * rhs[kk].to_f32(),
        };
        kk += 1;
    }
}

#[cfg(all(
    feature = "x86-fp-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[target_feature(enable = "avx512f")]
unsafe fn sgd_update_f16_f32_x86_avx512(data: &mut [f16], grad: &[f32], lr: f32) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let len = data.len();
    let lr_v = _mm512_set1_ps(lr);
    let mut kk = 0usize;
    while kk + 32 <= len {
        let w0 = unsafe { load_f16_as_f32x16_x86(data.as_ptr().add(kk)) };
        let w1 = unsafe { load_f16_as_f32x16_x86(data.as_ptr().add(kk + 16)) };
        let g0 = unsafe { _mm512_loadu_ps(grad.as_ptr().add(kk)) };
        let g1 = unsafe { _mm512_loadu_ps(grad.as_ptr().add(kk + 16)) };
        let updated0 = _mm512_fnmadd_ps(lr_v, g0, w0);
        let updated1 = _mm512_fnmadd_ps(lr_v, g1, w1);
        unsafe { store_f16_from_f32x16_x86(data.as_mut_ptr().add(kk), updated0) };
        unsafe { store_f16_from_f32x16_x86(data.as_mut_ptr().add(kk + 16), updated1) };
        kk += 32;
    }

    while kk + 16 <= len {
        let w = unsafe { load_f16_as_f32x16_x86(data.as_ptr().add(kk)) };
        let g = unsafe { _mm512_loadu_ps(grad.as_ptr().add(kk)) };
        let updated = _mm512_fnmadd_ps(lr_v, g, w);
        unsafe { store_f16_from_f32x16_x86(data.as_mut_ptr().add(kk), updated) };
        kk += 16;
    }

    while kk < len {
        data[kk] = f16::from_f32(data[kk].to_f32() - lr * grad[kk]);
        kk += 1;
    }
}

#[cfg(all(
    feature = "x86-fp-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[target_feature(enable = "avx512f")]
unsafe fn sum_f16_x86_avx512(x: &[f16]) -> f32 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let len = x.len();
    let mut kk = 0usize;
    let mut acc0 = _mm512_setzero_ps();
    let mut acc1 = _mm512_setzero_ps();

    while kk + 32 <= len {
        let x0 = unsafe { load_f16_as_f32x16_x86(x.as_ptr().add(kk)) };
        let x1 = unsafe { load_f16_as_f32x16_x86(x.as_ptr().add(kk + 16)) };
        acc0 = _mm512_add_ps(acc0, x0);
        acc1 = _mm512_add_ps(acc1, x1);
        kk += 32;
    }

    while kk + 16 <= len {
        let chunk = unsafe { load_f16_as_f32x16_x86(x.as_ptr().add(kk)) };
        acc0 = _mm512_add_ps(acc0, chunk);
        kk += 16;
    }

    let mut sum = unsafe { reduce_f32x16_x86(acc0) + reduce_f32x16_x86(acc1) };
    while kk < len {
        sum += x[kk].to_f32();
        kk += 1;
    }
    sum
}

#[cfg(all(
    feature = "x86-fp-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[target_feature(enable = "avx512f,avx512bf16,avx512vl")]
unsafe fn sgd_update_bf16_f32_x86_avx512(data: &mut [bf16], grad: &[f32], lr: f32) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let len = data.len();
    let lr_v = _mm512_set1_ps(lr);
    let mut kk = 0usize;
    while kk + 32 <= len {
        let w0 = unsafe { _mm512_cvtpbh_ps(load_bf16_x16_x86(data.as_ptr().add(kk))) };
        let w1 = unsafe { _mm512_cvtpbh_ps(load_bf16_x16_x86(data.as_ptr().add(kk + 16))) };
        let g0 = unsafe { _mm512_loadu_ps(grad.as_ptr().add(kk)) };
        let g1 = unsafe { _mm512_loadu_ps(grad.as_ptr().add(kk + 16)) };
        let updated0 = _mm512_fnmadd_ps(lr_v, g0, w0);
        let updated1 = _mm512_fnmadd_ps(lr_v, g1, w1);
        unsafe {
            store_bf16_x16_x86(
                data.as_mut_ptr().add(kk),
                convert_f32_to_bf16x16_x86(updated0),
            )
        };
        unsafe {
            store_bf16_x16_x86(
                data.as_mut_ptr().add(kk + 16),
                convert_f32_to_bf16x16_x86(updated1),
            )
        };
        kk += 32;
    }

    while kk + 16 <= len {
        let w = unsafe { _mm512_cvtpbh_ps(load_bf16_x16_x86(data.as_ptr().add(kk))) };
        let g = unsafe { _mm512_loadu_ps(grad.as_ptr().add(kk)) };
        let updated = _mm512_fnmadd_ps(lr_v, g, w);
        unsafe {
            store_bf16_x16_x86(
                data.as_mut_ptr().add(kk),
                convert_f32_to_bf16x16_x86(updated),
            )
        };
        kk += 16;
    }

    while kk < len {
        data[kk] = half::bf16::from_f32(data[kk].to_f32() - lr * grad[kk]);
        kk += 1;
    }
}

#[cfg(all(
    feature = "x86-fp-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[target_feature(enable = "avx512f,avx512bf16,avx512vl")]
unsafe fn sum_bf16_x86_avx512(x: &[bf16]) -> f32 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let len = x.len();
    let mut kk = 0usize;
    let mut acc0 = _mm512_setzero_ps();
    let mut acc1 = _mm512_setzero_ps();

    while kk + 32 <= len {
        let x0 = unsafe { _mm512_cvtpbh_ps(load_bf16_x16_x86(x.as_ptr().add(kk))) };
        let x1 = unsafe { _mm512_cvtpbh_ps(load_bf16_x16_x86(x.as_ptr().add(kk + 16))) };
        acc0 = _mm512_add_ps(acc0, x0);
        acc1 = _mm512_add_ps(acc1, x1);
        kk += 32;
    }

    while kk + 16 <= len {
        let chunk = unsafe { _mm512_cvtpbh_ps(load_bf16_x16_x86(x.as_ptr().add(kk))) };
        acc0 = _mm512_add_ps(acc0, chunk);
        kk += 16;
    }

    let mut sum = unsafe { reduce_f32x16_x86(acc0) + reduce_f32x16_x86(acc1) };
    while kk < len {
        sum += x[kk].to_f32();
        kk += 1;
    }
    sum
}

#[cfg(all(
    feature = "x86-fp-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[target_feature(enable = "avx512f")]
unsafe fn sgd_momentum_update_f16_f32_x86_avx512(
    data: &mut [f16],
    velocity: &mut [f32],
    grad: &[f32],
    lr: f32,
    momentum: f32,
) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let len = data.len();
    let lr_v = _mm512_set1_ps(lr);
    let momentum_v = _mm512_set1_ps(momentum);
    let mut kk = 0usize;
    while kk + 16 <= len {
        let w = unsafe { load_f16_as_f32x16_x86(data.as_ptr().add(kk)) };
        let old_v = unsafe { _mm512_loadu_ps(velocity.as_ptr().add(kk)) };
        let g = unsafe { _mm512_loadu_ps(grad.as_ptr().add(kk)) };
        let new_v = _mm512_fmadd_ps(momentum_v, old_v, g);
        unsafe { _mm512_storeu_ps(velocity.as_mut_ptr().add(kk), new_v) };
        let updated = _mm512_fnmadd_ps(lr_v, new_v, w);
        unsafe { store_f16_from_f32x16_x86(data.as_mut_ptr().add(kk), updated) };
        kk += 16;
    }

    while kk < len {
        velocity[kk] = momentum * velocity[kk] + grad[kk];
        data[kk] = f16::from_f32(data[kk].to_f32() - lr * velocity[kk]);
        kk += 1;
    }
}

#[cfg(all(
    feature = "x86-fp-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[target_feature(enable = "avx512f,avx512bf16,avx512vl")]
unsafe fn sgd_momentum_update_bf16_f32_x86_avx512(
    data: &mut [bf16],
    velocity: &mut [f32],
    grad: &[f32],
    lr: f32,
    momentum: f32,
) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let len = data.len();
    let lr_v = _mm512_set1_ps(lr);
    let momentum_v = _mm512_set1_ps(momentum);
    let mut kk = 0usize;
    while kk + 16 <= len {
        let w = unsafe { _mm512_cvtpbh_ps(load_bf16_x16_x86(data.as_ptr().add(kk))) };
        let old_v = unsafe { _mm512_loadu_ps(velocity.as_ptr().add(kk)) };
        let g = unsafe { _mm512_loadu_ps(grad.as_ptr().add(kk)) };
        let new_v = _mm512_fmadd_ps(momentum_v, old_v, g);
        unsafe { _mm512_storeu_ps(velocity.as_mut_ptr().add(kk), new_v) };
        let updated = _mm512_fnmadd_ps(lr_v, new_v, w);
        unsafe {
            store_bf16_x16_x86(
                data.as_mut_ptr().add(kk),
                convert_f32_to_bf16x16_x86(updated),
            )
        };
        kk += 16;
    }

    while kk < len {
        velocity[kk] = momentum * velocity[kk] + grad[kk];
        data[kk] = half::bf16::from_f32(data[kk].to_f32() - lr * velocity[kk]);
        kk += 1;
    }
}

#[cfg(all(
    feature = "x86-fp-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx512f")]
unsafe fn adam_update_f16_f32_x86_avx512(
    data: &mut [f16],
    exp_avg: &mut [f32],
    exp_avg_sq: &mut [f32],
    grad: &[f32],
    lr: f32,
    beta1: f32,
    beta2: f32,
    bias_correction1: f32,
    bias_correction2: f32,
    eps: f32,
) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let len = data.len();
    let lr_v = _mm512_set1_ps(lr);
    let beta1_v = _mm512_set1_ps(beta1);
    let one_minus_beta1_v = _mm512_set1_ps(1.0 - beta1);
    let beta2_v = _mm512_set1_ps(beta2);
    let one_minus_beta2_v = _mm512_set1_ps(1.0 - beta2);
    let inv_bias_correction1_v = _mm512_set1_ps(1.0 / bias_correction1);
    let inv_bias_correction2_v = _mm512_set1_ps(1.0 / bias_correction2);
    let eps_v = _mm512_set1_ps(eps);
    let mut kk = 0usize;

    while kk + 16 <= len {
        let w = unsafe { load_f16_as_f32x16_x86(data.as_ptr().add(kk)) };
        let old_m = unsafe { _mm512_loadu_ps(exp_avg.as_ptr().add(kk)) };
        let old_v = unsafe { _mm512_loadu_ps(exp_avg_sq.as_ptr().add(kk)) };
        let g = unsafe { _mm512_loadu_ps(grad.as_ptr().add(kk)) };
        let new_m = _mm512_fmadd_ps(beta1_v, old_m, _mm512_mul_ps(one_minus_beta1_v, g));
        let grad_sq = _mm512_mul_ps(g, g);
        let new_v = _mm512_fmadd_ps(beta2_v, old_v, _mm512_mul_ps(one_minus_beta2_v, grad_sq));
        unsafe { _mm512_storeu_ps(exp_avg.as_mut_ptr().add(kk), new_m) };
        unsafe { _mm512_storeu_ps(exp_avg_sq.as_mut_ptr().add(kk), new_v) };
        let m_hat = _mm512_mul_ps(new_m, inv_bias_correction1_v);
        let v_hat = _mm512_mul_ps(new_v, inv_bias_correction2_v);
        let denom = _mm512_add_ps(_mm512_sqrt_ps(v_hat), eps_v);
        let updated = _mm512_fnmadd_ps(lr_v, _mm512_div_ps(m_hat, denom), w);
        unsafe { store_f16_from_f32x16_x86(data.as_mut_ptr().add(kk), updated) };
        kk += 16;
    }

    while kk < len {
        exp_avg[kk] = beta1 * exp_avg[kk] + (1.0 - beta1) * grad[kk];
        exp_avg_sq[kk] = beta2 * exp_avg_sq[kk] + (1.0 - beta2) * grad[kk] * grad[kk];
        let m_hat = exp_avg[kk] / bias_correction1;
        let v_hat = exp_avg_sq[kk] / bias_correction2;
        data[kk] = f16::from_f32(data[kk].to_f32() - lr * (m_hat / (v_hat.sqrt() + eps)));
        kk += 1;
    }
}

#[cfg(all(
    feature = "x86-fp-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx512f,avx512bf16,avx512vl")]
unsafe fn adam_update_bf16_f32_x86_avx512(
    data: &mut [bf16],
    exp_avg: &mut [f32],
    exp_avg_sq: &mut [f32],
    grad: &[f32],
    lr: f32,
    beta1: f32,
    beta2: f32,
    bias_correction1: f32,
    bias_correction2: f32,
    eps: f32,
) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let len = data.len();
    let lr_v = _mm512_set1_ps(lr);
    let beta1_v = _mm512_set1_ps(beta1);
    let one_minus_beta1_v = _mm512_set1_ps(1.0 - beta1);
    let beta2_v = _mm512_set1_ps(beta2);
    let one_minus_beta2_v = _mm512_set1_ps(1.0 - beta2);
    let inv_bias_correction1_v = _mm512_set1_ps(1.0 / bias_correction1);
    let inv_bias_correction2_v = _mm512_set1_ps(1.0 / bias_correction2);
    let eps_v = _mm512_set1_ps(eps);
    let mut kk = 0usize;

    while kk + 16 <= len {
        let w = unsafe { _mm512_cvtpbh_ps(load_bf16_x16_x86(data.as_ptr().add(kk))) };
        let old_m = unsafe { _mm512_loadu_ps(exp_avg.as_ptr().add(kk)) };
        let old_v = unsafe { _mm512_loadu_ps(exp_avg_sq.as_ptr().add(kk)) };
        let g = unsafe { _mm512_loadu_ps(grad.as_ptr().add(kk)) };
        let new_m = _mm512_fmadd_ps(beta1_v, old_m, _mm512_mul_ps(one_minus_beta1_v, g));
        let grad_sq = _mm512_mul_ps(g, g);
        let new_v = _mm512_fmadd_ps(beta2_v, old_v, _mm512_mul_ps(one_minus_beta2_v, grad_sq));
        unsafe { _mm512_storeu_ps(exp_avg.as_mut_ptr().add(kk), new_m) };
        unsafe { _mm512_storeu_ps(exp_avg_sq.as_mut_ptr().add(kk), new_v) };
        let m_hat = _mm512_mul_ps(new_m, inv_bias_correction1_v);
        let v_hat = _mm512_mul_ps(new_v, inv_bias_correction2_v);
        let denom = _mm512_add_ps(_mm512_sqrt_ps(v_hat), eps_v);
        let updated = _mm512_fnmadd_ps(lr_v, _mm512_div_ps(m_hat, denom), w);
        unsafe {
            store_bf16_x16_x86(
                data.as_mut_ptr().add(kk),
                convert_f32_to_bf16x16_x86(updated),
            )
        };
        kk += 16;
    }

    while kk < len {
        exp_avg[kk] = beta1 * exp_avg[kk] + (1.0 - beta1) * grad[kk];
        exp_avg_sq[kk] = beta2 * exp_avg_sq[kk] + (1.0 - beta2) * grad[kk] * grad[kk];
        let m_hat = exp_avg[kk] / bias_correction1;
        let v_hat = exp_avg_sq[kk] / bias_correction2;
        data[kk] = half::bf16::from_f32(data[kk].to_f32() - lr * (m_hat / (v_hat.sqrt() + eps)));
        kk += 1;
    }
}

#[cfg(all(
    feature = "x86-fp-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn sgd_update_f16_f32_x86_avx2(data: &mut [f16], grad: &[f32], lr: f32) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let len = data.len();
    let lr_v = _mm256_set1_ps(lr);
    let mut kk = 0usize;
    while kk + 16 <= len {
        let w0 = unsafe { load_f16_as_f32x8_x86(data.as_ptr().add(kk)) };
        let w1 = unsafe { load_f16_as_f32x8_x86(data.as_ptr().add(kk + 8)) };
        let g0 = unsafe { _mm256_loadu_ps(grad.as_ptr().add(kk)) };
        let g1 = unsafe { _mm256_loadu_ps(grad.as_ptr().add(kk + 8)) };
        let updated0 = _mm256_fnmadd_ps(lr_v, g0, w0);
        let updated1 = _mm256_fnmadd_ps(lr_v, g1, w1);
        unsafe { store_f16_from_f32x8_x86(data.as_mut_ptr().add(kk), updated0) };
        unsafe { store_f16_from_f32x8_x86(data.as_mut_ptr().add(kk + 8), updated1) };
        kk += 16;
    }

    while kk + 8 <= len {
        let w = unsafe { load_f16_as_f32x8_x86(data.as_ptr().add(kk)) };
        let g = unsafe { _mm256_loadu_ps(grad.as_ptr().add(kk)) };
        let updated = _mm256_fnmadd_ps(lr_v, g, w);
        unsafe { store_f16_from_f32x8_x86(data.as_mut_ptr().add(kk), updated) };
        kk += 8;
    }

    while kk < len {
        data[kk] = f16::from_f32(data[kk].to_f32() - lr * grad[kk]);
        kk += 1;
    }
}

#[cfg(all(
    feature = "x86-fp-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn sgd_momentum_update_f16_f32_x86_avx2(
    data: &mut [f16],
    velocity: &mut [f32],
    grad: &[f32],
    lr: f32,
    momentum: f32,
) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let len = data.len();
    let lr_v = _mm256_set1_ps(lr);
    let momentum_v = _mm256_set1_ps(momentum);
    let mut kk = 0usize;
    while kk + 8 <= len {
        let w = unsafe { load_f16_as_f32x8_x86(data.as_ptr().add(kk)) };
        let old_v = unsafe { _mm256_loadu_ps(velocity.as_ptr().add(kk)) };
        let g = unsafe { _mm256_loadu_ps(grad.as_ptr().add(kk)) };
        let new_v = _mm256_fmadd_ps(momentum_v, old_v, g);
        unsafe { _mm256_storeu_ps(velocity.as_mut_ptr().add(kk), new_v) };
        let updated = _mm256_fnmadd_ps(lr_v, new_v, w);
        unsafe { store_f16_from_f32x8_x86(data.as_mut_ptr().add(kk), updated) };
        kk += 8;
    }

    while kk < len {
        velocity[kk] = momentum * velocity[kk] + grad[kk];
        data[kk] = f16::from_f32(data[kk].to_f32() - lr * velocity[kk]);
        kk += 1;
    }
}

#[cfg(all(
    feature = "x86-fp-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn adam_update_f16_f32_x86_avx2(
    data: &mut [f16],
    exp_avg: &mut [f32],
    exp_avg_sq: &mut [f32],
    grad: &[f32],
    lr: f32,
    beta1: f32,
    beta2: f32,
    bias_correction1: f32,
    bias_correction2: f32,
    eps: f32,
) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let len = data.len();
    let lr_v = _mm256_set1_ps(lr);
    let beta1_v = _mm256_set1_ps(beta1);
    let one_minus_beta1_v = _mm256_set1_ps(1.0 - beta1);
    let beta2_v = _mm256_set1_ps(beta2);
    let one_minus_beta2_v = _mm256_set1_ps(1.0 - beta2);
    let inv_bias_correction1_v = _mm256_set1_ps(1.0 / bias_correction1);
    let inv_bias_correction2_v = _mm256_set1_ps(1.0 / bias_correction2);
    let eps_v = _mm256_set1_ps(eps);
    let mut kk = 0usize;

    while kk + 8 <= len {
        let w = unsafe { load_f16_as_f32x8_x86(data.as_ptr().add(kk)) };
        let old_m = unsafe { _mm256_loadu_ps(exp_avg.as_ptr().add(kk)) };
        let old_v = unsafe { _mm256_loadu_ps(exp_avg_sq.as_ptr().add(kk)) };
        let g = unsafe { _mm256_loadu_ps(grad.as_ptr().add(kk)) };
        let new_m = _mm256_fmadd_ps(beta1_v, old_m, _mm256_mul_ps(one_minus_beta1_v, g));
        let grad_sq = _mm256_mul_ps(g, g);
        let new_v = _mm256_fmadd_ps(beta2_v, old_v, _mm256_mul_ps(one_minus_beta2_v, grad_sq));
        unsafe { _mm256_storeu_ps(exp_avg.as_mut_ptr().add(kk), new_m) };
        unsafe { _mm256_storeu_ps(exp_avg_sq.as_mut_ptr().add(kk), new_v) };
        let m_hat = _mm256_mul_ps(new_m, inv_bias_correction1_v);
        let v_hat = _mm256_mul_ps(new_v, inv_bias_correction2_v);
        let denom = _mm256_add_ps(_mm256_sqrt_ps(v_hat), eps_v);
        let updated = _mm256_fnmadd_ps(lr_v, _mm256_div_ps(m_hat, denom), w);
        unsafe { store_f16_from_f32x8_x86(data.as_mut_ptr().add(kk), updated) };
        kk += 8;
    }

    while kk < len {
        exp_avg[kk] = beta1 * exp_avg[kk] + (1.0 - beta1) * grad[kk];
        exp_avg_sq[kk] = beta2 * exp_avg_sq[kk] + (1.0 - beta2) * grad[kk] * grad[kk];
        let m_hat = exp_avg[kk] / bias_correction1;
        let v_hat = exp_avg_sq[kk] / bias_correction2;
        data[kk] = f16::from_f32(data[kk].to_f32() - lr * (m_hat / (v_hat.sqrt() + eps)));
        kk += 1;
    }
}

#[cfg(all(
    feature = "x86-fp-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[target_feature(enable = "avx2,fma")]
unsafe fn sgd_update_bf16_f32_x86_avx2(data: &mut [bf16], grad: &[f32], lr: f32) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let len = data.len();
    let lr_v = _mm256_set1_ps(lr);
    let mut kk = 0usize;
    while kk + 16 <= len {
        let w0 = unsafe { load_bf16_as_f32x8_x86(data.as_ptr().add(kk)) };
        let w1 = unsafe { load_bf16_as_f32x8_x86(data.as_ptr().add(kk + 8)) };
        let g0 = unsafe { _mm256_loadu_ps(grad.as_ptr().add(kk)) };
        let g1 = unsafe { _mm256_loadu_ps(grad.as_ptr().add(kk + 8)) };
        let updated0 = _mm256_fnmadd_ps(lr_v, g0, w0);
        let updated1 = _mm256_fnmadd_ps(lr_v, g1, w1);
        unsafe { store_bf16_from_f32x8_x86(data.as_mut_ptr().add(kk), updated0) };
        unsafe { store_bf16_from_f32x8_x86(data.as_mut_ptr().add(kk + 8), updated1) };
        kk += 16;
    }

    while kk + 8 <= len {
        let w = unsafe { load_bf16_as_f32x8_x86(data.as_ptr().add(kk)) };
        let g = unsafe { _mm256_loadu_ps(grad.as_ptr().add(kk)) };
        let updated = _mm256_fnmadd_ps(lr_v, g, w);
        unsafe { store_bf16_from_f32x8_x86(data.as_mut_ptr().add(kk), updated) };
        kk += 8;
    }

    while kk < len {
        data[kk] = half::bf16::from_f32(data[kk].to_f32() - lr * grad[kk]);
        kk += 1;
    }
}

#[cfg(all(
    feature = "x86-fp-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[target_feature(enable = "avx2,fma")]
unsafe fn sgd_momentum_update_bf16_f32_x86_avx2(
    data: &mut [bf16],
    velocity: &mut [f32],
    grad: &[f32],
    lr: f32,
    momentum: f32,
) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let len = data.len();
    let lr_v = _mm256_set1_ps(lr);
    let momentum_v = _mm256_set1_ps(momentum);
    let mut kk = 0usize;
    while kk + 8 <= len {
        let w = unsafe { load_bf16_as_f32x8_x86(data.as_ptr().add(kk)) };
        let old_v = unsafe { _mm256_loadu_ps(velocity.as_ptr().add(kk)) };
        let g = unsafe { _mm256_loadu_ps(grad.as_ptr().add(kk)) };
        let new_v = _mm256_fmadd_ps(momentum_v, old_v, g);
        unsafe { _mm256_storeu_ps(velocity.as_mut_ptr().add(kk), new_v) };
        let updated = _mm256_fnmadd_ps(lr_v, new_v, w);
        unsafe { store_bf16_from_f32x8_x86(data.as_mut_ptr().add(kk), updated) };
        kk += 8;
    }

    while kk < len {
        velocity[kk] = momentum * velocity[kk] + grad[kk];
        data[kk] = half::bf16::from_f32(data[kk].to_f32() - lr * velocity[kk]);
        kk += 1;
    }
}

#[cfg(all(
    feature = "x86-fp-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2,fma")]
unsafe fn adam_update_bf16_f32_x86_avx2(
    data: &mut [bf16],
    exp_avg: &mut [f32],
    exp_avg_sq: &mut [f32],
    grad: &[f32],
    lr: f32,
    beta1: f32,
    beta2: f32,
    bias_correction1: f32,
    bias_correction2: f32,
    eps: f32,
) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let len = data.len();
    let lr_v = _mm256_set1_ps(lr);
    let beta1_v = _mm256_set1_ps(beta1);
    let one_minus_beta1_v = _mm256_set1_ps(1.0 - beta1);
    let beta2_v = _mm256_set1_ps(beta2);
    let one_minus_beta2_v = _mm256_set1_ps(1.0 - beta2);
    let inv_bias_correction1_v = _mm256_set1_ps(1.0 / bias_correction1);
    let inv_bias_correction2_v = _mm256_set1_ps(1.0 / bias_correction2);
    let eps_v = _mm256_set1_ps(eps);
    let mut kk = 0usize;

    while kk + 8 <= len {
        let w = unsafe { load_bf16_as_f32x8_x86(data.as_ptr().add(kk)) };
        let old_m = unsafe { _mm256_loadu_ps(exp_avg.as_ptr().add(kk)) };
        let old_v = unsafe { _mm256_loadu_ps(exp_avg_sq.as_ptr().add(kk)) };
        let g = unsafe { _mm256_loadu_ps(grad.as_ptr().add(kk)) };
        let new_m = _mm256_fmadd_ps(beta1_v, old_m, _mm256_mul_ps(one_minus_beta1_v, g));
        let grad_sq = _mm256_mul_ps(g, g);
        let new_v = _mm256_fmadd_ps(beta2_v, old_v, _mm256_mul_ps(one_minus_beta2_v, grad_sq));
        unsafe { _mm256_storeu_ps(exp_avg.as_mut_ptr().add(kk), new_m) };
        unsafe { _mm256_storeu_ps(exp_avg_sq.as_mut_ptr().add(kk), new_v) };
        let m_hat = _mm256_mul_ps(new_m, inv_bias_correction1_v);
        let v_hat = _mm256_mul_ps(new_v, inv_bias_correction2_v);
        let denom = _mm256_add_ps(_mm256_sqrt_ps(v_hat), eps_v);
        let updated = _mm256_fnmadd_ps(lr_v, _mm256_div_ps(m_hat, denom), w);
        unsafe { store_bf16_from_f32x8_x86(data.as_mut_ptr().add(kk), updated) };
        kk += 8;
    }

    while kk < len {
        exp_avg[kk] = beta1 * exp_avg[kk] + (1.0 - beta1) * grad[kk];
        exp_avg_sq[kk] = beta2 * exp_avg_sq[kk] + (1.0 - beta2) * grad[kk] * grad[kk];
        let m_hat = exp_avg[kk] / bias_correction1;
        let v_hat = exp_avg_sq[kk] / bias_correction2;
        data[kk] = half::bf16::from_f32(data[kk].to_f32() - lr * (m_hat / (v_hat.sqrt() + eps)));
        kk += 1;
    }
}

#[cfg(all(
    feature = "x86-fp-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[target_feature(enable = "avx512f")]
unsafe fn dot_f32_x86_avx512(x: &[f32], row: &[f32]) -> f32 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let k_dim = x.len();
    let mut kk = 0usize;
    let mut acc0 = _mm512_setzero_ps();
    let mut acc1 = _mm512_setzero_ps();

    while kk + 32 <= k_dim {
        let row_lo = unsafe { _mm512_loadu_ps(row.as_ptr().add(kk)) };
        let row_hi = unsafe { _mm512_loadu_ps(row.as_ptr().add(kk + 16)) };
        let x_lo = unsafe { _mm512_loadu_ps(x.as_ptr().add(kk)) };
        let x_hi = unsafe { _mm512_loadu_ps(x.as_ptr().add(kk + 16)) };
        acc0 = _mm512_fmadd_ps(row_lo, x_lo, acc0);
        acc1 = _mm512_fmadd_ps(row_hi, x_hi, acc1);
        kk += 32;
    }

    while kk + 16 <= k_dim {
        let row_chunk = unsafe { _mm512_loadu_ps(row.as_ptr().add(kk)) };
        let x_chunk = unsafe { _mm512_loadu_ps(x.as_ptr().add(kk)) };
        acc0 = _mm512_fmadd_ps(row_chunk, x_chunk, acc0);
        kk += 16;
    }

    let mut sum = unsafe { reduce_f32x16_x86(acc0) + reduce_f32x16_x86(acc1) };
    while kk < k_dim {
        sum += row[kk] * x[kk];
        kk += 1;
    }
    sum
}

#[cfg(all(
    feature = "x86-fp-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[target_feature(enable = "avx512f")]
unsafe fn dot2_f32_x86_avx512(x: &[f32], row0: &[f32], row1: &[f32]) -> (f32, f32) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let k_dim = x.len();
    let mut kk = 0usize;
    let mut acc00 = _mm512_setzero_ps();
    let mut acc01 = _mm512_setzero_ps();
    let mut acc10 = _mm512_setzero_ps();
    let mut acc11 = _mm512_setzero_ps();

    while kk + 32 <= k_dim {
        let x_lo = unsafe { _mm512_loadu_ps(x.as_ptr().add(kk)) };
        let x_hi = unsafe { _mm512_loadu_ps(x.as_ptr().add(kk + 16)) };
        let row0_lo = unsafe { _mm512_loadu_ps(row0.as_ptr().add(kk)) };
        let row0_hi = unsafe { _mm512_loadu_ps(row0.as_ptr().add(kk + 16)) };
        let row1_lo = unsafe { _mm512_loadu_ps(row1.as_ptr().add(kk)) };
        let row1_hi = unsafe { _mm512_loadu_ps(row1.as_ptr().add(kk + 16)) };
        acc00 = _mm512_fmadd_ps(row0_lo, x_lo, acc00);
        acc01 = _mm512_fmadd_ps(row0_hi, x_hi, acc01);
        acc10 = _mm512_fmadd_ps(row1_lo, x_lo, acc10);
        acc11 = _mm512_fmadd_ps(row1_hi, x_hi, acc11);
        kk += 32;
    }

    while kk + 16 <= k_dim {
        let x_chunk = unsafe { _mm512_loadu_ps(x.as_ptr().add(kk)) };
        let row0_chunk = unsafe { _mm512_loadu_ps(row0.as_ptr().add(kk)) };
        let row1_chunk = unsafe { _mm512_loadu_ps(row1.as_ptr().add(kk)) };
        acc00 = _mm512_fmadd_ps(row0_chunk, x_chunk, acc00);
        acc10 = _mm512_fmadd_ps(row1_chunk, x_chunk, acc10);
        kk += 16;
    }

    let mut sum0 = unsafe { reduce_f32x16_x86(acc00) + reduce_f32x16_x86(acc01) };
    let mut sum1 = unsafe { reduce_f32x16_x86(acc10) + reduce_f32x16_x86(acc11) };
    while kk < k_dim {
        let xv = x[kk];
        sum0 += row0[kk] * xv;
        sum1 += row1[kk] * xv;
        kk += 1;
    }
    (sum0, sum1)
}

#[cfg(all(
    feature = "x86-fp-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[target_feature(enable = "avx512f")]
unsafe fn dot3_f32_x86_avx512(
    x: &[f32],
    row0: &[f32],
    row1: &[f32],
    row2: &[f32],
) -> (f32, f32, f32) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let k_dim = x.len();
    let mut kk = 0usize;
    let mut acc00 = _mm512_setzero_ps();
    let mut acc01 = _mm512_setzero_ps();
    let mut acc10 = _mm512_setzero_ps();
    let mut acc11 = _mm512_setzero_ps();
    let mut acc20 = _mm512_setzero_ps();
    let mut acc21 = _mm512_setzero_ps();

    while kk + 32 <= k_dim {
        let x_lo = unsafe { _mm512_loadu_ps(x.as_ptr().add(kk)) };
        let x_hi = unsafe { _mm512_loadu_ps(x.as_ptr().add(kk + 16)) };
        let row0_lo = unsafe { _mm512_loadu_ps(row0.as_ptr().add(kk)) };
        let row0_hi = unsafe { _mm512_loadu_ps(row0.as_ptr().add(kk + 16)) };
        let row1_lo = unsafe { _mm512_loadu_ps(row1.as_ptr().add(kk)) };
        let row1_hi = unsafe { _mm512_loadu_ps(row1.as_ptr().add(kk + 16)) };
        let row2_lo = unsafe { _mm512_loadu_ps(row2.as_ptr().add(kk)) };
        let row2_hi = unsafe { _mm512_loadu_ps(row2.as_ptr().add(kk + 16)) };
        acc00 = _mm512_fmadd_ps(row0_lo, x_lo, acc00);
        acc01 = _mm512_fmadd_ps(row0_hi, x_hi, acc01);
        acc10 = _mm512_fmadd_ps(row1_lo, x_lo, acc10);
        acc11 = _mm512_fmadd_ps(row1_hi, x_hi, acc11);
        acc20 = _mm512_fmadd_ps(row2_lo, x_lo, acc20);
        acc21 = _mm512_fmadd_ps(row2_hi, x_hi, acc21);
        kk += 32;
    }

    while kk + 16 <= k_dim {
        let x_chunk = unsafe { _mm512_loadu_ps(x.as_ptr().add(kk)) };
        let row0_chunk = unsafe { _mm512_loadu_ps(row0.as_ptr().add(kk)) };
        let row1_chunk = unsafe { _mm512_loadu_ps(row1.as_ptr().add(kk)) };
        let row2_chunk = unsafe { _mm512_loadu_ps(row2.as_ptr().add(kk)) };
        acc00 = _mm512_fmadd_ps(row0_chunk, x_chunk, acc00);
        acc10 = _mm512_fmadd_ps(row1_chunk, x_chunk, acc10);
        acc20 = _mm512_fmadd_ps(row2_chunk, x_chunk, acc20);
        kk += 16;
    }

    let mut sum0 = unsafe { reduce_f32x16_x86(acc00) + reduce_f32x16_x86(acc01) };
    let mut sum1 = unsafe { reduce_f32x16_x86(acc10) + reduce_f32x16_x86(acc11) };
    let mut sum2 = unsafe { reduce_f32x16_x86(acc20) + reduce_f32x16_x86(acc21) };
    while kk < k_dim {
        let xv = x[kk];
        sum0 += row0[kk] * xv;
        sum1 += row1[kk] * xv;
        sum2 += row2[kk] * xv;
        kk += 1;
    }
    (sum0, sum1, sum2)
}

#[cfg(all(
    feature = "x86-fp-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[target_feature(enable = "avx512f")]
unsafe fn dot_f32_f16_x86_avx512(x: &[f32], row: &[f16]) -> f32 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let k_dim = x.len();
    let mut kk = 0usize;
    let mut acc0 = _mm512_setzero_ps();
    let mut acc1 = _mm512_setzero_ps();

    while kk + 32 <= k_dim {
        let row_lo = unsafe { load_f16_as_f32x16_x86(row.as_ptr().add(kk)) };
        let row_hi = unsafe { load_f16_as_f32x16_x86(row.as_ptr().add(kk + 16)) };
        let x_lo = unsafe { _mm512_loadu_ps(x.as_ptr().add(kk)) };
        let x_hi = unsafe { _mm512_loadu_ps(x.as_ptr().add(kk + 16)) };
        acc0 = _mm512_fmadd_ps(row_lo, x_lo, acc0);
        acc1 = _mm512_fmadd_ps(row_hi, x_hi, acc1);
        kk += 32;
    }

    while kk + 16 <= k_dim {
        let row_chunk = unsafe { load_f16_as_f32x16_x86(row.as_ptr().add(kk)) };
        let x_chunk = unsafe { _mm512_loadu_ps(x.as_ptr().add(kk)) };
        acc0 = _mm512_fmadd_ps(row_chunk, x_chunk, acc0);
        kk += 16;
    }

    let mut sum = unsafe { reduce_f32x16_x86(acc0) + reduce_f32x16_x86(acc1) };
    while kk < k_dim {
        sum += row[kk].to_f32() * x[kk];
        kk += 1;
    }
    sum
}

#[cfg(all(
    feature = "x86-fp-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[target_feature(enable = "avx512f")]
unsafe fn dot2_f32_f16_x86_avx512(x: &[f32], row0: &[f16], row1: &[f16]) -> (f32, f32) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let k_dim = x.len();
    let mut kk = 0usize;
    let mut acc00 = _mm512_setzero_ps();
    let mut acc01 = _mm512_setzero_ps();
    let mut acc10 = _mm512_setzero_ps();
    let mut acc11 = _mm512_setzero_ps();

    while kk + 32 <= k_dim {
        let x_lo = unsafe { _mm512_loadu_ps(x.as_ptr().add(kk)) };
        let x_hi = unsafe { _mm512_loadu_ps(x.as_ptr().add(kk + 16)) };
        let row0_lo = unsafe { load_f16_as_f32x16_x86(row0.as_ptr().add(kk)) };
        let row0_hi = unsafe { load_f16_as_f32x16_x86(row0.as_ptr().add(kk + 16)) };
        let row1_lo = unsafe { load_f16_as_f32x16_x86(row1.as_ptr().add(kk)) };
        let row1_hi = unsafe { load_f16_as_f32x16_x86(row1.as_ptr().add(kk + 16)) };
        acc00 = _mm512_fmadd_ps(row0_lo, x_lo, acc00);
        acc01 = _mm512_fmadd_ps(row0_hi, x_hi, acc01);
        acc10 = _mm512_fmadd_ps(row1_lo, x_lo, acc10);
        acc11 = _mm512_fmadd_ps(row1_hi, x_hi, acc11);
        kk += 32;
    }

    while kk + 16 <= k_dim {
        let x_chunk = unsafe { _mm512_loadu_ps(x.as_ptr().add(kk)) };
        let row0_chunk = unsafe { load_f16_as_f32x16_x86(row0.as_ptr().add(kk)) };
        let row1_chunk = unsafe { load_f16_as_f32x16_x86(row1.as_ptr().add(kk)) };
        acc00 = _mm512_fmadd_ps(row0_chunk, x_chunk, acc00);
        acc10 = _mm512_fmadd_ps(row1_chunk, x_chunk, acc10);
        kk += 16;
    }

    let mut sum0 = unsafe { reduce_f32x16_x86(acc00) + reduce_f32x16_x86(acc01) };
    let mut sum1 = unsafe { reduce_f32x16_x86(acc10) + reduce_f32x16_x86(acc11) };
    while kk < k_dim {
        let xv = x[kk];
        sum0 += row0[kk].to_f32() * xv;
        sum1 += row1[kk].to_f32() * xv;
        kk += 1;
    }
    (sum0, sum1)
}

#[cfg(all(
    feature = "x86-fp-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[target_feature(enable = "avx512f")]
unsafe fn dot3_f32_f16_x86_avx512(
    x: &[f32],
    row0: &[f16],
    row1: &[f16],
    row2: &[f16],
) -> (f32, f32, f32) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let k_dim = x.len();
    let mut kk = 0usize;
    let mut acc00 = _mm512_setzero_ps();
    let mut acc01 = _mm512_setzero_ps();
    let mut acc10 = _mm512_setzero_ps();
    let mut acc11 = _mm512_setzero_ps();
    let mut acc20 = _mm512_setzero_ps();
    let mut acc21 = _mm512_setzero_ps();

    while kk + 32 <= k_dim {
        let x_lo = unsafe { _mm512_loadu_ps(x.as_ptr().add(kk)) };
        let x_hi = unsafe { _mm512_loadu_ps(x.as_ptr().add(kk + 16)) };
        let row0_lo = unsafe { load_f16_as_f32x16_x86(row0.as_ptr().add(kk)) };
        let row0_hi = unsafe { load_f16_as_f32x16_x86(row0.as_ptr().add(kk + 16)) };
        let row1_lo = unsafe { load_f16_as_f32x16_x86(row1.as_ptr().add(kk)) };
        let row1_hi = unsafe { load_f16_as_f32x16_x86(row1.as_ptr().add(kk + 16)) };
        let row2_lo = unsafe { load_f16_as_f32x16_x86(row2.as_ptr().add(kk)) };
        let row2_hi = unsafe { load_f16_as_f32x16_x86(row2.as_ptr().add(kk + 16)) };
        acc00 = _mm512_fmadd_ps(row0_lo, x_lo, acc00);
        acc01 = _mm512_fmadd_ps(row0_hi, x_hi, acc01);
        acc10 = _mm512_fmadd_ps(row1_lo, x_lo, acc10);
        acc11 = _mm512_fmadd_ps(row1_hi, x_hi, acc11);
        acc20 = _mm512_fmadd_ps(row2_lo, x_lo, acc20);
        acc21 = _mm512_fmadd_ps(row2_hi, x_hi, acc21);
        kk += 32;
    }

    while kk + 16 <= k_dim {
        let x_chunk = unsafe { _mm512_loadu_ps(x.as_ptr().add(kk)) };
        let row0_chunk = unsafe { load_f16_as_f32x16_x86(row0.as_ptr().add(kk)) };
        let row1_chunk = unsafe { load_f16_as_f32x16_x86(row1.as_ptr().add(kk)) };
        let row2_chunk = unsafe { load_f16_as_f32x16_x86(row2.as_ptr().add(kk)) };
        acc00 = _mm512_fmadd_ps(row0_chunk, x_chunk, acc00);
        acc10 = _mm512_fmadd_ps(row1_chunk, x_chunk, acc10);
        acc20 = _mm512_fmadd_ps(row2_chunk, x_chunk, acc20);
        kk += 16;
    }

    let mut sum0 = unsafe { reduce_f32x16_x86(acc00) + reduce_f32x16_x86(acc01) };
    let mut sum1 = unsafe { reduce_f32x16_x86(acc10) + reduce_f32x16_x86(acc11) };
    let mut sum2 = unsafe { reduce_f32x16_x86(acc20) + reduce_f32x16_x86(acc21) };
    while kk < k_dim {
        let xv = x[kk];
        sum0 += row0[kk].to_f32() * xv;
        sum1 += row1[kk].to_f32() * xv;
        sum2 += row2[kk].to_f32() * xv;
        kk += 1;
    }
    (sum0, sum1, sum2)
}

#[cfg(all(
    feature = "x86-fp-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[target_feature(enable = "avx512f")]
unsafe fn dot_f32_bf16_x86_avx512(x: &[f32], row: &[bf16]) -> f32 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let k_dim = x.len();
    let mut kk = 0usize;
    let mut acc0 = _mm512_setzero_ps();
    let mut acc1 = _mm512_setzero_ps();

    while kk + 32 <= k_dim {
        let row_lo = unsafe { load_bf16_as_f32x16_x86(row.as_ptr().add(kk)) };
        let row_hi = unsafe { load_bf16_as_f32x16_x86(row.as_ptr().add(kk + 16)) };
        let x_lo = unsafe { _mm512_loadu_ps(x.as_ptr().add(kk)) };
        let x_hi = unsafe { _mm512_loadu_ps(x.as_ptr().add(kk + 16)) };
        acc0 = _mm512_fmadd_ps(row_lo, x_lo, acc0);
        acc1 = _mm512_fmadd_ps(row_hi, x_hi, acc1);
        kk += 32;
    }

    while kk + 16 <= k_dim {
        let row_chunk = unsafe { load_bf16_as_f32x16_x86(row.as_ptr().add(kk)) };
        let x_chunk = unsafe { _mm512_loadu_ps(x.as_ptr().add(kk)) };
        acc0 = _mm512_fmadd_ps(row_chunk, x_chunk, acc0);
        kk += 16;
    }

    let mut sum = unsafe { reduce_f32x16_x86(acc0) + reduce_f32x16_x86(acc1) };
    while kk < k_dim {
        sum += row[kk].to_f32() * x[kk];
        kk += 1;
    }
    sum
}

#[cfg(all(
    feature = "x86-fp-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[target_feature(enable = "avx512f")]
unsafe fn dot2_f32_bf16_x86_avx512(x: &[f32], row0: &[bf16], row1: &[bf16]) -> (f32, f32) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let k_dim = x.len();
    let mut kk = 0usize;
    let mut acc00 = _mm512_setzero_ps();
    let mut acc01 = _mm512_setzero_ps();
    let mut acc10 = _mm512_setzero_ps();
    let mut acc11 = _mm512_setzero_ps();

    while kk + 32 <= k_dim {
        let x_lo = unsafe { _mm512_loadu_ps(x.as_ptr().add(kk)) };
        let x_hi = unsafe { _mm512_loadu_ps(x.as_ptr().add(kk + 16)) };
        let row0_lo = unsafe { load_bf16_as_f32x16_x86(row0.as_ptr().add(kk)) };
        let row0_hi = unsafe { load_bf16_as_f32x16_x86(row0.as_ptr().add(kk + 16)) };
        let row1_lo = unsafe { load_bf16_as_f32x16_x86(row1.as_ptr().add(kk)) };
        let row1_hi = unsafe { load_bf16_as_f32x16_x86(row1.as_ptr().add(kk + 16)) };
        acc00 = _mm512_fmadd_ps(row0_lo, x_lo, acc00);
        acc01 = _mm512_fmadd_ps(row0_hi, x_hi, acc01);
        acc10 = _mm512_fmadd_ps(row1_lo, x_lo, acc10);
        acc11 = _mm512_fmadd_ps(row1_hi, x_hi, acc11);
        kk += 32;
    }

    while kk + 16 <= k_dim {
        let x_chunk = unsafe { _mm512_loadu_ps(x.as_ptr().add(kk)) };
        let row0_chunk = unsafe { load_bf16_as_f32x16_x86(row0.as_ptr().add(kk)) };
        let row1_chunk = unsafe { load_bf16_as_f32x16_x86(row1.as_ptr().add(kk)) };
        acc00 = _mm512_fmadd_ps(row0_chunk, x_chunk, acc00);
        acc10 = _mm512_fmadd_ps(row1_chunk, x_chunk, acc10);
        kk += 16;
    }

    let mut sum0 = unsafe { reduce_f32x16_x86(acc00) + reduce_f32x16_x86(acc01) };
    let mut sum1 = unsafe { reduce_f32x16_x86(acc10) + reduce_f32x16_x86(acc11) };
    while kk < k_dim {
        let xv = x[kk];
        sum0 += row0[kk].to_f32() * xv;
        sum1 += row1[kk].to_f32() * xv;
        kk += 1;
    }
    (sum0, sum1)
}

#[cfg(all(
    feature = "x86-fp-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[target_feature(enable = "avx512f")]
unsafe fn dot3_f32_bf16_x86_avx512(
    x: &[f32],
    row0: &[bf16],
    row1: &[bf16],
    row2: &[bf16],
) -> (f32, f32, f32) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let k_dim = x.len();
    let mut kk = 0usize;
    let mut acc00 = _mm512_setzero_ps();
    let mut acc01 = _mm512_setzero_ps();
    let mut acc10 = _mm512_setzero_ps();
    let mut acc11 = _mm512_setzero_ps();
    let mut acc20 = _mm512_setzero_ps();
    let mut acc21 = _mm512_setzero_ps();

    while kk + 32 <= k_dim {
        let x_lo = unsafe { _mm512_loadu_ps(x.as_ptr().add(kk)) };
        let x_hi = unsafe { _mm512_loadu_ps(x.as_ptr().add(kk + 16)) };
        let row0_lo = unsafe { load_bf16_as_f32x16_x86(row0.as_ptr().add(kk)) };
        let row0_hi = unsafe { load_bf16_as_f32x16_x86(row0.as_ptr().add(kk + 16)) };
        let row1_lo = unsafe { load_bf16_as_f32x16_x86(row1.as_ptr().add(kk)) };
        let row1_hi = unsafe { load_bf16_as_f32x16_x86(row1.as_ptr().add(kk + 16)) };
        let row2_lo = unsafe { load_bf16_as_f32x16_x86(row2.as_ptr().add(kk)) };
        let row2_hi = unsafe { load_bf16_as_f32x16_x86(row2.as_ptr().add(kk + 16)) };
        acc00 = _mm512_fmadd_ps(row0_lo, x_lo, acc00);
        acc01 = _mm512_fmadd_ps(row0_hi, x_hi, acc01);
        acc10 = _mm512_fmadd_ps(row1_lo, x_lo, acc10);
        acc11 = _mm512_fmadd_ps(row1_hi, x_hi, acc11);
        acc20 = _mm512_fmadd_ps(row2_lo, x_lo, acc20);
        acc21 = _mm512_fmadd_ps(row2_hi, x_hi, acc21);
        kk += 32;
    }

    while kk + 16 <= k_dim {
        let x_chunk = unsafe { _mm512_loadu_ps(x.as_ptr().add(kk)) };
        let row0_chunk = unsafe { load_bf16_as_f32x16_x86(row0.as_ptr().add(kk)) };
        let row1_chunk = unsafe { load_bf16_as_f32x16_x86(row1.as_ptr().add(kk)) };
        let row2_chunk = unsafe { load_bf16_as_f32x16_x86(row2.as_ptr().add(kk)) };
        acc00 = _mm512_fmadd_ps(row0_chunk, x_chunk, acc00);
        acc10 = _mm512_fmadd_ps(row1_chunk, x_chunk, acc10);
        acc20 = _mm512_fmadd_ps(row2_chunk, x_chunk, acc20);
        kk += 16;
    }

    let mut sum0 = unsafe { reduce_f32x16_x86(acc00) + reduce_f32x16_x86(acc01) };
    let mut sum1 = unsafe { reduce_f32x16_x86(acc10) + reduce_f32x16_x86(acc11) };
    let mut sum2 = unsafe { reduce_f32x16_x86(acc20) + reduce_f32x16_x86(acc21) };
    while kk < k_dim {
        let xv = x[kk];
        sum0 += row0[kk].to_f32() * xv;
        sum1 += row1[kk].to_f32() * xv;
        sum2 += row2[kk].to_f32() * xv;
        kk += 1;
    }
    (sum0, sum1, sum2)
}

#[cfg(all(
    feature = "x86-fp-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[target_feature(enable = "avx512f,avx512vl,avx512bf16")]
unsafe fn dot_bf16_bf16_x86_avx512(x: &[bf16], row: &[bf16]) -> f32 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let k_dim = x.len();
    let mut kk = 0usize;
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();

    while kk + 32 <= k_dim {
        let x_lo = unsafe { load_bf16_x16_x86(x.as_ptr().add(kk)) };
        let x_hi = unsafe { load_bf16_x16_x86(x.as_ptr().add(kk + 16)) };
        let row_lo = unsafe { load_bf16_x16_x86(row.as_ptr().add(kk)) };
        let row_hi = unsafe { load_bf16_x16_x86(row.as_ptr().add(kk + 16)) };
        acc0 = _mm256_dpbf16_ps(acc0, row_lo, x_lo);
        acc1 = _mm256_dpbf16_ps(acc1, row_hi, x_hi);
        kk += 32;
    }

    while kk + 16 <= k_dim {
        let x_chunk = unsafe { load_bf16_x16_x86(x.as_ptr().add(kk)) };
        let row_chunk = unsafe { load_bf16_x16_x86(row.as_ptr().add(kk)) };
        acc0 = _mm256_dpbf16_ps(acc0, row_chunk, x_chunk);
        kk += 16;
    }

    let mut sum = unsafe { reduce_f32x8_x86(acc0) + reduce_f32x8_x86(acc1) };
    while kk < k_dim {
        sum += row[kk].to_f32() * x[kk].to_f32();
        kk += 1;
    }
    sum
}

#[cfg(all(
    feature = "x86-fp-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[target_feature(enable = "avx512f,avx512vl,avx512bf16")]
unsafe fn dot2_bf16_bf16_x86_avx512(x: &[bf16], row0: &[bf16], row1: &[bf16]) -> (f32, f32) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let k_dim = x.len();
    let mut kk = 0usize;
    let mut acc00 = _mm256_setzero_ps();
    let mut acc01 = _mm256_setzero_ps();
    let mut acc10 = _mm256_setzero_ps();
    let mut acc11 = _mm256_setzero_ps();

    while kk + 32 <= k_dim {
        let x_lo = unsafe { load_bf16_x16_x86(x.as_ptr().add(kk)) };
        let x_hi = unsafe { load_bf16_x16_x86(x.as_ptr().add(kk + 16)) };
        let row0_lo = unsafe { load_bf16_x16_x86(row0.as_ptr().add(kk)) };
        let row0_hi = unsafe { load_bf16_x16_x86(row0.as_ptr().add(kk + 16)) };
        let row1_lo = unsafe { load_bf16_x16_x86(row1.as_ptr().add(kk)) };
        let row1_hi = unsafe { load_bf16_x16_x86(row1.as_ptr().add(kk + 16)) };
        acc00 = _mm256_dpbf16_ps(acc00, row0_lo, x_lo);
        acc01 = _mm256_dpbf16_ps(acc01, row0_hi, x_hi);
        acc10 = _mm256_dpbf16_ps(acc10, row1_lo, x_lo);
        acc11 = _mm256_dpbf16_ps(acc11, row1_hi, x_hi);
        kk += 32;
    }

    while kk + 16 <= k_dim {
        let x_chunk = unsafe { load_bf16_x16_x86(x.as_ptr().add(kk)) };
        let row0_chunk = unsafe { load_bf16_x16_x86(row0.as_ptr().add(kk)) };
        let row1_chunk = unsafe { load_bf16_x16_x86(row1.as_ptr().add(kk)) };
        acc00 = _mm256_dpbf16_ps(acc00, row0_chunk, x_chunk);
        acc10 = _mm256_dpbf16_ps(acc10, row1_chunk, x_chunk);
        kk += 16;
    }

    let mut sum0 = unsafe { reduce_f32x8_x86(acc00) + reduce_f32x8_x86(acc01) };
    let mut sum1 = unsafe { reduce_f32x8_x86(acc10) + reduce_f32x8_x86(acc11) };
    while kk < k_dim {
        let xv = x[kk].to_f32();
        sum0 += row0[kk].to_f32() * xv;
        sum1 += row1[kk].to_f32() * xv;
        kk += 1;
    }
    (sum0, sum1)
}

#[cfg(all(
    feature = "x86-fp-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[target_feature(enable = "avx512f,avx512vl,avx512bf16")]
unsafe fn dot3_bf16_bf16_x86_avx512(
    x: &[bf16],
    row0: &[bf16],
    row1: &[bf16],
    row2: &[bf16],
) -> (f32, f32, f32) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let k_dim = x.len();
    let mut kk = 0usize;
    let mut acc00 = _mm256_setzero_ps();
    let mut acc01 = _mm256_setzero_ps();
    let mut acc10 = _mm256_setzero_ps();
    let mut acc11 = _mm256_setzero_ps();
    let mut acc20 = _mm256_setzero_ps();
    let mut acc21 = _mm256_setzero_ps();

    while kk + 32 <= k_dim {
        let x_lo = unsafe { load_bf16_x16_x86(x.as_ptr().add(kk)) };
        let x_hi = unsafe { load_bf16_x16_x86(x.as_ptr().add(kk + 16)) };
        let row0_lo = unsafe { load_bf16_x16_x86(row0.as_ptr().add(kk)) };
        let row0_hi = unsafe { load_bf16_x16_x86(row0.as_ptr().add(kk + 16)) };
        let row1_lo = unsafe { load_bf16_x16_x86(row1.as_ptr().add(kk)) };
        let row1_hi = unsafe { load_bf16_x16_x86(row1.as_ptr().add(kk + 16)) };
        let row2_lo = unsafe { load_bf16_x16_x86(row2.as_ptr().add(kk)) };
        let row2_hi = unsafe { load_bf16_x16_x86(row2.as_ptr().add(kk + 16)) };
        acc00 = _mm256_dpbf16_ps(acc00, row0_lo, x_lo);
        acc01 = _mm256_dpbf16_ps(acc01, row0_hi, x_hi);
        acc10 = _mm256_dpbf16_ps(acc10, row1_lo, x_lo);
        acc11 = _mm256_dpbf16_ps(acc11, row1_hi, x_hi);
        acc20 = _mm256_dpbf16_ps(acc20, row2_lo, x_lo);
        acc21 = _mm256_dpbf16_ps(acc21, row2_hi, x_hi);
        kk += 32;
    }

    while kk + 16 <= k_dim {
        let x_chunk = unsafe { load_bf16_x16_x86(x.as_ptr().add(kk)) };
        let row0_chunk = unsafe { load_bf16_x16_x86(row0.as_ptr().add(kk)) };
        let row1_chunk = unsafe { load_bf16_x16_x86(row1.as_ptr().add(kk)) };
        let row2_chunk = unsafe { load_bf16_x16_x86(row2.as_ptr().add(kk)) };
        acc00 = _mm256_dpbf16_ps(acc00, row0_chunk, x_chunk);
        acc10 = _mm256_dpbf16_ps(acc10, row1_chunk, x_chunk);
        acc20 = _mm256_dpbf16_ps(acc20, row2_chunk, x_chunk);
        kk += 16;
    }

    let mut sum0 = unsafe { reduce_f32x8_x86(acc00) + reduce_f32x8_x86(acc01) };
    let mut sum1 = unsafe { reduce_f32x8_x86(acc10) + reduce_f32x8_x86(acc11) };
    let mut sum2 = unsafe { reduce_f32x8_x86(acc20) + reduce_f32x8_x86(acc21) };
    while kk < k_dim {
        let xv = x[kk].to_f32();
        sum0 += row0[kk].to_f32() * xv;
        sum1 += row1[kk].to_f32() * xv;
        sum2 += row2[kk].to_f32() * xv;
        kk += 1;
    }
    (sum0, sum1, sum2)
}

#[cfg(all(
    feature = "x86-fp-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_bf16_bf16_x86_avx2(x: &[bf16], row: &[bf16]) -> f32 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let k_dim = x.len();
    let mut kk = 0usize;
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();

    while kk + 16 <= k_dim {
        let x_lo = unsafe { load_bf16_as_f32x8_x86(x.as_ptr().add(kk)) };
        let x_hi = unsafe { load_bf16_as_f32x8_x86(x.as_ptr().add(kk + 8)) };
        let row_lo = unsafe { load_bf16_as_f32x8_x86(row.as_ptr().add(kk)) };
        let row_hi = unsafe { load_bf16_as_f32x8_x86(row.as_ptr().add(kk + 8)) };
        acc0 = _mm256_fmadd_ps(row_lo, x_lo, acc0);
        acc1 = _mm256_fmadd_ps(row_hi, x_hi, acc1);
        kk += 16;
    }

    while kk + 8 <= k_dim {
        let x_chunk = unsafe { load_bf16_as_f32x8_x86(x.as_ptr().add(kk)) };
        let row_chunk = unsafe { load_bf16_as_f32x8_x86(row.as_ptr().add(kk)) };
        acc0 = _mm256_fmadd_ps(row_chunk, x_chunk, acc0);
        kk += 8;
    }

    let mut sum = unsafe { reduce_f32x8_x86(acc0) + reduce_f32x8_x86(acc1) };
    while kk < k_dim {
        sum += row[kk].to_f32() * x[kk].to_f32();
        kk += 1;
    }
    sum
}

#[cfg(all(
    feature = "x86-fp-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot2_bf16_bf16_x86_avx2(x: &[bf16], row0: &[bf16], row1: &[bf16]) -> (f32, f32) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let k_dim = x.len();
    let mut kk = 0usize;
    let mut acc00 = _mm256_setzero_ps();
    let mut acc01 = _mm256_setzero_ps();
    let mut acc10 = _mm256_setzero_ps();
    let mut acc11 = _mm256_setzero_ps();

    while kk + 16 <= k_dim {
        let x_lo = unsafe { load_bf16_as_f32x8_x86(x.as_ptr().add(kk)) };
        let x_hi = unsafe { load_bf16_as_f32x8_x86(x.as_ptr().add(kk + 8)) };
        let row0_lo = unsafe { load_bf16_as_f32x8_x86(row0.as_ptr().add(kk)) };
        let row0_hi = unsafe { load_bf16_as_f32x8_x86(row0.as_ptr().add(kk + 8)) };
        let row1_lo = unsafe { load_bf16_as_f32x8_x86(row1.as_ptr().add(kk)) };
        let row1_hi = unsafe { load_bf16_as_f32x8_x86(row1.as_ptr().add(kk + 8)) };
        acc00 = _mm256_fmadd_ps(row0_lo, x_lo, acc00);
        acc01 = _mm256_fmadd_ps(row0_hi, x_hi, acc01);
        acc10 = _mm256_fmadd_ps(row1_lo, x_lo, acc10);
        acc11 = _mm256_fmadd_ps(row1_hi, x_hi, acc11);
        kk += 16;
    }

    while kk + 8 <= k_dim {
        let x_chunk = unsafe { load_bf16_as_f32x8_x86(x.as_ptr().add(kk)) };
        let row0_chunk = unsafe { load_bf16_as_f32x8_x86(row0.as_ptr().add(kk)) };
        let row1_chunk = unsafe { load_bf16_as_f32x8_x86(row1.as_ptr().add(kk)) };
        acc00 = _mm256_fmadd_ps(row0_chunk, x_chunk, acc00);
        acc10 = _mm256_fmadd_ps(row1_chunk, x_chunk, acc10);
        kk += 8;
    }

    let mut sum0 = unsafe { reduce_f32x8_x86(acc00) + reduce_f32x8_x86(acc01) };
    let mut sum1 = unsafe { reduce_f32x8_x86(acc10) + reduce_f32x8_x86(acc11) };
    while kk < k_dim {
        let xv = x[kk].to_f32();
        sum0 += row0[kk].to_f32() * xv;
        sum1 += row1[kk].to_f32() * xv;
        kk += 1;
    }
    (sum0, sum1)
}

#[cfg(all(
    feature = "x86-fp-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot3_bf16_bf16_x86_avx2(
    x: &[bf16],
    row0: &[bf16],
    row1: &[bf16],
    row2: &[bf16],
) -> (f32, f32, f32) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let k_dim = x.len();
    let mut kk = 0usize;
    let mut acc00 = _mm256_setzero_ps();
    let mut acc01 = _mm256_setzero_ps();
    let mut acc10 = _mm256_setzero_ps();
    let mut acc11 = _mm256_setzero_ps();
    let mut acc20 = _mm256_setzero_ps();
    let mut acc21 = _mm256_setzero_ps();

    while kk + 16 <= k_dim {
        let x_lo = unsafe { load_bf16_as_f32x8_x86(x.as_ptr().add(kk)) };
        let x_hi = unsafe { load_bf16_as_f32x8_x86(x.as_ptr().add(kk + 8)) };
        let row0_lo = unsafe { load_bf16_as_f32x8_x86(row0.as_ptr().add(kk)) };
        let row0_hi = unsafe { load_bf16_as_f32x8_x86(row0.as_ptr().add(kk + 8)) };
        let row1_lo = unsafe { load_bf16_as_f32x8_x86(row1.as_ptr().add(kk)) };
        let row1_hi = unsafe { load_bf16_as_f32x8_x86(row1.as_ptr().add(kk + 8)) };
        let row2_lo = unsafe { load_bf16_as_f32x8_x86(row2.as_ptr().add(kk)) };
        let row2_hi = unsafe { load_bf16_as_f32x8_x86(row2.as_ptr().add(kk + 8)) };
        acc00 = _mm256_fmadd_ps(row0_lo, x_lo, acc00);
        acc01 = _mm256_fmadd_ps(row0_hi, x_hi, acc01);
        acc10 = _mm256_fmadd_ps(row1_lo, x_lo, acc10);
        acc11 = _mm256_fmadd_ps(row1_hi, x_hi, acc11);
        acc20 = _mm256_fmadd_ps(row2_lo, x_lo, acc20);
        acc21 = _mm256_fmadd_ps(row2_hi, x_hi, acc21);
        kk += 16;
    }

    while kk + 8 <= k_dim {
        let x_chunk = unsafe { load_bf16_as_f32x8_x86(x.as_ptr().add(kk)) };
        let row0_chunk = unsafe { load_bf16_as_f32x8_x86(row0.as_ptr().add(kk)) };
        let row1_chunk = unsafe { load_bf16_as_f32x8_x86(row1.as_ptr().add(kk)) };
        let row2_chunk = unsafe { load_bf16_as_f32x8_x86(row2.as_ptr().add(kk)) };
        acc00 = _mm256_fmadd_ps(row0_chunk, x_chunk, acc00);
        acc10 = _mm256_fmadd_ps(row1_chunk, x_chunk, acc10);
        acc20 = _mm256_fmadd_ps(row2_chunk, x_chunk, acc20);
        kk += 8;
    }

    let mut sum0 = unsafe { reduce_f32x8_x86(acc00) + reduce_f32x8_x86(acc01) };
    let mut sum1 = unsafe { reduce_f32x8_x86(acc10) + reduce_f32x8_x86(acc11) };
    let mut sum2 = unsafe { reduce_f32x8_x86(acc20) + reduce_f32x8_x86(acc21) };
    while kk < k_dim {
        let xv = x[kk].to_f32();
        sum0 += row0[kk].to_f32() * xv;
        sum1 += row1[kk].to_f32() * xv;
        sum2 += row2[kk].to_f32() * xv;
        kk += 1;
    }
    (sum0, sum1, sum2)
}

#[cfg(all(
    feature = "x86-fp-kernels-nightly",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[inline]
#[target_feature(enable = "avx512fp16")]
unsafe fn load_f16_x32_x86(ptr: *const f16) -> __m512h {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    unsafe { _mm512_loadu_ph(ptr as *const _) }
}

#[cfg(all(
    feature = "x86-fp-kernels-nightly",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx512fp16")]
unsafe fn store_f16_x32_x86(ptr: *mut f16, v: __m512h) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    unsafe { _mm512_storeu_ph(ptr as *mut _, v) };
}

#[cfg(all(
    feature = "x86-fp-kernels-nightly",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[inline]
#[target_feature(enable = "avx512fp16")]
unsafe fn reduce_f16x32_as_f32_x86(v: __m512h) -> f32 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    _mm512_reduce_add_ph(v) as f32
}

#[cfg(all(
    feature = "x86-fp-kernels-nightly",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[target_feature(enable = "avx512fp16")]
unsafe fn dot_f16_f16_x86_avx512(x: &[f16], row: &[f16]) -> f32 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let k_dim = x.len();
    let mut kk = 0usize;
    let mut acc0 = _mm512_setzero_ph();
    let mut acc1 = _mm512_setzero_ph();

    while kk + 64 <= k_dim {
        let x_lo = unsafe { load_f16_x32_x86(x.as_ptr().add(kk)) };
        let x_hi = unsafe { load_f16_x32_x86(x.as_ptr().add(kk + 32)) };
        let row_lo = unsafe { load_f16_x32_x86(row.as_ptr().add(kk)) };
        let row_hi = unsafe { load_f16_x32_x86(row.as_ptr().add(kk + 32)) };
        acc0 = _mm512_fmadd_ph(row_lo, x_lo, acc0);
        acc1 = _mm512_fmadd_ph(row_hi, x_hi, acc1);
        kk += 64;
    }

    while kk + 32 <= k_dim {
        let x_chunk = unsafe { load_f16_x32_x86(x.as_ptr().add(kk)) };
        let row_chunk = unsafe { load_f16_x32_x86(row.as_ptr().add(kk)) };
        acc0 = _mm512_fmadd_ph(row_chunk, x_chunk, acc0);
        kk += 32;
    }

    let mut sum = unsafe { reduce_f16x32_as_f32_x86(acc0) + reduce_f16x32_as_f32_x86(acc1) };
    while kk < k_dim {
        sum += row[kk].to_f32() * x[kk].to_f32();
        kk += 1;
    }
    sum
}

#[cfg(all(
    feature = "x86-fp-kernels-nightly",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[target_feature(enable = "avx512fp16")]
unsafe fn dot2_f16_f16_x86_avx512(x: &[f16], row0: &[f16], row1: &[f16]) -> (f32, f32) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let k_dim = x.len();
    let mut kk = 0usize;
    let mut acc00 = _mm512_setzero_ph();
    let mut acc01 = _mm512_setzero_ph();
    let mut acc10 = _mm512_setzero_ph();
    let mut acc11 = _mm512_setzero_ph();

    while kk + 64 <= k_dim {
        let x_lo = unsafe { load_f16_x32_x86(x.as_ptr().add(kk)) };
        let x_hi = unsafe { load_f16_x32_x86(x.as_ptr().add(kk + 32)) };
        let row0_lo = unsafe { load_f16_x32_x86(row0.as_ptr().add(kk)) };
        let row0_hi = unsafe { load_f16_x32_x86(row0.as_ptr().add(kk + 32)) };
        let row1_lo = unsafe { load_f16_x32_x86(row1.as_ptr().add(kk)) };
        let row1_hi = unsafe { load_f16_x32_x86(row1.as_ptr().add(kk + 32)) };
        acc00 = _mm512_fmadd_ph(row0_lo, x_lo, acc00);
        acc01 = _mm512_fmadd_ph(row0_hi, x_hi, acc01);
        acc10 = _mm512_fmadd_ph(row1_lo, x_lo, acc10);
        acc11 = _mm512_fmadd_ph(row1_hi, x_hi, acc11);
        kk += 64;
    }

    while kk + 32 <= k_dim {
        let x_chunk = unsafe { load_f16_x32_x86(x.as_ptr().add(kk)) };
        let row0_chunk = unsafe { load_f16_x32_x86(row0.as_ptr().add(kk)) };
        let row1_chunk = unsafe { load_f16_x32_x86(row1.as_ptr().add(kk)) };
        acc00 = _mm512_fmadd_ph(row0_chunk, x_chunk, acc00);
        acc10 = _mm512_fmadd_ph(row1_chunk, x_chunk, acc10);
        kk += 32;
    }

    let mut sum0 = unsafe { reduce_f16x32_as_f32_x86(acc00) + reduce_f16x32_as_f32_x86(acc01) };
    let mut sum1 = unsafe { reduce_f16x32_as_f32_x86(acc10) + reduce_f16x32_as_f32_x86(acc11) };
    while kk < k_dim {
        let xv = x[kk].to_f32();
        sum0 += row0[kk].to_f32() * xv;
        sum1 += row1[kk].to_f32() * xv;
        kk += 1;
    }
    (sum0, sum1)
}

#[cfg(all(
    feature = "x86-fp-kernels-nightly",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[target_feature(enable = "avx512fp16")]
unsafe fn dot3_f16_f16_x86_avx512(
    x: &[f16],
    row0: &[f16],
    row1: &[f16],
    row2: &[f16],
) -> (f32, f32, f32) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let k_dim = x.len();
    let mut kk = 0usize;
    let mut acc00 = _mm512_setzero_ph();
    let mut acc01 = _mm512_setzero_ph();
    let mut acc10 = _mm512_setzero_ph();
    let mut acc11 = _mm512_setzero_ph();
    let mut acc20 = _mm512_setzero_ph();
    let mut acc21 = _mm512_setzero_ph();

    while kk + 64 <= k_dim {
        let x_lo = unsafe { load_f16_x32_x86(x.as_ptr().add(kk)) };
        let x_hi = unsafe { load_f16_x32_x86(x.as_ptr().add(kk + 32)) };
        let row0_lo = unsafe { load_f16_x32_x86(row0.as_ptr().add(kk)) };
        let row0_hi = unsafe { load_f16_x32_x86(row0.as_ptr().add(kk + 32)) };
        let row1_lo = unsafe { load_f16_x32_x86(row1.as_ptr().add(kk)) };
        let row1_hi = unsafe { load_f16_x32_x86(row1.as_ptr().add(kk + 32)) };
        let row2_lo = unsafe { load_f16_x32_x86(row2.as_ptr().add(kk)) };
        let row2_hi = unsafe { load_f16_x32_x86(row2.as_ptr().add(kk + 32)) };
        acc00 = _mm512_fmadd_ph(row0_lo, x_lo, acc00);
        acc01 = _mm512_fmadd_ph(row0_hi, x_hi, acc01);
        acc10 = _mm512_fmadd_ph(row1_lo, x_lo, acc10);
        acc11 = _mm512_fmadd_ph(row1_hi, x_hi, acc11);
        acc20 = _mm512_fmadd_ph(row2_lo, x_lo, acc20);
        acc21 = _mm512_fmadd_ph(row2_hi, x_hi, acc21);
        kk += 64;
    }

    while kk + 32 <= k_dim {
        let x_chunk = unsafe { load_f16_x32_x86(x.as_ptr().add(kk)) };
        let row0_chunk = unsafe { load_f16_x32_x86(row0.as_ptr().add(kk)) };
        let row1_chunk = unsafe { load_f16_x32_x86(row1.as_ptr().add(kk)) };
        let row2_chunk = unsafe { load_f16_x32_x86(row2.as_ptr().add(kk)) };
        acc00 = _mm512_fmadd_ph(row0_chunk, x_chunk, acc00);
        acc10 = _mm512_fmadd_ph(row1_chunk, x_chunk, acc10);
        acc20 = _mm512_fmadd_ph(row2_chunk, x_chunk, acc20);
        kk += 32;
    }

    let mut sum0 = unsafe { reduce_f16x32_as_f32_x86(acc00) + reduce_f16x32_as_f32_x86(acc01) };
    let mut sum1 = unsafe { reduce_f16x32_as_f32_x86(acc10) + reduce_f16x32_as_f32_x86(acc11) };
    let mut sum2 = unsafe { reduce_f16x32_as_f32_x86(acc20) + reduce_f16x32_as_f32_x86(acc21) };
    while kk < k_dim {
        let xv = x[kk].to_f32();
        sum0 += row0[kk].to_f32() * xv;
        sum1 += row1[kk].to_f32() * xv;
        sum2 += row2[kk].to_f32() * xv;
        kk += 1;
    }
    (sum0, sum1, sum2)
}

#[cfg(all(
    feature = "x86-fp-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn reduce_f32x8_x86(v: __m256) -> f32 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let mut buf = [0.0f32; 8];
    unsafe {
        _mm256_storeu_ps(buf.as_mut_ptr(), v);
    }
    buf.iter().sum()
}

#[cfg(all(
    feature = "x86-fp-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[inline]
#[target_feature(enable = "avx2,f16c")]
unsafe fn load_f16_as_f32x8_x86(ptr: *const f16) -> __m256 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let raw = unsafe { _mm_loadu_si128(ptr as *const __m128i) };
    _mm256_cvtph_ps(raw)
}

#[cfg(all(
    feature = "x86-fp-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[inline]
#[target_feature(enable = "avx2,f16c")]
unsafe fn store_f16_from_f32x8_x86(ptr: *mut f16, v: __m256) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let raw = _mm256_cvtps_ph::<{ _MM_FROUND_TO_NEAREST_INT }>(v);
    unsafe { _mm_storeu_si128(ptr as *mut __m128i, raw) };
}

#[cfg(all(
    feature = "x86-fp-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn load_bf16_as_f32x8_x86(ptr: *const bf16) -> __m256 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let raw = unsafe { _mm_loadu_si128(ptr as *const __m128i) };
    let widened = _mm256_cvtepu16_epi32(raw);
    let bits = _mm256_slli_epi32(widened, 16);
    _mm256_castsi256_ps(bits)
}

#[cfg(all(
    feature = "x86-fp-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn store_bf16_from_f32x8_x86(ptr: *mut bf16, v: __m256) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let bits = _mm256_castps_si256(v);
    let lsb = _mm256_and_si256(_mm256_srli_epi32(bits, 16), _mm256_set1_epi32(1));
    let bias = _mm256_add_epi32(_mm256_set1_epi32(0x7fff), lsb);
    let rounded = _mm256_add_epi32(bits, bias);
    let bf16_bits = _mm256_srli_epi32(rounded, 16);
    let packed = _mm256_packus_epi32(bf16_bits, bf16_bits);
    let ordered = _mm256_permute4x64_epi64::<0b1101_1000>(packed);
    let low = _mm256_castsi256_si128(ordered);
    unsafe { _mm_storeu_si128(ptr as *mut __m128i, low) };
}

#[cfg(all(
    feature = "x86-fp-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn apply_f32_lowp_op_x86_avx2(
    lhs: __m256,
    rhs: __m256,
    lowp_on_lhs: bool,
    op: F32LowpElementwiseOp,
) -> __m256 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    match op {
        F32LowpElementwiseOp::Add => _mm256_add_ps(lhs, rhs),
        F32LowpElementwiseOp::Sub if lowp_on_lhs => _mm256_sub_ps(rhs, lhs),
        F32LowpElementwiseOp::Sub => _mm256_sub_ps(lhs, rhs),
    }
}

#[cfg(all(
    feature = "x86-fp-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[target_feature(enable = "avx2,f16c")]
unsafe fn f32_f16_to_f32_x86_avx2(
    lhs: &[f32],
    rhs: &[f16],
    lowp_on_lhs: bool,
    out: &mut [f32],
    op: F32LowpElementwiseOp,
) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let len = lhs.len();
    let mut kk = 0usize;
    while kk + 16 <= len {
        let a0 = unsafe { _mm256_loadu_ps(lhs.as_ptr().add(kk)) };
        let a1 = unsafe { _mm256_loadu_ps(lhs.as_ptr().add(kk + 8)) };
        let b0 = unsafe { load_f16_as_f32x8_x86(rhs.as_ptr().add(kk)) };
        let b1 = unsafe { load_f16_as_f32x8_x86(rhs.as_ptr().add(kk + 8)) };
        let y0 = unsafe { apply_f32_lowp_op_x86_avx2(a0, b0, lowp_on_lhs, op) };
        let y1 = unsafe { apply_f32_lowp_op_x86_avx2(a1, b1, lowp_on_lhs, op) };
        unsafe { _mm256_storeu_ps(out.as_mut_ptr().add(kk), y0) };
        unsafe { _mm256_storeu_ps(out.as_mut_ptr().add(kk + 8), y1) };
        kk += 16;
    }
    while kk + 8 <= len {
        let a = unsafe { _mm256_loadu_ps(lhs.as_ptr().add(kk)) };
        let b = unsafe { load_f16_as_f32x8_x86(rhs.as_ptr().add(kk)) };
        let y = unsafe { apply_f32_lowp_op_x86_avx2(a, b, lowp_on_lhs, op) };
        unsafe { _mm256_storeu_ps(out.as_mut_ptr().add(kk), y) };
        kk += 8;
    }
    while kk < len {
        out[kk] = match op {
            F32LowpElementwiseOp::Add => lhs[kk] + rhs[kk].to_f32(),
            F32LowpElementwiseOp::Sub if lowp_on_lhs => rhs[kk].to_f32() - lhs[kk],
            F32LowpElementwiseOp::Sub => lhs[kk] - rhs[kk].to_f32(),
        };
        kk += 1;
    }
}

#[cfg(all(
    feature = "x86-fp-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[target_feature(enable = "avx2")]
unsafe fn f32_bf16_to_f32_x86_avx2(
    lhs: &[f32],
    rhs: &[bf16],
    lowp_on_lhs: bool,
    out: &mut [f32],
    op: F32LowpElementwiseOp,
) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let len = lhs.len();
    let mut kk = 0usize;
    while kk + 16 <= len {
        let a0 = unsafe { _mm256_loadu_ps(lhs.as_ptr().add(kk)) };
        let a1 = unsafe { _mm256_loadu_ps(lhs.as_ptr().add(kk + 8)) };
        let b0 = unsafe { load_bf16_as_f32x8_x86(rhs.as_ptr().add(kk)) };
        let b1 = unsafe { load_bf16_as_f32x8_x86(rhs.as_ptr().add(kk + 8)) };
        let y0 = unsafe { apply_f32_lowp_op_x86_avx2(a0, b0, lowp_on_lhs, op) };
        let y1 = unsafe { apply_f32_lowp_op_x86_avx2(a1, b1, lowp_on_lhs, op) };
        unsafe { _mm256_storeu_ps(out.as_mut_ptr().add(kk), y0) };
        unsafe { _mm256_storeu_ps(out.as_mut_ptr().add(kk + 8), y1) };
        kk += 16;
    }
    while kk + 8 <= len {
        let a = unsafe { _mm256_loadu_ps(lhs.as_ptr().add(kk)) };
        let b = unsafe { load_bf16_as_f32x8_x86(rhs.as_ptr().add(kk)) };
        let y = unsafe { apply_f32_lowp_op_x86_avx2(a, b, lowp_on_lhs, op) };
        unsafe { _mm256_storeu_ps(out.as_mut_ptr().add(kk), y) };
        kk += 8;
    }
    while kk < len {
        out[kk] = match op {
            F32LowpElementwiseOp::Add => lhs[kk] + rhs[kk].to_f32(),
            F32LowpElementwiseOp::Sub if lowp_on_lhs => rhs[kk].to_f32() - lhs[kk],
            F32LowpElementwiseOp::Sub => lhs[kk] - rhs[kk].to_f32(),
        };
        kk += 1;
    }
}

#[cfg(all(
    feature = "x86-fp-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[target_feature(enable = "avx2,f16c")]
unsafe fn mul_f32_f16_to_f32_x86_avx2(lhs: &[f32], rhs: &[f16], out: &mut [f32]) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let len = lhs.len();
    let mut kk = 0usize;
    while kk + 16 <= len {
        let a0 = unsafe { _mm256_loadu_ps(lhs.as_ptr().add(kk)) };
        let a1 = unsafe { _mm256_loadu_ps(lhs.as_ptr().add(kk + 8)) };
        let b0 = unsafe { load_f16_as_f32x8_x86(rhs.as_ptr().add(kk)) };
        let b1 = unsafe { load_f16_as_f32x8_x86(rhs.as_ptr().add(kk + 8)) };
        unsafe { _mm256_storeu_ps(out.as_mut_ptr().add(kk), _mm256_mul_ps(a0, b0)) };
        unsafe { _mm256_storeu_ps(out.as_mut_ptr().add(kk + 8), _mm256_mul_ps(a1, b1)) };
        kk += 16;
    }
    while kk + 8 <= len {
        let a = unsafe { _mm256_loadu_ps(lhs.as_ptr().add(kk)) };
        let b = unsafe { load_f16_as_f32x8_x86(rhs.as_ptr().add(kk)) };
        unsafe { _mm256_storeu_ps(out.as_mut_ptr().add(kk), _mm256_mul_ps(a, b)) };
        kk += 8;
    }
    while kk < len {
        out[kk] = lhs[kk] * rhs[kk].to_f32();
        kk += 1;
    }
}

#[cfg(all(
    feature = "x86-fp-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[target_feature(enable = "avx2")]
unsafe fn mul_f32_bf16_to_f32_x86_avx2(lhs: &[f32], rhs: &[bf16], out: &mut [f32]) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let len = lhs.len();
    let mut kk = 0usize;
    while kk + 16 <= len {
        let a0 = unsafe { _mm256_loadu_ps(lhs.as_ptr().add(kk)) };
        let a1 = unsafe { _mm256_loadu_ps(lhs.as_ptr().add(kk + 8)) };
        let b0 = unsafe { load_bf16_as_f32x8_x86(rhs.as_ptr().add(kk)) };
        let b1 = unsafe { load_bf16_as_f32x8_x86(rhs.as_ptr().add(kk + 8)) };
        unsafe { _mm256_storeu_ps(out.as_mut_ptr().add(kk), _mm256_mul_ps(a0, b0)) };
        unsafe { _mm256_storeu_ps(out.as_mut_ptr().add(kk + 8), _mm256_mul_ps(a1, b1)) };
        kk += 16;
    }
    while kk + 8 <= len {
        let a = unsafe { _mm256_loadu_ps(lhs.as_ptr().add(kk)) };
        let b = unsafe { load_bf16_as_f32x8_x86(rhs.as_ptr().add(kk)) };
        unsafe { _mm256_storeu_ps(out.as_mut_ptr().add(kk), _mm256_mul_ps(a, b)) };
        kk += 8;
    }
    while kk < len {
        out[kk] = lhs[kk] * rhs[kk].to_f32();
        kk += 1;
    }
}

#[cfg(all(
    feature = "x86-fp-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[target_feature(enable = "avx2,f16c")]
unsafe fn f16_scalar_to_f32_x86_avx2(
    lhs: &[f16],
    rhs: f32,
    lowp_on_lhs: bool,
    out: &mut [f32],
    op: F32LowpElementwiseOp,
) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let len = lhs.len();
    let rhs_v = _mm256_set1_ps(rhs);
    let mut kk = 0usize;
    while kk + 16 <= len {
        let a0 = unsafe { load_f16_as_f32x8_x86(lhs.as_ptr().add(kk)) };
        let a1 = unsafe { load_f16_as_f32x8_x86(lhs.as_ptr().add(kk + 8)) };
        let y0 = unsafe { apply_f32_lowp_op_x86_avx2(rhs_v, a0, lowp_on_lhs, op) };
        let y1 = unsafe { apply_f32_lowp_op_x86_avx2(rhs_v, a1, lowp_on_lhs, op) };
        unsafe { _mm256_storeu_ps(out.as_mut_ptr().add(kk), y0) };
        unsafe { _mm256_storeu_ps(out.as_mut_ptr().add(kk + 8), y1) };
        kk += 16;
    }
    while kk + 8 <= len {
        let a = unsafe { load_f16_as_f32x8_x86(lhs.as_ptr().add(kk)) };
        let y = unsafe { apply_f32_lowp_op_x86_avx2(rhs_v, a, lowp_on_lhs, op) };
        unsafe { _mm256_storeu_ps(out.as_mut_ptr().add(kk), y) };
        kk += 8;
    }
    while kk < len {
        out[kk] = match op {
            F32LowpElementwiseOp::Add => rhs + lhs[kk].to_f32(),
            F32LowpElementwiseOp::Sub if lowp_on_lhs => lhs[kk].to_f32() - rhs,
            F32LowpElementwiseOp::Sub => rhs - lhs[kk].to_f32(),
        };
        kk += 1;
    }
}

#[cfg(all(
    feature = "x86-fp-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[target_feature(enable = "avx2")]
unsafe fn bf16_scalar_to_f32_x86_avx2(
    lhs: &[bf16],
    rhs: f32,
    lowp_on_lhs: bool,
    out: &mut [f32],
    op: F32LowpElementwiseOp,
) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let len = lhs.len();
    let rhs_v = _mm256_set1_ps(rhs);
    let mut kk = 0usize;
    while kk + 16 <= len {
        let a0 = unsafe { load_bf16_as_f32x8_x86(lhs.as_ptr().add(kk)) };
        let a1 = unsafe { load_bf16_as_f32x8_x86(lhs.as_ptr().add(kk + 8)) };
        let y0 = unsafe { apply_f32_lowp_op_x86_avx2(rhs_v, a0, lowp_on_lhs, op) };
        let y1 = unsafe { apply_f32_lowp_op_x86_avx2(rhs_v, a1, lowp_on_lhs, op) };
        unsafe { _mm256_storeu_ps(out.as_mut_ptr().add(kk), y0) };
        unsafe { _mm256_storeu_ps(out.as_mut_ptr().add(kk + 8), y1) };
        kk += 16;
    }
    while kk + 8 <= len {
        let a = unsafe { load_bf16_as_f32x8_x86(lhs.as_ptr().add(kk)) };
        let y = unsafe { apply_f32_lowp_op_x86_avx2(rhs_v, a, lowp_on_lhs, op) };
        unsafe { _mm256_storeu_ps(out.as_mut_ptr().add(kk), y) };
        kk += 8;
    }
    while kk < len {
        out[kk] = match op {
            F32LowpElementwiseOp::Add => rhs + lhs[kk].to_f32(),
            F32LowpElementwiseOp::Sub if lowp_on_lhs => lhs[kk].to_f32() - rhs,
            F32LowpElementwiseOp::Sub => rhs - lhs[kk].to_f32(),
        };
        kk += 1;
    }
}

#[cfg(all(
    feature = "x86-fp-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[target_feature(enable = "avx2,f16c")]
unsafe fn mul_f16_scalar_to_f32_x86_avx2(lhs: &[f16], rhs: f32, out: &mut [f32]) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let len = lhs.len();
    let rhs_v = _mm256_set1_ps(rhs);
    let mut kk = 0usize;
    while kk + 16 <= len {
        let a0 = unsafe { load_f16_as_f32x8_x86(lhs.as_ptr().add(kk)) };
        let a1 = unsafe { load_f16_as_f32x8_x86(lhs.as_ptr().add(kk + 8)) };
        unsafe { _mm256_storeu_ps(out.as_mut_ptr().add(kk), _mm256_mul_ps(a0, rhs_v)) };
        unsafe { _mm256_storeu_ps(out.as_mut_ptr().add(kk + 8), _mm256_mul_ps(a1, rhs_v)) };
        kk += 16;
    }
    while kk + 8 <= len {
        let a = unsafe { load_f16_as_f32x8_x86(lhs.as_ptr().add(kk)) };
        unsafe { _mm256_storeu_ps(out.as_mut_ptr().add(kk), _mm256_mul_ps(a, rhs_v)) };
        kk += 8;
    }
    while kk < len {
        out[kk] = lhs[kk].to_f32() * rhs;
        kk += 1;
    }
}

#[cfg(all(
    feature = "x86-fp-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[target_feature(enable = "avx2")]
unsafe fn mul_bf16_scalar_to_f32_x86_avx2(lhs: &[bf16], rhs: f32, out: &mut [f32]) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let len = lhs.len();
    let rhs_v = _mm256_set1_ps(rhs);
    let mut kk = 0usize;
    while kk + 16 <= len {
        let a0 = unsafe { load_bf16_as_f32x8_x86(lhs.as_ptr().add(kk)) };
        let a1 = unsafe { load_bf16_as_f32x8_x86(lhs.as_ptr().add(kk + 8)) };
        unsafe { _mm256_storeu_ps(out.as_mut_ptr().add(kk), _mm256_mul_ps(a0, rhs_v)) };
        unsafe { _mm256_storeu_ps(out.as_mut_ptr().add(kk + 8), _mm256_mul_ps(a1, rhs_v)) };
        kk += 16;
    }
    while kk + 8 <= len {
        let a = unsafe { load_bf16_as_f32x8_x86(lhs.as_ptr().add(kk)) };
        unsafe { _mm256_storeu_ps(out.as_mut_ptr().add(kk), _mm256_mul_ps(a, rhs_v)) };
        kk += 8;
    }
    while kk < len {
        out[kk] = lhs[kk].to_f32() * rhs;
        kk += 1;
    }
}

#[cfg(all(
    feature = "x86-fp-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[target_feature(enable = "avx2,f16c")]
unsafe fn elementwise_f16_f16_to_f32_x86_avx2(
    lhs: &[f16],
    rhs: &[f16],
    out: &mut [f32],
    op: FpElementwiseOp,
) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let len = lhs.len();
    let mut kk = 0usize;
    while kk + 16 <= len {
        let a0 = unsafe { load_f16_as_f32x8_x86(lhs.as_ptr().add(kk)) };
        let a1 = unsafe { load_f16_as_f32x8_x86(lhs.as_ptr().add(kk + 8)) };
        let b0 = unsafe { load_f16_as_f32x8_x86(rhs.as_ptr().add(kk)) };
        let b1 = unsafe { load_f16_as_f32x8_x86(rhs.as_ptr().add(kk + 8)) };
        let y0 = match op {
            FpElementwiseOp::Add => _mm256_add_ps(a0, b0),
            FpElementwiseOp::Sub => _mm256_sub_ps(a0, b0),
            FpElementwiseOp::Mul => _mm256_mul_ps(a0, b0),
        };
        let y1 = match op {
            FpElementwiseOp::Add => _mm256_add_ps(a1, b1),
            FpElementwiseOp::Sub => _mm256_sub_ps(a1, b1),
            FpElementwiseOp::Mul => _mm256_mul_ps(a1, b1),
        };
        unsafe { _mm256_storeu_ps(out.as_mut_ptr().add(kk), y0) };
        unsafe { _mm256_storeu_ps(out.as_mut_ptr().add(kk + 8), y1) };
        kk += 16;
    }
    while kk + 8 <= len {
        let a = unsafe { load_f16_as_f32x8_x86(lhs.as_ptr().add(kk)) };
        let b = unsafe { load_f16_as_f32x8_x86(rhs.as_ptr().add(kk)) };
        let y = match op {
            FpElementwiseOp::Add => _mm256_add_ps(a, b),
            FpElementwiseOp::Sub => _mm256_sub_ps(a, b),
            FpElementwiseOp::Mul => _mm256_mul_ps(a, b),
        };
        unsafe { _mm256_storeu_ps(out.as_mut_ptr().add(kk), y) };
        kk += 8;
    }
    while kk < len {
        out[kk] = match op {
            FpElementwiseOp::Add => lhs[kk].to_f32() + rhs[kk].to_f32(),
            FpElementwiseOp::Sub => lhs[kk].to_f32() - rhs[kk].to_f32(),
            FpElementwiseOp::Mul => lhs[kk].to_f32() * rhs[kk].to_f32(),
        };
        kk += 1;
    }
}

#[cfg(all(
    feature = "x86-fp-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[target_feature(enable = "avx2")]
unsafe fn elementwise_bf16_bf16_to_f32_x86_avx2(
    lhs: &[bf16],
    rhs: &[bf16],
    out: &mut [f32],
    op: FpElementwiseOp,
) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let len = lhs.len();
    let mut kk = 0usize;
    while kk + 16 <= len {
        let a0 = unsafe { load_bf16_as_f32x8_x86(lhs.as_ptr().add(kk)) };
        let a1 = unsafe { load_bf16_as_f32x8_x86(lhs.as_ptr().add(kk + 8)) };
        let b0 = unsafe { load_bf16_as_f32x8_x86(rhs.as_ptr().add(kk)) };
        let b1 = unsafe { load_bf16_as_f32x8_x86(rhs.as_ptr().add(kk + 8)) };
        let y0 = match op {
            FpElementwiseOp::Add => _mm256_add_ps(a0, b0),
            FpElementwiseOp::Sub => _mm256_sub_ps(a0, b0),
            FpElementwiseOp::Mul => _mm256_mul_ps(a0, b0),
        };
        let y1 = match op {
            FpElementwiseOp::Add => _mm256_add_ps(a1, b1),
            FpElementwiseOp::Sub => _mm256_sub_ps(a1, b1),
            FpElementwiseOp::Mul => _mm256_mul_ps(a1, b1),
        };
        unsafe { _mm256_storeu_ps(out.as_mut_ptr().add(kk), y0) };
        unsafe { _mm256_storeu_ps(out.as_mut_ptr().add(kk + 8), y1) };
        kk += 16;
    }
    while kk + 8 <= len {
        let a = unsafe { load_bf16_as_f32x8_x86(lhs.as_ptr().add(kk)) };
        let b = unsafe { load_bf16_as_f32x8_x86(rhs.as_ptr().add(kk)) };
        let y = match op {
            FpElementwiseOp::Add => _mm256_add_ps(a, b),
            FpElementwiseOp::Sub => _mm256_sub_ps(a, b),
            FpElementwiseOp::Mul => _mm256_mul_ps(a, b),
        };
        unsafe { _mm256_storeu_ps(out.as_mut_ptr().add(kk), y) };
        kk += 8;
    }
    while kk < len {
        out[kk] = match op {
            FpElementwiseOp::Add => lhs[kk].to_f32() + rhs[kk].to_f32(),
            FpElementwiseOp::Sub => lhs[kk].to_f32() - rhs[kk].to_f32(),
            FpElementwiseOp::Mul => lhs[kk].to_f32() * rhs[kk].to_f32(),
        };
        kk += 1;
    }
}

#[cfg(all(
    feature = "x86-fp-kernels-nightly",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[target_feature(enable = "avx512fp16")]
unsafe fn elementwise_f16_f16_x86_avx512(
    lhs: &[f16],
    rhs: &[f16],
    out: &mut [f16],
    op: FpElementwiseOp,
) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let len = lhs.len();
    let mut kk = 0usize;
    while kk + 32 <= len {
        let a = unsafe { load_f16_x32_x86(lhs.as_ptr().add(kk)) };
        let b = unsafe { load_f16_x32_x86(rhs.as_ptr().add(kk)) };
        let y = match op {
            FpElementwiseOp::Add => _mm512_add_ph(a, b),
            FpElementwiseOp::Sub => _mm512_sub_ph(a, b),
            FpElementwiseOp::Mul => _mm512_mul_ph(a, b),
        };
        unsafe { store_f16_x32_x86(out.as_mut_ptr().add(kk), y) };
        kk += 32;
    }

    while kk < len {
        out[kk] = match op {
            FpElementwiseOp::Add => f16::from_f32(lhs[kk].to_f32() + rhs[kk].to_f32()),
            FpElementwiseOp::Sub => f16::from_f32(lhs[kk].to_f32() - rhs[kk].to_f32()),
            FpElementwiseOp::Mul => f16::from_f32(lhs[kk].to_f32() * rhs[kk].to_f32()),
        };
        kk += 1;
    }
}

#[cfg(all(
    feature = "x86-fp-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn elementwise_f16_f16_x86_avx2(
    lhs: &[f16],
    rhs: &[f16],
    out: &mut [f16],
    op: FpElementwiseOp,
) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let len = lhs.len();
    let mut kk = 0usize;
    while kk + 16 <= len {
        let a0 = unsafe { load_f16_as_f32x8_x86(lhs.as_ptr().add(kk)) };
        let a1 = unsafe { load_f16_as_f32x8_x86(lhs.as_ptr().add(kk + 8)) };
        let b0 = unsafe { load_f16_as_f32x8_x86(rhs.as_ptr().add(kk)) };
        let b1 = unsafe { load_f16_as_f32x8_x86(rhs.as_ptr().add(kk + 8)) };
        let y0 = match op {
            FpElementwiseOp::Add => _mm256_add_ps(a0, b0),
            FpElementwiseOp::Sub => _mm256_sub_ps(a0, b0),
            FpElementwiseOp::Mul => _mm256_mul_ps(a0, b0),
        };
        let y1 = match op {
            FpElementwiseOp::Add => _mm256_add_ps(a1, b1),
            FpElementwiseOp::Sub => _mm256_sub_ps(a1, b1),
            FpElementwiseOp::Mul => _mm256_mul_ps(a1, b1),
        };
        unsafe { store_f16_from_f32x8_x86(out.as_mut_ptr().add(kk), y0) };
        unsafe { store_f16_from_f32x8_x86(out.as_mut_ptr().add(kk + 8), y1) };
        kk += 16;
    }

    while kk + 8 <= len {
        let a = unsafe { load_f16_as_f32x8_x86(lhs.as_ptr().add(kk)) };
        let b = unsafe { load_f16_as_f32x8_x86(rhs.as_ptr().add(kk)) };
        let y = match op {
            FpElementwiseOp::Add => _mm256_add_ps(a, b),
            FpElementwiseOp::Sub => _mm256_sub_ps(a, b),
            FpElementwiseOp::Mul => _mm256_mul_ps(a, b),
        };
        unsafe { store_f16_from_f32x8_x86(out.as_mut_ptr().add(kk), y) };
        kk += 8;
    }

    while kk < len {
        out[kk] = match op {
            FpElementwiseOp::Add => f16::from_f32(lhs[kk].to_f32() + rhs[kk].to_f32()),
            FpElementwiseOp::Sub => f16::from_f32(lhs[kk].to_f32() - rhs[kk].to_f32()),
            FpElementwiseOp::Mul => f16::from_f32(lhs[kk].to_f32() * rhs[kk].to_f32()),
        };
        kk += 1;
    }
}

#[cfg(all(
    feature = "x86-fp-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[target_feature(enable = "avx2,fma")]
unsafe fn elementwise_bf16_bf16_x86_avx2(
    lhs: &[bf16],
    rhs: &[bf16],
    out: &mut [bf16],
    op: FpElementwiseOp,
) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let len = lhs.len();
    let mut kk = 0usize;
    while kk + 16 <= len {
        let a0 = unsafe { load_bf16_as_f32x8_x86(lhs.as_ptr().add(kk)) };
        let a1 = unsafe { load_bf16_as_f32x8_x86(lhs.as_ptr().add(kk + 8)) };
        let b0 = unsafe { load_bf16_as_f32x8_x86(rhs.as_ptr().add(kk)) };
        let b1 = unsafe { load_bf16_as_f32x8_x86(rhs.as_ptr().add(kk + 8)) };
        let y0 = match op {
            FpElementwiseOp::Add => _mm256_add_ps(a0, b0),
            FpElementwiseOp::Sub => _mm256_sub_ps(a0, b0),
            FpElementwiseOp::Mul => _mm256_mul_ps(a0, b0),
        };
        let y1 = match op {
            FpElementwiseOp::Add => _mm256_add_ps(a1, b1),
            FpElementwiseOp::Sub => _mm256_sub_ps(a1, b1),
            FpElementwiseOp::Mul => _mm256_mul_ps(a1, b1),
        };
        unsafe { store_bf16_from_f32x8_x86(out.as_mut_ptr().add(kk), y0) };
        unsafe { store_bf16_from_f32x8_x86(out.as_mut_ptr().add(kk + 8), y1) };
        kk += 16;
    }

    while kk + 8 <= len {
        let a = unsafe { load_bf16_as_f32x8_x86(lhs.as_ptr().add(kk)) };
        let b = unsafe { load_bf16_as_f32x8_x86(rhs.as_ptr().add(kk)) };
        let y = match op {
            FpElementwiseOp::Add => _mm256_add_ps(a, b),
            FpElementwiseOp::Sub => _mm256_sub_ps(a, b),
            FpElementwiseOp::Mul => _mm256_mul_ps(a, b),
        };
        unsafe { store_bf16_from_f32x8_x86(out.as_mut_ptr().add(kk), y) };
        kk += 8;
    }

    while kk < len {
        out[kk] = match op {
            FpElementwiseOp::Add => half::bf16::from_f32(lhs[kk].to_f32() + rhs[kk].to_f32()),
            FpElementwiseOp::Sub => half::bf16::from_f32(lhs[kk].to_f32() - rhs[kk].to_f32()),
            FpElementwiseOp::Mul => half::bf16::from_f32(lhs[kk].to_f32() * rhs[kk].to_f32()),
        };
        kk += 1;
    }
}

#[cfg(all(
    feature = "x86-fp-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn sum_f16_x86_avx2(x: &[f16]) -> f32 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let len = x.len();
    let mut kk = 0usize;
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();

    while kk + 16 <= len {
        let x0 = unsafe { load_f16_as_f32x8_x86(x.as_ptr().add(kk)) };
        let x1 = unsafe { load_f16_as_f32x8_x86(x.as_ptr().add(kk + 8)) };
        acc0 = _mm256_add_ps(acc0, x0);
        acc1 = _mm256_add_ps(acc1, x1);
        kk += 16;
    }

    while kk + 8 <= len {
        let chunk = unsafe { load_f16_as_f32x8_x86(x.as_ptr().add(kk)) };
        acc0 = _mm256_add_ps(acc0, chunk);
        kk += 8;
    }

    let mut sum = unsafe { reduce_f32x8_x86(acc0) + reduce_f32x8_x86(acc1) };
    while kk < len {
        sum += x[kk].to_f32();
        kk += 1;
    }
    sum
}

#[cfg(all(
    feature = "x86-fp-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[target_feature(enable = "avx2,fma")]
unsafe fn sum_bf16_x86_avx2(x: &[bf16]) -> f32 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let len = x.len();
    let mut kk = 0usize;
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();

    while kk + 16 <= len {
        let x0 = unsafe { load_bf16_as_f32x8_x86(x.as_ptr().add(kk)) };
        let x1 = unsafe { load_bf16_as_f32x8_x86(x.as_ptr().add(kk + 8)) };
        acc0 = _mm256_add_ps(acc0, x0);
        acc1 = _mm256_add_ps(acc1, x1);
        kk += 16;
    }

    while kk + 8 <= len {
        let chunk = unsafe { load_bf16_as_f32x8_x86(x.as_ptr().add(kk)) };
        acc0 = _mm256_add_ps(acc0, chunk);
        kk += 8;
    }

    let mut sum = unsafe { reduce_f32x8_x86(acc0) + reduce_f32x8_x86(acc1) };
    while kk < len {
        sum += x[kk].to_f32();
        kk += 1;
    }
    sum
}

#[cfg(all(
    feature = "x86-fp-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn dot_f32_f16_x86_avx2(x: &[f32], row: &[f16]) -> f32 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let k_dim = x.len();
    let mut kk = 0usize;
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();

    while kk + 16 <= k_dim {
        let row_lo = unsafe { load_f16_as_f32x8_x86(row.as_ptr().add(kk)) };
        let row_hi = unsafe { load_f16_as_f32x8_x86(row.as_ptr().add(kk + 8)) };
        let x_lo = unsafe { _mm256_loadu_ps(x.as_ptr().add(kk)) };
        let x_hi = unsafe { _mm256_loadu_ps(x.as_ptr().add(kk + 8)) };
        acc0 = _mm256_fmadd_ps(row_lo, x_lo, acc0);
        acc1 = _mm256_fmadd_ps(row_hi, x_hi, acc1);
        kk += 16;
    }

    while kk + 8 <= k_dim {
        let row_chunk = unsafe { load_f16_as_f32x8_x86(row.as_ptr().add(kk)) };
        let x_chunk = unsafe { _mm256_loadu_ps(x.as_ptr().add(kk)) };
        acc0 = _mm256_fmadd_ps(row_chunk, x_chunk, acc0);
        kk += 8;
    }

    let mut sum = unsafe { reduce_f32x8_x86(acc0) + reduce_f32x8_x86(acc1) };
    while kk < k_dim {
        sum += row[kk].to_f32() * x[kk];
        kk += 1;
    }
    sum
}

#[cfg(all(
    feature = "x86-fp-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn dot2_f32_f16_x86_avx2(x: &[f32], row0: &[f16], row1: &[f16]) -> (f32, f32) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let k_dim = x.len();
    let mut kk = 0usize;
    let mut acc00 = _mm256_setzero_ps();
    let mut acc01 = _mm256_setzero_ps();
    let mut acc10 = _mm256_setzero_ps();
    let mut acc11 = _mm256_setzero_ps();

    while kk + 16 <= k_dim {
        let x_lo = unsafe { _mm256_loadu_ps(x.as_ptr().add(kk)) };
        let x_hi = unsafe { _mm256_loadu_ps(x.as_ptr().add(kk + 8)) };
        let row0_lo = unsafe { load_f16_as_f32x8_x86(row0.as_ptr().add(kk)) };
        let row0_hi = unsafe { load_f16_as_f32x8_x86(row0.as_ptr().add(kk + 8)) };
        let row1_lo = unsafe { load_f16_as_f32x8_x86(row1.as_ptr().add(kk)) };
        let row1_hi = unsafe { load_f16_as_f32x8_x86(row1.as_ptr().add(kk + 8)) };
        acc00 = _mm256_fmadd_ps(row0_lo, x_lo, acc00);
        acc01 = _mm256_fmadd_ps(row0_hi, x_hi, acc01);
        acc10 = _mm256_fmadd_ps(row1_lo, x_lo, acc10);
        acc11 = _mm256_fmadd_ps(row1_hi, x_hi, acc11);
        kk += 16;
    }

    while kk + 8 <= k_dim {
        let x_chunk = unsafe { _mm256_loadu_ps(x.as_ptr().add(kk)) };
        let row0_chunk = unsafe { load_f16_as_f32x8_x86(row0.as_ptr().add(kk)) };
        let row1_chunk = unsafe { load_f16_as_f32x8_x86(row1.as_ptr().add(kk)) };
        acc00 = _mm256_fmadd_ps(row0_chunk, x_chunk, acc00);
        acc10 = _mm256_fmadd_ps(row1_chunk, x_chunk, acc10);
        kk += 8;
    }

    let mut sum0 = unsafe { reduce_f32x8_x86(acc00) + reduce_f32x8_x86(acc01) };
    let mut sum1 = unsafe { reduce_f32x8_x86(acc10) + reduce_f32x8_x86(acc11) };
    while kk < k_dim {
        let xv = x[kk];
        sum0 += row0[kk].to_f32() * xv;
        sum1 += row1[kk].to_f32() * xv;
        kk += 1;
    }
    (sum0, sum1)
}

#[cfg(all(
    feature = "x86-fp-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn dot3_f32_f16_x86_avx2(
    x: &[f32],
    row0: &[f16],
    row1: &[f16],
    row2: &[f16],
) -> (f32, f32, f32) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let k_dim = x.len();
    let mut kk = 0usize;
    let mut acc00 = _mm256_setzero_ps();
    let mut acc01 = _mm256_setzero_ps();
    let mut acc10 = _mm256_setzero_ps();
    let mut acc11 = _mm256_setzero_ps();
    let mut acc20 = _mm256_setzero_ps();
    let mut acc21 = _mm256_setzero_ps();

    while kk + 16 <= k_dim {
        let x_lo = unsafe { _mm256_loadu_ps(x.as_ptr().add(kk)) };
        let x_hi = unsafe { _mm256_loadu_ps(x.as_ptr().add(kk + 8)) };
        let row0_lo = unsafe { load_f16_as_f32x8_x86(row0.as_ptr().add(kk)) };
        let row0_hi = unsafe { load_f16_as_f32x8_x86(row0.as_ptr().add(kk + 8)) };
        let row1_lo = unsafe { load_f16_as_f32x8_x86(row1.as_ptr().add(kk)) };
        let row1_hi = unsafe { load_f16_as_f32x8_x86(row1.as_ptr().add(kk + 8)) };
        let row2_lo = unsafe { load_f16_as_f32x8_x86(row2.as_ptr().add(kk)) };
        let row2_hi = unsafe { load_f16_as_f32x8_x86(row2.as_ptr().add(kk + 8)) };
        acc00 = _mm256_fmadd_ps(row0_lo, x_lo, acc00);
        acc01 = _mm256_fmadd_ps(row0_hi, x_hi, acc01);
        acc10 = _mm256_fmadd_ps(row1_lo, x_lo, acc10);
        acc11 = _mm256_fmadd_ps(row1_hi, x_hi, acc11);
        acc20 = _mm256_fmadd_ps(row2_lo, x_lo, acc20);
        acc21 = _mm256_fmadd_ps(row2_hi, x_hi, acc21);
        kk += 16;
    }

    while kk + 8 <= k_dim {
        let x_chunk = unsafe { _mm256_loadu_ps(x.as_ptr().add(kk)) };
        let row0_chunk = unsafe { load_f16_as_f32x8_x86(row0.as_ptr().add(kk)) };
        let row1_chunk = unsafe { load_f16_as_f32x8_x86(row1.as_ptr().add(kk)) };
        let row2_chunk = unsafe { load_f16_as_f32x8_x86(row2.as_ptr().add(kk)) };
        acc00 = _mm256_fmadd_ps(row0_chunk, x_chunk, acc00);
        acc10 = _mm256_fmadd_ps(row1_chunk, x_chunk, acc10);
        acc20 = _mm256_fmadd_ps(row2_chunk, x_chunk, acc20);
        kk += 8;
    }

    let mut sum0 = unsafe { reduce_f32x8_x86(acc00) + reduce_f32x8_x86(acc01) };
    let mut sum1 = unsafe { reduce_f32x8_x86(acc10) + reduce_f32x8_x86(acc11) };
    let mut sum2 = unsafe { reduce_f32x8_x86(acc20) + reduce_f32x8_x86(acc21) };
    while kk < k_dim {
        let xv = x[kk];
        sum0 += row0[kk].to_f32() * xv;
        sum1 += row1[kk].to_f32() * xv;
        sum2 += row2[kk].to_f32() * xv;
        kk += 1;
    }
    (sum0, sum1, sum2)
}

#[cfg(all(
    feature = "x86-fp-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_f32_bf16_x86_avx2(x: &[f32], row: &[bf16]) -> f32 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let k_dim = x.len();
    let mut kk = 0usize;
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();

    while kk + 16 <= k_dim {
        let row_lo = unsafe { load_bf16_as_f32x8_x86(row.as_ptr().add(kk)) };
        let row_hi = unsafe { load_bf16_as_f32x8_x86(row.as_ptr().add(kk + 8)) };
        let x_lo = unsafe { _mm256_loadu_ps(x.as_ptr().add(kk)) };
        let x_hi = unsafe { _mm256_loadu_ps(x.as_ptr().add(kk + 8)) };
        acc0 = _mm256_fmadd_ps(row_lo, x_lo, acc0);
        acc1 = _mm256_fmadd_ps(row_hi, x_hi, acc1);
        kk += 16;
    }

    while kk + 8 <= k_dim {
        let row_chunk = unsafe { load_bf16_as_f32x8_x86(row.as_ptr().add(kk)) };
        let x_chunk = unsafe { _mm256_loadu_ps(x.as_ptr().add(kk)) };
        acc0 = _mm256_fmadd_ps(row_chunk, x_chunk, acc0);
        kk += 8;
    }

    let mut sum = unsafe { reduce_f32x8_x86(acc0) + reduce_f32x8_x86(acc1) };
    while kk < k_dim {
        sum += row[kk].to_f32() * x[kk];
        kk += 1;
    }
    sum
}

#[cfg(all(
    feature = "x86-fp-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot2_f32_bf16_x86_avx2(x: &[f32], row0: &[bf16], row1: &[bf16]) -> (f32, f32) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let k_dim = x.len();
    let mut kk = 0usize;
    let mut acc00 = _mm256_setzero_ps();
    let mut acc01 = _mm256_setzero_ps();
    let mut acc10 = _mm256_setzero_ps();
    let mut acc11 = _mm256_setzero_ps();

    while kk + 16 <= k_dim {
        let x_lo = unsafe { _mm256_loadu_ps(x.as_ptr().add(kk)) };
        let x_hi = unsafe { _mm256_loadu_ps(x.as_ptr().add(kk + 8)) };
        let row0_lo = unsafe { load_bf16_as_f32x8_x86(row0.as_ptr().add(kk)) };
        let row0_hi = unsafe { load_bf16_as_f32x8_x86(row0.as_ptr().add(kk + 8)) };
        let row1_lo = unsafe { load_bf16_as_f32x8_x86(row1.as_ptr().add(kk)) };
        let row1_hi = unsafe { load_bf16_as_f32x8_x86(row1.as_ptr().add(kk + 8)) };
        acc00 = _mm256_fmadd_ps(row0_lo, x_lo, acc00);
        acc01 = _mm256_fmadd_ps(row0_hi, x_hi, acc01);
        acc10 = _mm256_fmadd_ps(row1_lo, x_lo, acc10);
        acc11 = _mm256_fmadd_ps(row1_hi, x_hi, acc11);
        kk += 16;
    }

    while kk + 8 <= k_dim {
        let x_chunk = unsafe { _mm256_loadu_ps(x.as_ptr().add(kk)) };
        let row0_chunk = unsafe { load_bf16_as_f32x8_x86(row0.as_ptr().add(kk)) };
        let row1_chunk = unsafe { load_bf16_as_f32x8_x86(row1.as_ptr().add(kk)) };
        acc00 = _mm256_fmadd_ps(row0_chunk, x_chunk, acc00);
        acc10 = _mm256_fmadd_ps(row1_chunk, x_chunk, acc10);
        kk += 8;
    }

    let mut sum0 = unsafe { reduce_f32x8_x86(acc00) + reduce_f32x8_x86(acc01) };
    let mut sum1 = unsafe { reduce_f32x8_x86(acc10) + reduce_f32x8_x86(acc11) };
    while kk < k_dim {
        let xv = x[kk];
        sum0 += row0[kk].to_f32() * xv;
        sum1 += row1[kk].to_f32() * xv;
        kk += 1;
    }
    (sum0, sum1)
}

#[cfg(all(
    feature = "x86-fp-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot3_f32_bf16_x86_avx2(
    x: &[f32],
    row0: &[bf16],
    row1: &[bf16],
    row2: &[bf16],
) -> (f32, f32, f32) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let k_dim = x.len();
    let mut kk = 0usize;
    let mut acc00 = _mm256_setzero_ps();
    let mut acc01 = _mm256_setzero_ps();
    let mut acc10 = _mm256_setzero_ps();
    let mut acc11 = _mm256_setzero_ps();
    let mut acc20 = _mm256_setzero_ps();
    let mut acc21 = _mm256_setzero_ps();

    while kk + 16 <= k_dim {
        let x_lo = unsafe { _mm256_loadu_ps(x.as_ptr().add(kk)) };
        let x_hi = unsafe { _mm256_loadu_ps(x.as_ptr().add(kk + 8)) };
        let row0_lo = unsafe { load_bf16_as_f32x8_x86(row0.as_ptr().add(kk)) };
        let row0_hi = unsafe { load_bf16_as_f32x8_x86(row0.as_ptr().add(kk + 8)) };
        let row1_lo = unsafe { load_bf16_as_f32x8_x86(row1.as_ptr().add(kk)) };
        let row1_hi = unsafe { load_bf16_as_f32x8_x86(row1.as_ptr().add(kk + 8)) };
        let row2_lo = unsafe { load_bf16_as_f32x8_x86(row2.as_ptr().add(kk)) };
        let row2_hi = unsafe { load_bf16_as_f32x8_x86(row2.as_ptr().add(kk + 8)) };
        acc00 = _mm256_fmadd_ps(row0_lo, x_lo, acc00);
        acc01 = _mm256_fmadd_ps(row0_hi, x_hi, acc01);
        acc10 = _mm256_fmadd_ps(row1_lo, x_lo, acc10);
        acc11 = _mm256_fmadd_ps(row1_hi, x_hi, acc11);
        acc20 = _mm256_fmadd_ps(row2_lo, x_lo, acc20);
        acc21 = _mm256_fmadd_ps(row2_hi, x_hi, acc21);
        kk += 16;
    }

    while kk + 8 <= k_dim {
        let x_chunk = unsafe { _mm256_loadu_ps(x.as_ptr().add(kk)) };
        let row0_chunk = unsafe { load_bf16_as_f32x8_x86(row0.as_ptr().add(kk)) };
        let row1_chunk = unsafe { load_bf16_as_f32x8_x86(row1.as_ptr().add(kk)) };
        let row2_chunk = unsafe { load_bf16_as_f32x8_x86(row2.as_ptr().add(kk)) };
        acc00 = _mm256_fmadd_ps(row0_chunk, x_chunk, acc00);
        acc10 = _mm256_fmadd_ps(row1_chunk, x_chunk, acc10);
        acc20 = _mm256_fmadd_ps(row2_chunk, x_chunk, acc20);
        kk += 8;
    }

    let mut sum0 = unsafe { reduce_f32x8_x86(acc00) + reduce_f32x8_x86(acc01) };
    let mut sum1 = unsafe { reduce_f32x8_x86(acc10) + reduce_f32x8_x86(acc11) };
    let mut sum2 = unsafe { reduce_f32x8_x86(acc20) + reduce_f32x8_x86(acc21) };
    while kk < k_dim {
        let xv = x[kk];
        sum0 += row0[kk].to_f32() * xv;
        sum1 += row1[kk].to_f32() * xv;
        sum2 += row2[kk].to_f32() * xv;
        kk += 1;
    }
    (sum0, sum1, sum2)
}

#[cfg(all(
    feature = "x86-fp-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn dot_f16_f16_x86_avx2(x: &[f16], row: &[f16]) -> f32 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let k_dim = x.len();
    let mut kk = 0usize;
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();

    while kk + 16 <= k_dim {
        let x_lo = unsafe { load_f16_as_f32x8_x86(x.as_ptr().add(kk)) };
        let x_hi = unsafe { load_f16_as_f32x8_x86(x.as_ptr().add(kk + 8)) };
        let row_lo = unsafe { load_f16_as_f32x8_x86(row.as_ptr().add(kk)) };
        let row_hi = unsafe { load_f16_as_f32x8_x86(row.as_ptr().add(kk + 8)) };
        acc0 = _mm256_fmadd_ps(row_lo, x_lo, acc0);
        acc1 = _mm256_fmadd_ps(row_hi, x_hi, acc1);
        kk += 16;
    }

    while kk + 8 <= k_dim {
        let x_chunk = unsafe { load_f16_as_f32x8_x86(x.as_ptr().add(kk)) };
        let row_chunk = unsafe { load_f16_as_f32x8_x86(row.as_ptr().add(kk)) };
        acc0 = _mm256_fmadd_ps(row_chunk, x_chunk, acc0);
        kk += 8;
    }

    let mut sum = unsafe { reduce_f32x8_x86(acc0) + reduce_f32x8_x86(acc1) };
    while kk < k_dim {
        sum += row[kk].to_f32() * x[kk].to_f32();
        kk += 1;
    }
    sum
}

#[cfg(all(
    feature = "x86-fp-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn dot2_f16_f16_x86_avx2(x: &[f16], row0: &[f16], row1: &[f16]) -> (f32, f32) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let k_dim = x.len();
    let mut kk = 0usize;
    let mut acc00 = _mm256_setzero_ps();
    let mut acc01 = _mm256_setzero_ps();
    let mut acc10 = _mm256_setzero_ps();
    let mut acc11 = _mm256_setzero_ps();

    while kk + 16 <= k_dim {
        let x_lo = unsafe { load_f16_as_f32x8_x86(x.as_ptr().add(kk)) };
        let x_hi = unsafe { load_f16_as_f32x8_x86(x.as_ptr().add(kk + 8)) };
        let row0_lo = unsafe { load_f16_as_f32x8_x86(row0.as_ptr().add(kk)) };
        let row0_hi = unsafe { load_f16_as_f32x8_x86(row0.as_ptr().add(kk + 8)) };
        let row1_lo = unsafe { load_f16_as_f32x8_x86(row1.as_ptr().add(kk)) };
        let row1_hi = unsafe { load_f16_as_f32x8_x86(row1.as_ptr().add(kk + 8)) };
        acc00 = _mm256_fmadd_ps(row0_lo, x_lo, acc00);
        acc01 = _mm256_fmadd_ps(row0_hi, x_hi, acc01);
        acc10 = _mm256_fmadd_ps(row1_lo, x_lo, acc10);
        acc11 = _mm256_fmadd_ps(row1_hi, x_hi, acc11);
        kk += 16;
    }

    while kk + 8 <= k_dim {
        let x_chunk = unsafe { load_f16_as_f32x8_x86(x.as_ptr().add(kk)) };
        let row0_chunk = unsafe { load_f16_as_f32x8_x86(row0.as_ptr().add(kk)) };
        let row1_chunk = unsafe { load_f16_as_f32x8_x86(row1.as_ptr().add(kk)) };
        acc00 = _mm256_fmadd_ps(row0_chunk, x_chunk, acc00);
        acc10 = _mm256_fmadd_ps(row1_chunk, x_chunk, acc10);
        kk += 8;
    }

    let mut sum0 = unsafe { reduce_f32x8_x86(acc00) + reduce_f32x8_x86(acc01) };
    let mut sum1 = unsafe { reduce_f32x8_x86(acc10) + reduce_f32x8_x86(acc11) };
    while kk < k_dim {
        let xv = x[kk].to_f32();
        sum0 += row0[kk].to_f32() * xv;
        sum1 += row1[kk].to_f32() * xv;
        kk += 1;
    }
    (sum0, sum1)
}

#[cfg(all(
    feature = "x86-fp-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn dot3_f16_f16_x86_avx2(
    x: &[f16],
    row0: &[f16],
    row1: &[f16],
    row2: &[f16],
) -> (f32, f32, f32) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let k_dim = x.len();
    let mut kk = 0usize;
    let mut acc00 = _mm256_setzero_ps();
    let mut acc01 = _mm256_setzero_ps();
    let mut acc10 = _mm256_setzero_ps();
    let mut acc11 = _mm256_setzero_ps();
    let mut acc20 = _mm256_setzero_ps();
    let mut acc21 = _mm256_setzero_ps();

    while kk + 16 <= k_dim {
        let x_lo = unsafe { load_f16_as_f32x8_x86(x.as_ptr().add(kk)) };
        let x_hi = unsafe { load_f16_as_f32x8_x86(x.as_ptr().add(kk + 8)) };
        let row0_lo = unsafe { load_f16_as_f32x8_x86(row0.as_ptr().add(kk)) };
        let row0_hi = unsafe { load_f16_as_f32x8_x86(row0.as_ptr().add(kk + 8)) };
        let row1_lo = unsafe { load_f16_as_f32x8_x86(row1.as_ptr().add(kk)) };
        let row1_hi = unsafe { load_f16_as_f32x8_x86(row1.as_ptr().add(kk + 8)) };
        let row2_lo = unsafe { load_f16_as_f32x8_x86(row2.as_ptr().add(kk)) };
        let row2_hi = unsafe { load_f16_as_f32x8_x86(row2.as_ptr().add(kk + 8)) };
        acc00 = _mm256_fmadd_ps(row0_lo, x_lo, acc00);
        acc01 = _mm256_fmadd_ps(row0_hi, x_hi, acc01);
        acc10 = _mm256_fmadd_ps(row1_lo, x_lo, acc10);
        acc11 = _mm256_fmadd_ps(row1_hi, x_hi, acc11);
        acc20 = _mm256_fmadd_ps(row2_lo, x_lo, acc20);
        acc21 = _mm256_fmadd_ps(row2_hi, x_hi, acc21);
        kk += 16;
    }

    while kk + 8 <= k_dim {
        let x_chunk = unsafe { load_f16_as_f32x8_x86(x.as_ptr().add(kk)) };
        let row0_chunk = unsafe { load_f16_as_f32x8_x86(row0.as_ptr().add(kk)) };
        let row1_chunk = unsafe { load_f16_as_f32x8_x86(row1.as_ptr().add(kk)) };
        let row2_chunk = unsafe { load_f16_as_f32x8_x86(row2.as_ptr().add(kk)) };
        acc00 = _mm256_fmadd_ps(row0_chunk, x_chunk, acc00);
        acc10 = _mm256_fmadd_ps(row1_chunk, x_chunk, acc10);
        acc20 = _mm256_fmadd_ps(row2_chunk, x_chunk, acc20);
        kk += 8;
    }

    let mut sum0 = unsafe { reduce_f32x8_x86(acc00) + reduce_f32x8_x86(acc01) };
    let mut sum1 = unsafe { reduce_f32x8_x86(acc10) + reduce_f32x8_x86(acc11) };
    let mut sum2 = unsafe { reduce_f32x8_x86(acc20) + reduce_f32x8_x86(acc21) };
    while kk < k_dim {
        let xv = x[kk].to_f32();
        sum0 += row0[kk].to_f32() * xv;
        sum1 += row1[kk].to_f32() * xv;
        sum2 += row2[kk].to_f32() * xv;
        kk += 1;
    }
    (sum0, sum1, sum2)
}

#[cfg(all(
    feature = "x86-fp-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_f32_x86_avx2(x: &[f32], row: &[f32]) -> f32 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let k_dim = x.len();
    let mut kk = 0usize;
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();

    while kk + 16 <= k_dim {
        let row_lo = unsafe { _mm256_loadu_ps(row.as_ptr().add(kk)) };
        let row_hi = unsafe { _mm256_loadu_ps(row.as_ptr().add(kk + 8)) };
        let x_lo = unsafe { _mm256_loadu_ps(x.as_ptr().add(kk)) };
        let x_hi = unsafe { _mm256_loadu_ps(x.as_ptr().add(kk + 8)) };
        acc0 = _mm256_fmadd_ps(row_lo, x_lo, acc0);
        acc1 = _mm256_fmadd_ps(row_hi, x_hi, acc1);
        kk += 16;
    }

    while kk + 8 <= k_dim {
        let row_chunk = unsafe { _mm256_loadu_ps(row.as_ptr().add(kk)) };
        let x_chunk = unsafe { _mm256_loadu_ps(x.as_ptr().add(kk)) };
        acc0 = _mm256_fmadd_ps(row_chunk, x_chunk, acc0);
        kk += 8;
    }

    let mut buf0 = [0.0f32; 8];
    let mut buf1 = [0.0f32; 8];
    unsafe {
        _mm256_storeu_ps(buf0.as_mut_ptr(), acc0);
        _mm256_storeu_ps(buf1.as_mut_ptr(), acc1);
    }

    let mut sum: f32 = buf0.iter().sum::<f32>() + buf1.iter().sum::<f32>();
    while kk < k_dim {
        sum += row[kk] * x[kk];
        kk += 1;
    }
    sum
}

#[cfg(all(
    feature = "x86-fp-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot2_f32_x86_avx2(x: &[f32], row0: &[f32], row1: &[f32]) -> (f32, f32) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let k_dim = x.len();
    let mut kk = 0usize;
    let mut acc00 = _mm256_setzero_ps();
    let mut acc01 = _mm256_setzero_ps();
    let mut acc10 = _mm256_setzero_ps();
    let mut acc11 = _mm256_setzero_ps();

    while kk + 16 <= k_dim {
        let x_lo = unsafe { _mm256_loadu_ps(x.as_ptr().add(kk)) };
        let x_hi = unsafe { _mm256_loadu_ps(x.as_ptr().add(kk + 8)) };
        let row0_lo = unsafe { _mm256_loadu_ps(row0.as_ptr().add(kk)) };
        let row0_hi = unsafe { _mm256_loadu_ps(row0.as_ptr().add(kk + 8)) };
        let row1_lo = unsafe { _mm256_loadu_ps(row1.as_ptr().add(kk)) };
        let row1_hi = unsafe { _mm256_loadu_ps(row1.as_ptr().add(kk + 8)) };
        acc00 = _mm256_fmadd_ps(row0_lo, x_lo, acc00);
        acc01 = _mm256_fmadd_ps(row0_hi, x_hi, acc01);
        acc10 = _mm256_fmadd_ps(row1_lo, x_lo, acc10);
        acc11 = _mm256_fmadd_ps(row1_hi, x_hi, acc11);
        kk += 16;
    }

    while kk + 8 <= k_dim {
        let x_chunk = unsafe { _mm256_loadu_ps(x.as_ptr().add(kk)) };
        let row0_chunk = unsafe { _mm256_loadu_ps(row0.as_ptr().add(kk)) };
        let row1_chunk = unsafe { _mm256_loadu_ps(row1.as_ptr().add(kk)) };
        acc00 = _mm256_fmadd_ps(row0_chunk, x_chunk, acc00);
        acc10 = _mm256_fmadd_ps(row1_chunk, x_chunk, acc10);
        kk += 8;
    }

    let mut buf00 = [0.0f32; 8];
    let mut buf01 = [0.0f32; 8];
    let mut buf10 = [0.0f32; 8];
    let mut buf11 = [0.0f32; 8];
    unsafe {
        _mm256_storeu_ps(buf00.as_mut_ptr(), acc00);
        _mm256_storeu_ps(buf01.as_mut_ptr(), acc01);
        _mm256_storeu_ps(buf10.as_mut_ptr(), acc10);
        _mm256_storeu_ps(buf11.as_mut_ptr(), acc11);
    }

    let mut sum0 = buf00.iter().sum::<f32>() + buf01.iter().sum::<f32>();
    let mut sum1 = buf10.iter().sum::<f32>() + buf11.iter().sum::<f32>();
    while kk < k_dim {
        let xv = x[kk];
        sum0 += row0[kk] * xv;
        sum1 += row1[kk] * xv;
        kk += 1;
    }
    (sum0, sum1)
}

#[cfg(all(
    feature = "x86-fp-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot3_f32_x86_avx2(
    x: &[f32],
    row0: &[f32],
    row1: &[f32],
    row2: &[f32],
) -> (f32, f32, f32) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let k_dim = x.len();
    let mut kk = 0usize;
    let mut acc00 = _mm256_setzero_ps();
    let mut acc01 = _mm256_setzero_ps();
    let mut acc10 = _mm256_setzero_ps();
    let mut acc11 = _mm256_setzero_ps();
    let mut acc20 = _mm256_setzero_ps();
    let mut acc21 = _mm256_setzero_ps();

    while kk + 16 <= k_dim {
        let x_lo = unsafe { _mm256_loadu_ps(x.as_ptr().add(kk)) };
        let x_hi = unsafe { _mm256_loadu_ps(x.as_ptr().add(kk + 8)) };
        let row0_lo = unsafe { _mm256_loadu_ps(row0.as_ptr().add(kk)) };
        let row0_hi = unsafe { _mm256_loadu_ps(row0.as_ptr().add(kk + 8)) };
        let row1_lo = unsafe { _mm256_loadu_ps(row1.as_ptr().add(kk)) };
        let row1_hi = unsafe { _mm256_loadu_ps(row1.as_ptr().add(kk + 8)) };
        let row2_lo = unsafe { _mm256_loadu_ps(row2.as_ptr().add(kk)) };
        let row2_hi = unsafe { _mm256_loadu_ps(row2.as_ptr().add(kk + 8)) };
        acc00 = _mm256_fmadd_ps(row0_lo, x_lo, acc00);
        acc01 = _mm256_fmadd_ps(row0_hi, x_hi, acc01);
        acc10 = _mm256_fmadd_ps(row1_lo, x_lo, acc10);
        acc11 = _mm256_fmadd_ps(row1_hi, x_hi, acc11);
        acc20 = _mm256_fmadd_ps(row2_lo, x_lo, acc20);
        acc21 = _mm256_fmadd_ps(row2_hi, x_hi, acc21);
        kk += 16;
    }

    while kk + 8 <= k_dim {
        let x_chunk = unsafe { _mm256_loadu_ps(x.as_ptr().add(kk)) };
        let row0_chunk = unsafe { _mm256_loadu_ps(row0.as_ptr().add(kk)) };
        let row1_chunk = unsafe { _mm256_loadu_ps(row1.as_ptr().add(kk)) };
        let row2_chunk = unsafe { _mm256_loadu_ps(row2.as_ptr().add(kk)) };
        acc00 = _mm256_fmadd_ps(row0_chunk, x_chunk, acc00);
        acc10 = _mm256_fmadd_ps(row1_chunk, x_chunk, acc10);
        acc20 = _mm256_fmadd_ps(row2_chunk, x_chunk, acc20);
        kk += 8;
    }

    let mut buf00 = [0.0f32; 8];
    let mut buf01 = [0.0f32; 8];
    let mut buf10 = [0.0f32; 8];
    let mut buf11 = [0.0f32; 8];
    let mut buf20 = [0.0f32; 8];
    let mut buf21 = [0.0f32; 8];
    unsafe {
        _mm256_storeu_ps(buf00.as_mut_ptr(), acc00);
        _mm256_storeu_ps(buf01.as_mut_ptr(), acc01);
        _mm256_storeu_ps(buf10.as_mut_ptr(), acc10);
        _mm256_storeu_ps(buf11.as_mut_ptr(), acc11);
        _mm256_storeu_ps(buf20.as_mut_ptr(), acc20);
        _mm256_storeu_ps(buf21.as_mut_ptr(), acc21);
    }

    let mut sum0 = buf00.iter().sum::<f32>() + buf01.iter().sum::<f32>();
    let mut sum1 = buf10.iter().sum::<f32>() + buf11.iter().sum::<f32>();
    let mut sum2 = buf20.iter().sum::<f32>() + buf21.iter().sum::<f32>();
    while kk < k_dim {
        let xv = x[kk];
        sum0 += row0[kk] * xv;
        sum1 += row1[kk] * xv;
        sum2 += row2[kk] * xv;
        kk += 1;
    }
    (sum0, sum1, sum2)
}

#[cfg(test)]
#[path = "../../tests/unit/ops/fp_kernels/tests.rs"]
mod tests;
