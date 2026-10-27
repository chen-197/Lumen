use crate::arch;

#[cfg(all(feature = "x86-int8-kernels", target_arch = "x86"))]
use std::arch::x86::{__m256, __m256i, __m512, __m512i};
#[cfg(all(feature = "x86-int8-kernels", target_arch = "x86_64"))]
use std::arch::x86_64::{__m256, __m256i, __m512, __m512i};

#[inline]
fn dot_len_matches(x: &[f32], row: &[i8]) -> bool {
    x.len() == row.len()
}

#[inline]
fn dot2_len_matches(x: &[f32], row0: &[i8], row1: &[i8]) -> bool {
    x.len() == row0.len() && x.len() == row1.len()
}

#[inline]
fn dot3_len_matches(x: &[f32], row0: &[i8], row1: &[i8], row2: &[i8]) -> bool {
    x.len() == row0.len() && x.len() == row1.len() && x.len() == row2.len()
}

#[inline]
fn dot_len_matches_same(x: &[i8], row: &[i8]) -> bool {
    x.len() == row.len()
}

#[inline]
fn dot2_len_matches_same(x: &[i8], row0: &[i8], row1: &[i8]) -> bool {
    x.len() == row0.len() && x.len() == row1.len()
}

#[inline]
fn dot3_len_matches_same(x: &[i8], row0: &[i8], row1: &[i8], row2: &[i8]) -> bool {
    x.len() == row0.len() && x.len() == row1.len() && x.len() == row2.len()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Int8KernelBackend {
    Portable,
    Arm64Neon,
    X86Avx512,
    X86Avx2,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum I8I8KernelBackend {
    Portable,
    X86Avx512,
    X86Avx2,
}

#[derive(Clone, Copy)]
pub struct I8ScaledRow<'a> {
    pub values: &'a [i8],
    pub scale: f32,
}

#[derive(Clone, Copy)]
pub struct I8RowBroadcast<'a> {
    pub lhs: I8ScaledRow<'a>,
    pub rhs: I8ScaledRow<'a>,
    pub last_dim: usize,
    pub vector_on_rhs: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum I8ElementwiseOp {
    Add,
    Sub,
    Mul,
}

#[inline]
pub fn active_int8_backend() -> Int8KernelBackend {
    if arch::arm64_i8_kernel_runtime_available() {
        Int8KernelBackend::Arm64Neon
    } else if arch::x86_avx512_i8_kernel_runtime_available() {
        Int8KernelBackend::X86Avx512
    } else if arch::x86_i8_kernel_runtime_available() {
        Int8KernelBackend::X86Avx2
    } else {
        Int8KernelBackend::Portable
    }
}

#[inline]
pub fn active_int8_backend_name() -> &'static str {
    match active_int8_backend() {
        Int8KernelBackend::Portable => "portable",
        Int8KernelBackend::Arm64Neon => "arm64-neon",
        Int8KernelBackend::X86Avx512 => "x86-avx512bw",
        Int8KernelBackend::X86Avx2 => "x86-avx2",
    }
}

#[inline]
pub fn active_i8_i8_backend() -> I8I8KernelBackend {
    if arch::x86_avx512_i8_i8_kernel_runtime_available() {
        I8I8KernelBackend::X86Avx512
    } else if arch::x86_i8_kernel_runtime_available() {
        I8I8KernelBackend::X86Avx2
    } else {
        I8I8KernelBackend::Portable
    }
}

#[inline]
pub fn active_i8_i8_backend_name() -> &'static str {
    match active_i8_i8_backend() {
        I8I8KernelBackend::Portable => "portable",
        I8I8KernelBackend::X86Avx512 => "x86-avx512bw",
        I8I8KernelBackend::X86Avx2 => "x86-avx2-i32acc",
    }
}

#[inline]
fn elementwise_len_matches<T>(lhs: &[T], rhs: &[T], out: &[T]) -> bool {
    lhs.len() == rhs.len() && lhs.len() == out.len()
}

#[inline]
fn row_broadcast_len_matches(args: I8RowBroadcast<'_>, out_len: usize) -> bool {
    if args.last_dim == 0 || !out_len.is_multiple_of(args.last_dim) {
        return false;
    }
    let expected_lhs = if args.vector_on_rhs {
        out_len
    } else {
        args.last_dim
    };
    let expected_rhs = if args.vector_on_rhs {
        args.last_dim
    } else {
        out_len
    };
    args.lhs.values.len() == expected_lhs && args.rhs.values.len() == expected_rhs
}

#[inline]
fn valid_scale(scale: f32) -> bool {
    scale.is_finite() && scale > 0.0
}

#[inline]
pub(crate) fn dynamic_i8_scale(max_abs: f32) -> f32 {
    if max_abs.is_finite() && max_abs > 0.0 {
        (max_abs / 127.0).max(f32::MIN_POSITIVE)
    } else {
        1.0
    }
}

#[inline]
pub fn add_i8_i8_arch(
    _lhs: &[i8],
    _lhs_scale: f32,
    _rhs: &[i8],
    _rhs_scale: f32,
    _out: &mut [i8],
) -> Option<f32> {
    elementwise_i8_i8_arch(
        _lhs,
        _lhs_scale,
        _rhs,
        _rhs_scale,
        _out,
        I8ElementwiseOp::Add,
    )
}

#[inline]
pub fn sub_i8_i8_arch(
    _lhs: &[i8],
    _lhs_scale: f32,
    _rhs: &[i8],
    _rhs_scale: f32,
    _out: &mut [i8],
) -> Option<f32> {
    elementwise_i8_i8_arch(
        _lhs,
        _lhs_scale,
        _rhs,
        _rhs_scale,
        _out,
        I8ElementwiseOp::Sub,
    )
}

#[inline]
pub fn mul_i8_i8_arch(
    _lhs: &[i8],
    _lhs_scale: f32,
    _rhs: &[i8],
    _rhs_scale: f32,
    _out: &mut [i8],
) -> Option<f32> {
    elementwise_i8_i8_arch(
        _lhs,
        _lhs_scale,
        _rhs,
        _rhs_scale,
        _out,
        I8ElementwiseOp::Mul,
    )
}

#[inline]
pub fn add_i8_i8_row_broadcast_arch(_args: I8RowBroadcast<'_>, _out: &mut [i8]) -> Option<f32> {
    elementwise_i8_i8_row_broadcast_arch(_args, _out, I8ElementwiseOp::Add)
}

#[inline]
pub fn sub_i8_i8_row_broadcast_arch(_args: I8RowBroadcast<'_>, _out: &mut [i8]) -> Option<f32> {
    elementwise_i8_i8_row_broadcast_arch(_args, _out, I8ElementwiseOp::Sub)
}

#[inline]
pub fn mul_i8_i8_row_broadcast_arch(_args: I8RowBroadcast<'_>, _out: &mut [i8]) -> Option<f32> {
    elementwise_i8_i8_row_broadcast_arch(_args, _out, I8ElementwiseOp::Mul)
}

#[inline]
pub fn add_i8_i8_to_f32_arch(
    _lhs: &[i8],
    _lhs_scale: f32,
    _rhs: &[i8],
    _rhs_scale: f32,
    _out: &mut [f32],
) -> bool {
    elementwise_i8_i8_to_f32_arch(
        _lhs,
        _lhs_scale,
        _rhs,
        _rhs_scale,
        _out,
        I8ElementwiseOp::Add,
    )
}

#[inline]
pub fn sub_i8_i8_to_f32_arch(
    _lhs: &[i8],
    _lhs_scale: f32,
    _rhs: &[i8],
    _rhs_scale: f32,
    _out: &mut [f32],
) -> bool {
    elementwise_i8_i8_to_f32_arch(
        _lhs,
        _lhs_scale,
        _rhs,
        _rhs_scale,
        _out,
        I8ElementwiseOp::Sub,
    )
}

#[inline]
pub fn mul_i8_i8_to_f32_arch(
    _lhs: &[i8],
    _lhs_scale: f32,
    _rhs: &[i8],
    _rhs_scale: f32,
    _out: &mut [f32],
) -> bool {
    elementwise_i8_i8_to_f32_arch(
        _lhs,
        _lhs_scale,
        _rhs,
        _rhs_scale,
        _out,
        I8ElementwiseOp::Mul,
    )
}

#[inline]
pub fn add_i8_i8_row_broadcast_to_f32_arch(_args: I8RowBroadcast<'_>, _out: &mut [f32]) -> bool {
    elementwise_i8_i8_row_broadcast_to_f32_arch(_args, _out, I8ElementwiseOp::Add)
}

#[inline]
pub fn sub_i8_i8_row_broadcast_to_f32_arch(_args: I8RowBroadcast<'_>, _out: &mut [f32]) -> bool {
    elementwise_i8_i8_row_broadcast_to_f32_arch(_args, _out, I8ElementwiseOp::Sub)
}

#[inline]
pub fn mul_i8_i8_row_broadcast_to_f32_arch(_args: I8RowBroadcast<'_>, _out: &mut [f32]) -> bool {
    elementwise_i8_i8_row_broadcast_to_f32_arch(_args, _out, I8ElementwiseOp::Mul)
}

#[inline]
fn elementwise_i8_i8_arch(
    _lhs: &[i8],
    _lhs_scale: f32,
    _rhs: &[i8],
    _rhs_scale: f32,
    _out: &mut [i8],
    _op: I8ElementwiseOp,
) -> Option<f32> {
    if !elementwise_len_matches(_lhs, _rhs, _out)
        || !valid_scale(_lhs_scale)
        || !valid_scale(_rhs_scale)
    {
        return None;
    }
    if _lhs.is_empty() {
        return Some(1.0);
    }

    match active_i8_i8_backend() {
        I8I8KernelBackend::Portable => None,
        #[cfg(all(
            feature = "x86-int8-kernels",
            any(target_arch = "x86_64", target_arch = "x86")
        ))]
        I8I8KernelBackend::X86Avx512 => Some(unsafe {
            elementwise_i8_i8_x86_avx512(_lhs, _lhs_scale, _rhs, _rhs_scale, _out, _op)
        }),
        #[cfg(all(
            feature = "x86-int8-kernels",
            any(target_arch = "x86_64", target_arch = "x86")
        ))]
        I8I8KernelBackend::X86Avx2 => Some(unsafe {
            elementwise_i8_i8_x86_avx2(_lhs, _lhs_scale, _rhs, _rhs_scale, _out, _op)
        }),
        #[allow(unreachable_patterns)]
        _ => None,
    }
}

#[inline]
fn elementwise_i8_i8_row_broadcast_arch(
    _args: I8RowBroadcast<'_>,
    _out: &mut [i8],
    _op: I8ElementwiseOp,
) -> Option<f32> {
    if !row_broadcast_len_matches(_args, _out.len())
        || !valid_scale(_args.lhs.scale)
        || !valid_scale(_args.rhs.scale)
    {
        return None;
    }
    if _out.is_empty() {
        return Some(1.0);
    }

    match active_i8_i8_backend() {
        I8I8KernelBackend::Portable => None,
        #[cfg(all(
            feature = "x86-int8-kernels",
            any(target_arch = "x86_64", target_arch = "x86")
        ))]
        I8I8KernelBackend::X86Avx512 => {
            Some(unsafe { elementwise_i8_i8_row_broadcast_x86_avx512(_args, _out, _op) })
        }
        #[cfg(all(
            feature = "x86-int8-kernels",
            any(target_arch = "x86_64", target_arch = "x86")
        ))]
        I8I8KernelBackend::X86Avx2 => {
            Some(unsafe { elementwise_i8_i8_row_broadcast_x86_avx2(_args, _out, _op) })
        }
        #[allow(unreachable_patterns)]
        _ => None,
    }
}

#[inline]
fn elementwise_i8_i8_to_f32_arch(
    _lhs: &[i8],
    _lhs_scale: f32,
    _rhs: &[i8],
    _rhs_scale: f32,
    _out: &mut [f32],
    _op: I8ElementwiseOp,
) -> bool {
    if _lhs.len() != _rhs.len()
        || _lhs.len() != _out.len()
        || !valid_scale(_lhs_scale)
        || !valid_scale(_rhs_scale)
    {
        return false;
    }

    match active_i8_i8_backend() {
        I8I8KernelBackend::Portable => false,
        #[cfg(all(
            feature = "x86-int8-kernels",
            any(target_arch = "x86_64", target_arch = "x86")
        ))]
        I8I8KernelBackend::X86Avx512 => {
            unsafe {
                elementwise_i8_i8_to_f32_x86_avx512(_lhs, _lhs_scale, _rhs, _rhs_scale, _out, _op)
            };
            true
        }
        #[cfg(all(
            feature = "x86-int8-kernels",
            any(target_arch = "x86_64", target_arch = "x86")
        ))]
        I8I8KernelBackend::X86Avx2 => {
            unsafe {
                elementwise_i8_i8_to_f32_x86_avx2(_lhs, _lhs_scale, _rhs, _rhs_scale, _out, _op)
            };
            true
        }
        #[allow(unreachable_patterns)]
        _ => false,
    }
}

#[inline]
fn elementwise_i8_i8_row_broadcast_to_f32_arch(
    _args: I8RowBroadcast<'_>,
    _out: &mut [f32],
    _op: I8ElementwiseOp,
) -> bool {
    if !row_broadcast_len_matches(_args, _out.len())
        || !valid_scale(_args.lhs.scale)
        || !valid_scale(_args.rhs.scale)
    {
        return false;
    }

    match active_i8_i8_backend() {
        I8I8KernelBackend::Portable => false,
        #[cfg(all(
            feature = "x86-int8-kernels",
            any(target_arch = "x86_64", target_arch = "x86")
        ))]
        I8I8KernelBackend::X86Avx512 => {
            unsafe { elementwise_i8_i8_row_broadcast_to_f32_x86_avx512(_args, _out, _op) };
            true
        }
        #[cfg(all(
            feature = "x86-int8-kernels",
            any(target_arch = "x86_64", target_arch = "x86")
        ))]
        I8I8KernelBackend::X86Avx2 => {
            unsafe { elementwise_i8_i8_row_broadcast_to_f32_x86_avx2(_args, _out, _op) };
            true
        }
        #[allow(unreachable_patterns)]
        _ => false,
    }
}

#[inline]
pub fn dot_f32_i8_arch(_x: &[f32], _row: &[i8], _scale: f32) -> Option<f32> {
    if !dot_len_matches(_x, _row) {
        return None;
    }

    match active_int8_backend() {
        Int8KernelBackend::Portable => None,
        #[cfg(all(feature = "arm64-int8-kernels", target_arch = "aarch64"))]
        Int8KernelBackend::Arm64Neon => Some(unsafe { dot_f32_i8_arm64_neon(_x, _row, _scale) }),
        #[cfg(all(
            feature = "x86-int8-kernels",
            any(target_arch = "x86_64", target_arch = "x86")
        ))]
        Int8KernelBackend::X86Avx512 => Some(unsafe { dot_f32_i8_x86_avx512(_x, _row, _scale) }),
        #[cfg(all(
            feature = "x86-int8-kernels",
            any(target_arch = "x86_64", target_arch = "x86")
        ))]
        Int8KernelBackend::X86Avx2 => Some(unsafe { dot_f32_i8_x86_avx2(_x, _row, _scale) }),
        #[allow(unreachable_patterns)]
        _ => None,
    }
}

#[inline]
pub fn dot2_f32_i8_arch(
    _x: &[f32],
    _row0: &[i8],
    _scale0: f32,
    _row1: &[i8],
    _scale1: f32,
) -> Option<(f32, f32)> {
    if !dot2_len_matches(_x, _row0, _row1) {
        return None;
    }

    match active_int8_backend() {
        Int8KernelBackend::Portable => None,
        #[cfg(all(feature = "arm64-int8-kernels", target_arch = "aarch64"))]
        Int8KernelBackend::Arm64Neon => {
            Some(unsafe { dot2_f32_i8_arm64_neon(_x, _row0, _scale0, _row1, _scale1) })
        }
        #[cfg(all(
            feature = "x86-int8-kernels",
            any(target_arch = "x86_64", target_arch = "x86")
        ))]
        Int8KernelBackend::X86Avx512 => {
            Some(unsafe { dot2_f32_i8_x86_avx512(_x, _row0, _scale0, _row1, _scale1) })
        }
        #[cfg(all(
            feature = "x86-int8-kernels",
            any(target_arch = "x86_64", target_arch = "x86")
        ))]
        Int8KernelBackend::X86Avx2 => {
            Some(unsafe { dot2_f32_i8_x86_avx2(_x, _row0, _scale0, _row1, _scale1) })
        }
        #[allow(unreachable_patterns)]
        _ => None,
    }
}

pub fn dot3_f32_i8_arch(
    _x: &[f32],
    _row0: &[i8],
    _scale0: f32,
    _row1: &[i8],
    _scale1: f32,
    _row2: &[i8],
    _scale2: f32,
) -> Option<(f32, f32, f32)> {
    if !dot3_len_matches(_x, _row0, _row1, _row2) {
        return None;
    }

    match active_int8_backend() {
        Int8KernelBackend::Portable => None,
        #[cfg(all(feature = "arm64-int8-kernels", target_arch = "aarch64"))]
        Int8KernelBackend::Arm64Neon => Some(unsafe {
            dot3_f32_i8_arm64_neon(_x, _row0, _scale0, _row1, _scale1, _row2, _scale2)
        }),
        #[cfg(all(
            feature = "x86-int8-kernels",
            any(target_arch = "x86_64", target_arch = "x86")
        ))]
        Int8KernelBackend::X86Avx512 => Some(unsafe {
            dot3_f32_i8_x86_avx512(_x, _row0, _scale0, _row1, _scale1, _row2, _scale2)
        }),
        #[cfg(all(
            feature = "x86-int8-kernels",
            any(target_arch = "x86_64", target_arch = "x86")
        ))]
        Int8KernelBackend::X86Avx2 => Some(unsafe {
            dot3_f32_i8_x86_avx2(_x, _row0, _scale0, _row1, _scale1, _row2, _scale2)
        }),
        #[allow(unreachable_patterns)]
        _ => None,
    }
}

#[inline]
pub fn dot_i8_i8_arch(_x: &[i8], _x_scale: f32, _row: &[i8], _row_scale: f32) -> Option<f32> {
    if !dot_len_matches_same(_x, _row) {
        return None;
    }

    match active_i8_i8_backend() {
        I8I8KernelBackend::Portable => None,
        #[cfg(all(
            feature = "x86-int8-kernels",
            any(target_arch = "x86_64", target_arch = "x86")
        ))]
        I8I8KernelBackend::X86Avx512 => {
            Some(unsafe { dot_i8_i8_x86_avx512(_x, _x_scale, _row, _row_scale) })
        }
        #[cfg(all(
            feature = "x86-int8-kernels",
            any(target_arch = "x86_64", target_arch = "x86")
        ))]
        I8I8KernelBackend::X86Avx2 => {
            Some(unsafe { dot_i8_i8_x86_avx2(_x, _x_scale, _row, _row_scale) })
        }
        #[allow(unreachable_patterns)]
        _ => None,
    }
}

#[inline]
pub fn dot2_i8_i8_arch(
    _x: &[i8],
    _x_scale: f32,
    _row0: &[i8],
    _scale0: f32,
    _row1: &[i8],
    _scale1: f32,
) -> Option<(f32, f32)> {
    if !dot2_len_matches_same(_x, _row0, _row1) {
        return None;
    }

    match active_i8_i8_backend() {
        I8I8KernelBackend::Portable => None,
        #[cfg(all(
            feature = "x86-int8-kernels",
            any(target_arch = "x86_64", target_arch = "x86")
        ))]
        I8I8KernelBackend::X86Avx512 => {
            Some(unsafe { dot2_i8_i8_x86_avx512(_x, _x_scale, _row0, _scale0, _row1, _scale1) })
        }
        #[cfg(all(
            feature = "x86-int8-kernels",
            any(target_arch = "x86_64", target_arch = "x86")
        ))]
        I8I8KernelBackend::X86Avx2 => {
            Some(unsafe { dot2_i8_i8_x86_avx2(_x, _x_scale, _row0, _scale0, _row1, _scale1) })
        }
        #[allow(unreachable_patterns)]
        _ => None,
    }
}

#[inline]
pub fn dot3_i8_i8_arch(
    _x: &[i8],
    _x_scale: f32,
    _rows: [I8ScaledRow<'_>; 3],
) -> Option<(f32, f32, f32)> {
    let [_row0, _row1, _row2] = _rows;
    if !dot3_len_matches_same(_x, _row0.values, _row1.values, _row2.values) {
        return None;
    }

    match active_i8_i8_backend() {
        I8I8KernelBackend::Portable => None,
        #[cfg(all(
            feature = "x86-int8-kernels",
            any(target_arch = "x86_64", target_arch = "x86")
        ))]
        I8I8KernelBackend::X86Avx512 => Some(unsafe { dot3_i8_i8_x86_avx512(_x, _x_scale, _rows) }),
        #[cfg(all(
            feature = "x86-int8-kernels",
            any(target_arch = "x86_64", target_arch = "x86")
        ))]
        I8I8KernelBackend::X86Avx2 => Some(unsafe { dot3_i8_i8_x86_avx2(_x, _x_scale, _rows) }),
        #[allow(unreachable_patterns)]
        _ => None,
    }
}

#[inline]
pub fn sum_i8_arch(_x: &[i8], _scale: f32) -> Option<f32> {
    if !_scale.is_finite() || _scale <= 0.0 {
        return None;
    }
    if _x.is_empty() {
        return Some(0.0);
    }

    match active_i8_i8_backend() {
        I8I8KernelBackend::Portable => None,
        #[cfg(all(
            feature = "x86-int8-kernels",
            any(target_arch = "x86_64", target_arch = "x86")
        ))]
        I8I8KernelBackend::X86Avx512 => Some(unsafe { sum_i8_x86_avx512(_x, _scale) }),
        #[cfg(all(
            feature = "x86-int8-kernels",
            any(target_arch = "x86_64", target_arch = "x86")
        ))]
        I8I8KernelBackend::X86Avx2 => Some(unsafe { sum_i8_x86_avx2(_x, _scale) }),
        #[allow(unreachable_patterns)]
        _ => None,
    }
}

#[inline]
pub fn mul_f32_i8_to_f32_arch(
    _lhs: &[f32],
    _rhs: &[i8],
    _rhs_scale: f32,
    _out: &mut [f32],
) -> bool {
    if _lhs.len() != _rhs.len()
        || _lhs.len() != _out.len()
        || !_rhs_scale.is_finite()
        || _rhs_scale <= 0.0
    {
        return false;
    }

    match active_int8_backend() {
        Int8KernelBackend::Portable | Int8KernelBackend::Arm64Neon => false,
        #[cfg(all(
            feature = "x86-int8-kernels",
            any(target_arch = "x86_64", target_arch = "x86")
        ))]
        Int8KernelBackend::X86Avx512 => {
            unsafe { mul_f32_i8_to_f32_x86_avx512(_lhs, _rhs, _rhs_scale, _out) };
            true
        }
        #[cfg(all(
            feature = "x86-int8-kernels",
            any(target_arch = "x86_64", target_arch = "x86")
        ))]
        Int8KernelBackend::X86Avx2 => {
            unsafe { mul_f32_i8_to_f32_x86_avx2(_lhs, _rhs, _rhs_scale, _out) };
            true
        }
        #[allow(unreachable_patterns)]
        _ => false,
    }
}

#[inline]
pub fn add_f32_i8_to_f32_arch(
    _lhs: &[f32],
    _rhs: &[i8],
    _rhs_scale: f32,
    _out: &mut [f32],
) -> bool {
    f32_i8_to_f32_arch(_lhs, _rhs, _rhs_scale, false, _out, F32I8ElementwiseOp::Add)
}

#[inline]
pub fn sub_f32_i8_to_f32_arch(
    _lhs: &[f32],
    _rhs: &[i8],
    _rhs_scale: f32,
    _lowp_on_lhs: bool,
    _out: &mut [f32],
) -> bool {
    f32_i8_to_f32_arch(
        _lhs,
        _rhs,
        _rhs_scale,
        _lowp_on_lhs,
        _out,
        F32I8ElementwiseOp::Sub,
    )
}

#[derive(Clone, Copy)]
enum F32I8ElementwiseOp {
    Add,
    Sub,
}

#[inline]
fn f32_i8_to_f32_arch(
    _lhs: &[f32],
    _rhs: &[i8],
    _rhs_scale: f32,
    _lowp_on_lhs: bool,
    _out: &mut [f32],
    _op: F32I8ElementwiseOp,
) -> bool {
    if _lhs.len() != _rhs.len()
        || _lhs.len() != _out.len()
        || !_rhs_scale.is_finite()
        || _rhs_scale <= 0.0
    {
        return false;
    }

    match active_int8_backend() {
        Int8KernelBackend::Portable | Int8KernelBackend::Arm64Neon => false,
        #[cfg(all(
            feature = "x86-int8-kernels",
            any(target_arch = "x86_64", target_arch = "x86")
        ))]
        Int8KernelBackend::X86Avx512 => {
            unsafe { f32_i8_to_f32_x86_avx512(_lhs, _rhs, _rhs_scale, _lowp_on_lhs, _out, _op) };
            true
        }
        #[cfg(all(
            feature = "x86-int8-kernels",
            any(target_arch = "x86_64", target_arch = "x86")
        ))]
        Int8KernelBackend::X86Avx2 => {
            unsafe { f32_i8_to_f32_x86_avx2(_lhs, _rhs, _rhs_scale, _lowp_on_lhs, _out, _op) };
            true
        }
        #[allow(unreachable_patterns)]
        _ => false,
    }
}

#[inline]
pub fn mul_i8_scalar_to_f32_arch(
    _lhs: &[i8],
    _lhs_scale: f32,
    _rhs: f32,
    _out: &mut [f32],
) -> bool {
    if _lhs.len() != _out.len() || !_lhs_scale.is_finite() || _lhs_scale <= 0.0 || !_rhs.is_finite()
    {
        return false;
    }

    match active_int8_backend() {
        Int8KernelBackend::Portable | Int8KernelBackend::Arm64Neon => false,
        #[cfg(all(
            feature = "x86-int8-kernels",
            any(target_arch = "x86_64", target_arch = "x86")
        ))]
        Int8KernelBackend::X86Avx512 => {
            unsafe { mul_i8_scalar_to_f32_x86_avx512(_lhs, _lhs_scale, _rhs, _out) };
            true
        }
        #[cfg(all(
            feature = "x86-int8-kernels",
            any(target_arch = "x86_64", target_arch = "x86")
        ))]
        Int8KernelBackend::X86Avx2 => {
            unsafe { mul_i8_scalar_to_f32_x86_avx2(_lhs, _lhs_scale, _rhs, _out) };
            true
        }
        #[allow(unreachable_patterns)]
        _ => false,
    }
}

#[inline]
pub fn add_i8_scalar_to_f32_arch(
    _lhs: &[i8],
    _lhs_scale: f32,
    _rhs: f32,
    _out: &mut [f32],
) -> bool {
    i8_scalar_to_f32_arch(_lhs, _lhs_scale, _rhs, false, _out, F32I8ElementwiseOp::Add)
}

#[inline]
pub fn sub_i8_scalar_to_f32_arch(
    _lhs: &[i8],
    _lhs_scale: f32,
    _rhs: f32,
    _lowp_on_lhs: bool,
    _out: &mut [f32],
) -> bool {
    i8_scalar_to_f32_arch(
        _lhs,
        _lhs_scale,
        _rhs,
        _lowp_on_lhs,
        _out,
        F32I8ElementwiseOp::Sub,
    )
}

#[inline]
fn i8_scalar_to_f32_arch(
    _lhs: &[i8],
    _lhs_scale: f32,
    _rhs: f32,
    _lowp_on_lhs: bool,
    _out: &mut [f32],
    _op: F32I8ElementwiseOp,
) -> bool {
    if _lhs.len() != _out.len() || !_lhs_scale.is_finite() || _lhs_scale <= 0.0 || !_rhs.is_finite()
    {
        return false;
    }

    match active_int8_backend() {
        Int8KernelBackend::Portable | Int8KernelBackend::Arm64Neon => false,
        #[cfg(all(
            feature = "x86-int8-kernels",
            any(target_arch = "x86_64", target_arch = "x86")
        ))]
        Int8KernelBackend::X86Avx512 => {
            unsafe { i8_scalar_to_f32_x86_avx512(_lhs, _lhs_scale, _rhs, _lowp_on_lhs, _out, _op) };
            true
        }
        #[cfg(all(
            feature = "x86-int8-kernels",
            any(target_arch = "x86_64", target_arch = "x86")
        ))]
        Int8KernelBackend::X86Avx2 => {
            unsafe { i8_scalar_to_f32_x86_avx2(_lhs, _lhs_scale, _rhs, _lowp_on_lhs, _out, _op) };
            true
        }
        #[allow(unreachable_patterns)]
        _ => false,
    }
}

#[inline]
pub fn sgd_update_i8_f32_arch(_data: &mut [i8], _scale: f32, _grad: &[f32], _lr: f32) -> bool {
    if _data.len() != _grad.len()
        || !_scale.is_finite()
        || _scale <= 0.0
        || !_scale.recip().is_finite()
    {
        return false;
    }

    match active_int8_backend() {
        Int8KernelBackend::Portable | Int8KernelBackend::Arm64Neon => false,
        #[cfg(all(
            feature = "x86-int8-kernels",
            any(target_arch = "x86_64", target_arch = "x86")
        ))]
        Int8KernelBackend::X86Avx512 => {
            unsafe { sgd_update_i8_f32_x86_avx512(_data, _scale, _grad, _lr) };
            true
        }
        #[cfg(all(
            feature = "x86-int8-kernels",
            any(target_arch = "x86_64", target_arch = "x86")
        ))]
        Int8KernelBackend::X86Avx2 => {
            unsafe { sgd_update_i8_f32_x86_avx2(_data, _scale, _grad, _lr) };
            true
        }
        #[allow(unreachable_patterns)]
        _ => false,
    }
}

#[inline]
pub fn sgd_momentum_update_i8_f32_arch(
    _data: &mut [i8],
    _scale: f32,
    _velocity: &mut [f32],
    _grad: &[f32],
    _lr: f32,
    _momentum: f32,
) -> bool {
    if _data.len() != _grad.len()
        || _data.len() != _velocity.len()
        || !_scale.is_finite()
        || _scale <= 0.0
        || !_scale.recip().is_finite()
    {
        return false;
    }

    match active_int8_backend() {
        Int8KernelBackend::Portable | Int8KernelBackend::Arm64Neon => false,
        #[cfg(all(
            feature = "x86-int8-kernels",
            any(target_arch = "x86_64", target_arch = "x86")
        ))]
        Int8KernelBackend::X86Avx512 => {
            unsafe {
                sgd_momentum_update_i8_f32_x86_avx512(
                    _data, _scale, _velocity, _grad, _lr, _momentum,
                )
            };
            true
        }
        #[cfg(all(
            feature = "x86-int8-kernels",
            any(target_arch = "x86_64", target_arch = "x86")
        ))]
        Int8KernelBackend::X86Avx2 => {
            unsafe {
                sgd_momentum_update_i8_f32_x86_avx2(_data, _scale, _velocity, _grad, _lr, _momentum)
            };
            true
        }
        #[allow(unreachable_patterns)]
        _ => false,
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub fn adam_update_i8_f32_arch(
    _data: &mut [i8],
    _scale: f32,
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
        || !_scale.is_finite()
        || _scale <= 0.0
        || !_scale.recip().is_finite()
        || !_bias_correction1.is_finite()
        || _bias_correction1 == 0.0
        || !_bias_correction2.is_finite()
        || _bias_correction2 == 0.0
    {
        return false;
    }

    match active_int8_backend() {
        Int8KernelBackend::Portable | Int8KernelBackend::Arm64Neon => false,
        #[cfg(all(
            feature = "x86-int8-kernels",
            any(target_arch = "x86_64", target_arch = "x86")
        ))]
        Int8KernelBackend::X86Avx512 => {
            unsafe {
                adam_update_i8_f32_x86_avx512(
                    _data,
                    _scale,
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
            true
        }
        #[cfg(all(
            feature = "x86-int8-kernels",
            any(target_arch = "x86_64", target_arch = "x86")
        ))]
        Int8KernelBackend::X86Avx2 => {
            unsafe {
                adam_update_i8_f32_x86_avx2(
                    _data,
                    _scale,
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
            true
        }
        #[allow(unreachable_patterns)]
        _ => false,
    }
}

#[cfg(all(feature = "arm64-int8-kernels", target_arch = "aarch64"))]
#[target_feature(enable = "neon")]
unsafe fn dot2_f32_i8_arm64_neon(
    x: &[f32],
    row0: &[i8],
    scale0: f32,
    row1: &[i8],
    scale1: f32,
) -> (f32, f32) {
    use std::arch::aarch64::*;

    let k_dim = x.len();
    let mut kk = 0usize;
    let mut acc00 = vdupq_n_f32(0.0);
    let mut acc01 = vdupq_n_f32(0.0);
    let mut acc10 = vdupq_n_f32(0.0);
    let mut acc11 = vdupq_n_f32(0.0);

    while kk + 8 <= k_dim {
        let row0_8 = unsafe { vld1_s8(row0.as_ptr().add(kk)) };
        let row1_8 = unsafe { vld1_s8(row1.as_ptr().add(kk)) };
        let row0_16 = vmovl_s8(row0_8);
        let row1_16 = vmovl_s8(row1_8);
        let row0_lo = vcvtq_f32_s32(vmovl_s16(vget_low_s16(row0_16)));
        let row0_hi = vcvtq_f32_s32(vmovl_s16(vget_high_s16(row0_16)));
        let row1_lo = vcvtq_f32_s32(vmovl_s16(vget_low_s16(row1_16)));
        let row1_hi = vcvtq_f32_s32(vmovl_s16(vget_high_s16(row1_16)));
        let x_lo = unsafe { vld1q_f32(x.as_ptr().add(kk)) };
        let x_hi = unsafe { vld1q_f32(x.as_ptr().add(kk + 4)) };
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
        sum0 += row0[kk] as f32 * xv;
        sum1 += row1[kk] as f32 * xv;
        kk += 1;
    }
    (sum0 * scale0, sum1 * scale1)
}

#[cfg(all(feature = "arm64-int8-kernels", target_arch = "aarch64"))]
#[target_feature(enable = "neon")]
unsafe fn dot3_f32_i8_arm64_neon(
    x: &[f32],
    row0: &[i8],
    scale0: f32,
    row1: &[i8],
    scale1: f32,
    row2: &[i8],
    scale2: f32,
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
        let row0_8 = unsafe { vld1_s8(row0.as_ptr().add(kk)) };
        let row1_8 = unsafe { vld1_s8(row1.as_ptr().add(kk)) };
        let row2_8 = unsafe { vld1_s8(row2.as_ptr().add(kk)) };
        let row0_16 = vmovl_s8(row0_8);
        let row1_16 = vmovl_s8(row1_8);
        let row2_16 = vmovl_s8(row2_8);
        let row0_lo = vcvtq_f32_s32(vmovl_s16(vget_low_s16(row0_16)));
        let row0_hi = vcvtq_f32_s32(vmovl_s16(vget_high_s16(row0_16)));
        let row1_lo = vcvtq_f32_s32(vmovl_s16(vget_low_s16(row1_16)));
        let row1_hi = vcvtq_f32_s32(vmovl_s16(vget_high_s16(row1_16)));
        let row2_lo = vcvtq_f32_s32(vmovl_s16(vget_low_s16(row2_16)));
        let row2_hi = vcvtq_f32_s32(vmovl_s16(vget_high_s16(row2_16)));
        let x_lo = unsafe { vld1q_f32(x.as_ptr().add(kk)) };
        let x_hi = unsafe { vld1q_f32(x.as_ptr().add(kk + 4)) };
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
        sum0 += row0[kk] as f32 * xv;
        sum1 += row1[kk] as f32 * xv;
        sum2 += row2[kk] as f32 * xv;
        kk += 1;
    }
    (sum0 * scale0, sum1 * scale1, sum2 * scale2)
}

#[cfg(all(feature = "arm64-int8-kernels", target_arch = "aarch64"))]
#[target_feature(enable = "neon")]
unsafe fn dot_f32_i8_arm64_neon(x: &[f32], row: &[i8], scale: f32) -> f32 {
    use std::arch::aarch64::*;

    let k_dim = x.len();
    let mut kk = 0usize;
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);

    while kk + 8 <= k_dim {
        let row8 = unsafe { vld1_s8(row.as_ptr().add(kk)) };
        let row16 = vmovl_s8(row8);
        let row_lo_i32 = vmovl_s16(vget_low_s16(row16));
        let row_hi_i32 = vmovl_s16(vget_high_s16(row16));
        let row_lo = vcvtq_f32_s32(row_lo_i32);
        let row_hi = vcvtq_f32_s32(row_hi_i32);
        let x_lo = unsafe { vld1q_f32(x.as_ptr().add(kk)) };
        let x_hi = unsafe { vld1q_f32(x.as_ptr().add(kk + 4)) };
        acc0 = vfmaq_f32(acc0, row_lo, x_lo);
        acc1 = vfmaq_f32(acc1, row_hi, x_hi);
        kk += 8;
    }

    let mut sum = vaddvq_f32(acc0) + vaddvq_f32(acc1);
    while kk < k_dim {
        sum += row[kk] as f32 * x[kk];
        kk += 1;
    }
    sum * scale
}

#[cfg(all(
    feature = "x86-int8-kernels",
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
    feature = "x86-int8-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn reduce_max_f32x8_x86(v: __m256) -> f32 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let mut buf = [0.0f32; 8];
    unsafe {
        _mm256_storeu_ps(buf.as_mut_ptr(), v);
    }
    buf.iter().copied().fold(0.0f32, f32::max)
}

#[cfg(all(
    feature = "x86-int8-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn load_i8_as_f32x8_x86(ptr: *const i8) -> __m256 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let raw = unsafe { _mm_loadl_epi64(ptr as *const __m128i) };
    _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(raw))
}

#[cfg(all(
    feature = "x86-int8-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[inline]
#[target_feature(enable = "avx,avx512f")]
unsafe fn apply_f32_i8_op_x86_avx512(
    lhs: __m512,
    rhs: __m512,
    lowp_on_lhs: bool,
    op: F32I8ElementwiseOp,
) -> __m512 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    match op {
        F32I8ElementwiseOp::Add => _mm512_add_ps(lhs, rhs),
        F32I8ElementwiseOp::Sub if lowp_on_lhs => _mm512_sub_ps(rhs, lhs),
        F32I8ElementwiseOp::Sub => _mm512_sub_ps(lhs, rhs),
    }
}

#[cfg(all(
    feature = "x86-int8-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[target_feature(enable = "avx,avx512f,avx512bw")]
unsafe fn f32_i8_to_f32_x86_avx512(
    lhs: &[f32],
    rhs: &[i8],
    rhs_scale: f32,
    lowp_on_lhs: bool,
    out: &mut [f32],
    op: F32I8ElementwiseOp,
) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let len = lhs.len();
    let scale_v = _mm512_set1_ps(rhs_scale);
    let mut idx = 0usize;
    while idx + 32 <= len {
        let a0 = unsafe { _mm512_loadu_ps(lhs.as_ptr().add(idx)) };
        let a1 = unsafe { _mm512_loadu_ps(lhs.as_ptr().add(idx + 16)) };
        let b0_packed = unsafe { _mm_loadu_si128(rhs.as_ptr().add(idx) as *const __m128i) };
        let b1_packed = unsafe { _mm_loadu_si128(rhs.as_ptr().add(idx + 16) as *const __m128i) };
        let b0 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(b0_packed)), scale_v);
        let b1 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(b1_packed)), scale_v);
        let y0 = unsafe { apply_f32_i8_op_x86_avx512(a0, b0, lowp_on_lhs, op) };
        let y1 = unsafe { apply_f32_i8_op_x86_avx512(a1, b1, lowp_on_lhs, op) };
        unsafe { _mm512_storeu_ps(out.as_mut_ptr().add(idx), y0) };
        unsafe { _mm512_storeu_ps(out.as_mut_ptr().add(idx + 16), y1) };
        idx += 32;
    }
    while idx + 16 <= len {
        let a = unsafe { _mm512_loadu_ps(lhs.as_ptr().add(idx)) };
        let b_packed = unsafe { _mm_loadu_si128(rhs.as_ptr().add(idx) as *const __m128i) };
        let b = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(b_packed)), scale_v);
        let y = unsafe { apply_f32_i8_op_x86_avx512(a, b, lowp_on_lhs, op) };
        unsafe { _mm512_storeu_ps(out.as_mut_ptr().add(idx), y) };
        idx += 16;
    }
    while idx < len {
        let b = (rhs[idx] as f32) * rhs_scale;
        out[idx] = match op {
            F32I8ElementwiseOp::Add => lhs[idx] + b,
            F32I8ElementwiseOp::Sub if lowp_on_lhs => b - lhs[idx],
            F32I8ElementwiseOp::Sub => lhs[idx] - b,
        };
        idx += 1;
    }
}

#[cfg(all(
    feature = "x86-int8-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[target_feature(enable = "avx,avx512f,avx512bw")]
unsafe fn mul_f32_i8_to_f32_x86_avx512(lhs: &[f32], rhs: &[i8], rhs_scale: f32, out: &mut [f32]) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let len = lhs.len();
    let scale_v = _mm512_set1_ps(rhs_scale);
    let mut idx = 0usize;
    while idx + 32 <= len {
        let a0 = unsafe { _mm512_loadu_ps(lhs.as_ptr().add(idx)) };
        let a1 = unsafe { _mm512_loadu_ps(lhs.as_ptr().add(idx + 16)) };
        let b0_packed = unsafe { _mm_loadu_si128(rhs.as_ptr().add(idx) as *const __m128i) };
        let b1_packed = unsafe { _mm_loadu_si128(rhs.as_ptr().add(idx + 16) as *const __m128i) };
        let b0 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(b0_packed)), scale_v);
        let b1 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(b1_packed)), scale_v);
        unsafe { _mm512_storeu_ps(out.as_mut_ptr().add(idx), _mm512_mul_ps(a0, b0)) };
        unsafe { _mm512_storeu_ps(out.as_mut_ptr().add(idx + 16), _mm512_mul_ps(a1, b1)) };
        idx += 32;
    }
    while idx + 16 <= len {
        let a = unsafe { _mm512_loadu_ps(lhs.as_ptr().add(idx)) };
        let b_packed = unsafe { _mm_loadu_si128(rhs.as_ptr().add(idx) as *const __m128i) };
        let b = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(b_packed)), scale_v);
        unsafe { _mm512_storeu_ps(out.as_mut_ptr().add(idx), _mm512_mul_ps(a, b)) };
        idx += 16;
    }
    while idx < len {
        out[idx] = lhs[idx] * (rhs[idx] as f32) * rhs_scale;
        idx += 1;
    }
}

#[cfg(all(
    feature = "x86-int8-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[target_feature(enable = "avx,avx512f,avx512bw")]
unsafe fn i8_scalar_to_f32_x86_avx512(
    lhs: &[i8],
    lhs_scale: f32,
    rhs: f32,
    lowp_on_lhs: bool,
    out: &mut [f32],
    op: F32I8ElementwiseOp,
) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let len = lhs.len();
    let scale_v = _mm512_set1_ps(lhs_scale);
    let rhs_v = _mm512_set1_ps(rhs);
    let mut idx = 0usize;
    while idx + 32 <= len {
        let a0_packed = unsafe { _mm_loadu_si128(lhs.as_ptr().add(idx) as *const __m128i) };
        let a1_packed = unsafe { _mm_loadu_si128(lhs.as_ptr().add(idx + 16) as *const __m128i) };
        let a0 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(a0_packed)), scale_v);
        let a1 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(a1_packed)), scale_v);
        let y0 = unsafe { apply_f32_i8_op_x86_avx512(rhs_v, a0, lowp_on_lhs, op) };
        let y1 = unsafe { apply_f32_i8_op_x86_avx512(rhs_v, a1, lowp_on_lhs, op) };
        unsafe { _mm512_storeu_ps(out.as_mut_ptr().add(idx), y0) };
        unsafe { _mm512_storeu_ps(out.as_mut_ptr().add(idx + 16), y1) };
        idx += 32;
    }
    while idx + 16 <= len {
        let a_packed = unsafe { _mm_loadu_si128(lhs.as_ptr().add(idx) as *const __m128i) };
        let a = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(a_packed)), scale_v);
        let y = unsafe { apply_f32_i8_op_x86_avx512(rhs_v, a, lowp_on_lhs, op) };
        unsafe { _mm512_storeu_ps(out.as_mut_ptr().add(idx), y) };
        idx += 16;
    }
    while idx < len {
        let a = (lhs[idx] as f32) * lhs_scale;
        out[idx] = match op {
            F32I8ElementwiseOp::Add => rhs + a,
            F32I8ElementwiseOp::Sub if lowp_on_lhs => a - rhs,
            F32I8ElementwiseOp::Sub => rhs - a,
        };
        idx += 1;
    }
}

#[cfg(all(
    feature = "x86-int8-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[target_feature(enable = "avx,avx512f,avx512bw")]
unsafe fn mul_i8_scalar_to_f32_x86_avx512(lhs: &[i8], lhs_scale: f32, rhs: f32, out: &mut [f32]) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let len = lhs.len();
    let scale_v = _mm512_set1_ps(lhs_scale * rhs);
    let mut idx = 0usize;
    while idx + 32 <= len {
        let a0_packed = unsafe { _mm_loadu_si128(lhs.as_ptr().add(idx) as *const __m128i) };
        let a1_packed = unsafe { _mm_loadu_si128(lhs.as_ptr().add(idx + 16) as *const __m128i) };
        let a0 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(a0_packed));
        let a1 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(a1_packed));
        unsafe { _mm512_storeu_ps(out.as_mut_ptr().add(idx), _mm512_mul_ps(a0, scale_v)) };
        unsafe { _mm512_storeu_ps(out.as_mut_ptr().add(idx + 16), _mm512_mul_ps(a1, scale_v)) };
        idx += 32;
    }
    while idx + 16 <= len {
        let a_packed = unsafe { _mm_loadu_si128(lhs.as_ptr().add(idx) as *const __m128i) };
        let a = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(a_packed));
        unsafe { _mm512_storeu_ps(out.as_mut_ptr().add(idx), _mm512_mul_ps(a, scale_v)) };
        idx += 16;
    }
    while idx < len {
        out[idx] = (lhs[idx] as f32) * lhs_scale * rhs;
        idx += 1;
    }
}

#[cfg(all(
    feature = "x86-int8-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn apply_f32_i8_op_x86_avx2(
    lhs: __m256,
    rhs: __m256,
    lowp_on_lhs: bool,
    op: F32I8ElementwiseOp,
) -> __m256 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    match op {
        F32I8ElementwiseOp::Add => _mm256_add_ps(lhs, rhs),
        F32I8ElementwiseOp::Sub if lowp_on_lhs => _mm256_sub_ps(rhs, lhs),
        F32I8ElementwiseOp::Sub => _mm256_sub_ps(lhs, rhs),
    }
}

#[cfg(all(
    feature = "x86-int8-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[target_feature(enable = "avx2")]
unsafe fn f32_i8_to_f32_x86_avx2(
    lhs: &[f32],
    rhs: &[i8],
    rhs_scale: f32,
    lowp_on_lhs: bool,
    out: &mut [f32],
    op: F32I8ElementwiseOp,
) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let len = lhs.len();
    let scale_v = _mm256_set1_ps(rhs_scale);
    let mut idx = 0usize;
    while idx + 16 <= len {
        let a0 = unsafe { _mm256_loadu_ps(lhs.as_ptr().add(idx)) };
        let a1 = unsafe { _mm256_loadu_ps(lhs.as_ptr().add(idx + 8)) };
        let b0 = _mm256_mul_ps(
            unsafe { load_i8_as_f32x8_x86(rhs.as_ptr().add(idx)) },
            scale_v,
        );
        let b1 = _mm256_mul_ps(
            unsafe { load_i8_as_f32x8_x86(rhs.as_ptr().add(idx + 8)) },
            scale_v,
        );
        let y0 = unsafe { apply_f32_i8_op_x86_avx2(a0, b0, lowp_on_lhs, op) };
        let y1 = unsafe { apply_f32_i8_op_x86_avx2(a1, b1, lowp_on_lhs, op) };
        unsafe { _mm256_storeu_ps(out.as_mut_ptr().add(idx), y0) };
        unsafe { _mm256_storeu_ps(out.as_mut_ptr().add(idx + 8), y1) };
        idx += 16;
    }
    while idx + 8 <= len {
        let a = unsafe { _mm256_loadu_ps(lhs.as_ptr().add(idx)) };
        let b = _mm256_mul_ps(
            unsafe { load_i8_as_f32x8_x86(rhs.as_ptr().add(idx)) },
            scale_v,
        );
        let y = unsafe { apply_f32_i8_op_x86_avx2(a, b, lowp_on_lhs, op) };
        unsafe { _mm256_storeu_ps(out.as_mut_ptr().add(idx), y) };
        idx += 8;
    }
    while idx < len {
        let b = (rhs[idx] as f32) * rhs_scale;
        out[idx] = match op {
            F32I8ElementwiseOp::Add => lhs[idx] + b,
            F32I8ElementwiseOp::Sub if lowp_on_lhs => b - lhs[idx],
            F32I8ElementwiseOp::Sub => lhs[idx] - b,
        };
        idx += 1;
    }
}

#[cfg(all(
    feature = "x86-int8-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[target_feature(enable = "avx2")]
unsafe fn mul_f32_i8_to_f32_x86_avx2(lhs: &[f32], rhs: &[i8], rhs_scale: f32, out: &mut [f32]) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let len = lhs.len();
    let scale_v = _mm256_set1_ps(rhs_scale);
    let mut idx = 0usize;
    while idx + 16 <= len {
        let a0 = unsafe { _mm256_loadu_ps(lhs.as_ptr().add(idx)) };
        let a1 = unsafe { _mm256_loadu_ps(lhs.as_ptr().add(idx + 8)) };
        let b0 = _mm256_mul_ps(
            unsafe { load_i8_as_f32x8_x86(rhs.as_ptr().add(idx)) },
            scale_v,
        );
        let b1 = _mm256_mul_ps(
            unsafe { load_i8_as_f32x8_x86(rhs.as_ptr().add(idx + 8)) },
            scale_v,
        );
        unsafe { _mm256_storeu_ps(out.as_mut_ptr().add(idx), _mm256_mul_ps(a0, b0)) };
        unsafe { _mm256_storeu_ps(out.as_mut_ptr().add(idx + 8), _mm256_mul_ps(a1, b1)) };
        idx += 16;
    }
    while idx + 8 <= len {
        let a = unsafe { _mm256_loadu_ps(lhs.as_ptr().add(idx)) };
        let b = _mm256_mul_ps(
            unsafe { load_i8_as_f32x8_x86(rhs.as_ptr().add(idx)) },
            scale_v,
        );
        unsafe { _mm256_storeu_ps(out.as_mut_ptr().add(idx), _mm256_mul_ps(a, b)) };
        idx += 8;
    }
    while idx < len {
        out[idx] = lhs[idx] * (rhs[idx] as f32) * rhs_scale;
        idx += 1;
    }
}

#[cfg(all(
    feature = "x86-int8-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[target_feature(enable = "avx2")]
unsafe fn i8_scalar_to_f32_x86_avx2(
    lhs: &[i8],
    lhs_scale: f32,
    rhs: f32,
    lowp_on_lhs: bool,
    out: &mut [f32],
    op: F32I8ElementwiseOp,
) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let len = lhs.len();
    let scale_v = _mm256_set1_ps(lhs_scale);
    let rhs_v = _mm256_set1_ps(rhs);
    let mut idx = 0usize;
    while idx + 16 <= len {
        let a0 = _mm256_mul_ps(
            unsafe { load_i8_as_f32x8_x86(lhs.as_ptr().add(idx)) },
            scale_v,
        );
        let a1 = _mm256_mul_ps(
            unsafe { load_i8_as_f32x8_x86(lhs.as_ptr().add(idx + 8)) },
            scale_v,
        );
        let y0 = unsafe { apply_f32_i8_op_x86_avx2(rhs_v, a0, lowp_on_lhs, op) };
        let y1 = unsafe { apply_f32_i8_op_x86_avx2(rhs_v, a1, lowp_on_lhs, op) };
        unsafe { _mm256_storeu_ps(out.as_mut_ptr().add(idx), y0) };
        unsafe { _mm256_storeu_ps(out.as_mut_ptr().add(idx + 8), y1) };
        idx += 16;
    }
    while idx + 8 <= len {
        let a = _mm256_mul_ps(
            unsafe { load_i8_as_f32x8_x86(lhs.as_ptr().add(idx)) },
            scale_v,
        );
        let y = unsafe { apply_f32_i8_op_x86_avx2(rhs_v, a, lowp_on_lhs, op) };
        unsafe { _mm256_storeu_ps(out.as_mut_ptr().add(idx), y) };
        idx += 8;
    }
    while idx < len {
        let a = (lhs[idx] as f32) * lhs_scale;
        out[idx] = match op {
            F32I8ElementwiseOp::Add => rhs + a,
            F32I8ElementwiseOp::Sub if lowp_on_lhs => a - rhs,
            F32I8ElementwiseOp::Sub => rhs - a,
        };
        idx += 1;
    }
}

#[cfg(all(
    feature = "x86-int8-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[target_feature(enable = "avx2")]
unsafe fn mul_i8_scalar_to_f32_x86_avx2(lhs: &[i8], lhs_scale: f32, rhs: f32, out: &mut [f32]) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let len = lhs.len();
    let scale_v = _mm256_set1_ps(lhs_scale * rhs);
    let mut idx = 0usize;
    while idx + 16 <= len {
        let a0 = unsafe { load_i8_as_f32x8_x86(lhs.as_ptr().add(idx)) };
        let a1 = unsafe { load_i8_as_f32x8_x86(lhs.as_ptr().add(idx + 8)) };
        unsafe { _mm256_storeu_ps(out.as_mut_ptr().add(idx), _mm256_mul_ps(a0, scale_v)) };
        unsafe { _mm256_storeu_ps(out.as_mut_ptr().add(idx + 8), _mm256_mul_ps(a1, scale_v)) };
        idx += 16;
    }
    while idx + 8 <= len {
        let a = unsafe { load_i8_as_f32x8_x86(lhs.as_ptr().add(idx)) };
        unsafe { _mm256_storeu_ps(out.as_mut_ptr().add(idx), _mm256_mul_ps(a, scale_v)) };
        idx += 8;
    }
    while idx < len {
        out[idx] = (lhs[idx] as f32) * lhs_scale * rhs;
        idx += 1;
    }
}

#[cfg(all(
    feature = "x86-int8-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[target_feature(enable = "avx,avx512f,avx512bw")]
unsafe fn elementwise_i8_i8_to_f32_x86_avx512(
    lhs: &[i8],
    lhs_scale: f32,
    rhs: &[i8],
    rhs_scale: f32,
    out: &mut [f32],
    op: I8ElementwiseOp,
) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let len = lhs.len();
    let lhs_scale_v = _mm512_set1_ps(lhs_scale);
    let rhs_scale_v = _mm512_set1_ps(rhs_scale);
    let mut idx = 0usize;
    while idx + 32 <= len {
        let y0 = unsafe {
            i8_elementwise_values_x16_x86(
                lhs.as_ptr().add(idx),
                lhs_scale_v,
                rhs.as_ptr().add(idx),
                rhs_scale_v,
                op,
            )
        };
        let y1 = unsafe {
            i8_elementwise_values_x16_x86(
                lhs.as_ptr().add(idx + 16),
                lhs_scale_v,
                rhs.as_ptr().add(idx + 16),
                rhs_scale_v,
                op,
            )
        };
        unsafe { _mm512_storeu_ps(out.as_mut_ptr().add(idx), y0) };
        unsafe { _mm512_storeu_ps(out.as_mut_ptr().add(idx + 16), y1) };
        idx += 32;
    }
    while idx + 16 <= len {
        let y = unsafe {
            i8_elementwise_values_x16_x86(
                lhs.as_ptr().add(idx),
                lhs_scale_v,
                rhs.as_ptr().add(idx),
                rhs_scale_v,
                op,
            )
        };
        unsafe { _mm512_storeu_ps(out.as_mut_ptr().add(idx), y) };
        idx += 16;
    }
    while idx < len {
        let lhs_v = (lhs[idx] as f32) * lhs_scale;
        let rhs_v = (rhs[idx] as f32) * rhs_scale;
        out[idx] = match op {
            I8ElementwiseOp::Add => lhs_v + rhs_v,
            I8ElementwiseOp::Sub => lhs_v - rhs_v,
            I8ElementwiseOp::Mul => lhs_v * rhs_v,
        };
        idx += 1;
    }
}

#[cfg(all(
    feature = "x86-int8-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[target_feature(enable = "avx2")]
unsafe fn elementwise_i8_i8_to_f32_x86_avx2(
    lhs: &[i8],
    lhs_scale: f32,
    rhs: &[i8],
    rhs_scale: f32,
    out: &mut [f32],
    op: I8ElementwiseOp,
) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let len = lhs.len();
    let lhs_scale_v = _mm256_set1_ps(lhs_scale);
    let rhs_scale_v = _mm256_set1_ps(rhs_scale);
    let mut idx = 0usize;
    while idx + 16 <= len {
        let y0 = unsafe {
            i8_elementwise_values_x8_x86(
                lhs.as_ptr().add(idx),
                lhs_scale_v,
                rhs.as_ptr().add(idx),
                rhs_scale_v,
                op,
            )
        };
        let y1 = unsafe {
            i8_elementwise_values_x8_x86(
                lhs.as_ptr().add(idx + 8),
                lhs_scale_v,
                rhs.as_ptr().add(idx + 8),
                rhs_scale_v,
                op,
            )
        };
        unsafe { _mm256_storeu_ps(out.as_mut_ptr().add(idx), y0) };
        unsafe { _mm256_storeu_ps(out.as_mut_ptr().add(idx + 8), y1) };
        idx += 16;
    }
    while idx + 8 <= len {
        let y = unsafe {
            i8_elementwise_values_x8_x86(
                lhs.as_ptr().add(idx),
                lhs_scale_v,
                rhs.as_ptr().add(idx),
                rhs_scale_v,
                op,
            )
        };
        unsafe { _mm256_storeu_ps(out.as_mut_ptr().add(idx), y) };
        idx += 8;
    }
    while idx < len {
        let lhs_v = (lhs[idx] as f32) * lhs_scale;
        let rhs_v = (rhs[idx] as f32) * rhs_scale;
        out[idx] = match op {
            I8ElementwiseOp::Add => lhs_v + rhs_v,
            I8ElementwiseOp::Sub => lhs_v - rhs_v,
            I8ElementwiseOp::Mul => lhs_v * rhs_v,
        };
        idx += 1;
    }
}

#[cfg(all(
    feature = "x86-int8-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[target_feature(enable = "avx,avx512f,avx512bw")]
unsafe fn elementwise_i8_i8_row_broadcast_to_f32_x86_avx512(
    args: I8RowBroadcast<'_>,
    out: &mut [f32],
    op: I8ElementwiseOp,
) {
    let rows = out.len() / args.last_dim;
    for row in 0..rows {
        let start = row * args.last_dim;
        let (lhs_row, rhs_row) = if args.vector_on_rhs {
            (
                &args.lhs.values[start..start + args.last_dim],
                args.rhs.values,
            )
        } else {
            (
                args.lhs.values,
                &args.rhs.values[start..start + args.last_dim],
            )
        };
        unsafe {
            elementwise_i8_i8_to_f32_x86_avx512(
                lhs_row,
                args.lhs.scale,
                rhs_row,
                args.rhs.scale,
                &mut out[start..start + args.last_dim],
                op,
            )
        };
    }
}

#[cfg(all(
    feature = "x86-int8-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[target_feature(enable = "avx2")]
unsafe fn elementwise_i8_i8_row_broadcast_to_f32_x86_avx2(
    args: I8RowBroadcast<'_>,
    out: &mut [f32],
    op: I8ElementwiseOp,
) {
    let rows = out.len() / args.last_dim;
    for row in 0..rows {
        let start = row * args.last_dim;
        let (lhs_row, rhs_row) = if args.vector_on_rhs {
            (
                &args.lhs.values[start..start + args.last_dim],
                args.rhs.values,
            )
        } else {
            (
                args.lhs.values,
                &args.rhs.values[start..start + args.last_dim],
            )
        };
        unsafe {
            elementwise_i8_i8_to_f32_x86_avx2(
                lhs_row,
                args.lhs.scale,
                rhs_row,
                args.rhs.scale,
                &mut out[start..start + args.last_dim],
                op,
            )
        };
    }
}

#[cfg(all(
    feature = "x86-int8-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[inline]
#[target_feature(enable = "avx,avx512f,avx512bw")]
unsafe fn i8_elementwise_values_x16_x86(
    lhs: *const i8,
    lhs_scale: __m512,
    rhs: *const i8,
    rhs_scale: __m512,
    op: I8ElementwiseOp,
) -> __m512 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let lhs_packed = unsafe { _mm_loadu_si128(lhs as *const __m128i) };
    let rhs_packed = unsafe { _mm_loadu_si128(rhs as *const __m128i) };
    let lhs_v = _mm512_mul_ps(
        _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(lhs_packed)),
        lhs_scale,
    );
    let rhs_v = _mm512_mul_ps(
        _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(rhs_packed)),
        rhs_scale,
    );
    match op {
        I8ElementwiseOp::Add => _mm512_add_ps(lhs_v, rhs_v),
        I8ElementwiseOp::Sub => _mm512_sub_ps(lhs_v, rhs_v),
        I8ElementwiseOp::Mul => _mm512_mul_ps(lhs_v, rhs_v),
    }
}

#[cfg(all(
    feature = "x86-int8-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[target_feature(enable = "avx,avx512f,avx512bw")]
unsafe fn elementwise_i8_i8_x86_avx512(
    lhs: &[i8],
    lhs_scale: f32,
    rhs: &[i8],
    rhs_scale: f32,
    out: &mut [i8],
    op: I8ElementwiseOp,
) -> f32 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let len = lhs.len();
    let lhs_scale_v = _mm512_set1_ps(lhs_scale);
    let rhs_scale_v = _mm512_set1_ps(rhs_scale);
    let zero = _mm512_setzero_ps();
    let mut max_abs_v = _mm512_setzero_ps();
    let mut idx = 0usize;

    while idx + 16 <= len {
        let value = unsafe {
            i8_elementwise_values_x16_x86(
                lhs.as_ptr().add(idx),
                lhs_scale_v,
                rhs.as_ptr().add(idx),
                rhs_scale_v,
                op,
            )
        };
        let abs_value = _mm512_max_ps(value, _mm512_sub_ps(zero, value));
        max_abs_v = _mm512_max_ps(max_abs_v, abs_value);
        idx += 16;
    }

    let mut max_abs = unsafe { reduce_max_f32x16_x86(max_abs_v) };
    while idx < len {
        let lhs_v = (lhs[idx] as f32) * lhs_scale;
        let rhs_v = (rhs[idx] as f32) * rhs_scale;
        let value = match op {
            I8ElementwiseOp::Add => lhs_v + rhs_v,
            I8ElementwiseOp::Sub => lhs_v - rhs_v,
            I8ElementwiseOp::Mul => lhs_v * rhs_v,
        };
        max_abs = max_abs.max(value.abs());
        idx += 1;
    }

    let out_scale = dynamic_i8_scale(max_abs);
    let inv_scale_v = _mm512_set1_ps(1.0 / out_scale);
    let min_q = _mm512_set1_epi32(-127);
    let max_q = _mm512_set1_epi32(127);
    idx = 0;

    while idx + 16 <= len {
        let value = unsafe {
            i8_elementwise_values_x16_x86(
                lhs.as_ptr().add(idx),
                lhs_scale_v,
                rhs.as_ptr().add(idx),
                rhs_scale_v,
                op,
            )
        };
        let requant = unsafe { requantize_i8_x86_avx512(_mm512_mul_ps(value, inv_scale_v)) };
        let requant = _mm512_min_epi32(_mm512_max_epi32(requant, min_q), max_q);
        let bytes = _mm512_cvtsepi32_epi8(requant);
        unsafe { _mm_storeu_si128(out.as_mut_ptr().add(idx) as *mut __m128i, bytes) };
        idx += 16;
    }

    let inv_scale = 1.0 / out_scale;
    while idx < len {
        let lhs_v = (lhs[idx] as f32) * lhs_scale;
        let rhs_v = (rhs[idx] as f32) * rhs_scale;
        let value = match op {
            I8ElementwiseOp::Add => lhs_v + rhs_v,
            I8ElementwiseOp::Sub => lhs_v - rhs_v,
            I8ElementwiseOp::Mul => lhs_v * rhs_v,
        };
        out[idx] = (value * inv_scale).round().clamp(-127.0, 127.0) as i8;
        idx += 1;
    }

    out_scale
}

#[cfg(all(
    feature = "x86-int8-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn i8_elementwise_values_x8_x86(
    lhs: *const i8,
    lhs_scale: __m256,
    rhs: *const i8,
    rhs_scale: __m256,
    op: I8ElementwiseOp,
) -> __m256 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let lhs_v = _mm256_mul_ps(unsafe { load_i8_as_f32x8_x86(lhs) }, lhs_scale);
    let rhs_v = _mm256_mul_ps(unsafe { load_i8_as_f32x8_x86(rhs) }, rhs_scale);
    match op {
        I8ElementwiseOp::Add => _mm256_add_ps(lhs_v, rhs_v),
        I8ElementwiseOp::Sub => _mm256_sub_ps(lhs_v, rhs_v),
        I8ElementwiseOp::Mul => _mm256_mul_ps(lhs_v, rhs_v),
    }
}

#[cfg(all(
    feature = "x86-int8-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[target_feature(enable = "avx2")]
unsafe fn elementwise_i8_i8_x86_avx2(
    lhs: &[i8],
    lhs_scale: f32,
    rhs: &[i8],
    rhs_scale: f32,
    out: &mut [i8],
    op: I8ElementwiseOp,
) -> f32 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let len = lhs.len();
    let lhs_scale_v = _mm256_set1_ps(lhs_scale);
    let rhs_scale_v = _mm256_set1_ps(rhs_scale);
    let sign_mask = _mm256_set1_ps(-0.0);
    let mut max_abs_v = _mm256_setzero_ps();
    let mut idx = 0usize;

    while idx + 8 <= len {
        let value = unsafe {
            i8_elementwise_values_x8_x86(
                lhs.as_ptr().add(idx),
                lhs_scale_v,
                rhs.as_ptr().add(idx),
                rhs_scale_v,
                op,
            )
        };
        let abs_value = _mm256_andnot_ps(sign_mask, value);
        max_abs_v = _mm256_max_ps(max_abs_v, abs_value);
        idx += 8;
    }

    let mut max_abs = unsafe { reduce_max_f32x8_x86(max_abs_v) };
    while idx < len {
        let lhs_v = (lhs[idx] as f32) * lhs_scale;
        let rhs_v = (rhs[idx] as f32) * rhs_scale;
        let value = match op {
            I8ElementwiseOp::Add => lhs_v + rhs_v,
            I8ElementwiseOp::Sub => lhs_v - rhs_v,
            I8ElementwiseOp::Mul => lhs_v * rhs_v,
        };
        max_abs = max_abs.max(value.abs());
        idx += 1;
    }

    let out_scale = dynamic_i8_scale(max_abs);
    let inv_scale_v = _mm256_set1_ps(1.0 / out_scale);
    let min_q = _mm256_set1_epi32(-127);
    let max_q = _mm256_set1_epi32(127);
    idx = 0;

    while idx + 8 <= len {
        let value = unsafe {
            i8_elementwise_values_x8_x86(
                lhs.as_ptr().add(idx),
                lhs_scale_v,
                rhs.as_ptr().add(idx),
                rhs_scale_v,
                op,
            )
        };
        let requant = unsafe { requantize_i8_x86_avx2(_mm256_mul_ps(value, inv_scale_v)) };
        let requant = _mm256_min_epi32(_mm256_max_epi32(requant, min_q), max_q);
        unsafe { store_i32x8_as_clamped_i8_x86(out, idx, requant) };
        idx += 8;
    }

    let inv_scale = 1.0 / out_scale;
    while idx < len {
        let lhs_v = (lhs[idx] as f32) * lhs_scale;
        let rhs_v = (rhs[idx] as f32) * rhs_scale;
        let value = match op {
            I8ElementwiseOp::Add => lhs_v + rhs_v,
            I8ElementwiseOp::Sub => lhs_v - rhs_v,
            I8ElementwiseOp::Mul => lhs_v * rhs_v,
        };
        out[idx] = (value * inv_scale).round().clamp(-127.0, 127.0) as i8;
        idx += 1;
    }

    out_scale
}

#[cfg(all(
    feature = "x86-int8-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[target_feature(enable = "avx,avx512f,avx512bw")]
unsafe fn elementwise_i8_i8_row_broadcast_x86_avx512(
    args: I8RowBroadcast<'_>,
    out: &mut [i8],
    op: I8ElementwiseOp,
) -> f32 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let rows = out.len() / args.last_dim;
    let lhs_scale_v = _mm512_set1_ps(args.lhs.scale);
    let rhs_scale_v = _mm512_set1_ps(args.rhs.scale);
    let zero = _mm512_setzero_ps();
    let mut max_abs_v = _mm512_setzero_ps();

    for row in 0..rows {
        let start = row * args.last_dim;
        let (lhs_row, rhs_row) = if args.vector_on_rhs {
            (
                &args.lhs.values[start..start + args.last_dim],
                args.rhs.values,
            )
        } else {
            (
                args.lhs.values,
                &args.rhs.values[start..start + args.last_dim],
            )
        };
        let mut idx = 0usize;
        while idx + 16 <= args.last_dim {
            let value = unsafe {
                i8_elementwise_values_x16_x86(
                    lhs_row.as_ptr().add(idx),
                    lhs_scale_v,
                    rhs_row.as_ptr().add(idx),
                    rhs_scale_v,
                    op,
                )
            };
            let abs_value = _mm512_max_ps(value, _mm512_sub_ps(zero, value));
            max_abs_v = _mm512_max_ps(max_abs_v, abs_value);
            idx += 16;
        }
    }

    let mut max_abs = unsafe { reduce_max_f32x16_x86(max_abs_v) };
    for row in 0..rows {
        let start = row * args.last_dim;
        let (lhs_row, rhs_row) = if args.vector_on_rhs {
            (
                &args.lhs.values[start..start + args.last_dim],
                args.rhs.values,
            )
        } else {
            (
                args.lhs.values,
                &args.rhs.values[start..start + args.last_dim],
            )
        };
        let mut idx = args.last_dim - (args.last_dim % 16);
        while idx < args.last_dim {
            let lhs_v = (lhs_row[idx] as f32) * args.lhs.scale;
            let rhs_v = (rhs_row[idx] as f32) * args.rhs.scale;
            let value = match op {
                I8ElementwiseOp::Add => lhs_v + rhs_v,
                I8ElementwiseOp::Sub => lhs_v - rhs_v,
                I8ElementwiseOp::Mul => lhs_v * rhs_v,
            };
            max_abs = max_abs.max(value.abs());
            idx += 1;
        }
    }

    let out_scale = dynamic_i8_scale(max_abs);
    let inv_scale_v = _mm512_set1_ps(1.0 / out_scale);
    let min_q = _mm512_set1_epi32(-127);
    let max_q = _mm512_set1_epi32(127);

    for row in 0..rows {
        let start = row * args.last_dim;
        let (lhs_row, rhs_row) = if args.vector_on_rhs {
            (
                &args.lhs.values[start..start + args.last_dim],
                args.rhs.values,
            )
        } else {
            (
                args.lhs.values,
                &args.rhs.values[start..start + args.last_dim],
            )
        };
        let mut idx = 0usize;
        while idx + 16 <= args.last_dim {
            let value = unsafe {
                i8_elementwise_values_x16_x86(
                    lhs_row.as_ptr().add(idx),
                    lhs_scale_v,
                    rhs_row.as_ptr().add(idx),
                    rhs_scale_v,
                    op,
                )
            };
            let requant = unsafe { requantize_i8_x86_avx512(_mm512_mul_ps(value, inv_scale_v)) };
            let requant = _mm512_min_epi32(_mm512_max_epi32(requant, min_q), max_q);
            let bytes = _mm512_cvtsepi32_epi8(requant);
            unsafe { _mm_storeu_si128(out.as_mut_ptr().add(start + idx) as *mut __m128i, bytes) };
            idx += 16;
        }

        let inv_scale = 1.0 / out_scale;
        while idx < args.last_dim {
            let lhs_v = (lhs_row[idx] as f32) * args.lhs.scale;
            let rhs_v = (rhs_row[idx] as f32) * args.rhs.scale;
            let value = match op {
                I8ElementwiseOp::Add => lhs_v + rhs_v,
                I8ElementwiseOp::Sub => lhs_v - rhs_v,
                I8ElementwiseOp::Mul => lhs_v * rhs_v,
            };
            out[start + idx] = (value * inv_scale).round().clamp(-127.0, 127.0) as i8;
            idx += 1;
        }
    }

    out_scale
}

#[cfg(all(
    feature = "x86-int8-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[target_feature(enable = "avx2")]
unsafe fn elementwise_i8_i8_row_broadcast_x86_avx2(
    args: I8RowBroadcast<'_>,
    out: &mut [i8],
    op: I8ElementwiseOp,
) -> f32 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let rows = out.len() / args.last_dim;
    let lhs_scale_v = _mm256_set1_ps(args.lhs.scale);
    let rhs_scale_v = _mm256_set1_ps(args.rhs.scale);
    let sign_mask = _mm256_set1_ps(-0.0);
    let mut max_abs_v = _mm256_setzero_ps();

    for row in 0..rows {
        let start = row * args.last_dim;
        let (lhs_row, rhs_row) = if args.vector_on_rhs {
            (
                &args.lhs.values[start..start + args.last_dim],
                args.rhs.values,
            )
        } else {
            (
                args.lhs.values,
                &args.rhs.values[start..start + args.last_dim],
            )
        };
        let mut idx = 0usize;
        while idx + 8 <= args.last_dim {
            let value = unsafe {
                i8_elementwise_values_x8_x86(
                    lhs_row.as_ptr().add(idx),
                    lhs_scale_v,
                    rhs_row.as_ptr().add(idx),
                    rhs_scale_v,
                    op,
                )
            };
            let abs_value = _mm256_andnot_ps(sign_mask, value);
            max_abs_v = _mm256_max_ps(max_abs_v, abs_value);
            idx += 8;
        }
    }

    let mut max_abs = unsafe { reduce_max_f32x8_x86(max_abs_v) };
    for row in 0..rows {
        let start = row * args.last_dim;
        let (lhs_row, rhs_row) = if args.vector_on_rhs {
            (
                &args.lhs.values[start..start + args.last_dim],
                args.rhs.values,
            )
        } else {
            (
                args.lhs.values,
                &args.rhs.values[start..start + args.last_dim],
            )
        };
        let mut idx = args.last_dim - (args.last_dim % 8);
        while idx < args.last_dim {
            let lhs_v = (lhs_row[idx] as f32) * args.lhs.scale;
            let rhs_v = (rhs_row[idx] as f32) * args.rhs.scale;
            let value = match op {
                I8ElementwiseOp::Add => lhs_v + rhs_v,
                I8ElementwiseOp::Sub => lhs_v - rhs_v,
                I8ElementwiseOp::Mul => lhs_v * rhs_v,
            };
            max_abs = max_abs.max(value.abs());
            idx += 1;
        }
    }

    let out_scale = dynamic_i8_scale(max_abs);
    let inv_scale_v = _mm256_set1_ps(1.0 / out_scale);
    let min_q = _mm256_set1_epi32(-127);
    let max_q = _mm256_set1_epi32(127);

    for row in 0..rows {
        let start = row * args.last_dim;
        let (lhs_row, rhs_row) = if args.vector_on_rhs {
            (
                &args.lhs.values[start..start + args.last_dim],
                args.rhs.values,
            )
        } else {
            (
                args.lhs.values,
                &args.rhs.values[start..start + args.last_dim],
            )
        };
        let mut idx = 0usize;
        while idx + 8 <= args.last_dim {
            let value = unsafe {
                i8_elementwise_values_x8_x86(
                    lhs_row.as_ptr().add(idx),
                    lhs_scale_v,
                    rhs_row.as_ptr().add(idx),
                    rhs_scale_v,
                    op,
                )
            };
            let requant = unsafe { requantize_i8_x86_avx2(_mm256_mul_ps(value, inv_scale_v)) };
            let requant = _mm256_min_epi32(_mm256_max_epi32(requant, min_q), max_q);
            unsafe { store_i32x8_as_clamped_i8_x86(out, start + idx, requant) };
            idx += 8;
        }

        let inv_scale = 1.0 / out_scale;
        while idx < args.last_dim {
            let lhs_v = (lhs_row[idx] as f32) * args.lhs.scale;
            let rhs_v = (rhs_row[idx] as f32) * args.rhs.scale;
            let value = match op {
                I8ElementwiseOp::Add => lhs_v + rhs_v,
                I8ElementwiseOp::Sub => lhs_v - rhs_v,
                I8ElementwiseOp::Mul => lhs_v * rhs_v,
            };
            out[start + idx] = (value * inv_scale).round().clamp(-127.0, 127.0) as i8;
            idx += 1;
        }
    }

    out_scale
}

#[cfg(all(
    feature = "x86-int8-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[inline]
#[target_feature(enable = "avx,avx512f,avx512bw")]
unsafe fn load_i8_as_i16x32_x86(ptr: *const i8) -> __m512i {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let raw = unsafe { _mm256_loadu_si256(ptr as *const __m256i) };
    _mm512_cvtepi8_epi16(raw)
}

#[cfg(all(
    feature = "x86-int8-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn load_i8_as_i16x16_x86(ptr: *const i8) -> __m256i {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let raw = unsafe { _mm_loadu_si128(ptr as *const __m128i) };
    _mm256_cvtepi8_epi16(raw)
}

#[cfg(all(
    feature = "x86-int8-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn reduce_i32x16_x86(v: __m512i) -> i32 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let mut buf = [0i32; 16];
    unsafe {
        _mm512_storeu_si512(buf.as_mut_ptr() as *mut __m512i, v);
    }
    buf.iter().sum()
}

#[cfg(all(
    feature = "x86-int8-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn reduce_max_f32x16_x86(v: __m512) -> f32 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let mut buf = [0.0f32; 16];
    unsafe {
        _mm512_storeu_ps(buf.as_mut_ptr(), v);
    }
    buf.iter().copied().fold(0.0f32, f32::max)
}

#[cfg(all(
    feature = "x86-int8-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn reduce_i32x8_x86(v: __m256i) -> i32 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let mut buf = [0i32; 8];
    unsafe {
        _mm256_storeu_si256(buf.as_mut_ptr() as *mut __m256i, v);
    }
    buf.iter().sum()
}

#[cfg(all(
    feature = "x86-int8-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[inline]
#[target_feature(enable = "avx,avx512f,avx512bw")]
unsafe fn dot_i8_i8_i32_x86_avx512(x: &[i8], row: &[i8]) -> i32 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let k_dim = x.len();
    let mut kk = 0usize;
    let mut acc0 = _mm512_setzero_si512();
    let mut acc1 = _mm512_setzero_si512();

    while kk + 64 <= k_dim {
        let x_lo = unsafe { load_i8_as_i16x32_x86(x.as_ptr().add(kk)) };
        let x_hi = unsafe { load_i8_as_i16x32_x86(x.as_ptr().add(kk + 32)) };
        let row_lo = unsafe { load_i8_as_i16x32_x86(row.as_ptr().add(kk)) };
        let row_hi = unsafe { load_i8_as_i16x32_x86(row.as_ptr().add(kk + 32)) };
        acc0 = _mm512_add_epi32(acc0, _mm512_madd_epi16(row_lo, x_lo));
        acc1 = _mm512_add_epi32(acc1, _mm512_madd_epi16(row_hi, x_hi));
        kk += 64;
    }

    while kk + 32 <= k_dim {
        let x_chunk = unsafe { load_i8_as_i16x32_x86(x.as_ptr().add(kk)) };
        let row_chunk = unsafe { load_i8_as_i16x32_x86(row.as_ptr().add(kk)) };
        acc0 = _mm512_add_epi32(acc0, _mm512_madd_epi16(row_chunk, x_chunk));
        kk += 32;
    }

    let mut sum = unsafe { reduce_i32x16_x86(acc0) + reduce_i32x16_x86(acc1) };
    while kk < k_dim {
        sum += row[kk] as i32 * x[kk] as i32;
        kk += 1;
    }
    sum
}

#[cfg(all(
    feature = "x86-int8-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[inline]
#[target_feature(enable = "avx,avx512f,avx512bw")]
unsafe fn dot2_i8_i8_i32_x86_avx512(x: &[i8], row0: &[i8], row1: &[i8]) -> (i32, i32) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let k_dim = x.len();
    let mut kk = 0usize;
    let mut acc00 = _mm512_setzero_si512();
    let mut acc01 = _mm512_setzero_si512();
    let mut acc10 = _mm512_setzero_si512();
    let mut acc11 = _mm512_setzero_si512();

    while kk + 64 <= k_dim {
        let x_lo = unsafe { load_i8_as_i16x32_x86(x.as_ptr().add(kk)) };
        let x_hi = unsafe { load_i8_as_i16x32_x86(x.as_ptr().add(kk + 32)) };
        let row0_lo = unsafe { load_i8_as_i16x32_x86(row0.as_ptr().add(kk)) };
        let row0_hi = unsafe { load_i8_as_i16x32_x86(row0.as_ptr().add(kk + 32)) };
        let row1_lo = unsafe { load_i8_as_i16x32_x86(row1.as_ptr().add(kk)) };
        let row1_hi = unsafe { load_i8_as_i16x32_x86(row1.as_ptr().add(kk + 32)) };
        acc00 = _mm512_add_epi32(acc00, _mm512_madd_epi16(row0_lo, x_lo));
        acc01 = _mm512_add_epi32(acc01, _mm512_madd_epi16(row0_hi, x_hi));
        acc10 = _mm512_add_epi32(acc10, _mm512_madd_epi16(row1_lo, x_lo));
        acc11 = _mm512_add_epi32(acc11, _mm512_madd_epi16(row1_hi, x_hi));
        kk += 64;
    }

    while kk + 32 <= k_dim {
        let x_chunk = unsafe { load_i8_as_i16x32_x86(x.as_ptr().add(kk)) };
        let row0_chunk = unsafe { load_i8_as_i16x32_x86(row0.as_ptr().add(kk)) };
        let row1_chunk = unsafe { load_i8_as_i16x32_x86(row1.as_ptr().add(kk)) };
        acc00 = _mm512_add_epi32(acc00, _mm512_madd_epi16(row0_chunk, x_chunk));
        acc10 = _mm512_add_epi32(acc10, _mm512_madd_epi16(row1_chunk, x_chunk));
        kk += 32;
    }

    let mut sum0 = unsafe { reduce_i32x16_x86(acc00) + reduce_i32x16_x86(acc01) };
    let mut sum1 = unsafe { reduce_i32x16_x86(acc10) + reduce_i32x16_x86(acc11) };
    while kk < k_dim {
        let xv = x[kk] as i32;
        sum0 += row0[kk] as i32 * xv;
        sum1 += row1[kk] as i32 * xv;
        kk += 1;
    }
    (sum0, sum1)
}

#[cfg(all(
    feature = "x86-int8-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[inline]
#[target_feature(enable = "avx,avx512f,avx512bw")]
unsafe fn dot3_i8_i8_i32_x86_avx512(
    x: &[i8],
    row0: &[i8],
    row1: &[i8],
    row2: &[i8],
) -> (i32, i32, i32) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let k_dim = x.len();
    let mut kk = 0usize;
    let mut acc00 = _mm512_setzero_si512();
    let mut acc01 = _mm512_setzero_si512();
    let mut acc10 = _mm512_setzero_si512();
    let mut acc11 = _mm512_setzero_si512();
    let mut acc20 = _mm512_setzero_si512();
    let mut acc21 = _mm512_setzero_si512();

    while kk + 64 <= k_dim {
        let x_lo = unsafe { load_i8_as_i16x32_x86(x.as_ptr().add(kk)) };
        let x_hi = unsafe { load_i8_as_i16x32_x86(x.as_ptr().add(kk + 32)) };
        let row0_lo = unsafe { load_i8_as_i16x32_x86(row0.as_ptr().add(kk)) };
        let row0_hi = unsafe { load_i8_as_i16x32_x86(row0.as_ptr().add(kk + 32)) };
        let row1_lo = unsafe { load_i8_as_i16x32_x86(row1.as_ptr().add(kk)) };
        let row1_hi = unsafe { load_i8_as_i16x32_x86(row1.as_ptr().add(kk + 32)) };
        let row2_lo = unsafe { load_i8_as_i16x32_x86(row2.as_ptr().add(kk)) };
        let row2_hi = unsafe { load_i8_as_i16x32_x86(row2.as_ptr().add(kk + 32)) };
        acc00 = _mm512_add_epi32(acc00, _mm512_madd_epi16(row0_lo, x_lo));
        acc01 = _mm512_add_epi32(acc01, _mm512_madd_epi16(row0_hi, x_hi));
        acc10 = _mm512_add_epi32(acc10, _mm512_madd_epi16(row1_lo, x_lo));
        acc11 = _mm512_add_epi32(acc11, _mm512_madd_epi16(row1_hi, x_hi));
        acc20 = _mm512_add_epi32(acc20, _mm512_madd_epi16(row2_lo, x_lo));
        acc21 = _mm512_add_epi32(acc21, _mm512_madd_epi16(row2_hi, x_hi));
        kk += 64;
    }

    while kk + 32 <= k_dim {
        let x_chunk = unsafe { load_i8_as_i16x32_x86(x.as_ptr().add(kk)) };
        let row0_chunk = unsafe { load_i8_as_i16x32_x86(row0.as_ptr().add(kk)) };
        let row1_chunk = unsafe { load_i8_as_i16x32_x86(row1.as_ptr().add(kk)) };
        let row2_chunk = unsafe { load_i8_as_i16x32_x86(row2.as_ptr().add(kk)) };
        acc00 = _mm512_add_epi32(acc00, _mm512_madd_epi16(row0_chunk, x_chunk));
        acc10 = _mm512_add_epi32(acc10, _mm512_madd_epi16(row1_chunk, x_chunk));
        acc20 = _mm512_add_epi32(acc20, _mm512_madd_epi16(row2_chunk, x_chunk));
        kk += 32;
    }

    let mut sum0 = unsafe { reduce_i32x16_x86(acc00) + reduce_i32x16_x86(acc01) };
    let mut sum1 = unsafe { reduce_i32x16_x86(acc10) + reduce_i32x16_x86(acc11) };
    let mut sum2 = unsafe { reduce_i32x16_x86(acc20) + reduce_i32x16_x86(acc21) };
    while kk < k_dim {
        let xv = x[kk] as i32;
        sum0 += row0[kk] as i32 * xv;
        sum1 += row1[kk] as i32 * xv;
        sum2 += row2[kk] as i32 * xv;
        kk += 1;
    }
    (sum0, sum1, sum2)
}

#[cfg(all(
    feature = "x86-int8-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[target_feature(enable = "avx,avx512f,avx512bw")]
unsafe fn dot_i8_i8_x86_avx512(x: &[i8], x_scale: f32, row: &[i8], row_scale: f32) -> f32 {
    let sum = unsafe { dot_i8_i8_i32_x86_avx512(x, row) };
    (sum as f32) * x_scale * row_scale
}

#[cfg(all(
    feature = "x86-int8-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[target_feature(enable = "avx,avx512f,avx512bw")]
unsafe fn sum_i8_x86_avx512(x: &[i8], scale: f32) -> f32 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let len = x.len();
    let ones = _mm512_set1_epi16(1);
    let mut kk = 0usize;
    let mut acc0 = _mm512_setzero_si512();
    let mut acc1 = _mm512_setzero_si512();

    while kk + 64 <= len {
        let x0 = unsafe { load_i8_as_i16x32_x86(x.as_ptr().add(kk)) };
        let x1 = unsafe { load_i8_as_i16x32_x86(x.as_ptr().add(kk + 32)) };
        acc0 = _mm512_add_epi32(acc0, _mm512_madd_epi16(x0, ones));
        acc1 = _mm512_add_epi32(acc1, _mm512_madd_epi16(x1, ones));
        kk += 64;
    }

    while kk + 32 <= len {
        let chunk = unsafe { load_i8_as_i16x32_x86(x.as_ptr().add(kk)) };
        acc0 = _mm512_add_epi32(acc0, _mm512_madd_epi16(chunk, ones));
        kk += 32;
    }

    let mut sum = unsafe { reduce_i32x16_x86(acc0) + reduce_i32x16_x86(acc1) };
    while kk < len {
        sum += x[kk] as i32;
        kk += 1;
    }
    (sum as f32) * scale
}

#[cfg(all(
    feature = "x86-int8-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[target_feature(enable = "avx,avx512f,avx512bw")]
unsafe fn dot2_i8_i8_x86_avx512(
    x: &[i8],
    x_scale: f32,
    row0: &[i8],
    scale0: f32,
    row1: &[i8],
    scale1: f32,
) -> (f32, f32) {
    let (sum0, sum1) = unsafe { dot2_i8_i8_i32_x86_avx512(x, row0, row1) };
    (
        (sum0 as f32) * x_scale * scale0,
        (sum1 as f32) * x_scale * scale1,
    )
}

#[cfg(all(
    feature = "x86-int8-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[target_feature(enable = "avx,avx512f,avx512bw")]
unsafe fn dot3_i8_i8_x86_avx512(
    x: &[i8],
    x_scale: f32,
    rows: [I8ScaledRow<'_>; 3],
) -> (f32, f32, f32) {
    let [row0, row1, row2] = rows;
    let (sum0, sum1, sum2) =
        unsafe { dot3_i8_i8_i32_x86_avx512(x, row0.values, row1.values, row2.values) };
    (
        (sum0 as f32) * x_scale * row0.scale,
        (sum1 as f32) * x_scale * row1.scale,
        (sum2 as f32) * x_scale * row2.scale,
    )
}

#[cfg(all(
    feature = "x86-int8-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn dot_i8_i8_i32_x86_avx2(x: &[i8], row: &[i8]) -> i32 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let k_dim = x.len();
    let mut kk = 0usize;
    let mut acc0 = _mm256_setzero_si256();
    let mut acc1 = _mm256_setzero_si256();

    while kk + 32 <= k_dim {
        let x_lo = unsafe { load_i8_as_i16x16_x86(x.as_ptr().add(kk)) };
        let x_hi = unsafe { load_i8_as_i16x16_x86(x.as_ptr().add(kk + 16)) };
        let row_lo = unsafe { load_i8_as_i16x16_x86(row.as_ptr().add(kk)) };
        let row_hi = unsafe { load_i8_as_i16x16_x86(row.as_ptr().add(kk + 16)) };
        acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(row_lo, x_lo));
        acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(row_hi, x_hi));
        kk += 32;
    }

    while kk + 16 <= k_dim {
        let x_chunk = unsafe { load_i8_as_i16x16_x86(x.as_ptr().add(kk)) };
        let row_chunk = unsafe { load_i8_as_i16x16_x86(row.as_ptr().add(kk)) };
        acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(row_chunk, x_chunk));
        kk += 16;
    }

    let mut sum = unsafe { reduce_i32x8_x86(acc0) + reduce_i32x8_x86(acc1) };
    while kk < k_dim {
        sum += row[kk] as i32 * x[kk] as i32;
        kk += 1;
    }
    sum
}

#[cfg(all(
    feature = "x86-int8-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn dot2_i8_i8_i32_x86_avx2(x: &[i8], row0: &[i8], row1: &[i8]) -> (i32, i32) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let k_dim = x.len();
    let mut kk = 0usize;
    let mut acc00 = _mm256_setzero_si256();
    let mut acc01 = _mm256_setzero_si256();
    let mut acc10 = _mm256_setzero_si256();
    let mut acc11 = _mm256_setzero_si256();

    while kk + 32 <= k_dim {
        let x_lo = unsafe { load_i8_as_i16x16_x86(x.as_ptr().add(kk)) };
        let x_hi = unsafe { load_i8_as_i16x16_x86(x.as_ptr().add(kk + 16)) };
        let row0_lo = unsafe { load_i8_as_i16x16_x86(row0.as_ptr().add(kk)) };
        let row0_hi = unsafe { load_i8_as_i16x16_x86(row0.as_ptr().add(kk + 16)) };
        let row1_lo = unsafe { load_i8_as_i16x16_x86(row1.as_ptr().add(kk)) };
        let row1_hi = unsafe { load_i8_as_i16x16_x86(row1.as_ptr().add(kk + 16)) };
        acc00 = _mm256_add_epi32(acc00, _mm256_madd_epi16(row0_lo, x_lo));
        acc01 = _mm256_add_epi32(acc01, _mm256_madd_epi16(row0_hi, x_hi));
        acc10 = _mm256_add_epi32(acc10, _mm256_madd_epi16(row1_lo, x_lo));
        acc11 = _mm256_add_epi32(acc11, _mm256_madd_epi16(row1_hi, x_hi));
        kk += 32;
    }

    while kk + 16 <= k_dim {
        let x_chunk = unsafe { load_i8_as_i16x16_x86(x.as_ptr().add(kk)) };
        let row0_chunk = unsafe { load_i8_as_i16x16_x86(row0.as_ptr().add(kk)) };
        let row1_chunk = unsafe { load_i8_as_i16x16_x86(row1.as_ptr().add(kk)) };
        acc00 = _mm256_add_epi32(acc00, _mm256_madd_epi16(row0_chunk, x_chunk));
        acc10 = _mm256_add_epi32(acc10, _mm256_madd_epi16(row1_chunk, x_chunk));
        kk += 16;
    }

    let mut sum0 = unsafe { reduce_i32x8_x86(acc00) + reduce_i32x8_x86(acc01) };
    let mut sum1 = unsafe { reduce_i32x8_x86(acc10) + reduce_i32x8_x86(acc11) };
    while kk < k_dim {
        let xv = x[kk] as i32;
        sum0 += row0[kk] as i32 * xv;
        sum1 += row1[kk] as i32 * xv;
        kk += 1;
    }
    (sum0, sum1)
}

#[cfg(all(
    feature = "x86-int8-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn dot3_i8_i8_i32_x86_avx2(
    x: &[i8],
    row0: &[i8],
    row1: &[i8],
    row2: &[i8],
) -> (i32, i32, i32) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let k_dim = x.len();
    let mut kk = 0usize;
    let mut acc00 = _mm256_setzero_si256();
    let mut acc01 = _mm256_setzero_si256();
    let mut acc10 = _mm256_setzero_si256();
    let mut acc11 = _mm256_setzero_si256();
    let mut acc20 = _mm256_setzero_si256();
    let mut acc21 = _mm256_setzero_si256();

    while kk + 32 <= k_dim {
        let x_lo = unsafe { load_i8_as_i16x16_x86(x.as_ptr().add(kk)) };
        let x_hi = unsafe { load_i8_as_i16x16_x86(x.as_ptr().add(kk + 16)) };
        let row0_lo = unsafe { load_i8_as_i16x16_x86(row0.as_ptr().add(kk)) };
        let row0_hi = unsafe { load_i8_as_i16x16_x86(row0.as_ptr().add(kk + 16)) };
        let row1_lo = unsafe { load_i8_as_i16x16_x86(row1.as_ptr().add(kk)) };
        let row1_hi = unsafe { load_i8_as_i16x16_x86(row1.as_ptr().add(kk + 16)) };
        let row2_lo = unsafe { load_i8_as_i16x16_x86(row2.as_ptr().add(kk)) };
        let row2_hi = unsafe { load_i8_as_i16x16_x86(row2.as_ptr().add(kk + 16)) };
        acc00 = _mm256_add_epi32(acc00, _mm256_madd_epi16(row0_lo, x_lo));
        acc01 = _mm256_add_epi32(acc01, _mm256_madd_epi16(row0_hi, x_hi));
        acc10 = _mm256_add_epi32(acc10, _mm256_madd_epi16(row1_lo, x_lo));
        acc11 = _mm256_add_epi32(acc11, _mm256_madd_epi16(row1_hi, x_hi));
        acc20 = _mm256_add_epi32(acc20, _mm256_madd_epi16(row2_lo, x_lo));
        acc21 = _mm256_add_epi32(acc21, _mm256_madd_epi16(row2_hi, x_hi));
        kk += 32;
    }

    while kk + 16 <= k_dim {
        let x_chunk = unsafe { load_i8_as_i16x16_x86(x.as_ptr().add(kk)) };
        let row0_chunk = unsafe { load_i8_as_i16x16_x86(row0.as_ptr().add(kk)) };
        let row1_chunk = unsafe { load_i8_as_i16x16_x86(row1.as_ptr().add(kk)) };
        let row2_chunk = unsafe { load_i8_as_i16x16_x86(row2.as_ptr().add(kk)) };
        acc00 = _mm256_add_epi32(acc00, _mm256_madd_epi16(row0_chunk, x_chunk));
        acc10 = _mm256_add_epi32(acc10, _mm256_madd_epi16(row1_chunk, x_chunk));
        acc20 = _mm256_add_epi32(acc20, _mm256_madd_epi16(row2_chunk, x_chunk));
        kk += 16;
    }

    let mut sum0 = unsafe { reduce_i32x8_x86(acc00) + reduce_i32x8_x86(acc01) };
    let mut sum1 = unsafe { reduce_i32x8_x86(acc10) + reduce_i32x8_x86(acc11) };
    let mut sum2 = unsafe { reduce_i32x8_x86(acc20) + reduce_i32x8_x86(acc21) };
    while kk < k_dim {
        let xv = x[kk] as i32;
        sum0 += row0[kk] as i32 * xv;
        sum1 += row1[kk] as i32 * xv;
        sum2 += row2[kk] as i32 * xv;
        kk += 1;
    }
    (sum0, sum1, sum2)
}

#[cfg(all(
    feature = "x86-int8-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[target_feature(enable = "avx2")]
unsafe fn dot_i8_i8_x86_avx2(x: &[i8], x_scale: f32, row: &[i8], row_scale: f32) -> f32 {
    let sum = unsafe { dot_i8_i8_i32_x86_avx2(x, row) };
    (sum as f32) * x_scale * row_scale
}

#[cfg(all(
    feature = "x86-int8-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[target_feature(enable = "avx2")]
unsafe fn sum_i8_x86_avx2(x: &[i8], scale: f32) -> f32 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let len = x.len();
    let ones = _mm256_set1_epi16(1);
    let mut kk = 0usize;
    let mut acc0 = _mm256_setzero_si256();
    let mut acc1 = _mm256_setzero_si256();

    while kk + 32 <= len {
        let x0 = unsafe { load_i8_as_i16x16_x86(x.as_ptr().add(kk)) };
        let x1 = unsafe { load_i8_as_i16x16_x86(x.as_ptr().add(kk + 16)) };
        acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(x0, ones));
        acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(x1, ones));
        kk += 32;
    }

    while kk + 16 <= len {
        let chunk = unsafe { load_i8_as_i16x16_x86(x.as_ptr().add(kk)) };
        acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(chunk, ones));
        kk += 16;
    }

    let mut sum = unsafe { reduce_i32x8_x86(acc0) + reduce_i32x8_x86(acc1) };
    while kk < len {
        sum += x[kk] as i32;
        kk += 1;
    }
    (sum as f32) * scale
}

#[cfg(all(
    feature = "x86-int8-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[target_feature(enable = "avx2")]
unsafe fn dot2_i8_i8_x86_avx2(
    x: &[i8],
    x_scale: f32,
    row0: &[i8],
    scale0: f32,
    row1: &[i8],
    scale1: f32,
) -> (f32, f32) {
    let (sum0, sum1) = unsafe { dot2_i8_i8_i32_x86_avx2(x, row0, row1) };
    (
        (sum0 as f32) * x_scale * scale0,
        (sum1 as f32) * x_scale * scale1,
    )
}

#[cfg(all(
    feature = "x86-int8-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[target_feature(enable = "avx2")]
unsafe fn dot3_i8_i8_x86_avx2(
    x: &[i8],
    x_scale: f32,
    rows: [I8ScaledRow<'_>; 3],
) -> (f32, f32, f32) {
    let [row0, row1, row2] = rows;
    let (sum0, sum1, sum2) =
        unsafe { dot3_i8_i8_i32_x86_avx2(x, row0.values, row1.values, row2.values) };
    (
        (sum0 as f32) * x_scale * row0.scale,
        (sum1 as f32) * x_scale * row1.scale,
        (sum2 as f32) * x_scale * row2.scale,
    )
}

#[cfg(all(
    feature = "x86-int8-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn store_i32x8_as_clamped_i8_x86(data: &mut [i8], idx: usize, values: __m256i) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let mut buf = [0i32; 8];
    unsafe { _mm256_storeu_si256(buf.as_mut_ptr() as *mut __m256i, values) };
    for (offset, value) in buf.iter().enumerate() {
        data[idx + offset] = (*value).clamp(-127, 127) as i8;
    }
}

// Match f32::round() (half away from zero), including adjacent representable
// values. Clamp in floating point before conversion so a large positive value
// cannot become the integer-conversion overflow sentinel and flip sign.
#[cfg(all(
    feature = "x86-int8-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn requantize_i8_x86_avx2(value: __m256) -> __m256i {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    let value = _mm256_and_ps(value, _mm256_cmp_ps::<{ _CMP_ORD_Q }>(value, value));
    let value = _mm256_min_ps(
        _mm256_max_ps(value, _mm256_set1_ps(-127.0)),
        _mm256_set1_ps(127.0),
    );
    let sign_mask = _mm256_set1_ps(-0.0);
    let magnitude = _mm256_andnot_ps(sign_mask, value);
    let base = _mm256_floor_ps(magnitude);
    let fraction = _mm256_sub_ps(magnitude, base);
    let increment = _mm256_and_ps(
        _mm256_cmp_ps::<{ _CMP_GE_OQ }>(fraction, _mm256_set1_ps(0.5)),
        _mm256_set1_ps(1.0),
    );
    let rounded = _mm256_or_ps(
        _mm256_add_ps(base, increment),
        _mm256_and_ps(sign_mask, value),
    );
    _mm256_cvttps_epi32(rounded)
}

#[cfg(all(
    feature = "x86-int8-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[inline]
#[target_feature(enable = "avx,avx512f,avx512bw")]
unsafe fn requantize_i8_x86_avx512(value: __m512) -> __m512i {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    let value = _mm512_maskz_mov_ps(_mm512_cmp_ps_mask::<{ _CMP_ORD_Q }>(value, value), value);
    let value = _mm512_min_ps(
        _mm512_max_ps(value, _mm512_set1_ps(-127.0)),
        _mm512_set1_ps(127.0),
    );
    let sign_mask = _mm512_set1_epi32(i32::MIN);
    let bits = _mm512_castps_si512(value);
    let magnitude = _mm512_castsi512_ps(_mm512_andnot_si512(sign_mask, bits));
    let base = _mm512_roundscale_ps::<{ _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC }>(magnitude);
    let fraction = _mm512_sub_ps(magnitude, base);
    let increment = _mm512_maskz_mov_ps(
        _mm512_cmp_ps_mask::<{ _CMP_GE_OQ }>(fraction, _mm512_set1_ps(0.5)),
        _mm512_set1_ps(1.0),
    );
    let rounded = _mm512_or_si512(
        _mm512_castps_si512(_mm512_add_ps(base, increment)),
        _mm512_and_si512(sign_mask, bits),
    );
    _mm512_cvttps_epi32(_mm512_castsi512_ps(rounded))
}

#[cfg(all(
    feature = "x86-int8-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[target_feature(enable = "avx,avx512f,avx512bw")]
unsafe fn sgd_update_i8_f32_x86_avx512(data: &mut [i8], scale: f32, grad: &[f32], lr: f32) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let len = data.len();
    let mut idx = 0usize;
    let scale_v = _mm512_set1_ps(scale);
    let inv_scale_v = _mm512_set1_ps(1.0 / scale);
    let lr_v = _mm512_set1_ps(lr);
    let min_q = _mm512_set1_epi32(-127);
    let max_q = _mm512_set1_epi32(127);

    while idx + 16 <= len {
        let packed = unsafe { _mm_loadu_si128(data.as_ptr().add(idx) as *const __m128i) };
        let q_i32 = _mm512_cvtepi8_epi32(packed);
        let weight = _mm512_mul_ps(_mm512_cvtepi32_ps(q_i32), scale_v);
        let grad_v = unsafe { _mm512_loadu_ps(grad.as_ptr().add(idx)) };
        let updated = _mm512_sub_ps(weight, _mm512_mul_ps(lr_v, grad_v));
        let requant = unsafe { requantize_i8_x86_avx512(_mm512_mul_ps(updated, inv_scale_v)) };
        let requant = _mm512_min_epi32(_mm512_max_epi32(requant, min_q), max_q);
        let bytes = _mm512_cvtsepi32_epi8(requant);
        unsafe { _mm_storeu_si128(data.as_mut_ptr().add(idx) as *mut __m128i, bytes) };
        idx += 16;
    }

    let inv_scale = 1.0 / scale;
    while idx < len {
        let updated = (data[idx] as f32) * scale - lr * grad[idx];
        data[idx] = (updated * inv_scale).round().clamp(-127.0, 127.0) as i8;
        idx += 1;
    }
}

#[cfg(all(
    feature = "x86-int8-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[target_feature(enable = "avx,avx512f,avx512bw")]
unsafe fn sgd_momentum_update_i8_f32_x86_avx512(
    data: &mut [i8],
    scale: f32,
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
    let mut idx = 0usize;
    let scale_v = _mm512_set1_ps(scale);
    let inv_scale_v = _mm512_set1_ps(1.0 / scale);
    let lr_v = _mm512_set1_ps(lr);
    let momentum_v = _mm512_set1_ps(momentum);
    let min_q = _mm512_set1_epi32(-127);
    let max_q = _mm512_set1_epi32(127);

    while idx + 16 <= len {
        let packed = unsafe { _mm_loadu_si128(data.as_ptr().add(idx) as *const __m128i) };
        let q_i32 = _mm512_cvtepi8_epi32(packed);
        let weight = _mm512_mul_ps(_mm512_cvtepi32_ps(q_i32), scale_v);
        let old_velocity = unsafe { _mm512_loadu_ps(velocity.as_ptr().add(idx)) };
        let grad_v = unsafe { _mm512_loadu_ps(grad.as_ptr().add(idx)) };
        let new_velocity = _mm512_add_ps(_mm512_mul_ps(momentum_v, old_velocity), grad_v);
        unsafe { _mm512_storeu_ps(velocity.as_mut_ptr().add(idx), new_velocity) };
        let updated = _mm512_sub_ps(weight, _mm512_mul_ps(lr_v, new_velocity));
        let requant = unsafe { requantize_i8_x86_avx512(_mm512_mul_ps(updated, inv_scale_v)) };
        let requant = _mm512_min_epi32(_mm512_max_epi32(requant, min_q), max_q);
        let bytes = _mm512_cvtsepi32_epi8(requant);
        unsafe { _mm_storeu_si128(data.as_mut_ptr().add(idx) as *mut __m128i, bytes) };
        idx += 16;
    }

    let inv_scale = 1.0 / scale;
    while idx < len {
        velocity[idx] = momentum * velocity[idx] + grad[idx];
        let updated = (data[idx] as f32) * scale - lr * velocity[idx];
        data[idx] = (updated * inv_scale).round().clamp(-127.0, 127.0) as i8;
        idx += 1;
    }
}

#[cfg(all(
    feature = "x86-int8-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx,avx512f,avx512bw")]
unsafe fn adam_update_i8_f32_x86_avx512(
    data: &mut [i8],
    scale: f32,
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
    let mut idx = 0usize;
    let scale_v = _mm512_set1_ps(scale);
    let inv_scale_v = _mm512_set1_ps(1.0 / scale);
    let lr_v = _mm512_set1_ps(lr);
    let beta1_v = _mm512_set1_ps(beta1);
    let one_minus_beta1_v = _mm512_set1_ps(1.0 - beta1);
    let beta2_v = _mm512_set1_ps(beta2);
    let one_minus_beta2_v = _mm512_set1_ps(1.0 - beta2);
    let inv_bias_correction1_v = _mm512_set1_ps(1.0 / bias_correction1);
    let inv_bias_correction2_v = _mm512_set1_ps(1.0 / bias_correction2);
    let eps_v = _mm512_set1_ps(eps);
    let min_q = _mm512_set1_epi32(-127);
    let max_q = _mm512_set1_epi32(127);

    while idx + 16 <= len {
        let packed = unsafe { _mm_loadu_si128(data.as_ptr().add(idx) as *const __m128i) };
        let q_i32 = _mm512_cvtepi8_epi32(packed);
        let weight = _mm512_mul_ps(_mm512_cvtepi32_ps(q_i32), scale_v);

        let old_m = unsafe { _mm512_loadu_ps(exp_avg.as_ptr().add(idx)) };
        let old_v = unsafe { _mm512_loadu_ps(exp_avg_sq.as_ptr().add(idx)) };
        let grad_v = unsafe { _mm512_loadu_ps(grad.as_ptr().add(idx)) };

        let new_m = _mm512_add_ps(
            _mm512_mul_ps(beta1_v, old_m),
            _mm512_mul_ps(one_minus_beta1_v, grad_v),
        );
        let grad_sq = _mm512_mul_ps(grad_v, grad_v);
        let new_v = _mm512_add_ps(
            _mm512_mul_ps(beta2_v, old_v),
            _mm512_mul_ps(one_minus_beta2_v, grad_sq),
        );
        unsafe { _mm512_storeu_ps(exp_avg.as_mut_ptr().add(idx), new_m) };
        unsafe { _mm512_storeu_ps(exp_avg_sq.as_mut_ptr().add(idx), new_v) };

        let m_hat = _mm512_mul_ps(new_m, inv_bias_correction1_v);
        let v_hat = _mm512_mul_ps(new_v, inv_bias_correction2_v);
        let denom = _mm512_add_ps(_mm512_sqrt_ps(v_hat), eps_v);
        let step = _mm512_mul_ps(lr_v, _mm512_div_ps(m_hat, denom));
        let updated = _mm512_sub_ps(weight, step);
        let requant = unsafe { requantize_i8_x86_avx512(_mm512_mul_ps(updated, inv_scale_v)) };
        let requant = _mm512_min_epi32(_mm512_max_epi32(requant, min_q), max_q);
        let bytes = _mm512_cvtsepi32_epi8(requant);
        unsafe { _mm_storeu_si128(data.as_mut_ptr().add(idx) as *mut __m128i, bytes) };
        idx += 16;
    }

    let inv_scale = 1.0 / scale;
    while idx < len {
        exp_avg[idx] = beta1 * exp_avg[idx] + (1.0 - beta1) * grad[idx];
        exp_avg_sq[idx] = beta2 * exp_avg_sq[idx] + (1.0 - beta2) * grad[idx] * grad[idx];
        let m_hat = exp_avg[idx] / bias_correction1;
        let v_hat = exp_avg_sq[idx] / bias_correction2;
        let updated = (data[idx] as f32) * scale - lr * (m_hat / (v_hat.sqrt() + eps));
        data[idx] = (updated * inv_scale).round().clamp(-127.0, 127.0) as i8;
        idx += 1;
    }
}

#[cfg(all(
    feature = "x86-int8-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[target_feature(enable = "avx2,fma")]
unsafe fn sgd_update_i8_f32_x86_avx2(data: &mut [i8], scale: f32, grad: &[f32], lr: f32) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let len = data.len();
    let mut idx = 0usize;
    let scale_v = _mm256_set1_ps(scale);
    let inv_scale_v = _mm256_set1_ps(1.0 / scale);
    let lr_v = _mm256_set1_ps(lr);
    let min_q = _mm256_set1_epi32(-127);
    let max_q = _mm256_set1_epi32(127);

    while idx + 8 <= len {
        let packed = unsafe { _mm_loadl_epi64(data.as_ptr().add(idx) as *const __m128i) };
        let q_i32 = _mm256_cvtepi8_epi32(packed);
        let weight = _mm256_mul_ps(_mm256_cvtepi32_ps(q_i32), scale_v);
        let grad_v = unsafe { _mm256_loadu_ps(grad.as_ptr().add(idx)) };
        let updated = _mm256_sub_ps(weight, _mm256_mul_ps(lr_v, grad_v));
        let requant = unsafe { requantize_i8_x86_avx2(_mm256_mul_ps(updated, inv_scale_v)) };
        let requant = _mm256_min_epi32(_mm256_max_epi32(requant, min_q), max_q);
        unsafe { store_i32x8_as_clamped_i8_x86(data, idx, requant) };
        idx += 8;
    }

    let inv_scale = 1.0 / scale;
    while idx < len {
        let updated = (data[idx] as f32) * scale - lr * grad[idx];
        data[idx] = (updated * inv_scale).round().clamp(-127.0, 127.0) as i8;
        idx += 1;
    }
}

#[cfg(all(
    feature = "x86-int8-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[target_feature(enable = "avx2,fma")]
unsafe fn sgd_momentum_update_i8_f32_x86_avx2(
    data: &mut [i8],
    scale: f32,
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
    let mut idx = 0usize;
    let scale_v = _mm256_set1_ps(scale);
    let inv_scale_v = _mm256_set1_ps(1.0 / scale);
    let lr_v = _mm256_set1_ps(lr);
    let momentum_v = _mm256_set1_ps(momentum);
    let min_q = _mm256_set1_epi32(-127);
    let max_q = _mm256_set1_epi32(127);

    while idx + 8 <= len {
        let packed = unsafe { _mm_loadl_epi64(data.as_ptr().add(idx) as *const __m128i) };
        let q_i32 = _mm256_cvtepi8_epi32(packed);
        let weight = _mm256_mul_ps(_mm256_cvtepi32_ps(q_i32), scale_v);
        let old_velocity = unsafe { _mm256_loadu_ps(velocity.as_ptr().add(idx)) };
        let grad_v = unsafe { _mm256_loadu_ps(grad.as_ptr().add(idx)) };
        let new_velocity = _mm256_add_ps(_mm256_mul_ps(momentum_v, old_velocity), grad_v);
        unsafe { _mm256_storeu_ps(velocity.as_mut_ptr().add(idx), new_velocity) };
        let updated = _mm256_sub_ps(weight, _mm256_mul_ps(lr_v, new_velocity));
        let requant = unsafe { requantize_i8_x86_avx2(_mm256_mul_ps(updated, inv_scale_v)) };
        let requant = _mm256_min_epi32(_mm256_max_epi32(requant, min_q), max_q);
        unsafe { store_i32x8_as_clamped_i8_x86(data, idx, requant) };
        idx += 8;
    }

    let inv_scale = 1.0 / scale;
    while idx < len {
        velocity[idx] = momentum * velocity[idx] + grad[idx];
        let updated = (data[idx] as f32) * scale - lr * velocity[idx];
        data[idx] = (updated * inv_scale).round().clamp(-127.0, 127.0) as i8;
        idx += 1;
    }
}

#[cfg(all(
    feature = "x86-int8-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2,fma")]
unsafe fn adam_update_i8_f32_x86_avx2(
    data: &mut [i8],
    scale: f32,
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
    let mut idx = 0usize;
    let scale_v = _mm256_set1_ps(scale);
    let inv_scale_v = _mm256_set1_ps(1.0 / scale);
    let lr_v = _mm256_set1_ps(lr);
    let beta1_v = _mm256_set1_ps(beta1);
    let one_minus_beta1_v = _mm256_set1_ps(1.0 - beta1);
    let beta2_v = _mm256_set1_ps(beta2);
    let one_minus_beta2_v = _mm256_set1_ps(1.0 - beta2);
    let inv_bias_correction1_v = _mm256_set1_ps(1.0 / bias_correction1);
    let inv_bias_correction2_v = _mm256_set1_ps(1.0 / bias_correction2);
    let eps_v = _mm256_set1_ps(eps);
    let min_q = _mm256_set1_epi32(-127);
    let max_q = _mm256_set1_epi32(127);

    while idx + 8 <= len {
        let packed = unsafe { _mm_loadl_epi64(data.as_ptr().add(idx) as *const __m128i) };
        let q_i32 = _mm256_cvtepi8_epi32(packed);
        let weight = _mm256_mul_ps(_mm256_cvtepi32_ps(q_i32), scale_v);

        let old_m = unsafe { _mm256_loadu_ps(exp_avg.as_ptr().add(idx)) };
        let old_v = unsafe { _mm256_loadu_ps(exp_avg_sq.as_ptr().add(idx)) };
        let grad_v = unsafe { _mm256_loadu_ps(grad.as_ptr().add(idx)) };

        let new_m = _mm256_add_ps(
            _mm256_mul_ps(beta1_v, old_m),
            _mm256_mul_ps(one_minus_beta1_v, grad_v),
        );
        let grad_sq = _mm256_mul_ps(grad_v, grad_v);
        let new_v = _mm256_add_ps(
            _mm256_mul_ps(beta2_v, old_v),
            _mm256_mul_ps(one_minus_beta2_v, grad_sq),
        );
        unsafe { _mm256_storeu_ps(exp_avg.as_mut_ptr().add(idx), new_m) };
        unsafe { _mm256_storeu_ps(exp_avg_sq.as_mut_ptr().add(idx), new_v) };

        let m_hat = _mm256_mul_ps(new_m, inv_bias_correction1_v);
        let v_hat = _mm256_mul_ps(new_v, inv_bias_correction2_v);
        let denom = _mm256_add_ps(_mm256_sqrt_ps(v_hat), eps_v);
        let step = _mm256_mul_ps(lr_v, _mm256_div_ps(m_hat, denom));
        let updated = _mm256_sub_ps(weight, step);
        let requant = unsafe { requantize_i8_x86_avx2(_mm256_mul_ps(updated, inv_scale_v)) };
        let requant = _mm256_min_epi32(_mm256_max_epi32(requant, min_q), max_q);
        unsafe { store_i32x8_as_clamped_i8_x86(data, idx, requant) };
        idx += 8;
    }

    let inv_scale = 1.0 / scale;
    while idx < len {
        exp_avg[idx] = beta1 * exp_avg[idx] + (1.0 - beta1) * grad[idx];
        exp_avg_sq[idx] = beta2 * exp_avg_sq[idx] + (1.0 - beta2) * grad[idx] * grad[idx];
        let m_hat = exp_avg[idx] / bias_correction1;
        let v_hat = exp_avg_sq[idx] / bias_correction2;
        let updated = (data[idx] as f32) * scale - lr * (m_hat / (v_hat.sqrt() + eps));
        data[idx] = (updated * inv_scale).round().clamp(-127.0, 127.0) as i8;
        idx += 1;
    }
}

#[cfg(all(
    feature = "x86-int8-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[inline]
#[target_feature(enable = "avx,avx512f,avx512bw")]
unsafe fn reduce_f32x16_sum_x86(v: __m512) -> f32 {
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
    feature = "x86-int8-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[inline]
#[target_feature(enable = "avx,avx512f,avx512bw")]
unsafe fn load_i8_as_f32x16_x86(ptr: *const i8) -> __m512 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let raw = unsafe { _mm_loadu_si128(ptr as *const __m128i) };
    _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(raw))
}

#[cfg(all(
    feature = "x86-int8-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[target_feature(enable = "avx,avx512f,avx512bw,fma")]
unsafe fn dot_f32_i8_sums_x86_avx512<const N: usize>(x: &[f32], rows: [&[i8]; N]) -> [f32; N] {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    debug_assert!(N > 0);
    let k_dim = x.len();
    let mut kk = 0usize;
    let zero = _mm512_setzero_ps();
    let mut acc0 = [zero; N];
    let mut acc1 = [zero; N];

    while kk + 32 <= k_dim {
        let x0 = unsafe { _mm512_loadu_ps(x.as_ptr().add(kk)) };
        let x1 = unsafe { _mm512_loadu_ps(x.as_ptr().add(kk + 16)) };
        for lane in 0..N {
            let row0 = unsafe { load_i8_as_f32x16_x86(rows[lane].as_ptr().add(kk)) };
            let row1 = unsafe { load_i8_as_f32x16_x86(rows[lane].as_ptr().add(kk + 16)) };
            acc0[lane] = _mm512_fmadd_ps(row0, x0, acc0[lane]);
            acc1[lane] = _mm512_fmadd_ps(row1, x1, acc1[lane]);
        }
        kk += 32;
    }

    while kk + 16 <= k_dim {
        let x0 = unsafe { _mm512_loadu_ps(x.as_ptr().add(kk)) };
        for lane in 0..N {
            let row0 = unsafe { load_i8_as_f32x16_x86(rows[lane].as_ptr().add(kk)) };
            acc0[lane] = _mm512_fmadd_ps(row0, x0, acc0[lane]);
        }
        kk += 16;
    }

    let mut sums = [0.0f32; N];
    for lane in 0..N {
        sums[lane] =
            unsafe { reduce_f32x16_sum_x86(acc0[lane]) + reduce_f32x16_sum_x86(acc1[lane]) };
    }

    while kk < k_dim {
        let xv = x[kk];
        for lane in 0..N {
            sums[lane] += rows[lane][kk] as f32 * xv;
        }
        kk += 1;
    }
    sums
}

#[cfg(all(
    feature = "x86-int8-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[target_feature(enable = "avx,avx512f,avx512bw,fma")]
unsafe fn dot_f32_i8_x86_avx512(x: &[f32], row: &[i8], scale: f32) -> f32 {
    let [sum] = unsafe { dot_f32_i8_sums_x86_avx512(x, [row]) };
    sum * scale
}

#[cfg(all(
    feature = "x86-int8-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[target_feature(enable = "avx,avx512f,avx512bw,fma")]
unsafe fn dot2_f32_i8_x86_avx512(
    x: &[f32],
    row0: &[i8],
    scale0: f32,
    row1: &[i8],
    scale1: f32,
) -> (f32, f32) {
    let [sum0, sum1] = unsafe { dot_f32_i8_sums_x86_avx512(x, [row0, row1]) };
    (sum0 * scale0, sum1 * scale1)
}

#[cfg(all(
    feature = "x86-int8-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[target_feature(enable = "avx,avx512f,avx512bw,fma")]
unsafe fn dot3_f32_i8_x86_avx512(
    x: &[f32],
    row0: &[i8],
    scale0: f32,
    row1: &[i8],
    scale1: f32,
    row2: &[i8],
    scale2: f32,
) -> (f32, f32, f32) {
    let [sum0, sum1, sum2] = unsafe { dot_f32_i8_sums_x86_avx512(x, [row0, row1, row2]) };
    (sum0 * scale0, sum1 * scale1, sum2 * scale2)
}

#[cfg(all(
    feature = "x86-int8-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_f32_i8_x86_avx2(x: &[f32], row: &[i8], scale: f32) -> f32 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let k_dim = x.len();
    let mut kk = 0usize;
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();

    while kk + 16 <= k_dim {
        let row_lo = unsafe { load_i8_as_f32x8_x86(row.as_ptr().add(kk)) };
        let row_hi = unsafe { load_i8_as_f32x8_x86(row.as_ptr().add(kk + 8)) };
        let x_lo = unsafe { _mm256_loadu_ps(x.as_ptr().add(kk)) };
        let x_hi = unsafe { _mm256_loadu_ps(x.as_ptr().add(kk + 8)) };
        acc0 = _mm256_fmadd_ps(row_lo, x_lo, acc0);
        acc1 = _mm256_fmadd_ps(row_hi, x_hi, acc1);
        kk += 16;
    }

    while kk + 8 <= k_dim {
        let row_f32 = unsafe { load_i8_as_f32x8_x86(row.as_ptr().add(kk)) };
        let x_chunk = unsafe { _mm256_loadu_ps(x.as_ptr().add(kk)) };
        acc0 = _mm256_fmadd_ps(row_f32, x_chunk, acc0);
        kk += 8;
    }

    let mut sum: f32 = unsafe { reduce_f32x8_x86(acc0) + reduce_f32x8_x86(acc1) };
    while kk < k_dim {
        sum += row[kk] as f32 * x[kk];
        kk += 1;
    }
    sum * scale
}

#[cfg(all(
    feature = "x86-int8-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot2_f32_i8_x86_avx2(
    x: &[f32],
    row0: &[i8],
    scale0: f32,
    row1: &[i8],
    scale1: f32,
) -> (f32, f32) {
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
        let row0_lo = unsafe { load_i8_as_f32x8_x86(row0.as_ptr().add(kk)) };
        let row0_hi = unsafe { load_i8_as_f32x8_x86(row0.as_ptr().add(kk + 8)) };
        let row1_lo = unsafe { load_i8_as_f32x8_x86(row1.as_ptr().add(kk)) };
        let row1_hi = unsafe { load_i8_as_f32x8_x86(row1.as_ptr().add(kk + 8)) };
        acc00 = _mm256_fmadd_ps(row0_lo, x_lo, acc00);
        acc01 = _mm256_fmadd_ps(row0_hi, x_hi, acc01);
        acc10 = _mm256_fmadd_ps(row1_lo, x_lo, acc10);
        acc11 = _mm256_fmadd_ps(row1_hi, x_hi, acc11);
        kk += 16;
    }

    while kk + 8 <= k_dim {
        let x_chunk = unsafe { _mm256_loadu_ps(x.as_ptr().add(kk)) };
        let row0_f32 = unsafe { load_i8_as_f32x8_x86(row0.as_ptr().add(kk)) };
        let row1_f32 = unsafe { load_i8_as_f32x8_x86(row1.as_ptr().add(kk)) };
        acc00 = _mm256_fmadd_ps(row0_f32, x_chunk, acc00);
        acc10 = _mm256_fmadd_ps(row1_f32, x_chunk, acc10);
        kk += 8;
    }

    let mut sum0 = unsafe { reduce_f32x8_x86(acc00) + reduce_f32x8_x86(acc01) };
    let mut sum1 = unsafe { reduce_f32x8_x86(acc10) + reduce_f32x8_x86(acc11) };
    while kk < k_dim {
        let xv = x[kk];
        sum0 += row0[kk] as f32 * xv;
        sum1 += row1[kk] as f32 * xv;
        kk += 1;
    }
    (sum0 * scale0, sum1 * scale1)
}

#[cfg(all(
    feature = "x86-int8-kernels",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot3_f32_i8_x86_avx2(
    x: &[f32],
    row0: &[i8],
    scale0: f32,
    row1: &[i8],
    scale1: f32,
    row2: &[i8],
    scale2: f32,
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
        let row0_lo = unsafe { load_i8_as_f32x8_x86(row0.as_ptr().add(kk)) };
        let row0_hi = unsafe { load_i8_as_f32x8_x86(row0.as_ptr().add(kk + 8)) };
        let row1_lo = unsafe { load_i8_as_f32x8_x86(row1.as_ptr().add(kk)) };
        let row1_hi = unsafe { load_i8_as_f32x8_x86(row1.as_ptr().add(kk + 8)) };
        let row2_lo = unsafe { load_i8_as_f32x8_x86(row2.as_ptr().add(kk)) };
        let row2_hi = unsafe { load_i8_as_f32x8_x86(row2.as_ptr().add(kk + 8)) };
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
        let row0_f32 = unsafe { load_i8_as_f32x8_x86(row0.as_ptr().add(kk)) };
        let row1_f32 = unsafe { load_i8_as_f32x8_x86(row1.as_ptr().add(kk)) };
        let row2_f32 = unsafe { load_i8_as_f32x8_x86(row2.as_ptr().add(kk)) };
        acc00 = _mm256_fmadd_ps(row0_f32, x_chunk, acc00);
        acc10 = _mm256_fmadd_ps(row1_f32, x_chunk, acc10);
        acc20 = _mm256_fmadd_ps(row2_f32, x_chunk, acc20);
        kk += 8;
    }

    let mut sum0 = unsafe { reduce_f32x8_x86(acc00) + reduce_f32x8_x86(acc01) };
    let mut sum1 = unsafe { reduce_f32x8_x86(acc10) + reduce_f32x8_x86(acc11) };
    let mut sum2 = unsafe { reduce_f32x8_x86(acc20) + reduce_f32x8_x86(acc21) };
    while kk < k_dim {
        let xv = x[kk];
        sum0 += row0[kk] as f32 * xv;
        sum1 += row1[kk] as f32 * xv;
        sum2 += row2[kk] as f32 * xv;
        kk += 1;
    }
    (sum0 * scale0, sum1 * scale1, sum2 * scale2)
}

#[cfg(test)]
#[path = "../../tests/unit/ops/int8_kernels/tests.rs"]
mod tests;
