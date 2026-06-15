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
    if _data.len() != _grad.len() || !_scale.is_finite() || _scale <= 0.0 {
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
        let requant = _mm512_cvtps_epi32(_mm512_mul_ps(value, inv_scale_v));
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
        let requant = _mm256_cvtps_epi32(_mm256_mul_ps(value, inv_scale_v));
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
            let requant = _mm512_cvtps_epi32(_mm512_mul_ps(value, inv_scale_v));
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
            let requant = _mm256_cvtps_epi32(_mm256_mul_ps(value, inv_scale_v));
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
        let requant = _mm512_cvtps_epi32(_mm512_mul_ps(updated, inv_scale_v));
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
        let requant = _mm512_cvtps_epi32(_mm512_mul_ps(updated, inv_scale_v));
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
        let requant = _mm512_cvtps_epi32(_mm512_mul_ps(updated, inv_scale_v));
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
        let requant = _mm256_cvtps_epi32(_mm256_mul_ps(updated, inv_scale_v));
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
        let requant = _mm256_cvtps_epi32(_mm256_mul_ps(updated, inv_scale_v));
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
        let requant = _mm256_cvtps_epi32(_mm256_mul_ps(updated, inv_scale_v));
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
mod tests {
    use super::*;

    fn patterned_i8(i: i32, mul: i32, add: i32) -> i8 {
        (((i * mul + add) % 255) - 127).clamp(-127, 127) as i8
    }

    #[test]
    fn backend_name_matches_backend_enum() {
        let name = active_int8_backend_name();
        match active_int8_backend() {
            Int8KernelBackend::Portable => assert_eq!(name, "portable"),
            Int8KernelBackend::Arm64Neon => assert_eq!(name, "arm64-neon"),
            Int8KernelBackend::X86Avx512 => assert_eq!(name, "x86-avx512bw"),
            Int8KernelBackend::X86Avx2 => assert_eq!(name, "x86-avx2"),
        }
    }

    #[test]
    fn dynamic_i8_scale_never_underflows_to_zero() {
        assert_eq!(dynamic_i8_scale(0.0), 1.0);
        assert_eq!(dynamic_i8_scale(f32::from_bits(1)), f32::MIN_POSITIVE);
        assert_eq!(dynamic_i8_scale(f32::INFINITY), 1.0);
    }

    #[test]
    fn i8_i8_backend_name_matches_runtime_path() {
        let backend = active_i8_i8_backend();
        let name = active_i8_i8_backend_name();

        if arch::x86_avx512_i8_i8_kernel_runtime_available() {
            assert_eq!(backend, I8I8KernelBackend::X86Avx512);
            assert_eq!(name, "x86-avx512bw");
        } else if arch::x86_i8_kernel_runtime_available() {
            assert_eq!(backend, I8I8KernelBackend::X86Avx2);
            assert_eq!(name, "x86-avx2-i32acc");
        } else {
            assert_eq!(backend, I8I8KernelBackend::Portable);
            assert_eq!(name, "portable");
        }
    }

    #[test]
    fn i8_i8_dispatch_consistency() {
        let x = [3i8, -2, 5, 1, -4, 6, -1, 2];
        let row0 = [3i8, -2, 5, 1, -4, 6, -1, 2];
        let row1 = [-1i8, 4, -3, 2, 5, -2, 7, -6];
        let row2 = [2i8, 1, -2, 3, -5, 4, 1, -3];
        let has_arch = active_i8_i8_backend() != I8I8KernelBackend::Portable;

        assert_eq!(dot_i8_i8_arch(&x, 0.5, &row0, 0.25).is_some(), has_arch);
        assert_eq!(sum_i8_arch(&x, 0.5).is_some(), has_arch);
        assert_eq!(
            dot2_i8_i8_arch(&x, 0.5, &row0, 0.25, &row1, 0.125).is_some(),
            has_arch
        );
        assert_eq!(
            dot3_i8_i8_arch(
                &x,
                0.5,
                [
                    I8ScaledRow {
                        values: &row0,
                        scale: 0.25
                    },
                    I8ScaledRow {
                        values: &row1,
                        scale: 0.125
                    },
                    I8ScaledRow {
                        values: &row2,
                        scale: 0.0625
                    },
                ]
            )
            .is_some(),
            has_arch
        );
    }

    #[test]
    fn i8_sum_arch_matches_scalar_reference() {
        let x = [
            3i8, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16, 17, -18, 19,
        ];
        let scale = 0.125f32;
        if let Some(got) = sum_i8_arch(&x, scale) {
            let expected = x.iter().map(|&v| (v as f32) * scale).sum::<f32>();
            assert!((got - expected).abs() <= 1e-6);
        }
        assert_eq!(sum_i8_arch(&[], scale), Some(0.0));
        assert_eq!(sum_i8_arch(&x, 0.0), None);
    }

    #[test]
    fn i8_mul_grad_arch_matches_scalar_reference() {
        let grad = [
            0.5f32, -1.0, 2.0, 0.25, -0.75, 1.5, -2.5, 3.0, 0.125, -0.5, 0.75, -1.25, 1.75, -2.25,
            2.75, -3.25, 3.75,
        ];
        let x = [
            3i8, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16, 17, -18, 19,
        ];
        let scale = 0.125f32;
        let mut out = [0.0f32; 17];

        if mul_f32_i8_to_f32_arch(&grad, &x, scale, &mut out) {
            for ((&got, &g), &xv) in out.iter().zip(grad.iter()).zip(x.iter()) {
                assert!((got - g * (xv as f32) * scale).abs() <= 1e-6);
            }
        }
        let scalar = -0.375f32;
        if mul_i8_scalar_to_f32_arch(&x, scale, scalar, &mut out) {
            for (&got, &xv) in out.iter().zip(x.iter()) {
                assert!((got - (xv as f32) * scale * scalar).abs() <= 1e-6);
            }
        }
        if add_i8_scalar_to_f32_arch(&x, scale, scalar, &mut out) {
            for (&got, &xv) in out.iter().zip(x.iter()) {
                assert!((got - (scalar + (xv as f32) * scale)).abs() <= 1e-6);
            }
        }
        if sub_i8_scalar_to_f32_arch(&x, scale, scalar, false, &mut out) {
            for (&got, &xv) in out.iter().zip(x.iter()) {
                assert!((got - (scalar - (xv as f32) * scale)).abs() <= 1e-6);
            }
        }
        if sub_i8_scalar_to_f32_arch(&x, scale, scalar, true, &mut out) {
            for (&got, &xv) in out.iter().zip(x.iter()) {
                assert!((got - ((xv as f32) * scale - scalar)).abs() <= 1e-6);
            }
        }
        assert!(!mul_f32_i8_to_f32_arch(&grad, &x, 0.0, &mut out));
        assert!(!mul_i8_scalar_to_f32_arch(&x, 0.0, scalar, &mut out));
        assert!(!mul_i8_scalar_to_f32_arch(&x, scale, f32::NAN, &mut out));
        assert!(!add_i8_scalar_to_f32_arch(&x, 0.0, scalar, &mut out));
        assert!(!sub_i8_scalar_to_f32_arch(
            &x,
            scale,
            f32::NAN,
            true,
            &mut out
        ));
    }

    #[test]
    fn i8_elementwise_arch_matches_dynamic_quantized_reference() {
        let lhs = [3i8, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16, 17];
        let rhs = [
            -2i8, 6, -8, 10, -12, 14, -16, 18, -20, 22, -24, 26, -28, 30, -32,
        ];
        let lhs_scale = 0.125f32;
        let rhs_scale = 0.0625f32;

        let reference = |op: I8ElementwiseOp| -> (Vec<i8>, f32) {
            let values = lhs
                .iter()
                .zip(rhs.iter())
                .map(|(&a, &b)| {
                    let av = (a as f32) * lhs_scale;
                    let bv = (b as f32) * rhs_scale;
                    match op {
                        I8ElementwiseOp::Add => av + bv,
                        I8ElementwiseOp::Sub => av - bv,
                        I8ElementwiseOp::Mul => av * bv,
                    }
                })
                .collect::<Vec<_>>();
            let max_abs = values.iter().copied().map(f32::abs).fold(0.0f32, f32::max);
            let scale = dynamic_i8_scale(max_abs);
            let inv_scale = 1.0 / scale;
            (
                values
                    .iter()
                    .map(|&v| (v * inv_scale).round().clamp(-127.0, 127.0) as i8)
                    .collect(),
                scale,
            )
        };

        for (op, run) in [
            (
                I8ElementwiseOp::Add,
                add_i8_i8_arch as fn(&[i8], f32, &[i8], f32, &mut [i8]) -> Option<f32>,
            ),
            (I8ElementwiseOp::Sub, sub_i8_i8_arch),
            (I8ElementwiseOp::Mul, mul_i8_i8_arch),
        ] {
            let mut out = vec![0i8; lhs.len()];
            if let Some(scale) = run(&lhs, lhs_scale, &rhs, rhs_scale, &mut out) {
                let (expected, expected_scale) = reference(op);
                assert!((scale - expected_scale).abs() <= 1e-6);
                assert_eq!(out, expected);
            }
        }
    }

    #[test]
    fn i8_elementwise_to_f32_arch_matches_scalar_reference() {
        let lhs = [3i8, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16, 17];
        let rhs = [
            -2i8, 6, -8, 10, -12, 14, -16, 18, -20, 22, -24, 26, -28, 30, -32,
        ];
        let lhs_scale = 0.125f32;
        let rhs_scale = 0.0625f32;
        let mut out = [0.0f32; 15];

        let reference = |a: i8, b: i8, op: I8ElementwiseOp| {
            let av = (a as f32) * lhs_scale;
            let bv = (b as f32) * rhs_scale;
            match op {
                I8ElementwiseOp::Add => av + bv,
                I8ElementwiseOp::Sub => av - bv,
                I8ElementwiseOp::Mul => av * bv,
            }
        };

        if add_i8_i8_to_f32_arch(&lhs, lhs_scale, &rhs, rhs_scale, &mut out) {
            for ((&got, &a), &b) in out.iter().zip(lhs.iter()).zip(rhs.iter()) {
                assert!((got - reference(a, b, I8ElementwiseOp::Add)).abs() <= 1e-6);
            }
        }
        if sub_i8_i8_to_f32_arch(&lhs, lhs_scale, &rhs, rhs_scale, &mut out) {
            for ((&got, &a), &b) in out.iter().zip(lhs.iter()).zip(rhs.iter()) {
                assert!((got - reference(a, b, I8ElementwiseOp::Sub)).abs() <= 1e-6);
            }
        }
        if mul_i8_i8_to_f32_arch(&lhs, lhs_scale, &rhs, rhs_scale, &mut out) {
            for ((&got, &a), &b) in out.iter().zip(lhs.iter()).zip(rhs.iter()) {
                assert!((got - reference(a, b, I8ElementwiseOp::Mul)).abs() <= 1e-6);
            }
        }

        assert!(!add_i8_i8_to_f32_arch(&lhs, 0.0, &rhs, rhs_scale, &mut out));
    }

    #[test]
    fn f32_i8_elementwise_to_f32_arch_matches_scalar_reference() {
        let lhs = [
            0.25f32, -1.5, 2.0, 0.75, -3.25, 4.5, -0.125, 1.875, 2.25, -2.75, 3.5, -4.25, 0.5,
            1.25, -1.0,
        ];
        let rhs = [
            -2i8, 6, -8, 10, -12, 14, -16, 18, -20, 22, -24, 26, -28, 30, -32,
        ];
        let scale = 0.0625f32;
        let mut out = [0.0f32; 15];

        if add_f32_i8_to_f32_arch(&lhs, &rhs, scale, &mut out) {
            for ((&got, &a), &b) in out.iter().zip(lhs.iter()).zip(rhs.iter()) {
                let expected = a + (b as f32) * scale;
                assert!((got - expected).abs() <= 1e-6);
            }
        }
        if sub_f32_i8_to_f32_arch(&lhs, &rhs, scale, false, &mut out) {
            for ((&got, &a), &b) in out.iter().zip(lhs.iter()).zip(rhs.iter()) {
                let expected = a - (b as f32) * scale;
                assert!((got - expected).abs() <= 1e-6);
            }
        }
        if sub_f32_i8_to_f32_arch(&lhs, &rhs, scale, true, &mut out) {
            for ((&got, &a), &b) in out.iter().zip(lhs.iter()).zip(rhs.iter()) {
                let expected = (b as f32) * scale - a;
                assert!((got - expected).abs() <= 1e-6);
            }
        }

        assert!(!add_f32_i8_to_f32_arch(
            &lhs[..lhs.len() - 1],
            &rhs,
            scale,
            &mut out
        ));
        assert!(!sub_f32_i8_to_f32_arch(&lhs, &rhs, 0.0, false, &mut out));
    }

    #[test]
    fn i8_row_broadcast_arch_matches_scalar_reference() {
        let matrix = [
            3i8, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16, 17, -18, -19, 20, -21, 22,
            -23, 24, -25, 26, -27, 28, -29, 30, -31, 32, -33, 34, 35, -36, 37, -38, 39, -40, 41,
            -42,
        ];
        let vector = [2i8, -3, 4, -5, 6, -7, 8, -9];
        let matrix_scale = 0.03125f32;
        let vector_scale = 0.0625f32;
        let last_dim = vector.len();

        let value = |a: i8, a_scale: f32, b: i8, b_scale: f32, op: I8ElementwiseOp| {
            let av = (a as f32) * a_scale;
            let bv = (b as f32) * b_scale;
            match op {
                I8ElementwiseOp::Add => av + bv,
                I8ElementwiseOp::Sub => av - bv,
                I8ElementwiseOp::Mul => av * bv,
            }
        };

        let check = |vector_on_rhs: bool, op: I8ElementwiseOp| {
            let args = I8RowBroadcast {
                lhs: if vector_on_rhs {
                    I8ScaledRow {
                        values: &matrix,
                        scale: matrix_scale,
                    }
                } else {
                    I8ScaledRow {
                        values: &vector,
                        scale: vector_scale,
                    }
                },
                rhs: if vector_on_rhs {
                    I8ScaledRow {
                        values: &vector,
                        scale: vector_scale,
                    }
                } else {
                    I8ScaledRow {
                        values: &matrix,
                        scale: matrix_scale,
                    }
                },
                last_dim,
                vector_on_rhs,
            };

            let values = (0..matrix.len())
                .map(|idx| {
                    if vector_on_rhs {
                        value(
                            matrix[idx],
                            matrix_scale,
                            vector[idx % last_dim],
                            vector_scale,
                            op,
                        )
                    } else {
                        value(
                            vector[idx % last_dim],
                            vector_scale,
                            matrix[idx],
                            matrix_scale,
                            op,
                        )
                    }
                })
                .collect::<Vec<_>>();
            let max_abs = values.iter().copied().map(f32::abs).fold(0.0f32, f32::max);
            let expected_scale = dynamic_i8_scale(max_abs);
            let inv_scale = 1.0 / expected_scale;
            let expected_i8 = values
                .iter()
                .map(|&v| (v * inv_scale).round().clamp(-127.0, 127.0) as i8)
                .collect::<Vec<_>>();

            let mut typed_out = vec![0i8; matrix.len()];
            let got_scale = match op {
                I8ElementwiseOp::Add => add_i8_i8_row_broadcast_arch(args, &mut typed_out),
                I8ElementwiseOp::Sub => sub_i8_i8_row_broadcast_arch(args, &mut typed_out),
                I8ElementwiseOp::Mul => mul_i8_i8_row_broadcast_arch(args, &mut typed_out),
            };
            let Some(got_scale) = got_scale else {
                return;
            };
            assert!((got_scale - expected_scale).abs() <= 1e-6);
            assert_eq!(typed_out, expected_i8);

            let mut f32_out = vec![0.0f32; matrix.len()];
            let handled = match op {
                I8ElementwiseOp::Add => add_i8_i8_row_broadcast_to_f32_arch(args, &mut f32_out),
                I8ElementwiseOp::Sub => sub_i8_i8_row_broadcast_to_f32_arch(args, &mut f32_out),
                I8ElementwiseOp::Mul => mul_i8_i8_row_broadcast_to_f32_arch(args, &mut f32_out),
            };
            assert!(handled);
            for (&got, &expected) in f32_out.iter().zip(values.iter()) {
                assert!((got - expected).abs() <= 1e-6);
            }
        };

        for vector_on_rhs in [true, false] {
            for op in [
                I8ElementwiseOp::Add,
                I8ElementwiseOp::Sub,
                I8ElementwiseOp::Mul,
            ] {
                check(vector_on_rhs, op);
            }
        }

        let bad_args = I8RowBroadcast {
            lhs: I8ScaledRow {
                values: &matrix,
                scale: matrix_scale,
            },
            rhs: I8ScaledRow {
                values: &vector,
                scale: vector_scale,
            },
            last_dim: 0,
            vector_on_rhs: true,
        };
        let mut out = vec![0i8; matrix.len()];
        assert!(add_i8_i8_row_broadcast_arch(bad_args, &mut out).is_none());
    }

    #[test]
    fn i8_i8_avx2_fallback_matches_scalar_reference_when_available() {
        #[cfg(all(
            feature = "x86-int8-kernels",
            any(target_arch = "x86_64", target_arch = "x86")
        ))]
        {
            if !arch::x86_i8_kernel_runtime_available() {
                return;
            }

            let x = [
                3i8, -2, 5, 1, -4, 6, -1, 2, 7, -8, 9, -10, 11, -12, 13, -14, 4, -3, 2,
            ];
            let row0 = [
                -7i8, 3, 12, -5, 9, -11, 15, 1, 8, -4, 6, 10, -13, 14, -2, 5, 2, -3, 7,
            ];
            let row1 = [
                4i8, -9, 6, 2, -8, 7, -5, 3, -1, 10, -6, 11, -12, 13, -4, 8, -2, 1, 5,
            ];
            let row2 = [
                -3i8, 2, -7, 9, -10, 6, 4, -1, 12, -8, 5, -6, 11, -13, 7, -2, 3, -4, 10,
            ];
            let x_scale = 0.0625f32;
            let scale0 = 0.125f32;
            let scale1 = 0.0625f32;
            let scale2 = 0.25f32;
            let scalar = |row: &[i8], scale: f32| -> f32 {
                x.iter()
                    .zip(row.iter())
                    .map(|(&xv, &rv)| xv as i32 * rv as i32)
                    .sum::<i32>() as f32
                    * x_scale
                    * scale
            };

            let single = unsafe { dot_i8_i8_x86_avx2(&x, x_scale, &row0, scale0) };
            let pair = unsafe { dot2_i8_i8_x86_avx2(&x, x_scale, &row0, scale0, &row1, scale1) };
            let triple = unsafe {
                dot3_i8_i8_x86_avx2(
                    &x,
                    x_scale,
                    [
                        I8ScaledRow {
                            values: &row0,
                            scale: scale0,
                        },
                        I8ScaledRow {
                            values: &row1,
                            scale: scale1,
                        },
                        I8ScaledRow {
                            values: &row2,
                            scale: scale2,
                        },
                    ],
                )
            };

            assert!((single - scalar(&row0, scale0)).abs() <= 1e-5);
            assert!((pair.0 - scalar(&row0, scale0)).abs() <= 1e-5);
            assert!((pair.1 - scalar(&row1, scale1)).abs() <= 1e-5);
            assert!((triple.0 - scalar(&row0, scale0)).abs() <= 1e-5);
            assert!((triple.1 - scalar(&row1, scale1)).abs() <= 1e-5);
            assert!((triple.2 - scalar(&row2, scale2)).abs() <= 1e-5);
        }
    }

    #[test]
    fn f32_i8_avx512_path_matches_scalar_reference_when_available() {
        #[cfg(all(
            feature = "x86-int8-kernels",
            any(target_arch = "x86_64", target_arch = "x86")
        ))]
        {
            if !arch::x86_avx512_i8_i8_kernel_runtime_available() {
                return;
            }

            let x: Vec<f32> = (0..67)
                .map(|i| ((i as f32 * 0.03125) - 1.0).sin() * 3.0)
                .collect();
            let row0: Vec<i8> = (0..67).map(|i| patterned_i8(i, 37, 11)).collect();
            let row1: Vec<i8> = (0..67).map(|i| patterned_i8(i, 53, 19)).collect();
            let row2: Vec<i8> = (0..67).map(|i| patterned_i8(i, 71, 23)).collect();
            let scale0 = 0.0078125f32;
            let scale1 = 0.01171875f32;
            let scale2 = 0.015625f32;
            let scalar = |row: &[i8], scale: f32| -> f32 {
                x.iter()
                    .zip(row.iter())
                    .map(|(&xv, &rv)| xv * rv as f32)
                    .sum::<f32>()
                    * scale
            };

            let single = unsafe { dot_f32_i8_x86_avx512(&x, &row0, scale0) };
            let pair = unsafe { dot2_f32_i8_x86_avx512(&x, &row0, scale0, &row1, scale1) };
            let triple =
                unsafe { dot3_f32_i8_x86_avx512(&x, &row0, scale0, &row1, scale1, &row2, scale2) };

            assert!((single - scalar(&row0, scale0)).abs() <= 1e-4);
            assert!((pair.0 - scalar(&row0, scale0)).abs() <= 1e-4);
            assert!((pair.1 - scalar(&row1, scale1)).abs() <= 1e-4);
            assert!((triple.0 - scalar(&row0, scale0)).abs() <= 1e-4);
            assert!((triple.1 - scalar(&row1, scale1)).abs() <= 1e-4);
            assert!((triple.2 - scalar(&row2, scale2)).abs() <= 1e-4);
        }
    }

    #[test]
    fn architecture_dispatch_consistency() {
        let x = [0.5f32, -1.0, 2.0, 0.25, -0.75, 1.5, -0.5, 3.0];
        let row0 = [3i8, -2, 5, 1, -4, 6, -1, 2];
        let row1 = [-1i8, 4, -3, 2, 5, -2, 7, -6];
        let row2 = [2i8, 1, -2, 3, -5, 4, 1, -3];
        let single = dot_f32_i8_arch(&x, &row0, 0.125);
        let triple = dot3_f32_i8_arch(&x, &row0, 0.125, &row1, 0.25, &row2, 0.5);

        if active_int8_backend() == Int8KernelBackend::Portable {
            assert!(single.is_none());
            assert!(triple.is_none());
        } else {
            assert!(single.is_some());
            assert!(triple.is_some());
        }
    }

    #[test]
    fn x86_int8_fast_paths_match_scalar_reference() {
        let x = [
            -1.25f32, 0.5, 2.0, -0.75, 1.5, -2.25, 3.0, 0.125, 1.75, -1.0, 0.625, 2.5, -3.5, 4.0,
            -0.875, 1.125, 0.333, -0.666, 1.999,
        ];
        let row0 = [
            -7i8, 3, 12, -5, 9, -11, 15, 1, 8, -4, 6, 10, -13, 14, -2, 5, 2, -3, 7,
        ];
        let row1 = [
            4i8, -9, 6, 2, -8, 7, -5, 3, -1, 10, -6, 11, -12, 13, -4, 8, -2, 1, 5,
        ];
        let row2 = [
            -3i8, 2, -7, 9, -10, 6, 4, -1, 12, -8, 5, -6, 11, -13, 7, -2, 3, -4, 10,
        ];
        let scale0 = 0.125f32;
        let scale1 = 0.0625f32;
        let scale2 = 0.25f32;

        let scalar = |row: &[i8], scale: f32| -> f32 {
            x.iter()
                .zip(row.iter())
                .map(|(&xv, &rv)| xv * rv as f32)
                .sum::<f32>()
                * scale
        };

        if let Some(sum) = dot_f32_i8_arch(&x, &row0, scale0) {
            assert!((sum - scalar(&row0, scale0)).abs() <= 1e-5);
        }
        if let Some((sum0, sum1)) = dot2_f32_i8_arch(&x, &row0, scale0, &row1, scale1) {
            assert!((sum0 - scalar(&row0, scale0)).abs() <= 1e-5);
            assert!((sum1 - scalar(&row1, scale1)).abs() <= 1e-5);
        }
        if let Some((sum0, sum1, sum2)) =
            dot3_f32_i8_arch(&x, &row0, scale0, &row1, scale1, &row2, scale2)
        {
            assert!((sum0 - scalar(&row0, scale0)).abs() <= 1e-5);
            assert!((sum1 - scalar(&row1, scale1)).abs() <= 1e-5);
            assert!((sum2 - scalar(&row2, scale2)).abs() <= 1e-5);
        }

        let x_i8 = [
            3i8, -2, 5, 1, -4, 6, -1, 2, 7, -8, 9, -10, 11, -12, 13, -14, 4, -3, 2,
        ];
        let x_scale = 0.0625f32;
        let scalar_i8 = |row: &[i8], scale: f32| -> f32 {
            x_i8.iter()
                .zip(row.iter())
                .map(|(&xv, &rv)| xv as i32 * rv as i32)
                .sum::<i32>() as f32
                * x_scale
                * scale
        };

        if let Some(sum) = dot_i8_i8_arch(&x_i8, x_scale, &row0, scale0) {
            assert!((sum - scalar_i8(&row0, scale0)).abs() <= 1e-5);
        }
        if let Some((sum0, sum1)) = dot2_i8_i8_arch(&x_i8, x_scale, &row0, scale0, &row1, scale1) {
            assert!((sum0 - scalar_i8(&row0, scale0)).abs() <= 1e-5);
            assert!((sum1 - scalar_i8(&row1, scale1)).abs() <= 1e-5);
        }
        if let Some((sum0, sum1, sum2)) = dot3_i8_i8_arch(
            &x_i8,
            x_scale,
            [
                I8ScaledRow {
                    values: &row0,
                    scale: scale0,
                },
                I8ScaledRow {
                    values: &row1,
                    scale: scale1,
                },
                I8ScaledRow {
                    values: &row2,
                    scale: scale2,
                },
            ],
        ) {
            assert!((sum0 - scalar_i8(&row0, scale0)).abs() <= 1e-5);
            assert!((sum1 - scalar_i8(&row1, scale1)).abs() <= 1e-5);
            assert!((sum2 - scalar_i8(&row2, scale2)).abs() <= 1e-5);
        }
    }

    #[test]
    fn length_mismatch_disables_arch_fast_path() {
        let x = [1.0f32, 2.0, 3.0, 4.0];
        let short = [1i8, 2];

        assert!(dot_f32_i8_arch(&x, &short, 0.5).is_none());
        assert!(dot2_f32_i8_arch(&x, &short, 0.5, &short, 0.25).is_none());
        assert!(dot3_f32_i8_arch(&x, &short, 0.5, &short, 0.25, &short, 0.125).is_none());
        let row = [1i8, 2, 3, 4];
        assert!(dot_i8_i8_arch(&row, 0.5, &short, 0.25).is_none());
        assert!(dot2_i8_i8_arch(&row, 0.5, &short, 0.25, &short, 0.125).is_none());
        assert!(
            dot3_i8_i8_arch(
                &row,
                0.5,
                [
                    I8ScaledRow {
                        values: &short,
                        scale: 0.25
                    },
                    I8ScaledRow {
                        values: &short,
                        scale: 0.125
                    },
                    I8ScaledRow {
                        values: &short,
                        scale: 0.0625
                    },
                ]
            )
            .is_none()
        );
        let mut update = [1i8, 2, 3, 4];
        assert!(!sgd_update_i8_f32_arch(&mut update, 0.5, &x[..2], 0.1));
        assert!(!sgd_update_i8_f32_arch(&mut update, 0.0, &x, 0.1));
        let mut velocity = [0.0f32; 4];
        assert!(!sgd_momentum_update_i8_f32_arch(
            &mut update,
            0.5,
            &mut velocity,
            &x[..2],
            0.1,
            0.9
        ));
        assert!(!sgd_momentum_update_i8_f32_arch(
            &mut update,
            0.0,
            &mut velocity,
            &x,
            0.1,
            0.9
        ));
        let mut exp_avg = [0.0f32; 4];
        let mut exp_avg_sq = [0.0f32; 4];
        assert!(!adam_update_i8_f32_arch(
            &mut update,
            0.5,
            &mut exp_avg,
            &mut exp_avg_sq,
            &x[..2],
            0.1,
            0.9,
            0.999,
            0.1,
            0.001,
            1e-8
        ));
        assert!(!adam_update_i8_f32_arch(
            &mut update,
            0.0,
            &mut exp_avg,
            &mut exp_avg_sq,
            &x,
            0.1,
            0.9,
            0.999,
            0.1,
            0.001,
            1e-8
        ));
    }

    #[test]
    fn i8_sgd_update_fast_path_matches_scalar_reference() {
        let scale = 0.0625f32;
        let lr = 0.03125f32;
        let mut data: Vec<i8> = (0..65).map(|i| patterned_i8(i, 37, 11)).collect();
        let grad: Vec<f32> = (0..data.len())
            .map(|i| ((i as f32 * 0.17).sin() * 3.0) + ((i % 7) as f32 - 3.0) * 0.125)
            .collect();
        let mut expected = data.clone();
        for (w, g) in expected.iter_mut().zip(grad.iter()) {
            let updated = (*w as f32) * scale - lr * *g;
            *w = (updated / scale).round().clamp(-127.0, 127.0) as i8;
        }

        if sgd_update_i8_f32_arch(&mut data, scale, &grad, lr) {
            assert_eq!(data, expected);
        }
    }

    #[test]
    fn i8_adam_update_fast_path_matches_scalar_reference() {
        let scale = 0.0625f32;
        let lr = 0.001f32;
        let beta1 = 0.9f32;
        let beta2 = 0.999f32;
        let bias_correction1 = 0.1f32;
        let bias_correction2 = 0.001f32;
        let eps = 1e-8f32;
        let mut data: Vec<i8> = (0..65).map(|i| patterned_i8(i, 71, 23)).collect();
        let mut exp_avg: Vec<f32> = (0..data.len())
            .map(|i| ((i as f32 * 0.07).sin() * 0.25) - 0.1)
            .collect();
        let mut exp_avg_sq: Vec<f32> = (0..data.len())
            .map(|i| ((i % 11) as f32 + 1.0) * 0.0001)
            .collect();
        let grad: Vec<f32> = (0..data.len())
            .map(|i| ((i as f32 * 0.19).cos() * 0.75) + ((i % 3) as f32 - 1.0) * 0.05)
            .collect();
        let mut expected_data = data.clone();
        let mut expected_m = exp_avg.clone();
        let mut expected_v = exp_avg_sq.clone();
        for (((w, m), v), g) in expected_data
            .iter_mut()
            .zip(expected_m.iter_mut())
            .zip(expected_v.iter_mut())
            .zip(grad.iter())
        {
            *m = beta1 * *m + (1.0 - beta1) * *g;
            *v = beta2 * *v + (1.0 - beta2) * *g * *g;
            let m_hat = *m / bias_correction1;
            let v_hat = *v / bias_correction2;
            let updated = (*w as f32) * scale - lr * (m_hat / (v_hat.sqrt() + eps));
            *w = (updated / scale).round().clamp(-127.0, 127.0) as i8;
        }

        if adam_update_i8_f32_arch(
            &mut data,
            scale,
            &mut exp_avg,
            &mut exp_avg_sq,
            &grad,
            lr,
            beta1,
            beta2,
            bias_correction1,
            bias_correction2,
            eps,
        ) {
            assert_eq!(data, expected_data);
            for (actual, expected) in exp_avg.iter().zip(expected_m.iter()) {
                assert!((actual - expected).abs() <= 1e-6);
            }
            for (actual, expected) in exp_avg_sq.iter().zip(expected_v.iter()) {
                assert!((actual - expected).abs() <= 1e-6);
            }
        }
    }

    #[test]
    fn i8_sgd_momentum_update_fast_path_matches_scalar_reference() {
        let scale = 0.0625f32;
        let lr = 0.03125f32;
        let momentum = 0.875f32;
        let mut data: Vec<i8> = (0..65).map(|i| patterned_i8(i, 53, 19)).collect();
        let mut velocity: Vec<f32> = (0..data.len())
            .map(|i| ((i as f32 * 0.11).cos() * 0.5) - 0.25)
            .collect();
        let grad: Vec<f32> = (0..data.len())
            .map(|i| ((i as f32 * 0.13).sin() * 2.0) + ((i % 5) as f32 - 2.0) * 0.25)
            .collect();
        let mut expected_data = data.clone();
        let mut expected_velocity = velocity.clone();
        for ((w, v), g) in expected_data
            .iter_mut()
            .zip(expected_velocity.iter_mut())
            .zip(grad.iter())
        {
            *v = momentum * *v + *g;
            let updated = (*w as f32) * scale - lr * *v;
            *w = (updated / scale).round().clamp(-127.0, 127.0) as i8;
        }

        if sgd_momentum_update_i8_f32_arch(&mut data, scale, &mut velocity, &grad, lr, momentum) {
            assert_eq!(data, expected_data);
            for (actual, expected) in velocity.iter().zip(expected_velocity.iter()) {
                assert!((actual - expected).abs() <= 1e-6);
            }
        }
    }

    #[test]
    fn i8_update_avx2_fallbacks_match_scalar_reference_when_available() {
        #[cfg(all(
            feature = "x86-int8-kernels",
            any(target_arch = "x86_64", target_arch = "x86")
        ))]
        {
            if !arch::x86_i8_kernel_runtime_available() {
                return;
            }

            let scale = 0.0625f32;
            let lr = 0.03125f32;
            let mut data: Vec<i8> = (0..65).map(|i| patterned_i8(i, 37, 11)).collect();
            let grad: Vec<f32> = (0..data.len())
                .map(|i| ((i as f32 * 0.17).sin() * 3.0) + ((i % 7) as f32 - 3.0) * 0.125)
                .collect();

            let mut expected_sgd = data.clone();
            for (w, g) in expected_sgd.iter_mut().zip(grad.iter()) {
                let updated = (*w as f32) * scale - lr * *g;
                *w = (updated / scale).round().clamp(-127.0, 127.0) as i8;
            }
            unsafe { sgd_update_i8_f32_x86_avx2(&mut data, scale, &grad, lr) };
            assert_eq!(data, expected_sgd);

            let momentum = 0.875f32;
            let mut data: Vec<i8> = (0..65).map(|i| patterned_i8(i, 53, 19)).collect();
            let mut velocity: Vec<f32> = (0..data.len())
                .map(|i| ((i as f32 * 0.11).cos() * 0.5) - 0.25)
                .collect();
            let mut expected_data = data.clone();
            let mut expected_velocity = velocity.clone();
            for ((w, v), g) in expected_data
                .iter_mut()
                .zip(expected_velocity.iter_mut())
                .zip(grad.iter())
            {
                *v = momentum * *v + *g;
                let updated = (*w as f32) * scale - lr * *v;
                *w = (updated / scale).round().clamp(-127.0, 127.0) as i8;
            }
            unsafe {
                sgd_momentum_update_i8_f32_x86_avx2(
                    &mut data,
                    scale,
                    &mut velocity,
                    &grad,
                    lr,
                    momentum,
                )
            };
            assert_eq!(data, expected_data);
            for (actual, expected) in velocity.iter().zip(expected_velocity.iter()) {
                assert!((actual - expected).abs() <= 1e-6);
            }

            let lr = 0.001f32;
            let beta1 = 0.9f32;
            let beta2 = 0.999f32;
            let bias_correction1 = 0.1f32;
            let bias_correction2 = 0.001f32;
            let eps = 1e-8f32;
            let mut data: Vec<i8> = (0..65).map(|i| patterned_i8(i, 71, 23)).collect();
            let mut exp_avg: Vec<f32> = (0..data.len())
                .map(|i| ((i as f32 * 0.07).sin() * 0.25) - 0.1)
                .collect();
            let mut exp_avg_sq: Vec<f32> = (0..data.len())
                .map(|i| ((i % 11) as f32 + 1.0) * 0.0001)
                .collect();
            let mut expected_data = data.clone();
            let mut expected_m = exp_avg.clone();
            let mut expected_v = exp_avg_sq.clone();
            for (((w, m), v), g) in expected_data
                .iter_mut()
                .zip(expected_m.iter_mut())
                .zip(expected_v.iter_mut())
                .zip(grad.iter())
            {
                *m = beta1 * *m + (1.0 - beta1) * *g;
                *v = beta2 * *v + (1.0 - beta2) * *g * *g;
                let m_hat = *m / bias_correction1;
                let v_hat = *v / bias_correction2;
                let updated = (*w as f32) * scale - lr * (m_hat / (v_hat.sqrt() + eps));
                *w = (updated / scale).round().clamp(-127.0, 127.0) as i8;
            }
            unsafe {
                adam_update_i8_f32_x86_avx2(
                    &mut data,
                    scale,
                    &mut exp_avg,
                    &mut exp_avg_sq,
                    &grad,
                    lr,
                    beta1,
                    beta2,
                    bias_correction1,
                    bias_correction2,
                    eps,
                )
            };
            assert_eq!(data, expected_data);
            for (actual, expected) in exp_avg.iter().zip(expected_m.iter()) {
                assert!((actual - expected).abs() <= 1e-6);
            }
            for (actual, expected) in exp_avg_sq.iter().zip(expected_v.iter()) {
                assert!((actual - expected).abs() <= 1e-6);
            }
        }
    }

    #[test]
    #[ignore = "performance sanity test; run with --ignored --nocapture"]
    fn perf_i8_i8_dot2_dot3_arch_paths() {
        let len = 8192usize;
        let iters = 4096usize;
        let make_i8 = |seed: usize| -> Vec<i8> {
            (0..len)
                .map(|i| {
                    let v = ((i.wrapping_mul(1103515245usize) ^ seed).rotate_left(7) & 255) as i32
                        - 128;
                    v.clamp(-127, 127) as i8
                })
                .collect()
        };
        let x = make_i8(0x1234);
        let row0 = make_i8(0x2345);
        let row1 = make_i8(0x3456);
        let row2 = make_i8(0x4567);
        let x_f32 = (0..len)
            .map(|i| ((i * 97 % 257) as f32 - 128.0) / 64.0)
            .collect::<Vec<_>>();
        let x_scale = 0.03125f32;
        let scale0 = 0.015625f32;
        let scale1 = 0.0078125f32;
        let scale2 = 0.00390625f32;
        let row_broadcast_last_dim = 256usize;
        let row_broadcast_vector = &row1[..row_broadcast_last_dim];
        let grad = (0..len)
            .map(|i| ((i * 41 % 233) as f32 - 116.0) / 233.0)
            .collect::<Vec<_>>();
        let mut grad_out = vec![0.0f32; len];
        let mut fwd_out = vec![0.0f32; len];
        let row_broadcast_args = I8RowBroadcast {
            lhs: I8ScaledRow {
                values: &x,
                scale: x_scale,
            },
            rhs: I8ScaledRow {
                values: row_broadcast_vector,
                scale: scale1,
            },
            last_dim: row_broadcast_last_dim,
            vector_on_rhs: true,
        };

        if dot2_i8_i8_arch(&x, x_scale, &row0, scale0, &row1, scale1).is_none() {
            eprintln!(
                "perf_i8_i8_dot2_dot3 skipped backend={}",
                active_i8_i8_backend_name()
            );
            return;
        }

        let start = std::time::Instant::now();
        let mut sink = 0.0f32;
        for _ in 0..iters {
            let (a, b) =
                dot2_i8_i8_arch(&x, x_scale, &row0, scale0, &row1, scale1).expect("dot2 arch");
            sink += a + b;
        }
        let dot2_elapsed = start.elapsed();

        let start = std::time::Instant::now();
        for _ in 0..iters {
            let (a, b, c) = dot3_i8_i8_arch(
                &x,
                x_scale,
                [
                    I8ScaledRow {
                        values: &row0,
                        scale: scale0,
                    },
                    I8ScaledRow {
                        values: &row1,
                        scale: scale1,
                    },
                    I8ScaledRow {
                        values: &row2,
                        scale: scale2,
                    },
                ],
            )
            .expect("dot3 arch");
            sink += a + b + c;
        }
        let dot3_elapsed = start.elapsed();

        let start = std::time::Instant::now();
        for _ in 0..iters {
            let (a, b) =
                dot2_f32_i8_arch(&x_f32, &row0, scale0, &row1, scale1).expect("f32_i8 dot2 arch");
            sink += a + b;
        }
        let f32_i8_dot2_elapsed = start.elapsed();

        let start = std::time::Instant::now();
        for _ in 0..iters {
            let (a, b, c) = dot3_f32_i8_arch(&x_f32, &row0, scale0, &row1, scale1, &row2, scale2)
                .expect("f32_i8 dot3 arch");
            sink += a + b + c;
        }
        let f32_i8_dot3_elapsed = start.elapsed();

        let start = std::time::Instant::now();
        for _ in 0..iters {
            sink += sum_i8_arch(&x, x_scale).expect("sum arch");
        }
        let sum_elapsed = start.elapsed();

        let mut out = vec![0i8; len];
        let start = std::time::Instant::now();
        for _ in 0..iters {
            sink += add_i8_i8_arch(&x, x_scale, &row0, scale0, &mut out).expect("add arch");
        }
        let add_elapsed = start.elapsed();

        let start = std::time::Instant::now();
        for _ in 0..iters {
            sink += mul_i8_i8_arch(&x, x_scale, &row0, scale0, &mut out).expect("mul arch");
        }
        let mul_elapsed = start.elapsed();

        let start = std::time::Instant::now();
        for _ in 0..iters {
            sink += mul_i8_i8_row_broadcast_arch(row_broadcast_args, &mut out)
                .expect("row-broadcast mul arch");
        }
        let row_broadcast_mul_elapsed = start.elapsed();

        let start = std::time::Instant::now();
        for _ in 0..iters {
            assert!(mul_f32_i8_to_f32_arch(&grad, &x, x_scale, &mut grad_out));
            sink += grad_out[0];
        }
        let mul_grad_elapsed = start.elapsed();

        let start = std::time::Instant::now();
        for _ in 0..iters {
            assert!(mul_i8_scalar_to_f32_arch(
                &x,
                x_scale,
                -0.375,
                &mut grad_out
            ));
            sink += grad_out[0];
        }
        let scalar_mul_elapsed = start.elapsed();

        let start = std::time::Instant::now();
        for _ in 0..iters {
            assert!(mul_i8_i8_to_f32_arch(
                &x,
                x_scale,
                &row0,
                scale0,
                &mut fwd_out
            ));
            sink += fwd_out[0];
        }
        let fwd_mul_elapsed = start.elapsed();

        let start = std::time::Instant::now();
        for _ in 0..iters {
            assert!(mul_i8_i8_row_broadcast_to_f32_arch(
                row_broadcast_args,
                &mut fwd_out
            ));
            sink += fwd_out[0];
        }
        let row_broadcast_fwd_mul_elapsed = start.elapsed();

        eprintln!(
            "perf_i8_i8_dot2_dot3 backend={} len={len} iters={iters} dot2={:.3}us dot3={:.3}us f32_i8_dot2={:.3}us f32_i8_dot3={:.3}us sum={:.3}us add={:.3}us mul={:.3}us row_broadcast_mul={:.3}us mul_grad={:.3}us scalar_mul_f32={:.3}us fwd_mul_f32={:.3}us row_broadcast_fwd_mul_f32={:.3}us sink={sink}",
            active_i8_i8_backend_name(),
            dot2_elapsed.as_secs_f64() * 1e6 / iters as f64,
            dot3_elapsed.as_secs_f64() * 1e6 / iters as f64,
            f32_i8_dot2_elapsed.as_secs_f64() * 1e6 / iters as f64,
            f32_i8_dot3_elapsed.as_secs_f64() * 1e6 / iters as f64,
            sum_elapsed.as_secs_f64() * 1e6 / iters as f64,
            add_elapsed.as_secs_f64() * 1e6 / iters as f64,
            mul_elapsed.as_secs_f64() * 1e6 / iters as f64,
            row_broadcast_mul_elapsed.as_secs_f64() * 1e6 / iters as f64,
            mul_grad_elapsed.as_secs_f64() * 1e6 / iters as f64,
            scalar_mul_elapsed.as_secs_f64() * 1e6 / iters as f64,
            fwd_mul_elapsed.as_secs_f64() * 1e6 / iters as f64,
            row_broadcast_fwd_mul_elapsed.as_secs_f64() * 1e6 / iters as f64,
        );
    }
}
