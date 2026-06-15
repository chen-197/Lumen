use crate::autograd::{
    StoragePreference, Tensor, TensorData, TensorStorageOwned, TensorStorageView,
    assert_native_device_support, assert_same_device, is_no_grad, is_strict_device_execution,
};
use crate::ops::cuda;
use crate::ops::fp_kernels::{
    dot_bf16_bf16_arch, dot_f16_f16_arch, dot_f32_arch, dot_f32_bf16_arch, dot_f32_f16_arch,
    dot2_bf16_bf16_arch, dot2_f16_f16_arch, dot2_f32_arch, dot2_f32_bf16_arch, dot2_f32_f16_arch,
    dot3_bf16_bf16_arch, dot3_f16_f16_arch, dot3_f32_arch, dot3_f32_bf16_arch, dot3_f32_f16_arch,
};
use crate::ops::int8_kernels::{
    I8ScaledRow, dot_f32_i8_arch, dot_i8_i8_arch, dot2_f32_i8_arch, dot2_i8_i8_arch,
    dot3_f32_i8_arch, dot3_i8_i8_arch,
};
use crate::precision::DType;
use half::{bf16, f16, slice::HalfFloatSliceExt};
use ndarray::linalg::general_mat_mul;
use ndarray::{Array2, Array4, ArrayD, Ix2, Ix4, IxDyn, Zip};
use rayon::prelude::*;
use std::cell::RefCell;
use std::rc::Rc;

#[cfg(all(feature = "arm64-int8-kernels", target_arch = "aarch64"))]
const MATVEC_BLOCK_ROWS: usize = 32;
#[cfg(not(all(feature = "arm64-int8-kernels", target_arch = "aarch64")))]
const MATVEC_BLOCK_ROWS: usize = 16;

#[cfg(all(feature = "arm64-int8-kernels", target_arch = "aarch64"))]
const SILU_I8_BLOCK_ROWS: usize = 64;
#[cfg(not(all(feature = "arm64-int8-kernels", target_arch = "aarch64")))]
const SILU_I8_BLOCK_ROWS: usize = 32;

const ARGMAX_BLOCK_ROWS: usize = 32;

#[cfg(all(feature = "arm64-int8-kernels", target_arch = "aarch64"))]
const MATVEC_I8_PAR_CHUNK_ROWS: usize = 128;
#[cfg(not(all(feature = "arm64-int8-kernels", target_arch = "aarch64")))]
const MATVEC_I8_PAR_CHUNK_ROWS: usize = 64;

#[cfg(all(feature = "arm64-int8-kernels", target_arch = "aarch64"))]
const QKV_I8_PAR_CHUNK_ROWS: usize = 64;
#[cfg(not(all(feature = "arm64-int8-kernels", target_arch = "aarch64")))]
const QKV_I8_PAR_CHUNK_ROWS: usize = 64;

#[cfg(all(feature = "arm64-fp-kernels", target_arch = "aarch64"))]
const MIXED_ROW_PAR_CHUNK_ROWS: usize = 32;
#[cfg(not(all(feature = "arm64-fp-kernels", target_arch = "aarch64")))]
const MIXED_ROW_PAR_CHUNK_ROWS: usize = 16;

const MATVEC_BLOCK_THRESHOLD: usize = 16384;
#[cfg(all(feature = "arm64-int8-kernels", target_arch = "aarch64"))]
const MATVEC_PAR_THRESHOLD: usize = 1024;
#[cfg(not(all(feature = "arm64-int8-kernels", target_arch = "aarch64")))]
const MATVEC_PAR_THRESHOLD: usize = 256;

type CudaMatmulBackwardHostDevice = (
    (ndarray::ArrayD<f32>, cuda::CudaBuffer),
    (ndarray::ArrayD<f32>, cuda::CudaBuffer),
);

#[derive(Clone, Copy)]
struct BatchMatmulDims {
    b: usize,
    h: usize,
    m: usize,
    k: usize,
    n: usize,
}

impl BatchMatmulDims {
    #[inline]
    fn batch_count(self) -> Option<usize> {
        self.b.checked_mul(self.h)
    }

    #[inline]
    fn out_shape(self) -> [usize; 4] {
        [self.b, self.h, self.m, self.n]
    }
}

#[derive(Clone, Copy)]
struct I8ScaledSlice<'a> {
    values: &'a [i8],
    scale: f32,
}

#[derive(Clone, Copy)]
struct I8QkvBlock<'a> {
    q: I8ScaledSlice<'a>,
    k: I8ScaledSlice<'a>,
    v: I8ScaledSlice<'a>,
}

#[derive(Clone, Copy)]
struct I8DualBlock<'a> {
    left: I8ScaledSlice<'a>,
    right: I8ScaledSlice<'a>,
}

struct DualOutMut<'a> {
    left: &'a mut [f32],
    right: &'a mut [f32],
}

struct QkvOutMut<'a> {
    q: &'a mut [f32],
    k: &'a mut [f32],
    v: &'a mut [f32],
}

#[inline]
fn should_use_mixed_matvec_block_kernel(n_rows: usize) -> bool {
    #[cfg(all(feature = "arm64-fp-kernels", target_arch = "aarch64"))]
    {
        let _ = n_rows;
        false
    }

    #[cfg(not(all(feature = "arm64-fp-kernels", target_arch = "aarch64")))]
    {
        let _ = n_rows;
        false
    }
}

fn should_use_mixed_dual_block_kernel(n_rows: usize) -> bool {
    #[cfg(all(feature = "arm64-fp-kernels", target_arch = "aarch64"))]
    {
        let _ = n_rows;
        false
    }

    #[cfg(not(all(feature = "arm64-fp-kernels", target_arch = "aarch64")))]
    {
        let _ = n_rows;
        false
    }
}

#[inline]
fn should_use_argmax_block_kernel(n_rows: usize) -> bool {
    #[cfg(all(feature = "arm64-fp-kernels", target_arch = "aarch64"))]
    {
        let _ = n_rows;
        false
    }

    #[cfg(not(all(feature = "arm64-fp-kernels", target_arch = "aarch64")))]
    {
        n_rows >= MATVEC_BLOCK_THRESHOLD
    }
}

thread_local! {
    static F16_TO_F32_SCRATCH: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    static BF16_TO_F32_SCRATCH: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    static I8_TO_F32_SCRATCH: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

pub trait DotElem: Copy + Send + Sync {
    fn to_f32(self) -> f32;
}

impl DotElem for f32 {
    #[inline]
    fn to_f32(self) -> f32 {
        self
    }
}

impl DotElem for bf16 {
    #[inline]
    fn to_f32(self) -> f32 {
        self.to_f32()
    }
}

impl DotElem for f16 {
    #[inline]
    fn to_f32(self) -> f32 {
        self.to_f32()
    }
}

#[derive(Clone, Copy)]
pub enum SliceRef<'a> {
    F32(&'a [f32]),
    F16(&'a [f16]),
    BF16(&'a [bf16]),
    I8(&'a [i8], f32),
}

#[inline]
fn should_use_i8_block_kernel(n_rows: usize, k_dim: usize) -> bool {
    #[cfg(all(feature = "arm64-int8-kernels", target_arch = "aarch64"))]
    {
        let _ = k_dim;
        n_rows >= MATVEC_BLOCK_THRESHOLD
    }

    #[cfg(not(all(feature = "arm64-int8-kernels", target_arch = "aarch64")))]
    {
        let _ = (n_rows, k_dim);
        false
    }
}

#[inline]
fn should_use_i8_silu_block_kernel(n_rows: usize, k_dim: usize) -> bool {
    #[cfg(all(feature = "arm64-int8-kernels", target_arch = "aarch64"))]
    {
        let _ = (n_rows, k_dim);
        false
    }

    #[cfg(not(all(feature = "arm64-int8-kernels", target_arch = "aarch64")))]
    {
        let _ = (n_rows, k_dim);
        false
    }
}

#[inline]
fn should_use_i8_matmul_block_kernel(n_rows: usize, k_dim: usize) -> bool {
    #[cfg(all(feature = "arm64-int8-kernels", target_arch = "aarch64"))]
    {
        let _ = (n_rows, k_dim);
        false
    }

    #[cfg(not(all(feature = "arm64-int8-kernels", target_arch = "aarch64")))]
    {
        let _ = (n_rows, k_dim);
        false
    }
}

#[inline]
fn should_use_i8_qkv_row4_kernel() -> bool {
    #[cfg(all(feature = "arm64-int8-kernels", target_arch = "aarch64"))]
    {
        false
    }

    #[cfg(not(all(feature = "arm64-int8-kernels", target_arch = "aarch64")))]
    {
        false
    }
}

#[inline]
pub(crate) fn with_f16_input_as_f32<R>(x: &[f16], f: impl FnOnce(&[f32]) -> R) -> R {
    F16_TO_F32_SCRATCH.with(|scratch| {
        if let Ok(mut scratch) = scratch.try_borrow_mut() {
            if scratch.len() < x.len() {
                scratch.resize(x.len(), 0.0);
            }
            x.convert_to_f32_slice(&mut scratch[..x.len()]);
            return f(&scratch[..x.len()]);
        }

        let mut fallback = vec![0.0f32; x.len()];
        x.convert_to_f32_slice(&mut fallback);
        f(&fallback)
    })
}

#[inline]
pub(crate) fn with_bf16_input_as_f32<R>(x: &[bf16], f: impl FnOnce(&[f32]) -> R) -> R {
    BF16_TO_F32_SCRATCH.with(|scratch| {
        if let Ok(mut scratch) = scratch.try_borrow_mut() {
            if scratch.len() < x.len() {
                scratch.resize(x.len(), 0.0);
            }
            x.convert_to_f32_slice(&mut scratch[..x.len()]);
            return f(&scratch[..x.len()]);
        }

        let mut fallback = vec![0.0f32; x.len()];
        x.convert_to_f32_slice(&mut fallback);
        f(&fallback)
    })
}

#[inline]
pub(crate) fn with_i8_input_as_f32<R>(x: &[i8], scale: f32, f: impl FnOnce(&[f32]) -> R) -> R {
    I8_TO_F32_SCRATCH.with(|scratch| {
        if let Ok(mut scratch) = scratch.try_borrow_mut() {
            if scratch.len() < x.len() {
                scratch.resize(x.len(), 0.0);
            }
            for (dst, &src) in scratch[..x.len()].iter_mut().zip(x.iter()) {
                *dst = src as f32 * scale;
            }
            return f(&scratch[..x.len()]);
        }

        let mut fallback = vec![0.0f32; x.len()];
        for (dst, &src) in fallback.iter_mut().zip(x.iter()) {
            *dst = src as f32 * scale;
        }
        f(&fallback)
    })
}

#[inline]
pub(crate) fn dot_unrolled(x: &[f32], row: &[f32]) -> f32 {
    if let Some(sum) = dot_f32_arch(x, row) {
        return sum;
    }

    let mut s0 = 0.0f32;
    let mut s1 = 0.0f32;
    let mut s2 = 0.0f32;
    let mut s3 = 0.0f32;
    let mut kk = 0usize;
    let k_dim = x.len();

    while kk + 8 <= k_dim {
        s0 += row[kk] * x[kk] + row[kk + 4] * x[kk + 4];
        s1 += row[kk + 1] * x[kk + 1] + row[kk + 5] * x[kk + 5];
        s2 += row[kk + 2] * x[kk + 2] + row[kk + 6] * x[kk + 6];
        s3 += row[kk + 3] * x[kk + 3] + row[kk + 7] * x[kk + 7];
        kk += 8;
    }

    while kk + 4 <= k_dim {
        s0 += row[kk] * x[kk];
        s1 += row[kk + 1] * x[kk + 1];
        s2 += row[kk + 2] * x[kk + 2];
        s3 += row[kk + 3] * x[kk + 3];
        kk += 4;
    }

    let mut sum = s0 + s1 + s2 + s3;
    while kk < k_dim {
        sum += row[kk] * x[kk];
        kk += 1;
    }
    sum
}

#[inline]
pub(crate) fn dot_unrolled_f32_bf16(x: &[f32], row: &[bf16]) -> f32 {
    if let Some(sum) = dot_f32_bf16_arch(x, row) {
        return sum;
    }

    let mut s0 = 0.0f32;
    let mut s1 = 0.0f32;
    let mut s2 = 0.0f32;
    let mut s3 = 0.0f32;
    let mut kk = 0usize;
    let k_dim = x.len();

    while kk + 8 <= k_dim {
        s0 += row[kk].to_f32() * x[kk] + row[kk + 4].to_f32() * x[kk + 4];
        s1 += row[kk + 1].to_f32() * x[kk + 1] + row[kk + 5].to_f32() * x[kk + 5];
        s2 += row[kk + 2].to_f32() * x[kk + 2] + row[kk + 6].to_f32() * x[kk + 6];
        s3 += row[kk + 3].to_f32() * x[kk + 3] + row[kk + 7].to_f32() * x[kk + 7];
        kk += 8;
    }

    while kk + 4 <= k_dim {
        s0 += row[kk].to_f32() * x[kk];
        s1 += row[kk + 1].to_f32() * x[kk + 1];
        s2 += row[kk + 2].to_f32() * x[kk + 2];
        s3 += row[kk + 3].to_f32() * x[kk + 3];
        kk += 4;
    }

    let mut sum = s0 + s1 + s2 + s3;
    while kk < k_dim {
        sum += row[kk].to_f32() * x[kk];
        kk += 1;
    }
    sum
}

#[inline]
pub(crate) fn dot_unrolled_f32_f16(x: &[f32], row: &[f16]) -> f32 {
    if let Some(sum) = dot_f32_f16_arch(x, row) {
        return sum;
    }

    let mut s0 = 0.0f32;
    let mut s1 = 0.0f32;
    let mut s2 = 0.0f32;
    let mut s3 = 0.0f32;
    let mut kk = 0usize;
    let k_dim = x.len();

    while kk + 8 <= k_dim {
        s0 += row[kk].to_f32() * x[kk] + row[kk + 4].to_f32() * x[kk + 4];
        s1 += row[kk + 1].to_f32() * x[kk + 1] + row[kk + 5].to_f32() * x[kk + 5];
        s2 += row[kk + 2].to_f32() * x[kk + 2] + row[kk + 6].to_f32() * x[kk + 6];
        s3 += row[kk + 3].to_f32() * x[kk + 3] + row[kk + 7].to_f32() * x[kk + 7];
        kk += 8;
    }

    while kk + 4 <= k_dim {
        s0 += row[kk].to_f32() * x[kk];
        s1 += row[kk + 1].to_f32() * x[kk + 1];
        s2 += row[kk + 2].to_f32() * x[kk + 2];
        s3 += row[kk + 3].to_f32() * x[kk + 3];
        kk += 4;
    }

    let mut sum = s0 + s1 + s2 + s3;
    while kk < k_dim {
        sum += row[kk].to_f32() * x[kk];
        kk += 1;
    }
    sum
}

#[inline]
fn dot_unrolled_f32_i8_portable(x: &[f32], row: &[i8], scale: f32) -> f32 {
    let mut s0 = 0.0f32;
    let mut s1 = 0.0f32;
    let mut s2 = 0.0f32;
    let mut s3 = 0.0f32;
    let mut s4 = 0.0f32;
    let mut s5 = 0.0f32;
    let mut s6 = 0.0f32;
    let mut s7 = 0.0f32;
    let mut kk = 0usize;
    let k_dim = x.len();

    while kk + 16 <= k_dim {
        s0 += row[kk] as f32 * x[kk] + row[kk + 8] as f32 * x[kk + 8];
        s1 += row[kk + 1] as f32 * x[kk + 1] + row[kk + 9] as f32 * x[kk + 9];
        s2 += row[kk + 2] as f32 * x[kk + 2] + row[kk + 10] as f32 * x[kk + 10];
        s3 += row[kk + 3] as f32 * x[kk + 3] + row[kk + 11] as f32 * x[kk + 11];
        s4 += row[kk + 4] as f32 * x[kk + 4] + row[kk + 12] as f32 * x[kk + 12];
        s5 += row[kk + 5] as f32 * x[kk + 5] + row[kk + 13] as f32 * x[kk + 13];
        s6 += row[kk + 6] as f32 * x[kk + 6] + row[kk + 14] as f32 * x[kk + 14];
        s7 += row[kk + 7] as f32 * x[kk + 7] + row[kk + 15] as f32 * x[kk + 15];
        kk += 16;
    }

    while kk + 8 <= k_dim {
        s0 += row[kk] as f32 * x[kk] + row[kk + 4] as f32 * x[kk + 4];
        s1 += row[kk + 1] as f32 * x[kk + 1] + row[kk + 5] as f32 * x[kk + 5];
        s2 += row[kk + 2] as f32 * x[kk + 2] + row[kk + 6] as f32 * x[kk + 6];
        s3 += row[kk + 3] as f32 * x[kk + 3] + row[kk + 7] as f32 * x[kk + 7];
        kk += 8;
    }

    while kk + 4 <= k_dim {
        s0 += row[kk] as f32 * x[kk];
        s1 += row[kk + 1] as f32 * x[kk + 1];
        s2 += row[kk + 2] as f32 * x[kk + 2];
        s3 += row[kk + 3] as f32 * x[kk + 3];
        kk += 4;
    }

    let mut sum = s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7;
    while kk < k_dim {
        sum += row[kk] as f32 * x[kk];
        kk += 1;
    }
    sum * scale
}

#[inline]
fn dot_unrolled_i8_i8(x: &[i8], x_scale: f32, row: &[i8], w_scale: f32) -> f32 {
    if let Some(sum) = dot_i8_i8_arch(x, x_scale, row, w_scale) {
        return sum;
    }

    let mut s0 = 0i32;
    let mut s1 = 0i32;
    let mut s2 = 0i32;
    let mut s3 = 0i32;
    let mut s4 = 0i32;
    let mut s5 = 0i32;
    let mut s6 = 0i32;
    let mut s7 = 0i32;
    let mut kk = 0usize;
    let k_dim = x.len();

    while kk + 16 <= k_dim {
        s0 += row[kk] as i32 * x[kk] as i32 + row[kk + 8] as i32 * x[kk + 8] as i32;
        s1 += row[kk + 1] as i32 * x[kk + 1] as i32 + row[kk + 9] as i32 * x[kk + 9] as i32;
        s2 += row[kk + 2] as i32 * x[kk + 2] as i32 + row[kk + 10] as i32 * x[kk + 10] as i32;
        s3 += row[kk + 3] as i32 * x[kk + 3] as i32 + row[kk + 11] as i32 * x[kk + 11] as i32;
        s4 += row[kk + 4] as i32 * x[kk + 4] as i32 + row[kk + 12] as i32 * x[kk + 12] as i32;
        s5 += row[kk + 5] as i32 * x[kk + 5] as i32 + row[kk + 13] as i32 * x[kk + 13] as i32;
        s6 += row[kk + 6] as i32 * x[kk + 6] as i32 + row[kk + 14] as i32 * x[kk + 14] as i32;
        s7 += row[kk + 7] as i32 * x[kk + 7] as i32 + row[kk + 15] as i32 * x[kk + 15] as i32;
        kk += 16;
    }

    while kk + 8 <= k_dim {
        s0 += row[kk] as i32 * x[kk] as i32 + row[kk + 4] as i32 * x[kk + 4] as i32;
        s1 += row[kk + 1] as i32 * x[kk + 1] as i32 + row[kk + 5] as i32 * x[kk + 5] as i32;
        s2 += row[kk + 2] as i32 * x[kk + 2] as i32 + row[kk + 6] as i32 * x[kk + 6] as i32;
        s3 += row[kk + 3] as i32 * x[kk + 3] as i32 + row[kk + 7] as i32 * x[kk + 7] as i32;
        kk += 8;
    }

    while kk + 4 <= k_dim {
        s0 += row[kk] as i32 * x[kk] as i32;
        s1 += row[kk + 1] as i32 * x[kk + 1] as i32;
        s2 += row[kk + 2] as i32 * x[kk + 2] as i32;
        s3 += row[kk + 3] as i32 * x[kk + 3] as i32;
        kk += 4;
    }

    let mut sum = s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7;
    while kk < k_dim {
        sum += row[kk] as i32 * x[kk] as i32;
        kk += 1;
    }
    (sum as f32) * x_scale * w_scale
}

#[inline]
fn dot_unrolled_bf16_bf16_scalar(x: &[bf16], row: &[bf16]) -> f32 {
    let mut s0 = 0.0f32;
    let mut s1 = 0.0f32;
    let mut s2 = 0.0f32;
    let mut s3 = 0.0f32;
    let mut kk = 0usize;
    let k_dim = x.len();

    while kk + 4 <= k_dim {
        s0 += row[kk].to_f32() * x[kk].to_f32();
        s1 += row[kk + 1].to_f32() * x[kk + 1].to_f32();
        s2 += row[kk + 2].to_f32() * x[kk + 2].to_f32();
        s3 += row[kk + 3].to_f32() * x[kk + 3].to_f32();
        kk += 4;
    }

    let mut sum = s0 + s1 + s2 + s3;
    while kk < k_dim {
        sum += row[kk].to_f32() * x[kk].to_f32();
        kk += 1;
    }
    sum
}

#[inline]
fn dot_unrolled_f16_f16_scalar(x: &[f16], row: &[f16]) -> f32 {
    let mut s0 = 0.0f32;
    let mut s1 = 0.0f32;
    let mut s2 = 0.0f32;
    let mut s3 = 0.0f32;
    let mut kk = 0usize;
    let k_dim = x.len();

    while kk + 4 <= k_dim {
        s0 += row[kk].to_f32() * x[kk].to_f32();
        s1 += row[kk + 1].to_f32() * x[kk + 1].to_f32();
        s2 += row[kk + 2].to_f32() * x[kk + 2].to_f32();
        s3 += row[kk + 3].to_f32() * x[kk + 3].to_f32();
        kk += 4;
    }

    let mut sum = s0 + s1 + s2 + s3;
    while kk < k_dim {
        sum += row[kk].to_f32() * x[kk].to_f32();
        kk += 1;
    }
    sum
}

#[inline]
fn dot2_unrolled_i8_i8(
    x: &[i8],
    x_scale: f32,
    row0: &[i8],
    scale0: f32,
    row1: &[i8],
    scale1: f32,
) -> (f32, f32) {
    if let Some(sum) = dot2_i8_i8_arch(x, x_scale, row0, scale0, row1, scale1) {
        return sum;
    }

    let mut a0 = 0i32;
    let mut a1 = 0i32;
    let mut b0 = 0i32;
    let mut b1 = 0i32;
    let mut c0 = 0i32;
    let mut c1 = 0i32;
    let mut d0 = 0i32;
    let mut d1 = 0i32;
    let mut e0 = 0i32;
    let mut e1 = 0i32;
    let mut f0 = 0i32;
    let mut f1 = 0i32;
    let mut g0 = 0i32;
    let mut g1 = 0i32;
    let mut h0 = 0i32;
    let mut h1 = 0i32;
    let mut kk = 0usize;
    let k_dim = x.len();

    while kk + 16 <= k_dim {
        let x0 = x[kk] as i32;
        let x1 = x[kk + 1] as i32;
        let x2 = x[kk + 2] as i32;
        let x3 = x[kk + 3] as i32;
        let x4 = x[kk + 4] as i32;
        let x5 = x[kk + 5] as i32;
        let x6 = x[kk + 6] as i32;
        let x7 = x[kk + 7] as i32;
        let x8 = x[kk + 8] as i32;
        let x9 = x[kk + 9] as i32;
        let x10 = x[kk + 10] as i32;
        let x11 = x[kk + 11] as i32;
        let x12 = x[kk + 12] as i32;
        let x13 = x[kk + 13] as i32;
        let x14 = x[kk + 14] as i32;
        let x15 = x[kk + 15] as i32;

        a0 += row0[kk] as i32 * x0;
        a1 += row1[kk] as i32 * x0;
        b0 += row0[kk + 1] as i32 * x1;
        b1 += row1[kk + 1] as i32 * x1;
        c0 += row0[kk + 2] as i32 * x2;
        c1 += row1[kk + 2] as i32 * x2;
        d0 += row0[kk + 3] as i32 * x3;
        d1 += row1[kk + 3] as i32 * x3;
        e0 += row0[kk + 4] as i32 * x4;
        e1 += row1[kk + 4] as i32 * x4;
        f0 += row0[kk + 5] as i32 * x5;
        f1 += row1[kk + 5] as i32 * x5;
        g0 += row0[kk + 6] as i32 * x6;
        g1 += row1[kk + 6] as i32 * x6;
        h0 += row0[kk + 7] as i32 * x7;
        h1 += row1[kk + 7] as i32 * x7;
        a0 += row0[kk + 8] as i32 * x8;
        a1 += row1[kk + 8] as i32 * x8;
        b0 += row0[kk + 9] as i32 * x9;
        b1 += row1[kk + 9] as i32 * x9;
        c0 += row0[kk + 10] as i32 * x10;
        c1 += row1[kk + 10] as i32 * x10;
        d0 += row0[kk + 11] as i32 * x11;
        d1 += row1[kk + 11] as i32 * x11;
        e0 += row0[kk + 12] as i32 * x12;
        e1 += row1[kk + 12] as i32 * x12;
        f0 += row0[kk + 13] as i32 * x13;
        f1 += row1[kk + 13] as i32 * x13;
        g0 += row0[kk + 14] as i32 * x14;
        g1 += row1[kk + 14] as i32 * x14;
        h0 += row0[kk + 15] as i32 * x15;
        h1 += row1[kk + 15] as i32 * x15;
        kk += 16;
    }

    while kk + 8 <= k_dim {
        let x0 = x[kk] as i32;
        let x1 = x[kk + 1] as i32;
        let x2 = x[kk + 2] as i32;
        let x3 = x[kk + 3] as i32;
        let x4 = x[kk + 4] as i32;
        let x5 = x[kk + 5] as i32;
        let x6 = x[kk + 6] as i32;
        let x7 = x[kk + 7] as i32;

        a0 += row0[kk] as i32 * x0 + row0[kk + 4] as i32 * x4;
        a1 += row1[kk] as i32 * x0 + row1[kk + 4] as i32 * x4;
        b0 += row0[kk + 1] as i32 * x1 + row0[kk + 5] as i32 * x5;
        b1 += row1[kk + 1] as i32 * x1 + row1[kk + 5] as i32 * x5;
        c0 += row0[kk + 2] as i32 * x2 + row0[kk + 6] as i32 * x6;
        c1 += row1[kk + 2] as i32 * x2 + row1[kk + 6] as i32 * x6;
        d0 += row0[kk + 3] as i32 * x3 + row0[kk + 7] as i32 * x7;
        d1 += row1[kk + 3] as i32 * x3 + row1[kk + 7] as i32 * x7;
        kk += 8;
    }

    while kk + 4 <= k_dim {
        let x0 = x[kk] as i32;
        let x1 = x[kk + 1] as i32;
        let x2 = x[kk + 2] as i32;
        let x3 = x[kk + 3] as i32;
        a0 += row0[kk] as i32 * x0;
        a1 += row1[kk] as i32 * x0;
        b0 += row0[kk + 1] as i32 * x1;
        b1 += row1[kk + 1] as i32 * x1;
        c0 += row0[kk + 2] as i32 * x2;
        c1 += row1[kk + 2] as i32 * x2;
        d0 += row0[kk + 3] as i32 * x3;
        d1 += row1[kk + 3] as i32 * x3;
        kk += 4;
    }

    let mut sum0 = a0 + b0 + c0 + d0 + e0 + f0 + g0 + h0;
    let mut sum1 = a1 + b1 + c1 + d1 + e1 + f1 + g1 + h1;
    while kk < k_dim {
        let xv = x[kk] as i32;
        sum0 += row0[kk] as i32 * xv;
        sum1 += row1[kk] as i32 * xv;
        kk += 1;
    }
    (
        (sum0 as f32) * x_scale * scale0,
        (sum1 as f32) * x_scale * scale1,
    )
}

#[inline]
fn dot3_unrolled_i8_i8(x: &[i8], x_scale: f32, rows: [I8ScaledSlice<'_>; 3]) -> (f32, f32, f32) {
    let [row0, row1, row2] = rows;
    if let Some(sum) = dot3_i8_i8_arch(
        x,
        x_scale,
        [
            I8ScaledRow {
                values: row0.values,
                scale: row0.scale,
            },
            I8ScaledRow {
                values: row1.values,
                scale: row1.scale,
            },
            I8ScaledRow {
                values: row2.values,
                scale: row2.scale,
            },
        ],
    ) {
        return sum;
    }
    let scale0 = row0.scale;
    let scale1 = row1.scale;
    let scale2 = row2.scale;
    let row0 = row0.values;
    let row1 = row1.values;
    let row2 = row2.values;

    let mut a0 = 0i32;
    let mut a1 = 0i32;
    let mut a2 = 0i32;
    let mut b0 = 0i32;
    let mut b1 = 0i32;
    let mut b2 = 0i32;
    let mut c0 = 0i32;
    let mut c1 = 0i32;
    let mut c2 = 0i32;
    let mut d0 = 0i32;
    let mut d1 = 0i32;
    let mut d2 = 0i32;
    let mut e0 = 0i32;
    let mut e1 = 0i32;
    let mut e2 = 0i32;
    let mut f0 = 0i32;
    let mut f1 = 0i32;
    let mut f2 = 0i32;
    let mut g0 = 0i32;
    let mut g1 = 0i32;
    let mut g2 = 0i32;
    let mut h0 = 0i32;
    let mut h1 = 0i32;
    let mut h2 = 0i32;
    let mut kk = 0usize;
    let k_dim = x.len();

    while kk + 16 <= k_dim {
        let x0 = x[kk] as i32;
        let x1 = x[kk + 1] as i32;
        let x2 = x[kk + 2] as i32;
        let x3 = x[kk + 3] as i32;
        let x4 = x[kk + 4] as i32;
        let x5 = x[kk + 5] as i32;
        let x6 = x[kk + 6] as i32;
        let x7 = x[kk + 7] as i32;
        let x8 = x[kk + 8] as i32;
        let x9 = x[kk + 9] as i32;
        let x10 = x[kk + 10] as i32;
        let x11 = x[kk + 11] as i32;
        let x12 = x[kk + 12] as i32;
        let x13 = x[kk + 13] as i32;
        let x14 = x[kk + 14] as i32;
        let x15 = x[kk + 15] as i32;

        a0 += row0[kk] as i32 * x0;
        a1 += row1[kk] as i32 * x0;
        a2 += row2[kk] as i32 * x0;
        b0 += row0[kk + 1] as i32 * x1;
        b1 += row1[kk + 1] as i32 * x1;
        b2 += row2[kk + 1] as i32 * x1;
        c0 += row0[kk + 2] as i32 * x2;
        c1 += row1[kk + 2] as i32 * x2;
        c2 += row2[kk + 2] as i32 * x2;
        d0 += row0[kk + 3] as i32 * x3;
        d1 += row1[kk + 3] as i32 * x3;
        d2 += row2[kk + 3] as i32 * x3;
        e0 += row0[kk + 4] as i32 * x4;
        e1 += row1[kk + 4] as i32 * x4;
        e2 += row2[kk + 4] as i32 * x4;
        f0 += row0[kk + 5] as i32 * x5;
        f1 += row1[kk + 5] as i32 * x5;
        f2 += row2[kk + 5] as i32 * x5;
        g0 += row0[kk + 6] as i32 * x6;
        g1 += row1[kk + 6] as i32 * x6;
        g2 += row2[kk + 6] as i32 * x6;
        h0 += row0[kk + 7] as i32 * x7;
        h1 += row1[kk + 7] as i32 * x7;
        h2 += row2[kk + 7] as i32 * x7;
        a0 += row0[kk + 8] as i32 * x8;
        a1 += row1[kk + 8] as i32 * x8;
        a2 += row2[kk + 8] as i32 * x8;
        b0 += row0[kk + 9] as i32 * x9;
        b1 += row1[kk + 9] as i32 * x9;
        b2 += row2[kk + 9] as i32 * x9;
        c0 += row0[kk + 10] as i32 * x10;
        c1 += row1[kk + 10] as i32 * x10;
        c2 += row2[kk + 10] as i32 * x10;
        d0 += row0[kk + 11] as i32 * x11;
        d1 += row1[kk + 11] as i32 * x11;
        d2 += row2[kk + 11] as i32 * x11;
        e0 += row0[kk + 12] as i32 * x12;
        e1 += row1[kk + 12] as i32 * x12;
        e2 += row2[kk + 12] as i32 * x12;
        f0 += row0[kk + 13] as i32 * x13;
        f1 += row1[kk + 13] as i32 * x13;
        f2 += row2[kk + 13] as i32 * x13;
        g0 += row0[kk + 14] as i32 * x14;
        g1 += row1[kk + 14] as i32 * x14;
        g2 += row2[kk + 14] as i32 * x14;
        h0 += row0[kk + 15] as i32 * x15;
        h1 += row1[kk + 15] as i32 * x15;
        h2 += row2[kk + 15] as i32 * x15;
        kk += 16;
    }

    while kk + 8 <= k_dim {
        let x0 = x[kk] as i32;
        let x1 = x[kk + 1] as i32;
        let x2 = x[kk + 2] as i32;
        let x3 = x[kk + 3] as i32;
        let x4 = x[kk + 4] as i32;
        let x5 = x[kk + 5] as i32;
        let x6 = x[kk + 6] as i32;
        let x7 = x[kk + 7] as i32;

        a0 += row0[kk] as i32 * x0 + row0[kk + 4] as i32 * x4;
        a1 += row1[kk] as i32 * x0 + row1[kk + 4] as i32 * x4;
        a2 += row2[kk] as i32 * x0 + row2[kk + 4] as i32 * x4;
        b0 += row0[kk + 1] as i32 * x1 + row0[kk + 5] as i32 * x5;
        b1 += row1[kk + 1] as i32 * x1 + row1[kk + 5] as i32 * x5;
        b2 += row2[kk + 1] as i32 * x1 + row2[kk + 5] as i32 * x5;
        c0 += row0[kk + 2] as i32 * x2 + row0[kk + 6] as i32 * x6;
        c1 += row1[kk + 2] as i32 * x2 + row1[kk + 6] as i32 * x6;
        c2 += row2[kk + 2] as i32 * x2 + row2[kk + 6] as i32 * x6;
        d0 += row0[kk + 3] as i32 * x3 + row0[kk + 7] as i32 * x7;
        d1 += row1[kk + 3] as i32 * x3 + row1[kk + 7] as i32 * x7;
        d2 += row2[kk + 3] as i32 * x3 + row2[kk + 7] as i32 * x7;
        kk += 8;
    }

    while kk + 4 <= k_dim {
        let x0 = x[kk] as i32;
        let x1 = x[kk + 1] as i32;
        let x2 = x[kk + 2] as i32;
        let x3 = x[kk + 3] as i32;
        a0 += row0[kk] as i32 * x0;
        a1 += row1[kk] as i32 * x0;
        a2 += row2[kk] as i32 * x0;
        b0 += row0[kk + 1] as i32 * x1;
        b1 += row1[kk + 1] as i32 * x1;
        b2 += row2[kk + 1] as i32 * x1;
        c0 += row0[kk + 2] as i32 * x2;
        c1 += row1[kk + 2] as i32 * x2;
        c2 += row2[kk + 2] as i32 * x2;
        d0 += row0[kk + 3] as i32 * x3;
        d1 += row1[kk + 3] as i32 * x3;
        d2 += row2[kk + 3] as i32 * x3;
        kk += 4;
    }

    let mut sum0 = a0 + b0 + c0 + d0 + e0 + f0 + g0 + h0;
    let mut sum1 = a1 + b1 + c1 + d1 + e1 + f1 + g1 + h1;
    let mut sum2 = a2 + b2 + c2 + d2 + e2 + f2 + g2 + h2;
    while kk < k_dim {
        let xv = x[kk] as i32;
        sum0 += row0[kk] as i32 * xv;
        sum1 += row1[kk] as i32 * xv;
        sum2 += row2[kk] as i32 * xv;
        kk += 1;
    }
    (
        (sum0 as f32) * x_scale * scale0,
        (sum1 as f32) * x_scale * scale1,
        (sum2 as f32) * x_scale * scale2,
    )
}

#[inline]
pub(crate) fn dot_unrolled_f32_i8(x: &[f32], row: &[i8], scale: f32) -> f32 {
    if let Some(sum) = dot_f32_i8_arch(x, row, scale) {
        sum
    } else {
        dot_unrolled_f32_i8_portable(x, row, scale)
    }
}

#[inline]
fn matvec_rowmajor_serial(
    x: &[f32],
    w_rowmajor: &[f32],
    n_rows: usize,
    k_dim: usize,
    out: &mut [f32],
) {
    for i in 0..n_rows {
        let row = &w_rowmajor[i * k_dim..(i + 1) * k_dim];
        out[i] = dot_unrolled(x, row);
    }
}

#[inline]
fn matvec_rowmajor_serial_f32_bf16(
    x: &[f32],
    w_rowmajor: &[bf16],
    n_rows: usize,
    k_dim: usize,
    out: &mut [f32],
) {
    for i in 0..n_rows {
        let row = &w_rowmajor[i * k_dim..(i + 1) * k_dim];
        out[i] = dot_unrolled_f32_bf16(x, row);
    }
}

#[inline]
fn matvec_rowmajor_serial_f32_f16(
    x: &[f32],
    w_rowmajor: &[f16],
    n_rows: usize,
    k_dim: usize,
    out: &mut [f32],
) {
    for i in 0..n_rows {
        let row = &w_rowmajor[i * k_dim..(i + 1) * k_dim];
        out[i] = dot_unrolled_f32_f16(x, row);
    }
}

#[inline]
fn matvec_rowmajor_serial_f32_i8(
    x: &[f32],
    w_rowmajor: &[i8],
    scale: f32,
    n_rows: usize,
    k_dim: usize,
    out: &mut [f32],
) {
    #[cfg(all(feature = "arm64-int8-kernels", target_arch = "aarch64"))]
    {
        let mut i = 0usize;
        while i + 1 < n_rows {
            let row0 = &w_rowmajor[i * k_dim..(i + 1) * k_dim];
            let row1 = &w_rowmajor[(i + 1) * k_dim..(i + 2) * k_dim];
            let (s0, s1) = dot2_unrolled_f32_i8(x, row0, scale, row1, scale);
            out[i] = s0;
            out[i + 1] = s1;
            i += 2;
        }
        if i < n_rows {
            let row = &w_rowmajor[i * k_dim..(i + 1) * k_dim];
            out[i] = dot_unrolled_f32_i8(x, row, scale);
        }
        return;
    }

    #[cfg(not(all(feature = "arm64-int8-kernels", target_arch = "aarch64")))]
    {
        for i in 0..n_rows {
            let row = &w_rowmajor[i * k_dim..(i + 1) * k_dim];
            out[i] = dot_unrolled_f32_i8(x, row, scale);
        }
    }
}

fn matvec_rowmajor_rowwise_parallel(
    x: &[f32],
    w_rowmajor: &[f32],
    _n_rows: usize,
    k_dim: usize,
    out: &mut [f32],
) {
    out.par_iter_mut().enumerate().for_each(|(i, out_val)| {
        let row = &w_rowmajor[i * k_dim..(i + 1) * k_dim];
        *out_val = dot_unrolled(x, row);
    });
}

#[inline]
fn matvec_rowmajor_rowwise_parallel_f32_bf16(
    x: &[f32],
    w_rowmajor: &[bf16],
    _n_rows: usize,
    k_dim: usize,
    out: &mut [f32],
) {
    out.par_chunks_mut(MIXED_ROW_PAR_CHUNK_ROWS)
        .enumerate()
        .for_each(|(chunk_idx, out_chunk)| {
            let row_start = chunk_idx * MIXED_ROW_PAR_CHUNK_ROWS;
            let mut offset = 0usize;
            while offset + 1 < out_chunk.len() {
                let row0_idx = row_start + offset;
                let row1_idx = row0_idx + 1;
                let row0 = &w_rowmajor[row0_idx * k_dim..(row0_idx + 1) * k_dim];
                let row1 = &w_rowmajor[row1_idx * k_dim..(row1_idx + 1) * k_dim];
                let (s0, s1) = dot2_unrolled_f32_bf16(x, row0, row1);
                out_chunk[offset] = s0;
                out_chunk[offset + 1] = s1;
                offset += 2;
            }
            if offset < out_chunk.len() {
                let row_idx = row_start + offset;
                let row = &w_rowmajor[row_idx * k_dim..(row_idx + 1) * k_dim];
                out_chunk[offset] = dot_unrolled_f32_bf16(x, row);
            }
        });
}

#[inline]
fn matvec_rowmajor_rowwise_parallel_f32_f16(
    x: &[f32],
    w_rowmajor: &[f16],
    _n_rows: usize,
    k_dim: usize,
    out: &mut [f32],
) {
    out.par_chunks_mut(MIXED_ROW_PAR_CHUNK_ROWS)
        .enumerate()
        .for_each(|(chunk_idx, out_chunk)| {
            let row_start = chunk_idx * MIXED_ROW_PAR_CHUNK_ROWS;
            let mut offset = 0usize;
            while offset + 1 < out_chunk.len() {
                let row0_idx = row_start + offset;
                let row1_idx = row0_idx + 1;
                let row0 = &w_rowmajor[row0_idx * k_dim..(row0_idx + 1) * k_dim];
                let row1 = &w_rowmajor[row1_idx * k_dim..(row1_idx + 1) * k_dim];
                let (s0, s1) = dot2_unrolled_f32_f16(x, row0, row1);
                out_chunk[offset] = s0;
                out_chunk[offset + 1] = s1;
                offset += 2;
            }
            if offset < out_chunk.len() {
                let row_idx = row_start + offset;
                let row = &w_rowmajor[row_idx * k_dim..(row_idx + 1) * k_dim];
                out_chunk[offset] = dot_unrolled_f32_f16(x, row);
            }
        });
}

#[inline]
fn matvec_rowmajor_rowwise_parallel_f32_i8(
    x: &[f32],
    w_rowmajor: &[i8],
    scale: f32,
    _n_rows: usize,
    k_dim: usize,
    out: &mut [f32],
) {
    out.par_chunks_mut(MATVEC_I8_PAR_CHUNK_ROWS)
        .enumerate()
        .for_each(|(chunk_idx, out_chunk)| {
            let row_start = chunk_idx * MATVEC_I8_PAR_CHUNK_ROWS;

            #[cfg(all(feature = "arm64-int8-kernels", target_arch = "aarch64"))]
            {
                let mut offset = 0usize;
                while offset + 1 < out_chunk.len() {
                    let row0_idx = row_start + offset;
                    let row1_idx = row0_idx + 1;
                    let row0 = &w_rowmajor[row0_idx * k_dim..(row0_idx + 1) * k_dim];
                    let row1 = &w_rowmajor[row1_idx * k_dim..(row1_idx + 1) * k_dim];
                    let (s0, s1) = dot2_unrolled_f32_i8(x, row0, scale, row1, scale);
                    out_chunk[offset] = s0;
                    out_chunk[offset + 1] = s1;
                    offset += 2;
                }
                if offset < out_chunk.len() {
                    let row_idx = row_start + offset;
                    let row = &w_rowmajor[row_idx * k_dim..(row_idx + 1) * k_dim];
                    out_chunk[offset] = dot_unrolled_f32_i8(x, row, scale);
                }
            }

            #[cfg(not(all(feature = "arm64-int8-kernels", target_arch = "aarch64")))]
            {
                for (offset, out_val) in out_chunk.iter_mut().enumerate() {
                    let row_idx = row_start + offset;
                    let row = &w_rowmajor[row_idx * k_dim..(row_idx + 1) * k_dim];
                    *out_val = dot_unrolled_f32_i8(x, row, scale);
                }
            }
        });
}

#[inline]
fn matvec_rowmajor_block_parallel(
    x: &[f32],
    w_rowmajor: &[f32],
    _n_rows: usize,
    k_dim: usize,
    out: &mut [f32],
) {
    out.par_chunks_mut(MATVEC_BLOCK_ROWS)
        .enumerate()
        .for_each(|(block_idx, out_chunk)| {
            let row_start = block_idx * MATVEC_BLOCK_ROWS;
            let rows = out_chunk.len();
            let w_block = &w_rowmajor[row_start * k_dim..(row_start + rows) * k_dim];
            let mut acc = [0.0f32; MATVEC_BLOCK_ROWS];

            let mut kk = 0usize;
            while kk + 8 <= k_dim {
                let x0 = x[kk];
                let x1 = x[kk + 1];
                let x2 = x[kk + 2];
                let x3 = x[kk + 3];
                let x4 = x[kk + 4];
                let x5 = x[kk + 5];
                let x6 = x[kk + 6];
                let x7 = x[kk + 7];
                for (r, acc_r) in acc.iter_mut().enumerate().take(rows) {
                    let base = r * k_dim + kk;
                    *acc_r += w_block[base] * x0
                        + w_block[base + 1] * x1
                        + w_block[base + 2] * x2
                        + w_block[base + 3] * x3
                        + w_block[base + 4] * x4
                        + w_block[base + 5] * x5
                        + w_block[base + 6] * x6
                        + w_block[base + 7] * x7;
                }
                kk += 8;
            }

            while kk < k_dim {
                let xv = x[kk];
                for (r, acc_r) in acc.iter_mut().enumerate().take(rows) {
                    *acc_r += w_block[r * k_dim + kk] * xv;
                }
                kk += 1;
            }

            out_chunk.copy_from_slice(&acc[..rows]);
        });
}

#[inline]
fn matvec_rowmajor_block_parallel_f32_bf16(
    x: &[f32],
    w_rowmajor: &[bf16],
    _n_rows: usize,
    k_dim: usize,
    out: &mut [f32],
) {
    out.par_chunks_mut(MATVEC_BLOCK_ROWS)
        .enumerate()
        .for_each(|(block_idx, out_chunk)| {
            let row_start = block_idx * MATVEC_BLOCK_ROWS;
            let rows = out_chunk.len();
            let w_block = &w_rowmajor[row_start * k_dim..(row_start + rows) * k_dim];
            let mut acc = [0.0f32; MATVEC_BLOCK_ROWS];

            let mut kk = 0usize;
            while kk + 8 <= k_dim {
                let x0 = x[kk];
                let x1 = x[kk + 1];
                let x2 = x[kk + 2];
                let x3 = x[kk + 3];
                let x4 = x[kk + 4];
                let x5 = x[kk + 5];
                let x6 = x[kk + 6];
                let x7 = x[kk + 7];
                for (r, acc_r) in acc.iter_mut().enumerate().take(rows) {
                    let base = r * k_dim + kk;
                    *acc_r += w_block[base].to_f32() * x0
                        + w_block[base + 1].to_f32() * x1
                        + w_block[base + 2].to_f32() * x2
                        + w_block[base + 3].to_f32() * x3
                        + w_block[base + 4].to_f32() * x4
                        + w_block[base + 5].to_f32() * x5
                        + w_block[base + 6].to_f32() * x6
                        + w_block[base + 7].to_f32() * x7;
                }
                kk += 8;
            }

            while kk < k_dim {
                let xv = x[kk];
                for (r, acc_r) in acc.iter_mut().enumerate().take(rows) {
                    *acc_r += w_block[r * k_dim + kk].to_f32() * xv;
                }
                kk += 1;
            }

            out_chunk.copy_from_slice(&acc[..rows]);
        });
}

fn matvec_rowmajor_block_parallel_f32_i8(
    x: &[f32],
    w_rowmajor: &[i8],
    scale: f32,
    _n_rows: usize,
    k_dim: usize,
    out: &mut [f32],
) {
    out.par_chunks_mut(MATVEC_BLOCK_ROWS)
        .enumerate()
        .for_each(|(block_idx, out_chunk)| {
            let row_start = block_idx * MATVEC_BLOCK_ROWS;
            let rows = out_chunk.len();
            let w_block = &w_rowmajor[row_start * k_dim..(row_start + rows) * k_dim];
            let mut acc = [0.0f32; MATVEC_BLOCK_ROWS];

            let mut kk = 0usize;
            while kk + 16 <= k_dim {
                let x0 = x[kk];
                let x1 = x[kk + 1];
                let x2 = x[kk + 2];
                let x3 = x[kk + 3];
                let x4 = x[kk + 4];
                let x5 = x[kk + 5];
                let x6 = x[kk + 6];
                let x7 = x[kk + 7];
                let x8 = x[kk + 8];
                let x9 = x[kk + 9];
                let x10 = x[kk + 10];
                let x11 = x[kk + 11];
                let x12 = x[kk + 12];
                let x13 = x[kk + 13];
                let x14 = x[kk + 14];
                let x15 = x[kk + 15];
                for (r, acc_r) in acc.iter_mut().enumerate().take(rows) {
                    let base = r * k_dim + kk;
                    *acc_r += w_block[base] as f32 * x0
                        + w_block[base + 1] as f32 * x1
                        + w_block[base + 2] as f32 * x2
                        + w_block[base + 3] as f32 * x3
                        + w_block[base + 4] as f32 * x4
                        + w_block[base + 5] as f32 * x5
                        + w_block[base + 6] as f32 * x6
                        + w_block[base + 7] as f32 * x7
                        + w_block[base + 8] as f32 * x8
                        + w_block[base + 9] as f32 * x9
                        + w_block[base + 10] as f32 * x10
                        + w_block[base + 11] as f32 * x11
                        + w_block[base + 12] as f32 * x12
                        + w_block[base + 13] as f32 * x13
                        + w_block[base + 14] as f32 * x14
                        + w_block[base + 15] as f32 * x15;
                }
                kk += 16;
            }

            while kk + 8 <= k_dim {
                let x0 = x[kk];
                let x1 = x[kk + 1];
                let x2 = x[kk + 2];
                let x3 = x[kk + 3];
                let x4 = x[kk + 4];
                let x5 = x[kk + 5];
                let x6 = x[kk + 6];
                let x7 = x[kk + 7];
                for (r, acc_r) in acc.iter_mut().enumerate().take(rows) {
                    let base = r * k_dim + kk;
                    *acc_r += w_block[base] as f32 * x0
                        + w_block[base + 1] as f32 * x1
                        + w_block[base + 2] as f32 * x2
                        + w_block[base + 3] as f32 * x3
                        + w_block[base + 4] as f32 * x4
                        + w_block[base + 5] as f32 * x5
                        + w_block[base + 6] as f32 * x6
                        + w_block[base + 7] as f32 * x7;
                }
                kk += 8;
            }

            while kk < k_dim {
                let xv = x[kk];
                for (r, acc_r) in acc.iter_mut().enumerate().take(rows) {
                    *acc_r += w_block[r * k_dim + kk] as f32 * xv;
                }
                kk += 1;
            }

            for r in 0..rows {
                out_chunk[r] = acc[r] * scale;
            }
        });
}

pub fn matvec_rowmajor_parallel(
    x: &[f32],
    w_rowmajor: &[f32],
    n_rows: usize,
    k_dim: usize,
    out: &mut [f32],
) {
    assert_eq!(x.len(), k_dim, "x len / k_dim mismatch");
    assert_eq!(w_rowmajor.len(), n_rows * k_dim, "weight size mismatch");
    assert_eq!(out.len(), n_rows, "out size mismatch");

    if n_rows < MATVEC_PAR_THRESHOLD {
        matvec_rowmajor_serial(x, w_rowmajor, n_rows, k_dim, out);
    } else if n_rows >= MATVEC_BLOCK_THRESHOLD {
        matvec_rowmajor_block_parallel(x, w_rowmajor, n_rows, k_dim, out);
    } else {
        matvec_rowmajor_rowwise_parallel(x, w_rowmajor, n_rows, k_dim, out);
    }
}

#[inline]
pub fn matvec_rowmajor_parallel_f32_bf16(
    x: &[f32],
    w_rowmajor: &[bf16],
    n_rows: usize,
    k_dim: usize,
    out: &mut [f32],
) {
    assert_eq!(x.len(), k_dim, "x len / k_dim mismatch");
    assert_eq!(w_rowmajor.len(), n_rows * k_dim, "weight size mismatch");
    assert_eq!(out.len(), n_rows, "out size mismatch");

    if n_rows < MATVEC_PAR_THRESHOLD {
        matvec_rowmajor_serial_f32_bf16(x, w_rowmajor, n_rows, k_dim, out);
    } else if should_use_mixed_matvec_block_kernel(n_rows) {
        matvec_rowmajor_block_parallel_f32_bf16(x, w_rowmajor, n_rows, k_dim, out);
    } else {
        matvec_rowmajor_rowwise_parallel_f32_bf16(x, w_rowmajor, n_rows, k_dim, out);
    }
}

#[inline]
pub fn matvec_rowmajor_parallel_f32_f16(
    x: &[f32],
    w_rowmajor: &[f16],
    n_rows: usize,
    k_dim: usize,
    out: &mut [f32],
) {
    assert_eq!(x.len(), k_dim, "x len / k_dim mismatch");
    assert_eq!(w_rowmajor.len(), n_rows * k_dim, "weight size mismatch");
    assert_eq!(out.len(), n_rows, "out size mismatch");

    if n_rows < MATVEC_PAR_THRESHOLD {
        matvec_rowmajor_serial_f32_f16(x, w_rowmajor, n_rows, k_dim, out);
    } else {
        matvec_rowmajor_rowwise_parallel_f32_f16(x, w_rowmajor, n_rows, k_dim, out);
    }
}

#[inline]
pub fn matvec_rowmajor_parallel_f32_i8(
    x: &[f32],
    w_rowmajor: &[i8],
    scale: f32,
    n_rows: usize,
    k_dim: usize,
    out: &mut [f32],
) {
    assert_eq!(x.len(), k_dim, "x len / k_dim mismatch");
    assert_eq!(w_rowmajor.len(), n_rows * k_dim, "weight size mismatch");
    assert_eq!(out.len(), n_rows, "out size mismatch");

    if n_rows < MATVEC_PAR_THRESHOLD {
        matvec_rowmajor_serial_f32_i8(x, w_rowmajor, scale, n_rows, k_dim, out);
    } else if should_use_i8_block_kernel(n_rows, k_dim) {
        matvec_rowmajor_block_parallel_f32_i8(x, w_rowmajor, scale, n_rows, k_dim, out);
    } else {
        matvec_rowmajor_rowwise_parallel_f32_i8(x, w_rowmajor, scale, n_rows, k_dim, out);
    }
}

#[inline]
pub fn matvec_rowmajor_parallel_f32_i8_matmul(
    x: &[f32],
    w_rowmajor: &[i8],
    scale: f32,
    n_rows: usize,
    k_dim: usize,
    out: &mut [f32],
) {
    assert_eq!(x.len(), k_dim, "x len / k_dim mismatch");
    assert_eq!(w_rowmajor.len(), n_rows * k_dim, "weight size mismatch");
    assert_eq!(out.len(), n_rows, "out size mismatch");

    if n_rows < MATVEC_PAR_THRESHOLD {
        matvec_rowmajor_serial_f32_i8(x, w_rowmajor, scale, n_rows, k_dim, out);
    } else if should_use_i8_matmul_block_kernel(n_rows, k_dim) {
        matvec_rowmajor_block_parallel_f32_i8(x, w_rowmajor, scale, n_rows, k_dim, out);
    } else {
        matvec_rowmajor_rowwise_parallel_f32_i8(x, w_rowmajor, scale, n_rows, k_dim, out);
    }
}

#[inline]
pub fn matvec_rowmajor_parallel_bf16_f32(
    x: &[bf16],
    w_rowmajor: &[f32],
    n_rows: usize,
    k_dim: usize,
    out: &mut [f32],
) {
    with_bf16_input_as_f32(x, |x_f32| {
        matvec_rowmajor_parallel(x_f32, w_rowmajor, n_rows, k_dim, out);
    });
}

#[inline]
pub fn matvec_rowmajor_parallel_bf16_bf16(
    x: &[bf16],
    w_rowmajor: &[bf16],
    n_rows: usize,
    k_dim: usize,
    out: &mut [f32],
) {
    assert_eq!(x.len(), k_dim, "x len / k_dim mismatch");
    assert_eq!(w_rowmajor.len(), n_rows * k_dim, "weight size mismatch");
    assert_eq!(out.len(), n_rows, "out size mismatch");

    if crate::arch::x86_avx512_bf16_kernel_runtime_available() {
        if n_rows < MATVEC_PAR_THRESHOLD {
            let mut i = 0usize;
            while i + 2 < n_rows {
                let row0 = &w_rowmajor[i * k_dim..(i + 1) * k_dim];
                let row1 = &w_rowmajor[(i + 1) * k_dim..(i + 2) * k_dim];
                let row2 = &w_rowmajor[(i + 2) * k_dim..(i + 3) * k_dim];
                if let Some((s0, s1, s2)) = dot3_bf16_bf16_arch(x, row0, row1, row2) {
                    out[i] = s0;
                    out[i + 1] = s1;
                    out[i + 2] = s2;
                    i += 3;
                    continue;
                }
                break;
            }
            while i + 1 < n_rows {
                let row0 = &w_rowmajor[i * k_dim..(i + 1) * k_dim];
                let row1 = &w_rowmajor[(i + 1) * k_dim..(i + 2) * k_dim];
                if let Some((s0, s1)) = dot2_bf16_bf16_arch(x, row0, row1) {
                    out[i] = s0;
                    out[i + 1] = s1;
                    i += 2;
                    continue;
                }
                break;
            }
            while i < n_rows {
                let row = &w_rowmajor[i * k_dim..(i + 1) * k_dim];
                if let Some(sum) = dot_bf16_bf16_arch(x, row) {
                    out[i] = sum;
                    i += 1;
                    continue;
                }
                break;
            }
            if i == n_rows {
                return;
            }
        } else {
            out.par_iter_mut().enumerate().for_each(|(i, out_val)| {
                let row = &w_rowmajor[i * k_dim..(i + 1) * k_dim];
                *out_val = dot_bf16_bf16_arch(x, row)
                    .unwrap_or_else(|| dot_unrolled_bf16_bf16_scalar(x, row));
            });
            return;
        }
    }

    with_bf16_input_as_f32(x, |x_f32| {
        matvec_rowmajor_parallel_f32_bf16(x_f32, w_rowmajor, n_rows, k_dim, out);
    });
}

#[inline]
pub fn matvec_rowmajor_parallel_f16_f16(
    x: &[f16],
    w_rowmajor: &[f16],
    n_rows: usize,
    k_dim: usize,
    out: &mut [f32],
) {
    assert_eq!(x.len(), k_dim, "x len / k_dim mismatch");
    assert_eq!(w_rowmajor.len(), n_rows * k_dim, "weight size mismatch");
    assert_eq!(out.len(), n_rows, "out size mismatch");

    if crate::arch::x86_avx512_fp16_kernel_runtime_available() {
        if n_rows < MATVEC_PAR_THRESHOLD {
            let mut i = 0usize;
            while i + 2 < n_rows {
                let row0 = &w_rowmajor[i * k_dim..(i + 1) * k_dim];
                let row1 = &w_rowmajor[(i + 1) * k_dim..(i + 2) * k_dim];
                let row2 = &w_rowmajor[(i + 2) * k_dim..(i + 3) * k_dim];
                if let Some((s0, s1, s2)) = dot3_f16_f16_arch(x, row0, row1, row2) {
                    out[i] = s0;
                    out[i + 1] = s1;
                    out[i + 2] = s2;
                    i += 3;
                    continue;
                }
                break;
            }
            while i + 1 < n_rows {
                let row0 = &w_rowmajor[i * k_dim..(i + 1) * k_dim];
                let row1 = &w_rowmajor[(i + 1) * k_dim..(i + 2) * k_dim];
                if let Some((s0, s1)) = dot2_f16_f16_arch(x, row0, row1) {
                    out[i] = s0;
                    out[i + 1] = s1;
                    i += 2;
                    continue;
                }
                break;
            }
            while i < n_rows {
                let row = &w_rowmajor[i * k_dim..(i + 1) * k_dim];
                if let Some(sum) = dot_f16_f16_arch(x, row) {
                    out[i] = sum;
                    i += 1;
                    continue;
                }
                break;
            }
            if i == n_rows {
                return;
            }
        } else {
            out.par_iter_mut().enumerate().for_each(|(i, out_val)| {
                let row = &w_rowmajor[i * k_dim..(i + 1) * k_dim];
                *out_val =
                    dot_f16_f16_arch(x, row).unwrap_or_else(|| dot_unrolled_f16_f16_scalar(x, row));
            });
            return;
        }
    }

    with_f16_input_as_f32(x, |x_f32| {
        matvec_rowmajor_parallel_f32_f16(x_f32, w_rowmajor, n_rows, k_dim, out);
    });
}

#[inline]
pub fn matvec_rowmajor_parallel_bf16_i8(
    x: &[bf16],
    w_rowmajor: &[i8],
    scale: f32,
    n_rows: usize,
    k_dim: usize,
    out: &mut [f32],
) {
    with_bf16_input_as_f32(x, |x_f32| {
        matvec_rowmajor_parallel_f32_i8(x_f32, w_rowmajor, scale, n_rows, k_dim, out);
    });
}

#[inline]
pub fn matvec_rowmajor_parallel_i8_f32(
    x: &[i8],
    x_scale: f32,
    w_rowmajor: &[f32],
    n_rows: usize,
    k_dim: usize,
    out: &mut [f32],
) {
    with_i8_input_as_f32(x, x_scale, |x_f32| {
        matvec_rowmajor_parallel(x_f32, w_rowmajor, n_rows, k_dim, out);
    });
}

#[inline]
pub fn matvec_rowmajor_parallel_i8_bf16(
    x: &[i8],
    x_scale: f32,
    w_rowmajor: &[bf16],
    n_rows: usize,
    k_dim: usize,
    out: &mut [f32],
) {
    with_i8_input_as_f32(x, x_scale, |x_f32| {
        matvec_rowmajor_parallel_f32_bf16(x_f32, w_rowmajor, n_rows, k_dim, out);
    });
}

#[inline]
pub fn matvec_rowmajor_parallel_i8_i8(
    x: &[i8],
    x_scale: f32,
    w_rowmajor: &[i8],
    w_scale: f32,
    n_rows: usize,
    k_dim: usize,
    out: &mut [f32],
) {
    assert_eq!(x.len(), k_dim, "x len / k_dim mismatch");
    assert_eq!(w_rowmajor.len(), n_rows * k_dim, "weight size mismatch");
    assert_eq!(out.len(), n_rows, "out size mismatch");

    if n_rows < MATVEC_PAR_THRESHOLD {
        #[cfg(all(feature = "arm64-int8-kernels", target_arch = "aarch64"))]
        {
            let mut i = 0usize;
            while i + 1 < n_rows {
                let row0 = &w_rowmajor[i * k_dim..(i + 1) * k_dim];
                let row1 = &w_rowmajor[(i + 1) * k_dim..(i + 2) * k_dim];
                let (s0, s1) = dot2_unrolled_i8_i8(x, x_scale, row0, w_scale, row1, w_scale);
                out[i] = s0;
                out[i + 1] = s1;
                i += 2;
            }
            if i < n_rows {
                let row = &w_rowmajor[i * k_dim..(i + 1) * k_dim];
                out[i] = dot_unrolled_i8_i8(x, x_scale, row, w_scale);
            }
        }

        #[cfg(not(all(feature = "arm64-int8-kernels", target_arch = "aarch64")))]
        {
            for i in 0..n_rows {
                let row = &w_rowmajor[i * k_dim..(i + 1) * k_dim];
                out[i] = dot_unrolled_i8_i8(x, x_scale, row, w_scale);
            }
        }
    } else {
        out.par_chunks_mut(MATVEC_I8_PAR_CHUNK_ROWS)
            .enumerate()
            .for_each(|(chunk_idx, out_chunk)| {
                let row_start = chunk_idx * MATVEC_I8_PAR_CHUNK_ROWS;

                #[cfg(all(feature = "arm64-int8-kernels", target_arch = "aarch64"))]
                {
                    let mut offset = 0usize;
                    while offset + 1 < out_chunk.len() {
                        let row0_idx = row_start + offset;
                        let row1_idx = row0_idx + 1;
                        let row0 = &w_rowmajor[row0_idx * k_dim..(row0_idx + 1) * k_dim];
                        let row1 = &w_rowmajor[row1_idx * k_dim..(row1_idx + 1) * k_dim];
                        let (s0, s1) =
                            dot2_unrolled_i8_i8(x, x_scale, row0, w_scale, row1, w_scale);
                        out_chunk[offset] = s0;
                        out_chunk[offset + 1] = s1;
                        offset += 2;
                    }
                    if offset < out_chunk.len() {
                        let row_idx = row_start + offset;
                        let row = &w_rowmajor[row_idx * k_dim..(row_idx + 1) * k_dim];
                        out_chunk[offset] = dot_unrolled_i8_i8(x, x_scale, row, w_scale);
                    }
                }

                #[cfg(not(all(feature = "arm64-int8-kernels", target_arch = "aarch64")))]
                {
                    for (offset, out_val) in out_chunk.iter_mut().enumerate() {
                        let row_idx = row_start + offset;
                        let row = &w_rowmajor[row_idx * k_dim..(row_idx + 1) * k_dim];
                        *out_val = dot_unrolled_i8_i8(x, x_scale, row, w_scale);
                    }
                }
            });
    }
}

#[inline]
pub fn matvec_rowmajor_parallel_mixed(
    x: SliceRef<'_>,
    w_rowmajor: SliceRef<'_>,
    n_rows: usize,
    k_dim: usize,
    out: &mut [f32],
) {
    assert_eq!(out.len(), n_rows, "out size mismatch");

    match (x, w_rowmajor) {
        (SliceRef::F32(x), SliceRef::F32(w)) => matvec_rowmajor_parallel(x, w, n_rows, k_dim, out),
        (SliceRef::F32(x), SliceRef::F16(w)) => {
            matvec_rowmajor_parallel_f32_f16(x, w, n_rows, k_dim, out);
        }
        (SliceRef::F32(x), SliceRef::BF16(w)) => {
            matvec_rowmajor_parallel_f32_bf16(x, w, n_rows, k_dim, out);
        }
        (SliceRef::F32(x), SliceRef::I8(w, scale)) => {
            matvec_rowmajor_parallel_f32_i8(x, w, scale, n_rows, k_dim, out);
        }
        (SliceRef::F16(x), SliceRef::F32(w)) => {
            with_f16_input_as_f32(x, |x_f32| {
                matvec_rowmajor_parallel(x_f32, w, n_rows, k_dim, out);
            });
        }
        (SliceRef::F16(x), SliceRef::F16(w)) => {
            matvec_rowmajor_parallel_f16_f16(x, w, n_rows, k_dim, out);
        }
        (SliceRef::F16(x), SliceRef::BF16(w)) => {
            with_f16_input_as_f32(x, |x_f32| {
                matvec_rowmajor_parallel_f32_bf16(x_f32, w, n_rows, k_dim, out);
            });
        }
        (SliceRef::F16(x), SliceRef::I8(w, scale)) => {
            with_f16_input_as_f32(x, |x_f32| {
                matvec_rowmajor_parallel_f32_i8(x_f32, w, scale, n_rows, k_dim, out);
            });
        }
        (SliceRef::BF16(x), SliceRef::F32(w)) => {
            matvec_rowmajor_parallel_bf16_f32(x, w, n_rows, k_dim, out);
        }
        (SliceRef::BF16(x), SliceRef::BF16(w)) => {
            matvec_rowmajor_parallel_bf16_bf16(x, w, n_rows, k_dim, out);
        }
        (SliceRef::BF16(x), SliceRef::I8(w, scale)) => {
            matvec_rowmajor_parallel_bf16_i8(x, w, scale, n_rows, k_dim, out);
        }
        (SliceRef::BF16(x), SliceRef::F16(w)) => {
            with_bf16_input_as_f32(x, |x_f32| {
                matvec_rowmajor_parallel_f32_f16(x_f32, w, n_rows, k_dim, out);
            });
        }
        (SliceRef::I8(x, scale), SliceRef::F32(w)) => {
            matvec_rowmajor_parallel_i8_f32(x, scale, w, n_rows, k_dim, out);
        }
        (SliceRef::I8(x, scale), SliceRef::F16(w)) => {
            with_i8_input_as_f32(x, scale, |x_f32| {
                matvec_rowmajor_parallel_f32_f16(x_f32, w, n_rows, k_dim, out)
            });
        }
        (SliceRef::I8(x, scale), SliceRef::BF16(w)) => {
            matvec_rowmajor_parallel_i8_bf16(x, scale, w, n_rows, k_dim, out);
        }
        (SliceRef::I8(x, x_scale), SliceRef::I8(w, w_scale)) => {
            matvec_rowmajor_parallel_i8_i8(x, x_scale, w, w_scale, n_rows, k_dim, out);
        }
    }
}

#[inline]
pub(crate) fn dot2_unrolled(x: &[f32], row0: &[f32], row1: &[f32]) -> (f32, f32) {
    if let Some(sum) = dot2_f32_arch(x, row0, row1) {
        return sum;
    }

    let mut a0 = 0.0f32;
    let mut a1 = 0.0f32;
    let mut b0 = 0.0f32;
    let mut b1 = 0.0f32;
    let mut c0 = 0.0f32;
    let mut c1 = 0.0f32;
    let mut d0 = 0.0f32;
    let mut d1 = 0.0f32;
    let mut kk = 0usize;
    let k_dim = x.len();

    while kk + 8 <= k_dim {
        let x0 = x[kk];
        let x1 = x[kk + 1];
        let x2 = x[kk + 2];
        let x3 = x[kk + 3];
        let x4 = x[kk + 4];
        let x5 = x[kk + 5];
        let x6 = x[kk + 6];
        let x7 = x[kk + 7];

        a0 += row0[kk] * x0 + row0[kk + 4] * x4;
        a1 += row1[kk] * x0 + row1[kk + 4] * x4;
        b0 += row0[kk + 1] * x1 + row0[kk + 5] * x5;
        b1 += row1[kk + 1] * x1 + row1[kk + 5] * x5;
        c0 += row0[kk + 2] * x2 + row0[kk + 6] * x6;
        c1 += row1[kk + 2] * x2 + row1[kk + 6] * x6;
        d0 += row0[kk + 3] * x3 + row0[kk + 7] * x7;
        d1 += row1[kk + 3] * x3 + row1[kk + 7] * x7;
        kk += 8;
    }

    while kk + 4 <= k_dim {
        let x0 = x[kk];
        let x1 = x[kk + 1];
        let x2 = x[kk + 2];
        let x3 = x[kk + 3];
        a0 += row0[kk] * x0;
        a1 += row1[kk] * x0;
        b0 += row0[kk + 1] * x1;
        b1 += row1[kk + 1] * x1;
        c0 += row0[kk + 2] * x2;
        c1 += row1[kk + 2] * x2;
        d0 += row0[kk + 3] * x3;
        d1 += row1[kk + 3] * x3;
        kk += 4;
    }

    let mut sum0 = a0 + b0 + c0 + d0;
    let mut sum1 = a1 + b1 + c1 + d1;
    while kk < k_dim {
        let xv = x[kk];
        sum0 += row0[kk] * xv;
        sum1 += row1[kk] * xv;
        kk += 1;
    }
    (sum0, sum1)
}

#[inline]
pub(crate) fn dot2_unrolled_f32_bf16(x: &[f32], row0: &[bf16], row1: &[bf16]) -> (f32, f32) {
    if let Some(sum) = dot2_f32_bf16_arch(x, row0, row1) {
        return sum;
    }

    let mut a0 = 0.0f32;
    let mut a1 = 0.0f32;
    let mut b0 = 0.0f32;
    let mut b1 = 0.0f32;
    let mut c0 = 0.0f32;
    let mut c1 = 0.0f32;
    let mut d0 = 0.0f32;
    let mut d1 = 0.0f32;
    let mut kk = 0usize;
    let k_dim = x.len();

    while kk + 8 <= k_dim {
        let x0 = x[kk];
        let x1 = x[kk + 1];
        let x2 = x[kk + 2];
        let x3 = x[kk + 3];
        let x4 = x[kk + 4];
        let x5 = x[kk + 5];
        let x6 = x[kk + 6];
        let x7 = x[kk + 7];

        a0 += row0[kk].to_f32() * x0 + row0[kk + 4].to_f32() * x4;
        a1 += row1[kk].to_f32() * x0 + row1[kk + 4].to_f32() * x4;
        b0 += row0[kk + 1].to_f32() * x1 + row0[kk + 5].to_f32() * x5;
        b1 += row1[kk + 1].to_f32() * x1 + row1[kk + 5].to_f32() * x5;
        c0 += row0[kk + 2].to_f32() * x2 + row0[kk + 6].to_f32() * x6;
        c1 += row1[kk + 2].to_f32() * x2 + row1[kk + 6].to_f32() * x6;
        d0 += row0[kk + 3].to_f32() * x3 + row0[kk + 7].to_f32() * x7;
        d1 += row1[kk + 3].to_f32() * x3 + row1[kk + 7].to_f32() * x7;
        kk += 8;
    }

    while kk + 4 <= k_dim {
        let x0 = x[kk];
        let x1 = x[kk + 1];
        let x2 = x[kk + 2];
        let x3 = x[kk + 3];
        a0 += row0[kk].to_f32() * x0;
        a1 += row1[kk].to_f32() * x0;
        b0 += row0[kk + 1].to_f32() * x1;
        b1 += row1[kk + 1].to_f32() * x1;
        c0 += row0[kk + 2].to_f32() * x2;
        c1 += row1[kk + 2].to_f32() * x2;
        d0 += row0[kk + 3].to_f32() * x3;
        d1 += row1[kk + 3].to_f32() * x3;
        kk += 4;
    }

    let mut sum0 = a0 + b0 + c0 + d0;
    let mut sum1 = a1 + b1 + c1 + d1;
    while kk < k_dim {
        let xv = x[kk];
        sum0 += row0[kk].to_f32() * xv;
        sum1 += row1[kk].to_f32() * xv;
        kk += 1;
    }
    (sum0, sum1)
}

#[inline]
pub(crate) fn dot2_unrolled_f32_f16(x: &[f32], row0: &[f16], row1: &[f16]) -> (f32, f32) {
    if let Some(sum) = dot2_f32_f16_arch(x, row0, row1) {
        return sum;
    }

    let mut a0 = 0.0f32;
    let mut a1 = 0.0f32;
    let mut b0 = 0.0f32;
    let mut b1 = 0.0f32;
    let mut c0 = 0.0f32;
    let mut c1 = 0.0f32;
    let mut d0 = 0.0f32;
    let mut d1 = 0.0f32;
    let mut kk = 0usize;
    let k_dim = x.len();

    while kk + 8 <= k_dim {
        let x0 = x[kk];
        let x1 = x[kk + 1];
        let x2 = x[kk + 2];
        let x3 = x[kk + 3];
        let x4 = x[kk + 4];
        let x5 = x[kk + 5];
        let x6 = x[kk + 6];
        let x7 = x[kk + 7];

        a0 += row0[kk].to_f32() * x0 + row0[kk + 4].to_f32() * x4;
        a1 += row1[kk].to_f32() * x0 + row1[kk + 4].to_f32() * x4;
        b0 += row0[kk + 1].to_f32() * x1 + row0[kk + 5].to_f32() * x5;
        b1 += row1[kk + 1].to_f32() * x1 + row1[kk + 5].to_f32() * x5;
        c0 += row0[kk + 2].to_f32() * x2 + row0[kk + 6].to_f32() * x6;
        c1 += row1[kk + 2].to_f32() * x2 + row1[kk + 6].to_f32() * x6;
        d0 += row0[kk + 3].to_f32() * x3 + row0[kk + 7].to_f32() * x7;
        d1 += row1[kk + 3].to_f32() * x3 + row1[kk + 7].to_f32() * x7;
        kk += 8;
    }

    while kk + 4 <= k_dim {
        let x0 = x[kk];
        let x1 = x[kk + 1];
        let x2 = x[kk + 2];
        let x3 = x[kk + 3];
        a0 += row0[kk].to_f32() * x0;
        a1 += row1[kk].to_f32() * x0;
        b0 += row0[kk + 1].to_f32() * x1;
        b1 += row1[kk + 1].to_f32() * x1;
        c0 += row0[kk + 2].to_f32() * x2;
        c1 += row1[kk + 2].to_f32() * x2;
        d0 += row0[kk + 3].to_f32() * x3;
        d1 += row1[kk + 3].to_f32() * x3;
        kk += 4;
    }

    let mut sum0 = a0 + b0 + c0 + d0;
    let mut sum1 = a1 + b1 + c1 + d1;
    while kk < k_dim {
        let xv = x[kk];
        sum0 += row0[kk].to_f32() * xv;
        sum1 += row1[kk].to_f32() * xv;
        kk += 1;
    }
    (sum0, sum1)
}

#[inline]
pub(crate) fn dot2_unrolled_f32_i8(
    x: &[f32],
    row0: &[i8],
    scale0: f32,
    row1: &[i8],
    scale1: f32,
) -> (f32, f32) {
    if let Some(sum) = dot2_f32_i8_arch(x, row0, scale0, row1, scale1) {
        return sum;
    }

    let mut a0 = 0.0f32;
    let mut a1 = 0.0f32;
    let mut b0 = 0.0f32;
    let mut b1 = 0.0f32;
    let mut c0 = 0.0f32;
    let mut c1 = 0.0f32;
    let mut d0 = 0.0f32;
    let mut d1 = 0.0f32;
    let mut e0 = 0.0f32;
    let mut e1 = 0.0f32;
    let mut f0 = 0.0f32;
    let mut f1 = 0.0f32;
    let mut g0 = 0.0f32;
    let mut g1 = 0.0f32;
    let mut h0 = 0.0f32;
    let mut h1 = 0.0f32;
    let mut kk = 0usize;
    let k_dim = x.len();

    while kk + 16 <= k_dim {
        let x0 = x[kk];
        let x1 = x[kk + 1];
        let x2 = x[kk + 2];
        let x3 = x[kk + 3];
        let x4 = x[kk + 4];
        let x5 = x[kk + 5];
        let x6 = x[kk + 6];
        let x7 = x[kk + 7];
        let x8 = x[kk + 8];
        let x9 = x[kk + 9];
        let x10 = x[kk + 10];
        let x11 = x[kk + 11];
        let x12 = x[kk + 12];
        let x13 = x[kk + 13];
        let x14 = x[kk + 14];
        let x15 = x[kk + 15];

        a0 += row0[kk] as f32 * x0;
        a1 += row1[kk] as f32 * x0;
        b0 += row0[kk + 1] as f32 * x1;
        b1 += row1[kk + 1] as f32 * x1;
        c0 += row0[kk + 2] as f32 * x2;
        c1 += row1[kk + 2] as f32 * x2;
        d0 += row0[kk + 3] as f32 * x3;
        d1 += row1[kk + 3] as f32 * x3;
        e0 += row0[kk + 4] as f32 * x4;
        e1 += row1[kk + 4] as f32 * x4;
        f0 += row0[kk + 5] as f32 * x5;
        f1 += row1[kk + 5] as f32 * x5;
        g0 += row0[kk + 6] as f32 * x6;
        g1 += row1[kk + 6] as f32 * x6;
        h0 += row0[kk + 7] as f32 * x7;
        h1 += row1[kk + 7] as f32 * x7;
        a0 += row0[kk + 8] as f32 * x8;
        a1 += row1[kk + 8] as f32 * x8;
        b0 += row0[kk + 9] as f32 * x9;
        b1 += row1[kk + 9] as f32 * x9;
        c0 += row0[kk + 10] as f32 * x10;
        c1 += row1[kk + 10] as f32 * x10;
        d0 += row0[kk + 11] as f32 * x11;
        d1 += row1[kk + 11] as f32 * x11;
        e0 += row0[kk + 12] as f32 * x12;
        e1 += row1[kk + 12] as f32 * x12;
        f0 += row0[kk + 13] as f32 * x13;
        f1 += row1[kk + 13] as f32 * x13;
        g0 += row0[kk + 14] as f32 * x14;
        g1 += row1[kk + 14] as f32 * x14;
        h0 += row0[kk + 15] as f32 * x15;
        h1 += row1[kk + 15] as f32 * x15;
        kk += 16;
    }

    while kk + 8 <= k_dim {
        let x0 = x[kk];
        let x1 = x[kk + 1];
        let x2 = x[kk + 2];
        let x3 = x[kk + 3];
        let x4 = x[kk + 4];
        let x5 = x[kk + 5];
        let x6 = x[kk + 6];
        let x7 = x[kk + 7];

        a0 += row0[kk] as f32 * x0 + row0[kk + 4] as f32 * x4;
        a1 += row1[kk] as f32 * x0 + row1[kk + 4] as f32 * x4;
        b0 += row0[kk + 1] as f32 * x1 + row0[kk + 5] as f32 * x5;
        b1 += row1[kk + 1] as f32 * x1 + row1[kk + 5] as f32 * x5;
        c0 += row0[kk + 2] as f32 * x2 + row0[kk + 6] as f32 * x6;
        c1 += row1[kk + 2] as f32 * x2 + row1[kk + 6] as f32 * x6;
        d0 += row0[kk + 3] as f32 * x3 + row0[kk + 7] as f32 * x7;
        d1 += row1[kk + 3] as f32 * x3 + row1[kk + 7] as f32 * x7;
        kk += 8;
    }

    while kk + 4 <= k_dim {
        let x0 = x[kk];
        let x1 = x[kk + 1];
        let x2 = x[kk + 2];
        let x3 = x[kk + 3];
        a0 += row0[kk] as f32 * x0;
        a1 += row1[kk] as f32 * x0;
        b0 += row0[kk + 1] as f32 * x1;
        b1 += row1[kk + 1] as f32 * x1;
        c0 += row0[kk + 2] as f32 * x2;
        c1 += row1[kk + 2] as f32 * x2;
        d0 += row0[kk + 3] as f32 * x3;
        d1 += row1[kk + 3] as f32 * x3;
        kk += 4;
    }

    let mut sum0 = a0 + b0 + c0 + d0 + e0 + f0 + g0 + h0;
    let mut sum1 = a1 + b1 + c1 + d1 + e1 + f1 + g1 + h1;
    while kk < k_dim {
        let xv = x[kk];
        sum0 += row0[kk] as f32 * xv;
        sum1 += row1[kk] as f32 * xv;
        kk += 1;
    }
    (sum0 * scale0, sum1 * scale1)
}

#[inline]
pub(crate) fn dot3_unrolled(
    x: &[f32],
    row0: &[f32],
    row1: &[f32],
    row2: &[f32],
) -> (f32, f32, f32) {
    if let Some(sum) = dot3_f32_arch(x, row0, row1, row2) {
        return sum;
    }

    let mut a0 = 0.0f32;
    let mut a1 = 0.0f32;
    let mut a2 = 0.0f32;
    let mut b0 = 0.0f32;
    let mut b1 = 0.0f32;
    let mut b2 = 0.0f32;
    let mut c0 = 0.0f32;
    let mut c1 = 0.0f32;
    let mut c2 = 0.0f32;
    let mut d0 = 0.0f32;
    let mut d1 = 0.0f32;
    let mut d2 = 0.0f32;
    let mut e0 = 0.0f32;
    let mut e1 = 0.0f32;
    let mut e2 = 0.0f32;
    let mut f0 = 0.0f32;
    let mut f1 = 0.0f32;
    let mut f2 = 0.0f32;
    let mut g0 = 0.0f32;
    let mut g1 = 0.0f32;
    let mut g2 = 0.0f32;
    let mut h0 = 0.0f32;
    let mut h1 = 0.0f32;
    let mut h2 = 0.0f32;
    let mut kk = 0usize;
    let k_dim = x.len();

    while kk + 16 <= k_dim {
        let x0 = x[kk];
        let x1 = x[kk + 1];
        let x2 = x[kk + 2];
        let x3 = x[kk + 3];
        let x4 = x[kk + 4];
        let x5 = x[kk + 5];
        let x6 = x[kk + 6];
        let x7 = x[kk + 7];
        let x8 = x[kk + 8];
        let x9 = x[kk + 9];
        let x10 = x[kk + 10];
        let x11 = x[kk + 11];
        let x12 = x[kk + 12];
        let x13 = x[kk + 13];
        let x14 = x[kk + 14];
        let x15 = x[kk + 15];

        a0 += row0[kk] * x0;
        a1 += row1[kk] * x0;
        a2 += row2[kk] * x0;
        b0 += row0[kk + 1] * x1;
        b1 += row1[kk + 1] * x1;
        b2 += row2[kk + 1] * x1;
        c0 += row0[kk + 2] * x2;
        c1 += row1[kk + 2] * x2;
        c2 += row2[kk + 2] * x2;
        d0 += row0[kk + 3] * x3;
        d1 += row1[kk + 3] * x3;
        d2 += row2[kk + 3] * x3;
        e0 += row0[kk + 4] * x4;
        e1 += row1[kk + 4] * x4;
        e2 += row2[kk + 4] * x4;
        f0 += row0[kk + 5] * x5;
        f1 += row1[kk + 5] * x5;
        f2 += row2[kk + 5] * x5;
        g0 += row0[kk + 6] * x6;
        g1 += row1[kk + 6] * x6;
        g2 += row2[kk + 6] * x6;
        h0 += row0[kk + 7] * x7;
        h1 += row1[kk + 7] * x7;
        h2 += row2[kk + 7] * x7;
        a0 += row0[kk + 8] * x8;
        a1 += row1[kk + 8] * x8;
        a2 += row2[kk + 8] * x8;
        b0 += row0[kk + 9] * x9;
        b1 += row1[kk + 9] * x9;
        b2 += row2[kk + 9] * x9;
        c0 += row0[kk + 10] * x10;
        c1 += row1[kk + 10] * x10;
        c2 += row2[kk + 10] * x10;
        d0 += row0[kk + 11] * x11;
        d1 += row1[kk + 11] * x11;
        d2 += row2[kk + 11] * x11;
        e0 += row0[kk + 12] * x12;
        e1 += row1[kk + 12] * x12;
        e2 += row2[kk + 12] * x12;
        f0 += row0[kk + 13] * x13;
        f1 += row1[kk + 13] * x13;
        f2 += row2[kk + 13] * x13;
        g0 += row0[kk + 14] * x14;
        g1 += row1[kk + 14] * x14;
        g2 += row2[kk + 14] * x14;
        h0 += row0[kk + 15] * x15;
        h1 += row1[kk + 15] * x15;
        h2 += row2[kk + 15] * x15;
        kk += 16;
    }

    while kk + 8 <= k_dim {
        let x0 = x[kk];
        let x1 = x[kk + 1];
        let x2 = x[kk + 2];
        let x3 = x[kk + 3];
        let x4 = x[kk + 4];
        let x5 = x[kk + 5];
        let x6 = x[kk + 6];
        let x7 = x[kk + 7];

        a0 += row0[kk] * x0 + row0[kk + 4] * x4;
        a1 += row1[kk] * x0 + row1[kk + 4] * x4;
        a2 += row2[kk] * x0 + row2[kk + 4] * x4;
        b0 += row0[kk + 1] * x1 + row0[kk + 5] * x5;
        b1 += row1[kk + 1] * x1 + row1[kk + 5] * x5;
        b2 += row2[kk + 1] * x1 + row2[kk + 5] * x5;
        c0 += row0[kk + 2] * x2 + row0[kk + 6] * x6;
        c1 += row1[kk + 2] * x2 + row1[kk + 6] * x6;
        c2 += row2[kk + 2] * x2 + row2[kk + 6] * x6;
        d0 += row0[kk + 3] * x3 + row0[kk + 7] * x7;
        d1 += row1[kk + 3] * x3 + row1[kk + 7] * x7;
        d2 += row2[kk + 3] * x3 + row2[kk + 7] * x7;
        kk += 8;
    }

    while kk + 4 <= k_dim {
        let x0 = x[kk];
        let x1 = x[kk + 1];
        let x2 = x[kk + 2];
        let x3 = x[kk + 3];
        a0 += row0[kk] * x0;
        a1 += row1[kk] * x0;
        a2 += row2[kk] * x0;
        b0 += row0[kk + 1] * x1;
        b1 += row1[kk + 1] * x1;
        b2 += row2[kk + 1] * x1;
        c0 += row0[kk + 2] * x2;
        c1 += row1[kk + 2] * x2;
        c2 += row2[kk + 2] * x2;
        d0 += row0[kk + 3] * x3;
        d1 += row1[kk + 3] * x3;
        d2 += row2[kk + 3] * x3;
        kk += 4;
    }

    let mut sum0 = a0 + b0 + c0 + d0 + e0 + f0 + g0 + h0;
    let mut sum1 = a1 + b1 + c1 + d1 + e1 + f1 + g1 + h1;
    let mut sum2 = a2 + b2 + c2 + d2 + e2 + f2 + g2 + h2;
    while kk < k_dim {
        let xv = x[kk];
        sum0 += row0[kk] * xv;
        sum1 += row1[kk] * xv;
        sum2 += row2[kk] * xv;
        kk += 1;
    }
    (sum0, sum1, sum2)
}

#[inline]
pub(crate) fn dot3_unrolled_f32_bf16(
    x: &[f32],
    row0: &[bf16],
    row1: &[bf16],
    row2: &[bf16],
) -> (f32, f32, f32) {
    if let Some(sum) = dot3_f32_bf16_arch(x, row0, row1, row2) {
        return sum;
    }

    let mut a0 = 0.0f32;
    let mut a1 = 0.0f32;
    let mut a2 = 0.0f32;
    let mut b0 = 0.0f32;
    let mut b1 = 0.0f32;
    let mut b2 = 0.0f32;
    let mut c0 = 0.0f32;
    let mut c1 = 0.0f32;
    let mut c2 = 0.0f32;
    let mut d0 = 0.0f32;
    let mut d1 = 0.0f32;
    let mut d2 = 0.0f32;
    let mut e0 = 0.0f32;
    let mut e1 = 0.0f32;
    let mut e2 = 0.0f32;
    let mut f0 = 0.0f32;
    let mut f1 = 0.0f32;
    let mut f2 = 0.0f32;
    let mut g0 = 0.0f32;
    let mut g1 = 0.0f32;
    let mut g2 = 0.0f32;
    let mut h0 = 0.0f32;
    let mut h1 = 0.0f32;
    let mut h2 = 0.0f32;
    let mut kk = 0usize;
    let k_dim = x.len();

    while kk + 16 <= k_dim {
        let x0 = x[kk];
        let x1 = x[kk + 1];
        let x2 = x[kk + 2];
        let x3 = x[kk + 3];
        let x4 = x[kk + 4];
        let x5 = x[kk + 5];
        let x6 = x[kk + 6];
        let x7 = x[kk + 7];
        let x8 = x[kk + 8];
        let x9 = x[kk + 9];
        let x10 = x[kk + 10];
        let x11 = x[kk + 11];
        let x12 = x[kk + 12];
        let x13 = x[kk + 13];
        let x14 = x[kk + 14];
        let x15 = x[kk + 15];

        a0 += row0[kk].to_f32() * x0;
        a1 += row1[kk].to_f32() * x0;
        a2 += row2[kk].to_f32() * x0;
        b0 += row0[kk + 1].to_f32() * x1;
        b1 += row1[kk + 1].to_f32() * x1;
        b2 += row2[kk + 1].to_f32() * x1;
        c0 += row0[kk + 2].to_f32() * x2;
        c1 += row1[kk + 2].to_f32() * x2;
        c2 += row2[kk + 2].to_f32() * x2;
        d0 += row0[kk + 3].to_f32() * x3;
        d1 += row1[kk + 3].to_f32() * x3;
        d2 += row2[kk + 3].to_f32() * x3;
        e0 += row0[kk + 4].to_f32() * x4;
        e1 += row1[kk + 4].to_f32() * x4;
        e2 += row2[kk + 4].to_f32() * x4;
        f0 += row0[kk + 5].to_f32() * x5;
        f1 += row1[kk + 5].to_f32() * x5;
        f2 += row2[kk + 5].to_f32() * x5;
        g0 += row0[kk + 6].to_f32() * x6;
        g1 += row1[kk + 6].to_f32() * x6;
        g2 += row2[kk + 6].to_f32() * x6;
        h0 += row0[kk + 7].to_f32() * x7;
        h1 += row1[kk + 7].to_f32() * x7;
        h2 += row2[kk + 7].to_f32() * x7;
        a0 += row0[kk + 8].to_f32() * x8;
        a1 += row1[kk + 8].to_f32() * x8;
        a2 += row2[kk + 8].to_f32() * x8;
        b0 += row0[kk + 9].to_f32() * x9;
        b1 += row1[kk + 9].to_f32() * x9;
        b2 += row2[kk + 9].to_f32() * x9;
        c0 += row0[kk + 10].to_f32() * x10;
        c1 += row1[kk + 10].to_f32() * x10;
        c2 += row2[kk + 10].to_f32() * x10;
        d0 += row0[kk + 11].to_f32() * x11;
        d1 += row1[kk + 11].to_f32() * x11;
        d2 += row2[kk + 11].to_f32() * x11;
        e0 += row0[kk + 12].to_f32() * x12;
        e1 += row1[kk + 12].to_f32() * x12;
        e2 += row2[kk + 12].to_f32() * x12;
        f0 += row0[kk + 13].to_f32() * x13;
        f1 += row1[kk + 13].to_f32() * x13;
        f2 += row2[kk + 13].to_f32() * x13;
        g0 += row0[kk + 14].to_f32() * x14;
        g1 += row1[kk + 14].to_f32() * x14;
        g2 += row2[kk + 14].to_f32() * x14;
        h0 += row0[kk + 15].to_f32() * x15;
        h1 += row1[kk + 15].to_f32() * x15;
        h2 += row2[kk + 15].to_f32() * x15;
        kk += 16;
    }

    while kk + 8 <= k_dim {
        let x0 = x[kk];
        let x1 = x[kk + 1];
        let x2 = x[kk + 2];
        let x3 = x[kk + 3];
        let x4 = x[kk + 4];
        let x5 = x[kk + 5];
        let x6 = x[kk + 6];
        let x7 = x[kk + 7];

        a0 += row0[kk].to_f32() * x0 + row0[kk + 4].to_f32() * x4;
        a1 += row1[kk].to_f32() * x0 + row1[kk + 4].to_f32() * x4;
        a2 += row2[kk].to_f32() * x0 + row2[kk + 4].to_f32() * x4;
        b0 += row0[kk + 1].to_f32() * x1 + row0[kk + 5].to_f32() * x5;
        b1 += row1[kk + 1].to_f32() * x1 + row1[kk + 5].to_f32() * x5;
        b2 += row2[kk + 1].to_f32() * x1 + row2[kk + 5].to_f32() * x5;
        c0 += row0[kk + 2].to_f32() * x2 + row0[kk + 6].to_f32() * x6;
        c1 += row1[kk + 2].to_f32() * x2 + row1[kk + 6].to_f32() * x6;
        c2 += row2[kk + 2].to_f32() * x2 + row2[kk + 6].to_f32() * x6;
        d0 += row0[kk + 3].to_f32() * x3 + row0[kk + 7].to_f32() * x7;
        d1 += row1[kk + 3].to_f32() * x3 + row1[kk + 7].to_f32() * x7;
        d2 += row2[kk + 3].to_f32() * x3 + row2[kk + 7].to_f32() * x7;
        kk += 8;
    }

    while kk + 4 <= k_dim {
        let x0 = x[kk];
        let x1 = x[kk + 1];
        let x2 = x[kk + 2];
        let x3 = x[kk + 3];
        a0 += row0[kk].to_f32() * x0;
        a1 += row1[kk].to_f32() * x0;
        a2 += row2[kk].to_f32() * x0;
        b0 += row0[kk + 1].to_f32() * x1;
        b1 += row1[kk + 1].to_f32() * x1;
        b2 += row2[kk + 1].to_f32() * x1;
        c0 += row0[kk + 2].to_f32() * x2;
        c1 += row1[kk + 2].to_f32() * x2;
        c2 += row2[kk + 2].to_f32() * x2;
        d0 += row0[kk + 3].to_f32() * x3;
        d1 += row1[kk + 3].to_f32() * x3;
        d2 += row2[kk + 3].to_f32() * x3;
        kk += 4;
    }

    let mut sum0 = a0 + b0 + c0 + d0 + e0 + f0 + g0 + h0;
    let mut sum1 = a1 + b1 + c1 + d1 + e1 + f1 + g1 + h1;
    let mut sum2 = a2 + b2 + c2 + d2 + e2 + f2 + g2 + h2;
    while kk < k_dim {
        let xv = x[kk];
        sum0 += row0[kk].to_f32() * xv;
        sum1 += row1[kk].to_f32() * xv;
        sum2 += row2[kk].to_f32() * xv;
        kk += 1;
    }
    (sum0, sum1, sum2)
}

#[inline]
pub(crate) fn dot3_unrolled_f32_f16(
    x: &[f32],
    row0: &[f16],
    row1: &[f16],
    row2: &[f16],
) -> (f32, f32, f32) {
    if let Some(sum) = dot3_f32_f16_arch(x, row0, row1, row2) {
        return sum;
    }

    let mut a0 = 0.0f32;
    let mut a1 = 0.0f32;
    let mut a2 = 0.0f32;
    let mut b0 = 0.0f32;
    let mut b1 = 0.0f32;
    let mut b2 = 0.0f32;
    let mut c0 = 0.0f32;
    let mut c1 = 0.0f32;
    let mut c2 = 0.0f32;
    let mut d0 = 0.0f32;
    let mut d1 = 0.0f32;
    let mut d2 = 0.0f32;
    let mut kk = 0usize;
    let k_dim = x.len();

    while kk + 8 <= k_dim {
        let x0 = x[kk];
        let x1 = x[kk + 1];
        let x2 = x[kk + 2];
        let x3 = x[kk + 3];
        let x4 = x[kk + 4];
        let x5 = x[kk + 5];
        let x6 = x[kk + 6];
        let x7 = x[kk + 7];

        a0 += row0[kk].to_f32() * x0 + row0[kk + 4].to_f32() * x4;
        a1 += row1[kk].to_f32() * x0 + row1[kk + 4].to_f32() * x4;
        a2 += row2[kk].to_f32() * x0 + row2[kk + 4].to_f32() * x4;
        b0 += row0[kk + 1].to_f32() * x1 + row0[kk + 5].to_f32() * x5;
        b1 += row1[kk + 1].to_f32() * x1 + row1[kk + 5].to_f32() * x5;
        b2 += row2[kk + 1].to_f32() * x1 + row2[kk + 5].to_f32() * x5;
        c0 += row0[kk + 2].to_f32() * x2 + row0[kk + 6].to_f32() * x6;
        c1 += row1[kk + 2].to_f32() * x2 + row1[kk + 6].to_f32() * x6;
        c2 += row2[kk + 2].to_f32() * x2 + row2[kk + 6].to_f32() * x6;
        d0 += row0[kk + 3].to_f32() * x3 + row0[kk + 7].to_f32() * x7;
        d1 += row1[kk + 3].to_f32() * x3 + row1[kk + 7].to_f32() * x7;
        d2 += row2[kk + 3].to_f32() * x3 + row2[kk + 7].to_f32() * x7;
        kk += 8;
    }

    while kk + 4 <= k_dim {
        let x0 = x[kk];
        let x1 = x[kk + 1];
        let x2 = x[kk + 2];
        let x3 = x[kk + 3];
        a0 += row0[kk].to_f32() * x0;
        a1 += row1[kk].to_f32() * x0;
        a2 += row2[kk].to_f32() * x0;
        b0 += row0[kk + 1].to_f32() * x1;
        b1 += row1[kk + 1].to_f32() * x1;
        b2 += row2[kk + 1].to_f32() * x1;
        c0 += row0[kk + 2].to_f32() * x2;
        c1 += row1[kk + 2].to_f32() * x2;
        c2 += row2[kk + 2].to_f32() * x2;
        d0 += row0[kk + 3].to_f32() * x3;
        d1 += row1[kk + 3].to_f32() * x3;
        d2 += row2[kk + 3].to_f32() * x3;
        kk += 4;
    }

    let mut sum0 = a0 + b0 + c0 + d0;
    let mut sum1 = a1 + b1 + c1 + d1;
    let mut sum2 = a2 + b2 + c2 + d2;
    while kk < k_dim {
        let xv = x[kk];
        sum0 += row0[kk].to_f32() * xv;
        sum1 += row1[kk].to_f32() * xv;
        sum2 += row2[kk].to_f32() * xv;
        kk += 1;
    }
    (sum0, sum1, sum2)
}

#[inline]
fn dot3_unrolled_f32_i8_portable(
    x: &[f32],
    row0: &[i8],
    scale0: f32,
    row1: &[i8],
    scale1: f32,
    row2: &[i8],
    scale2: f32,
) -> (f32, f32, f32) {
    let mut a0 = 0.0f32;
    let mut a1 = 0.0f32;
    let mut a2 = 0.0f32;
    let mut b0 = 0.0f32;
    let mut b1 = 0.0f32;
    let mut b2 = 0.0f32;
    let mut c0 = 0.0f32;
    let mut c1 = 0.0f32;
    let mut c2 = 0.0f32;
    let mut d0 = 0.0f32;
    let mut d1 = 0.0f32;
    let mut d2 = 0.0f32;
    let mut e0 = 0.0f32;
    let mut e1 = 0.0f32;
    let mut e2 = 0.0f32;
    let mut f0 = 0.0f32;
    let mut f1 = 0.0f32;
    let mut f2 = 0.0f32;
    let mut g0 = 0.0f32;
    let mut g1 = 0.0f32;
    let mut g2 = 0.0f32;
    let mut h0 = 0.0f32;
    let mut h1 = 0.0f32;
    let mut h2 = 0.0f32;
    let mut kk = 0usize;
    let k_dim = x.len();

    while kk + 16 <= k_dim {
        let x0 = x[kk];
        let x1 = x[kk + 1];
        let x2 = x[kk + 2];
        let x3 = x[kk + 3];
        let x4 = x[kk + 4];
        let x5 = x[kk + 5];
        let x6 = x[kk + 6];
        let x7 = x[kk + 7];
        let x8 = x[kk + 8];
        let x9 = x[kk + 9];
        let x10 = x[kk + 10];
        let x11 = x[kk + 11];
        let x12 = x[kk + 12];
        let x13 = x[kk + 13];
        let x14 = x[kk + 14];
        let x15 = x[kk + 15];

        a0 += row0[kk] as f32 * x0;
        a1 += row1[kk] as f32 * x0;
        a2 += row2[kk] as f32 * x0;
        b0 += row0[kk + 1] as f32 * x1;
        b1 += row1[kk + 1] as f32 * x1;
        b2 += row2[kk + 1] as f32 * x1;
        c0 += row0[kk + 2] as f32 * x2;
        c1 += row1[kk + 2] as f32 * x2;
        c2 += row2[kk + 2] as f32 * x2;
        d0 += row0[kk + 3] as f32 * x3;
        d1 += row1[kk + 3] as f32 * x3;
        d2 += row2[kk + 3] as f32 * x3;
        e0 += row0[kk + 4] as f32 * x4;
        e1 += row1[kk + 4] as f32 * x4;
        e2 += row2[kk + 4] as f32 * x4;
        f0 += row0[kk + 5] as f32 * x5;
        f1 += row1[kk + 5] as f32 * x5;
        f2 += row2[kk + 5] as f32 * x5;
        g0 += row0[kk + 6] as f32 * x6;
        g1 += row1[kk + 6] as f32 * x6;
        g2 += row2[kk + 6] as f32 * x6;
        h0 += row0[kk + 7] as f32 * x7;
        h1 += row1[kk + 7] as f32 * x7;
        h2 += row2[kk + 7] as f32 * x7;
        a0 += row0[kk + 8] as f32 * x8;
        a1 += row1[kk + 8] as f32 * x8;
        a2 += row2[kk + 8] as f32 * x8;
        b0 += row0[kk + 9] as f32 * x9;
        b1 += row1[kk + 9] as f32 * x9;
        b2 += row2[kk + 9] as f32 * x9;
        c0 += row0[kk + 10] as f32 * x10;
        c1 += row1[kk + 10] as f32 * x10;
        c2 += row2[kk + 10] as f32 * x10;
        d0 += row0[kk + 11] as f32 * x11;
        d1 += row1[kk + 11] as f32 * x11;
        d2 += row2[kk + 11] as f32 * x11;
        e0 += row0[kk + 12] as f32 * x12;
        e1 += row1[kk + 12] as f32 * x12;
        e2 += row2[kk + 12] as f32 * x12;
        f0 += row0[kk + 13] as f32 * x13;
        f1 += row1[kk + 13] as f32 * x13;
        f2 += row2[kk + 13] as f32 * x13;
        g0 += row0[kk + 14] as f32 * x14;
        g1 += row1[kk + 14] as f32 * x14;
        g2 += row2[kk + 14] as f32 * x14;
        h0 += row0[kk + 15] as f32 * x15;
        h1 += row1[kk + 15] as f32 * x15;
        h2 += row2[kk + 15] as f32 * x15;
        kk += 16;
    }

    while kk + 8 <= k_dim {
        let x0 = x[kk];
        let x1 = x[kk + 1];
        let x2 = x[kk + 2];
        let x3 = x[kk + 3];
        let x4 = x[kk + 4];
        let x5 = x[kk + 5];
        let x6 = x[kk + 6];
        let x7 = x[kk + 7];

        a0 += row0[kk] as f32 * x0 + row0[kk + 4] as f32 * x4;
        a1 += row1[kk] as f32 * x0 + row1[kk + 4] as f32 * x4;
        a2 += row2[kk] as f32 * x0 + row2[kk + 4] as f32 * x4;
        b0 += row0[kk + 1] as f32 * x1 + row0[kk + 5] as f32 * x5;
        b1 += row1[kk + 1] as f32 * x1 + row1[kk + 5] as f32 * x5;
        b2 += row2[kk + 1] as f32 * x1 + row2[kk + 5] as f32 * x5;
        c0 += row0[kk + 2] as f32 * x2 + row0[kk + 6] as f32 * x6;
        c1 += row1[kk + 2] as f32 * x2 + row1[kk + 6] as f32 * x6;
        c2 += row2[kk + 2] as f32 * x2 + row2[kk + 6] as f32 * x6;
        d0 += row0[kk + 3] as f32 * x3 + row0[kk + 7] as f32 * x7;
        d1 += row1[kk + 3] as f32 * x3 + row1[kk + 7] as f32 * x7;
        d2 += row2[kk + 3] as f32 * x3 + row2[kk + 7] as f32 * x7;
        kk += 8;
    }

    while kk + 4 <= k_dim {
        let x0 = x[kk];
        let x1 = x[kk + 1];
        let x2 = x[kk + 2];
        let x3 = x[kk + 3];
        a0 += row0[kk] as f32 * x0;
        a1 += row1[kk] as f32 * x0;
        a2 += row2[kk] as f32 * x0;
        b0 += row0[kk + 1] as f32 * x1;
        b1 += row1[kk + 1] as f32 * x1;
        b2 += row2[kk + 1] as f32 * x1;
        c0 += row0[kk + 2] as f32 * x2;
        c1 += row1[kk + 2] as f32 * x2;
        c2 += row2[kk + 2] as f32 * x2;
        d0 += row0[kk + 3] as f32 * x3;
        d1 += row1[kk + 3] as f32 * x3;
        d2 += row2[kk + 3] as f32 * x3;
        kk += 4;
    }

    let mut sum0 = a0 + b0 + c0 + d0 + e0 + f0 + g0 + h0;
    let mut sum1 = a1 + b1 + c1 + d1 + e1 + f1 + g1 + h1;
    let mut sum2 = a2 + b2 + c2 + d2 + e2 + f2 + g2 + h2;
    while kk < k_dim {
        let xv = x[kk];
        sum0 += row0[kk] as f32 * xv;
        sum1 += row1[kk] as f32 * xv;
        sum2 += row2[kk] as f32 * xv;
        kk += 1;
    }
    (sum0 * scale0, sum1 * scale1, sum2 * scale2)
}

#[inline]
pub(crate) fn dot3_unrolled_f32_i8(
    x: &[f32],
    row0: &[i8],
    scale0: f32,
    row1: &[i8],
    scale1: f32,
    row2: &[i8],
    scale2: f32,
) -> (f32, f32, f32) {
    if let Some(sum) = dot3_f32_i8_arch(x, row0, scale0, row1, scale1, row2, scale2) {
        sum
    } else {
        dot3_unrolled_f32_i8_portable(x, row0, scale0, row1, scale1, row2, scale2)
    }
}

#[inline]
fn dot3_rows_unrolled_f32_i8(
    x: &[f32],
    weights: I8QkvBlock<'_>,
    k_dim: usize,
    rows: usize,
    out: QkvOutMut<'_>,
) {
    let I8QkvBlock { q, k, v } = weights;
    let q_block = q.values;
    let k_block = k.values;
    let v_block = v.values;
    let q_scale = q.scale;
    let k_scale = k.scale;
    let v_scale = v.scale;
    let QkvOutMut {
        q: q_out,
        k: k_out,
        v: v_out,
    } = out;
    debug_assert!(rows <= 4);
    debug_assert_eq!(q_out.len(), rows);
    debug_assert_eq!(k_out.len(), rows);
    debug_assert_eq!(v_out.len(), rows);

    let mut q_acc = [0.0f32; 4];
    let mut k_acc = [0.0f32; 4];
    let mut v_acc = [0.0f32; 4];
    let mut kk = 0usize;

    while kk + 16 <= k_dim {
        let x0 = x[kk];
        let x1 = x[kk + 1];
        let x2 = x[kk + 2];
        let x3 = x[kk + 3];
        let x4 = x[kk + 4];
        let x5 = x[kk + 5];
        let x6 = x[kk + 6];
        let x7 = x[kk + 7];
        let x8 = x[kk + 8];
        let x9 = x[kk + 9];
        let x10 = x[kk + 10];
        let x11 = x[kk + 11];
        let x12 = x[kk + 12];
        let x13 = x[kk + 13];
        let x14 = x[kk + 14];
        let x15 = x[kk + 15];
        for r in 0..rows {
            let base = r * k_dim + kk;
            q_acc[r] += q_block[base] as f32 * x0
                + q_block[base + 1] as f32 * x1
                + q_block[base + 2] as f32 * x2
                + q_block[base + 3] as f32 * x3
                + q_block[base + 4] as f32 * x4
                + q_block[base + 5] as f32 * x5
                + q_block[base + 6] as f32 * x6
                + q_block[base + 7] as f32 * x7
                + q_block[base + 8] as f32 * x8
                + q_block[base + 9] as f32 * x9
                + q_block[base + 10] as f32 * x10
                + q_block[base + 11] as f32 * x11
                + q_block[base + 12] as f32 * x12
                + q_block[base + 13] as f32 * x13
                + q_block[base + 14] as f32 * x14
                + q_block[base + 15] as f32 * x15;
            k_acc[r] += k_block[base] as f32 * x0
                + k_block[base + 1] as f32 * x1
                + k_block[base + 2] as f32 * x2
                + k_block[base + 3] as f32 * x3
                + k_block[base + 4] as f32 * x4
                + k_block[base + 5] as f32 * x5
                + k_block[base + 6] as f32 * x6
                + k_block[base + 7] as f32 * x7
                + k_block[base + 8] as f32 * x8
                + k_block[base + 9] as f32 * x9
                + k_block[base + 10] as f32 * x10
                + k_block[base + 11] as f32 * x11
                + k_block[base + 12] as f32 * x12
                + k_block[base + 13] as f32 * x13
                + k_block[base + 14] as f32 * x14
                + k_block[base + 15] as f32 * x15;
            v_acc[r] += v_block[base] as f32 * x0
                + v_block[base + 1] as f32 * x1
                + v_block[base + 2] as f32 * x2
                + v_block[base + 3] as f32 * x3
                + v_block[base + 4] as f32 * x4
                + v_block[base + 5] as f32 * x5
                + v_block[base + 6] as f32 * x6
                + v_block[base + 7] as f32 * x7
                + v_block[base + 8] as f32 * x8
                + v_block[base + 9] as f32 * x9
                + v_block[base + 10] as f32 * x10
                + v_block[base + 11] as f32 * x11
                + v_block[base + 12] as f32 * x12
                + v_block[base + 13] as f32 * x13
                + v_block[base + 14] as f32 * x14
                + v_block[base + 15] as f32 * x15;
        }
        kk += 16;
    }

    while kk + 8 <= k_dim {
        let x0 = x[kk];
        let x1 = x[kk + 1];
        let x2 = x[kk + 2];
        let x3 = x[kk + 3];
        let x4 = x[kk + 4];
        let x5 = x[kk + 5];
        let x6 = x[kk + 6];
        let x7 = x[kk + 7];
        for r in 0..rows {
            let base = r * k_dim + kk;
            q_acc[r] += q_block[base] as f32 * x0
                + q_block[base + 1] as f32 * x1
                + q_block[base + 2] as f32 * x2
                + q_block[base + 3] as f32 * x3
                + q_block[base + 4] as f32 * x4
                + q_block[base + 5] as f32 * x5
                + q_block[base + 6] as f32 * x6
                + q_block[base + 7] as f32 * x7;
            k_acc[r] += k_block[base] as f32 * x0
                + k_block[base + 1] as f32 * x1
                + k_block[base + 2] as f32 * x2
                + k_block[base + 3] as f32 * x3
                + k_block[base + 4] as f32 * x4
                + k_block[base + 5] as f32 * x5
                + k_block[base + 6] as f32 * x6
                + k_block[base + 7] as f32 * x7;
            v_acc[r] += v_block[base] as f32 * x0
                + v_block[base + 1] as f32 * x1
                + v_block[base + 2] as f32 * x2
                + v_block[base + 3] as f32 * x3
                + v_block[base + 4] as f32 * x4
                + v_block[base + 5] as f32 * x5
                + v_block[base + 6] as f32 * x6
                + v_block[base + 7] as f32 * x7;
        }
        kk += 8;
    }

    while kk < k_dim {
        let xv = x[kk];
        for r in 0..rows {
            let base = r * k_dim + kk;
            q_acc[r] += q_block[base] as f32 * xv;
            k_acc[r] += k_block[base] as f32 * xv;
            v_acc[r] += v_block[base] as f32 * xv;
        }
        kk += 1;
    }

    for r in 0..rows {
        q_out[r] = q_acc[r] * q_scale;
        k_out[r] = k_acc[r] * k_scale;
        v_out[r] = v_acc[r] * v_scale;
    }
}

#[inline]
fn row_slice(rowmajor: SliceRef<'_>, row_idx: usize, k_dim: usize) -> SliceRef<'_> {
    let start = row_idx * k_dim;
    let end = start + k_dim;
    match rowmajor {
        SliceRef::F32(w) => SliceRef::F32(&w[start..end]),
        SliceRef::F16(w) => SliceRef::F16(&w[start..end]),
        SliceRef::BF16(w) => SliceRef::BF16(&w[start..end]),
        SliceRef::I8(w, scale) => SliceRef::I8(&w[start..end], scale),
    }
}

#[inline]
fn dot_unrolled_from_slice(x: &[f32], row: SliceRef<'_>) -> f32 {
    match row {
        SliceRef::F32(row) => dot_unrolled(x, row),
        SliceRef::F16(row) => dot_unrolled_f32_f16(x, row),
        SliceRef::BF16(row) => dot_unrolled_f32_bf16(x, row),
        SliceRef::I8(row, scale) => dot_unrolled_f32_i8(x, row, scale),
    }
}

#[inline]
pub fn dual_matvec_rowmajor_parallel(
    x: &[f32],
    w0_rowmajor: &[f32],
    w1_rowmajor: &[f32],
    n_rows: usize,
    k_dim: usize,
    out0: &mut [f32],
    out1: &mut [f32],
) {
    assert_eq!(x.len(), k_dim, "x len / k_dim mismatch");
    assert_eq!(w0_rowmajor.len(), n_rows * k_dim, "weight0 size mismatch");
    assert_eq!(w1_rowmajor.len(), n_rows * k_dim, "weight1 size mismatch");
    assert_eq!(out0.len(), n_rows, "out0 size mismatch");
    assert_eq!(out1.len(), n_rows, "out1 size mismatch");

    if n_rows < MATVEC_PAR_THRESHOLD {
        for i in 0..n_rows {
            let row0 = &w0_rowmajor[i * k_dim..(i + 1) * k_dim];
            let row1 = &w1_rowmajor[i * k_dim..(i + 1) * k_dim];
            let (s0, s1) = dot2_unrolled(x, row0, row1);
            out0[i] = s0;
            out1[i] = s1;
        }
    } else {
        out0.par_iter_mut()
            .zip(out1.par_iter_mut())
            .enumerate()
            .for_each(|(i, (dst0, dst1))| {
                let row0 = &w0_rowmajor[i * k_dim..(i + 1) * k_dim];
                let row1 = &w1_rowmajor[i * k_dim..(i + 1) * k_dim];
                let (s0, s1) = dot2_unrolled(x, row0, row1);
                *dst0 = s0;
                *dst1 = s1;
            });
    }
}

#[inline]
pub(crate) fn dual_matvec_rowmajor_parallel_f32_bf16(
    x: &[f32],
    w0_rowmajor: &[bf16],
    w1_rowmajor: &[bf16],
    n_rows: usize,
    k_dim: usize,
    out0: &mut [f32],
    out1: &mut [f32],
) {
    assert_eq!(x.len(), k_dim, "x len / k_dim mismatch");
    assert_eq!(w0_rowmajor.len(), n_rows * k_dim, "weight0 size mismatch");
    assert_eq!(w1_rowmajor.len(), n_rows * k_dim, "weight1 size mismatch");
    assert_eq!(out0.len(), n_rows, "out0 size mismatch");
    assert_eq!(out1.len(), n_rows, "out1 size mismatch");

    if n_rows < MATVEC_PAR_THRESHOLD {
        for i in 0..n_rows {
            let row0 = &w0_rowmajor[i * k_dim..(i + 1) * k_dim];
            let row1 = &w1_rowmajor[i * k_dim..(i + 1) * k_dim];
            let (s0, s1) = dot2_unrolled_f32_bf16(x, row0, row1);
            out0[i] = s0;
            out1[i] = s1;
        }
    } else if should_use_mixed_dual_block_kernel(n_rows) {
        dual_matvec_rowmajor_block_parallel_f32_bf16(
            x,
            w0_rowmajor,
            w1_rowmajor,
            n_rows,
            k_dim,
            out0,
            out1,
        );
    } else {
        out0.par_chunks_mut(MIXED_ROW_PAR_CHUNK_ROWS)
            .zip(out1.par_chunks_mut(MIXED_ROW_PAR_CHUNK_ROWS))
            .enumerate()
            .for_each(|(chunk_idx, (out0_chunk, out1_chunk))| {
                let row_start = chunk_idx * MIXED_ROW_PAR_CHUNK_ROWS;
                for (offset, (dst0, dst1)) in
                    out0_chunk.iter_mut().zip(out1_chunk.iter_mut()).enumerate()
                {
                    let row_idx = row_start + offset;
                    let row0 = &w0_rowmajor[row_idx * k_dim..(row_idx + 1) * k_dim];
                    let row1 = &w1_rowmajor[row_idx * k_dim..(row_idx + 1) * k_dim];
                    let (s0, s1) = dot2_unrolled_f32_bf16(x, row0, row1);
                    *dst0 = s0;
                    *dst1 = s1;
                }
            });
    }
}

#[inline]
pub(crate) fn dual_matvec_rowmajor_parallel_f32_f16(
    x: &[f32],
    w0_rowmajor: &[f16],
    w1_rowmajor: &[f16],
    n_rows: usize,
    k_dim: usize,
    out0: &mut [f32],
    out1: &mut [f32],
) {
    assert_eq!(x.len(), k_dim, "x len / k_dim mismatch");
    assert_eq!(w0_rowmajor.len(), n_rows * k_dim, "weight0 size mismatch");
    assert_eq!(w1_rowmajor.len(), n_rows * k_dim, "weight1 size mismatch");
    assert_eq!(out0.len(), n_rows, "out0 size mismatch");
    assert_eq!(out1.len(), n_rows, "out1 size mismatch");

    if n_rows < MATVEC_PAR_THRESHOLD {
        for i in 0..n_rows {
            let row0 = &w0_rowmajor[i * k_dim..(i + 1) * k_dim];
            let row1 = &w1_rowmajor[i * k_dim..(i + 1) * k_dim];
            let (s0, s1) = dot2_unrolled_f32_f16(x, row0, row1);
            out0[i] = s0;
            out1[i] = s1;
        }
    } else {
        out0.par_chunks_mut(MIXED_ROW_PAR_CHUNK_ROWS)
            .zip(out1.par_chunks_mut(MIXED_ROW_PAR_CHUNK_ROWS))
            .enumerate()
            .for_each(|(chunk_idx, (out0_chunk, out1_chunk))| {
                let row_start = chunk_idx * MIXED_ROW_PAR_CHUNK_ROWS;
                for (offset, (dst0, dst1)) in
                    out0_chunk.iter_mut().zip(out1_chunk.iter_mut()).enumerate()
                {
                    let row_idx = row_start + offset;
                    let row0 = &w0_rowmajor[row_idx * k_dim..(row_idx + 1) * k_dim];
                    let row1 = &w1_rowmajor[row_idx * k_dim..(row_idx + 1) * k_dim];
                    let (s0, s1) = dot2_unrolled_f32_f16(x, row0, row1);
                    *dst0 = s0;
                    *dst1 = s1;
                }
            });
    }
}

#[inline]
pub(crate) fn dual_matvec_rowmajor_parallel_f32_i8(
    x: &[f32],
    w0_rowmajor: &[i8],
    w0_scale: f32,
    w1_rowmajor: &[i8],
    w1_scale: f32,
    n_rows: usize,
    k_dim: usize,
    out0: &mut [f32],
    out1: &mut [f32],
) {
    assert_eq!(x.len(), k_dim, "x len / k_dim mismatch");
    assert_eq!(w0_rowmajor.len(), n_rows * k_dim, "weight0 size mismatch");
    assert_eq!(w1_rowmajor.len(), n_rows * k_dim, "weight1 size mismatch");
    assert_eq!(out0.len(), n_rows, "out0 size mismatch");
    assert_eq!(out1.len(), n_rows, "out1 size mismatch");

    if n_rows < MATVEC_PAR_THRESHOLD {
        for i in 0..n_rows {
            let row0 = &w0_rowmajor[i * k_dim..(i + 1) * k_dim];
            let row1 = &w1_rowmajor[i * k_dim..(i + 1) * k_dim];
            let (s0, s1) = dot2_unrolled_f32_i8(x, row0, w0_scale, row1, w1_scale);
            out0[i] = s0;
            out1[i] = s1;
        }
    } else if should_use_i8_block_kernel(n_rows, k_dim) {
        dual_matvec_rowmajor_block_parallel_f32_i8(
            x,
            I8DualBlock {
                left: I8ScaledSlice {
                    values: w0_rowmajor,
                    scale: w0_scale,
                },
                right: I8ScaledSlice {
                    values: w1_rowmajor,
                    scale: w1_scale,
                },
            },
            k_dim,
            DualOutMut {
                left: out0,
                right: out1,
            },
        );
    } else {
        out0.par_chunks_mut(MATVEC_I8_PAR_CHUNK_ROWS)
            .zip(out1.par_chunks_mut(MATVEC_I8_PAR_CHUNK_ROWS))
            .enumerate()
            .for_each(|(chunk_idx, (out0_chunk, out1_chunk))| {
                let row_start = chunk_idx * MATVEC_I8_PAR_CHUNK_ROWS;
                for (offset, (dst0, dst1)) in
                    out0_chunk.iter_mut().zip(out1_chunk.iter_mut()).enumerate()
                {
                    let row_idx = row_start + offset;
                    let row0 = &w0_rowmajor[row_idx * k_dim..(row_idx + 1) * k_dim];
                    let row1 = &w1_rowmajor[row_idx * k_dim..(row_idx + 1) * k_dim];
                    let (s0, s1) = dot2_unrolled_f32_i8(x, row0, w0_scale, row1, w1_scale);
                    *dst0 = s0;
                    *dst1 = s1;
                }
            });
    }
}

#[inline]
fn dual_matvec_rowmajor_block_parallel_f32_bf16(
    x: &[f32],
    w0_rowmajor: &[bf16],
    w1_rowmajor: &[bf16],
    _n_rows: usize,
    k_dim: usize,
    out0: &mut [f32],
    out1: &mut [f32],
) {
    out0.par_chunks_mut(MATVEC_BLOCK_ROWS)
        .zip(out1.par_chunks_mut(MATVEC_BLOCK_ROWS))
        .enumerate()
        .for_each(|(block_idx, (out0_chunk, out1_chunk))| {
            let row_start = block_idx * MATVEC_BLOCK_ROWS;
            let rows = out0_chunk.len();
            let w0_block = &w0_rowmajor[row_start * k_dim..(row_start + rows) * k_dim];
            let w1_block = &w1_rowmajor[row_start * k_dim..(row_start + rows) * k_dim];
            let mut acc0 = [0.0f32; MATVEC_BLOCK_ROWS];
            let mut acc1 = [0.0f32; MATVEC_BLOCK_ROWS];

            let mut kk = 0usize;
            while kk + 8 <= k_dim {
                let x0 = x[kk];
                let x1 = x[kk + 1];
                let x2 = x[kk + 2];
                let x3 = x[kk + 3];
                let x4 = x[kk + 4];
                let x5 = x[kk + 5];
                let x6 = x[kk + 6];
                let x7 = x[kk + 7];
                for r in 0..rows {
                    let base = r * k_dim + kk;
                    acc0[r] += w0_block[base].to_f32() * x0
                        + w0_block[base + 1].to_f32() * x1
                        + w0_block[base + 2].to_f32() * x2
                        + w0_block[base + 3].to_f32() * x3
                        + w0_block[base + 4].to_f32() * x4
                        + w0_block[base + 5].to_f32() * x5
                        + w0_block[base + 6].to_f32() * x6
                        + w0_block[base + 7].to_f32() * x7;
                    acc1[r] += w1_block[base].to_f32() * x0
                        + w1_block[base + 1].to_f32() * x1
                        + w1_block[base + 2].to_f32() * x2
                        + w1_block[base + 3].to_f32() * x3
                        + w1_block[base + 4].to_f32() * x4
                        + w1_block[base + 5].to_f32() * x5
                        + w1_block[base + 6].to_f32() * x6
                        + w1_block[base + 7].to_f32() * x7;
                }
                kk += 8;
            }

            while kk < k_dim {
                let xv = x[kk];
                for r in 0..rows {
                    let base = r * k_dim + kk;
                    acc0[r] += w0_block[base].to_f32() * xv;
                    acc1[r] += w1_block[base].to_f32() * xv;
                }
                kk += 1;
            }

            out0_chunk.copy_from_slice(&acc0[..rows]);
            out1_chunk.copy_from_slice(&acc1[..rows]);
        });
}

fn dual_matvec_rowmajor_block_parallel_f32_i8(
    x: &[f32],
    weights: I8DualBlock<'_>,
    k_dim: usize,
    out: DualOutMut<'_>,
) {
    let w0_rowmajor = weights.left.values;
    let w0_scale = weights.left.scale;
    let w1_rowmajor = weights.right.values;
    let w1_scale = weights.right.scale;
    let out0 = out.left;
    let out1 = out.right;

    out0.par_chunks_mut(MATVEC_BLOCK_ROWS)
        .zip(out1.par_chunks_mut(MATVEC_BLOCK_ROWS))
        .enumerate()
        .for_each(|(block_idx, (out0_chunk, out1_chunk))| {
            let row_start = block_idx * MATVEC_BLOCK_ROWS;
            let rows = out0_chunk.len();
            let w0_block = &w0_rowmajor[row_start * k_dim..(row_start + rows) * k_dim];
            let w1_block = &w1_rowmajor[row_start * k_dim..(row_start + rows) * k_dim];
            let mut acc0 = [0.0f32; MATVEC_BLOCK_ROWS];
            let mut acc1 = [0.0f32; MATVEC_BLOCK_ROWS];

            let mut kk = 0usize;
            while kk + 8 <= k_dim {
                let x0 = x[kk];
                let x1 = x[kk + 1];
                let x2 = x[kk + 2];
                let x3 = x[kk + 3];
                let x4 = x[kk + 4];
                let x5 = x[kk + 5];
                let x6 = x[kk + 6];
                let x7 = x[kk + 7];
                for r in 0..rows {
                    let base = r * k_dim + kk;
                    acc0[r] += w0_block[base] as f32 * x0
                        + w0_block[base + 1] as f32 * x1
                        + w0_block[base + 2] as f32 * x2
                        + w0_block[base + 3] as f32 * x3
                        + w0_block[base + 4] as f32 * x4
                        + w0_block[base + 5] as f32 * x5
                        + w0_block[base + 6] as f32 * x6
                        + w0_block[base + 7] as f32 * x7;
                    acc1[r] += w1_block[base] as f32 * x0
                        + w1_block[base + 1] as f32 * x1
                        + w1_block[base + 2] as f32 * x2
                        + w1_block[base + 3] as f32 * x3
                        + w1_block[base + 4] as f32 * x4
                        + w1_block[base + 5] as f32 * x5
                        + w1_block[base + 6] as f32 * x6
                        + w1_block[base + 7] as f32 * x7;
                }
                kk += 8;
            }

            while kk < k_dim {
                let xv = x[kk];
                for r in 0..rows {
                    let base = r * k_dim + kk;
                    acc0[r] += w0_block[base] as f32 * xv;
                    acc1[r] += w1_block[base] as f32 * xv;
                }
                kk += 1;
            }

            for r in 0..rows {
                out0_chunk[r] = acc0[r] * w0_scale;
                out1_chunk[r] = acc1[r] * w1_scale;
            }
        });
}

#[inline]
fn dual_matvec_rowmajor_parallel_bf16_f32(
    x: &[bf16],
    w0_rowmajor: &[f32],
    w1_rowmajor: &[f32],
    n_rows: usize,
    k_dim: usize,
    out0: &mut [f32],
    out1: &mut [f32],
) {
    with_bf16_input_as_f32(x, |x_f32| {
        dual_matvec_rowmajor_parallel(x_f32, w0_rowmajor, w1_rowmajor, n_rows, k_dim, out0, out1);
    });
}

#[inline]
fn dual_matvec_rowmajor_parallel_bf16_bf16(
    x: &[bf16],
    w0_rowmajor: &[bf16],
    w1_rowmajor: &[bf16],
    n_rows: usize,
    k_dim: usize,
    out0: &mut [f32],
    out1: &mut [f32],
) {
    assert_eq!(x.len(), k_dim, "x len / k_dim mismatch");
    assert_eq!(w0_rowmajor.len(), n_rows * k_dim, "weight0 size mismatch");
    assert_eq!(w1_rowmajor.len(), n_rows * k_dim, "weight1 size mismatch");
    assert_eq!(out0.len(), n_rows, "out0 size mismatch");
    assert_eq!(out1.len(), n_rows, "out1 size mismatch");

    if crate::arch::x86_avx512_bf16_kernel_runtime_available() {
        if n_rows < MATVEC_PAR_THRESHOLD {
            for i in 0..n_rows {
                let row0 = &w0_rowmajor[i * k_dim..(i + 1) * k_dim];
                let row1 = &w1_rowmajor[i * k_dim..(i + 1) * k_dim];
                let (s0, s1) = dot2_bf16_bf16_arch(x, row0, row1).unwrap_or_else(|| {
                    (
                        dot_unrolled_bf16_bf16_scalar(x, row0),
                        dot_unrolled_bf16_bf16_scalar(x, row1),
                    )
                });
                out0[i] = s0;
                out1[i] = s1;
            }
        } else {
            out0.par_iter_mut()
                .zip(out1.par_iter_mut())
                .enumerate()
                .for_each(|(i, (dst0, dst1))| {
                    let row0 = &w0_rowmajor[i * k_dim..(i + 1) * k_dim];
                    let row1 = &w1_rowmajor[i * k_dim..(i + 1) * k_dim];
                    let (s0, s1) = dot2_bf16_bf16_arch(x, row0, row1).unwrap_or_else(|| {
                        (
                            dot_unrolled_bf16_bf16_scalar(x, row0),
                            dot_unrolled_bf16_bf16_scalar(x, row1),
                        )
                    });
                    *dst0 = s0;
                    *dst1 = s1;
                });
        }
        return;
    }

    with_bf16_input_as_f32(x, |x_f32| {
        dual_matvec_rowmajor_parallel_f32_bf16(
            x_f32,
            w0_rowmajor,
            w1_rowmajor,
            n_rows,
            k_dim,
            out0,
            out1,
        );
    });
}

#[inline]
fn dual_matvec_rowmajor_parallel_f16_f16(
    x: &[f16],
    w0_rowmajor: &[f16],
    w1_rowmajor: &[f16],
    n_rows: usize,
    k_dim: usize,
    out0: &mut [f32],
    out1: &mut [f32],
) {
    assert_eq!(x.len(), k_dim, "x len / k_dim mismatch");
    assert_eq!(w0_rowmajor.len(), n_rows * k_dim, "weight0 size mismatch");
    assert_eq!(w1_rowmajor.len(), n_rows * k_dim, "weight1 size mismatch");
    assert_eq!(out0.len(), n_rows, "out0 size mismatch");
    assert_eq!(out1.len(), n_rows, "out1 size mismatch");

    if crate::arch::x86_avx512_fp16_kernel_runtime_available() {
        if n_rows < MATVEC_PAR_THRESHOLD {
            for i in 0..n_rows {
                let row0 = &w0_rowmajor[i * k_dim..(i + 1) * k_dim];
                let row1 = &w1_rowmajor[i * k_dim..(i + 1) * k_dim];
                let (s0, s1) = dot2_f16_f16_arch(x, row0, row1).unwrap_or_else(|| {
                    (
                        dot_unrolled_f16_f16_scalar(x, row0),
                        dot_unrolled_f16_f16_scalar(x, row1),
                    )
                });
                out0[i] = s0;
                out1[i] = s1;
            }
        } else {
            out0.par_iter_mut()
                .zip(out1.par_iter_mut())
                .enumerate()
                .for_each(|(i, (dst0, dst1))| {
                    let row0 = &w0_rowmajor[i * k_dim..(i + 1) * k_dim];
                    let row1 = &w1_rowmajor[i * k_dim..(i + 1) * k_dim];
                    let (s0, s1) = dot2_f16_f16_arch(x, row0, row1).unwrap_or_else(|| {
                        (
                            dot_unrolled_f16_f16_scalar(x, row0),
                            dot_unrolled_f16_f16_scalar(x, row1),
                        )
                    });
                    *dst0 = s0;
                    *dst1 = s1;
                });
        }
        return;
    }

    with_f16_input_as_f32(x, |x_f32| {
        dual_matvec_rowmajor_parallel_f32_f16(
            x_f32,
            w0_rowmajor,
            w1_rowmajor,
            n_rows,
            k_dim,
            out0,
            out1,
        );
    });
}

#[inline]
fn dual_matvec_rowmajor_parallel_bf16_i8(
    x: &[bf16],
    w0_rowmajor: &[i8],
    w0_scale: f32,
    w1_rowmajor: &[i8],
    w1_scale: f32,
    n_rows: usize,
    k_dim: usize,
    out0: &mut [f32],
    out1: &mut [f32],
) {
    with_bf16_input_as_f32(x, |x_f32| {
        dual_matvec_rowmajor_parallel_f32_i8(
            x_f32,
            w0_rowmajor,
            w0_scale,
            w1_rowmajor,
            w1_scale,
            n_rows,
            k_dim,
            out0,
            out1,
        );
    });
}

#[inline]
fn dual_matvec_rowmajor_parallel_i8_f32(
    x: &[i8],
    x_scale: f32,
    w0_rowmajor: &[f32],
    w1_rowmajor: &[f32],
    n_rows: usize,
    k_dim: usize,
    out0: &mut [f32],
    out1: &mut [f32],
) {
    with_i8_input_as_f32(x, x_scale, |x_f32| {
        dual_matvec_rowmajor_parallel(x_f32, w0_rowmajor, w1_rowmajor, n_rows, k_dim, out0, out1);
    });
}

#[inline]
fn dual_matvec_rowmajor_parallel_i8_bf16(
    x: &[i8],
    x_scale: f32,
    w0_rowmajor: &[bf16],
    w1_rowmajor: &[bf16],
    n_rows: usize,
    k_dim: usize,
    out0: &mut [f32],
    out1: &mut [f32],
) {
    with_i8_input_as_f32(x, x_scale, |x_f32| {
        dual_matvec_rowmajor_parallel_f32_bf16(
            x_f32,
            w0_rowmajor,
            w1_rowmajor,
            n_rows,
            k_dim,
            out0,
            out1,
        );
    });
}

#[inline]
fn dual_matvec_rowmajor_parallel_i8_i8(
    x: &[i8],
    x_scale: f32,
    w0_rowmajor: &[i8],
    w0_scale: f32,
    w1_rowmajor: &[i8],
    w1_scale: f32,
    n_rows: usize,
    k_dim: usize,
    out0: &mut [f32],
    out1: &mut [f32],
) {
    assert_eq!(x.len(), k_dim, "x len / k_dim mismatch");
    assert_eq!(w0_rowmajor.len(), n_rows * k_dim, "weight0 size mismatch");
    assert_eq!(w1_rowmajor.len(), n_rows * k_dim, "weight1 size mismatch");
    assert_eq!(out0.len(), n_rows, "out0 size mismatch");
    assert_eq!(out1.len(), n_rows, "out1 size mismatch");

    if n_rows < MATVEC_PAR_THRESHOLD {
        for i in 0..n_rows {
            let row0 = &w0_rowmajor[i * k_dim..(i + 1) * k_dim];
            let row1 = &w1_rowmajor[i * k_dim..(i + 1) * k_dim];
            let (s0, s1) = dot2_unrolled_i8_i8(x, x_scale, row0, w0_scale, row1, w1_scale);
            out0[i] = s0;
            out1[i] = s1;
        }
    } else {
        out0.par_chunks_mut(MATVEC_I8_PAR_CHUNK_ROWS)
            .zip(out1.par_chunks_mut(MATVEC_I8_PAR_CHUNK_ROWS))
            .enumerate()
            .for_each(|(chunk_idx, (out0_chunk, out1_chunk))| {
                let row_start = chunk_idx * MATVEC_I8_PAR_CHUNK_ROWS;
                for offset in 0..out0_chunk.len() {
                    let row_idx = row_start + offset;
                    let row0 = &w0_rowmajor[row_idx * k_dim..(row_idx + 1) * k_dim];
                    let row1 = &w1_rowmajor[row_idx * k_dim..(row_idx + 1) * k_dim];
                    let (s0, s1) = dot2_unrolled_i8_i8(x, x_scale, row0, w0_scale, row1, w1_scale);
                    out0_chunk[offset] = s0;
                    out1_chunk[offset] = s1;
                }
            });
    }
}

fn dual_matvec_rowmajor_parallel_mixed_f32_input(
    x: &[f32],
    w0_rowmajor: SliceRef<'_>,
    w1_rowmajor: SliceRef<'_>,
    n_rows: usize,
    k_dim: usize,
    out0: &mut [f32],
    out1: &mut [f32],
) {
    if n_rows < MATVEC_PAR_THRESHOLD {
        for i in 0..n_rows {
            out0[i] = dot_unrolled_from_slice(x, row_slice(w0_rowmajor, i, k_dim));
            out1[i] = dot_unrolled_from_slice(x, row_slice(w1_rowmajor, i, k_dim));
        }
    } else {
        out0.par_iter_mut()
            .zip(out1.par_iter_mut())
            .enumerate()
            .for_each(|(i, (out0_val, out1_val))| {
                *out0_val = dot_unrolled_from_slice(x, row_slice(w0_rowmajor, i, k_dim));
                *out1_val = dot_unrolled_from_slice(x, row_slice(w1_rowmajor, i, k_dim));
            });
    }
}

#[inline]
pub fn dual_matvec_rowmajor_parallel_mixed(
    x: SliceRef<'_>,
    w0_rowmajor: SliceRef<'_>,
    w1_rowmajor: SliceRef<'_>,
    n_rows: usize,
    k_dim: usize,
    out0: &mut [f32],
    out1: &mut [f32],
) {
    match (x, w0_rowmajor, w1_rowmajor) {
        (SliceRef::F32(x), SliceRef::F32(w0), SliceRef::F32(w1)) => {
            dual_matvec_rowmajor_parallel(x, w0, w1, n_rows, k_dim, out0, out1)
        }
        (SliceRef::F32(x), SliceRef::F16(w0), SliceRef::F16(w1)) => {
            dual_matvec_rowmajor_parallel_f32_f16(x, w0, w1, n_rows, k_dim, out0, out1)
        }
        (SliceRef::F32(x), SliceRef::BF16(w0), SliceRef::BF16(w1)) => {
            dual_matvec_rowmajor_parallel_f32_bf16(x, w0, w1, n_rows, k_dim, out0, out1)
        }
        (SliceRef::F32(x), SliceRef::I8(w0, s0), SliceRef::I8(w1, s1)) => {
            dual_matvec_rowmajor_parallel_f32_i8(x, w0, s0, w1, s1, n_rows, k_dim, out0, out1)
        }
        (SliceRef::F16(x), SliceRef::F32(w0), SliceRef::F32(w1)) => {
            with_f16_input_as_f32(x, |x_f32| {
                dual_matvec_rowmajor_parallel(x_f32, w0, w1, n_rows, k_dim, out0, out1)
            })
        }
        (SliceRef::F16(x), SliceRef::F16(w0), SliceRef::F16(w1)) => {
            dual_matvec_rowmajor_parallel_f16_f16(x, w0, w1, n_rows, k_dim, out0, out1)
        }
        (SliceRef::F16(x), SliceRef::BF16(w0), SliceRef::BF16(w1)) => {
            with_f16_input_as_f32(x, |x_f32| {
                dual_matvec_rowmajor_parallel_f32_bf16(x_f32, w0, w1, n_rows, k_dim, out0, out1)
            })
        }
        (SliceRef::F16(x), SliceRef::I8(w0, s0), SliceRef::I8(w1, s1)) => {
            with_f16_input_as_f32(x, |x_f32| {
                dual_matvec_rowmajor_parallel_f32_i8(
                    x_f32, w0, s0, w1, s1, n_rows, k_dim, out0, out1,
                )
            })
        }
        (SliceRef::BF16(x), SliceRef::F32(w0), SliceRef::F32(w1)) => {
            dual_matvec_rowmajor_parallel_bf16_f32(x, w0, w1, n_rows, k_dim, out0, out1)
        }
        (SliceRef::BF16(x), SliceRef::BF16(w0), SliceRef::BF16(w1)) => {
            dual_matvec_rowmajor_parallel_bf16_bf16(x, w0, w1, n_rows, k_dim, out0, out1)
        }
        (SliceRef::BF16(x), SliceRef::F16(w0), SliceRef::F16(w1)) => {
            with_bf16_input_as_f32(x, |x_f32| {
                dual_matvec_rowmajor_parallel_f32_f16(x_f32, w0, w1, n_rows, k_dim, out0, out1)
            })
        }
        (SliceRef::BF16(x), SliceRef::I8(w0, s0), SliceRef::I8(w1, s1)) => {
            dual_matvec_rowmajor_parallel_bf16_i8(x, w0, s0, w1, s1, n_rows, k_dim, out0, out1)
        }
        (SliceRef::I8(x, sx), SliceRef::F32(w0), SliceRef::F32(w1)) => {
            dual_matvec_rowmajor_parallel_i8_f32(x, sx, w0, w1, n_rows, k_dim, out0, out1)
        }
        (SliceRef::I8(x, sx), SliceRef::F16(w0), SliceRef::F16(w1)) => {
            with_i8_input_as_f32(x, sx, |x_f32| {
                dual_matvec_rowmajor_parallel_f32_f16(x_f32, w0, w1, n_rows, k_dim, out0, out1)
            })
        }
        (SliceRef::I8(x, sx), SliceRef::BF16(w0), SliceRef::BF16(w1)) => {
            dual_matvec_rowmajor_parallel_i8_bf16(x, sx, w0, w1, n_rows, k_dim, out0, out1)
        }
        (SliceRef::I8(x, sx), SliceRef::I8(w0, s0), SliceRef::I8(w1, s1)) => {
            dual_matvec_rowmajor_parallel_i8_i8(x, sx, w0, s0, w1, s1, n_rows, k_dim, out0, out1)
        }
        (SliceRef::F32(x), w0, w1) => {
            dual_matvec_rowmajor_parallel_mixed_f32_input(x, w0, w1, n_rows, k_dim, out0, out1)
        }
        (SliceRef::F16(x), w0, w1) => {
            with_f16_input_as_f32(x, |x_f32| {
                dual_matvec_rowmajor_parallel_mixed_f32_input(
                    x_f32, w0, w1, n_rows, k_dim, out0, out1,
                )
            });
        }
        (SliceRef::BF16(x), w0, w1) => {
            with_bf16_input_as_f32(x, |x_f32| {
                dual_matvec_rowmajor_parallel_mixed_f32_input(
                    x_f32, w0, w1, n_rows, k_dim, out0, out1,
                )
            });
        }
        (SliceRef::I8(x, scale), w0, w1) => {
            with_i8_input_as_f32(x, scale, |x_f32| {
                dual_matvec_rowmajor_parallel_mixed_f32_input(
                    x_f32, w0, w1, n_rows, k_dim, out0, out1,
                )
            });
        }
    }
}

#[inline]
pub(crate) fn qkv_matvec_rowmajor_parallel(
    x: &[f32],
    q_rowmajor: &[f32],
    k_rowmajor: &[f32],
    v_rowmajor: &[f32],
    q_rows: usize,
    kv_rows: usize,
    k_dim: usize,
    q_out: &mut [f32],
    k_out: &mut [f32],
    v_out: &mut [f32],
) {
    assert_eq!(x.len(), k_dim, "x len / k_dim mismatch");
    assert_eq!(q_rowmajor.len(), q_rows * k_dim, "Q weight size mismatch");
    assert_eq!(k_rowmajor.len(), kv_rows * k_dim, "K weight size mismatch");
    assert_eq!(v_rowmajor.len(), kv_rows * k_dim, "V weight size mismatch");
    assert_eq!(q_out.len(), q_rows, "Q output size mismatch");
    assert_eq!(k_out.len(), kv_rows, "K output size mismatch");
    assert_eq!(v_out.len(), kv_rows, "V output size mismatch");

    let shared_rows = q_rows.min(kv_rows);
    if shared_rows < MATVEC_PAR_THRESHOLD {
        for i in 0..shared_rows {
            let q_row = &q_rowmajor[i * k_dim..(i + 1) * k_dim];
            let k_row = &k_rowmajor[i * k_dim..(i + 1) * k_dim];
            let v_row = &v_rowmajor[i * k_dim..(i + 1) * k_dim];
            let (q, k, v) = dot3_unrolled(x, q_row, k_row, v_row);
            q_out[i] = q;
            k_out[i] = k;
            v_out[i] = v;
        }
    } else {
        q_out[..shared_rows]
            .par_iter_mut()
            .zip(k_out[..shared_rows].par_iter_mut())
            .zip(v_out[..shared_rows].par_iter_mut())
            .enumerate()
            .for_each(|(i, ((q_dst, k_dst), v_dst))| {
                let q_row = &q_rowmajor[i * k_dim..(i + 1) * k_dim];
                let k_row = &k_rowmajor[i * k_dim..(i + 1) * k_dim];
                let v_row = &v_rowmajor[i * k_dim..(i + 1) * k_dim];
                let (q, k, v) = dot3_unrolled(x, q_row, k_row, v_row);
                *q_dst = q;
                *k_dst = k;
                *v_dst = v;
            });
    }

    if q_rows > shared_rows {
        matvec_rowmajor_parallel(
            x,
            &q_rowmajor[shared_rows * k_dim..],
            q_rows - shared_rows,
            k_dim,
            &mut q_out[shared_rows..],
        );
    }
    if kv_rows > shared_rows {
        dual_matvec_rowmajor_parallel(
            x,
            &k_rowmajor[shared_rows * k_dim..],
            &v_rowmajor[shared_rows * k_dim..],
            kv_rows - shared_rows,
            k_dim,
            &mut k_out[shared_rows..],
            &mut v_out[shared_rows..],
        );
    }
}

#[inline]
pub(crate) fn qkv_matvec_rowmajor_parallel_f32_bf16(
    x: &[f32],
    q_rowmajor: &[bf16],
    k_rowmajor: &[bf16],
    v_rowmajor: &[bf16],
    q_rows: usize,
    kv_rows: usize,
    k_dim: usize,
    q_out: &mut [f32],
    k_out: &mut [f32],
    v_out: &mut [f32],
) {
    assert_eq!(x.len(), k_dim, "x len / k_dim mismatch");
    assert_eq!(q_rowmajor.len(), q_rows * k_dim, "Q weight size mismatch");
    assert_eq!(k_rowmajor.len(), kv_rows * k_dim, "K weight size mismatch");
    assert_eq!(v_rowmajor.len(), kv_rows * k_dim, "V weight size mismatch");
    assert_eq!(q_out.len(), q_rows, "Q output size mismatch");
    assert_eq!(k_out.len(), kv_rows, "K output size mismatch");
    assert_eq!(v_out.len(), kv_rows, "V output size mismatch");

    let shared_rows = q_rows.min(kv_rows);
    if shared_rows < MATVEC_PAR_THRESHOLD {
        for i in 0..shared_rows {
            let q_row = &q_rowmajor[i * k_dim..(i + 1) * k_dim];
            let k_row = &k_rowmajor[i * k_dim..(i + 1) * k_dim];
            let v_row = &v_rowmajor[i * k_dim..(i + 1) * k_dim];
            let (q, k, v) = dot3_unrolled_f32_bf16(x, q_row, k_row, v_row);
            q_out[i] = q;
            k_out[i] = k;
            v_out[i] = v;
        }
    } else {
        q_out[..shared_rows]
            .par_iter_mut()
            .zip(k_out[..shared_rows].par_iter_mut())
            .zip(v_out[..shared_rows].par_iter_mut())
            .enumerate()
            .for_each(|(i, ((q_dst, k_dst), v_dst))| {
                let q_row = &q_rowmajor[i * k_dim..(i + 1) * k_dim];
                let k_row = &k_rowmajor[i * k_dim..(i + 1) * k_dim];
                let v_row = &v_rowmajor[i * k_dim..(i + 1) * k_dim];
                let (q, k, v) = dot3_unrolled_f32_bf16(x, q_row, k_row, v_row);
                *q_dst = q;
                *k_dst = k;
                *v_dst = v;
            });
    }

    if q_rows > shared_rows {
        matvec_rowmajor_parallel_f32_bf16(
            x,
            &q_rowmajor[shared_rows * k_dim..],
            q_rows - shared_rows,
            k_dim,
            &mut q_out[shared_rows..],
        );
    }
    if kv_rows > shared_rows {
        dual_matvec_rowmajor_parallel_f32_bf16(
            x,
            &k_rowmajor[shared_rows * k_dim..],
            &v_rowmajor[shared_rows * k_dim..],
            kv_rows - shared_rows,
            k_dim,
            &mut k_out[shared_rows..],
            &mut v_out[shared_rows..],
        );
    }
}

#[inline]
pub(crate) fn qkv_matvec_rowmajor_parallel_f32_f16(
    x: &[f32],
    q_rowmajor: &[f16],
    k_rowmajor: &[f16],
    v_rowmajor: &[f16],
    q_rows: usize,
    kv_rows: usize,
    k_dim: usize,
    q_out: &mut [f32],
    k_out: &mut [f32],
    v_out: &mut [f32],
) {
    assert_eq!(x.len(), k_dim, "x len / k_dim mismatch");
    assert_eq!(q_rowmajor.len(), q_rows * k_dim, "Q weight size mismatch");
    assert_eq!(k_rowmajor.len(), kv_rows * k_dim, "K weight size mismatch");
    assert_eq!(v_rowmajor.len(), kv_rows * k_dim, "V weight size mismatch");
    assert_eq!(q_out.len(), q_rows, "Q output size mismatch");
    assert_eq!(k_out.len(), kv_rows, "K output size mismatch");
    assert_eq!(v_out.len(), kv_rows, "V output size mismatch");

    let shared_rows = q_rows.min(kv_rows);
    if shared_rows < MATVEC_PAR_THRESHOLD {
        for i in 0..shared_rows {
            let q_row = &q_rowmajor[i * k_dim..(i + 1) * k_dim];
            let k_row = &k_rowmajor[i * k_dim..(i + 1) * k_dim];
            let v_row = &v_rowmajor[i * k_dim..(i + 1) * k_dim];
            let (q, k, v) = dot3_unrolled_f32_f16(x, q_row, k_row, v_row);
            q_out[i] = q;
            k_out[i] = k;
            v_out[i] = v;
        }
    } else {
        q_out[..shared_rows]
            .par_iter_mut()
            .zip(k_out[..shared_rows].par_iter_mut())
            .zip(v_out[..shared_rows].par_iter_mut())
            .enumerate()
            .for_each(|(i, ((q_dst, k_dst), v_dst))| {
                let q_row = &q_rowmajor[i * k_dim..(i + 1) * k_dim];
                let k_row = &k_rowmajor[i * k_dim..(i + 1) * k_dim];
                let v_row = &v_rowmajor[i * k_dim..(i + 1) * k_dim];
                let (q, k, v) = dot3_unrolled_f32_f16(x, q_row, k_row, v_row);
                *q_dst = q;
                *k_dst = k;
                *v_dst = v;
            });
    }

    if q_rows > shared_rows {
        matvec_rowmajor_parallel_f32_f16(
            x,
            &q_rowmajor[shared_rows * k_dim..],
            q_rows - shared_rows,
            k_dim,
            &mut q_out[shared_rows..],
        );
    }
    if kv_rows > shared_rows {
        dual_matvec_rowmajor_parallel_f32_f16(
            x,
            &k_rowmajor[shared_rows * k_dim..],
            &v_rowmajor[shared_rows * k_dim..],
            kv_rows - shared_rows,
            k_dim,
            &mut k_out[shared_rows..],
            &mut v_out[shared_rows..],
        );
    }
}

#[inline]
pub(crate) fn qkv_matvec_rowmajor_parallel_f32_i8(
    x: &[f32],
    q_rowmajor: &[i8],
    q_scale: f32,
    k_rowmajor: &[i8],
    k_scale: f32,
    v_rowmajor: &[i8],
    v_scale: f32,
    q_rows: usize,
    kv_rows: usize,
    k_dim: usize,
    q_out: &mut [f32],
    k_out: &mut [f32],
    v_out: &mut [f32],
) {
    assert_eq!(x.len(), k_dim, "x len / k_dim mismatch");
    assert_eq!(q_rowmajor.len(), q_rows * k_dim, "Q weight size mismatch");
    assert_eq!(k_rowmajor.len(), kv_rows * k_dim, "K weight size mismatch");
    assert_eq!(v_rowmajor.len(), kv_rows * k_dim, "V weight size mismatch");
    assert_eq!(q_out.len(), q_rows, "Q output size mismatch");
    assert_eq!(k_out.len(), kv_rows, "K output size mismatch");
    assert_eq!(v_out.len(), kv_rows, "V output size mismatch");

    let shared_rows = q_rows.min(kv_rows);
    if shared_rows < MATVEC_PAR_THRESHOLD {
        for i in 0..shared_rows {
            let q_row = &q_rowmajor[i * k_dim..(i + 1) * k_dim];
            let k_row = &k_rowmajor[i * k_dim..(i + 1) * k_dim];
            let v_row = &v_rowmajor[i * k_dim..(i + 1) * k_dim];
            let (q, k, v) = dot3_unrolled_f32_i8(x, q_row, q_scale, k_row, k_scale, v_row, v_scale);
            q_out[i] = q;
            k_out[i] = k;
            v_out[i] = v;
        }
    } else if should_use_i8_block_kernel(shared_rows, k_dim) {
        qkv_matvec_rowmajor_block_parallel_f32_i8(
            x,
            I8QkvBlock {
                q: I8ScaledSlice {
                    values: q_rowmajor,
                    scale: q_scale,
                },
                k: I8ScaledSlice {
                    values: k_rowmajor,
                    scale: k_scale,
                },
                v: I8ScaledSlice {
                    values: v_rowmajor,
                    scale: v_scale,
                },
            },
            k_dim,
            QkvOutMut {
                q: &mut q_out[..shared_rows],
                k: &mut k_out[..shared_rows],
                v: &mut v_out[..shared_rows],
            },
        );
    } else {
        q_out[..shared_rows]
            .par_chunks_mut(QKV_I8_PAR_CHUNK_ROWS)
            .zip(k_out[..shared_rows].par_chunks_mut(QKV_I8_PAR_CHUNK_ROWS))
            .zip(v_out[..shared_rows].par_chunks_mut(QKV_I8_PAR_CHUNK_ROWS))
            .enumerate()
            .for_each(|(chunk_idx, ((q_chunk, k_chunk), v_chunk))| {
                let row_start = chunk_idx * QKV_I8_PAR_CHUNK_ROWS;
                let mut offset = 0usize;
                if should_use_i8_qkv_row4_kernel() {
                    while offset + 4 <= q_chunk.len() {
                        let row_idx = row_start + offset;
                        dot3_rows_unrolled_f32_i8(
                            x,
                            I8QkvBlock {
                                q: I8ScaledSlice {
                                    values: &q_rowmajor[row_idx * k_dim..(row_idx + 4) * k_dim],
                                    scale: q_scale,
                                },
                                k: I8ScaledSlice {
                                    values: &k_rowmajor[row_idx * k_dim..(row_idx + 4) * k_dim],
                                    scale: k_scale,
                                },
                                v: I8ScaledSlice {
                                    values: &v_rowmajor[row_idx * k_dim..(row_idx + 4) * k_dim],
                                    scale: v_scale,
                                },
                            },
                            k_dim,
                            4,
                            QkvOutMut {
                                q: &mut q_chunk[offset..offset + 4],
                                k: &mut k_chunk[offset..offset + 4],
                                v: &mut v_chunk[offset..offset + 4],
                            },
                        );
                        offset += 4;
                    }
                }
                while offset < q_chunk.len() {
                    let row_idx = row_start + offset;
                    let q_row = &q_rowmajor[row_idx * k_dim..(row_idx + 1) * k_dim];
                    let k_row = &k_rowmajor[row_idx * k_dim..(row_idx + 1) * k_dim];
                    let v_row = &v_rowmajor[row_idx * k_dim..(row_idx + 1) * k_dim];
                    let (q, k, v) =
                        dot3_unrolled_f32_i8(x, q_row, q_scale, k_row, k_scale, v_row, v_scale);
                    q_chunk[offset] = q;
                    k_chunk[offset] = k;
                    v_chunk[offset] = v;
                    offset += 1;
                }
            });
    }

    if q_rows > shared_rows {
        matvec_rowmajor_parallel_f32_i8(
            x,
            &q_rowmajor[shared_rows * k_dim..],
            q_scale,
            q_rows - shared_rows,
            k_dim,
            &mut q_out[shared_rows..],
        );
    }
    if kv_rows > shared_rows {
        dual_matvec_rowmajor_parallel_f32_i8(
            x,
            &k_rowmajor[shared_rows * k_dim..],
            k_scale,
            &v_rowmajor[shared_rows * k_dim..],
            v_scale,
            kv_rows - shared_rows,
            k_dim,
            &mut k_out[shared_rows..],
            &mut v_out[shared_rows..],
        );
    }
}

#[inline]
pub(crate) fn qkv_matvec_rowmajor_parallel_i8_i8(
    x: &[i8],
    x_scale: f32,
    q_rowmajor: &[i8],
    q_scale: f32,
    k_rowmajor: &[i8],
    k_scale: f32,
    v_rowmajor: &[i8],
    v_scale: f32,
    q_rows: usize,
    kv_rows: usize,
    k_dim: usize,
    q_out: &mut [f32],
    k_out: &mut [f32],
    v_out: &mut [f32],
) {
    assert_eq!(x.len(), k_dim, "x len / k_dim mismatch");
    assert_eq!(q_rowmajor.len(), q_rows * k_dim, "Q weight size mismatch");
    assert_eq!(k_rowmajor.len(), kv_rows * k_dim, "K weight size mismatch");
    assert_eq!(v_rowmajor.len(), kv_rows * k_dim, "V weight size mismatch");
    assert_eq!(q_out.len(), q_rows, "Q output size mismatch");
    assert_eq!(k_out.len(), kv_rows, "K output size mismatch");
    assert_eq!(v_out.len(), kv_rows, "V output size mismatch");

    let shared_rows = q_rows.min(kv_rows);
    if shared_rows < MATVEC_PAR_THRESHOLD {
        for i in 0..shared_rows {
            let q_row = &q_rowmajor[i * k_dim..(i + 1) * k_dim];
            let k_row = &k_rowmajor[i * k_dim..(i + 1) * k_dim];
            let v_row = &v_rowmajor[i * k_dim..(i + 1) * k_dim];
            let (q, k, v) = dot3_unrolled_i8_i8(
                x,
                x_scale,
                [
                    I8ScaledSlice {
                        values: q_row,
                        scale: q_scale,
                    },
                    I8ScaledSlice {
                        values: k_row,
                        scale: k_scale,
                    },
                    I8ScaledSlice {
                        values: v_row,
                        scale: v_scale,
                    },
                ],
            );
            q_out[i] = q;
            k_out[i] = k;
            v_out[i] = v;
        }
    } else {
        q_out[..shared_rows]
            .par_chunks_mut(QKV_I8_PAR_CHUNK_ROWS)
            .zip(k_out[..shared_rows].par_chunks_mut(QKV_I8_PAR_CHUNK_ROWS))
            .zip(v_out[..shared_rows].par_chunks_mut(QKV_I8_PAR_CHUNK_ROWS))
            .enumerate()
            .for_each(|(chunk_idx, ((q_chunk, k_chunk), v_chunk))| {
                let row_start = chunk_idx * QKV_I8_PAR_CHUNK_ROWS;
                for offset in 0..q_chunk.len() {
                    let row_idx = row_start + offset;
                    let q_row = &q_rowmajor[row_idx * k_dim..(row_idx + 1) * k_dim];
                    let k_row = &k_rowmajor[row_idx * k_dim..(row_idx + 1) * k_dim];
                    let v_row = &v_rowmajor[row_idx * k_dim..(row_idx + 1) * k_dim];
                    let (q, k, v) = dot3_unrolled_i8_i8(
                        x,
                        x_scale,
                        [
                            I8ScaledSlice {
                                values: q_row,
                                scale: q_scale,
                            },
                            I8ScaledSlice {
                                values: k_row,
                                scale: k_scale,
                            },
                            I8ScaledSlice {
                                values: v_row,
                                scale: v_scale,
                            },
                        ],
                    );
                    q_chunk[offset] = q;
                    k_chunk[offset] = k;
                    v_chunk[offset] = v;
                }
            });
    }

    if q_rows > shared_rows {
        matvec_rowmajor_parallel_i8_i8(
            x,
            x_scale,
            &q_rowmajor[shared_rows * k_dim..],
            q_scale,
            q_rows - shared_rows,
            k_dim,
            &mut q_out[shared_rows..],
        );
    }
    if kv_rows > shared_rows {
        dual_matvec_rowmajor_parallel_i8_i8(
            x,
            x_scale,
            &k_rowmajor[shared_rows * k_dim..],
            k_scale,
            &v_rowmajor[shared_rows * k_dim..],
            v_scale,
            kv_rows - shared_rows,
            k_dim,
            &mut k_out[shared_rows..],
            &mut v_out[shared_rows..],
        );
    }
}

fn qkv_matvec_rowmajor_block_parallel_f32_i8(
    x: &[f32],
    weights: I8QkvBlock<'_>,
    k_dim: usize,
    out: QkvOutMut<'_>,
) {
    let q_rowmajor = weights.q.values;
    let q_scale = weights.q.scale;
    let k_rowmajor = weights.k.values;
    let k_scale = weights.k.scale;
    let v_rowmajor = weights.v.values;
    let v_scale = weights.v.scale;
    let q_out = out.q;
    let k_out = out.k;
    let v_out = out.v;

    q_out
        .par_chunks_mut(MATVEC_BLOCK_ROWS)
        .zip(k_out.par_chunks_mut(MATVEC_BLOCK_ROWS))
        .zip(v_out.par_chunks_mut(MATVEC_BLOCK_ROWS))
        .enumerate()
        .for_each(|(block_idx, ((q_chunk, k_chunk), v_chunk))| {
            let row_start = block_idx * MATVEC_BLOCK_ROWS;
            let rows = q_chunk.len();
            let q_block = &q_rowmajor[row_start * k_dim..(row_start + rows) * k_dim];
            let k_block = &k_rowmajor[row_start * k_dim..(row_start + rows) * k_dim];
            let v_block = &v_rowmajor[row_start * k_dim..(row_start + rows) * k_dim];
            let mut q_acc = [0.0f32; MATVEC_BLOCK_ROWS];
            let mut k_acc = [0.0f32; MATVEC_BLOCK_ROWS];
            let mut v_acc = [0.0f32; MATVEC_BLOCK_ROWS];

            let mut kk = 0usize;
            while kk + 16 <= k_dim {
                let x0 = x[kk];
                let x1 = x[kk + 1];
                let x2 = x[kk + 2];
                let x3 = x[kk + 3];
                let x4 = x[kk + 4];
                let x5 = x[kk + 5];
                let x6 = x[kk + 6];
                let x7 = x[kk + 7];
                let x8 = x[kk + 8];
                let x9 = x[kk + 9];
                let x10 = x[kk + 10];
                let x11 = x[kk + 11];
                let x12 = x[kk + 12];
                let x13 = x[kk + 13];
                let x14 = x[kk + 14];
                let x15 = x[kk + 15];
                for r in 0..rows {
                    let base = r * k_dim + kk;
                    q_acc[r] += q_block[base] as f32 * x0
                        + q_block[base + 1] as f32 * x1
                        + q_block[base + 2] as f32 * x2
                        + q_block[base + 3] as f32 * x3
                        + q_block[base + 4] as f32 * x4
                        + q_block[base + 5] as f32 * x5
                        + q_block[base + 6] as f32 * x6
                        + q_block[base + 7] as f32 * x7
                        + q_block[base + 8] as f32 * x8
                        + q_block[base + 9] as f32 * x9
                        + q_block[base + 10] as f32 * x10
                        + q_block[base + 11] as f32 * x11
                        + q_block[base + 12] as f32 * x12
                        + q_block[base + 13] as f32 * x13
                        + q_block[base + 14] as f32 * x14
                        + q_block[base + 15] as f32 * x15;
                    k_acc[r] += k_block[base] as f32 * x0
                        + k_block[base + 1] as f32 * x1
                        + k_block[base + 2] as f32 * x2
                        + k_block[base + 3] as f32 * x3
                        + k_block[base + 4] as f32 * x4
                        + k_block[base + 5] as f32 * x5
                        + k_block[base + 6] as f32 * x6
                        + k_block[base + 7] as f32 * x7
                        + k_block[base + 8] as f32 * x8
                        + k_block[base + 9] as f32 * x9
                        + k_block[base + 10] as f32 * x10
                        + k_block[base + 11] as f32 * x11
                        + k_block[base + 12] as f32 * x12
                        + k_block[base + 13] as f32 * x13
                        + k_block[base + 14] as f32 * x14
                        + k_block[base + 15] as f32 * x15;
                    v_acc[r] += v_block[base] as f32 * x0
                        + v_block[base + 1] as f32 * x1
                        + v_block[base + 2] as f32 * x2
                        + v_block[base + 3] as f32 * x3
                        + v_block[base + 4] as f32 * x4
                        + v_block[base + 5] as f32 * x5
                        + v_block[base + 6] as f32 * x6
                        + v_block[base + 7] as f32 * x7
                        + v_block[base + 8] as f32 * x8
                        + v_block[base + 9] as f32 * x9
                        + v_block[base + 10] as f32 * x10
                        + v_block[base + 11] as f32 * x11
                        + v_block[base + 12] as f32 * x12
                        + v_block[base + 13] as f32 * x13
                        + v_block[base + 14] as f32 * x14
                        + v_block[base + 15] as f32 * x15;
                }
                kk += 16;
            }

            while kk + 8 <= k_dim {
                let x0 = x[kk];
                let x1 = x[kk + 1];
                let x2 = x[kk + 2];
                let x3 = x[kk + 3];
                let x4 = x[kk + 4];
                let x5 = x[kk + 5];
                let x6 = x[kk + 6];
                let x7 = x[kk + 7];
                for r in 0..rows {
                    let base = r * k_dim + kk;
                    q_acc[r] += q_block[base] as f32 * x0
                        + q_block[base + 1] as f32 * x1
                        + q_block[base + 2] as f32 * x2
                        + q_block[base + 3] as f32 * x3
                        + q_block[base + 4] as f32 * x4
                        + q_block[base + 5] as f32 * x5
                        + q_block[base + 6] as f32 * x6
                        + q_block[base + 7] as f32 * x7;
                    k_acc[r] += k_block[base] as f32 * x0
                        + k_block[base + 1] as f32 * x1
                        + k_block[base + 2] as f32 * x2
                        + k_block[base + 3] as f32 * x3
                        + k_block[base + 4] as f32 * x4
                        + k_block[base + 5] as f32 * x5
                        + k_block[base + 6] as f32 * x6
                        + k_block[base + 7] as f32 * x7;
                    v_acc[r] += v_block[base] as f32 * x0
                        + v_block[base + 1] as f32 * x1
                        + v_block[base + 2] as f32 * x2
                        + v_block[base + 3] as f32 * x3
                        + v_block[base + 4] as f32 * x4
                        + v_block[base + 5] as f32 * x5
                        + v_block[base + 6] as f32 * x6
                        + v_block[base + 7] as f32 * x7;
                }
                kk += 8;
            }

            while kk < k_dim {
                let xv = x[kk];
                for r in 0..rows {
                    let base = r * k_dim + kk;
                    q_acc[r] += q_block[base] as f32 * xv;
                    k_acc[r] += k_block[base] as f32 * xv;
                    v_acc[r] += v_block[base] as f32 * xv;
                }
                kk += 1;
            }

            for r in 0..rows {
                q_chunk[r] = q_acc[r] * q_scale;
                k_chunk[r] = k_acc[r] * k_scale;
                v_chunk[r] = v_acc[r] * v_scale;
            }
        });
}

#[inline]
pub fn dual_matvec_silu_mul_rowmajor_parallel(
    x: &[f32],
    gate_w_rowmajor: &[f32],
    up_w_rowmajor: &[f32],
    n_rows: usize,
    k_dim: usize,
    out: &mut [f32],
) {
    assert_eq!(x.len(), k_dim, "x len / k_dim mismatch");
    assert_eq!(
        gate_w_rowmajor.len(),
        n_rows * k_dim,
        "gate weight size mismatch"
    );
    assert_eq!(
        up_w_rowmajor.len(),
        n_rows * k_dim,
        "up weight size mismatch"
    );
    assert_eq!(out.len(), n_rows, "out size mismatch");

    if n_rows < MATVEC_PAR_THRESHOLD {
        for i in 0..n_rows {
            let gate_row = &gate_w_rowmajor[i * k_dim..(i + 1) * k_dim];
            let up_row = &up_w_rowmajor[i * k_dim..(i + 1) * k_dim];
            let (g, u) = dot2_unrolled(x, gate_row, up_row);
            let sig = 1.0 / (1.0 + (-g).exp());
            out[i] = (g * sig) * u;
        }
    } else {
        out.par_iter_mut().enumerate().for_each(|(i, out_val)| {
            let gate_row = &gate_w_rowmajor[i * k_dim..(i + 1) * k_dim];
            let up_row = &up_w_rowmajor[i * k_dim..(i + 1) * k_dim];
            let (g, u) = dot2_unrolled(x, gate_row, up_row);
            let sig = 1.0 / (1.0 + (-g).exp());
            *out_val = (g * sig) * u;
        });
    }
}

#[inline]
pub(crate) fn dual_matvec_silu_mul_rowmajor_parallel_f32_bf16(
    x: &[f32],
    gate_w_rowmajor: &[bf16],
    up_w_rowmajor: &[bf16],
    n_rows: usize,
    k_dim: usize,
    out: &mut [f32],
) {
    assert_eq!(x.len(), k_dim, "x len / k_dim mismatch");
    assert_eq!(
        gate_w_rowmajor.len(),
        n_rows * k_dim,
        "gate weight size mismatch"
    );
    assert_eq!(
        up_w_rowmajor.len(),
        n_rows * k_dim,
        "up weight size mismatch"
    );
    assert_eq!(out.len(), n_rows, "out size mismatch");

    if n_rows < MATVEC_PAR_THRESHOLD {
        for i in 0..n_rows {
            let gate_row = &gate_w_rowmajor[i * k_dim..(i + 1) * k_dim];
            let up_row = &up_w_rowmajor[i * k_dim..(i + 1) * k_dim];
            let (g, u) = dot2_unrolled_f32_bf16(x, gate_row, up_row);
            let sig = 1.0 / (1.0 + (-g).exp());
            out[i] = (g * sig) * u;
        }
    } else {
        out.par_chunks_mut(MIXED_ROW_PAR_CHUNK_ROWS)
            .enumerate()
            .for_each(|(chunk_idx, out_chunk)| {
                let row_start = chunk_idx * MIXED_ROW_PAR_CHUNK_ROWS;
                for (offset, out_val) in out_chunk.iter_mut().enumerate() {
                    let row_idx = row_start + offset;
                    let gate_row = &gate_w_rowmajor[row_idx * k_dim..(row_idx + 1) * k_dim];
                    let up_row = &up_w_rowmajor[row_idx * k_dim..(row_idx + 1) * k_dim];
                    let (g, u) = dot2_unrolled_f32_bf16(x, gate_row, up_row);
                    let sig = 1.0 / (1.0 + (-g).exp());
                    *out_val = (g * sig) * u;
                }
            });
    }
}

#[inline]
pub(crate) fn dual_matvec_silu_mul_rowmajor_parallel_f32_f16(
    x: &[f32],
    gate_w_rowmajor: &[f16],
    up_w_rowmajor: &[f16],
    n_rows: usize,
    k_dim: usize,
    out: &mut [f32],
) {
    assert_eq!(x.len(), k_dim, "x len / k_dim mismatch");
    assert_eq!(
        gate_w_rowmajor.len(),
        n_rows * k_dim,
        "gate weight size mismatch"
    );
    assert_eq!(
        up_w_rowmajor.len(),
        n_rows * k_dim,
        "up weight size mismatch"
    );
    assert_eq!(out.len(), n_rows, "out size mismatch");

    if n_rows < MATVEC_PAR_THRESHOLD {
        for i in 0..n_rows {
            let gate_row = &gate_w_rowmajor[i * k_dim..(i + 1) * k_dim];
            let up_row = &up_w_rowmajor[i * k_dim..(i + 1) * k_dim];
            let (g, u) = dot2_unrolled_f32_f16(x, gate_row, up_row);
            let sig = 1.0 / (1.0 + (-g).exp());
            out[i] = (g * sig) * u;
        }
    } else {
        out.par_chunks_mut(MIXED_ROW_PAR_CHUNK_ROWS)
            .enumerate()
            .for_each(|(chunk_idx, out_chunk)| {
                let row_start = chunk_idx * MIXED_ROW_PAR_CHUNK_ROWS;
                for (offset, out_val) in out_chunk.iter_mut().enumerate() {
                    let row_idx = row_start + offset;
                    let gate_row = &gate_w_rowmajor[row_idx * k_dim..(row_idx + 1) * k_dim];
                    let up_row = &up_w_rowmajor[row_idx * k_dim..(row_idx + 1) * k_dim];
                    let (g, u) = dot2_unrolled_f32_f16(x, gate_row, up_row);
                    let sig = 1.0 / (1.0 + (-g).exp());
                    *out_val = (g * sig) * u;
                }
            });
    }
}

#[inline]
pub(crate) fn dual_matvec_silu_mul_rowmajor_parallel_f32_i8(
    x: &[f32],
    gate_w_rowmajor: &[i8],
    gate_scale: f32,
    up_w_rowmajor: &[i8],
    up_scale: f32,
    n_rows: usize,
    k_dim: usize,
    out: &mut [f32],
) {
    assert_eq!(x.len(), k_dim, "x len / k_dim mismatch");
    assert_eq!(
        gate_w_rowmajor.len(),
        n_rows * k_dim,
        "gate weight size mismatch"
    );
    assert_eq!(
        up_w_rowmajor.len(),
        n_rows * k_dim,
        "up weight size mismatch"
    );
    assert_eq!(out.len(), n_rows, "out size mismatch");

    if n_rows < MATVEC_PAR_THRESHOLD {
        for i in 0..n_rows {
            let gate_row = &gate_w_rowmajor[i * k_dim..(i + 1) * k_dim];
            let up_row = &up_w_rowmajor[i * k_dim..(i + 1) * k_dim];
            let (g, u) = dot2_unrolled_f32_i8(x, gate_row, gate_scale, up_row, up_scale);
            let sig = 1.0 / (1.0 + (-g).exp());
            out[i] = (g * sig) * u;
        }
    } else if should_use_i8_silu_block_kernel(n_rows, k_dim) {
        dual_matvec_silu_mul_rowmajor_block_parallel_f32_i8(
            x,
            I8DualBlock {
                left: I8ScaledSlice {
                    values: gate_w_rowmajor,
                    scale: gate_scale,
                },
                right: I8ScaledSlice {
                    values: up_w_rowmajor,
                    scale: up_scale,
                },
            },
            k_dim,
            out,
        );
    } else {
        out.par_chunks_mut(MATVEC_I8_PAR_CHUNK_ROWS)
            .enumerate()
            .for_each(|(chunk_idx, out_chunk)| {
                let row_start = chunk_idx * MATVEC_I8_PAR_CHUNK_ROWS;
                for (offset, out_val) in out_chunk.iter_mut().enumerate() {
                    let row_idx = row_start + offset;
                    let gate_row = &gate_w_rowmajor[row_idx * k_dim..(row_idx + 1) * k_dim];
                    let up_row = &up_w_rowmajor[row_idx * k_dim..(row_idx + 1) * k_dim];
                    let (g, u) = dot2_unrolled_f32_i8(x, gate_row, gate_scale, up_row, up_scale);
                    let sig = 1.0 / (1.0 + (-g).exp());
                    *out_val = (g * sig) * u;
                }
            });
    }
}

#[inline]
pub(crate) fn dual_matvec_silu_mul_rowmajor_parallel_i8_i8(
    x: &[i8],
    x_scale: f32,
    gate_w_rowmajor: &[i8],
    gate_scale: f32,
    up_w_rowmajor: &[i8],
    up_scale: f32,
    n_rows: usize,
    k_dim: usize,
    out: &mut [f32],
) {
    assert_eq!(x.len(), k_dim, "x len / k_dim mismatch");
    assert_eq!(
        gate_w_rowmajor.len(),
        n_rows * k_dim,
        "gate weight size mismatch"
    );
    assert_eq!(
        up_w_rowmajor.len(),
        n_rows * k_dim,
        "up weight size mismatch"
    );
    assert_eq!(out.len(), n_rows, "out size mismatch");

    if n_rows < MATVEC_PAR_THRESHOLD {
        for i in 0..n_rows {
            let gate_row = &gate_w_rowmajor[i * k_dim..(i + 1) * k_dim];
            let up_row = &up_w_rowmajor[i * k_dim..(i + 1) * k_dim];
            let (g, u) = dot2_unrolled_i8_i8(x, x_scale, gate_row, gate_scale, up_row, up_scale);
            let sig = 1.0 / (1.0 + (-g).exp());
            out[i] = (g * sig) * u;
        }
    } else {
        out.par_chunks_mut(MATVEC_I8_PAR_CHUNK_ROWS)
            .enumerate()
            .for_each(|(chunk_idx, out_chunk)| {
                let row_start = chunk_idx * MATVEC_I8_PAR_CHUNK_ROWS;
                for (offset, out_val) in out_chunk.iter_mut().enumerate() {
                    let row_idx = row_start + offset;
                    let gate_row = &gate_w_rowmajor[row_idx * k_dim..(row_idx + 1) * k_dim];
                    let up_row = &up_w_rowmajor[row_idx * k_dim..(row_idx + 1) * k_dim];
                    let (g, u) =
                        dot2_unrolled_i8_i8(x, x_scale, gate_row, gate_scale, up_row, up_scale);
                    let sig = 1.0 / (1.0 + (-g).exp());
                    *out_val = (g * sig) * u;
                }
            });
    }
}

fn dual_matvec_silu_mul_rowmajor_block_parallel_f32_i8(
    x: &[f32],
    weights: I8DualBlock<'_>,
    k_dim: usize,
    out: &mut [f32],
) {
    let gate_w_rowmajor = weights.left.values;
    let gate_scale = weights.left.scale;
    let up_w_rowmajor = weights.right.values;
    let up_scale = weights.right.scale;

    out.par_chunks_mut(SILU_I8_BLOCK_ROWS)
        .enumerate()
        .for_each(|(block_idx, out_chunk)| {
            let row_start = block_idx * SILU_I8_BLOCK_ROWS;
            let rows = out_chunk.len();
            let gate_block = &gate_w_rowmajor[row_start * k_dim..(row_start + rows) * k_dim];
            let up_block = &up_w_rowmajor[row_start * k_dim..(row_start + rows) * k_dim];
            let mut gate_acc = [0.0f32; SILU_I8_BLOCK_ROWS];
            let mut up_acc = [0.0f32; SILU_I8_BLOCK_ROWS];

            let mut kk = 0usize;
            while kk + 16 <= k_dim {
                let x0 = x[kk];
                let x1 = x[kk + 1];
                let x2 = x[kk + 2];
                let x3 = x[kk + 3];
                let x4 = x[kk + 4];
                let x5 = x[kk + 5];
                let x6 = x[kk + 6];
                let x7 = x[kk + 7];
                let x8 = x[kk + 8];
                let x9 = x[kk + 9];
                let x10 = x[kk + 10];
                let x11 = x[kk + 11];
                let x12 = x[kk + 12];
                let x13 = x[kk + 13];
                let x14 = x[kk + 14];
                let x15 = x[kk + 15];
                for r in 0..rows {
                    let base = r * k_dim + kk;
                    gate_acc[r] += gate_block[base] as f32 * x0
                        + gate_block[base + 1] as f32 * x1
                        + gate_block[base + 2] as f32 * x2
                        + gate_block[base + 3] as f32 * x3
                        + gate_block[base + 4] as f32 * x4
                        + gate_block[base + 5] as f32 * x5
                        + gate_block[base + 6] as f32 * x6
                        + gate_block[base + 7] as f32 * x7
                        + gate_block[base + 8] as f32 * x8
                        + gate_block[base + 9] as f32 * x9
                        + gate_block[base + 10] as f32 * x10
                        + gate_block[base + 11] as f32 * x11
                        + gate_block[base + 12] as f32 * x12
                        + gate_block[base + 13] as f32 * x13
                        + gate_block[base + 14] as f32 * x14
                        + gate_block[base + 15] as f32 * x15;
                    up_acc[r] += up_block[base] as f32 * x0
                        + up_block[base + 1] as f32 * x1
                        + up_block[base + 2] as f32 * x2
                        + up_block[base + 3] as f32 * x3
                        + up_block[base + 4] as f32 * x4
                        + up_block[base + 5] as f32 * x5
                        + up_block[base + 6] as f32 * x6
                        + up_block[base + 7] as f32 * x7
                        + up_block[base + 8] as f32 * x8
                        + up_block[base + 9] as f32 * x9
                        + up_block[base + 10] as f32 * x10
                        + up_block[base + 11] as f32 * x11
                        + up_block[base + 12] as f32 * x12
                        + up_block[base + 13] as f32 * x13
                        + up_block[base + 14] as f32 * x14
                        + up_block[base + 15] as f32 * x15;
                }
                kk += 16;
            }

            while kk + 8 <= k_dim {
                let x0 = x[kk];
                let x1 = x[kk + 1];
                let x2 = x[kk + 2];
                let x3 = x[kk + 3];
                let x4 = x[kk + 4];
                let x5 = x[kk + 5];
                let x6 = x[kk + 6];
                let x7 = x[kk + 7];
                for r in 0..rows {
                    let base = r * k_dim + kk;
                    gate_acc[r] += gate_block[base] as f32 * x0
                        + gate_block[base + 1] as f32 * x1
                        + gate_block[base + 2] as f32 * x2
                        + gate_block[base + 3] as f32 * x3
                        + gate_block[base + 4] as f32 * x4
                        + gate_block[base + 5] as f32 * x5
                        + gate_block[base + 6] as f32 * x6
                        + gate_block[base + 7] as f32 * x7;
                    up_acc[r] += up_block[base] as f32 * x0
                        + up_block[base + 1] as f32 * x1
                        + up_block[base + 2] as f32 * x2
                        + up_block[base + 3] as f32 * x3
                        + up_block[base + 4] as f32 * x4
                        + up_block[base + 5] as f32 * x5
                        + up_block[base + 6] as f32 * x6
                        + up_block[base + 7] as f32 * x7;
                }
                kk += 8;
            }

            while kk < k_dim {
                let xv = x[kk];
                for r in 0..rows {
                    let base = r * k_dim + kk;
                    gate_acc[r] += gate_block[base] as f32 * xv;
                    up_acc[r] += up_block[base] as f32 * xv;
                }
                kk += 1;
            }

            for r in 0..rows {
                let g = gate_acc[r] * gate_scale;
                let sig = 1.0 / (1.0 + (-g).exp());
                out_chunk[r] = (g * sig) * (up_acc[r] * up_scale);
            }
        });
}

fn dual_matvec_silu_mul_rowmajor_parallel_bf16_f32(
    x: &[bf16],
    gate_w_rowmajor: &[f32],
    up_w_rowmajor: &[f32],
    n_rows: usize,
    k_dim: usize,
    out: &mut [f32],
) {
    with_bf16_input_as_f32(x, |x_f32| {
        dual_matvec_silu_mul_rowmajor_parallel(
            x_f32,
            gate_w_rowmajor,
            up_w_rowmajor,
            n_rows,
            k_dim,
            out,
        );
    });
}

#[inline]
fn dual_matvec_silu_mul_rowmajor_parallel_bf16_bf16(
    x: &[bf16],
    gate_w_rowmajor: &[bf16],
    up_w_rowmajor: &[bf16],
    n_rows: usize,
    k_dim: usize,
    out: &mut [f32],
) {
    assert_eq!(x.len(), k_dim, "x len / k_dim mismatch");
    assert_eq!(
        gate_w_rowmajor.len(),
        n_rows * k_dim,
        "gate weight size mismatch"
    );
    assert_eq!(
        up_w_rowmajor.len(),
        n_rows * k_dim,
        "up weight size mismatch"
    );
    assert_eq!(out.len(), n_rows, "out size mismatch");

    if crate::arch::x86_avx512_bf16_kernel_runtime_available() {
        let compute = |i: usize| {
            let gate = &gate_w_rowmajor[i * k_dim..(i + 1) * k_dim];
            let up = &up_w_rowmajor[i * k_dim..(i + 1) * k_dim];
            let (g, u) = dot2_bf16_bf16_arch(x, gate, up).unwrap_or_else(|| {
                (
                    dot_unrolled_bf16_bf16_scalar(x, gate),
                    dot_unrolled_bf16_bf16_scalar(x, up),
                )
            });
            let sig = 1.0 / (1.0 + (-g).exp());
            (g * sig) * u
        };

        if n_rows < MATVEC_PAR_THRESHOLD {
            for (i, out_val) in out.iter_mut().enumerate() {
                *out_val = compute(i);
            }
        } else {
            out.par_iter_mut()
                .enumerate()
                .for_each(|(i, out_val)| *out_val = compute(i));
        }
        return;
    }

    with_bf16_input_as_f32(x, |x_f32| {
        dual_matvec_silu_mul_rowmajor_parallel_f32_bf16(
            x_f32,
            gate_w_rowmajor,
            up_w_rowmajor,
            n_rows,
            k_dim,
            out,
        );
    });
}

#[inline]
fn dual_matvec_silu_mul_rowmajor_parallel_f16_f16(
    x: &[f16],
    gate_w_rowmajor: &[f16],
    up_w_rowmajor: &[f16],
    n_rows: usize,
    k_dim: usize,
    out: &mut [f32],
) {
    assert_eq!(x.len(), k_dim, "x len / k_dim mismatch");
    assert_eq!(
        gate_w_rowmajor.len(),
        n_rows * k_dim,
        "gate weight size mismatch"
    );
    assert_eq!(
        up_w_rowmajor.len(),
        n_rows * k_dim,
        "up weight size mismatch"
    );
    assert_eq!(out.len(), n_rows, "out size mismatch");

    if crate::arch::x86_avx512_fp16_kernel_runtime_available() {
        let compute = |i: usize| {
            let gate = &gate_w_rowmajor[i * k_dim..(i + 1) * k_dim];
            let up = &up_w_rowmajor[i * k_dim..(i + 1) * k_dim];
            let (g, u) = dot2_f16_f16_arch(x, gate, up).unwrap_or_else(|| {
                (
                    dot_unrolled_f16_f16_scalar(x, gate),
                    dot_unrolled_f16_f16_scalar(x, up),
                )
            });
            let sig = 1.0 / (1.0 + (-g).exp());
            (g * sig) * u
        };

        if n_rows < MATVEC_PAR_THRESHOLD {
            for (i, out_val) in out.iter_mut().enumerate() {
                *out_val = compute(i);
            }
        } else {
            out.par_iter_mut()
                .enumerate()
                .for_each(|(i, out_val)| *out_val = compute(i));
        }
        return;
    }

    with_f16_input_as_f32(x, |x_f32| {
        dual_matvec_silu_mul_rowmajor_parallel_f32_f16(
            x_f32,
            gate_w_rowmajor,
            up_w_rowmajor,
            n_rows,
            k_dim,
            out,
        );
    });
}

#[inline]
fn dual_matvec_silu_mul_rowmajor_parallel_bf16_i8(
    x: &[bf16],
    gate_w_rowmajor: &[i8],
    gate_scale: f32,
    up_w_rowmajor: &[i8],
    up_scale: f32,
    n_rows: usize,
    k_dim: usize,
    out: &mut [f32],
) {
    with_bf16_input_as_f32(x, |x_f32| {
        dual_matvec_silu_mul_rowmajor_parallel_f32_i8(
            x_f32,
            gate_w_rowmajor,
            gate_scale,
            up_w_rowmajor,
            up_scale,
            n_rows,
            k_dim,
            out,
        );
    });
}

#[inline]
fn dual_matvec_silu_mul_rowmajor_parallel_i8_f32(
    x: &[i8],
    x_scale: f32,
    gate_w_rowmajor: &[f32],
    up_w_rowmajor: &[f32],
    n_rows: usize,
    k_dim: usize,
    out: &mut [f32],
) {
    with_i8_input_as_f32(x, x_scale, |x_f32| {
        dual_matvec_silu_mul_rowmajor_parallel(
            x_f32,
            gate_w_rowmajor,
            up_w_rowmajor,
            n_rows,
            k_dim,
            out,
        );
    });
}

#[inline]
fn dual_matvec_silu_mul_rowmajor_parallel_i8_bf16(
    x: &[i8],
    x_scale: f32,
    gate_w_rowmajor: &[bf16],
    up_w_rowmajor: &[bf16],
    n_rows: usize,
    k_dim: usize,
    out: &mut [f32],
) {
    with_i8_input_as_f32(x, x_scale, |x_f32| {
        dual_matvec_silu_mul_rowmajor_parallel_f32_bf16(
            x_f32,
            gate_w_rowmajor,
            up_w_rowmajor,
            n_rows,
            k_dim,
            out,
        );
    });
}

fn dual_matvec_silu_mul_rowmajor_parallel_mixed_f32_input(
    x: &[f32],
    gate_w_rowmajor: SliceRef<'_>,
    up_w_rowmajor: SliceRef<'_>,
    n_rows: usize,
    k_dim: usize,
    out: &mut [f32],
) {
    if n_rows < MATVEC_PAR_THRESHOLD {
        for (i, out_val) in out.iter_mut().enumerate().take(n_rows) {
            let g = dot_unrolled_from_slice(x, row_slice(gate_w_rowmajor, i, k_dim));
            let u = dot_unrolled_from_slice(x, row_slice(up_w_rowmajor, i, k_dim));
            let sig = 1.0 / (1.0 + (-g).exp());
            *out_val = (g * sig) * u;
        }
    } else {
        out.par_iter_mut().enumerate().for_each(|(i, out_val)| {
            let g = dot_unrolled_from_slice(x, row_slice(gate_w_rowmajor, i, k_dim));
            let u = dot_unrolled_from_slice(x, row_slice(up_w_rowmajor, i, k_dim));
            let sig = 1.0 / (1.0 + (-g).exp());
            *out_val = (g * sig) * u;
        });
    }
}

#[inline]
pub fn dual_matvec_silu_mul_rowmajor_parallel_mixed(
    x: SliceRef<'_>,
    gate_w_rowmajor: SliceRef<'_>,
    up_w_rowmajor: SliceRef<'_>,
    n_rows: usize,
    k_dim: usize,
    out: &mut [f32],
) {
    match (x, gate_w_rowmajor, up_w_rowmajor) {
        (SliceRef::F32(x), SliceRef::F32(gate), SliceRef::F32(up)) => {
            dual_matvec_silu_mul_rowmajor_parallel(x, gate, up, n_rows, k_dim, out)
        }
        (SliceRef::F32(x), SliceRef::F16(gate), SliceRef::F16(up)) => {
            dual_matvec_silu_mul_rowmajor_parallel_f32_f16(x, gate, up, n_rows, k_dim, out)
        }
        (SliceRef::F32(x), SliceRef::BF16(gate), SliceRef::BF16(up)) => {
            dual_matvec_silu_mul_rowmajor_parallel_f32_bf16(x, gate, up, n_rows, k_dim, out)
        }
        (SliceRef::F32(x), SliceRef::I8(gate, gs), SliceRef::I8(up, us)) => {
            dual_matvec_silu_mul_rowmajor_parallel_f32_i8(x, gate, gs, up, us, n_rows, k_dim, out)
        }
        (SliceRef::F16(x), SliceRef::F32(gate), SliceRef::F32(up)) => {
            with_f16_input_as_f32(x, |x_f32| {
                dual_matvec_silu_mul_rowmajor_parallel(x_f32, gate, up, n_rows, k_dim, out)
            })
        }
        (SliceRef::F16(x), SliceRef::F16(gate), SliceRef::F16(up)) => {
            dual_matvec_silu_mul_rowmajor_parallel_f16_f16(x, gate, up, n_rows, k_dim, out)
        }
        (SliceRef::F16(x), SliceRef::BF16(gate), SliceRef::BF16(up)) => {
            with_f16_input_as_f32(x, |x_f32| {
                dual_matvec_silu_mul_rowmajor_parallel_f32_bf16(x_f32, gate, up, n_rows, k_dim, out)
            })
        }
        (SliceRef::F16(x), SliceRef::I8(gate, gs), SliceRef::I8(up, us)) => {
            with_f16_input_as_f32(x, |x_f32| {
                dual_matvec_silu_mul_rowmajor_parallel_f32_i8(
                    x_f32, gate, gs, up, us, n_rows, k_dim, out,
                )
            })
        }
        (SliceRef::BF16(x), SliceRef::F32(gate), SliceRef::F32(up)) => {
            dual_matvec_silu_mul_rowmajor_parallel_bf16_f32(x, gate, up, n_rows, k_dim, out)
        }
        (SliceRef::BF16(x), SliceRef::BF16(gate), SliceRef::BF16(up)) => {
            dual_matvec_silu_mul_rowmajor_parallel_bf16_bf16(x, gate, up, n_rows, k_dim, out)
        }
        (SliceRef::BF16(x), SliceRef::F16(gate), SliceRef::F16(up)) => {
            with_bf16_input_as_f32(x, |x_f32| {
                dual_matvec_silu_mul_rowmajor_parallel_f32_f16(x_f32, gate, up, n_rows, k_dim, out)
            })
        }
        (SliceRef::BF16(x), SliceRef::I8(gate, gs), SliceRef::I8(up, us)) => {
            dual_matvec_silu_mul_rowmajor_parallel_bf16_i8(x, gate, gs, up, us, n_rows, k_dim, out)
        }
        (SliceRef::I8(x, xs), SliceRef::F32(gate), SliceRef::F32(up)) => {
            dual_matvec_silu_mul_rowmajor_parallel_i8_f32(x, xs, gate, up, n_rows, k_dim, out)
        }
        (SliceRef::I8(x, xs), SliceRef::F16(gate), SliceRef::F16(up)) => {
            with_i8_input_as_f32(x, xs, |x_f32| {
                dual_matvec_silu_mul_rowmajor_parallel_f32_f16(x_f32, gate, up, n_rows, k_dim, out)
            })
        }
        (SliceRef::I8(x, xs), SliceRef::BF16(gate), SliceRef::BF16(up)) => {
            dual_matvec_silu_mul_rowmajor_parallel_i8_bf16(x, xs, gate, up, n_rows, k_dim, out)
        }
        (SliceRef::I8(x, xs), SliceRef::I8(gate, gs), SliceRef::I8(up, us)) => {
            dual_matvec_silu_mul_rowmajor_parallel_i8_i8(
                x, xs, gate, gs, up, us, n_rows, k_dim, out,
            )
        }
        (SliceRef::F32(x), gate, up) => {
            dual_matvec_silu_mul_rowmajor_parallel_mixed_f32_input(x, gate, up, n_rows, k_dim, out)
        }
        (SliceRef::F16(x), gate, up) => {
            with_f16_input_as_f32(x, |x_f32| {
                dual_matvec_silu_mul_rowmajor_parallel_mixed_f32_input(
                    x_f32, gate, up, n_rows, k_dim, out,
                )
            });
        }
        (SliceRef::BF16(x), gate, up) => {
            with_bf16_input_as_f32(x, |x_f32| {
                dual_matvec_silu_mul_rowmajor_parallel_mixed_f32_input(
                    x_f32, gate, up, n_rows, k_dim, out,
                )
            });
        }
        (SliceRef::I8(x, scale), gate, up) => {
            with_i8_input_as_f32(x, scale, |x_f32| {
                dual_matvec_silu_mul_rowmajor_parallel_mixed_f32_input(
                    x_f32, gate, up, n_rows, k_dim, out,
                )
            });
        }
    }
}

#[inline]
pub fn matvec_argmax_rowmajor_parallel(
    x: &[f32],
    w_rowmajor: &[f32],
    n_rows: usize,
    k_dim: usize,
) -> usize {
    assert_eq!(x.len(), k_dim, "x len / k_dim mismatch");
    assert_eq!(w_rowmajor.len(), n_rows * k_dim, "weight size mismatch");

    if should_use_argmax_block_kernel(n_rows) {
        let n_blocks = n_rows.div_ceil(ARGMAX_BLOCK_ROWS);
        return (0..n_blocks)
            .into_par_iter()
            .map(|block_idx| {
                let row_start = block_idx * ARGMAX_BLOCK_ROWS;
                let rows = (n_rows - row_start).min(ARGMAX_BLOCK_ROWS);
                let w_block = &w_rowmajor[row_start * k_dim..(row_start + rows) * k_dim];
                let mut acc = [0.0f32; ARGMAX_BLOCK_ROWS];

                let mut kk = 0usize;
                while kk + 8 <= k_dim {
                    let x0 = x[kk];
                    let x1 = x[kk + 1];
                    let x2 = x[kk + 2];
                    let x3 = x[kk + 3];
                    let x4 = x[kk + 4];
                    let x5 = x[kk + 5];
                    let x6 = x[kk + 6];
                    let x7 = x[kk + 7];
                    for (r, acc_r) in acc.iter_mut().enumerate().take(rows) {
                        let base = r * k_dim + kk;
                        *acc_r += w_block[base] * x0
                            + w_block[base + 1] * x1
                            + w_block[base + 2] * x2
                            + w_block[base + 3] * x3
                            + w_block[base + 4] * x4
                            + w_block[base + 5] * x5
                            + w_block[base + 6] * x6
                            + w_block[base + 7] * x7;
                    }
                    kk += 8;
                }

                while kk < k_dim {
                    let xv = x[kk];
                    for (r, acc_r) in acc.iter_mut().enumerate().take(rows) {
                        *acc_r += w_block[r * k_dim + kk] * xv;
                    }
                    kk += 1;
                }

                let mut best = (row_start, f32::NEG_INFINITY);
                for (r, &acc_r) in acc.iter().enumerate().take(rows) {
                    let cand = (row_start + r, acc_r);
                    if cand.1 > best.1 {
                        best = cand;
                    }
                }
                best
            })
            .reduce(
                || (0usize, f32::NEG_INFINITY),
                |a, b| if a.1 >= b.1 { a } else { b },
            )
            .0;
    }

    if n_rows < MATVEC_PAR_THRESHOLD {
        let mut best = (0usize, f32::NEG_INFINITY);
        for i in 0..n_rows {
            let row = &w_rowmajor[i * k_dim..(i + 1) * k_dim];
            let score = dot_unrolled(x, row);
            if score > best.1 {
                best = (i, score);
            }
        }
        best.0
    } else {
        (0..n_rows)
            .into_par_iter()
            .map(|i| {
                let row = &w_rowmajor[i * k_dim..(i + 1) * k_dim];
                (i, dot_unrolled(x, row))
            })
            .reduce(
                || (0usize, f32::NEG_INFINITY),
                |a, b| if a.1 >= b.1 { a } else { b },
            )
            .0
    }
}

#[inline]
fn matvec_argmax_rowmajor_parallel_f32_bf16(
    x: &[f32],
    w_rowmajor: &[bf16],
    n_rows: usize,
    k_dim: usize,
) -> usize {
    assert_eq!(x.len(), k_dim, "x len / k_dim mismatch");
    assert_eq!(w_rowmajor.len(), n_rows * k_dim, "weight size mismatch");

    if n_rows < MATVEC_PAR_THRESHOLD {
        let mut best = (0usize, f32::NEG_INFINITY);
        for i in 0..n_rows {
            let row = &w_rowmajor[i * k_dim..(i + 1) * k_dim];
            let score = dot_unrolled_f32_bf16(x, row);
            if score > best.1 {
                best = (i, score);
            }
        }
        best.0
    } else {
        (0..n_rows)
            .into_par_iter()
            .map(|i| {
                let row = &w_rowmajor[i * k_dim..(i + 1) * k_dim];
                (i, dot_unrolled_f32_bf16(x, row))
            })
            .reduce(
                || (0usize, f32::NEG_INFINITY),
                |a, b| if a.1 >= b.1 { a } else { b },
            )
            .0
    }
}

#[inline]
fn matvec_argmax_rowmajor_parallel_f32_f16(
    x: &[f32],
    w_rowmajor: &[f16],
    n_rows: usize,
    k_dim: usize,
) -> usize {
    assert_eq!(x.len(), k_dim, "x len / k_dim mismatch");
    assert_eq!(w_rowmajor.len(), n_rows * k_dim, "weight size mismatch");

    if n_rows < MATVEC_PAR_THRESHOLD {
        let mut best = (0usize, f32::NEG_INFINITY);
        for i in 0..n_rows {
            let row = &w_rowmajor[i * k_dim..(i + 1) * k_dim];
            let score = dot_unrolled_f32_f16(x, row);
            if score > best.1 {
                best = (i, score);
            }
        }
        best.0
    } else {
        (0..n_rows)
            .into_par_iter()
            .map(|i| {
                let row = &w_rowmajor[i * k_dim..(i + 1) * k_dim];
                (i, dot_unrolled_f32_f16(x, row))
            })
            .reduce(
                || (0usize, f32::NEG_INFINITY),
                |a, b| if a.1 >= b.1 { a } else { b },
            )
            .0
    }
}

#[inline]
fn matvec_argmax_rowmajor_parallel_f32_i8(
    x: &[f32],
    w_rowmajor: &[i8],
    scale: f32,
    n_rows: usize,
    k_dim: usize,
) -> usize {
    assert_eq!(x.len(), k_dim, "x len / k_dim mismatch");
    assert_eq!(w_rowmajor.len(), n_rows * k_dim, "weight size mismatch");

    if n_rows >= MATVEC_BLOCK_THRESHOLD {
        let n_blocks = n_rows.div_ceil(ARGMAX_BLOCK_ROWS);
        return (0..n_blocks)
            .into_par_iter()
            .map(|block_idx| {
                let row_start = block_idx * ARGMAX_BLOCK_ROWS;
                let rows = (n_rows - row_start).min(ARGMAX_BLOCK_ROWS);
                let w_block = &w_rowmajor[row_start * k_dim..(row_start + rows) * k_dim];
                let mut acc = [0.0f32; ARGMAX_BLOCK_ROWS];

                let mut kk = 0usize;
                while kk + 8 <= k_dim {
                    let x0 = x[kk];
                    let x1 = x[kk + 1];
                    let x2 = x[kk + 2];
                    let x3 = x[kk + 3];
                    let x4 = x[kk + 4];
                    let x5 = x[kk + 5];
                    let x6 = x[kk + 6];
                    let x7 = x[kk + 7];
                    for (r, acc_r) in acc.iter_mut().enumerate().take(rows) {
                        let base = r * k_dim + kk;
                        *acc_r += w_block[base] as f32 * x0
                            + w_block[base + 1] as f32 * x1
                            + w_block[base + 2] as f32 * x2
                            + w_block[base + 3] as f32 * x3
                            + w_block[base + 4] as f32 * x4
                            + w_block[base + 5] as f32 * x5
                            + w_block[base + 6] as f32 * x6
                            + w_block[base + 7] as f32 * x7;
                    }
                    kk += 8;
                }

                while kk < k_dim {
                    let xv = x[kk];
                    for (r, acc_r) in acc.iter_mut().enumerate().take(rows) {
                        *acc_r += w_block[r * k_dim + kk] as f32 * xv;
                    }
                    kk += 1;
                }

                let mut best = (row_start, f32::NEG_INFINITY);
                for (r, value) in acc[..rows].iter().enumerate() {
                    let cand = (row_start + r, *value * scale);
                    if cand.1 > best.1 {
                        best = cand;
                    }
                }
                best
            })
            .reduce(
                || (0usize, f32::NEG_INFINITY),
                |a, b| if a.1 >= b.1 { a } else { b },
            )
            .0;
    }

    if n_rows < MATVEC_PAR_THRESHOLD {
        let mut best = (0usize, f32::NEG_INFINITY);
        for i in 0..n_rows {
            let row = &w_rowmajor[i * k_dim..(i + 1) * k_dim];
            let score = dot_unrolled_f32_i8(x, row, scale);
            if score > best.1 {
                best = (i, score);
            }
        }
        best.0
    } else {
        (0..n_rows)
            .into_par_iter()
            .map(|i| {
                let row = &w_rowmajor[i * k_dim..(i + 1) * k_dim];
                (i, dot_unrolled_f32_i8(x, row, scale))
            })
            .reduce(
                || (0usize, f32::NEG_INFINITY),
                |a, b| if a.1 >= b.1 { a } else { b },
            )
            .0
    }
}

#[inline]
fn matvec_argmax_rowmajor_parallel_bf16_f32(
    x: &[bf16],
    w_rowmajor: &[f32],
    n_rows: usize,
    k_dim: usize,
) -> usize {
    with_bf16_input_as_f32(x, |x_f32| {
        matvec_argmax_rowmajor_parallel(x_f32, w_rowmajor, n_rows, k_dim)
    })
}

#[inline]
fn matvec_argmax_rowmajor_parallel_bf16_bf16(
    x: &[bf16],
    w_rowmajor: &[bf16],
    n_rows: usize,
    k_dim: usize,
) -> usize {
    assert_eq!(x.len(), k_dim, "x len / k_dim mismatch");
    assert_eq!(w_rowmajor.len(), n_rows * k_dim, "weight size mismatch");

    if crate::arch::x86_avx512_bf16_kernel_runtime_available() {
        if n_rows < MATVEC_PAR_THRESHOLD {
            let mut best = (0usize, f32::NEG_INFINITY);
            for i in 0..n_rows {
                let row = &w_rowmajor[i * k_dim..(i + 1) * k_dim];
                let score = dot_bf16_bf16_arch(x, row)
                    .unwrap_or_else(|| dot_unrolled_bf16_bf16_scalar(x, row));
                if score > best.1 {
                    best = (i, score);
                }
            }
            return best.0;
        }

        return (0..n_rows)
            .into_par_iter()
            .map(|i| {
                let row = &w_rowmajor[i * k_dim..(i + 1) * k_dim];
                (
                    i,
                    dot_bf16_bf16_arch(x, row)
                        .unwrap_or_else(|| dot_unrolled_bf16_bf16_scalar(x, row)),
                )
            })
            .reduce(
                || (0usize, f32::NEG_INFINITY),
                |a, b| if a.1 >= b.1 { a } else { b },
            )
            .0;
    }

    with_bf16_input_as_f32(x, |x_f32| {
        matvec_argmax_rowmajor_parallel_f32_bf16(x_f32, w_rowmajor, n_rows, k_dim)
    })
}

#[inline]
fn matvec_argmax_rowmajor_parallel_f16_f16(
    x: &[f16],
    w_rowmajor: &[f16],
    n_rows: usize,
    k_dim: usize,
) -> usize {
    assert_eq!(x.len(), k_dim, "x len / k_dim mismatch");
    assert_eq!(w_rowmajor.len(), n_rows * k_dim, "weight size mismatch");

    if crate::arch::x86_avx512_fp16_kernel_runtime_available() {
        if n_rows < MATVEC_PAR_THRESHOLD {
            let mut best = (0usize, f32::NEG_INFINITY);
            for i in 0..n_rows {
                let row = &w_rowmajor[i * k_dim..(i + 1) * k_dim];
                let score =
                    dot_f16_f16_arch(x, row).unwrap_or_else(|| dot_unrolled_f16_f16_scalar(x, row));
                if score > best.1 {
                    best = (i, score);
                }
            }
            return best.0;
        }

        return (0..n_rows)
            .into_par_iter()
            .map(|i| {
                let row = &w_rowmajor[i * k_dim..(i + 1) * k_dim];
                (
                    i,
                    dot_f16_f16_arch(x, row).unwrap_or_else(|| dot_unrolled_f16_f16_scalar(x, row)),
                )
            })
            .reduce(
                || (0usize, f32::NEG_INFINITY),
                |a, b| if a.1 >= b.1 { a } else { b },
            )
            .0;
    }

    with_f16_input_as_f32(x, |x_f32| {
        matvec_argmax_rowmajor_parallel_f32_f16(x_f32, w_rowmajor, n_rows, k_dim)
    })
}

#[inline]
fn matvec_argmax_rowmajor_parallel_bf16_i8(
    x: &[bf16],
    w_rowmajor: &[i8],
    scale: f32,
    n_rows: usize,
    k_dim: usize,
) -> usize {
    with_bf16_input_as_f32(x, |x_f32| {
        matvec_argmax_rowmajor_parallel_f32_i8(x_f32, w_rowmajor, scale, n_rows, k_dim)
    })
}

#[inline]
fn matvec_argmax_rowmajor_parallel_i8_f32(
    x: &[i8],
    x_scale: f32,
    w_rowmajor: &[f32],
    n_rows: usize,
    k_dim: usize,
) -> usize {
    with_i8_input_as_f32(x, x_scale, |x_f32| {
        matvec_argmax_rowmajor_parallel(x_f32, w_rowmajor, n_rows, k_dim)
    })
}

#[inline]
fn matvec_argmax_rowmajor_parallel_i8_bf16(
    x: &[i8],
    x_scale: f32,
    w_rowmajor: &[bf16],
    n_rows: usize,
    k_dim: usize,
) -> usize {
    with_i8_input_as_f32(x, x_scale, |x_f32| {
        matvec_argmax_rowmajor_parallel_f32_bf16(x_f32, w_rowmajor, n_rows, k_dim)
    })
}

#[inline]
fn matvec_argmax_rowmajor_parallel_i8_i8(
    x: &[i8],
    x_scale: f32,
    w_rowmajor: &[i8],
    w_scale: f32,
    n_rows: usize,
    k_dim: usize,
) -> usize {
    assert_eq!(x.len(), k_dim, "x len / k_dim mismatch");
    assert_eq!(w_rowmajor.len(), n_rows * k_dim, "weight size mismatch");

    let choose_best = |best: &mut (usize, f32), row_idx: usize, score: f32| {
        if score > best.1 {
            *best = (row_idx, score);
        }
    };

    if n_rows < MATVEC_PAR_THRESHOLD {
        let mut best = (0usize, f32::NEG_INFINITY);
        for i in 0..n_rows {
            let row = &w_rowmajor[i * k_dim..(i + 1) * k_dim];
            let score = dot_unrolled_i8_i8(x, x_scale, row, w_scale);
            choose_best(&mut best, i, score);
        }
        best.0
    } else {
        (0..n_rows)
            .into_par_iter()
            .map(|i| {
                let row = &w_rowmajor[i * k_dim..(i + 1) * k_dim];
                (i, dot_unrolled_i8_i8(x, x_scale, row, w_scale))
            })
            .reduce(
                || (0usize, f32::NEG_INFINITY),
                |a, b| if a.1 >= b.1 { a } else { b },
            )
            .0
    }
}

pub fn matvec_argmax_rowmajor_parallel_mixed(
    x: SliceRef<'_>,
    w_rowmajor: SliceRef<'_>,
    n_rows: usize,
    k_dim: usize,
) -> usize {
    match (x, w_rowmajor) {
        (SliceRef::F32(x), SliceRef::F32(w)) => {
            matvec_argmax_rowmajor_parallel(x, w, n_rows, k_dim)
        }
        (SliceRef::F32(x), SliceRef::F16(w)) => {
            matvec_argmax_rowmajor_parallel_f32_f16(x, w, n_rows, k_dim)
        }
        (SliceRef::F32(x), SliceRef::BF16(w)) => {
            matvec_argmax_rowmajor_parallel_f32_bf16(x, w, n_rows, k_dim)
        }
        (SliceRef::F32(x), SliceRef::I8(w, scale)) => {
            matvec_argmax_rowmajor_parallel_f32_i8(x, w, scale, n_rows, k_dim)
        }
        (SliceRef::F16(x), SliceRef::F32(w)) => with_f16_input_as_f32(x, |x_f32| {
            matvec_argmax_rowmajor_parallel(x_f32, w, n_rows, k_dim)
        }),
        (SliceRef::F16(x), SliceRef::F16(w)) => {
            matvec_argmax_rowmajor_parallel_f16_f16(x, w, n_rows, k_dim)
        }
        (SliceRef::F16(x), SliceRef::BF16(w)) => with_f16_input_as_f32(x, |x_f32| {
            matvec_argmax_rowmajor_parallel_f32_bf16(x_f32, w, n_rows, k_dim)
        }),
        (SliceRef::F16(x), SliceRef::I8(w, scale)) => with_f16_input_as_f32(x, |x_f32| {
            matvec_argmax_rowmajor_parallel_f32_i8(x_f32, w, scale, n_rows, k_dim)
        }),
        (SliceRef::BF16(x), SliceRef::F32(w)) => {
            matvec_argmax_rowmajor_parallel_bf16_f32(x, w, n_rows, k_dim)
        }
        (SliceRef::BF16(x), SliceRef::F16(w)) => with_bf16_input_as_f32(x, |x_f32| {
            matvec_argmax_rowmajor_parallel_f32_f16(x_f32, w, n_rows, k_dim)
        }),
        (SliceRef::BF16(x), SliceRef::BF16(w)) => {
            matvec_argmax_rowmajor_parallel_bf16_bf16(x, w, n_rows, k_dim)
        }
        (SliceRef::BF16(x), SliceRef::I8(w, scale)) => {
            matvec_argmax_rowmajor_parallel_bf16_i8(x, w, scale, n_rows, k_dim)
        }
        (SliceRef::I8(x, scale), SliceRef::F32(w)) => {
            matvec_argmax_rowmajor_parallel_i8_f32(x, scale, w, n_rows, k_dim)
        }
        (SliceRef::I8(x, scale), SliceRef::F16(w)) => with_i8_input_as_f32(x, scale, |x_f32| {
            matvec_argmax_rowmajor_parallel_f32_f16(x_f32, w, n_rows, k_dim)
        }),
        (SliceRef::I8(x, scale), SliceRef::BF16(w)) => {
            matvec_argmax_rowmajor_parallel_i8_bf16(x, scale, w, n_rows, k_dim)
        }
        (SliceRef::I8(x, x_scale), SliceRef::I8(w, w_scale)) => {
            matvec_argmax_rowmajor_parallel_i8_i8(x, x_scale, w, w_scale, n_rows, k_dim)
        }
    }
}

fn matmul_rows_f32_bf16(
    a_view: ndarray::ArrayViewD<'_, f32>,
    b_view: ndarray::ArrayViewD<'_, bf16>,
    m_dim: usize,
    k_dim: usize,
    n_dim: usize,
) -> Array2<f32> {
    let b_2d = b_view.into_dimensionality::<Ix2>().unwrap();
    let b_owned;
    let b_slice: &[bf16] = if let Some(s) = b_2d.as_slice() {
        s
    } else {
        b_owned = b_2d.as_standard_layout().to_owned();
        b_owned
            .as_slice()
            .expect("standard-layout matmul RHS should be contiguous")
    };

    if m_dim == 1 {
        let a_flat = a_view
            .clone()
            .into_shape(k_dim)
            .expect("single-row lhs reshape failed");
        let a_owned;
        let a_slice: &[f32] = if let Some(s) = a_flat.as_slice() {
            s
        } else {
            a_owned = a_flat.iter().copied().collect::<Vec<f32>>();
            a_owned.as_slice()
        };
        let mut out_vec = vec![0.0f32; n_dim];
        matvec_rowmajor_parallel_f32_bf16(a_slice, b_slice, n_dim, k_dim, &mut out_vec);
        return Array2::from_shape_vec((1, n_dim), out_vec)
            .expect("decode matvec shape build failed");
    }

    let mut res = Array2::<f32>::zeros((m_dim, n_dim));
    if let Ok(a_2d) = a_view.clone().into_shape((m_dim, k_dim)) {
        Zip::from(res.outer_iter_mut())
            .and(a_2d.outer_iter())
            .par_for_each(|mut out_row, a_row| {
                let a_owned;
                let a_slice: &[f32] = if let Some(s) = a_row.as_slice() {
                    s
                } else {
                    a_owned = a_row.iter().copied().collect::<Vec<f32>>();
                    a_owned.as_slice()
                };
                let out_slice = out_row
                    .as_slice_mut()
                    .expect("matmul output row should be contiguous");
                matvec_rowmajor_serial_f32_bf16(a_slice, b_slice, n_dim, k_dim, out_slice);
            });
    } else {
        let a_2d_owned = a_view
            .to_owned()
            .into_shape((m_dim, k_dim))
            .expect("Reshape A failed");
        Zip::from(res.outer_iter_mut())
            .and(a_2d_owned.outer_iter())
            .par_for_each(|mut out_row, a_row| {
                let a_owned;
                let a_slice: &[f32] = if let Some(s) = a_row.as_slice() {
                    s
                } else {
                    a_owned = a_row.iter().copied().collect::<Vec<f32>>();
                    a_owned.as_slice()
                };
                let out_slice = out_row
                    .as_slice_mut()
                    .expect("matmul output row should be contiguous");
                matvec_rowmajor_serial_f32_bf16(a_slice, b_slice, n_dim, k_dim, out_slice);
            });
    }
    res
}

fn matmul_rows_f32_f16(
    a_view: ndarray::ArrayViewD<'_, f32>,
    b_view: ndarray::ArrayViewD<'_, f16>,
    m_dim: usize,
    k_dim: usize,
    n_dim: usize,
) -> Array2<f32> {
    let b_2d = b_view.into_dimensionality::<Ix2>().unwrap();
    let b_owned;
    let b_slice: &[f16] = if let Some(s) = b_2d.as_slice() {
        s
    } else {
        b_owned = b_2d.as_standard_layout().to_owned();
        b_owned
            .as_slice()
            .expect("standard-layout matmul RHS should be contiguous")
    };

    #[cfg(all(feature = "arm64-fp-kernels", target_arch = "aarch64"))]
    if m_dim == 1 {
        let a_flat = a_view
            .clone()
            .into_shape(k_dim)
            .expect("single-row lhs reshape failed");
        let a_owned;
        let a_slice: &[f32] = if let Some(s) = a_flat.as_slice() {
            s
        } else {
            a_owned = a_flat.iter().copied().collect::<Vec<f32>>();
            a_owned.as_slice()
        };
        let mut out_vec = vec![0.0f32; n_dim];
        matvec_rowmajor_parallel_f32_f16(a_slice, b_slice, n_dim, k_dim, &mut out_vec);
        return Array2::from_shape_vec((1, n_dim), out_vec)
            .expect("decode matvec shape build failed");
    }

    let mut res = Array2::<f32>::zeros((m_dim, n_dim));
    if let Ok(a_2d) = a_view.clone().into_shape((m_dim, k_dim)) {
        Zip::from(res.outer_iter_mut())
            .and(a_2d.outer_iter())
            .par_for_each(|mut out_row, a_row| {
                let a_owned;
                let a_slice: &[f32] = if let Some(s) = a_row.as_slice() {
                    s
                } else {
                    a_owned = a_row.iter().copied().collect::<Vec<f32>>();
                    a_owned.as_slice()
                };
                let out_slice = out_row
                    .as_slice_mut()
                    .expect("matmul output row should be contiguous");
                matvec_rowmajor_serial_f32_f16(a_slice, b_slice, n_dim, k_dim, out_slice);
            });
    } else {
        let a_2d_owned = a_view
            .to_owned()
            .into_shape((m_dim, k_dim))
            .expect("Reshape A failed");
        Zip::from(res.outer_iter_mut())
            .and(a_2d_owned.outer_iter())
            .par_for_each(|mut out_row, a_row| {
                let a_owned;
                let a_slice: &[f32] = if let Some(s) = a_row.as_slice() {
                    s
                } else {
                    a_owned = a_row.iter().copied().collect::<Vec<f32>>();
                    a_owned.as_slice()
                };
                let out_slice = out_row
                    .as_slice_mut()
                    .expect("matmul output row should be contiguous");
                matvec_rowmajor_serial_f32_f16(a_slice, b_slice, n_dim, k_dim, out_slice);
            });
    }
    res
}

fn matmul_rows_f16_f32(
    a_view: ndarray::ArrayViewD<'_, f16>,
    b_view: ndarray::ArrayViewD<'_, f32>,
    m_dim: usize,
    k_dim: usize,
    n_dim: usize,
) -> Array2<f32> {
    let b_2d = b_view.into_dimensionality::<Ix2>().unwrap();
    let b_owned;
    let b_slice: &[f32] = if let Some(s) = b_2d.as_slice() {
        s
    } else {
        b_owned = b_2d.as_standard_layout().to_owned();
        b_owned
            .as_slice()
            .expect("standard-layout matmul RHS should be contiguous")
    };

    #[cfg(all(feature = "arm64-fp-kernels", target_arch = "aarch64"))]
    if m_dim == 1 {
        let a_flat = a_view
            .clone()
            .into_shape(k_dim)
            .expect("single-row lhs reshape failed");
        let a_owned;
        let a_slice: &[f16] = if let Some(s) = a_flat.as_slice() {
            s
        } else {
            a_owned = a_flat.iter().copied().collect::<Vec<f16>>();
            a_owned.as_slice()
        };
        let mut out_vec = vec![0.0f32; n_dim];
        with_f16_input_as_f32(a_slice, |a_f32| {
            matvec_rowmajor_parallel(a_f32, b_slice, n_dim, k_dim, &mut out_vec);
        });
        return Array2::from_shape_vec((1, n_dim), out_vec)
            .expect("decode matvec shape build failed");
    }

    let mut res = Array2::<f32>::zeros((m_dim, n_dim));
    if let Ok(a_2d) = a_view.clone().into_shape((m_dim, k_dim)) {
        Zip::from(res.outer_iter_mut())
            .and(a_2d.outer_iter())
            .par_for_each(|mut out_row, a_row| {
                let a_owned;
                let a_slice: &[f16] = if let Some(s) = a_row.as_slice() {
                    s
                } else {
                    a_owned = a_row.iter().copied().collect::<Vec<f16>>();
                    a_owned.as_slice()
                };
                let out_slice = out_row
                    .as_slice_mut()
                    .expect("matmul output row should be contiguous");
                with_f16_input_as_f32(a_slice, |a_f32| {
                    matvec_rowmajor_serial(a_f32, b_slice, n_dim, k_dim, out_slice);
                });
            });
    } else {
        let a_2d_owned = a_view
            .to_owned()
            .into_shape((m_dim, k_dim))
            .expect("Reshape A failed");
        Zip::from(res.outer_iter_mut())
            .and(a_2d_owned.outer_iter())
            .par_for_each(|mut out_row, a_row| {
                let a_slice = a_row.as_slice().expect("owned row must be contiguous");
                let out_slice = out_row
                    .as_slice_mut()
                    .expect("matmul output row should be contiguous");
                with_f16_input_as_f32(a_slice, |a_f32| {
                    matvec_rowmajor_serial(a_f32, b_slice, n_dim, k_dim, out_slice);
                });
            });
    }
    res
}

fn matmul_rows_f16_f16(
    a_view: ndarray::ArrayViewD<'_, f16>,
    b_view: ndarray::ArrayViewD<'_, f16>,
    m_dim: usize,
    k_dim: usize,
    n_dim: usize,
) -> Array2<f32> {
    let b_2d = b_view.into_dimensionality::<Ix2>().unwrap();
    let b_owned;
    let b_slice: &[f16] = if let Some(s) = b_2d.as_slice() {
        s
    } else {
        b_owned = b_2d.as_standard_layout().to_owned();
        b_owned
            .as_slice()
            .expect("standard-layout matmul RHS should be contiguous")
    };

    #[cfg(all(feature = "arm64-fp-kernels", target_arch = "aarch64"))]
    if m_dim == 1 {
        let a_flat = a_view
            .clone()
            .into_shape(k_dim)
            .expect("single-row lhs reshape failed");
        let a_owned;
        let a_slice: &[f16] = if let Some(s) = a_flat.as_slice() {
            s
        } else {
            a_owned = a_flat.iter().copied().collect::<Vec<f16>>();
            a_owned.as_slice()
        };
        let mut out_vec = vec![0.0f32; n_dim];
        matvec_rowmajor_parallel_f16_f16(a_slice, b_slice, n_dim, k_dim, &mut out_vec);
        return Array2::from_shape_vec((1, n_dim), out_vec)
            .expect("decode matvec shape build failed");
    }

    let mut res = Array2::<f32>::zeros((m_dim, n_dim));
    if let Ok(a_2d) = a_view.clone().into_shape((m_dim, k_dim)) {
        Zip::from(res.outer_iter_mut())
            .and(a_2d.outer_iter())
            .par_for_each(|mut out_row, a_row| {
                let a_owned;
                let a_slice: &[f16] = if let Some(s) = a_row.as_slice() {
                    s
                } else {
                    a_owned = a_row.iter().copied().collect::<Vec<f16>>();
                    a_owned.as_slice()
                };
                let out_slice = out_row
                    .as_slice_mut()
                    .expect("matmul output row should be contiguous");
                if crate::arch::x86_avx512_fp16_kernel_runtime_available() {
                    matvec_rowmajor_parallel_f16_f16(a_slice, b_slice, n_dim, k_dim, out_slice);
                } else {
                    with_f16_input_as_f32(a_slice, |a_f32| {
                        matvec_rowmajor_serial_f32_f16(a_f32, b_slice, n_dim, k_dim, out_slice);
                    });
                }
            });
    } else {
        let a_2d_owned = a_view
            .to_owned()
            .into_shape((m_dim, k_dim))
            .expect("Reshape A failed");
        Zip::from(res.outer_iter_mut())
            .and(a_2d_owned.outer_iter())
            .par_for_each(|mut out_row, a_row| {
                let a_slice = a_row.as_slice().expect("owned row must be contiguous");
                let out_slice = out_row
                    .as_slice_mut()
                    .expect("matmul output row should be contiguous");
                if crate::arch::x86_avx512_fp16_kernel_runtime_available() {
                    matvec_rowmajor_parallel_f16_f16(a_slice, b_slice, n_dim, k_dim, out_slice);
                } else {
                    with_f16_input_as_f32(a_slice, |a_f32| {
                        matvec_rowmajor_serial_f32_f16(a_f32, b_slice, n_dim, k_dim, out_slice);
                    });
                }
            });
    }
    res
}

fn matmul_rows_f16_slice(
    a_view: ndarray::ArrayViewD<'_, f16>,
    b_rowmajor: SliceRef<'_>,
    m_dim: usize,
    k_dim: usize,
    n_dim: usize,
) -> Array2<f32> {
    #[cfg(all(feature = "arm64-fp-kernels", target_arch = "aarch64"))]
    if m_dim == 1 {
        let a_flat = a_view
            .clone()
            .into_shape(k_dim)
            .expect("single-row lhs reshape failed");
        let a_owned;
        let a_slice: &[f16] = if let Some(s) = a_flat.as_slice() {
            s
        } else {
            a_owned = a_flat.iter().copied().collect::<Vec<f16>>();
            a_owned.as_slice()
        };
        let mut out_vec = vec![0.0f32; n_dim];
        matvec_rowmajor_parallel_mixed(
            SliceRef::F16(a_slice),
            b_rowmajor,
            n_dim,
            k_dim,
            &mut out_vec,
        );
        return Array2::from_shape_vec((1, n_dim), out_vec)
            .expect("decode matvec shape build failed");
    }

    let mut res = Array2::<f32>::zeros((m_dim, n_dim));
    if let Ok(a_2d) = a_view.clone().into_shape((m_dim, k_dim)) {
        Zip::from(res.outer_iter_mut())
            .and(a_2d.outer_iter())
            .par_for_each(|mut out_row, a_row| {
                let a_owned;
                let a_slice: &[f16] = if let Some(s) = a_row.as_slice() {
                    s
                } else {
                    a_owned = a_row.iter().copied().collect::<Vec<f16>>();
                    a_owned.as_slice()
                };
                let out_slice = out_row
                    .as_slice_mut()
                    .expect("output row should be contiguous");
                matvec_rowmajor_parallel_mixed(
                    SliceRef::F16(a_slice),
                    b_rowmajor,
                    n_dim,
                    k_dim,
                    out_slice,
                );
            });
    } else {
        let a_2d = a_view
            .to_owned()
            .into_shape((m_dim, k_dim))
            .expect("Reshape A failed");
        Zip::from(res.outer_iter_mut())
            .and(a_2d.outer_iter())
            .par_for_each(|mut out_row, a_row| {
                let a_slice = a_row.as_slice().expect("owned row must be contiguous");
                let out_slice = out_row
                    .as_slice_mut()
                    .expect("output row should be contiguous");
                matvec_rowmajor_parallel_mixed(
                    SliceRef::F16(a_slice),
                    b_rowmajor,
                    n_dim,
                    k_dim,
                    out_slice,
                );
            });
    }
    res
}

fn matmul_rows_i8_slice(
    a_view: ndarray::ArrayViewD<'_, i8>,
    a_scale: f32,
    b_rowmajor: SliceRef<'_>,
    m_dim: usize,
    k_dim: usize,
    n_dim: usize,
) -> Array2<f32> {
    let mut res = Array2::<f32>::zeros((m_dim, n_dim));
    if let Ok(a_2d) = a_view.clone().into_shape((m_dim, k_dim)) {
        Zip::from(res.outer_iter_mut())
            .and(a_2d.outer_iter())
            .par_for_each(|mut out_row, a_row| {
                let a_owned;
                let a_slice: &[i8] = if let Some(s) = a_row.as_slice() {
                    s
                } else {
                    a_owned = a_row.iter().copied().collect::<Vec<i8>>();
                    a_owned.as_slice()
                };
                let out_slice = out_row
                    .as_slice_mut()
                    .expect("output row should be contiguous");
                matvec_rowmajor_parallel_mixed(
                    SliceRef::I8(a_slice, a_scale),
                    b_rowmajor,
                    n_dim,
                    k_dim,
                    out_slice,
                );
            });
    } else {
        let a_2d = a_view
            .to_owned()
            .into_shape((m_dim, k_dim))
            .expect("Reshape A failed");
        Zip::from(res.outer_iter_mut())
            .and(a_2d.outer_iter())
            .par_for_each(|mut out_row, a_row| {
                let a_slice = a_row.as_slice().expect("owned row must be contiguous");
                let out_slice = out_row
                    .as_slice_mut()
                    .expect("output row should be contiguous");
                matvec_rowmajor_parallel_mixed(
                    SliceRef::I8(a_slice, a_scale),
                    b_rowmajor,
                    n_dim,
                    k_dim,
                    out_slice,
                );
            });
    }
    res
}

fn matmul_rows_bf16_slice(
    a_view: ndarray::ArrayViewD<'_, bf16>,
    b_rowmajor: SliceRef<'_>,
    m_dim: usize,
    k_dim: usize,
    n_dim: usize,
) -> Array2<f32> {
    #[cfg(all(feature = "arm64-fp-kernels", target_arch = "aarch64"))]
    if m_dim == 1 {
        let a_flat = a_view
            .clone()
            .into_shape(k_dim)
            .expect("single-row lhs reshape failed");
        let a_owned;
        let a_slice: &[bf16] = if let Some(s) = a_flat.as_slice() {
            s
        } else {
            a_owned = a_flat.iter().copied().collect::<Vec<bf16>>();
            a_owned.as_slice()
        };
        let mut out_vec = vec![0.0f32; n_dim];
        matvec_rowmajor_parallel_mixed(
            SliceRef::BF16(a_slice),
            b_rowmajor,
            n_dim,
            k_dim,
            &mut out_vec,
        );
        return Array2::from_shape_vec((1, n_dim), out_vec)
            .expect("decode matvec shape build failed");
    }

    let mut res = Array2::<f32>::zeros((m_dim, n_dim));
    if let Ok(a_2d) = a_view.clone().into_shape((m_dim, k_dim)) {
        Zip::from(res.outer_iter_mut())
            .and(a_2d.outer_iter())
            .par_for_each(|mut out_row, a_row| {
                let a_owned;
                let a_slice: &[bf16] = if let Some(s) = a_row.as_slice() {
                    s
                } else {
                    a_owned = a_row.iter().copied().collect::<Vec<bf16>>();
                    a_owned.as_slice()
                };
                let out_slice = out_row
                    .as_slice_mut()
                    .expect("output row should be contiguous");
                matvec_rowmajor_parallel_mixed(
                    SliceRef::BF16(a_slice),
                    b_rowmajor,
                    n_dim,
                    k_dim,
                    out_slice,
                );
            });
    } else {
        let a_2d = a_view
            .to_owned()
            .into_shape((m_dim, k_dim))
            .expect("Reshape A failed");
        Zip::from(res.outer_iter_mut())
            .and(a_2d.outer_iter())
            .par_for_each(|mut out_row, a_row| {
                let a_slice = a_row.as_slice().expect("owned row must be contiguous");
                let out_slice = out_row
                    .as_slice_mut()
                    .expect("output row should be contiguous");
                matvec_rowmajor_parallel_mixed(
                    SliceRef::BF16(a_slice),
                    b_rowmajor,
                    n_dim,
                    k_dim,
                    out_slice,
                );
            });
    }
    res
}

fn matmul_rows_bf16_f32(
    a_view: ndarray::ArrayViewD<'_, bf16>,
    b_view: ndarray::ArrayViewD<'_, f32>,
    m_dim: usize,
    k_dim: usize,
    n_dim: usize,
) -> Array2<f32> {
    let b_2d = b_view.into_dimensionality::<Ix2>().unwrap();
    let b_owned;
    let b_slice: &[f32] = if let Some(s) = b_2d.as_slice() {
        s
    } else {
        b_owned = b_2d.as_standard_layout().to_owned();
        b_owned
            .as_slice()
            .expect("standard-layout matmul RHS should be contiguous")
    };

    #[cfg(all(feature = "arm64-fp-kernels", target_arch = "aarch64"))]
    if m_dim == 1 {
        let a_flat = a_view
            .clone()
            .into_shape(k_dim)
            .expect("single-row lhs reshape failed");
        let a_owned;
        let a_slice: &[bf16] = if let Some(s) = a_flat.as_slice() {
            s
        } else {
            a_owned = a_flat.iter().copied().collect::<Vec<bf16>>();
            a_owned.as_slice()
        };
        let mut out_vec = vec![0.0f32; n_dim];
        with_bf16_input_as_f32(a_slice, |a_f32| {
            matvec_rowmajor_parallel(a_f32, b_slice, n_dim, k_dim, &mut out_vec);
        });
        return Array2::from_shape_vec((1, n_dim), out_vec)
            .expect("decode matvec shape build failed");
    }

    let mut res = Array2::<f32>::zeros((m_dim, n_dim));
    if let Ok(a_2d) = a_view.clone().into_shape((m_dim, k_dim)) {
        Zip::from(res.outer_iter_mut())
            .and(a_2d.outer_iter())
            .par_for_each(|mut out_row, a_row| {
                let a_owned;
                let a_slice: &[bf16] = if let Some(s) = a_row.as_slice() {
                    s
                } else {
                    a_owned = a_row.iter().copied().collect::<Vec<bf16>>();
                    a_owned.as_slice()
                };
                let out_slice = out_row
                    .as_slice_mut()
                    .expect("matmul output row should be contiguous");
                with_bf16_input_as_f32(a_slice, |a_f32| {
                    matvec_rowmajor_serial(a_f32, b_slice, n_dim, k_dim, out_slice);
                });
            });
    } else {
        let a_2d_owned = a_view
            .to_owned()
            .into_shape((m_dim, k_dim))
            .expect("Reshape A failed");
        Zip::from(res.outer_iter_mut())
            .and(a_2d_owned.outer_iter())
            .par_for_each(|mut out_row, a_row| {
                let a_owned;
                let a_slice: &[bf16] = if let Some(s) = a_row.as_slice() {
                    s
                } else {
                    a_owned = a_row.iter().copied().collect::<Vec<bf16>>();
                    a_owned.as_slice()
                };
                let out_slice = out_row
                    .as_slice_mut()
                    .expect("matmul output row should be contiguous");
                with_bf16_input_as_f32(a_slice, |a_f32| {
                    matvec_rowmajor_serial(a_f32, b_slice, n_dim, k_dim, out_slice);
                });
            });
    }
    res
}

fn matmul_rows_bf16_bf16(
    a_view: ndarray::ArrayViewD<'_, bf16>,
    b_view: ndarray::ArrayViewD<'_, bf16>,
    m_dim: usize,
    k_dim: usize,
    n_dim: usize,
) -> Array2<f32> {
    let b_2d = b_view.into_dimensionality::<Ix2>().unwrap();
    let b_owned;
    let b_slice: &[bf16] = if let Some(s) = b_2d.as_slice() {
        s
    } else {
        b_owned = b_2d.as_standard_layout().to_owned();
        b_owned
            .as_slice()
            .expect("standard-layout matmul RHS should be contiguous")
    };

    #[cfg(all(feature = "arm64-fp-kernels", target_arch = "aarch64"))]
    if m_dim == 1 {
        let a_flat = a_view
            .clone()
            .into_shape(k_dim)
            .expect("single-row lhs reshape failed");
        let a_owned;
        let a_slice: &[bf16] = if let Some(s) = a_flat.as_slice() {
            s
        } else {
            a_owned = a_flat.iter().copied().collect::<Vec<bf16>>();
            a_owned.as_slice()
        };
        let mut out_vec = vec![0.0f32; n_dim];
        with_bf16_input_as_f32(a_slice, |a_f32| {
            matvec_rowmajor_parallel_f32_bf16(a_f32, b_slice, n_dim, k_dim, &mut out_vec);
        });
        return Array2::from_shape_vec((1, n_dim), out_vec)
            .expect("decode matvec shape build failed");
    }

    let mut res = Array2::<f32>::zeros((m_dim, n_dim));
    if let Ok(a_2d) = a_view.clone().into_shape((m_dim, k_dim)) {
        Zip::from(res.outer_iter_mut())
            .and(a_2d.outer_iter())
            .par_for_each(|mut out_row, a_row| {
                let a_owned;
                let a_slice: &[bf16] = if let Some(s) = a_row.as_slice() {
                    s
                } else {
                    a_owned = a_row.iter().copied().collect::<Vec<bf16>>();
                    a_owned.as_slice()
                };
                let out_slice = out_row
                    .as_slice_mut()
                    .expect("matmul output row should be contiguous");
                with_bf16_input_as_f32(a_slice, |a_f32| {
                    matvec_rowmajor_serial_f32_bf16(a_f32, b_slice, n_dim, k_dim, out_slice);
                });
            });
    } else {
        let a_2d_owned = a_view
            .to_owned()
            .into_shape((m_dim, k_dim))
            .expect("Reshape A failed");
        Zip::from(res.outer_iter_mut())
            .and(a_2d_owned.outer_iter())
            .par_for_each(|mut out_row, a_row| {
                let a_owned;
                let a_slice: &[bf16] = if let Some(s) = a_row.as_slice() {
                    s
                } else {
                    a_owned = a_row.iter().copied().collect::<Vec<bf16>>();
                    a_owned.as_slice()
                };
                let out_slice = out_row
                    .as_slice_mut()
                    .expect("matmul output row should be contiguous");
                with_bf16_input_as_f32(a_slice, |a_f32| {
                    matvec_rowmajor_serial_f32_bf16(a_f32, b_slice, n_dim, k_dim, out_slice);
                });
            });
    }
    res
}

fn matmul_rows_f32_i8(
    a_view: ndarray::ArrayViewD<'_, f32>,
    b_view: ndarray::ArrayViewD<'_, i8>,
    scale: f32,
    m_dim: usize,
    k_dim: usize,
    n_dim: usize,
) -> Array2<f32> {
    let b_2d = b_view.into_dimensionality::<Ix2>().unwrap();
    let b_owned;
    let b_slice: &[i8] = if let Some(s) = b_2d.as_slice() {
        s
    } else {
        b_owned = b_2d.as_standard_layout().to_owned();
        b_owned
            .as_slice()
            .expect("standard-layout matmul RHS should be contiguous")
    };

    let mut res = Array2::<f32>::zeros((m_dim, n_dim));
    if let Ok(a_2d) = a_view.clone().into_shape((m_dim, k_dim)) {
        Zip::from(res.outer_iter_mut())
            .and(a_2d.outer_iter())
            .par_for_each(|mut out_row, a_row| {
                let a_owned;
                let a_slice: &[f32] = if let Some(s) = a_row.as_slice() {
                    s
                } else {
                    a_owned = a_row.iter().copied().collect::<Vec<f32>>();
                    a_owned.as_slice()
                };
                let out_slice = out_row
                    .as_slice_mut()
                    .expect("matmul output row should be contiguous");
                matvec_rowmajor_serial_f32_i8(a_slice, b_slice, scale, n_dim, k_dim, out_slice);
            });
    } else {
        let a_2d_owned = a_view
            .to_owned()
            .into_shape((m_dim, k_dim))
            .expect("Reshape A failed");
        Zip::from(res.outer_iter_mut())
            .and(a_2d_owned.outer_iter())
            .par_for_each(|mut out_row, a_row| {
                let a_owned;
                let a_slice: &[f32] = if let Some(s) = a_row.as_slice() {
                    s
                } else {
                    a_owned = a_row.iter().copied().collect::<Vec<f32>>();
                    a_owned.as_slice()
                };
                let out_slice = out_row
                    .as_slice_mut()
                    .expect("matmul output row should be contiguous");
                matvec_rowmajor_serial_f32_i8(a_slice, b_slice, scale, n_dim, k_dim, out_slice);
            });
    }
    res
}

fn matmul_rows_bf16_i8(
    a_view: ndarray::ArrayViewD<'_, bf16>,
    b_view: ndarray::ArrayViewD<'_, i8>,
    scale: f32,
    m_dim: usize,
    k_dim: usize,
    n_dim: usize,
) -> Array2<f32> {
    let b_2d = b_view.into_dimensionality::<Ix2>().unwrap();
    let b_owned;
    let b_slice: &[i8] = if let Some(s) = b_2d.as_slice() {
        s
    } else {
        b_owned = b_2d.as_standard_layout().to_owned();
        b_owned
            .as_slice()
            .expect("standard-layout matmul RHS should be contiguous")
    };

    let mut res = Array2::<f32>::zeros((m_dim, n_dim));
    if let Ok(a_2d) = a_view.clone().into_shape((m_dim, k_dim)) {
        Zip::from(res.outer_iter_mut())
            .and(a_2d.outer_iter())
            .par_for_each(|mut out_row, a_row| {
                let a_owned;
                let a_slice: &[bf16] = if let Some(s) = a_row.as_slice() {
                    s
                } else {
                    a_owned = a_row.iter().copied().collect::<Vec<bf16>>();
                    a_owned.as_slice()
                };
                let out_slice = out_row
                    .as_slice_mut()
                    .expect("matmul output row should be contiguous");
                with_bf16_input_as_f32(a_slice, |a_f32| {
                    matvec_rowmajor_serial_f32_i8(a_f32, b_slice, scale, n_dim, k_dim, out_slice);
                });
            });
    } else {
        let a_2d_owned = a_view
            .to_owned()
            .into_shape((m_dim, k_dim))
            .expect("Reshape A failed");
        Zip::from(res.outer_iter_mut())
            .and(a_2d_owned.outer_iter())
            .par_for_each(|mut out_row, a_row| {
                let a_owned;
                let a_slice: &[bf16] = if let Some(s) = a_row.as_slice() {
                    s
                } else {
                    a_owned = a_row.iter().copied().collect::<Vec<bf16>>();
                    a_owned.as_slice()
                };
                let out_slice = out_row
                    .as_slice_mut()
                    .expect("matmul output row should be contiguous");
                with_bf16_input_as_f32(a_slice, |a_f32| {
                    matvec_rowmajor_serial_f32_i8(a_f32, b_slice, scale, n_dim, k_dim, out_slice);
                });
            });
    }
    res
}

fn try_cuda_matmul_buffer(
    a: &Tensor,
    b: &Tensor,
    m_dim: usize,
    k_dim: usize,
    n_dim: usize,
) -> Option<cuda::CudaBuffer> {
    if a.device() != crate::autograd::Device::Cuda || b.device() != crate::autograd::Device::Cuda {
        return None;
    }
    if let Some(buffer) = try_cuda_matmul_native_low_precision_buffer(a, b, m_dim, k_dim, n_dim) {
        return Some(buffer);
    }
    let force_cuda = is_strict_device_execution();
    if !force_cuda && !cuda::should_accelerate_matmul(m_dim, n_dim, k_dim) {
        return None;
    }
    let cuda_out = a.with_cuda_f32_buffer(|a_buf| {
        b.with_cuda_f32_buffer(|b_buf| cuda::matmul_f32_no_host(a_buf, b_buf, m_dim, n_dim, k_dim))
    });
    Some(match cuda_out {
        Ok(out) => out,
        Err(err) => {
            if force_cuda {
                panic!("CUDA matmul failed in strict device execution mode: {err}");
            }
            return None;
        }
    })
}

fn try_cuda_matmul_native_low_precision_buffer(
    a: &Tensor,
    b: &Tensor,
    m_dim: usize,
    k_dim: usize,
    n_dim: usize,
) -> Option<cuda::CudaBuffer> {
    let a_native = a.cloned_cuda_native_lowp_buffer();
    let b_native = b.cloned_cuda_native_lowp_buffer();

    if let (Some((DType::BF16, a_buffer, _)), Some((DType::I8, b_buffer, Some(b_scale)))) =
        (&a_native, &b_native)
    {
        return cuda::matmul_bf16_i8_buffer_no_host(
            a_buffer, b_buffer, *b_scale, m_dim, n_dim, k_dim,
        )
        .ok();
    }

    if let (Some((DType::F16, a_buffer, _)), Some((DType::I8, b_buffer, Some(b_scale)))) =
        (&a_native, &b_native)
    {
        return cuda::matmul_f16_i8_buffer_no_host(
            a_buffer, b_buffer, *b_scale, m_dim, n_dim, k_dim,
        )
        .ok();
    }

    if a.dtype() == DType::F32
        && let (Some(a_buffer), Some((DType::I8, b_buffer, Some(b_scale)))) =
            (a.cloned_cuda_f32_buffer(), &b_native)
    {
        return cuda::matmul_f32_i8_buffer_no_host(
            &a_buffer, b_buffer, *b_scale, m_dim, n_dim, k_dim,
        )
        .ok();
    }

    if let (Some((DType::I8, a_buffer, Some(a_scale))), Some((DType::BF16, b_buffer, _))) =
        (&a_native, &b_native)
    {
        return cuda::matmul_i8_bf16_buffer_no_host(
            a_buffer, *a_scale, b_buffer, m_dim, n_dim, k_dim,
        )
        .ok();
    }

    if let (Some((DType::I8, a_buffer, Some(a_scale))), Some((DType::F16, b_buffer, _))) =
        (&a_native, &b_native)
    {
        return cuda::matmul_i8_f16_buffer_no_host(
            a_buffer, *a_scale, b_buffer, m_dim, n_dim, k_dim,
        )
        .ok();
    }

    if b.dtype() == DType::F32
        && let (Some((DType::I8, a_buffer, Some(a_scale))), Some(b_buffer)) =
            (&a_native, b.cloned_cuda_f32_buffer())
    {
        return cuda::matmul_i8_f32_buffer_no_host(
            a_buffer, *a_scale, &b_buffer, m_dim, n_dim, k_dim,
        )
        .ok();
    }

    if let (Some((a_dtype, a_buffer, a_scale)), Some((b_dtype, b_buffer, b_scale))) =
        (&a_native, &b_native)
        && a_dtype == b_dtype
    {
        return match a_dtype {
            DType::BF16 => {
                cuda::matmul_bf16_buffer_no_host(a_buffer, b_buffer, m_dim, n_dim, k_dim).ok()
            }
            DType::F16 => {
                cuda::matmul_f16_buffer_no_host(a_buffer, b_buffer, m_dim, n_dim, k_dim).ok()
            }
            DType::I8 => cuda::matmul_i8_buffer_no_host(
                a_buffer,
                (*a_scale)?,
                b_buffer,
                (*b_scale)?,
                m_dim,
                n_dim,
                k_dim,
            )
            .ok(),
            DType::F32 => None,
        };
    }

    match (a.native_storage_owned(), b.native_storage_owned()) {
        (TensorStorageOwned::BF16(a_data), TensorStorageOwned::BF16(b_data)) => {
            let a_bits = a_data.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
            let b_bits = b_data.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
            cuda::matmul_bf16_host_no_host(&a_bits, &b_bits, m_dim, n_dim, k_dim).ok()
        }
        (TensorStorageOwned::F16(a_data), TensorStorageOwned::F16(b_data)) => {
            let a_bits = a_data.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
            let b_bits = b_data.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
            cuda::matmul_f16_host_no_host(&a_bits, &b_bits, m_dim, n_dim, k_dim).ok()
        }
        (TensorStorageOwned::I8(a_data, a_scale), TensorStorageOwned::I8(b_data, b_scale)) => {
            let a_values = a_data.iter().copied().collect::<Vec<_>>();
            let b_values = b_data.iter().copied().collect::<Vec<_>>();
            cuda::matmul_i8_host_no_host(
                &a_values, a_scale, &b_values, b_scale, m_dim, n_dim, k_dim,
            )
            .ok()
        }
        _ => None,
    }
}

fn try_cuda_matmul(
    a: &Tensor,
    b: &Tensor,
    m_dim: usize,
    n_dim: usize,
    k_dim: usize,
    output_dtype: DType,
) -> Option<Tensor> {
    let mut out_shape = a.shape_vec();
    let last_idx = out_shape.len() - 1;
    out_shape[last_idx] = n_dim;

    if output_dtype == DType::I8
        && let Some((buffer, scale)) =
            try_cuda_matmul_i8_typed_output_buffer(a, b, m_dim, k_dim, n_dim)
    {
        return Some(Tensor::from_cuda_native_lowp_buffer_no_host_with_dtype(
            &out_shape,
            buffer,
            a.device(),
            output_dtype,
            Some(scale),
        ));
    }

    if matches!(output_dtype, DType::F16 | DType::BF16)
        && let Some(buffer) = try_cuda_matmul_native_low_precision_typed_output_buffer(
            a,
            b,
            m_dim,
            k_dim,
            n_dim,
            output_dtype,
        )
    {
        return Some(Tensor::from_cuda_native_lowp_buffer_no_host_with_dtype(
            &out_shape,
            buffer,
            a.device(),
            output_dtype,
            None,
        ));
    }

    let buffer = try_cuda_matmul_buffer(a, b, m_dim, k_dim, n_dim)?;

    if output_dtype == DType::F32 {
        return Some(Tensor::from_cuda_f32_buffer_no_host(
            &out_shape,
            buffer,
            a.device(),
        ));
    }

    if matches!(output_dtype, DType::F16 | DType::BF16) {
        return Some(Tensor::from_cuda_f32_buffer_no_host_with_dtype(
            &out_shape,
            buffer,
            a.device(),
            output_dtype,
        ));
    }

    if output_dtype == DType::I8 {
        let (i8_buffer, scale) = cuda::quantize_f32_to_i8_dynamic_no_host(&buffer).ok()?;
        return Some(Tensor::from_cuda_native_lowp_buffer_no_host_with_dtype(
            &out_shape,
            i8_buffer,
            a.device(),
            output_dtype,
            Some(scale),
        ));
    }

    None
}

fn try_cuda_matmul_i8_typed_output_buffer(
    a: &Tensor,
    b: &Tensor,
    m_dim: usize,
    k_dim: usize,
    n_dim: usize,
) -> Option<(cuda::CudaBuffer, f32)> {
    if a.device() != crate::autograd::Device::Cuda || b.device() != crate::autograd::Device::Cuda {
        return None;
    }
    let (Some((DType::I8, a_buffer, Some(a_scale))), Some((DType::I8, b_buffer, Some(b_scale)))) = (
        a.cloned_cuda_native_lowp_buffer(),
        b.cloned_cuda_native_lowp_buffer(),
    ) else {
        return None;
    };
    cuda::matmul_i8_typed_output_buffer_no_host(
        &a_buffer, a_scale, &b_buffer, b_scale, m_dim, n_dim, k_dim,
    )
    .ok()
}

fn try_cuda_matmul_native_low_precision_typed_output_buffer(
    a: &Tensor,
    b: &Tensor,
    m_dim: usize,
    k_dim: usize,
    n_dim: usize,
    output_dtype: DType,
) -> Option<cuda::CudaBuffer> {
    if a.device() != crate::autograd::Device::Cuda || b.device() != crate::autograd::Device::Cuda {
        return None;
    }
    let (Some((a_dtype, a_buffer, _)), Some((b_dtype, b_buffer, _))) = (
        a.cloned_cuda_native_lowp_buffer(),
        b.cloned_cuda_native_lowp_buffer(),
    ) else {
        return None;
    };
    if a_dtype != b_dtype || a_dtype != output_dtype {
        return None;
    }

    match a_dtype {
        DType::BF16 => {
            cuda::matmul_bf16_typed_output_buffer_no_host(&a_buffer, &b_buffer, m_dim, n_dim, k_dim)
                .ok()
        }
        DType::F16 => {
            cuda::matmul_f16_typed_output_buffer_no_host(&a_buffer, &b_buffer, m_dim, n_dim, k_dim)
                .ok()
        }
        DType::F32 | DType::I8 => None,
    }
}

fn try_cuda_batch_matmul_buffer(
    lhs: &Tensor,
    rhs: &Tensor,
    b: usize,
    h: usize,
    m: usize,
    k: usize,
    n: usize,
) -> Option<cuda::CudaBuffer> {
    if lhs.device() != crate::autograd::Device::Cuda
        || rhs.device() != crate::autograd::Device::Cuda
    {
        return None;
    }
    let batch_count = b.checked_mul(h)?;
    if let Some(buffer) =
        try_cuda_batch_matmul_native_low_precision_buffer(lhs, rhs, batch_count, m, k, n)
    {
        return Some(buffer);
    }
    let force_cuda = is_strict_device_execution();
    if !force_cuda && !cuda::should_accelerate_batch_matmul(batch_count, m, n, k) {
        return None;
    }
    let cuda_out = lhs.with_cuda_f32_buffer(|lhs_buf| {
        rhs.with_cuda_f32_buffer(|rhs_buf| {
            cuda::batch_matmul_f32_no_host(lhs_buf, rhs_buf, batch_count, m, n, k)
        })
    });
    Some(match cuda_out {
        Ok(out) => out,
        Err(err) => {
            if force_cuda {
                panic!("CUDA batch_matmul failed in strict device execution mode: {err}");
            }
            return None;
        }
    })
}

fn try_cuda_batch_matmul_native_low_precision_buffer(
    lhs: &Tensor,
    rhs: &Tensor,
    batch_count: usize,
    m: usize,
    k: usize,
    n: usize,
) -> Option<cuda::CudaBuffer> {
    let rhs_native = rhs.cloned_cuda_native_lowp_buffer();
    if let (Some((DType::BF16, lhs_buffer, _)), Some((DType::I8, rhs_buffer, Some(rhs_scale)))) =
        (lhs.cloned_cuda_native_lowp_buffer(), &rhs_native)
    {
        return cuda::batch_matmul_bf16_i8_buffer_no_host(
            &lhs_buffer,
            rhs_buffer,
            *rhs_scale,
            batch_count,
            m,
            n,
            k,
        )
        .ok();
    }

    if let (Some((DType::F16, lhs_buffer, _)), Some((DType::I8, rhs_buffer, Some(rhs_scale)))) =
        (lhs.cloned_cuda_native_lowp_buffer(), &rhs_native)
    {
        return cuda::batch_matmul_f16_i8_buffer_no_host(
            &lhs_buffer,
            rhs_buffer,
            *rhs_scale,
            batch_count,
            m,
            n,
            k,
        )
        .ok();
    }

    if lhs.dtype() == DType::F32
        && let (Some(lhs_buffer), Some((DType::I8, rhs_buffer, Some(rhs_scale)))) =
            (lhs.cloned_cuda_f32_buffer(), &rhs_native)
    {
        return cuda::batch_matmul_f32_i8_buffer_no_host(
            &lhs_buffer,
            rhs_buffer,
            *rhs_scale,
            batch_count,
            m,
            n,
            k,
        )
        .ok();
    }

    let lhs_native = lhs.cloned_cuda_native_lowp_buffer();
    if let (Some((DType::I8, lhs_buffer, Some(lhs_scale))), Some((DType::BF16, rhs_buffer, _))) =
        (&lhs_native, &rhs_native)
    {
        return cuda::batch_matmul_i8_bf16_buffer_no_host(
            lhs_buffer,
            *lhs_scale,
            rhs_buffer,
            batch_count,
            m,
            n,
            k,
        )
        .ok();
    }

    if let (Some((DType::I8, lhs_buffer, Some(lhs_scale))), Some((DType::F16, rhs_buffer, _))) =
        (&lhs_native, &rhs_native)
    {
        return cuda::batch_matmul_i8_f16_buffer_no_host(
            lhs_buffer,
            *lhs_scale,
            rhs_buffer,
            batch_count,
            m,
            n,
            k,
        )
        .ok();
    }

    if rhs.dtype() == DType::F32
        && let (Some((DType::I8, lhs_buffer, Some(lhs_scale))), Some(rhs_buffer)) =
            (&lhs_native, rhs.cloned_cuda_f32_buffer())
    {
        return cuda::batch_matmul_i8_f32_buffer_no_host(
            lhs_buffer,
            *lhs_scale,
            &rhs_buffer,
            batch_count,
            m,
            n,
            k,
        )
        .ok();
    }

    let (Some((lhs_dtype, lhs_buffer, lhs_scale)), Some((rhs_dtype, rhs_buffer, rhs_scale))) =
        (lhs_native, rhs_native)
    else {
        return None;
    };
    if lhs_dtype != rhs_dtype {
        return None;
    }

    match lhs_dtype {
        DType::BF16 => {
            cuda::batch_matmul_bf16_buffer_no_host(&lhs_buffer, &rhs_buffer, batch_count, m, n, k)
                .ok()
        }
        DType::F16 => {
            cuda::batch_matmul_f16_buffer_no_host(&lhs_buffer, &rhs_buffer, batch_count, m, n, k)
                .ok()
        }
        DType::I8 => cuda::batch_matmul_i8_buffer_no_host(
            &lhs_buffer,
            lhs_scale?,
            &rhs_buffer,
            rhs_scale?,
            batch_count,
            m,
            n,
            k,
        )
        .ok(),
        DType::F32 => None,
    }
}

fn try_cuda_batch_matmul(
    lhs: &Tensor,
    rhs: &Tensor,
    dims: BatchMatmulDims,
    output_dtype: DType,
) -> Option<Tensor> {
    let out_shape = dims.out_shape();
    if output_dtype == DType::I8
        && let Some((buffer, scale)) = try_cuda_batch_matmul_i8_typed_output_buffer(lhs, rhs, dims)
    {
        return Some(Tensor::from_cuda_native_lowp_buffer_no_host_with_dtype(
            &out_shape,
            buffer,
            lhs.device(),
            output_dtype,
            Some(scale),
        ));
    }

    if matches!(output_dtype, DType::F16 | DType::BF16)
        && let Some(buffer) = try_cuda_batch_matmul_native_low_precision_typed_output_buffer(
            lhs,
            rhs,
            dims,
            output_dtype,
        )
    {
        return Some(Tensor::from_cuda_native_lowp_buffer_no_host_with_dtype(
            &out_shape,
            buffer,
            lhs.device(),
            output_dtype,
            None,
        ));
    }

    let buffer = try_cuda_batch_matmul_buffer(lhs, rhs, dims.b, dims.h, dims.m, dims.k, dims.n)?;
    if output_dtype == DType::F32 {
        return Some(Tensor::from_cuda_f32_buffer_no_host(
            &out_shape,
            buffer,
            lhs.device(),
        ));
    }

    if matches!(output_dtype, DType::F16 | DType::BF16) {
        return Some(Tensor::from_cuda_f32_buffer_no_host_with_dtype(
            &out_shape,
            buffer,
            lhs.device(),
            output_dtype,
        ));
    }

    if output_dtype == DType::I8 {
        let (i8_buffer, scale) = cuda::quantize_f32_to_i8_dynamic_no_host(&buffer).ok()?;
        return Some(Tensor::from_cuda_native_lowp_buffer_no_host_with_dtype(
            &out_shape,
            i8_buffer,
            lhs.device(),
            output_dtype,
            Some(scale),
        ));
    }

    None
}

fn try_cuda_batch_matmul_i8_typed_output_buffer(
    lhs: &Tensor,
    rhs: &Tensor,
    dims: BatchMatmulDims,
) -> Option<(cuda::CudaBuffer, f32)> {
    if lhs.device() != crate::autograd::Device::Cuda
        || rhs.device() != crate::autograd::Device::Cuda
    {
        return None;
    }
    let (
        Some((DType::I8, lhs_buffer, Some(lhs_scale))),
        Some((DType::I8, rhs_buffer, Some(rhs_scale))),
    ) = (
        lhs.cloned_cuda_native_lowp_buffer(),
        rhs.cloned_cuda_native_lowp_buffer(),
    )
    else {
        return None;
    };
    let batch_count = dims.batch_count()?;
    cuda::batch_matmul_i8_typed_output_buffer_no_host(
        &lhs_buffer,
        lhs_scale,
        &rhs_buffer,
        rhs_scale,
        batch_count,
        dims.m,
        dims.n,
        dims.k,
    )
    .ok()
}

fn try_cuda_batch_matmul_native_low_precision_typed_output_buffer(
    lhs: &Tensor,
    rhs: &Tensor,
    dims: BatchMatmulDims,
    output_dtype: DType,
) -> Option<cuda::CudaBuffer> {
    if lhs.device() != crate::autograd::Device::Cuda
        || rhs.device() != crate::autograd::Device::Cuda
    {
        return None;
    }
    let batch_count = dims.batch_count()?;
    let (Some((lhs_dtype, lhs_buffer, _)), Some((rhs_dtype, rhs_buffer, _))) = (
        lhs.cloned_cuda_native_lowp_buffer(),
        rhs.cloned_cuda_native_lowp_buffer(),
    ) else {
        return None;
    };
    if lhs_dtype != rhs_dtype || lhs_dtype != output_dtype {
        return None;
    }

    match lhs_dtype {
        DType::BF16 => cuda::batch_matmul_bf16_typed_output_buffer_no_host(
            &lhs_buffer,
            &rhs_buffer,
            batch_count,
            dims.m,
            dims.n,
            dims.k,
        )
        .ok(),
        DType::F16 => cuda::batch_matmul_f16_typed_output_buffer_no_host(
            &lhs_buffer,
            &rhs_buffer,
            batch_count,
            dims.m,
            dims.n,
            dims.k,
        )
        .ok(),
        DType::F32 | DType::I8 => None,
    }
}

fn try_cuda_training_matmul_backward(
    grad: &ndarray::ArrayViewD<'_, f32>,
    cuda_grad: Option<cuda::CudaBuffer>,
    a: &Tensor,
    b: &Tensor,
    m_dim: usize,
    k_dim: usize,
    n_dim: usize,
) -> Result<CudaMatmulBackwardHostDevice, String> {
    let (da_buf, db_buf) =
        try_cuda_training_matmul_backward_buffers(grad, cuda_grad, a, b, m_dim, k_dim, n_dim)?;
    let da_host = cuda::download_f32(&da_buf)?;
    let db_host = cuda::download_f32(&db_buf)?;

    let da = Array2::from_shape_vec((m_dim, k_dim), da_host)
        .expect("CUDA matmul backward dA shape build failed")
        .into_shape(a.shape_vec())
        .expect("CUDA matmul backward dA reshape failed")
        .into_dyn();
    let db = Array2::from_shape_vec((n_dim, k_dim), db_host)
        .expect("CUDA matmul backward dB shape build failed")
        .into_dyn();
    Ok(((da, da_buf), (db, db_buf)))
}

fn try_cuda_training_matmul_backward_buffers(
    grad: &ndarray::ArrayViewD<'_, f32>,
    cuda_grad: Option<cuda::CudaBuffer>,
    a: &Tensor,
    b: &Tensor,
    m_dim: usize,
    k_dim: usize,
    n_dim: usize,
) -> Result<(cuda::CudaBuffer, cuda::CudaBuffer), String> {
    let grad_buf = match cuda_grad {
        Some(buffer) if buffer.len() == grad.len() => buffer,
        _ => cuda::upload_f32(&grad.iter().copied().collect::<Vec<_>>())?,
    };

    if let (Some((DType::BF16, a_buffer, _)), Some((DType::I8, b_buffer, b_scale))) = (
        a.cloned_cuda_native_lowp_buffer(),
        b.cloned_cuda_native_lowp_buffer(),
    ) {
        return cuda::matmul_backward_bf16_i8_no_host(
            &grad_buf,
            &a_buffer,
            &b_buffer,
            b_scale.ok_or_else(|| "CUDA I8 rhs scale missing".to_string())?,
            m_dim,
            k_dim,
            n_dim,
        );
    }

    if let (Some((DType::F16, a_buffer, _)), Some((DType::I8, b_buffer, b_scale))) = (
        a.cloned_cuda_native_lowp_buffer(),
        b.cloned_cuda_native_lowp_buffer(),
    ) {
        return cuda::matmul_backward_f16_i8_no_host(
            &grad_buf,
            &a_buffer,
            &b_buffer,
            b_scale.ok_or_else(|| "CUDA I8 rhs scale missing".to_string())?,
            m_dim,
            k_dim,
            n_dim,
        );
    }

    if let Some((DType::I8, b_buffer, b_scale)) = b.cloned_cuda_native_lowp_buffer()
        && a.dtype() == DType::F32
    {
        let a_buffer = a
            .cloned_cuda_f32_buffer()
            .ok_or_else(|| "CUDA F32xI8 matmul backward expected lhs buffer".to_string())?;
        return cuda::matmul_backward_f32_i8_no_host(
            &grad_buf,
            &a_buffer,
            &b_buffer,
            b_scale.ok_or_else(|| "CUDA I8 rhs scale missing".to_string())?,
            m_dim,
            k_dim,
            n_dim,
        );
    }

    if let (Some((DType::I8, a_buffer, a_scale)), Some((DType::BF16, b_buffer, _))) = (
        a.cloned_cuda_native_lowp_buffer(),
        b.cloned_cuda_native_lowp_buffer(),
    ) {
        return cuda::matmul_backward_i8_bf16_no_host(
            &grad_buf,
            &a_buffer,
            a_scale.ok_or_else(|| "CUDA I8 lhs scale missing".to_string())?,
            &b_buffer,
            m_dim,
            k_dim,
            n_dim,
        );
    }

    if let (Some((DType::I8, a_buffer, a_scale)), Some((DType::F16, b_buffer, _))) = (
        a.cloned_cuda_native_lowp_buffer(),
        b.cloned_cuda_native_lowp_buffer(),
    ) {
        return cuda::matmul_backward_i8_f16_no_host(
            &grad_buf,
            &a_buffer,
            a_scale.ok_or_else(|| "CUDA I8 lhs scale missing".to_string())?,
            &b_buffer,
            m_dim,
            k_dim,
            n_dim,
        );
    }

    if let Some((DType::I8, a_buffer, a_scale)) = a.cloned_cuda_native_lowp_buffer()
        && b.dtype() == DType::F32
    {
        let b_buffer = b
            .cloned_cuda_f32_buffer()
            .ok_or_else(|| "CUDA I8xF32 matmul backward expected rhs buffer".to_string())?;
        return cuda::matmul_backward_i8_f32_no_host(
            &grad_buf,
            &a_buffer,
            a_scale.ok_or_else(|| "CUDA I8 lhs scale missing".to_string())?,
            &b_buffer,
            m_dim,
            k_dim,
            n_dim,
        );
    }

    let a_buf = a
        .cloned_cuda_f32_buffer()
        .ok_or_else(|| "CUDA matmul backward expected lhs resident buffer".to_string())?;
    let b_buf = b
        .cloned_cuda_f32_buffer()
        .ok_or_else(|| "CUDA matmul backward expected rhs resident buffer".to_string())?;

    cuda::matmul_backward_f32_no_host(&grad_buf, &a_buf, &b_buf, m_dim, k_dim, n_dim)
}

fn try_cuda_training_batch_matmul_backward(
    grad: &ndarray::ArrayViewD<'_, f32>,
    cuda_grad: Option<cuda::CudaBuffer>,
    lhs: &Tensor,
    rhs: &Tensor,
    dims: BatchMatmulDims,
) -> Result<CudaMatmulBackwardHostDevice, String> {
    let (d_lhs_buf, d_rhs_buf) =
        try_cuda_training_batch_matmul_backward_buffers(grad, cuda_grad, lhs, rhs, dims)?;
    let d_lhs_host = cuda::download_f32(&d_lhs_buf)?;
    let d_rhs_host = cuda::download_f32(&d_rhs_buf)?;

    let d_lhs = Array4::from_shape_vec((dims.b, dims.h, dims.m, dims.k), d_lhs_host)
        .expect("CUDA batch_matmul backward dLHS shape build failed")
        .into_dyn();
    let d_rhs = Array4::from_shape_vec((dims.b, dims.h, dims.k, dims.n), d_rhs_host)
        .expect("CUDA batch_matmul backward dRHS shape build failed")
        .into_dyn();
    Ok(((d_lhs, d_lhs_buf), (d_rhs, d_rhs_buf)))
}

fn try_cuda_training_batch_matmul_backward_buffers(
    grad: &ndarray::ArrayViewD<'_, f32>,
    cuda_grad: Option<cuda::CudaBuffer>,
    lhs: &Tensor,
    rhs: &Tensor,
    dims: BatchMatmulDims,
) -> Result<(cuda::CudaBuffer, cuda::CudaBuffer), String> {
    let batch_count = dims
        .batch_count()
        .ok_or_else(|| "CUDA batch_matmul backward batch count overflow".to_string())?;
    let grad_buf = match cuda_grad {
        Some(buffer) if buffer.len() == grad.len() => buffer,
        _ => cuda::upload_f32(&grad.iter().copied().collect::<Vec<_>>())?,
    };

    if let (Some((DType::BF16, lhs_buffer, _)), Some((DType::I8, rhs_buffer, rhs_scale))) = (
        lhs.cloned_cuda_native_lowp_buffer(),
        rhs.cloned_cuda_native_lowp_buffer(),
    ) {
        return cuda::batch_matmul_backward_bf16_i8_no_host(
            &grad_buf,
            &lhs_buffer,
            &rhs_buffer,
            rhs_scale.ok_or_else(|| "CUDA I8 rhs scale missing".to_string())?,
            batch_count,
            dims.m,
            dims.k,
            dims.n,
        );
    }

    if let (Some((DType::F16, lhs_buffer, _)), Some((DType::I8, rhs_buffer, rhs_scale))) = (
        lhs.cloned_cuda_native_lowp_buffer(),
        rhs.cloned_cuda_native_lowp_buffer(),
    ) {
        return cuda::batch_matmul_backward_f16_i8_no_host(
            &grad_buf,
            &lhs_buffer,
            &rhs_buffer,
            rhs_scale.ok_or_else(|| "CUDA I8 rhs scale missing".to_string())?,
            dims.batch_count()
                .ok_or_else(|| "CUDA batch_matmul backward batch size overflow".to_string())?,
            dims.m,
            dims.k,
            dims.n,
        );
    }

    if let Some((DType::I8, rhs_buffer, rhs_scale)) = rhs.cloned_cuda_native_lowp_buffer()
        && lhs.dtype() == DType::F32
    {
        let lhs_buffer = lhs
            .cloned_cuda_f32_buffer()
            .ok_or_else(|| "CUDA F32xI8 batch_matmul backward expected lhs buffer".to_string())?;
        return cuda::batch_matmul_backward_f32_i8_no_host(
            &grad_buf,
            &lhs_buffer,
            &rhs_buffer,
            rhs_scale.ok_or_else(|| "CUDA I8 rhs scale missing".to_string())?,
            batch_count,
            dims.m,
            dims.k,
            dims.n,
        );
    }

    if let (Some((DType::I8, lhs_buffer, lhs_scale)), Some((DType::BF16, rhs_buffer, _))) = (
        lhs.cloned_cuda_native_lowp_buffer(),
        rhs.cloned_cuda_native_lowp_buffer(),
    ) {
        return cuda::batch_matmul_backward_i8_bf16_no_host(
            &grad_buf,
            &lhs_buffer,
            lhs_scale.ok_or_else(|| "CUDA I8 lhs scale missing".to_string())?,
            &rhs_buffer,
            batch_count,
            dims.m,
            dims.k,
            dims.n,
        );
    }

    if let (Some((DType::I8, lhs_buffer, lhs_scale)), Some((DType::F16, rhs_buffer, _))) = (
        lhs.cloned_cuda_native_lowp_buffer(),
        rhs.cloned_cuda_native_lowp_buffer(),
    ) {
        return cuda::batch_matmul_backward_i8_f16_no_host(
            &grad_buf,
            &lhs_buffer,
            lhs_scale.ok_or_else(|| "CUDA I8 lhs scale missing".to_string())?,
            &rhs_buffer,
            batch_count,
            dims.m,
            dims.k,
            dims.n,
        );
    }

    if let Some((DType::I8, lhs_buffer, lhs_scale)) = lhs.cloned_cuda_native_lowp_buffer()
        && rhs.dtype() == DType::F32
    {
        let rhs_buffer = rhs
            .cloned_cuda_f32_buffer()
            .ok_or_else(|| "CUDA I8xF32 batch_matmul backward expected rhs buffer".to_string())?;
        return cuda::batch_matmul_backward_i8_f32_no_host(
            &grad_buf,
            &lhs_buffer,
            lhs_scale.ok_or_else(|| "CUDA I8 lhs scale missing".to_string())?,
            &rhs_buffer,
            batch_count,
            dims.m,
            dims.k,
            dims.n,
        );
    }

    if let (Some((lhs_dtype, lhs_buffer, lhs_scale)), Some((rhs_dtype, rhs_buffer, rhs_scale))) = (
        lhs.cloned_cuda_native_lowp_buffer(),
        rhs.cloned_cuda_native_lowp_buffer(),
    ) && lhs_dtype == rhs_dtype
        && lhs_dtype != DType::F32
    {
        return match lhs_dtype {
            DType::BF16 => cuda::batch_matmul_backward_bf16_no_host(
                &grad_buf,
                &lhs_buffer,
                &rhs_buffer,
                batch_count,
                dims.m,
                dims.k,
                dims.n,
            ),
            DType::F16 => cuda::batch_matmul_backward_f16_no_host(
                &grad_buf,
                &lhs_buffer,
                &rhs_buffer,
                batch_count,
                dims.m,
                dims.k,
                dims.n,
            ),
            DType::I8 => cuda::batch_matmul_backward_i8_no_host(
                &grad_buf,
                &lhs_buffer,
                lhs_scale.ok_or_else(|| "CUDA I8 lhs scale missing".to_string())?,
                &rhs_buffer,
                rhs_scale.ok_or_else(|| "CUDA I8 rhs scale missing".to_string())?,
                batch_count,
                dims.m,
                dims.k,
                dims.n,
            ),
            DType::F32 => unreachable!("F32 is excluded by the low-precision guard"),
        };
    }
    let lhs_buf = lhs
        .cloned_cuda_f32_buffer()
        .ok_or_else(|| "CUDA batch_matmul backward expected lhs resident buffer".to_string())?;
    let rhs_buf = rhs
        .cloned_cuda_f32_buffer()
        .ok_or_else(|| "CUDA batch_matmul backward expected rhs resident buffer".to_string())?;

    cuda::batch_matmul_backward_f32_no_host(
        &grad_buf,
        &lhs_buf,
        &rhs_buf,
        batch_count,
        dims.m,
        dims.k,
        dims.n,
    )
}

fn matmul_forward_cpu_native(
    a: &Tensor,
    b: &Tensor,
    m_dim: usize,
    k_dim: usize,
    n_dim: usize,
) -> Array2<f32> {
    let input_dtype = a.dtype();
    if b.dtype() == DType::I8 {
        return match b.native_storage_owned() {
            TensorStorageOwned::I8(b_data, scale) => {
                let b_2d = b_data
                    .view()
                    .into_dimensionality::<Ix2>()
                    .expect("matmul RHS must be 2D [N, K]");
                let b_owned;
                let b_slice: &[i8] = if let Some(s) = b_2d.as_slice() {
                    s
                } else {
                    b_owned = b_2d.as_standard_layout().to_owned();
                    b_owned
                        .as_slice()
                        .expect("standard-layout matmul RHS should be contiguous")
                };

                if input_dtype == DType::I8 {
                    match a.native_storage_owned() {
                        TensorStorageOwned::I8(a_data, a_scale) => {
                            if m_dim == 1 {
                                let a_owned;
                                let a_vec: &[i8] = if let Some(s) = a_data.as_slice() {
                                    s
                                } else {
                                    a_owned = a_data.iter().copied().collect::<Vec<i8>>();
                                    a_owned.as_slice()
                                };

                                let mut out_vec = vec![0.0f32; n_dim];
                                matvec_rowmajor_parallel_i8_i8(
                                    a_vec,
                                    a_scale,
                                    b_slice,
                                    scale,
                                    n_dim,
                                    k_dim,
                                    &mut out_vec,
                                );
                                Array2::from_shape_vec((1, n_dim), out_vec)
                                    .expect("decode matvec shape build failed")
                            } else {
                                matmul_rows_i8_slice(
                                    a_data.view().into_dyn(),
                                    a_scale,
                                    SliceRef::I8(b_slice, scale),
                                    m_dim,
                                    k_dim,
                                    n_dim,
                                )
                            }
                        }
                        TensorStorageOwned::F32(_)
                        | TensorStorageOwned::F16(_)
                        | TensorStorageOwned::BF16(_) => {
                            unreachable!("checked i8 lhs above")
                        }
                    }
                } else {
                    a.with_storage_view_preferring(StoragePreference::Native, |a_view| {
                        if m_dim == 1 {
                            match a_view {
                                TensorStorageView::F32(a_view) => {
                                    let a_owned;
                                    let a_vec: &[f32] = if let Some(s) = a_view.as_slice() {
                                        s
                                    } else {
                                        a_owned = a_view.iter().copied().collect::<Vec<f32>>();
                                        a_owned.as_slice()
                                    };

                                    let mut out_vec = vec![0.0f32; n_dim];
                                    matvec_rowmajor_parallel_f32_i8_matmul(
                                        a_vec,
                                        b_slice,
                                        scale,
                                        n_dim,
                                        k_dim,
                                        &mut out_vec,
                                    );
                                    Array2::from_shape_vec((1, n_dim), out_vec)
                                        .expect("decode matvec shape build failed")
                                }
                                TensorStorageView::F16(a_view) => {
                                    let a_owned;
                                    let a_vec: &[f16] = if let Some(s) = a_view.as_slice() {
                                        s
                                    } else {
                                        a_owned = a_view.iter().copied().collect::<Vec<f16>>();
                                        a_owned.as_slice()
                                    };

                                    let mut out_vec = vec![0.0f32; n_dim];
                                    with_f16_input_as_f32(a_vec, |a_f32| {
                                        matvec_rowmajor_parallel_f32_i8_matmul(
                                            a_f32,
                                            b_slice,
                                            scale,
                                            n_dim,
                                            k_dim,
                                            &mut out_vec,
                                        );
                                    });
                                    Array2::from_shape_vec((1, n_dim), out_vec)
                                        .expect("decode matvec shape build failed")
                                }
                                TensorStorageView::BF16(a_view) => {
                                    let a_owned;
                                    let a_vec: &[bf16] = if let Some(s) = a_view.as_slice() {
                                        s
                                    } else {
                                        a_owned = a_view.iter().copied().collect::<Vec<bf16>>();
                                        a_owned.as_slice()
                                    };

                                    let mut out_vec = vec![0.0f32; n_dim];
                                    with_bf16_input_as_f32(a_vec, |a_f32| {
                                        matvec_rowmajor_parallel_f32_i8_matmul(
                                            a_f32,
                                            b_slice,
                                            scale,
                                            n_dim,
                                            k_dim,
                                            &mut out_vec,
                                        );
                                    });
                                    Array2::from_shape_vec((1, n_dim), out_vec)
                                        .expect("decode matvec shape build failed")
                                }
                            }
                        } else {
                            let b_view = b_data.view().into_dyn();
                            match a_view {
                                TensorStorageView::F32(a_view) => {
                                    matmul_rows_f32_i8(a_view, b_view, scale, m_dim, k_dim, n_dim)
                                }
                                TensorStorageView::F16(a_view) => matmul_rows_f16_slice(
                                    a_view,
                                    SliceRef::I8(b_slice, scale),
                                    m_dim,
                                    k_dim,
                                    n_dim,
                                ),
                                TensorStorageView::BF16(a_view) => {
                                    matmul_rows_bf16_i8(a_view, b_view, scale, m_dim, k_dim, n_dim)
                                }
                            }
                        }
                    })
                }
            }
            TensorStorageOwned::F32(_)
            | TensorStorageOwned::F16(_)
            | TensorStorageOwned::BF16(_) => unreachable!("checked i8 RHS above"),
        };
    }

    a.with_storage_view_preferring(StoragePreference::Native, |a_view| {
        b.with_storage_view_preferring(StoragePreference::Native, |b_view| match (a_view, b_view) {
            (TensorStorageView::F32(a_view), TensorStorageView::F32(b_view)) => {
                if m_dim == 1 {
                    let a_owned;
                    let a_vec: &[f32] = if let Some(s) = a_view.as_slice() {
                        s
                    } else {
                        a_owned = a_view.iter().copied().collect::<Vec<f32>>();
                        a_owned.as_slice()
                    };

                    let b_2d = b_view.into_dimensionality::<Ix2>().unwrap();
                    let b_owned;
                    let b_slice: &[f32] = if let Some(s) = b_2d.as_slice() {
                        s
                    } else {
                        b_owned = b_2d.as_standard_layout().to_owned();
                        b_owned
                            .as_slice()
                            .expect("standard-layout matmul RHS should be contiguous")
                    };

                    let mut out_vec = vec![0.0f32; n_dim];
                    matvec_rowmajor_parallel(a_vec, b_slice, n_dim, k_dim, &mut out_vec);
                    Array2::from_shape_vec((1, n_dim), out_vec)
                        .expect("decode matvec shape build failed")
                } else {
                    let b_2d = b_view.into_dimensionality::<Ix2>().unwrap();
                    let mut res = Array2::<f32>::zeros((m_dim, n_dim));
                    if let Ok(a_2d_view) = a_view.clone().into_shape((m_dim, k_dim)) {
                        general_mat_mul(1.0, &a_2d_view, &b_2d.t(), 0.0, &mut res);
                    } else {
                        let a_2d_owned = a_view
                            .to_owned()
                            .into_shape((m_dim, k_dim))
                            .expect("Reshape A failed");
                        general_mat_mul(1.0, &a_2d_owned, &b_2d.t(), 0.0, &mut res);
                    }
                    res
                }
            }
            (TensorStorageView::F32(a_view), TensorStorageView::F16(b_view)) => {
                matmul_rows_f32_f16(a_view, b_view, m_dim, k_dim, n_dim)
            }
            (TensorStorageView::F32(a_view), TensorStorageView::BF16(b_view)) => {
                matmul_rows_f32_bf16(a_view, b_view, m_dim, k_dim, n_dim)
            }
            (TensorStorageView::F16(a_view), TensorStorageView::F32(b_view)) => {
                matmul_rows_f16_f32(a_view, b_view, m_dim, k_dim, n_dim)
            }
            (TensorStorageView::F16(a_view), TensorStorageView::F16(b_view)) => {
                matmul_rows_f16_f16(a_view, b_view, m_dim, k_dim, n_dim)
            }
            (TensorStorageView::F16(a_view), TensorStorageView::BF16(b_view)) => {
                let b_2d = b_view.into_dimensionality::<Ix2>().unwrap();
                matmul_rows_f16_slice(
                    a_view,
                    SliceRef::BF16(
                        b_2d.as_slice()
                            .expect("standard-layout matmul RHS should be contiguous"),
                    ),
                    m_dim,
                    k_dim,
                    n_dim,
                )
            }
            (TensorStorageView::BF16(a_view), TensorStorageView::F32(b_view)) => {
                matmul_rows_bf16_f32(a_view, b_view, m_dim, k_dim, n_dim)
            }
            (TensorStorageView::BF16(a_view), TensorStorageView::F16(b_view)) => {
                let b_2d = b_view.into_dimensionality::<Ix2>().unwrap();
                matmul_rows_bf16_slice(
                    a_view,
                    SliceRef::F16(
                        b_2d.as_slice()
                            .expect("standard-layout matmul RHS should be contiguous"),
                    ),
                    m_dim,
                    k_dim,
                    n_dim,
                )
            }
            (TensorStorageView::BF16(a_view), TensorStorageView::BF16(b_view)) => {
                matmul_rows_bf16_bf16(a_view, b_view, m_dim, k_dim, n_dim)
            }
        })
    })
}

fn f16_view_to_f32_owned(view: ndarray::ArrayViewD<'_, f16>) -> ArrayD<f32> {
    let shape = view.shape().to_vec();
    let mut raw = vec![0.0f32; view.len()];
    if let Some(slice) = view.as_slice_memory_order() {
        slice.convert_to_f32_slice(&mut raw);
    } else {
        for (dst, src) in raw.iter_mut().zip(view.iter()) {
            *dst = src.to_f32();
        }
    }
    ArrayD::from_shape_vec(IxDyn(&shape), raw).expect("Failed to build f16 f32 compute view")
}

fn bf16_view_to_f32_owned(view: ndarray::ArrayViewD<'_, bf16>) -> ArrayD<f32> {
    let shape = view.shape().to_vec();
    let mut raw = vec![0.0f32; view.len()];
    if let Some(slice) = view.as_slice_memory_order() {
        slice.convert_to_f32_slice(&mut raw);
    } else {
        for (dst, src) in raw.iter_mut().zip(view.iter()) {
            *dst = src.to_f32();
        }
    }
    ArrayD::from_shape_vec(IxDyn(&shape), raw).expect("Failed to build bf16 f32 compute view")
}

fn with_native_f32_compute_view<R>(
    tensor: &Tensor,
    f: impl FnOnce(ndarray::ArrayViewD<'_, f32>) -> R,
) -> R {
    tensor.with_storage_view_preferring(StoragePreference::Native, |view| match view {
        TensorStorageView::F32(view) => f(view),
        TensorStorageView::F16(view) => {
            let owned = f16_view_to_f32_owned(view);
            f(owned.view())
        }
        TensorStorageView::BF16(view) => {
            let owned = bf16_view_to_f32_owned(view);
            f(owned.view())
        }
    })
}

fn matmul_backward_f16_slices(
    a: &[f16],
    b: &[f16],
    grad: &[f32],
    m_dim: usize,
    k_dim: usize,
    n_dim: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut da = vec![0.0f32; m_dim * k_dim];
    da.par_chunks_mut(k_dim)
        .enumerate()
        .for_each(|(m, da_row)| {
            let grad_row = &grad[m * n_dim..(m + 1) * n_dim];
            for n in 0..n_dim {
                let g = grad_row[n];
                let b_row = &b[n * k_dim..(n + 1) * k_dim];
                for (dst, &bv) in da_row.iter_mut().zip(b_row.iter()) {
                    *dst += g * bv.to_f32();
                }
            }
        });

    let mut db = vec![0.0f32; n_dim * k_dim];
    db.par_chunks_mut(k_dim)
        .enumerate()
        .for_each(|(n, db_row)| {
            for m in 0..m_dim {
                let g = grad[m * n_dim + n];
                let a_row = &a[m * k_dim..(m + 1) * k_dim];
                for (dst, &av) in db_row.iter_mut().zip(a_row.iter()) {
                    *dst += g * av.to_f32();
                }
            }
        });

    (da, db)
}

fn matmul_backward_bf16_slices(
    a: &[bf16],
    b: &[bf16],
    grad: &[f32],
    m_dim: usize,
    k_dim: usize,
    n_dim: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut da = vec![0.0f32; m_dim * k_dim];
    da.par_chunks_mut(k_dim)
        .enumerate()
        .for_each(|(m, da_row)| {
            let grad_row = &grad[m * n_dim..(m + 1) * n_dim];
            for n in 0..n_dim {
                let g = grad_row[n];
                let b_row = &b[n * k_dim..(n + 1) * k_dim];
                for (dst, &bv) in da_row.iter_mut().zip(b_row.iter()) {
                    *dst += g * bv.to_f32();
                }
            }
        });

    let mut db = vec![0.0f32; n_dim * k_dim];
    db.par_chunks_mut(k_dim)
        .enumerate()
        .for_each(|(n, db_row)| {
            for m in 0..m_dim {
                let g = grad[m * n_dim + n];
                let a_row = &a[m * k_dim..(m + 1) * k_dim];
                for (dst, &av) in db_row.iter_mut().zip(a_row.iter()) {
                    *dst += g * av.to_f32();
                }
            }
        });

    (da, db)
}

fn matmul_backward_i8_slices(
    a: &[i8],
    a_scale: f32,
    b: &[i8],
    b_scale: f32,
    grad: &[f32],
    m_dim: usize,
    k_dim: usize,
    n_dim: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut da = vec![0.0f32; m_dim * k_dim];
    da.par_chunks_mut(k_dim)
        .enumerate()
        .for_each(|(m, da_row)| {
            let grad_row = &grad[m * n_dim..(m + 1) * n_dim];
            for n in 0..n_dim {
                let g = grad_row[n] * b_scale;
                let b_row = &b[n * k_dim..(n + 1) * k_dim];
                for (dst, &bv) in da_row.iter_mut().zip(b_row.iter()) {
                    *dst += g * (bv as f32);
                }
            }
        });

    let mut db = vec![0.0f32; n_dim * k_dim];
    db.par_chunks_mut(k_dim)
        .enumerate()
        .for_each(|(n, db_row)| {
            for m in 0..m_dim {
                let g = grad[m * n_dim + n] * a_scale;
                let a_row = &a[m * k_dim..(m + 1) * k_dim];
                for (dst, &av) in db_row.iter_mut().zip(a_row.iter()) {
                    *dst += g * (av as f32);
                }
            }
        });

    (da, db)
}

fn matmul_backward_rhs_i8_slices<T: DotElem>(
    a: &[T],
    b: &[i8],
    b_scale: f32,
    grad: &[f32],
    m_dim: usize,
    k_dim: usize,
    n_dim: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut da = vec![0.0f32; m_dim * k_dim];
    da.par_chunks_mut(k_dim)
        .enumerate()
        .for_each(|(m, da_row)| {
            let grad_row = &grad[m * n_dim..(m + 1) * n_dim];
            for n in 0..n_dim {
                let g = grad_row[n] * b_scale;
                let b_row = &b[n * k_dim..(n + 1) * k_dim];
                for (dst, &bv) in da_row.iter_mut().zip(b_row.iter()) {
                    *dst += g * (bv as f32);
                }
            }
        });

    let mut db = vec![0.0f32; n_dim * k_dim];
    db.par_chunks_mut(k_dim)
        .enumerate()
        .for_each(|(n, db_row)| {
            for m in 0..m_dim {
                let g = grad[m * n_dim + n];
                let a_row = &a[m * k_dim..(m + 1) * k_dim];
                for (dst, &av) in db_row.iter_mut().zip(a_row.iter()) {
                    *dst += g * av.to_f32();
                }
            }
        });

    (da, db)
}

fn matmul_backward_lhs_i8_slices<T: DotElem>(
    a: &[i8],
    a_scale: f32,
    b: &[T],
    grad: &[f32],
    m_dim: usize,
    k_dim: usize,
    n_dim: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut da = vec![0.0f32; m_dim * k_dim];
    da.par_chunks_mut(k_dim)
        .enumerate()
        .for_each(|(m, da_row)| {
            let grad_row = &grad[m * n_dim..(m + 1) * n_dim];
            for n in 0..n_dim {
                let g = grad_row[n];
                let b_row = &b[n * k_dim..(n + 1) * k_dim];
                for (dst, &bv) in da_row.iter_mut().zip(b_row.iter()) {
                    *dst += g * bv.to_f32();
                }
            }
        });

    let mut db = vec![0.0f32; n_dim * k_dim];
    db.par_chunks_mut(k_dim)
        .enumerate()
        .for_each(|(n, db_row)| {
            for m in 0..m_dim {
                let g = grad[m * n_dim + n] * a_scale;
                let a_row = &a[m * k_dim..(m + 1) * k_dim];
                for (dst, &av) in db_row.iter_mut().zip(a_row.iter()) {
                    *dst += g * (av as f32);
                }
            }
        });

    (da, db)
}

#[allow(clippy::too_many_arguments)]
fn batch_matmul_backward_typed_slices<T: DotElem>(
    lhs: &[T],
    rhs: &[T],
    grad: &[f32],
    batch_count: usize,
    m_dim: usize,
    k_dim: usize,
    n_dim: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut d_lhs = vec![0.0f32; batch_count * m_dim * k_dim];
    d_lhs
        .par_chunks_mut(k_dim)
        .enumerate()
        .for_each(|(batch_row, d_lhs_row)| {
            let batch = batch_row / m_dim;
            let row = batch_row - batch * m_dim;
            let grad_batch = &grad[batch * m_dim * n_dim..(batch + 1) * m_dim * n_dim];
            let rhs_batch = &rhs[batch * k_dim * n_dim..(batch + 1) * k_dim * n_dim];
            let grad_row = &grad_batch[row * n_dim..(row + 1) * n_dim];
            for kk in 0..k_dim {
                let mut acc = 0.0f32;
                for col in 0..n_dim {
                    acc += grad_row[col] * rhs_batch[kk * n_dim + col].to_f32();
                }
                d_lhs_row[kk] = acc;
            }
        });

    let mut d_rhs = vec![0.0f32; batch_count * k_dim * n_dim];
    d_rhs
        .par_chunks_mut(n_dim)
        .enumerate()
        .for_each(|(batch_kk, d_rhs_row)| {
            let batch = batch_kk / k_dim;
            let kk = batch_kk - batch * k_dim;
            let lhs_batch = &lhs[batch * m_dim * k_dim..(batch + 1) * m_dim * k_dim];
            let grad_batch = &grad[batch * m_dim * n_dim..(batch + 1) * m_dim * n_dim];
            for col in 0..n_dim {
                let mut acc = 0.0f32;
                for row in 0..m_dim {
                    acc += lhs_batch[row * k_dim + kk].to_f32() * grad_batch[row * n_dim + col];
                }
                d_rhs_row[col] = acc;
            }
        });

    (d_lhs, d_rhs)
}

#[allow(clippy::too_many_arguments)]
fn batch_matmul_backward_i8_slices(
    lhs: &[i8],
    lhs_scale: f32,
    rhs: &[i8],
    rhs_scale: f32,
    grad: &[f32],
    batch_count: usize,
    m_dim: usize,
    k_dim: usize,
    n_dim: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut d_lhs = vec![0.0f32; batch_count * m_dim * k_dim];
    d_lhs
        .par_chunks_mut(k_dim)
        .enumerate()
        .for_each(|(batch_row, d_lhs_row)| {
            let batch = batch_row / m_dim;
            let row = batch_row - batch * m_dim;
            let grad_batch = &grad[batch * m_dim * n_dim..(batch + 1) * m_dim * n_dim];
            let rhs_batch = &rhs[batch * k_dim * n_dim..(batch + 1) * k_dim * n_dim];
            let grad_row = &grad_batch[row * n_dim..(row + 1) * n_dim];
            for kk in 0..k_dim {
                let mut acc = 0.0f32;
                for col in 0..n_dim {
                    acc += grad_row[col] * (rhs_batch[kk * n_dim + col] as f32) * rhs_scale;
                }
                d_lhs_row[kk] = acc;
            }
        });

    let mut d_rhs = vec![0.0f32; batch_count * k_dim * n_dim];
    d_rhs
        .par_chunks_mut(n_dim)
        .enumerate()
        .for_each(|(batch_kk, d_rhs_row)| {
            let batch = batch_kk / k_dim;
            let kk = batch_kk - batch * k_dim;
            let lhs_batch = &lhs[batch * m_dim * k_dim..(batch + 1) * m_dim * k_dim];
            let grad_batch = &grad[batch * m_dim * n_dim..(batch + 1) * m_dim * n_dim];
            for col in 0..n_dim {
                let mut acc = 0.0f32;
                for row in 0..m_dim {
                    acc += (lhs_batch[row * k_dim + kk] as f32)
                        * lhs_scale
                        * grad_batch[row * n_dim + col];
                }
                d_rhs_row[col] = acc;
            }
        });

    (d_lhs, d_rhs)
}

#[allow(clippy::too_many_arguments)]
fn batch_matmul_backward_rhs_i8_slices<T: DotElem>(
    lhs: &[T],
    rhs: &[i8],
    rhs_scale: f32,
    grad: &[f32],
    batch_count: usize,
    m_dim: usize,
    k_dim: usize,
    n_dim: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut d_lhs = vec![0.0f32; batch_count * m_dim * k_dim];
    d_lhs
        .par_chunks_mut(k_dim)
        .enumerate()
        .for_each(|(batch_row, d_lhs_row)| {
            let batch = batch_row / m_dim;
            let row = batch_row - batch * m_dim;
            let grad_batch = &grad[batch * m_dim * n_dim..(batch + 1) * m_dim * n_dim];
            let rhs_batch = &rhs[batch * k_dim * n_dim..(batch + 1) * k_dim * n_dim];
            let grad_row = &grad_batch[row * n_dim..(row + 1) * n_dim];
            for kk in 0..k_dim {
                let mut acc = 0.0f32;
                for col in 0..n_dim {
                    acc += grad_row[col] * (rhs_batch[kk * n_dim + col] as f32) * rhs_scale;
                }
                d_lhs_row[kk] = acc;
            }
        });

    let mut d_rhs = vec![0.0f32; batch_count * k_dim * n_dim];
    d_rhs
        .par_chunks_mut(n_dim)
        .enumerate()
        .for_each(|(batch_kk, d_rhs_row)| {
            let batch = batch_kk / k_dim;
            let kk = batch_kk - batch * k_dim;
            let lhs_batch = &lhs[batch * m_dim * k_dim..(batch + 1) * m_dim * k_dim];
            let grad_batch = &grad[batch * m_dim * n_dim..(batch + 1) * m_dim * n_dim];
            for col in 0..n_dim {
                let mut acc = 0.0f32;
                for row in 0..m_dim {
                    acc += lhs_batch[row * k_dim + kk].to_f32() * grad_batch[row * n_dim + col];
                }
                d_rhs_row[col] = acc;
            }
        });

    (d_lhs, d_rhs)
}

#[allow(clippy::too_many_arguments)]
fn batch_matmul_backward_lhs_i8_slices<T: DotElem>(
    lhs: &[i8],
    lhs_scale: f32,
    rhs: &[T],
    grad: &[f32],
    batch_count: usize,
    m_dim: usize,
    k_dim: usize,
    n_dim: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut d_lhs = vec![0.0f32; batch_count * m_dim * k_dim];
    d_lhs
        .par_chunks_mut(k_dim)
        .enumerate()
        .for_each(|(batch_row, d_lhs_row)| {
            let batch = batch_row / m_dim;
            let row = batch_row - batch * m_dim;
            let grad_batch = &grad[batch * m_dim * n_dim..(batch + 1) * m_dim * n_dim];
            let rhs_batch = &rhs[batch * k_dim * n_dim..(batch + 1) * k_dim * n_dim];
            let grad_row = &grad_batch[row * n_dim..(row + 1) * n_dim];
            for kk in 0..k_dim {
                let mut acc = 0.0f32;
                for col in 0..n_dim {
                    acc += grad_row[col] * rhs_batch[kk * n_dim + col].to_f32();
                }
                d_lhs_row[kk] = acc;
            }
        });

    let mut d_rhs = vec![0.0f32; batch_count * k_dim * n_dim];
    d_rhs
        .par_chunks_mut(n_dim)
        .enumerate()
        .for_each(|(batch_kk, d_rhs_row)| {
            let batch = batch_kk / k_dim;
            let kk = batch_kk - batch * k_dim;
            let lhs_batch = &lhs[batch * m_dim * k_dim..(batch + 1) * m_dim * k_dim];
            let grad_batch = &grad[batch * m_dim * n_dim..(batch + 1) * m_dim * n_dim];
            for col in 0..n_dim {
                let mut acc = 0.0f32;
                for row in 0..m_dim {
                    acc += (lhs_batch[row * k_dim + kk] as f32)
                        * lhs_scale
                        * grad_batch[row * n_dim + col];
                }
                d_rhs_row[col] = acc;
            }
        });

    (d_lhs, d_rhs)
}

fn try_matmul_backward_cpu_native_lowp(
    a: &Tensor,
    b: &Tensor,
    grad: &ndarray::ArrayViewD<'_, f32>,
    m_dim: usize,
    k_dim: usize,
    n_dim: usize,
) -> Option<(ndarray::ArrayD<f32>, ndarray::ArrayD<f32>)> {
    if grad.len() != m_dim * n_dim {
        return None;
    }

    macro_rules! slice_or_owned {
        ($array:expr, $owned:ident) => {
            if let Some(slice) = $array.as_slice_memory_order() {
                slice
            } else {
                $owned = $array.iter().copied().collect::<Vec<_>>();
                $owned.as_slice()
            }
        };
    }

    let grad_owned;
    let grad_slice = if let Some(slice) = grad.as_slice_memory_order() {
        slice
    } else {
        grad_owned = grad.iter().copied().collect::<Vec<_>>();
        grad_owned.as_slice()
    };
    let a_shape = a.shape_vec();

    let (da, db) = match (a.native_storage_owned(), b.native_storage_owned()) {
        (TensorStorageOwned::F16(a_data), TensorStorageOwned::F16(b_data)) => {
            let a_owned;
            let b_owned;
            matmul_backward_f16_slices(
                slice_or_owned!(a_data, a_owned),
                slice_or_owned!(b_data, b_owned),
                grad_slice,
                m_dim,
                k_dim,
                n_dim,
            )
        }
        (TensorStorageOwned::BF16(a_data), TensorStorageOwned::BF16(b_data)) => {
            let a_owned;
            let b_owned;
            matmul_backward_bf16_slices(
                slice_or_owned!(a_data, a_owned),
                slice_or_owned!(b_data, b_owned),
                grad_slice,
                m_dim,
                k_dim,
                n_dim,
            )
        }
        (TensorStorageOwned::I8(a_data, a_scale), TensorStorageOwned::I8(b_data, b_scale)) => {
            let a_owned;
            let b_owned;
            matmul_backward_i8_slices(
                slice_or_owned!(a_data, a_owned),
                a_scale,
                slice_or_owned!(b_data, b_owned),
                b_scale,
                grad_slice,
                m_dim,
                k_dim,
                n_dim,
            )
        }
        (TensorStorageOwned::F32(a_data), TensorStorageOwned::I8(b_data, b_scale)) => {
            let a_owned;
            let b_owned;
            matmul_backward_rhs_i8_slices(
                slice_or_owned!(a_data, a_owned),
                slice_or_owned!(b_data, b_owned),
                b_scale,
                grad_slice,
                m_dim,
                k_dim,
                n_dim,
            )
        }
        (TensorStorageOwned::F16(a_data), TensorStorageOwned::I8(b_data, b_scale)) => {
            let a_owned;
            let b_owned;
            matmul_backward_rhs_i8_slices(
                slice_or_owned!(a_data, a_owned),
                slice_or_owned!(b_data, b_owned),
                b_scale,
                grad_slice,
                m_dim,
                k_dim,
                n_dim,
            )
        }
        (TensorStorageOwned::BF16(a_data), TensorStorageOwned::I8(b_data, b_scale)) => {
            let a_owned;
            let b_owned;
            matmul_backward_rhs_i8_slices(
                slice_or_owned!(a_data, a_owned),
                slice_or_owned!(b_data, b_owned),
                b_scale,
                grad_slice,
                m_dim,
                k_dim,
                n_dim,
            )
        }
        (TensorStorageOwned::I8(a_data, a_scale), TensorStorageOwned::F32(b_data)) => {
            let a_owned;
            let b_owned;
            matmul_backward_lhs_i8_slices(
                slice_or_owned!(a_data, a_owned),
                a_scale,
                slice_or_owned!(b_data, b_owned),
                grad_slice,
                m_dim,
                k_dim,
                n_dim,
            )
        }
        (TensorStorageOwned::I8(a_data, a_scale), TensorStorageOwned::F16(b_data)) => {
            let a_owned;
            let b_owned;
            matmul_backward_lhs_i8_slices(
                slice_or_owned!(a_data, a_owned),
                a_scale,
                slice_or_owned!(b_data, b_owned),
                grad_slice,
                m_dim,
                k_dim,
                n_dim,
            )
        }
        (TensorStorageOwned::I8(a_data, a_scale), TensorStorageOwned::BF16(b_data)) => {
            let a_owned;
            let b_owned;
            matmul_backward_lhs_i8_slices(
                slice_or_owned!(a_data, a_owned),
                a_scale,
                slice_or_owned!(b_data, b_owned),
                grad_slice,
                m_dim,
                k_dim,
                n_dim,
            )
        }
        _ => return None,
    };

    Some((
        Array2::from_shape_vec((m_dim, k_dim), da)
            .ok()?
            .into_shape(a_shape)
            .ok()?
            .into_dyn(),
        Array2::from_shape_vec((n_dim, k_dim), db).ok()?.into_dyn(),
    ))
}

fn matmul_backward_cpu_f32(
    a: &Tensor,
    b: &Tensor,
    grad: &ndarray::ArrayViewD<'_, f32>,
    m_dim: usize,
    k_dim: usize,
    n_dim: usize,
) -> (ndarray::ArrayD<f32>, ndarray::ArrayD<f32>) {
    if let Some(grads) = try_matmul_backward_cpu_native_lowp(a, b, grad, m_dim, k_dim, n_dim) {
        return grads;
    }

    let g_len = grad.len();
    let g_m = g_len / n_dim;
    let grad_2d = grad
        .view()
        .into_shape((g_m, n_dim))
        .expect("Grad reshape failed: non-contiguous gradient?");
    let a_shape = a.shape_vec();

    with_native_f32_compute_view(a, |a_view| {
        with_native_f32_compute_view(b, |b_view| {
            let b_2d = b_view.into_dimensionality::<Ix2>().unwrap();

            let mut da_2d = Array2::<f32>::zeros((m_dim, k_dim));
            general_mat_mul(1.0, &grad_2d, &b_2d, 0.0, &mut da_2d);

            let mut db_2d = Array2::<f32>::zeros((n_dim, k_dim));
            if let Ok(a_2d) = a_view.clone().into_shape((m_dim, k_dim)) {
                general_mat_mul(1.0, &grad_2d.t(), &a_2d, 0.0, &mut db_2d);
            } else {
                let a_2d_owned = a_view.to_owned().into_shape((m_dim, k_dim)).unwrap();
                general_mat_mul(1.0, &grad_2d.t(), &a_2d_owned, 0.0, &mut db_2d);
            }

            (
                da_2d.into_shape(a_shape).unwrap().into_dyn(),
                db_2d.into_dyn(),
            )
        })
    })
}

// A[..., K] @ B^T, where B is [N(out), K(in)]
// output: [..., N]
pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor {
    let output_device = assert_same_device(a, b, "matmul");
    let build_graph = !is_no_grad() && (a.requires_grad() || b.requires_grad());
    let cuda_native_supported = output_device == crate::autograd::Device::Cuda;
    assert_native_device_support(output_device, "matmul", cuda_native_supported);

    let a_shape = a.shape_vec();
    let b_shape = b.shape_vec();
    let a_len = a.len();

    if b_shape.len() != 2 {
        panic!("MatMul RHS must be 2D, got {:?}", b_shape);
    }

    let k_dim_a = a_shape[a_shape.len() - 1];
    let n_dim = b_shape[0];
    let k_dim_b = b_shape[1];

    if k_dim_a != k_dim_b {
        panic!(
            "MatMul shape mismatch: a {:?} (K={}) vs b {:?} (K={})",
            a_shape, k_dim_a, b_shape, k_dim_b
        );
    }

    let m_dim = a_len / k_dim_a;

    if build_graph
        && output_device == crate::autograd::Device::Cuda
        && let Some(buffer) = try_cuda_matmul_buffer(a, b, m_dim, k_dim_a, n_dim)
    {
        let mut out_shape = a_shape.clone();
        let last_idx = out_shape.len() - 1;
        out_shape[last_idx] = n_dim;

        let a_clone = a.clone();
        let b_clone = b.clone();
        let output_self = Rc::new(RefCell::new(None::<Tensor>));
        let output_self_for_backward = output_self.clone();
        let tensor = Tensor(Rc::new(RefCell::new(TensorData {
            data: ndarray::ArrayD::<f32>::zeros(ndarray::IxDyn(&out_shape)).into_shared(),
            f16_data: None,
            bf16_data: None,
            i8_data: None,
            cuda_f32_data: Some(buffer),
            cuda_f16_data: None,
            cuda_bf16_data: None,
            cuda_i8_data: None,
            i8_scale: None,
            has_f32_data: false,
            storage_dtype: crate::precision::DType::F32,
            cache_dirty: false,
            is_parameter: false,
            grad: None,
            cuda_f32_grad: None,
            parents: vec![a_clone.clone(), b_clone.clone()],
            requires_grad: true,
            backward_op: Some(std::rc::Rc::new(move |grad: &ndarray::ArrayViewD<f32>| {
                let cuda_grad = output_self_for_backward
                    .borrow()
                    .as_ref()
                    .and_then(|output| output.cloned_cuda_f32_grad())
                    .filter(|buffer| buffer.len() == grad.len());
                if is_strict_device_execution() {
                    match try_cuda_training_matmul_backward_buffers(
                        grad,
                        cuda_grad.clone(),
                        &a_clone,
                        &b_clone,
                        m_dim,
                        k_dim_a,
                        n_dim,
                    ) {
                        Ok((da_buf, db_buf)) => {
                            a_clone.add_cuda_grad_buffer_only(da_buf);
                            b_clone.add_cuda_grad_buffer_only(db_buf);
                            return;
                        }
                        Err(err) => {
                            panic!(
                                "CUDA matmul backward failed in strict device execution mode: {err}"
                            );
                        }
                    }
                }
                let cuda_result = try_cuda_training_matmul_backward(
                    grad, cuda_grad, &a_clone, &b_clone, m_dim, k_dim_a, n_dim,
                );
                match cuda_result {
                    Ok(((da, da_buf), (db, db_buf))) => {
                        a_clone.add_grad_with_cuda_buffer(da, Some(da_buf));
                        b_clone.add_grad_with_cuda_buffer(db, Some(db_buf));
                    }
                    Err(err) => {
                        if is_strict_device_execution() {
                            panic!(
                                "CUDA matmul backward failed in strict device execution mode: {err}"
                            );
                        }
                        let (da, db) = matmul_backward_cpu_f32(
                            &a_clone, &b_clone, grad, m_dim, k_dim_a, n_dim,
                        );
                        a_clone.add_grad(da);
                        b_clone.add_grad(db);
                    }
                }
            })),
            device: output_device,
        })));
        *output_self.borrow_mut() = Some(tensor.clone());
        return tensor;
    }

    if !build_graph {
        let output_dtype = if a.dtype() == b.dtype() {
            a.dtype()
        } else {
            DType::F32
        };
        if let Some(cuda_out) = try_cuda_matmul(a, b, m_dim, n_dim, k_dim_a, output_dtype) {
            return cuda_out;
        }
        let res_2d = matmul_forward_cpu_native(a, b, m_dim, k_dim_a, n_dim);

        let mut out_shape = a_shape.clone();
        let last_idx = out_shape.len() - 1;
        out_shape[last_idx] = n_dim;
        return Tensor::from_f32_data_no_grad_with_device_dtype(
            res_2d.into_shape(out_shape).unwrap().into_dyn(),
            output_dtype,
            output_device,
        );
    }

    let res_2d = matmul_forward_cpu_native(a, b, m_dim, k_dim_a, n_dim);

    let mut out_shape = a_shape.clone();
    let last_idx = out_shape.len() - 1;
    out_shape[last_idx] = n_dim;

    let result = res_2d.into_shape(out_shape).unwrap().into_dyn();

    let a_clone = a.clone();
    let b_clone = b.clone();

    Tensor(Rc::new(RefCell::new(TensorData {
        data: result.into_shared(),
        f16_data: None,
        bf16_data: None,
        i8_data: None,
        cuda_f32_data: None,
        cuda_f16_data: None,
        cuda_bf16_data: None,
        cuda_i8_data: None,
        i8_scale: None,
        has_f32_data: true,
        storage_dtype: crate::precision::DType::F32,
        cache_dirty: false,
        is_parameter: false,
        grad: None,
        cuda_f32_grad: None,
        parents: vec![a_clone.clone(), b_clone.clone()],
        requires_grad: true,
        backward_op: Some(std::rc::Rc::new(move |grad: &ndarray::ArrayViewD<f32>| {
            let (da, db) = matmul_backward_cpu_f32(&a_clone, &b_clone, grad, m_dim, k_dim_a, n_dim);
            a_clone.add_grad(da);
            b_clone.add_grad(db);
        })),
        device: output_device,
    })))
}

fn batch_matmul_forward_cpu_f32(
    lhs: &Tensor,
    rhs: &Tensor,
    b: usize,
    h: usize,
    m: usize,
    n: usize,
) -> ndarray::ArrayD<f32> {
    with_native_f32_compute_view(lhs, |lhs_view| {
        with_native_f32_compute_view(rhs, |rhs_view| {
            let lhs_view = lhs_view.into_dimensionality::<Ix4>().unwrap();
            let rhs_view = rhs_view.into_dimensionality::<Ix4>().unwrap();
            let mut output = Array4::<f32>::zeros((b, h, m, n));

            Zip::from(output.outer_iter_mut())
                .and(lhs_view.outer_iter())
                .and(rhs_view.outer_iter())
                .for_each(|mut out_batch, lhs_batch, rhs_batch| {
                    Zip::from(out_batch.outer_iter_mut())
                        .and(lhs_batch.outer_iter())
                        .and(rhs_batch.outer_iter())
                        .for_each(|mut out_mat, lhs_mat, rhs_mat| {
                            general_mat_mul(1.0, &lhs_mat, &rhs_mat, 0.0, &mut out_mat);
                        });
                });

            output.into_dyn()
        })
    })
}

#[allow(clippy::too_many_arguments)]
fn try_batch_matmul_backward_cpu_native_lowp(
    lhs: &Tensor,
    rhs: &Tensor,
    grad: &ndarray::ArrayViewD<'_, f32>,
    b: usize,
    h: usize,
    m: usize,
    k: usize,
    n: usize,
) -> Option<(ndarray::ArrayD<f32>, ndarray::ArrayD<f32>)> {
    if grad.len() != b * h * m * n {
        return None;
    }

    macro_rules! slice_or_owned {
        ($array:expr, $owned:ident) => {
            if let Some(slice) = $array.as_slice_memory_order() {
                slice
            } else {
                $owned = $array.iter().copied().collect::<Vec<_>>();
                $owned.as_slice()
            }
        };
    }

    let grad_owned;
    let grad_slice = if let Some(slice) = grad.as_slice_memory_order() {
        slice
    } else {
        grad_owned = grad.iter().copied().collect::<Vec<_>>();
        grad_owned.as_slice()
    };
    let batch_count = b.checked_mul(h)?;

    let (d_lhs, d_rhs) = match (lhs.native_storage_owned(), rhs.native_storage_owned()) {
        (TensorStorageOwned::F16(lhs_data), TensorStorageOwned::F16(rhs_data)) => {
            let lhs_owned;
            let rhs_owned;
            batch_matmul_backward_typed_slices(
                slice_or_owned!(lhs_data, lhs_owned),
                slice_or_owned!(rhs_data, rhs_owned),
                grad_slice,
                batch_count,
                m,
                k,
                n,
            )
        }
        (TensorStorageOwned::BF16(lhs_data), TensorStorageOwned::BF16(rhs_data)) => {
            let lhs_owned;
            let rhs_owned;
            batch_matmul_backward_typed_slices(
                slice_or_owned!(lhs_data, lhs_owned),
                slice_or_owned!(rhs_data, rhs_owned),
                grad_slice,
                batch_count,
                m,
                k,
                n,
            )
        }
        (
            TensorStorageOwned::I8(lhs_data, lhs_scale),
            TensorStorageOwned::I8(rhs_data, rhs_scale),
        ) => {
            let lhs_owned;
            let rhs_owned;
            batch_matmul_backward_i8_slices(
                slice_or_owned!(lhs_data, lhs_owned),
                lhs_scale,
                slice_or_owned!(rhs_data, rhs_owned),
                rhs_scale,
                grad_slice,
                batch_count,
                m,
                k,
                n,
            )
        }
        (TensorStorageOwned::F32(lhs_data), TensorStorageOwned::I8(rhs_data, rhs_scale)) => {
            let lhs_owned;
            let rhs_owned;
            batch_matmul_backward_rhs_i8_slices(
                slice_or_owned!(lhs_data, lhs_owned),
                slice_or_owned!(rhs_data, rhs_owned),
                rhs_scale,
                grad_slice,
                batch_count,
                m,
                k,
                n,
            )
        }
        (TensorStorageOwned::F16(lhs_data), TensorStorageOwned::I8(rhs_data, rhs_scale)) => {
            let lhs_owned;
            let rhs_owned;
            batch_matmul_backward_rhs_i8_slices(
                slice_or_owned!(lhs_data, lhs_owned),
                slice_or_owned!(rhs_data, rhs_owned),
                rhs_scale,
                grad_slice,
                batch_count,
                m,
                k,
                n,
            )
        }
        (TensorStorageOwned::BF16(lhs_data), TensorStorageOwned::I8(rhs_data, rhs_scale)) => {
            let lhs_owned;
            let rhs_owned;
            batch_matmul_backward_rhs_i8_slices(
                slice_or_owned!(lhs_data, lhs_owned),
                slice_or_owned!(rhs_data, rhs_owned),
                rhs_scale,
                grad_slice,
                batch_count,
                m,
                k,
                n,
            )
        }
        (TensorStorageOwned::I8(lhs_data, lhs_scale), TensorStorageOwned::F32(rhs_data)) => {
            let lhs_owned;
            let rhs_owned;
            batch_matmul_backward_lhs_i8_slices(
                slice_or_owned!(lhs_data, lhs_owned),
                lhs_scale,
                slice_or_owned!(rhs_data, rhs_owned),
                grad_slice,
                batch_count,
                m,
                k,
                n,
            )
        }
        (TensorStorageOwned::I8(lhs_data, lhs_scale), TensorStorageOwned::F16(rhs_data)) => {
            let lhs_owned;
            let rhs_owned;
            batch_matmul_backward_lhs_i8_slices(
                slice_or_owned!(lhs_data, lhs_owned),
                lhs_scale,
                slice_or_owned!(rhs_data, rhs_owned),
                grad_slice,
                batch_count,
                m,
                k,
                n,
            )
        }
        (TensorStorageOwned::I8(lhs_data, lhs_scale), TensorStorageOwned::BF16(rhs_data)) => {
            let lhs_owned;
            let rhs_owned;
            batch_matmul_backward_lhs_i8_slices(
                slice_or_owned!(lhs_data, lhs_owned),
                lhs_scale,
                slice_or_owned!(rhs_data, rhs_owned),
                grad_slice,
                batch_count,
                m,
                k,
                n,
            )
        }
        _ => return None,
    };

    Some((
        Array4::from_shape_vec((b, h, m, k), d_lhs).ok()?.into_dyn(),
        Array4::from_shape_vec((b, h, k, n), d_rhs).ok()?.into_dyn(),
    ))
}

#[allow(clippy::too_many_arguments)]
fn batch_matmul_backward_cpu_f32(
    lhs: &Tensor,
    rhs: &Tensor,
    grad: &ndarray::ArrayViewD<'_, f32>,
    b: usize,
    h: usize,
    m: usize,
    k: usize,
    n: usize,
) -> (ndarray::ArrayD<f32>, ndarray::ArrayD<f32>) {
    if let Some(grads) = try_batch_matmul_backward_cpu_native_lowp(lhs, rhs, grad, b, h, m, k, n) {
        return grads;
    }

    let grad_view = grad.view().into_dimensionality::<Ix4>().unwrap();
    with_native_f32_compute_view(lhs, |lhs_view| {
        with_native_f32_compute_view(rhs, |rhs_view| {
            let lhs_view = lhs_view.into_dimensionality::<Ix4>().unwrap();
            let rhs_view = rhs_view.into_dimensionality::<Ix4>().unwrap();

            let mut d_lhs = Array4::<f32>::zeros((b, h, m, k));
            Zip::from(d_lhs.outer_iter_mut())
                .and(grad_view.outer_iter())
                .and(rhs_view.outer_iter())
                .for_each(|mut d_l_b, g_b, r_b| {
                    Zip::from(d_l_b.outer_iter_mut())
                        .and(g_b.outer_iter())
                        .and(r_b.outer_iter())
                        .for_each(|mut d_l_mat, g_mat, r_mat| {
                            general_mat_mul(1.0, &g_mat, &r_mat.t(), 0.0, &mut d_l_mat);
                        });
                });

            let mut d_rhs = Array4::<f32>::zeros((b, h, k, n));
            Zip::from(d_rhs.outer_iter_mut())
                .and(lhs_view.outer_iter())
                .and(grad_view.outer_iter())
                .for_each(|mut d_r_b, l_b, g_b| {
                    Zip::from(d_r_b.outer_iter_mut())
                        .and(l_b.outer_iter())
                        .and(g_b.outer_iter())
                        .for_each(|mut d_r_mat, l_mat, g_mat| {
                            general_mat_mul(1.0, &l_mat.t(), &g_mat, 0.0, &mut d_r_mat);
                        });
                });

            (d_lhs.into_dyn(), d_rhs.into_dyn())
        })
    })
}

// lhs: [B, H, M, K]
// rhs: [B, H, K, N]
// out: [B, H, M, N]
pub fn batch_matmul(lhs: &Tensor, rhs: &Tensor) -> Tensor {
    let output_device = assert_same_device(lhs, rhs, "batch_matmul");
    let build_graph = !is_no_grad() && (lhs.requires_grad() || rhs.requires_grad());
    let cuda_native_supported = output_device == crate::autograd::Device::Cuda;
    assert_native_device_support(output_device, "batch_matmul", cuda_native_supported);

    let lhs_shape = lhs.shape_vec();
    let rhs_shape = rhs.shape_vec();
    assert_eq!(lhs_shape.len(), 4, "batch_matmul lhs must be [B,H,M,K]");
    assert_eq!(rhs_shape.len(), 4, "batch_matmul rhs must be [B,H,K,N]");

    let (b, h, m, k) = (lhs_shape[0], lhs_shape[1], lhs_shape[2], lhs_shape[3]);
    let (b2, h2, k2, n) = (rhs_shape[0], rhs_shape[1], rhs_shape[2], rhs_shape[3]);

    assert_eq!(b, b2, "batch dim mismatch");
    assert_eq!(h, h2, "head dim mismatch");
    assert_eq!(k, k2, "k dim mismatch");
    let dims = BatchMatmulDims { b, h, m, k, n };

    if build_graph
        && output_device == crate::autograd::Device::Cuda
        && let Some(buffer) = try_cuda_batch_matmul_buffer(lhs, rhs, b, h, m, k, n)
    {
        let lhs_clone = lhs.clone();
        let rhs_clone = rhs.clone();
        let out_shape = vec![b, h, m, n];
        let output_self = Rc::new(RefCell::new(None::<Tensor>));
        let output_self_for_backward = output_self.clone();
        let tensor = Tensor(Rc::new(RefCell::new(TensorData {
            data: ndarray::ArrayD::<f32>::zeros(ndarray::IxDyn(&out_shape)).into_shared(),
            f16_data: None,
            bf16_data: None,
            i8_data: None,
            cuda_f32_data: Some(buffer),
            cuda_f16_data: None,
            cuda_bf16_data: None,
            cuda_i8_data: None,
            i8_scale: None,
            has_f32_data: false,
            storage_dtype: crate::precision::DType::F32,
            cache_dirty: false,
            is_parameter: false,
            grad: None,
            cuda_f32_grad: None,
            parents: vec![lhs_clone.clone(), rhs_clone.clone()],
            backward_op: Some(std::rc::Rc::new(move |grad: &ndarray::ArrayViewD<f32>| {
                let cuda_grad = output_self_for_backward
                    .borrow()
                    .as_ref()
                    .and_then(|output| output.cloned_cuda_f32_grad())
                    .filter(|buffer| buffer.len() == grad.len());
                if is_strict_device_execution() {
                    match try_cuda_training_batch_matmul_backward_buffers(
                        grad,
                        cuda_grad.clone(),
                        &lhs_clone,
                        &rhs_clone,
                        dims,
                    ) {
                        Ok((d_lhs_buf, d_rhs_buf)) => {
                            lhs_clone.add_cuda_grad_buffer_only(d_lhs_buf);
                            rhs_clone.add_cuda_grad_buffer_only(d_rhs_buf);
                            return;
                        }
                        Err(err) => {
                            panic!(
                                "CUDA batch_matmul backward failed in strict device execution mode: {err}"
                            );
                        }
                    }
                }
                let cuda_result = try_cuda_training_batch_matmul_backward(
                    grad, cuda_grad, &lhs_clone, &rhs_clone, dims,
                );
                match cuda_result {
                    Ok(((d_lhs, d_lhs_buf), (d_rhs, d_rhs_buf))) => {
                        lhs_clone.add_grad_with_cuda_buffer(d_lhs, Some(d_lhs_buf));
                        rhs_clone.add_grad_with_cuda_buffer(d_rhs, Some(d_rhs_buf));
                    }
                    Err(err) => {
                        if is_strict_device_execution() {
                            panic!(
                                "CUDA batch_matmul backward failed in strict device execution mode: {err}"
                            );
                        }
                        let (d_lhs, d_rhs) = batch_matmul_backward_cpu_f32(
                            &lhs_clone, &rhs_clone, grad, b, h, m, k, n,
                        );
                        lhs_clone.add_grad(d_lhs);
                        rhs_clone.add_grad(d_rhs);
                    }
                }
            })),
            requires_grad: true,
            device: output_device,
        })));
        *output_self.borrow_mut() = Some(tensor.clone());
        return tensor;
    }

    if !build_graph {
        let output_dtype = if lhs.dtype() == rhs.dtype() {
            lhs.dtype()
        } else {
            DType::F32
        };
        if let Some(cuda_out) = try_cuda_batch_matmul(lhs, rhs, dims, output_dtype) {
            return cuda_out;
        }
        let output_dyn = batch_matmul_forward_cpu_f32(lhs, rhs, b, h, m, n);

        return Tensor::from_f32_data_no_grad_with_device_dtype(
            output_dyn,
            output_dtype,
            output_device,
        );
    }

    let output_dyn = batch_matmul_forward_cpu_f32(lhs, rhs, b, h, m, n);

    let lhs_clone = lhs.clone();
    let rhs_clone = rhs.clone();

    Tensor(Rc::new(RefCell::new(TensorData {
        data: output_dyn.into_shared(),
        f16_data: None,
        bf16_data: None,
        i8_data: None,
        cuda_f32_data: None,
        cuda_f16_data: None,
        cuda_bf16_data: None,
        cuda_i8_data: None,
        i8_scale: None,
        has_f32_data: true,
        storage_dtype: crate::precision::DType::F32,
        cache_dirty: false,
        is_parameter: false,
        grad: None,
        cuda_f32_grad: None,
        parents: vec![lhs_clone.clone(), rhs_clone.clone()],
        backward_op: Some(std::rc::Rc::new(move |grad: &ndarray::ArrayViewD<f32>| {
            let (d_lhs, d_rhs) =
                batch_matmul_backward_cpu_f32(&lhs_clone, &rhs_clone, grad, b, h, m, k, n);
            lhs_clone.add_grad(d_lhs);
            rhs_clone.add_grad(d_rhs);
        })),
        requires_grad: true,
        device: output_device,
    })))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::no_grad;
    #[cfg(feature = "cuda")]
    use crate::autograd::set_strict_device_execution;
    use crate::ops::arithmetic::sum;
    use crate::precision::{PrecisionConfig, with_precision_config};
    #[cfg(feature = "cuda")]
    use ndarray::{Array, IxDyn};

    fn sample_f32(len: usize) -> Vec<f32> {
        (0..len)
            .map(|i| (((i * 17 + 11) % 29) as f32) / 13.0 - 1.0)
            .collect()
    }

    fn to_bf16(src: &[f32]) -> Vec<bf16> {
        src.iter().map(|&v| bf16::from_f32(v)).collect()
    }

    fn to_f16(src: &[f32]) -> Vec<f16> {
        src.iter().map(|&v| f16::from_f32(v)).collect()
    }

    fn bf16_to_f32(src: &[bf16]) -> Vec<f32> {
        src.iter().map(|&v| v.to_f32()).collect()
    }

    #[cfg(feature = "cuda")]
    fn make_grad_tensor(shape: &[usize], data: Vec<f32>) -> Tensor {
        Tensor::from_data_with_grad_flag(
            Array::from_shape_vec(IxDyn(shape), data)
                .expect("test tensor shape mismatch")
                .into_dyn(),
            true,
        )
    }

    fn f16_to_f32(src: &[f16]) -> Vec<f32> {
        src.iter().map(|&v| v.to_f32()).collect()
    }

    fn i8_storage(src: &[f32]) -> (Vec<f32>, Vec<i8>, f32) {
        let tensor = make_tensor(&[src.len()], src.to_vec(), DType::I8);
        match tensor.native_storage_owned() {
            TensorStorageOwned::I8(data, scale) => (
                tensor.data_ref().iter().copied().collect(),
                data.iter().copied().collect(),
                scale,
            ),
            TensorStorageOwned::F32(_)
            | TensorStorageOwned::F16(_)
            | TensorStorageOwned::BF16(_) => {
                panic!("expected i8 storage")
            }
        }
    }

    fn assert_close(lhs: &[f32], rhs: &[f32], tol: f32) {
        assert_eq!(lhs.len(), rhs.len());
        for (idx, (&a, &b)) in lhs.iter().zip(rhs.iter()).enumerate() {
            assert!(
                (a - b).abs() <= tol,
                "mismatch at {idx}: lhs={a}, rhs={b}, tol={tol}"
            );
        }
    }

    #[test]
    fn bf16_input_matvec_matches_quantized_reference() {
        let k_dim = 11usize;
        let n_rows = 7usize;
        let x = sample_f32(k_dim);
        let x_bf16 = to_bf16(&x);
        let w = sample_f32(n_rows * k_dim);
        let w_bf16 = to_bf16(&w);

        let mut ref_out = vec![0.0f32; n_rows];
        let mut out = vec![0.0f32; n_rows];

        let x_q = bf16_to_f32(&x_bf16);
        matvec_rowmajor_parallel(&x_q, &w, n_rows, k_dim, &mut ref_out);
        matvec_rowmajor_parallel_mixed(
            SliceRef::BF16(&x_bf16),
            SliceRef::F32(&w),
            n_rows,
            k_dim,
            &mut out,
        );
        assert_close(&ref_out, &out, 1e-5);

        let w_q = bf16_to_f32(&w_bf16);
        matvec_rowmajor_parallel(&x_q, &w_q, n_rows, k_dim, &mut ref_out);
        matvec_rowmajor_parallel_mixed(
            SliceRef::BF16(&x_bf16),
            SliceRef::BF16(&w_bf16),
            n_rows,
            k_dim,
            &mut out,
        );
        assert_close(&ref_out, &out, 1e-5);
    }

    #[test]
    fn bf16_input_dual_matvec_matches_quantized_reference() {
        let k_dim = 13usize;
        let n_rows = 5usize;
        let x = sample_f32(k_dim);
        let x_bf16 = to_bf16(&x);
        let w0 = sample_f32(n_rows * k_dim);
        let w1 = sample_f32(n_rows * k_dim)
            .into_iter()
            .map(|v| v * 0.7 - 0.1)
            .collect::<Vec<_>>();
        let w0_bf16 = to_bf16(&w0);
        let w1_bf16 = to_bf16(&w1);
        let x_q = bf16_to_f32(&x_bf16);

        let mut ref0 = vec![0.0f32; n_rows];
        let mut ref1 = vec![0.0f32; n_rows];
        let mut out0 = vec![0.0f32; n_rows];
        let mut out1 = vec![0.0f32; n_rows];

        dual_matvec_rowmajor_parallel(&x_q, &w0, &w1, n_rows, k_dim, &mut ref0, &mut ref1);
        dual_matvec_rowmajor_parallel_mixed(
            SliceRef::BF16(&x_bf16),
            SliceRef::F32(&w0),
            SliceRef::F32(&w1),
            n_rows,
            k_dim,
            &mut out0,
            &mut out1,
        );
        assert_close(&ref0, &out0, 1e-5);
        assert_close(&ref1, &out1, 1e-5);

        let w0_q = bf16_to_f32(&w0_bf16);
        let w1_q = bf16_to_f32(&w1_bf16);
        dual_matvec_rowmajor_parallel(&x_q, &w0_q, &w1_q, n_rows, k_dim, &mut ref0, &mut ref1);
        dual_matvec_rowmajor_parallel_mixed(
            SliceRef::BF16(&x_bf16),
            SliceRef::BF16(&w0_bf16),
            SliceRef::BF16(&w1_bf16),
            n_rows,
            k_dim,
            &mut out0,
            &mut out1,
        );
        assert_close(&ref0, &out0, 1e-5);
        assert_close(&ref1, &out1, 1e-5);
    }

    #[test]
    fn bf16_input_silu_matches_quantized_reference() {
        let k_dim = 9usize;
        let n_rows = 6usize;
        let x = sample_f32(k_dim);
        let x_bf16 = to_bf16(&x);
        let gate = sample_f32(n_rows * k_dim);
        let up = sample_f32(n_rows * k_dim)
            .into_iter()
            .map(|v| v * -0.5 + 0.2)
            .collect::<Vec<_>>();
        let gate_bf16 = to_bf16(&gate);
        let up_bf16 = to_bf16(&up);
        let x_q = bf16_to_f32(&x_bf16);

        let mut ref_out = vec![0.0f32; n_rows];
        let mut out = vec![0.0f32; n_rows];

        dual_matvec_silu_mul_rowmajor_parallel(&x_q, &gate, &up, n_rows, k_dim, &mut ref_out);
        dual_matvec_silu_mul_rowmajor_parallel_mixed(
            SliceRef::BF16(&x_bf16),
            SliceRef::F32(&gate),
            SliceRef::F32(&up),
            n_rows,
            k_dim,
            &mut out,
        );
        assert_close(&ref_out, &out, 1e-5);

        let gate_q = bf16_to_f32(&gate_bf16);
        let up_q = bf16_to_f32(&up_bf16);
        dual_matvec_silu_mul_rowmajor_parallel(&x_q, &gate_q, &up_q, n_rows, k_dim, &mut ref_out);
        dual_matvec_silu_mul_rowmajor_parallel_mixed(
            SliceRef::BF16(&x_bf16),
            SliceRef::BF16(&gate_bf16),
            SliceRef::BF16(&up_bf16),
            n_rows,
            k_dim,
            &mut out,
        );
        assert_close(&ref_out, &out, 1e-5);
    }

    #[test]
    fn bf16_input_argmax_matches_quantized_reference() {
        let k_dim = 10usize;
        let n_rows = 12usize;
        let x = sample_f32(k_dim);
        let x_bf16 = to_bf16(&x);
        let w = sample_f32(n_rows * k_dim);
        let w_bf16 = to_bf16(&w);
        let x_q = bf16_to_f32(&x_bf16);
        let w_q = bf16_to_f32(&w_bf16);

        let idx_f32 = matvec_argmax_rowmajor_parallel(&x_q, &w, n_rows, k_dim);
        let idx_bf16f32 = matvec_argmax_rowmajor_parallel_mixed(
            SliceRef::BF16(&x_bf16),
            SliceRef::F32(&w),
            n_rows,
            k_dim,
        );
        assert_eq!(idx_f32, idx_bf16f32);

        let idx_q = matvec_argmax_rowmajor_parallel(&x_q, &w_q, n_rows, k_dim);
        let idx_bf16bf16 = matvec_argmax_rowmajor_parallel_mixed(
            SliceRef::BF16(&x_bf16),
            SliceRef::BF16(&w_bf16),
            n_rows,
            k_dim,
        );
        assert_eq!(idx_q, idx_bf16bf16);
    }

    #[test]
    fn f16_input_matvec_matches_quantized_reference() {
        let k_dim = 11usize;
        let n_rows = 7usize;
        let x = sample_f32(k_dim);
        let x_f16 = to_f16(&x);
        let w = sample_f32(n_rows * k_dim);
        let w_f16 = to_f16(&w);

        let mut ref_out = vec![0.0f32; n_rows];
        let mut out = vec![0.0f32; n_rows];

        let x_q = f16_to_f32(&x_f16);
        matvec_rowmajor_parallel(&x_q, &w, n_rows, k_dim, &mut ref_out);
        matvec_rowmajor_parallel_mixed(
            SliceRef::F16(&x_f16),
            SliceRef::F32(&w),
            n_rows,
            k_dim,
            &mut out,
        );
        assert_close(&ref_out, &out, 1e-5);

        let w_q = f16_to_f32(&w_f16);
        matvec_rowmajor_parallel(&x_q, &w_q, n_rows, k_dim, &mut ref_out);
        matvec_rowmajor_parallel_mixed(
            SliceRef::F16(&x_f16),
            SliceRef::F16(&w_f16),
            n_rows,
            k_dim,
            &mut out,
        );
        assert_close(&ref_out, &out, 1e-5);
    }

    #[test]
    fn f16_input_dual_matvec_matches_quantized_reference() {
        let k_dim = 13usize;
        let n_rows = 5usize;
        let x = sample_f32(k_dim);
        let x_f16 = to_f16(&x);
        let w0 = sample_f32(n_rows * k_dim);
        let w1 = sample_f32(n_rows * k_dim)
            .into_iter()
            .map(|v| v * 0.7 - 0.1)
            .collect::<Vec<_>>();
        let w0_f16 = to_f16(&w0);
        let w1_f16 = to_f16(&w1);
        let x_q = f16_to_f32(&x_f16);

        let mut ref0 = vec![0.0f32; n_rows];
        let mut ref1 = vec![0.0f32; n_rows];
        let mut out0 = vec![0.0f32; n_rows];
        let mut out1 = vec![0.0f32; n_rows];

        dual_matvec_rowmajor_parallel(&x_q, &w0, &w1, n_rows, k_dim, &mut ref0, &mut ref1);
        dual_matvec_rowmajor_parallel_mixed(
            SliceRef::F16(&x_f16),
            SliceRef::F32(&w0),
            SliceRef::F32(&w1),
            n_rows,
            k_dim,
            &mut out0,
            &mut out1,
        );
        assert_close(&ref0, &out0, 1e-5);
        assert_close(&ref1, &out1, 1e-5);

        let w0_q = f16_to_f32(&w0_f16);
        let w1_q = f16_to_f32(&w1_f16);
        dual_matvec_rowmajor_parallel(&x_q, &w0_q, &w1_q, n_rows, k_dim, &mut ref0, &mut ref1);
        dual_matvec_rowmajor_parallel_mixed(
            SliceRef::F16(&x_f16),
            SliceRef::F16(&w0_f16),
            SliceRef::F16(&w1_f16),
            n_rows,
            k_dim,
            &mut out0,
            &mut out1,
        );
        assert_close(&ref0, &out0, 1e-5);
        assert_close(&ref1, &out1, 1e-5);
    }

    #[test]
    fn f16_input_silu_matches_quantized_reference() {
        let k_dim = 9usize;
        let n_rows = 6usize;
        let x = sample_f32(k_dim);
        let x_f16 = to_f16(&x);
        let gate = sample_f32(n_rows * k_dim);
        let up = sample_f32(n_rows * k_dim)
            .into_iter()
            .map(|v| v * 0.5 + 0.2)
            .collect::<Vec<_>>();
        let gate_f16 = to_f16(&gate);
        let up_f16 = to_f16(&up);
        let x_q = f16_to_f32(&x_f16);

        let mut ref_out = vec![0.0f32; n_rows];
        let mut out = vec![0.0f32; n_rows];

        dual_matvec_silu_mul_rowmajor_parallel(&x_q, &gate, &up, n_rows, k_dim, &mut ref_out);
        dual_matvec_silu_mul_rowmajor_parallel_mixed(
            SliceRef::F16(&x_f16),
            SliceRef::F32(&gate),
            SliceRef::F32(&up),
            n_rows,
            k_dim,
            &mut out,
        );
        assert_close(&ref_out, &out, 1e-5);

        let gate_q = f16_to_f32(&gate_f16);
        let up_q = f16_to_f32(&up_f16);
        dual_matvec_silu_mul_rowmajor_parallel(&x_q, &gate_q, &up_q, n_rows, k_dim, &mut ref_out);
        dual_matvec_silu_mul_rowmajor_parallel_mixed(
            SliceRef::F16(&x_f16),
            SliceRef::F16(&gate_f16),
            SliceRef::F16(&up_f16),
            n_rows,
            k_dim,
            &mut out,
        );
        assert_close(&ref_out, &out, 1e-5);
    }

    #[test]
    fn f16_input_argmax_matches_quantized_reference() {
        let k_dim = 11usize;
        let n_rows = 7usize;
        let x = sample_f32(k_dim);
        let x_f16 = to_f16(&x);
        let w = sample_f32(n_rows * k_dim);
        let w_f16 = to_f16(&w);
        let x_q = f16_to_f32(&x_f16);
        let w_q = f16_to_f32(&w_f16);

        let idx_f32 = matvec_argmax_rowmajor_parallel(&x_q, &w, n_rows, k_dim);
        let idx_f16f32 = matvec_argmax_rowmajor_parallel_mixed(
            SliceRef::F16(&x_f16),
            SliceRef::F32(&w),
            n_rows,
            k_dim,
        );
        assert_eq!(idx_f32, idx_f16f32);

        let idx_q = matvec_argmax_rowmajor_parallel(&x_q, &w_q, n_rows, k_dim);
        let idx_f16f16 = matvec_argmax_rowmajor_parallel_mixed(
            SliceRef::F16(&x_f16),
            SliceRef::F16(&w_f16),
            n_rows,
            k_dim,
        );
        assert_eq!(idx_q, idx_f16f16);
    }

    #[test]
    fn i8_weight_matvec_matches_quantized_reference() {
        let k_dim = 11usize;
        let n_rows = 7usize;
        let x = sample_f32(k_dim);
        let w = sample_f32(n_rows * k_dim);
        let (w_q, w_i8, w_scale) = i8_storage(&w);

        let mut ref_out = vec![0.0f32; n_rows];
        let mut out = vec![0.0f32; n_rows];

        matvec_rowmajor_parallel(&x, &w_q, n_rows, k_dim, &mut ref_out);
        matvec_rowmajor_parallel_mixed(
            SliceRef::F32(&x),
            SliceRef::I8(&w_i8, w_scale),
            n_rows,
            k_dim,
            &mut out,
        );
        assert_close(&ref_out, &out, 1e-5);
    }

    #[test]
    fn i8_weight_dual_matvec_matches_quantized_reference() {
        let k_dim = 13usize;
        let n_rows = 5usize;
        let x = sample_f32(k_dim);
        let w0 = sample_f32(n_rows * k_dim);
        let w1 = sample_f32(n_rows * k_dim)
            .into_iter()
            .map(|v| v * 0.7 - 0.1)
            .collect::<Vec<_>>();
        let (w0_q, w0_i8, w0_scale) = i8_storage(&w0);
        let (w1_q, w1_i8, w1_scale) = i8_storage(&w1);

        let mut ref0 = vec![0.0f32; n_rows];
        let mut ref1 = vec![0.0f32; n_rows];
        let mut out0 = vec![0.0f32; n_rows];
        let mut out1 = vec![0.0f32; n_rows];

        dual_matvec_rowmajor_parallel(&x, &w0_q, &w1_q, n_rows, k_dim, &mut ref0, &mut ref1);
        dual_matvec_rowmajor_parallel_mixed(
            SliceRef::F32(&x),
            SliceRef::I8(&w0_i8, w0_scale),
            SliceRef::I8(&w1_i8, w1_scale),
            n_rows,
            k_dim,
            &mut out0,
            &mut out1,
        );
        assert_close(&ref0, &out0, 1e-5);
        assert_close(&ref1, &out1, 1e-5);
    }

    #[test]
    fn i8_weight_silu_matches_quantized_reference() {
        let k_dim = 9usize;
        let n_rows = 6usize;
        let x = sample_f32(k_dim);
        let gate = sample_f32(n_rows * k_dim);
        let up = sample_f32(n_rows * k_dim)
            .into_iter()
            .map(|v| v * -0.5 + 0.2)
            .collect::<Vec<_>>();
        let (gate_q, gate_i8, gate_scale) = i8_storage(&gate);
        let (up_q, up_i8, up_scale) = i8_storage(&up);

        let mut ref_out = vec![0.0f32; n_rows];
        let mut out = vec![0.0f32; n_rows];

        dual_matvec_silu_mul_rowmajor_parallel(&x, &gate_q, &up_q, n_rows, k_dim, &mut ref_out);
        dual_matvec_silu_mul_rowmajor_parallel_mixed(
            SliceRef::F32(&x),
            SliceRef::I8(&gate_i8, gate_scale),
            SliceRef::I8(&up_i8, up_scale),
            n_rows,
            k_dim,
            &mut out,
        );
        assert_close(&ref_out, &out, 1e-5);
    }

    #[test]
    fn i8_weight_argmax_matches_quantized_reference() {
        let k_dim = 10usize;
        let n_rows = 12usize;
        let x = sample_f32(k_dim);
        let w = sample_f32(n_rows * k_dim);
        let (w_q, w_i8, w_scale) = i8_storage(&w);

        let idx_q = matvec_argmax_rowmajor_parallel(&x, &w_q, n_rows, k_dim);
        let idx_i8 = matvec_argmax_rowmajor_parallel_mixed(
            SliceRef::F32(&x),
            SliceRef::I8(&w_i8, w_scale),
            n_rows,
            k_dim,
        );
        assert_eq!(idx_q, idx_i8);
    }

    #[test]
    fn nested_bf16_input_conversion_is_reentrant() {
        let lhs = to_bf16(&sample_f32(5));
        let rhs = to_bf16(&sample_f32(3));
        let lhs_q = bf16_to_f32(&lhs);
        let rhs_q = bf16_to_f32(&rhs);

        with_bf16_input_as_f32(&lhs, |lhs_f32| {
            assert_close(lhs_f32, &lhs_q, 1e-6);
            with_bf16_input_as_f32(&rhs, |rhs_f32| {
                assert_close(rhs_f32, &rhs_q, 1e-6);
            });
        });
    }

    fn make_tensor(shape: &[usize], data: Vec<f32>, dtype: DType) -> Tensor {
        let t = Tensor::from_array_no_grad(
            ndarray::Array::from_shape_vec(ndarray::IxDyn(shape), data)
                .expect("test tensor shape mismatch")
                .into_dyn(),
        );
        t.cast_inplace(dtype);
        t
    }

    fn make_training_tensor(shape: &[usize], data: Vec<f32>, dtype: DType) -> Tensor {
        Tensor::new_with_dtype(
            ndarray::Array::from_shape_vec(ndarray::IxDyn(shape), data)
                .expect("test tensor shape mismatch")
                .into_dyn(),
            dtype,
        )
    }

    fn make_training_parameter(shape: &[usize], data: Vec<f32>, dtype: DType) -> Tensor {
        Tensor::parameter_with_dtype(
            ndarray::Array::from_shape_vec(ndarray::IxDyn(shape), data)
                .expect("test tensor shape mismatch")
                .into_dyn(),
            dtype,
        )
    }

    #[test]
    fn matmul_training_forward_keeps_low_precision_storage_until_backward() {
        for allow_parameter_dtype_copies in [false, true] {
            with_precision_config(
                PrecisionConfig {
                    parameter_dtype: DType::F32,
                    runtime_dtype: DType::F32,
                    allow_parameter_dtype_copies,
                },
                || {
                    for dtype in [DType::F16, DType::BF16, DType::I8] {
                        let a = make_training_tensor(
                            &[2, 4],
                            vec![1.0, -2.0, 0.5, 3.0, -1.0, 0.25, 1.5, -0.5],
                            dtype,
                        );
                        let b = make_training_parameter(
                            &[3, 4],
                            vec![
                                1.0, 0.0, -1.0, 2.0, 0.5, 1.5, -0.5, 0.25, -1.0, 2.0, 1.0, -0.5,
                            ],
                            dtype,
                        );

                        assert_eq!(a.dtype(), dtype);
                        assert_eq!(b.dtype(), dtype);
                        assert!(!a.has_host_f32_data());
                        assert!(!b.has_host_f32_data());

                        let out = matmul(&a, &b);
                        assert_eq!(out.dtype(), DType::F32);
                        assert_eq!(out.shape_vec(), vec![2, 3]);
                        assert!(
                            !a.has_host_f32_data(),
                            "lhs {dtype:?} materialized during forward"
                        );
                        assert!(
                            !b.has_host_f32_data(),
                            "rhs parameter {dtype:?} materialized during forward"
                        );

                        sum(&out).backward();
                        let a_grad = a.grad().expect("lhs grad");
                        let b_grad = b.grad().expect("rhs grad");
                        assert_eq!(a_grad.shape(), &[2, 4]);
                        assert_eq!(b_grad.shape(), &[3, 4]);
                        assert!(a_grad.iter().all(|v| v.is_finite()));
                        assert!(b_grad.iter().all(|v| v.is_finite()));
                        assert_eq!(a.dtype(), dtype);
                        assert_eq!(b.dtype(), dtype);
                        assert!(
                            !a.has_host_f32_data(),
                            "lhs {dtype:?} materialized during backward"
                        );
                        assert!(
                            !b.has_host_f32_data(),
                            "rhs parameter {dtype:?} materialized during backward"
                        );
                    }
                },
            );
        }
    }

    #[test]
    fn matmul_backward_reads_native_low_precision_without_f32_materialization() {
        fn decoded(tensor: &Tensor) -> Vec<f32> {
            match tensor.native_storage_owned() {
                TensorStorageOwned::F32(data) => data.iter().copied().collect(),
                TensorStorageOwned::F16(data) => data.iter().map(|v| v.to_f32()).collect(),
                TensorStorageOwned::BF16(data) => data.iter().map(|v| v.to_f32()).collect(),
                TensorStorageOwned::I8(data, scale) => {
                    data.iter().map(|&v| (v as f32) * scale).collect()
                }
            }
        }

        let grad = ndarray::Array::from_shape_vec(
            ndarray::IxDyn(&[2, 3]),
            vec![1.0f32, -0.5, 0.25, 2.0, -1.0, 0.5],
        )
        .expect("grad shape mismatch")
        .into_dyn();

        for (lhs_dtype, rhs_dtype) in [
            (DType::F16, DType::F16),
            (DType::BF16, DType::BF16),
            (DType::I8, DType::I8),
            (DType::F32, DType::I8),
            (DType::F16, DType::I8),
            (DType::BF16, DType::I8),
            (DType::I8, DType::F32),
            (DType::I8, DType::F16),
            (DType::I8, DType::BF16),
        ] {
            let a = make_tensor(
                &[2, 4],
                vec![1.0, -2.0, 0.5, 3.0, -1.0, 0.25, 1.5, -0.5],
                lhs_dtype,
            );
            let b = make_tensor(
                &[3, 4],
                vec![
                    1.0, 0.0, -1.0, 2.0, 0.5, 1.5, -0.5, 0.25, -1.0, 2.0, 1.0, -0.5,
                ],
                rhs_dtype,
            );
            let lhs_was_lowp = lhs_dtype != DType::F32;
            let rhs_was_lowp = rhs_dtype != DType::F32;
            if lhs_was_lowp {
                assert!(!a.has_host_f32_data());
            }
            if rhs_was_lowp {
                assert!(!b.has_host_f32_data());
            }

            let (da, db) = matmul_backward_cpu_f32(&a, &b, &grad.view(), 2, 4, 3);
            if lhs_was_lowp {
                assert!(
                    !a.has_host_f32_data(),
                    "lhs {lhs_dtype:?} materialized during matmul backward"
                );
            }
            if rhs_was_lowp {
                assert!(
                    !b.has_host_f32_data(),
                    "rhs {rhs_dtype:?} materialized during matmul backward"
                );
            }

            let a_vals = decoded(&a);
            let b_vals = decoded(&b);
            let grad_vals = grad.as_slice_memory_order().unwrap();
            for m in 0..2 {
                for k in 0..4 {
                    let expected = (0..3)
                        .map(|n| grad_vals[m * 3 + n] * b_vals[n * 4 + k])
                        .sum::<f32>();
                    let got = da.as_slice_memory_order().unwrap()[m * 4 + k];
                    assert!(
                        (got - expected).abs() <= 0.08,
                        "lhs={lhs_dtype:?} rhs={rhs_dtype:?} da[{m},{k}] got {got}, expected {expected}"
                    );
                }
            }
            for n in 0..3 {
                for k in 0..4 {
                    let expected = (0..2)
                        .map(|m| grad_vals[m * 3 + n] * a_vals[m * 4 + k])
                        .sum::<f32>();
                    let got = db.as_slice_memory_order().unwrap()[n * 4 + k];
                    assert!(
                        (got - expected).abs() <= 0.08,
                        "lhs={lhs_dtype:?} rhs={rhs_dtype:?} db[{n},{k}] got {got}, expected {expected}"
                    );
                }
            }
        }
    }

    #[test]
    fn batch_matmul_backward_reads_native_low_precision_without_f32_materialization() {
        fn decoded(tensor: &Tensor) -> Vec<f32> {
            match tensor.native_storage_owned() {
                TensorStorageOwned::F32(data) => data.iter().copied().collect(),
                TensorStorageOwned::F16(data) => data.iter().map(|v| v.to_f32()).collect(),
                TensorStorageOwned::BF16(data) => data.iter().map(|v| v.to_f32()).collect(),
                TensorStorageOwned::I8(data, scale) => {
                    data.iter().map(|&v| (v as f32) * scale).collect()
                }
            }
        }

        let grad = ndarray::Array::from_shape_vec(
            ndarray::IxDyn(&[1, 2, 2, 3]),
            vec![
                1.0f32, -0.5, 0.25, 2.0, -1.0, 0.5, -0.75, 1.25, -1.5, 0.5, 0.75, -0.25,
            ],
        )
        .expect("grad shape mismatch")
        .into_dyn();

        for (lhs_dtype, rhs_dtype) in [
            (DType::F16, DType::F16),
            (DType::BF16, DType::BF16),
            (DType::I8, DType::I8),
            (DType::F32, DType::I8),
            (DType::F16, DType::I8),
            (DType::BF16, DType::I8),
            (DType::I8, DType::F32),
            (DType::I8, DType::F16),
            (DType::I8, DType::BF16),
        ] {
            let lhs = make_tensor(
                &[1, 2, 2, 4],
                vec![
                    1.0, -2.0, 0.5, 3.0, -1.0, 0.25, 1.5, -0.5, 0.75, -1.25, 2.0, -2.5, 1.25, 0.5,
                    -0.75, 2.25,
                ],
                lhs_dtype,
            );
            let rhs = make_tensor(
                &[1, 2, 4, 3],
                vec![
                    1.0, 0.0, -1.0, 2.0, 0.5, 1.5, -0.5, 0.25, -1.0, 2.0, 1.0, -0.5, -0.25, 0.75,
                    1.25, -1.5, 0.5, -0.75, 1.75, -1.25, 0.25, 0.5, -2.0, 1.0,
                ],
                rhs_dtype,
            );
            let lhs_was_lowp = lhs_dtype != DType::F32;
            let rhs_was_lowp = rhs_dtype != DType::F32;
            if lhs_was_lowp {
                assert!(!lhs.has_host_f32_data());
            }
            if rhs_was_lowp {
                assert!(!rhs.has_host_f32_data());
            }

            let (d_lhs, d_rhs) =
                batch_matmul_backward_cpu_f32(&lhs, &rhs, &grad.view(), 1, 2, 2, 4, 3);

            if lhs_was_lowp {
                assert!(
                    !lhs.has_host_f32_data(),
                    "lhs {lhs_dtype:?} materialized during batch backward"
                );
            }
            if rhs_was_lowp {
                assert!(
                    !rhs.has_host_f32_data(),
                    "rhs {rhs_dtype:?} materialized during batch backward"
                );
            }

            let lhs_vals = decoded(&lhs);
            let rhs_vals = decoded(&rhs);
            let grad_vals = grad.as_slice_memory_order().unwrap();
            let d_lhs_vals = d_lhs.as_slice_memory_order().unwrap();
            let d_rhs_vals = d_rhs.as_slice_memory_order().unwrap();
            for batch in 0..2 {
                for row in 0..2 {
                    for kk in 0..4 {
                        let expected = (0..3)
                            .map(|col| {
                                grad_vals[batch * 2 * 3 + row * 3 + col]
                                    * rhs_vals[batch * 4 * 3 + kk * 3 + col]
                            })
                            .sum::<f32>();
                        let got = d_lhs_vals[batch * 2 * 4 + row * 4 + kk];
                        assert!(
                            (got - expected).abs() <= 0.08,
                            "lhs={lhs_dtype:?} rhs={rhs_dtype:?} d_lhs[{batch},{row},{kk}] got {got}, expected {expected}"
                        );
                    }
                }
                for kk in 0..4 {
                    for col in 0..3 {
                        let expected = (0..2)
                            .map(|row| {
                                lhs_vals[batch * 2 * 4 + row * 4 + kk]
                                    * grad_vals[batch * 2 * 3 + row * 3 + col]
                            })
                            .sum::<f32>();
                        let got = d_rhs_vals[batch * 4 * 3 + kk * 3 + col];
                        assert!(
                            (got - expected).abs() <= 0.08,
                            "lhs={lhs_dtype:?} rhs={rhs_dtype:?} d_rhs[{batch},{kk},{col}] got {got}, expected {expected}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn batch_matmul_training_forward_keeps_low_precision_storage_until_backward() {
        for allow_parameter_dtype_copies in [false, true] {
            with_precision_config(
                PrecisionConfig {
                    parameter_dtype: DType::F32,
                    runtime_dtype: DType::F32,
                    allow_parameter_dtype_copies,
                },
                || {
                    for dtype in [DType::F16, DType::BF16, DType::I8] {
                        let lhs = make_training_tensor(
                            &[1, 1, 2, 3],
                            vec![1.0, -2.0, 0.5, 3.0, -1.0, 2.0],
                            dtype,
                        );
                        let rhs = make_training_parameter(
                            &[1, 1, 3, 2],
                            vec![0.5, 1.0, -1.5, 2.0, 0.25, -0.75],
                            dtype,
                        );

                        assert!(!lhs.has_host_f32_data());
                        assert!(!rhs.has_host_f32_data());

                        let out = batch_matmul(&lhs, &rhs);
                        assert_eq!(out.dtype(), DType::F32);
                        assert_eq!(out.shape_vec(), vec![1, 1, 2, 2]);
                        assert!(
                            !lhs.has_host_f32_data(),
                            "lhs {dtype:?} materialized during forward"
                        );
                        assert!(
                            !rhs.has_host_f32_data(),
                            "rhs parameter {dtype:?} materialized during forward"
                        );

                        sum(&out).backward();
                        let lhs_grad = lhs.grad().expect("lhs grad");
                        let rhs_grad = rhs.grad().expect("rhs grad");
                        assert_eq!(lhs_grad.shape(), &[1, 1, 2, 3]);
                        assert_eq!(rhs_grad.shape(), &[1, 1, 3, 2]);
                        assert!(lhs_grad.iter().all(|v| v.is_finite()));
                        assert!(rhs_grad.iter().all(|v| v.is_finite()));
                        assert_eq!(lhs.dtype(), dtype);
                        assert_eq!(rhs.dtype(), dtype);
                        assert!(
                            !lhs.has_host_f32_data(),
                            "lhs {dtype:?} materialized during backward"
                        );
                        assert!(
                            !rhs.has_host_f32_data(),
                            "rhs parameter {dtype:?} materialized during backward"
                        );
                    }
                },
            );
        }
    }

    #[test]
    fn matmul_no_grad_preserves_bf16_output_dtype() {
        let a = make_tensor(&[1, 4], vec![1.0, -2.0, 0.5, 3.0], DType::BF16);
        let b = make_tensor(
            &[3, 4],
            vec![
                1.0, 0.0, -1.0, 2.0, 0.5, 1.5, -0.5, 0.25, -1.0, 2.0, 1.0, -0.5,
            ],
            DType::BF16,
        );

        let ref_out = no_grad(|| {
            matmul(
                &make_tensor(&[1, 4], vec![1.0, -2.0, 0.5, 3.0], DType::F32),
                &make_tensor(
                    &[3, 4],
                    vec![
                        1.0, 0.0, -1.0, 2.0, 0.5, 1.5, -0.5, 0.25, -1.0, 2.0, 1.0, -0.5,
                    ],
                    DType::F32,
                ),
            )
        });
        let out = no_grad(|| matmul(&a, &b));

        assert_eq!(a.dtype(), DType::BF16);
        assert_eq!(b.dtype(), DType::BF16);
        assert_eq!(out.dtype(), DType::BF16);

        let ref_vals = ref_out
            .data_ref()
            .iter()
            .map(|&v| bf16::from_f32(v).to_f32())
            .collect::<Vec<_>>();
        out.with_storage_view(|view| match view {
            TensorStorageView::BF16(view) => {
                let vals = view.iter().map(|v| v.to_f32()).collect::<Vec<_>>();
                assert_eq!(vals, ref_vals);
            }
            TensorStorageView::F16(_) => panic!("bf16 matmul output should stay bf16 in no-grad"),
            TensorStorageView::F32(_) => panic!("bf16 matmul output should stay bf16 in no-grad"),
        });
    }

    #[test]
    fn matmul_same_dtype_bf16_parameter_uses_native_storage_for_generic_gemm() {
        with_precision_config(
            PrecisionConfig {
                parameter_dtype: DType::BF16,
                runtime_dtype: DType::F32,
                allow_parameter_dtype_copies: true,
            },
            || {
                let a = make_tensor(&[1, 4], vec![1.0, -2.0, 0.5, 3.0], DType::BF16);
                let b = Tensor::parameter_with_dtype(
                    ndarray::Array::from_shape_vec(
                        ndarray::IxDyn(&[3, 4]),
                        vec![
                            1.0, 0.0, -1.0, 2.0, 0.5, 1.5, -0.5, 0.25, -1.0, 2.0, 1.0, -0.5,
                        ],
                    )
                    .expect("parameter shape mismatch")
                    .into_dyn(),
                    DType::BF16,
                );

                {
                    let inner = b.0.borrow();
                    assert!(
                        !inner.has_f32_data,
                        "bf16 parameter should start without cached f32 copy"
                    );
                }

                let out = no_grad(|| matmul(&a, &b));
                assert_eq!(out.dtype(), DType::BF16);

                let inner = b.0.borrow();
                assert!(
                    !inner.has_f32_data,
                    "generic bf16 matmul should read native parameter storage without caching f32"
                );
            },
        );
    }

    #[test]
    fn matmul_mixed_f32_input_uses_native_low_precision_parameter_storage() {
        with_precision_config(
            PrecisionConfig {
                parameter_dtype: DType::BF16,
                runtime_dtype: DType::F32,
                allow_parameter_dtype_copies: true,
            },
            || {
                let a = make_tensor(&[1, 4], vec![1.0, -2.0, 0.5, 3.0], DType::F32);
                let b = Tensor::parameter_with_dtype(
                    ndarray::Array::from_shape_vec(
                        ndarray::IxDyn(&[3, 4]),
                        vec![
                            1.0, 0.0, -1.0, 2.0, 0.5, 1.5, -0.5, 0.25, -1.0, 2.0, 1.0, -0.5,
                        ],
                    )
                    .expect("parameter shape mismatch")
                    .into_dyn(),
                    DType::BF16,
                );

                {
                    let inner = b.0.borrow();
                    assert!(
                        !inner.has_f32_data,
                        "bf16 parameter should start without cached f32 copy"
                    );
                }

                let out = no_grad(|| matmul(&a, &b));
                assert_eq!(out.dtype(), DType::F32);

                let inner = b.0.borrow();
                assert!(
                    !inner.has_f32_data,
                    "mixed f32 input should read native low-precision parameter storage without caching f32"
                );
            },
        );
    }

    #[test]
    fn batch_matmul_no_grad_preserves_bf16_output_dtype() {
        let lhs = make_tensor(
            &[1, 1, 2, 3],
            vec![1.0, -2.0, 0.5, 3.0, -1.0, 2.0],
            DType::BF16,
        );
        let rhs = make_tensor(
            &[1, 1, 3, 2],
            vec![0.5, 1.0, -1.5, 2.0, 0.25, -0.75],
            DType::BF16,
        );

        let ref_out = no_grad(|| {
            batch_matmul(
                &make_tensor(
                    &[1, 1, 2, 3],
                    vec![1.0, -2.0, 0.5, 3.0, -1.0, 2.0],
                    DType::F32,
                ),
                &make_tensor(
                    &[1, 1, 3, 2],
                    vec![0.5, 1.0, -1.5, 2.0, 0.25, -0.75],
                    DType::F32,
                ),
            )
        });
        let out = no_grad(|| batch_matmul(&lhs, &rhs));

        assert_eq!(lhs.dtype(), DType::BF16);
        assert_eq!(rhs.dtype(), DType::BF16);
        assert_eq!(out.dtype(), DType::BF16);

        let ref_vals = ref_out
            .data_ref()
            .iter()
            .map(|&v| bf16::from_f32(v).to_f32())
            .collect::<Vec<_>>();
        out.with_storage_view(|view| match view {
            TensorStorageView::BF16(view) => {
                let vals = view.iter().map(|v| v.to_f32()).collect::<Vec<_>>();
                assert_eq!(vals, ref_vals);
            }
            TensorStorageView::F16(_) => {
                panic!("bf16 batch_matmul output should stay bf16 in no-grad")
            }
            TensorStorageView::F32(_) => {
                panic!("bf16 batch_matmul output should stay bf16 in no-grad")
            }
        });
    }

    #[test]
    fn matmul_no_grad_preserves_f16_output_dtype() {
        let a = make_tensor(&[1, 4], vec![1.0, -2.0, 0.5, 3.0], DType::F16);
        let b = make_tensor(
            &[3, 4],
            vec![
                1.0, 0.0, -1.0, 2.0, 0.5, 1.5, -0.5, 0.25, -1.0, 2.0, 1.0, -0.5,
            ],
            DType::F16,
        );

        let ref_out = no_grad(|| {
            matmul(
                &make_tensor(&[1, 4], vec![1.0, -2.0, 0.5, 3.0], DType::F32),
                &make_tensor(
                    &[3, 4],
                    vec![
                        1.0, 0.0, -1.0, 2.0, 0.5, 1.5, -0.5, 0.25, -1.0, 2.0, 1.0, -0.5,
                    ],
                    DType::F32,
                ),
            )
        });
        let out = no_grad(|| matmul(&a, &b));

        assert_eq!(a.dtype(), DType::F16);
        assert_eq!(b.dtype(), DType::F16);
        assert_eq!(out.dtype(), DType::F16);

        let ref_vals = ref_out
            .data_ref()
            .iter()
            .map(|&v| f16::from_f32(v).to_f32())
            .collect::<Vec<_>>();
        out.with_storage_view(|view| match view {
            TensorStorageView::F16(view) => {
                let vals = view.iter().map(|v| v.to_f32()).collect::<Vec<_>>();
                assert_eq!(vals, ref_vals);
            }
            TensorStorageView::BF16(_) => panic!("f16 matmul output should stay f16 in no-grad"),
            TensorStorageView::F32(_) => panic!("f16 matmul output should stay f16 in no-grad"),
        });
    }

    #[test]
    fn matmul_no_grad_preserves_i8_output_dtype() {
        let a = make_tensor(&[1, 4], vec![1.0, -2.0, 0.5, 3.0], DType::I8);
        let b = make_tensor(
            &[3, 4],
            vec![
                1.0, 0.0, -1.0, 2.0, 0.5, 1.5, -0.5, 0.25, -1.0, 2.0, 1.0, -0.5,
            ],
            DType::I8,
        );

        let ref_a_q = make_tensor(&[1, 4], vec![1.0, -2.0, 0.5, 3.0], DType::I8);
        let ref_b_q = make_tensor(
            &[3, 4],
            vec![
                1.0, 0.0, -1.0, 2.0, 0.5, 1.5, -0.5, 0.25, -1.0, 2.0, 1.0, -0.5,
            ],
            DType::I8,
        );
        let a_q = ref_a_q.data_ref().iter().copied().collect::<Vec<_>>();
        let b_q = ref_b_q.data_ref().iter().copied().collect::<Vec<_>>();
        let ref_out = no_grad(|| {
            matmul(
                &make_tensor(&[1, 4], a_q, DType::F32),
                &make_tensor(&[3, 4], b_q, DType::F32),
            )
        });
        let out = no_grad(|| matmul(&a, &b));

        assert_eq!(out.dtype(), DType::I8);

        let expected = make_tensor(
            &[1, 3],
            ref_out.data_ref().iter().copied().collect(),
            DType::I8,
        );
        let ref_vals = expected.data_ref().iter().copied().collect::<Vec<_>>();
        let out_vals = out.data_ref().iter().copied().collect::<Vec<_>>();
        assert_eq!(out_vals.len(), ref_vals.len());
        for (actual, expected) in out_vals.iter().zip(ref_vals.iter()) {
            assert!(
                (actual - expected).abs() < 1e-5,
                "i8 matmul output drifted: actual={actual}, expected={expected}"
            );
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_matmul_matches_cpu_reference() {
        if !crate::ops::cuda::is_available() {
            return;
        }

        crate::ops::cuda::set_enabled(true);
        let a = make_tensor(&[8, 32], sample_f32(8 * 32), DType::F32);
        let b = make_tensor(&[6, 32], sample_f32(6 * 32), DType::F32);

        let out = no_grad(|| matmul(&a, &b));
        crate::ops::cuda::set_enabled(false);
        let reference = no_grad(|| matmul(&a, &b));

        let out_vals = out.data_ref().iter().copied().collect::<Vec<_>>();
        let ref_vals = reference.data_ref().iter().copied().collect::<Vec<_>>();
        assert_eq!(out_vals.len(), ref_vals.len());
        for (got, expect) in out_vals.iter().zip(ref_vals.iter()) {
            assert!((got - expect).abs() < 1e-3, "got {got}, expect {expect}");
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_batch_matmul_matches_cpu_reference() {
        if !crate::ops::cuda::is_available() {
            return;
        }

        crate::ops::cuda::set_enabled(true);
        let lhs = make_tensor(&[2, 3, 4, 16], sample_f32(2 * 3 * 4 * 16), DType::F32);
        let rhs = make_tensor(&[2, 3, 16, 5], sample_f32(2 * 3 * 16 * 5), DType::F32);

        let out = no_grad(|| batch_matmul(&lhs, &rhs));
        crate::ops::cuda::set_enabled(false);
        let reference = no_grad(|| batch_matmul(&lhs, &rhs));

        let out_vals = out.data_ref().iter().copied().collect::<Vec<_>>();
        let ref_vals = reference.data_ref().iter().copied().collect::<Vec<_>>();
        assert_eq!(out_vals.len(), ref_vals.len());
        for (got, expect) in out_vals.iter().zip(ref_vals.iter()) {
            assert!((got - expect).abs() < 1e-3, "got {got}, expect {expect}");
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_matmul_preserves_bf16_dtype_in_strict_mode() {
        if !crate::ops::cuda::is_available() {
            return;
        }

        let a = make_tensor(&[1, 4], vec![1.0, -2.0, 0.5, 3.0], DType::BF16).to_cuda();
        let b = make_tensor(
            &[3, 4],
            vec![
                1.0, 0.0, -1.0, 2.0, 0.5, 1.5, -0.5, 0.25, -1.0, 2.0, 1.0, -0.5,
            ],
            DType::BF16,
        )
        .to_cuda();
        assert!(a.cloned_cuda_native_lowp_buffer().is_some());
        assert!(b.cloned_cuda_native_lowp_buffer().is_some());
        assert!(!a.has_host_f32_data());
        assert!(!b.has_host_f32_data());

        crate::ops::cuda::set_enabled(true);
        crate::autograd::set_strict_device_execution(true);
        let out = no_grad(|| matmul(&a, &b));
        crate::autograd::set_strict_device_execution(false);
        crate::ops::cuda::set_enabled(false);

        let reference = no_grad(|| matmul(&a.to_cpu(), &b.to_cpu()));
        assert!(out.is_cuda());
        assert_eq!(out.dtype(), DType::BF16);
        assert!(
            !out.has_host_f32_data(),
            "bf16 CUDA matmul should keep output resident until host data is requested"
        );
        {
            let inner = out.0.borrow();
            assert!(
                inner.cuda_f32_data.is_none(),
                "bf16 CUDA matmul should write native bf16 output, not a resident f32 buffer"
            );
            assert!(
                inner.cuda_bf16_data.is_some(),
                "bf16 CUDA matmul should keep resident bf16 storage"
            );
        }
        for (got, expect) in out.data_ref().iter().zip(reference.data_ref().iter()) {
            assert!((got - expect).abs() < 2e-2, "got {got}, expect {expect}");
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_matmul_preserves_f16_dtype_in_strict_mode() {
        if !crate::ops::cuda::is_available() {
            return;
        }

        let a = make_tensor(&[1, 4], vec![1.0, -2.0, 0.5, 3.0], DType::F16).to_cuda();
        let b = make_tensor(
            &[3, 4],
            vec![
                1.0, 0.0, -1.0, 2.0, 0.5, 1.5, -0.5, 0.25, -1.0, 2.0, 1.0, -0.5,
            ],
            DType::F16,
        )
        .to_cuda();
        assert!(a.cloned_cuda_native_lowp_buffer().is_some());
        assert!(b.cloned_cuda_native_lowp_buffer().is_some());
        assert!(!a.has_host_f32_data());
        assert!(!b.has_host_f32_data());

        crate::ops::cuda::set_enabled(true);
        crate::autograd::set_strict_device_execution(true);
        let out = no_grad(|| matmul(&a, &b));
        crate::autograd::set_strict_device_execution(false);
        crate::ops::cuda::set_enabled(false);

        let reference = no_grad(|| matmul(&a.to_cpu(), &b.to_cpu()));
        assert!(out.is_cuda());
        assert_eq!(out.dtype(), DType::F16);
        assert!(
            !out.has_host_f32_data(),
            "f16 CUDA matmul should keep output resident until host data is requested"
        );
        {
            let inner = out.0.borrow();
            assert!(
                inner.cuda_f32_data.is_none(),
                "f16 CUDA matmul should write native f16 output, not a resident f32 buffer"
            );
            assert!(
                inner.cuda_f16_data.is_some(),
                "f16 CUDA matmul should keep resident f16 storage"
            );
        }
        for (got, expect) in out.data_ref().iter().zip(reference.data_ref().iter()) {
            assert!((got - expect).abs() < 2e-2, "got {got}, expect {expect}");
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_bf16_matmul_native_view_materializes_cuda_values() {
        if !crate::ops::cuda::is_available() {
            return;
        }

        let a = make_tensor(&[1, 4], vec![1.0, -2.0, 0.5, 3.0], DType::BF16).to_cuda();
        let b = make_tensor(
            &[3, 4],
            vec![
                1.0, 0.0, -1.0, 2.0, 0.5, 1.5, -0.5, 0.25, -1.0, 2.0, 1.0, -0.5,
            ],
            DType::BF16,
        )
        .to_cuda();
        assert!(a.cloned_cuda_native_lowp_buffer().is_some());
        assert!(b.cloned_cuda_native_lowp_buffer().is_some());
        assert!(!a.has_host_f32_data());
        assert!(!b.has_host_f32_data());

        crate::ops::cuda::set_enabled(true);
        crate::autograd::set_strict_device_execution(true);
        let out = no_grad(|| matmul(&a, &b));
        crate::autograd::set_strict_device_execution(false);
        crate::ops::cuda::set_enabled(false);

        assert!(out.is_cuda());
        assert_eq!(out.dtype(), DType::BF16);
        assert!(!out.has_host_f32_data());

        let bf16_buffer = {
            let inner = out.0.borrow();
            assert!(inner.cuda_f32_data.is_none());
            inner
                .cuda_bf16_data
                .clone()
                .expect("bf16 matmul should keep resident bf16 output")
        };
        let bits = crate::ops::cuda::download_u16_storage(&bf16_buffer)
            .expect("download bf16 matmul bits");
        let vals = bits
            .iter()
            .map(|&bits| bf16::from_bits(bits).to_f32())
            .collect::<Vec<_>>();
        assert!(
            vals.iter().any(|v| v.abs() > 1e-6),
            "resident CUDA bf16 values should not be an all-zero placeholder"
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_matmul_preserves_i8_dtype_in_strict_mode() {
        if !crate::ops::cuda::is_available() {
            return;
        }

        let a = make_tensor(&[1, 4], vec![1.0, -2.0, 0.5, 3.0], DType::I8).to_cuda();
        let b = make_tensor(
            &[3, 4],
            vec![
                1.0, 0.0, -1.0, 2.0, 0.5, 1.5, -0.5, 0.25, -1.0, 2.0, 1.0, -0.5,
            ],
            DType::I8,
        )
        .to_cuda();
        assert!(a.cloned_cuda_native_lowp_buffer().is_some());
        assert!(b.cloned_cuda_native_lowp_buffer().is_some());
        assert!(!a.has_host_f32_data());
        assert!(!b.has_host_f32_data());

        crate::ops::cuda::set_enabled(true);
        crate::autograd::set_strict_device_execution(true);
        let out = no_grad(|| matmul(&a, &b));
        crate::autograd::set_strict_device_execution(false);
        crate::ops::cuda::set_enabled(false);

        let reference = no_grad(|| matmul(&a.to_cpu(), &b.to_cpu()));
        assert!(out.is_cuda());
        assert_eq!(out.dtype(), DType::I8);
        {
            let inner = out.0.borrow();
            assert!(
                !inner.has_f32_data,
                "i8 CUDA matmul output should not eagerly materialize host f32 data"
            );
            assert!(
                inner.cuda_f32_data.is_none(),
                "i8 CUDA matmul output should quantize away the temporary f32 buffer"
            );
            assert!(
                inner.cuda_i8_data.is_some(),
                "i8 CUDA matmul output should keep resident i8 storage"
            );
            assert!(
                inner.i8_scale.is_some(),
                "i8 CUDA matmul output should record dynamic quantization scale"
            );
        }
        for (got, expect) in out.data_ref().iter().zip(reference.data_ref().iter()) {
            assert!((got - expect).abs() < 1e-5, "got {got}, expect {expect}");
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_resident_i8_matmul_matches_quantized_reference() {
        if !crate::ops::cuda::is_available() {
            return;
        }

        let m = 9;
        let n = 19;
        let k = 35;
        let scale = 0.03125f32;
        let a = (0..m * k)
            .map(|i| ((((i * 17 + 3) % 255) as i32) - 127) as i8)
            .collect::<Vec<_>>();
        let b = (0..n * k)
            .map(|i| ((((i * 29 + 11) % 255) as i32) - 127) as i8)
            .collect::<Vec<_>>();
        let a_buf = crate::ops::cuda::upload_i8_storage(&a).expect("upload i8 matmul a");
        let b_buf = crate::ops::cuda::upload_i8_storage(&b).expect("upload i8 matmul b");

        let out = crate::ops::cuda::matmul_i8_buffer_no_host(&a_buf, scale, &b_buf, scale, m, n, k)
            .expect("CUDA resident i8 matmul");
        let got = crate::ops::cuda::download_f32(&out).expect("download resident i8 matmul");

        let mut expected = vec![0.0f32; m * n];
        for row in 0..m {
            for col in 0..n {
                let mut acc = 0i32;
                for kk in 0..k {
                    acc += (a[row * k + kk] as i32) * (b[col * k + kk] as i32);
                }
                expected[row * n + col] = (acc as f32) * scale * scale;
            }
        }

        for (actual, expected) in got.iter().zip(expected.iter()) {
            assert_eq!(actual, expected, "resident tiled i8 matmul kernel drifted");
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_resident_bf16_i8_matmul_matches_cpu_reference_without_host_f32() {
        if !crate::ops::cuda::is_available() {
            return;
        }

        let a = make_tensor(
            &[4, 7],
            vec![
                1.0, -2.0, 0.5, 3.0, -1.25, 0.75, 2.5, -0.5, 1.5, -3.0, 0.25, 2.0, -1.0, 0.125,
                3.5, -0.75, 1.25, -2.25, 0.5, 1.0, -1.5, 0.0, 2.75, -0.25, 1.75, -3.5, 0.625, 2.25,
            ],
            DType::BF16,
        )
        .to_cuda();
        let b = make_tensor(
            &[5, 7],
            vec![
                0.5, -1.0, 2.0, -0.25, 1.5, -2.5, 0.75, -1.5, 0.25, 1.25, -2.0, 0.5, 3.0, -0.75,
                2.5, -0.5, -1.25, 1.0, -3.0, 0.125, 1.75, -2.25, 1.5, 0.5, -0.625, 2.25, -1.0, 3.5,
                1.0, 2.0, -3.0, 0.75, -1.5, 0.25, -2.5,
            ],
            DType::I8,
        )
        .to_cuda();
        assert_eq!(a.dtype(), DType::BF16);
        assert_eq!(b.dtype(), DType::I8);
        assert!(a.cloned_cuda_native_lowp_buffer().is_some());
        assert!(b.cloned_cuda_native_lowp_buffer().is_some());
        assert!(!a.has_host_f32_data());
        assert!(!b.has_host_f32_data());

        crate::ops::cuda::set_enabled(true);
        crate::autograd::set_strict_device_execution(true);
        let out = no_grad(|| matmul(&a, &b));
        crate::autograd::set_strict_device_execution(false);
        crate::ops::cuda::set_enabled(false);

        assert!(out.is_cuda());
        assert_eq!(out.dtype(), DType::F32);
        assert!(!a.has_host_f32_data());
        assert!(
            !b.has_host_f32_data(),
            "mixed BF16xI8 CUDA matmul should not materialize I8 weights as host f32"
        );
        assert!(
            !out.has_host_f32_data(),
            "mixed BF16xI8 CUDA matmul should keep f32 output resident until requested"
        );
        {
            let inner = out.0.borrow();
            assert!(inner.cuda_f32_data.is_some());
            assert!(inner.cuda_bf16_data.is_none());
            assert!(inner.cuda_i8_data.is_none());
        }

        let reference = no_grad(|| matmul(&a.to_cpu(), &b.to_cpu()));
        for (got, expect) in out.data_ref().iter().zip(reference.data_ref().iter()) {
            assert!(
                (got - expect).abs() < 3e-2,
                "mixed BF16xI8 CUDA matmul drifted: got={got}, expect={expect}"
            );
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_resident_f16_i8_matmul_matches_cpu_reference_without_host_f32() {
        if !crate::ops::cuda::is_available() {
            return;
        }

        let a = make_tensor(&[4, 7], sample_f32(28), DType::F16).to_cuda();
        let b = make_tensor(&[5, 7], sample_f32(35), DType::I8).to_cuda();
        assert_eq!(a.dtype(), DType::F16);
        assert_eq!(b.dtype(), DType::I8);
        assert!(a.cloned_cuda_native_lowp_buffer().is_some());
        assert!(b.cloned_cuda_native_lowp_buffer().is_some());
        assert!(!a.has_host_f32_data());
        assert!(!b.has_host_f32_data());

        crate::ops::cuda::set_enabled(true);
        crate::autograd::set_strict_device_execution(true);
        let out = no_grad(|| matmul(&a, &b));
        crate::autograd::set_strict_device_execution(false);
        crate::ops::cuda::set_enabled(false);

        assert!(out.is_cuda());
        assert_eq!(out.dtype(), DType::F32);
        assert!(!a.has_host_f32_data());
        assert!(
            !b.has_host_f32_data(),
            "mixed F16xI8 CUDA matmul should not materialize I8 weights as host f32"
        );
        assert!(
            !out.has_host_f32_data(),
            "mixed F16xI8 CUDA matmul should keep f32 output resident until requested"
        );

        let reference = no_grad(|| matmul(&a.to_cpu(), &b.to_cpu()));
        for (got, expect) in out.data_ref().iter().zip(reference.data_ref().iter()) {
            assert!(
                (got - expect).abs() < 3e-2,
                "mixed F16xI8 CUDA matmul drifted: got={got}, expect={expect}"
            );
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_resident_f32_i8_matmul_matches_cpu_reference_without_host_f32_weight() {
        if !crate::ops::cuda::is_available() {
            return;
        }

        let a = make_tensor(&[3, 6], sample_f32(18), DType::F32).to_cuda();
        let b = make_tensor(
            &[4, 6],
            vec![
                0.5, -1.0, 2.0, -0.25, 1.5, -2.5, -1.5, 0.25, 1.25, -2.0, 0.5, 3.0, 2.5, -0.5,
                -1.25, 1.0, -3.0, 0.125, -2.25, 1.5, 0.5, -0.625, 2.25, -1.0,
            ],
            DType::I8,
        )
        .to_cuda();
        assert_eq!(a.dtype(), DType::F32);
        assert_eq!(b.dtype(), DType::I8);
        assert!(a.cloned_cuda_f32_buffer().is_some());
        assert!(b.cloned_cuda_native_lowp_buffer().is_some());
        assert!(!b.has_host_f32_data());

        crate::ops::cuda::set_enabled(true);
        crate::autograd::set_strict_device_execution(true);
        let out = no_grad(|| matmul(&a, &b));
        crate::autograd::set_strict_device_execution(false);
        crate::ops::cuda::set_enabled(false);

        assert!(out.is_cuda());
        assert_eq!(out.dtype(), DType::F32);
        assert!(
            !b.has_host_f32_data(),
            "F32xI8 CUDA matmul should not materialize I8 weights as host f32"
        );
        assert!(
            !out.has_host_f32_data(),
            "F32xI8 CUDA matmul should keep f32 output resident until requested"
        );
        let reference = no_grad(|| matmul(&a.to_cpu(), &b.to_cpu()));
        for (got, expect) in out.data_ref().iter().zip(reference.data_ref().iter()) {
            assert!(
                (got - expect).abs() < 3e-2,
                "mixed F32xI8 CUDA matmul drifted: got={got}, expect={expect}"
            );
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_resident_i8_floatlike_matmul_matches_cpu_reference_without_host_f32_input() {
        if !crate::ops::cuda::is_available() {
            return;
        }

        let k = 31;
        let n = 9;
        for (m, rhs_dtype) in [1usize, 5]
            .into_iter()
            .flat_map(|m| [DType::F16, DType::BF16, DType::F32].map(|rhs_dtype| (m, rhs_dtype)))
        {
            let a = make_tensor(&[m, k], sample_f32(m * k), DType::I8).to_cuda();
            let b = make_tensor(&[n, k], sample_f32(n * k), rhs_dtype).to_cuda();
            assert_eq!(a.dtype(), DType::I8);
            assert_eq!(b.dtype(), rhs_dtype);
            assert!(a.cloned_cuda_native_lowp_buffer().is_some());
            assert!(!a.has_host_f32_data());
            if rhs_dtype != DType::F32 {
                assert!(b.cloned_cuda_native_lowp_buffer().is_some());
                assert!(!b.has_host_f32_data());
            } else {
                assert!(b.cloned_cuda_f32_buffer().is_some());
            }

            crate::ops::cuda::set_enabled(true);
            crate::autograd::set_strict_device_execution(true);
            let out = no_grad(|| matmul(&a, &b));
            crate::autograd::set_strict_device_execution(false);
            crate::ops::cuda::set_enabled(false);

            assert!(out.is_cuda());
            assert_eq!(out.dtype(), DType::F32);
            assert!(
                !a.has_host_f32_data(),
                "I8x{rhs_dtype:?} CUDA matmul should not materialize I8 input as host f32"
            );
            assert!(
                !out.has_host_f32_data(),
                "I8x{rhs_dtype:?} CUDA matmul should keep f32 output resident until requested"
            );

            let reference = no_grad(|| matmul(&a.to_cpu(), &b.to_cpu()));
            for (got, expect) in out.data_ref().iter().zip(reference.data_ref().iter()) {
                assert!(
                    (got - expect).abs() < 5e-2,
                    "mixed I8x{rhs_dtype:?} CUDA matmul drifted for m={m}: got={got}, expect={expect}"
                );
            }
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_resident_single_row_float_i8_matvec_matches_cpu_reference_without_host_f32_weight() {
        if !crate::ops::cuda::is_available() {
            return;
        }

        let k = 257;
        let n = 7;
        for dtype in [DType::F16, DType::BF16, DType::F32] {
            let a = make_tensor(&[1, k], sample_f32(k), dtype).to_cuda();
            let b = make_tensor(&[n, k], sample_f32(n * k), DType::I8).to_cuda();
            assert_eq!(a.dtype(), dtype);
            assert_eq!(b.dtype(), DType::I8);
            assert!(b.cloned_cuda_native_lowp_buffer().is_some());
            assert!(!b.has_host_f32_data());

            crate::ops::cuda::set_enabled(true);
            crate::autograd::set_strict_device_execution(true);
            let out = no_grad(|| matmul(&a, &b));
            crate::autograd::set_strict_device_execution(false);
            crate::ops::cuda::set_enabled(false);

            assert!(out.is_cuda());
            assert_eq!(out.dtype(), DType::F32);
            assert!(
                !b.has_host_f32_data(),
                "{dtype:?}xI8 CUDA matvec should not materialize I8 weights as host f32"
            );
            let reference = no_grad(|| matmul(&a.to_cpu(), &b.to_cpu()));
            for (got, expect) in out.data_ref().iter().zip(reference.data_ref().iter()) {
                assert!(
                    (got - expect).abs() < 5e-2,
                    "single-row {dtype:?}xI8 CUDA matvec drifted: got={got}, expect={expect}"
                );
            }
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_resident_mixed_i8_argmax_matches_cpu_logits_argmax_without_host_f32_weight() {
        if !crate::ops::cuda::is_available() {
            return;
        }

        let k = 19;
        let vocab = 13;
        for (batch, dtype) in [DType::F16, DType::BF16, DType::F32, DType::I8]
            .into_iter()
            .flat_map(|dtype| [(1usize, dtype), (2usize, dtype)])
        {
            let a = make_tensor(&[batch, k], sample_f32(batch * k), dtype).to_cuda();
            let b = make_tensor(&[vocab, k], sample_f32(vocab * k), DType::I8).to_cuda();
            let (weight_dtype, weight_buf, weight_scale) = b
                .cloned_cuda_native_lowp_buffer()
                .expect("resident i8 weight");
            assert_eq!(weight_dtype, DType::I8);
            assert!(!b.has_host_f32_data());

            let got = if dtype == DType::BF16 {
                let (input_dtype, input_buf, _) = a
                    .cloned_cuda_native_lowp_buffer()
                    .expect("resident bf16 input");
                assert_eq!(input_dtype, DType::BF16);
                crate::ops::cuda::matvec_argmax_bf16_i8(
                    &input_buf,
                    &weight_buf,
                    weight_scale.expect("i8 weight scale"),
                    batch,
                    vocab,
                    k,
                )
            } else if dtype == DType::F16 {
                let (input_dtype, input_buf, _) = a
                    .cloned_cuda_native_lowp_buffer()
                    .expect("resident f16 input");
                assert_eq!(input_dtype, DType::F16);
                crate::ops::cuda::matvec_argmax_f16_i8(
                    &input_buf,
                    &weight_buf,
                    weight_scale.expect("i8 weight scale"),
                    batch,
                    vocab,
                    k,
                )
            } else if dtype == DType::I8 {
                let (input_dtype, input_buf, input_scale) = a
                    .cloned_cuda_native_lowp_buffer()
                    .expect("resident i8 input");
                assert_eq!(input_dtype, DType::I8);
                crate::ops::cuda::matvec_argmax_i8_i8(
                    &input_buf,
                    input_scale.expect("i8 input scale"),
                    &weight_buf,
                    weight_scale.expect("i8 weight scale"),
                    batch,
                    vocab,
                    k,
                )
            } else {
                let input_buf = a.cloned_cuda_f32_buffer().expect("resident f32 input");
                crate::ops::cuda::matvec_argmax_f32_i8(
                    &input_buf,
                    &weight_buf,
                    weight_scale.expect("i8 weight scale"),
                    batch,
                    vocab,
                    k,
                )
            }
            .expect("CUDA mixed argmax");

            let reference = no_grad(|| matmul(&a.to_cpu(), &b.to_cpu()));
            let logits = reference.data_ref();
            let expected = (0..batch)
                .map(|row| {
                    let row_logits = logits.slice(ndarray::s![row, ..]);
                    row_logits
                        .iter()
                        .enumerate()
                        .max_by(|(_, lhs), (_, rhs)| lhs.total_cmp(rhs))
                        .map(|(idx, _)| idx)
                        .expect("non-empty vocab")
                })
                .collect::<Vec<_>>();

            assert_eq!(got, expected, "{dtype:?}xI8 CUDA argmax drifted");
            assert!(
                !b.has_host_f32_data(),
                "{dtype:?}xI8 CUDA argmax should not materialize I8 weights as host f32"
            );
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_i8_weight_argmax_rejects_invalid_scales() {
        if !crate::ops::cuda::is_available() {
            return;
        }

        let batch = 2;
        let vocab = 3;
        let hidden = 5;
        let input_u16 = crate::ops::cuda::upload_u16_storage(&vec![0; batch * hidden])
            .expect("upload u16 input");
        let input_f32 =
            crate::ops::cuda::upload_f32(&vec![0.0; batch * hidden]).expect("upload f32 input");
        let input_i8 =
            crate::ops::cuda::upload_i8_storage(&vec![0; batch * hidden]).expect("upload i8 input");
        let weight_i8 = crate::ops::cuda::upload_i8_storage(&vec![0; vocab * hidden])
            .expect("upload i8 weight");

        assert!(
            crate::ops::cuda::matvec_argmax_bf16_i8(
                &input_u16,
                &weight_i8,
                f32::NAN,
                batch,
                vocab,
                hidden,
            )
            .is_err()
        );
        assert!(
            crate::ops::cuda::matvec_argmax_f16_i8(
                &input_u16, &weight_i8, 0.0, batch, vocab, hidden,
            )
            .is_err()
        );
        assert!(
            crate::ops::cuda::matvec_argmax_f32_i8(
                &input_f32, &weight_i8, -1.0, batch, vocab, hidden,
            )
            .is_err()
        );
        assert!(
            crate::ops::cuda::matvec_argmax_i8_i8(
                &input_i8,
                f32::INFINITY,
                &weight_i8,
                0.05,
                batch,
                vocab,
                hidden,
            )
            .is_err()
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_resident_i8_matmul_typed_output_matches_dynamic_quantized_reference() {
        if !crate::ops::cuda::is_available() {
            return;
        }

        let m = 9;
        let n = 19;
        let k = 35;
        let a_scale = 0.03125f32;
        let b_scale = 0.046875f32;
        let a = (0..m * k)
            .map(|i| ((((i * 17 + 3) % 255) as i32) - 127) as i8)
            .collect::<Vec<_>>();
        let b = (0..n * k)
            .map(|i| ((((i * 29 + 11) % 255) as i32) - 127) as i8)
            .collect::<Vec<_>>();
        let a_buf = crate::ops::cuda::upload_i8_storage(&a).expect("upload typed i8 matmul a");
        let b_buf = crate::ops::cuda::upload_i8_storage(&b).expect("upload typed i8 matmul b");

        let (out, got_scale) = crate::ops::cuda::matmul_i8_typed_output_buffer_no_host(
            &a_buf, a_scale, &b_buf, b_scale, m, n, k,
        )
        .expect("CUDA resident typed-output i8 matmul");
        let got = crate::ops::cuda::download_i8_storage(&out).expect("download typed i8 matmul");

        let mut values = vec![0.0f32; m * n];
        let mut max_abs = 0.0f32;
        for row in 0..m {
            for col in 0..n {
                let mut acc = 0i32;
                for kk in 0..k {
                    acc += (a[row * k + kk] as i32) * (b[col * k + kk] as i32);
                }
                let value = (acc as f32) * a_scale * b_scale;
                values[row * n + col] = value;
                max_abs = max_abs.max(value.abs());
            }
        }
        let expected_scale = if max_abs > 0.0 {
            (max_abs / 127.0).max(f32::MIN_POSITIVE)
        } else {
            1.0
        };
        assert!(
            (got_scale - expected_scale).abs() <= 1e-6,
            "typed i8 matmul scale drifted: got={got_scale}, expected={expected_scale}"
        );

        let expected = values
            .iter()
            .map(|value| {
                let q = (value / expected_scale).round().clamp(-127.0, 127.0);
                q as i8
            })
            .collect::<Vec<_>>();
        assert_eq!(got, expected, "typed i8 matmul quantized output drifted");
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_resident_i8_matmul_typed_output_zero_absmax_uses_unit_scale() {
        if !crate::ops::cuda::is_available() {
            return;
        }

        let m = 5;
        let n = 7;
        let k = 11;
        let a_buf =
            crate::ops::cuda::upload_i8_storage(&vec![0; m * k]).expect("upload zero i8 matmul a");
        let b_buf =
            crate::ops::cuda::upload_i8_storage(&vec![0; n * k]).expect("upload zero i8 matmul b");

        let (out, scale) = crate::ops::cuda::matmul_i8_typed_output_buffer_no_host(
            &a_buf, 0.03125, &b_buf, 0.046875, m, n, k,
        )
        .expect("CUDA zero typed-output i8 matmul");
        let got =
            crate::ops::cuda::download_i8_storage(&out).expect("download zero typed i8 matmul");

        assert_eq!(scale, 1.0);
        assert_eq!(got, vec![0; m * n]);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_resident_i8_batch_matmul_typed_output_matches_dynamic_quantized_reference() {
        if !crate::ops::cuda::is_available() {
            return;
        }

        let batch_count = 2;
        let m = 3;
        let n = 4;
        let k = 5;
        let lhs_scale = 0.03125f32;
        let rhs_scale = 0.046875f32;
        let lhs = (0..batch_count * m * k)
            .map(|i| ((((i * 17 + 3) % 255) as i32) - 127) as i8)
            .collect::<Vec<_>>();
        let rhs = (0..batch_count * k * n)
            .map(|i| ((((i * 29 + 11) % 255) as i32) - 127) as i8)
            .collect::<Vec<_>>();
        let lhs_buf = crate::ops::cuda::upload_i8_storage(&lhs).expect("upload typed i8 batch lhs");
        let rhs_buf = crate::ops::cuda::upload_i8_storage(&rhs).expect("upload typed i8 batch rhs");

        let (out, got_scale) = crate::ops::cuda::batch_matmul_i8_typed_output_buffer_no_host(
            &lhs_buf,
            lhs_scale,
            &rhs_buf,
            rhs_scale,
            batch_count,
            m,
            n,
            k,
        )
        .expect("CUDA resident typed-output i8 batch matmul");
        let got =
            crate::ops::cuda::download_i8_storage(&out).expect("download typed i8 batch matmul");

        let mut values = vec![0.0f32; batch_count * m * n];
        let mut max_abs = 0.0f32;
        for batch in 0..batch_count {
            for row in 0..m {
                for col in 0..n {
                    let mut acc = 0i32;
                    for kk in 0..k {
                        acc += (lhs[batch * m * k + row * k + kk] as i32)
                            * (rhs[batch * k * n + kk * n + col] as i32);
                    }
                    let idx = batch * m * n + row * n + col;
                    let value = (acc as f32) * lhs_scale * rhs_scale;
                    values[idx] = value;
                    max_abs = max_abs.max(value.abs());
                }
            }
        }
        let expected_scale = if max_abs > 0.0 {
            (max_abs / 127.0).max(f32::MIN_POSITIVE)
        } else {
            1.0
        };
        assert!(
            (got_scale - expected_scale).abs() <= 1e-6,
            "typed i8 batch matmul scale drifted: got={got_scale}, expected={expected_scale}"
        );

        let expected = values
            .iter()
            .map(|value| {
                let q = (value / expected_scale).round().clamp(-127.0, 127.0);
                q as i8
            })
            .collect::<Vec<_>>();
        assert_eq!(
            got, expected,
            "typed i8 batch matmul quantized output drifted"
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_resident_i8_batch_matmul_f32_output_handles_tiled_tail_dimensions() {
        if !crate::ops::cuda::is_available() {
            return;
        }

        let batch_count = 2;
        let m = 5;
        let n = 7;
        let k = 35;
        let lhs_scale = 0.03125f32;
        let rhs_scale = 0.046875f32;
        let lhs = (0..batch_count * m * k)
            .map(|i| ((((i * 17 + 3) % 255) as i32) - 127) as i8)
            .collect::<Vec<_>>();
        let rhs = (0..batch_count * k * n)
            .map(|i| ((((i * 29 + 11) % 255) as i32) - 127) as i8)
            .collect::<Vec<_>>();
        let lhs_buf = crate::ops::cuda::upload_i8_storage(&lhs).expect("upload tiled i8 batch lhs");
        let rhs_buf = crate::ops::cuda::upload_i8_storage(&rhs).expect("upload tiled i8 batch rhs");

        let out = crate::ops::cuda::batch_matmul_i8_buffer_no_host(
            &lhs_buf,
            lhs_scale,
            &rhs_buf,
            rhs_scale,
            batch_count,
            m,
            n,
            k,
        )
        .expect("CUDA resident f32-output i8 batch matmul");
        let got = crate::ops::cuda::download_f32(&out).expect("download tiled i8 batch matmul");

        for batch in 0..batch_count {
            for row in 0..m {
                for col in 0..n {
                    let mut acc = 0i32;
                    for kk in 0..k {
                        acc += (lhs[batch * m * k + row * k + kk] as i32)
                            * (rhs[batch * k * n + kk * n + col] as i32);
                    }
                    let idx = batch * m * n + row * n + col;
                    let expected = (acc as f32) * lhs_scale * rhs_scale;
                    assert_eq!(
                        got[idx], expected,
                        "tiled i8 batch_matmul drifted at batch={batch}, row={row}, col={col}"
                    );
                }
            }
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_batch_matmul_preserves_bf16_dtype_in_strict_mode() {
        if !crate::ops::cuda::is_available() {
            return;
        }

        let lhs = make_tensor(
            &[1, 1, 2, 3],
            vec![1.0, -2.0, 0.5, 3.0, -1.0, 2.0],
            DType::BF16,
        )
        .to_cuda();
        let rhs = make_tensor(
            &[1, 1, 3, 2],
            vec![0.5, 1.0, -1.5, 2.0, 0.25, -0.75],
            DType::BF16,
        )
        .to_cuda();

        crate::ops::cuda::set_enabled(true);
        crate::autograd::set_strict_device_execution(true);
        let out = no_grad(|| batch_matmul(&lhs, &rhs));
        crate::autograd::set_strict_device_execution(false);
        crate::ops::cuda::set_enabled(false);

        let reference = no_grad(|| batch_matmul(&lhs.to_cpu(), &rhs.to_cpu()));
        assert!(out.is_cuda());
        assert_eq!(out.dtype(), DType::BF16);
        assert!(
            !out.has_host_f32_data(),
            "bf16 CUDA batch_matmul should keep output resident until host data is requested"
        );
        {
            let inner = out.0.borrow();
            assert!(
                inner.cuda_f32_data.is_none(),
                "bf16 CUDA batch_matmul should write native bf16 output, not a resident f32 buffer"
            );
            assert!(
                inner.cuda_bf16_data.is_some(),
                "bf16 CUDA batch_matmul should keep resident bf16 storage"
            );
        }
        for (got, expect) in out.data_ref().iter().zip(reference.data_ref().iter()) {
            assert!((got - expect).abs() < 2e-2, "got {got}, expect {expect}");
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_batch_matmul_mixed_i8_matches_cpu_reference_without_host_f32_rhs() {
        if !crate::ops::cuda::is_available() {
            return;
        }

        let shape_lhs = [2, 3, 5, 131];
        let shape_rhs = [2, 3, 131, 7];
        let lhs_len = shape_lhs.iter().product::<usize>();
        let rhs_len = shape_rhs.iter().product::<usize>();
        for dtype in [DType::F16, DType::BF16, DType::F32] {
            let lhs = make_tensor(&shape_lhs, sample_f32(lhs_len), dtype).to_cuda();
            let rhs = make_tensor(&shape_rhs, sample_f32(rhs_len), DType::I8).to_cuda();
            assert_eq!(lhs.dtype(), dtype);
            assert_eq!(rhs.dtype(), DType::I8);
            assert!(rhs.cloned_cuda_native_lowp_buffer().is_some());
            assert!(!rhs.has_host_f32_data());

            crate::ops::cuda::set_enabled(true);
            crate::autograd::set_strict_device_execution(true);
            let out = no_grad(|| batch_matmul(&lhs, &rhs));
            crate::autograd::set_strict_device_execution(false);
            crate::ops::cuda::set_enabled(false);

            assert!(out.is_cuda());
            assert_eq!(out.dtype(), DType::F32);
            assert!(
                !rhs.has_host_f32_data(),
                "{dtype:?}xI8 CUDA batch_matmul should not materialize I8 rhs as host f32"
            );
            assert!(
                !out.has_host_f32_data(),
                "{dtype:?}xI8 CUDA batch_matmul should keep f32 output resident until requested"
            );

            let reference = no_grad(|| batch_matmul(&lhs.to_cpu(), &rhs.to_cpu()));
            for (got, expect) in out.data_ref().iter().zip(reference.data_ref().iter()) {
                assert!(
                    (got - expect).abs() < 5e-2,
                    "mixed {dtype:?}xI8 CUDA batch_matmul drifted: got={got}, expect={expect}"
                );
            }
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_batch_matmul_i8_floatlike_matches_cpu_reference_without_host_f32_lhs() {
        if !crate::ops::cuda::is_available() {
            return;
        }

        let shape_lhs = [2, 3, 5, 131];
        let shape_rhs = [2, 3, 131, 7];
        let lhs_len = shape_lhs.iter().product::<usize>();
        let rhs_len = shape_rhs.iter().product::<usize>();
        for dtype in [DType::F16, DType::BF16, DType::F32] {
            let lhs = make_tensor(&shape_lhs, sample_f32(lhs_len), DType::I8).to_cuda();
            let rhs = make_tensor(&shape_rhs, sample_f32(rhs_len), dtype).to_cuda();
            assert_eq!(lhs.dtype(), DType::I8);
            assert_eq!(rhs.dtype(), dtype);
            assert!(lhs.cloned_cuda_native_lowp_buffer().is_some());
            assert!(!lhs.has_host_f32_data());
            if dtype != DType::F32 {
                assert!(rhs.cloned_cuda_native_lowp_buffer().is_some());
                assert!(!rhs.has_host_f32_data());
            } else {
                assert!(rhs.cloned_cuda_f32_buffer().is_some());
            }

            crate::ops::cuda::set_enabled(true);
            crate::autograd::set_strict_device_execution(true);
            let out = no_grad(|| batch_matmul(&lhs, &rhs));
            crate::autograd::set_strict_device_execution(false);
            crate::ops::cuda::set_enabled(false);

            assert!(out.is_cuda());
            assert_eq!(out.dtype(), DType::F32);
            assert!(
                !lhs.has_host_f32_data(),
                "I8x{dtype:?} CUDA batch_matmul should not materialize I8 lhs as host f32"
            );
            assert!(
                !out.has_host_f32_data(),
                "I8x{dtype:?} CUDA batch_matmul should keep f32 output resident until requested"
            );

            let reference = no_grad(|| batch_matmul(&lhs.to_cpu(), &rhs.to_cpu()));
            for (got, expect) in out.data_ref().iter().zip(reference.data_ref().iter()) {
                assert!(
                    (got - expect).abs() < 5e-2,
                    "mixed I8x{dtype:?} CUDA batch_matmul drifted: got={got}, expect={expect}"
                );
            }
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_batch_matmul_lowp_uses_resident_forward_buffer_in_strict_mode() {
        if !crate::ops::cuda::is_available() {
            return;
        }

        crate::ops::cuda::set_enabled(true);
        crate::autograd::set_strict_device_execution(true);

        for dtype in [DType::F16, DType::BF16, DType::I8] {
            let lhs =
                make_tensor(&[1, 1, 2, 3], vec![1.0, -2.0, 0.5, 3.0, -1.0, 2.0], dtype).to_cuda();
            let rhs =
                make_tensor(&[1, 1, 3, 2], vec![0.5, 1.0, -1.5, 2.0, 0.25, -0.75], dtype).to_cuda();
            assert!(lhs.cloned_cuda_native_lowp_buffer().is_some());
            assert!(rhs.cloned_cuda_native_lowp_buffer().is_some());
            assert!(!lhs.has_host_f32_data());
            assert!(!rhs.has_host_f32_data());

            let out = no_grad(|| batch_matmul(&lhs, &rhs));

            assert!(out.is_cuda());
            assert_eq!(out.dtype(), dtype);
            {
                let inner = out.0.borrow();
                match dtype {
                    DType::F16 => {
                        assert!(inner.cuda_f32_data.is_none());
                        assert!(inner.cuda_f16_data.is_some());
                    }
                    DType::BF16 => {
                        assert!(inner.cuda_f32_data.is_none());
                        assert!(inner.cuda_bf16_data.is_some());
                    }
                    DType::I8 => {
                        assert!(inner.cuda_f32_data.is_none());
                        assert!(inner.cuda_i8_data.is_some());
                    }
                    DType::F32 => unreachable!("test only covers low precision dtypes"),
                }
            }
            assert!(!lhs.has_host_f32_data());
            assert!(!rhs.has_host_f32_data());
            let reference = no_grad(|| batch_matmul(&lhs.to_cpu(), &rhs.to_cpu()));
            for (got, expect) in out.data_ref().iter().zip(reference.data_ref().iter()) {
                assert!(
                    (got - expect).abs() <= 0.08,
                    "{dtype:?} batch_matmul got {got}, expect {expect}"
                );
            }
        }

        crate::autograd::set_strict_device_execution(false);
        crate::ops::cuda::set_enabled(false);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_batch_matmul_lowp_backward_outputs_f32_grads() {
        if !crate::ops::cuda::is_available() {
            return;
        }

        crate::ops::cuda::set_enabled(true);
        crate::autograd::set_strict_device_execution(true);

        for dtype in [DType::F16, DType::BF16, DType::I8] {
            let lhs = make_grad_tensor(&[1, 1, 2, 3], vec![1.0, -2.0, 0.5, 3.0, -1.0, 2.0]);
            lhs.cast_inplace(dtype);
            let lhs = lhs.to_cuda();
            let rhs = make_grad_tensor(&[1, 1, 3, 2], vec![0.5, 1.0, -1.5, 2.0, 0.25, -0.75]);
            rhs.cast_inplace(dtype);
            let rhs = rhs.to_cuda();

            let loss = sum(&batch_matmul(&lhs, &rhs));
            loss.backward();

            assert!(lhs.cloned_cuda_f32_grad().is_some());
            assert!(rhs.cloned_cuda_f32_grad().is_some());
            assert!(!lhs.has_host_f32_data());
            assert!(!rhs.has_host_f32_data());
            let lhs_grad = lhs.grad().expect("CUDA batch_matmul lhs grad");
            let rhs_grad = rhs.grad().expect("CUDA batch_matmul rhs grad");
            for (got, expected) in lhs_grad.iter().zip([1.5f32, 0.5, -0.5, 1.5, 0.5, -0.5]) {
                assert!(
                    (got - expected).abs() <= 0.08,
                    "{dtype:?} lhs grad got {got}, expected {expected}"
                );
            }
            for (got, expected) in rhs_grad.iter().zip([4.0f32, 4.0, -3.0, -3.0, 2.5, 2.5]) {
                assert!(
                    (got - expected).abs() <= 0.08,
                    "{dtype:?} rhs grad got {got}, expected {expected}"
                );
            }
        }

        crate::autograd::set_strict_device_execution(false);
        crate::ops::cuda::set_enabled(false);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_batch_matmul_mixed_i8_backward_reads_typed_data_and_writes_f32_grads() {
        if !crate::ops::cuda::is_available() {
            return;
        }

        let lhs_data = vec![1.0, -2.0, 0.5, 3.0, -1.0, 2.0];
        let rhs_data = vec![0.5, 1.0, -1.5, 2.0, 0.25, -0.75];

        crate::ops::cuda::set_enabled(true);
        crate::autograd::set_strict_device_execution(true);

        for lhs_dtype in [DType::F16, DType::F32, DType::BF16] {
            let lhs_cpu = make_grad_tensor(&[1, 1, 2, 3], lhs_data.clone());
            lhs_cpu.cast_inplace(lhs_dtype);
            let rhs_cpu = make_grad_tensor(&[1, 1, 3, 2], rhs_data.clone());
            rhs_cpu.cast_inplace(DType::I8);

            let lhs_cuda = make_grad_tensor(&[1, 1, 2, 3], lhs_data.clone());
            lhs_cuda.cast_inplace(lhs_dtype);
            let lhs_cuda = lhs_cuda.to_cuda();
            let rhs_cuda = make_grad_tensor(&[1, 1, 3, 2], rhs_data.clone());
            rhs_cuda.cast_inplace(DType::I8);
            let rhs_cuda = rhs_cuda.to_cuda();

            let loss_cuda = sum(&batch_matmul(&lhs_cuda, &rhs_cuda));
            loss_cuda.backward();

            let loss_cpu = sum(&batch_matmul(&lhs_cpu, &rhs_cpu));
            loss_cpu.backward();

            assert!(lhs_cuda.cloned_cuda_f32_grad().is_some());
            assert!(rhs_cuda.cloned_cuda_f32_grad().is_some());
            assert!(!rhs_cuda.has_host_f32_data());
            assert!(!lhs_cuda.has_host_grad());
            assert!(!rhs_cuda.has_host_grad());

            let lhs_grad = lhs_cuda.grad().expect("CUDA mixed batch_matmul lhs grad");
            let rhs_grad = rhs_cuda.grad().expect("CUDA mixed batch_matmul rhs grad");
            let lhs_ref = lhs_cpu.grad().expect("CPU mixed batch_matmul lhs grad");
            let rhs_ref = rhs_cpu.grad().expect("CPU mixed batch_matmul rhs grad");
            for (got, expect) in lhs_grad.iter().zip(lhs_ref.iter()) {
                assert!(
                    (got - expect).abs() <= 0.08,
                    "{lhs_dtype:?}xI8 lhs grad got {got}, expect {expect}"
                );
            }
            for (got, expect) in rhs_grad.iter().zip(rhs_ref.iter()) {
                assert!(
                    (got - expect).abs() <= 0.08,
                    "{lhs_dtype:?}xI8 rhs grad got {got}, expect {expect}"
                );
            }
        }

        crate::autograd::set_strict_device_execution(false);
        crate::ops::cuda::set_enabled(false);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_batch_matmul_i8_floatlike_backward_reads_typed_data_and_writes_f32_grads() {
        if !crate::ops::cuda::is_available() {
            return;
        }

        let lhs_data = vec![1.0, -2.0, 0.5, 3.0, -1.0, 2.0];
        let rhs_data = vec![0.5, 1.0, -1.5, 2.0, 0.25, -0.75];

        crate::ops::cuda::set_enabled(true);
        crate::autograd::set_strict_device_execution(true);

        for rhs_dtype in [DType::F16, DType::F32, DType::BF16] {
            let lhs_cpu = make_grad_tensor(&[1, 1, 2, 3], lhs_data.clone());
            lhs_cpu.cast_inplace(DType::I8);
            let rhs_cpu = make_grad_tensor(&[1, 1, 3, 2], rhs_data.clone());
            rhs_cpu.cast_inplace(rhs_dtype);

            let lhs_cuda = make_grad_tensor(&[1, 1, 2, 3], lhs_data.clone());
            lhs_cuda.cast_inplace(DType::I8);
            let lhs_cuda = lhs_cuda.to_cuda();
            let rhs_cuda = make_grad_tensor(&[1, 1, 3, 2], rhs_data.clone());
            rhs_cuda.cast_inplace(rhs_dtype);
            let rhs_cuda = rhs_cuda.to_cuda();

            let loss_cuda = sum(&batch_matmul(&lhs_cuda, &rhs_cuda));
            loss_cuda.backward();

            let loss_cpu = sum(&batch_matmul(&lhs_cpu, &rhs_cpu));
            loss_cpu.backward();

            assert!(lhs_cuda.cloned_cuda_f32_grad().is_some());
            assert!(rhs_cuda.cloned_cuda_f32_grad().is_some());
            assert!(!lhs_cuda.has_host_f32_data());
            assert!(!lhs_cuda.has_host_grad());
            assert!(!rhs_cuda.has_host_grad());

            let lhs_grad = lhs_cuda
                .grad()
                .expect("CUDA I8xfloatlike batch_matmul lhs grad");
            let rhs_grad = rhs_cuda
                .grad()
                .expect("CUDA I8xfloatlike batch_matmul rhs grad");
            let lhs_ref = lhs_cpu
                .grad()
                .expect("CPU I8xfloatlike batch_matmul lhs grad");
            let rhs_ref = rhs_cpu
                .grad()
                .expect("CPU I8xfloatlike batch_matmul rhs grad");
            for (got, expect) in lhs_grad.iter().zip(lhs_ref.iter()) {
                assert!(
                    (got - expect).abs() <= 0.08,
                    "I8x{rhs_dtype:?} batch_matmul lhs grad got {got}, expect {expect}"
                );
            }
            for (got, expect) in rhs_grad.iter().zip(rhs_ref.iter()) {
                assert!(
                    (got - expect).abs() <= 0.08,
                    "I8x{rhs_dtype:?} batch_matmul rhs grad got {got}, expect {expect}"
                );
            }
        }

        crate::autograd::set_strict_device_execution(false);
        crate::ops::cuda::set_enabled(false);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_batch_matmul_lowp_backward_handles_asymmetric_gradient_sizes() {
        if !crate::ops::cuda::is_available() {
            return;
        }

        crate::ops::cuda::set_enabled(true);
        crate::autograd::set_strict_device_execution(true);

        for (m, n, lhs_dtype, rhs_dtype) in [
            (2, 5, DType::BF16, DType::BF16),
            (5, 2, DType::F16, DType::I8),
            (2, 5, DType::I8, DType::F16),
            (5, 2, DType::I8, DType::I8),
        ] {
            let k = 3;
            let lhs_data = sample_f32(m * k);
            let rhs_data = sample_f32(k * n);

            let lhs_cpu = make_grad_tensor(&[1, 1, m, k], lhs_data.clone());
            lhs_cpu.cast_inplace(lhs_dtype);
            let rhs_cpu = make_grad_tensor(&[1, 1, k, n], rhs_data.clone());
            rhs_cpu.cast_inplace(rhs_dtype);

            let lhs_cuda = make_grad_tensor(&[1, 1, m, k], lhs_data);
            lhs_cuda.cast_inplace(lhs_dtype);
            let lhs_cuda = lhs_cuda.to_cuda();
            let rhs_cuda = make_grad_tensor(&[1, 1, k, n], rhs_data);
            rhs_cuda.cast_inplace(rhs_dtype);
            let rhs_cuda = rhs_cuda.to_cuda();

            sum(&batch_matmul(&lhs_cuda, &rhs_cuda)).backward();
            sum(&batch_matmul(&lhs_cpu, &rhs_cpu)).backward();

            assert!(lhs_cuda.cloned_cuda_f32_grad().is_some());
            assert!(rhs_cuda.cloned_cuda_f32_grad().is_some());
            assert!(!lhs_cuda.has_host_grad());
            assert!(!rhs_cuda.has_host_grad());
            let lhs_grad = lhs_cuda.grad().expect("CUDA asymmetric batch lhs grad");
            let rhs_grad = rhs_cuda.grad().expect("CUDA asymmetric batch rhs grad");
            let lhs_ref = lhs_cpu.grad().expect("CPU asymmetric batch lhs grad");
            let rhs_ref = rhs_cpu.grad().expect("CPU asymmetric batch rhs grad");
            assert_eq!(lhs_grad.len(), m * k);
            assert_eq!(rhs_grad.len(), k * n);
            for (got, expect) in lhs_grad.iter().zip(lhs_ref.iter()) {
                assert!(
                    (got - expect).abs() <= 0.08,
                    "{lhs_dtype:?}x{rhs_dtype:?} asymmetric lhs grad got {got}, expect {expect}"
                );
            }
            for (got, expect) in rhs_grad.iter().zip(rhs_ref.iter()) {
                assert!(
                    (got - expect).abs() <= 0.08,
                    "{lhs_dtype:?}x{rhs_dtype:?} asymmetric rhs grad got {got}, expect {expect}"
                );
            }
        }

        crate::autograd::set_strict_device_execution(false);
        crate::ops::cuda::set_enabled(false);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_matmul_mixed_i8_backward_reads_typed_data_and_writes_f32_grads() {
        if !crate::ops::cuda::is_available() {
            return;
        }

        let lhs_data = vec![1.0, -2.0, 0.5, 3.0, -1.0, 2.0];
        let rhs_data = vec![0.5, 1.0, -1.5, 2.0, 0.25, -0.75, 1.25, -0.5, 0.75];

        crate::ops::cuda::set_enabled(true);
        crate::autograd::set_strict_device_execution(true);

        for lhs_dtype in [DType::F16, DType::F32, DType::BF16] {
            let lhs_cpu = make_grad_tensor(&[2, 3], lhs_data.clone());
            lhs_cpu.cast_inplace(lhs_dtype);
            let rhs_cpu = make_grad_tensor(&[3, 3], rhs_data.clone());
            rhs_cpu.cast_inplace(DType::I8);

            let lhs_cuda = make_grad_tensor(&[2, 3], lhs_data.clone());
            lhs_cuda.cast_inplace(lhs_dtype);
            let lhs_cuda = lhs_cuda.to_cuda();
            let rhs_cuda = make_grad_tensor(&[3, 3], rhs_data.clone());
            rhs_cuda.cast_inplace(DType::I8);
            let rhs_cuda = rhs_cuda.to_cuda();

            let loss_cuda = sum(&matmul(&lhs_cuda, &rhs_cuda));
            loss_cuda.backward();

            let loss_cpu = sum(&matmul(&lhs_cpu, &rhs_cpu));
            loss_cpu.backward();

            assert!(lhs_cuda.cloned_cuda_f32_grad().is_some());
            assert!(rhs_cuda.cloned_cuda_f32_grad().is_some());
            assert!(!rhs_cuda.has_host_f32_data());
            assert!(!lhs_cuda.has_host_grad());
            assert!(!rhs_cuda.has_host_grad());

            let lhs_grad = lhs_cuda.grad().expect("CUDA mixed matmul lhs grad");
            let rhs_grad = rhs_cuda.grad().expect("CUDA mixed matmul rhs grad");
            let lhs_ref = lhs_cpu.grad().expect("CPU mixed matmul lhs grad");
            let rhs_ref = rhs_cpu.grad().expect("CPU mixed matmul rhs grad");
            for (got, expect) in lhs_grad.iter().zip(lhs_ref.iter()) {
                assert!(
                    (got - expect).abs() <= 0.08,
                    "{lhs_dtype:?}xI8 matmul lhs grad got {got}, expect {expect}"
                );
            }
            for (got, expect) in rhs_grad.iter().zip(rhs_ref.iter()) {
                assert!(
                    (got - expect).abs() <= 0.08,
                    "{lhs_dtype:?}xI8 matmul rhs grad got {got}, expect {expect}"
                );
            }
        }

        crate::autograd::set_strict_device_execution(false);
        crate::ops::cuda::set_enabled(false);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_matmul_i8_floatlike_backward_reads_typed_data_and_writes_f32_grads() {
        if !crate::ops::cuda::is_available() {
            return;
        }

        let lhs_data = vec![1.0, -2.0, 0.5, 3.0, -1.0, 2.0];
        let rhs_data = vec![0.5, 1.0, -1.5, 2.0, 0.25, -0.75, 1.25, -0.5, 0.75];

        crate::ops::cuda::set_enabled(true);
        crate::autograd::set_strict_device_execution(true);

        for rhs_dtype in [DType::F16, DType::F32, DType::BF16] {
            let lhs_cpu = make_grad_tensor(&[2, 3], lhs_data.clone());
            lhs_cpu.cast_inplace(DType::I8);
            let rhs_cpu = make_grad_tensor(&[3, 3], rhs_data.clone());
            rhs_cpu.cast_inplace(rhs_dtype);

            let lhs_cuda = make_grad_tensor(&[2, 3], lhs_data.clone());
            lhs_cuda.cast_inplace(DType::I8);
            let lhs_cuda = lhs_cuda.to_cuda();
            let rhs_cuda = make_grad_tensor(&[3, 3], rhs_data.clone());
            rhs_cuda.cast_inplace(rhs_dtype);
            let rhs_cuda = rhs_cuda.to_cuda();

            let loss_cuda = sum(&matmul(&lhs_cuda, &rhs_cuda));
            loss_cuda.backward();

            let loss_cpu = sum(&matmul(&lhs_cpu, &rhs_cpu));
            loss_cpu.backward();

            assert!(lhs_cuda.cloned_cuda_f32_grad().is_some());
            assert!(rhs_cuda.cloned_cuda_f32_grad().is_some());
            assert!(!lhs_cuda.has_host_f32_data());
            assert!(!lhs_cuda.has_host_grad());
            assert!(!rhs_cuda.has_host_grad());

            let lhs_grad = lhs_cuda.grad().expect("CUDA I8xfloatlike matmul lhs grad");
            let rhs_grad = rhs_cuda.grad().expect("CUDA I8xfloatlike matmul rhs grad");
            let lhs_ref = lhs_cpu.grad().expect("CPU I8xfloatlike matmul lhs grad");
            let rhs_ref = rhs_cpu.grad().expect("CPU I8xfloatlike matmul rhs grad");
            for (got, expect) in lhs_grad.iter().zip(lhs_ref.iter()) {
                assert!(
                    (got - expect).abs() <= 0.08,
                    "I8x{rhs_dtype:?} matmul lhs grad got {got}, expect {expect}"
                );
            }
            for (got, expect) in rhs_grad.iter().zip(rhs_ref.iter()) {
                assert!(
                    (got - expect).abs() <= 0.08,
                    "I8x{rhs_dtype:?} matmul rhs grad got {got}, expect {expect}"
                );
            }
        }

        crate::autograd::set_strict_device_execution(false);
        crate::ops::cuda::set_enabled(false);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_matmul_backward_matches_cpu_reference_in_strict_mode() {
        if !crate::ops::cuda::is_available() {
            return;
        }

        let a_data = vec![1.0, -2.0, 0.5, 3.0, -1.0, 2.0];
        let b_data = vec![0.5, 1.0, -1.5, 2.0, 0.25, -0.75, 1.25, -0.5, 0.75];
        let a_cpu = make_grad_tensor(&[2, 3], a_data.clone());
        let b_cpu = make_grad_tensor(&[3, 3], b_data.clone());
        let a_cuda = make_grad_tensor(&[2, 3], a_data).to_cuda();
        let b_cuda = make_grad_tensor(&[3, 3], b_data).to_cuda();

        crate::ops::cuda::set_enabled(true);
        set_strict_device_execution(true);
        let loss_cuda = sum(&matmul(&a_cuda, &b_cuda));
        loss_cuda.backward();
        set_strict_device_execution(false);
        crate::ops::cuda::set_enabled(false);

        let loss_cpu = sum(&matmul(&a_cpu, &b_cpu));
        loss_cpu.backward();

        assert!(!a_cuda.has_host_grad());
        assert!(!b_cuda.has_host_grad());
        assert!(a_cuda.cloned_cuda_f32_grad().is_some());
        assert!(b_cuda.cloned_cuda_f32_grad().is_some());
        let a_cuda_grad = a_cuda.grad().expect("cuda lhs grad");
        let b_cuda_grad = b_cuda.grad().expect("cuda rhs grad");
        let a_cpu_grad = a_cpu.grad().expect("cpu lhs grad");
        let b_cpu_grad = b_cpu.grad().expect("cpu rhs grad");
        for (got, expect) in a_cuda_grad.iter().zip(a_cpu_grad.iter()) {
            assert!(
                (got - expect).abs() < 1e-4,
                "lhs grad got {got}, expect {expect}"
            );
        }
        for (got, expect) in b_cuda_grad.iter().zip(b_cpu_grad.iter()) {
            assert!(
                (got - expect).abs() < 1e-4,
                "rhs grad got {got}, expect {expect}"
            );
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_batch_matmul_backward_matches_cpu_reference_in_strict_mode() {
        if !crate::ops::cuda::is_available() {
            return;
        }

        let lhs_data = vec![1.0, -2.0, 0.5, 3.0, -1.0, 2.0];
        let rhs_data = vec![0.5, 1.0, -1.5, 2.0, 0.25, -0.75];
        let lhs_cpu = make_grad_tensor(&[1, 1, 2, 3], lhs_data.clone());
        let rhs_cpu = make_grad_tensor(&[1, 1, 3, 2], rhs_data.clone());
        let lhs_cuda = make_grad_tensor(&[1, 1, 2, 3], lhs_data).to_cuda();
        let rhs_cuda = make_grad_tensor(&[1, 1, 3, 2], rhs_data).to_cuda();

        crate::ops::cuda::set_enabled(true);
        set_strict_device_execution(true);
        let loss_cuda = sum(&batch_matmul(&lhs_cuda, &rhs_cuda));
        loss_cuda.backward();
        set_strict_device_execution(false);
        crate::ops::cuda::set_enabled(false);

        let loss_cpu = sum(&batch_matmul(&lhs_cpu, &rhs_cpu));
        loss_cpu.backward();

        assert!(!lhs_cuda.has_host_grad());
        assert!(!rhs_cuda.has_host_grad());
        assert!(lhs_cuda.cloned_cuda_f32_grad().is_some());
        assert!(rhs_cuda.cloned_cuda_f32_grad().is_some());
        let lhs_cuda_grad = lhs_cuda.grad().expect("cuda lhs grad");
        let rhs_cuda_grad = rhs_cuda.grad().expect("cuda rhs grad");
        let lhs_cpu_grad = lhs_cpu.grad().expect("cpu lhs grad");
        let rhs_cpu_grad = rhs_cpu.grad().expect("cpu rhs grad");
        for (got, expect) in lhs_cuda_grad.iter().zip(lhs_cpu_grad.iter()) {
            assert!(
                (got - expect).abs() < 1e-4,
                "lhs grad got {got}, expect {expect}"
            );
        }
        for (got, expect) in rhs_cuda_grad.iter().zip(rhs_cpu_grad.iter()) {
            assert!(
                (got - expect).abs() < 1e-4,
                "rhs grad got {got}, expect {expect}"
            );
        }
    }

    #[test]
    #[ignore = "performance sanity test; run with --ignored --nocapture"]
    fn cpu_batch_matmul_backward_lowp_perf_smoke() {
        let b = 4;
        let h = 4;
        let m = 16;
        let k = 64;
        let n = 32;
        let lhs_data = (0..b * h * m * k)
            .map(|i| (i as f32 % 251.0) / 41.0 - 2.75)
            .collect::<Vec<_>>();
        let rhs_data = (0..b * h * k * n)
            .map(|i| ((i * 23) as f32 % 239.0) / 53.0 - 2.25)
            .collect::<Vec<_>>();
        let grad = ndarray::Array::from_shape_vec(
            ndarray::IxDyn(&[b, h, m, n]),
            (0..b * h * m * n)
                .map(|i| ((i * 11) as f32 % 89.0) / 29.0 - 1.25)
                .collect::<Vec<_>>(),
        )
        .expect("grad shape mismatch")
        .into_dyn();

        let measure = |lhs_dtype: DType, rhs_dtype: DType| {
            let lhs = make_tensor(&[b, h, m, k], lhs_data.clone(), lhs_dtype);
            let rhs = make_tensor(&[b, h, k, n], rhs_data.clone(), rhs_dtype);
            let start = std::time::Instant::now();
            let (d_lhs, d_rhs) =
                batch_matmul_backward_cpu_f32(&lhs, &rhs, &grad.view(), b, h, m, k, n);
            let elapsed_us = start.elapsed().as_secs_f64() * 1.0e6;
            assert_eq!(d_lhs.shape(), &[b, h, m, k]);
            assert_eq!(d_rhs.shape(), &[b, h, k, n]);
            assert!(d_lhs.iter().all(|v| v.is_finite()));
            assert!(d_rhs.iter().all(|v| v.is_finite()));
            elapsed_us
        };

        let bf16_bf16_us = measure(DType::BF16, DType::BF16);
        let i8_i8_us = measure(DType::I8, DType::I8);
        let bf16_i8_us = measure(DType::BF16, DType::I8);
        let f32_i8_us = measure(DType::F32, DType::I8);

        println!(
            "cpu batch_matmul backward b={b} h={h} m={m} n={n} k={k}: bf16xbf16={bf16_bf16_us:.1}us, i8xi8={i8_i8_us:.1}us, bf16xi8={bf16_i8_us:.1}us, f32xi8={f32_i8_us:.1}us"
        );

        let mm_m = 64;
        let mm_n = 64;
        let mm_k = 128;
        let mm_lhs = (0..mm_m * mm_k)
            .map(|i| (i as f32 % 251.0) / 43.0 - 2.5)
            .collect::<Vec<_>>();
        let mm_rhs = (0..mm_n * mm_k)
            .map(|i| ((i * 19) as f32 % 239.0) / 47.0 - 2.0)
            .collect::<Vec<_>>();
        let mm_grad = ndarray::Array::from_shape_vec(
            ndarray::IxDyn(&[mm_m, mm_n]),
            (0..mm_m * mm_n)
                .map(|i| ((i * 13) as f32 % 97.0) / 31.0 - 1.5)
                .collect::<Vec<_>>(),
        )
        .expect("matmul grad shape mismatch")
        .into_dyn();

        let measure_mm = |lhs_dtype: DType, rhs_dtype: DType| {
            let lhs = make_tensor(&[mm_m, mm_k], mm_lhs.clone(), lhs_dtype);
            let rhs = make_tensor(&[mm_n, mm_k], mm_rhs.clone(), rhs_dtype);
            let start = std::time::Instant::now();
            let (d_lhs, d_rhs) =
                matmul_backward_cpu_f32(&lhs, &rhs, &mm_grad.view(), mm_m, mm_k, mm_n);
            let elapsed_us = start.elapsed().as_secs_f64() * 1.0e6;
            assert_eq!(d_lhs.shape(), &[mm_m, mm_k]);
            assert_eq!(d_rhs.shape(), &[mm_n, mm_k]);
            assert!(d_lhs.iter().all(|v| v.is_finite()));
            assert!(d_rhs.iter().all(|v| v.is_finite()));
            elapsed_us
        };

        let mm_bf16_bf16_us = measure_mm(DType::BF16, DType::BF16);
        let mm_i8_i8_us = measure_mm(DType::I8, DType::I8);
        let mm_bf16_i8_us = measure_mm(DType::BF16, DType::I8);
        let mm_f32_i8_us = measure_mm(DType::F32, DType::I8);

        println!(
            "cpu matmul backward m={mm_m} n={mm_n} k={mm_k}: bf16xbf16={mm_bf16_bf16_us:.1}us, i8xi8={mm_i8_i8_us:.1}us, bf16xi8={mm_bf16_i8_us:.1}us, f32xi8={mm_f32_i8_us:.1}us"
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "performance sanity test; run with --ignored --nocapture"]
    fn cuda_resident_lowp_matmul_perf_smoke() {
        if !crate::ops::cuda::is_available() {
            return;
        }

        fn median_cuda_us(mut run: impl FnMut()) -> f64 {
            run();
            crate::ops::cuda::synchronize().expect("sync CUDA performance warmup");
            let mut samples = Vec::with_capacity(7);
            for _ in 0..7 {
                let start = std::time::Instant::now();
                run();
                crate::ops::cuda::synchronize().expect("sync CUDA performance sample");
                samples.push(start.elapsed().as_secs_f64() * 1.0e6);
            }
            samples.sort_by(f64::total_cmp);
            samples[samples.len() / 2]
        }

        let m = 128;
        let n = 128;
        let k = 256;
        let a = (0..m * k)
            .map(|i| (i as f32 % 251.0) / 43.0 - 2.5)
            .collect::<Vec<_>>();
        let b = (0..n * k)
            .map(|i| ((i * 19) as f32 % 239.0) / 47.0 - 2.0)
            .collect::<Vec<_>>();

        let a_f32 = crate::ops::cuda::upload_f32(&a).expect("upload matmul a f32");
        let b_f32 = crate::ops::cuda::upload_f32(&b).expect("upload matmul b f32");
        let f32_out = crate::ops::cuda::matmul_f32_no_host(&a_f32, &b_f32, m, n, k)
            .expect("CUDA f32 matmul perf path");
        let f32_us = median_cuda_us(|| {
            let _ = crate::ops::cuda::matmul_f32_no_host(&a_f32, &b_f32, m, n, k)
                .expect("CUDA f32 matmul performance sample");
        });
        let f32_vals = crate::ops::cuda::download_f32(&f32_out).expect("download f32 matmul");

        let a_f16 = a
            .iter()
            .map(|&v| f16::from_f32(v).to_bits())
            .collect::<Vec<_>>();
        let b_f16 = b
            .iter()
            .map(|&v| f16::from_f32(v).to_bits())
            .collect::<Vec<_>>();
        let a_f16_buf = crate::ops::cuda::upload_u16_storage(&a_f16).expect("upload matmul a f16");
        let b_f16_buf = crate::ops::cuda::upload_u16_storage(&b_f16).expect("upload matmul b f16");
        let f16_out = crate::ops::cuda::matmul_f16_buffer_no_host(&a_f16_buf, &b_f16_buf, m, n, k)
            .expect("CUDA resident f16 matmul perf path");
        let f16_us = median_cuda_us(|| {
            let _ = crate::ops::cuda::matmul_f16_buffer_no_host(&a_f16_buf, &b_f16_buf, m, n, k)
                .expect("CUDA resident f16 matmul performance sample");
        });
        let f16_vals = crate::ops::cuda::download_f32(&f16_out).expect("download f16 matmul");

        let a_bf16 = a
            .iter()
            .map(|&v| bf16::from_f32(v).to_bits())
            .collect::<Vec<_>>();
        let b_bf16 = b
            .iter()
            .map(|&v| bf16::from_f32(v).to_bits())
            .collect::<Vec<_>>();
        let a_bf16_buf =
            crate::ops::cuda::upload_u16_storage(&a_bf16).expect("upload matmul a bf16");
        let b_bf16_buf =
            crate::ops::cuda::upload_u16_storage(&b_bf16).expect("upload matmul b bf16");
        let bf16_out =
            crate::ops::cuda::matmul_bf16_buffer_no_host(&a_bf16_buf, &b_bf16_buf, m, n, k)
                .expect("CUDA resident bf16 matmul perf path");
        let bf16_us = median_cuda_us(|| {
            let _ = crate::ops::cuda::matmul_bf16_buffer_no_host(&a_bf16_buf, &b_bf16_buf, m, n, k)
                .expect("CUDA resident bf16 matmul performance sample");
        });
        let bf16_vals = crate::ops::cuda::download_f32(&bf16_out).expect("download bf16 matmul");

        let a_i8 = a
            .iter()
            .map(|&v| (v / 0.05).round().clamp(-127.0, 127.0) as i8)
            .collect::<Vec<_>>();
        let b_i8 = b
            .iter()
            .map(|&v| (v / 0.05).round().clamp(-127.0, 127.0) as i8)
            .collect::<Vec<_>>();
        let a_i8_buf = crate::ops::cuda::upload_i8_storage(&a_i8).expect("upload matmul a i8");
        let b_i8_buf = crate::ops::cuda::upload_i8_storage(&b_i8).expect("upload matmul b i8");
        let i8_out =
            crate::ops::cuda::matmul_i8_buffer_no_host(&a_i8_buf, 0.05, &b_i8_buf, 0.05, m, n, k)
                .expect("CUDA resident i8 matmul perf path");
        let i8_us = median_cuda_us(|| {
            let _ = crate::ops::cuda::matmul_i8_buffer_no_host(
                &a_i8_buf, 0.05, &b_i8_buf, 0.05, m, n, k,
            )
            .expect("CUDA resident i8 matmul performance sample");
        });
        let i8_vals = crate::ops::cuda::download_f32(&i8_out).expect("download i8 matmul");
        let i8_typed_us = median_cuda_us(|| {
            let _ = crate::ops::cuda::matmul_i8_typed_output_buffer_no_host(
                &a_i8_buf, 0.05, &b_i8_buf, 0.05, m, n, k,
            )
            .expect("CUDA resident typed-output i8 matmul performance sample");
        });
        let mut i8_quant_ref = vec![0.0f32; m * n];
        for row in 0..m {
            for col in 0..n {
                let mut acc = 0i32;
                for kk in 0..k {
                    acc += (a_i8[row * k + kk] as i32) * (b_i8[col * k + kk] as i32);
                }
                i8_quant_ref[row * n + col] = (acc as f32) * 0.05 * 0.05;
            }
        }

        let max_err = |vals: &[f32]| {
            vals.iter()
                .zip(f32_vals.iter())
                .map(|(&got, &expect)| (got - expect).abs())
                .fold(0.0f32, f32::max)
        };
        let max_err_against = |vals: &[f32], reference: &[f32]| {
            vals.iter()
                .zip(reference.iter())
                .map(|(&got, &expect)| (got - expect).abs())
                .fold(0.0f32, f32::max)
        };

        println!(
            "cuda resident lowp matmul m={m} n={n} k={k}: f32={f32_us:.1}us, f16={f16_us:.1}us max_err={:.5}, bf16={bf16_us:.1}us max_err={:.5}, i8_f32_out={i8_us:.1}us i8_typed_out={i8_typed_us:.1}us kernel_err={:.5} quant_err={:.5}",
            max_err(&f16_vals),
            max_err(&bf16_vals),
            max_err_against(&i8_vals, &i8_quant_ref),
            max_err(&i8_vals)
        );

        let f16_i8_out =
            crate::ops::cuda::matmul_f16_i8_buffer_no_host(&a_f16_buf, &b_i8_buf, 0.05, m, n, k)
                .expect("CUDA F16xI8 matmul perf path");
        let f16_i8_us = median_cuda_us(|| {
            let _ = crate::ops::cuda::matmul_f16_i8_buffer_no_host(
                &a_f16_buf, &b_i8_buf, 0.05, m, n, k,
            )
            .expect("CUDA F16xI8 matmul performance sample");
        });
        let f16_i8_vals =
            crate::ops::cuda::download_f32(&f16_i8_out).expect("download F16xI8 matmul");
        let bf16_i8_out =
            crate::ops::cuda::matmul_bf16_i8_buffer_no_host(&a_bf16_buf, &b_i8_buf, 0.05, m, n, k)
                .expect("CUDA BF16xI8 matmul perf path");
        let bf16_i8_us = median_cuda_us(|| {
            let _ = crate::ops::cuda::matmul_bf16_i8_buffer_no_host(
                &a_bf16_buf,
                &b_i8_buf,
                0.05,
                m,
                n,
                k,
            )
            .expect("CUDA BF16xI8 matmul performance sample");
        });
        let bf16_i8_vals =
            crate::ops::cuda::download_f32(&bf16_i8_out).expect("download BF16xI8 matmul");
        let f32_i8_out =
            crate::ops::cuda::matmul_f32_i8_buffer_no_host(&a_f32, &b_i8_buf, 0.05, m, n, k)
                .expect("CUDA F32xI8 matmul perf path");
        let f32_i8_us = median_cuda_us(|| {
            let _ =
                crate::ops::cuda::matmul_f32_i8_buffer_no_host(&a_f32, &b_i8_buf, 0.05, m, n, k)
                    .expect("CUDA F32xI8 matmul performance sample");
        });
        let f32_i8_vals =
            crate::ops::cuda::download_f32(&f32_i8_out).expect("download F32xI8 matmul");
        println!(
            "cuda resident mixed matmul m={m} n={n} k={k}: f16xi8={f16_i8_us:.1}us quant_err={:.5} f32_err={:.5}, bf16xi8={bf16_i8_us:.1}us quant_err={:.5} f32_err={:.5}, f32xi8={f32_i8_us:.1}us quant_err={:.5} f32_err={:.5}",
            max_err_against(&f16_i8_vals, &i8_quant_ref),
            max_err(&f16_i8_vals),
            max_err_against(&bf16_i8_vals, &i8_quant_ref),
            max_err(&bf16_i8_vals),
            max_err_against(&f32_i8_vals, &i8_quant_ref),
            max_err(&f32_i8_vals)
        );

        let i8_f16_out =
            crate::ops::cuda::matmul_i8_f16_buffer_no_host(&a_i8_buf, 0.05, &b_f16_buf, m, n, k)
                .expect("CUDA I8xF16 matmul perf path");
        let i8_f16_us = median_cuda_us(|| {
            let _ = crate::ops::cuda::matmul_i8_f16_buffer_no_host(
                &a_i8_buf, 0.05, &b_f16_buf, m, n, k,
            )
            .expect("CUDA I8xF16 matmul performance sample");
        });
        let i8_f16_vals =
            crate::ops::cuda::download_f32(&i8_f16_out).expect("download I8xF16 matmul");
        let i8_bf16_out =
            crate::ops::cuda::matmul_i8_bf16_buffer_no_host(&a_i8_buf, 0.05, &b_bf16_buf, m, n, k)
                .expect("CUDA I8xBF16 matmul perf path");
        let i8_bf16_us = median_cuda_us(|| {
            let _ = crate::ops::cuda::matmul_i8_bf16_buffer_no_host(
                &a_i8_buf,
                0.05,
                &b_bf16_buf,
                m,
                n,
                k,
            )
            .expect("CUDA I8xBF16 matmul performance sample");
        });
        let i8_bf16_vals =
            crate::ops::cuda::download_f32(&i8_bf16_out).expect("download I8xBF16 matmul");
        let i8_f32_out =
            crate::ops::cuda::matmul_i8_f32_buffer_no_host(&a_i8_buf, 0.05, &b_f32, m, n, k)
                .expect("CUDA I8xF32 matmul perf path");
        let i8_f32_us = median_cuda_us(|| {
            let _ =
                crate::ops::cuda::matmul_i8_f32_buffer_no_host(&a_i8_buf, 0.05, &b_f32, m, n, k)
                    .expect("CUDA I8xF32 matmul performance sample");
        });
        let i8_f32_vals =
            crate::ops::cuda::download_f32(&i8_f32_out).expect("download I8xF32 matmul");

        let mut i8_lhs_ref = vec![0.0f32; m * n];
        for row in 0..m {
            for col in 0..n {
                let mut acc = 0.0f32;
                for kk in 0..k {
                    acc += (a_i8[row * k + kk] as f32) * 0.05 * b[col * k + kk];
                }
                i8_lhs_ref[row * n + col] = acc;
            }
        }
        println!(
            "cuda resident mixed matmul mirrored m={m} n={n} k={k}: i8xf16={i8_f16_us:.1}us quant_err={:.5} f32_err={:.5}, i8xbf16={i8_bf16_us:.1}us quant_err={:.5} f32_err={:.5}, i8xf32={i8_f32_us:.1}us quant_err={:.5} f32_err={:.5}",
            max_err_against(&i8_f16_vals, &i8_lhs_ref),
            max_err(&i8_f16_vals),
            max_err_against(&i8_bf16_vals, &i8_lhs_ref),
            max_err(&i8_bf16_vals),
            max_err_against(&i8_f32_vals, &i8_lhs_ref),
            max_err(&i8_f32_vals)
        );

        let batch_count = 8;
        let bm = 16;
        let bn = 32;
        let bk = 64;
        let lhs = (0..batch_count * bm * bk)
            .map(|i| (i as f32 % 251.0) / 41.0 - 2.75)
            .collect::<Vec<_>>();
        let rhs = (0..batch_count * bk * bn)
            .map(|i| ((i * 23) as f32 % 239.0) / 53.0 - 2.25)
            .collect::<Vec<_>>();
        let lhs_f32 = crate::ops::cuda::upload_f32(&lhs).expect("upload batch lhs f32");
        let rhs_f32 = crate::ops::cuda::upload_f32(&rhs).expect("upload batch rhs f32");
        let batch_f32_out =
            crate::ops::cuda::batch_matmul_f32_no_host(&lhs_f32, &rhs_f32, batch_count, bm, bn, bk)
                .expect("CUDA f32 batch_matmul perf path");
        let batch_f32_us = median_cuda_us(|| {
            let _ = crate::ops::cuda::batch_matmul_f32_no_host(
                &lhs_f32,
                &rhs_f32,
                batch_count,
                bm,
                bn,
                bk,
            )
            .expect("CUDA f32 batch_matmul performance sample");
        });
        let batch_f32_vals =
            crate::ops::cuda::download_f32(&batch_f32_out).expect("download f32 batch_matmul");

        let lhs_bf16 = lhs
            .iter()
            .map(|&v| bf16::from_f32(v).to_bits())
            .collect::<Vec<_>>();
        let lhs_f16 = lhs
            .iter()
            .map(|&v| f16::from_f32(v).to_bits())
            .collect::<Vec<_>>();
        let rhs_i8 = rhs
            .iter()
            .map(|&v| (v / 0.05).round().clamp(-127.0, 127.0) as i8)
            .collect::<Vec<_>>();
        let lhs_i8 = lhs
            .iter()
            .map(|&v| (v / 0.05).round().clamp(-127.0, 127.0) as i8)
            .collect::<Vec<_>>();
        let rhs_bf16 = rhs
            .iter()
            .map(|&v| bf16::from_f32(v).to_bits())
            .collect::<Vec<_>>();
        let rhs_f16 = rhs
            .iter()
            .map(|&v| f16::from_f32(v).to_bits())
            .collect::<Vec<_>>();
        let lhs_bf16_buf =
            crate::ops::cuda::upload_u16_storage(&lhs_bf16).expect("upload batch lhs bf16");
        let lhs_f16_buf =
            crate::ops::cuda::upload_u16_storage(&lhs_f16).expect("upload batch lhs f16");
        let lhs_i8_buf = crate::ops::cuda::upload_i8_storage(&lhs_i8).expect("upload batch lhs i8");
        let rhs_i8_buf = crate::ops::cuda::upload_i8_storage(&rhs_i8).expect("upload batch rhs i8");
        let rhs_bf16_buf =
            crate::ops::cuda::upload_u16_storage(&rhs_bf16).expect("upload batch rhs bf16");
        let rhs_f16_buf =
            crate::ops::cuda::upload_u16_storage(&rhs_f16).expect("upload batch rhs f16");
        let batch_f16_i8_out = crate::ops::cuda::batch_matmul_f16_i8_buffer_no_host(
            &lhs_f16_buf,
            &rhs_i8_buf,
            0.05,
            batch_count,
            bm,
            bn,
            bk,
        )
        .expect("CUDA F16xI8 batch_matmul perf path");
        let batch_f16_i8_us = median_cuda_us(|| {
            let _ = crate::ops::cuda::batch_matmul_f16_i8_buffer_no_host(
                &lhs_f16_buf,
                &rhs_i8_buf,
                0.05,
                batch_count,
                bm,
                bn,
                bk,
            )
            .expect("CUDA F16xI8 batch_matmul performance sample");
        });
        let batch_f16_i8_vals = crate::ops::cuda::download_f32(&batch_f16_i8_out)
            .expect("download F16xI8 batch_matmul");

        let batch_bf16_i8_out = crate::ops::cuda::batch_matmul_bf16_i8_buffer_no_host(
            &lhs_bf16_buf,
            &rhs_i8_buf,
            0.05,
            batch_count,
            bm,
            bn,
            bk,
        )
        .expect("CUDA BF16xI8 batch_matmul perf path");
        let batch_bf16_i8_us = median_cuda_us(|| {
            let _ = crate::ops::cuda::batch_matmul_bf16_i8_buffer_no_host(
                &lhs_bf16_buf,
                &rhs_i8_buf,
                0.05,
                batch_count,
                bm,
                bn,
                bk,
            )
            .expect("CUDA BF16xI8 batch_matmul performance sample");
        });
        let batch_bf16_i8_vals = crate::ops::cuda::download_f32(&batch_bf16_i8_out)
            .expect("download BF16xI8 batch_matmul");

        let batch_f32_i8_out = crate::ops::cuda::batch_matmul_f32_i8_buffer_no_host(
            &lhs_f32,
            &rhs_i8_buf,
            0.05,
            batch_count,
            bm,
            bn,
            bk,
        )
        .expect("CUDA F32xI8 batch_matmul perf path");
        let batch_f32_i8_us = median_cuda_us(|| {
            let _ = crate::ops::cuda::batch_matmul_f32_i8_buffer_no_host(
                &lhs_f32,
                &rhs_i8_buf,
                0.05,
                batch_count,
                bm,
                bn,
                bk,
            )
            .expect("CUDA F32xI8 batch_matmul performance sample");
        });
        let batch_f32_i8_vals = crate::ops::cuda::download_f32(&batch_f32_i8_out)
            .expect("download F32xI8 batch_matmul");

        let mut batch_quant_ref = vec![0.0f32; batch_count * bm * bn];
        for batch in 0..batch_count {
            for row in 0..bm {
                for col in 0..bn {
                    let mut acc = 0.0f32;
                    for kk in 0..bk {
                        acc += lhs[batch * bm * bk + row * bk + kk]
                            * (rhs_i8[batch * bk * bn + kk * bn + col] as f32)
                            * 0.05;
                    }
                    batch_quant_ref[batch * bm * bn + row * bn + col] = acc;
                }
            }
        }
        println!(
            "cuda resident mixed batch_matmul batch={batch_count} m={bm} n={bn} k={bk}: f32={batch_f32_us:.1}us, f16xi8={batch_f16_i8_us:.1}us quant_err={:.5} f32_err={:.5}, bf16xi8={batch_bf16_i8_us:.1}us quant_err={:.5} f32_err={:.5}, f32xi8={batch_f32_i8_us:.1}us quant_err={:.5} f32_err={:.5}",
            max_err_against(&batch_f16_i8_vals, &batch_quant_ref),
            max_err_against(&batch_f16_i8_vals, &batch_f32_vals),
            max_err_against(&batch_bf16_i8_vals, &batch_quant_ref),
            max_err_against(&batch_bf16_i8_vals, &batch_f32_vals),
            max_err_against(&batch_f32_i8_vals, &batch_quant_ref),
            max_err_against(&batch_f32_i8_vals, &batch_f32_vals)
        );

        let batch_i8_f16_out = crate::ops::cuda::batch_matmul_i8_f16_buffer_no_host(
            &lhs_i8_buf,
            0.05,
            &rhs_f16_buf,
            batch_count,
            bm,
            bn,
            bk,
        )
        .expect("CUDA I8xF16 batch_matmul perf path");
        let batch_i8_f16_us = median_cuda_us(|| {
            let _ = crate::ops::cuda::batch_matmul_i8_f16_buffer_no_host(
                &lhs_i8_buf,
                0.05,
                &rhs_f16_buf,
                batch_count,
                bm,
                bn,
                bk,
            )
            .expect("CUDA I8xF16 batch_matmul performance sample");
        });
        let batch_i8_f16_vals = crate::ops::cuda::download_f32(&batch_i8_f16_out)
            .expect("download I8xF16 batch_matmul");

        let batch_i8_bf16_out = crate::ops::cuda::batch_matmul_i8_bf16_buffer_no_host(
            &lhs_i8_buf,
            0.05,
            &rhs_bf16_buf,
            batch_count,
            bm,
            bn,
            bk,
        )
        .expect("CUDA I8xBF16 batch_matmul perf path");
        let batch_i8_bf16_us = median_cuda_us(|| {
            let _ = crate::ops::cuda::batch_matmul_i8_bf16_buffer_no_host(
                &lhs_i8_buf,
                0.05,
                &rhs_bf16_buf,
                batch_count,
                bm,
                bn,
                bk,
            )
            .expect("CUDA I8xBF16 batch_matmul performance sample");
        });
        let batch_i8_bf16_vals = crate::ops::cuda::download_f32(&batch_i8_bf16_out)
            .expect("download I8xBF16 batch_matmul");

        let batch_i8_f32_out = crate::ops::cuda::batch_matmul_i8_f32_buffer_no_host(
            &lhs_i8_buf,
            0.05,
            &rhs_f32,
            batch_count,
            bm,
            bn,
            bk,
        )
        .expect("CUDA I8xF32 batch_matmul perf path");
        let batch_i8_f32_us = median_cuda_us(|| {
            let _ = crate::ops::cuda::batch_matmul_i8_f32_buffer_no_host(
                &lhs_i8_buf,
                0.05,
                &rhs_f32,
                batch_count,
                bm,
                bn,
                bk,
            )
            .expect("CUDA I8xF32 batch_matmul performance sample");
        });
        let batch_i8_f32_vals = crate::ops::cuda::download_f32(&batch_i8_f32_out)
            .expect("download I8xF32 batch_matmul");

        let mut batch_i8_lhs_ref = vec![0.0f32; batch_count * bm * bn];
        for batch in 0..batch_count {
            for row in 0..bm {
                for col in 0..bn {
                    let mut acc = 0.0f32;
                    for kk in 0..bk {
                        acc += (lhs_i8[batch * bm * bk + row * bk + kk] as f32)
                            * 0.05
                            * rhs[batch * bk * bn + kk * bn + col];
                    }
                    batch_i8_lhs_ref[batch * bm * bn + row * bn + col] = acc;
                }
            }
        }
        println!(
            "cuda resident mixed batch_matmul mirrored batch={batch_count} m={bm} n={bn} k={bk}: i8xf16={batch_i8_f16_us:.1}us quant_err={:.5} f32_err={:.5}, i8xbf16={batch_i8_bf16_us:.1}us quant_err={:.5} f32_err={:.5}, i8xf32={batch_i8_f32_us:.1}us quant_err={:.5} f32_err={:.5}",
            max_err_against(&batch_i8_f16_vals, &batch_i8_lhs_ref),
            max_err_against(&batch_i8_f16_vals, &batch_f32_vals),
            max_err_against(&batch_i8_bf16_vals, &batch_i8_lhs_ref),
            max_err_against(&batch_i8_bf16_vals, &batch_f32_vals),
            max_err_against(&batch_i8_f32_vals, &batch_i8_lhs_ref),
            max_err_against(&batch_i8_f32_vals, &batch_f32_vals)
        );

        let batch_i8_i8_f32_us = median_cuda_us(|| {
            let _ = crate::ops::cuda::batch_matmul_i8_buffer_no_host(
                &lhs_i8_buf,
                0.05,
                &rhs_i8_buf,
                0.05,
                batch_count,
                bm,
                bn,
                bk,
            )
            .expect("CUDA I8xI8 f32-output batch_matmul performance sample");
        });

        let batch_i8_i8_typed_us = median_cuda_us(|| {
            let _ = crate::ops::cuda::batch_matmul_i8_typed_output_buffer_no_host(
                &lhs_i8_buf,
                0.05,
                &rhs_i8_buf,
                0.05,
                batch_count,
                bm,
                bn,
                bk,
            )
            .expect("CUDA I8xI8 typed-output batch_matmul performance sample");
        });
        println!(
            "cuda resident I8xI8 batch_matmul batch={batch_count} m={bm} n={bn} k={bk}: f32_out={batch_i8_i8_f32_us:.1}us, typed_i8_out={batch_i8_i8_typed_us:.1}us"
        );

        let grad = (0..m * n)
            .map(|i| ((i * 13) as f32 % 97.0) / 31.0 - 1.5)
            .collect::<Vec<_>>();
        let grad_buf = crate::ops::cuda::upload_f32(&grad).expect("upload matmul grad");
        let backward_f16_i8_us = median_cuda_us(|| {
            let _ = crate::ops::cuda::matmul_backward_f16_i8_no_host(
                &grad_buf, &a_f16_buf, &b_i8_buf, 0.05, m, k, n,
            )
            .expect("CUDA F16xI8 matmul backward performance sample");
        });
        let backward_bf16_i8_us = median_cuda_us(|| {
            let _ = crate::ops::cuda::matmul_backward_bf16_i8_no_host(
                &grad_buf,
                &a_bf16_buf,
                &b_i8_buf,
                0.05,
                m,
                k,
                n,
            )
            .expect("CUDA BF16xI8 matmul backward performance sample");
        });
        let backward_f32_i8_us = median_cuda_us(|| {
            let _ = crate::ops::cuda::matmul_backward_f32_i8_no_host(
                &grad_buf, &a_f32, &b_i8_buf, 0.05, m, k, n,
            )
            .expect("CUDA F32xI8 matmul backward performance sample");
        });
        let backward_i8_f16_us = median_cuda_us(|| {
            let _ = crate::ops::cuda::matmul_backward_i8_f16_no_host(
                &grad_buf, &a_i8_buf, 0.05, &b_f16_buf, m, k, n,
            )
            .expect("CUDA I8xF16 matmul backward performance sample");
        });
        let backward_i8_bf16_us = median_cuda_us(|| {
            let _ = crate::ops::cuda::matmul_backward_i8_bf16_no_host(
                &grad_buf,
                &a_i8_buf,
                0.05,
                &b_bf16_buf,
                m,
                k,
                n,
            )
            .expect("CUDA I8xBF16 matmul backward performance sample");
        });
        let backward_i8_f32_us = median_cuda_us(|| {
            let _ = crate::ops::cuda::matmul_backward_i8_f32_no_host(
                &grad_buf, &a_i8_buf, 0.05, &b_f32, m, k, n,
            )
            .expect("CUDA I8xF32 matmul backward performance sample");
        });

        let batch_grad = (0..batch_count * bm * bn)
            .map(|i| ((i * 11) as f32 % 89.0) / 29.0 - 1.25)
            .collect::<Vec<_>>();
        let batch_grad_buf =
            crate::ops::cuda::upload_f32(&batch_grad).expect("upload batch_matmul grad");
        let batch_backward_f16_i8_us = median_cuda_us(|| {
            let _ = crate::ops::cuda::batch_matmul_backward_f16_i8_no_host(
                &batch_grad_buf,
                &lhs_f16_buf,
                &rhs_i8_buf,
                0.05,
                batch_count,
                bm,
                bk,
                bn,
            )
            .expect("CUDA F16xI8 batch_matmul backward performance sample");
        });
        let batch_backward_bf16_i8_us = median_cuda_us(|| {
            let _ = crate::ops::cuda::batch_matmul_backward_bf16_i8_no_host(
                &batch_grad_buf,
                &lhs_bf16_buf,
                &rhs_i8_buf,
                0.05,
                batch_count,
                bm,
                bk,
                bn,
            )
            .expect("CUDA BF16xI8 batch_matmul backward performance sample");
        });
        let batch_backward_f32_i8_us = median_cuda_us(|| {
            let _ = crate::ops::cuda::batch_matmul_backward_f32_i8_no_host(
                &batch_grad_buf,
                &lhs_f32,
                &rhs_i8_buf,
                0.05,
                batch_count,
                bm,
                bk,
                bn,
            )
            .expect("CUDA F32xI8 batch_matmul backward performance sample");
        });
        let batch_backward_i8_f16_us = median_cuda_us(|| {
            let _ = crate::ops::cuda::batch_matmul_backward_i8_f16_no_host(
                &batch_grad_buf,
                &lhs_i8_buf,
                0.05,
                &rhs_f16_buf,
                batch_count,
                bm,
                bk,
                bn,
            )
            .expect("CUDA I8xF16 batch_matmul backward performance sample");
        });
        let batch_backward_i8_bf16_us = median_cuda_us(|| {
            let _ = crate::ops::cuda::batch_matmul_backward_i8_bf16_no_host(
                &batch_grad_buf,
                &lhs_i8_buf,
                0.05,
                &rhs_bf16_buf,
                batch_count,
                bm,
                bk,
                bn,
            )
            .expect("CUDA I8xBF16 batch_matmul backward performance sample");
        });
        let batch_backward_i8_f32_us = median_cuda_us(|| {
            let _ = crate::ops::cuda::batch_matmul_backward_i8_f32_no_host(
                &batch_grad_buf,
                &lhs_i8_buf,
                0.05,
                &rhs_f32,
                batch_count,
                bm,
                bk,
                bn,
            )
            .expect("CUDA I8xF32 batch_matmul backward performance sample");
        });

        println!(
            "cuda resident mixed backward: matmul f16xi8={backward_f16_i8_us:.1}us bf16xi8={backward_bf16_i8_us:.1}us f32xi8={backward_f32_i8_us:.1}us i8xf16={backward_i8_f16_us:.1}us i8xbf16={backward_i8_bf16_us:.1}us i8xf32={backward_i8_f32_us:.1}us; batch_matmul f16xi8={batch_backward_f16_i8_us:.1}us bf16xi8={batch_backward_bf16_i8_us:.1}us f32xi8={batch_backward_f32_i8_us:.1}us i8xf16={batch_backward_i8_f16_us:.1}us i8xbf16={batch_backward_i8_bf16_us:.1}us i8xf32={batch_backward_i8_f32_us:.1}us"
        );
    }
}
