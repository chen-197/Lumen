//! Matrix products, projections, and dtype-aware CPU/CUDA routing.

mod conversion;
mod policy;
pub(crate) use conversion::{with_bf16_input_as_f32, with_f16_input_as_f32, with_i8_input_as_f32};
mod dot;
pub(crate) use dot::{
    dot_unrolled, dot_unrolled_f32_bf16, dot_unrolled_f32_f16, dot_unrolled_f32_i8, dot2_unrolled,
    dot2_unrolled_f32_bf16, dot2_unrolled_f32_f16, dot2_unrolled_f32_i8, dot3_unrolled,
    dot3_unrolled_f32_bf16, dot3_unrolled_f32_f16, dot3_unrolled_f32_i8,
};
mod matvec;
pub use matvec::{
    matvec_rowmajor_parallel, matvec_rowmajor_parallel_bf16_bf16,
    matvec_rowmajor_parallel_bf16_f32, matvec_rowmajor_parallel_bf16_i8,
    matvec_rowmajor_parallel_f16_f16, matvec_rowmajor_parallel_f32_bf16,
    matvec_rowmajor_parallel_f32_f16, matvec_rowmajor_parallel_f32_i8,
    matvec_rowmajor_parallel_f32_i8_matmul, matvec_rowmajor_parallel_i8_bf16,
    matvec_rowmajor_parallel_i8_f32, matvec_rowmajor_parallel_i8_i8,
    matvec_rowmajor_parallel_mixed,
};
mod projection;
pub use projection::{
    dual_matvec_rowmajor_parallel, dual_matvec_rowmajor_parallel_mixed,
    dual_matvec_silu_mul_rowmajor_parallel, dual_matvec_silu_mul_rowmajor_parallel_mixed,
};
pub(crate) use projection::{
    dual_matvec_silu_mul_rowmajor_parallel_f32_bf16,
    dual_matvec_silu_mul_rowmajor_parallel_f32_f16, dual_matvec_silu_mul_rowmajor_parallel_f32_i8,
    dual_matvec_silu_mul_rowmajor_parallel_i8_i8, qkv_matvec_rowmajor_parallel,
    qkv_matvec_rowmajor_parallel_f32_bf16, qkv_matvec_rowmajor_parallel_f32_f16,
    qkv_matvec_rowmajor_parallel_f32_i8, qkv_matvec_rowmajor_parallel_i8_i8,
};
mod argmax;
pub use argmax::{matvec_argmax_rowmajor_parallel, matvec_argmax_rowmajor_parallel_mixed};
mod cuda_route;
mod layout;
mod prefill;
mod rows;
mod standard_cpu;
mod tensor;
pub use tensor::{batch_matmul, matmul, matmul_transposed_rhs};
mod backward;

use crate::autograd::{
    StoragePreference, Tensor, TensorData, TensorStorageOwned, TensorStorageView, is_no_grad,
};
use crate::device::{assert_native_device_support, assert_same_device, is_strict_device_execution};
use crate::ops::fp_kernels::{
    dot_bf16_bf16_arch, dot_f16_f16_arch, dot_f32_arch, dot_f32_bf16_arch, dot_f32_f16_arch,
    dot2_bf16_bf16_arch, dot2_f16_f16_arch, dot2_f32_arch, dot2_f32_bf16_arch, dot2_f32_f16_arch,
    dot3_bf16_bf16_arch, dot3_f16_f16_arch, dot3_f32_arch, dot3_f32_bf16_arch, dot3_f32_f16_arch,
};
use crate::ops::int8_kernels::{
    I8ScaledRow, dot_f32_i8_arch, dot_i8_i8_arch, dot2_f32_i8_arch, dot2_i8_i8_arch,
    dot3_f32_i8_arch, dot3_i8_i8_arch,
};
use crate::ops::{cpu_gemm, cuda};
use crate::precision::DType;
use half::{bf16, f16, slice::HalfFloatSliceExt};
use ndarray::linalg::general_mat_mul;
use ndarray::{Array2, Array4, Ix2, Ix4};
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
// Short reductions do not amortize native dispatch and scalar tails well.
const ARGMAX_NATIVE_MIN_K: usize = 128;

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

thread_local! {
    static F16_TO_F32_SCRATCH: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    static BF16_TO_F32_SCRATCH: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    static I8_TO_F32_SCRATCH: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    // Two operands can be borrowed together. Deeper reentry uses a temporary.
    static COMPUTE_F32_SCRATCH: [RefCell<Vec<f32>>; 2] = const {
        [RefCell::new(Vec::new()), RefCell::new(Vec::new())]
    };
}

// Bound retained conversion storage to 2 MiB per calling thread. Larger inputs
// still work, using a temporary that is released when the numerical call ends.
const MAX_COMPUTE_SCRATCH_ELEMENTS: usize = 256 * 1024;

pub trait DotElem: Copy + Send + Sync {
    fn to_f32(self) -> f32;

    /// Copy a contiguous input, using a bulk conversion when available.
    fn copy_to_f32(input: &[Self], output: &mut [f32]) {
        assert_eq!(input.len(), output.len());
        for (dst, &src) in output.iter_mut().zip(input) {
            *dst = src.to_f32();
        }
    }
}

impl DotElem for f32 {
    #[inline]
    fn to_f32(self) -> f32 {
        self
    }

    fn copy_to_f32(input: &[Self], output: &mut [f32]) {
        output.copy_from_slice(input);
    }
}

impl DotElem for bf16 {
    #[inline]
    fn to_f32(self) -> f32 {
        self.to_f32()
    }

    fn copy_to_f32(input: &[Self], output: &mut [f32]) {
        input.convert_to_f32_slice(output);
    }
}

impl DotElem for f16 {
    #[inline]
    fn to_f32(self) -> f32 {
        self.to_f32()
    }

    fn copy_to_f32(input: &[Self], output: &mut [f32]) {
        input.convert_to_f32_slice(output);
    }
}

#[derive(Clone, Copy)]
pub enum SliceRef<'a> {
    F32(&'a [f32]),
    F16(&'a [f16]),
    BF16(&'a [bf16]),
    I8(&'a [i8], f32),
}

#[cfg(test)]
#[path = "../../../tests/unit/ops/matmul/empty.rs"]
mod empty_tests;

#[cfg(test)]
#[path = "../../../tests/unit/ops/matmul/backward.rs"]
mod backward_tests;

#[cfg(test)]
#[path = "../../../tests/unit/ops/matmul/tests.rs"]
mod tests;

#[cfg(test)]
#[path = "../../../tests/unit/ops/matmul/standard.rs"]
mod standard_tests;
