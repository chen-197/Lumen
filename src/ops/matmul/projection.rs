use super::dot::{
    dot_unrolled_bf16_bf16_scalar, dot_unrolled_f16_f16_scalar, dot_unrolled_from_slice,
    dot2_unrolled_i8_i8, dot3_rows_unrolled_f32_i8, dot3_unrolled_i8_i8,
};
use super::policy::{
    should_use_i8_block_kernel, should_use_i8_qkv_row4_kernel, should_use_i8_silu_block_kernel,
    should_use_mixed_dual_block_kernel,
};
use super::*;

#[inline]
pub(super) fn row_slice(rowmajor: SliceRef<'_>, row_idx: usize, k_dim: usize) -> SliceRef<'_> {
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

    if !crate::cpu::should_parallelize(n_rows, k_dim) {
        for i in 0..n_rows {
            let row0 = &w0_rowmajor[i * k_dim..(i + 1) * k_dim];
            let row1 = &w1_rowmajor[i * k_dim..(i + 1) * k_dim];
            let (s0, s1) = dot2_unrolled(x, row0, row1);
            out0[i] = s0;
            out1[i] = s1;
        }
    } else {
        crate::cpu::for_each_chunk2_mut(
            &mut out0[..],
            &mut out1[..],
            1,
            k_dim.saturating_mul(2),
            |(i, (dst0, dst1))| {
                let dst0 = &mut dst0[0];
                let dst1 = &mut dst1[0];
                let row0 = &w0_rowmajor[i * k_dim..(i + 1) * k_dim];
                let row1 = &w1_rowmajor[i * k_dim..(i + 1) * k_dim];
                let (s0, s1) = dot2_unrolled(x, row0, row1);
                *dst0 = s0;
                *dst1 = s1;
            },
        );
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

    if !crate::cpu::should_parallelize(n_rows, k_dim) {
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
        crate::cpu::for_each_chunk2_mut(
            &mut out0[..],
            &mut out1[..],
            MIXED_ROW_PAR_CHUNK_ROWS,
            k_dim
                .saturating_mul(MIXED_ROW_PAR_CHUNK_ROWS)
                .saturating_mul(2),
            |(chunk_idx, (out0_chunk, out1_chunk))| {
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
            },
        );
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

    if !crate::cpu::should_parallelize(n_rows, k_dim) {
        for i in 0..n_rows {
            let row0 = &w0_rowmajor[i * k_dim..(i + 1) * k_dim];
            let row1 = &w1_rowmajor[i * k_dim..(i + 1) * k_dim];
            let (s0, s1) = dot2_unrolled_f32_f16(x, row0, row1);
            out0[i] = s0;
            out1[i] = s1;
        }
    } else {
        crate::cpu::for_each_chunk2_mut(
            &mut out0[..],
            &mut out1[..],
            MIXED_ROW_PAR_CHUNK_ROWS,
            k_dim
                .saturating_mul(MIXED_ROW_PAR_CHUNK_ROWS)
                .saturating_mul(2),
            |(chunk_idx, (out0_chunk, out1_chunk))| {
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
            },
        );
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

    if !crate::cpu::should_parallelize(n_rows, k_dim) {
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
        crate::cpu::for_each_chunk2_mut(
            &mut out0[..],
            &mut out1[..],
            MATVEC_I8_PAR_CHUNK_ROWS,
            k_dim
                .saturating_mul(MATVEC_I8_PAR_CHUNK_ROWS)
                .saturating_mul(2),
            |(chunk_idx, (out0_chunk, out1_chunk))| {
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
            },
        );
    }
}

#[inline]
pub(super) fn dual_matvec_rowmajor_block_parallel_f32_bf16(
    x: &[f32],
    w0_rowmajor: &[bf16],
    w1_rowmajor: &[bf16],
    _n_rows: usize,
    k_dim: usize,
    out0: &mut [f32],
    out1: &mut [f32],
) {
    crate::cpu::for_each_chunk2_mut(
        &mut out0[..],
        &mut out1[..],
        MATVEC_BLOCK_ROWS,
        k_dim.saturating_mul(MATVEC_BLOCK_ROWS).saturating_mul(2),
        |(block_idx, (out0_chunk, out1_chunk))| {
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
        },
    );
}

pub(super) fn dual_matvec_rowmajor_block_parallel_f32_i8(
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

    crate::cpu::for_each_chunk2_mut(
        &mut out0[..],
        &mut out1[..],
        MATVEC_BLOCK_ROWS,
        k_dim.saturating_mul(MATVEC_BLOCK_ROWS).saturating_mul(2),
        |(block_idx, (out0_chunk, out1_chunk))| {
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
        },
    );
}

#[inline]
pub(super) fn dual_matvec_rowmajor_parallel_bf16_f32(
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
pub(super) fn dual_matvec_rowmajor_parallel_bf16_bf16(
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
        if !crate::cpu::should_parallelize(n_rows, k_dim) {
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
            crate::cpu::for_each_chunk2_mut(
                &mut out0[..],
                &mut out1[..],
                1,
                k_dim.saturating_mul(2),
                |(i, (dst0, dst1))| {
                    let dst0 = &mut dst0[0];
                    let dst1 = &mut dst1[0];
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
                },
            );
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
pub(super) fn dual_matvec_rowmajor_parallel_f16_f16(
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
        if !crate::cpu::should_parallelize(n_rows, k_dim) {
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
            crate::cpu::for_each_chunk2_mut(
                &mut out0[..],
                &mut out1[..],
                1,
                k_dim.saturating_mul(2),
                |(i, (dst0, dst1))| {
                    let dst0 = &mut dst0[0];
                    let dst1 = &mut dst1[0];
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
                },
            );
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
pub(super) fn dual_matvec_rowmajor_parallel_bf16_i8(
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
pub(super) fn dual_matvec_rowmajor_parallel_i8_f32(
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
pub(super) fn dual_matvec_rowmajor_parallel_i8_bf16(
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
pub(super) fn dual_matvec_rowmajor_parallel_i8_i8(
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

    if !crate::cpu::should_parallelize(n_rows, k_dim) {
        for i in 0..n_rows {
            let row0 = &w0_rowmajor[i * k_dim..(i + 1) * k_dim];
            let row1 = &w1_rowmajor[i * k_dim..(i + 1) * k_dim];
            let (s0, s1) = dot2_unrolled_i8_i8(x, x_scale, row0, w0_scale, row1, w1_scale);
            out0[i] = s0;
            out1[i] = s1;
        }
    } else {
        crate::cpu::for_each_chunk2_mut(
            &mut out0[..],
            &mut out1[..],
            MATVEC_I8_PAR_CHUNK_ROWS,
            k_dim
                .saturating_mul(MATVEC_I8_PAR_CHUNK_ROWS)
                .saturating_mul(2),
            |(chunk_idx, (out0_chunk, out1_chunk))| {
                let row_start = chunk_idx * MATVEC_I8_PAR_CHUNK_ROWS;
                for offset in 0..out0_chunk.len() {
                    let row_idx = row_start + offset;
                    let row0 = &w0_rowmajor[row_idx * k_dim..(row_idx + 1) * k_dim];
                    let row1 = &w1_rowmajor[row_idx * k_dim..(row_idx + 1) * k_dim];
                    let (s0, s1) = dot2_unrolled_i8_i8(x, x_scale, row0, w0_scale, row1, w1_scale);
                    out0_chunk[offset] = s0;
                    out1_chunk[offset] = s1;
                }
            },
        );
    }
}

pub(super) fn dual_matvec_rowmajor_parallel_mixed_f32_input(
    x: &[f32],
    w0_rowmajor: SliceRef<'_>,
    w1_rowmajor: SliceRef<'_>,
    n_rows: usize,
    k_dim: usize,
    out0: &mut [f32],
    out1: &mut [f32],
) {
    if !crate::cpu::should_parallelize(n_rows, k_dim) {
        for i in 0..n_rows {
            out0[i] = dot_unrolled_from_slice(x, row_slice(w0_rowmajor, i, k_dim));
            out1[i] = dot_unrolled_from_slice(x, row_slice(w1_rowmajor, i, k_dim));
        }
    } else {
        crate::cpu::for_each_chunk2_mut(
            &mut out0[..],
            &mut out1[..],
            1,
            k_dim.saturating_mul(2),
            |(i, (out0_val, out1_val))| {
                let out0_val = &mut out0_val[0];
                let out1_val = &mut out1_val[0];
                *out0_val = dot_unrolled_from_slice(x, row_slice(w0_rowmajor, i, k_dim));
                *out1_val = dot_unrolled_from_slice(x, row_slice(w1_rowmajor, i, k_dim));
            },
        );
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
    if !crate::cpu::should_parallelize(shared_rows, k_dim.saturating_mul(3)) {
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
        crate::cpu::for_each_chunk3_mut(
            &mut q_out[..shared_rows],
            &mut k_out[..shared_rows],
            &mut v_out[..shared_rows],
            1,
            k_dim.saturating_mul(3),
            |(i, ((q_dst, k_dst), v_dst))| {
                let q_dst = &mut q_dst[0];
                let k_dst = &mut k_dst[0];
                let v_dst = &mut v_dst[0];
                let q_row = &q_rowmajor[i * k_dim..(i + 1) * k_dim];
                let k_row = &k_rowmajor[i * k_dim..(i + 1) * k_dim];
                let v_row = &v_rowmajor[i * k_dim..(i + 1) * k_dim];
                let (q, k, v) = dot3_unrolled(x, q_row, k_row, v_row);
                *q_dst = q;
                *k_dst = k;
                *v_dst = v;
            },
        );
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
    if !crate::cpu::should_parallelize(shared_rows, k_dim.saturating_mul(3)) {
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
        crate::cpu::for_each_chunk3_mut(
            &mut q_out[..shared_rows],
            &mut k_out[..shared_rows],
            &mut v_out[..shared_rows],
            1,
            k_dim.saturating_mul(3),
            |(i, ((q_dst, k_dst), v_dst))| {
                let q_dst = &mut q_dst[0];
                let k_dst = &mut k_dst[0];
                let v_dst = &mut v_dst[0];
                let q_row = &q_rowmajor[i * k_dim..(i + 1) * k_dim];
                let k_row = &k_rowmajor[i * k_dim..(i + 1) * k_dim];
                let v_row = &v_rowmajor[i * k_dim..(i + 1) * k_dim];
                let (q, k, v) = dot3_unrolled_f32_bf16(x, q_row, k_row, v_row);
                *q_dst = q;
                *k_dst = k;
                *v_dst = v;
            },
        );
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
    if !crate::cpu::should_parallelize(shared_rows, k_dim.saturating_mul(3)) {
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
        crate::cpu::for_each_chunk3_mut(
            &mut q_out[..shared_rows],
            &mut k_out[..shared_rows],
            &mut v_out[..shared_rows],
            1,
            k_dim.saturating_mul(3),
            |(i, ((q_dst, k_dst), v_dst))| {
                let q_dst = &mut q_dst[0];
                let k_dst = &mut k_dst[0];
                let v_dst = &mut v_dst[0];
                let q_row = &q_rowmajor[i * k_dim..(i + 1) * k_dim];
                let k_row = &k_rowmajor[i * k_dim..(i + 1) * k_dim];
                let v_row = &v_rowmajor[i * k_dim..(i + 1) * k_dim];
                let (q, k, v) = dot3_unrolled_f32_f16(x, q_row, k_row, v_row);
                *q_dst = q;
                *k_dst = k;
                *v_dst = v;
            },
        );
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
    if !crate::cpu::should_parallelize(shared_rows, k_dim.saturating_mul(3)) {
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
        crate::cpu::for_each_chunk3_mut(
            &mut q_out[..shared_rows],
            &mut k_out[..shared_rows],
            &mut v_out[..shared_rows],
            QKV_I8_PAR_CHUNK_ROWS,
            k_dim
                .saturating_mul(QKV_I8_PAR_CHUNK_ROWS)
                .saturating_mul(3),
            |(chunk_idx, ((q_chunk, k_chunk), v_chunk))| {
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
            },
        );
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
    if !crate::cpu::should_parallelize(shared_rows, k_dim.saturating_mul(3)) {
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
        crate::cpu::for_each_chunk3_mut(
            &mut q_out[..shared_rows],
            &mut k_out[..shared_rows],
            &mut v_out[..shared_rows],
            QKV_I8_PAR_CHUNK_ROWS,
            k_dim
                .saturating_mul(QKV_I8_PAR_CHUNK_ROWS)
                .saturating_mul(3),
            |(chunk_idx, ((q_chunk, k_chunk), v_chunk))| {
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
            },
        );
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

pub(super) fn qkv_matvec_rowmajor_block_parallel_f32_i8(
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

    crate::cpu::for_each_chunk3_mut(
        &mut q_out[..],
        &mut k_out[..],
        &mut v_out[..],
        MATVEC_BLOCK_ROWS,
        k_dim.saturating_mul(MATVEC_BLOCK_ROWS).saturating_mul(3),
        |(block_idx, ((q_chunk, k_chunk), v_chunk))| {
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
        },
    );
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

    if !crate::cpu::should_parallelize(n_rows, k_dim) {
        for i in 0..n_rows {
            let gate_row = &gate_w_rowmajor[i * k_dim..(i + 1) * k_dim];
            let up_row = &up_w_rowmajor[i * k_dim..(i + 1) * k_dim];
            let (g, u) = dot2_unrolled(x, gate_row, up_row);
            let sig = 1.0 / (1.0 + (-g).exp());
            out[i] = (g * sig) * u;
        }
    } else {
        crate::cpu::for_each_chunk_mut(&mut out[..], 1, k_dim.saturating_mul(1), |(i, out_val)| {
            let out_val = &mut out_val[0];
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

    if !crate::cpu::should_parallelize(n_rows, k_dim) {
        for i in 0..n_rows {
            let gate_row = &gate_w_rowmajor[i * k_dim..(i + 1) * k_dim];
            let up_row = &up_w_rowmajor[i * k_dim..(i + 1) * k_dim];
            let (g, u) = dot2_unrolled_f32_bf16(x, gate_row, up_row);
            let sig = 1.0 / (1.0 + (-g).exp());
            out[i] = (g * sig) * u;
        }
    } else {
        crate::cpu::for_each_chunk_mut(
            &mut out[..],
            MIXED_ROW_PAR_CHUNK_ROWS,
            k_dim
                .saturating_mul(MIXED_ROW_PAR_CHUNK_ROWS)
                .saturating_mul(1),
            |(chunk_idx, out_chunk)| {
                let row_start = chunk_idx * MIXED_ROW_PAR_CHUNK_ROWS;
                for (offset, out_val) in out_chunk.iter_mut().enumerate() {
                    let row_idx = row_start + offset;
                    let gate_row = &gate_w_rowmajor[row_idx * k_dim..(row_idx + 1) * k_dim];
                    let up_row = &up_w_rowmajor[row_idx * k_dim..(row_idx + 1) * k_dim];
                    let (g, u) = dot2_unrolled_f32_bf16(x, gate_row, up_row);
                    let sig = 1.0 / (1.0 + (-g).exp());
                    *out_val = (g * sig) * u;
                }
            },
        );
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

    if !crate::cpu::should_parallelize(n_rows, k_dim) {
        for i in 0..n_rows {
            let gate_row = &gate_w_rowmajor[i * k_dim..(i + 1) * k_dim];
            let up_row = &up_w_rowmajor[i * k_dim..(i + 1) * k_dim];
            let (g, u) = dot2_unrolled_f32_f16(x, gate_row, up_row);
            let sig = 1.0 / (1.0 + (-g).exp());
            out[i] = (g * sig) * u;
        }
    } else {
        crate::cpu::for_each_chunk_mut(
            &mut out[..],
            MIXED_ROW_PAR_CHUNK_ROWS,
            k_dim
                .saturating_mul(MIXED_ROW_PAR_CHUNK_ROWS)
                .saturating_mul(1),
            |(chunk_idx, out_chunk)| {
                let row_start = chunk_idx * MIXED_ROW_PAR_CHUNK_ROWS;
                for (offset, out_val) in out_chunk.iter_mut().enumerate() {
                    let row_idx = row_start + offset;
                    let gate_row = &gate_w_rowmajor[row_idx * k_dim..(row_idx + 1) * k_dim];
                    let up_row = &up_w_rowmajor[row_idx * k_dim..(row_idx + 1) * k_dim];
                    let (g, u) = dot2_unrolled_f32_f16(x, gate_row, up_row);
                    let sig = 1.0 / (1.0 + (-g).exp());
                    *out_val = (g * sig) * u;
                }
            },
        );
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

    if !crate::cpu::should_parallelize(n_rows, k_dim) {
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
        crate::cpu::for_each_chunk_mut(
            &mut out[..],
            MATVEC_I8_PAR_CHUNK_ROWS,
            k_dim
                .saturating_mul(MATVEC_I8_PAR_CHUNK_ROWS)
                .saturating_mul(1),
            |(chunk_idx, out_chunk)| {
                let row_start = chunk_idx * MATVEC_I8_PAR_CHUNK_ROWS;
                for (offset, out_val) in out_chunk.iter_mut().enumerate() {
                    let row_idx = row_start + offset;
                    let gate_row = &gate_w_rowmajor[row_idx * k_dim..(row_idx + 1) * k_dim];
                    let up_row = &up_w_rowmajor[row_idx * k_dim..(row_idx + 1) * k_dim];
                    let (g, u) = dot2_unrolled_f32_i8(x, gate_row, gate_scale, up_row, up_scale);
                    let sig = 1.0 / (1.0 + (-g).exp());
                    *out_val = (g * sig) * u;
                }
            },
        );
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

    if !crate::cpu::should_parallelize(n_rows, k_dim) {
        for i in 0..n_rows {
            let gate_row = &gate_w_rowmajor[i * k_dim..(i + 1) * k_dim];
            let up_row = &up_w_rowmajor[i * k_dim..(i + 1) * k_dim];
            let (g, u) = dot2_unrolled_i8_i8(x, x_scale, gate_row, gate_scale, up_row, up_scale);
            let sig = 1.0 / (1.0 + (-g).exp());
            out[i] = (g * sig) * u;
        }
    } else {
        crate::cpu::for_each_chunk_mut(
            &mut out[..],
            MATVEC_I8_PAR_CHUNK_ROWS,
            k_dim
                .saturating_mul(MATVEC_I8_PAR_CHUNK_ROWS)
                .saturating_mul(1),
            |(chunk_idx, out_chunk)| {
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
            },
        );
    }
}

pub(super) fn dual_matvec_silu_mul_rowmajor_block_parallel_f32_i8(
    x: &[f32],
    weights: I8DualBlock<'_>,
    k_dim: usize,
    out: &mut [f32],
) {
    let gate_w_rowmajor = weights.left.values;
    let gate_scale = weights.left.scale;
    let up_w_rowmajor = weights.right.values;
    let up_scale = weights.right.scale;

    crate::cpu::for_each_chunk_mut(
        &mut out[..],
        SILU_I8_BLOCK_ROWS,
        k_dim.saturating_mul(SILU_I8_BLOCK_ROWS).saturating_mul(1),
        |(block_idx, out_chunk)| {
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
        },
    );
}

pub(super) fn dual_matvec_silu_mul_rowmajor_parallel_bf16_f32(
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
pub(super) fn dual_matvec_silu_mul_rowmajor_parallel_bf16_bf16(
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

        if !crate::cpu::should_parallelize(n_rows, k_dim) {
            for (i, out_val) in out.iter_mut().enumerate() {
                *out_val = compute(i);
            }
        } else {
            crate::cpu::for_each_chunk_mut(out, 1, k_dim.saturating_mul(2), |(i, values)| {
                values[0] = compute(i)
            });
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
pub(super) fn dual_matvec_silu_mul_rowmajor_parallel_f16_f16(
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

        if !crate::cpu::should_parallelize(n_rows, k_dim) {
            for (i, out_val) in out.iter_mut().enumerate() {
                *out_val = compute(i);
            }
        } else {
            crate::cpu::for_each_chunk_mut(out, 1, k_dim.saturating_mul(2), |(i, values)| {
                values[0] = compute(i)
            });
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
pub(super) fn dual_matvec_silu_mul_rowmajor_parallel_bf16_i8(
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
pub(super) fn dual_matvec_silu_mul_rowmajor_parallel_i8_f32(
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
pub(super) fn dual_matvec_silu_mul_rowmajor_parallel_i8_bf16(
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

pub(super) fn dual_matvec_silu_mul_rowmajor_parallel_mixed_f32_input(
    x: &[f32],
    gate_w_rowmajor: SliceRef<'_>,
    up_w_rowmajor: SliceRef<'_>,
    n_rows: usize,
    k_dim: usize,
    out: &mut [f32],
) {
    if !crate::cpu::should_parallelize(n_rows, k_dim) {
        for (i, out_val) in out.iter_mut().enumerate().take(n_rows) {
            let g = dot_unrolled_from_slice(x, row_slice(gate_w_rowmajor, i, k_dim));
            let u = dot_unrolled_from_slice(x, row_slice(up_w_rowmajor, i, k_dim));
            let sig = 1.0 / (1.0 + (-g).exp());
            *out_val = (g * sig) * u;
        }
    } else {
        crate::cpu::for_each_chunk_mut(&mut out[..], 1, k_dim.saturating_mul(1), |(i, out_val)| {
            let out_val = &mut out_val[0];
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
