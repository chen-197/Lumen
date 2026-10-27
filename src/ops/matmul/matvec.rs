#[cfg(all(feature = "arm64-int8-kernels", target_arch = "aarch64"))]
use super::dot::dot2_unrolled_i8_i8;
use super::dot::{dot_unrolled_bf16_bf16_scalar, dot_unrolled_f16_f16_scalar, dot_unrolled_i8_i8};
use super::policy::{
    should_use_i8_block_kernel, should_use_i8_matmul_block_kernel,
    should_use_mixed_matvec_block_kernel,
};
use super::*;

#[inline]
pub(super) fn matvec_rowmajor_serial(
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
pub(super) fn matvec_rowmajor_serial_f32_bf16(
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
pub(super) fn matvec_rowmajor_serial_f32_f16(
    x: &[f32],
    w_rowmajor: &[f16],
    n_rows: usize,
    k_dim: usize,
    out: &mut [f32],
) {
    let mut first = 0;
    // Short reductions benefit from sharing the activation loads. Keep the
    // existing long-K kernel, whose cache/register tradeoff differs.
    while k_dim <= 512 && first + 3 <= n_rows {
        let row0 = &w_rowmajor[first * k_dim..(first + 1) * k_dim];
        let row1 = &w_rowmajor[(first + 1) * k_dim..(first + 2) * k_dim];
        let row2 = &w_rowmajor[(first + 2) * k_dim..(first + 3) * k_dim];
        let Some((a, b, c)) = dot3_f32_f16_arch(x, row0, row1, row2) else {
            break;
        };
        out[first..first + 3].copy_from_slice(&[a, b, c]);
        first += 3;
    }
    for i in first..n_rows {
        let row = &w_rowmajor[i * k_dim..(i + 1) * k_dim];
        out[i] = dot_unrolled_f32_f16(x, row);
    }
}

#[inline]
pub(super) fn matvec_rowmajor_serial_f32_i8(
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
        let mut first = 0;
        // Short reductions benefit from sharing the activation loads. Keep the
        // existing long-K kernel, whose cache/register tradeoff differs.
        while k_dim <= 512 && first + 3 <= n_rows {
            let row0 = &w_rowmajor[first * k_dim..(first + 1) * k_dim];
            let row1 = &w_rowmajor[(first + 1) * k_dim..(first + 2) * k_dim];
            let row2 = &w_rowmajor[(first + 2) * k_dim..(first + 3) * k_dim];
            let Some((a, b, c)) = dot3_f32_i8_arch(x, row0, scale, row1, scale, row2, scale) else {
                break;
            };
            out[first..first + 3].copy_from_slice(&[a, b, c]);
            first += 3;
        }
        for i in first..n_rows {
            let row = &w_rowmajor[i * k_dim..(i + 1) * k_dim];
            out[i] = dot_unrolled_f32_i8(x, row, scale);
        }
    }
}

pub(super) fn matvec_rowmajor_rowwise_parallel(
    x: &[f32],
    w_rowmajor: &[f32],
    _n_rows: usize,
    k_dim: usize,
    out: &mut [f32],
) {
    crate::cpu::for_each_chunk_mut(&mut out[..], 1, k_dim.saturating_mul(1), |(i, out_val)| {
        let out_val = &mut out_val[0];
        let row = &w_rowmajor[i * k_dim..(i + 1) * k_dim];
        *out_val = dot_unrolled(x, row);
    });
}

#[inline]
pub(super) fn matvec_rowmajor_rowwise_parallel_f32_bf16(
    x: &[f32],
    w_rowmajor: &[bf16],
    _n_rows: usize,
    k_dim: usize,
    out: &mut [f32],
) {
    crate::cpu::for_each_chunk_mut(
        &mut out[..],
        MIXED_ROW_PAR_CHUNK_ROWS,
        k_dim
            .saturating_mul(MIXED_ROW_PAR_CHUNK_ROWS)
            .saturating_mul(1),
        |(chunk_idx, out_chunk)| {
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
        },
    );
}

#[inline]
pub(super) fn matvec_rowmajor_rowwise_parallel_f32_f16(
    x: &[f32],
    w_rowmajor: &[f16],
    _n_rows: usize,
    k_dim: usize,
    out: &mut [f32],
) {
    crate::cpu::for_each_chunk_mut(
        &mut out[..],
        MIXED_ROW_PAR_CHUNK_ROWS,
        k_dim
            .saturating_mul(MIXED_ROW_PAR_CHUNK_ROWS)
            .saturating_mul(1),
        |(chunk_idx, out_chunk)| {
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
        },
    );
}

#[inline]
pub(super) fn matvec_rowmajor_rowwise_parallel_f32_i8(
    x: &[f32],
    w_rowmajor: &[i8],
    scale: f32,
    _n_rows: usize,
    k_dim: usize,
    out: &mut [f32],
) {
    crate::cpu::for_each_chunk_mut(
        &mut out[..],
        MATVEC_I8_PAR_CHUNK_ROWS,
        k_dim
            .saturating_mul(MATVEC_I8_PAR_CHUNK_ROWS)
            .saturating_mul(1),
        |(chunk_idx, out_chunk)| {
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
        },
    );
}

#[inline]
pub(super) fn matvec_rowmajor_block_parallel(
    x: &[f32],
    w_rowmajor: &[f32],
    _n_rows: usize,
    k_dim: usize,
    out: &mut [f32],
) {
    crate::cpu::for_each_chunk_mut(
        &mut out[..],
        MATVEC_BLOCK_ROWS,
        k_dim.saturating_mul(MATVEC_BLOCK_ROWS).saturating_mul(1),
        |(block_idx, out_chunk)| {
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
        },
    );
}

#[inline]
pub(super) fn matvec_rowmajor_block_parallel_f32_bf16(
    x: &[f32],
    w_rowmajor: &[bf16],
    _n_rows: usize,
    k_dim: usize,
    out: &mut [f32],
) {
    crate::cpu::for_each_chunk_mut(
        &mut out[..],
        MATVEC_BLOCK_ROWS,
        k_dim.saturating_mul(MATVEC_BLOCK_ROWS).saturating_mul(1),
        |(block_idx, out_chunk)| {
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
        },
    );
}

pub(super) fn matvec_rowmajor_block_parallel_f32_i8(
    x: &[f32],
    w_rowmajor: &[i8],
    scale: f32,
    _n_rows: usize,
    k_dim: usize,
    out: &mut [f32],
) {
    crate::cpu::for_each_chunk_mut(
        &mut out[..],
        MATVEC_BLOCK_ROWS,
        k_dim.saturating_mul(MATVEC_BLOCK_ROWS).saturating_mul(1),
        |(block_idx, out_chunk)| {
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
        },
    );
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

    if !crate::cpu::should_parallelize(n_rows, k_dim) {
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

    if !crate::cpu::should_parallelize(n_rows, k_dim) {
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

    if !crate::cpu::should_parallelize(n_rows, k_dim) {
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

    if !crate::cpu::should_parallelize(n_rows, k_dim) {
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

    if !crate::cpu::should_parallelize(n_rows, k_dim) {
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
        if !crate::cpu::should_parallelize(n_rows, k_dim) {
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
            crate::cpu::for_each_chunk_mut(
                &mut out[..],
                1,
                k_dim.saturating_mul(1),
                |(i, out_val)| {
                    let out_val = &mut out_val[0];
                    let row = &w_rowmajor[i * k_dim..(i + 1) * k_dim];
                    *out_val = dot_bf16_bf16_arch(x, row)
                        .unwrap_or_else(|| dot_unrolled_bf16_bf16_scalar(x, row));
                },
            );
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
        if !crate::cpu::should_parallelize(n_rows, k_dim) {
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
            crate::cpu::for_each_chunk_mut(
                &mut out[..],
                1,
                k_dim.saturating_mul(1),
                |(i, out_val)| {
                    let out_val = &mut out_val[0];
                    let row = &w_rowmajor[i * k_dim..(i + 1) * k_dim];
                    *out_val = dot_f16_f16_arch(x, row)
                        .unwrap_or_else(|| dot_unrolled_f16_f16_scalar(x, row));
                },
            );
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

    if !crate::cpu::should_parallelize(n_rows, k_dim) {
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
        crate::cpu::for_each_chunk_mut(
            &mut out[..],
            MATVEC_I8_PAR_CHUNK_ROWS,
            k_dim
                .saturating_mul(MATVEC_I8_PAR_CHUNK_ROWS)
                .saturating_mul(1),
            |(chunk_idx, out_chunk)| {
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
            },
        );
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
