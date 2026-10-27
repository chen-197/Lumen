use super::dot::{dot_unrolled_bf16_bf16_scalar, dot_unrolled_f16_f16_scalar, dot_unrolled_i8_i8};
use super::policy::should_use_argmax_block_kernel;
use super::*;

// Keep complete rows contiguous for the native dot kernels. Grouping rows amortizes
// scheduling and reductions without materializing the vocabulary-sized logits.
#[inline]
pub(super) fn argmax_native_row_blocks(
    n_rows: usize,
    k_dim: usize,
    score: impl Fn(usize) -> f32 + Send + Sync,
) -> usize {
    crate::cpu::map_reduce(
        n_rows.div_ceil(ARGMAX_BLOCK_ROWS),
        k_dim.saturating_mul(ARGMAX_BLOCK_ROWS),
        |block| {
            let start = block * ARGMAX_BLOCK_ROWS;
            let end = (start + ARGMAX_BLOCK_ROWS).min(n_rows);
            let mut best = (0, f32::NEG_INFINITY);
            for row in start..end {
                let value = score(row);
                if value > best.1 {
                    best = (row, value);
                }
            }
            best
        },
        || (0, f32::NEG_INFINITY),
        |a, b| {
            if b.1 > a.1 || (b.1 == a.1 && b.0 < a.0) {
                b
            } else {
                a
            }
        },
    )
    .0
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
        if k_dim >= ARGMAX_NATIVE_MIN_K
            && crate::ops::fp_kernels::active_float_backend()
                != crate::ops::fp_kernels::FloatKernelBackend::Portable
        {
            return argmax_native_row_blocks(n_rows, k_dim, |row| {
                dot_unrolled(x, &w_rowmajor[row * k_dim..(row + 1) * k_dim])
            });
        }
        let n_blocks = n_rows.div_ceil(ARGMAX_BLOCK_ROWS);
        return crate::cpu::map_reduce(
            n_blocks,
            k_dim.saturating_mul(ARGMAX_BLOCK_ROWS),
            |block_idx| {
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
            },
            || (0usize, f32::NEG_INFINITY),
            |a, b| if a.1 >= b.1 { a } else { b },
        )
        .0;
    }

    if !crate::cpu::should_parallelize(n_rows, k_dim) {
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
        crate::cpu::map_reduce(
            n_rows,
            k_dim,
            |i| {
                let row = &w_rowmajor[i * k_dim..(i + 1) * k_dim];
                (i, dot_unrolled(x, row))
            },
            || (0usize, f32::NEG_INFINITY),
            |a, b| if a.1 >= b.1 { a } else { b },
        )
        .0
    }
}

#[inline]
pub(super) fn matvec_argmax_rowmajor_parallel_f32_bf16(
    x: &[f32],
    w_rowmajor: &[bf16],
    n_rows: usize,
    k_dim: usize,
) -> usize {
    assert_eq!(x.len(), k_dim, "x len / k_dim mismatch");
    assert_eq!(w_rowmajor.len(), n_rows * k_dim, "weight size mismatch");

    if !crate::cpu::should_parallelize(n_rows, k_dim) {
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
        crate::cpu::map_reduce(
            n_rows,
            k_dim,
            |i| {
                let row = &w_rowmajor[i * k_dim..(i + 1) * k_dim];
                (i, dot_unrolled_f32_bf16(x, row))
            },
            || (0usize, f32::NEG_INFINITY),
            |a, b| if a.1 >= b.1 { a } else { b },
        )
        .0
    }
}

#[inline]
pub(super) fn matvec_argmax_rowmajor_parallel_f32_f16(
    x: &[f32],
    w_rowmajor: &[f16],
    n_rows: usize,
    k_dim: usize,
) -> usize {
    assert_eq!(x.len(), k_dim, "x len / k_dim mismatch");
    assert_eq!(w_rowmajor.len(), n_rows * k_dim, "weight size mismatch");

    if !crate::cpu::should_parallelize(n_rows, k_dim) {
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
        crate::cpu::map_reduce(
            n_rows,
            k_dim,
            |i| {
                let row = &w_rowmajor[i * k_dim..(i + 1) * k_dim];
                (i, dot_unrolled_f32_f16(x, row))
            },
            || (0usize, f32::NEG_INFINITY),
            |a, b| if a.1 >= b.1 { a } else { b },
        )
        .0
    }
}

#[inline]
pub(super) fn matvec_argmax_rowmajor_parallel_f32_i8(
    x: &[f32],
    w_rowmajor: &[i8],
    scale: f32,
    n_rows: usize,
    k_dim: usize,
) -> usize {
    assert_eq!(x.len(), k_dim, "x len / k_dim mismatch");
    assert_eq!(w_rowmajor.len(), n_rows * k_dim, "weight size mismatch");

    if n_rows >= MATVEC_BLOCK_THRESHOLD {
        if k_dim >= ARGMAX_NATIVE_MIN_K
            && crate::ops::int8_kernels::active_int8_backend()
                != crate::ops::int8_kernels::Int8KernelBackend::Portable
        {
            return argmax_native_row_blocks(n_rows, k_dim, |row| {
                dot_unrolled_f32_i8(x, &w_rowmajor[row * k_dim..(row + 1) * k_dim], scale)
            });
        }
        let n_blocks = n_rows.div_ceil(ARGMAX_BLOCK_ROWS);
        return crate::cpu::map_reduce(
            n_blocks,
            k_dim.saturating_mul(ARGMAX_BLOCK_ROWS),
            |block_idx| {
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
            },
            || (0usize, f32::NEG_INFINITY),
            |a, b| if a.1 >= b.1 { a } else { b },
        )
        .0;
    }

    if !crate::cpu::should_parallelize(n_rows, k_dim) {
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
        crate::cpu::map_reduce(
            n_rows,
            k_dim,
            |i| {
                let row = &w_rowmajor[i * k_dim..(i + 1) * k_dim];
                (i, dot_unrolled_f32_i8(x, row, scale))
            },
            || (0usize, f32::NEG_INFINITY),
            |a, b| if a.1 >= b.1 { a } else { b },
        )
        .0
    }
}

#[inline]
pub(super) fn matvec_argmax_rowmajor_parallel_bf16_f32(
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
pub(super) fn matvec_argmax_rowmajor_parallel_bf16_bf16(
    x: &[bf16],
    w_rowmajor: &[bf16],
    n_rows: usize,
    k_dim: usize,
) -> usize {
    assert_eq!(x.len(), k_dim, "x len / k_dim mismatch");
    assert_eq!(w_rowmajor.len(), n_rows * k_dim, "weight size mismatch");

    if crate::arch::x86_avx512_bf16_kernel_runtime_available() {
        if !crate::cpu::should_parallelize(n_rows, k_dim) {
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

        return crate::cpu::map_reduce(
            n_rows,
            k_dim,
            |i| {
                let row = &w_rowmajor[i * k_dim..(i + 1) * k_dim];
                (
                    i,
                    dot_bf16_bf16_arch(x, row)
                        .unwrap_or_else(|| dot_unrolled_bf16_bf16_scalar(x, row)),
                )
            },
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
pub(super) fn matvec_argmax_rowmajor_parallel_f16_f16(
    x: &[f16],
    w_rowmajor: &[f16],
    n_rows: usize,
    k_dim: usize,
) -> usize {
    assert_eq!(x.len(), k_dim, "x len / k_dim mismatch");
    assert_eq!(w_rowmajor.len(), n_rows * k_dim, "weight size mismatch");

    if crate::arch::x86_avx512_fp16_kernel_runtime_available() {
        if !crate::cpu::should_parallelize(n_rows, k_dim) {
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

        return crate::cpu::map_reduce(
            n_rows,
            k_dim,
            |i| {
                let row = &w_rowmajor[i * k_dim..(i + 1) * k_dim];
                (
                    i,
                    dot_f16_f16_arch(x, row).unwrap_or_else(|| dot_unrolled_f16_f16_scalar(x, row)),
                )
            },
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
pub(super) fn matvec_argmax_rowmajor_parallel_bf16_i8(
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
pub(super) fn matvec_argmax_rowmajor_parallel_i8_f32(
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
pub(super) fn matvec_argmax_rowmajor_parallel_i8_bf16(
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
pub(super) fn matvec_argmax_rowmajor_parallel_i8_i8(
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

    if !crate::cpu::should_parallelize(n_rows, k_dim) {
        let mut best = (0usize, f32::NEG_INFINITY);
        for i in 0..n_rows {
            let row = &w_rowmajor[i * k_dim..(i + 1) * k_dim];
            let score = dot_unrolled_i8_i8(x, x_scale, row, w_scale);
            choose_best(&mut best, i, score);
        }
        best.0
    } else {
        crate::cpu::map_reduce(
            n_rows,
            k_dim,
            |i| {
                let row = &w_rowmajor[i * k_dim..(i + 1) * k_dim];
                (i, dot_unrolled_i8_i8(x, x_scale, row, w_scale))
            },
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
