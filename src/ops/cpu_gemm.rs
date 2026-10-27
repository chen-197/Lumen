//! CPU matrix scheduling and temporary native low-precision weight panels.
use super::matmul::SliceRef;
use half::f16;
use std::cell::RefCell;

/// Schedule independent rows/heads without changing the reduction dimension.
/// Views retain their strides; no complete weight transpose is materialized.
pub(super) fn f32_batched_matmul(
    a: ndarray::ArrayView4<'_, f32>,
    b: ndarray::ArrayView4<'_, f32>,
    mut out: ndarray::ArrayViewMut4<'_, f32>,
) {
    use ndarray::linalg::general_mat_mul;
    use ndarray::{Axis, s};
    let (batches, heads, rows, k) = a.dim();
    let n = b.shape()[3];
    assert_eq!(b.dim(), (batches, heads, k, n));
    assert_eq!(out.dim(), (batches, heads, rows, n));
    let head_jobs = batches.saturating_mul(heads);
    let work = rows.saturating_mul(k).saturating_mul(n);
    // GEMM packing and worker wakeup need a much coarser grain than a dot row.
    // This portable work heuristic only reduces the caller's available budget.
    const MIN_GEMM_WORK: usize = 4 * 1024 * 1024;
    let workers = crate::cpu::parallelism().min(
        head_jobs
            .saturating_mul(work)
            .div_euclid(MIN_GEMM_WORK)
            .max(1),
    );
    if workers == 1 {
        for bb in 0..batches {
            for hh in 0..heads {
                general_mat_mul(
                    1.0,
                    &a.slice(s![bb, hh, .., ..]),
                    &b.slice(s![bb, hh, .., ..]),
                    0.0,
                    &mut out.slice_mut(s![bb, hh, .., ..]),
                );
            }
        }
        return;
    }
    // Keep row/column tiles large enough to amortize packing. Short-wide
    // matrices split columns instead of forcing undersized row tiles.
    let desired = workers.div_ceil(head_jobs.max(1));
    let column_blocks = if rows / 16 < desired && n >= desired.saturating_mul(128) {
        desired
    } else {
        1
    };
    let blocks = if column_blocks > 1 {
        1
    } else {
        desired.min((rows / 16).max(1)).max(1)
    };
    let rows_per_job = rows.div_ceil(blocks).max(1);
    let cols_per_job = n.div_ceil(column_blocks).max(1);
    let mut jobs = Vec::new();
    let mut batches_out = out.view_mut();
    for bb in 0..batches {
        let (batch, tail) = batches_out.split_at(Axis(0), 1);
        batches_out = tail;
        let mut heads_out = batch.index_axis_move(Axis(0), 0);
        for hh in 0..heads {
            let (head, tail) = heads_out.split_at(Axis(0), 1);
            heads_out = tail;
            let mut remaining = head.index_axis_move(Axis(0), 0);
            for first in (0..rows).step_by(rows_per_job) {
                let count = (rows - first).min(rows_per_job);
                let (part, tail) = remaining.split_at(Axis(0), count);
                remaining = tail;
                let mut columns = part;
                for col in (0..n).step_by(cols_per_job) {
                    let count = (n - col).min(cols_per_job);
                    let (part, tail) = columns.split_at(Axis(1), count);
                    columns = tail;
                    jobs.push((bb, hh, first, col, part));
                }
            }
        }
    }
    let per_group = jobs.len().div_ceil(workers).max(1);
    crate::cpu::for_each_chunk_mut(
        &mut jobs,
        per_group,
        rows_per_job
            .saturating_mul(k)
            .saturating_mul(cols_per_job)
            .saturating_mul(per_group),
        |(_, group)| {
            for job in group {
                let (bb, hh, first, col, output) = job;
                let (bb, hh, first, col) = (*bb, *hh, *first, *col);
                let count = output.nrows();
                let cols = output.ncols();
                general_mat_mul(
                    1.0,
                    &a.slice(s![bb, hh, first..first + count, ..]),
                    &b.slice(s![bb, hh, .., col..col + cols]),
                    0.0,
                    output,
                );
            }
        },
    );
}

pub(super) fn f32_matmul(
    a: ndarray::ArrayView2<'_, f32>,
    b: ndarray::ArrayView2<'_, f32>,
    out: ndarray::ArrayViewMut2<'_, f32>,
) {
    use ndarray::Axis;
    f32_batched_matmul(
        a.insert_axis(Axis(0)).insert_axis(Axis(0)),
        b.insert_axis(Axis(0)).insert_axis(Axis(0)),
        out.insert_axis(Axis(0)).insert_axis(Axis(0)),
    );
}

pub(super) const CHANNELS: usize = 16;

thread_local! {
    static HALF_PANEL: RefCell<Vec<f16>> = const { RefCell::new(Vec::new()) };
    static INT_PANEL: RefCell<Vec<i8>> = const { RefCell::new(Vec::new()) };
}

pub(super) fn available(weights: SliceRef<'_>) -> bool {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    match weights {
        SliceRef::F16(_) => crate::arch::x86_fp16_kernel_runtime_available(),
        SliceRef::I8(_, _) => {
            crate::arch::x86_i8_kernel_runtime_available()
                && (std::is_x86_feature_detected!("avx512f")
                    || std::is_x86_feature_detected!("f16c"))
        }
        _ => false,
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        let _ = weights;
        false
    }
}

pub(super) fn panel(
    input: &[f32],
    weights: SliceRef<'_>,
    k: usize,
    n: usize,
    first: usize,
    output: &mut [f32],
) {
    assert!(available(weights));
    assert_eq!(input.len() % k, 0);
    assert_eq!(output.len(), input.len() / k * CHANNELS);
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let wide = std::is_x86_feature_detected!("avx512f");
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    let wide = false;
    match weights {
        SliceRef::F16(w) => HALF_PANEL.with(|slot| {
            let mut packed = slot.borrow_mut();
            run::<f16, true>(input, w, 1., k, n, first, output, &mut packed, wide);
        }),
        SliceRef::I8(w, scale) => INT_PANEL.with(|slot| {
            let mut packed = slot.borrow_mut();
            run::<i8, false>(input, w, scale, k, n, first, output, &mut packed, wide);
        }),
        _ => unreachable!(),
    }
}

fn run<T: Copy + Default, const HALF: bool>(
    input: &[f32],
    weights: &[T],
    scale: f32,
    k: usize,
    n: usize,
    first: usize,
    output: &mut [f32],
    packed: &mut Vec<T>,
    wide: bool,
) {
    assert_eq!(weights.len(), n * k);
    assert_eq!(std::mem::size_of::<T>(), if HALF { 2 } else { 1 });
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        let lanes = if wide { 16 } else { 8 };
        packed.resize(k * lanes, T::default());
        for col in (0..CHANNELS).step_by(lanes) {
            let valid = n.saturating_sub(first + col).min(lanes);
            // The small transpose stays in L1 while consuming contiguous K.
            for kb in (0..k).step_by(32) {
                for channel in 0..lanes {
                    for kk in kb..(kb + 32).min(k) {
                        packed[kk * lanes + channel] = if channel < valid {
                            weights[(first + col + channel) * k + kk]
                        } else {
                            T::default()
                        };
                    }
                }
            }
            let m = input.len() / k;
            for token in (0..m).step_by(8) {
                let rows = (m - token).min(8);
                let input = &input[token * k..(token + rows) * k];
                let output = &mut output[token * CHANNELS..(token + rows) * CHANNELS];
                macro_rules! compute {
                    ($rows:literal) => {
                        // available() checked the ISA; packed has K*lanes
                        // native elements, input/output lengths checked above.
                        unsafe {
                            if wide {
                                kernel512::<$rows, HALF>(
                                    input,
                                    packed.as_ptr().cast(),
                                    k,
                                    col,
                                    scale,
                                    output,
                                );
                            } else {
                                kernel256::<$rows, HALF>(
                                    input,
                                    packed.as_ptr().cast(),
                                    k,
                                    col,
                                    scale,
                                    output,
                                );
                            }
                        }
                    };
                }
                match rows {
                    1 => compute!(1),
                    2 => compute!(2),
                    3 => compute!(3),
                    4 => compute!(4),
                    5 => compute!(5),
                    6 => compute!(6),
                    7 => compute!(7),
                    8 => compute!(8),
                    _ => unreachable!(),
                }
            }
        }
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        let _ = (input, scale, first, output, packed, wide);
        unreachable!();
    }
}

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
macro_rules! define_kernel {
    ($name:ident, $features:literal, $lanes:literal, $zero:ident, $broadcast:ident, $fma:ident, $add:ident, $mul:ident, $store:ident, $load_half:expr, $load_int:expr) => {
        #[target_feature(enable = $features)]
        unsafe fn $name<const M: usize, const HALF: bool>(
            input: &[f32],
            packed: *const u8,
            k: usize,
            col: usize,
            scale: f32,
            output: &mut [f32],
        ) {
            let mut totals = [$zero(); 8];
            // Bound sequential F32 rounding error without widening the data.
            for kb in (0..k).step_by(128) {
                let mut sums = [$zero(); 8];
                for kk in kb..(kb + 128).min(k) {
                    let weight = if HALF {
                        unsafe { ($load_half)(packed.add(kk * $lanes * 2)) }
                    } else {
                        unsafe { ($load_int)(packed.add(kk * $lanes)) }
                    };
                    macro_rules! row {
                        ($row:literal) => {
                            if M > $row {
                                let x = $broadcast(unsafe { *input.get_unchecked($row * k + kk) });
                                sums[$row] = $fma(weight, x, sums[$row]);
                            }
                        };
                    }
                    row!(0);
                    row!(1);
                    row!(2);
                    row!(3);
                    row!(4);
                    row!(5);
                    row!(6);
                    row!(7);
                }
                for row in 0..M {
                    totals[row] = $add(totals[row], sums[row]);
                }
            }
            for (row, sum) in totals.iter().enumerate().take(M) {
                let sum = if HALF {
                    *sum
                } else {
                    $mul(*sum, $broadcast(scale))
                };
                unsafe { $store(output.as_mut_ptr().add(row * CHANNELS + col), sum) };
            }
        }
    };
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
define_kernel!(
    kernel512,
    "avx512f",
    16,
    _mm512_setzero_ps,
    _mm512_set1_ps,
    _mm512_fmadd_ps,
    _mm512_add_ps,
    _mm512_mul_ps,
    _mm512_storeu_ps,
    |p: *const u8| _mm512_cvtph_ps(_mm256_loadu_si256(p.cast())),
    |p: *const u8| _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128(p.cast())))
);

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
define_kernel!(
    kernel256,
    "avx2,fma,f16c",
    8,
    _mm256_setzero_ps,
    _mm256_set1_ps,
    _mm256_fmadd_ps,
    _mm256_add_ps,
    _mm256_mul_ps,
    _mm256_storeu_ps,
    |p: *const u8| _mm256_cvtph_ps(_mm_loadu_si128(p.cast())),
    |p: *const u8| _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(p.cast())))
);

#[cfg(all(test, any(target_arch = "x86", target_arch = "x86_64")))]
#[path = "../../tests/unit/ops/cpu_gemm/tests.rs"]
mod tests;
