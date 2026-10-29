//! CPU matrix scheduling and temporary native low-precision weight panels.
use super::matmul::SliceRef;
use half::{bf16, f16};
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
    static BF16_F32_PANEL: RefCell<Vec<bf16>> = const { RefCell::new(Vec::new()) };
    static BF16_PANEL: RefCell<Vec<u32>> = const { RefCell::new(Vec::new()) };
}

/// Each packed lane contains a K pair for one output channel. Reuse that
/// weight vector across several input rows instead of reducing each dot alone.
pub(super) fn bf16_panel(
    input_pairs: &[u32],
    weights: &[bf16],
    k: usize,
    n: usize,
    first: usize,
    output: &mut [f32],
) {
    assert!(crate::arch::x86_avx512_bf16_kernel_runtime_available());
    let pairs = k.div_ceil(2);
    assert!(k > 0 && first < n);
    assert_eq!(weights.len(), n * k);
    assert_eq!(input_pairs.len() % pairs, 0);
    let m = input_pairs.len() / pairs;
    assert_eq!(output.len(), m * CHANNELS);
    BF16_PANEL.with(|slot| {
        let mut packed = slot.borrow_mut();
        packed.resize(pairs * CHANNELS, 0);
        pack_bf16_panel(weights, k, n, first, &mut packed);
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        for token in (0..m).step_by(8) {
            let rows = (m - token).min(8);
            let input = &input_pairs[token * pairs..(token + rows) * pairs];
            let output = &mut output[token * CHANNELS..(token + rows) * CHANNELS];
            macro_rules! compute {
                ($rows:literal) => {
                    // Runtime ISA check above; all packed/input/output extents
                    // include their padded channel and odd-K tails.
                    unsafe { bf16_kernel512::<$rows>(input, &packed, pairs, output) }
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
    });
}

// Transpose pairs as integer bits: packing must not round, flush subnormals,
// or change NaN payloads. A 4x4 tile reads contiguous pairs from four channels.
fn pack_bf16_panel(weights: &[bf16], k: usize, n: usize, first: usize, packed: &mut [u32]) {
    assert!(k > 0 && first < n);
    assert_eq!(weights.len(), n * k);
    assert_eq!(packed.len(), k.div_ceil(2) * CHANNELS);
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if std::is_x86_feature_detected!("sse2") {
        // The extents above cover every complete 4-channel/4-pair tile;
        // the helper handles incomplete tiles and odd K with scalar loads.
        unsafe { pack_bf16_panel_sse2(weights, k, n, first, packed) };
        return;
    }
    for pair in 0..k.div_ceil(2) {
        for channel in 0..CHANNELS {
            packed[pair * CHANNELS + channel] =
                bf16_weight_pair(weights, k, n, first + channel, pair);
        }
    }
}

fn bf16_weight_pair(weights: &[bf16], k: usize, n: usize, channel: usize, pair: usize) -> u32 {
    if channel >= n {
        return 0;
    }
    let offset = channel * k + pair * 2;
    let low = weights[offset].to_bits() as u32;
    let high = if pair * 2 + 1 < k {
        weights[offset + 1].to_bits() as u32
    } else {
        0
    };
    low | (high << 16)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn pack_bf16_panel_sse2(
    weights: &[bf16],
    k: usize,
    n: usize,
    first: usize,
    packed: &mut [u32],
) {
    let pairs = k.div_ceil(2);
    let valid = (n - first).min(CHANNELS);
    // Only complete pairs may be loaded; an odd final element is zero-padded.
    let full_pairs = (k / 2) / 4 * 4;
    for channel in (0..CHANNELS).step_by(4) {
        let end = if channel + 4 <= valid { full_pairs } else { 0 };
        for pair in (0..end).step_by(4) {
            unsafe {
                let source = weights.as_ptr().add((first + channel) * k + pair * 2);
                let a = _mm_loadu_si128(source.cast());
                let b = _mm_loadu_si128(source.add(k).cast());
                let c = _mm_loadu_si128(source.add(k * 2).cast());
                let d = _mm_loadu_si128(source.add(k * 3).cast());
                let ab_lo = _mm_unpacklo_epi32(a, b);
                let ab_hi = _mm_unpackhi_epi32(a, b);
                let cd_lo = _mm_unpacklo_epi32(c, d);
                let cd_hi = _mm_unpackhi_epi32(c, d);
                let destination = packed.as_mut_ptr().add(pair * CHANNELS + channel);
                _mm_storeu_si128(destination.cast(), _mm_unpacklo_epi64(ab_lo, cd_lo));
                _mm_storeu_si128(
                    destination.add(CHANNELS).cast(),
                    _mm_unpackhi_epi64(ab_lo, cd_lo),
                );
                _mm_storeu_si128(
                    destination.add(CHANNELS * 2).cast(),
                    _mm_unpacklo_epi64(ab_hi, cd_hi),
                );
                _mm_storeu_si128(
                    destination.add(CHANNELS * 3).cast(),
                    _mm_unpackhi_epi64(ab_hi, cd_hi),
                );
            }
        }
        // Overwrite padding on every use, including when TLS scratch shrinks.
        for pair in end..pairs {
            for c in channel..channel + 4 {
                packed[pair * CHANNELS + c] = bf16_weight_pair(weights, k, n, first + c, pair);
            }
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx512bf16")]
unsafe fn bf16_kernel512<const M: usize>(
    input: &[u32],
    packed: &[u32],
    pairs: usize,
    output: &mut [f32],
) {
    let mut totals = [_mm512_setzero_ps(); 8];
    for block in (0..pairs).step_by(64) {
        let mut sums = [_mm512_setzero_ps(); 8];
        for pair in block..(block + 64).min(pairs) {
            let weight: __m512bh = unsafe {
                std::mem::transmute::<__m512i, __m512bh>(_mm512_loadu_si512(
                    packed.as_ptr().add(pair * CHANNELS).cast(),
                ))
            };
            macro_rules! row {
                ($row:literal) => {
                    if M > $row {
                        let bits = unsafe { *input.get_unchecked($row * pairs + pair) };
                        let x: __m512bh = unsafe {
                            std::mem::transmute::<__m512i, __m512bh>(_mm512_set1_epi32(bits as i32))
                        };
                        sums[$row] = _mm512_dpbf16_ps(sums[$row], weight, x);
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
            totals[row] = _mm512_add_ps(totals[row], sums[row]);
        }
    }
    for (row, sum) in totals.iter().enumerate().take(M) {
        unsafe { _mm512_storeu_ps(output.as_mut_ptr().add(row * CHANNELS), *sum) };
    }
}

pub(super) fn available(weights: SliceRef<'_>) -> bool {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    match weights {
        SliceRef::F16(_) => crate::arch::x86_fp16_kernel_runtime_available(),
        SliceRef::BF16(_) => crate::arch::x86_fp_kernel_runtime_available(),
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
            run::<f16, true, false>(input, w, 1., k, n, first, output, &mut packed, wide);
        }),
        SliceRef::BF16(w) => BF16_F32_PANEL.with(|slot| {
            let mut packed = slot.borrow_mut();
            run::<bf16, true, true>(input, w, 1., k, n, first, output, &mut packed, wide);
        }),
        SliceRef::I8(w, scale) => INT_PANEL.with(|slot| {
            let mut packed = slot.borrow_mut();
            run::<i8, false, false>(input, w, scale, k, n, first, output, &mut packed, wide);
        }),
        _ => unreachable!(),
    }
}

fn run<T: Copy + Default, const HALF: bool, const BFLOAT: bool>(
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
                                kernel512::<$rows, HALF, BFLOAT>(
                                    input,
                                    packed.as_ptr().cast(),
                                    k,
                                    col,
                                    scale,
                                    output,
                                );
                            } else if BFLOAT {
                                kernel256_bfloat::<$rows, true, true>(
                                    input,
                                    packed.as_ptr().cast(),
                                    k,
                                    col,
                                    scale,
                                    output,
                                );
                            } else {
                                kernel256::<$rows, HALF, BFLOAT>(
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
    ($name:ident, $features:literal, $lanes:literal, $zero:ident, $broadcast:ident, $fma:ident, $add:ident, $mul:ident, $store:ident, $load_half:expr, $load_bfloat:expr, $load_int:expr) => {
        #[target_feature(enable = $features)]
        unsafe fn $name<const M: usize, const HALF: bool, const BFLOAT: bool>(
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
                        if BFLOAT {
                            unsafe { ($load_bfloat)(packed.add(kk * $lanes * 2)) }
                        } else {
                            unsafe { ($load_half)(packed.add(kk * $lanes * 2)) }
                        }
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
    |p: *const u8| _mm512_castsi512_ps(_mm512_slli_epi32::<16>(_mm512_cvtepu16_epi32(
        _mm256_loadu_si256(p.cast())
    ))),
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
    |p: *const u8| _mm256_castsi256_ps(_mm256_slli_epi32::<16>(_mm256_cvtepu16_epi32(
        _mm_loadu_si128(p.cast())
    ))),
    |p: *const u8| _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(p.cast())))
);

// BF16 expansion needs AVX2/FMA but does not require F16C.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
define_kernel!(
    kernel256_bfloat,
    "avx2,fma",
    8,
    _mm256_setzero_ps,
    _mm256_set1_ps,
    _mm256_fmadd_ps,
    _mm256_add_ps,
    _mm256_mul_ps,
    _mm256_storeu_ps,
    |p: *const u8| _mm256_castsi256_ps(_mm256_slli_epi32::<16>(_mm256_cvtepu16_epi32(
        _mm_loadu_si128(p.cast())
    ))),
    |p: *const u8| _mm256_castsi256_ps(_mm256_slli_epi32::<16>(_mm256_cvtepu16_epi32(
        _mm_loadu_si128(p.cast())
    ))),
    |p: *const u8| _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(p.cast())))
);

#[cfg(all(test, any(target_arch = "x86", target_arch = "x86_64")))]
#[path = "../../tests/unit/ops/cpu_gemm/tests.rs"]
mod tests;
