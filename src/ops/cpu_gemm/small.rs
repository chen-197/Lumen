//! Short-row standard products: stream consecutive RHS columns without packing.
use super::{SliceRef, bf16, f16};
use ndarray::Array2;

pub(in crate::ops) fn available(weights: SliceRef<'_>) -> bool {
    match weights {
        SliceRef::F32(_) | SliceRef::BF16(_) => crate::arch::x86_fp_kernel_runtime_available(),
        SliceRef::F16(_) => crate::arch::x86_fp16_kernel_runtime_available(),
        SliceRef::I8(_, _) => crate::arch::x86_i8_kernel_runtime_available(),
    }
}

pub(in crate::ops) fn matmul(
    input: &[f32],
    weights: SliceRef<'_>,
    m: usize,
    k: usize,
    n: usize,
) -> Option<Array2<f32>> {
    if !available(weights) {
        return None;
    }
    assert!(k > 0 && n > 0);
    assert_eq!(input.len(), m * k);
    Some(match weights {
        SliceRef::F32(w) => compute(input, w, 1., m, k, n),
        SliceRef::F16(w) => compute(input, w, 1., m, k, n),
        SliceRef::BF16(w) => compute(input, w, 1., m, k, n),
        SliceRef::I8(w, scale) => compute(input, w, scale, m, k, n),
    })
}

trait Weight: Copy + Send + Sync {
    const KIND: usize;
    fn f32(self) -> f32;
}
impl Weight for f32 {
    const KIND: usize = 0;
    fn f32(self) -> f32 {
        self
    }
}
impl Weight for f16 {
    const KIND: usize = 1;
    fn f32(self) -> f32 {
        self.to_f32()
    }
}
impl Weight for bf16 {
    const KIND: usize = 2;
    fn f32(self) -> f32 {
        self.to_f32()
    }
}
impl Weight for i8 {
    const KIND: usize = 3;
    fn f32(self) -> f32 {
        self as f32
    }
}

fn compute<T: Weight>(
    input: &[f32],
    weights: &[T],
    scale: f32,
    m: usize,
    k: usize,
    n: usize,
) -> Array2<f32> {
    assert_eq!(weights.len(), k * n);
    let mut result = Array2::zeros((m, n));
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let wide = std::is_x86_feature_detected!("avx512f");
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    let wide = false;
    // Independent output spans respect the caller's pool and task budget.
    // Spans may cross a logical row; split those before entering the kernel.
    let columns = if n <= 256 { n } else { 128 };
    let compute = |(job, out): (usize, &mut [f32])| {
        let mut offset = 0;
        while offset < out.len() {
            let index = job * columns + offset;
            let (row, col) = (index / n, index % n);
            let count = (n - col).min(out.len() - offset);
            compute_row(
                &input[row * k..(row + 1) * k],
                weights,
                n,
                col,
                scale,
                &mut out[offset..offset + count],
                wide,
            );
            offset += count;
        }
    };
    // Worker wakeup costs dominate short cache-resident products. This work
    // estimate only reduces the caller's budget; it assumes no CPU/cache model.
    if m.saturating_mul(k).saturating_mul(n) < 1024 * 1024 {
        result
            .as_slice_mut()
            .unwrap()
            .chunks_mut(columns)
            .enumerate()
            .for_each(compute);
    } else {
        crate::cpu::for_each_chunk_mut(
            result.as_slice_mut().unwrap(),
            columns,
            k.saturating_mul(columns),
            compute,
        );
    }
    result
}

fn compute_row<T: Weight>(
    input: &[f32],
    weights: &[T],
    n: usize,
    first: usize,
    scale: f32,
    out: &mut [f32],
    wide: bool,
) {
    assert_eq!(weights.len(), input.len() * n);
    assert!(first <= n && out.len() <= n - first);
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let vector_end = {
        let lanes = if wide { 16 } else { 8 };
        let vector_end = out.len() / lanes * lanes;
        // The typed slices and checks above cover every complete SIMD load.
        // The caller checks AVX2/FMA, F16C where needed, and AVX-512F for wide.
        unsafe {
            macro_rules! run {
                ($kind:literal) => {
                    if wide {
                        wide_row::<$kind>(
                            input,
                            weights.as_ptr().cast(),
                            n,
                            first,
                            scale,
                            &mut out[..vector_end],
                        );
                    } else if T::KIND == 1 {
                        narrow_half_row::<$kind>(
                            input,
                            weights.as_ptr().cast(),
                            n,
                            first,
                            scale,
                            &mut out[..vector_end],
                        );
                    } else {
                        narrow_row::<$kind>(
                            input,
                            weights.as_ptr().cast(),
                            n,
                            first,
                            scale,
                            &mut out[..vector_end],
                        );
                    }
                };
            }
            match T::KIND {
                0 => run!(0),
                1 => run!(1),
                2 => run!(2),
                3 => run!(3),
                _ => unreachable!(),
            }
        }
        vector_end
    };
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    let vector_end = {
        let _ = (wide, T::KIND);
        0
    };
    // Dispatch above requires FMA. Keep scalar tails in an FMA-enabled
    // function too, so portable builds do not call a software fmaf per term.
    unsafe {
        tail(input, weights, n, first, scale, out, vector_end);
    }
}

#[cfg_attr(
    any(target_arch = "x86", target_arch = "x86_64"),
    target_feature(enable = "fma")
)]
unsafe fn tail<T: Weight>(
    input: &[f32],
    weights: &[T],
    n: usize,
    first: usize,
    scale: f32,
    out: &mut [f32],
    vector_end: usize,
) {
    for (col, value) in out.iter_mut().enumerate().skip(vector_end) {
        let mut total = 0.;
        for block in (0..input.len()).step_by(128) {
            let mut sum = 0.;
            for kk in block..(block + 128).min(input.len()) {
                sum = input[kk].mul_add(weights[kk * n + first + col].f32(), sum);
            }
            total += sum;
        }
        *value = total * scale;
    }
}

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
macro_rules! define_row {
    ($name:ident,$features:literal,$lanes:literal,$zero:ident,$broadcast:ident,$fma:ident,$add:ident,$mul:ident,$store:ident,$float:expr,$half:expr,$bfloat:expr,$int:expr) => {
        #[target_feature(enable=$features)]
        unsafe fn $name<const KIND: usize>(
            input: &[f32],
            weights: *const u8,
            stride: usize,
            first: usize,
            scale: f32,
            output: &mut [f32],
        ) {
            let vectors = output.len() / $lanes;
            for group in (0..vectors).step_by(4) {
                let count = (vectors - group).min(4);
                let mut totals = [$zero(); 4];
                for block in (0..input.len()).step_by(128) {
                    let mut sums = [$zero(); 4];
                    for kk in block..(block + 128).min(input.len()) {
                        let x = $broadcast(input[kk]);
                        for v in 0..count {
                            let element = kk * stride + first + (group + v) * $lanes;
                            let w = unsafe {
                                match KIND {
                                    0 => ($float)(weights.add(element * 4)),
                                    1 => ($half)(weights.add(element * 2)),
                                    2 => ($bfloat)(weights.add(element * 2)),
                                    3 => ($int)(weights.add(element)),
                                    _ => unreachable!(),
                                }
                            };
                            sums[v] = $fma(w, x, sums[v]);
                        }
                    }
                    for v in 0..count {
                        totals[v] = $add(totals[v], sums[v]);
                    }
                }
                for v in 0..count {
                    let sum = if KIND == 3 {
                        $mul(totals[v], $broadcast(scale))
                    } else {
                        totals[v]
                    };
                    unsafe {
                        $store(output.as_mut_ptr().add((group + v) * $lanes), sum);
                    }
                }
            }
        }
    };
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
define_row!(
    wide_row,
    "avx512f",
    16,
    _mm512_setzero_ps,
    _mm512_set1_ps,
    _mm512_fmadd_ps,
    _mm512_add_ps,
    _mm512_mul_ps,
    _mm512_storeu_ps,
    |p: *const u8| _mm512_loadu_ps(p.cast()),
    |p: *const u8| _mm512_cvtph_ps(_mm256_loadu_si256(p.cast())),
    |p: *const u8| _mm512_castsi512_ps(_mm512_slli_epi32::<16>(_mm512_cvtepu16_epi32(
        _mm256_loadu_si256(p.cast())
    ))),
    |p: *const u8| _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128(p.cast())))
);

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
define_row!(
    narrow_half_row,
    "avx2,fma,f16c",
    8,
    _mm256_setzero_ps,
    _mm256_set1_ps,
    _mm256_fmadd_ps,
    _mm256_add_ps,
    _mm256_mul_ps,
    _mm256_storeu_ps,
    |p: *const u8| _mm256_loadu_ps(p.cast()),
    |p: *const u8| _mm256_cvtph_ps(_mm_loadu_si128(p.cast())),
    |p: *const u8| _mm256_castsi256_ps(_mm256_slli_epi32::<16>(_mm256_cvtepu16_epi32(
        _mm_loadu_si128(p.cast())
    ))),
    |p: *const u8| _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(p.cast())))
);

// F16 uses narrow_half_row; this variant needs no F16C for F32/BF16/I8.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
define_row!(
    narrow_row,
    "avx2,fma",
    8,
    _mm256_setzero_ps,
    _mm256_set1_ps,
    _mm256_fmadd_ps,
    _mm256_add_ps,
    _mm256_mul_ps,
    _mm256_storeu_ps,
    |p: *const u8| _mm256_loadu_ps(p.cast()),
    |p: *const u8| _mm256_castsi256_ps(_mm256_slli_epi32::<16>(_mm256_cvtepu16_epi32(
        _mm_loadu_si128(p.cast())
    ))),
    |p: *const u8| _mm256_castsi256_ps(_mm256_slli_epi32::<16>(_mm256_cvtepu16_epi32(
        _mm_loadu_si128(p.cast())
    ))),
    |p: *const u8| _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(p.cast())))
);

#[cfg(all(test, any(target_arch = "x86", target_arch = "x86_64")))]
#[path = "../../../tests/unit/ops/cpu_gemm/small.rs"]
mod tests;
