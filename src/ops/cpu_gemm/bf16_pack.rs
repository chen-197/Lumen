//! Integer-only BF16 K-pair packing for transposed-RHS matrix products.
use super::{CHANNELS, bf16};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// Transpose pairs as integer bits: packing must not round, flush subnormals,
// or change NaN payloads. AVX2 uses 8x8 pair tiles, with SSE2 4x4 and scalar
// fallbacks for CPUs or panel extents that do not support the larger tiles.
pub(super) fn pack(weights: &[bf16], k: usize, n: usize, first: usize, packed: &mut [u32]) {
    assert!(k > 0 && first < n);
    assert_eq!(weights.len(), n * k);
    assert_eq!(packed.len(), k.div_ceil(2) * CHANNELS);
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if (n - first).min(CHANNELS) == CHANNELS && k >= 16 && std::is_x86_feature_detected!("avx2") {
        // Full 8-channel/8-pair tiles read only complete input pairs. The
        // remaining pairs are copied scalarly, including odd-K zero padding.
        unsafe { pack_avx2(weights, k, n, first, packed) };
        return;
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if std::is_x86_feature_detected!("sse2") {
        // The extents above cover every complete 4-channel/4-pair tile;
        // the helper handles incomplete tiles and odd K with scalar loads.
        unsafe { pack_sse2(weights, k, n, first, packed) };
        return;
    }
    for pair in 0..k.div_ceil(2) {
        for channel in 0..CHANNELS {
            packed[pair * CHANNELS + channel] = weight_pair(weights, k, n, first + channel, pair);
        }
    }
}

fn weight_pair(weights: &[bf16], k: usize, n: usize, channel: usize, pair: usize) -> u32 {
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
unsafe fn pack_sse2(weights: &[bf16], k: usize, n: usize, first: usize, packed: &mut [u32]) {
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
                packed[pair * CHANNELS + c] = weight_pair(weights, k, n, first + c, pair);
            }
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
// Transpose eight channels by eight K pairs as integer representations.
#[target_feature(enable = "avx2")]
unsafe fn pack_avx2(weights: &[bf16], k: usize, n: usize, first: usize, packed: &mut [u32]) {
    let pairs = k.div_ceil(2);
    let valid = (n - first).min(16);
    let full_pairs = k / 2 / 8 * 8;
    for channel in (0..16).step_by(8) {
        let end = if channel + 8 <= valid { full_pairs } else { 0 };
        for pair in (0..end).step_by(8) {
            unsafe {
                let source = weights.as_ptr().add((first + channel) * k + pair * 2);
                let a = _mm256_loadu_si256(source.cast());
                let b = _mm256_loadu_si256(source.add(k).cast());
                let c = _mm256_loadu_si256(source.add(k * 2).cast());
                let d = _mm256_loadu_si256(source.add(k * 3).cast());
                let e = _mm256_loadu_si256(source.add(k * 4).cast());
                let f = _mm256_loadu_si256(source.add(k * 5).cast());
                let g = _mm256_loadu_si256(source.add(k * 6).cast());
                let h = _mm256_loadu_si256(source.add(k * 7).cast());
                let ab0 = _mm256_unpacklo_epi32(a, b);
                let ab1 = _mm256_unpackhi_epi32(a, b);
                let cd0 = _mm256_unpacklo_epi32(c, d);
                let cd1 = _mm256_unpackhi_epi32(c, d);
                let ef0 = _mm256_unpacklo_epi32(e, f);
                let ef1 = _mm256_unpackhi_epi32(e, f);
                let gh0 = _mm256_unpacklo_epi32(g, h);
                let gh1 = _mm256_unpackhi_epi32(g, h);
                let abcd0 = _mm256_unpacklo_epi64(ab0, cd0);
                let abcd1 = _mm256_unpackhi_epi64(ab0, cd0);
                let abcd2 = _mm256_unpacklo_epi64(ab1, cd1);
                let abcd3 = _mm256_unpackhi_epi64(ab1, cd1);
                let efgh0 = _mm256_unpacklo_epi64(ef0, gh0);
                let efgh1 = _mm256_unpackhi_epi64(ef0, gh0);
                let efgh2 = _mm256_unpacklo_epi64(ef1, gh1);
                let efgh3 = _mm256_unpackhi_epi64(ef1, gh1);
                let destination = packed.as_mut_ptr().add(pair * 16 + channel);
                macro_rules! store {
                    ($i:literal,$x:ident,$y:ident) => {
                        _mm256_storeu_si256(
                            destination.add($i * 16).cast(),
                            _mm256_permute2x128_si256::<0x20>($x, $y),
                        );
                        _mm256_storeu_si256(
                            destination.add(($i + 4) * 16).cast(),
                            _mm256_permute2x128_si256::<0x31>($x, $y),
                        );
                    };
                }
                store!(0, abcd0, efgh0);
                store!(1, abcd1, efgh1);
                store!(2, abcd2, efgh2);
                store!(3, abcd3, efgh3);
            }
        }
        for pair in end..pairs {
            for c in channel..channel + 8 {
                packed[pair * 16 + c] = if c < valid {
                    let offset = (first + c) * k + pair * 2;
                    let low = weights[offset].to_bits() as u32;
                    let high = if pair * 2 + 1 < k {
                        weights[offset + 1].to_bits() as u32
                    } else {
                        0
                    };
                    low | high << 16
                } else {
                    0
                };
            }
        }
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/ops/cpu_gemm/bf16_pack.rs"]
mod tests;
