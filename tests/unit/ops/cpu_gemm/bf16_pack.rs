use super::*;

#[test]
fn integer_packers_preserve_all_bits_unaligned_inputs_and_reused_tails() {
    let mut actual = Vec::new();
    // Multiplication by an odd integer visits all u16 representations in the
    // first case, including signed zeros, subnormals and every NaN payload.
    for (k, n) in std::iter::once((257usize, 257usize)).chain(
        [5633usize, 129, 128, 33, 32, 31, 17, 16, 15, 9, 8, 7, 2, 1]
            .into_iter()
            .flat_map(|k| [25usize, 17, 16, 15, 12, 9, 8, 7, 4, 3, 1].map(|n| (k, n))),
    ) {
        for offset in [0, 1, 3] {
            let storage: Vec<_> = (0..k * n + offset)
                .map(|i| bf16::from_bits((i as u16).wrapping_mul(197)))
                .collect();
            let weights = &storage[offset..];
            actual.resize(k.div_ceil(2) * CHANNELS, u32::MAX);
            for first in (0..n).step_by(CHANNELS) {
                let mut check = |packer: &dyn Fn(&mut [u32])| {
                    actual.fill(u32::MAX);
                    packer(&mut actual);
                    for pair in 0..k.div_ceil(2) {
                        for channel in 0..CHANNELS {
                            let bits = |half: usize| {
                                if first + channel < n && 2 * pair + half < k {
                                    weights[(first + channel) * k + 2 * pair + half].to_bits()
                                        as u32
                                } else {
                                    0
                                }
                            };
                            assert_eq!(actual[pair * CHANNELS + channel], bits(0) | bits(1) << 16);
                        }
                    }
                };
                check(&|out| pack(weights, k, n, first, out));
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                {
                    if std::is_x86_feature_detected!("sse2") {
                        // All slice extents meet the same checked dispatcher
                        // contract; force this fallback even in native builds.
                        check(&|out| unsafe { pack_sse2(weights, k, n, first, out) });
                    }
                    if std::is_x86_feature_detected!("avx2") {
                        check(&|out| unsafe { pack_avx2(weights, k, n, first, out) });
                    }
                }
            }
        }
    }
}
