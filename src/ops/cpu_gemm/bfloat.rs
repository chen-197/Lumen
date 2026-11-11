//! BF16 K-pair products write disjoint column views in the final output array.
use super::{CHANNELS, bf16, pack_bf16_panel};
use ndarray::{Array2, ArrayView2, ArrayViewMut2, Axis};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::cell::RefCell;

thread_local! {
    static PANEL: RefCell<Vec<u32>> = const { RefCell::new(Vec::new()) };
}

pub(in crate::ops) fn matmul(input: &[u32], weights: &[bf16], k: usize, n: usize) -> Array2<f32> {
    assert_eq!(weights.len(), n * k);
    product(input, k, n, |first, packed| {
        pack_bf16_panel(weights, k, n, first, packed);
    })
}

pub(in crate::ops) fn standard_matmul(input: &[u32], weights: ArrayView2<'_, bf16>) -> Array2<f32> {
    let (k, n) = weights.dim();
    product(input, k, n, |first, packed| {
        pack_standard(weights, first, packed);
    })
}

fn product(
    input: &[u32],
    k: usize,
    n: usize,
    pack: impl Fn(usize, &mut [u32]) + Send + Sync,
) -> Array2<f32> {
    assert!(k > 0);
    let pairs = k.div_ceil(2);
    assert_eq!(input.len() % pairs, 0);
    let m = input.len() / pairs;
    let mut output = Array2::zeros((m, n));
    if m == 0 || n == 0 {
        return output;
    }
    // Keep nearby column writes in one task, while retaining enough groups for
    // the caller's available workers. Only numerical array views enter the pool.
    let panels = n.div_ceil(CHANNELS);
    let capacity = crate::cpu::parallelism();
    let mut per_group = (panels / capacity).clamp(1, 4);
    // Capped dispatch groups iterator items with ceil division. Prefer a group
    // count divisible by capacity so grouping does not leave extra workers idle.
    while per_group > 1 && !panels.div_ceil(per_group).is_multiple_of(capacity) {
        per_group -= 1;
    }
    let columns = CHANNELS * per_group;
    crate::cpu::for_each_indexed(
        output.axis_chunks_iter_mut(Axis(1), columns),
        k.saturating_mul(columns).saturating_mul(m),
        |(group, mut output)| {
            for (panel, output) in output.axis_chunks_iter_mut(Axis(1), CHANNELS).enumerate() {
                let first = group * columns + panel * CHANNELS;
                panel_packed(input, k, output, |packed| pack(first, packed));
            }
        },
    );
    output
}

// The padded contiguous interface remains useful for direct-kernel references.
#[cfg(test)]
pub(super) fn panel(
    input: &[u32],
    weights: &[bf16],
    k: usize,
    n: usize,
    first: usize,
    output: &mut [f32],
) {
    assert!(k > 0);
    let m = input.len() / k.div_ceil(2);
    let output = ArrayViewMut2::from_shape((m, CHANNELS), output).unwrap();
    panel_packed(input, k, output, |packed| {
        pack_bf16_panel(weights, k, n, first, packed);
    });
}

fn panel_packed(
    input: &[u32],
    k: usize,
    mut output: ArrayViewMut2<'_, f32>,
    pack: impl FnOnce(&mut [u32]),
) {
    assert!(crate::arch::x86_avx512_bf16_kernel_runtime_available());
    assert!(k > 0);
    let pairs = k.div_ceil(2);
    assert_eq!(input.len() % pairs, 0);
    let m = input.len() / pairs;
    let columns = output.ncols();
    assert_eq!(output.nrows(), m);
    assert!((1..=CHANNELS).contains(&columns));
    // Empty ndarray slices can have zero strides; no pointer is dereferenced.
    if m == 0 {
        return;
    }
    // Slicing a singleton column can also normalize its unused stride to zero.
    assert!(columns == 1 || output.strides()[1] == 1);
    assert!(m <= 1 || output.strides()[0] >= columns as isize);
    let stride = output.strides()[0] as usize;
    let destination = output.as_mut_ptr();
    PANEL.with(|slot| {
        let mut packed = slot.borrow_mut();
        packed.resize(pairs * CHANNELS, 0);
        pack(&mut packed);
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        for token in (0..m).step_by(8) {
            let rows = (m - token).min(8);
            let input = &input[token * pairs..(token + rows) * pairs];
            macro_rules! compute {
                ($rows:literal) => {
                    // ISA and input/packed extents are checked above. This
                    // unique ndarray view contains all rows and active columns;
                    // masked stores touch only those columns, even at the tail.
                    unsafe {
                        bf16_kernel512::<$rows>(
                            input,
                            &packed,
                            pairs,
                            destination.add(token * stride),
                            stride,
                            columns,
                        )
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
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        let _ = (stride, destination, columns);
    });
}

pub(super) fn pack_standard(
    weights: ndarray::ArrayView2<'_, bf16>,
    first: usize,
    packed: &mut [u32],
) {
    let (k, n) = weights.dim();
    assert!(k > 0 && first < n);
    assert_eq!(packed.len(), k.div_ceil(2) * CHANNELS);
    packed.fill(0);
    let valid = (n - first).min(CHANNELS);
    for pair in 0..k.div_ceil(2) {
        for channel in 0..valid {
            let low = weights[(pair * 2, first + channel)].to_bits() as u32;
            let high = if pair * 2 + 1 < k {
                weights[(pair * 2 + 1, first + channel)].to_bits() as u32
            } else {
                0
            };
            packed[pair * CHANNELS + channel] = low | high << 16;
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx512bf16")]
unsafe fn bf16_kernel512<const M: usize>(
    input: &[u32],
    packed: &[u32],
    pairs: usize,
    output: *mut f32,
    stride: usize,
    columns: usize,
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
        // Each worker owns a disjoint output view. Partial last panels use a
        // mask so no store reaches another row or a neighboring column panel.
        unsafe {
            let destination = output.add(row * stride);
            if columns == CHANNELS {
                _mm512_storeu_ps(destination, *sum);
            } else {
                _mm512_mask_storeu_ps(destination, ((1u32 << columns) - 1) as u16, *sum);
            }
        }
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/ops/cpu_gemm/bfloat.rs"]
mod tests;
