use super::*;
use ndarray::s;

fn inputs(m: usize, k: usize) -> (Vec<bf16>, Vec<u32>) {
    let values: Vec<_> = (0..m * k)
        .map(|i| bf16::from_f32(((i * 7 % 19) as f32 - 9.) / 31.))
        .collect();
    let pairs = k.div_ceil(2);
    let mut packed = vec![0; m * pairs];
    for (i, value) in values.iter().enumerate() {
        packed[(i / k) * pairs + (i % k) / 2] |= (value.to_bits() as u32) << ((i % k % 2) * 16);
    }
    (values, packed)
}

#[test]
fn direct_output_views_preserve_guards_tails_and_scalar_reference() {
    if !crate::arch::x86_avx512_bf16_kernel_runtime_available() {
        return;
    }
    let guard = f32::from_bits(0x7fc12345);
    for m in [0, 1, 7, 8, 9, 17] {
        for k in [1, 127, 128, 129, 513] {
            let (values, input) = inputs(m, k);
            let weights: Vec<_> = (0..CHANNELS * k)
                .map(|i| bf16::from_f32(((i * 13 % 23) as f32 - 11.) / 37.))
                .collect();
            for columns in 1..=CHANNELS {
                let mut output = Array2::from_elem((m + 2, columns + 7), guard);
                panel_packed(
                    &input,
                    k,
                    output.slice_mut(s![1..m + 1, 3..columns + 3]),
                    |packed| pack_bf16_panel(&weights, k, CHANNELS, 0, packed),
                );
                for ((row, col), &actual) in output.indexed_iter() {
                    if row == 0 || row > m || col < 3 || col >= columns + 3 {
                        assert_eq!(actual.to_bits(), guard.to_bits(), "guard overwritten");
                    } else {
                        let expected: f64 = values[(row - 1) * k..row * k]
                            .iter()
                            .zip(&weights[(col - 3) * k..(col - 2) * k])
                            .map(|(x, w)| x.to_f32() as f64 * w.to_f32() as f64)
                            .sum();
                        assert!(
                            (actual as f64 - expected).abs() < 1e-4,
                            "M={m} K={k} columns={columns} row={row} col={col}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn direct_products_match_both_rhs_layouts_across_task_budgets_and_empty_axes() {
    if !crate::arch::x86_avx512_bf16_kernel_runtime_available() {
        return;
    }
    use crate::cpu::CpuKernelContext;
    use std::{num::NonZeroUsize, sync::Arc};
    let pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap(),
    );
    for m in [0, 1, 7, 8, 9, 17] {
        for k in [1, 129] {
            let (values, input) = inputs(m, k);
            for n in [0, 1, 15, 16, 17, 31, 32, 33, 128, 129, 191, 255, 256, 257] {
                let weights = Array2::from_shape_fn((n, k), |(row, col)| {
                    bf16::from_f32((((row * k + col) * 13 % 23) as f32 - 11.) / 37.)
                });
                let mut reference = None;
                for limit in [1, 4] {
                    let context = CpuKernelContext::with_pool(pool.clone())
                        .with_task_limit(NonZeroUsize::new(limit).unwrap());
                    context.scope(|| {
                        let actual = matmul(&input, weights.as_slice().unwrap(), k, n);
                        let standard = standard_matmul(&input, weights.t());
                        assert_eq!(actual.dim(), (m, n));
                        assert_eq!(actual, standard);
                        if let Some(expected) = &reference {
                            assert_eq!(&actual, expected);
                        } else {
                            for ((row, col), &value) in actual.indexed_iter() {
                                let expected: f64 = values[row * k..(row + 1) * k]
                                    .iter()
                                    .zip(weights.row(col))
                                    .map(|(x, w)| x.to_f32() as f64 * w.to_f32() as f64)
                                    .sum();
                                assert!((value as f64 - expected).abs() < 1e-4);
                            }
                            reference = Some(actual);
                        }
                    });
                }
            }
        }
    }
}

#[test]
fn unsafe_store_wrapper_rejects_incompatible_views_before_packing() {
    if !crate::arch::x86_avx512_bf16_kernel_runtime_available() {
        return;
    }
    use std::panic::{AssertUnwindSafe, catch_unwind};
    let packed = std::cell::Cell::new(false);
    let (_, input) = inputs(2, 1);
    let mut output = Array2::zeros((2, 34));
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            panel_packed(&input, 1, output.slice_mut(s![.., ..17]), |_| {
                packed.set(true)
            });
        }))
        .is_err()
    );
    assert!(!packed.get());
    let mut output = Array2::zeros((2, 32));
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            panel_packed(&input, 1, output.slice_mut(s![.., ..;2]), |_| {
                packed.set(true)
            });
        }))
        .is_err()
    );
    assert!(!packed.get());
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            panel_packed(&input, 1, output.slice_mut(s![..;-1, ..16]), |_| {
                packed.set(true)
            });
        }))
        .is_err()
    );
    assert!(!packed.get());
}
