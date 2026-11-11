use super::*;
use crate::autograd::no_grad;
use ndarray::{Array2, Array3, ArrayViewD, s};

#[test]
fn narrow_bf16_prefill_dispatch_preserves_layouts_precision_and_fallbacks() {
    use crate::cpu::CpuKernelContext;
    use std::{num::NonZeroUsize, sync::Arc};
    let available = crate::arch::x86_avx512_bf16_kernel_runtime_available();
    let pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap(),
    );
    for (m, k, n, parallel_selected) in [
        (8, 512, 128, false),
        (14, 513, 129, false),
        (33, 1024, 257, false),
        (46, 2048, 256, false),
        (64, 2048, 256, true),
        (128, 2049, 271, false),
    ] {
        let a = Array2::from_shape_fn((m, k), |(r, c)| {
            bf16::from_f32(((r * k + c) % 29) as f32 / 37. - 0.4)
        });
        let w = Array2::from_shape_fn((n, k), |(r, c)| {
            bf16::from_f32(((r * k + c) % 31) as f32 / 41. - 0.4)
        });
        for strided in [false, true] {
            let ordered = |x: &Array2<bf16>| {
                if strided {
                    x.t().as_standard_layout().to_owned().reversed_axes()
                } else {
                    x.clone()
                }
            };
            let input = Tensor::from_shared_bf16_no_grad_with_device(
                ordered(&a).into_dyn().into_shared(),
                crate::device::Device::Cpu,
            );
            let weight = Tensor::from_shared_bf16_no_grad_with_device(
                ordered(&w).into_dyn().into_shared(),
                crate::device::Device::Cpu,
            );
            for limit in [1, 4] {
                let context = CpuKernelContext::with_pool(pool.clone())
                    .with_task_limit(NonZeroUsize::new(limit).unwrap());
                context.scope(|| {
                    no_grad(|| {
                        let result = try_prefill_gemm(&input, &weight, m, k, n);
                        let selected = available && (limit == 1 || parallel_selected);
                        assert_eq!(result.is_some(), selected);
                        if let Some(result) = result {
                            for ((row, col), &actual) in result.indexed_iter() {
                                let expected: f64 = a
                                    .row(row)
                                    .iter()
                                    .zip(w.row(col))
                                    .map(|(x, w)| x.to_f32() as f64 * w.to_f32() as f64)
                                    .sum();
                                assert!((actual as f64 - expected).abs() < 2e-4);
                            }
                        }
                        let output = matmul_transposed_rhs(&input, &weight);
                        assert_eq!(output.dtype(), DType::BF16);
                        assert_eq!(output.shape_vec(), vec![m, n]);
                        assert_eq!(input.dtype(), DType::BF16);
                        assert_eq!(weight.dtype(), DType::BF16);
                    })
                });
            }
            // This dispatcher is inference-only. Frozen inputs do not turn a
            // normal call into an implicit inference/precision scope.
            assert!(try_prefill_gemm(&input, &weight, m, k, n).is_none());
        }
    }
    for (m, k, n) in [(7, 512, 128), (8, 511, 128), (8, 512, 127), (1, 2048, 256)] {
        let input = Tensor::parameter_with_dtype(Array2::zeros((m, k)).into_dyn(), DType::BF16);
        let weight = Tensor::parameter_with_dtype(Array2::zeros((n, k)).into_dyn(), DType::BF16);
        assert!(no_grad(|| try_prefill_gemm(&input, &weight, m, k, n)).is_none());
    }
    for dtype in [DType::F32, DType::F16, DType::I8] {
        let input = Tensor::parameter_with_dtype(Array2::ones((8, 512)).into_dyn(), dtype);
        let weight = Tensor::parameter_with_dtype(Array2::ones((128, 512)).into_dyn(), DType::BF16);
        assert!(no_grad(|| try_prefill_gemm(&input, &weight, 8, 512, 128)).is_none());
    }
}

#[test]
fn bf16_input_pairs_match_previous_conversion_across_bits_and_strides() {
    fn check<A: DotElem>(view: ArrayViewD<'_, A>, m: usize, k: usize) {
        // Independent flat-index reference retains the original conversion and
        // row-boundary rule, including zero padding after an odd final K.
        let mut expected = vec![0u32; m * k.div_ceil(2)];
        for (i, &value) in view.iter().enumerate() {
            let bits = bf16::from_f32(value.to_f32()).to_bits() as u32;
            expected[(i / k) * k.div_ceil(2) + (i % k) / 2] |= bits << ((i % k % 2) * 16);
        }
        assert_eq!(pack_bf16_inputs(view, m, k), expected);
    }

    fn layouts<A: DotElem>(array: Array2<A>) {
        let (m, k) = array.dim();
        check(array.view().into_dyn(), m, k);
        let strided = array.t().as_standard_layout().to_owned().reversed_axes();
        check(strided.view().into_dyn(), m, k);
        check(array.slice(s![..;-1, ..;-1]).into_dyn(), m, k);
        if m > 0 {
            let row = array.slice(s![0..1, ..]);
            check(row.broadcast((m, k)).unwrap().into_dyn(), m, k);
        }
    }

    // The first shape visits every BF16/F16 bit pattern. Additional shapes
    // exercise pair and row tails without carrying padding into the next row.
    for (m, k) in [(257usize, 257usize)].into_iter().chain(
        [0usize, 1, 2, 3, 8, 9]
            .into_iter()
            .flat_map(|m| [1usize, 2, 3, 7, 8, 9, 31, 32, 33, 129, 513].map(|k| (m, k))),
    ) {
        let bits = |i: usize| (i as u16).wrapping_mul(197);
        layouts(Array2::from_shape_fn((m, k), |(row, col)| {
            bf16::from_bits(bits(row * k + col))
        }));
        layouts(Array2::from_shape_fn((m, k), |(row, col)| {
            f16::from_bits(bits(row * k + col))
        }));
        layouts(Array2::from_shape_fn((m, k), |(row, col)| {
            let i = row * k + col;
            f32::from_bits(((bits(i) as u32) << 16) | (i as u16 as u32))
        }));
    }

    for k in [1, 2, 9, 32, 513] {
        let batched = Array3::from_shape_fn((2, 3, k), |(b, row, col)| {
            bf16::from_bits(((b * 3 * k + row * k + col) as u16).wrapping_mul(197))
        });
        check(batched.view().into_dyn(), 6, k);
        check(batched.view().permuted_axes([1, 0, 2]).into_dyn(), 6, k);
        check(batched.slice(s![..;-1, .., ..;-1]).into_dyn(), 6, k);
    }
}
