use super::backward::{batch_matmul_backward_cpu_f32, matmul_backward_cpu_f32};
use super::conversion::with_lowp_f32_compute_view;
use super::matvec::{matvec_rowmajor_serial_f32_f16, matvec_rowmajor_serial_f32_i8};
use super::prefill::{prefill_gemm_panels, try_prefill_gemm};
use super::rows::matmul_row_fallback;
use super::*;

#[test]
fn serial_lowp_triplets_match_f64_reference_with_tails() {
    for n in [0, 1, 2, 3, 4, 5, 7, 515] {
        for k in [31, 512, 513] {
            let x = sample_f32(k);
            let w = sample_f32(n * k);
            let half: Vec<f16> = w.iter().copied().map(f16::from_f32).collect();
            let quant: Vec<i8> = w.iter().map(|x| (x * 23.).round() as i8).collect();
            let mut got = vec![0.; n];
            matvec_rowmajor_serial_f32_f16(&x, &half, n, k, &mut got);
            for (row, &got) in got.iter().enumerate() {
                let expected: f64 = x
                    .iter()
                    .zip(&half[row * k..(row + 1) * k])
                    .map(|(&a, &b)| a as f64 * b.to_f32() as f64)
                    .sum();
                assert!(
                    (got as f64 - expected).abs() < 1e-4,
                    "F16 {got} vs {expected}"
                );
            }
            matvec_rowmajor_serial_f32_i8(&x, &quant, 0.037, n, k, &mut got);
            for (row, &got) in got.iter().enumerate() {
                let expected = x
                    .iter()
                    .zip(&quant[row * k..(row + 1) * k])
                    .map(|(&a, &b)| a as f64 * b as f64)
                    .sum::<f64>()
                    * 0.037f32 as f64;
                assert!(
                    (got as f64 - expected).abs() < 1e-4,
                    "I8 {got} vs {expected}"
                );
            }
        }
    }
}

#[test]
fn lowp_batch_triplets_keep_native_storage_and_strided_inputs() {
    let m = 33;
    for k in [512, 513, 2049] {
        let n = if k >= 2048 { 2051 } else { 515 };
        let backing = Array2::from_shape_vec((k, m), sample_f32(m * k)).unwrap();
        for input_dtype in [DType::F32, DType::F16, DType::BF16] {
            for dtype in [DType::F16, DType::I8] {
                let a = Tensor::from_array_no_grad(backing.t().to_owned().into_dyn());
                a.cast_inplace(input_dtype);
                let b = make_tensor(&[n, k], sample_f32(n * k), dtype);
                let mut expected = a
                    .detach()
                    .data()
                    .into_shape((m, k))
                    .unwrap()
                    .dot(&b.detach().data().into_shape((n, k)).unwrap().t());
                if input_dtype == DType::F16 && dtype == DType::F16 {
                    expected.mapv_inplace(|x| f16::from_f32(x).to_f32());
                }
                let got = no_grad(|| matmul(&a, &b));
                for (&actual, &reference) in got.data().iter().zip(expected.iter()) {
                    // F32 summation orders can straddle an F16 rounding
                    // midpoint. Permit one F16 ULP only for F16 output.
                    let half_neighbor = input_dtype == DType::F16
                        && dtype == DType::F16
                        && f16::from_f32(actual)
                            .to_bits()
                            .abs_diff(f16::from_f32(reference).to_bits())
                            <= 1;
                    assert!(
                        (actual - reference).abs() <= 4e-4 || half_neighbor,
                        "{input_dtype:?} x {dtype:?}, K={k}: {actual} vs {reference}"
                    );
                }
                assert_eq!(b.dtype(), dtype);
                assert!(!b.0.borrow().has_f32_data);
            }
        }
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_matmul_backward_outputs_release_without_reference_cycles() {
    if !cuda::is_available() {
        return;
    }
    cuda::set_enabled(true);
    let _strict = crate::autograd::set_strict_device_execution_scoped(true);
    for batched in [false, true] {
        let (a_shape, b_shape) = if batched {
            (vec![1, 1, 2, 3], vec![1, 1, 3, 2])
        } else {
            (vec![2, 3], vec![2, 3])
        };
        let a = make_grad_tensor(&a_shape, vec![1.; 6]).to_cuda();
        let b = make_grad_tensor(&b_shape, vec![0.5; 6]).to_cuda();
        let output = if batched {
            batch_matmul(&a, &b)
        } else {
            matmul(&a, &b)
        };
        let weak = Rc::downgrade(&output.0);
        output.backward();
        assert!(a.cloned_cuda_f32_grad().is_some());
        assert!(b.cloned_cuda_f32_grad().is_some());
        drop(output);
        assert!(
            weak.upgrade().is_none(),
            "backward closure retained its output"
        );
    }
    cuda::set_enabled(false);
}

#[test]
fn prefill_panels_match_scalar_for_tail_and_strided_weights() {
    let (m, k, n) = (10, 544, 515);
    for k in [513, k] {
        for input_dtype in [DType::F32, DType::F16, DType::BF16] {
            for weight_dtype in [DType::F32, DType::BF16] {
                let a = make_tensor(&[m, k], sample_f32(m * k), input_dtype);
                let b = make_tensor(&[n, k], sample_f32(n * k), weight_dtype);
                let reference_input = a.detach().data().into_shape((m, k)).unwrap();
                let expected =
                    reference_input.dot(&b.detach().data().into_shape((n, k)).unwrap().t());
                let output = no_grad(|| try_prefill_gemm(&a, &b, m, k, n).unwrap());
                assert_close(
                    output.as_slice().unwrap(),
                    expected.as_slice().unwrap(),
                    2e-4,
                );
                let weights = b.detach().data().into_shape((n, k)).unwrap();
                for row in 0..m {
                    for col in 0..n {
                        let expected: f64 = (0..k)
                            .map(|j| reference_input[[row, j]] as f64 * weights[[col, j]] as f64)
                            .sum();
                        let actual = output[[row, col]] as f64;
                        assert!(
                            (actual - expected).abs() <= 8e-6 + 8e-6 * expected.abs(),
                            "prefill {input_dtype:?} x {weight_dtype:?}: {actual} vs {expected}"
                        );
                    }
                }
                assert_eq!(a.dtype(), input_dtype);
                assert_eq!(b.dtype(), weight_dtype);
            }
        }
    }
    let a = Array2::from_shape_vec((m, k), sample_f32(m * k)).unwrap();
    let b = Array2::from_shape_vec((k, n), sample_f32(n * k)).unwrap();
    let expected = a.dot(&b);
    let output = prefill_gemm_panels(a.view().into_dyn(), b.t().into_dyn(), (1.0, 1.0), (m, k, n));
    assert_close(
        output.as_slice().unwrap(),
        expected.as_slice().unwrap(),
        2e-4,
    );
}

#[test]
fn bf16_prefill_mixed_inputs_match_f64_at_block_dispatch_boundary() {
    let (m, k, n) = (17, 1025, 515);
    for input_dtype in [DType::F32, DType::F16, DType::BF16] {
        let a = Tensor::from_array_no_grad(
            Array2::from_shape_vec((k, m), sample_f32(m * k))
                .unwrap()
                .reversed_axes()
                .into_dyn(),
        );
        a.cast_inplace(input_dtype);
        let b = Tensor::from_array_no_grad(
            Array2::from_shape_vec((k, n), sample_f32(n * k))
                .unwrap()
                .reversed_axes()
                .into_dyn(),
        );
        b.cast_inplace(DType::BF16);
        let x = a.detach().data().into_shape((m, k)).unwrap();
        let w = b.detach().data().into_shape((n, k)).unwrap();
        let output = no_grad(|| try_prefill_gemm(&a, &b, m, k, n).unwrap());
        for row in 0..m {
            for col in 0..n {
                let reference: f64 = (0..k)
                    .map(|j| x[[row, j]] as f64 * w[[col, j]] as f64)
                    .sum();
                let actual = output[[row, col]] as f64;
                assert!(
                    (actual - reference).abs() <= 8e-6 + 8e-6 * reference.abs(),
                    "{input_dtype:?} x BF16 row={row} col={col}: {actual} vs {reference}"
                );
            }
        }
        assert_eq!(a.dtype(), input_dtype);
        assert_eq!(b.dtype(), DType::BF16);
    }
}

#[test]
fn bf16_prefill_large_row_counts_keep_native_and_mixed_precision() {
    for (m, k, n) in [(257, 513, 515), (513, 1025, 519)] {
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            let a = make_tensor(&[m, k], sample_f32(m * k), dtype);
            let b = make_tensor(&[n, k], sample_f32(n * k), DType::BF16);
            let x = a.detach().data().into_shape((m, k)).unwrap();
            let w = b.detach().data().into_shape((n, k)).unwrap();
            for tasks in [1, 3] {
                let policy = crate::cpu::CpuKernelContext::default()
                    .with_task_limit(std::num::NonZeroUsize::new(tasks).unwrap());
                let output = policy
                    .scope(|| no_grad(|| try_prefill_gemm(&a, &b, m, k, n)))
                    .expect("large BF16 matrices must retain the prefill dispatcher");
                assert_eq!(output.dim(), (m, n));
                assert!(output.iter().all(|v| v.is_finite()));
                // Probe both sides of the former boundary, multiple row tiles
                // and the last partial row/channel against represented FP64.
                for row in [0, 7, 8, 255, 256, m - 1] {
                    for col in 0..n {
                        let expected: f64 = (0..k)
                            .map(|j| x[[row, j]] as f64 * w[[col, j]] as f64)
                            .sum();
                        let actual = output[[row, col]] as f64;
                        assert!(
                            (actual - expected).abs() <= 8e-6 + 8e-6 * expected.abs(),
                            "{dtype:?} large BF16 tasks={tasks} M={m} K={k} row={row} col={col}: {actual} vs {expected}"
                        );
                    }
                }
            }
            assert_eq!(a.dtype(), dtype);
            assert_eq!(b.dtype(), DType::BF16);
        }
    }
}
use crate::autograd::no_grad;
#[cfg(feature = "cuda")]
use crate::autograd::set_strict_device_execution;
use crate::ops::arithmetic::sum;
use crate::precision::{PrecisionConfig, with_precision_config};
#[cfg(feature = "cuda")]
use ndarray::{Array, IxDyn};

#[test]
fn row_fallback_respects_pool_budget_grain_and_nested_work() {
    use crate::accuracy_gate::ordered;
    use crate::cpu::CpuKernelContext;
    use std::num::NonZeroUsize;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    let pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(3)
            .build()
            .unwrap(),
    );
    let caller = std::thread::current().id();
    for budget in [1, 2, 7] {
        let context = CpuKernelContext::with_pool(pool.clone())
            .with_task_limit(NonZeroUsize::new(budget).unwrap());
        for (m, k, n) in [
            (0, 5, 7),
            (1, 129, 67),
            (3, 5, 7),
            (37, 129, 67),
            (37, 9, 0),
        ] {
            for layout in 0..3 {
                let input = (0..m * k).map(|i| (i % 31) as f32).collect::<Vec<_>>();
                let input = ordered(&[m, k], &input, layout);
                let active = AtomicUsize::new(0);
                let peak = AtomicUsize::new(0);
                let calls = AtomicUsize::new(0);
                let output = context.scope(|| {
                    matmul_row_fallback(input.view(), m, k, n, |row, out| {
                        calls.fetch_add(1, Ordering::SeqCst);
                        let count = active.fetch_add(1, Ordering::SeqCst) + 1;
                        peak.fetch_max(count, Ordering::SeqCst);
                        let id = std::thread::current().id();
                        if budget == 1 || m == 1 || m * k * n < 65536 {
                            assert_eq!(id, caller);
                        } else {
                            assert!(pool.current_thread_index().is_some());
                        }
                        if m == 1 {
                            assert_eq!(crate::cpu::parallelism(), budget.min(3));
                        } else {
                            assert_eq!(crate::cpu::parallelism(), 1);
                            let mut nested = [0; 17];
                            crate::cpu::for_each_chunk_mut(
                                &mut nested,
                                1,
                                100_000,
                                |(_, values)| {
                                    assert_eq!(std::thread::current().id(), id);
                                    values.fill(1);
                                },
                            );
                        }
                        out.fill(row.iter().sum());
                        active.fetch_sub(1, Ordering::SeqCst);
                    })
                });
                assert_eq!(calls.load(Ordering::SeqCst), if n == 0 { 0 } else { m });
                assert!(peak.load(Ordering::SeqCst) <= budget.min(3));
                for row in 0..m {
                    let expected: f32 = input.index_axis(ndarray::Axis(0), row).iter().sum();
                    assert!(output.row(row).iter().all(|&value| value == expected));
                }
            }
        }
    }
}

#[test]
fn compute_conversion_buffers_bound_retention_and_survive_nested_calls() {
    std::thread::spawn(|| {
        let a = ndarray::Array::from_elem(257, f16::from_f32(1.25)).into_dyn();
        let b = ndarray::Array::from_elem(67, bf16::from_f32(-2.5)).into_dyn();
        let c = ndarray::Array::from_elem(19, f16::from_f32(0.5)).into_dyn();
        let pointer = with_lowp_f32_compute_view(a.view(), |first| {
            with_lowp_f32_compute_view(b.view(), |second| {
                with_lowp_f32_compute_view(c.view(), |third| {
                    assert!(first.iter().all(|&x| x == 1.25));
                    assert!(second.iter().all(|&x| x == -2.5));
                    assert!(third.iter().all(|&x| x == 0.5));
                    COMPUTE_F32_SCRATCH.with(|slots| {
                        assert!(slots.iter().all(|s| s.try_borrow_mut().is_err()));
                    });
                });
            });
            first.as_ptr() as usize
        });
        with_lowp_f32_compute_view(b.view(), |view| {
            assert_eq!(view.as_ptr() as usize, pointer);
            assert!(view.iter().all(|&x| x == -2.5));
        });
        let large =
            ndarray::Array::from_elem(MAX_COMPUTE_SCRATCH_ELEMENTS + 1, f16::ONE).into_dyn();
        with_lowp_f32_compute_view(large.view(), |view| {
            assert_eq!(view.len(), large.len());
            assert!(view.iter().all(|&x| x == 1.0));
            COMPUTE_F32_SCRATCH.with(|slots| {
                assert!(slots.iter().all(|s| s.try_borrow_mut().is_ok()));
            });
        });
        for len in [1025, 131_073, MAX_COMPUTE_SCRATCH_ELEMENTS, 3] {
            let a = ndarray::Array::from_elem(len, bf16::from_f32(0.25)).into_dyn();
            with_lowp_f32_compute_view(a.view(), |view| {
                assert!(view.iter().all(|&x| x == 0.25));
            });
        }
        let result = std::panic::catch_unwind(|| {
            with_lowp_f32_compute_view(a.view(), |_| panic!("conversion scope probe"));
        });
        assert!(result.is_err());
        with_lowp_f32_compute_view(c.view(), |view| {
            assert!(view.iter().all(|&x| x == 0.5));
        });
        COMPUTE_F32_SCRATCH.with(|slots| {
            for slot in slots {
                assert!(slot.borrow().capacity() <= MAX_COMPUTE_SCRATCH_ELEMENTS);
            }
        });
    })
    .join()
    .unwrap();
}

#[test]
fn reused_compute_views_refresh_mutated_mixed_precision_inputs_and_gradients() {
    use crate::accuracy_gate::{check, ordered, tensor, values};
    use crate::autograd::Device;
    let (m, k, n) = (7, 5, 3);
    for xd in [DType::F16, DType::BF16] {
        for wd in [DType::F32, DType::F16, DType::BF16] {
            for layout in 0..3 {
                let x = tensor(&[1, 1, m, k], &vec![0.; m * k], xd, layout, Device::Cpu);
                let w = tensor(&[1, 1, k, n], &vec![0.; k * n], wd, layout, Device::Cpu);
                for step in 0..4 {
                    let xv = (0..m * k)
                        .map(|i| ((i * 17 + step * 13) % 37) as f32 * 0.00371 - 0.05)
                        .collect::<Vec<_>>();
                    let wv = (0..k * n)
                        .map(|i| ((i * 11 + step * 7) % 41) as f32 * 0.00713 - 0.09)
                        .collect::<Vec<_>>();
                    let next_x = tensor(&[1, 1, m, k], &xv, xd, layout, Device::Cpu);
                    let next_w = tensor(&[1, 1, k, n], &wv, wd, layout, Device::Cpu);
                    x.set_array_f32_with_dtype(
                        ordered(&[1, 1, m, k], &values(&next_x), layout),
                        xd,
                    );
                    w.set_array_f32_with_dtype(
                        ordered(&[1, 1, k, n], &values(&next_w), layout),
                        wd,
                    );
                    let xv = values(&x);
                    let wv = values(&w);
                    let expected = (0..m * n)
                        .map(|i| {
                            (0..k)
                                .map(|j| xv[i / n * k + j] as f64 * wv[j * n + i % n] as f64)
                                .sum::<f64>()
                        })
                        .collect::<Vec<_>>();
                    let inference = no_grad(|| batch_matmul(&x, &w));
                    let round = |v: f64| match inference.dtype() {
                        DType::F16 => f16::from_f32(v as f32).to_f32() as f64,
                        DType::BF16 => bf16::from_f32(v as f32).to_f32() as f64,
                        _ => v,
                    };
                    check(
                        "refreshed native inference",
                        &values(&inference),
                        &expected.iter().copied().map(round).collect::<Vec<_>>(),
                        1e-7,
                        1e-6,
                    );
                    x.zero_grad();
                    w.zero_grad();
                    let output = batch_matmul(&x, &w);
                    check(
                        "refreshed native training",
                        &values(&output),
                        &expected,
                        1e-7,
                        1e-6,
                    );
                    output.backward();
                    let dx = (0..m * k)
                        .map(|i| (0..n).map(|j| wv[i % k * n + j] as f64).sum::<f64>())
                        .collect::<Vec<_>>();
                    let dw = (0..k * n)
                        .map(|i| (0..m).map(|j| xv[j * k + i / n] as f64).sum::<f64>())
                        .collect::<Vec<_>>();
                    check(
                        "refreshed input gradient",
                        &x.grad().unwrap().iter().copied().collect::<Vec<_>>(),
                        &dx,
                        1e-7,
                        1e-6,
                    );
                    check(
                        "refreshed weight gradient",
                        &w.grad().unwrap().iter().copied().collect::<Vec<_>>(),
                        &dw,
                        1e-7,
                        1e-6,
                    );
                }
            }
        }
    }
}

fn sample_f32(len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| (((i * 17 + 11) % 29) as f32) / 13.0 - 1.0)
        .collect()
}

fn to_bf16(src: &[f32]) -> Vec<bf16> {
    src.iter().map(|&v| bf16::from_f32(v)).collect()
}

fn to_f16(src: &[f32]) -> Vec<f16> {
    src.iter().map(|&v| f16::from_f32(v)).collect()
}

fn bf16_to_f32(src: &[bf16]) -> Vec<f32> {
    src.iter().map(|&v| v.to_f32()).collect()
}

#[cfg(feature = "cuda")]
fn make_grad_tensor(shape: &[usize], data: Vec<f32>) -> Tensor {
    Tensor::from_data_with_grad_flag(
        Array::from_shape_vec(IxDyn(shape), data)
            .expect("test tensor shape mismatch")
            .into_dyn(),
        true,
    )
}

fn f16_to_f32(src: &[f16]) -> Vec<f32> {
    src.iter().map(|&v| v.to_f32()).collect()
}

fn i8_storage(src: &[f32]) -> (Vec<f32>, Vec<i8>, f32) {
    let tensor = make_tensor(&[src.len()], src.to_vec(), DType::I8);
    match tensor.native_storage_owned() {
        TensorStorageOwned::I8(data, scale) => (
            tensor.data_ref().iter().copied().collect(),
            data.iter().copied().collect(),
            scale,
        ),
        TensorStorageOwned::F32(_) | TensorStorageOwned::F16(_) | TensorStorageOwned::BF16(_) => {
            panic!("expected i8 storage")
        }
    }
}

fn assert_close(lhs: &[f32], rhs: &[f32], tol: f32) {
    assert_eq!(lhs.len(), rhs.len());
    for (idx, (&a, &b)) in lhs.iter().zip(rhs.iter()).enumerate() {
        assert!(
            (a - b).abs() <= tol,
            "mismatch at {idx}: lhs={a}, rhs={b}, tol={tol}"
        );
    }
}

#[test]
fn bf16_input_matvec_matches_quantized_reference() {
    let k_dim = 11usize;
    let n_rows = 7usize;
    let x = sample_f32(k_dim);
    let x_bf16 = to_bf16(&x);
    let w = sample_f32(n_rows * k_dim);
    let w_bf16 = to_bf16(&w);

    let mut ref_out = vec![0.0f32; n_rows];
    let mut out = vec![0.0f32; n_rows];

    let x_q = bf16_to_f32(&x_bf16);
    matvec_rowmajor_parallel(&x_q, &w, n_rows, k_dim, &mut ref_out);
    matvec_rowmajor_parallel_mixed(
        SliceRef::BF16(&x_bf16),
        SliceRef::F32(&w),
        n_rows,
        k_dim,
        &mut out[..],
    );
    assert_close(&ref_out, &out, 1e-5);

    let w_q = bf16_to_f32(&w_bf16);
    matvec_rowmajor_parallel(&x_q, &w_q, n_rows, k_dim, &mut ref_out);
    matvec_rowmajor_parallel_mixed(
        SliceRef::BF16(&x_bf16),
        SliceRef::BF16(&w_bf16),
        n_rows,
        k_dim,
        &mut out[..],
    );
    assert_close(&ref_out, &out, 1e-5);
}

#[test]
fn bf16_input_dual_matvec_matches_quantized_reference() {
    let k_dim = 13usize;
    let n_rows = 5usize;
    let x = sample_f32(k_dim);
    let x_bf16 = to_bf16(&x);
    let w0 = sample_f32(n_rows * k_dim);
    let w1 = sample_f32(n_rows * k_dim)
        .into_iter()
        .map(|v| v * 0.7 - 0.1)
        .collect::<Vec<_>>();
    let w0_bf16 = to_bf16(&w0);
    let w1_bf16 = to_bf16(&w1);
    let x_q = bf16_to_f32(&x_bf16);

    let mut ref0 = vec![0.0f32; n_rows];
    let mut ref1 = vec![0.0f32; n_rows];
    let mut out0 = vec![0.0f32; n_rows];
    let mut out1 = vec![0.0f32; n_rows];

    dual_matvec_rowmajor_parallel(&x_q, &w0, &w1, n_rows, k_dim, &mut ref0, &mut ref1);
    dual_matvec_rowmajor_parallel_mixed(
        SliceRef::BF16(&x_bf16),
        SliceRef::F32(&w0),
        SliceRef::F32(&w1),
        n_rows,
        k_dim,
        &mut out0[..],
        &mut out1[..],
    );
    assert_close(&ref0, &out0, 1e-5);
    assert_close(&ref1, &out1, 1e-5);

    let w0_q = bf16_to_f32(&w0_bf16);
    let w1_q = bf16_to_f32(&w1_bf16);
    dual_matvec_rowmajor_parallel(&x_q, &w0_q, &w1_q, n_rows, k_dim, &mut ref0, &mut ref1);
    dual_matvec_rowmajor_parallel_mixed(
        SliceRef::BF16(&x_bf16),
        SliceRef::BF16(&w0_bf16),
        SliceRef::BF16(&w1_bf16),
        n_rows,
        k_dim,
        &mut out0[..],
        &mut out1[..],
    );
    assert_close(&ref0, &out0, 1e-5);
    assert_close(&ref1, &out1, 1e-5);
}

#[test]
fn bf16_input_silu_matches_quantized_reference() {
    let k_dim = 9usize;
    let n_rows = 6usize;
    let x = sample_f32(k_dim);
    let x_bf16 = to_bf16(&x);
    let gate = sample_f32(n_rows * k_dim);
    let up = sample_f32(n_rows * k_dim)
        .into_iter()
        .map(|v| v * -0.5 + 0.2)
        .collect::<Vec<_>>();
    let gate_bf16 = to_bf16(&gate);
    let up_bf16 = to_bf16(&up);
    let x_q = bf16_to_f32(&x_bf16);

    let mut ref_out = vec![0.0f32; n_rows];
    let mut out = vec![0.0f32; n_rows];

    dual_matvec_silu_mul_rowmajor_parallel(&x_q, &gate, &up, n_rows, k_dim, &mut ref_out);
    dual_matvec_silu_mul_rowmajor_parallel_mixed(
        SliceRef::BF16(&x_bf16),
        SliceRef::F32(&gate),
        SliceRef::F32(&up),
        n_rows,
        k_dim,
        &mut out[..],
    );
    assert_close(&ref_out, &out, 1e-5);

    let gate_q = bf16_to_f32(&gate_bf16);
    let up_q = bf16_to_f32(&up_bf16);
    dual_matvec_silu_mul_rowmajor_parallel(&x_q, &gate_q, &up_q, n_rows, k_dim, &mut ref_out);
    dual_matvec_silu_mul_rowmajor_parallel_mixed(
        SliceRef::BF16(&x_bf16),
        SliceRef::BF16(&gate_bf16),
        SliceRef::BF16(&up_bf16),
        n_rows,
        k_dim,
        &mut out[..],
    );
    assert_close(&ref_out, &out, 1e-5);
}

#[test]
fn bf16_input_argmax_matches_quantized_reference() {
    let k_dim = 10usize;
    let n_rows = 12usize;
    let x = sample_f32(k_dim);
    let x_bf16 = to_bf16(&x);
    let w = sample_f32(n_rows * k_dim);
    let w_bf16 = to_bf16(&w);
    let x_q = bf16_to_f32(&x_bf16);
    let w_q = bf16_to_f32(&w_bf16);

    let idx_f32 = matvec_argmax_rowmajor_parallel(&x_q, &w, n_rows, k_dim);
    let idx_bf16f32 = matvec_argmax_rowmajor_parallel_mixed(
        SliceRef::BF16(&x_bf16),
        SliceRef::F32(&w),
        n_rows,
        k_dim,
    );
    assert_eq!(idx_f32, idx_bf16f32);

    let idx_q = matvec_argmax_rowmajor_parallel(&x_q, &w_q, n_rows, k_dim);
    let idx_bf16bf16 = matvec_argmax_rowmajor_parallel_mixed(
        SliceRef::BF16(&x_bf16),
        SliceRef::BF16(&w_bf16),
        n_rows,
        k_dim,
    );
    assert_eq!(idx_q, idx_bf16bf16);
}

#[test]
fn f16_input_matvec_matches_quantized_reference() {
    let k_dim = 11usize;
    let n_rows = 7usize;
    let x = sample_f32(k_dim);
    let x_f16 = to_f16(&x);
    let w = sample_f32(n_rows * k_dim);
    let w_f16 = to_f16(&w);

    let mut ref_out = vec![0.0f32; n_rows];
    let mut out = vec![0.0f32; n_rows];

    let x_q = f16_to_f32(&x_f16);
    matvec_rowmajor_parallel(&x_q, &w, n_rows, k_dim, &mut ref_out);
    matvec_rowmajor_parallel_mixed(
        SliceRef::F16(&x_f16),
        SliceRef::F32(&w),
        n_rows,
        k_dim,
        &mut out[..],
    );
    assert_close(&ref_out, &out, 1e-5);

    let w_q = f16_to_f32(&w_f16);
    matvec_rowmajor_parallel(&x_q, &w_q, n_rows, k_dim, &mut ref_out);
    matvec_rowmajor_parallel_mixed(
        SliceRef::F16(&x_f16),
        SliceRef::F16(&w_f16),
        n_rows,
        k_dim,
        &mut out[..],
    );
    assert_close(&ref_out, &out, 1e-5);
}

#[test]
fn f16_input_dual_matvec_matches_quantized_reference() {
    let k_dim = 13usize;
    let n_rows = 5usize;
    let x = sample_f32(k_dim);
    let x_f16 = to_f16(&x);
    let w0 = sample_f32(n_rows * k_dim);
    let w1 = sample_f32(n_rows * k_dim)
        .into_iter()
        .map(|v| v * 0.7 - 0.1)
        .collect::<Vec<_>>();
    let w0_f16 = to_f16(&w0);
    let w1_f16 = to_f16(&w1);
    let x_q = f16_to_f32(&x_f16);

    let mut ref0 = vec![0.0f32; n_rows];
    let mut ref1 = vec![0.0f32; n_rows];
    let mut out0 = vec![0.0f32; n_rows];
    let mut out1 = vec![0.0f32; n_rows];

    dual_matvec_rowmajor_parallel(&x_q, &w0, &w1, n_rows, k_dim, &mut ref0, &mut ref1);
    dual_matvec_rowmajor_parallel_mixed(
        SliceRef::F16(&x_f16),
        SliceRef::F32(&w0),
        SliceRef::F32(&w1),
        n_rows,
        k_dim,
        &mut out0[..],
        &mut out1[..],
    );
    assert_close(&ref0, &out0, 1e-5);
    assert_close(&ref1, &out1, 1e-5);

    let w0_q = f16_to_f32(&w0_f16);
    let w1_q = f16_to_f32(&w1_f16);
    dual_matvec_rowmajor_parallel(&x_q, &w0_q, &w1_q, n_rows, k_dim, &mut ref0, &mut ref1);
    dual_matvec_rowmajor_parallel_mixed(
        SliceRef::F16(&x_f16),
        SliceRef::F16(&w0_f16),
        SliceRef::F16(&w1_f16),
        n_rows,
        k_dim,
        &mut out0[..],
        &mut out1[..],
    );
    assert_close(&ref0, &out0, 1e-5);
    assert_close(&ref1, &out1, 1e-5);
}

#[test]
fn f16_input_silu_matches_quantized_reference() {
    let k_dim = 9usize;
    let n_rows = 6usize;
    let x = sample_f32(k_dim);
    let x_f16 = to_f16(&x);
    let gate = sample_f32(n_rows * k_dim);
    let up = sample_f32(n_rows * k_dim)
        .into_iter()
        .map(|v| v * 0.5 + 0.2)
        .collect::<Vec<_>>();
    let gate_f16 = to_f16(&gate);
    let up_f16 = to_f16(&up);
    let x_q = f16_to_f32(&x_f16);

    let mut ref_out = vec![0.0f32; n_rows];
    let mut out = vec![0.0f32; n_rows];

    dual_matvec_silu_mul_rowmajor_parallel(&x_q, &gate, &up, n_rows, k_dim, &mut ref_out);
    dual_matvec_silu_mul_rowmajor_parallel_mixed(
        SliceRef::F16(&x_f16),
        SliceRef::F32(&gate),
        SliceRef::F32(&up),
        n_rows,
        k_dim,
        &mut out[..],
    );
    assert_close(&ref_out, &out, 1e-5);

    let gate_q = f16_to_f32(&gate_f16);
    let up_q = f16_to_f32(&up_f16);
    dual_matvec_silu_mul_rowmajor_parallel(&x_q, &gate_q, &up_q, n_rows, k_dim, &mut ref_out);
    dual_matvec_silu_mul_rowmajor_parallel_mixed(
        SliceRef::F16(&x_f16),
        SliceRef::F16(&gate_f16),
        SliceRef::F16(&up_f16),
        n_rows,
        k_dim,
        &mut out[..],
    );
    assert_close(&ref_out, &out, 1e-5);
}

#[test]
fn f16_input_argmax_matches_quantized_reference() {
    let k_dim = 11usize;
    let n_rows = 7usize;
    let x = sample_f32(k_dim);
    let x_f16 = to_f16(&x);
    let w = sample_f32(n_rows * k_dim);
    let w_f16 = to_f16(&w);
    let x_q = f16_to_f32(&x_f16);
    let w_q = f16_to_f32(&w_f16);

    let idx_f32 = matvec_argmax_rowmajor_parallel(&x_q, &w, n_rows, k_dim);
    let idx_f16f32 = matvec_argmax_rowmajor_parallel_mixed(
        SliceRef::F16(&x_f16),
        SliceRef::F32(&w),
        n_rows,
        k_dim,
    );
    assert_eq!(idx_f32, idx_f16f32);

    let idx_q = matvec_argmax_rowmajor_parallel(&x_q, &w_q, n_rows, k_dim);
    let idx_f16f16 = matvec_argmax_rowmajor_parallel_mixed(
        SliceRef::F16(&x_f16),
        SliceRef::F16(&w_f16),
        n_rows,
        k_dim,
    );
    assert_eq!(idx_q, idx_f16f16);
}

#[test]
fn i8_weight_matvec_matches_quantized_reference() {
    let k_dim = 11usize;
    let n_rows = 7usize;
    let x = sample_f32(k_dim);
    let w = sample_f32(n_rows * k_dim);
    let (w_q, w_i8, w_scale) = i8_storage(&w);

    let mut ref_out = vec![0.0f32; n_rows];
    let mut out = vec![0.0f32; n_rows];

    matvec_rowmajor_parallel(&x, &w_q, n_rows, k_dim, &mut ref_out);
    matvec_rowmajor_parallel_mixed(
        SliceRef::F32(&x),
        SliceRef::I8(&w_i8, w_scale),
        n_rows,
        k_dim,
        &mut out[..],
    );
    assert_close(&ref_out, &out, 1e-5);
}

#[test]
fn i8_weight_dual_matvec_matches_quantized_reference() {
    let k_dim = 13usize;
    let n_rows = 5usize;
    let x = sample_f32(k_dim);
    let w0 = sample_f32(n_rows * k_dim);
    let w1 = sample_f32(n_rows * k_dim)
        .into_iter()
        .map(|v| v * 0.7 - 0.1)
        .collect::<Vec<_>>();
    let (w0_q, w0_i8, w0_scale) = i8_storage(&w0);
    let (w1_q, w1_i8, w1_scale) = i8_storage(&w1);

    let mut ref0 = vec![0.0f32; n_rows];
    let mut ref1 = vec![0.0f32; n_rows];
    let mut out0 = vec![0.0f32; n_rows];
    let mut out1 = vec![0.0f32; n_rows];

    dual_matvec_rowmajor_parallel(&x, &w0_q, &w1_q, n_rows, k_dim, &mut ref0, &mut ref1);
    dual_matvec_rowmajor_parallel_mixed(
        SliceRef::F32(&x),
        SliceRef::I8(&w0_i8, w0_scale),
        SliceRef::I8(&w1_i8, w1_scale),
        n_rows,
        k_dim,
        &mut out0[..],
        &mut out1[..],
    );
    assert_close(&ref0, &out0, 1e-5);
    assert_close(&ref1, &out1, 1e-5);
}

#[test]
fn i8_weight_silu_matches_quantized_reference() {
    let k_dim = 9usize;
    let n_rows = 6usize;
    let x = sample_f32(k_dim);
    let gate = sample_f32(n_rows * k_dim);
    let up = sample_f32(n_rows * k_dim)
        .into_iter()
        .map(|v| v * -0.5 + 0.2)
        .collect::<Vec<_>>();
    let (gate_q, gate_i8, gate_scale) = i8_storage(&gate);
    let (up_q, up_i8, up_scale) = i8_storage(&up);

    let mut ref_out = vec![0.0f32; n_rows];
    let mut out = vec![0.0f32; n_rows];

    dual_matvec_silu_mul_rowmajor_parallel(&x, &gate_q, &up_q, n_rows, k_dim, &mut ref_out);
    dual_matvec_silu_mul_rowmajor_parallel_mixed(
        SliceRef::F32(&x),
        SliceRef::I8(&gate_i8, gate_scale),
        SliceRef::I8(&up_i8, up_scale),
        n_rows,
        k_dim,
        &mut out[..],
    );
    assert_close(&ref_out, &out, 1e-5);
}

#[test]
fn large_vocabulary_argmax_keeps_first_tie_and_native_weight_storage() {
    let pool = std::sync::Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(3)
            .build()
            .unwrap(),
    );
    for k in [31, 128, 513] {
        let n = MATVEC_BLOCK_THRESHOLD + 3;
        let x = vec![1.0; k];
        let x_f16 = vec![f16::ONE; k];
        let x_bf16 = vec![bf16::ONE; k];
        let mut w = vec![-2i8; n * k];
        for row in [33, 65, n - 1] {
            w[row * k..(row + 1) * k].fill(-1);
        }
        let w_f32: Vec<_> = w.iter().map(|&v| v as f32 * 0.25).collect();
        for limit in [1, 3] {
            crate::cpu::CpuKernelContext::with_pool(pool.clone())
                .with_task_limit(std::num::NonZeroUsize::new(limit).unwrap())
                .scope(|| {
                    assert_eq!(matvec_argmax_rowmajor_parallel(&x, &w_f32, n, k), 33);
                    for input in [
                        SliceRef::F32(&x),
                        SliceRef::F16(&x_f16),
                        SliceRef::BF16(&x_bf16),
                    ] {
                        assert_eq!(
                            matvec_argmax_rowmajor_parallel_mixed(
                                input,
                                SliceRef::I8(&w, 0.25),
                                n,
                                k
                            ),
                            33,
                        );
                    }
                });
        }
    }
}

#[test]
fn large_vocabulary_argmax_handles_empty_nan_infinity_and_tail() {
    let n = MATVEC_BLOCK_THRESHOLD + 3;
    let k = 129;
    let x = vec![1.0; k];
    let mut w = vec![f32::NAN; n * k];
    assert_eq!(matvec_argmax_rowmajor_parallel(&x, &w, n, k), 0);
    w[(n - 1) * k..].fill(1.0);
    assert_eq!(matvec_argmax_rowmajor_parallel(&x, &w, n, k), n - 1);
    w[33 * k..34 * k].fill(f32::INFINITY);
    w[65 * k..66 * k].fill(f32::INFINITY);
    assert_eq!(matvec_argmax_rowmajor_parallel(&x, &w, n, k), 33);
    w.fill(f32::NEG_INFINITY);
    assert_eq!(matvec_argmax_rowmajor_parallel(&x, &w, n, k), 0);
    assert_eq!(matvec_argmax_rowmajor_parallel(&x, &[], 0, k), 0);
    assert_eq!(matvec_argmax_rowmajor_parallel(&[], &[], n, 0), 0);
}

#[test]
fn i8_weight_argmax_matches_quantized_reference() {
    let k_dim = 10usize;
    let n_rows = 12usize;
    let x = sample_f32(k_dim);
    let w = sample_f32(n_rows * k_dim);
    let (w_q, w_i8, w_scale) = i8_storage(&w);

    let idx_q = matvec_argmax_rowmajor_parallel(&x, &w_q, n_rows, k_dim);
    let idx_i8 = matvec_argmax_rowmajor_parallel_mixed(
        SliceRef::F32(&x),
        SliceRef::I8(&w_i8, w_scale),
        n_rows,
        k_dim,
    );
    assert_eq!(idx_q, idx_i8);
}

#[test]
fn nested_bf16_input_conversion_is_reentrant() {
    let lhs = to_bf16(&sample_f32(5));
    let rhs = to_bf16(&sample_f32(3));
    let lhs_q = bf16_to_f32(&lhs);
    let rhs_q = bf16_to_f32(&rhs);

    with_bf16_input_as_f32(&lhs, |lhs_f32| {
        assert_close(lhs_f32, &lhs_q, 1e-6);
        with_bf16_input_as_f32(&rhs, |rhs_f32| {
            assert_close(rhs_f32, &rhs_q, 1e-6);
        });
    });
}

fn make_tensor(shape: &[usize], data: Vec<f32>, dtype: DType) -> Tensor {
    let t = Tensor::from_array_no_grad(
        ndarray::Array::from_shape_vec(ndarray::IxDyn(shape), data)
            .expect("test tensor shape mismatch")
            .into_dyn(),
    );
    t.cast_inplace(dtype);
    t
}

fn make_training_tensor(shape: &[usize], data: Vec<f32>, dtype: DType) -> Tensor {
    Tensor::new_with_dtype(
        ndarray::Array::from_shape_vec(ndarray::IxDyn(shape), data)
            .expect("test tensor shape mismatch")
            .into_dyn(),
        dtype,
    )
}

fn make_training_parameter(shape: &[usize], data: Vec<f32>, dtype: DType) -> Tensor {
    Tensor::parameter_with_dtype(
        ndarray::Array::from_shape_vec(ndarray::IxDyn(shape), data)
            .expect("test tensor shape mismatch")
            .into_dyn(),
        dtype,
    )
}

#[test]
fn matmul_training_forward_keeps_low_precision_storage_until_backward() {
    for allow_parameter_dtype_copies in [false, true] {
        with_precision_config(
            PrecisionConfig {
                parameter_dtype: DType::F32,
                runtime_dtype: DType::F32,
                allow_parameter_dtype_copies,
            },
            || {
                for dtype in [DType::F16, DType::BF16, DType::I8] {
                    let a = make_training_tensor(
                        &[2, 4],
                        vec![1.0, -2.0, 0.5, 3.0, -1.0, 0.25, 1.5, -0.5],
                        dtype,
                    );
                    let b = make_training_parameter(
                        &[3, 4],
                        vec![
                            1.0, 0.0, -1.0, 2.0, 0.5, 1.5, -0.5, 0.25, -1.0, 2.0, 1.0, -0.5,
                        ],
                        dtype,
                    );

                    assert_eq!(a.dtype(), dtype);
                    assert_eq!(b.dtype(), dtype);
                    assert!(!a.has_host_f32_data());
                    assert!(!b.has_host_f32_data());

                    let out = matmul(&a, &b);
                    assert_eq!(out.dtype(), DType::F32);
                    assert_eq!(out.shape_vec(), vec![2, 3]);
                    assert!(
                        !a.has_host_f32_data(),
                        "lhs {dtype:?} materialized during forward"
                    );
                    assert!(
                        !b.has_host_f32_data(),
                        "rhs parameter {dtype:?} materialized during forward"
                    );

                    sum(&out).backward();
                    let a_grad = a.grad().expect("lhs grad");
                    let b_grad = b.grad().expect("rhs grad");
                    assert_eq!(a_grad.shape(), &[2, 4]);
                    assert_eq!(b_grad.shape(), &[3, 4]);
                    assert!(a_grad.iter().all(|v| v.is_finite()));
                    assert!(b_grad.iter().all(|v| v.is_finite()));
                    assert_eq!(a.dtype(), dtype);
                    assert_eq!(b.dtype(), dtype);
                    assert!(
                        !a.has_host_f32_data(),
                        "lhs {dtype:?} materialized during backward"
                    );
                    assert!(
                        !b.has_host_f32_data(),
                        "rhs parameter {dtype:?} materialized during backward"
                    );
                }
            },
        );
    }
}

#[test]
fn matmul_backward_reads_native_low_precision_without_f32_materialization() {
    fn decoded(tensor: &Tensor) -> Vec<f32> {
        match tensor.native_storage_owned() {
            TensorStorageOwned::F32(data) => data.iter().copied().collect(),
            TensorStorageOwned::F16(data) => data.iter().map(|v| v.to_f32()).collect(),
            TensorStorageOwned::BF16(data) => data.iter().map(|v| v.to_f32()).collect(),
            TensorStorageOwned::I8(data, scale) => {
                data.iter().map(|&v| (v as f32) * scale).collect()
            }
        }
    }

    let grad = ndarray::Array::from_shape_vec(
        ndarray::IxDyn(&[2, 3]),
        vec![1.0f32, -0.5, 0.25, 2.0, -1.0, 0.5],
    )
    .expect("grad shape mismatch")
    .into_dyn();

    for (lhs_dtype, rhs_dtype) in [
        (DType::F16, DType::F16),
        (DType::BF16, DType::BF16),
        (DType::I8, DType::I8),
        (DType::F32, DType::I8),
        (DType::F16, DType::I8),
        (DType::BF16, DType::I8),
        (DType::I8, DType::F32),
        (DType::I8, DType::F16),
        (DType::I8, DType::BF16),
    ] {
        let a = make_tensor(
            &[2, 4],
            vec![1.0, -2.0, 0.5, 3.0, -1.0, 0.25, 1.5, -0.5],
            lhs_dtype,
        );
        let b = make_tensor(
            &[3, 4],
            vec![
                1.0, 0.0, -1.0, 2.0, 0.5, 1.5, -0.5, 0.25, -1.0, 2.0, 1.0, -0.5,
            ],
            rhs_dtype,
        );
        let lhs_was_lowp = lhs_dtype != DType::F32;
        let rhs_was_lowp = rhs_dtype != DType::F32;
        if lhs_was_lowp {
            assert!(!a.has_host_f32_data());
        }
        if rhs_was_lowp {
            assert!(!b.has_host_f32_data());
        }

        let (da, db) = matmul_backward_cpu_f32(&a, &b, &grad.view(), 2, 4, 3);
        if lhs_was_lowp {
            assert!(
                !a.has_host_f32_data(),
                "lhs {lhs_dtype:?} materialized during matmul backward"
            );
        }
        if rhs_was_lowp {
            assert!(
                !b.has_host_f32_data(),
                "rhs {rhs_dtype:?} materialized during matmul backward"
            );
        }

        let a_vals = decoded(&a);
        let b_vals = decoded(&b);
        let grad_vals = grad.as_slice_memory_order().unwrap();
        for m in 0..2 {
            for k in 0..4 {
                let expected = (0..3)
                    .map(|n| grad_vals[m * 3 + n] * b_vals[n * 4 + k])
                    .sum::<f32>();
                let got = da.as_slice_memory_order().unwrap()[m * 4 + k];
                assert!(
                    (got - expected).abs() <= 0.08,
                    "lhs={lhs_dtype:?} rhs={rhs_dtype:?} da[{m},{k}] got {got}, expected {expected}"
                );
            }
        }
        for n in 0..3 {
            for k in 0..4 {
                let expected = (0..2)
                    .map(|m| grad_vals[m * 3 + n] * a_vals[m * 4 + k])
                    .sum::<f32>();
                let got = db.as_slice_memory_order().unwrap()[n * 4 + k];
                assert!(
                    (got - expected).abs() <= 0.08,
                    "lhs={lhs_dtype:?} rhs={rhs_dtype:?} db[{n},{k}] got {got}, expected {expected}"
                );
            }
        }
    }
}

#[test]
fn batch_matmul_backward_reads_native_low_precision_without_f32_materialization() {
    fn decoded(tensor: &Tensor) -> Vec<f32> {
        match tensor.native_storage_owned() {
            TensorStorageOwned::F32(data) => data.iter().copied().collect(),
            TensorStorageOwned::F16(data) => data.iter().map(|v| v.to_f32()).collect(),
            TensorStorageOwned::BF16(data) => data.iter().map(|v| v.to_f32()).collect(),
            TensorStorageOwned::I8(data, scale) => {
                data.iter().map(|&v| (v as f32) * scale).collect()
            }
        }
    }

    let grad = ndarray::Array::from_shape_vec(
        ndarray::IxDyn(&[1, 2, 2, 3]),
        vec![
            1.0f32, -0.5, 0.25, 2.0, -1.0, 0.5, -0.75, 1.25, -1.5, 0.5, 0.75, -0.25,
        ],
    )
    .expect("grad shape mismatch")
    .into_dyn();

    for (lhs_dtype, rhs_dtype) in [
        (DType::F16, DType::F16),
        (DType::BF16, DType::BF16),
        (DType::I8, DType::I8),
        (DType::F32, DType::I8),
        (DType::F16, DType::I8),
        (DType::BF16, DType::I8),
        (DType::I8, DType::F32),
        (DType::I8, DType::F16),
        (DType::I8, DType::BF16),
    ] {
        let lhs = make_tensor(
            &[1, 2, 2, 4],
            vec![
                1.0, -2.0, 0.5, 3.0, -1.0, 0.25, 1.5, -0.5, 0.75, -1.25, 2.0, -2.5, 1.25, 0.5,
                -0.75, 2.25,
            ],
            lhs_dtype,
        );
        let rhs = make_tensor(
            &[1, 2, 4, 3],
            vec![
                1.0, 0.0, -1.0, 2.0, 0.5, 1.5, -0.5, 0.25, -1.0, 2.0, 1.0, -0.5, -0.25, 0.75, 1.25,
                -1.5, 0.5, -0.75, 1.75, -1.25, 0.25, 0.5, -2.0, 1.0,
            ],
            rhs_dtype,
        );
        let lhs_was_lowp = lhs_dtype != DType::F32;
        let rhs_was_lowp = rhs_dtype != DType::F32;
        if lhs_was_lowp {
            assert!(!lhs.has_host_f32_data());
        }
        if rhs_was_lowp {
            assert!(!rhs.has_host_f32_data());
        }

        let (d_lhs, d_rhs) = batch_matmul_backward_cpu_f32(&lhs, &rhs, &grad.view(), 1, 2, 2, 4, 3);

        if lhs_was_lowp {
            assert!(
                !lhs.has_host_f32_data(),
                "lhs {lhs_dtype:?} materialized during batch backward"
            );
        }
        if rhs_was_lowp {
            assert!(
                !rhs.has_host_f32_data(),
                "rhs {rhs_dtype:?} materialized during batch backward"
            );
        }

        let lhs_vals = decoded(&lhs);
        let rhs_vals = decoded(&rhs);
        let grad_vals = grad.as_slice_memory_order().unwrap();
        let d_lhs_vals = d_lhs.as_slice_memory_order().unwrap();
        let d_rhs_vals = d_rhs.as_slice_memory_order().unwrap();
        for batch in 0..2 {
            for row in 0..2 {
                for kk in 0..4 {
                    let expected = (0..3)
                        .map(|col| {
                            grad_vals[batch * 2 * 3 + row * 3 + col]
                                * rhs_vals[batch * 4 * 3 + kk * 3 + col]
                        })
                        .sum::<f32>();
                    let got = d_lhs_vals[batch * 2 * 4 + row * 4 + kk];
                    assert!(
                        (got - expected).abs() <= 0.08,
                        "lhs={lhs_dtype:?} rhs={rhs_dtype:?} d_lhs[{batch},{row},{kk}] got {got}, expected {expected}"
                    );
                }
            }
            for kk in 0..4 {
                for col in 0..3 {
                    let expected = (0..2)
                        .map(|row| {
                            lhs_vals[batch * 2 * 4 + row * 4 + kk]
                                * grad_vals[batch * 2 * 3 + row * 3 + col]
                        })
                        .sum::<f32>();
                    let got = d_rhs_vals[batch * 4 * 3 + kk * 3 + col];
                    assert!(
                        (got - expected).abs() <= 0.08,
                        "lhs={lhs_dtype:?} rhs={rhs_dtype:?} d_rhs[{batch},{kk},{col}] got {got}, expected {expected}"
                    );
                }
            }
        }
    }
}

#[test]
fn batch_matmul_training_forward_keeps_low_precision_storage_until_backward() {
    for allow_parameter_dtype_copies in [false, true] {
        with_precision_config(
            PrecisionConfig {
                parameter_dtype: DType::F32,
                runtime_dtype: DType::F32,
                allow_parameter_dtype_copies,
            },
            || {
                for dtype in [DType::F16, DType::BF16, DType::I8] {
                    let lhs = make_training_tensor(
                        &[1, 1, 2, 3],
                        vec![1.0, -2.0, 0.5, 3.0, -1.0, 2.0],
                        dtype,
                    );
                    let rhs = make_training_parameter(
                        &[1, 1, 3, 2],
                        vec![0.5, 1.0, -1.5, 2.0, 0.25, -0.75],
                        dtype,
                    );

                    assert!(!lhs.has_host_f32_data());
                    assert!(!rhs.has_host_f32_data());

                    let out = batch_matmul(&lhs, &rhs);
                    assert_eq!(out.dtype(), DType::F32);
                    assert_eq!(out.shape_vec(), vec![1, 1, 2, 2]);
                    assert!(
                        !lhs.has_host_f32_data(),
                        "lhs {dtype:?} materialized during forward"
                    );
                    assert!(
                        !rhs.has_host_f32_data(),
                        "rhs parameter {dtype:?} materialized during forward"
                    );

                    sum(&out).backward();
                    let lhs_grad = lhs.grad().expect("lhs grad");
                    let rhs_grad = rhs.grad().expect("rhs grad");
                    assert_eq!(lhs_grad.shape(), &[1, 1, 2, 3]);
                    assert_eq!(rhs_grad.shape(), &[1, 1, 3, 2]);
                    assert!(lhs_grad.iter().all(|v| v.is_finite()));
                    assert!(rhs_grad.iter().all(|v| v.is_finite()));
                    assert_eq!(lhs.dtype(), dtype);
                    assert_eq!(rhs.dtype(), dtype);
                    assert!(
                        !lhs.has_host_f32_data(),
                        "lhs {dtype:?} materialized during backward"
                    );
                    assert!(
                        !rhs.has_host_f32_data(),
                        "rhs parameter {dtype:?} materialized during backward"
                    );
                }
            },
        );
    }
}

#[test]
fn matmul_no_grad_preserves_bf16_output_dtype() {
    let a = make_tensor(&[1, 4], vec![1.0, -2.0, 0.5, 3.0], DType::BF16);
    let b = make_tensor(
        &[3, 4],
        vec![
            1.0, 0.0, -1.0, 2.0, 0.5, 1.5, -0.5, 0.25, -1.0, 2.0, 1.0, -0.5,
        ],
        DType::BF16,
    );

    let ref_out = no_grad(|| {
        matmul(
            &make_tensor(&[1, 4], vec![1.0, -2.0, 0.5, 3.0], DType::F32),
            &make_tensor(
                &[3, 4],
                vec![
                    1.0, 0.0, -1.0, 2.0, 0.5, 1.5, -0.5, 0.25, -1.0, 2.0, 1.0, -0.5,
                ],
                DType::F32,
            ),
        )
    });
    let out = no_grad(|| matmul(&a, &b));

    assert_eq!(a.dtype(), DType::BF16);
    assert_eq!(b.dtype(), DType::BF16);
    assert_eq!(out.dtype(), DType::BF16);

    let ref_vals = ref_out
        .data_ref()
        .iter()
        .map(|&v| bf16::from_f32(v).to_f32())
        .collect::<Vec<_>>();
    out.with_storage_view(|view| match view {
        TensorStorageView::BF16(view) => {
            let vals = view.iter().map(|v| v.to_f32()).collect::<Vec<_>>();
            assert_eq!(vals, ref_vals);
        }
        TensorStorageView::F16(_) => panic!("bf16 matmul output should stay bf16 in no-grad"),
        TensorStorageView::F32(_) => panic!("bf16 matmul output should stay bf16 in no-grad"),
    });
}

#[test]
fn matmul_same_dtype_bf16_parameter_uses_native_storage_for_generic_gemm() {
    with_precision_config(
        PrecisionConfig {
            parameter_dtype: DType::BF16,
            runtime_dtype: DType::F32,
            allow_parameter_dtype_copies: true,
        },
        || {
            let a = make_tensor(&[1, 4], vec![1.0, -2.0, 0.5, 3.0], DType::BF16);
            let b = Tensor::parameter_with_dtype(
                ndarray::Array::from_shape_vec(
                    ndarray::IxDyn(&[3, 4]),
                    vec![
                        1.0, 0.0, -1.0, 2.0, 0.5, 1.5, -0.5, 0.25, -1.0, 2.0, 1.0, -0.5,
                    ],
                )
                .expect("parameter shape mismatch")
                .into_dyn(),
                DType::BF16,
            );

            {
                let inner = b.0.borrow();
                assert!(
                    !inner.has_f32_data,
                    "bf16 parameter should start without cached f32 copy"
                );
            }

            let out = no_grad(|| matmul(&a, &b));
            assert_eq!(out.dtype(), DType::BF16);

            let inner = b.0.borrow();
            assert!(
                !inner.has_f32_data,
                "generic bf16 matmul should read native parameter storage without caching f32"
            );
        },
    );
}

#[test]
fn matmul_mixed_f32_input_uses_native_low_precision_parameter_storage() {
    with_precision_config(
        PrecisionConfig {
            parameter_dtype: DType::BF16,
            runtime_dtype: DType::F32,
            allow_parameter_dtype_copies: true,
        },
        || {
            let a = make_tensor(&[1, 4], vec![1.0, -2.0, 0.5, 3.0], DType::F32);
            let b = Tensor::parameter_with_dtype(
                ndarray::Array::from_shape_vec(
                    ndarray::IxDyn(&[3, 4]),
                    vec![
                        1.0, 0.0, -1.0, 2.0, 0.5, 1.5, -0.5, 0.25, -1.0, 2.0, 1.0, -0.5,
                    ],
                )
                .expect("parameter shape mismatch")
                .into_dyn(),
                DType::BF16,
            );

            {
                let inner = b.0.borrow();
                assert!(
                    !inner.has_f32_data,
                    "bf16 parameter should start without cached f32 copy"
                );
            }

            let out = no_grad(|| matmul(&a, &b));
            assert_eq!(out.dtype(), DType::F32);

            let inner = b.0.borrow();
            assert!(
                !inner.has_f32_data,
                "mixed f32 input should read native low-precision parameter storage without caching f32"
            );
        },
    );
}

#[test]
fn batch_matmul_no_grad_preserves_bf16_output_dtype() {
    let lhs = make_tensor(
        &[1, 1, 2, 3],
        vec![1.0, -2.0, 0.5, 3.0, -1.0, 2.0],
        DType::BF16,
    );
    let rhs = make_tensor(
        &[1, 1, 3, 2],
        vec![0.5, 1.0, -1.5, 2.0, 0.25, -0.75],
        DType::BF16,
    );

    let ref_out = no_grad(|| {
        batch_matmul(
            &make_tensor(
                &[1, 1, 2, 3],
                vec![1.0, -2.0, 0.5, 3.0, -1.0, 2.0],
                DType::F32,
            ),
            &make_tensor(
                &[1, 1, 3, 2],
                vec![0.5, 1.0, -1.5, 2.0, 0.25, -0.75],
                DType::F32,
            ),
        )
    });
    let out = no_grad(|| batch_matmul(&lhs, &rhs));

    assert_eq!(lhs.dtype(), DType::BF16);
    assert_eq!(rhs.dtype(), DType::BF16);
    assert_eq!(out.dtype(), DType::BF16);

    let ref_vals = ref_out
        .data_ref()
        .iter()
        .map(|&v| bf16::from_f32(v).to_f32())
        .collect::<Vec<_>>();
    out.with_storage_view(|view| match view {
        TensorStorageView::BF16(view) => {
            let vals = view.iter().map(|v| v.to_f32()).collect::<Vec<_>>();
            assert_eq!(vals, ref_vals);
        }
        TensorStorageView::F16(_) => {
            panic!("bf16 batch_matmul output should stay bf16 in no-grad")
        }
        TensorStorageView::F32(_) => {
            panic!("bf16 batch_matmul output should stay bf16 in no-grad")
        }
    });
}

#[test]
fn matmul_no_grad_preserves_f16_output_dtype() {
    let a = make_tensor(&[1, 4], vec![1.0, -2.0, 0.5, 3.0], DType::F16);
    let b = make_tensor(
        &[3, 4],
        vec![
            1.0, 0.0, -1.0, 2.0, 0.5, 1.5, -0.5, 0.25, -1.0, 2.0, 1.0, -0.5,
        ],
        DType::F16,
    );

    let ref_out = no_grad(|| {
        matmul(
            &make_tensor(&[1, 4], vec![1.0, -2.0, 0.5, 3.0], DType::F32),
            &make_tensor(
                &[3, 4],
                vec![
                    1.0, 0.0, -1.0, 2.0, 0.5, 1.5, -0.5, 0.25, -1.0, 2.0, 1.0, -0.5,
                ],
                DType::F32,
            ),
        )
    });
    let out = no_grad(|| matmul(&a, &b));

    assert_eq!(a.dtype(), DType::F16);
    assert_eq!(b.dtype(), DType::F16);
    assert_eq!(out.dtype(), DType::F16);

    let ref_vals = ref_out
        .data_ref()
        .iter()
        .map(|&v| f16::from_f32(v).to_f32())
        .collect::<Vec<_>>();
    out.with_storage_view(|view| match view {
        TensorStorageView::F16(view) => {
            let vals = view.iter().map(|v| v.to_f32()).collect::<Vec<_>>();
            assert_eq!(vals, ref_vals);
        }
        TensorStorageView::BF16(_) => panic!("f16 matmul output should stay f16 in no-grad"),
        TensorStorageView::F32(_) => panic!("f16 matmul output should stay f16 in no-grad"),
    });
}

#[test]
fn matmul_no_grad_preserves_i8_output_dtype() {
    let a = make_tensor(&[1, 4], vec![1.0, -2.0, 0.5, 3.0], DType::I8);
    let b = make_tensor(
        &[3, 4],
        vec![
            1.0, 0.0, -1.0, 2.0, 0.5, 1.5, -0.5, 0.25, -1.0, 2.0, 1.0, -0.5,
        ],
        DType::I8,
    );

    let ref_a_q = make_tensor(&[1, 4], vec![1.0, -2.0, 0.5, 3.0], DType::I8);
    let ref_b_q = make_tensor(
        &[3, 4],
        vec![
            1.0, 0.0, -1.0, 2.0, 0.5, 1.5, -0.5, 0.25, -1.0, 2.0, 1.0, -0.5,
        ],
        DType::I8,
    );
    let a_q = ref_a_q.data_ref().iter().copied().collect::<Vec<_>>();
    let b_q = ref_b_q.data_ref().iter().copied().collect::<Vec<_>>();
    let ref_out = no_grad(|| {
        matmul(
            &make_tensor(&[1, 4], a_q, DType::F32),
            &make_tensor(&[3, 4], b_q, DType::F32),
        )
    });
    let out = no_grad(|| matmul(&a, &b));

    assert_eq!(out.dtype(), DType::I8);

    let expected = make_tensor(
        &[1, 3],
        ref_out.data_ref().iter().copied().collect(),
        DType::I8,
    );
    let ref_vals = expected.data_ref().iter().copied().collect::<Vec<_>>();
    let out_vals = out.data_ref().iter().copied().collect::<Vec<_>>();
    assert_eq!(out_vals.len(), ref_vals.len());
    for (actual, expected) in out_vals.iter().zip(ref_vals.iter()) {
        assert!(
            (actual - expected).abs() < 1e-5,
            "i8 matmul output drifted: actual={actual}, expected={expected}"
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_matmul_matches_cpu_reference() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    crate::ops::cuda::set_enabled(true);
    let a = make_tensor(&[8, 32], sample_f32(8 * 32), DType::F32);
    let b = make_tensor(&[6, 32], sample_f32(6 * 32), DType::F32);

    let out = no_grad(|| matmul(&a, &b));
    crate::ops::cuda::set_enabled(false);
    let reference = no_grad(|| matmul(&a, &b));

    let out_vals = out.data_ref().iter().copied().collect::<Vec<_>>();
    let ref_vals = reference.data_ref().iter().copied().collect::<Vec<_>>();
    assert_eq!(out_vals.len(), ref_vals.len());
    for (got, expect) in out_vals.iter().zip(ref_vals.iter()) {
        assert!((got - expect).abs() < 1e-3, "got {got}, expect {expect}");
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_batch_matmul_matches_cpu_reference() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    crate::ops::cuda::set_enabled(true);
    let lhs = make_tensor(&[2, 3, 4, 16], sample_f32(2 * 3 * 4 * 16), DType::F32);
    let rhs = make_tensor(&[2, 3, 16, 5], sample_f32(2 * 3 * 16 * 5), DType::F32);

    let out = no_grad(|| batch_matmul(&lhs, &rhs));
    crate::ops::cuda::set_enabled(false);
    let reference = no_grad(|| batch_matmul(&lhs, &rhs));

    let out_vals = out.data_ref().iter().copied().collect::<Vec<_>>();
    let ref_vals = reference.data_ref().iter().copied().collect::<Vec<_>>();
    assert_eq!(out_vals.len(), ref_vals.len());
    for (got, expect) in out_vals.iter().zip(ref_vals.iter()) {
        assert!((got - expect).abs() < 1e-3, "got {got}, expect {expect}");
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_matmul_preserves_bf16_dtype_in_strict_mode() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let a = make_tensor(&[1, 4], vec![1.0, -2.0, 0.5, 3.0], DType::BF16).to_cuda();
    let b = make_tensor(
        &[3, 4],
        vec![
            1.0, 0.0, -1.0, 2.0, 0.5, 1.5, -0.5, 0.25, -1.0, 2.0, 1.0, -0.5,
        ],
        DType::BF16,
    )
    .to_cuda();
    assert!(a.cloned_cuda_native_lowp_buffer().is_some());
    assert!(b.cloned_cuda_native_lowp_buffer().is_some());
    assert!(!a.has_host_f32_data());
    assert!(!b.has_host_f32_data());

    crate::ops::cuda::set_enabled(true);
    crate::autograd::set_strict_device_execution(true);
    let out = no_grad(|| matmul(&a, &b));
    crate::autograd::set_strict_device_execution(false);
    crate::ops::cuda::set_enabled(false);

    let reference = no_grad(|| matmul(&a.to_cpu(), &b.to_cpu()));
    assert!(out.is_cuda());
    assert_eq!(out.dtype(), DType::BF16);
    assert!(
        !out.has_host_f32_data(),
        "bf16 CUDA matmul should keep output resident until host data is requested"
    );
    {
        let inner = out.0.borrow();
        assert!(
            inner.cuda_f32_data.is_none(),
            "bf16 CUDA matmul should write native bf16 output, not a resident f32 buffer"
        );
        assert!(
            inner.cuda_bf16_data.is_some(),
            "bf16 CUDA matmul should keep resident bf16 storage"
        );
    }
    for (got, expect) in out.data_ref().iter().zip(reference.data_ref().iter()) {
        assert!((got - expect).abs() < 2e-2, "got {got}, expect {expect}");
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_matmul_preserves_f16_dtype_in_strict_mode() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let a = make_tensor(&[1, 4], vec![1.0, -2.0, 0.5, 3.0], DType::F16).to_cuda();
    let b = make_tensor(
        &[3, 4],
        vec![
            1.0, 0.0, -1.0, 2.0, 0.5, 1.5, -0.5, 0.25, -1.0, 2.0, 1.0, -0.5,
        ],
        DType::F16,
    )
    .to_cuda();
    assert!(a.cloned_cuda_native_lowp_buffer().is_some());
    assert!(b.cloned_cuda_native_lowp_buffer().is_some());
    assert!(!a.has_host_f32_data());
    assert!(!b.has_host_f32_data());

    crate::ops::cuda::set_enabled(true);
    crate::autograd::set_strict_device_execution(true);
    let out = no_grad(|| matmul(&a, &b));
    crate::autograd::set_strict_device_execution(false);
    crate::ops::cuda::set_enabled(false);

    let reference = no_grad(|| matmul(&a.to_cpu(), &b.to_cpu()));
    assert!(out.is_cuda());
    assert_eq!(out.dtype(), DType::F16);
    assert!(
        !out.has_host_f32_data(),
        "f16 CUDA matmul should keep output resident until host data is requested"
    );
    {
        let inner = out.0.borrow();
        assert!(
            inner.cuda_f32_data.is_none(),
            "f16 CUDA matmul should write native f16 output, not a resident f32 buffer"
        );
        assert!(
            inner.cuda_f16_data.is_some(),
            "f16 CUDA matmul should keep resident f16 storage"
        );
    }
    for (got, expect) in out.data_ref().iter().zip(reference.data_ref().iter()) {
        assert!((got - expect).abs() < 2e-2, "got {got}, expect {expect}");
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_bf16_matmul_native_view_materializes_cuda_values() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let a = make_tensor(&[1, 4], vec![1.0, -2.0, 0.5, 3.0], DType::BF16).to_cuda();
    let b = make_tensor(
        &[3, 4],
        vec![
            1.0, 0.0, -1.0, 2.0, 0.5, 1.5, -0.5, 0.25, -1.0, 2.0, 1.0, -0.5,
        ],
        DType::BF16,
    )
    .to_cuda();
    assert!(a.cloned_cuda_native_lowp_buffer().is_some());
    assert!(b.cloned_cuda_native_lowp_buffer().is_some());
    assert!(!a.has_host_f32_data());
    assert!(!b.has_host_f32_data());

    crate::ops::cuda::set_enabled(true);
    crate::autograd::set_strict_device_execution(true);
    let out = no_grad(|| matmul(&a, &b));
    crate::autograd::set_strict_device_execution(false);
    crate::ops::cuda::set_enabled(false);

    assert!(out.is_cuda());
    assert_eq!(out.dtype(), DType::BF16);
    assert!(!out.has_host_f32_data());

    let bf16_buffer = {
        let inner = out.0.borrow();
        assert!(inner.cuda_f32_data.is_none());
        inner
            .cuda_bf16_data
            .clone()
            .expect("bf16 matmul should keep resident bf16 output")
    };
    let bits =
        crate::ops::cuda::download_u16_storage(&bf16_buffer).expect("download bf16 matmul bits");
    let vals = bits
        .iter()
        .map(|&bits| bf16::from_bits(bits).to_f32())
        .collect::<Vec<_>>();
    assert!(
        vals.iter().any(|v| v.abs() > 1e-6),
        "resident CUDA bf16 values should not be an all-zero placeholder"
    );
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_matmul_preserves_i8_dtype_in_strict_mode() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let a = make_tensor(&[1, 4], vec![1.0, -2.0, 0.5, 3.0], DType::I8).to_cuda();
    let b = make_tensor(
        &[3, 4],
        vec![
            1.0, 0.0, -1.0, 2.0, 0.5, 1.5, -0.5, 0.25, -1.0, 2.0, 1.0, -0.5,
        ],
        DType::I8,
    )
    .to_cuda();
    assert!(a.cloned_cuda_native_lowp_buffer().is_some());
    assert!(b.cloned_cuda_native_lowp_buffer().is_some());
    assert!(!a.has_host_f32_data());
    assert!(!b.has_host_f32_data());

    crate::ops::cuda::set_enabled(true);
    crate::autograd::set_strict_device_execution(true);
    let out = no_grad(|| matmul(&a, &b));
    crate::autograd::set_strict_device_execution(false);
    crate::ops::cuda::set_enabled(false);

    let reference = no_grad(|| matmul(&a.to_cpu(), &b.to_cpu()));
    assert!(out.is_cuda());
    assert_eq!(out.dtype(), DType::I8);
    {
        let inner = out.0.borrow();
        assert!(
            !inner.has_f32_data,
            "i8 CUDA matmul output should not eagerly materialize host f32 data"
        );
        assert!(
            inner.cuda_f32_data.is_none(),
            "i8 CUDA matmul output should quantize away the temporary f32 buffer"
        );
        assert!(
            inner.cuda_i8_data.is_some(),
            "i8 CUDA matmul output should keep resident i8 storage"
        );
        assert!(
            inner.i8_scale.is_some(),
            "i8 CUDA matmul output should record dynamic quantization scale"
        );
    }
    for (got, expect) in out.data_ref().iter().zip(reference.data_ref().iter()) {
        assert!((got - expect).abs() < 1e-5, "got {got}, expect {expect}");
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_resident_i8_matmul_matches_quantized_reference() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let m = 9;
    let n = 19;
    let k = 35;
    let scale = 0.03125f32;
    let a = (0..m * k)
        .map(|i| ((((i * 17 + 3) % 255) as i32) - 127) as i8)
        .collect::<Vec<_>>();
    let b = (0..n * k)
        .map(|i| ((((i * 29 + 11) % 255) as i32) - 127) as i8)
        .collect::<Vec<_>>();
    let a_buf = crate::ops::cuda::upload_i8_storage(&a).expect("upload i8 matmul a");
    let b_buf = crate::ops::cuda::upload_i8_storage(&b).expect("upload i8 matmul b");

    let out = crate::ops::cuda::matmul_i8_buffer_no_host(&a_buf, scale, &b_buf, scale, m, n, k)
        .expect("CUDA resident i8 matmul");
    let got = crate::ops::cuda::download_f32(&out).expect("download resident i8 matmul");

    let mut expected = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0i32;
            for kk in 0..k {
                acc += (a[row * k + kk] as i32) * (b[col * k + kk] as i32);
            }
            expected[row * n + col] = (acc as f32) * scale * scale;
        }
    }

    for (actual, expected) in got.iter().zip(expected.iter()) {
        assert_eq!(actual, expected, "resident tiled i8 matmul kernel drifted");
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_resident_aligned_i8_matmul_matches_i32_reference() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let (m, n, k) = (11, 20, 36);
    let a_scale = 0.03125f32;
    let b_scale = 0.0625f32;
    let a = (0..m * k)
        .map(|i| ((((i * 17 + 3) % 255) as i32) - 127) as i8)
        .collect::<Vec<_>>();
    let b = (0..n * k)
        .map(|i| ((((i * 29 + 11) % 255) as i32) - 127) as i8)
        .collect::<Vec<_>>();
    let a_buf = crate::ops::cuda::upload_i8_storage(&a).expect("upload aligned i8 matmul a");
    let b_buf = crate::ops::cuda::upload_i8_storage(&b).expect("upload aligned i8 matmul b");
    let out = crate::ops::cuda::matmul_i8_buffer_no_host(&a_buf, a_scale, &b_buf, b_scale, m, n, k)
        .expect("CUDA aligned resident i8 matmul");
    let got = crate::ops::cuda::download_f32(&out).expect("download aligned i8 matmul");

    for row in 0..m {
        for col in 0..n {
            let acc = (0..k)
                .map(|kk| (a[row * k + kk] as i32) * (b[col * k + kk] as i32))
                .sum::<i32>();
            let expected = (acc as f32) * (a_scale * b_scale);
            assert_eq!(
                got[row * n + col],
                expected,
                "aligned cuBLAS i8 matmul drifted at row={row}, col={col}"
            );
        }
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_resident_f32_i8_gemm_preserves_activations_and_weight_tile_tails() {
    if !crate::ops::cuda::is_available() {
        return;
    }
    // Unaligned dimensions, a prefill shape, and matrices crossing the
    // cache-aware tile budget or its 32 MiB upper bound, with odd tails.
    for (m, n, k) in [
        (17, 129, 513),
        (46, 256, 2048),
        (8, 3073, 2048),
        (8, 4097, 2048),
        (127, 129, 129),
        (128, 129, 129),
        (129, 129, 513),
        (178, 3073, 2048),
        (128, 4097, 2048),
    ] {
        for revision in 0..2 {
            let scale = if revision == 0 { 0.013f32 } else { 0.023f32 };
            let mut a = (0..m * k)
                .map(|i| ((i * 37 % 1009) as f32 - 504.0) / 379.0)
                .collect::<Vec<_>>();
            let mut b = (0..n * k)
                .map(|i| (((i * 29 + revision * 11) % 255) as i32 - 127) as i8)
                .collect::<Vec<_>>();
            a[..k].fill(0.0);
            a[0] = 1000.0;
            a[1] = 1.0e-4;
            a[2] = -7.0e-5;
            b[..k].fill(0);
            b[1] = 79;
            b[2] = -101;
            let a_buf = crate::ops::cuda::upload_f32(&a).expect("upload F32 activation");
            let b_buf = crate::ops::cuda::upload_i8_storage(&b).expect("upload I8 weight");
            let out =
                crate::ops::cuda::matmul_f32_i8_buffer_no_host(&a_buf, &b_buf, scale, m, n, k)
                    .expect("F32xI8 tiled weight cuBLAS GEMM");
            let got = crate::ops::cuda::download_f32(&out).unwrap();
            let tiny_expected = a[1] * (b[1] as f32 * scale) + a[2] * (b[2] as f32 * scale);
            assert!(
                (got[0] - tiny_expected).abs() < 1.0e-8,
                "small F32 activations were rounded away: {} vs {tiny_expected}",
                got[0]
            );
            for row in 0..m {
                for col in 0..n {
                    let expected = (0..k)
                        .map(|kk| {
                            let weight = b[col * k + kk] as f32 * scale;
                            a[row * k + kk] as f64 * weight as f64
                        })
                        .sum::<f64>();
                    let tolerance = 3.0e-4 + 3.0e-5 * expected.abs();
                    assert!(
                        (got[row * n + col] as f64 - expected).abs() <= tolerance,
                        "F32xI8 GEMM m={m} n={n} k={k} revision={revision} row={row} col={col}: {} vs {expected}",
                        got[row * n + col]
                    );
                }
            }
        }
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_resident_f32_i8_gemm_unpacks_signed_bytes_and_partial_vectors_exactly() {
    if !crate::ops::cuda::is_available() {
        return;
    }
    for m in [8, 128] {
        let n = 129;
        for k in [1025, 1026, 1027] {
            let positions = [0, 1, 2, 3, k - 1, k - 2, k - 3, k / 2];
            let columns: Vec<_> = (0..m).map(|row| positions[row % positions.len()]).collect();
            let mut a = vec![0.0f32; m * k];
            for (row, &column) in columns.iter().enumerate() {
                a[row * k + column] = 1.0;
            }
            let b: Vec<_> = (0..n * k).map(|i| ((i % 256) as i32 - 128) as i8).collect();
            let input = crate::ops::cuda::upload_f32(&a).unwrap();
            let weight = crate::ops::cuda::upload_i8_storage(&b).unwrap();
            for scale in [0.013f32, 0.023, 1e-30, 1e30] {
                let output =
                    crate::ops::cuda::matmul_f32_i8_buffer_no_host(&input, &weight, scale, m, n, k)
                        .unwrap();
                let got = crate::ops::cuda::download_f32(&output).unwrap();
                for (row, &column) in columns.iter().enumerate() {
                    for col in 0..n {
                        let expected = f32::from(b[col * k + column]) * scale;
                        assert_eq!(
                            got[row * n + col].to_bits(),
                            expected.to_bits(),
                            "signed byte/tail mismatch at k={k}, row={row}, col={col}, scale={scale}"
                        );
                    }
                }
            }
        }
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_resident_f32_i8_gemm_training_updates_tiled_weights_and_decreases_loss() {
    use crate::autograd::set_strict_device_execution_scoped;
    use crate::loss::MSELoss;
    use crate::optim::{Optimizer, SGD};
    use crate::precision::ParameterQuantization;
    if !crate::ops::cuda::is_available() {
        return;
    }
    let _cuda_guard = crate::ops::cuda::set_enabled_scoped(true);
    let _strict_guard = set_strict_device_execution_scoped(true);
    for m in [8, 128] {
        let (n, k) = (3073, 2048);
        let amplitude = if m == 8 { 32.0 } else { 128.0 };
        let mut values = vec![0.0f32; m * k];
        for row in 0..m {
            let column = if m == 8 { row * 257 } else { (row * 17) % k };
            values[row * k + column] = amplitude;
        }
        let input = Tensor::from_array_no_grad(
            ndarray::Array::from_shape_vec(ndarray::IxDyn(&[m, k]), values).unwrap(),
        )
        .to_cuda();
        let target = Tensor::from_array_no_grad(
            ndarray::Array::from_shape_vec(
                ndarray::IxDyn(&[m, n]),
                (0..m * n)
                    .map(|i| ((i * 13 % 97) as f32 - 48.0) / 48.0)
                    .collect(),
            )
            .unwrap(),
        )
        .to_cuda();
        let weight = Tensor::parameter_with_quantization(
            ndarray::Array::zeros(ndarray::IxDyn(&[n, k])),
            ParameterQuantization::Int8.with_scale(0.001),
        )
        .to_cuda();
        let mut optimizer = SGD::new_with_dtype(vec![weight.clone()], 2.0, DType::F32)
            .with_momentum(0.3)
            .with_dynamic_i8_master_weights();
        let mut losses = Vec::new();
        for _ in 0..32 {
            let output = matmul(&input, &weight);
            let loss = MSELoss::apply(&output, &target);
            losses.push(loss.data_ref().iter().copied().next().unwrap());
            loss.backward();
            assert!(weight.cloned_cuda_f32_grad().is_some());
            assert!(!weight.has_host_grad());
            optimizer.step();
            optimizer.zero_grad();
            assert_eq!(weight.dtype(), DType::I8);
            assert!(weight.quantization_scale().unwrap().is_finite());
        }
        println!(
            "tiled_i8_gemm_training {}",
            serde_json::json!({
                "m": m, "n": n, "k": k, "losses": losses,
            })
        );
        assert!(losses.iter().all(|v| v.is_finite()));
        assert!(losses[31] < losses[0] * 0.01);
        let head = losses[..8].iter().sum::<f32>();
        let tail = losses[24..].iter().sum::<f32>();
        assert!(tail < head * 0.01);
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_resident_f32_i8_gemm_does_not_hide_nonfinite_activations() {
    if !crate::ops::cuda::is_available() {
        return;
    }
    for m in [8, 128] {
        let (n, k) = (512, 512);
        let mut a = vec![0.01f32; m * k];
        a[0] = f32::NAN;
        a[k + 3] = f32::INFINITY;
        let b = vec![1i8; n * k];
        let a_buf = crate::ops::cuda::upload_f32(&a).unwrap();
        let b_buf = crate::ops::cuda::upload_i8_storage(&b).unwrap();
        let out =
            crate::ops::cuda::matmul_f32_i8_buffer_no_host(&a_buf, &b_buf, 0.013, m, n, k).unwrap();
        let got = crate::ops::cuda::download_f32(&out).unwrap();
        assert!(got[..2 * n].iter().all(|x| !x.is_finite()));
        assert!(got[2 * n..].iter().all(|x| x.is_finite()));
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_resident_bf16_i8_dynamic_matmul_stays_within_quantization_bound() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let (m, n, k) = (8, 12, 32);
    let b_scale = 0.03125f32;
    let a = (0..m * k)
        .map(|i| bf16::from_f32(((i * 37 % 257) as f32 - 128.0) / 31.0))
        .collect::<Vec<_>>();
    let b = (0..n * k)
        .map(|i| ((((i * 29 + 11) % 255) as i32) - 127) as i8)
        .collect::<Vec<_>>();
    let a_bits = a.iter().map(|value| value.to_bits()).collect::<Vec<_>>();
    let a_buf =
        crate::ops::cuda::upload_u16_storage(&a_bits).expect("upload dynamic bf16 matmul a");
    let b_buf = crate::ops::cuda::upload_i8_storage(&b).expect("upload dynamic i8 matmul b");
    let out =
        crate::ops::cuda::matmul_bf16_i8_dynamic_buffer_no_host(&a_buf, &b_buf, b_scale, m, n, k)
            .expect("CUDA dynamic BF16xI8 matmul");
    let got = crate::ops::cuda::download_f32(&out).expect("download dynamic BF16xI8 matmul");

    for row in 0..m {
        let row_max = a[row * k..(row + 1) * k]
            .iter()
            .map(|value| value.to_f32().abs())
            .fold(0.0f32, f32::max);
        let activation_scale = (row_max / 127.0).max(f32::MIN_POSITIVE);
        for col in 0..n {
            let mut expected = 0.0f32;
            let mut weight_l1 = 0.0f32;
            for kk in 0..k {
                let weight = (b[col * k + kk] as f32) * b_scale;
                expected += a[row * k + kk].to_f32() * weight;
                weight_l1 += weight.abs();
            }
            let error_bound = 0.5 * activation_scale * weight_l1 + 2.0e-4;
            let error = (got[row * n + col] - expected).abs();
            assert!(
                error <= error_bound,
                "dynamic BF16xI8 error {error} exceeded bound {error_bound} at row={row}, col={col}"
            );
        }
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_resident_f16_i8_dynamic_matmul_stays_within_quantization_bound() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let (m, n, k) = (8, 12, 32);
    let b_scale = 0.03125f32;
    let a = (0..m * k)
        .map(|i| f16::from_f32(((i * 37 % 257) as f32 - 128.0) / 31.0))
        .collect::<Vec<_>>();
    let b = (0..n * k)
        .map(|i| ((((i * 29 + 11) % 255) as i32) - 127) as i8)
        .collect::<Vec<_>>();
    let a_bits = a.iter().map(|value| value.to_bits()).collect::<Vec<_>>();
    let a_buf = crate::ops::cuda::upload_u16_storage(&a_bits).expect("upload dynamic f16 matmul a");
    let b_buf = crate::ops::cuda::upload_i8_storage(&b).expect("upload dynamic i8 matmul b");
    let out =
        crate::ops::cuda::matmul_f16_i8_dynamic_buffer_no_host(&a_buf, &b_buf, b_scale, m, n, k)
            .expect("CUDA dynamic F16xI8 matmul");
    let got = crate::ops::cuda::download_f32(&out).expect("download dynamic F16xI8 matmul");

    for row in 0..m {
        let row_max = a[row * k..(row + 1) * k]
            .iter()
            .map(|value| value.to_f32().abs())
            .fold(0.0f32, f32::max);
        let activation_scale = (row_max / 127.0).max(f32::MIN_POSITIVE);
        for col in 0..n {
            let mut expected = 0.0f32;
            let mut weight_l1 = 0.0f32;
            for kk in 0..k {
                let weight = (b[col * k + kk] as f32) * b_scale;
                expected += a[row * k + kk].to_f32() * weight;
                weight_l1 += weight.abs();
            }
            let error_bound = 0.5 * activation_scale * weight_l1 + 2.0e-4;
            let error = (got[row * n + col] - expected).abs();
            assert!(
                error <= error_bound,
                "dynamic F16xI8 error {error} exceeded bound {error_bound} at row={row}, col={col}"
            );
        }
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_dynamic_floatlike_i8_matmul_preserves_non_finite_failure_signal() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let (m, n, k) = (8, 12, 32);
    let mut a = vec![bf16::from_f32(0.5); m * k];
    a[7] = bf16::INFINITY;
    let b = vec![1i8; n * k];
    let a_bits = a.iter().map(|value| value.to_bits()).collect::<Vec<_>>();
    let a_buf =
        crate::ops::cuda::upload_u16_storage(&a_bits).expect("upload non-finite bf16 matmul a");
    let b_buf = crate::ops::cuda::upload_i8_storage(&b).expect("upload non-finite i8 matmul b");
    let out =
        crate::ops::cuda::matmul_bf16_i8_dynamic_buffer_no_host(&a_buf, &b_buf, 0.03125, m, n, k)
            .expect("CUDA dynamic non-finite BF16xI8 matmul");
    let got = crate::ops::cuda::download_f32(&out).expect("download non-finite BF16xI8 matmul");

    assert!(
        got[..n].iter().all(|value| !value.is_finite()),
        "dynamic quantization must not hide a non-finite activation row"
    );
    assert!(
        got[n..].iter().all(|value| value.is_finite()),
        "finite activation rows should remain finite"
    );
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_training_bf16_i8_matmul_uses_direct_forward_for_f32_backward_consistency() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let (m, n, k) = (8, 12, 32);
    let a_values = (0..m * k)
        .map(|i| ((i * 37 % 257) as f32 - 128.0) / 31.0)
        .collect::<Vec<_>>();
    let b_values = (0..n * k)
        .map(|i| ((i * 29 % 251) as f32 - 125.0) / 29.0)
        .collect::<Vec<_>>();
    let a_cpu = make_training_tensor(&[m, k], a_values, DType::BF16);
    let b_cpu = make_training_tensor(&[n, k], b_values, DType::I8);
    let a_cuda = a_cpu.to_cuda();
    let b_cuda = b_cpu.to_cuda();

    crate::ops::cuda::set_enabled(true);
    crate::autograd::set_strict_device_execution(true);
    let out_cuda = matmul(&a_cuda, &b_cuda);
    crate::autograd::set_strict_device_execution(false);
    crate::ops::cuda::set_enabled(false);
    let out_cpu = matmul(&a_cpu, &b_cpu);

    for (got, expected) in out_cuda.data_ref().iter().zip(out_cpu.data_ref().iter()) {
        assert!(
            (got - expected).abs() < 3e-2,
            "training BF16xI8 forward drifted from direct CPU function: got={got}, expected={expected}"
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_resident_bf16_i8_matmul_matches_cpu_reference_without_host_f32() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let a = make_tensor(
        &[4, 7],
        vec![
            1.0, -2.0, 0.5, 3.0, -1.25, 0.75, 2.5, -0.5, 1.5, -3.0, 0.25, 2.0, -1.0, 0.125, 3.5,
            -0.75, 1.25, -2.25, 0.5, 1.0, -1.5, 0.0, 2.75, -0.25, 1.75, -3.5, 0.625, 2.25,
        ],
        DType::BF16,
    )
    .to_cuda();
    let b = make_tensor(
        &[5, 7],
        vec![
            0.5, -1.0, 2.0, -0.25, 1.5, -2.5, 0.75, -1.5, 0.25, 1.25, -2.0, 0.5, 3.0, -0.75, 2.5,
            -0.5, -1.25, 1.0, -3.0, 0.125, 1.75, -2.25, 1.5, 0.5, -0.625, 2.25, -1.0, 3.5, 1.0,
            2.0, -3.0, 0.75, -1.5, 0.25, -2.5,
        ],
        DType::I8,
    )
    .to_cuda();
    assert_eq!(a.dtype(), DType::BF16);
    assert_eq!(b.dtype(), DType::I8);
    assert!(a.cloned_cuda_native_lowp_buffer().is_some());
    assert!(b.cloned_cuda_native_lowp_buffer().is_some());
    assert!(!a.has_host_f32_data());
    assert!(!b.has_host_f32_data());

    crate::ops::cuda::set_enabled(true);
    crate::autograd::set_strict_device_execution(true);
    let out = no_grad(|| matmul(&a, &b));
    crate::autograd::set_strict_device_execution(false);
    crate::ops::cuda::set_enabled(false);

    assert!(out.is_cuda());
    assert_eq!(out.dtype(), DType::F32);
    assert!(!a.has_host_f32_data());
    assert!(
        !b.has_host_f32_data(),
        "mixed BF16xI8 CUDA matmul should not materialize I8 weights as host f32"
    );
    assert!(
        !out.has_host_f32_data(),
        "mixed BF16xI8 CUDA matmul should keep f32 output resident until requested"
    );
    {
        let inner = out.0.borrow();
        assert!(inner.cuda_f32_data.is_some());
        assert!(inner.cuda_bf16_data.is_none());
        assert!(inner.cuda_i8_data.is_none());
    }

    let reference = no_grad(|| matmul(&a.to_cpu(), &b.to_cpu()));
    for (got, expect) in out.data_ref().iter().zip(reference.data_ref().iter()) {
        assert!(
            (got - expect).abs() < 3e-2,
            "mixed BF16xI8 CUDA matmul drifted: got={got}, expect={expect}"
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_resident_f16_i8_matmul_matches_cpu_reference_without_host_f32() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let a = make_tensor(&[4, 7], sample_f32(28), DType::F16).to_cuda();
    let b = make_tensor(&[5, 7], sample_f32(35), DType::I8).to_cuda();
    assert_eq!(a.dtype(), DType::F16);
    assert_eq!(b.dtype(), DType::I8);
    assert!(a.cloned_cuda_native_lowp_buffer().is_some());
    assert!(b.cloned_cuda_native_lowp_buffer().is_some());
    assert!(!a.has_host_f32_data());
    assert!(!b.has_host_f32_data());

    crate::ops::cuda::set_enabled(true);
    crate::autograd::set_strict_device_execution(true);
    let out = no_grad(|| matmul(&a, &b));
    crate::autograd::set_strict_device_execution(false);
    crate::ops::cuda::set_enabled(false);

    assert!(out.is_cuda());
    assert_eq!(out.dtype(), DType::F32);
    assert!(!a.has_host_f32_data());
    assert!(
        !b.has_host_f32_data(),
        "mixed F16xI8 CUDA matmul should not materialize I8 weights as host f32"
    );
    assert!(
        !out.has_host_f32_data(),
        "mixed F16xI8 CUDA matmul should keep f32 output resident until requested"
    );

    let reference = no_grad(|| matmul(&a.to_cpu(), &b.to_cpu()));
    for (got, expect) in out.data_ref().iter().zip(reference.data_ref().iter()) {
        assert!(
            (got - expect).abs() < 3e-2,
            "mixed F16xI8 CUDA matmul drifted: got={got}, expect={expect}"
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_resident_f32_i8_matmul_matches_cpu_reference_without_host_f32_weight() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let a = make_tensor(&[3, 6], sample_f32(18), DType::F32).to_cuda();
    let b = make_tensor(
        &[4, 6],
        vec![
            0.5, -1.0, 2.0, -0.25, 1.5, -2.5, -1.5, 0.25, 1.25, -2.0, 0.5, 3.0, 2.5, -0.5, -1.25,
            1.0, -3.0, 0.125, -2.25, 1.5, 0.5, -0.625, 2.25, -1.0,
        ],
        DType::I8,
    )
    .to_cuda();
    assert_eq!(a.dtype(), DType::F32);
    assert_eq!(b.dtype(), DType::I8);
    assert!(a.cloned_cuda_f32_buffer().is_some());
    assert!(b.cloned_cuda_native_lowp_buffer().is_some());
    assert!(!b.has_host_f32_data());

    crate::ops::cuda::set_enabled(true);
    crate::autograd::set_strict_device_execution(true);
    let out = no_grad(|| matmul(&a, &b));
    crate::autograd::set_strict_device_execution(false);
    crate::ops::cuda::set_enabled(false);

    assert!(out.is_cuda());
    assert_eq!(out.dtype(), DType::F32);
    assert!(
        !b.has_host_f32_data(),
        "F32xI8 CUDA matmul should not materialize I8 weights as host f32"
    );
    assert!(
        !out.has_host_f32_data(),
        "F32xI8 CUDA matmul should keep f32 output resident until requested"
    );
    let reference = no_grad(|| matmul(&a.to_cpu(), &b.to_cpu()));
    for (got, expect) in out.data_ref().iter().zip(reference.data_ref().iter()) {
        assert!(
            (got - expect).abs() < 3e-2,
            "mixed F32xI8 CUDA matmul drifted: got={got}, expect={expect}"
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_resident_i8_floatlike_matmul_matches_cpu_reference_without_host_f32_input() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let k = 31;
    let n = 9;
    for (m, rhs_dtype) in [1usize, 5]
        .into_iter()
        .flat_map(|m| [DType::F16, DType::BF16, DType::F32].map(|rhs_dtype| (m, rhs_dtype)))
    {
        let a = make_tensor(&[m, k], sample_f32(m * k), DType::I8).to_cuda();
        let b = make_tensor(&[n, k], sample_f32(n * k), rhs_dtype).to_cuda();
        assert_eq!(a.dtype(), DType::I8);
        assert_eq!(b.dtype(), rhs_dtype);
        assert!(a.cloned_cuda_native_lowp_buffer().is_some());
        assert!(!a.has_host_f32_data());
        if rhs_dtype != DType::F32 {
            assert!(b.cloned_cuda_native_lowp_buffer().is_some());
            assert!(!b.has_host_f32_data());
        } else {
            assert!(b.cloned_cuda_f32_buffer().is_some());
        }

        crate::ops::cuda::set_enabled(true);
        crate::autograd::set_strict_device_execution(true);
        let out = no_grad(|| matmul(&a, &b));
        crate::autograd::set_strict_device_execution(false);
        crate::ops::cuda::set_enabled(false);

        assert!(out.is_cuda());
        assert_eq!(out.dtype(), DType::F32);
        assert!(
            !a.has_host_f32_data(),
            "I8x{rhs_dtype:?} CUDA matmul should not materialize I8 input as host f32"
        );
        assert!(
            !out.has_host_f32_data(),
            "I8x{rhs_dtype:?} CUDA matmul should keep f32 output resident until requested"
        );

        let reference = no_grad(|| matmul(&a.to_cpu(), &b.to_cpu()));
        for (got, expect) in out.data_ref().iter().zip(reference.data_ref().iter()) {
            assert!(
                (got - expect).abs() < 5e-2,
                "mixed I8x{rhs_dtype:?} CUDA matmul drifted for m={m}: got={got}, expect={expect}"
            );
        }
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_resident_single_row_float_i8_matvec_matches_cpu_reference_without_host_f32_weight() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let k = 257;
    let n = 7;
    for dtype in [DType::F16, DType::BF16, DType::F32] {
        let a = make_tensor(&[1, k], sample_f32(k), dtype).to_cuda();
        let b = make_tensor(&[n, k], sample_f32(n * k), DType::I8).to_cuda();
        assert_eq!(a.dtype(), dtype);
        assert_eq!(b.dtype(), DType::I8);
        assert!(b.cloned_cuda_native_lowp_buffer().is_some());
        assert!(!b.has_host_f32_data());

        crate::ops::cuda::set_enabled(true);
        crate::autograd::set_strict_device_execution(true);
        let out = no_grad(|| matmul(&a, &b));
        crate::autograd::set_strict_device_execution(false);
        crate::ops::cuda::set_enabled(false);

        assert!(out.is_cuda());
        assert_eq!(out.dtype(), DType::F32);
        assert!(
            !b.has_host_f32_data(),
            "{dtype:?}xI8 CUDA matvec should not materialize I8 weights as host f32"
        );
        let reference = no_grad(|| matmul(&a.to_cpu(), &b.to_cpu()));
        for (got, expect) in out.data_ref().iter().zip(reference.data_ref().iter()) {
            assert!(
                (got - expect).abs() < 5e-2,
                "single-row {dtype:?}xI8 CUDA matvec drifted: got={got}, expect={expect}"
            );
        }
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_resident_mixed_i8_argmax_matches_cpu_logits_argmax_without_host_f32_weight() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let k = 19;
    let vocab = 13;
    for (batch, dtype) in [DType::F16, DType::BF16, DType::F32, DType::I8]
        .into_iter()
        .flat_map(|dtype| [(1usize, dtype), (2usize, dtype)])
    {
        let a = make_tensor(&[batch, k], sample_f32(batch * k), dtype).to_cuda();
        let b = make_tensor(&[vocab, k], sample_f32(vocab * k), DType::I8).to_cuda();
        let (weight_dtype, weight_buf, weight_scale) = b
            .cloned_cuda_native_lowp_buffer()
            .expect("resident i8 weight");
        assert_eq!(weight_dtype, DType::I8);
        assert!(!b.has_host_f32_data());

        let got = if dtype == DType::BF16 {
            let (input_dtype, input_buf, _) = a
                .cloned_cuda_native_lowp_buffer()
                .expect("resident bf16 input");
            assert_eq!(input_dtype, DType::BF16);
            crate::ops::cuda::matvec_argmax_bf16_i8(
                &input_buf,
                &weight_buf,
                weight_scale.expect("i8 weight scale"),
                batch,
                vocab,
                k,
            )
        } else if dtype == DType::F16 {
            let (input_dtype, input_buf, _) = a
                .cloned_cuda_native_lowp_buffer()
                .expect("resident f16 input");
            assert_eq!(input_dtype, DType::F16);
            crate::ops::cuda::matvec_argmax_f16_i8(
                &input_buf,
                &weight_buf,
                weight_scale.expect("i8 weight scale"),
                batch,
                vocab,
                k,
            )
        } else if dtype == DType::I8 {
            let (input_dtype, input_buf, input_scale) = a
                .cloned_cuda_native_lowp_buffer()
                .expect("resident i8 input");
            assert_eq!(input_dtype, DType::I8);
            crate::ops::cuda::matvec_argmax_i8_i8(
                &input_buf,
                input_scale.expect("i8 input scale"),
                &weight_buf,
                weight_scale.expect("i8 weight scale"),
                batch,
                vocab,
                k,
            )
        } else {
            let input_buf = a.cloned_cuda_f32_buffer().expect("resident f32 input");
            crate::ops::cuda::matvec_argmax_f32_i8(
                &input_buf,
                &weight_buf,
                weight_scale.expect("i8 weight scale"),
                batch,
                vocab,
                k,
            )
        }
        .expect("CUDA mixed argmax");

        let reference = no_grad(|| matmul(&a.to_cpu(), &b.to_cpu()));
        let logits = reference.data_ref();
        let expected = (0..batch)
            .map(|row| {
                let row_logits = logits.slice(ndarray::s![row, ..]);
                row_logits
                    .iter()
                    .enumerate()
                    .max_by(|(_, lhs), (_, rhs)| lhs.total_cmp(rhs))
                    .map(|(idx, _)| idx)
                    .expect("non-empty vocab")
            })
            .collect::<Vec<_>>();

        assert_eq!(got, expected, "{dtype:?}xI8 CUDA argmax drifted");
        assert!(
            !b.has_host_f32_data(),
            "{dtype:?}xI8 CUDA argmax should not materialize I8 weights as host f32"
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_i8_weight_argmax_rejects_invalid_scales() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let batch = 2;
    let vocab = 3;
    let hidden = 5;
    let input_u16 =
        crate::ops::cuda::upload_u16_storage(&vec![0; batch * hidden]).expect("upload u16 input");
    let input_f32 =
        crate::ops::cuda::upload_f32(&vec![0.0; batch * hidden]).expect("upload f32 input");
    let input_i8 =
        crate::ops::cuda::upload_i8_storage(&vec![0; batch * hidden]).expect("upload i8 input");
    let weight_i8 =
        crate::ops::cuda::upload_i8_storage(&vec![0; vocab * hidden]).expect("upload i8 weight");

    assert!(
        crate::ops::cuda::matvec_argmax_bf16_i8(
            &input_u16,
            &weight_i8,
            f32::NAN,
            batch,
            vocab,
            hidden,
        )
        .is_err()
    );
    assert!(
        crate::ops::cuda::matvec_argmax_f16_i8(&input_u16, &weight_i8, 0.0, batch, vocab, hidden,)
            .is_err()
    );
    assert!(
        crate::ops::cuda::matvec_argmax_f32_i8(&input_f32, &weight_i8, -1.0, batch, vocab, hidden,)
            .is_err()
    );
    assert!(
        crate::ops::cuda::matvec_argmax_i8_i8(
            &input_i8,
            f32::INFINITY,
            &weight_i8,
            0.05,
            batch,
            vocab,
            hidden,
        )
        .is_err()
    );
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_resident_i8_matmul_typed_output_matches_dynamic_quantized_reference() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let m = 9;
    let n = 19;
    let k = 35;
    let a_scale = 0.03125f32;
    let b_scale = 0.046875f32;
    let a = (0..m * k)
        .map(|i| ((((i * 17 + 3) % 255) as i32) - 127) as i8)
        .collect::<Vec<_>>();
    let b = (0..n * k)
        .map(|i| ((((i * 29 + 11) % 255) as i32) - 127) as i8)
        .collect::<Vec<_>>();
    let a_buf = crate::ops::cuda::upload_i8_storage(&a).expect("upload typed i8 matmul a");
    let b_buf = crate::ops::cuda::upload_i8_storage(&b).expect("upload typed i8 matmul b");

    let (out, got_scale) = crate::ops::cuda::matmul_i8_typed_output_buffer_no_host(
        &a_buf, a_scale, &b_buf, b_scale, m, n, k,
    )
    .expect("CUDA resident typed-output i8 matmul");
    let got = crate::ops::cuda::download_i8_storage(&out).expect("download typed i8 matmul");

    let mut values = vec![0.0f32; m * n];
    let mut max_abs = 0.0f32;
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0i32;
            for kk in 0..k {
                acc += (a[row * k + kk] as i32) * (b[col * k + kk] as i32);
            }
            let value = (acc as f32) * a_scale * b_scale;
            values[row * n + col] = value;
            max_abs = max_abs.max(value.abs());
        }
    }
    let expected_scale = if max_abs > 0.0 {
        (max_abs / 127.0).max(f32::MIN_POSITIVE)
    } else {
        1.0
    };
    assert!(
        (got_scale - expected_scale).abs() <= 1e-6,
        "typed i8 matmul scale drifted: got={got_scale}, expected={expected_scale}"
    );

    let expected = values
        .iter()
        .map(|value| {
            let q = (value / expected_scale).round().clamp(-127.0, 127.0);
            q as i8
        })
        .collect::<Vec<_>>();
    assert_eq!(got, expected, "typed i8 matmul quantized output drifted");
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_resident_i8_matmul_typed_output_zero_absmax_uses_unit_scale() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let m = 5;
    let n = 7;
    let k = 11;
    let a_buf =
        crate::ops::cuda::upload_i8_storage(&vec![0; m * k]).expect("upload zero i8 matmul a");
    let b_buf =
        crate::ops::cuda::upload_i8_storage(&vec![0; n * k]).expect("upload zero i8 matmul b");

    let (out, scale) = crate::ops::cuda::matmul_i8_typed_output_buffer_no_host(
        &a_buf, 0.03125, &b_buf, 0.046875, m, n, k,
    )
    .expect("CUDA zero typed-output i8 matmul");
    let got = crate::ops::cuda::download_i8_storage(&out).expect("download zero typed i8 matmul");

    assert_eq!(scale, 1.0);
    assert_eq!(got, vec![0; m * n]);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_resident_i8_batch_matmul_typed_output_matches_dynamic_quantized_reference() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let batch_count = 2;
    let m = 3;
    let n = 4;
    let k = 5;
    let lhs_scale = 0.03125f32;
    let rhs_scale = 0.046875f32;
    let lhs = (0..batch_count * m * k)
        .map(|i| ((((i * 17 + 3) % 255) as i32) - 127) as i8)
        .collect::<Vec<_>>();
    let rhs = (0..batch_count * k * n)
        .map(|i| ((((i * 29 + 11) % 255) as i32) - 127) as i8)
        .collect::<Vec<_>>();
    let lhs_buf = crate::ops::cuda::upload_i8_storage(&lhs).expect("upload typed i8 batch lhs");
    let rhs_buf = crate::ops::cuda::upload_i8_storage(&rhs).expect("upload typed i8 batch rhs");

    let (out, got_scale) = crate::ops::cuda::batch_matmul_i8_typed_output_buffer_no_host(
        &lhs_buf,
        lhs_scale,
        &rhs_buf,
        rhs_scale,
        batch_count,
        m,
        n,
        k,
    )
    .expect("CUDA resident typed-output i8 batch matmul");
    let got = crate::ops::cuda::download_i8_storage(&out).expect("download typed i8 batch matmul");

    let mut values = vec![0.0f32; batch_count * m * n];
    let mut max_abs = 0.0f32;
    for batch in 0..batch_count {
        for row in 0..m {
            for col in 0..n {
                let mut acc = 0i32;
                for kk in 0..k {
                    acc += (lhs[batch * m * k + row * k + kk] as i32)
                        * (rhs[batch * k * n + kk * n + col] as i32);
                }
                let idx = batch * m * n + row * n + col;
                let value = (acc as f32) * lhs_scale * rhs_scale;
                values[idx] = value;
                max_abs = max_abs.max(value.abs());
            }
        }
    }
    let expected_scale = if max_abs > 0.0 {
        (max_abs / 127.0).max(f32::MIN_POSITIVE)
    } else {
        1.0
    };
    assert!(
        (got_scale - expected_scale).abs() <= 1e-6,
        "typed i8 batch matmul scale drifted: got={got_scale}, expected={expected_scale}"
    );

    let expected = values
        .iter()
        .map(|value| {
            let q = (value / expected_scale).round().clamp(-127.0, 127.0);
            q as i8
        })
        .collect::<Vec<_>>();
    assert_eq!(
        got, expected,
        "typed i8 batch matmul quantized output drifted"
    );
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_resident_i8_batch_matmul_f32_output_handles_tiled_tail_dimensions() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let batch_count = 2;
    let m = 5;
    let n = 7;
    let k = 35;
    let lhs_scale = 0.03125f32;
    let rhs_scale = 0.046875f32;
    let lhs = (0..batch_count * m * k)
        .map(|i| ((((i * 17 + 3) % 255) as i32) - 127) as i8)
        .collect::<Vec<_>>();
    let rhs = (0..batch_count * k * n)
        .map(|i| ((((i * 29 + 11) % 255) as i32) - 127) as i8)
        .collect::<Vec<_>>();
    let lhs_buf = crate::ops::cuda::upload_i8_storage(&lhs).expect("upload tiled i8 batch lhs");
    let rhs_buf = crate::ops::cuda::upload_i8_storage(&rhs).expect("upload tiled i8 batch rhs");

    let out = crate::ops::cuda::batch_matmul_i8_buffer_no_host(
        &lhs_buf,
        lhs_scale,
        &rhs_buf,
        rhs_scale,
        batch_count,
        m,
        n,
        k,
    )
    .expect("CUDA resident f32-output i8 batch matmul");
    let got = crate::ops::cuda::download_f32(&out).expect("download tiled i8 batch matmul");

    for batch in 0..batch_count {
        for row in 0..m {
            for col in 0..n {
                let mut acc = 0i32;
                for kk in 0..k {
                    acc += (lhs[batch * m * k + row * k + kk] as i32)
                        * (rhs[batch * k * n + kk * n + col] as i32);
                }
                let idx = batch * m * n + row * n + col;
                let expected = (acc as f32) * lhs_scale * rhs_scale;
                assert_eq!(
                    got[idx], expected,
                    "tiled i8 batch_matmul drifted at batch={batch}, row={row}, col={col}"
                );
            }
        }
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_resident_aligned_i8_batch_matmul_matches_i32_reference() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let (batch_count, m, n, k) = (2, 32, 64, 256);
    let lhs_scale = 0.03125f32;
    let rhs_scale = 0.0625f32;
    let lhs = (0..batch_count * m * k)
        .map(|i| ((((i * 17 + 3) % 255) as i32) - 127) as i8)
        .collect::<Vec<_>>();
    let rhs = (0..batch_count * k * n)
        .map(|i| ((((i * 29 + 11) % 255) as i32) - 127) as i8)
        .collect::<Vec<_>>();
    let lhs_buf = crate::ops::cuda::upload_i8_storage(&lhs).expect("upload aligned i8 batch lhs");
    let rhs_buf = crate::ops::cuda::upload_i8_storage(&rhs).expect("upload aligned i8 batch rhs");
    let out = crate::ops::cuda::batch_matmul_i8_buffer_no_host(
        &lhs_buf,
        lhs_scale,
        &rhs_buf,
        rhs_scale,
        batch_count,
        m,
        n,
        k,
    )
    .expect("CUDA aligned resident i8 batch matmul");
    let got = crate::ops::cuda::download_f32(&out).expect("download aligned i8 batch matmul");

    for batch in 0..batch_count {
        for row in 0..m {
            for col in 0..n {
                let acc = (0..k)
                    .map(|kk| {
                        (lhs[batch * m * k + row * k + kk] as i32)
                            * (rhs[batch * k * n + kk * n + col] as i32)
                    })
                    .sum::<i32>();
                let idx = batch * m * n + row * n + col;
                assert_eq!(
                    got[idx],
                    (acc as f32) * (lhs_scale * rhs_scale),
                    "aligned cuBLAS i8 batch_matmul drifted at batch={batch}, row={row}, col={col}"
                );
            }
        }
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_batch_matmul_preserves_bf16_dtype_in_strict_mode() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let lhs = make_tensor(
        &[1, 1, 2, 3],
        vec![1.0, -2.0, 0.5, 3.0, -1.0, 2.0],
        DType::BF16,
    )
    .to_cuda();
    let rhs = make_tensor(
        &[1, 1, 3, 2],
        vec![0.5, 1.0, -1.5, 2.0, 0.25, -0.75],
        DType::BF16,
    )
    .to_cuda();

    crate::ops::cuda::set_enabled(true);
    crate::autograd::set_strict_device_execution(true);
    let out = no_grad(|| batch_matmul(&lhs, &rhs));
    crate::autograd::set_strict_device_execution(false);
    crate::ops::cuda::set_enabled(false);

    let reference = no_grad(|| batch_matmul(&lhs.to_cpu(), &rhs.to_cpu()));
    assert!(out.is_cuda());
    assert_eq!(out.dtype(), DType::BF16);
    assert!(
        !out.has_host_f32_data(),
        "bf16 CUDA batch_matmul should keep output resident until host data is requested"
    );
    {
        let inner = out.0.borrow();
        assert!(
            inner.cuda_f32_data.is_none(),
            "bf16 CUDA batch_matmul should write native bf16 output, not a resident f32 buffer"
        );
        assert!(
            inner.cuda_bf16_data.is_some(),
            "bf16 CUDA batch_matmul should keep resident bf16 storage"
        );
    }
    for (got, expect) in out.data_ref().iter().zip(reference.data_ref().iter()) {
        assert!((got - expect).abs() < 2e-2, "got {got}, expect {expect}");
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_batch_matmul_mixed_i8_matches_cpu_reference_without_host_f32_rhs() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let shape_lhs = [2, 3, 5, 131];
    let shape_rhs = [2, 3, 131, 7];
    let lhs_len = shape_lhs.iter().product::<usize>();
    let rhs_len = shape_rhs.iter().product::<usize>();
    for dtype in [DType::F16, DType::BF16, DType::F32] {
        let lhs = make_tensor(&shape_lhs, sample_f32(lhs_len), dtype).to_cuda();
        let rhs = make_tensor(&shape_rhs, sample_f32(rhs_len), DType::I8).to_cuda();
        assert_eq!(lhs.dtype(), dtype);
        assert_eq!(rhs.dtype(), DType::I8);
        assert!(rhs.cloned_cuda_native_lowp_buffer().is_some());
        assert!(!rhs.has_host_f32_data());

        crate::ops::cuda::set_enabled(true);
        crate::autograd::set_strict_device_execution(true);
        let out = no_grad(|| batch_matmul(&lhs, &rhs));
        crate::autograd::set_strict_device_execution(false);
        crate::ops::cuda::set_enabled(false);

        assert!(out.is_cuda());
        assert_eq!(out.dtype(), DType::F32);
        assert!(
            !rhs.has_host_f32_data(),
            "{dtype:?}xI8 CUDA batch_matmul should not materialize I8 rhs as host f32"
        );
        assert!(
            !out.has_host_f32_data(),
            "{dtype:?}xI8 CUDA batch_matmul should keep f32 output resident until requested"
        );

        let reference = no_grad(|| batch_matmul(&lhs.to_cpu(), &rhs.to_cpu()));
        for (got, expect) in out.data_ref().iter().zip(reference.data_ref().iter()) {
            assert!(
                (got - expect).abs() < 5e-2,
                "mixed {dtype:?}xI8 CUDA batch_matmul drifted: got={got}, expect={expect}"
            );
        }
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_batch_matmul_i8_floatlike_matches_cpu_reference_without_host_f32_lhs() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let shape_lhs = [2, 3, 5, 131];
    let shape_rhs = [2, 3, 131, 7];
    let lhs_len = shape_lhs.iter().product::<usize>();
    let rhs_len = shape_rhs.iter().product::<usize>();
    for dtype in [DType::F16, DType::BF16, DType::F32] {
        let lhs = make_tensor(&shape_lhs, sample_f32(lhs_len), DType::I8).to_cuda();
        let rhs = make_tensor(&shape_rhs, sample_f32(rhs_len), dtype).to_cuda();
        assert_eq!(lhs.dtype(), DType::I8);
        assert_eq!(rhs.dtype(), dtype);
        assert!(lhs.cloned_cuda_native_lowp_buffer().is_some());
        assert!(!lhs.has_host_f32_data());
        if dtype != DType::F32 {
            assert!(rhs.cloned_cuda_native_lowp_buffer().is_some());
            assert!(!rhs.has_host_f32_data());
        } else {
            assert!(rhs.cloned_cuda_f32_buffer().is_some());
        }

        crate::ops::cuda::set_enabled(true);
        crate::autograd::set_strict_device_execution(true);
        let out = no_grad(|| batch_matmul(&lhs, &rhs));
        crate::autograd::set_strict_device_execution(false);
        crate::ops::cuda::set_enabled(false);

        assert!(out.is_cuda());
        assert_eq!(out.dtype(), DType::F32);
        assert!(
            !lhs.has_host_f32_data(),
            "I8x{dtype:?} CUDA batch_matmul should not materialize I8 lhs as host f32"
        );
        assert!(
            !out.has_host_f32_data(),
            "I8x{dtype:?} CUDA batch_matmul should keep f32 output resident until requested"
        );

        let reference = no_grad(|| batch_matmul(&lhs.to_cpu(), &rhs.to_cpu()));
        for (got, expect) in out.data_ref().iter().zip(reference.data_ref().iter()) {
            assert!(
                (got - expect).abs() < 5e-2,
                "mixed I8x{dtype:?} CUDA batch_matmul drifted: got={got}, expect={expect}"
            );
        }
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_batch_matmul_lowp_uses_resident_forward_buffer_in_strict_mode() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    crate::ops::cuda::set_enabled(true);
    crate::autograd::set_strict_device_execution(true);

    for dtype in [DType::F16, DType::BF16, DType::I8] {
        let lhs = make_tensor(&[1, 1, 2, 3], vec![1.0, -2.0, 0.5, 3.0, -1.0, 2.0], dtype).to_cuda();
        let rhs =
            make_tensor(&[1, 1, 3, 2], vec![0.5, 1.0, -1.5, 2.0, 0.25, -0.75], dtype).to_cuda();
        assert!(lhs.cloned_cuda_native_lowp_buffer().is_some());
        assert!(rhs.cloned_cuda_native_lowp_buffer().is_some());
        assert!(!lhs.has_host_f32_data());
        assert!(!rhs.has_host_f32_data());

        let out = no_grad(|| batch_matmul(&lhs, &rhs));

        assert!(out.is_cuda());
        assert_eq!(out.dtype(), dtype);
        {
            let inner = out.0.borrow();
            match dtype {
                DType::F16 => {
                    assert!(inner.cuda_f32_data.is_none());
                    assert!(inner.cuda_f16_data.is_some());
                }
                DType::BF16 => {
                    assert!(inner.cuda_f32_data.is_none());
                    assert!(inner.cuda_bf16_data.is_some());
                }
                DType::I8 => {
                    assert!(inner.cuda_f32_data.is_none());
                    assert!(inner.cuda_i8_data.is_some());
                }
                DType::F32 => unreachable!("test only covers low precision dtypes"),
            }
        }
        assert!(!lhs.has_host_f32_data());
        assert!(!rhs.has_host_f32_data());
        let reference = no_grad(|| batch_matmul(&lhs.to_cpu(), &rhs.to_cpu()));
        for (got, expect) in out.data_ref().iter().zip(reference.data_ref().iter()) {
            assert!(
                (got - expect).abs() <= 0.08,
                "{dtype:?} batch_matmul got {got}, expect {expect}"
            );
        }
    }

    crate::autograd::set_strict_device_execution(false);
    crate::ops::cuda::set_enabled(false);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_batch_matmul_lowp_backward_outputs_f32_grads() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    crate::ops::cuda::set_enabled(true);
    crate::autograd::set_strict_device_execution(true);

    for dtype in [DType::F16, DType::BF16, DType::I8] {
        let lhs = make_grad_tensor(&[1, 1, 2, 3], vec![1.0, -2.0, 0.5, 3.0, -1.0, 2.0]);
        lhs.cast_inplace(dtype);
        let lhs = lhs.to_cuda();
        let rhs = make_grad_tensor(&[1, 1, 3, 2], vec![0.5, 1.0, -1.5, 2.0, 0.25, -0.75]);
        rhs.cast_inplace(dtype);
        let rhs = rhs.to_cuda();

        let loss = sum(&batch_matmul(&lhs, &rhs));
        loss.backward();

        assert!(lhs.cloned_cuda_f32_grad().is_some());
        assert!(rhs.cloned_cuda_f32_grad().is_some());
        assert!(!lhs.has_host_f32_data());
        assert!(!rhs.has_host_f32_data());
        let lhs_grad = lhs.grad().expect("CUDA batch_matmul lhs grad");
        let rhs_grad = rhs.grad().expect("CUDA batch_matmul rhs grad");
        for (got, expected) in lhs_grad.iter().zip([1.5f32, 0.5, -0.5, 1.5, 0.5, -0.5]) {
            assert!(
                (got - expected).abs() <= 0.08,
                "{dtype:?} lhs grad got {got}, expected {expected}"
            );
        }
        for (got, expected) in rhs_grad.iter().zip([4.0f32, 4.0, -3.0, -3.0, 2.5, 2.5]) {
            assert!(
                (got - expected).abs() <= 0.08,
                "{dtype:?} rhs grad got {got}, expected {expected}"
            );
        }
    }

    crate::autograd::set_strict_device_execution(false);
    crate::ops::cuda::set_enabled(false);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_batch_matmul_mixed_i8_backward_reads_typed_data_and_writes_f32_grads() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let lhs_data = vec![1.0, -2.0, 0.5, 3.0, -1.0, 2.0];
    let rhs_data = vec![0.5, 1.0, -1.5, 2.0, 0.25, -0.75];

    crate::ops::cuda::set_enabled(true);
    crate::autograd::set_strict_device_execution(true);

    for lhs_dtype in [DType::F16, DType::F32, DType::BF16] {
        let lhs_cpu = make_grad_tensor(&[1, 1, 2, 3], lhs_data.clone());
        lhs_cpu.cast_inplace(lhs_dtype);
        let rhs_cpu = make_grad_tensor(&[1, 1, 3, 2], rhs_data.clone());
        rhs_cpu.cast_inplace(DType::I8);

        let lhs_cuda = make_grad_tensor(&[1, 1, 2, 3], lhs_data.clone());
        lhs_cuda.cast_inplace(lhs_dtype);
        let lhs_cuda = lhs_cuda.to_cuda();
        let rhs_cuda = make_grad_tensor(&[1, 1, 3, 2], rhs_data.clone());
        rhs_cuda.cast_inplace(DType::I8);
        let rhs_cuda = rhs_cuda.to_cuda();

        let loss_cuda = sum(&batch_matmul(&lhs_cuda, &rhs_cuda));
        loss_cuda.backward();

        let loss_cpu = sum(&batch_matmul(&lhs_cpu, &rhs_cpu));
        loss_cpu.backward();

        assert!(lhs_cuda.cloned_cuda_f32_grad().is_some());
        assert!(rhs_cuda.cloned_cuda_f32_grad().is_some());
        assert!(!rhs_cuda.has_host_f32_data());
        assert!(!lhs_cuda.has_host_grad());
        assert!(!rhs_cuda.has_host_grad());

        let lhs_grad = lhs_cuda.grad().expect("CUDA mixed batch_matmul lhs grad");
        let rhs_grad = rhs_cuda.grad().expect("CUDA mixed batch_matmul rhs grad");
        let lhs_ref = lhs_cpu.grad().expect("CPU mixed batch_matmul lhs grad");
        let rhs_ref = rhs_cpu.grad().expect("CPU mixed batch_matmul rhs grad");
        for (got, expect) in lhs_grad.iter().zip(lhs_ref.iter()) {
            assert!(
                (got - expect).abs() <= 0.08,
                "{lhs_dtype:?}xI8 lhs grad got {got}, expect {expect}"
            );
        }
        for (got, expect) in rhs_grad.iter().zip(rhs_ref.iter()) {
            assert!(
                (got - expect).abs() <= 0.08,
                "{lhs_dtype:?}xI8 rhs grad got {got}, expect {expect}"
            );
        }
    }

    crate::autograd::set_strict_device_execution(false);
    crate::ops::cuda::set_enabled(false);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_batch_matmul_i8_floatlike_backward_reads_typed_data_and_writes_f32_grads() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let lhs_data = vec![1.0, -2.0, 0.5, 3.0, -1.0, 2.0];
    let rhs_data = vec![0.5, 1.0, -1.5, 2.0, 0.25, -0.75];

    crate::ops::cuda::set_enabled(true);
    crate::autograd::set_strict_device_execution(true);

    for rhs_dtype in [DType::F16, DType::F32, DType::BF16] {
        let lhs_cpu = make_grad_tensor(&[1, 1, 2, 3], lhs_data.clone());
        lhs_cpu.cast_inplace(DType::I8);
        let rhs_cpu = make_grad_tensor(&[1, 1, 3, 2], rhs_data.clone());
        rhs_cpu.cast_inplace(rhs_dtype);

        let lhs_cuda = make_grad_tensor(&[1, 1, 2, 3], lhs_data.clone());
        lhs_cuda.cast_inplace(DType::I8);
        let lhs_cuda = lhs_cuda.to_cuda();
        let rhs_cuda = make_grad_tensor(&[1, 1, 3, 2], rhs_data.clone());
        rhs_cuda.cast_inplace(rhs_dtype);
        let rhs_cuda = rhs_cuda.to_cuda();

        let loss_cuda = sum(&batch_matmul(&lhs_cuda, &rhs_cuda));
        loss_cuda.backward();

        let loss_cpu = sum(&batch_matmul(&lhs_cpu, &rhs_cpu));
        loss_cpu.backward();

        assert!(lhs_cuda.cloned_cuda_f32_grad().is_some());
        assert!(rhs_cuda.cloned_cuda_f32_grad().is_some());
        assert!(!lhs_cuda.has_host_f32_data());
        assert!(!lhs_cuda.has_host_grad());
        assert!(!rhs_cuda.has_host_grad());

        let lhs_grad = lhs_cuda
            .grad()
            .expect("CUDA I8xfloatlike batch_matmul lhs grad");
        let rhs_grad = rhs_cuda
            .grad()
            .expect("CUDA I8xfloatlike batch_matmul rhs grad");
        let lhs_ref = lhs_cpu
            .grad()
            .expect("CPU I8xfloatlike batch_matmul lhs grad");
        let rhs_ref = rhs_cpu
            .grad()
            .expect("CPU I8xfloatlike batch_matmul rhs grad");
        for (got, expect) in lhs_grad.iter().zip(lhs_ref.iter()) {
            assert!(
                (got - expect).abs() <= 0.08,
                "I8x{rhs_dtype:?} batch_matmul lhs grad got {got}, expect {expect}"
            );
        }
        for (got, expect) in rhs_grad.iter().zip(rhs_ref.iter()) {
            assert!(
                (got - expect).abs() <= 0.08,
                "I8x{rhs_dtype:?} batch_matmul rhs grad got {got}, expect {expect}"
            );
        }
    }

    crate::autograd::set_strict_device_execution(false);
    crate::ops::cuda::set_enabled(false);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_batch_matmul_lowp_backward_handles_asymmetric_gradient_sizes() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    crate::ops::cuda::set_enabled(true);
    crate::autograd::set_strict_device_execution(true);

    for (m, n, lhs_dtype, rhs_dtype) in [
        (2, 5, DType::BF16, DType::BF16),
        (5, 2, DType::F16, DType::I8),
        (2, 5, DType::I8, DType::F16),
        (5, 2, DType::I8, DType::I8),
    ] {
        let k = 3;
        let lhs_data = sample_f32(m * k);
        let rhs_data = sample_f32(k * n);

        let lhs_cpu = make_grad_tensor(&[1, 1, m, k], lhs_data.clone());
        lhs_cpu.cast_inplace(lhs_dtype);
        let rhs_cpu = make_grad_tensor(&[1, 1, k, n], rhs_data.clone());
        rhs_cpu.cast_inplace(rhs_dtype);

        let lhs_cuda = make_grad_tensor(&[1, 1, m, k], lhs_data);
        lhs_cuda.cast_inplace(lhs_dtype);
        let lhs_cuda = lhs_cuda.to_cuda();
        let rhs_cuda = make_grad_tensor(&[1, 1, k, n], rhs_data);
        rhs_cuda.cast_inplace(rhs_dtype);
        let rhs_cuda = rhs_cuda.to_cuda();

        sum(&batch_matmul(&lhs_cuda, &rhs_cuda)).backward();
        sum(&batch_matmul(&lhs_cpu, &rhs_cpu)).backward();

        assert!(lhs_cuda.cloned_cuda_f32_grad().is_some());
        assert!(rhs_cuda.cloned_cuda_f32_grad().is_some());
        assert!(!lhs_cuda.has_host_grad());
        assert!(!rhs_cuda.has_host_grad());
        let lhs_grad = lhs_cuda.grad().expect("CUDA asymmetric batch lhs grad");
        let rhs_grad = rhs_cuda.grad().expect("CUDA asymmetric batch rhs grad");
        let lhs_ref = lhs_cpu.grad().expect("CPU asymmetric batch lhs grad");
        let rhs_ref = rhs_cpu.grad().expect("CPU asymmetric batch rhs grad");
        assert_eq!(lhs_grad.len(), m * k);
        assert_eq!(rhs_grad.len(), k * n);
        for (got, expect) in lhs_grad.iter().zip(lhs_ref.iter()) {
            assert!(
                (got - expect).abs() <= 0.08,
                "{lhs_dtype:?}x{rhs_dtype:?} asymmetric lhs grad got {got}, expect {expect}"
            );
        }
        for (got, expect) in rhs_grad.iter().zip(rhs_ref.iter()) {
            assert!(
                (got - expect).abs() <= 0.08,
                "{lhs_dtype:?}x{rhs_dtype:?} asymmetric rhs grad got {got}, expect {expect}"
            );
        }
    }

    crate::autograd::set_strict_device_execution(false);
    crate::ops::cuda::set_enabled(false);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_matmul_mixed_i8_backward_reads_typed_data_and_writes_f32_grads() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let lhs_data = vec![1.0, -2.0, 0.5, 3.0, -1.0, 2.0];
    let rhs_data = vec![0.5, 1.0, -1.5, 2.0, 0.25, -0.75, 1.25, -0.5, 0.75];

    crate::ops::cuda::set_enabled(true);
    crate::autograd::set_strict_device_execution(true);

    for lhs_dtype in [DType::F16, DType::F32, DType::BF16] {
        let lhs_cpu = make_grad_tensor(&[2, 3], lhs_data.clone());
        lhs_cpu.cast_inplace(lhs_dtype);
        let rhs_cpu = make_grad_tensor(&[3, 3], rhs_data.clone());
        rhs_cpu.cast_inplace(DType::I8);

        let lhs_cuda = make_grad_tensor(&[2, 3], lhs_data.clone());
        lhs_cuda.cast_inplace(lhs_dtype);
        let lhs_cuda = lhs_cuda.to_cuda();
        let rhs_cuda = make_grad_tensor(&[3, 3], rhs_data.clone());
        rhs_cuda.cast_inplace(DType::I8);
        let rhs_cuda = rhs_cuda.to_cuda();

        let loss_cuda = sum(&matmul(&lhs_cuda, &rhs_cuda));
        loss_cuda.backward();

        let loss_cpu = sum(&matmul(&lhs_cpu, &rhs_cpu));
        loss_cpu.backward();

        assert!(lhs_cuda.cloned_cuda_f32_grad().is_some());
        assert!(rhs_cuda.cloned_cuda_f32_grad().is_some());
        assert!(!rhs_cuda.has_host_f32_data());
        assert!(!lhs_cuda.has_host_grad());
        assert!(!rhs_cuda.has_host_grad());

        let lhs_grad = lhs_cuda.grad().expect("CUDA mixed matmul lhs grad");
        let rhs_grad = rhs_cuda.grad().expect("CUDA mixed matmul rhs grad");
        let lhs_ref = lhs_cpu.grad().expect("CPU mixed matmul lhs grad");
        let rhs_ref = rhs_cpu.grad().expect("CPU mixed matmul rhs grad");
        for (got, expect) in lhs_grad.iter().zip(lhs_ref.iter()) {
            assert!(
                (got - expect).abs() <= 0.08,
                "{lhs_dtype:?}xI8 matmul lhs grad got {got}, expect {expect}"
            );
        }
        for (got, expect) in rhs_grad.iter().zip(rhs_ref.iter()) {
            assert!(
                (got - expect).abs() <= 0.08,
                "{lhs_dtype:?}xI8 matmul rhs grad got {got}, expect {expect}"
            );
        }
    }

    crate::autograd::set_strict_device_execution(false);
    crate::ops::cuda::set_enabled(false);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_matmul_i8_floatlike_backward_reads_typed_data_and_writes_f32_grads() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let lhs_data = vec![1.0, -2.0, 0.5, 3.0, -1.0, 2.0];
    let rhs_data = vec![0.5, 1.0, -1.5, 2.0, 0.25, -0.75, 1.25, -0.5, 0.75];

    crate::ops::cuda::set_enabled(true);
    crate::autograd::set_strict_device_execution(true);

    for rhs_dtype in [DType::F16, DType::F32, DType::BF16] {
        let lhs_cpu = make_grad_tensor(&[2, 3], lhs_data.clone());
        lhs_cpu.cast_inplace(DType::I8);
        let rhs_cpu = make_grad_tensor(&[3, 3], rhs_data.clone());
        rhs_cpu.cast_inplace(rhs_dtype);

        let lhs_cuda = make_grad_tensor(&[2, 3], lhs_data.clone());
        lhs_cuda.cast_inplace(DType::I8);
        let lhs_cuda = lhs_cuda.to_cuda();
        let rhs_cuda = make_grad_tensor(&[3, 3], rhs_data.clone());
        rhs_cuda.cast_inplace(rhs_dtype);
        let rhs_cuda = rhs_cuda.to_cuda();

        let loss_cuda = sum(&matmul(&lhs_cuda, &rhs_cuda));
        loss_cuda.backward();

        let loss_cpu = sum(&matmul(&lhs_cpu, &rhs_cpu));
        loss_cpu.backward();

        assert!(lhs_cuda.cloned_cuda_f32_grad().is_some());
        assert!(rhs_cuda.cloned_cuda_f32_grad().is_some());
        assert!(!lhs_cuda.has_host_f32_data());
        assert!(!lhs_cuda.has_host_grad());
        assert!(!rhs_cuda.has_host_grad());

        let lhs_grad = lhs_cuda.grad().expect("CUDA I8xfloatlike matmul lhs grad");
        let rhs_grad = rhs_cuda.grad().expect("CUDA I8xfloatlike matmul rhs grad");
        let lhs_ref = lhs_cpu.grad().expect("CPU I8xfloatlike matmul lhs grad");
        let rhs_ref = rhs_cpu.grad().expect("CPU I8xfloatlike matmul rhs grad");
        for (got, expect) in lhs_grad.iter().zip(lhs_ref.iter()) {
            assert!(
                (got - expect).abs() <= 0.08,
                "I8x{rhs_dtype:?} matmul lhs grad got {got}, expect {expect}"
            );
        }
        for (got, expect) in rhs_grad.iter().zip(rhs_ref.iter()) {
            assert!(
                (got - expect).abs() <= 0.08,
                "I8x{rhs_dtype:?} matmul rhs grad got {got}, expect {expect}"
            );
        }
    }

    crate::autograd::set_strict_device_execution(false);
    crate::ops::cuda::set_enabled(false);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_matmul_backward_matches_cpu_reference_in_strict_mode() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let a_data = vec![1.0, -2.0, 0.5, 3.0, -1.0, 2.0];
    let b_data = vec![0.5, 1.0, -1.5, 2.0, 0.25, -0.75, 1.25, -0.5, 0.75];
    let a_cpu = make_grad_tensor(&[2, 3], a_data.clone());
    let b_cpu = make_grad_tensor(&[3, 3], b_data.clone());
    let a_cuda = make_grad_tensor(&[2, 3], a_data).to_cuda();
    let b_cuda = make_grad_tensor(&[3, 3], b_data).to_cuda();

    crate::ops::cuda::set_enabled(true);
    set_strict_device_execution(true);
    let loss_cuda = sum(&matmul(&a_cuda, &b_cuda));
    loss_cuda.backward();
    set_strict_device_execution(false);
    crate::ops::cuda::set_enabled(false);

    let loss_cpu = sum(&matmul(&a_cpu, &b_cpu));
    loss_cpu.backward();

    assert!(!a_cuda.has_host_grad());
    assert!(!b_cuda.has_host_grad());
    assert!(a_cuda.cloned_cuda_f32_grad().is_some());
    assert!(b_cuda.cloned_cuda_f32_grad().is_some());
    let a_cuda_grad = a_cuda.grad().expect("cuda lhs grad");
    let b_cuda_grad = b_cuda.grad().expect("cuda rhs grad");
    let a_cpu_grad = a_cpu.grad().expect("cpu lhs grad");
    let b_cpu_grad = b_cpu.grad().expect("cpu rhs grad");
    for (got, expect) in a_cuda_grad.iter().zip(a_cpu_grad.iter()) {
        assert!(
            (got - expect).abs() < 1e-4,
            "lhs grad got {got}, expect {expect}"
        );
    }
    for (got, expect) in b_cuda_grad.iter().zip(b_cpu_grad.iter()) {
        assert!(
            (got - expect).abs() < 1e-4,
            "rhs grad got {got}, expect {expect}"
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_batch_matmul_backward_matches_cpu_reference_in_strict_mode() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let lhs_data = vec![1.0, -2.0, 0.5, 3.0, -1.0, 2.0];
    let rhs_data = vec![0.5, 1.0, -1.5, 2.0, 0.25, -0.75];
    let lhs_cpu = make_grad_tensor(&[1, 1, 2, 3], lhs_data.clone());
    let rhs_cpu = make_grad_tensor(&[1, 1, 3, 2], rhs_data.clone());
    let lhs_cuda = make_grad_tensor(&[1, 1, 2, 3], lhs_data).to_cuda();
    let rhs_cuda = make_grad_tensor(&[1, 1, 3, 2], rhs_data).to_cuda();

    crate::ops::cuda::set_enabled(true);
    set_strict_device_execution(true);
    let loss_cuda = sum(&batch_matmul(&lhs_cuda, &rhs_cuda));
    loss_cuda.backward();
    set_strict_device_execution(false);
    crate::ops::cuda::set_enabled(false);

    let loss_cpu = sum(&batch_matmul(&lhs_cpu, &rhs_cpu));
    loss_cpu.backward();

    assert!(!lhs_cuda.has_host_grad());
    assert!(!rhs_cuda.has_host_grad());
    assert!(lhs_cuda.cloned_cuda_f32_grad().is_some());
    assert!(rhs_cuda.cloned_cuda_f32_grad().is_some());
    let lhs_cuda_grad = lhs_cuda.grad().expect("cuda lhs grad");
    let rhs_cuda_grad = rhs_cuda.grad().expect("cuda rhs grad");
    let lhs_cpu_grad = lhs_cpu.grad().expect("cpu lhs grad");
    let rhs_cpu_grad = rhs_cpu.grad().expect("cpu rhs grad");
    for (got, expect) in lhs_cuda_grad.iter().zip(lhs_cpu_grad.iter()) {
        assert!(
            (got - expect).abs() < 1e-4,
            "lhs grad got {got}, expect {expect}"
        );
    }
    for (got, expect) in rhs_cuda_grad.iter().zip(rhs_cpu_grad.iter()) {
        assert!(
            (got - expect).abs() < 1e-4,
            "rhs grad got {got}, expect {expect}"
        );
    }
}

#[test]
#[ignore = "performance sanity test; run with --ignored --nocapture"]
fn cpu_batch_matmul_backward_lowp_perf_smoke() {
    let b = 4;
    let h = 4;
    let m = 16;
    let k = 64;
    let n = 32;
    let lhs_data = (0..b * h * m * k)
        .map(|i| (i as f32 % 251.0) / 41.0 - 2.75)
        .collect::<Vec<_>>();
    let rhs_data = (0..b * h * k * n)
        .map(|i| ((i * 23) as f32 % 239.0) / 53.0 - 2.25)
        .collect::<Vec<_>>();
    let grad = ndarray::Array::from_shape_vec(
        ndarray::IxDyn(&[b, h, m, n]),
        (0..b * h * m * n)
            .map(|i| ((i * 11) as f32 % 89.0) / 29.0 - 1.25)
            .collect::<Vec<_>>(),
    )
    .expect("grad shape mismatch")
    .into_dyn();

    let measure = |lhs_dtype: DType, rhs_dtype: DType| {
        let lhs = make_tensor(&[b, h, m, k], lhs_data.clone(), lhs_dtype);
        let rhs = make_tensor(&[b, h, k, n], rhs_data.clone(), rhs_dtype);
        let start = std::time::Instant::now();
        let (d_lhs, d_rhs) = batch_matmul_backward_cpu_f32(&lhs, &rhs, &grad.view(), b, h, m, k, n);
        let elapsed_us = start.elapsed().as_secs_f64() * 1.0e6;
        assert_eq!(d_lhs.shape(), &[b, h, m, k]);
        assert_eq!(d_rhs.shape(), &[b, h, k, n]);
        assert!(d_lhs.iter().all(|v| v.is_finite()));
        assert!(d_rhs.iter().all(|v| v.is_finite()));
        elapsed_us
    };

    let bf16_bf16_us = measure(DType::BF16, DType::BF16);
    let i8_i8_us = measure(DType::I8, DType::I8);
    let bf16_i8_us = measure(DType::BF16, DType::I8);
    let f32_i8_us = measure(DType::F32, DType::I8);

    println!(
        "cpu batch_matmul backward b={b} h={h} m={m} n={n} k={k}: bf16xbf16={bf16_bf16_us:.1}us, i8xi8={i8_i8_us:.1}us, bf16xi8={bf16_i8_us:.1}us, f32xi8={f32_i8_us:.1}us"
    );

    let mm_m = 64;
    let mm_n = 64;
    let mm_k = 128;
    let mm_lhs = (0..mm_m * mm_k)
        .map(|i| (i as f32 % 251.0) / 43.0 - 2.5)
        .collect::<Vec<_>>();
    let mm_rhs = (0..mm_n * mm_k)
        .map(|i| ((i * 19) as f32 % 239.0) / 47.0 - 2.0)
        .collect::<Vec<_>>();
    let mm_grad = ndarray::Array::from_shape_vec(
        ndarray::IxDyn(&[mm_m, mm_n]),
        (0..mm_m * mm_n)
            .map(|i| ((i * 13) as f32 % 97.0) / 31.0 - 1.5)
            .collect::<Vec<_>>(),
    )
    .expect("matmul grad shape mismatch")
    .into_dyn();

    let measure_mm = |lhs_dtype: DType, rhs_dtype: DType| {
        let lhs = make_tensor(&[mm_m, mm_k], mm_lhs.clone(), lhs_dtype);
        let rhs = make_tensor(&[mm_n, mm_k], mm_rhs.clone(), rhs_dtype);
        let start = std::time::Instant::now();
        let (d_lhs, d_rhs) = matmul_backward_cpu_f32(&lhs, &rhs, &mm_grad.view(), mm_m, mm_k, mm_n);
        let elapsed_us = start.elapsed().as_secs_f64() * 1.0e6;
        assert_eq!(d_lhs.shape(), &[mm_m, mm_k]);
        assert_eq!(d_rhs.shape(), &[mm_n, mm_k]);
        assert!(d_lhs.iter().all(|v| v.is_finite()));
        assert!(d_rhs.iter().all(|v| v.is_finite()));
        elapsed_us
    };

    let mm_bf16_bf16_us = measure_mm(DType::BF16, DType::BF16);
    let mm_i8_i8_us = measure_mm(DType::I8, DType::I8);
    let mm_bf16_i8_us = measure_mm(DType::BF16, DType::I8);
    let mm_f32_i8_us = measure_mm(DType::F32, DType::I8);

    println!(
        "cpu matmul backward m={mm_m} n={mm_n} k={mm_k}: bf16xbf16={mm_bf16_bf16_us:.1}us, i8xi8={mm_i8_i8_us:.1}us, bf16xi8={mm_bf16_i8_us:.1}us, f32xi8={mm_f32_i8_us:.1}us"
    );
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "performance sanity test; run with --ignored --nocapture"]
fn cuda_resident_lowp_matmul_perf_smoke() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    fn median_cuda_us(mut run: impl FnMut()) -> f64 {
        run();
        crate::ops::cuda::synchronize().expect("sync CUDA performance warmup");
        let mut samples = Vec::with_capacity(7);
        for _ in 0..7 {
            let start = std::time::Instant::now();
            run();
            crate::ops::cuda::synchronize().expect("sync CUDA performance sample");
            samples.push(start.elapsed().as_secs_f64() * 1.0e6);
        }
        samples.sort_by(f64::total_cmp);
        samples[samples.len() / 2]
    }

    let m = 128;
    let n = 128;
    let k = 256;
    let a = (0..m * k)
        .map(|i| (i as f32 % 251.0) / 43.0 - 2.5)
        .collect::<Vec<_>>();
    let b = (0..n * k)
        .map(|i| ((i * 19) as f32 % 239.0) / 47.0 - 2.0)
        .collect::<Vec<_>>();

    let a_f32 = crate::ops::cuda::upload_f32(&a).expect("upload matmul a f32");
    let b_f32 = crate::ops::cuda::upload_f32(&b).expect("upload matmul b f32");
    let f32_out = crate::ops::cuda::matmul_f32_no_host(&a_f32, &b_f32, m, n, k)
        .expect("CUDA f32 matmul perf path");
    let f32_us = median_cuda_us(|| {
        let _ = crate::ops::cuda::matmul_f32_no_host(&a_f32, &b_f32, m, n, k)
            .expect("CUDA f32 matmul performance sample");
    });
    let f32_vals = crate::ops::cuda::download_f32(&f32_out).expect("download f32 matmul");

    let a_f16 = a
        .iter()
        .map(|&v| f16::from_f32(v).to_bits())
        .collect::<Vec<_>>();
    let b_f16 = b
        .iter()
        .map(|&v| f16::from_f32(v).to_bits())
        .collect::<Vec<_>>();
    let a_f16_buf = crate::ops::cuda::upload_u16_storage(&a_f16).expect("upload matmul a f16");
    let b_f16_buf = crate::ops::cuda::upload_u16_storage(&b_f16).expect("upload matmul b f16");
    let f16_out = crate::ops::cuda::matmul_f16_buffer_no_host(&a_f16_buf, &b_f16_buf, m, n, k)
        .expect("CUDA resident f16 matmul perf path");
    let f16_us = median_cuda_us(|| {
        let _ = crate::ops::cuda::matmul_f16_buffer_no_host(&a_f16_buf, &b_f16_buf, m, n, k)
            .expect("CUDA resident f16 matmul performance sample");
    });
    let f16_vals = crate::ops::cuda::download_f32(&f16_out).expect("download f16 matmul");

    let a_bf16 = a
        .iter()
        .map(|&v| bf16::from_f32(v).to_bits())
        .collect::<Vec<_>>();
    let b_bf16 = b
        .iter()
        .map(|&v| bf16::from_f32(v).to_bits())
        .collect::<Vec<_>>();
    let a_bf16_buf = crate::ops::cuda::upload_u16_storage(&a_bf16).expect("upload matmul a bf16");
    let b_bf16_buf = crate::ops::cuda::upload_u16_storage(&b_bf16).expect("upload matmul b bf16");
    let bf16_out = crate::ops::cuda::matmul_bf16_buffer_no_host(&a_bf16_buf, &b_bf16_buf, m, n, k)
        .expect("CUDA resident bf16 matmul perf path");
    let bf16_us = median_cuda_us(|| {
        let _ = crate::ops::cuda::matmul_bf16_buffer_no_host(&a_bf16_buf, &b_bf16_buf, m, n, k)
            .expect("CUDA resident bf16 matmul performance sample");
    });
    let bf16_vals = crate::ops::cuda::download_f32(&bf16_out).expect("download bf16 matmul");

    let a_i8 = a
        .iter()
        .map(|&v| (v / 0.05).round().clamp(-127.0, 127.0) as i8)
        .collect::<Vec<_>>();
    let b_i8 = b
        .iter()
        .map(|&v| (v / 0.05).round().clamp(-127.0, 127.0) as i8)
        .collect::<Vec<_>>();
    let a_i8_buf = crate::ops::cuda::upload_i8_storage(&a_i8).expect("upload matmul a i8");
    let b_i8_buf = crate::ops::cuda::upload_i8_storage(&b_i8).expect("upload matmul b i8");
    let i8_out =
        crate::ops::cuda::matmul_i8_buffer_no_host(&a_i8_buf, 0.05, &b_i8_buf, 0.05, m, n, k)
            .expect("CUDA resident i8 matmul perf path");
    let i8_us = median_cuda_us(|| {
        let _ =
            crate::ops::cuda::matmul_i8_buffer_no_host(&a_i8_buf, 0.05, &b_i8_buf, 0.05, m, n, k)
                .expect("CUDA resident i8 matmul performance sample");
    });
    let i8_vals = crate::ops::cuda::download_f32(&i8_out).expect("download i8 matmul");
    let i8_typed_us = median_cuda_us(|| {
        let _ = crate::ops::cuda::matmul_i8_typed_output_buffer_no_host(
            &a_i8_buf, 0.05, &b_i8_buf, 0.05, m, n, k,
        )
        .expect("CUDA resident typed-output i8 matmul performance sample");
    });
    let mut i8_quant_ref = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0i32;
            for kk in 0..k {
                acc += (a_i8[row * k + kk] as i32) * (b_i8[col * k + kk] as i32);
            }
            i8_quant_ref[row * n + col] = (acc as f32) * 0.05 * 0.05;
        }
    }

    let max_err = |vals: &[f32]| {
        vals.iter()
            .zip(f32_vals.iter())
            .map(|(&got, &expect)| (got - expect).abs())
            .fold(0.0f32, f32::max)
    };
    let max_err_against = |vals: &[f32], reference: &[f32]| {
        vals.iter()
            .zip(reference.iter())
            .map(|(&got, &expect)| (got - expect).abs())
            .fold(0.0f32, f32::max)
    };

    println!(
        "cuda resident lowp matmul m={m} n={n} k={k}: f32={f32_us:.1}us, f16={f16_us:.1}us max_err={:.5}, bf16={bf16_us:.1}us max_err={:.5}, i8_f32_out={i8_us:.1}us i8_typed_out={i8_typed_us:.1}us kernel_err={:.5} quant_err={:.5}",
        max_err(&f16_vals),
        max_err(&bf16_vals),
        max_err_against(&i8_vals, &i8_quant_ref),
        max_err(&i8_vals)
    );

    let f16_i8_out = crate::ops::cuda::matmul_f16_i8_dynamic_buffer_no_host(
        &a_f16_buf, &b_i8_buf, 0.05, m, n, k,
    )
    .expect("CUDA dynamic F16xI8 matmul perf path");
    let f16_i8_us = median_cuda_us(|| {
        let _ = crate::ops::cuda::matmul_f16_i8_dynamic_buffer_no_host(
            &a_f16_buf, &b_i8_buf, 0.05, m, n, k,
        )
        .expect("CUDA dynamic F16xI8 matmul performance sample");
    });
    let f16_i8_vals = crate::ops::cuda::download_f32(&f16_i8_out).expect("download F16xI8 matmul");
    let bf16_i8_out = crate::ops::cuda::matmul_bf16_i8_dynamic_buffer_no_host(
        &a_bf16_buf,
        &b_i8_buf,
        0.05,
        m,
        n,
        k,
    )
    .expect("CUDA dynamic BF16xI8 matmul perf path");
    let bf16_i8_us = median_cuda_us(|| {
        let _ = crate::ops::cuda::matmul_bf16_i8_dynamic_buffer_no_host(
            &a_bf16_buf,
            &b_i8_buf,
            0.05,
            m,
            n,
            k,
        )
        .expect("CUDA dynamic BF16xI8 matmul performance sample");
    });
    let bf16_i8_vals =
        crate::ops::cuda::download_f32(&bf16_i8_out).expect("download BF16xI8 matmul");
    let f32_i8_out =
        crate::ops::cuda::matmul_f32_i8_buffer_no_host(&a_f32, &b_i8_buf, 0.05, m, n, k)
            .expect("CUDA F32xI8 matmul perf path");
    let f32_i8_us = median_cuda_us(|| {
        let _ = crate::ops::cuda::matmul_f32_i8_buffer_no_host(&a_f32, &b_i8_buf, 0.05, m, n, k)
            .expect("CUDA F32xI8 matmul performance sample");
    });
    let f32_i8_vals = crate::ops::cuda::download_f32(&f32_i8_out).expect("download F32xI8 matmul");
    println!(
        "cuda resident mixed matmul m={m} n={n} k={k}: f16xi8={f16_i8_us:.1}us quant_err={:.5} f32_err={:.5}, bf16xi8={bf16_i8_us:.1}us quant_err={:.5} f32_err={:.5}, f32xi8={f32_i8_us:.1}us quant_err={:.5} f32_err={:.5}",
        max_err_against(&f16_i8_vals, &i8_quant_ref),
        max_err(&f16_i8_vals),
        max_err_against(&bf16_i8_vals, &i8_quant_ref),
        max_err(&bf16_i8_vals),
        max_err_against(&f32_i8_vals, &i8_quant_ref),
        max_err(&f32_i8_vals)
    );

    let i8_f16_out =
        crate::ops::cuda::matmul_i8_f16_buffer_no_host(&a_i8_buf, 0.05, &b_f16_buf, m, n, k)
            .expect("CUDA I8xF16 matmul perf path");
    let i8_f16_us = median_cuda_us(|| {
        let _ =
            crate::ops::cuda::matmul_i8_f16_buffer_no_host(&a_i8_buf, 0.05, &b_f16_buf, m, n, k)
                .expect("CUDA I8xF16 matmul performance sample");
    });
    let i8_f16_vals = crate::ops::cuda::download_f32(&i8_f16_out).expect("download I8xF16 matmul");
    let i8_bf16_out =
        crate::ops::cuda::matmul_i8_bf16_buffer_no_host(&a_i8_buf, 0.05, &b_bf16_buf, m, n, k)
            .expect("CUDA I8xBF16 matmul perf path");
    let i8_bf16_us = median_cuda_us(|| {
        let _ =
            crate::ops::cuda::matmul_i8_bf16_buffer_no_host(&a_i8_buf, 0.05, &b_bf16_buf, m, n, k)
                .expect("CUDA I8xBF16 matmul performance sample");
    });
    let i8_bf16_vals =
        crate::ops::cuda::download_f32(&i8_bf16_out).expect("download I8xBF16 matmul");
    let i8_f32_out =
        crate::ops::cuda::matmul_i8_f32_buffer_no_host(&a_i8_buf, 0.05, &b_f32, m, n, k)
            .expect("CUDA I8xF32 matmul perf path");
    let i8_f32_us = median_cuda_us(|| {
        let _ = crate::ops::cuda::matmul_i8_f32_buffer_no_host(&a_i8_buf, 0.05, &b_f32, m, n, k)
            .expect("CUDA I8xF32 matmul performance sample");
    });
    let i8_f32_vals = crate::ops::cuda::download_f32(&i8_f32_out).expect("download I8xF32 matmul");

    let mut i8_lhs_ref = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += (a_i8[row * k + kk] as f32) * 0.05 * b[col * k + kk];
            }
            i8_lhs_ref[row * n + col] = acc;
        }
    }
    println!(
        "cuda resident mixed matmul mirrored m={m} n={n} k={k}: i8xf16={i8_f16_us:.1}us quant_err={:.5} f32_err={:.5}, i8xbf16={i8_bf16_us:.1}us quant_err={:.5} f32_err={:.5}, i8xf32={i8_f32_us:.1}us quant_err={:.5} f32_err={:.5}",
        max_err_against(&i8_f16_vals, &i8_lhs_ref),
        max_err(&i8_f16_vals),
        max_err_against(&i8_bf16_vals, &i8_lhs_ref),
        max_err(&i8_bf16_vals),
        max_err_against(&i8_f32_vals, &i8_lhs_ref),
        max_err(&i8_f32_vals)
    );

    let batch_count = 8;
    let bm = 16;
    let bn = 32;
    let bk = 64;
    let lhs = (0..batch_count * bm * bk)
        .map(|i| (i as f32 % 251.0) / 41.0 - 2.75)
        .collect::<Vec<_>>();
    let rhs = (0..batch_count * bk * bn)
        .map(|i| ((i * 23) as f32 % 239.0) / 53.0 - 2.25)
        .collect::<Vec<_>>();
    let lhs_f32 = crate::ops::cuda::upload_f32(&lhs).expect("upload batch lhs f32");
    let rhs_f32 = crate::ops::cuda::upload_f32(&rhs).expect("upload batch rhs f32");
    let batch_f32_out =
        crate::ops::cuda::batch_matmul_f32_no_host(&lhs_f32, &rhs_f32, batch_count, bm, bn, bk)
            .expect("CUDA f32 batch_matmul perf path");
    let batch_f32_us = median_cuda_us(|| {
        let _ =
            crate::ops::cuda::batch_matmul_f32_no_host(&lhs_f32, &rhs_f32, batch_count, bm, bn, bk)
                .expect("CUDA f32 batch_matmul performance sample");
    });
    let batch_f32_vals =
        crate::ops::cuda::download_f32(&batch_f32_out).expect("download f32 batch_matmul");

    let lhs_bf16 = lhs
        .iter()
        .map(|&v| bf16::from_f32(v).to_bits())
        .collect::<Vec<_>>();
    let lhs_f16 = lhs
        .iter()
        .map(|&v| f16::from_f32(v).to_bits())
        .collect::<Vec<_>>();
    let rhs_i8 = rhs
        .iter()
        .map(|&v| (v / 0.05).round().clamp(-127.0, 127.0) as i8)
        .collect::<Vec<_>>();
    let lhs_i8 = lhs
        .iter()
        .map(|&v| (v / 0.05).round().clamp(-127.0, 127.0) as i8)
        .collect::<Vec<_>>();
    let rhs_bf16 = rhs
        .iter()
        .map(|&v| bf16::from_f32(v).to_bits())
        .collect::<Vec<_>>();
    let rhs_f16 = rhs
        .iter()
        .map(|&v| f16::from_f32(v).to_bits())
        .collect::<Vec<_>>();
    let lhs_bf16_buf =
        crate::ops::cuda::upload_u16_storage(&lhs_bf16).expect("upload batch lhs bf16");
    let lhs_f16_buf = crate::ops::cuda::upload_u16_storage(&lhs_f16).expect("upload batch lhs f16");
    let lhs_i8_buf = crate::ops::cuda::upload_i8_storage(&lhs_i8).expect("upload batch lhs i8");
    let rhs_i8_buf = crate::ops::cuda::upload_i8_storage(&rhs_i8).expect("upload batch rhs i8");
    let rhs_bf16_buf =
        crate::ops::cuda::upload_u16_storage(&rhs_bf16).expect("upload batch rhs bf16");
    let rhs_f16_buf = crate::ops::cuda::upload_u16_storage(&rhs_f16).expect("upload batch rhs f16");
    let batch_f16_i8_out = crate::ops::cuda::batch_matmul_f16_i8_buffer_no_host(
        &lhs_f16_buf,
        &rhs_i8_buf,
        0.05,
        batch_count,
        bm,
        bn,
        bk,
    )
    .expect("CUDA F16xI8 batch_matmul perf path");
    let batch_f16_i8_us = median_cuda_us(|| {
        let _ = crate::ops::cuda::batch_matmul_f16_i8_buffer_no_host(
            &lhs_f16_buf,
            &rhs_i8_buf,
            0.05,
            batch_count,
            bm,
            bn,
            bk,
        )
        .expect("CUDA F16xI8 batch_matmul performance sample");
    });
    let batch_f16_i8_vals =
        crate::ops::cuda::download_f32(&batch_f16_i8_out).expect("download F16xI8 batch_matmul");

    let batch_bf16_i8_out = crate::ops::cuda::batch_matmul_bf16_i8_buffer_no_host(
        &lhs_bf16_buf,
        &rhs_i8_buf,
        0.05,
        batch_count,
        bm,
        bn,
        bk,
    )
    .expect("CUDA BF16xI8 batch_matmul perf path");
    let batch_bf16_i8_us = median_cuda_us(|| {
        let _ = crate::ops::cuda::batch_matmul_bf16_i8_buffer_no_host(
            &lhs_bf16_buf,
            &rhs_i8_buf,
            0.05,
            batch_count,
            bm,
            bn,
            bk,
        )
        .expect("CUDA BF16xI8 batch_matmul performance sample");
    });
    let batch_bf16_i8_vals =
        crate::ops::cuda::download_f32(&batch_bf16_i8_out).expect("download BF16xI8 batch_matmul");

    let batch_f32_i8_out = crate::ops::cuda::batch_matmul_f32_i8_buffer_no_host(
        &lhs_f32,
        &rhs_i8_buf,
        0.05,
        batch_count,
        bm,
        bn,
        bk,
    )
    .expect("CUDA F32xI8 batch_matmul perf path");
    let batch_f32_i8_us = median_cuda_us(|| {
        let _ = crate::ops::cuda::batch_matmul_f32_i8_buffer_no_host(
            &lhs_f32,
            &rhs_i8_buf,
            0.05,
            batch_count,
            bm,
            bn,
            bk,
        )
        .expect("CUDA F32xI8 batch_matmul performance sample");
    });
    let batch_f32_i8_vals =
        crate::ops::cuda::download_f32(&batch_f32_i8_out).expect("download F32xI8 batch_matmul");

    let mut batch_quant_ref = vec![0.0f32; batch_count * bm * bn];
    for batch in 0..batch_count {
        for row in 0..bm {
            for col in 0..bn {
                let mut acc = 0.0f32;
                for kk in 0..bk {
                    acc += lhs[batch * bm * bk + row * bk + kk]
                        * (rhs_i8[batch * bk * bn + kk * bn + col] as f32)
                        * 0.05;
                }
                batch_quant_ref[batch * bm * bn + row * bn + col] = acc;
            }
        }
    }
    println!(
        "cuda resident mixed batch_matmul batch={batch_count} m={bm} n={bn} k={bk}: f32={batch_f32_us:.1}us, f16xi8={batch_f16_i8_us:.1}us quant_err={:.5} f32_err={:.5}, bf16xi8={batch_bf16_i8_us:.1}us quant_err={:.5} f32_err={:.5}, f32xi8={batch_f32_i8_us:.1}us quant_err={:.5} f32_err={:.5}",
        max_err_against(&batch_f16_i8_vals, &batch_quant_ref),
        max_err_against(&batch_f16_i8_vals, &batch_f32_vals),
        max_err_against(&batch_bf16_i8_vals, &batch_quant_ref),
        max_err_against(&batch_bf16_i8_vals, &batch_f32_vals),
        max_err_against(&batch_f32_i8_vals, &batch_quant_ref),
        max_err_against(&batch_f32_i8_vals, &batch_f32_vals)
    );

    let batch_i8_f16_out = crate::ops::cuda::batch_matmul_i8_f16_buffer_no_host(
        &lhs_i8_buf,
        0.05,
        &rhs_f16_buf,
        batch_count,
        bm,
        bn,
        bk,
    )
    .expect("CUDA I8xF16 batch_matmul perf path");
    let batch_i8_f16_us = median_cuda_us(|| {
        let _ = crate::ops::cuda::batch_matmul_i8_f16_buffer_no_host(
            &lhs_i8_buf,
            0.05,
            &rhs_f16_buf,
            batch_count,
            bm,
            bn,
            bk,
        )
        .expect("CUDA I8xF16 batch_matmul performance sample");
    });
    let batch_i8_f16_vals =
        crate::ops::cuda::download_f32(&batch_i8_f16_out).expect("download I8xF16 batch_matmul");

    let batch_i8_bf16_out = crate::ops::cuda::batch_matmul_i8_bf16_buffer_no_host(
        &lhs_i8_buf,
        0.05,
        &rhs_bf16_buf,
        batch_count,
        bm,
        bn,
        bk,
    )
    .expect("CUDA I8xBF16 batch_matmul perf path");
    let batch_i8_bf16_us = median_cuda_us(|| {
        let _ = crate::ops::cuda::batch_matmul_i8_bf16_buffer_no_host(
            &lhs_i8_buf,
            0.05,
            &rhs_bf16_buf,
            batch_count,
            bm,
            bn,
            bk,
        )
        .expect("CUDA I8xBF16 batch_matmul performance sample");
    });
    let batch_i8_bf16_vals =
        crate::ops::cuda::download_f32(&batch_i8_bf16_out).expect("download I8xBF16 batch_matmul");

    let batch_i8_f32_out = crate::ops::cuda::batch_matmul_i8_f32_buffer_no_host(
        &lhs_i8_buf,
        0.05,
        &rhs_f32,
        batch_count,
        bm,
        bn,
        bk,
    )
    .expect("CUDA I8xF32 batch_matmul perf path");
    let batch_i8_f32_us = median_cuda_us(|| {
        let _ = crate::ops::cuda::batch_matmul_i8_f32_buffer_no_host(
            &lhs_i8_buf,
            0.05,
            &rhs_f32,
            batch_count,
            bm,
            bn,
            bk,
        )
        .expect("CUDA I8xF32 batch_matmul performance sample");
    });
    let batch_i8_f32_vals =
        crate::ops::cuda::download_f32(&batch_i8_f32_out).expect("download I8xF32 batch_matmul");

    let mut batch_i8_lhs_ref = vec![0.0f32; batch_count * bm * bn];
    for batch in 0..batch_count {
        for row in 0..bm {
            for col in 0..bn {
                let mut acc = 0.0f32;
                for kk in 0..bk {
                    acc += (lhs_i8[batch * bm * bk + row * bk + kk] as f32)
                        * 0.05
                        * rhs[batch * bk * bn + kk * bn + col];
                }
                batch_i8_lhs_ref[batch * bm * bn + row * bn + col] = acc;
            }
        }
    }
    println!(
        "cuda resident mixed batch_matmul mirrored batch={batch_count} m={bm} n={bn} k={bk}: i8xf16={batch_i8_f16_us:.1}us quant_err={:.5} f32_err={:.5}, i8xbf16={batch_i8_bf16_us:.1}us quant_err={:.5} f32_err={:.5}, i8xf32={batch_i8_f32_us:.1}us quant_err={:.5} f32_err={:.5}",
        max_err_against(&batch_i8_f16_vals, &batch_i8_lhs_ref),
        max_err_against(&batch_i8_f16_vals, &batch_f32_vals),
        max_err_against(&batch_i8_bf16_vals, &batch_i8_lhs_ref),
        max_err_against(&batch_i8_bf16_vals, &batch_f32_vals),
        max_err_against(&batch_i8_f32_vals, &batch_i8_lhs_ref),
        max_err_against(&batch_i8_f32_vals, &batch_f32_vals)
    );

    let batch_i8_i8_f32_us = median_cuda_us(|| {
        let _ = crate::ops::cuda::batch_matmul_i8_buffer_no_host(
            &lhs_i8_buf,
            0.05,
            &rhs_i8_buf,
            0.05,
            batch_count,
            bm,
            bn,
            bk,
        )
        .expect("CUDA I8xI8 f32-output batch_matmul performance sample");
    });

    let batch_i8_i8_typed_us = median_cuda_us(|| {
        let _ = crate::ops::cuda::batch_matmul_i8_typed_output_buffer_no_host(
            &lhs_i8_buf,
            0.05,
            &rhs_i8_buf,
            0.05,
            batch_count,
            bm,
            bn,
            bk,
        )
        .expect("CUDA I8xI8 typed-output batch_matmul performance sample");
    });
    println!(
        "cuda resident I8xI8 batch_matmul batch={batch_count} m={bm} n={bn} k={bk}: f32_out={batch_i8_i8_f32_us:.1}us, typed_i8_out={batch_i8_i8_typed_us:.1}us"
    );

    let grad = (0..m * n)
        .map(|i| ((i * 13) as f32 % 97.0) / 31.0 - 1.5)
        .collect::<Vec<_>>();
    let grad_buf = crate::ops::cuda::upload_f32(&grad).expect("upload matmul grad");
    let backward_f16_i8_us = median_cuda_us(|| {
        let _ = crate::ops::cuda::matmul_backward_f16_i8_no_host(
            &grad_buf, &a_f16_buf, &b_i8_buf, 0.05, m, k, n,
        )
        .expect("CUDA F16xI8 matmul backward performance sample");
    });
    let backward_bf16_i8_us = median_cuda_us(|| {
        let _ = crate::ops::cuda::matmul_backward_bf16_i8_no_host(
            &grad_buf,
            &a_bf16_buf,
            &b_i8_buf,
            0.05,
            m,
            k,
            n,
        )
        .expect("CUDA BF16xI8 matmul backward performance sample");
    });
    let backward_f32_i8_us = median_cuda_us(|| {
        let _ = crate::ops::cuda::matmul_backward_f32_i8_no_host(
            &grad_buf, &a_f32, &b_i8_buf, 0.05, m, k, n,
        )
        .expect("CUDA F32xI8 matmul backward performance sample");
    });
    let backward_i8_f16_us = median_cuda_us(|| {
        let _ = crate::ops::cuda::matmul_backward_i8_f16_no_host(
            &grad_buf, &a_i8_buf, 0.05, &b_f16_buf, m, k, n,
        )
        .expect("CUDA I8xF16 matmul backward performance sample");
    });
    let backward_i8_bf16_us = median_cuda_us(|| {
        let _ = crate::ops::cuda::matmul_backward_i8_bf16_no_host(
            &grad_buf,
            &a_i8_buf,
            0.05,
            &b_bf16_buf,
            m,
            k,
            n,
        )
        .expect("CUDA I8xBF16 matmul backward performance sample");
    });
    let backward_i8_f32_us = median_cuda_us(|| {
        let _ = crate::ops::cuda::matmul_backward_i8_f32_no_host(
            &grad_buf, &a_i8_buf, 0.05, &b_f32, m, k, n,
        )
        .expect("CUDA I8xF32 matmul backward performance sample");
    });

    let batch_grad = (0..batch_count * bm * bn)
        .map(|i| ((i * 11) as f32 % 89.0) / 29.0 - 1.25)
        .collect::<Vec<_>>();
    let batch_grad_buf =
        crate::ops::cuda::upload_f32(&batch_grad).expect("upload batch_matmul grad");
    let batch_backward_f16_i8_us = median_cuda_us(|| {
        let _ = crate::ops::cuda::batch_matmul_backward_f16_i8_no_host(
            &batch_grad_buf,
            &lhs_f16_buf,
            &rhs_i8_buf,
            0.05,
            batch_count,
            bm,
            bk,
            bn,
        )
        .expect("CUDA F16xI8 batch_matmul backward performance sample");
    });
    let batch_backward_bf16_i8_us = median_cuda_us(|| {
        let _ = crate::ops::cuda::batch_matmul_backward_bf16_i8_no_host(
            &batch_grad_buf,
            &lhs_bf16_buf,
            &rhs_i8_buf,
            0.05,
            batch_count,
            bm,
            bk,
            bn,
        )
        .expect("CUDA BF16xI8 batch_matmul backward performance sample");
    });
    let batch_backward_f32_i8_us = median_cuda_us(|| {
        let _ = crate::ops::cuda::batch_matmul_backward_f32_i8_no_host(
            &batch_grad_buf,
            &lhs_f32,
            &rhs_i8_buf,
            0.05,
            batch_count,
            bm,
            bk,
            bn,
        )
        .expect("CUDA F32xI8 batch_matmul backward performance sample");
    });
    let batch_backward_i8_f16_us = median_cuda_us(|| {
        let _ = crate::ops::cuda::batch_matmul_backward_i8_f16_no_host(
            &batch_grad_buf,
            &lhs_i8_buf,
            0.05,
            &rhs_f16_buf,
            batch_count,
            bm,
            bk,
            bn,
        )
        .expect("CUDA I8xF16 batch_matmul backward performance sample");
    });
    let batch_backward_i8_bf16_us = median_cuda_us(|| {
        let _ = crate::ops::cuda::batch_matmul_backward_i8_bf16_no_host(
            &batch_grad_buf,
            &lhs_i8_buf,
            0.05,
            &rhs_bf16_buf,
            batch_count,
            bm,
            bk,
            bn,
        )
        .expect("CUDA I8xBF16 batch_matmul backward performance sample");
    });
    let batch_backward_i8_f32_us = median_cuda_us(|| {
        let _ = crate::ops::cuda::batch_matmul_backward_i8_f32_no_host(
            &batch_grad_buf,
            &lhs_i8_buf,
            0.05,
            &rhs_f32,
            batch_count,
            bm,
            bk,
            bn,
        )
        .expect("CUDA I8xF32 batch_matmul backward performance sample");
    });

    println!(
        "cuda resident mixed backward: matmul f16xi8={backward_f16_i8_us:.1}us bf16xi8={backward_bf16_i8_us:.1}us f32xi8={backward_f32_i8_us:.1}us i8xf16={backward_i8_f16_us:.1}us i8xbf16={backward_i8_bf16_us:.1}us i8xf32={backward_i8_f32_us:.1}us; batch_matmul f16xi8={batch_backward_f16_i8_us:.1}us bf16xi8={batch_backward_bf16_i8_us:.1}us f32xi8={batch_backward_f32_i8_us:.1}us i8xf16={batch_backward_i8_f16_us:.1}us i8xbf16={batch_backward_i8_bf16_us:.1}us i8xf32={batch_backward_i8_f32_us:.1}us"
    );
}
