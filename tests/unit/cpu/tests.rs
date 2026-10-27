use super::*;
use crate::autograd::{Tensor, is_no_grad, no_grad};
use crate::ops::matmul::{
    SliceRef, dual_matvec_rowmajor_parallel_mixed, matvec_argmax_rowmajor_parallel_mixed,
    matvec_rowmajor_parallel_mixed,
};
use half::{bf16, f16};
use std::collections::HashSet;
use std::sync::Mutex;

fn pool(workers: usize) -> Arc<rayon::ThreadPool> {
    Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .unwrap(),
    )
}

#[test]
fn inference_phases_share_pool_and_restore_nested_policy_on_panic() {
    let workers = pool(3);
    let context = CpuInferenceContext::with_pool(workers.clone())
        .with_prefill_task_limit(NonZeroUsize::new(2).unwrap())
        .with_decode_task_limit(NonZeroUsize::new(1).unwrap());
    let outer = CpuKernelContext::default().with_task_limit(NonZeroUsize::new(3).unwrap());
    outer.scope(|| {
        let caller = std::thread::current().id();
        context.prefill(|| {
            CONTEXT.with(|slot| {
                let current = slot.borrow();
                assert!(Arc::ptr_eq(current.pool.as_ref().unwrap(), &workers));
                assert_eq!(current.task_limit.unwrap().get(), 2);
            });
            let mut output = [0; 17];
            for_each_chunk_mut(&mut output, 1, 100_000, |(_, value)| {
                assert!(workers.current_thread_index().is_some());
                value.fill(1);
            });
            assert_eq!(output, [1; 17]);
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                context.decode(|| {
                    for_each_chunk_mut(&mut output, 1, 100_000, |(_, value)| {
                        assert_eq!(std::thread::current().id(), caller);
                        value.fill(2);
                    });
                    panic!("restore phase policy");
                });
            }));
            assert!(result.is_err());
            assert_eq!(output, [2; 17]);
            assert_eq!(
                CONTEXT.with(|slot| slot.borrow().task_limit.unwrap().get()),
                2
            );
        });
        assert_eq!(
            CONTEXT.with(|slot| slot.borrow().task_limit.unwrap().get()),
            3
        );
    });
}

#[test]
fn inference_phases_preserve_caller_precision_and_non_send_tensor() {
    use crate::autograd::{Tensor, is_no_grad, no_grad};
    use crate::precision::{DType, default_activation_dtype, with_runtime_component_dtypes};
    let base = CpuKernelContext::with_pool(pool(2)).with_task_limit(NonZeroUsize::new(1).unwrap());
    let context = CpuInferenceContext::new(base);
    let tensor = Tensor::new(ndarray::Array::from_vec(vec![1., 2.]).into_dyn());
    no_grad(|| {
        with_runtime_component_dtypes(Some(DType::BF16), None, || {
            let output = context.prefill(|| {
                context.decode(|| {
                    assert!(is_no_grad());
                    assert_eq!(default_activation_dtype(), DType::BF16);
                    assert_eq!(
                        CONTEXT.with(|slot| slot.borrow().task_limit.unwrap().get()),
                        1
                    );
                    tensor.clone()
                })
            });
            assert_eq!(output.data(), tensor.data());
        })
    });
}

#[test]
fn small_jobs_and_serial_policy_stay_on_caller() {
    let ctx = CpuKernelContext::with_pool(pool(3));
    let caller = std::thread::current().id();
    ctx.scope(|| {
        let mut out = [0; 7];
        for_each_chunk_mut(&mut out, 3, 1, |(i, values)| {
            assert_eq!(std::thread::current().id(), caller);
            values.fill(i);
        });
        assert_eq!(out, [0, 0, 0, 1, 1, 1, 2]);
    });
    ctx.with_task_limit(NonZeroUsize::new(1).unwrap())
        .scope(|| {
            let mut out = [0; 11];
            for_each_chunk_mut(&mut out, 1, usize::MAX, |(_, _)| {
                assert_eq!(std::thread::current().id(), caller);
            });
        });
}

#[test]
fn scope_keeps_non_send_tensors_and_autograd_state_on_caller() {
    let ctx = CpuKernelContext::with_pool(pool(3));
    no_grad(|| {
        let x = Tensor::new(ndarray::Array::from_vec(vec![1., 2.]).into_dyn());
        let result = ctx.scope(|| {
            assert!(is_no_grad());
            x.clone()
        });
        assert_eq!(result.data(), x.data());
        assert!(is_no_grad());
    });
}

#[test]
fn scope_restores_policy_after_nesting_and_unwind() {
    let original = CONTEXT.with(|slot| slot.borrow().task_limit);
    let serial = CpuKernelContext::default().with_task_limit(NonZeroUsize::new(1).unwrap());
    serial.scope(|| {
        assert_eq!(
            CONTEXT.with(|slot| slot.borrow().task_limit.unwrap().get()),
            1
        );
        CpuKernelContext::default().scope(|| {
            assert!(CONTEXT.with(|slot| slot.borrow().task_limit.is_none()));
        });
        assert_eq!(
            CONTEXT.with(|slot| slot.borrow().task_limit.unwrap().get()),
            1
        );
    });
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        serial.scope(|| panic!("unwind policy"))
    }));
    assert!(result.is_err());
    assert_eq!(CONTEXT.with(|slot| slot.borrow().task_limit), original);
    assert!(!IN_KERNEL.with(Cell::get));
}

#[test]
fn shared_pool_is_reused_and_nested_kernel_work_is_serial() {
    let workers = pool(3);
    let ctx = CpuKernelContext::with_pool(workers.clone());
    for _ in 0..2 {
        ctx.scope(|| {
            let mut out = [0; 17];
            for_each_chunk_mut(&mut out, 3, 100_000, |(i, values)| {
                assert_eq!(
                    workers.current_thread_index(),
                    rayon::current_thread_index()
                );
                assert!(workers.current_thread_index().is_some());
                let id = std::thread::current().id();
                let mut inner = [0; 5];
                for_each_chunk_mut(&mut inner, 1, usize::MAX, |(_, v)| {
                    assert_eq!(std::thread::current().id(), id);
                    v.fill(1);
                });
                values.fill(i);
            });
            assert_eq!(out[16], 5);
        });
    }
}

#[test]
fn concurrent_contexts_share_one_worker_pool() {
    let workers = pool(3);
    let ids = Mutex::new(HashSet::new());
    std::thread::scope(|scope| {
        for _ in 0..4 {
            let ids = &ids;
            let workers = workers.clone();
            scope.spawn(move || {
                CpuKernelContext::with_pool(workers.clone()).scope(|| {
                    let mut out = [0; 31];
                    for_each_chunk_mut(&mut out, 1, 100_000, |(_, _)| {
                        assert!(workers.current_thread_index().is_some());
                        ids.lock().unwrap().insert(std::thread::current().id());
                    });
                });
            });
        }
    });
    assert!(ids.into_inner().unwrap().len() <= 3);
}

#[test]
fn task_limit_bounds_active_groups_and_argmax_ties_are_stable() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let ctx = CpuKernelContext::with_pool(pool(7)).with_task_limit(NonZeroUsize::new(2).unwrap());
    let active = AtomicUsize::new(0);
    let peak = AtomicUsize::new(0);
    ctx.scope(|| {
        assert_eq!(parallelism(), 2);
        let mut out = vec![0; 97];
        for_each_chunk_mut(&mut out, 3, 100_000, |(i, values)| {
            assert_eq!(parallelism(), 1);
            let current = active.fetch_add(1, Ordering::SeqCst) + 1;
            peak.fetch_max(current, Ordering::SeqCst);
            values.fill(i);
            active.fetch_sub(1, Ordering::SeqCst);
        });
        assert!(peak.load(Ordering::SeqCst) <= 2);
        for (i, value) in out.iter().enumerate() {
            assert_eq!(*value, i / 3);
        }
        assert_eq!(
            map_reduce(
                97,
                100_000,
                |i| (i, 1.),
                || (0, f32::NEG_INFINITY),
                |a, b| if a.1 >= b.1 { a } else { b }
            )
            .0,
            0
        );
    });
}

#[test]
fn indexed_views_keep_pool_budget_and_nested_execution_for_strided_outputs() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let workers = pool(3);
    for limit in [
        None,
        NonZeroUsize::new(1),
        NonZeroUsize::new(2),
        NonZeroUsize::new(5),
    ] {
        let mut context = CpuKernelContext::with_pool(workers.clone());
        if let Some(limit) = limit {
            context = context.with_task_limit(limit);
        }
        let active = AtomicUsize::new(0);
        let peak = AtomicUsize::new(0);
        let caller = std::thread::current().id();
        let mut out = ndarray::Array2::<usize>::zeros((3, 31));
        context.scope(|| {
            for_each_indexed(
                out.axis_iter_mut(ndarray::Axis(1)),
                100_000,
                |(i, mut column)| {
                    if limit.is_some_and(|l| l.get() == 1) {
                        assert_eq!(std::thread::current().id(), caller);
                    } else {
                        assert!(workers.current_thread_index().is_some());
                    }
                    let count = active.fetch_add(1, Ordering::SeqCst) + 1;
                    peak.fetch_max(count, Ordering::SeqCst);
                    let id = std::thread::current().id();
                    let mut nested = ndarray::Array2::<usize>::zeros((2, 5));
                    for_each_indexed(
                        nested.axis_iter_mut(ndarray::Axis(1)),
                        100_000,
                        |(_, mut values)| {
                            assert_eq!(std::thread::current().id(), id);
                            values.fill(7);
                        },
                    );
                    assert!(nested.iter().all(|v| *v == 7));
                    column.fill(i);
                    active.fetch_sub(1, Ordering::SeqCst);
                },
            )
        });
        assert!(peak.load(Ordering::SeqCst) <= limit.map_or(3, |l| l.get().min(3)));
        for ((_, col), value) in out.indexed_iter() {
            assert_eq!(*value, col);
        }
    }
}

#[test]
fn planner_handles_empty_overflow_and_uneven_budgets() {
    assert_eq!(task_count(0, 0, 0, 0), 1);
    assert_eq!(task_count(usize::MAX, usize::MAX, 7, 3), 3);
    assert_eq!(task_count(1000, 1, 32, 32), 1);
    for workers in [1, 2, 3, 7] {
        let ctx = CpuKernelContext::with_pool(pool(workers));
        ctx.scope(|| {
            for n in [0, 1, 9, 31] {
                let mut a = vec![0; n];
                let mut b = a.clone();
                let mut c = a.clone();
                for_each_chunk3_mut(&mut a, &mut b, &mut c, 7, 100_000, |(i, ((a, b), c))| {
                    a.fill(i);
                    b.fill(i + 1);
                    c.fill(i + 2);
                });
                for i in 0..n {
                    assert_eq!((a[i], b[i], c[i]), (i / 7, i / 7 + 1, i / 7 + 2));
                }
                assert_eq!(
                    map_reduce(n, 100_000, |i| i, || 0, |a, b| a + b),
                    (0..n).sum::<usize>()
                );
            }
        });
    }
}

#[test]
fn projection_dtypes_tails_and_argmax_match_scalar_across_pools() {
    for workers in [1, 2, 3, 7] {
        let ctx = CpuKernelContext::with_pool(pool(workers));
        for (n, k) in [(9, 31), (257, 517)] {
            let x: Vec<f32> = (0..k).map(|i| ((i % 13) as f32 - 6.) / 16.).collect();
            let w: Vec<f32> = (0..n * k).map(|i| ((i % 23) as f32 - 11.) / 32.).collect();
            let xf: Vec<f16> = x.iter().copied().map(f16::from_f32).collect();
            let xb: Vec<bf16> = x.iter().copied().map(bf16::from_f32).collect();
            let wf: Vec<f16> = w.iter().copied().map(f16::from_f32).collect();
            let wb: Vec<bf16> = w.iter().copied().map(bf16::from_f32).collect();
            let xi: Vec<i8> = x.iter().map(|v| (v * 16.) as i8).collect();
            let wi: Vec<i8> = w.iter().map(|v| (v * 32.) as i8).collect();
            for (input, weight) in [
                (SliceRef::F32(&x), SliceRef::F32(&w)),
                (SliceRef::F16(&xf), SliceRef::F16(&wf)),
                (SliceRef::BF16(&xb), SliceRef::BF16(&wb)),
                (SliceRef::I8(&xi, 1. / 16.), SliceRef::I8(&wi, 1. / 32.)),
                (SliceRef::F32(&x), SliceRef::I8(&wi, 1. / 32.)),
            ] {
                let expected: Vec<f32> = w
                    .chunks(k)
                    .map(|row| row.iter().zip(&x).map(|(a, b)| a * b).sum())
                    .collect();
                let mut out = vec![0.; n];
                let mut other = out.clone();
                ctx.scope(|| {
                    matvec_rowmajor_parallel_mixed(input, weight, n, k, &mut out);
                    for (a, b) in out.iter().zip(&expected) {
                        assert!((a - b).abs() < 0.002, "{a} vs {b}");
                    }
                    dual_matvec_rowmajor_parallel_mixed(
                        input, weight, weight, n, k, &mut out, &mut other,
                    );
                    for (a, b) in out.iter().zip(&other) {
                        assert!((a - b).abs() < 0.002);
                    }
                    let idx = matvec_argmax_rowmajor_parallel_mixed(input, weight, n, k);
                    assert!(
                        (out[idx] - out.iter().copied().fold(f32::NEG_INFINITY, f32::max)).abs()
                            < 0.002
                    );
                });
            }
        }
    }
}
