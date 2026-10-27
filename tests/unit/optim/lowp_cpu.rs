use super::conditional_tests::values;
use super::*;
use crate::cpu::CpuKernelContext;
use crate::precision::{PrecisionConfig, with_precision_config};
use std::{num::NonZeroUsize, sync::Arc};

fn layout<T: Clone + Default>(values: &[T], order: usize) -> ArrayD<T> {
    match order {
        0 => ArrayD::from_shape_vec(IxDyn(&[2, 3]), values.to_vec()).unwrap(),
        1 => ArrayD::from_shape_fn(IxDyn(&[3, 2]), |i| values[i[1] * 3 + i[0]].clone())
            .reversed_axes(),
        _ => ArrayD::from_shape_fn(IxDyn(&[2, 6]), |i| values[i[0] * 3 + i[1] / 2].clone())
            .slice_move(ndarray::s![.., ..;2])
            .into_dyn(),
    }
}

fn tensor(values: &[f32], dtype: DType, order: usize) -> Tensor {
    let t = Tensor::parameter_with_dtype(layout(values, 0), DType::F32);
    match dtype {
        DType::F32 => t.set_array_f32_with_dtype(layout(values, order), dtype),
        DType::F16 => t.set_array_f16_with_dtype(
            layout(
                &values
                    .iter()
                    .map(|v| half::f16::from_f32(*v))
                    .collect::<Vec<_>>(),
                order,
            ),
            dtype,
        ),
        DType::BF16 => t.set_array_bf16_with_dtype(
            layout(
                &values
                    .iter()
                    .map(|v| half::bf16::from_f32(*v))
                    .collect::<Vec<_>>(),
                order,
            ),
            dtype,
        ),
        DType::I8 => t.set_array_i8_with_dtype(
            layout(
                &values
                    .iter()
                    .map(|v| (*v * 100.0).round() as i8)
                    .collect::<Vec<_>>(),
                order,
            ),
            0.01,
            dtype,
        ),
    }
    t
}

fn quantize(x: f32, dtype: DType) -> f32 {
    match dtype {
        DType::F32 => x,
        DType::F16 => half::f16::from_f32(x).to_f32(),
        DType::BF16 => half::bf16::from_f32(x).to_f32(),
        DType::I8 => (x * 100.0).round().clamp(-127.0, 127.0) * 0.01,
    }
}

#[test]
fn lowp_cpu_layout_cache_cow_and_gradient_semantics() {
    let _cpu = cuda::set_enabled_scoped(false);
    with_precision_config(
        PrecisionConfig {
            parameter_dtype: DType::F32,
            runtime_dtype: DType::F32,
            allow_parameter_dtype_copies: true,
        },
        || {
            for dtype in [DType::F32, DType::F16, DType::BF16, DType::I8] {
                for state in [DType::F16, DType::BF16] {
                    // Mix row-major, column-major, and genuinely strided operands.
                    for orders in [[0, 0, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [2, 1, 2, 2]] {
                        for adam in [false, true] {
                            let w = tensor(&[0.8, -0.4, 0.2, -0.1, 0.6, -0.3], dtype, orders[0]);
                            let m = tensor(&[0.0; 6], state, orders[1]);
                            let v = tensor(&[0.0; 6], state, orders[2]);
                            let mut expected = values(&w);
                            let mut em = vec![0.0f32; 6];
                            let mut ev = em.clone();
                            // Retain storage aliases, so the update must obey ndarray COW.
                            let held = [&w, &m, &v].map(|t| {
                                let i = t.0.borrow();
                                (
                                    i.data.clone(),
                                    i.f16_data.clone(),
                                    i.bf16_data.clone(),
                                    i.i8_data.clone(),
                                )
                            });
                            let old = [&w, &m, &v].map(|t| t.export_raw());
                            for step in 1..=8 {
                                // Exercise previously materialized read-only F32 caches.
                                for t in [&w, &m, &v] {
                                    t.with_storage_view_preferring(
                                        StoragePreference::F32Compute,
                                        |view| assert!(matches!(view, TensorStorageView::F32(_))),
                                    );
                                }
                                let g: Vec<_> = (0..6)
                                    .map(|j| {
                                        if step == 4 {
                                            0.0
                                        } else {
                                            [0.3, -0.7, 0.5, -0.2, 0.1, -0.4][j] * step as f32
                                        }
                                    })
                                    .collect();
                                w.0.borrow_mut().grad = Some(layout(&g, orders[3]).into_shared());
                                let c = lowp_cpu::AdamConfig {
                                    lr: 0.01,
                                    beta1: 0.9,
                                    beta2: 0.999,
                                    correction1: 1.0 - 0.9f32.powi(step),
                                    correction2: 1.0 - 0.999f32.powi(step),
                                    eps: 1e-8,
                                };
                                if adam {
                                    assert!(lowp_cpu::adam(&w, &m, &v, c));
                                } else {
                                    assert!(lowp_cpu::momentum(&w, &m, 0.01, 0.9));
                                }
                                for j in 0..6 {
                                    let nm = if adam {
                                        0.9 * em[j] + (1.0 - 0.9) * g[j]
                                    } else {
                                        0.9 * em[j] + g[j]
                                    };
                                    let nv = 0.999 * ev[j] + (1.0 - 0.999) * g[j] * g[j];
                                    let delta = if adam {
                                        (nm / c.correction1) / ((nv / c.correction2).sqrt() + c.eps)
                                    } else {
                                        nm
                                    };
                                    expected[j] = quantize(expected[j] - 0.01 * delta, dtype);
                                    em[j] = quantize(nm, state);
                                    if adam {
                                        ev[j] = quantize(nv, state);
                                    }
                                }
                                for (got, expected) in values(&w).iter().zip(&expected) {
                                    assert!(
                                        (got - expected).abs() < 2e-6,
                                        "{dtype:?}/{state:?} {orders:?} adam={adam}: {got} != {expected}"
                                    );
                                }
                                assert_eq!(values(&m), em);
                                assert_eq!(values(&v), ev);
                                assert_eq!(w.dtype(), dtype);
                                assert_eq!(m.dtype(), state);
                                assert_eq!(v.dtype(), state);
                                assert!(w.has_pending_grad());
                                assert_eq!(
                                    w.grad_arc().unwrap().iter().copied().collect::<Vec<_>>(),
                                    g
                                );
                                for t in [&w, &m] {
                                    if t.dtype() != DType::F32 {
                                        assert!(!t.has_host_f32_data());
                                    }
                                }
                                // Rebuilding the compute cache must observe the new native values.
                                w.with_storage_view_preferring(
                                    StoragePreference::F32Compute,
                                    |view| match view {
                                        TensorStorageView::F32(data) => assert_eq!(
                                            data.iter().copied().collect::<Vec<_>>(),
                                            expected.clone()
                                        ),
                                        _ => unreachable!(),
                                    },
                                );
                            }
                            for (index, (f32data, f16data, bf16data, i8data)) in
                                held.into_iter().enumerate()
                            {
                                let t = Tensor::new(ArrayD::zeros(IxDyn(&[0])));
                                {
                                    let mut i = t.0.borrow_mut();
                                    i.data = f32data;
                                    i.f16_data = f16data;
                                    i.bf16_data = bf16data;
                                    i.i8_data = i8data;
                                    i.storage_dtype = old[index].1;
                                    i.i8_scale = if old[index].1 == DType::I8 {
                                        Some(0.01)
                                    } else {
                                        None
                                    };
                                }
                                assert_eq!(
                                    bincode::serialize(&t.export_raw()).unwrap(),
                                    bincode::serialize(&old[index]).unwrap()
                                );
                            }
                        }
                    }
                }
            }
        },
    );
}

#[test]
fn lowp_cpu_parallel_tail_matches_serial_and_skips_missing_gradients() {
    let _cpu = cuda::set_enabled_scoped(false);
    let pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap(),
    );
    for state in [DType::F16, DType::BF16] {
        for adam in [false, true] {
            let make = || {
                let w = Tensor::parameter_with_dtype(
                    ArrayD::from_shape_fn(IxDyn(&[32769]), |i| (i[0] % 19) as f32 * 0.01),
                    DType::F32,
                );
                let m = Tensor::new_with_dtype(ArrayD::zeros(IxDyn(&[32769])), state);
                let v = Tensor::new_with_dtype(ArrayD::zeros(IxDyn(&[32769])), state);
                (w, m, v)
            };
            let serial = make();
            let parallel = make();
            for step in 1..=6 {
                for (tuple, limit) in [(&serial, 1), (&parallel, 2)] {
                    let (w, m, v) = tuple;
                    if step == 3 {
                        w.zero_grad();
                    } else {
                        w.0.borrow_mut().grad = Some(
                            ArrayD::from_shape_fn(IxDyn(&[32769]), |i| {
                                ((i[0] % 17) as f32 - 8.0) * 0.125
                            })
                            .into_shared(),
                        );
                    }
                    let before =
                        bincode::serialize(&[w.export_raw(), m.export_raw(), v.export_raw()])
                            .unwrap();
                    let changed = CpuKernelContext::with_pool(pool.clone())
                        .with_task_limit(NonZeroUsize::new(limit).unwrap())
                        .scope(|| {
                            if adam {
                                lowp_cpu::adam(
                                    w,
                                    m,
                                    v,
                                    lowp_cpu::AdamConfig {
                                        lr: 0.01,
                                        beta1: 0.9,
                                        beta2: 0.999,
                                        correction1: 1.0 - 0.9f32.powi(step),
                                        correction2: 1.0 - 0.999f32.powi(step),
                                        eps: 1e-8,
                                    },
                                )
                            } else {
                                lowp_cpu::momentum(w, m, 0.01, 0.9)
                            }
                        });
                    assert_eq!(changed, step != 3);
                    if step == 3 {
                        assert_eq!(
                            bincode::serialize(&[w.export_raw(), m.export_raw(), v.export_raw()])
                                .unwrap(),
                            before
                        );
                    }
                }
                for (a, b) in [
                    (&serial.0, &parallel.0),
                    (&serial.1, &parallel.1),
                    (&serial.2, &parallel.2),
                ] {
                    assert_eq!(
                        bincode::serialize(&a.export_raw()).unwrap(),
                        bincode::serialize(&b.export_raw()).unwrap()
                    );
                }
            }
            assert!(!lowp_cpu::momentum(&serial.1, &serial.1, 0.01, 0.9));
            assert!(!lowp_cpu::adam(
                &serial.0,
                &serial.1,
                &serial.1,
                lowp_cpu::AdamConfig {
                    lr: 0.01,
                    beta1: 0.9,
                    beta2: 0.999,
                    correction1: 0.1,
                    correction2: 0.001,
                    eps: 1e-8
                }
            ));
        }
    }
}
