use super::*;
use crate::cpu::CpuKernelContext;
use crate::precision::{PrecisionConfig, with_precision_config};
use ndarray::{Array, IxDyn};
use std::{num::NonZeroUsize, sync::Arc};

#[test]
fn input_gradient_rows_observe_pool_grain_budget_and_nested_execution() {
    let pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(3)
            .build()
            .unwrap(),
    );
    let caller = std::thread::current().id();
    for budget in [0, 1, 2, 7] {
        let context = CpuKernelContext::with_pool(pool.clone());
        let context = if let Some(limit) = NonZeroUsize::new(budget) {
            context.with_task_limit(limit)
        } else {
            context
        };
        context.scope(|| {
            for (rows, dim) in [(0, 513), (2, 129), (65, 513)] {
                let x = Array2::from_shape_fn((rows, dim), |(r, c)| (r * dim + c) as f32);
                let g = x.mapv(|v| -v);
                let mut dx = Array2::zeros((rows, dim));
                let parallel = crate::cpu::should_parallelize(rows, dim * 8);
                let active = std::sync::atomic::AtomicUsize::new(0);
                let peak = std::sync::atomic::AtomicUsize::new(0);
                for_each_input_grad_row(&mut dx, x.view(), g.view(), |mut out, x, g| {
                    use std::sync::atomic::Ordering;
                    if parallel {
                        assert!(pool.current_thread_index().is_some());
                    } else {
                        assert_eq!(std::thread::current().id(), caller);
                    }
                    assert_eq!(crate::cpu::parallelism(), 1);
                    let count = active.fetch_add(1, Ordering::SeqCst) + 1;
                    peak.fetch_max(count, Ordering::SeqCst);
                    let thread = std::thread::current().id();
                    let mut inner = Array2::zeros((1, dim));
                    let xx = x.insert_axis(ndarray::Axis(0));
                    let gg = g.insert_axis(ndarray::Axis(0));
                    for_each_input_grad_row(&mut inner, xx, gg, |_, _, _| {
                        assert_eq!(std::thread::current().id(), thread);
                    });
                    for ((out, &x), &g) in out.iter_mut().zip(x).zip(g) {
                        *out = x - g;
                    }
                    active.fetch_sub(1, Ordering::SeqCst);
                });
                assert!(
                    peak.load(std::sync::atomic::Ordering::SeqCst)
                        <= if budget == 0 { 3 } else { budget.min(3) }
                );
                assert_eq!(dx, x * 2.);
                let fold_parallel = crate::cpu::should_parallelize(rows, dim * 4);
                let visits = (0..rows)
                    .map(|_| std::sync::atomic::AtomicUsize::new(0))
                    .collect::<Vec<_>>();
                let active = std::sync::atomic::AtomicUsize::new(0);
                let peak = std::sync::atomic::AtomicUsize::new(0);
                let total = crate::cpu::fold_reduce(
                    rows,
                    dim * 4,
                    || {
                        assert_eq!(crate::cpu::parallelism(), 1);
                        0usize
                    },
                    |acc, row| {
                        use std::sync::atomic::Ordering;
                        if fold_parallel {
                            assert!(pool.current_thread_index().is_some());
                        } else {
                            assert_eq!(std::thread::current().id(), caller);
                        }
                        visits[row].fetch_add(1, Ordering::SeqCst);
                        let count = active.fetch_add(1, Ordering::SeqCst) + 1;
                        peak.fetch_max(count, Ordering::SeqCst);
                        if crate::cpu::should_parallelize(rows, dim * 4) {
                            unreachable!("numeric folds must suppress nested scheduling");
                        }
                        assert_eq!(crate::cpu::parallelism(), 1);
                        let thread = std::thread::current().id();
                        let nested = crate::cpu::fold_reduce(
                            2,
                            usize::MAX,
                            || 0,
                            |a, i| {
                                assert_eq!(std::thread::current().id(), thread);
                                a + i
                            },
                            |a, b| a + b,
                        );
                        assert_eq!(nested, 1);
                        active.fetch_sub(1, Ordering::SeqCst);
                        acc + row
                    },
                    |a, b| {
                        assert_eq!(crate::cpu::parallelism(), 1);
                        a + b
                    },
                );
                assert_eq!(total, rows * rows.saturating_sub(1) / 2);
                assert!(
                    visits
                        .iter()
                        .all(|v| v.load(std::sync::atomic::Ordering::SeqCst) == 1)
                );
                assert!(
                    peak.load(std::sync::atomic::Ordering::SeqCst)
                        <= if budget == 0 { 3 } else { budget.min(3) }
                );
            }
        });
    }
}

#[test]
fn backward_policy_preserves_native_layouts_row_bits_and_fp64_gradients() {
    use crate::{
        accuracy_gate::{check, tensor, values},
        device::Device,
    };
    let _cpu = cuda::set_enabled_scoped(false);
    let pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(3)
            .build()
            .unwrap(),
    );
    with_precision_config(
        PrecisionConfig {
            parameter_dtype: DType::F32,
            runtime_dtype: DType::F32,
            allow_parameter_dtype_copies: false,
        },
        || {
            for input_dtype in [DType::F32, DType::F16, DType::BF16, DType::I8] {
                for weight_dtype in [DType::F32, DType::F16, DType::BF16, DType::I8] {
                    for (rows, dim, layout) in [
                        (2, 17, 0),
                        (43, 513, 1),
                        (43, 513, 2),
                        (0, 17, 0),
                        (3, 1, 2),
                    ] {
                        let input = tensor(
                            &[rows, dim],
                            &(0..rows * dim)
                                .map(|i| ((i * 17 + 3) % 251) as f32 / 53. - 2.3)
                                .collect::<Vec<_>>(),
                            input_dtype,
                            layout,
                            Device::Cpu,
                        );
                        let weight = tensor(
                            &[dim],
                            &(0..dim)
                                .map(|i| ((i * 13 + 11) % 97) as f32 / 59. - 0.8)
                                .collect::<Vec<_>>(),
                            weight_dtype,
                            layout,
                            Device::Cpu,
                        );
                        let x = values(&input);
                        let w = values(&weight);
                        let seed = Array::from_shape_fn(IxDyn(&[rows, dim]), |i| {
                            (((i[0] * dim + i[1]) * 19 + 7) % 127) as f32 / 127. - 0.4
                        });
                        let g = seed.iter().copied().collect::<Vec<_>>();
                        let norm = RMSNorm { weight, eps: 1e-5 };
                        let mut dx64 = vec![0.; x.len()];
                        let mut dw64 = vec![0.; dim];
                        let mut dx_bits = vec![0; x.len()];
                        for r in 0..rows {
                            let offset = r * dim;
                            let row = &x[offset..offset + dim];
                            let inv64 = (row.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>()
                                / dim as f64
                                + f64::from(norm.eps))
                            .sqrt()
                            .recip();
                            let dot64 = (0..dim)
                                .map(|j| {
                                    f64::from(g[offset + j])
                                        * f64::from(w[j])
                                        * f64::from(row[j])
                                        * inv64
                                })
                                .sum::<f64>()
                                / dim as f64;
                            let sum_sq = row.iter().fold(0.0f32, |acc, &v| acc + v * v);
                            let inv = 1.0 / (sum_sq / dim as f32 + norm.eps).sqrt();
                            let mut dot = 0.0f32;
                            for j in 0..dim {
                                dot += (g[offset + j] * w[j]) * (row[j] * inv);
                            }
                            let mean_dot = dot * (1.0 / dim as f32);
                            for j in 0..dim {
                                dx64[offset + j] = inv64
                                    * (f64::from(g[offset + j]) * f64::from(w[j])
                                        - f64::from(row[j]) * inv64 * dot64);
                                dw64[j] += f64::from(g[offset + j]) * f64::from(row[j]) * inv64;
                                dx_bits[offset + j] = (inv
                                    * (g[offset + j] * w[j] - (row[j] * inv) * mean_dot))
                                    .to_bits();
                            }
                        }
                        for budget in [1, 3] {
                            input.zero_grad();
                            norm.weight.zero_grad();
                            let context = CpuKernelContext::with_pool(pool.clone())
                                .with_task_limit(NonZeroUsize::new(budget).unwrap());
                            let output = context.scope(|| norm.forward(input.clone()));
                            context.scope(|| output.backward_with_grad(seed.clone()));
                            let dx = input.grad().unwrap();
                            let dw = norm.weight.grad().unwrap();
                            let dx = dx.iter().copied().collect::<Vec<_>>();
                            let dw = dw.iter().copied().collect::<Vec<_>>();
                            assert_eq!(dx.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), dx_bits);
                            check("rms dx", &dx, &dx64, 3e-6, 3e-5);
                            check("rms dw", &dw, &dw64, 3e-6, 3e-5);
                            assert_eq!(input.dtype(), input_dtype);
                            assert_eq!(norm.weight.dtype(), weight_dtype);
                            if rows == 2 && budget == 3 {
                                println!(
                                    "rms_backward_policy {}",
                                    serde_json::json!({"input_dtype":format!("{input_dtype:?}"),"weight_dtype":format!("{weight_dtype:?}"),"rows":rows,"dim":dim,"input":x,"weight":w,"seed":g,"dx":dx,"dw":dw})
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
fn scoped_input_and_weight_training_descends_for_sgd_and_adam() {
    use crate::{
        accuracy_gate::tensor,
        device::Device,
        loss::MSELoss,
        optim::{Adam, Optimizer, SGD},
    };
    let _cpu = cuda::set_enabled_scoped(false);
    let pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(3)
            .build()
            .unwrap(),
    );
    let context = CpuKernelContext::with_pool(pool).with_task_limit(NonZeroUsize::new(3).unwrap());
    with_precision_config(
        PrecisionConfig {
            parameter_dtype: DType::F32,
            runtime_dtype: DType::F32,
            allow_parameter_dtype_copies: false,
        },
        || {
            for dtype in [DType::F32, DType::F16, DType::BF16] {
                for adam in [false, true] {
                    let (rows, dim) = (65, 129);
                    let input = tensor(
                        &[rows, dim],
                        &(0..rows * dim)
                            .map(|i| ((i * 37 % 251) as f32 - 125.) / 43.)
                            .collect::<Vec<_>>(),
                        dtype,
                        1,
                        Device::Cpu,
                    );
                    let norm = RMSNorm::new_with_dtype(dim, 1e-5, dtype);
                    let target = Tensor::from_array_no_grad(Array::zeros(IxDyn(&[rows, dim])));
                    let params = vec![input.clone(), norm.weight.clone()];
                    let mut optimizer: Box<dyn Optimizer> = if adam {
                        Box::new(Adam::new_with_dtype(params, 0.03, DType::F32))
                    } else {
                        Box::new(SGD::new_with_dtype(params, 10., DType::F32))
                    };
                    let mut losses = Vec::new();
                    for step in 0..=48 {
                        let loss =
                            context.scope(|| MSELoss::apply(&norm.forward(input.clone()), &target));
                        losses.push(*loss.detach().data().iter().next().unwrap());
                        if step < 48 {
                            context.scope(|| {
                                loss.backward();
                                optimizer.step();
                                optimizer.zero_grad();
                            });
                        }
                        assert_eq!(input.dtype(), dtype);
                        assert_eq!(norm.weight.dtype(), dtype);
                    }
                    assert!(losses.iter().all(|v| v.is_finite()));
                    assert!(losses[48] < losses[0] * 0.1);
                    assert!(
                        losses[41..].iter().sum::<f32>() < losses[..8].iter().sum::<f32>() * 0.1
                    );
                    println!(
                        "rms_backward_training {}",
                        serde_json::json!({"dtype":format!("{dtype:?}"),"adam":adam,"losses":losses})
                    );
                }
            }
        },
    );
}
