use super::*;
use crate::accuracy_gate::{check, run, tensor, values};
use crate::autograd::{Device, no_grad};

fn case(device: Device, xd: DType, td: DType, order: usize, pattern: usize, flags: usize) {
    let labels = [
        [0.2, 0.3, 0.5, 0.1, 0.6, 0.3],
        [0.0, 0.5, 1.5, 0.25, 0.25, 0.0],
        [0.0, 0.0, 0.0, 0.1, 0.2, 0.7],
        [-0.2, 0.5, 0.7, 1.0, -1.0, 0.0],
    ];
    let x = tensor(
        &[2, 3],
        &[1000.0, 0.0, -1000.0, 3.0, 2.0, -3.0],
        xd,
        order,
        Device::Cpu,
    );
    let t = tensor(&[2, 3], &labels[pattern], td, order, Device::Cpu);
    let xv = values(&x);
    let tv = values(&t);
    let x_grad = flags == 1 || flags == 3;
    let t_grad = flags == 2 || flags == 3;
    x.0.borrow_mut().requires_grad = x_grad;
    t.0.borrow_mut().requires_grad = t_grad;
    if device == Device::Cuda {
        x.to_device_inplace(device);
        t.to_device_inplace(device);
    }
    let mut expected_loss = 0.0;
    let mut expected_dx = Vec::new();
    let mut expected_dt = Vec::new();
    for (row, target) in xv.chunks(3).zip(tv.chunks(3)) {
        let max = row
            .iter()
            .map(|v| *v as f64)
            .fold(f64::NEG_INFINITY, f64::max);
        let sum = row.iter().map(|v| (*v as f64 - max).exp()).sum::<f64>();
        let mass = target.iter().map(|v| *v as f64).sum::<f64>();
        for (&v, &label) in row.iter().zip(target) {
            let nll = (max - v as f64) + sum.ln();
            expected_loss += label as f64 * nll / 2.0;
            expected_dx.push((mass * (v as f64 - max).exp() / sum - label as f64) * 0.375);
            expected_dt.push(nll * 0.375);
        }
    }
    let loss = if flags == 0 {
        no_grad(|| CrossEntropyLoss::apply(&x, &t))
    } else {
        CrossEntropyLoss::apply(&x, &t)
    };
    check(
        "dense CE loss",
        &values(&loss),
        &[expected_loss],
        2e-6,
        2e-6,
    );
    assert_eq!(loss.requires_grad(), flags != 0);
    if flags != 0 {
        let backward = loss.0.borrow().backward_op.clone().unwrap();
        backward(&arr0(0.75).into_dyn().view());
    }
    if device == Device::Cuda {
        if x_grad {
            assert!(x.cloned_cuda_f32_grad().is_some());
            assert!(!x.has_host_grad());
        }
        if t_grad {
            assert!(t.cloned_cuda_f32_grad().is_some());
            assert!(!t.has_host_grad());
        }
    }
    let dx = x.grad().map(|a| a.iter().copied().collect::<Vec<_>>());
    let dt = t.grad().map(|a| a.iter().copied().collect::<Vec<_>>());
    assert_eq!(dx.is_some(), x_grad, "logits gradient presence");
    assert_eq!(dt.is_some(), t_grad, "target gradient presence");
    if let Some(dx) = &dx {
        check("dense CE dx", dx, &expected_dx, 3e-6, 3e-6);
    }
    if let Some(dt) = &dt {
        check("dense CE dt", dt, &expected_dt, 3e-6, 3e-6);
    }
    if std::env::var_os("LUMEN_OPERATOR_AUDIT").is_some() {
        println!(
            "DENSE_CE_AUDIT={}",
            serde_json::json!({
                "device":format!("{device:?}"),"xd":format!("{xd:?}"),"td":format!("{td:?}"),
                "order":order,"pattern":pattern,"flags":flags,"x":xv,"target":tv,
                "loss":values(&loss),"dx":dx,"dt":dt,"upstream":0.75,
            })
        );
    }
}

fn matrix(device: Device) {
    for xd in [DType::F32, DType::F16, DType::BF16, DType::I8] {
        for td in [DType::F32, DType::F16, DType::BF16, DType::I8] {
            for order in 0..3 {
                for pattern in 0..4 {
                    for flags in 0..4 {
                        case(device, xd, td, order, pattern, flags);
                    }
                }
            }
        }
    }
}

#[test]
fn dense_ce_weighted_logits_gradient_cpu() {
    run(Device::Cpu, |d| case(d, DType::F32, DType::F32, 0, 1, 1));
}

#[test]
fn dense_ce_target_gradient_cpu() {
    run(Device::Cpu, |d| case(d, DType::F32, DType::F32, 0, 0, 2));
}

#[test]
fn dense_ce_matrix_cpu() {
    run(Device::Cpu, matrix);
}

#[cfg(feature = "cuda")]
#[test]
fn dense_ce_matrix_cuda() {
    run(Device::Cuda, matrix);
}

fn aliased_targets(device: Device) {
    for dtype in [DType::F32, DType::F16, DType::BF16, DType::I8] {
        for order in 0..3 {
            let x = tensor(
                &[2, 3],
                &[0.2, 0.1, 0.7, 1.0, 0.5, -0.2],
                dtype,
                order,
                Device::Cpu,
            );
            let initial = values(&x);
            if device == Device::Cuda {
                x.to_device_inplace(device);
            }
            let mut expected = Vec::new();
            for row in initial.chunks(3) {
                let max = row
                    .iter()
                    .map(|v| *v as f64)
                    .fold(f64::NEG_INFINITY, f64::max);
                let sum = row.iter().map(|v| (*v as f64 - max).exp()).sum::<f64>();
                let mass = row.iter().map(|v| *v as f64).sum::<f64>();
                for &value in row {
                    let v = value as f64;
                    expected.push((mass * (v - max).exp() / sum - v + (max - v) + sum.ln()) / 2.0);
                }
            }
            CrossEntropyLoss::apply(&x, &x).backward();
            let grad = x.grad().expect("aliased target and logits gradient");
            check(
                "aliased dense CE",
                &grad.iter().copied().collect::<Vec<_>>(),
                &expected,
                3e-6,
                3e-6,
            );
        }
    }
}

#[test]
fn dense_ce_aliased_target_cpu() {
    run(Device::Cpu, aliased_targets);
}

#[cfg(feature = "cuda")]
#[test]
fn dense_ce_aliased_target_cuda() {
    run(Device::Cuda, aliased_targets);
}

#[test]
fn cached_parameter_compute_views_allow_same_tensor_aliases() {
    run(Device::Cpu, |_| {
        for dtype in [DType::F16, DType::BF16, DType::I8] {
            let x = tensor(
                &[2, 3],
                &[0.2, 0.1, 0.7, 1.0, 0.5, -0.2],
                dtype,
                1,
                Device::Cpu,
            );
            for preference in [StoragePreference::F32Compute, StoragePreference::Auto] {
                x.with_storage_view_preferring(StoragePreference::F32Compute, |outer| {
                    let TensorStorageView::F32(outer) = outer else {
                        unreachable!()
                    };
                    x.with_storage_view_preferring(preference, |inner| {
                        let TensorStorageView::F32(inner) = inner else {
                            unreachable!()
                        };
                        assert_eq!(outer, inner);
                    });
                });
                assert_eq!(x.dtype(), dtype);
            }
            let loss = MSELoss::apply(&x, &x);
            loss.backward();
            assert_eq!(values(&loss), vec![0.0]);
            assert!(x.grad().unwrap().iter().all(|v| *v == 0.0));
        }
    });
}

fn mse_requested_gradients(device: Device) {
    crate::precision::with_precision_config(
        crate::precision::PrecisionConfig {
            parameter_dtype: DType::F32,
            runtime_dtype: DType::F32,
            allow_parameter_dtype_copies: false,
        },
        || {
            for xd in [DType::F32, DType::F16, DType::BF16, DType::I8] {
                for td in [DType::F32, DType::F16, DType::BF16, DType::I8] {
                    for order in 0..3 {
                        for flags in 1..4 {
                            let x = tensor(
                                &[2, 3],
                                &[0.2, 0.1, 0.7, 1.0, 0.5, -0.2],
                                xd,
                                order,
                                Device::Cpu,
                            );
                            let t = tensor(
                                &[2, 3],
                                &[-0.1, 0.8, -0.3, 0.4, 0.2, 0.1],
                                td,
                                order,
                                Device::Cpu,
                            );
                            let xv = values(&x);
                            let tv = values(&t);
                            let want_x = flags != 2;
                            let want_t = flags != 1;
                            x.0.borrow_mut().requires_grad = want_x;
                            t.0.borrow_mut().requires_grad = want_t;
                            if device == Device::Cuda {
                                x.to_device_inplace(device);
                                t.to_device_inplace(device);
                            }
                            MSELoss::apply(&x, &t).backward();
                            assert_eq!(x.dtype(), xd, "MSE must preserve input storage");
                            assert_eq!(t.dtype(), td, "MSE must preserve target storage");
                            if device == Device::Cpu {
                                if xd != DType::F32 {
                                    assert!(!x.has_host_f32_data());
                                }
                                if td != DType::F32 {
                                    assert!(!t.has_host_f32_data());
                                }
                            }
                            let dx = x.grad();
                            let dt = t.grad();
                            assert_eq!(dx.is_some(), want_x);
                            assert_eq!(dt.is_some(), want_t);
                            let expected = xv
                                .iter()
                                .zip(&tv)
                                .map(|(&a, &b)| 2.0 * (a as f64 - b as f64) / 6.0)
                                .collect::<Vec<_>>();
                            if let Some(dx) = dx {
                                check(
                                    "MSE requested dx",
                                    &dx.iter().copied().collect::<Vec<_>>(),
                                    &expected,
                                    3e-6,
                                    3e-6,
                                );
                            }
                            if let Some(dt) = dt {
                                check(
                                    "MSE requested dt",
                                    &dt.iter().copied().collect::<Vec<_>>(),
                                    &expected.iter().map(|v| -v).collect::<Vec<_>>(),
                                    3e-6,
                                    3e-6,
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
fn mse_native_storage_and_requested_gradients_cpu() {
    run(Device::Cpu, mse_requested_gradients);
}

#[cfg(feature = "cuda")]
#[test]
fn mse_native_storage_and_requested_gradients_cuda() {
    run(Device::Cuda, mse_requested_gradients);
}
