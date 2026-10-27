use super::*;
use crate::autograd::{
    Device, StoragePreference, TensorStorageView, set_strict_device_execution_scoped,
};
use crate::precision::{ParameterQuantization, PrecisionConfig, with_precision_config};

pub(super) fn values(tensor: &Tensor) -> Vec<f32> {
    if tensor.is_cuda() {
        cuda::download_f32(&tensor.cloned_cuda_f32_buffer().unwrap()).unwrap()
    } else {
        tensor.with_storage_view_preferring(StoragePreference::Native, |view| match view {
            TensorStorageView::F32(data) => data.iter().copied().collect(),
            TensorStorageView::F16(data) => data.iter().map(|v| v.to_f32()).collect(),
            TensorStorageView::BF16(data) => data.iter().map(|v| v.to_f32()).collect(),
        })
    }
}

fn rounded(value: f64, dtype: DType, scale: f32) -> f64 {
    let x = value as f32;
    (match dtype {
        DType::F32 => x,
        DType::F16 => half::f16::from_f32(x).to_f32(),
        DType::BF16 => half::bf16::from_f32(x).to_f32(),
        DType::I8 => (x / scale).round().clamp(-127.0, 127.0) * scale,
    }) as f64
}

fn active(step: usize, index: usize) -> bool {
    match step {
        0 | 5 | 10 => false,
        1 | 2 | 6 => index.is_multiple_of(2),
        4 | 8 => index % 2 == 1,
        _ => true,
    }
}

fn check_conditional_updates(device: Device, dtype: DType, state: DType, policy: &str, len: usize) {
    let _enabled = cuda::set_enabled_scoped(device == Device::Cuda);
    let _strict = set_strict_device_execution_scoped(device == Device::Cuda);
    let params = (0..16)
        .map(|_| {
            let data = ArrayD::from_shape_vec(
                IxDyn(&[len]),
                (0..len).map(|j| [1.0, -0.5, 0.25, -0.125][j % 4]).collect(),
            )
            .unwrap();
            let tensor = if dtype == DType::I8 {
                Tensor::parameter_with_quantization(
                    data,
                    ParameterQuantization::Int8.with_scale(0.01),
                )
            } else {
                Tensor::parameter_with_dtype(data, dtype)
            };
            tensor.to_device(device)
        })
        .collect::<Vec<_>>();
    let mut opt = Adam::new_with_dtype(params.clone(), 0.01, state);
    opt = match policy {
        "fixed" => opt.with_fp32_master_weights(),
        "dynamic" => opt.with_dynamic_i8_master_weights(),
        _ => opt,
    };
    let initial = values(&params[0]);
    let mut masters = vec![[0.0f64; 4]; 16];
    for weights in &mut masters {
        for (j, weight) in weights.iter_mut().enumerate() {
            *weight = initial[j] as f64;
        }
    }
    let mut first = vec![[0.0f64; 4]; 16];
    let mut second = first.clone();
    let mut counts = [0i32; 16];
    let mut expected = masters.clone();
    let trace_enabled = std::env::var_os("LUMEN_ADAM_AUDIT").is_some();
    let mut trace = Vec::new();
    for step in 0..12 {
        opt.zero_grad();
        for (index, tensor) in params.iter().enumerate() {
            if !active(step, index) {
                continue;
            }
            // A present all-zero gradient must age moments; a missing gradient must not.
            let gradient = |j: usize| {
                if step == 4 {
                    0.0
                } else {
                    [0.25, -0.5, 1.0, -0.125][j % 4] * (1.0 + step as f32 * 0.125)
                }
            };
            let data = (0..len).map(gradient).collect::<Vec<_>>();
            let array = ArrayD::from_shape_vec(IxDyn(&[len]), data.clone()).unwrap();
            if device == Device::Cuda {
                tensor.add_grad_with_cuda_buffer(array, Some(cuda::upload_f32(&data).unwrap()));
            } else {
                tensor.add_grad(array);
            }
            counts[index] += 1;
            for j in 0..4 {
                let grad = gradient(j) as f64;
                // Independent scalar recurrence: corrections use each parameter's age.
                let beta1 = 0.9f32 as f64;
                let beta2 = 0.999f32 as f64;
                let m = beta1 * first[index][j] + (1.0 - beta1) * grad;
                let v = beta2 * second[index][j] + (1.0 - beta2) * grad * grad;
                first[index][j] = rounded(m, state, 1.0);
                second[index][j] = rounded(v, state, 1.0);
                masters[index][j] -= 0.01 * (m / (1.0 - beta1.powi(counts[index])))
                    / ((v / (1.0 - beta2.powi(counts[index]))).sqrt() + 1e-8);
            }
            let scale = if dtype == DType::I8 && policy == "dynamic" {
                masters[index]
                    .iter()
                    .map(|v| (*v as f32).abs())
                    .fold(0.0f32, f32::max)
                    / 127.0
            } else {
                tensor.quantization_scale().unwrap_or(1.0)
            };
            for j in 0..4 {
                expected[index][j] = rounded(masters[index][j], dtype, scale);
            }
            if policy == "native" {
                masters[index] = expected[index];
            }
        }
        opt.step();
        if device == Device::Cuda && len >= 32768 {
            assert_eq!(
                opt.last_cuda_batched_update_count,
                (0..16).filter(|&index| active(step, index)).count(),
                "same-age peers should retain pointer batching"
            );
        }
        for (index, tensor) in params.iter().enumerate() {
            assert_eq!(tensor.dtype(), dtype);
            if device == Device::Cuda && counts[index] > 0 {
                assert!(
                    !tensor.has_host_f32_data(),
                    "{dtype:?} {policy} step={step} param={index}: updated parameters stay resident"
                );
                assert!(!tensor.has_host_grad());
                assert!(!tensor.has_pending_grad());
                for moment in [&opt.exp_avg[index], &opt.exp_avg_sq[index]] {
                    let moment = moment.as_ref().unwrap();
                    assert!(moment.is_cuda());
                    assert_eq!(moment.dtype(), state);
                    assert!(!moment.has_host_f32_data());
                    if state != DType::F32 {
                        assert!(moment.cloned_cuda_native_lowp_buffer().is_some());
                        assert!(
                            moment.0.borrow().cuda_f32_data.is_none(),
                            "no expanded F32 state cache"
                        );
                    }
                }
            }
            let scale = tensor.quantization_scale().unwrap_or(0.0) as f64;
            let tolerance = if dtype == DType::I8 {
                scale * 1.01 + 2e-6
            } else if dtype == DType::BF16 {
                0.008
            } else if dtype == DType::F16 {
                0.001
            } else if state != DType::F32 {
                4e-4
            } else {
                3e-6
            };
            for (j, got) in values(tensor).iter().enumerate() {
                assert!(
                    (*got as f64 - expected[index][j % 4]).abs() <= tolerance,
                    "{device:?} {dtype:?}/{state:?} {policy} step={step} param={index} value={j}: {got} vs {}",
                    expected[index][j % 4]
                );
            }
        }
        if trace_enabled {
            trace.push(serde_json::json!({
                "step":step,"parameter_steps":opt.parameter_steps,
                "weights":params.iter().map(|p|values(p)[..4].to_vec()).collect::<Vec<_>>()
            }));
        }
    }
    if trace_enabled {
        println!(
            "conditional_adam_audit {}",
            serde_json::json!({
                "device":format!("{device:?}"),"dtype":format!("{dtype:?}"),
                "state_dtype":format!("{state:?}"),"policy":policy,"len":len,
                "initial":initial[..4],"trace":trace
            })
        );
    }
}

fn matrix(device: Device) {
    with_precision_config(
        PrecisionConfig {
            parameter_dtype: DType::F32,
            runtime_dtype: DType::F32,
            allow_parameter_dtype_copies: false,
        },
        || {
            for state in [DType::F32, DType::F16, DType::BF16] {
                check_conditional_updates(device, DType::F32, state, "native", 4);
                if state != DType::F32 {
                    for dtype in [DType::F16, DType::BF16, DType::I8] {
                        check_conditional_updates(device, dtype, state, "native", 4);
                    }
                }
            }
            for dtype in [DType::F16, DType::BF16, DType::I8] {
                for policy in ["native", "fixed"] {
                    check_conditional_updates(device, dtype, DType::F32, policy, 4);
                }
            }
            check_conditional_updates(device, DType::I8, DType::F32, "dynamic", 4);
        },
    );
}

#[test]
fn cpu_adam_uses_parameter_update_age() {
    matrix(Device::Cpu);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_adam_uses_parameter_update_age_and_retains_batching() {
    if !cuda::is_available() {
        return;
    }
    matrix(Device::Cuda);
    for (dtype, policy) in [(DType::F32, "native"), (DType::I8, "dynamic")] {
        check_conditional_updates(Device::Cuda, dtype, DType::F32, policy, 32768);
    }
    for (dtype, state) in [(DType::F32, DType::F16), (DType::I8, DType::BF16)] {
        check_conditional_updates(Device::Cuda, dtype, state, "native", 32769);
    }
}

#[test]
fn conditional_adam_training_descends_with_native_and_master_weights() {
    use crate::loss::MSELoss;
    for device in [Device::Cpu, Device::Cuda] {
        if device == Device::Cuda && !cuda::is_available() {
            continue;
        }
        let _enabled = cuda::set_enabled_scoped(device == Device::Cuda);
        let _strict = set_strict_device_execution_scoped(device == Device::Cuda);
        with_precision_config(
            PrecisionConfig {
                parameter_dtype: DType::F32,
                runtime_dtype: DType::F32,
                allow_parameter_dtype_copies: false,
            },
            || {
                for (dtype, state_dtype) in [
                    (DType::F32, DType::F32),
                    (DType::F16, DType::F32),
                    (DType::BF16, DType::F32),
                    (DType::I8, DType::F32),
                    (DType::F32, DType::F16),
                    (DType::F32, DType::BF16),
                ] {
                    let params = (0..8)
                        .map(|_| {
                            Tensor::parameter_with_dtype(
                                ArrayD::from_shape_vec(IxDyn(&[4]), vec![0.1, -0.1, 0.05, -0.05])
                                    .unwrap(),
                                dtype,
                            )
                            .to_device(device)
                        })
                        .collect::<Vec<_>>();
                    let target_values = [0.8f32, -0.7, 0.6, -0.5];
                    let target = Tensor::from_array_no_grad(
                        ArrayD::from_shape_vec(IxDyn(&[4]), target_values.to_vec()).unwrap(),
                    )
                    .to_device(device);
                    let initial = values(&params[0]);
                    let mut opt = Adam::new_with_dtype(params.clone(), 0.04, state_dtype);
                    if dtype == DType::I8 {
                        opt = opt.with_dynamic_i8_master_weights();
                    } else if dtype != DType::F32 {
                        opt = opt.with_fp32_master_weights();
                    }
                    let mut losses = Vec::new();
                    for step in 0..=80 {
                        let objective = params
                            .iter()
                            .flat_map(values)
                            .enumerate()
                            .map(|(j, x)| {
                                let err = x as f64 - target_values[j % 4] as f64;
                                err * err / 32.0
                            })
                            .sum::<f64>();
                        losses.push(objective);
                        if step == 80 {
                            break;
                        }
                        opt.zero_grad();
                        for (index, param) in params.iter().enumerate() {
                            let enabled = step % 17 != 0
                                && if index < 4 {
                                    step % 4 != 3
                                } else {
                                    step >= 7 && step % 3 != 1
                                };
                            if enabled {
                                MSELoss::apply(param, &target).backward();
                            }
                        }
                        opt.step();
                    }
                    assert!(losses.iter().all(|v| v.is_finite()));
                    assert!(
                        losses[80] < losses[0] * 0.05,
                        "{device:?} {dtype:?}: {losses:?}"
                    );
                    assert!(losses[71..].iter().sum::<f64>() < losses[..10].iter().sum::<f64>());
                    println!(
                        "conditional_training_audit {}",
                        serde_json::json!({
                            "device":format!("{device:?}"),"dtype":format!("{dtype:?}"),
                            "state_dtype":format!("{state_dtype:?}"),"optimizer":"adam",
                            "initial":initial,"target":target_values,"losses":losses,
                            "parameter_steps":opt.parameter_steps
                        })
                    );
                }
            },
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_lowp_momentum_state_matches_scalar_and_stays_resident() {
    if !cuda::is_available() {
        return;
    }
    let _enabled = cuda::set_enabled_scoped(true);
    let _strict = set_strict_device_execution_scoped(true);
    for state in [DType::F16, DType::BF16] {
        for dtype in [DType::F32, DType::F16, DType::BF16, DType::I8] {
            for len in [3, 32769] {
                let params = (0..16)
                    .map(|_| {
                        let data = ArrayD::from_shape_vec(
                            IxDyn(&[len]),
                            (0..len).map(|j| [1.0, -0.5, 0.25, -0.125][j % 4]).collect(),
                        )
                        .unwrap();
                        (if dtype == DType::I8 {
                            Tensor::parameter_with_quantization(
                                data,
                                ParameterQuantization::Int8.with_scale(0.01),
                            )
                        } else {
                            Tensor::parameter_with_dtype(data, dtype)
                        })
                        .to_cuda()
                    })
                    .collect::<Vec<_>>();
                let initial = values(&params[0]);
                let mut expected = (0..16)
                    .map(|_| {
                        initial
                            .iter()
                            .take(4)
                            .map(|x| *x as f64)
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                let mut velocity = vec![vec![0.0f64; initial.len().min(4)]; 16];
                let mut seen = [false; 16];
                let mut opt = SGD::new_with_dtype(params.clone(), 0.01, state).with_momentum(0.75);
                let mut trace = Vec::new();
                for step in 0..8 {
                    opt.zero_grad();
                    for (index, param) in params.iter().enumerate() {
                        if !active(step, index) {
                            continue;
                        }
                        seen[index] = true;
                        let gradient = |j: usize| {
                            if step == 4 {
                                0.0
                            } else {
                                [0.25, -0.5, 1.0, -0.125][j % 4] * (1.0 + step as f32 * 0.125)
                            }
                        };
                        let g = (0..len).map(gradient).collect::<Vec<_>>();
                        param.add_grad_with_cuda_buffer(
                            ArrayD::from_shape_vec(IxDyn(&[len]), g.clone()).unwrap(),
                            Some(cuda::upload_f32(&g).unwrap()),
                        );
                        for j in 0..expected[index].len() {
                            let next = 0.75 * velocity[index][j] + gradient(j) as f64;
                            velocity[index][j] = rounded(next, state, 1.0);
                            expected[index][j] =
                                rounded(expected[index][j] - 0.01 * next, dtype, 0.01);
                        }
                    }
                    opt.step();
                    if len >= 32768 {
                        assert_eq!(
                            opt.last_cuda_batched_update_count,
                            (0..16).filter(|&i| active(step, i)).count()
                        );
                    }
                    for (index, param) in params.iter().enumerate() {
                        assert_eq!(param.dtype(), dtype);
                        if seen[index] {
                            assert!(!param.has_host_f32_data() && !param.has_host_grad());
                            let saved = opt.velocities[index].as_ref().unwrap();
                            assert_eq!(saved.dtype(), state);
                            assert!(saved.is_cuda());
                            assert!(!saved.has_host_f32_data());
                            assert!(saved.0.borrow().cuda_f32_data.is_none());
                            assert!(saved.cloned_cuda_native_lowp_buffer().is_some());
                        }
                        let tolerance = match dtype {
                            DType::F32 => 3e-6,
                            DType::F16 => 0.001,
                            DType::BF16 => 0.008,
                            DType::I8 => 0.010001,
                        };
                        for (j, x) in values(param).iter().enumerate() {
                            assert!(
                                (*x as f64 - expected[index][j % expected[index].len()]).abs()
                                    <= tolerance,
                                "momentum {dtype:?}/{state:?} len={len} step={step} index={index} coordinate={j}: {x} vs {}",
                                expected[index][j % expected[index].len()]
                            );
                        }
                    }
                    if std::env::var_os("LUMEN_ADAM_AUDIT").is_some() {
                        trace.push(serde_json::json!({"step":step,"weights":params.iter().map(|p|values(p).into_iter().take(4).collect::<Vec<_>>()).collect::<Vec<_>>()}));
                    }
                }
                if !trace.is_empty() {
                    println!(
                        "conditional_momentum_audit {}",
                        serde_json::json!({"device":"Cuda","dtype":format!("{dtype:?}"),"state_dtype":format!("{state:?}"),"len":len,"initial":initial.into_iter().take(4).collect::<Vec<_>>(),"trace":trace})
                    );
                }
            }
        }
    }
}

#[test]
fn lowp_momentum_state_training_descends() {
    use crate::loss::MSELoss;
    for device in [Device::Cpu, Device::Cuda] {
        if device == Device::Cuda && !cuda::is_available() {
            continue;
        }
        let _enabled = cuda::set_enabled_scoped(device == Device::Cuda);
        let _strict = set_strict_device_execution_scoped(device == Device::Cuda);
        for state in [DType::F16, DType::BF16] {
            let p = Tensor::parameter_with_dtype(
                ArrayD::from_shape_vec(IxDyn(&[4]), vec![0.1, -0.1, 0.05, -0.05]).unwrap(),
                DType::F32,
            )
            .to_device(device);
            let target_values = vec![0.8f32, -0.7, 0.6, -0.5];
            let target = Tensor::from_array_no_grad(
                ArrayD::from_shape_vec(IxDyn(&[4]), target_values.clone()).unwrap(),
            )
            .to_device(device);
            let initial = values(&p);
            let mut opt = SGD::new_with_dtype(vec![p.clone()], 0.1, state).with_momentum(0.9);
            let mut losses = Vec::new();
            for step in 0..=80 {
                losses.push(
                    values(&p)
                        .iter()
                        .zip(&target_values)
                        .map(|(a, b)| (*a as f64 - *b as f64).powi(2) / 4.0)
                        .sum::<f64>(),
                );
                if step == 80 {
                    break;
                }
                opt.zero_grad();
                if step % 7 != 0 {
                    MSELoss::apply(&p, &target).backward();
                }
                opt.step();
                if device == Device::Cuda && step > 0 {
                    assert!(!p.has_host_f32_data() && !p.has_host_grad());
                    let v = opt.velocities[0].as_ref().unwrap();
                    assert!(!v.has_host_f32_data());
                    assert!(v.0.borrow().cuda_f32_data.is_none());
                }
            }
            assert!(losses.iter().all(|x| x.is_finite()) && losses[80] < losses[0] * 0.01);
            println!(
                "conditional_training_audit {}",
                serde_json::json!({"optimizer":"sgd","device":format!("{device:?}"),"dtype":"F32","state_dtype":format!("{state:?}"),"initial":initial,"target":target_values,"losses":losses})
            );
        }
    }
}

#[cfg(feature = "cuda")]
#[test]
fn lowp_cuda_optimizer_rejects_invalid_metadata_before_update() {
    if !cuda::is_available() {
        return;
    }
    let p = cuda::upload_f32(&[1.0, 2.0, 3.0]).unwrap();
    let g = cuda::upload_f32(&[0.1, 0.2, 0.3]).unwrap();
    let zeros = cuda::fill_scalar_f32_buffer(3, 0.0).unwrap();
    let state = cuda::f32_to_lowp_storage_no_host(&zeros, DType::F16).unwrap();
    let bad = cuda::alloc_storage(2).unwrap();
    let config = cuda::LowpOptimizerConfig {
        state_dtype: DType::F16,
        lr: 0.1,
        beta1: 0.9,
        beta2: 0.999,
        correction1: 0.1,
        correction2: 0.001,
        eps: 1e-8,
    };
    let run =
        |params: &[cuda::CudaBuffer],
         grads: &[cuda::CudaBuffer],
         m: &[cuda::CudaBuffer],
         v: Option<&[cuda::CudaBuffer]>,
         cfg| cuda::optimizer_lowp_state_update_no_host(params, grads, m, v, cfg);
    assert!(run(&[], &[], &[], None, config).is_err());
    assert!(
        run(
            std::slice::from_ref(&p),
            &[],
            std::slice::from_ref(&state),
            None,
            config
        )
        .is_err()
    );
    assert!(
        run(
            std::slice::from_ref(&p),
            std::slice::from_ref(&g),
            std::slice::from_ref(&bad),
            None,
            config
        )
        .is_err()
    );
    assert!(
        run(
            std::slice::from_ref(&p),
            std::slice::from_ref(&g),
            std::slice::from_ref(&state),
            Some(&[]),
            config
        )
        .is_err()
    );
    let unsupported = cuda::LowpOptimizerConfig {
        state_dtype: DType::F32,
        ..config
    };
    assert!(
        run(
            std::slice::from_ref(&p),
            std::slice::from_ref(&g),
            std::slice::from_ref(&state),
            None,
            unsupported
        )
        .is_err()
    );
    assert_eq!(cuda::download_f32(&p).unwrap(), vec![1.0, 2.0, 3.0]);
}
