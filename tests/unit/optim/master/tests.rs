use super::*;
use crate::optim::{Adam, Optimizer, SGD};
use ndarray::{ArrayD, IxDyn};

fn values(param: &Tensor) -> ArrayD<f32> {
    if param.is_cuda() {
        let buffer = param.cloned_cuda_f32_buffer().unwrap();
        ArrayD::from_shape_vec(
            IxDyn(&param.shape_vec()),
            cuda::download_f32(&buffer).unwrap(),
        )
        .unwrap()
    } else {
        param.with_storage_view_preferring(StoragePreference::Native, |v| match v {
            TensorStorageView::F32(v) => v.to_owned(),
            TensorStorageView::F16(v) => v.mapv(|x| x.to_f32()),
            TensorStorageView::BF16(v) => v.mapv(|x| x.to_f32()),
        })
    }
}

fn optimizer(param: &Tensor, kind: &str, lr: f32, master: bool) -> Box<dyn Optimizer> {
    if kind == "adam" {
        let opt = Adam::new_with_dtype(vec![param.clone()], lr, DType::F32);
        Box::new(if master {
            opt.with_fp32_master_weights()
        } else {
            opt
        })
    } else {
        let opt = SGD::new_with_dtype(vec![param.clone()], lr, DType::F32)
            .with_momentum(if kind == "momentum" { 0.5 } else { 0.0 });
        Box::new(if master {
            opt.with_fp32_master_weights()
        } else {
            opt
        })
    }
}

fn subquantum_steps(device: Device) {
    let _enabled = cuda::set_enabled_scoped(device == Device::Cuda);
    let _strict = crate::device::set_strict_device_execution_scoped(device == Device::Cuda);
    for dtype in [DType::F16, DType::BF16, DType::I8] {
        let quantum = match dtype {
            DType::F16 => 1.0 / 1024.0,
            DType::BF16 => 1.0 / 128.0,
            DType::I8 => 0.02,
            _ => unreachable!(),
        };
        for kind in ["sgd", "momentum", "adam"] {
            for master in [false, true] {
                let make_param = |dtype| {
                    let param = Tensor::parameter_with_dtype(
                        ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, -1.0]).unwrap(),
                        dtype,
                    );
                    if dtype == DType::I8 {
                        param.set_f32_slice_with_quantization(
                            &[2],
                            &[1.0, -1.0],
                            ParameterQuantization::Int8.with_scale(quantum),
                        );
                    }
                    param.to_device(device)
                };
                let param = make_param(dtype);
                let reference = make_param(DType::F32);
                let mut opt = optimizer(&param, kind, quantum / 16.0, master);
                let mut reference_opt = optimizer(&reference, kind, quantum / 16.0, false);
                for _ in 0..32 {
                    for p in [&param, &reference] {
                        p.zero_grad();
                        let grad = ArrayD::from_elem(IxDyn(&[2]), 1.0);
                        if device == Device::Cuda {
                            let buffer = cuda::upload_f32(&[1.0, 1.0]).unwrap();
                            p.add_grad_with_cuda_buffer(grad, Some(buffer));
                        } else {
                            p.add_grad(grad);
                        }
                    }
                    opt.step();
                    reference_opt.step();
                    assert_eq!(param.dtype(), dtype);
                    assert!(
                        !param.has_host_f32_data(),
                        "forward tensor must stay native"
                    );
                    if device == Device::Cuda {
                        assert!(!param.has_host_grad());
                        assert!(param.cloned_cuda_native_lowp_buffer().is_some());
                        assert!(!param.has_pending_grad(), "CUDA consumes gradients");
                    }
                }
                let actual = values(&param);
                let expected = reference.data();
                if master {
                    for (a, b) in actual.iter().zip(expected.iter()) {
                        assert!(
                            (a - b).abs() <= quantum * 0.51,
                            "{device:?} {dtype:?} {kind} master={master}: {a} vs F32 {b}"
                        );
                    }
                } else {
                    assert_eq!(
                        actual.iter().copied().collect::<Vec<_>>(),
                        vec![1.0, -1.0],
                        "sub-quantum native updates should reproduce the rounding plateau"
                    );
                }
                println!(
                    "subquantum {device:?} {dtype:?} {kind} master={master} actual={actual:?} f32={expected:?}"
                );
                opt.zero_grad();
                opt.step();
                assert_eq!(
                    values(&param),
                    actual,
                    "no pending grad must not move a master"
                );
            }
        }
    }
}

#[test]
fn cpu_master_weights_accumulate_subquantum_updates() {
    subquantum_steps(Device::Cpu);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_master_weights_accumulate_subquantum_updates() {
    if !cuda::is_available() {
        return;
    }
    subquantum_steps(Device::Cuda);
}

#[test]
fn master_weights_are_lazy_and_keep_original_parameter_handles() {
    let p = Tensor::parameter_with_dtype(ArrayD::from_elem(IxDyn(&[2]), 1.0), DType::I8);
    let mut opt =
        Adam::new_with_dtype(vec![p.clone()], 0.001, DType::F32).with_fp32_master_weights();
    opt.step();
    assert!(opt.master_weights.entries.as_ref().unwrap()[0].is_none());
    assert!(std::rc::Rc::ptr_eq(&p.0, &opt.params()[0].0));
    p.add_grad(ArrayD::from_elem(IxDyn(&[2]), 1.0));
    opt.step();
    assert!(opt.master_weights.entries.as_ref().unwrap()[0].is_some());
}

#[test]
fn dynamic_i8_masters_allow_weights_to_grow_beyond_initial_range() {
    for device in [Device::Cpu, Device::Cuda] {
        if device == Device::Cuda && !cuda::is_available() {
            continue;
        }
        let _enabled = cuda::set_enabled_scoped(device == Device::Cuda);
        for kind in ["sgd", "momentum", "adam"] {
            let p = Tensor::parameter_with_dtype(ArrayD::from_elem(IxDyn(&[2]), 1.0), DType::I8)
                .to_device(device);
            let r = Tensor::parameter_with_dtype(ArrayD::from_elem(IxDyn(&[2]), 1.0), DType::F32)
                .to_device(device);
            let initial_scale = p.quantization_scale().unwrap();
            let mut opt: Box<dyn Optimizer> = if kind == "adam" {
                Box::new(
                    Adam::new_with_dtype(vec![p.clone()], 0.2, DType::F32)
                        .with_dynamic_i8_master_weights(),
                )
            } else {
                Box::new(
                    SGD::new_with_dtype(vec![p.clone()], 0.2, DType::F32)
                        .with_momentum(if kind == "momentum" { 0.5 } else { 0.0 })
                        .with_dynamic_i8_master_weights(),
                )
            };
            let mut reference = optimizer(&r, kind, 0.2, false);
            for _ in 0..4 {
                for t in [&p, &r] {
                    t.zero_grad();
                    let g = ArrayD::from_elem(IxDyn(&[2]), -1.0);
                    if device == Device::Cuda {
                        t.add_grad_with_cuda_buffer(
                            g,
                            Some(cuda::upload_f32(&[-1.0, -1.0]).unwrap()),
                        );
                    } else {
                        t.add_grad(g);
                    }
                }
                opt.step();
                reference.step();
                assert_eq!(p.dtype(), DType::I8);
                assert!(!p.has_host_f32_data());
                let scale = p.quantization_scale().unwrap();
                assert!(scale > initial_scale);
                for (a, b) in values(&p).iter().zip(values(&r).iter()) {
                    assert!(
                        (a - b).abs() <= scale * 0.51,
                        "{device:?} {kind}: {a} vs {b}"
                    );
                }
            }
        }
    }
}

#[test]
fn i8_adam_master_training_escapes_initial_range_plateau() {
    use crate::layers::Linear;
    use crate::loss::MSELoss;
    use crate::module::Module;
    use crate::precision::{PrecisionConfig, with_precision_config};
    let pattern = |len: usize, scale: f32, phase: usize| {
        (0..len)
            .map(|i| {
                let x = ((i + phase) % 9) as f32 - 4.0;
                if x == 0.0 { scale * 0.5 } else { x * scale }
            })
            .collect::<Vec<_>>()
    };
    for device in [Device::Cpu, Device::Cuda] {
        if device == Device::Cuda && !cuda::is_available() {
            continue;
        }
        let _enabled = cuda::set_enabled_scoped(device == Device::Cuda);
        let _strict = crate::device::set_strict_device_execution_scoped(device == Device::Cuda);
        with_precision_config(
            PrecisionConfig {
                parameter_dtype: DType::I8,
                runtime_dtype: DType::BF16,
                allow_parameter_dtype_copies: false,
            },
            || {
                let mut final_losses = Vec::new();
                for policy in ["native", "fixed_master", "dynamic_master"] {
                    let tensor = |shape: &[usize], data: Vec<f32>| {
                        let t = Tensor::from_array_no_grad(
                            ArrayD::from_shape_vec(IxDyn(shape), data).unwrap(),
                        );
                        t.cast_inplace(DType::I8);
                        t.to_device(device)
                    };
                    let x = tensor(&[4, 192], pattern(4 * 192, 0.020, 7));
                    let y = tensor(&[4, 192], pattern(4 * 192, 0.070, 17));
                    let mut shards = Vec::new();
                    for i in 0..8 {
                        let layer = Linear::new_with_dtype(192, 192, DType::I8);
                        layer.weight.set_array_f32_with_dtype(
                            ArrayD::from_shape_vec(
                                IxDyn(&[192, 192]),
                                pattern(192 * 192, 0.00025, i + 13),
                            )
                            .unwrap(),
                            DType::I8,
                        );
                        layer.bias.as_ref().unwrap().set_array_f32_with_dtype(
                            ArrayD::from_shape_vec(IxDyn(&[192]), pattern(192, 0.00015, i + 19))
                                .unwrap(),
                            DType::I8,
                        );
                        if device == Device::Cuda {
                            layer.to_cuda();
                        }
                        shards.push(layer);
                    }
                    let params = shards
                        .iter()
                        .flat_map(Module::parameters)
                        .collect::<Vec<_>>();
                    let mut opt = Adam::new_with_dtype(params.clone(), 0.006, DType::F32);
                    opt = match policy {
                        "fixed_master" => opt.with_fp32_master_weights(),
                        "dynamic_master" => opt.with_dynamic_i8_master_weights(),
                        _ => opt,
                    };
                    let initial_scales = params
                        .iter()
                        .map(Tensor::quantization_scale)
                        .collect::<Vec<_>>();
                    let mut losses = Vec::new();
                    let mut audit = Vec::new();
                    for step in 0..24 {
                        let before = params.iter().map(values).collect::<Vec<_>>();
                        let mut pred = shards[0].forward(x.clone());
                        for shard in &shards[1..] {
                            pred = &pred + &shard.forward(x.clone());
                        }
                        let loss = MSELoss::apply(&pred, &y);
                        losses.push(loss.data().iter().next().copied().unwrap());
                        loss.backward();
                        if device == Device::Cuda {
                            assert!(
                                params
                                    .iter()
                                    .all(|p| p.cloned_cuda_f32_grad().is_some()
                                        && !p.has_host_grad())
                            );
                        }
                        opt.step();
                        if device == Device::Cuda {
                            assert!(opt.last_cuda_batched_update_count >= 8);
                            assert!(
                                params
                                    .iter()
                                    .all(|p| !p.has_host_f32_data() && !p.has_host_grad())
                            );
                        }
                        let mut changed = 0usize;
                        let mut saturated = 0usize;
                        let mut total = 0usize;
                        for (p, old) in params.iter().zip(&before) {
                            let after = values(p);
                            let bound = p.quantization_scale().unwrap() * 127.0;
                            for (a, b) in after.iter().zip(old.iter()) {
                                changed += usize::from(a != b);
                                saturated += usize::from(a.abs() >= bound * 0.999);
                                total += 1;
                            }
                        }
                        audit.push(serde_json::json!({"step": step,"changed":changed,"saturated":saturated,"total":total,"weight_scale":params[0].quantization_scale()}));
                        opt.zero_grad();
                    }
                    println!(
                        "training_audit {}",
                        serde_json::json!({"device":format!("{device:?}"),"policy":policy,"initial_scales":initial_scales,"losses":losses,"updates":audit})
                    );
                    assert!(losses.iter().all(|v| v.is_finite()));
                    final_losses.push(*losses.last().unwrap());
                }
                assert!(
                    final_losses[2] < final_losses[0] * 0.2,
                    "dynamic masters must escape the native plateau: {device:?} {final_losses:?}"
                );
            },
        );
    }
}
