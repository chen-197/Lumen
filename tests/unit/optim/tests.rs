use super::*;
use crate::precision::{PrecisionConfig, set_default_runtime_dtype, with_precision_config};

fn malformed_gradient_parameters(on_cuda: bool, dtype: DType, bad_shape: &[usize]) -> Vec<Tensor> {
    let params = (0..2)
        .map(|_| {
            let tensor = Tensor::parameter_with_dtype(ArrayD::ones(IxDyn(&[2, 2])), dtype);
            if on_cuda { tensor.to_cuda() } else { tensor }
        })
        .collect::<Vec<_>>();
    params[0].add_grad(ArrayD::ones(IxDyn(&[2, 2])));
    *params[1].grad_mut() = Some(ArrayD::ones(IxDyn(bad_shape)).into_shared());
    params
}

#[test]
fn regression_optimizer_sgd_rejects_bad_gradient_before_any_update() {
    use std::panic::{AssertUnwindSafe, catch_unwind};
    for on_cuda in [false, true] {
        if on_cuda && !cuda::is_available() {
            continue;
        }
        let _cuda = cuda::set_enabled_scoped(on_cuda);
        let _strict = crate::autograd::set_strict_device_execution_scoped(on_cuda);
        for dtype in [DType::F32, DType::F16, DType::BF16, DType::I8] {
            for state in [DType::F32, DType::F16, DType::BF16] {
                for momentum in [0., 0.9] {
                    for master in [false, true] {
                        if master && state != DType::F32 {
                            continue;
                        }
                        for shape in [&[2][..], &[4][..]] {
                            let params = malformed_gradient_parameters(on_cuda, dtype, shape);
                            let before = params
                                .iter()
                                .map(super::conditional_tests::values)
                                .collect::<Vec<_>>();
                            let mut opt = SGD::new_with_dtype(params.clone(), 0.1, state)
                                .with_momentum(momentum);
                            if master {
                                opt = opt.with_fp32_master_weights();
                            }
                            assert!(
                                catch_unwind(AssertUnwindSafe(|| opt.step())).is_err(),
                                "SGD accepted malformed gradient"
                            );
                            assert!(!opt.has_stepped, "SGD changed state before validation");
                            assert!(opt.velocities.iter().all(Option::is_none));
                            for (param, expected) in params.iter().zip(before) {
                                assert_eq!(super::conditional_tests::values(param), expected);
                                assert_eq!(param.dtype(), dtype);
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn regression_optimizer_adam_rejects_bad_gradient_before_any_update() {
    use std::panic::{AssertUnwindSafe, catch_unwind};
    for on_cuda in [false, true] {
        if on_cuda && !cuda::is_available() {
            continue;
        }
        let _cuda = cuda::set_enabled_scoped(on_cuda);
        let _strict = crate::autograd::set_strict_device_execution_scoped(on_cuda);
        for dtype in [DType::F32, DType::F16, DType::BF16, DType::I8] {
            for state in [DType::F32, DType::F16, DType::BF16] {
                for master in [false, true] {
                    if master && state != DType::F32 {
                        continue;
                    }
                    for shape in [&[2][..], &[4][..]] {
                        let params = malformed_gradient_parameters(on_cuda, dtype, shape);
                        let before = params
                            .iter()
                            .map(super::conditional_tests::values)
                            .collect::<Vec<_>>();
                        let mut opt = Adam::new_with_dtype(params.clone(), 0.1, state);
                        if master {
                            opt = opt.with_fp32_master_weights();
                        }
                        assert!(
                            catch_unwind(AssertUnwindSafe(|| opt.step())).is_err(),
                            "Adam accepted malformed gradient"
                        );
                        assert_eq!(opt.step_count, 0);
                        assert_eq!(opt.parameter_steps, vec![0, 0]);
                        assert!(
                            opt.exp_avg
                                .iter()
                                .chain(&opt.exp_avg_sq)
                                .all(Option::is_none)
                        );
                        for (param, expected) in params.iter().zip(before) {
                            assert_eq!(super::conditional_tests::values(param), expected);
                            assert_eq!(param.dtype(), dtype);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn sgd_default_construction_captures_runtime_dtype_for_future_state() {
    with_precision_config(
        PrecisionConfig {
            parameter_dtype: DType::F32,
            runtime_dtype: DType::BF16,
            allow_parameter_dtype_copies: false,
        },
        || {
            let param = Tensor::parameter(ArrayD::from_elem(IxDyn(&[2]), 1.0));
            let mut opt = SGD::new(vec![param.clone()], 0.1).with_momentum(0.9);
            set_default_runtime_dtype(DType::F32);
            param.add_grad(ArrayD::from_elem(IxDyn(&[2]), 0.5));
            opt.step();

            assert_eq!(opt.state_dtype(), DType::BF16);
            assert_eq!(
                opt.velocities[0].as_ref().expect("velocity state").dtype(),
                DType::BF16
            );
        },
    );
}

#[test]
fn sgd_explicit_state_dtype_overrides_global_default() {
    with_precision_config(
        PrecisionConfig {
            parameter_dtype: DType::F32,
            runtime_dtype: DType::BF16,
            allow_parameter_dtype_copies: false,
        },
        || {
            let param = Tensor::parameter(ArrayD::from_elem(IxDyn(&[2]), 1.0));
            let mut opt =
                SGD::new_with_dtype(vec![param.clone()], 0.1, DType::F32).with_momentum(0.9);
            param.add_grad(ArrayD::from_elem(IxDyn(&[2]), 0.5));
            opt.step();

            assert_eq!(opt.state_dtype(), DType::F32);
            assert_eq!(
                opt.velocities[0].as_ref().expect("velocity state").dtype(),
                DType::F32
            );
        },
    );
}

#[test]
fn cpu_sgd_step_updates_low_precision_parameters_without_f32_cache() {
    let initial = vec![1.0, -2.0, 0.5, 3.0];
    let grad_values = vec![0.5, -1.0, 2.0, 0.25];
    let expected = [0.95, -1.9, 0.3, 2.975];

    for dtype in [DType::F16, DType::BF16, DType::I8] {
        let param = Tensor::parameter_with_dtype(
            ArrayD::from_shape_vec(IxDyn(&[4]), initial.clone()).unwrap(),
            dtype,
        );
        assert_eq!(param.dtype(), dtype);
        assert!(
            !param.has_host_f32_data(),
            "{dtype:?} parameter should start without a host f32 cache"
        );

        param.add_grad(ArrayD::from_shape_vec(IxDyn(&[4]), grad_values.clone()).unwrap());
        let mut opt = SGD::new_with_dtype(vec![param.clone()], 0.1, DType::F32);
        opt.step();

        assert_eq!(param.dtype(), dtype);
        assert!(
            !param.has_host_f32_data(),
            "{dtype:?} SGD update should stay on native CPU storage"
        );

        let values = param.data_ref().iter().copied().collect::<Vec<_>>();
        for (got, expect) in values.iter().zip(expected.iter()) {
            assert!(
                (got - expect).abs() <= 0.03,
                "{dtype:?} CPU SGD update got {got}, expected approximately {expect}"
            );
        }
    }
}

#[test]
fn cpu_sgd_momentum_step_updates_low_precision_parameters_without_f32_cache() {
    let initial = vec![1.0, -2.0, 0.5, 3.0];
    let grad_values = vec![0.5, -1.0, 2.0, 0.25];
    let expected = [0.95, -1.9, 0.3, 2.975];

    for dtype in [DType::F16, DType::BF16, DType::I8] {
        let param = Tensor::parameter_with_dtype(
            ArrayD::from_shape_vec(IxDyn(&[4]), initial.clone()).unwrap(),
            dtype,
        );
        param.add_grad(ArrayD::from_shape_vec(IxDyn(&[4]), grad_values.clone()).unwrap());

        let mut opt = SGD::new_with_dtype(vec![param.clone()], 0.1, DType::F32).with_momentum(0.9);
        opt.step();

        assert_eq!(param.dtype(), dtype);
        assert!(
            !param.has_host_f32_data(),
            "{dtype:?} momentum SGD update should stay on native CPU storage"
        );
        assert_eq!(
            opt.velocities[0].as_ref().expect("velocity").dtype(),
            DType::F32
        );

        let values = param.data_ref().iter().copied().collect::<Vec<_>>();
        for (got, expect) in values.iter().zip(expected.iter()) {
            assert!(
                (got - expect).abs() <= 0.03,
                "{dtype:?} CPU momentum SGD update got {got}, expected approximately {expect}"
            );
        }
    }
}

#[test]
fn cpu_adam_step_updates_low_precision_parameters_without_f32_cache() {
    let initial = vec![1.0, -2.0, 0.5, 3.0];
    let grad_values = vec![0.5, -1.0, 2.0, 0.25];
    let expected = [0.9, -1.9, 0.4, 2.9];

    for dtype in [DType::F16, DType::BF16, DType::I8] {
        let param = Tensor::parameter_with_dtype(
            ArrayD::from_shape_vec(IxDyn(&[4]), initial.clone()).unwrap(),
            dtype,
        );
        param.add_grad(ArrayD::from_shape_vec(IxDyn(&[4]), grad_values.clone()).unwrap());

        let mut opt = Adam::new_with_dtype(vec![param.clone()], 0.1, DType::F32);
        opt.step();

        assert_eq!(param.dtype(), dtype);
        assert!(
            !param.has_host_f32_data(),
            "{dtype:?} Adam update should stay on native CPU storage"
        );
        assert_eq!(
            opt.exp_avg[0].as_ref().expect("exp_avg").dtype(),
            DType::F32
        );
        assert_eq!(
            opt.exp_avg_sq[0].as_ref().expect("exp_avg_sq").dtype(),
            DType::F32
        );

        let values = param.data_ref().iter().copied().collect::<Vec<_>>();
        for (got, expect) in values.iter().zip(expected.iter()) {
            assert!(
                (got - expect).abs() <= 0.04,
                "{dtype:?} CPU Adam update got {got}, expected approximately {expect}"
            );
        }
    }
}

#[test]
#[ignore = "performance sanity test; run with --ignored --nocapture"]
fn perf_low_precision_cpu_optimizer_updates_keep_f32_grad_state() {
    let len = 8192usize;
    let iters = 128usize;
    let shape = IxDyn(&[len]);
    let initial = (0..len)
        .map(|i| ((i * 17 + 5) % 257) as f32 / 257.0 - 0.5)
        .collect::<Vec<_>>();
    let grad_values = (0..len)
        .map(|i| ((i * 29 + 11) % 127) as f32 * 1e-4)
        .collect::<Vec<_>>();
    let grad = ArrayD::from_shape_vec(shape.clone(), grad_values).unwrap();

    let bench_sgd = |dtype: DType| {
        let param = Tensor::parameter_with_dtype(
            ArrayD::from_shape_vec(shape.clone(), initial.clone()).unwrap(),
            dtype,
        );
        let mut opt = SGD::new_with_dtype(vec![param.clone()], 1e-4, DType::F32);
        let start = std::time::Instant::now();
        for _ in 0..iters {
            param.zero_grad();
            param.add_grad(grad.clone());
            opt.step();
        }
        assert_eq!(param.dtype(), dtype);
        if dtype != DType::F32 {
            assert!(!param.has_host_f32_data());
        }
        start.elapsed()
    };

    let bench_sgd_momentum = |dtype: DType| {
        let param = Tensor::parameter_with_dtype(
            ArrayD::from_shape_vec(shape.clone(), initial.clone()).unwrap(),
            dtype,
        );
        let mut opt = SGD::new_with_dtype(vec![param.clone()], 1e-4, DType::F32).with_momentum(0.9);
        let start = std::time::Instant::now();
        for _ in 0..iters {
            param.zero_grad();
            param.add_grad(grad.clone());
            opt.step();
        }
        assert_eq!(param.dtype(), dtype);
        if dtype != DType::F32 {
            assert!(!param.has_host_f32_data());
        }
        start.elapsed()
    };

    let bench_adam = |dtype: DType| {
        let param = Tensor::parameter_with_dtype(
            ArrayD::from_shape_vec(shape.clone(), initial.clone()).unwrap(),
            dtype,
        );
        let mut opt = Adam::new_with_dtype(vec![param.clone()], 1e-4, DType::F32);
        let start = std::time::Instant::now();
        for _ in 0..iters {
            param.zero_grad();
            param.add_grad(grad.clone());
            opt.step();
        }
        assert_eq!(param.dtype(), dtype);
        if dtype != DType::F32 {
            assert!(!param.has_host_f32_data());
        }
        start.elapsed()
    };

    let sgd_f32 = bench_sgd(DType::F32);
    let sgd_bf16 = bench_sgd(DType::BF16);
    let sgd_f16 = bench_sgd(DType::F16);
    let sgd_i8 = bench_sgd(DType::I8);
    let sgd_momentum_f32 = bench_sgd_momentum(DType::F32);
    let sgd_momentum_bf16 = bench_sgd_momentum(DType::BF16);
    let sgd_momentum_f16 = bench_sgd_momentum(DType::F16);
    let sgd_momentum_i8 = bench_sgd_momentum(DType::I8);
    let adam_f32 = bench_adam(DType::F32);
    let adam_bf16 = bench_adam(DType::BF16);
    let adam_f16 = bench_adam(DType::F16);
    let adam_i8 = bench_adam(DType::I8);

    let us = |duration: std::time::Duration| duration.as_secs_f64() * 1e6 / iters as f64;
    eprintln!(
        "perf_optimizer len={len} iters={iters} sgd_us: f32={:.3} bf16={:.3} f16={:.3} i8={:.3}; sgd_momentum_us: f32={:.3} bf16={:.3} f16={:.3} i8={:.3}; adam_us: f32={:.3} bf16={:.3} f16={:.3} i8={:.3}",
        us(sgd_f32),
        us(sgd_bf16),
        us(sgd_f16),
        us(sgd_i8),
        us(sgd_momentum_f32),
        us(sgd_momentum_bf16),
        us(sgd_momentum_f16),
        us(sgd_momentum_i8),
        us(adam_f32),
        us(adam_bf16),
        us(adam_f16),
        us(adam_i8),
    );
}

#[test]
fn adam_default_construction_captures_runtime_dtype_for_future_state() {
    with_precision_config(
        PrecisionConfig {
            parameter_dtype: DType::F32,
            runtime_dtype: DType::BF16,
            allow_parameter_dtype_copies: false,
        },
        || {
            let param = Tensor::parameter(ArrayD::from_elem(IxDyn(&[2]), 1.0));
            let mut opt = Adam::new(vec![param.clone()], 0.1);
            set_default_runtime_dtype(DType::F32);
            param.add_grad(ArrayD::from_elem(IxDyn(&[2]), 0.25));
            opt.step();

            assert_eq!(opt.state_dtype(), DType::BF16);
            assert_eq!(
                opt.exp_avg[0].as_ref().expect("exp_avg").dtype(),
                DType::BF16
            );
            assert_eq!(
                opt.exp_avg_sq[0].as_ref().expect("exp_avg_sq").dtype(),
                DType::BF16
            );
        },
    );
}

#[cfg(feature = "cuda")]
#[test]
fn sgd_step_uses_cuda_grad_and_keeps_parameter_resident() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    crate::ops::cuda::set_enabled(true);
    let param = Tensor::parameter(
        ArrayD::from_shape_vec(IxDyn(&[4]), vec![1.0, -2.0, 0.5, 3.0]).expect("parameter shape"),
    )
    .to_cuda();
    let grad = ArrayD::from_shape_vec(IxDyn(&[4]), vec![0.5, -1.0, 2.0, 0.25]).expect("grad shape");
    let grad_buffer =
        crate::ops::cuda::upload_f32(grad.as_slice().expect("contiguous grad")).unwrap();
    param.add_grad_with_cuda_buffer(grad, Some(grad_buffer));

    let mut opt = SGD::new(vec![param.clone()], 0.1);
    opt.step();
    crate::ops::cuda::set_enabled(false);

    assert!(param.is_cuda());
    assert!(param.cloned_cuda_f32_buffer().is_some());
    let values = param.data_ref().iter().copied().collect::<Vec<_>>();
    let expected = [0.95, -1.9, 0.3, 2.975];
    for (got, expect) in values.iter().zip(expected.iter()) {
        assert!(
            (got - expect).abs() <= 1e-6,
            "SGD CUDA update got {got}, expected {expect}"
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
fn sgd_step_accepts_bf16_parameter_on_cuda_fast_path() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let initial = vec![1.0, -2.0, 0.5, 3.0];
    let grad_values = vec![0.5, -1.0, 2.0, 0.25];

    let cpu_param = Tensor::parameter_with_dtype(
        ArrayD::from_shape_vec(IxDyn(&[4]), initial.clone()).unwrap(),
        DType::BF16,
    );
    cpu_param.add_grad(ArrayD::from_shape_vec(IxDyn(&[4]), grad_values.clone()).unwrap());
    let mut cpu_opt = SGD::new(vec![cpu_param.clone()], 0.1);
    cpu_opt.step();
    let cpu_values = cpu_param.data_ref().iter().copied().collect::<Vec<_>>();

    crate::ops::cuda::set_enabled(true);
    let cuda_param = Tensor::parameter_with_dtype(
        ArrayD::from_shape_vec(IxDyn(&[4]), initial).unwrap(),
        DType::BF16,
    )
    .to_cuda();
    let grad = ArrayD::from_shape_vec(IxDyn(&[4]), grad_values).unwrap();
    let grad_buffer =
        crate::ops::cuda::upload_f32(grad.as_slice().expect("contiguous grad")).unwrap();
    cuda_param.add_grad_with_cuda_buffer(grad, Some(grad_buffer));

    let mut cuda_opt = SGD::new(vec![cuda_param.clone()], 0.1);
    cuda_opt.step();
    crate::ops::cuda::set_enabled(false);

    assert!(cuda_param.is_cuda());
    assert!(cuda_param.cloned_cuda_f32_buffer().is_some());
    assert_eq!(cuda_param.dtype(), DType::BF16);

    let cuda_values = cuda_param.data_ref().iter().copied().collect::<Vec<_>>();
    for (got, expect) in cuda_values.iter().zip(cpu_values.iter()) {
        assert!(
            (got - expect).abs() <= 1e-6,
            "BF16 SGD CUDA update got {got}, expected {expect}"
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
fn sgd_step_preserves_f16_and_i8_parameters_on_cuda_fast_path() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let initial = vec![1.0, -2.0, 0.5, 3.0];
    let grad_values = vec![0.5, -1.0, 2.0, 0.25];

    for dtype in [DType::F16, DType::I8] {
        let cpu_param = Tensor::parameter_with_dtype(
            ArrayD::from_shape_vec(IxDyn(&[4]), initial.clone()).unwrap(),
            dtype,
        );
        cpu_param.add_grad(ArrayD::from_shape_vec(IxDyn(&[4]), grad_values.clone()).unwrap());
        let mut cpu_opt = SGD::new_with_dtype(vec![cpu_param.clone()], 0.1, DType::F32);
        cpu_opt.step();
        let cpu_values = cpu_param.data_ref().iter().copied().collect::<Vec<_>>();

        crate::ops::cuda::set_enabled(true);
        let cuda_param = Tensor::parameter_with_dtype(
            ArrayD::from_shape_vec(IxDyn(&[4]), initial.clone()).unwrap(),
            dtype,
        )
        .to_cuda();
        let grad = ArrayD::from_shape_vec(IxDyn(&[4]), grad_values.clone()).unwrap();
        let grad_buffer =
            crate::ops::cuda::upload_f32(grad.as_slice().expect("contiguous grad")).unwrap();
        cuda_param.add_grad_with_cuda_buffer(grad, Some(grad_buffer));

        let mut cuda_opt = SGD::new_with_dtype(vec![cuda_param.clone()], 0.1, DType::F32);
        cuda_opt.step();
        crate::ops::cuda::set_enabled(false);

        assert!(cuda_param.is_cuda());
        assert!(cuda_param.cloned_cuda_f32_buffer().is_some());
        assert!(cuda_param.cloned_cuda_native_lowp_buffer().is_some());
        assert!(!cuda_param.has_host_f32_data());
        assert_eq!(cuda_param.dtype(), dtype);

        let cuda_values = cuda_param.data_ref().iter().copied().collect::<Vec<_>>();
        let tolerance = if dtype == DType::F16 { 1e-3 } else { 0.04 };
        for (got, expect) in cuda_values.iter().zip(cpu_values.iter()) {
            assert!(
                (got - expect).abs() <= tolerance,
                "{dtype:?} SGD CUDA update got {got}, expected {expect}"
            );
        }
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_sgd_step_clears_consumed_grad() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    crate::ops::cuda::set_enabled(true);
    let param = Tensor::parameter(
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 2.0]).expect("parameter shape"),
    )
    .to_cuda();
    let grad = ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.5, -1.0]).expect("grad shape");
    let grad_buffer =
        crate::ops::cuda::upload_f32(grad.as_slice().expect("contiguous grad")).unwrap();
    param.add_grad_with_cuda_buffer(grad, Some(grad_buffer));

    let mut opt = SGD::new(vec![param.clone()], 0.1);
    opt.step();
    let after_first = param.data_ref().iter().copied().collect::<Vec<_>>();
    opt.step();
    let after_second = param.data_ref().iter().copied().collect::<Vec<_>>();
    crate::ops::cuda::set_enabled(false);

    assert_eq!(after_first.len(), after_second.len());
    for (got, expect) in after_second.iter().zip(after_first.iter()) {
        assert!(
            (got - expect).abs() <= 1e-6,
            "CUDA SGD reused a consumed grad: got {got}, expected {expect}"
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
fn adam_step_uses_cuda_grad_and_keeps_state_resident() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let initial = vec![1.0, -2.0, 0.5, 3.0];
    let grad_values = vec![0.25, -0.5, 1.0, -0.125];

    let cpu_param =
        Tensor::parameter(ArrayD::from_shape_vec(IxDyn(&[4]), initial.clone()).unwrap());
    cpu_param.add_grad(ArrayD::from_shape_vec(IxDyn(&[4]), grad_values.clone()).unwrap());
    let mut cpu_opt = Adam::new_with_dtype(vec![cpu_param.clone()], 0.1, DType::F32);
    cpu_opt.step();
    let cpu_values = cpu_param.data_ref().iter().copied().collect::<Vec<_>>();

    crate::ops::cuda::set_enabled(true);
    let cuda_param =
        Tensor::parameter(ArrayD::from_shape_vec(IxDyn(&[4]), initial).unwrap()).to_cuda();
    let grad = ArrayD::from_shape_vec(IxDyn(&[4]), grad_values).unwrap();
    let grad_buffer =
        crate::ops::cuda::upload_f32(grad.as_slice().expect("contiguous grad")).unwrap();
    cuda_param.add_grad_with_cuda_buffer(grad, Some(grad_buffer));

    let mut cuda_opt = Adam::new_with_dtype(vec![cuda_param.clone()], 0.1, DType::F32);
    cuda_opt.step();
    crate::ops::cuda::set_enabled(false);

    assert!(cuda_param.is_cuda());
    assert!(cuda_param.cloned_cuda_f32_buffer().is_some());
    assert!(
        cuda_opt.exp_avg[0]
            .as_ref()
            .expect("exp_avg")
            .cloned_cuda_f32_buffer()
            .is_some()
    );
    assert!(
        cuda_opt.exp_avg_sq[0]
            .as_ref()
            .expect("exp_avg_sq")
            .cloned_cuda_f32_buffer()
            .is_some()
    );

    let cuda_values = cuda_param.data_ref().iter().copied().collect::<Vec<_>>();
    for (got, expect) in cuda_values.iter().zip(cpu_values.iter()) {
        assert!(
            (got - expect).abs() <= 1e-6,
            "Adam CUDA update got {got}, expected {expect}"
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
fn adam_step_accepts_bf16_parameter_with_f32_state_on_cuda_fast_path() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let initial = vec![1.0, -2.0, 0.5, 3.0];
    let grad_values = vec![0.25, -0.5, 1.0, -0.125];

    let cpu_param = Tensor::parameter_with_dtype(
        ArrayD::from_shape_vec(IxDyn(&[4]), initial.clone()).unwrap(),
        DType::BF16,
    );
    cpu_param.add_grad(ArrayD::from_shape_vec(IxDyn(&[4]), grad_values.clone()).unwrap());
    let mut cpu_opt = Adam::new_with_dtype(vec![cpu_param.clone()], 0.1, DType::F32);
    cpu_opt.step();
    let cpu_values = cpu_param.data_ref().iter().copied().collect::<Vec<_>>();
    let cpu_exp_avg = cpu_opt.exp_avg[0]
        .as_ref()
        .expect("cpu exp_avg")
        .data_ref()
        .iter()
        .copied()
        .collect::<Vec<_>>();
    let cpu_exp_avg_sq = cpu_opt.exp_avg_sq[0]
        .as_ref()
        .expect("cpu exp_avg_sq")
        .data_ref()
        .iter()
        .copied()
        .collect::<Vec<_>>();

    crate::ops::cuda::set_enabled(true);
    let cuda_param = Tensor::parameter_with_dtype(
        ArrayD::from_shape_vec(IxDyn(&[4]), initial).unwrap(),
        DType::BF16,
    )
    .to_cuda();
    let grad = ArrayD::from_shape_vec(IxDyn(&[4]), grad_values).unwrap();
    let grad_buffer =
        crate::ops::cuda::upload_f32(grad.as_slice().expect("contiguous grad")).unwrap();
    cuda_param.add_grad_with_cuda_buffer(grad, Some(grad_buffer));

    let mut cuda_opt = Adam::new_with_dtype(vec![cuda_param.clone()], 0.1, DType::F32);
    cuda_opt.step();
    crate::ops::cuda::set_enabled(false);

    assert!(cuda_param.is_cuda());
    assert!(cuda_param.cloned_cuda_f32_buffer().is_some());
    assert!(cuda_param.cloned_cuda_native_lowp_buffer().is_some());
    assert!(!cuda_param.has_host_f32_data());
    assert_eq!(cuda_param.dtype(), DType::BF16);
    let cuda_exp_avg_tensor = cuda_opt.exp_avg[0].as_ref().expect("cuda exp_avg");
    let cuda_exp_avg_sq_tensor = cuda_opt.exp_avg_sq[0].as_ref().expect("cuda exp_avg_sq");
    assert!(cuda_exp_avg_tensor.is_cuda());
    assert!(cuda_exp_avg_sq_tensor.is_cuda());
    assert!(cuda_exp_avg_tensor.cloned_cuda_f32_buffer().is_some());
    assert!(cuda_exp_avg_sq_tensor.cloned_cuda_f32_buffer().is_some());

    let cuda_values = cuda_param.data_ref().iter().copied().collect::<Vec<_>>();
    let cuda_exp_avg = cuda_exp_avg_tensor
        .data_ref()
        .iter()
        .copied()
        .collect::<Vec<_>>();
    let cuda_exp_avg_sq = cuda_exp_avg_sq_tensor
        .data_ref()
        .iter()
        .copied()
        .collect::<Vec<_>>();
    for (got, expect) in cuda_values.iter().zip(cpu_values.iter()) {
        assert!(
            (got - expect).abs() <= 1e-6,
            "BF16 Adam CUDA update got {got}, expected {expect}"
        );
    }
    for (got, expect) in cuda_exp_avg.iter().zip(cpu_exp_avg.iter()) {
        assert!(
            (got - expect).abs() <= 1e-6,
            "BF16 Adam CUDA exp_avg got {got}, expected {expect}"
        );
    }
    for (got, expect) in cuda_exp_avg_sq.iter().zip(cpu_exp_avg_sq.iter()) {
        assert!(
            (got - expect).abs() <= 1e-6,
            "BF16 Adam CUDA exp_avg_sq got {got}, expected {expect}"
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
fn adam_step_preserves_lowp_cuda_native_buffer_after_update() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    crate::ops::cuda::set_enabled(true);
    for dtype in [DType::F16, DType::BF16, DType::I8] {
        let param = Tensor::parameter_with_dtype(
            ArrayD::from_shape_vec(IxDyn(&[4]), vec![0.2, -0.1, 0.3, -0.2]).unwrap(),
            dtype,
        )
        .to_cuda();
        let grad = ArrayD::from_shape_vec(IxDyn(&[4]), vec![0.05, -0.03, 0.02, -0.04]).unwrap();
        let grad_buffer =
            crate::ops::cuda::upload_f32(grad.as_slice().expect("contiguous grad")).unwrap();
        param.add_grad_with_cuda_buffer(grad, Some(grad_buffer));

        let mut opt = Adam::new_with_dtype(vec![param.clone()], 0.01, DType::F32);
        opt.step();

        assert_eq!(param.dtype(), dtype);
        assert!(
            param.cloned_cuda_f32_buffer().is_some(),
            "{dtype:?} Adam update should keep a CUDA f32 optimizer buffer"
        );
        assert!(
            param.cloned_cuda_native_lowp_buffer().is_some(),
            "{dtype:?} Adam update should keep a native lowp CUDA buffer"
        );
        assert!(
            !param.has_host_f32_data(),
            "{dtype:?} Adam update should not materialize host f32 parameter data"
        );
    }
    crate::ops::cuda::set_enabled(false);
}

#[cfg(feature = "cuda")]
#[test]
fn sgd_momentum_step_uses_cuda_grad_and_keeps_velocity_resident() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let initial = vec![1.0, -2.0, 0.5, 3.0];
    let grad_values = vec![0.5, -1.0, 2.0, 0.25];

    let cpu_param =
        Tensor::parameter(ArrayD::from_shape_vec(IxDyn(&[4]), initial.clone()).unwrap());
    cpu_param.add_grad(ArrayD::from_shape_vec(IxDyn(&[4]), grad_values.clone()).unwrap());
    let mut cpu_opt =
        SGD::new_with_dtype(vec![cpu_param.clone()], 0.1, DType::F32).with_momentum(0.9);
    cpu_opt.step();
    let cpu_values = cpu_param.data_ref().iter().copied().collect::<Vec<_>>();
    let cpu_velocity = cpu_opt.velocities[0]
        .as_ref()
        .expect("cpu velocity")
        .data_ref()
        .iter()
        .copied()
        .collect::<Vec<_>>();

    crate::ops::cuda::set_enabled(true);
    let cuda_param =
        Tensor::parameter(ArrayD::from_shape_vec(IxDyn(&[4]), initial).unwrap()).to_cuda();
    let grad = ArrayD::from_shape_vec(IxDyn(&[4]), grad_values).unwrap();
    let grad_buffer =
        crate::ops::cuda::upload_f32(grad.as_slice().expect("contiguous grad")).unwrap();
    cuda_param.add_grad_with_cuda_buffer(grad, Some(grad_buffer));

    let mut cuda_opt =
        SGD::new_with_dtype(vec![cuda_param.clone()], 0.1, DType::F32).with_momentum(0.9);
    cuda_opt.step();
    crate::ops::cuda::set_enabled(false);

    assert!(cuda_param.is_cuda());
    assert!(cuda_param.cloned_cuda_f32_buffer().is_some());
    let cuda_velocity_tensor = cuda_opt.velocities[0].as_ref().expect("cuda velocity");
    assert!(cuda_velocity_tensor.is_cuda());
    assert!(cuda_velocity_tensor.cloned_cuda_f32_buffer().is_some());

    let cuda_values = cuda_param.data_ref().iter().copied().collect::<Vec<_>>();
    let cuda_velocity = cuda_velocity_tensor
        .data_ref()
        .iter()
        .copied()
        .collect::<Vec<_>>();
    for (got, expect) in cuda_values.iter().zip(cpu_values.iter()) {
        assert!(
            (got - expect).abs() <= 1e-6,
            "SGD momentum CUDA update got {got}, expected {expect}"
        );
    }
    for (got, expect) in cuda_velocity.iter().zip(cpu_velocity.iter()) {
        assert!(
            (got - expect).abs() <= 1e-6,
            "SGD momentum CUDA velocity got {got}, expected {expect}"
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
fn sgd_momentum_step_accepts_bf16_parameter_on_cuda_fast_path() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let initial = vec![1.0, -2.0, 0.5, 3.0];
    let grad_values = vec![0.5, -1.0, 2.0, 0.25];

    let cpu_param = Tensor::parameter_with_dtype(
        ArrayD::from_shape_vec(IxDyn(&[4]), initial.clone()).unwrap(),
        DType::BF16,
    );
    cpu_param.add_grad(ArrayD::from_shape_vec(IxDyn(&[4]), grad_values.clone()).unwrap());
    let mut cpu_opt =
        SGD::new_with_dtype(vec![cpu_param.clone()], 0.1, DType::F32).with_momentum(0.9);
    cpu_opt.step();
    let cpu_values = cpu_param.data_ref().iter().copied().collect::<Vec<_>>();
    let cpu_velocity = cpu_opt.velocities[0]
        .as_ref()
        .expect("cpu velocity")
        .data_ref()
        .iter()
        .copied()
        .collect::<Vec<_>>();

    crate::ops::cuda::set_enabled(true);
    let cuda_param = Tensor::parameter_with_dtype(
        ArrayD::from_shape_vec(IxDyn(&[4]), initial).unwrap(),
        DType::BF16,
    )
    .to_cuda();
    let grad = ArrayD::from_shape_vec(IxDyn(&[4]), grad_values).unwrap();
    let grad_buffer =
        crate::ops::cuda::upload_f32(grad.as_slice().expect("contiguous grad")).unwrap();
    cuda_param.add_grad_with_cuda_buffer(grad, Some(grad_buffer));

    let mut cuda_opt =
        SGD::new_with_dtype(vec![cuda_param.clone()], 0.1, DType::F32).with_momentum(0.9);
    cuda_opt.step();
    crate::ops::cuda::set_enabled(false);

    assert!(cuda_param.is_cuda());
    assert!(cuda_param.cloned_cuda_f32_buffer().is_some());
    assert_eq!(cuda_param.dtype(), DType::BF16);
    let cuda_velocity_tensor = cuda_opt.velocities[0].as_ref().expect("cuda velocity");
    assert!(cuda_velocity_tensor.is_cuda());
    assert!(cuda_velocity_tensor.cloned_cuda_f32_buffer().is_some());

    let cuda_values = cuda_param.data_ref().iter().copied().collect::<Vec<_>>();
    let cuda_velocity = cuda_velocity_tensor
        .data_ref()
        .iter()
        .copied()
        .collect::<Vec<_>>();
    for (got, expect) in cuda_values.iter().zip(cpu_values.iter()) {
        assert!(
            (got - expect).abs() <= 1e-6,
            "BF16 SGD momentum CUDA update got {got}, expected {expect}"
        );
    }
    for (got, expect) in cuda_velocity.iter().zip(cpu_velocity.iter()) {
        assert!(
            (got - expect).abs() <= 1e-6,
            "BF16 SGD momentum CUDA velocity got {got}, expected {expect}"
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
fn sgd_step_batches_multiple_cuda_parameters() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let p0_initial = vec![1.0, -2.0, 0.5, 3.0];
    let p1_initial = vec![0.25, -0.75, 1.5];
    let g0_values = vec![0.5, -1.0, 2.0, 0.25];
    let g1_values = vec![-0.25, 0.5, 1.25];

    let p0_cpu =
        Tensor::parameter(ArrayD::from_shape_vec(IxDyn(&[4]), p0_initial.clone()).unwrap());
    let p1_cpu =
        Tensor::parameter(ArrayD::from_shape_vec(IxDyn(&[3]), p1_initial.clone()).unwrap());
    p0_cpu.add_grad(ArrayD::from_shape_vec(IxDyn(&[4]), g0_values.clone()).unwrap());
    p1_cpu.add_grad(ArrayD::from_shape_vec(IxDyn(&[3]), g1_values.clone()).unwrap());
    let mut cpu_opt = SGD::new(vec![p0_cpu.clone(), p1_cpu.clone()], 0.1);
    cpu_opt.step();

    crate::ops::cuda::set_enabled(true);
    let p0_cuda =
        Tensor::parameter(ArrayD::from_shape_vec(IxDyn(&[4]), p0_initial).unwrap()).to_cuda();
    let p1_cuda =
        Tensor::parameter(ArrayD::from_shape_vec(IxDyn(&[3]), p1_initial).unwrap()).to_cuda();
    let g0 = ArrayD::from_shape_vec(IxDyn(&[4]), g0_values).unwrap();
    let g1 = ArrayD::from_shape_vec(IxDyn(&[3]), g1_values).unwrap();
    let g0_buf = crate::ops::cuda::upload_f32(g0.as_slice().unwrap()).unwrap();
    let g1_buf = crate::ops::cuda::upload_f32(g1.as_slice().unwrap()).unwrap();
    p0_cuda.add_grad_with_cuda_buffer(g0, Some(g0_buf));
    p1_cuda.add_grad_with_cuda_buffer(g1, Some(g1_buf));
    let mut cuda_opt = SGD::new(vec![p0_cuda.clone(), p1_cuda.clone()], 0.1);
    cuda_opt.step();
    crate::ops::cuda::set_enabled(false);

    for (got, expect) in p0_cuda.data_ref().iter().zip(p0_cpu.data_ref().iter()) {
        assert!((got - expect).abs() <= 1e-6);
    }
    for (got, expect) in p1_cuda.data_ref().iter().zip(p1_cpu.data_ref().iter()) {
        assert!((got - expect).abs() <= 1e-6);
    }
}

#[cfg(feature = "cuda")]
#[test]
fn adam_step_batches_multiple_cuda_parameters_with_f32_state() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let p0_initial = vec![1.0, -2.0, 0.5, 3.0];
    let p1_initial = vec![0.25, -0.75, 1.5];
    let g0_values = vec![0.25, -0.5, 1.0, -0.125];
    let g1_values = vec![-0.25, 0.5, 1.25];

    let p0_cpu =
        Tensor::parameter(ArrayD::from_shape_vec(IxDyn(&[4]), p0_initial.clone()).unwrap());
    let p1_cpu =
        Tensor::parameter(ArrayD::from_shape_vec(IxDyn(&[3]), p1_initial.clone()).unwrap());
    p0_cpu.add_grad(ArrayD::from_shape_vec(IxDyn(&[4]), g0_values.clone()).unwrap());
    p1_cpu.add_grad(ArrayD::from_shape_vec(IxDyn(&[3]), g1_values.clone()).unwrap());
    let mut cpu_opt = Adam::new_with_dtype(vec![p0_cpu.clone(), p1_cpu.clone()], 0.1, DType::F32);
    cpu_opt.step();

    crate::ops::cuda::set_enabled(true);
    let p0_cuda =
        Tensor::parameter(ArrayD::from_shape_vec(IxDyn(&[4]), p0_initial).unwrap()).to_cuda();
    let p1_cuda =
        Tensor::parameter(ArrayD::from_shape_vec(IxDyn(&[3]), p1_initial).unwrap()).to_cuda();
    let g0 = ArrayD::from_shape_vec(IxDyn(&[4]), g0_values).unwrap();
    let g1 = ArrayD::from_shape_vec(IxDyn(&[3]), g1_values).unwrap();
    let g0_buf = crate::ops::cuda::upload_f32(g0.as_slice().unwrap()).unwrap();
    let g1_buf = crate::ops::cuda::upload_f32(g1.as_slice().unwrap()).unwrap();
    p0_cuda.add_grad_with_cuda_buffer(g0, Some(g0_buf));
    p1_cuda.add_grad_with_cuda_buffer(g1, Some(g1_buf));
    let mut cuda_opt =
        Adam::new_with_dtype(vec![p0_cuda.clone(), p1_cuda.clone()], 0.1, DType::F32);
    cuda_opt.step();
    crate::ops::cuda::set_enabled(false);

    for (got, expect) in p0_cuda.data_ref().iter().zip(p0_cpu.data_ref().iter()) {
        assert!((got - expect).abs() <= 1e-6);
    }
    for (got, expect) in p1_cuda.data_ref().iter().zip(p1_cpu.data_ref().iter()) {
        assert!((got - expect).abs() <= 1e-6);
    }
    assert!(
        cuda_opt.exp_avg[0]
            .as_ref()
            .expect("first exp_avg")
            .cloned_cuda_f32_buffer()
            .is_some()
    );
    assert!(
        cuda_opt.exp_avg[1]
            .as_ref()
            .expect("second exp_avg")
            .cloned_cuda_f32_buffer()
            .is_some()
    );
}

#[test]
#[should_panic(expected = "Optimizer state currently only supports floating dtypes")]
fn optimizer_state_rejects_integer_dtype() {
    let _ = SGD::new_with_dtype(Vec::new(), 0.1, DType::I8);
}
