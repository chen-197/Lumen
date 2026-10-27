use super::*;
use crate::autograd::no_grad;
use crate::precision::{PrecisionConfig, with_precision_config};
use ndarray::{Array, IxDyn};

fn make_tensor(shape: &[usize], data: Vec<f32>, dtype: DType) -> Tensor {
    let t = Tensor::from_array_no_grad(
        Array::from_shape_vec(IxDyn(shape), data)
            .expect("test tensor shape mismatch")
            .into_dyn(),
    );
    t.cast_inplace(dtype);
    t
}

#[test]
fn rms_norm_training_preserves_cpu_native_input_and_parameter_storage() {
    with_precision_config(
        PrecisionConfig {
            parameter_dtype: DType::F32,
            runtime_dtype: DType::F32,
            allow_parameter_dtype_copies: false,
        },
        || {
            for dtype in [DType::F32, DType::F16, DType::BF16, DType::I8] {
                let norm = RMSNorm::new_with_dtype(129, 1e-5, dtype);
                let scale = norm.weight.quantization_scale();
                let input = make_tensor(
                    &[3, 129],
                    (0..3 * 129)
                        .map(|i| ((i * 7 % 97) as f32 - 48.0) / 17.0)
                        .collect(),
                    DType::BF16,
                );
                let target = make_tensor(&[3, 129], vec![0.5; 3 * 129], DType::F32);
                for _ in 0..2 {
                    let output = norm.forward(input.clone());
                    assert_eq!(input.dtype(), DType::BF16);
                    assert_eq!(norm.weight.dtype(), dtype);
                    crate::loss::MSELoss::apply(&output, &target).backward();
                    assert_eq!(input.dtype(), DType::BF16);
                    assert_eq!(norm.weight.dtype(), dtype);
                    assert_eq!(norm.weight.quantization_scale(), scale);
                    assert!(
                        norm.weight
                            .grad_ref()
                            .as_ref()
                            .unwrap()
                            .iter()
                            .all(|v| v.is_finite())
                    );
                    norm.weight.zero_grad();
                }
            }
        },
    );
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_rms_norm_cooperative_training_loss_decreases_across_parameter_types() {
    use crate::autograd::{Device, set_strict_device_execution_scoped};
    use crate::loss::MSELoss;
    use crate::optim::{Adam, Optimizer, SGD};
    if !cuda::is_available() {
        return;
    }
    let dim = 257;
    let rows = 3;
    let values: Vec<_> = (0..rows * dim)
        .map(|i| ((i * 37 % 251) as f32 - 125.0) / 43.0)
        .collect();
    let rounded: Vec<_> = values.iter().map(|&v| bf16::from_f32(v).to_f32()).collect();
    let mut targets = Vec::with_capacity(rows * dim);
    for row in rounded.chunks_exact(dim) {
        let inv = (row.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>() / dim as f64
            + f64::from(1e-5f32))
        .sqrt()
        .recip();
        targets.extend(
            row.iter()
                .enumerate()
                .map(|(j, &v)| (f64::from(v) * inv * (0.2 + 0.5 * (j % 17) as f64 / 16.0)) as f32),
        );
    }
    for device in [Device::Cpu, Device::Cuda] {
        let _cuda_guard = cuda::set_enabled_scoped(device == Device::Cuda);
        let _strict_guard = set_strict_device_execution_scoped(device == Device::Cuda);
        for dtype in [DType::F32, DType::F16, DType::BF16, DType::I8] {
            for algorithm in ["sgd_momentum", "adam"] {
                let norm = RMSNorm::new_with_dtype(dim, 1e-5, dtype);
                norm.weight.to_device_inplace(device);
                let input =
                    make_tensor(&[rows, dim], values.clone(), DType::BF16).to_device(device);
                let target =
                    make_tensor(&[rows, dim], targets.clone(), DType::F32).to_device(device);
                let params = vec![norm.weight.clone()];
                let mut optimizer: Box<dyn Optimizer> = if algorithm == "adam" {
                    Box::new(
                        Adam::new_with_dtype(params, 0.03, DType::F32)
                            .with_dynamic_i8_master_weights(),
                    )
                } else {
                    Box::new(
                        SGD::new_with_dtype(params, 10.0, DType::F32)
                            .with_momentum(0.3)
                            .with_dynamic_i8_master_weights(),
                    )
                };
                let mut losses = Vec::new();
                for _ in 0..48 {
                    let output = norm.forward(input.clone());
                    let loss = MSELoss::apply(&output, &target);
                    losses.push(loss.data_ref().iter().copied().next().unwrap());
                    loss.backward();
                    if device == Device::Cuda {
                        assert!(norm.weight.cloned_cuda_f32_grad().is_some());
                        assert!(!norm.weight.has_host_grad());
                    }
                    optimizer.step();
                    optimizer.zero_grad();
                    assert_eq!(norm.weight.dtype(), dtype);
                }
                assert!(losses.iter().all(|v| v.is_finite()));
                let head = losses[..8].iter().sum::<f32>() / 8.0;
                let tail = losses[40..].iter().sum::<f32>() / 8.0;
                assert!(
                    tail < head * 0.1,
                    "{device:?}/{dtype:?}/{algorithm}: {head} -> {tail}"
                );
                assert!(losses[47] < losses[0] * 0.1);
                println!(
                    "rmsnorm_training {}",
                    serde_json::json!({
                        "device": format!("{device:?}"), "dtype": format!("{dtype:?}"),
                        "optimizer": algorithm, "losses": losses,
                    })
                );
            }
        }
    }
}

#[test]
fn rms_norm_no_grad_preserves_bf16_input_dtype() {
    let norm = RMSNorm::new(4, 1e-5);
    norm.weight.set_array_f32_with_dtype(
        Array::from_shape_vec(IxDyn(&[4]), vec![1.0, 0.5, 1.5, 2.0])
            .expect("weight shape mismatch")
            .into_dyn(),
        DType::BF16,
    );

    let input_f32 = make_tensor(&[1, 1, 4], vec![1.0, -2.0, 3.0, -4.0], DType::F32);
    let input_bf16 = make_tensor(&[1, 1, 4], vec![1.0, -2.0, 3.0, -4.0], DType::BF16);

    let ref_out = no_grad(|| norm.forward(input_f32));
    let bf16_out = no_grad(|| norm.forward(input_bf16));

    assert_eq!(bf16_out.dtype(), DType::BF16);
    let ref_vals = ref_out
        .data_ref()
        .iter()
        .map(|&v| bf16::from_f32(v).to_f32())
        .collect::<Vec<_>>();
    bf16_out.with_storage_view(|view| match view {
        TensorStorageView::BF16(view) => {
            let vals = view.iter().map(|v| v.to_f32()).collect::<Vec<_>>();
            assert_eq!(vals, ref_vals);
        }
        TensorStorageView::F16(_) => panic!("bf16 RMSNorm output should stay bf16 in no-grad"),
        TensorStorageView::F32(_) => panic!("bf16 RMSNorm output should stay bf16 in no-grad"),
    });
}

#[test]
fn rms_norm_explicit_dtype_overrides_global_default() {
    with_precision_config(
        PrecisionConfig {
            parameter_dtype: DType::BF16,
            runtime_dtype: DType::F32,
            allow_parameter_dtype_copies: false,
        },
        || {
            let norm = RMSNorm::new_with_dtype(4, 1e-5, DType::F32);
            assert_eq!(norm.weight.dtype(), DType::F32);
        },
    );
}

#[test]
fn rms_norm_no_grad_preserves_i8_input_dtype() {
    let norm = RMSNorm::new(4, 1e-5);
    norm.weight.set_array_f32_with_dtype(
        Array::from_shape_vec(IxDyn(&[4]), vec![1.0, 0.5, 1.5, 2.0])
            .expect("weight shape mismatch")
            .into_dyn(),
        DType::BF16,
    );

    let input_i8 = make_tensor(&[1, 1, 4], vec![1.0, -2.0, 3.0, -4.0], DType::I8);
    let out = no_grad(|| norm.forward(input_i8));
    assert_eq!(out.dtype(), DType::I8);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_rms_norm_matches_cpu_reference_in_strict_mode() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let weight_values = vec![1.0, 0.5, 1.5, 2.0];
    let norm = RMSNorm::new(4, 1e-5);
    norm.weight.set_array_f32_with_dtype(
        Array::from_shape_vec(IxDyn(&[4]), weight_values.clone())
            .expect("weight shape mismatch")
            .into_dyn(),
        DType::BF16,
    );
    norm.weight.to_cuda_inplace();
    let norm_ref = RMSNorm::new(4, 1e-5);
    norm_ref.weight.set_array_f32_with_dtype(
        Array::from_shape_vec(IxDyn(&[4]), weight_values)
            .expect("weight shape mismatch")
            .into_dyn(),
        DType::BF16,
    );

    let input = make_tensor(
        &[1, 2, 4],
        vec![1.0, -2.0, 3.0, -4.0, 0.5, 1.5, -2.5, 4.0],
        DType::BF16,
    )
    .to_cuda();
    assert!(input.0.borrow().cuda_f32_data.is_none());
    assert!(norm.weight.0.borrow().cuda_f32_data.is_none());

    crate::ops::cuda::set_enabled(true);
    crate::autograd::set_strict_device_execution(true);
    let cuda_out = no_grad(|| norm.forward(input.clone()));
    crate::autograd::set_strict_device_execution(false);
    crate::ops::cuda::set_enabled(false);
    assert!(input.0.borrow().cuda_f32_data.is_none());
    assert!(norm.weight.0.borrow().cuda_f32_data.is_none());

    let cpu_out = no_grad(|| norm_ref.forward(input.to_cpu()));
    assert!(cuda_out.is_cuda());
    assert_eq!(cuda_out.dtype(), DType::BF16);
    {
        let inner = cuda_out.0.borrow();
        assert!(
            !inner.has_f32_data,
            "CUDA RMSNorm no-grad output should not eagerly materialize host f32 data"
        );
        assert!(
            inner.cuda_f32_data.is_none(),
            "CUDA RMSNorm no-grad output should not keep a resident f32 compute buffer for bf16 output"
        );
        assert!(
            inner.cuda_bf16_data.is_some(),
            "CUDA RMSNorm no-grad output should keep resident bf16 storage"
        );
    }

    for (got, expect) in cuda_out.data_ref().iter().zip(cpu_out.data_ref().iter()) {
        assert!((got - expect).abs() < 2e-2, "got {got}, expect {expect}");
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_empty_rms_norm_stays_on_cuda_in_strict_mode() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let norm = RMSNorm::new(4, 1e-5);
    norm.weight.set_array_f32_with_dtype(
        Array::from_shape_vec(IxDyn(&[4]), vec![1.0, 0.5, 1.5, 2.0])
            .expect("weight shape mismatch")
            .into_dyn(),
        DType::BF16,
    );
    norm.weight.to_cuda_inplace();
    let input = make_tensor(&[0, 4], vec![], DType::BF16).to_cuda();

    crate::ops::cuda::set_enabled(true);
    crate::autograd::set_strict_device_execution(true);
    let out = no_grad(|| norm.forward(input.clone()));
    crate::autograd::set_strict_device_execution(false);
    crate::ops::cuda::set_enabled(false);

    assert!(out.is_cuda());
    assert_eq!(out.dtype(), DType::BF16);
    assert_eq!(out.shape_vec(), vec![0, 4]);
    assert_eq!(out.len(), 0);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_rms_norm_backward_matches_cpu_reference_in_strict_mode() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let weight_values = vec![1.0, 0.5, 1.5, 2.0];
    let input_values = vec![1.0, -2.0, 3.0, -4.0, 0.5, 1.5, -2.5, 4.0];
    let coeff_values = vec![0.5, -1.0, 2.0, 0.25, -0.75, 1.5, -0.5, 0.75];

    let norm_cuda = RMSNorm::new(4, 1e-5);
    norm_cuda.weight.set_array_f32_with_dtype(
        Array::from_shape_vec(IxDyn(&[4]), weight_values.clone())
            .expect("weight shape mismatch")
            .into_dyn(),
        DType::F32,
    );
    norm_cuda.weight.to_cuda_inplace();
    let input_cpu = Tensor::from_data_with_grad_flag(
        Array::from_shape_vec(IxDyn(&[1, 2, 4]), input_values.clone())
            .expect("input shape mismatch")
            .into_dyn(),
        true,
    );
    let coeff_cpu = Tensor::from_data_with_grad_flag(
        Array::from_shape_vec(IxDyn(&[1, 2, 4]), coeff_values)
            .expect("coeff shape mismatch")
            .into_dyn(),
        false,
    );
    let input_cuda = input_cpu.to_cuda();
    let coeff_cuda = coeff_cpu.to_cuda();

    crate::ops::cuda::set_enabled(true);
    crate::autograd::set_strict_device_execution(true);
    let cuda_out = norm_cuda.forward(input_cuda.clone());
    let cuda_loss = crate::ops::arithmetic::sum(&(&cuda_out * &coeff_cuda));
    cuda_loss.backward();
    assert!(input_cuda.cloned_cuda_f32_grad().is_some());
    assert!(norm_cuda.weight.cloned_cuda_f32_grad().is_some());
    assert!(!input_cuda.has_host_grad());
    assert!(!norm_cuda.weight.has_host_grad());
    crate::autograd::set_strict_device_execution(false);
    crate::ops::cuda::set_enabled(false);

    let norm_cpu = RMSNorm::new(4, 1e-5);
    norm_cpu.weight.set_array_f32_with_dtype(
        Array::from_shape_vec(IxDyn(&[4]), weight_values)
            .expect("weight shape mismatch")
            .into_dyn(),
        DType::F32,
    );
    let cpu_out = norm_cpu.forward(input_cpu.clone());
    let cpu_loss = crate::ops::arithmetic::sum(&(&cpu_out * &coeff_cpu));
    cpu_loss.backward();

    let input_cuda_grad = input_cuda.grad().expect("cuda RMSNorm input grad");
    let input_cpu_grad = input_cpu.grad().expect("cpu RMSNorm input grad");
    let weight_cuda_grad = norm_cuda.weight.grad().expect("cuda RMSNorm weight grad");
    let weight_cpu_grad = norm_cpu.weight.grad().expect("cpu RMSNorm weight grad");
    for (got, expect) in input_cuda_grad.iter().zip(input_cpu_grad.iter()) {
        assert!(
            (got - expect).abs() < 1e-5,
            "input got {got}, expect {expect}"
        );
    }
    for (got, expect) in weight_cuda_grad.iter().zip(weight_cpu_grad.iter()) {
        assert!(
            (got - expect).abs() < 1e-5,
            "weight got {got}, expect {expect}"
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_rms_norm_lowp_backward_uses_resident_inputs_and_f32_grads() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let weight_values = vec![1.0, 0.5, 1.5, 2.0];
    let input_values = vec![1.0, -2.0, 3.0, -4.0, 0.5, 1.5, -2.5, 4.0];
    let coeff_values = vec![0.5, -1.0, 2.0, 0.25, -0.75, 1.5, -0.5, 0.75];

    let norm_cuda = RMSNorm::new(4, 1e-5);
    norm_cuda.weight.set_array_f32_with_dtype(
        Array::from_shape_vec(IxDyn(&[4]), weight_values.clone())
            .expect("weight shape mismatch")
            .into_dyn(),
        DType::BF16,
    );
    norm_cuda.weight.to_cuda_inplace();

    let input_cuda = Tensor::from_data_with_grad_flag(
        Array::from_shape_vec(IxDyn(&[1, 2, 4]), input_values.clone())
            .expect("input shape mismatch")
            .into_dyn(),
        true,
    );
    input_cuda.cast_inplace(DType::BF16);
    let input_cuda = input_cuda.to_cuda();
    let coeff_cuda = Tensor::from_data_with_grad_flag(
        Array::from_shape_vec(IxDyn(&[1, 2, 4]), coeff_values.clone())
            .expect("coeff shape mismatch")
            .into_dyn(),
        false,
    )
    .to_cuda();

    assert!(input_cuda.0.borrow().cuda_f32_data.is_none());
    assert!(norm_cuda.weight.0.borrow().cuda_f32_data.is_none());

    crate::ops::cuda::set_enabled(true);
    crate::autograd::set_strict_device_execution(true);
    let cuda_out = norm_cuda.forward(input_cuda.clone());
    let cuda_loss = crate::ops::arithmetic::sum(&(&cuda_out * &coeff_cuda));
    cuda_loss.backward();
    crate::autograd::set_strict_device_execution(false);
    crate::ops::cuda::set_enabled(false);

    assert!(input_cuda.0.borrow().cuda_f32_data.is_none());
    assert!(norm_cuda.weight.0.borrow().cuda_f32_data.is_none());
    assert!(input_cuda.cloned_cuda_f32_grad().is_some());
    assert!(norm_cuda.weight.cloned_cuda_f32_grad().is_some());
    assert!(!input_cuda.has_host_grad());
    assert!(!norm_cuda.weight.has_host_grad());

    let norm_cpu = RMSNorm::new(4, 1e-5);
    norm_cpu.weight.set_array_f32_with_dtype(
        Array::from_shape_vec(IxDyn(&[4]), weight_values)
            .expect("weight shape mismatch")
            .into_dyn(),
        DType::BF16,
    );
    let input_cpu = Tensor::from_data_with_grad_flag(
        Array::from_shape_vec(IxDyn(&[1, 2, 4]), input_values)
            .expect("input shape mismatch")
            .into_dyn(),
        true,
    );
    input_cpu.cast_inplace(DType::BF16);
    let coeff_cpu = Tensor::from_data_with_grad_flag(
        Array::from_shape_vec(IxDyn(&[1, 2, 4]), coeff_values)
            .expect("coeff shape mismatch")
            .into_dyn(),
        false,
    );
    let cpu_out = norm_cpu.forward(input_cpu.clone());
    let cpu_loss = crate::ops::arithmetic::sum(&(&cpu_out * &coeff_cpu));
    cpu_loss.backward();

    let input_cuda_grad = input_cuda.grad().expect("cuda RMSNorm input grad");
    let input_cpu_grad = input_cpu.grad().expect("cpu RMSNorm input grad");
    let weight_cuda_grad = norm_cuda.weight.grad().expect("cuda RMSNorm weight grad");
    let weight_cpu_grad = norm_cpu.weight.grad().expect("cpu RMSNorm weight grad");
    for (got, expect) in input_cuda_grad.iter().zip(input_cpu_grad.iter()) {
        assert!(
            (got - expect).abs() < 3e-2,
            "input got {got}, expect {expect}"
        );
    }
    for (got, expect) in weight_cuda_grad.iter().zip(weight_cpu_grad.iter()) {
        assert!(
            (got - expect).abs() < 3e-2,
            "weight got {got}, expect {expect}"
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_rms_norm_preserves_i8_dtype_in_strict_mode() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let weight_values = vec![1.0, 0.5, 1.5, 2.0];
    let norm = RMSNorm::new(4, 1e-5);
    norm.weight.set_array_f32_with_dtype(
        Array::from_shape_vec(IxDyn(&[4]), weight_values.clone())
            .expect("weight shape mismatch")
            .into_dyn(),
        DType::BF16,
    );
    norm.weight.to_cuda_inplace();
    let norm_ref = RMSNorm::new(4, 1e-5);
    norm_ref.weight.set_array_f32_with_dtype(
        Array::from_shape_vec(IxDyn(&[4]), weight_values)
            .expect("weight shape mismatch")
            .into_dyn(),
        DType::BF16,
    );

    let input = make_tensor(
        &[1, 2, 4],
        vec![1.0, -2.0, 3.0, -4.0, 0.5, 1.5, -2.5, 4.0],
        DType::I8,
    )
    .to_cuda();
    assert!(input.0.borrow().cuda_f32_data.is_none());
    assert!(norm.weight.0.borrow().cuda_f32_data.is_none());

    crate::ops::cuda::set_enabled(true);
    crate::autograd::set_strict_device_execution(true);
    let cuda_out = no_grad(|| norm.forward(input.clone()));
    crate::autograd::set_strict_device_execution(false);
    crate::ops::cuda::set_enabled(false);
    assert!(input.0.borrow().cuda_f32_data.is_none());
    assert!(norm.weight.0.borrow().cuda_f32_data.is_none());

    let cpu_out = no_grad(|| norm_ref.forward(input.to_cpu()));
    assert!(cuda_out.is_cuda());
    assert_eq!(cuda_out.dtype(), DType::I8);
    {
        let inner = cuda_out.0.borrow();
        assert!(
            !inner.has_f32_data,
            "CUDA RMSNorm I8 no-grad output should not eagerly materialize host f32 data"
        );
        assert!(
            inner.cuda_f32_data.is_none(),
            "CUDA RMSNorm I8 no-grad output should not keep a resident f32 compute buffer"
        );
        assert!(
            inner.cuda_i8_data.is_some(),
            "CUDA RMSNorm I8 no-grad output should keep resident i8 storage"
        );
        assert!(
            inner.i8_scale.is_some(),
            "CUDA RMSNorm I8 no-grad output should record dynamic quantization scale"
        );
    }
    for (got, expect) in cuda_out.data_ref().iter().zip(cpu_out.data_ref().iter()) {
        assert!((got - expect).abs() < 1e-5, "got {got}, expect {expect}");
    }
}
