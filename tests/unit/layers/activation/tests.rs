use super::*;

#[cfg(feature = "cuda")]
#[test]
fn cuda_f32_unary_inference_keeps_inputs_resident() {
    assert!(
        cuda::is_available(),
        "CUDA activation check requires a working runtime"
    );
    let _cuda = cuda::set_enabled_scoped(true);
    let _strict = crate::device::set_strict_device_execution_scoped(true);
    for len in [7, 16385] {
        let data = (0..len)
            .map(|i| (i % 97) as f32 / 12. - 4.)
            .collect::<Vec<_>>();
        for op in [
            cuda::UnaryOp::Relu,
            cuda::UnaryOp::Sigmoid,
            cuda::UnaryOp::Tanh,
            cuda::UnaryOp::Silu,
            cuda::UnaryOp::Gelu,
        ] {
            let input = Tensor::from_cuda_f32_buffer_no_host(
                &[len],
                cuda::upload_f32(&data).unwrap(),
                Device::Cuda,
            );
            let before = cuda::test_host_downloads();
            let output = crate::autograd::no_grad(|| {
                unary_no_grad(&input, op, |_| panic!("unexpected CPU unary fallback"))
            });
            assert_eq!(
                cuda::test_host_downloads(),
                before,
                "{op:?}/{len} downloaded a resident input"
            );
            assert_eq!(output.device(), Device::Cuda);
            assert_eq!(output.dtype(), DType::F32);
            assert!(!output.requires_grad());
            let expected = data
                .iter()
                .map(|x| {
                    let x = *x as f64;
                    match op {
                        cuda::UnaryOp::Relu => x.max(0.),
                        cuda::UnaryOp::Sigmoid => 1. / (1. + (-x).exp()),
                        cuda::UnaryOp::Tanh => x.tanh(),
                        cuda::UnaryOp::Silu => x / (1. + (-x).exp()),
                        cuda::UnaryOp::Gelu => {
                            0.5 * x
                                * (1.
                                    + ((2. / std::f64::consts::PI).sqrt()
                                        * (x + 0.044715 * x.powi(3)))
                                    .tanh())
                        }
                    }
                })
                .collect::<Vec<_>>();
            crate::accuracy_gate::check(
                "resident F32 unary",
                &cuda::download_f32(&output.cloned_cuda_f32_buffer().unwrap()).unwrap(),
                &expected,
                2e-6,
                2e-5,
            );
        }
    }
}
use crate::autograd::no_grad;
use crate::precision::DType;
use half::bf16;
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

#[cfg(feature = "cuda")]
fn make_training_tensor(shape: &[usize], data: Vec<f32>) -> Tensor {
    Tensor::from_data_with_grad_flag(
        Array::from_shape_vec(IxDyn(shape), data)
            .expect("test tensor shape mismatch")
            .into_dyn(),
        true,
    )
}

#[cfg(feature = "cuda")]
fn make_training_tensor_with_dtype(shape: &[usize], data: Vec<f32>, dtype: DType) -> Tensor {
    let tensor = make_training_tensor(shape, data);
    tensor.cast_inplace(dtype);
    tensor
}

#[cfg(feature = "cuda")]
fn assert_cuda_activation_backward_matches_cpu<M: Module>(
    make_module: impl Fn() -> M,
    input_values: &[f32],
    tolerance: f32,
) {
    crate::ops::cuda::set_enabled(false);
    crate::device::set_strict_device_execution(false);
    let cpu_input = make_training_tensor(&[64, 256], input_values.to_vec());
    let cpu_out = make_module().forward(cpu_input.clone());
    cpu_out.backward();
    let cpu_grad = cpu_input
        .grad()
        .expect("CPU activation backward should populate input grad");

    crate::ops::cuda::set_enabled(true);
    crate::device::set_strict_device_execution(true);
    let cuda_input = make_training_tensor(&[64, 256], input_values.to_vec()).to_cuda();
    let cuda_out = make_module().forward(cuda_input.clone());
    assert!(cuda_out.is_cuda());
    cuda_out.backward();
    assert!(cuda_input.cloned_cuda_f32_grad().is_some());
    assert!(!cuda_input.has_host_grad());
    let cuda_grad = cuda_input
        .grad()
        .expect("CUDA activation backward should populate input grad");

    crate::device::set_strict_device_execution(false);
    crate::ops::cuda::set_enabled(false);

    for (idx, (got, expect)) in cuda_grad.iter().zip(cpu_grad.iter()).enumerate() {
        assert!(
            (got - expect).abs() <= tolerance,
            "activation grad mismatch at {idx}: got {got}, expect {expect}"
        );
    }
}

#[test]
fn relu_no_grad_preserves_bf16_dtype() {
    let input = make_tensor(&[3], vec![-1.0, 0.5, 2.0], DType::BF16);
    let out = no_grad(|| ReLU::new().forward(input.clone()));

    assert_eq!(input.dtype(), DType::BF16);
    assert_eq!(out.dtype(), DType::BF16);
}

#[test]
fn softmax_no_grad_preserves_bf16_dtype() {
    let input = make_tensor(&[1, 4], vec![1.0, 2.0, 3.0, 4.0], DType::BF16);
    let ref_out = no_grad(|| {
        Softmax::new(1).forward(make_tensor(&[1, 4], vec![1.0, 2.0, 3.0, 4.0], DType::F32))
    });
    let out = no_grad(|| Softmax::new(1).forward(input.clone()));

    assert_eq!(input.dtype(), DType::BF16);
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
        TensorStorageView::F16(_) => panic!("bf16 softmax output should stay bf16 in no-grad"),
        TensorStorageView::F32(_) => panic!("bf16 softmax output should stay bf16 in no-grad"),
    });
}

#[test]
fn relu_no_grad_preserves_i8_dtype() {
    let input = make_tensor(&[3], vec![-1.0, 0.5, 2.0], DType::I8);
    let out = no_grad(|| ReLU::new().forward(input.clone()));

    assert_eq!(input.dtype(), DType::I8);
    assert_eq!(out.dtype(), DType::I8);
    let vals = out.data_ref().iter().copied().collect::<Vec<_>>();
    assert!(vals[0].abs() <= 0.02);
    assert!((vals[1] - 0.5).abs() <= 0.02);
    assert!((vals[2] - 2.0).abs() <= 0.02);
}

#[test]
fn softmax_no_grad_promotes_i8_to_f32() {
    let input = make_tensor(&[1, 4], vec![1.0, 2.0, 3.0, 4.0], DType::I8);
    let out = no_grad(|| Softmax::new(1).forward(input.clone()));

    assert_eq!(input.dtype(), DType::I8);
    assert_eq!(out.dtype(), DType::F32);
    let vals = out.data_ref().iter().copied().collect::<Vec<_>>();
    let sum: f32 = vals.iter().sum();
    assert!((sum - 1.0).abs() <= 1e-5);
}

#[test]
#[should_panic(expected = "only supports the last dimension")]
fn softmax_rejects_non_last_axis() {
    let input = make_tensor(&[2, 3], (0..6).map(|v| v as f32).collect(), DType::F32);
    no_grad(|| {
        let _ = Softmax::new(0).forward(input);
    });
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_relu_and_softmax_match_cpu_reference() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    crate::ops::cuda::set_enabled(true);
    let input = make_tensor(
        &[64, 256],
        (0..(64 * 256))
            .map(|i| (i as f32 % 37.0) / 9.0 - 2.0)
            .collect(),
        DType::F32,
    )
    .to_cuda();

    let relu_out = no_grad(|| ReLU::new().forward(input.clone()));
    let softmax_out = no_grad(|| Softmax::new(1).forward(input.clone()));
    assert!(relu_out.is_cuda());
    assert!(softmax_out.is_cuda());

    crate::ops::cuda::set_enabled(false);
    let relu_ref = no_grad(|| ReLU::new().forward(input.to_cpu()));
    let softmax_ref = no_grad(|| Softmax::new(1).forward(input.to_cpu()));

    for (got, expect) in relu_out.data_ref().iter().zip(relu_ref.data_ref().iter()) {
        assert!(
            (got - expect).abs() < 1e-5,
            "relu got {got}, expect {expect}"
        );
    }
    for (got, expect) in softmax_out
        .data_ref()
        .iter()
        .zip(softmax_ref.data_ref().iter())
    {
        assert!(
            (got - expect).abs() < 1e-4,
            "softmax got {got}, expect {expect}"
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_lowp_relu_uses_resident_forward_buffer_in_strict_mode() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    crate::ops::cuda::set_enabled(true);
    crate::device::set_strict_device_execution(true);

    for dtype in [DType::F16, DType::BF16, DType::I8] {
        let input = make_tensor(&[2, 2], vec![-1.0, 0.0, 0.5, 2.0], dtype).to_cuda();
        assert!(input.cloned_cuda_native_lowp_buffer().is_some());
        assert!(!input.has_host_f32_data());

        let out = no_grad(|| ReLU::new().forward(input.clone()));

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
                    assert!(inner.i8_scale.is_some());
                }
                DType::F32 => unreachable!("test only covers low precision dtypes"),
            }
        }
        assert!(!out.has_host_f32_data());
        assert!(!input.has_host_f32_data());
        let vals = out.data_ref().iter().copied().collect::<Vec<_>>();
        for (got, expected) in vals.iter().zip([0.0f32, 0.0, 0.5, 2.0]) {
            assert!(
                (got - expected).abs() <= 0.03,
                "{dtype:?} relu got {got}, expected {expected}"
            );
        }
    }

    crate::device::set_strict_device_execution(false);
    crate::ops::cuda::set_enabled(false);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_lowp_softmax_uses_resident_forward_buffer_in_strict_mode() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    crate::ops::cuda::set_enabled(true);
    crate::device::set_strict_device_execution(true);

    for dtype in [DType::F16, DType::BF16, DType::I8] {
        let values = (0..(64 * 256))
            .map(|i| (i as f32 % 29.0) / 11.0 - 1.3)
            .collect::<Vec<_>>();
        let input = make_tensor(&[64, 256], values.clone(), dtype).to_cuda();
        assert!(input.cloned_cuda_native_lowp_buffer().is_some());
        assert!(input.0.borrow().cuda_f32_data.is_none());

        let out = no_grad(|| Softmax::new(1).forward(input.clone()));

        assert!(out.is_cuda());
        assert_eq!(
            out.dtype(),
            if dtype == DType::I8 {
                DType::F32
            } else {
                dtype
            }
        );
        assert!(input.0.borrow().cuda_f32_data.is_none());
        assert!(!input.has_host_f32_data());
        if dtype == DType::I8 {
            assert!(out.cloned_cuda_f32_buffer().is_some());
        } else {
            assert!(out.cloned_cuda_native_lowp_buffer().is_some());
            assert!(out.0.borrow().cuda_f32_data.is_none());
        }
        assert!(!out.has_host_f32_data());

        let cpu_input = make_tensor(&[64, 256], values, dtype);
        let cpu_out = no_grad(|| Softmax::new(1).forward(cpu_input));
        let tol = if dtype == DType::F16 { 2e-3 } else { 8e-3 };
        for (got, expect) in out.data_ref().iter().zip(cpu_out.data_ref().iter()) {
            assert!(
                (got - expect).abs() <= tol,
                "{dtype:?} softmax got {got}, expected {expect}"
            );
        }
    }

    crate::device::set_strict_device_execution(false);
    crate::ops::cuda::set_enabled(false);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_lowp_relu_backward_keeps_f32_grad_and_no_operand_f32_materialization() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    crate::ops::cuda::set_enabled(true);
    crate::device::set_strict_device_execution(true);

    for dtype in [DType::F16, DType::BF16, DType::I8] {
        let input =
            make_training_tensor_with_dtype(&[2, 2], vec![-1.0, 0.0, 0.5, 2.0], dtype).to_cuda();
        assert!(input.cloned_cuda_native_lowp_buffer().is_some());

        let out = ReLU::new().forward(input.clone());
        out.backward();

        assert!(input.cloned_cuda_f32_grad().is_some());
        assert!(!input.has_host_f32_data());
        let grad = input.grad().expect("CUDA lowp relu grad");
        for (got, expected) in grad.iter().zip([0.0f32, 0.0, 1.0, 1.0]) {
            assert!(
                (got - expected).abs() <= 1e-6,
                "{dtype:?} relu grad got {got}, expected {expected}"
            );
        }
    }

    crate::device::set_strict_device_execution(false);
    crate::ops::cuda::set_enabled(false);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_empty_activation_forward_backward_stays_stable_in_strict_mode() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    crate::ops::cuda::set_enabled(true);
    crate::device::set_strict_device_execution(true);

    let relu_input = make_tensor(&[0, 4], vec![], DType::BF16).to_cuda();
    let relu_out = no_grad(|| ReLU::new().forward(relu_input.clone()));
    assert!(relu_out.is_cuda());
    assert_eq!(relu_out.dtype(), DType::BF16);
    assert_eq!(relu_out.shape_vec(), vec![0, 4]);

    let softmax_input = make_tensor(&[0, 4], vec![], DType::F32).to_cuda();
    let softmax_out = no_grad(|| Softmax::new(1).forward(softmax_input.clone()));
    assert!(softmax_out.is_cuda());
    assert_eq!(softmax_out.shape_vec(), vec![0, 4]);

    let relu_train = make_training_tensor(&[0, 4], vec![]).to_cuda();
    let relu_train_out = ReLU::new().forward(relu_train.clone());
    assert!(relu_train_out.is_cuda());
    relu_train_out.backward();
    let relu_grad = relu_train
        .grad()
        .expect("empty CUDA relu grad should be recorded");
    assert_eq!(relu_grad.len(), 0);

    let softmax_train = make_training_tensor(&[0, 4], vec![]).to_cuda();
    let softmax_train_out = Softmax::new(1).forward(softmax_train.clone());
    assert!(softmax_train_out.is_cuda());
    softmax_train_out.backward();
    let softmax_grad = softmax_train
        .grad()
        .expect("empty CUDA softmax grad should be recorded");
    assert_eq!(softmax_grad.len(), 0);

    crate::device::set_strict_device_execution(false);
    crate::ops::cuda::set_enabled(false);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_activation_backward_matches_cpu_reference_in_strict_mode() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let values = (0..(64 * 256))
        .map(|i| {
            let phase = (i as f32 % 53.0) / 13.0 - 2.0;
            phase + ((i / 53) as f32 % 7.0) * 0.03125
        })
        .collect::<Vec<_>>();

    assert_cuda_activation_backward_matches_cpu(ReLU::new, &values, 1e-5);
    assert_cuda_activation_backward_matches_cpu(Sigmoid::new, &values, 1e-4);
    assert_cuda_activation_backward_matches_cpu(Tanh::new, &values, 1e-4);
    assert_cuda_activation_backward_matches_cpu(SiLU::new, &values, 2e-4);
    assert_cuda_activation_backward_matches_cpu(Gelu::new, &values, 2e-4);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_softmax_backward_matches_cpu_reference_in_strict_mode() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let shape = [32, 128];
    let values = (0..(shape[0] * shape[1]))
        .map(|i| (i as f32 % 41.0) / 11.0 - 1.7)
        .collect::<Vec<_>>();
    let upstream = (0..(shape[0] * shape[1]))
        .map(|i| ((i * 17) as f32 % 29.0) / 19.0 - 0.6)
        .collect::<Vec<_>>();
    let upstream_grad = Array::from_shape_vec(IxDyn(&shape), upstream)
        .expect("test grad shape mismatch")
        .into_dyn();

    crate::ops::cuda::set_enabled(false);
    crate::device::set_strict_device_execution(false);
    let cpu_input = make_training_tensor(&shape, values.clone());
    let cpu_out = Softmax::new(1).forward(cpu_input.clone());
    cpu_out.add_grad(upstream_grad.clone());
    cpu_out.backward();
    let cpu_grad = cpu_input
        .grad()
        .expect("CPU softmax backward should populate input grad");

    crate::ops::cuda::set_enabled(true);
    crate::device::set_strict_device_execution(true);
    let cuda_input = make_training_tensor(&shape, values).to_cuda();
    let cuda_out = Softmax::new(1).forward(cuda_input.clone());
    assert!(cuda_out.is_cuda());
    cuda_out.add_grad(upstream_grad);
    cuda_out.backward();
    assert!(cuda_input.cloned_cuda_f32_grad().is_some());
    assert!(!cuda_input.has_host_grad());
    let cuda_grad = cuda_input
        .grad()
        .expect("CUDA softmax backward should populate input grad");

    crate::device::set_strict_device_execution(false);
    crate::ops::cuda::set_enabled(false);

    for (idx, (got, expect)) in cuda_grad.iter().zip(cpu_grad.iter()).enumerate() {
        assert!(
            (got - expect).abs() <= 2e-4,
            "softmax grad mismatch at {idx}: got {got}, expect {expect}"
        );
    }
}
