use super::*;
use crate::autograd::no_grad;
#[cfg(feature = "cuda")]
use crate::device::set_strict_device_execution;
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
fn mse_loss_no_grad_returns_f32_scalar_without_materializing_bf16_input() {
    let output = make_tensor(&[2], vec![1.0, 3.0], DType::BF16);
    let target = make_tensor(&[2], vec![2.0, 1.0], DType::BF16);

    let loss = no_grad(|| MSELoss::apply(&output, &target));

    assert_eq!(loss.dtype(), DType::F32);
    assert!(!loss.requires_grad());
    assert_eq!(output.dtype(), DType::BF16);
    assert_eq!(target.dtype(), DType::BF16);
    assert!((loss.data_ref().first().copied().unwrap_or_default() - 2.5).abs() <= 1e-6);
}

#[test]
fn cross_entropy_no_grad_accepts_i8_logits_and_returns_f32_scalar() {
    let logits = make_tensor(&[2, 3], vec![2.0, 0.0, -1.0, -1.0, 0.0, 2.0], DType::I8);
    let target = make_tensor(&[2, 3], vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0], DType::F32);

    let loss = no_grad(|| CrossEntropyLoss::apply(&logits, &target));

    assert_eq!(loss.dtype(), DType::F32);
    assert!(!loss.requires_grad());
    assert_eq!(logits.dtype(), DType::I8);
    assert!(
        loss.data_ref()
            .first()
            .copied()
            .unwrap_or_default()
            .is_finite()
    );
}

#[test]
#[should_panic(expected = "mse_loss expects at least one element")]
fn mse_loss_rejects_empty_input() {
    let output = make_tensor(&[0, 4], vec![], DType::F32);
    let target = make_tensor(&[0, 4], vec![], DType::F32);
    let _ = no_grad(|| MSELoss::apply(&output, &target));
}

#[test]
#[should_panic(expected = "batch size must be greater than zero")]
fn cross_entropy_rejects_empty_batch() {
    let logits = make_tensor(&[0, 4], vec![], DType::F32);
    let target = make_tensor(&[0, 4], vec![], DType::F32);
    let _ = no_grad(|| CrossEntropyLoss::apply(&logits, &target));
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_mse_loss_backward_matches_cpu_reference_in_strict_mode() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let output_cpu = Tensor::from_data_with_grad_flag(
        Array::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, -2.0, 0.5, 3.0, -1.0, 2.0])
            .unwrap()
            .into_dyn(),
        true,
    );
    let target_cpu = Tensor::from_data_with_grad_flag(
        Array::from_shape_vec(IxDyn(&[2, 3]), vec![0.5, -1.5, 0.25, 2.5, -0.5, 1.0])
            .unwrap()
            .into_dyn(),
        true,
    );
    let output_cuda = output_cpu.to_cuda();
    let target_cuda = target_cpu.to_cuda();

    crate::ops::cuda::set_enabled(true);
    set_strict_device_execution(true);
    let loss_cuda = MSELoss::apply(&output_cuda, &target_cuda);
    loss_cuda.backward();
    assert!(output_cuda.cloned_cuda_f32_grad().is_some());
    assert!(target_cuda.cloned_cuda_f32_grad().is_some());
    assert!(!output_cuda.has_host_grad());
    assert!(!target_cuda.has_host_grad());
    set_strict_device_execution(false);
    crate::ops::cuda::set_enabled(false);

    let loss_cpu = MSELoss::apply(&output_cpu, &target_cpu);
    loss_cpu.backward();

    let out_cuda_grad = output_cuda.grad().expect("cuda output grad");
    let out_cpu_grad = output_cpu.grad().expect("cpu output grad");
    let tar_cuda_grad = target_cuda.grad().expect("cuda target grad");
    let tar_cpu_grad = target_cpu.grad().expect("cpu target grad");
    for (got, expect) in out_cuda_grad.iter().zip(out_cpu_grad.iter()) {
        assert!(
            (got - expect).abs() < 1e-5,
            "output grad got {got}, expect {expect}"
        );
    }
    for (got, expect) in tar_cuda_grad.iter().zip(tar_cpu_grad.iter()) {
        assert!(
            (got - expect).abs() < 1e-5,
            "target grad got {got}, expect {expect}"
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_mse_loss_lowp_inputs_read_resident_buffers_and_write_f32_grads() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let output_data = vec![1.0, -2.0, 0.5, 3.0, -1.0, 2.0];
    let target_data = vec![0.5, -1.5, 0.25, 2.5, -0.5, 1.0];
    let output_cpu = Tensor::parameter_with_dtype(
        Array::from_shape_vec(IxDyn(&[2, 3]), output_data.clone())
            .unwrap()
            .into_dyn(),
        DType::BF16,
    );
    let target_cpu = Tensor::parameter_with_dtype(
        Array::from_shape_vec(IxDyn(&[2, 3]), target_data.clone())
            .unwrap()
            .into_dyn(),
        DType::I8,
    );
    let output_cuda = Tensor::parameter_with_dtype(
        Array::from_shape_vec(IxDyn(&[2, 3]), output_data)
            .unwrap()
            .into_dyn(),
        DType::BF16,
    )
    .to_cuda();
    let target_cuda = Tensor::parameter_with_dtype(
        Array::from_shape_vec(IxDyn(&[2, 3]), target_data)
            .unwrap()
            .into_dyn(),
        DType::I8,
    )
    .to_cuda();
    assert!(output_cuda.0.borrow().cuda_f32_data.is_none());
    assert!(target_cuda.0.borrow().cuda_f32_data.is_none());
    assert!(output_cuda.0.borrow().cuda_bf16_data.is_some());
    assert!(target_cuda.0.borrow().cuda_i8_data.is_some());

    crate::ops::cuda::set_enabled(true);
    set_strict_device_execution(true);
    let loss_cuda = MSELoss::apply(&output_cuda, &target_cuda);
    loss_cuda.backward();
    assert!(output_cuda.0.borrow().cuda_f32_data.is_none());
    assert!(target_cuda.0.borrow().cuda_f32_data.is_none());
    assert!(output_cuda.0.borrow().cuda_bf16_data.is_some());
    assert!(target_cuda.0.borrow().cuda_i8_data.is_some());
    assert!(output_cuda.cloned_cuda_f32_grad().is_some());
    assert!(target_cuda.cloned_cuda_f32_grad().is_some());
    assert!(!output_cuda.has_host_grad());
    assert!(!target_cuda.has_host_grad());
    set_strict_device_execution(false);
    crate::ops::cuda::set_enabled(false);

    let loss_cpu = MSELoss::apply(&output_cpu, &target_cpu);
    loss_cpu.backward();
    let got_loss = loss_cuda.data_ref().first().copied().unwrap_or_default();
    let expect_loss = loss_cpu.data_ref().first().copied().unwrap_or_default();
    assert!(
        (got_loss - expect_loss).abs() < 3e-2,
        "loss got {got_loss}, expect {expect_loss}"
    );

    let got_output_grad = output_cuda.grad().expect("cuda lowp output grad");
    let expect_output_grad = output_cpu.grad().expect("cpu lowp output grad");
    for (got, expect) in got_output_grad.iter().zip(expect_output_grad.iter()) {
        assert!(
            (got - expect).abs() < 3e-2,
            "output grad got {got}, expect {expect}"
        );
    }
    let got_target_grad = target_cuda.grad().expect("cuda lowp target grad");
    let expect_target_grad = target_cpu.grad().expect("cpu lowp target grad");
    for (got, expect) in got_target_grad.iter().zip(expect_target_grad.iter()) {
        assert!(
            (got - expect).abs() < 3e-2,
            "target grad got {got}, expect {expect}"
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_cross_entropy_backward_matches_cpu_reference_in_strict_mode() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let logits_cpu = Tensor::from_data_with_grad_flag(
        Array::from_shape_vec(IxDyn(&[2, 3]), vec![2.0, 0.0, -1.0, -1.0, 0.0, 2.0])
            .unwrap()
            .into_dyn(),
        true,
    );
    let targets_cpu = Tensor::from_data_with_grad_flag(
        Array::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0])
            .unwrap()
            .into_dyn(),
        true,
    );
    let logits_cuda = logits_cpu.to_cuda();
    let targets_cuda = targets_cpu.to_cuda();

    crate::ops::cuda::set_enabled(true);
    set_strict_device_execution(true);
    let loss_cuda = CrossEntropyLoss::apply(&logits_cuda, &targets_cuda);
    loss_cuda.backward();
    assert!(logits_cuda.cloned_cuda_f32_grad().is_some());
    assert!(!logits_cuda.has_host_grad());
    set_strict_device_execution(false);
    crate::ops::cuda::set_enabled(false);

    let loss_cpu = CrossEntropyLoss::apply(&logits_cpu, &targets_cpu);
    loss_cpu.backward();

    let logits_cuda_grad = logits_cuda.grad().expect("cuda logits grad");
    let logits_cpu_grad = logits_cpu.grad().expect("cpu logits grad");
    for (got, expect) in logits_cuda_grad.iter().zip(logits_cpu_grad.iter()) {
        assert!(
            (got - expect).abs() < 1e-5,
            "logits grad got {got}, expect {expect}"
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_cross_entropy_lowp_logits_and_target_read_resident_buffers() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let logits_data = vec![2.0, 0.0, -1.0, -1.0, 0.0, 2.0];
    let target_data = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
    let logits_cpu = Tensor::parameter_with_dtype(
        Array::from_shape_vec(IxDyn(&[2, 3]), logits_data.clone())
            .unwrap()
            .into_dyn(),
        DType::BF16,
    );
    let targets_cpu = make_tensor(&[2, 3], target_data.clone(), DType::I8);
    let logits_cuda = Tensor::parameter_with_dtype(
        Array::from_shape_vec(IxDyn(&[2, 3]), logits_data)
            .unwrap()
            .into_dyn(),
        DType::BF16,
    )
    .to_cuda();
    let targets_cuda = make_tensor(&[2, 3], target_data, DType::I8).to_cuda();
    assert!(logits_cuda.0.borrow().cuda_f32_data.is_none());
    assert!(targets_cuda.0.borrow().cuda_f32_data.is_none());
    assert!(logits_cuda.0.borrow().cuda_bf16_data.is_some());
    assert!(targets_cuda.0.borrow().cuda_i8_data.is_some());

    crate::ops::cuda::set_enabled(true);
    set_strict_device_execution(true);
    let loss_cuda = CrossEntropyLoss::apply(&logits_cuda, &targets_cuda);
    loss_cuda.backward();
    assert!(logits_cuda.0.borrow().cuda_f32_data.is_none());
    assert!(targets_cuda.0.borrow().cuda_f32_data.is_none());
    assert!(logits_cuda.0.borrow().cuda_bf16_data.is_some());
    assert!(targets_cuda.0.borrow().cuda_i8_data.is_some());
    assert!(logits_cuda.cloned_cuda_f32_grad().is_some());
    assert!(!logits_cuda.has_host_grad());
    set_strict_device_execution(false);
    crate::ops::cuda::set_enabled(false);

    let loss_cpu = CrossEntropyLoss::apply(&logits_cpu, &targets_cpu);
    loss_cpu.backward();
    let got_loss = loss_cuda.data_ref().first().copied().unwrap_or_default();
    let expect_loss = loss_cpu.data_ref().first().copied().unwrap_or_default();
    assert!(
        (got_loss - expect_loss).abs() < 3e-2,
        "loss got {got_loss}, expect {expect_loss}"
    );

    let got_grad = logits_cuda.grad().expect("cuda lowp logits grad");
    let expect_grad = logits_cpu.grad().expect("cpu lowp logits grad");
    for (got, expect) in got_grad.iter().zip(expect_grad.iter()) {
        assert!(
            (got - expect).abs() < 3e-2,
            "grad got {got}, expect {expect}"
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_cross_entropy_forward_keeps_resident_scalar_loss_buffer() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let logits_cpu = Tensor::from_data_with_grad_flag(
        Array::from_shape_vec(IxDyn(&[2, 3]), vec![2.0, 0.0, -1.0, -1.0, 0.0, 2.0])
            .unwrap()
            .into_dyn(),
        false,
    );
    let targets_cpu = Tensor::from_data_with_grad_flag(
        Array::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0])
            .unwrap()
            .into_dyn(),
        false,
    );
    let logits_cuda = logits_cpu.to_cuda();
    let targets_cuda = targets_cpu.to_cuda();

    crate::ops::cuda::set_enabled(true);
    set_strict_device_execution(true);
    let loss_cuda = no_grad(|| CrossEntropyLoss::apply(&logits_cuda, &targets_cuda));
    let loss_buffer = loss_cuda
        .cloned_cuda_f32_buffer()
        .expect("CUDA cross_entropy loss should keep a resident scalar buffer");
    assert_eq!(loss_buffer.len(), 1);
    set_strict_device_execution(false);
    crate::ops::cuda::set_enabled(false);

    let loss_cpu = CrossEntropyLoss::apply(&logits_cpu, &targets_cpu);
    let got = loss_cuda.data_ref().first().copied().unwrap_or_default();
    let expect = loss_cpu.data_ref().first().copied().unwrap_or_default();
    assert!(
        (got - expect).abs() < 1e-5,
        "loss got {got}, expect {expect}"
    );
}
