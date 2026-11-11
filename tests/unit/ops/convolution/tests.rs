use super::conv2d;
#[cfg(feature = "cuda")]
use super::max_pool2d;
use crate::autograd::{Tensor, no_grad};
use crate::precision::DType;
use ndarray::{Array, IxDyn};

fn make_tensor(shape: &[usize], data: Vec<f32>, dtype: DType) -> Tensor {
    Tensor::from_f32_data_no_grad_with_dtype(
        Array::from_shape_vec(IxDyn(shape), data)
            .expect("tensor shape mismatch")
            .into_dyn(),
        dtype,
    )
}

#[cfg(feature = "cuda")]
fn make_grad_tensor(shape: &[usize], data: Vec<f32>) -> Tensor {
    Tensor::from_data_with_grad_flag(
        Array::from_shape_vec(IxDyn(shape), data)
            .expect("tensor shape mismatch")
            .into_dyn(),
        true,
    )
}

#[cfg(feature = "cuda")]
fn sample_f32(len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| (((i * 13 + 5) % 29) as f32) / 12.0 - 1.1)
        .collect()
}

#[cfg(feature = "cuda")]
fn assert_cuda_native_resident_without_f32(tensor: &Tensor, dtype: DType) {
    let inner = tensor.0.borrow();
    assert!(inner.cuda_f32_data.is_none());
    match dtype {
        DType::F16 => assert!(inner.cuda_f16_data.is_some()),
        DType::BF16 => assert!(inner.cuda_bf16_data.is_some()),
        DType::I8 => assert!(inner.cuda_i8_data.is_some()),
        DType::F32 => assert!(inner.cuda_f32_data.is_some()),
    }
}

#[test]
#[should_panic(expected = "kernel dimensions must be greater than zero")]
fn conv2d_rejects_zero_sized_kernel() {
    let input = make_tensor(&[1, 1, 4, 4], vec![0.0; 16], DType::F32);
    let weight = make_tensor(&[1, 1, 0, 3], vec![], DType::F32);
    let _ = no_grad(|| conv2d(&input, &weight, None, (1, 1), (0, 0)));
}

#[test]
#[should_panic(expected = "channels must be greater than zero")]
fn conv2d_rejects_zero_channels() {
    let input = make_tensor(&[1, 0, 4, 4], vec![], DType::F32);
    let weight = make_tensor(&[1, 0, 3, 3], vec![], DType::F32);
    let _ = no_grad(|| conv2d(&input, &weight, None, (1, 1), (0, 0)));
}

#[test]
#[should_panic(expected = "stride must be greater than zero")]
fn conv2d_rejects_zero_stride() {
    let input = make_tensor(&[1, 1, 4, 4], vec![0.0; 16], DType::F32);
    let weight = make_tensor(&[1, 1, 3, 3], vec![0.0; 9], DType::F32);
    let _ = no_grad(|| conv2d(&input, &weight, None, (0, 1), (0, 0)));
}

#[test]
#[should_panic(expected = "padded height overflow")]
fn conv2d_rejects_padding_overflow() {
    let input = make_tensor(&[1, 1, 1, 1], vec![0.0], DType::F32);
    let weight = make_tensor(&[1, 1, 1, 1], vec![0.0], DType::F32);
    let _ = no_grad(|| conv2d(&input, &weight, None, (1, 1), (usize::MAX, 0)));
}

#[test]
#[should_panic(expected = "size overflow")]
fn conv2d_rejects_aggregate_size_overflow() {
    let input = make_tensor(&[1, 1, 1, 1], vec![0.0], DType::F32);
    let weight = make_tensor(&[1, 1, 1, 1], vec![0.0], DType::F32);
    let padding = usize::MAX / 4;
    let _ = no_grad(|| conv2d(&input, &weight, None, (1, 1), (padding, padding)));
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_empty_batch_conv2d_no_grad_stays_on_cuda_in_strict_mode() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let input = make_tensor(&[0, 1, 4, 4], vec![], DType::F32).to_cuda();
    let weight = make_tensor(&[2, 1, 3, 3], sample_f32(18), DType::F32).to_cuda();
    let bias = make_tensor(&[2], vec![0.1, -0.2], DType::F32).to_cuda();

    crate::ops::cuda::set_enabled(true);
    crate::device::set_strict_device_execution(true);
    let out = no_grad(|| conv2d(&input, &weight, Some(&bias), (1, 1), (1, 1)));
    crate::device::set_strict_device_execution(false);
    crate::ops::cuda::set_enabled(false);

    assert!(out.is_cuda());
    assert_eq!(out.shape_vec(), vec![0, 2, 4, 4]);
    assert_eq!(out.len(), 0);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_empty_batch_max_pool2d_no_grad_stays_on_cuda_in_strict_mode() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let input = make_tensor(&[0, 2, 4, 4], vec![], DType::F32).to_cuda();

    crate::ops::cuda::set_enabled(true);
    crate::device::set_strict_device_execution(true);
    let out = no_grad(|| max_pool2d(&input, (2, 2), (1, 1)));
    crate::device::set_strict_device_execution(false);
    crate::ops::cuda::set_enabled(false);

    assert!(out.is_cuda());
    assert_eq!(out.shape_vec(), vec![0, 2, 3, 3]);
    assert_eq!(out.len(), 0);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_conv2d_matches_cpu_reference_in_strict_mode() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let input = make_tensor(
        &[1, 2, 4, 4],
        vec![
            1.0, 0.5, -1.0, 2.0, 0.0, 1.5, 3.0, -0.5, 2.0, -1.5, 0.25, 1.25, -2.0, 0.75, 1.0, 0.5,
            -0.5, 1.0, 0.0, 2.0, 1.5, -1.0, 0.5, 0.25, -0.75, 2.5, 1.0, -1.5, 0.5, 1.0, -0.25, 1.5,
        ],
        DType::F32,
    );
    let weight = make_tensor(
        &[2, 2, 3, 3],
        vec![
            0.25, -0.5, 1.0, 0.75, 0.0, -0.25, 1.5, -1.0, 0.5, -0.5, 1.0, 0.25, 0.0, -0.75, 0.5,
            1.25, -0.25, -0.5, 0.5, 0.25, -1.0, 1.0, -0.5, 0.75, -0.25, 0.5, 1.25, -0.75, 1.0,
            -0.25, 0.5, 0.0, -0.5, 0.75, 0.25, 1.0,
        ],
        DType::F32,
    );
    let bias = make_tensor(&[2], vec![0.1, -0.2], DType::F32);

    crate::ops::cuda::set_enabled(true);
    crate::device::set_strict_device_execution(true);
    let cuda_out = no_grad(|| {
        conv2d(
            &input.to_cuda(),
            &weight.to_cuda(),
            Some(&bias.to_cuda()),
            (1, 1),
            (1, 1),
        )
    });
    crate::device::set_strict_device_execution(false);
    crate::ops::cuda::set_enabled(false);

    let cpu_out = no_grad(|| conv2d(&input, &weight, Some(&bias), (1, 1), (1, 1)));
    assert!(cuda_out.is_cuda());
    assert!(!cuda_out.requires_grad());
    assert_eq!(cuda_out.shape_vec(), cpu_out.shape_vec());

    for (got, expect) in cuda_out.data_ref().iter().zip(cpu_out.data_ref().iter()) {
        assert!((got - expect).abs() < 1e-4, "got {got}, expect {expect}");
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_conv2d_backward_matches_cpu_reference_in_strict_mode() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let input_cpu = make_grad_tensor(&[1, 2, 4, 4], sample_f32(32));
    let weight_cpu = make_grad_tensor(
        &[3, 2, 3, 3],
        sample_f32(54).into_iter().map(|v| v * 0.5).collect(),
    );
    let bias_cpu = make_grad_tensor(&[3], vec![0.1, -0.2, 0.3]);
    let coeff_cpu = Tensor::from_data_with_grad_flag(
        Array::from_shape_vec(
            IxDyn(&[1, 3, 4, 4]),
            sample_f32(48).into_iter().map(|v| v * 0.25).collect(),
        )
        .expect("tensor shape mismatch")
        .into_dyn(),
        false,
    );

    let input_cuda = input_cpu.to_cuda();
    let weight_cuda = weight_cpu.to_cuda();
    let bias_cuda = bias_cpu.to_cuda();
    let coeff_cuda = coeff_cpu.to_cuda();

    crate::ops::cuda::set_enabled(true);
    crate::device::set_strict_device_execution(true);
    let out_cuda = conv2d(&input_cuda, &weight_cuda, Some(&bias_cuda), (1, 1), (1, 1));
    assert!(out_cuda.is_cuda());
    let loss_cuda = crate::ops::arithmetic::sum(&(&out_cuda * &coeff_cuda));
    loss_cuda.backward();
    assert!(input_cuda.cloned_cuda_f32_grad().is_some());
    assert!(weight_cuda.cloned_cuda_f32_grad().is_some());
    assert!(bias_cuda.cloned_cuda_f32_grad().is_some());
    assert!(!input_cuda.has_host_grad());
    assert!(!weight_cuda.has_host_grad());
    assert!(!bias_cuda.has_host_grad());
    crate::device::set_strict_device_execution(false);
    crate::ops::cuda::set_enabled(false);

    let out_cpu = conv2d(&input_cpu, &weight_cpu, Some(&bias_cpu), (1, 1), (1, 1));
    let loss_cpu = crate::ops::arithmetic::sum(&(&out_cpu * &coeff_cpu));
    loss_cpu.backward();

    let input_cuda_grad = input_cuda.grad().expect("cuda conv2d input grad");
    let input_cpu_grad = input_cpu.grad().expect("cpu conv2d input grad");
    for (got, expect) in input_cuda_grad.iter().zip(input_cpu_grad.iter()) {
        assert!((got - expect).abs() < 1e-4, "got {got}, expect {expect}");
    }
    let weight_cuda_grad = weight_cuda.grad().expect("cuda conv2d weight grad");
    let weight_cpu_grad = weight_cpu.grad().expect("cpu conv2d weight grad");
    for (got, expect) in weight_cuda_grad.iter().zip(weight_cpu_grad.iter()) {
        assert!((got - expect).abs() < 1e-4, "got {got}, expect {expect}");
    }
    let bias_cuda_grad = bias_cuda.grad().expect("cuda conv2d bias grad");
    let bias_cpu_grad = bias_cpu.grad().expect("cpu conv2d bias grad");
    for (got, expect) in bias_cuda_grad.iter().zip(bias_cpu_grad.iter()) {
        assert!((got - expect).abs() < 1e-4, "got {got}, expect {expect}");
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_max_pool2d_matches_cpu_reference_in_strict_mode() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let input = make_tensor(
        &[1, 2, 4, 4],
        vec![
            1.0, -0.5, 0.25, 2.0, -1.0, 0.75, 1.5, -0.25, 0.5, 1.25, -0.75, 0.0, 2.5, -1.5, 0.5,
            1.0, -0.75, 1.5, 0.25, -1.25, 1.0, -0.5, 2.0, 0.75, -1.5, 0.5, 1.25, -0.25, 0.0, 2.5,
            -2.0, 1.75,
        ],
        DType::F32,
    );

    crate::ops::cuda::set_enabled(true);
    crate::device::set_strict_device_execution(true);
    let cuda_out = no_grad(|| max_pool2d(&input.to_cuda(), (2, 2), (1, 1)));
    crate::device::set_strict_device_execution(false);
    crate::ops::cuda::set_enabled(false);

    let cpu_out = no_grad(|| max_pool2d(&input, (2, 2), (1, 1)));
    assert!(cuda_out.is_cuda());
    assert!(!cuda_out.requires_grad());
    assert_eq!(cuda_out.shape_vec(), cpu_out.shape_vec());
    for (got, expect) in cuda_out.data_ref().iter().zip(cpu_out.data_ref().iter()) {
        assert!((got - expect).abs() < 1e-5, "got {got}, expect {expect}");
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_max_pool2d_bf16_reads_native_input_in_strict_mode() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let input = make_tensor(
        &[1, 1, 4, 4],
        vec![
            1.0, -0.5, 0.25, 2.0, -1.0, 0.75, 1.5, -0.25, 0.5, 1.25, -0.75, 0.0, 2.5, -1.5, 0.5,
            1.0,
        ],
        DType::BF16,
    )
    .to_cuda();
    assert!(input.0.borrow().cuda_f32_data.is_none());
    assert!(input.0.borrow().cuda_bf16_data.is_some());

    crate::ops::cuda::set_enabled(true);
    crate::device::set_strict_device_execution(true);
    let cuda_out = no_grad(|| max_pool2d(&input, (2, 2), (1, 1)));
    crate::device::set_strict_device_execution(false);
    crate::ops::cuda::set_enabled(false);

    assert!(input.0.borrow().cuda_f32_data.is_none());
    assert!(input.0.borrow().cuda_bf16_data.is_some());
    let cpu_out = no_grad(|| max_pool2d(&input.to_cpu(), (2, 2), (1, 1)));
    assert!(cuda_out.is_cuda());
    assert_eq!(cuda_out.dtype(), DType::F32);
    assert_eq!(cuda_out.shape_vec(), cpu_out.shape_vec());
    for (got, expect) in cuda_out.data_ref().iter().zip(cpu_out.data_ref().iter()) {
        assert!((got - expect).abs() < 2e-2, "got {got}, expect {expect}");
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_max_pool2d_backward_matches_cpu_reference_in_strict_mode() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let input_cpu = make_grad_tensor(&[1, 2, 4, 4], sample_f32(32));
    let coeff_cpu = Tensor::from_data_with_grad_flag(
        Array::from_shape_vec(
            IxDyn(&[1, 2, 3, 3]),
            sample_f32(18).into_iter().map(|v| v * 0.25).collect(),
        )
        .expect("tensor shape mismatch")
        .into_dyn(),
        false,
    );
    let input_cuda = input_cpu.to_cuda();
    let coeff_cuda = coeff_cpu.to_cuda();

    crate::ops::cuda::set_enabled(true);
    crate::device::set_strict_device_execution(true);
    let cuda_out = max_pool2d(&input_cuda, (2, 2), (1, 1));
    assert!(cuda_out.is_cuda());
    let cuda_loss = crate::ops::arithmetic::sum(&(&cuda_out * &coeff_cuda));
    cuda_loss.backward();
    assert!(input_cuda.cloned_cuda_f32_grad().is_some());
    assert!(!input_cuda.has_host_grad());
    crate::device::set_strict_device_execution(false);
    crate::ops::cuda::set_enabled(false);

    let cpu_out = max_pool2d(&input_cpu, (2, 2), (1, 1));
    let cpu_loss = crate::ops::arithmetic::sum(&(&cpu_out * &coeff_cpu));
    cpu_loss.backward();

    let cuda_grad = input_cuda.grad().expect("cuda max_pool2d input grad");
    let cpu_grad = input_cpu.grad().expect("cpu max_pool2d input grad");
    for (got, expect) in cuda_grad.iter().zip(cpu_grad.iter()) {
        assert!((got - expect).abs() < 1e-5, "got {got}, expect {expect}");
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_max_pool2d_backward_bf16_reads_native_input_in_strict_mode() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let input_data = vec![
        1.0, -0.5, 0.25, 2.0, -1.0, 0.75, 1.5, -0.25, 0.5, 1.25, -0.75, 0.0, 2.5, -1.5, 0.5, 1.0,
    ];
    let input_cpu = Tensor::parameter_with_dtype(
        Array::from_shape_vec(IxDyn(&[1, 1, 4, 4]), input_data.clone())
            .expect("tensor shape mismatch")
            .into_dyn(),
        DType::BF16,
    );
    let input_cuda = Tensor::parameter_with_dtype(
        Array::from_shape_vec(IxDyn(&[1, 1, 4, 4]), input_data)
            .expect("tensor shape mismatch")
            .into_dyn(),
        DType::BF16,
    )
    .to_cuda();
    let coeff_cpu = Tensor::from_data_with_grad_flag(
        Array::from_shape_vec(IxDyn(&[1, 1, 3, 3]), sample_f32(9))
            .expect("tensor shape mismatch")
            .into_dyn(),
        false,
    );
    let coeff_cuda = coeff_cpu.to_cuda();
    assert!(input_cuda.0.borrow().cuda_f32_data.is_none());
    assert!(input_cuda.0.borrow().cuda_bf16_data.is_some());

    crate::ops::cuda::set_enabled(true);
    crate::device::set_strict_device_execution(true);
    let cuda_out = max_pool2d(&input_cuda, (2, 2), (1, 1));
    let cuda_loss = crate::ops::arithmetic::sum(&(&cuda_out * &coeff_cuda));
    cuda_loss.backward();
    assert!(input_cuda.cloned_cuda_f32_grad().is_some());
    assert!(!input_cuda.has_host_grad());
    assert!(input_cuda.0.borrow().cuda_bf16_data.is_some());
    crate::device::set_strict_device_execution(false);
    crate::ops::cuda::set_enabled(false);

    let cpu_out = max_pool2d(&input_cpu, (2, 2), (1, 1));
    let cpu_loss = crate::ops::arithmetic::sum(&(&cpu_out * &coeff_cpu));
    cpu_loss.backward();

    let cuda_grad = input_cuda.grad().expect("cuda bf16 max_pool2d input grad");
    let cpu_grad = input_cpu.grad().expect("cpu bf16 max_pool2d input grad");
    for (got, expect) in cuda_grad.iter().zip(cpu_grad.iter()) {
        assert!((got - expect).abs() < 1e-5, "got {got}, expect {expect}");
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_conv2d_accepts_lowp_inputs_in_strict_mode() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    for dtype in [DType::F16, DType::BF16, DType::I8] {
        let input = make_tensor(
            &[1, 1, 4, 4],
            vec![
                1.0, -0.5, 0.25, 2.0, -1.0, 0.75, 1.5, -0.25, 0.5, 1.25, -0.75, 0.0, 2.5, -1.5,
                0.5, 1.0,
            ],
            dtype,
        );
        let weight = make_tensor(
            &[2, 1, 3, 3],
            vec![
                0.5, -0.25, 1.0, -0.75, 0.25, 0.5, 1.25, -1.0, 0.0, -0.5, 1.0, 0.25, 0.75, -0.5,
                0.0, 0.5, -0.25, 1.5,
            ],
            dtype,
        );
        let bias = make_tensor(&[2], vec![0.1, -0.15], dtype);
        let input_cuda = input.to_cuda();
        let weight_cuda = weight.to_cuda();
        let bias_cuda = bias.to_cuda();
        assert_cuda_native_resident_without_f32(&input_cuda, dtype);
        assert_cuda_native_resident_without_f32(&weight_cuda, dtype);
        assert_cuda_native_resident_without_f32(&bias_cuda, dtype);

        crate::ops::cuda::set_enabled(true);
        crate::device::set_strict_device_execution(true);
        let cuda_out =
            no_grad(|| conv2d(&input_cuda, &weight_cuda, Some(&bias_cuda), (1, 1), (1, 1)));
        crate::device::set_strict_device_execution(false);
        crate::ops::cuda::set_enabled(false);
        assert_cuda_native_resident_without_f32(&input_cuda, dtype);
        assert_cuda_native_resident_without_f32(&weight_cuda, dtype);
        assert_cuda_native_resident_without_f32(&bias_cuda, dtype);

        let cpu_out = no_grad(|| conv2d(&input, &weight, Some(&bias), (1, 1), (1, 1)));
        assert!(cuda_out.is_cuda());
        assert_eq!(cuda_out.dtype(), DType::F32);
        for (got, expect) in cuda_out.data_ref().iter().zip(cpu_out.data_ref().iter()) {
            assert!(
                (got - expect).abs() < 3e-2,
                "dtype {dtype:?}: got {got}, expect {expect}"
            );
        }
    }
}
