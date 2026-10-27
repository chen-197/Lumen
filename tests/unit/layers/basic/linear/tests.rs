use super::*;
use crate::autograd::no_grad;
#[cfg(feature = "cuda")]
use crate::autograd::set_strict_device_execution;
use crate::ops::arithmetic::sum;
use crate::precision::{
    ParameterQuantization, PrecisionConfig, with_parameter_quantization, with_precision_config,
};
use ndarray::{ArrayD, IxDyn};

fn training_tensor_with_dtype(shape: &[usize], data: Vec<f32>, dtype: DType) -> Tensor {
    let tensor = Tensor::from_data_with_grad_flag(
        ArrayD::from_shape_vec(IxDyn(shape), data).expect("tensor shape"),
        true,
    );
    tensor.cast_inplace(dtype);
    tensor
}

fn install_linear_values(linear: &Linear, dtype: DType) {
    linear.weight.set_array_f32_with_dtype(
        ArrayD::from_shape_vec(
            IxDyn(&[3, 4]),
            vec![
                0.25, -0.5, 0.75, 1.0, -1.25, 0.5, 1.5, -0.25, 0.875, -1.0, 0.375, 0.625,
            ],
        )
        .expect("weight shape"),
        dtype,
    );
    linear
        .bias
        .as_ref()
        .expect("linear bias")
        .set_array_f32_with_dtype(
            ArrayD::from_shape_vec(IxDyn(&[3]), vec![0.125, -0.25, 0.375]).expect("bias shape"),
            dtype,
        );
}

fn assert_no_host_f32_cache(tensor: &Tensor, label: &str) {
    assert!(
        !tensor.has_host_f32_data(),
        "{label} should keep native low-precision storage without host f32 cache"
    );
}

fn assert_grad_close(got: &Tensor, expect: &Tensor, label: &str, tol: f32) {
    let got_grad = got.grad().unwrap_or_else(|| panic!("{label} grad missing"));
    let expect_grad = expect
        .grad()
        .unwrap_or_else(|| panic!("{label} reference grad missing"));
    for (idx, (got, expect)) in got_grad.iter().zip(expect_grad.iter()).enumerate() {
        assert!(
            (got - expect).abs() <= tol,
            "{label} grad[{idx}] got {got}, expected {expect}"
        );
    }
}

#[test]
fn linear_explicit_dtype_overrides_global_default() {
    with_precision_config(
        PrecisionConfig {
            parameter_dtype: DType::BF16,
            runtime_dtype: DType::F32,
            allow_parameter_dtype_copies: false,
        },
        || {
            let linear = Linear::new_with_dtype(4, 3, DType::F32);
            assert_eq!(linear.weight.dtype(), DType::F32);
            assert_eq!(
                linear.bias.as_ref().expect("linear bias").dtype(),
                DType::F32
            );
        },
    );
}

#[test]
fn linear_decode_i8_no_copy_uses_native_weight_without_materializing_f32_cache() {
    with_precision_config(
        PrecisionConfig {
            parameter_dtype: DType::I8,
            runtime_dtype: DType::F32,
            allow_parameter_dtype_copies: false,
        },
        || {
            let linear = Linear::new_no_bias_with_dtype(4, 3, DType::I8);
            linear.weight.set_array_i8_with_dtype(
                ArrayD::from_shape_vec(
                    IxDyn(&[3, 4]),
                    vec![2i8, -1, 0, 4, -3, 1, 2, -2, 1, 1, -4, 3],
                )
                .expect("weight shape"),
                0.25,
                DType::I8,
            );
            {
                let inner = linear.weight.0.borrow();
                assert_eq!(inner.storage_dtype, DType::I8);
                assert!(
                    !inner.has_f32_data,
                    "i8 weight should start without f32 cache"
                );
            }

            let input = [1.0f32, -2.0, 0.5, 3.0];
            let mut out = [0.0f32; 3];
            no_grad(|| {
                linear.forward_decode_slice_no_bias_into(&input, &mut out);
            });

            let expected = [
                (2.0 * 1.0 + -1.0 * -2.0 + 0.0 * 0.5 + 4.0 * 3.0) * 0.25,
                (-3.0 * 1.0 + 1.0 * -2.0 + 2.0 * 0.5 + -2.0 * 3.0) * 0.25,
                (1.0 * 1.0 + 1.0 * -2.0 + -4.0 * 0.5 + 3.0 * 3.0) * 0.25,
            ];
            for (got, want) in out.iter().zip(expected.iter()) {
                assert!((got - want).abs() < 1e-5, "got {got}, expected {want}");
            }

            let inner = linear.weight.0.borrow();
            assert!(
                !inner.has_f32_data,
                "native i8 decode path should not materialize f32 cache"
            );
        },
    );
}

#[test]
fn linear_decode_rows_no_bias_matches_single_row_decode() {
    let linear = Linear::new_no_bias_with_dtype(4, 3, DType::BF16);
    linear.weight.set_array_f32_with_dtype(
        ArrayD::from_shape_vec(
            IxDyn(&[3, 4]),
            vec![
                1.0, 0.0, -1.0, 2.0, 0.5, 1.5, -0.5, 0.25, -1.0, 2.0, 1.0, -0.5,
            ],
        )
        .expect("weight shape"),
        DType::BF16,
    );
    let input = vec![1.0f32, -2.0, 0.5, 3.0, -0.25, 0.75, 2.0, -1.5];

    let out = no_grad(|| linear.forward_decode_rows_no_bias(&input, 2));
    let first = no_grad(|| linear.forward_decode_slice_no_bias(&input[..4]));
    let second = no_grad(|| linear.forward_decode_slice_no_bias(&input[4..]));
    let out_vals = out.data_ref().iter().copied().collect::<Vec<_>>();
    let first_vals = first.data_ref().iter().copied().collect::<Vec<_>>();
    let second_vals = second.data_ref().iter().copied().collect::<Vec<_>>();

    assert_eq!(out.shape_vec(), vec![2, 1, 3]);
    assert_eq!(&out_vals[..3], first_vals.as_slice());
    assert_eq!(&out_vals[3..], second_vals.as_slice());
}

#[test]
fn linear_default_construction_can_follow_global_quantization() {
    with_precision_config(
        PrecisionConfig {
            parameter_dtype: DType::F32,
            runtime_dtype: DType::F32,
            allow_parameter_dtype_copies: false,
        },
        || {
            with_parameter_quantization(ParameterQuantization::Int8, || {
                let linear = Linear::new(4, 3);
                assert_eq!(linear.weight.dtype(), DType::I8);
                assert_eq!(
                    linear.bias.as_ref().expect("linear bias").dtype(),
                    DType::I8
                );
            });
        },
    );
}

#[test]
fn linear_training_lowp_forward_keeps_f32_grads_without_host_cache() {
    for dtype in [DType::F16, DType::BF16, DType::I8] {
        let input = training_tensor_with_dtype(
            &[2, 4],
            vec![1.0, -2.0, 0.5, 3.0, -1.0, 2.0, 0.25, -0.75],
            dtype,
        );
        let linear = Linear::new_with_dtype(4, 3, dtype);
        install_linear_values(&linear, dtype);

        assert_no_host_f32_cache(&input, "linear input");
        assert_no_host_f32_cache(&linear.weight, "linear weight");
        assert_no_host_f32_cache(linear.bias.as_ref().expect("linear bias"), "linear bias");

        let out = linear.forward(input.clone());
        assert_eq!(
            out.dtype(),
            DType::F32,
            "training Linear output should promote {dtype:?} data to f32 for backward"
        );
        sum(&out).backward();

        assert_no_host_f32_cache(&input, "linear input");
        assert_no_host_f32_cache(&linear.weight, "linear weight");
        assert_no_host_f32_cache(linear.bias.as_ref().expect("linear bias"), "linear bias");
        assert!(input.cloned_cuda_f32_grad().is_none());
        assert!(linear.weight.cloned_cuda_f32_grad().is_none());
        assert!(
            linear
                .bias
                .as_ref()
                .expect("linear bias")
                .cloned_cuda_f32_grad()
                .is_none()
        );

        let reference_input = training_tensor_with_dtype(
            &[2, 4],
            vec![1.0, -2.0, 0.5, 3.0, -1.0, 2.0, 0.25, -0.75],
            DType::F32,
        );
        let reference = Linear::new_with_dtype(4, 3, DType::F32);
        install_linear_values(&reference, DType::F32);
        sum(&reference.forward(reference_input.clone())).backward();

        let tol = match dtype {
            DType::F16 => 2e-3,
            DType::BF16 => 1.5e-2,
            DType::I8 => 0.12,
            DType::F32 => unreachable!(),
        };
        assert_grad_close(&input, &reference_input, "linear input", tol);
        assert_grad_close(&linear.weight, &reference.weight, "linear weight", tol);
        assert_grad_close(
            linear.bias.as_ref().expect("linear bias"),
            reference.bias.as_ref().expect("reference bias"),
            "linear bias",
            tol,
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_linear_training_lowp_forward_uses_resident_data_and_f32_grads() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    crate::ops::cuda::set_enabled(true);
    set_strict_device_execution(true);

    for dtype in [DType::F16, DType::BF16, DType::I8] {
        let input = training_tensor_with_dtype(
            &[2, 4],
            vec![1.0, -2.0, 0.5, 3.0, -1.0, 2.0, 0.25, -0.75],
            dtype,
        )
        .to_cuda();
        let linear = Linear::new_with_dtype(4, 3, dtype);
        install_linear_values(&linear, dtype);
        linear.to_cuda();

        assert!(input.cloned_cuda_native_lowp_buffer().is_some());
        assert!(linear.weight.cloned_cuda_native_lowp_buffer().is_some());
        assert!(
            linear
                .bias
                .as_ref()
                .expect("linear bias")
                .cloned_cuda_native_lowp_buffer()
                .is_some()
        );

        let out = linear.forward(input.clone());
        assert_eq!(
            out.dtype(),
            DType::F32,
            "CUDA training Linear output should be f32 for {dtype:?}"
        );
        assert!(out.cloned_cuda_f32_buffer().is_some());
        sum(&out).backward();

        assert!(input.cloned_cuda_f32_grad().is_some());
        assert!(linear.weight.cloned_cuda_f32_grad().is_some());
        assert!(
            linear
                .bias
                .as_ref()
                .expect("linear bias")
                .cloned_cuda_f32_grad()
                .is_some()
        );
        assert_no_host_f32_cache(&input, "CUDA linear input");
        assert_no_host_f32_cache(&linear.weight, "CUDA linear weight");
        assert_no_host_f32_cache(
            linear.bias.as_ref().expect("linear bias"),
            "CUDA linear bias",
        );

        let reference_input = training_tensor_with_dtype(
            &[2, 4],
            vec![1.0, -2.0, 0.5, 3.0, -1.0, 2.0, 0.25, -0.75],
            DType::F32,
        );
        let reference = Linear::new_with_dtype(4, 3, DType::F32);
        install_linear_values(&reference, DType::F32);
        sum(&reference.forward(reference_input.clone())).backward();

        let tol = match dtype {
            DType::F16 => 2e-3,
            DType::BF16 => 1.5e-2,
            DType::I8 => 0.12,
            DType::F32 => unreachable!(),
        };
        assert_grad_close(&input, &reference_input, "CUDA linear input", tol);
        assert_grad_close(&linear.weight, &reference.weight, "CUDA linear weight", tol);
        assert_grad_close(
            linear.bias.as_ref().expect("linear bias"),
            reference.bias.as_ref().expect("reference bias"),
            "CUDA linear bias",
            tol,
        );
    }

    set_strict_device_execution(false);
    crate::ops::cuda::set_enabled(false);
}
