use super::*;
use crate::autograd::no_grad;
#[cfg(feature = "cuda")]
use crate::device::set_strict_device_execution;
use crate::precision::DType;
use ndarray::{Array, IxDyn};

fn check_no_grad_shape_identity(single_cat: bool) {
    let _cuda = crate::ops::cuda::set_enabled_scoped(true);
    let _strict = crate::device::set_strict_device_execution_scoped(true);
    for on_cuda in [false, true] {
        if on_cuda && !crate::ops::cuda::is_available() {
            continue;
        }
        for dtype in [DType::F32, DType::F16, DType::BF16, DType::I8] {
            let leaf = Tensor::new_with_dtype(
                Array::from_shape_vec(IxDyn(&[2, 3]), vec![-3., -2., -1., 0., 1., 2.]).unwrap(),
                dtype,
            );
            let leaf = if on_cuda { leaf.to_cuda() } else { leaf };
            for use_intermediate in [false, true] {
                let input = if use_intermediate {
                    reshape(&leaf, vec![3, 2])
                } else {
                    leaf.clone()
                };
                let shape = input.shape_vec();
                let expected_dtype = input.dtype();
                let expected = input.to_cpu().data();
                let output = no_grad(|| {
                    if single_cat {
                        cat(std::slice::from_ref(&input), 0)
                    } else {
                        slice_last_dim(&input, 0, shape[1])
                    }
                });
                assert!(
                    !output.requires_grad(),
                    "shape identity retained requires_grad"
                );
                assert!(
                    output.0.borrow().parents.is_empty(),
                    "shape identity retained parents"
                );
                assert!(output.0.borrow().backward_op.is_none());
                assert!(!Rc::ptr_eq(&output.0, &input.0));
                assert!(input.requires_grad());
                assert_eq!(output.dtype(), expected_dtype);
                assert_eq!(output.shape_vec(), shape);
                assert_eq!(output.device(), input.device());
                assert_eq!(output.data(), expected);
                leaf.zero_grad();
                output.backward();
                assert!(
                    leaf.grad().is_none(),
                    "no_grad shape output propagated to its source"
                );
            }
        }
    }
}

#[test]
fn regression_shape_no_grad_single_cat_is_untracked() {
    check_no_grad_shape_identity(true);
}

#[test]
fn regression_shape_no_grad_full_slice_is_untracked() {
    check_no_grad_shape_identity(false);
}

#[test]
fn regression_shape_empty_slice_preserves_resident_input_and_gradients() {
    let _cuda = crate::ops::cuda::set_enabled_scoped(true);
    let _strict = crate::device::set_strict_device_execution_scoped(true);
    for on_cuda in [false, true] {
        if on_cuda && !crate::ops::cuda::is_available() {
            continue;
        }
        for dtype in [DType::F32, DType::F16, DType::BF16, DType::I8] {
            for prior_gradient in [false, true] {
                let source = make_tensor(&[3, 2], vec![1.; 6], dtype);
                let source = if on_cuda { source.to_cuda() } else { source };
                let input = no_grad(|| permute(&source, vec![1, 0]));
                let scale = input.quantization_scale();
                let frozen_output = slice_last_dim(&input, 1, 1);
                assert!(!frozen_output.requires_grad());
                assert_eq!(frozen_output.dtype(), dtype);
                assert_eq!(frozen_output.quantization_scale(), scale);
                input.0.borrow_mut().requires_grad = true;
                let had_host_data = input.0.borrow().has_f32_data;
                if on_cuda {
                    assert!(!had_host_data);
                }
                if prior_gradient {
                    if on_cuda {
                        input.backward_with_cuda_grad(cuda::upload_f32(&[1.; 6]).unwrap());
                        assert!(!input.has_host_grad());
                    } else {
                        input.add_grad(Array::ones(IxDyn(&[2, 3])));
                    }
                }
                let output = slice_last_dim(&input, 1, 1);
                assert_eq!(output.shape_vec(), vec![2, 0]);
                assert!(output.requires_grad());
                assert_eq!(input.dtype(), dtype);
                assert_eq!(input.quantization_scale(), scale);
                assert!(
                    input.0.borrow().has_f32_data == had_host_data,
                    "empty slice materialized source data"
                );
                output.backward();
                if on_cuda {
                    assert!(
                        !input.has_host_grad(),
                        "zero VJP materialized resident gradients"
                    );
                    assert!(input.cloned_cuda_f32_grad().is_some());
                }
                assert_eq!(
                    input.grad().unwrap(),
                    Array::from_elem(IxDyn(&[2, 3]), if prior_gradient { 1. } else { 0. })
                );
            }
        }
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_cat_with_empty_inputs_keeps_inference_resident() {
    if !crate::ops::cuda::is_available() {
        return;
    }
    let _cuda = crate::ops::cuda::set_enabled_scoped(true);
    let _strict = crate::device::set_strict_device_execution_scoped(true);
    for dtype in [DType::F32, DType::F16, DType::BF16, DType::I8] {
        let full = make_tensor(&[2, 1, 3], vec![-2., -1., 0., 1., 2., 3.], dtype).to_cuda();
        let empty = make_tensor(&[2, 0, 3], vec![], dtype).to_cuda();
        // Equal I8 scales must preserve I8, including empty inputs with no buffer.
        if dtype == DType::I8 {
            empty.0.borrow_mut().i8_scale = full.quantization_scale();
        }
        for inputs in [
            vec![empty.clone(), full.clone()],
            vec![full.clone(), empty.clone()],
            vec![
                empty.clone(),
                full.clone(),
                empty.clone(),
                full.clone(),
                empty.clone(),
            ],
        ] {
            let cpu_inputs = inputs.iter().map(Tensor::to_cpu).collect::<Vec<_>>();
            let expected = no_grad(|| cat(&cpu_inputs, 1));
            let output = no_grad(|| cat(&inputs, 1));
            assert!(output.is_cuda());
            assert_eq!(output.dtype(), expected.dtype());
            assert!(!output.requires_grad());
            assert!(
                !output.0.borrow().has_f32_data,
                "cat materialized host data for {dtype:?}"
            );
            assert_eq!(output.shape_vec(), expected.shape_vec());
            assert_eq!(*output.data_ref(), *expected.data_ref());
        }
    }
}

#[test]
fn cat_empty_inputs_preserve_dtype_promotion() {
    let _cuda = crate::ops::cuda::set_enabled_scoped(true);
    let _strict = crate::device::set_strict_device_execution_scoped(true);
    for on_cuda in [false, true] {
        if on_cuda && !crate::ops::cuda::is_available() {
            continue;
        }
        for matching_scale in [false, true] {
            let empty = Tensor::from_i8_data_no_grad(
                Array::zeros(IxDyn(&[2, 0])).into_shared(),
                if matching_scale { 0.25 } else { 0.5 },
            );
            let full = Tensor::from_i8_data_no_grad(
                Array::from_shape_vec(IxDyn(&[2, 2]), vec![-4, -1, 2, 6])
                    .unwrap()
                    .into_shared(),
                0.25,
            );
            let inputs = if on_cuda {
                vec![empty.to_cuda(), full.to_cuda()]
            } else {
                vec![empty, full]
            };
            let output = cat(&inputs, 1);
            assert_eq!(
                output.dtype(),
                if matching_scale {
                    DType::I8
                } else {
                    DType::F32
                }
            );
            assert!(!output.requires_grad());
            assert_eq!(output.shape_vec(), vec![2, 2]);
            assert_eq!(
                output.data_ref().iter().copied().collect::<Vec<_>>(),
                vec![-1., -0.25, 0.5, 1.5]
            );
        }
        let empty = make_tensor(&[2, 0], vec![], DType::F16);
        let full = make_tensor(&[2, 2], vec![-1., 0., 1., 2.], DType::BF16);
        let inputs = if on_cuda {
            vec![empty.to_cuda(), full.to_cuda()]
        } else {
            vec![empty, full]
        };
        let output = cat(&inputs, 1);
        assert_eq!(output.dtype(), DType::F32);
        assert!(!output.requires_grad());
        assert_eq!(
            output.data_ref().iter().copied().collect::<Vec<_>>(),
            vec![-1., 0., 1., 2.]
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_cat_with_empty_inputs_preserves_training_gradients() {
    if !crate::ops::cuda::is_available() {
        return;
    }
    let _cuda = crate::ops::cuda::set_enabled_scoped(true);
    let _strict = crate::device::set_strict_device_execution_scoped(true);
    for strict in [false, true] {
        let _strict = crate::device::set_strict_device_execution_scoped(strict);
        for dtype in [DType::F32, DType::F16, DType::BF16, DType::I8] {
            let empty = Tensor::new_with_dtype(Array::zeros(IxDyn(&[2, 0, 3])), dtype).to_cuda();
            let full = Tensor::new_with_dtype(Array::ones(IxDyn(&[2, 1, 3])), dtype).to_cuda();
            let frozen = make_tensor(&[2, 1, 3], vec![2.; 6], dtype).to_cuda();
            let output = cat(
                &[
                    empty.clone(),
                    full.clone(),
                    empty.clone(),
                    frozen.clone(),
                    full.clone(),
                ],
                1,
            );
            let coefficients =
                Array::from_shape_vec(IxDyn(&[2, 3, 3]), (1..=18).map(|v| v as f32).collect())
                    .unwrap();
            let weights = Tensor::from_array_no_grad(coefficients.clone()).to_cuda();
            let loss = crate::ops::arithmetic::sum(&(&output * &weights));
            loss.backward();
            assert_eq!(empty.grad().unwrap().shape(), &[2, 0, 3]);
            assert!(frozen.grad().is_none());
            assert!(full.cloned_cuda_f32_grad().is_some());
            let expected = &coefficients.slice_axis(Axis(1), Slice::from(0..1))
                + &coefficients.slice_axis(Axis(1), Slice::from(2..3));
            assert_eq!(full.grad().unwrap(), expected);
            // A trainable empty input still creates a graph even when every
            // nonempty input is frozen.
            empty.zero_grad();
            let output = cat(&[empty.clone(), frozen.clone()], 1);
            assert!(output.requires_grad());
            output.backward();
            assert_eq!(empty.grad().unwrap().shape(), &[2, 0, 3]);
            assert!(frozen.grad().is_none());
        }
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_empty_shape_training_preserves_gradient_shapes() {
    if !crate::ops::cuda::is_available() {
        return;
    }
    let _cuda = crate::ops::cuda::set_enabled_scoped(true);
    let _strict = crate::device::set_strict_device_execution_scoped(true);
    for shape in [[0, 3, 4], [2, 0, 4], [2, 3, 0]] {
        for dtype in [DType::F32, DType::F16, DType::BF16, DType::I8] {
            let input = Tensor::new_with_dtype(Array::zeros(IxDyn(&shape)), dtype).to_cuda();
            let rhs = Tensor::new_with_dtype(Array::zeros(IxDyn(&shape)), dtype).to_cuda();
            for route in 0..4 {
                input.zero_grad();
                rhs.zero_grad();
                let output = match route {
                    0 => reshape(&input, shape.iter().map(|&n| n as i32).collect()),
                    1 => permute(&input, vec![2, 0, 1]),
                    2 => cat(&[input.clone(), rhs.clone()], 0),
                    _ => slice_last_dim(&input, 0, shape[2]),
                };
                assert!(output.is_cuda());
                assert!(output.is_empty());
                assert!(output.requires_grad());
                output.backward();
                assert_eq!(input.grad().unwrap().shape(), &shape);
                if route == 2 {
                    assert_eq!(rhs.grad().unwrap().shape(), &shape);
                }
            }
            let frozen = Tensor::from_array_no_grad(Array::zeros(IxDyn(&shape))).to_cuda();
            let output = slice_last_dim(&frozen, 0, shape[2]);
            assert!(output.is_cuda());
            assert!(!output.requires_grad());
        }
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_zero_width_slice_backward_preserves_existing_gradients() {
    if !crate::ops::cuda::is_available() {
        return;
    }
    let _cuda = crate::ops::cuda::set_enabled_scoped(true);
    let _strict = crate::device::set_strict_device_execution_scoped(true);
    let input = Tensor::new(Array::ones(IxDyn(&[2, 3]))).to_cuda();
    input.add_grad(Array::ones(IxDyn(&[2, 3])));
    let output = slice_last_dim(&input, 1, 1);
    assert!(output.is_cuda());
    assert_eq!(output.shape_vec(), vec![2, 0]);
    output.backward();
    assert_eq!(input.grad().unwrap(), Array::<f32, _>::ones(IxDyn(&[2, 3])));
}

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
fn reshape_supports_single_inferred_dimension() {
    let input = make_tensor(&[2, 3, 4], (0..24).map(|v| v as f32).collect(), DType::F32);
    let out = no_grad(|| reshape(&input, vec![2, -1]));
    assert_eq!(out.shape_vec(), vec![2, 12]);
}

#[test]
#[should_panic(expected = "only supports one inferred dimension")]
fn reshape_rejects_multiple_inferred_dimensions() {
    let input = make_tensor(&[2, 3, 4], (0..24).map(|v| v as f32).collect(), DType::F32);
    no_grad(|| {
        let _ = reshape(&input, vec![-1, -1]);
    });
}

#[test]
#[should_panic(expected = "must be >= -1")]
fn reshape_rejects_invalid_negative_dimension() {
    let input = make_tensor(&[2, 3, 4], (0..24).map(|v| v as f32).collect(), DType::F32);
    no_grad(|| {
        let _ = reshape(&input, vec![2, -2, 6]);
    });
}

#[test]
#[should_panic(expected = "dimension product overflow")]
fn reshape_rejects_dimension_product_overflow() {
    let input = make_tensor(&[1], vec![0.0], DType::F32);
    no_grad(|| {
        let _ = reshape(&input, vec![i32::MAX, i32::MAX, i32::MAX]);
    });
}

#[test]
#[should_panic(expected = "Concat axis length overflow")]
fn cat_output_shape_rejects_axis_length_overflow() {
    let _ = cat_output_shape(&[vec![usize::MAX], vec![1]], 0);
}

#[test]
#[should_panic(expected = "slice_last_dim end out of bounds")]
fn slice_last_dim_rejects_out_of_bounds_end() {
    let input = make_tensor(&[2, 3], (0..6).map(|v| v as f32).collect(), DType::F32);
    no_grad(|| {
        let _ = slice_last_dim(&input, 0, 4);
    });
}

#[test]
fn bf16_shape_ops_preserve_native_dtype_in_no_grad() {
    let input = make_tensor(
        &[2, 3, 4],
        (0..24).map(|v| v as f32 * 0.25).collect(),
        DType::BF16,
    );

    let reshaped = no_grad(|| reshape(&input, vec![2, -1]));
    let permuted = no_grad(|| permute(&input, vec![2, 0, 1]));
    let sliced = no_grad(|| slice_last_dim(&input, 1, 3));

    assert_eq!(reshaped.dtype(), DType::BF16);
    assert_eq!(permuted.dtype(), DType::BF16);
    assert_eq!(sliced.dtype(), DType::BF16);
}

#[test]
fn bf16_cat_preserves_native_dtype_in_no_grad() {
    let lhs = make_tensor(&[1, 2], vec![0.0, 1.0], DType::BF16);
    let rhs = make_tensor(&[1, 2], vec![2.0, 3.0], DType::BF16);

    let out = no_grad(|| cat(&[lhs, rhs], 0));
    assert_eq!(out.shape_vec(), vec![2, 2]);
    assert_eq!(out.dtype(), DType::BF16);
}

#[test]
fn i8_shape_ops_preserve_native_dtype_in_no_grad() {
    let input = make_tensor(
        &[2, 3, 4],
        (0..24).map(|v| v as f32 * 0.125 - 1.0).collect(),
        DType::I8,
    );

    let reshaped = no_grad(|| reshape(&input, vec![2, -1]));
    let permuted = no_grad(|| permute(&input, vec![2, 0, 1]));
    let sliced = no_grad(|| slice_last_dim(&input, 1, 3));
    let cat_out = no_grad(|| cat(&[input.clone(), input.clone()], 0));

    assert_eq!(reshaped.dtype(), DType::I8);
    assert_eq!(permuted.dtype(), DType::I8);
    assert_eq!(sliced.dtype(), DType::I8);
    assert_eq!(cat_out.dtype(), DType::I8);

    match reshaped.native_storage_owned() {
        TensorStorageOwned::I8(_, scale) => assert!(scale > 0.0),
        TensorStorageOwned::F32(_) | TensorStorageOwned::F16(_) | TensorStorageOwned::BF16(_) => {
            panic!("reshape should keep i8 storage")
        }
    }
}

#[test]
#[should_panic(expected = "Permute axes must be unique")]
fn permute_rejects_duplicate_axes() {
    let input = make_tensor(&[2, 3, 4], (0..24).map(|v| v as f32).collect(), DType::F32);
    no_grad(|| {
        let _ = permute(&input, vec![0, 1, 1]);
    });
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_reshape_reuses_resident_buffer() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let input = make_tensor(&[2, 4], (0..8).map(|v| v as f32).collect(), DType::F32).to_cuda();
    let before_handle = {
        let inner = input.0.borrow();
        inner
            .cuda_f32_data
            .as_ref()
            .expect("cuda tensor should have resident buffer")
            .handle()
    };

    let reshaped = no_grad(|| reshape(&input, vec![4, 2]));
    assert!(reshaped.is_cuda());
    let after_handle = {
        let inner = reshaped.0.borrow();
        inner
            .cuda_f32_data
            .as_ref()
            .expect("reshaped cuda tensor should keep resident buffer")
            .handle()
    };
    assert_eq!(before_handle, after_handle);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_reshape_reuses_native_bf16_buffer_in_strict_mode() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let input = make_tensor(
        &[2, 4],
        (0..8).map(|v| v as f32 * 0.25).collect(),
        DType::BF16,
    )
    .to_cuda();
    let before_handle = {
        let inner = input.0.borrow();
        assert!(inner.cuda_f32_data.is_none());
        inner
            .cuda_bf16_data
            .as_ref()
            .expect("cuda bf16 tensor should have resident native buffer")
            .handle()
    };

    crate::device::set_strict_device_execution(true);
    let reshaped = no_grad(|| reshape(&input, vec![4, 2]));
    crate::device::set_strict_device_execution(false);

    assert!(reshaped.is_cuda());
    assert_eq!(reshaped.dtype(), DType::BF16);
    assert_eq!(reshaped.shape_vec(), vec![4, 2]);
    let after_handle = {
        let inner = reshaped.0.borrow();
        assert!(inner.cuda_f32_data.is_none());
        inner
            .cuda_bf16_data
            .as_ref()
            .expect("reshaped cuda bf16 tensor should keep native buffer")
            .handle()
    };
    assert_eq!(before_handle, after_handle);
    let cpu = no_grad(|| reshape(&input.to_cpu(), vec![4, 2]));
    for (got, expect) in reshaped.data_ref().iter().zip(cpu.data_ref().iter()) {
        assert!((got - expect).abs() < 2e-2, "got {got}, expect {expect}");
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_reshape_backward_keeps_grad_resident_in_strict_mode() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let input_cpu = Tensor::from_data_with_grad_flag(
        Array::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, -2.0, 0.5, 3.0, -1.0, 2.0])
            .unwrap()
            .into_dyn(),
        true,
    );
    let coeff_cpu = Tensor::from_data_with_grad_flag(
        Array::from_shape_vec(IxDyn(&[3, 2]), vec![0.5, -1.0, 2.0, 0.25, -0.75, 1.5])
            .unwrap()
            .into_dyn(),
        false,
    );
    let input_cuda = input_cpu.to_cuda();
    let coeff_cuda = coeff_cpu.to_cuda();

    crate::ops::cuda::set_enabled(true);
    set_strict_device_execution(true);
    let cuda_out = reshape(&input_cuda, vec![3, 2]);
    let loss_cuda = crate::ops::arithmetic::sum(&(&cuda_out * &coeff_cuda));
    loss_cuda.backward();
    assert!(input_cuda.cloned_cuda_f32_grad().is_some());
    assert!(!input_cuda.has_host_grad());
    set_strict_device_execution(false);
    crate::ops::cuda::set_enabled(false);

    let cpu_out = reshape(&input_cpu, vec![3, 2]);
    let loss_cpu = crate::ops::arithmetic::sum(&(&cpu_out * &coeff_cpu));
    loss_cpu.backward();

    let cuda_grad = input_cuda.grad().expect("cuda reshape grad");
    let cpu_grad = input_cpu.grad().expect("cpu reshape grad");
    for (got, expect) in cuda_grad.iter().zip(cpu_grad.iter()) {
        assert!((got - expect).abs() < 1e-5, "got {got}, expect {expect}");
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_permute_matches_cpu_reference_in_strict_mode() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let input = make_tensor(
        &[2, 3, 4],
        (0..24).map(|v| v as f32 * 0.5).collect(),
        DType::BF16,
    )
    .to_cuda();
    assert!(input.0.borrow().cuda_f32_data.is_none());
    assert!(input.0.borrow().cuda_bf16_data.is_some());

    crate::device::set_strict_device_execution(true);
    let cuda_out = no_grad(|| permute(&input, vec![2, 0, 1]));
    crate::device::set_strict_device_execution(false);

    let cpu_out = no_grad(|| permute(&input.to_cpu(), vec![2, 0, 1]));
    assert!(cuda_out.is_cuda());
    assert_eq!(cuda_out.dtype(), DType::BF16);
    assert!(cuda_out.0.borrow().cuda_f32_data.is_none());
    assert!(cuda_out.0.borrow().cuda_bf16_data.is_some());
    assert_eq!(cuda_out.shape_vec(), cpu_out.shape_vec());
    for (got, expect) in cuda_out.data_ref().iter().zip(cpu_out.data_ref().iter()) {
        assert!((got - expect).abs() < 2e-2, "got {got}, expect {expect}");
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_permute_backward_matches_cpu_reference_in_strict_mode() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let input_cpu = Tensor::from_data_with_grad_flag(
        Array::from_shape_vec(
            IxDyn(&[2, 3, 4]),
            (0..24).map(|v| v as f32 * 0.25 - 2.0).collect(),
        )
        .unwrap()
        .into_dyn(),
        true,
    );
    let coeff_cpu = Tensor::from_data_with_grad_flag(
        Array::from_shape_vec(
            IxDyn(&[4, 2, 3]),
            (0..24).map(|v| (v as f32 % 7.0) * 0.2 - 0.5).collect(),
        )
        .unwrap()
        .into_dyn(),
        false,
    );
    let input_cuda = input_cpu.to_cuda();
    let coeff_cuda = coeff_cpu.to_cuda();

    crate::ops::cuda::set_enabled(true);
    set_strict_device_execution(true);
    let cuda_out = permute(&input_cuda, vec![2, 0, 1]);
    let loss_cuda = crate::ops::arithmetic::sum(&(&cuda_out * &coeff_cuda));
    loss_cuda.backward();
    assert!(input_cuda.cloned_cuda_f32_grad().is_some());
    assert!(!input_cuda.has_host_grad());
    set_strict_device_execution(false);
    crate::ops::cuda::set_enabled(false);

    let cpu_out = permute(&input_cpu, vec![2, 0, 1]);
    let loss_cpu = crate::ops::arithmetic::sum(&(&cpu_out * &coeff_cpu));
    loss_cpu.backward();

    let cuda_grad = input_cuda.grad().expect("cuda permute grad");
    let cpu_grad = input_cpu.grad().expect("cpu permute grad");
    for (got, expect) in cuda_grad.iter().zip(cpu_grad.iter()) {
        assert!((got - expect).abs() < 1e-5, "got {got}, expect {expect}");
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_slice_last_dim_matches_cpu_reference_in_strict_mode() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let input = make_tensor(
        &[2, 3, 5],
        (0..30).map(|v| v as f32 * 0.25 - 1.0).collect(),
        DType::BF16,
    )
    .to_cuda();
    assert!(input.0.borrow().cuda_f32_data.is_none());
    assert!(input.0.borrow().cuda_bf16_data.is_some());

    crate::device::set_strict_device_execution(true);
    let cuda_out = no_grad(|| slice_last_dim(&input, 1, 4));
    crate::device::set_strict_device_execution(false);

    let cpu_out = no_grad(|| slice_last_dim(&input.to_cpu(), 1, 4));
    assert!(cuda_out.is_cuda());
    assert_eq!(cuda_out.dtype(), DType::BF16);
    assert!(cuda_out.0.borrow().cuda_f32_data.is_none());
    assert!(cuda_out.0.borrow().cuda_bf16_data.is_some());
    assert_eq!(cuda_out.shape_vec(), cpu_out.shape_vec());
    for (got, expect) in cuda_out.data_ref().iter().zip(cpu_out.data_ref().iter()) {
        assert!((got - expect).abs() < 2e-2, "got {got}, expect {expect}");
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_slice_last_dim_empty_no_grad_stays_on_cuda_in_strict_mode() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let input = make_tensor(
        &[2, 3, 5],
        (0..30).map(|v| v as f32 * 0.25 - 1.0).collect(),
        DType::BF16,
    )
    .to_cuda();

    crate::device::set_strict_device_execution(true);
    let out = no_grad(|| slice_last_dim(&input, 2, 2));
    crate::device::set_strict_device_execution(false);

    assert!(out.is_cuda());
    assert_eq!(out.dtype(), DType::BF16);
    assert_eq!(out.shape_vec(), vec![2, 3, 0]);
    assert_eq!(out.len(), 0);
    assert_eq!(out.data_ref().len(), 0);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_empty_shape_ops_chain_stays_on_cuda_in_strict_mode() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let lhs = make_tensor(
        &[2, 3, 5],
        (0..30).map(|v| v as f32 * 0.25 - 1.0).collect(),
        DType::BF16,
    )
    .to_cuda();
    let rhs = make_tensor(
        &[2, 3, 5],
        (0..30).map(|v| v as f32 * -0.125 + 0.5).collect(),
        DType::BF16,
    )
    .to_cuda();

    crate::device::set_strict_device_execution(true);
    let out = no_grad(|| {
        let lhs_empty = slice_last_dim(&lhs, 2, 2);
        let rhs_empty = slice_last_dim(&rhs, 2, 2);
        let lhs_permuted = permute(&lhs_empty, vec![1, 0, 2]);
        let rhs_permuted = permute(&rhs_empty, vec![1, 0, 2]);
        let joined = cat(&[lhs_permuted, rhs_permuted], 0);
        reshape(&joined, vec![6, 2, 0])
    });
    crate::device::set_strict_device_execution(false);

    assert!(out.is_cuda());
    assert_eq!(out.dtype(), DType::BF16);
    assert_eq!(out.shape_vec(), vec![6, 2, 0]);
    assert_eq!(out.len(), 0);
    assert_eq!(out.data_ref().len(), 0);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_slice_last_dim_backward_matches_cpu_reference_in_strict_mode() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let input_cpu = Tensor::from_data_with_grad_flag(
        Array::from_shape_vec(
            IxDyn(&[2, 3, 5]),
            (0..30).map(|v| v as f32 * 0.1 - 1.0).collect(),
        )
        .unwrap()
        .into_dyn(),
        true,
    );
    let coeff_cpu = Tensor::from_data_with_grad_flag(
        Array::from_shape_vec(
            IxDyn(&[2, 3, 3]),
            (0..18).map(|v| (v as f32 % 5.0) * 0.3 - 0.4).collect(),
        )
        .unwrap()
        .into_dyn(),
        false,
    );
    let input_cuda = input_cpu.to_cuda();
    let coeff_cuda = coeff_cpu.to_cuda();

    crate::ops::cuda::set_enabled(true);
    set_strict_device_execution(true);
    let cuda_out = slice_last_dim(&input_cuda, 1, 4);
    let loss_cuda = crate::ops::arithmetic::sum(&(&cuda_out * &coeff_cuda));
    loss_cuda.backward();
    assert!(input_cuda.cloned_cuda_f32_grad().is_some());
    assert!(!input_cuda.has_host_grad());
    set_strict_device_execution(false);
    crate::ops::cuda::set_enabled(false);

    let cpu_out = slice_last_dim(&input_cpu, 1, 4);
    let loss_cpu = crate::ops::arithmetic::sum(&(&cpu_out * &coeff_cpu));
    loss_cpu.backward();

    let cuda_grad = input_cuda.grad().expect("cuda slice grad");
    let cpu_grad = input_cpu.grad().expect("cpu slice grad");
    for (got, expect) in cuda_grad.iter().zip(cpu_grad.iter()) {
        assert!((got - expect).abs() < 1e-5, "got {got}, expect {expect}");
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_cat_matches_cpu_reference_in_strict_mode() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let lhs = make_tensor(
        &[2, 1, 3],
        vec![0.0, 1.0, 2.0, -1.0, -2.0, -3.0],
        DType::BF16,
    )
    .to_cuda();
    let rhs = make_tensor(
        &[2, 2, 3],
        vec![
            3.0, 4.0, 5.0, 6.0, 7.0, 8.0, -4.0, -5.0, -6.0, 1.5, 2.5, 3.5,
        ],
        DType::BF16,
    )
    .to_cuda();
    assert!(lhs.0.borrow().cuda_f32_data.is_none());
    assert!(rhs.0.borrow().cuda_f32_data.is_none());
    assert!(lhs.0.borrow().cuda_bf16_data.is_some());
    assert!(rhs.0.borrow().cuda_bf16_data.is_some());

    set_strict_device_execution(true);
    let cuda_out = no_grad(|| cat(&[lhs.clone(), rhs.clone()], 1));
    set_strict_device_execution(false);

    let cpu_out = no_grad(|| cat(&[lhs.to_cpu(), rhs.to_cpu()], 1));
    assert!(cuda_out.is_cuda());
    assert_eq!(cuda_out.dtype(), DType::BF16);
    assert!(cuda_out.0.borrow().cuda_f32_data.is_none());
    assert!(cuda_out.0.borrow().cuda_bf16_data.is_some());
    assert_eq!(cuda_out.shape_vec(), cpu_out.shape_vec());
    for (got, expect) in cuda_out.data_ref().iter().zip(cpu_out.data_ref().iter()) {
        assert!((got - expect).abs() < 2e-2, "got {got}, expect {expect}");
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_cat_backward_matches_cpu_reference_in_strict_mode() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let lhs_cpu = Tensor::from_data_with_grad_flag(
        Array::from_shape_vec(IxDyn(&[2, 1, 3]), vec![0.0, 1.0, 2.0, -1.0, -2.0, -3.0])
            .unwrap()
            .into_dyn(),
        true,
    );
    let rhs_cpu = Tensor::from_data_with_grad_flag(
        Array::from_shape_vec(
            IxDyn(&[2, 2, 3]),
            vec![
                3.0, 4.0, 5.0, 6.0, 7.0, 8.0, -4.0, -5.0, -6.0, 1.5, 2.5, 3.5,
            ],
        )
        .unwrap()
        .into_dyn(),
        true,
    );
    let coeff_cpu = Tensor::from_data_with_grad_flag(
        Array::from_shape_vec(
            IxDyn(&[2, 3, 3]),
            (0..18).map(|v| (v as f32 % 7.0) * 0.25 - 0.75).collect(),
        )
        .unwrap()
        .into_dyn(),
        false,
    );
    let lhs_cuda = lhs_cpu.to_cuda();
    let rhs_cuda = rhs_cpu.to_cuda();
    let coeff_cuda = coeff_cpu.to_cuda();

    crate::ops::cuda::set_enabled(true);
    set_strict_device_execution(true);
    let cuda_out = cat(&[lhs_cuda.clone(), rhs_cuda.clone()], 1);
    let loss_cuda = crate::ops::arithmetic::sum(&(&cuda_out * &coeff_cuda));
    loss_cuda.backward();
    assert!(lhs_cuda.cloned_cuda_f32_grad().is_some());
    assert!(rhs_cuda.cloned_cuda_f32_grad().is_some());
    assert!(!lhs_cuda.has_host_grad());
    assert!(!rhs_cuda.has_host_grad());
    set_strict_device_execution(false);
    crate::ops::cuda::set_enabled(false);

    let cpu_out = cat(&[lhs_cpu.clone(), rhs_cpu.clone()], 1);
    let loss_cpu = crate::ops::arithmetic::sum(&(&cpu_out * &coeff_cpu));
    loss_cpu.backward();

    let lhs_cuda_grad = lhs_cuda.grad().expect("cuda lhs cat grad");
    let lhs_cpu_grad = lhs_cpu.grad().expect("cpu lhs cat grad");
    let rhs_cuda_grad = rhs_cuda.grad().expect("cuda rhs cat grad");
    let rhs_cpu_grad = rhs_cpu.grad().expect("cpu rhs cat grad");
    for (got, expect) in lhs_cuda_grad.iter().zip(lhs_cpu_grad.iter()) {
        assert!(
            (got - expect).abs() < 1e-5,
            "lhs got {got}, expect {expect}"
        );
    }
    for (got, expect) in rhs_cuda_grad.iter().zip(rhs_cpu_grad.iter()) {
        assert!(
            (got - expect).abs() < 1e-5,
            "rhs got {got}, expect {expect}"
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_cat_preserves_i8_dtype_in_strict_mode() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let lhs = make_tensor(&[1, 2], vec![-2.0, 1.0], DType::I8).to_cuda();
    let rhs = make_tensor(&[1, 2], vec![2.0, -1.0], DType::I8).to_cuda();
    assert!(lhs.0.borrow().cuda_f32_data.is_none());
    assert!(rhs.0.borrow().cuda_f32_data.is_none());
    assert!(lhs.0.borrow().cuda_i8_data.is_some());
    assert!(rhs.0.borrow().cuda_i8_data.is_some());

    set_strict_device_execution(true);
    let out = no_grad(|| cat(&[lhs, rhs], 0));
    set_strict_device_execution(false);

    assert!(out.is_cuda());
    assert_eq!(out.dtype(), DType::I8);
    assert!(out.0.borrow().cuda_f32_data.is_none());
    assert!(out.0.borrow().cuda_i8_data.is_some());
}
