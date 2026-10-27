use super::*;
use crate::precision::{
    DType, ParameterQuantization, PrecisionConfig, with_parameter_quantization,
    with_precision_config,
};

#[test]
fn manual_i8_quantization_boundary_with_subnormal_scale() {
    for scale in [
        f32::from_bits(1),
        f32::from_bits(0x10000),
        f32::MIN_POSITIVE,
    ] {
        let values = [0.0, scale, -scale, 2.0 * scale, -2.0 * scale];
        let expected = [0, 1, -1, 2, -2];
        let array = ArrayD::from_shape_vec(IxDyn(&[5]), values.to_vec()).unwrap();
        let (raw, actual_scale) = Tensor::quantize_f32_to_i8(&array, Some(scale));
        assert_eq!(actual_scale, scale);
        assert_eq!(raw.as_slice().unwrap(), expected);
        let mut output = [0; 5];
        Tensor::quantize_f32_values_into_i8_slice(&values, Some(scale), &mut output);
        assert_eq!(output, expected);
        let halves = values.map(f16::from_f32);
        Tensor::quantize_f16_values_into_i8_slice(&halves, Some(scale), &mut output);
        assert_eq!(output, [0; 5]);
        let brain = values.map(bf16::from_f32);
        Tensor::quantize_bf16_values_into_i8_slice(&brain, Some(scale), &mut output);
        let expected = brain.map(|v| (v.to_f32() as f64 / scale as f64).round() as i8);
        assert_eq!(output, expected);
    }
}

#[test]
fn i8_optimizer_quantization_boundary_keeps_zero_updates() {
    let scale = f32::from_bits(0x10000);
    for state_dtype in [DType::F32, DType::F16, DType::BF16] {
        for kind in ["sgd", "momentum", "adam"] {
            let input = Tensor::parameter_with_quantization(
                ArrayD::from_shape_vec(IxDyn(&[67]), vec![scale; 67]).unwrap(),
                ParameterQuantization::Int8.with_scale(scale),
            );
            input.add_grad(ArrayD::zeros(IxDyn(&[67])));
            use crate::optim::{Adam, Optimizer, SGD};
            let mut optimizer: Box<dyn Optimizer> = match kind {
                "sgd" => Box::new(SGD::new_with_dtype(vec![input.clone()], 0.1, state_dtype)),
                "momentum" => Box::new(
                    SGD::new_with_dtype(vec![input.clone()], 0.1, state_dtype).with_momentum(0.9),
                ),
                _ => Box::new(Adam::new_with_dtype(vec![input.clone()], 0.1, state_dtype)),
            };
            optimizer.step();
            let (_, _, raw) = input.export_raw();
            match raw {
                TensorRawData::I8 { values, .. } => {
                    assert_eq!(values, vec![1; 67], "{kind}/{state_dtype:?}")
                }
                _ => panic!("expected I8 storage"),
            }
        }
    }
}

#[test]
#[should_panic(expected = "shape element count overflow")]
fn checked_shape_len_rejects_overflow() {
    let _ = Tensor::checked_shape_len(&[usize::MAX, 2], "test tensor");
}

#[test]
fn parameter_creation_follows_precision_scope() {
    with_precision_config(
        PrecisionConfig {
            parameter_dtype: DType::BF16,
            runtime_dtype: DType::F32,
            allow_parameter_dtype_copies: false,
        },
        || {
            let param = Tensor::parameter(ArrayD::from_elem(IxDyn(&[2, 3]), 1.0));
            assert_eq!(param.dtype(), DType::BF16);
            param.with_storage_view(|view| match view {
                TensorStorageView::BF16(view) => assert_eq!(view.shape(), &[2, 3]),
                TensorStorageView::F16(_) => {
                    panic!("bf16 parameter unexpectedly exposed as f16")
                }
                TensorStorageView::F32(_) => {
                    panic!("bf16 parameter unexpectedly exposed as f32")
                }
            });
        },
    );
}

#[test]
fn no_grad_scope_is_thread_local() {
    set_inference_mode(false);

    let (tx, rx) = std::sync::mpsc::channel();
    let handle = std::thread::spawn(move || {
        let _guard = NoGradGuard::enter();
        tx.send(is_no_grad())
            .expect("send thread-local no_grad state");
        std::thread::sleep(std::time::Duration::from_millis(25));
        tx.send(is_no_grad())
            .expect("send thread-local no_grad state");
    });

    assert!(
        rx.recv().expect("receive spawned thread state"),
        "spawned thread should observe its own no_grad guard"
    );
    assert!(
        !is_no_grad(),
        "main thread should not inherit spawned thread's no_grad guard"
    );
    assert!(
        rx.recv().expect("receive spawned thread state"),
        "spawned thread should remain in no_grad until its guard drops"
    );
    assert!(
        !is_no_grad(),
        "main thread should still remain outside no_grad"
    );

    handle.join().expect("join no_grad thread");
    set_inference_mode(false);
}

#[test]
fn inference_mode_is_thread_local() {
    set_inference_mode(false);

    let (tx, rx) = std::sync::mpsc::channel();
    let handle = std::thread::spawn(move || {
        set_inference_mode(true);
        tx.send(is_inference_mode())
            .expect("send thread-local inference state");
        std::thread::sleep(std::time::Duration::from_millis(25));
        tx.send(is_inference_mode())
            .expect("send thread-local inference state");
        set_inference_mode(false);
    });

    assert!(
        rx.recv().expect("receive spawned thread state"),
        "spawned thread should observe its own inference mode"
    );
    assert!(
        !is_inference_mode(),
        "main thread should not inherit spawned thread's inference mode"
    );
    assert!(
        rx.recv().expect("receive spawned thread state"),
        "spawned thread should remain in inference mode until it resets"
    );
    assert!(
        !is_inference_mode(),
        "main thread should still remain outside inference mode"
    );

    handle.join().expect("join inference thread");
    set_inference_mode(false);
}

#[test]
fn strict_device_execution_is_thread_local() {
    set_strict_device_execution(false);

    let (tx, rx) = std::sync::mpsc::channel();
    let handle = std::thread::spawn(move || {
        set_strict_device_execution(true);
        tx.send(is_strict_device_execution())
            .expect("send thread-local strict device state");
        std::thread::sleep(std::time::Duration::from_millis(25));
        tx.send(is_strict_device_execution())
            .expect("send thread-local strict device state");
        set_strict_device_execution(false);
    });

    assert!(
        rx.recv().expect("receive spawned thread state"),
        "spawned thread should observe its own strict device flag"
    );
    assert!(
        !is_strict_device_execution(),
        "main thread should not inherit spawned thread's strict device flag"
    );
    assert!(
        rx.recv().expect("receive spawned thread state"),
        "spawned thread should remain strict until it resets"
    );
    assert!(
        !is_strict_device_execution(),
        "main thread should still remain outside strict device execution"
    );

    handle.join().expect("join strict device thread");
    set_strict_device_execution(false);
}

#[test]
fn parameter_storage_view_prefers_f32_cache_when_copies_allowed() {
    with_precision_config(
        PrecisionConfig {
            parameter_dtype: DType::BF16,
            runtime_dtype: DType::F32,
            allow_parameter_dtype_copies: true,
        },
        || {
            let param = Tensor::parameter(ArrayD::from_elem(IxDyn(&[4]), 2.0));
            assert_eq!(param.dtype(), DType::BF16);
            param.with_storage_view(|view| match view {
                TensorStorageView::F32(view) => assert_eq!(view.len(), 4),
                TensorStorageView::F16(_) => {
                    panic!("bf16 parameter should expose cached f32 view")
                }
                TensorStorageView::BF16(_) => {
                    panic!("bf16 parameter should expose cached f32 view")
                }
            });
            assert_eq!(param.dtype(), DType::BF16, "storage dtype should stay bf16");
        },
    );
}

#[test]
fn native_storage_view_preserves_bf16_parameter_storage() {
    with_precision_config(
        PrecisionConfig {
            parameter_dtype: DType::BF16,
            runtime_dtype: DType::F32,
            allow_parameter_dtype_copies: true,
        },
        || {
            let param = Tensor::parameter(ArrayD::from_elem(IxDyn(&[4]), 3.0));
            param.with_storage_view_preferring(StoragePreference::Native, |view| match view {
                TensorStorageView::BF16(view) => assert_eq!(view.len(), 4),
                TensorStorageView::F16(_) => {
                    panic!("native preference should keep bf16 storage")
                }
                TensorStorageView::F32(_) => {
                    panic!("native preference should keep bf16 storage")
                }
            });
        },
    );
}

#[test]
fn input_dtype_dispatch_prefers_native_same_dtype_parameter_view() {
    with_precision_config(
        PrecisionConfig {
            parameter_dtype: DType::BF16,
            runtime_dtype: DType::F32,
            allow_parameter_dtype_copies: true,
        },
        || {
            let param = Tensor::parameter(ArrayD::from_elem(IxDyn(&[4]), 3.0));

            param.with_storage_view_for_input_dtype(DType::BF16, |view| match view {
                TensorStorageView::BF16(view) => assert_eq!(view.len(), 4),
                TensorStorageView::F16(_) => {
                    panic!("same-dtype bf16 input should keep native bf16 parameter view")
                }
                TensorStorageView::F32(_) => {
                    panic!("same-dtype bf16 input should not be hijacked by cached f32 view")
                }
            });
        },
    );
}

#[test]
fn input_dtype_dispatch_keeps_cached_f32_view_for_mixed_parameter_use() {
    with_precision_config(
        PrecisionConfig {
            parameter_dtype: DType::BF16,
            runtime_dtype: DType::F32,
            allow_parameter_dtype_copies: true,
        },
        || {
            let param = Tensor::parameter(ArrayD::from_elem(IxDyn(&[4]), 3.0));

            param.with_storage_view_for_input_dtype(DType::F32, |view| {
                #[cfg(all(feature = "arm64-fp-kernels", target_arch = "aarch64"))]
                match view {
                    TensorStorageView::BF16(view) => assert_eq!(view.len(), 4),
                    TensorStorageView::F16(_) => {
                        panic!("mixed f32 generic-dispatch on arm should keep native bf16 parameter view")
                    }
                    TensorStorageView::F32(_) => {
                        panic!("mixed f32 generic-dispatch on arm should keep native bf16 parameter view")
                    }
                }

                #[cfg(not(all(feature = "arm64-fp-kernels", target_arch = "aarch64")))]
                match view {
                    TensorStorageView::F32(view) => assert_eq!(view.len(), 4),
                    TensorStorageView::F16(_) => panic!(
                        "mixed f32 input should still be allowed to use cached f32 parameter view"
                    ),
                    TensorStorageView::BF16(_) => panic!(
                        "mixed f32 input should still be allowed to use cached f32 parameter view"
                    ),
                }
            });
        },
    );
}

#[test]
fn generic_matmul_route_prefers_native_low_precision_parameter_view_on_arm() {
    with_precision_config(
        PrecisionConfig {
            parameter_dtype: DType::BF16,
            runtime_dtype: DType::F32,
            allow_parameter_dtype_copies: true,
        },
        || {
            let param = Tensor::parameter(ArrayD::from_elem(IxDyn(&[4]), 3.0));

            param.with_storage_view_for_input_dtype_and_route(
                DType::F32,
                KernelRouteClass::GenericMatmul,
                |view| {
                    #[cfg(all(feature = "arm64-fp-kernels", target_arch = "aarch64"))]
                    match view {
                        TensorStorageView::BF16(view) => assert_eq!(view.len(), 4),
                        TensorStorageView::F16(_) => {
                            panic!("generic matmul on arm should keep native bf16 parameter view")
                        }
                        TensorStorageView::F32(_) => {
                            panic!("generic matmul on arm should keep native bf16 parameter view")
                        }
                    }

                    #[cfg(not(all(feature = "arm64-fp-kernels", target_arch = "aarch64")))]
                    match view {
                        TensorStorageView::F32(view) => assert_eq!(view.len(), 4),
                        TensorStorageView::F16(_) => {
                            panic!("generic matmul should still be allowed to use cached f32 parameter view")
                        }
                        TensorStorageView::BF16(_) => {
                            panic!("generic matmul should still be allowed to use cached f32 parameter view")
                        }
                    }
                },
            );
        },
    );
}

#[test]
fn argmax_route_prefers_native_same_dtype_parameter_view() {
    with_precision_config(
        PrecisionConfig {
            parameter_dtype: DType::BF16,
            runtime_dtype: DType::F32,
            allow_parameter_dtype_copies: true,
        },
        || {
            let param = Tensor::parameter(ArrayD::from_elem(IxDyn(&[4]), 3.0));

            param.with_storage_view_for_input_dtype_and_route(
                DType::BF16,
                KernelRouteClass::Argmax,
                |view| match view {
                    TensorStorageView::BF16(view) => assert_eq!(view.len(), 4),
                    TensorStorageView::F16(_) => {
                        panic!("same-dtype argmax should keep native bf16 parameter view")
                    }
                    TensorStorageView::F32(_) => {
                        panic!("same-dtype argmax should keep native bf16 parameter view")
                    }
                },
            );
        },
    );
}

#[test]
fn decode_route_prefers_cached_f32_view_for_same_f16_parameter_use() {
    with_precision_config(
        PrecisionConfig {
            parameter_dtype: DType::F16,
            runtime_dtype: DType::F32,
            allow_parameter_dtype_copies: true,
        },
        || {
            let param = Tensor::parameter(ArrayD::from_elem(IxDyn(&[4]), 3.0));

            param.with_storage_view_for_input_dtype(DType::F16, |view| match view {
                TensorStorageView::F32(view) => assert_eq!(view.len(), 4),
                TensorStorageView::F16(_) => {
                    panic!(
                        "same-dtype f16 decode should currently prefer cached f32 parameter view"
                    )
                }
                TensorStorageView::BF16(_) => {
                    panic!(
                        "same-dtype f16 decode should currently prefer cached f32 parameter view"
                    )
                }
            });
        },
    );
}

#[test]
fn argmax_route_prefers_cached_f32_view_for_same_f16_parameter_use() {
    with_precision_config(
        PrecisionConfig {
            parameter_dtype: DType::F16,
            runtime_dtype: DType::F32,
            allow_parameter_dtype_copies: true,
        },
        || {
            let param = Tensor::parameter(ArrayD::from_elem(IxDyn(&[4]), 3.0));

            param.with_storage_view_for_input_dtype_and_route(
                DType::F16,
                KernelRouteClass::Argmax,
                |view| match view {
                    TensorStorageView::F32(view) => assert_eq!(view.len(), 4),
                    TensorStorageView::F16(_) => {
                        panic!("same-dtype f16 argmax should currently prefer cached f32 parameter view")
                    }
                    TensorStorageView::BF16(_) => {
                        panic!("same-dtype f16 argmax should currently prefer cached f32 parameter view")
                    }
                },
            );
        },
    );
}

#[test]
fn f32_compute_preference_materializes_compute_view() {
    with_precision_config(
        PrecisionConfig {
            parameter_dtype: DType::BF16,
            runtime_dtype: DType::F32,
            allow_parameter_dtype_copies: false,
        },
        || {
            let tensor = Tensor::new_with_dtype(ArrayD::from_elem(IxDyn(&[3]), 1.5), DType::BF16);
            tensor.with_storage_view_preferring(StoragePreference::F32Compute, |view| match view {
                TensorStorageView::F32(view) => assert_eq!(view.len(), 3),
                TensorStorageView::F16(_) => {
                    panic!("f32 compute preference should expose f32 view")
                }
                TensorStorageView::BF16(_) => {
                    panic!("f32 compute preference should expose f32 view")
                }
            });
            assert_eq!(
                tensor.dtype(),
                DType::BF16,
                "compute view should not mutate bf16 storage"
            );
        },
    );
}

#[test]
fn f32_compute_preference_is_reentrant() {
    with_precision_config(
        PrecisionConfig {
            parameter_dtype: DType::BF16,
            runtime_dtype: DType::F32,
            allow_parameter_dtype_copies: false,
        },
        || {
            let lhs = Tensor::new_with_dtype(ArrayD::from_elem(IxDyn(&[2]), 1.0), DType::BF16);
            let rhs = Tensor::new_with_dtype(ArrayD::from_elem(IxDyn(&[3]), 2.0), DType::BF16);
            lhs.with_storage_view_preferring(StoragePreference::F32Compute, |lhs_view| {
                match lhs_view {
                    TensorStorageView::F32(lhs_view) => {
                        assert_eq!(lhs_view.len(), 2);
                        rhs.with_storage_view_preferring(
                            StoragePreference::F32Compute,
                            |rhs_view| match rhs_view {
                                TensorStorageView::F32(rhs_view) => {
                                    assert_eq!(rhs_view.len(), 3)
                                }
                                TensorStorageView::F16(_) => {
                                    panic!("nested f32 compute preference should expose f32 view")
                                }
                                TensorStorageView::BF16(_) => {
                                    panic!("nested f32 compute preference should expose f32 view")
                                }
                            },
                        );
                    }
                    TensorStorageView::F16(_) => {
                        panic!("f32 compute preference should expose f32 view")
                    }
                    TensorStorageView::BF16(_) => {
                        panic!("f32 compute preference should expose f32 view")
                    }
                }
            });
        },
    );
}

#[test]
fn parameter_creation_supports_i8_scope() {
    with_precision_config(
        PrecisionConfig {
            parameter_dtype: DType::I8,
            runtime_dtype: DType::F32,
            allow_parameter_dtype_copies: false,
        },
        || {
            let param = Tensor::parameter(
                ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, -0.5, 0.25]).unwrap(),
            );
            assert_eq!(param.dtype(), DType::I8);
            match param.native_storage_owned() {
                TensorStorageOwned::I8(data, scale) => {
                    assert_eq!(data.len(), 3);
                    assert!(scale > 0.0);
                }
                TensorStorageOwned::F32(_)
                | TensorStorageOwned::F16(_)
                | TensorStorageOwned::BF16(_) => {
                    panic!("i8 parameter should keep native i8 storage")
                }
            }
            param.with_storage_view_preferring(StoragePreference::F32Compute, |view| match view {
                TensorStorageView::F32(view) => assert_eq!(view.len(), 3),
                TensorStorageView::F16(_) => panic!("i8 compute view should expose f32"),
                TensorStorageView::BF16(_) => panic!("i8 compute view should expose f32"),
            });
            assert_eq!(
                param.dtype(),
                DType::I8,
                "compute view should not mutate i8 storage"
            );
        },
    );
}

#[test]
fn parameter_creation_can_follow_global_quantization_setting() {
    with_precision_config(
        PrecisionConfig {
            parameter_dtype: DType::F32,
            runtime_dtype: DType::F32,
            allow_parameter_dtype_copies: false,
        },
        || {
            with_parameter_quantization(ParameterQuantization::Int8, || {
                let param = Tensor::parameter(
                    ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, -0.5, 0.25]).unwrap(),
                );
                assert_eq!(param.dtype(), DType::I8);
            });
        },
    );
}

#[test]
fn parameter_creation_can_follow_manual_quantization_scale() {
    with_precision_config(
        PrecisionConfig {
            parameter_dtype: DType::F32,
            runtime_dtype: DType::F32,
            allow_parameter_dtype_copies: false,
        },
        || {
            with_parameter_quantization(ParameterQuantization::Int8.with_scale(0.5), || {
                let param = Tensor::parameter(
                    ArrayD::from_shape_vec(IxDyn(&[4]), vec![0.9, -1.1, 1.6, -2.6]).unwrap(),
                );
                assert_eq!(param.dtype(), DType::I8);
                assert_eq!(param.quantization_scale(), Some(0.5));
                let loaded = param.data();
                let expected = [1.0f32, -1.0, 1.5, -2.5];
                for (got, want) in loaded.iter().zip(expected.iter()) {
                    assert!((got - want).abs() <= 1e-6, "got {got}, want {want}");
                }
            });
        },
    );
}

#[test]
fn set_i8_slice_with_dtype_reuses_existing_i8_placeholder_storage() {
    let tensor =
        Tensor::parameter_placeholder_with_quantization(&[2, 2], ParameterQuantization::Int8);

    let before_ptr = {
        let inner = tensor.0.borrow();
        inner
            .i8_data
            .as_ref()
            .expect("placeholder should allocate i8 storage")
            .as_slice_memory_order()
            .expect("placeholder i8 storage should be contiguous")
            .as_ptr()
    };

    tensor.set_i8_slice_with_dtype(&[2, 2], &[4, -8, 7, 9], 0.5, DType::I8);

    let inner = tensor.0.borrow();
    let after_slice = inner
        .i8_data
        .as_ref()
        .expect("tensor should retain i8 storage")
        .as_slice_memory_order()
        .expect("tensor i8 storage should remain contiguous");
    assert_eq!(before_ptr, after_slice.as_ptr());
    assert_eq!(after_slice, &[4, -8, 7, 9]);
    assert_eq!(inner.i8_scale, Some(0.5));
    assert!(!inner.has_f32_data);
}

#[test]
fn parameter_explicit_dtype_overrides_global_default() {
    with_precision_config(
        PrecisionConfig {
            parameter_dtype: DType::BF16,
            runtime_dtype: DType::F32,
            allow_parameter_dtype_copies: false,
        },
        || {
            let param = Tensor::parameter_with_dtype(
                ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, -0.5, 0.25]).unwrap(),
                DType::F32,
            );
            assert_eq!(param.dtype(), DType::F32);
            param.with_storage_view_preferring(StoragePreference::Native, |view| match view {
                TensorStorageView::F32(view) => assert_eq!(view.len(), 3),
                TensorStorageView::F16(_) => {
                    panic!("explicit parameter dtype should override global bf16 default")
                }
                TensorStorageView::BF16(_) => {
                    panic!("explicit parameter dtype should override global bf16 default")
                }
            });
        },
    );
}

#[test]
fn parameter_explicit_dtype_overrides_global_quantization() {
    with_parameter_quantization(ParameterQuantization::Int8, || {
        let param = Tensor::parameter_with_dtype(
            ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, -0.5, 0.25]).unwrap(),
            DType::F32,
        );
        assert_eq!(param.dtype(), DType::F32);
    });
}

#[test]
fn i8_export_import_preserves_dtype_and_values() {
    let src = Tensor::new_with_dtype(
        ArrayD::from_shape_vec(IxDyn(&[4]), vec![1.0, -2.0, 0.5, 3.25]).unwrap(),
        DType::I8,
    );
    let (shape, dtype, raw) = src.export_raw();
    let dst = Tensor::from_array_no_grad(ArrayD::zeros(IxDyn(&[4])));
    dst.import_raw(shape, dtype, raw)
        .expect("i8 export/import should succeed");

    assert_eq!(dst.dtype(), DType::I8);
    let src_vals = src.data();
    let dst_vals = dst.data();
    for (&lhs, &rhs) in src_vals.iter().zip(dst_vals.iter()) {
        assert!((lhs - rhs).abs() <= 1e-6, "lhs={lhs}, rhs={rhs}");
    }
}

#[test]
fn i8_quantization_rejects_non_finite_values() {
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            Tensor::new_with_dtype(
                ArrayD::from_shape_vec(IxDyn(&[1]), vec![value]).unwrap(),
                DType::I8,
            )
        }));
        assert!(result.is_err(), "I8 quantization accepted {value}");
    }
}

#[test]
fn i8_quantization_keeps_subnormal_scale_positive() {
    let tensor = Tensor::new_with_dtype(
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![f32::from_bits(1)]).unwrap(),
        DType::I8,
    );
    let (_, _, raw) = tensor.export_raw();
    match raw {
        TensorRawData::I8 { scale, .. } => assert_eq!(scale, f32::MIN_POSITIVE),
        _ => panic!("I8 tensor exported non-I8 storage"),
    }
}

#[test]
fn import_raw_returns_error_on_dtype_mismatch() {
    let dst = Tensor::from_array_no_grad(ArrayD::zeros(IxDyn(&[2])));
    let err = dst
        .import_raw(vec![2], DType::BF16, TensorRawData::F32(vec![1.0, -2.0]))
        .expect_err("dtype mismatch should return an error");
    assert!(
        err.contains("dtype BF16 with f32 data"),
        "unexpected error: {err}"
    );
}

#[cfg(feature = "cuda")]
#[test]
fn to_cuda_and_back_preserves_tensor_metadata() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let tensor = Tensor::new_with_dtype(
        ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0, -2.0, 0.5, 3.25]).unwrap(),
        DType::BF16,
    );
    let cuda_tensor = tensor.to_cuda();
    assert_eq!(cuda_tensor.device(), Device::Cuda);
    assert_eq!(cuda_tensor.dtype(), DType::BF16);
    assert_eq!(cuda_tensor.shape_vec(), vec![2, 2]);

    let cpu_tensor = cuda_tensor.to_cpu();
    assert_eq!(cpu_tensor.device(), Device::Cpu);
    assert_eq!(cpu_tensor.dtype(), DType::BF16);
    assert_eq!(cpu_tensor.shape_vec(), vec![2, 2]);
}

#[cfg(feature = "cuda")]
#[test]
fn to_cuda_materializes_resident_buffer_and_to_cpu_releases_it() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let tensor = Tensor::from_array_no_grad(
        ArrayD::from_shape_vec(IxDyn(&[4]), vec![1.0, -2.0, 0.5, 3.25]).unwrap(),
    );
    assert!(tensor.0.borrow().cuda_f32_data.is_none());

    tensor.to_cuda_inplace();
    {
        let inner = tensor.0.borrow();
        assert_eq!(inner.device, Device::Cuda);
        assert!(inner.cuda_f32_data.is_some());
    }

    tensor.to_cpu_inplace();
    {
        let inner = tensor.0.borrow();
        assert_eq!(inner.device, Device::Cpu);
        assert!(inner.cuda_f32_data.is_none());
    }
}

#[cfg(feature = "cuda")]
#[test]
fn empty_tensor_to_cuda_preserves_metadata_without_resident_buffer() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let tensor = Tensor::new_with_dtype(
        ArrayD::from_shape_vec(IxDyn(&[2, 0, 3]), Vec::<f32>::new()).unwrap(),
        DType::BF16,
    );
    let cuda_tensor = tensor.to_cuda();

    assert_eq!(cuda_tensor.device(), Device::Cuda);
    assert_eq!(cuda_tensor.dtype(), DType::BF16);
    assert_eq!(cuda_tensor.shape_vec(), vec![2, 0, 3]);
    assert_eq!(cuda_tensor.len(), 0);
    assert!(cuda_tensor.cloned_cuda_f32_buffer().is_none());
}

#[cfg(feature = "cuda")]
#[test]
fn host_mutation_invalidates_resident_cuda_buffer() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let tensor =
        Tensor::from_array_no_grad(ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 2.0]).unwrap());
    tensor.to_cuda_inplace();
    assert!(tensor.0.borrow().cuda_f32_data.is_some());

    tensor.set_raw_data(vec![2], vec![3.0, 4.0]);
    assert!(tensor.0.borrow().cuda_f32_data.is_none());

    tensor.to_cuda_inplace();
    assert!(tensor.0.borrow().cuda_f32_data.is_some());
}
