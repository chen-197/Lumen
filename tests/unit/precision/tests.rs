use super::*;

#[test]
fn inference_quantization_is_explicit_nested_and_thread_local() {
    assert!(!allow_dynamic_i8_activations());
    with_inference_activation_quantization(InferenceActivationQuantization::AllowDynamicI8, || {
        assert!(allow_dynamic_i8_activations());
        std::thread::spawn(|| assert!(!allow_dynamic_i8_activations()))
            .join()
            .unwrap();
        let result = std::panic::catch_unwind(|| {
            with_inference_activation_quantization(
                InferenceActivationQuantization::Disabled,
                || {
                    assert!(!allow_dynamic_i8_activations());
                    panic!("restore policy after unwind");
                },
            );
        });
        assert!(result.is_err());
        assert!(allow_dynamic_i8_activations());
    });
    assert!(!allow_dynamic_i8_activations());
}

#[test]
fn precision_scope_restores_previous_config() {
    let _guard = precision_guard(PrecisionConfig {
        parameter_dtype: DType::F32,
        runtime_dtype: DType::F32,
        allow_parameter_dtype_copies: false,
    });

    with_precision_config(
        PrecisionConfig {
            parameter_dtype: DType::BF16,
            runtime_dtype: DType::BF16,
            allow_parameter_dtype_copies: true,
        },
        || {
            assert_eq!(precision_config().parameter_dtype, DType::BF16);
            assert_eq!(precision_config().runtime_dtype, DType::BF16);
            assert!(precision_config().allow_parameter_dtype_copies);
        },
    );

    assert_eq!(precision_config().parameter_dtype, DType::F32);
    assert_eq!(precision_config().runtime_dtype, DType::F32);
    assert!(!precision_config().allow_parameter_dtype_copies);
}

#[test]
fn precision_config_is_thread_local() {
    set_precision_config(PrecisionConfig::default());

    let scoped = PrecisionConfig {
        parameter_dtype: DType::BF16,
        runtime_dtype: DType::F16,
        allow_parameter_dtype_copies: true,
    };
    let (tx, rx) = std::sync::mpsc::channel();
    let handle = std::thread::spawn(move || {
        let _guard = precision_guard(scoped);
        tx.send(precision_config())
            .expect("send thread-local precision config");
        std::thread::sleep(std::time::Duration::from_millis(25));
        tx.send(precision_config())
            .expect("send thread-local precision config");
    });

    assert_eq!(rx.recv().expect("receive spawned precision config"), scoped);
    assert_eq!(
        precision_config(),
        PrecisionConfig::default(),
        "main thread should not inherit spawned thread's precision config"
    );
    assert_eq!(rx.recv().expect("receive spawned precision config"), scoped);
    assert_eq!(
        precision_config(),
        PrecisionConfig::default(),
        "main thread should keep its own precision config"
    );

    handle.join().expect("join precision config thread");
    set_precision_config(PrecisionConfig::default());
}

#[test]
fn nested_precision_scopes_restore_in_stack_order() {
    let _guard = precision_guard(PrecisionConfig::default());

    let outer = PrecisionConfig {
        parameter_dtype: DType::BF16,
        runtime_dtype: DType::BF16,
        allow_parameter_dtype_copies: false,
    };
    let inner = PrecisionConfig {
        parameter_dtype: DType::F32,
        runtime_dtype: DType::F32,
        allow_parameter_dtype_copies: true,
    };

    with_precision_config(outer, || {
        assert_eq!(precision_config(), outer);
        with_precision_config(inner, || {
            assert_eq!(precision_config(), inner);
        });
        assert_eq!(precision_config(), outer);
    });

    assert_eq!(precision_config(), PrecisionConfig::default());
}

#[test]
fn parameter_quantization_scope_restores_previous_setting() {
    let _precision_guard = precision_guard(PrecisionConfig {
        parameter_dtype: DType::F32,
        runtime_dtype: DType::F32,
        allow_parameter_dtype_copies: false,
    });
    let _quantization_guard = parameter_quantization_guard(ParameterQuantization::Disabled);

    with_parameter_quantization(ParameterQuantization::Int8, || {
        assert_eq!(
            default_parameter_quantization(),
            ParameterQuantization::Int8
        );
        assert!(parameter_quantization_enabled());
        assert_eq!(default_parameter_storage_dtype(), DType::I8);
    });

    assert_eq!(
        default_parameter_quantization(),
        ParameterQuantization::Disabled
    );
    assert_eq!(default_parameter_storage_dtype(), DType::F32);
}

#[test]
fn enabling_parameter_quantization_defaults_parameter_storage_to_i8() {
    let _precision_guard = precision_guard(PrecisionConfig {
        parameter_dtype: DType::F32,
        runtime_dtype: DType::F32,
        allow_parameter_dtype_copies: false,
    });
    let _quantization_guard = parameter_quantization_guard(ParameterQuantization::Disabled);

    set_parameter_quantization_enabled(true);
    assert!(parameter_quantization_enabled());
    assert_eq!(default_parameter_storage_dtype(), DType::I8);

    set_parameter_quantization_enabled(false);
    assert!(!parameter_quantization_enabled());
    assert_eq!(default_parameter_storage_dtype(), DType::F32);
}

#[test]
fn parameter_quantization_can_store_manual_scale() {
    let _precision_guard = precision_guard(PrecisionConfig {
        parameter_dtype: DType::F32,
        runtime_dtype: DType::F32,
        allow_parameter_dtype_copies: false,
    });
    let _quantization_guard = parameter_quantization_guard(ParameterQuantization::Disabled);
    let quantization = ParameterQuantization::Int8.with_scale(0.25);
    set_default_parameter_quantization(quantization);

    assert_eq!(default_parameter_quantization(), quantization);
    assert_eq!(default_parameter_quantization_dtype(), Some(DType::I8));
    assert_eq!(default_parameter_quantization_scale(), Some(0.25));
}

#[test]
fn activation_and_kv_cache_defaults_follow_runtime_until_overridden() {
    let _precision_guard = precision_guard(PrecisionConfig {
        parameter_dtype: DType::F32,
        runtime_dtype: DType::F32,
        allow_parameter_dtype_copies: false,
    });
    reset_default_activation_dtype_to_runtime();
    reset_default_kv_cache_dtype_to_runtime();

    assert_eq!(default_activation_dtype(), DType::F32);
    assert_eq!(default_kv_cache_dtype(), DType::F32);
    assert!(activation_dtype_follows_runtime());
    assert!(kv_cache_dtype_follows_runtime());

    set_default_runtime_dtype(DType::BF16);
    assert_eq!(default_runtime_dtype(), DType::BF16);
    assert_eq!(default_activation_dtype(), DType::BF16);
    assert_eq!(default_kv_cache_dtype(), DType::BF16);
}

#[test]
fn activation_and_kv_cache_defaults_can_override_runtime_independently() {
    let _precision_guard = precision_guard(PrecisionConfig {
        parameter_dtype: DType::F32,
        runtime_dtype: DType::F32,
        allow_parameter_dtype_copies: false,
    });
    reset_default_activation_dtype_to_runtime();
    reset_default_kv_cache_dtype_to_runtime();

    set_default_activation_dtype(DType::F16);
    set_default_kv_cache_dtype(DType::BF16);
    set_default_runtime_dtype(DType::F32);

    assert_eq!(default_runtime_dtype(), DType::F32);
    assert_eq!(default_activation_dtype(), DType::F16);
    assert_eq!(default_kv_cache_dtype(), DType::BF16);
    assert!(!activation_dtype_follows_runtime());
    assert!(!kv_cache_dtype_follows_runtime());

    reset_default_activation_dtype_to_runtime();
    reset_default_kv_cache_dtype_to_runtime();
    assert_eq!(default_activation_dtype(), DType::F32);
    assert_eq!(default_kv_cache_dtype(), DType::F32);
    assert!(activation_dtype_follows_runtime());
    assert!(kv_cache_dtype_follows_runtime());
}

#[test]
fn runtime_component_dtype_scope_restores_previous_defaults_and_follow_flags() {
    let _precision_guard = precision_guard(PrecisionConfig {
        parameter_dtype: DType::F32,
        runtime_dtype: DType::BF16,
        allow_parameter_dtype_copies: false,
    });
    reset_default_activation_dtype_to_runtime();
    set_default_kv_cache_dtype(DType::F16);

    with_runtime_component_dtypes(Some(DType::F32), Some(DType::BF16), || {
        assert_eq!(default_runtime_dtype(), DType::BF16);
        assert_eq!(default_activation_dtype(), DType::F32);
        assert_eq!(default_kv_cache_dtype(), DType::BF16);
        assert!(!activation_dtype_follows_runtime());
        assert!(!kv_cache_dtype_follows_runtime());
    });

    assert_eq!(default_runtime_dtype(), DType::BF16);
    assert_eq!(default_activation_dtype(), DType::BF16);
    assert_eq!(default_kv_cache_dtype(), DType::F16);
    assert!(activation_dtype_follows_runtime());
    assert!(!kv_cache_dtype_follows_runtime());
}
