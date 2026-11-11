use super::*;
use crate::autograd::no_grad;
use crate::device::set_strict_device_execution;
#[cfg(feature = "cuda")]
use crate::loss::CrossEntropyLoss;
use crate::precision::{
    ParameterQuantization, PrecisionConfig, set_default_runtime_dtype, with_parameter_quantization,
    with_precision_config, with_runtime_component_dtypes,
};
use ndarray::ArrayD;
use ndarray::IxDyn;

fn test_config() -> LlamaConfig {
    LlamaConfig {
        vocab_size: 32,
        hidden_size: 8,
        intermediate_size: 16,
        num_hidden_layers: 1,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        rms_norm_eps: 1e-5,
        max_seq_len: 8,
        rope_theta: 10000.0,
    }
}

fn input_ids(shape: &[usize], ids: Vec<f32>) -> Tensor {
    Tensor::from_array_no_grad(ArrayD::from_shape_vec(IxDyn(shape), ids).expect("input_ids shape"))
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_activation_cast_to_f32_materializes_requested_dtype() {
    if !cuda::is_available() {
        return;
    }
    let _enabled = cuda::set_enabled_scoped(true);
    let _strict = crate::device::set_strict_device_execution_scoped(true);
    for dtype in [DType::F16, DType::BF16, DType::I8] {
        let values = ArrayD::from_shape_vec(IxDyn(&[1, 1, 4]), vec![0.5, -1.0, 1.5, 2.0]).unwrap();
        let reference =
            Tensor::from_f32_data_no_grad_with_device_dtype(values.clone(), dtype, Device::Cpu);
        let expected = reference.data_ref().iter().copied().collect::<Vec<_>>();
        let source = Tensor::from_f32_data_no_grad_with_device_dtype(values, dtype, Device::Cuda);
        let output = no_grad(|| cast_no_grad_tensor_to_dtype(source.clone(), DType::F32));
        assert_eq!(output.dtype(), DType::F32, "cast from {dtype:?}");
        assert_eq!(output.device(), Device::Cuda);
        assert_eq!(source.dtype(), dtype, "casting must preserve the source");
        assert_eq!(output.shape_vec(), vec![1, 1, 4]);
        assert_eq!(
            output.data_ref().iter().copied().collect::<Vec<_>>(),
            expected
        );
    }
}

#[cfg(feature = "cuda")]
fn one_hot_targets(rows: usize, cols: usize) -> Tensor {
    let mut data = vec![0.0; rows * cols];
    for row in 0..rows {
        data[row * cols + (row * 5 + 3) % cols] = 1.0;
    }
    Tensor::from_array_no_grad(
        ArrayD::from_shape_vec(IxDyn(&[rows, cols]), data).expect("target shape"),
    )
}

#[test]
fn llama_model_explicit_dtype_overrides_global_default() {
    with_precision_config(
        PrecisionConfig {
            parameter_dtype: DType::BF16,
            runtime_dtype: DType::F32,
            allow_parameter_dtype_copies: false,
        },
        || {
            let model = LlamaModel::new_with_dtype(test_config(), DType::F32);
            for param in model.parameters() {
                assert_eq!(param.dtype(), DType::F32);
            }
        },
    );
}

#[test]
fn llama_model_default_construction_captures_runtime_dtype_for_future_caches() {
    let config = test_config();

    with_precision_config(
        PrecisionConfig {
            parameter_dtype: DType::F32,
            runtime_dtype: DType::BF16,
            allow_parameter_dtype_copies: false,
        },
        || {
            let model = LlamaModel::new(config.clone());
            set_default_runtime_dtype(DType::F32);
            let caches = model.init_kv_caches(1);
            assert_eq!(caches[0].borrow().dtype, DType::BF16);
            for param in model.parameters() {
                assert_eq!(param.dtype(), DType::F32);
            }
        },
    );
}

#[test]
fn llama_model_explicit_dtypes_override_global_defaults() {
    let config = test_config();

    with_precision_config(
        PrecisionConfig {
            parameter_dtype: DType::BF16,
            runtime_dtype: DType::F32,
            allow_parameter_dtype_copies: false,
        },
        || {
            let model = LlamaModel::new_with_dtypes(config, DType::F32, DType::BF16);
            for param in model.parameters() {
                assert_eq!(param.dtype(), DType::F32);
            }
            let caches = model.init_kv_caches(1);
            assert_eq!(caches[0].borrow().dtype, DType::BF16);
        },
    );
}

#[test]
fn llama_model_default_construction_can_override_activation_and_kv_cache_dtypes_independently() {
    let config = test_config();

    with_precision_config(
        PrecisionConfig {
            parameter_dtype: DType::F32,
            runtime_dtype: DType::F32,
            allow_parameter_dtype_copies: false,
        },
        || {
            with_runtime_component_dtypes(Some(DType::BF16), Some(DType::F16), || {
                let model = LlamaModel::new(config.clone());
                assert_eq!(model.activation_dtype(), DType::BF16);
                assert_eq!(model.kv_cache_dtype(), DType::F16);

                let caches = model.init_kv_caches(1);
                assert_eq!(caches[0].borrow().dtype, DType::F16);
                for param in model.parameters() {
                    assert_eq!(param.dtype(), DType::F32);
                }
            });
        },
    );
}

#[test]
fn last_hidden_token_tensor_preserves_bf16_storage() {
    no_grad(|| {
        let hidden = Tensor::new_with_dtype(
            ArrayD::from_shape_vec(
                IxDyn(&[1, 3, 4]),
                vec![
                    0.1, 0.2, 0.3, 0.4, 1.0, 1.1, 1.2, 1.3, -0.5, -0.4, -0.3, -0.2,
                ],
            )
            .expect("hidden shape"),
            DType::BF16,
        );
        let last = last_hidden_token_tensor(hidden);
        assert_eq!(last.shape_vec(), vec![1, 1, 4]);
        match last.native_storage_owned() {
            TensorStorageOwned::BF16(data) => {
                let values = data.iter().map(|v| v.to_f32()).collect::<Vec<_>>();
                assert_eq!(values.len(), 4);
                assert!((values[0] + 0.5).abs() < 0.02);
                assert!((values[3] + 0.2).abs() < 0.02);
            }
            TensorStorageOwned::F32(_) => {
                panic!("last hidden token should not materialize f32")
            }
            TensorStorageOwned::F16(_) => panic!("last hidden token should stay bf16"),
            TensorStorageOwned::I8(_, _) => panic!("last hidden token should stay bf16"),
        }
    });
}

#[test]
fn llama_model_allows_i8_activation_dtype_with_float_kv_cache() {
    let model =
        LlamaModel::new_with_runtime_dtypes(test_config(), DType::I8, DType::I8, DType::BF16);
    assert_eq!(model.activation_dtype(), DType::I8);
    assert_eq!(model.kv_cache_dtype(), DType::BF16);
}

#[test]
#[should_panic(expected = "vocab_size must be > 0")]
fn llama_model_rejects_invalid_config() {
    let mut config = test_config();
    config.vocab_size = 0;
    let _ = LlamaModel::new(config);
}

#[test]
#[should_panic(expected = "inference input_ids must be [B,S]")]
fn llama_model_rejects_non_matrix_inference_ids() {
    let model = LlamaModel::new(test_config());
    let mut caches = model.init_kv_caches(1);
    no_grad(|| {
        let _ = model.forward(input_ids(&[1], vec![0.0]), &mut caches, 0);
    });
}

#[test]
#[should_panic(expected = "Llama layer 0 KV cache shape mismatch")]
fn llama_model_rejects_cache_batch_mismatch() {
    let model = LlamaModel::new(test_config());
    let mut caches = model.init_kv_caches(2);
    no_grad(|| {
        let _ = model.forward(input_ids(&[1, 1], vec![0.0]), &mut caches, 0);
    });
}

#[test]
fn forward_last_argmax_batch_matches_logits_argmax() {
    crate::ops::cuda::set_enabled(false);
    set_strict_device_execution(false);
    with_precision_config(PrecisionConfig::default(), || {
        with_runtime_component_dtypes(Some(DType::F32), Some(DType::F32), || {
            with_parameter_quantization(ParameterQuantization::Disabled, || {
                let model = LlamaModel::new_with_dtype(test_config(), DType::BF16);
                // Exact route equality requires unrounded logits. Other layers
                // remain BF16; the dedicated tie regression covers BF16 output.
                model.lm_head.weight.cast_inplace(DType::F32);
                let input = input_ids(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
                let mut logits_caches = model.init_kv_caches(2);
                let mut argmax_caches = model.init_kv_caches(2);

                let logits =
                    no_grad(|| model.forward_last_logits(input.clone(), &mut logits_caches, 0));
                let got = no_grad(|| model.forward_last_argmax_batch(input, &mut argmax_caches, 0));
                let logits_vals = logits.data_ref().iter().copied().collect::<Vec<_>>();
                let vocab = test_config().vocab_size;
                let expected = logits_vals
                    .chunks(vocab)
                    .map(|row| {
                        row.iter()
                            .enumerate()
                            .max_by(|(i, a), (j, b)| {
                                a.partial_cmp(b)
                                    .unwrap_or(std::cmp::Ordering::Equal)
                                    .then_with(|| j.cmp(i))
                            })
                            .map(|(idx, _)| idx)
                            .expect("logits row must not be empty")
                    })
                    .collect::<Vec<_>>();

                assert_eq!(got, expected);
            });
        });
    });
}

#[test]
fn bf16_rounded_logits_can_tie_while_f32_argmax_has_a_unique_winner() {
    let _cuda = cuda::set_enabled_scoped(false);
    let _strict = crate::device::set_strict_device_execution_scoped(false);
    with_precision_config(PrecisionConfig::default(), || {
        let model = LlamaModel::new_with_dtype(test_config(), DType::BF16);
        let mut weights = vec![0.; 32 * 8];
        weights[0] = 1.;
        weights[1] = 0.5;
        weights[8] = 1.;
        weights[9] = 0.25;
        model.lm_head.weight.set_array_f32_with_dtype(
            ArrayD::from_shape_vec(IxDyn(&[32, 8]), weights).unwrap(),
            DType::BF16,
        );
        let hidden = Tensor::from_f32_data_no_grad_with_dtype(
            ArrayD::from_shape_vec(
                IxDyn(&[1, 1, 8]),
                vec![1., 1. / 128., 0., 0., 0., 0., 0., 0.],
            )
            .unwrap(),
            DType::BF16,
        );
        no_grad(|| {
            let got = model.lm_head_argmax_batch_from_last_hidden(&hidden);
            let logits = model.lm_head_logits_from_last_hidden(hidden, 1, 8);
            assert_eq!(logits.dtype(), DType::BF16);
            let values = logits.data();
            assert_eq!(values[[0, 0, 0]], 1.);
            assert_eq!(values[[0, 0, 1]], 1.);
            // Exact F32 dot scores are 1 + 1/256 and 1 + 1/512.
            assert_eq!(got, [0]);
        });
    });
}

#[test]
fn forward_last_argmax_single_batch_keeps_old_return_type() {
    let model = LlamaModel::new_with_dtype(test_config(), DType::F16);
    let input = input_ids(&[1, 3], vec![1.0, 2.0, 3.0]);
    let mut batch_caches = model.init_kv_caches(1);
    let mut single_caches = model.init_kv_caches(1);

    let batch = no_grad(|| model.forward_last_argmax_batch(input.clone(), &mut batch_caches, 0));
    let single = no_grad(|| model.forward_last_argmax(input, &mut single_caches, 0));

    assert_eq!(batch, vec![single]);
}

#[cfg(feature = "cuda")]
#[test]
fn llama_forward_last_argmax_batch_matches_logits_on_cuda_in_strict_mode() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let config = test_config();
    let model = LlamaModel::new_with_dtype(config.clone(), DType::BF16);
    model.lm_head.weight.cast_inplace(DType::F32);
    model.to_cuda();
    let input = input_ids(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).to_cuda();
    let mut logits_caches = model.init_kv_caches(2);
    let mut argmax_caches = model.init_kv_caches(2);

    set_strict_device_execution(true);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        no_grad(|| {
            let logits = model.forward_last_logits(input.clone(), &mut logits_caches, 0);
            let got = model.forward_last_argmax_batch(input, &mut argmax_caches, 0);
            (logits, got)
        })
    }));
    set_strict_device_execution(false);
    let (logits, got) = result.unwrap();

    assert!(logits.is_cuda());
    let logits_vals = logits.data_ref().iter().copied().collect::<Vec<_>>();
    let expected = logits_vals
        .chunks(config.vocab_size)
        .map(|row| {
            row.iter()
                .enumerate()
                .max_by(|(i, a), (j, b)| {
                    a.partial_cmp(b)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| j.cmp(i))
                })
                .map(|(idx, _)| idx)
                .expect("logits row must not be empty")
        })
        .collect::<Vec<_>>();

    assert_eq!(got, expected);
}

#[cfg(feature = "cuda")]
#[test]
fn llama_forward_last_argmax_f16_activation_i8_lm_head_matches_cpu_logits_in_strict_mode() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let config = test_config();
    let model =
        LlamaModel::new_with_runtime_dtypes(config.clone(), DType::I8, DType::F16, DType::F16);
    let input = input_ids(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    let mut logits_caches = model.init_kv_caches(2);
    let logits = no_grad(|| model.forward_last_logits(input.clone(), &mut logits_caches, 0));
    let logits_vals = logits.data_ref().iter().copied().collect::<Vec<_>>();
    let expected = logits_vals
        .chunks(config.vocab_size)
        .map(|row| {
            row.iter()
                .enumerate()
                .max_by(|(i, a), (j, b)| {
                    a.partial_cmp(b)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| j.cmp(i))
                })
                .map(|(idx, _)| idx)
                .expect("logits row must not be empty")
        })
        .collect::<Vec<_>>();

    model.to_cuda();
    let mut argmax_caches = model.init_kv_caches(2);
    let input = input.to_cuda();

    set_strict_device_execution(true);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        no_grad(|| model.forward_last_argmax_batch(input, &mut argmax_caches, 0))
    }));
    set_strict_device_execution(false);
    let got = result.unwrap();

    assert_eq!(got, expected);
}

#[test]
#[should_panic(expected = "KV cache count mismatch")]
fn llama_model_reset_rejects_cache_count_mismatch() {
    let model = LlamaModel::new(test_config());
    let mut caches = model.init_kv_caches(1);
    caches.pop();
    model.reset_kv_caches(&mut caches);
}

#[cfg(feature = "cuda")]
#[test]
fn llama_model_init_kv_caches_follow_model_device() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let model = LlamaModel::new_with_dtype(test_config(), DType::BF16);
    model.to_cuda();
    let caches = model.init_kv_caches(1);
    assert!(caches[0].borrow().k.is_cuda());
    assert!(caches[0].borrow().v.is_cuda());
}

#[cfg(feature = "cuda")]
fn assert_llama_forward_last_logits_stays_on_cuda(dtype: DType) {
    let model = LlamaModel::new_with_dtype(test_config(), dtype);
    model.to_cuda();
    let mut caches = model.init_kv_caches(1);
    let input_cuda = input_ids(&[1, 4], vec![1.0, 2.0, 3.0, 4.0]).to_cuda();

    set_strict_device_execution(true);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        no_grad(|| model.forward_last_logits(input_cuda, &mut caches, 0))
    }));
    set_strict_device_execution(false);
    let logits = result.unwrap();

    assert!(logits.is_cuda());
    assert_eq!(logits.shape_vec(), vec![1, 1, test_config().vocab_size]);
    assert!(logits.cloned_cuda_f32_buffer().is_some());
    assert_eq!(logits.dtype(), dtype);
    assert!(caches[0].borrow().k.is_cuda());
    assert!(caches[0].borrow().v.is_cuda());
}

#[cfg(feature = "cuda")]
#[test]
fn llama_forward_last_logits_stays_on_cuda_in_strict_mode() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    for dtype in [DType::F32, DType::F16, DType::BF16] {
        assert_llama_forward_last_logits_stays_on_cuda(dtype);
    }
}

#[cfg(feature = "cuda")]
fn assert_llama_prefill_then_decode_stays_on_cuda(dtype: DType) {
    let model = LlamaModel::new_with_dtype(test_config(), dtype);
    model.to_cuda();
    let mut caches = model.init_kv_caches(1);
    let prefill_cuda = input_ids(&[1, 4], vec![1.0, 2.0, 3.0, 4.0]).to_cuda();
    let decode_cuda = input_ids(&[1, 1], vec![5.0]).to_cuda();

    set_strict_device_execution(true);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        no_grad(|| {
            let prefill_logits = model.forward_last_logits(prefill_cuda, &mut caches, 0);
            let decode_logits = model.forward_last_logits(decode_cuda, &mut caches, 4);
            (prefill_logits, decode_logits)
        })
    }));
    set_strict_device_execution(false);
    let (prefill_logits, decode_logits) = result.unwrap();

    assert!(prefill_logits.is_cuda());
    assert!(decode_logits.is_cuda());
    assert_eq!(prefill_logits.dtype(), dtype);
    assert_eq!(decode_logits.dtype(), dtype);
    assert_eq!(
        prefill_logits.shape_vec(),
        vec![1, 1, test_config().vocab_size]
    );
    assert_eq!(
        decode_logits.shape_vec(),
        vec![1, 1, test_config().vocab_size]
    );
    assert!(prefill_logits.cloned_cuda_f32_buffer().is_some());
    assert!(decode_logits.cloned_cuda_f32_buffer().is_some());
    assert!(caches[0].borrow().k.is_cuda());
    assert!(caches[0].borrow().v.is_cuda());
}

#[cfg(feature = "cuda")]
#[test]
fn llama_prefill_then_decode_stays_on_cuda_in_strict_mode() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    for dtype in [DType::F32, DType::F16, DType::BF16] {
        assert_llama_prefill_then_decode_stays_on_cuda(dtype);
    }
}

#[cfg(feature = "cuda")]
fn assert_llama_model_training_backward_stays_on_cuda(dtype: DType) {
    let config = test_config();
    let model = LlamaModel::new_with_dtype(config.clone(), dtype);
    model.to_cuda();
    let input_cuda = input_ids(&[1, 4], vec![1.0, 2.0, 3.0, 4.0]).to_cuda();
    let targets_cuda = one_hot_targets(4, config.vocab_size).to_cuda();

    set_strict_device_execution(true);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let logits = model.forward_train(input_cuda);
        assert!(logits.is_cuda());
        let loss = CrossEntropyLoss::apply(
            &logits.reshape(vec![-1, config.vocab_size as i32]),
            &targets_cuda,
        );
        loss.backward();
    }));
    set_strict_device_execution(false);
    result.unwrap();

    let params = model.parameters();
    assert!(!params.is_empty());
    for param in params {
        assert!(param.cloned_cuda_f32_grad().is_some());
        assert!(!param.has_host_grad());
    }
}

#[cfg(feature = "cuda")]
#[test]
fn llama_model_training_backward_stays_on_cuda_in_strict_mode() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    for dtype in [DType::F32, DType::F16, DType::BF16] {
        assert_llama_model_training_backward_stays_on_cuda(dtype);
    }
}

#[cfg(feature = "cuda")]
#[test]
fn llama_mlp_decode_fast_path_stays_on_cuda_and_matches_cpu() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let config = test_config();
    let mlp = LlamaMLP::new_with_dtype(&config, DType::BF16);
    let mlp_ref = LlamaMLP::new_with_dtype(&config, DType::BF16);

    for ((dst, src), shape) in [
        (
            (&mlp.gate_proj.weight, &mlp_ref.gate_proj.weight),
            vec![config.intermediate_size, config.hidden_size],
        ),
        (
            (&mlp.up_proj.weight, &mlp_ref.up_proj.weight),
            vec![config.intermediate_size, config.hidden_size],
        ),
        (
            (&mlp.down_proj.weight, &mlp_ref.down_proj.weight),
            vec![config.hidden_size, config.intermediate_size],
        ),
    ] {
        let data = (0..shape.iter().product::<usize>())
            .map(|i| (i as f32 * 0.03125) - 0.5)
            .collect::<Vec<_>>();
        let arr = ArrayD::from_shape_vec(IxDyn(&shape), data).expect("weight shape");
        dst.set_array_f32_with_dtype(arr.clone(), DType::BF16);
        src.set_array_f32_with_dtype(arr, DType::BF16);
    }

    mlp.gate_proj.weight.to_cuda_inplace();
    mlp.up_proj.weight.to_cuda_inplace();
    mlp.down_proj.weight.to_cuda_inplace();

    let input = Tensor::new_with_dtype(
        ArrayD::from_shape_vec(
            IxDyn(&[1, 1, config.hidden_size]),
            (0..config.hidden_size)
                .map(|i| (i as f32 * 0.125) - 0.25)
                .collect(),
        )
        .expect("input shape"),
        DType::BF16,
    )
    .to_cuda();

    crate::device::set_strict_device_execution(true);
    let cuda_out = no_grad(|| mlp.forward(input.clone()));
    crate::device::set_strict_device_execution(false);

    let cpu_out = no_grad(|| mlp_ref.forward(input.to_cpu()));
    assert!(cuda_out.is_cuda());
    assert_eq!(cuda_out.shape_vec(), cpu_out.shape_vec());
    for (got, expect) in cuda_out.data_ref().iter().zip(cpu_out.data_ref().iter()) {
        assert!((got - expect).abs() < 4e-2, "got {got}, expect {expect}");
    }
}
