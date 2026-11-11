use super::*;
use crate::{
    autograd::{Tensor, no_grad},
    device::Device,
    module::Module,
};
use ndarray::{ArrayD, IxDyn};

fn config() -> CausalLMConfig {
    serde_json::from_value(serde_json::json!({"model_type":"qwen3","vocab_size":17,
        "hidden_size":8,"intermediate_size":12,"num_hidden_layers":2,
        "num_attention_heads":2,"num_key_value_heads":1,"head_dim":6,
        "rms_norm_eps":1e-6,"max_position_embeddings":16,"rope_theta":10000.,
        "tie_word_embeddings":true,"eos_token_id":[15,16],"hidden_act":"silu"}))
    .unwrap()
}
fn ids(values: &[f32], device: Device) -> Tensor {
    Tensor::from_array_no_grad(
        ArrayD::from_shape_vec(IxDyn(&[1, values.len()]), values.to_vec()).unwrap(),
    )
    .to_device(device)
}

#[test]
fn pretrained_rejects_unsupported_configuration_before_construction() {
    let cfg = config();
    assert!(cfg.validate().is_ok());
    for field in [
        "model_type",
        "attention_bias",
        "rope_scaling",
        "use_sliding_window",
        "head_dim",
        "eos_token_id",
    ] {
        let mut value =
            serde_json::to_value(serde_json::json!({"model_type":"qwen3","vocab_size":17,
            "hidden_size":8,"intermediate_size":12,"num_hidden_layers":2,"num_attention_heads":2,
            "num_key_value_heads":1,"head_dim":6,"rms_norm_eps":1e-6,
            "max_position_embeddings":16,"rope_theta":10000.}))
            .unwrap();
        value[field] = match field {
            "model_type" => serde_json::json!("qwen3_moe"),
            "head_dim" => serde_json::json!(3),
            "eos_token_id" => serde_json::json!(99),
            "rope_scaling" => serde_json::json!({"type":"linear","factor":2.}),
            _ => serde_json::json!(true),
        };
        let bad: CausalLMConfig = serde_json::from_value(value).unwrap();
        assert!(bad.validate().is_err(), "{field}");
    }
    assert!(cfg.build(17, DType::F32, DType::F32, DType::F32).is_err());
}

#[test]
fn regression_pretrained_invalid_runtime_returns_error() {
    let cfg = config();
    let result = std::panic::catch_unwind(|| cfg.build(16, DType::F32, DType::F32, DType::I8));
    assert!(
        matches!(result, Ok(Err(_))),
        "build should return Err for an integer KV dtype"
    );
}

#[cfg(feature = "cuda")]
#[test]
fn regression_qwen3_cuda_lowp_cache_native_snapshot_is_current() {
    if !crate::ops::cuda::is_available() {
        return;
    }
    let _enabled = crate::ops::cuda::set_enabled_scoped(true);
    let _strict = crate::device::set_strict_device_execution_scoped(true);
    for dtype in [DType::F16, DType::BF16] {
        let model = config().build(16, DType::F32, DType::F32, dtype).unwrap();
        model.to_cuda();
        let mut caches = model.init_kv_caches(1);
        let _ = no_grad(|| model.forward(ids(&[1., 2., 3.], Device::Cuda), &mut caches, 0));
        for cache in &caches {
            let cache = cache.borrow();
            let mirror =
                crate::ops::cuda::download_f32(&cache.k.cloned_cuda_f32_buffer().unwrap()).unwrap();
            assert!(mirror.iter().any(|value| value.abs() > 1e-4));
            let snapshot = cache.k.to_cpu();
            let native = snapshot.data_ref().iter().copied().collect::<Vec<_>>();
            assert!(
                native.iter().any(|value| value.abs() > 1e-4),
                "{dtype:?}: native cache lost all written keys"
            );
            for (a, b) in mirror.iter().zip(&native) {
                let represented = match dtype {
                    DType::F16 => half::f16::from_f32(*a).to_f32(),
                    DType::BF16 => half::bf16::from_f32(*a).to_f32(),
                    _ => unreachable!(),
                };
                assert_eq!(
                    *a, represented,
                    "{dtype:?}: computation cache did not respect its dtype"
                );
                assert_eq!(
                    represented, *b,
                    "{dtype:?}: cache snapshot differs from represented keys"
                );
            }
        }
        let mut migrated = model.init_kv_caches(1);
        let _ = no_grad(|| model.forward(ids(&[1., 2., 3.], Device::Cuda), &mut migrated, 0));
        for cache in &migrated {
            cache.borrow_mut().to_device_inplace(Device::Cpu);
            cache.borrow_mut().to_device_inplace(Device::Cuda);
        }
        let continued = no_grad(|| model.forward(ids(&[4.], Device::Cuda), &mut caches, 0));
        let restored = no_grad(|| model.forward(ids(&[4.], Device::Cuda), &mut migrated, 0));
        for (a, b) in continued.data_ref().iter().zip(restored.data_ref().iter()) {
            assert!(
                (a - b).abs() < 2e-5,
                "{dtype:?}: migrating KV cache changed continued inference: {a} vs {b}"
            );
        }
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_kv_rounding_only_touches_appended_rows() {
    if !crate::ops::cuda::is_available() {
        return;
    }
    let values = (0..2 * 3 * 7 * 5)
        .map(|i| ((i as f32) * 0.1317).sin() * 1.3721)
        .collect::<Vec<_>>();
    for dtype in [DType::F16, DType::BF16] {
        let buffer = crate::ops::cuda::upload_f32(&values).unwrap();
        assert!(
            crate::ops::cuda::round_kv_cache_rows_f32(&buffer, 2, 3, 3, 7, 5, 6, dtype).is_err()
        );
        assert_eq!(crate::ops::cuda::download_f32(&buffer).unwrap(), values);
        crate::ops::cuda::round_kv_cache_rows_f32(&buffer, 2, 3, 3, 7, 5, 2, dtype).unwrap();
        let actual = crate::ops::cuda::download_f32(&buffer).unwrap();
        for (i, (&a, &value)) in actual.iter().zip(&values).enumerate() {
            let position = (i / 5) % 7;
            let expected = if (2..5).contains(&position) {
                match dtype {
                    DType::F16 => half::f16::from_f32(value).to_f32(),
                    DType::BF16 => half::bf16::from_f32(value).to_f32(),
                    _ => unreachable!(),
                }
            } else {
                value
            };
            assert_eq!(a, expected, "{dtype:?} at element {i}");
        }
    }
}

fn cache_and_gradients(device: Device) {
    let _enabled = crate::ops::cuda::set_enabled_scoped(device == Device::Cuda);
    let _strict = crate::device::set_strict_device_execution_scoped(device == Device::Cuda);
    for dtype in [DType::F32, DType::F16, DType::BF16, DType::I8] {
        let model = config().build(16, dtype, DType::F32, DType::F32).unwrap();
        for (name, p) in model.checkpoint_parameters() {
            let values = (0..p.shape_vec().iter().product::<usize>())
                .map(|i| {
                    if name.contains("norm") {
                        0.9 + (i % 5) as f32 * 0.03
                    } else {
                        ((i * 7 + name.len()) as f32 * 0.17).sin() * 0.16
                    }
                })
                .collect::<Vec<_>>();
            let shape = p.shape_vec();
            p.set_array_f32_with_dtype(
                ArrayD::from_shape_vec(IxDyn(&shape), values).unwrap(),
                dtype,
            );
        }
        model.to_device(device);
        let named = model.named_parameters();
        assert!(std::rc::Rc::ptr_eq(
            &named["lm_head.weight"].0,
            &named["model.embed_tokens.weight"].0
        ));
        assert_eq!(model.parameters().len() + 1, named.len());
        let sequence = [1., 3., 2., 5., 4.];
        let full =
            no_grad(|| model.forward(ids(&sequence, device), &mut model.init_kv_caches(1), 0));
        for chunks in [vec![2, 1, 2], vec![1, 1, 1, 1, 1]] {
            let mut caches = model.init_kv_caches(1);
            let mut got = Vec::new();
            let mut offset = 0;
            for length in chunks {
                let output = no_grad(|| {
                    model.forward(
                        ids(&sequence[offset..offset + length], device),
                        &mut caches,
                        0,
                    )
                });
                got.extend(output.data_ref().iter().copied());
                offset += length;
            }
            assert_eq!(caches[0].borrow().len, 5);
            for (&a, &b) in full.data_ref().iter().zip(&got) {
                assert!((a - b).abs() < 3e-4, "{device:?}/{dtype:?}: {a} vs {b}");
            }
            model.reset_kv_caches(&mut caches);
            assert_eq!(caches[0].borrow().len, 0);
        }
        let train = model.forward_train(ids(&sequence, device));
        for (&a, &b) in train.data_ref().iter().zip(full.data_ref().iter()) {
            assert!((a - b).abs() < 3e-4, "train/infer {dtype:?}: {a} vs {b}");
        }
        train.backward();
        for (name, p) in model.checkpoint_parameters() {
            let gradient = p
                .grad()
                .unwrap_or_else(|| panic!("missing gradient: {name}"));
            assert_eq!(gradient.shape(), p.shape_vec());
            assert!(gradient.iter().all(|x| x.is_finite()));
        }
        let qnorm = model.named_parameters()["model.layers.0.self_attn.q_norm.weight"]
            .grad()
            .unwrap();
        assert!(qnorm.iter().any(|x| x.abs() > 1e-9));
        if dtype == DType::F32 && std::env::var_os("LUMEN_OPERATOR_AUDIT").is_some() {
            let parameters=model.checkpoint_parameters().into_iter().map(|(name,p)| {
                (name,serde_json::json!({"shape":p.shape_vec(),"values":p.data_ref().iter().copied().collect::<Vec<_>>(),
                    "gradient":p.grad().unwrap().iter().copied().collect::<Vec<_>>()}))
            }).collect::<std::collections::BTreeMap<_,_>>();
            println!(
                "QWEN3_AUDIT={}",
                serde_json::json!({"device":format!("{device:?}"),"parameters":parameters,
                "input_ids":[1,3,2,5,4],"heads":2,"kv_heads":1,"head_dim":6,"layers":2,"eps":1e-6,
                "rope_theta":10000.,"logits":train.data_ref().iter().copied().collect::<Vec<_>>()})
            );
        }
    }
}

#[test]
fn qwen3_cpu_expanded_heads_tied_weights_cache_and_gradients() {
    cache_and_gradients(Device::Cpu);
}
#[cfg(feature = "cuda")]
#[test]
fn qwen3_cuda_expanded_heads_tied_weights_cache_and_gradients() {
    if crate::ops::cuda::is_available() {
        cache_and_gradients(Device::Cuda);
    }
}

#[test]
fn qwen3_text_chat_format_and_stops() {
    let format = crate::tokenizer::ChatFormat::Qwen3 {
        enable_thinking: false,
    };
    assert_eq!(
        format.first_turn("Help.", "Hi"),
        "<|im_start|>system\nHelp.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    );
    assert!(
        format
            .next_turn("Next")
            .starts_with("<|im_end|>\n<|im_start|>user\nNext")
    );
    assert!(
        !crate::tokenizer::ChatFormat::Qwen3 {
            enable_thinking: true
        }
        .first_turn("Help.", "Hi")
        .contains("</think>")
    );
    assert_eq!(config().eos_token_id.unwrap().values(), vec![15, 16]);
}

fn descending_training(device: Device) {
    use crate::{
        loss::MSELoss,
        optim::{Adam, Optimizer, SGD},
    };
    let _enabled = crate::ops::cuda::set_enabled_scoped(device == Device::Cuda);
    let _strict = crate::device::set_strict_device_execution_scoped(device == Device::Cuda);
    for adam in [false, true] {
        let model = config()
            .build(16, DType::F32, DType::F32, DType::F32)
            .unwrap();
        model.to_device(device);
        let mut optimizer: Box<dyn Optimizer> = if adam {
            Box::new(Adam::new_with_dtype(model.parameters(), 0.015, DType::F32))
        } else {
            Box::new(SGD::new_with_dtype(model.parameters(), 0.04, DType::F32))
        };
        let target = Tensor::from_array_no_grad(ArrayD::from_elem(IxDyn(&[1, 3, 17]), 0.1))
            .to_device(device);
        let mut losses = Vec::new();
        for _ in 0..80 {
            optimizer.zero_grad();
            let loss = MSELoss::apply(&model.forward_train(ids(&[1., 2., 3.], device)), &target);
            losses.push(*loss.data_ref().first().unwrap());
            loss.backward();
            optimizer.step();
        }
        assert!(losses.iter().all(|loss| loss.is_finite()));
        assert!(
            losses[79] < losses[0] * 0.4,
            "{device:?}/adam={adam}: {} -> {}",
            losses[0],
            losses[79]
        );
        assert!(losses[60..].iter().sum::<f32>() < losses[..20].iter().sum::<f32>());
    }
}
#[test]
fn qwen3_cpu_shared_weight_training_descends() {
    descending_training(Device::Cpu);
}
#[cfg(feature = "cuda")]
#[test]
fn qwen3_cuda_shared_weight_training_descends() {
    if crate::ops::cuda::is_available() {
        descending_training(Device::Cuda);
    }
}
