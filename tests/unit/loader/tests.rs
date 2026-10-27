use super::*;
use crate::autograd::Tensor;
use crate::precision::with_parameter_quantization;
use ndarray::{ArrayD, IxDyn};
use safetensors::tensor::{TensorView, serialize_to_file};
use std::collections::HashMap;
use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

fn bytes_from_f32(data: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for value in data {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn bytes_from_f16(data: &[f16]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(data.len() * 2);
    for value in data {
        bytes.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    bytes
}

fn write_safetensor(
    name: &str,
    dtype: safetensors::Dtype,
    shape: Vec<usize>,
    bytes: &[u8],
) -> std::path::PathBuf {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::env::temp_dir().join(format!("lumen_loader_{stamp}.safetensors"));
    let view = TensorView::new(dtype, shape, bytes).expect("failed to build tensor view");
    serialize_to_file(vec![(name.to_string(), view)], &None, &path)
        .expect("failed to write safetensors file");
    path
}

fn bytes_from_i8(data: &[i8]) -> Vec<u8> {
    data.iter().map(|&v| v as u8).collect()
}

fn write_safetensors(
    entries: Vec<(String, safetensors::Dtype, Vec<usize>, Vec<u8>)>,
) -> std::path::PathBuf {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::env::temp_dir().join(format!("lumen_loader_multi_{stamp}.safetensors"));
    let mut views = Vec::with_capacity(entries.len());
    for (name, dtype, shape, bytes) in &entries {
        let view =
            TensorView::new(*dtype, shape.clone(), bytes).expect("failed to build tensor view");
        views.push((name.clone(), view));
    }
    serialize_to_file(views, &None, &path).expect("failed to write safetensors file");
    path
}

#[test]
fn float_weights_can_be_quantized_directly_during_load() {
    let data = vec![1.0f32, -2.0, 3.5, 4.25];
    let bytes = bytes_from_f32(&data);
    let path = write_safetensor("weight", safetensors::Dtype::F32, vec![2, 2], &bytes);

    let param = Tensor::parameter_with_dtype(ArrayD::zeros(IxDyn(&[2, 2])), DType::F32);
    let mut params = HashMap::new();
    params.insert("weight".to_string(), param.clone());

    with_parameter_quantization(ParameterQuantization::Int8, || {
        ModelLoader::load_llama_weights(&path, &params).unwrap();
    });

    assert_eq!(param.dtype(), DType::I8);
    let loaded = param.data();
    for (&lhs, &rhs) in loaded.iter().zip(data.iter()) {
        assert!((lhs - rhs).abs() <= 0.05, "lhs={lhs}, rhs={rhs}");
    }

    let _ = fs::remove_file(path);
}

#[test]
fn explicit_load_options_override_global_quantization_setting() {
    let data = vec![1.0f32, -2.0, 3.5, 4.25];
    let bytes = bytes_from_f32(&data);
    let path = write_safetensor("weight", safetensors::Dtype::F32, vec![2, 2], &bytes);

    let param = Tensor::parameter_with_dtype(ArrayD::zeros(IxDyn(&[2, 2])), DType::F32);
    let mut params = HashMap::new();
    params.insert("weight".to_string(), param.clone());

    with_parameter_quantization(ParameterQuantization::Int8, || {
        ModelLoader::load_llama_weights_with_options(
            &path,
            &params,
            WeightLoadOptions {
                float_source_quantization: ParameterQuantization::Disabled,
                stream_from_disk: false,
            },
        )
        .unwrap();
    });

    assert_eq!(param.dtype(), DType::F32);
    let loaded = param.data();
    for (&lhs, &rhs) in loaded.iter().zip(data.iter()) {
        assert!((lhs - rhs).abs() <= 1e-6, "lhs={lhs}, rhs={rhs}");
    }

    let _ = fs::remove_file(path);
}

#[test]
fn f16_weights_can_be_loaded_and_quantized_to_i8_directly() {
    let data_f32 = [1.0f32, -2.0, 3.5, 4.25];
    let data_f16 = data_f32
        .iter()
        .map(|&v| f16::from_f32(v))
        .collect::<Vec<_>>();
    let bytes = bytes_from_f16(&data_f16);
    let path = write_safetensor("weight", safetensors::Dtype::F16, vec![2, 2], &bytes);

    let param = Tensor::parameter_with_dtype(ArrayD::zeros(IxDyn(&[2, 2])), DType::F32);
    let mut params = HashMap::new();
    params.insert("weight".to_string(), param.clone());

    ModelLoader::load_llama_weights_with_options(
        &path,
        &params,
        WeightLoadOptions {
            float_source_quantization: ParameterQuantization::Int8,
            stream_from_disk: false,
        },
    )
    .unwrap();

    assert_eq!(param.dtype(), DType::I8);
    let loaded = param.data();
    for (&lhs, &rhs) in loaded.iter().zip(data_f32.iter()) {
        assert!((lhs - rhs).abs() <= 0.05, "lhs={lhs}, rhs={rhs}");
    }

    let _ = fs::remove_file(path);
}

#[test]
fn float_weights_can_be_quantized_with_manual_scale_during_load() {
    let data = vec![0.9f32, -1.1, 1.6, -2.6];
    let bytes = bytes_from_f32(&data);
    let path = write_safetensor("weight", safetensors::Dtype::F32, vec![2, 2], &bytes);

    let param = Tensor::parameter_with_dtype(ArrayD::zeros(IxDyn(&[2, 2])), DType::F32);
    let mut params = HashMap::new();
    params.insert("weight".to_string(), param.clone());

    ModelLoader::load_llama_weights_with_options(
        &path,
        &params,
        WeightLoadOptions {
            float_source_quantization: ParameterQuantization::Int8.with_scale(0.5),
            stream_from_disk: false,
        },
    )
    .unwrap();

    assert_eq!(param.dtype(), DType::I8);
    assert_eq!(param.quantization_scale(), Some(0.5));
    let loaded = param.data();
    let expected = [1.0f32, -1.0, 1.5, -2.5];
    for (got, want) in loaded.iter().zip(expected.iter()) {
        assert!((got - want).abs() <= 1e-6, "got {got}, want {want}");
    }

    let _ = fs::remove_file(path);
}

#[test]
fn i8_weights_can_load_with_companion_scale_tensor() {
    let quantized = vec![4i8, -8, 7, 9];
    let expected = [2.0f32, -4.0, 3.5, 4.5];
    let scale = 0.5f32;
    let path = write_safetensors(vec![
        (
            "weight".to_string(),
            safetensors::Dtype::I8,
            vec![2, 2],
            bytes_from_i8(&quantized),
        ),
        (
            "weight.scale".to_string(),
            safetensors::Dtype::F32,
            vec![1],
            bytes_from_f32(&[scale]),
        ),
    ]);

    let param = Tensor::parameter_with_dtype(ArrayD::zeros(IxDyn(&[2, 2])), DType::I8);
    let mut params = HashMap::new();
    params.insert("weight".to_string(), param.clone());

    ModelLoader::load_llama_weights(&path, &params).unwrap();

    assert_eq!(param.dtype(), DType::I8);
    assert_eq!(param.quantization_scale(), Some(scale));
    let loaded = param.data();
    for (&lhs, &rhs) in loaded.iter().zip(expected.iter()) {
        assert!((lhs - rhs).abs() <= 1e-6, "lhs={lhs}, rhs={rhs}");
    }

    let _ = fs::remove_file(path);
}

#[test]
fn i8_weights_can_stream_from_disk_with_companion_scale_tensor() {
    let quantized = vec![4i8, -8, 7, 9];
    let expected = [2.0f32, -4.0, 3.5, 4.5];
    let scale = 0.5f32;
    let path = write_safetensors(vec![
        (
            "weight".to_string(),
            safetensors::Dtype::I8,
            vec![2, 2],
            bytes_from_i8(&quantized),
        ),
        (
            "weight.scale".to_string(),
            safetensors::Dtype::F32,
            vec![1],
            bytes_from_f32(&[scale]),
        ),
    ]);

    let param = Tensor::parameter_with_dtype(ArrayD::zeros(IxDyn(&[2, 2])), DType::I8);
    let mut params = HashMap::new();
    params.insert("weight".to_string(), param.clone());

    ModelLoader::load_llama_weights_with_options(
        &path,
        &params,
        WeightLoadOptions {
            float_source_quantization: ParameterQuantization::Disabled,
            stream_from_disk: true,
        },
    )
    .unwrap();

    assert_eq!(param.dtype(), DType::I8);
    assert_eq!(param.quantization_scale(), Some(scale));
    let loaded = param.data();
    for (&lhs, &rhs) in loaded.iter().zip(expected.iter()) {
        assert!((lhs - rhs).abs() <= 1e-6, "lhs={lhs}, rhs={rhs}");
    }

    let _ = fs::remove_file(path);
}

#[test]
fn i8_weights_require_companion_scale_tensor_for_mmap_load() {
    let path = write_safetensors(vec![(
        "weight".to_string(),
        safetensors::Dtype::I8,
        vec![2, 2],
        bytes_from_i8(&[4i8, -8, 7, 9]),
    )]);

    let param = Tensor::parameter_with_dtype(ArrayD::zeros(IxDyn(&[2, 2])), DType::I8);
    let mut params = HashMap::new();
    params.insert("weight".to_string(), param);

    let err = ModelLoader::load_llama_weights(&path, &params)
        .expect_err("missing i8 scale should return an error");
    assert!(
        err.to_string()
            .contains("missing required companion scale tensor"),
        "unexpected error: {err}"
    );

    let _ = fs::remove_file(path);
}

#[test]
fn i8_weights_require_companion_scale_tensor_for_streaming_load() {
    let path = write_safetensors(vec![(
        "weight".to_string(),
        safetensors::Dtype::I8,
        vec![2, 2],
        bytes_from_i8(&[4i8, -8, 7, 9]),
    )]);

    let param = Tensor::parameter_with_dtype(ArrayD::zeros(IxDyn(&[2, 2])), DType::I8);
    let mut params = HashMap::new();
    params.insert("weight".to_string(), param);

    let err = ModelLoader::load_llama_weights_with_options(
        &path,
        &params,
        WeightLoadOptions {
            float_source_quantization: ParameterQuantization::Disabled,
            stream_from_disk: true,
        },
    )
    .expect_err("missing i8 scale should return an error");
    assert!(
        err.to_string()
            .contains("missing required companion scale tensor"),
        "unexpected error: {err}"
    );

    let _ = fs::remove_file(path);
}
