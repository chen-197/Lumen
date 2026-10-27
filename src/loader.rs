use crate::autograd::Tensor;
use crate::precision::{DType, ParameterQuantization, default_parameter_quantization};
use half::{bf16, f16};
use memmap2::MmapOptions;
use ndarray::{Array, IxDyn};
use safetensors::SafeTensors;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WeightLoadOptions {
    pub float_source_quantization: ParameterQuantization,
    pub stream_from_disk: bool,
}

impl Default for WeightLoadOptions {
    fn default() -> Self {
        Self::from_global_defaults()
    }
}

impl WeightLoadOptions {
    #[inline]
    pub fn from_global_defaults() -> Self {
        Self {
            float_source_quantization: default_parameter_quantization(),
            stream_from_disk: false,
        }
    }

    #[inline]
    fn override_target_dtype(
        self,
        source_dtype: safetensors::Dtype,
        current_target_dtype: DType,
    ) -> DType {
        if matches!(
            source_dtype,
            safetensors::Dtype::F32 | safetensors::Dtype::F16 | safetensors::Dtype::BF16
        ) && let Some(dtype) = self.float_source_quantization.storage_dtype()
        {
            return dtype;
        }
        current_target_dtype
    }

    #[inline]
    fn quantization_for_float_target_dtype(
        self,
        target_dtype: DType,
    ) -> Option<ParameterQuantization> {
        if self.float_source_quantization.is_enabled() {
            return Some(self.float_source_quantization);
        }
        if target_dtype.is_integer() {
            return Some(ParameterQuantization::new(target_dtype));
        }
        None
    }
}

#[derive(Clone, Debug, Deserialize)]
struct StreamingTensorInfo {
    dtype: safetensors::Dtype,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

#[derive(Debug, Deserialize)]
struct StreamingHeader {
    #[serde(rename = "__metadata__", default)]
    metadata: Option<HashMap<String, String>>,
    #[serde(flatten)]
    tensors: HashMap<String, StreamingTensorInfo>,
}

pub struct ModelLoader;

#[derive(Default)]
struct LoadReport {
    loaded: usize,
    quantized_on_load: usize,
    missing: Vec<String>,
}

impl LoadReport {
    #[inline]
    fn record_loaded(&mut self, quantized_on_load: bool) {
        self.loaded += 1;
        if quantized_on_load {
            self.quantized_on_load += 1;
        }
    }

    #[inline]
    fn record_missing(&mut self, name: &str) {
        self.missing.push(name.to_string());
    }

    fn print_summary(&self) {
        println!(
            "Loaded {} tensors{}.",
            self.loaded,
            if self.quantized_on_load > 0 {
                format!(", {} quantized on load", self.quantized_on_load)
            } else {
                String::new()
            }
        );
        if !self.missing.is_empty() {
            let preview = self.missing.iter().take(5).cloned().collect::<Vec<_>>();
            let suffix = if self.missing.len() > preview.len() {
                format!(" ... (+{} more)", self.missing.len() - preview.len())
            } else {
                String::new()
            };
            println!(
                "Warning: {} parameters were missing from the checkpoint: {}{}",
                self.missing.len(),
                preview.join(", "),
                suffix
            );
        }
    }
}

impl ModelLoader {
    fn find_i8_scale_tensor_name(tensors: &SafeTensors<'_>, weight_name: &str) -> Option<String> {
        let dot = format!("{weight_name}.scale");
        if tensors.tensor(&dot).is_ok() {
            return Some(dot);
        }
        let underscore = format!("{weight_name}_scale");
        if tensors.tensor(&underscore).is_ok() {
            return Some(underscore);
        }
        None
    }

    fn find_i8_scale_tensor_name_in_map(
        tensors: &HashMap<String, StreamingTensorInfo>,
        weight_name: &str,
    ) -> Option<String> {
        let dot = format!("{weight_name}.scale");
        if tensors.contains_key(&dot) {
            return Some(dot);
        }
        let underscore = format!("{weight_name}_scale");
        if tensors.contains_key(&underscore) {
            return Some(underscore);
        }
        None
    }

    fn decode_scalar_scale_from_bytes(
        dtype: safetensors::Dtype,
        data_bytes: &[u8],
        scale_name: &str,
        scale_shape: &[usize],
    ) -> Result<f32, Box<dyn std::error::Error>> {
        let scale_len = scale_shape.iter().product::<usize>();
        if scale_len != 1 {
            return Err(format!(
                "Scale tensor {} must contain exactly one element, got shape {:?}",
                scale_name, scale_shape
            )
            .into());
        }

        let scale = match dtype {
            safetensors::Dtype::F32 => {
                if data_bytes.len() != 4 {
                    return Err(format!(
                        "Scale tensor {} expected 4 bytes for F32, got {}",
                        scale_name,
                        data_bytes.len()
                    )
                    .into());
                }
                f32::from_le_bytes(
                    data_bytes[0..4]
                        .try_into()
                        .expect("slice with exact f32 byte length"),
                )
            }
            safetensors::Dtype::F16 => {
                if data_bytes.len() != 2 {
                    return Err(format!(
                        "Scale tensor {} expected 2 bytes for F16, got {}",
                        scale_name,
                        data_bytes.len()
                    )
                    .into());
                }
                f16::from_bits(u16::from_le_bytes(
                    data_bytes[0..2]
                        .try_into()
                        .expect("slice with exact f16 byte length"),
                ))
                .to_f32()
            }
            safetensors::Dtype::BF16 => {
                if data_bytes.len() != 2 {
                    return Err(format!(
                        "Scale tensor {} expected 2 bytes for BF16, got {}",
                        scale_name,
                        data_bytes.len()
                    )
                    .into());
                }
                bf16::from_bits(u16::from_le_bytes(
                    data_bytes[0..2]
                        .try_into()
                        .expect("slice with exact bf16 byte length"),
                ))
                .to_f32()
            }
            other => {
                return Err(
                    format!("Unsupported scale dtype {:?} for {}", other, scale_name).into(),
                );
            }
        };

        if !scale.is_finite() || scale <= 0.0 {
            return Err(format!(
                "Scale tensor {} must be finite and > 0, got {}",
                scale_name, scale
            )
            .into());
        }

        Ok(scale)
    }

    fn read_scalar_scale(
        tensors: &SafeTensors<'_>,
        scale_name: &str,
    ) -> Result<f32, Box<dyn std::error::Error>> {
        let scale_view = tensors.tensor(scale_name)?;
        Self::decode_scalar_scale_from_bytes(
            scale_view.dtype(),
            scale_view.data(),
            scale_name,
            scale_view.shape(),
        )
    }

    fn decode_f32_bytes(
        data_bytes: &[u8],
        name: &str,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        if !data_bytes.len().is_multiple_of(4) {
            return Err(format!(
                "Tensor {} has invalid F32 byte length {}",
                name,
                data_bytes.len()
            )
            .into());
        }
        Ok(data_bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("f32 chunk should be 4 bytes")))
            .collect())
    }

    fn decode_f16_bytes(
        data_bytes: &[u8],
        name: &str,
    ) -> Result<Vec<f16>, Box<dyn std::error::Error>> {
        if !data_bytes.len().is_multiple_of(2) {
            return Err(format!(
                "Tensor {} has invalid F16 byte length {}",
                name,
                data_bytes.len()
            )
            .into());
        }
        Ok(data_bytes
            .chunks_exact(2)
            .map(|chunk| {
                f16::from_bits(u16::from_le_bytes(
                    chunk.try_into().expect("f16 chunk should be 2 bytes"),
                ))
            })
            .collect())
    }

    fn decode_bf16_bytes(
        data_bytes: &[u8],
        name: &str,
    ) -> Result<Vec<bf16>, Box<dyn std::error::Error>> {
        if !data_bytes.len().is_multiple_of(2) {
            return Err(format!(
                "Tensor {} has invalid BF16 byte length {}",
                name,
                data_bytes.len()
            )
            .into());
        }
        Ok(data_bytes
            .chunks_exact(2)
            .map(|chunk| {
                bf16::from_bits(u16::from_le_bytes(
                    chunk.try_into().expect("bf16 chunk should be 2 bytes"),
                ))
            })
            .collect())
    }

    fn read_streaming_header(
        file: &mut File,
    ) -> Result<(u64, HashMap<String, StreamingTensorInfo>), Box<dyn std::error::Error>> {
        let mut len_bytes = [0u8; 8];
        file.seek(SeekFrom::Start(0))?;
        file.read_exact(&mut len_bytes)?;
        let header_len_u64 = u64::from_le_bytes(len_bytes);
        let header_len: usize = header_len_u64
            .try_into()
            .map_err(|_| "safetensors header length overflows usize")?;
        let mut header_bytes = vec![0u8; header_len];
        file.read_exact(&mut header_bytes)?;
        let header: StreamingHeader = serde_json::from_slice(&header_bytes)?;
        let _metadata = header.metadata;
        Ok((8 + header_len_u64, header.tensors))
    }

    fn read_streaming_tensor_bytes(
        file: &mut File,
        data_offset_base: u64,
        info: &StreamingTensorInfo,
        buffer: &mut Vec<u8>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let start = info.data_offsets[0] as u64;
        let end = info.data_offsets[1] as u64;
        if end < start {
            return Err("invalid safetensors tensor offsets".into());
        }
        let len = (end - start) as usize;
        buffer.resize(len, 0);
        file.seek(SeekFrom::Start(data_offset_base + start))?;
        file.read_exact(buffer)?;
        Ok(())
    }

    fn read_streaming_scale(
        file: &mut File,
        data_offset_base: u64,
        tensors: &HashMap<String, StreamingTensorInfo>,
        scale_name: &str,
        buffer: &mut Vec<u8>,
    ) -> Result<f32, Box<dyn std::error::Error>> {
        let info = tensors
            .get(scale_name)
            .ok_or_else(|| format!("Scale tensor {} not found", scale_name))?;
        Self::read_streaming_tensor_bytes(file, data_offset_base, info, buffer)?;
        Self::decode_scalar_scale_from_bytes(info.dtype, buffer, scale_name, &info.shape)
    }

    pub fn load_llama_weights<P: AsRef<Path>>(
        path: P,

        model_params: &std::collections::HashMap<String, Tensor>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        Self::load_llama_weights_with_options(
            path,
            model_params,
            WeightLoadOptions::from_global_defaults(),
        )
    }

    fn load_llama_weights_streaming<P: AsRef<Path>>(
        path: P,
        model_params: &std::collections::HashMap<String, Tensor>,
        options: WeightLoadOptions,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut file = File::open(path)?;
        let (data_offset_base, tensor_infos) = Self::read_streaming_header(&mut file)?;

        println!("Loading weights...");

        let mut ordered = model_params
            .iter()
            .filter_map(|(name, tensor_target)| {
                tensor_infos
                    .get(name)
                    .map(|info| (info.data_offsets[0], name.as_str(), tensor_target, info))
            })
            .collect::<Vec<_>>();
        ordered.sort_by_key(|(offset, _, _, _)| *offset);

        let mut buffer = Vec::new();
        let mut scale_buffer = Vec::new();
        let mut report = LoadReport::default();

        for (_, name, tensor_target, info) in ordered {
            let source_dtype = info.dtype;
            let target_shape = tensor_target.shape_vec();
            if info.shape != target_shape {
                return Err(format!(
                    "Shape mismatch for {}: safetensors={:?}, target={:?}",
                    name, info.shape, target_shape
                )
                .into());
            }

            let target_dtype = options.override_target_dtype(source_dtype, tensor_target.dtype());
            let target_quantization = if matches!(
                source_dtype,
                safetensors::Dtype::F32 | safetensors::Dtype::F16 | safetensors::Dtype::BF16
            ) {
                options.quantization_for_float_target_dtype(target_dtype)
            } else {
                None
            };
            let direct_quantized_load =
                source_dtype != safetensors::Dtype::I8 && target_dtype.is_integer();

            Self::read_streaming_tensor_bytes(&mut file, data_offset_base, info, &mut buffer)?;

            match source_dtype {
                safetensors::Dtype::F32 => {
                    let decoded = Self::decode_f32_bytes(&buffer, name)?;
                    if let Some(quantization) = target_quantization {
                        tensor_target.set_f32_slice_with_quantization(
                            &target_shape,
                            &decoded,
                            quantization,
                        );
                    } else {
                        let source_array = Array::from_shape_vec(IxDyn(&target_shape), decoded)
                            .map_err(|e| format!("Shape mismatch for {}: {}", name, e))?;
                        tensor_target.set_array_f32_with_dtype(source_array, target_dtype);
                    }
                }
                safetensors::Dtype::F16 => {
                    let decoded = Self::decode_f16_bytes(&buffer, name)?;
                    if let Some(quantization) = target_quantization {
                        tensor_target.set_f16_slice_with_quantization(
                            &target_shape,
                            &decoded,
                            quantization,
                        );
                    } else {
                        let source_array = Array::from_shape_vec(IxDyn(&target_shape), decoded)
                            .map_err(|e| format!("Shape mismatch for {}: {}", name, e))?;
                        tensor_target.set_array_f16_with_dtype(source_array, target_dtype);
                    }
                }
                safetensors::Dtype::BF16 => {
                    let decoded = Self::decode_bf16_bytes(&buffer, name)?;
                    if let Some(quantization) = target_quantization {
                        tensor_target.set_bf16_slice_with_quantization(
                            &target_shape,
                            &decoded,
                            quantization,
                        );
                    } else {
                        let source_array = Array::from_shape_vec(IxDyn(&target_shape), decoded)
                            .map_err(|e| format!("Shape mismatch for {}: {}", name, e))?;
                        tensor_target.set_array_bf16_with_dtype(source_array, target_dtype);
                    }
                }
                safetensors::Dtype::I8 => {
                    let i8_data = buffer.iter().map(|&b| b as i8).collect::<Vec<_>>();
                    let scale_name = Self::find_i8_scale_tensor_name_in_map(&tensor_infos, name);
                    let scale = if let Some(scale_name) = scale_name.as_deref() {
                        Self::read_streaming_scale(
                            &mut file,
                            data_offset_base,
                            &tensor_infos,
                            scale_name,
                            &mut scale_buffer,
                        )?
                    } else {
                        return Err(format!(
                            "I8 tensor {} is missing required companion scale tensor (expected {}.scale or {}_scale)",
                            name, name, name
                        )
                        .into());
                    };
                    tensor_target.set_i8_slice_with_dtype(
                        &target_shape,
                        &i8_data,
                        scale,
                        target_dtype,
                    );
                }
                _ => {
                    return Err(
                        format!("Unsupported dtype: {:?} for {}", source_dtype, name).into(),
                    );
                }
            }
            report.record_loaded(direct_quantized_load);
        }

        for name in model_params.keys() {
            if !tensor_infos.contains_key(name) {
                report.record_missing(name);
            }
        }

        report.print_summary();

        Ok(())
    }

    pub fn load_llama_weights_with_options<P: AsRef<Path>>(
        path: P,
        model_params: &std::collections::HashMap<String, Tensor>,
        options: WeightLoadOptions,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if options.stream_from_disk {
            return Self::load_llama_weights_streaming(path, model_params, options);
        }

        let file = File::open(path)?;

        let mmap = unsafe { MmapOptions::new().map(&file)? };

        let tensors = SafeTensors::deserialize(&mmap)?;

        println!("Loading weights...");
        let mut report = LoadReport::default();

        for (name, tensor_target) in model_params {
            if let Ok(view) = tensors.tensor(name) {
                let source_dtype = view.dtype();
                let data_bytes = view.data();
                let source_shape = view.shape().to_vec();
                let target_shape = tensor_target.shape_vec();
                if source_shape != target_shape {
                    return Err(format!(
                        "Shape mismatch for {}: safetensors={:?}, target={:?}",
                        name, source_shape, target_shape
                    )
                    .into());
                }
                let target_dtype =
                    options.override_target_dtype(source_dtype, tensor_target.dtype());
                let target_quantization = if matches!(
                    source_dtype,
                    safetensors::Dtype::F32 | safetensors::Dtype::F16 | safetensors::Dtype::BF16
                ) {
                    options.quantization_for_float_target_dtype(target_dtype)
                } else {
                    None
                };
                let direct_quantized_load =
                    source_dtype != safetensors::Dtype::I8 && target_dtype.is_integer();

                match source_dtype {
                    safetensors::Dtype::F32 => {
                        let decoded = Self::decode_f32_bytes(data_bytes, name)?;

                        if let Some(quantization) = target_quantization {
                            tensor_target.set_f32_slice_with_quantization(
                                &target_shape,
                                &decoded,
                                quantization,
                            );
                        } else {
                            let source_array = Array::from_shape_vec(IxDyn(&target_shape), decoded)
                                .map_err(|e| format!("Shape mismatch for {}: {}", name, e))?;
                            tensor_target.set_array_f32_with_dtype(source_array, target_dtype);
                        }
                    }

                    safetensors::Dtype::F16 => {
                        let decoded = Self::decode_f16_bytes(data_bytes, name)?;

                        if let Some(quantization) = target_quantization {
                            tensor_target.set_f16_slice_with_quantization(
                                &target_shape,
                                &decoded,
                                quantization,
                            );
                        } else {
                            let source_array = Array::from_shape_vec(IxDyn(&target_shape), decoded)
                                .map_err(|e| format!("Shape mismatch for {}: {}", name, e))?;
                            tensor_target.set_array_f16_with_dtype(source_array, target_dtype);
                        }
                    }

                    safetensors::Dtype::BF16 => {
                        let decoded = Self::decode_bf16_bytes(data_bytes, name)?;

                        if let Some(quantization) = target_quantization {
                            tensor_target.set_bf16_slice_with_quantization(
                                &target_shape,
                                &decoded,
                                quantization,
                            );
                        } else {
                            let source_array = Array::from_shape_vec(IxDyn(&target_shape), decoded)
                                .map_err(|e| format!("Shape mismatch for {}: {}", name, e))?;
                            tensor_target.set_array_bf16_with_dtype(source_array, target_dtype);
                        }
                    }

                    safetensors::Dtype::I8 => {
                        let i8_data = data_bytes.iter().map(|&b| b as i8).collect::<Vec<_>>();

                        let scale_name = Self::find_i8_scale_tensor_name(&tensors, name);
                        let scale = if let Some(scale_name) = scale_name.as_deref() {
                            Self::read_scalar_scale(&tensors, scale_name)?
                        } else {
                            return Err(format!(
                                "I8 tensor {} is missing required companion scale tensor (expected {}.scale or {}_scale)",
                                name, name, name
                            )
                            .into());
                        };

                        tensor_target.set_i8_slice_with_dtype(
                            &target_shape,
                            &i8_data,
                            scale,
                            target_dtype,
                        );
                    }

                    _ => {
                        return Err(
                            format!("Unsupported dtype: {:?} for {}", source_dtype, name).into(),
                        );
                    }
                }
                report.record_loaded(direct_quantized_load);
            } else {
                report.record_missing(name);
            }
        }

        report.print_summary();

        Ok(())
    }
}

#[cfg(test)]
#[path = "../tests/unit/loader/tests.rs"]
mod tests;
