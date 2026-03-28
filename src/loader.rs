use crate::autograd::Tensor;
use crate::infer_mode::is_pure_bf16_infer_loading;
use crate::models::LlamaModel;
use half::bf16;
use memmap2::MmapOptions;
use safetensors::SafeTensors;
use std::fs::File;
use std::path::Path;

pub struct ModelLoader;

impl ModelLoader {
    pub fn load_llama_weights<P: AsRef<Path>>(
        path: P,
        model_params: &std::collections::HashMap<String, Tensor>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        let tensors = SafeTensors::deserialize(&mmap)?;

        println!("--- Loading Weights ---");

        for (name, tensor_target) in model_params {
            if let Ok(view) = tensors.tensor(name) {
                let dtype = view.dtype();
                let data_bytes = view.data();
                let mut target_data = tensor_target.0.borrow_mut();

                match dtype {
                    safetensors::Dtype::F32 => {
                        let f32_data: &[f32] = unsafe {
                            std::slice::from_raw_parts(
                                data_bytes.as_ptr() as *const f32,
                                data_bytes.len() / 4,
                            )
                        };

                        let target_shape = target_data.data.shape().to_vec();
                        let source_array =
                            ndarray::Array::from_shape_vec(target_shape, f32_data.to_vec())
                                .map_err(|e| format!("Shape mismatch for {}: {}", name, e))?;

                        target_data.data.assign(&source_array.into_dyn());
                    }
                    safetensors::Dtype::BF16 => {
                        let bf16_data: &[bf16] = unsafe {
                            std::slice::from_raw_parts(
                                data_bytes.as_ptr() as *const bf16,
                                data_bytes.len() / 2,
                            )
                        };

                        let f32_vec: Vec<f32> = bf16_data.iter().map(|&x| x.to_f32()).collect();
                        let target_shape = target_data.data.shape().to_vec();

                        let source_array = ndarray::Array::from_shape_vec(target_shape, f32_vec)
                            .map_err(|e| format!("Shape mismatch for {}: {}", name, e))?;

                        target_data.data.assign(&source_array.into_dyn());
                    }
                    _ => return Err(format!("Unsupported dtype: {:?} for {}", dtype, name).into()),
                }

                println!("✅ Loaded: {}", name);
            } else {
                println!(
                    "⚠️ Warning: Parameter {} not found in safetensors file",
                    name
                );
            }
        }

        Ok(())
    }

    pub fn load_llama_weights_direct<P: AsRef<Path>>(
        path: P,
        model: &mut LlamaModel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        let tensors = SafeTensors::deserialize(&mmap)?;
        let named_params = model.named_parameters();

        println!("--- Loading Weights ---");

        for name in tensors.names() {
            let view = tensors.tensor(name)?;
            let dtype = view.dtype();
            let shape = view.shape().to_vec();
            let data_bytes = view.data();

            let is_big_mat = name == "model.embed_tokens.weight"
                || name == "lm_head.weight"
                || name.contains(".self_attn.q_proj.weight")
                || name.contains(".self_attn.k_proj.weight")
                || name.contains(".self_attn.v_proj.weight")
                || name.contains(".self_attn.o_proj.weight")
                || name.contains(".mlp.gate_proj.weight")
                || name.contains(".mlp.up_proj.weight")
                || name.contains(".mlp.down_proj.weight");

            if is_pure_bf16_infer_loading() && is_big_mat {
                let bf16_bits: Vec<u16> = match dtype {
                    safetensors::Dtype::BF16 => {
                        let src: &[u16] = unsafe {
                            std::slice::from_raw_parts(
                                data_bytes.as_ptr() as *const u16,
                                data_bytes.len() / 2,
                            )
                        };
                        src.to_vec()
                    }
                    safetensors::Dtype::F32 => {
                        let src: &[f32] = unsafe {
                            std::slice::from_raw_parts(
                                data_bytes.as_ptr() as *const f32,
                                data_bytes.len() / 4,
                            )
                        };
                        src.iter().map(|&x| bf16::from_f32(x).to_bits()).collect()
                    }
                    _ => {
                        return Err(
                            format!("Unsupported dtype: {:?} for {}", dtype, name).into(),
                        )
                    }
                };

                model
                    .load_named_bf16_infer_weight_direct(name, shape, bf16_bits)
                    .map_err(|e| format!("failed direct BF16 infer load for {}: {}", name, e))?;

                println!("✅ Loaded direct BF16 infer: {}", name);
                continue;
            }

            if let Some(tensor_target) = named_params.get(name) {
                let mut target_data = tensor_target.0.borrow_mut();

                match dtype {
                    safetensors::Dtype::F32 => {
                        let f32_data: &[f32] = unsafe {
                            std::slice::from_raw_parts(
                                data_bytes.as_ptr() as *const f32,
                                data_bytes.len() / 4,
                            )
                        };

                        let target_shape = target_data.data.shape().to_vec();
                        let source_array =
                            ndarray::Array::from_shape_vec(target_shape, f32_data.to_vec())
                                .map_err(|e| format!("Shape mismatch for {}: {}", name, e))?;

                        target_data.data.assign(&source_array.into_dyn());
                    }
                    safetensors::Dtype::BF16 => {
                        let bf16_data: &[bf16] = unsafe {
                            std::slice::from_raw_parts(
                                data_bytes.as_ptr() as *const bf16,
                                data_bytes.len() / 2,
                            )
                        };

                        let f32_vec: Vec<f32> = bf16_data.iter().map(|&x| x.to_f32()).collect();
                        let target_shape = target_data.data.shape().to_vec();

                        let source_array = ndarray::Array::from_shape_vec(target_shape, f32_vec)
                            .map_err(|e| format!("Shape mismatch for {}: {}", name, e))?;

                        target_data.data.assign(&source_array.into_dyn());
                    }
                    _ => return Err(format!("Unsupported dtype: {:?} for {}", dtype, name).into()),
                }

                println!("✅ Loaded normal: {}", name);
            } else {
                println!("⚠️ Warning: Parameter {} not found in model", name);
            }
        }

        Ok(())
    }
}
