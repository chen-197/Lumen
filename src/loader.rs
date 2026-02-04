use crate::autograd::Tensor;
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

                // 获取目标张量的可变引用以填充数据
                let mut target_data = tensor_target.0.borrow_mut();

                match dtype {
                    safetensors::Dtype::F32 => {
                        let f32_data: &[f32] = unsafe {
                            std::slice::from_raw_parts(
                                data_bytes.as_ptr() as *const f32,
                                data_bytes.len() / 4,
                            )
                        };

                        // 获取目标张量的当前形状
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
}
