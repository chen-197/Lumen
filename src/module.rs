// src/module.rs
use crate::autograd::{Tensor, TensorRawData, set_inference_mode};
use crate::device::Device;
use crate::precision::{DType, ParameterQuantization, default_parameter_quantization};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};

#[derive(Serialize, Deserialize)]
pub struct CheckpointTensor {
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub raw: TensorRawData,
}

#[derive(Serialize, Deserialize)]
pub struct ModelCheckpoint {
    pub params: Vec<CheckpointTensor>,
}

fn write_checkpoint(mut writer: impl Write, checkpoint: &ModelCheckpoint) -> std::io::Result<()> {
    bincode::serialize_into(&mut writer, checkpoint).map_err(std::io::Error::other)?;
    writer.flush()
}

#[cfg(test)]
#[path = "../tests/unit/module/checkpoint_io_tests.rs"]
mod checkpoint_io_tests;

pub trait Module {
    fn forward(&self, input: Tensor) -> Tensor;
    fn parameters(&self) -> Vec<Tensor>;

    // 训练模式：允许构图
    fn train_mode(&mut self) {
        set_inference_mode(false);
    }

    // 推理模式：禁止构图（等价 no_grad）
    fn eval_mode(&mut self) {
        set_inference_mode(true);
    }

    fn save(&self, path: &str) -> std::io::Result<()> {
        let params = self.parameters();
        let mut data_list = Vec::new();
        for p in params {
            let (shape, dtype, raw) = p.export_raw();
            data_list.push(CheckpointTensor { shape, dtype, raw });
        }
        let checkpoint = ModelCheckpoint { params: data_list };
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        write_checkpoint(writer, &checkpoint)?;
        println!("Model saved to {} (Binary format)", path);
        Ok(())
    }

    fn load(&self, path: &str) -> std::io::Result<()> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let checkpoint: ModelCheckpoint =
            bincode::deserialize_from(reader).map_err(std::io::Error::other)?;

        let my_params = self.parameters();
        if checkpoint.params.len() != my_params.len() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "Load failed: parameter count mismatch (checkpoint={}, module={})",
                    checkpoint.params.len(),
                    my_params.len()
                ),
            ));
        }

        // Reject malformed later parameters before replacing any earlier values.
        for tensor_blob in &checkpoint.params {
            Tensor::validate_raw(&tensor_blob.shape, tensor_blob.dtype, &tensor_blob.raw)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        }
        for (param, tensor_blob) in my_params.iter().zip(checkpoint.params) {
            param
                .import_raw(tensor_blob.shape, tensor_blob.dtype, tensor_blob.raw)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        }

        println!("Model loaded from {} (Binary format)", path);
        Ok(())
    }

    fn cast_parameters(&self, dtype: DType) {
        for param in self.parameters() {
            param.cast_inplace(dtype);
        }
    }

    fn quantize_parameters(&self, dtype: DType) {
        assert!(
            dtype.is_integer(),
            "quantize_parameters currently expects integer dtype, got {:?}",
            dtype
        );
        for param in self.parameters() {
            param.quantize_inplace(dtype);
        }
    }

    fn dequantize_parameters(&self, dtype: DType) {
        assert!(
            dtype.is_float(),
            "dequantize_parameters currently expects floating dtype, got {:?}",
            dtype
        );
        for param in self.parameters() {
            param.dequantize_inplace(dtype);
        }
    }

    fn apply_parameter_quantization(&self, quantization: ParameterQuantization) {
        if !quantization.is_enabled() {
            return;
        }
        for param in self.parameters() {
            param.quantize_inplace_with_quantization(quantization);
        }
    }

    fn apply_default_parameter_quantization(&self) {
        self.apply_parameter_quantization(default_parameter_quantization());
    }

    fn to_device(&self, device: Device) {
        for param in self.parameters() {
            param.to_device_inplace(device);
        }
    }

    fn to_cpu(&self) {
        self.to_device(Device::Cpu);
    }

    fn to_cuda(&self) {
        self.to_device(Device::Cuda);
    }
}

pub struct Sequential {
    layers: Vec<Box<dyn Module>>,
}

impl Sequential {
    pub fn new(layers: Vec<Box<dyn Module>>) -> Self {
        Sequential { layers }
    }
}

impl Module for Sequential {
    fn forward(&self, mut input: Tensor) -> Tensor {
        for layer in &self.layers {
            input = layer.forward(input);
        }
        input
    }

    fn parameters(&self) -> Vec<Tensor> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }

    fn train_mode(&mut self) {
        // 先设置全局模式，再递归
        set_inference_mode(false);
        for l in &mut self.layers {
            l.train_mode();
        }
    }

    fn eval_mode(&mut self) {
        // 先设置全局模式，再递归
        set_inference_mode(true);
        for l in &mut self.layers {
            l.eval_mode();
        }
    }
}

#[macro_export]
macro_rules! sequential {
    ($($layer:expr),* $(,)?) => {
        $crate::module::Sequential::new(vec![
            $(Box::new($layer)),*
        ])
    };
}

#[cfg(test)]
#[path = "../tests/unit/module/tests.rs"]
mod tests;
