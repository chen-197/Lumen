// src/module.rs
use crate::autograd::{Tensor, set_inference_mode};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter};

#[derive(Serialize, Deserialize)]
pub struct ModelCheckpoint {
    pub params: Vec<(Vec<usize>, Vec<f32>)>,
}

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
            data_list.push(p.get_raw_data());
        }
        let checkpoint = ModelCheckpoint { params: data_list };
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        bincode::serialize_into(writer, &checkpoint)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        println!("Model saved to {} (Binary format)", path);
        Ok(())
    }

    fn load(&self, path: &str) -> std::io::Result<()> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let checkpoint: ModelCheckpoint = bincode::deserialize_from(reader)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

        let my_params = self.parameters();
        if checkpoint.params.len() != my_params.len() {
            panic!("Load failed: Parameter count mismatch.");
        }

        for (param, (shape, data)) in my_params.iter().zip(checkpoint.params.into_iter()) {
            param.set_raw_data(shape, data);
        }

        println!("Model loaded from {} (Binary format)", path);
        Ok(())
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
