// src/layers/linear.rs
use crate::autograd::Tensor;
use crate::init::{tensor_init, InitType}; 
use crate::module::Module;
use crate::ops::matmul::matmul;

pub struct Linear {
    pub weight: Tensor,       // shape: [out_features, in_features]
    pub bias: Option<Tensor>, // shape: [out_features]
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        // 注意：为了对齐 PyTorch/HF nn.Linear.weight 的布局，weight 存成 [out, in]
        let weight = tensor_init(vec![out_features, in_features], InitType::KaimingNormal);

        let bias = tensor_init(vec![out_features], InitType::Zeros);

        Linear {
            weight,
            bias: Some(bias),
        }
    }

    pub fn new_no_bias(in_features: usize, out_features: usize) -> Self {
        let weight = tensor_init(vec![out_features, in_features], InitType::KaimingNormal);

        Linear { weight, bias: None }
    }
}

impl Module for Linear {
    fn forward(&self, input: Tensor) -> Tensor {
        let y = matmul(&input, &self.weight);

        if let Some(bias) = &self.bias {
            y + bias.clone() // bias: [out]
        } else {
            y
        }
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = vec![self.weight.clone()];
        if let Some(b) = &self.bias {
            params.push(b.clone());
        }
        params
    }
}
