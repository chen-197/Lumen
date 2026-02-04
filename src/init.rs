use crate::autograd::Tensor;
use ndarray::ArrayD;
use ndarray_rand::RandomExt; 
use ndarray_rand::rand_distr::{Normal, Uniform};

pub enum InitType {
    XavierUniform, // For Tanh/Sigmoid (Glorot)
    KaimingNormal, // For ReLU/GELU (He)
    Zeros,         // For Bias
    Ones,          // For LayerNorm/RMSNorm weights
}

pub fn tensor_init(shape: Vec<usize>, init_type: InitType) -> Tensor {
    let shape_ref = shape.as_slice();

    let data = match init_type {
        InitType::Zeros => ArrayD::zeros(shape_ref),
        
        InitType::Ones => ArrayD::ones(shape_ref),
        
        InitType::XavierUniform => {
            let fan_in = shape[0] as f32;
            let fan_out = if shape.len() > 1 { shape[1] } else { shape[0] } as f32;
            let limit = (6.0 / (fan_in + fan_out)).sqrt();
            ArrayD::random(shape_ref, Uniform::new(-limit, limit))
        },
        
        InitType::KaimingNormal => {
            let fan_in = shape[0] as f32;
            let std = (2.0 / fan_in).sqrt();
            ArrayD::random(shape_ref, Normal::new(0.0, std).unwrap())
        }
    };
    
    Tensor::new(data)
}