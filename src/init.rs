use crate::autograd::Tensor;
use ndarray::{Array, ArrayD, IxDyn};
use ndarray_rand::rand_distr::{Normal, Uniform};
use ndarray_rand::RandomExt;

pub enum InitType {
    XavierUniform, // For Tanh/Sigmoid (Glorot)
    KaimingNormal, // For ReLU/GELU (He)
    Zeros,         // For Bias
    Ones,          // For LayerNorm/RMSNorm weights
}

pub fn tensor_init(shape: Vec<usize>, init_type: InitType) -> Tensor {
    let shape_dyn = IxDyn(shape.as_slice());

    let data = match init_type {
        InitType::Zeros => ArrayD::zeros(shape_dyn.clone()),
        
        InitType::Ones => ArrayD::ones(shape_dyn.clone()),
        
        InitType::XavierUniform => {
            let fan_in = shape[0] as f32;
            let fan_out = if shape.len() > 1 { shape[1] } else { shape[0] } as f32;
            let limit = (6.0 / (fan_in + fan_out)).sqrt();
            // ndarray 0.16: RandomExt is implemented on Array, not ArrayD::random.
            Array::random(shape_dyn.clone(), Uniform::new(-limit, limit)).into_dyn()
        },
        
        InitType::KaimingNormal => {
            let fan_in = shape[0] as f32;
            let std = (2.0 / fan_in).sqrt();
            Array::random(shape_dyn, Normal::new(0.0, std).unwrap()).into_dyn()
        }
    };
    
    Tensor::new(data)
}