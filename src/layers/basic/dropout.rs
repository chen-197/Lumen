use crate::autograd::Tensor;
use crate::module::Module;
use ndarray::prelude::*;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Bernoulli;

pub struct Dropout { p: f32, training: bool }
impl Dropout { pub fn new(p: f32) -> Self { Dropout { p, training: true } } }
impl Module for Dropout {
    fn forward(&self, input: Tensor) -> Tensor {
        if self.training && self.p > 0.0 {
            let shape = input.data().shape().to_vec();
            let dist = Bernoulli::new(1.0 - self.p as f64).unwrap();
            let mask_arr = Array::random(ndarray::IxDyn(&shape), dist).mapv(|x| if x { 1.0f32 } else { 0.0f32 });
            let scale = 1.0 / (1.0 - self.p);
            let mask = mask_arr * scale;
            let mask_tensor = Tensor::new(mask.into_dyn());
            input * mask_tensor 
        } else { input }
    }
    fn parameters(&self) -> Vec<Tensor> { vec![] }
    fn train_mode(&mut self) { self.training = true; }
    fn eval_mode(&mut self) { self.training = false; }
}