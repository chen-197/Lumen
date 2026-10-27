use crate::autograd::Tensor;
use crate::module::Module;
use ndarray::prelude::*;
use ndarray_rand::RandomExt;
use ndarray_rand::rand::{SeedableRng, rngs::StdRng};
use ndarray_rand::rand_distr::Bernoulli;

pub struct Dropout {
    p: f32,
    training: bool,
    seed: Option<u64>,
}
impl Dropout {
    pub fn new(p: f32) -> Self {
        assert!(
            (0.0..1.0).contains(&p),
            "Dropout probability must be in [0, 1), got {p}"
        );
        Dropout {
            p,
            training: true,
            seed: None,
        }
    }

    pub fn new_with_seed(p: f32, seed: u64) -> Self {
        let mut dropout = Self::new(p);
        dropout.seed = Some(seed);
        dropout
    }
}
impl Module for Dropout {
    fn forward(&self, input: Tensor) -> Tensor {
        if self.training && self.p > 0.0 {
            let shape = input.shape_vec();
            let dist = Bernoulli::new(1.0 - self.p as f64).unwrap();
            let mask_arr = if let Some(seed) = self.seed {
                let mut rng = StdRng::seed_from_u64(seed);
                Array::random_using(ndarray::IxDyn(&shape), dist, &mut rng)
            } else {
                Array::random(ndarray::IxDyn(&shape), dist)
            }
            .mapv(|x| if x { 1.0f32 } else { 0.0f32 });
            let scale = 1.0 / (1.0 - self.p);
            let mask = mask_arr * scale;
            let mask_tensor =
                Tensor::from_data_with_grad_flag(mask.into_dyn(), false).to_device(input.device());
            input * mask_tensor
        } else {
            input
        }
    }
    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }
    fn train_mode(&mut self) {
        self.training = true;
    }
    fn eval_mode(&mut self) {
        self.training = false;
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/layers/basic/dropout/tests.rs"]
mod tests;
