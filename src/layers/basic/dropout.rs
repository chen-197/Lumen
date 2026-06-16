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
mod tests {
    use super::*;
    #[cfg(feature = "cuda")]
    use crate::autograd::{Tensor, set_strict_device_execution};
    use ndarray::{Array, IxDyn};

    #[test]
    #[should_panic(expected = "Dropout probability must be in [0, 1)")]
    fn dropout_rejects_probability_one() {
        let _ = Dropout::new(1.0);
    }

    #[test]
    fn seeded_dropout_reuses_same_mask() {
        let dropout = Dropout::new_with_seed(0.4, 42);
        let input = Tensor::from_data_with_grad_flag(
            Array::from_shape_vec(
                IxDyn(&[2, 5]),
                vec![0.5, -0.3, 0.8, 1.1, -1.4, 0.7, 0.2, -0.9, 1.3, -0.6],
            )
            .expect("input shape mismatch")
            .into_dyn(),
            false,
        );

        let first = dropout.forward(input.clone());
        let second = dropout.forward(input);
        assert_eq!(first.data_ref().as_slice(), second.data_ref().as_slice());
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_dropout_backward_keeps_input_grad_resident_in_strict_mode() {
        if !crate::ops::cuda::is_available() {
            return;
        }

        let dropout = Dropout::new(0.5);
        let input_cpu = Tensor::from_data_with_grad_flag(
            Array::from_shape_vec(IxDyn(&[2, 4]), (0..8).map(|i| i as f32 - 3.0).collect())
                .expect("input shape mismatch")
                .into_dyn(),
            true,
        );
        let coeff_cpu = Tensor::from_data_with_grad_flag(
            Array::from_shape_vec(
                IxDyn(&[2, 4]),
                vec![0.25, -0.5, 0.75, -1.0, 1.25, -1.5, 1.75, -2.0],
            )
            .expect("coeff shape mismatch")
            .into_dyn(),
            false,
        );
        let input_cuda = input_cpu.to_cuda();
        let coeff_cuda = coeff_cpu.to_cuda();

        crate::ops::cuda::set_enabled(true);
        set_strict_device_execution(true);
        let out = dropout.forward(input_cuda.clone());
        assert!(out.is_cuda());
        let loss = crate::ops::arithmetic::sum(&(&out * &coeff_cuda));
        loss.backward();
        assert!(input_cuda.cloned_cuda_f32_grad().is_some());
        set_strict_device_execution(false);
        crate::ops::cuda::set_enabled(false);

        let grad = input_cuda.grad().expect("cuda dropout input grad");
        let coeff = coeff_cpu.data_ref();
        for (got, expect_coeff) in grad.iter().zip(coeff.iter()) {
            let scaled = expect_coeff * 2.0;
            assert!(
                got.abs() < 1e-6 || (got - scaled).abs() < 1e-6,
                "dropout grad should be 0 or scaled coeff, got {got}, coeff {expect_coeff}"
            );
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_dropout_bf16_input_keeps_native_storage_and_f32_grad_in_strict_mode() {
        if !crate::ops::cuda::is_available() {
            return;
        }

        let dropout = Dropout::new(0.5);
        let input_cuda = Tensor::parameter_with_dtype(
            Array::from_shape_vec(IxDyn(&[2, 4]), (0..8).map(|i| i as f32 - 3.0).collect())
                .expect("input shape mismatch")
                .into_dyn(),
            crate::precision::DType::BF16,
        )
        .to_cuda();
        let coeff_cuda = Tensor::from_data_with_grad_flag(
            Array::from_shape_vec(
                IxDyn(&[2, 4]),
                vec![0.25, -0.5, 0.75, -1.0, 1.25, -1.5, 1.75, -2.0],
            )
            .expect("coeff shape mismatch")
            .into_dyn(),
            false,
        )
        .to_cuda();
        assert!(input_cuda.0.borrow().cuda_f32_data.is_none());
        assert!(input_cuda.0.borrow().cuda_bf16_data.is_some());

        crate::ops::cuda::set_enabled(true);
        set_strict_device_execution(true);
        let out = dropout.forward(input_cuda.clone());
        assert!(out.is_cuda());
        assert!(input_cuda.0.borrow().cuda_f32_data.is_none());
        let loss = crate::ops::arithmetic::sum(&(&out * &coeff_cuda));
        loss.backward();
        assert!(input_cuda.0.borrow().cuda_f32_data.is_none());
        assert!(input_cuda.0.borrow().cuda_bf16_data.is_some());
        assert!(input_cuda.cloned_cuda_f32_grad().is_some());
        assert!(!input_cuda.has_host_grad());
        set_strict_device_execution(false);
        crate::ops::cuda::set_enabled(false);
    }
}
