use crate::autograd::Tensor;
use crate::module::Module;
use crate::ops::shape::reshape;
pub struct Flatten;
impl Flatten {
    pub fn new() -> Self {
        Flatten
    }
}

impl Default for Flatten {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for Flatten {
    fn forward(&self, input: Tensor) -> Tensor {
        let shape = input.shape_vec();
        assert!(!shape.is_empty(), "Flatten expects at least 1D input");
        // Derive the trailing size from dimensions, including empty batches.
        // With batch zero, reshape([0, -1]) cannot infer that size from len().
        let features = shape[1..]
            .iter()
            .try_fold(1usize, |size, &dim| size.checked_mul(dim))
            .expect("Flatten feature dimension overflow");
        let batch = i32::try_from(shape[0]).expect("Flatten batch dimension exceeds i32");
        let features = i32::try_from(features).expect("Flatten feature dimension exceeds i32");
        reshape(&input, vec![batch, features])
    }
    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/layers/basic/flatten/tests.rs"]
mod tests;
