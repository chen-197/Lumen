use crate::autograd::Tensor;
use crate::module::Module;
use crate::ops::shape::reshape;
pub struct Flatten;
impl Flatten { pub fn new() -> Self { Flatten } }
impl Module for Flatten {
    fn forward(&self, input: Tensor) -> Tensor {
        // Avoid cloning the full tensor data just to read the shape.
        let b = input.data_ref().shape()[0];
        reshape(&input, vec![b as i32, -1])
    }
    fn parameters(&self) -> Vec<Tensor> { vec![] }
}