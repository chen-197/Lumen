use super::*;

impl Tensor {
    pub fn sum(&self) -> Tensor {
        crate::ops::arithmetic::sum(self)
    }

    pub fn reshape(&self, shape: Vec<i32>) -> Tensor {
        crate::ops::shape::reshape(self, shape)
    }

    pub fn permute(&self, axes: Vec<usize>) -> Tensor {
        crate::ops::shape::permute(self, axes)
    }

    pub fn transpose(&self, dim0: usize, dim1: usize) -> Tensor {
        let ndim = self.ndim();
        let mut axes: Vec<usize> = (0..ndim).collect();
        axes.swap(dim0, dim1);
        self.permute(axes)
    }
}
