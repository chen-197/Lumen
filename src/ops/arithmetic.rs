// src/ops/arithmetic.rs
use crate::autograd::{is_no_grad, Tensor, TensorData};
use ndarray::{ArrayD, ArrayViewD, Zip};
use std::cell::RefCell;
use std::ops::{Add, Mul, Sub};
use std::rc::Rc;

fn reduce_gradient(grad: ArrayViewD<'_, f32>, target_shape: &[usize]) -> ArrayD<f32> {
    if grad.shape() == target_shape {
        return grad.to_owned().into_dyn();
    }

    let mut res = grad.to_owned().into_dyn();
    let g_ndim = res.ndim();
    let t_ndim = target_shape.len();

    if g_ndim > t_ndim {
        for _ in 0..(g_ndim - t_ndim) {
            res = res.sum_axis(ndarray::Axis(0));
        }
    }

    for i in 0..res.ndim() {
        if target_shape[i] == 1 && res.shape()[i] > 1 {
            let summed = res.sum_axis(ndarray::Axis(i));
            res = summed.insert_axis(ndarray::Axis(i));
        } else if target_shape[i] != res.shape()[i] {
            panic!(
                "Gradient shape mismatch. Grad: {:?}, Target: {:?}",
                grad.shape(),
                target_shape
            );
        }
    }

    if res.shape() != target_shape {
        if res.len() == target_shape.iter().product::<usize>() {
            return res.into_shape(target_shape).unwrap();
        }
        panic!("Reduction failed.");
    }

    res
}

impl Add for Tensor {
    type Output = Tensor;
    fn add(self, rhs: Tensor) -> Tensor {
        let data = (&*self.data_ref() + &*rhs.data_ref()).into_dyn();
        let build_graph = !is_no_grad() && (self.requires_grad() || rhs.requires_grad());

        if !build_graph {
            return Tensor::from_array_no_grad(data);
        }

        let lhs = self.clone();
        let rhs = rhs.clone();

        Tensor(Rc::new(RefCell::new(TensorData {
            data: data.into_shared(),
            grad: None,
            parents: vec![self.clone(), rhs.clone()],
            backward_op: Some(std::rc::Rc::new(move |grad| {
                let l_shape = lhs.data_ref().shape().to_vec();
                let r_shape = rhs.data_ref().shape().to_vec();
                lhs.add_grad(reduce_gradient(grad.view(), &l_shape));
                rhs.add_grad(reduce_gradient(grad.view(), &r_shape));
            })),
            requires_grad: true,
        })))
    }
}
impl<'a, 'b> Add<&'b Tensor> for &'a Tensor {
    type Output = Tensor;
    fn add(self, rhs: &'b Tensor) -> Tensor {
        self.clone() + rhs.clone()
    }
}

impl Sub for Tensor {
    type Output = Tensor;
    fn sub(self, rhs: Tensor) -> Tensor {
        let data = (&*self.data_ref() - &*rhs.data_ref()).into_dyn();
        let build_graph = !is_no_grad() && (self.requires_grad() || rhs.requires_grad());

        if !build_graph {
            return Tensor::from_array_no_grad(data);
        }

        let lhs = self.clone();
        let rhs = rhs.clone();

        Tensor(Rc::new(RefCell::new(TensorData {
            data: data.into_shared(),
            grad: None,
            parents: vec![self.clone(), rhs.clone()],
            backward_op: Some(std::rc::Rc::new(move |grad| {
                let l_shape = lhs.data_ref().shape().to_vec();
                let r_shape = rhs.data_ref().shape().to_vec();
                lhs.add_grad(reduce_gradient(grad.view(), &l_shape));

                let grad_neg = Zip::from(grad).par_map_collect(|&x| -x);
                rhs.add_grad(reduce_gradient(grad_neg.view(), &r_shape));
            })),
            requires_grad: true,
        })))
    }
}
impl<'a, 'b> Sub<&'b Tensor> for &'a Tensor {
    type Output = Tensor;
    fn sub(self, rhs: &'b Tensor) -> Tensor {
        self.clone() - rhs.clone()
    }
}

impl Mul for Tensor {
    type Output = Tensor;
    fn mul(self, rhs: Tensor) -> Tensor {
        let data = (&*self.data_ref() * &*rhs.data_ref()).into_dyn();
        let build_graph = !is_no_grad() && (self.requires_grad() || rhs.requires_grad());

        if !build_graph {
            return Tensor::from_array_no_grad(data);
        }

        let lhs = self.clone();
        let rhs = rhs.clone();

        Tensor(Rc::new(RefCell::new(TensorData {
            data: data.into_shared(),
            grad: None,
            parents: vec![self.clone(), rhs.clone()],
            backward_op: Some(std::rc::Rc::new(move |grad| {
                let (g_lhs, g_rhs, l_shape, r_shape) = {
                    let a_data = lhs.data_ref();
                    let b_data = rhs.data_ref();

                    let (g_lhs, g_rhs) = if grad.shape() == a_data.shape() && grad.shape() == b_data.shape() {
                        let gl = Zip::from(grad).and(&*b_data).par_map_collect(|&g, &b| g * b);
                        let gr = Zip::from(grad).and(&*a_data).par_map_collect(|&g, &a| g * a);
                        (gl, gr)
                    } else {
                        (grad * &*b_data, grad * &*a_data)
                    };

                    (g_lhs, g_rhs, a_data.shape().to_vec(), b_data.shape().to_vec())
                };
                lhs.add_grad(reduce_gradient(g_lhs.view(), &l_shape));
                rhs.add_grad(reduce_gradient(g_rhs.view(), &r_shape));
            })),
            requires_grad: true,
        })))
    }
}
impl<'a, 'b> Mul<&'b Tensor> for &'a Tensor {
    type Output = Tensor;
    fn mul(self, rhs: &'b Tensor) -> Tensor {
        self.clone() * rhs.clone()
    }
}

pub fn sum(input: &Tensor) -> Tensor {
    let sum_val = input.data_ref().sum();
    let result = ndarray::arr0(sum_val).into_dyn();
    let build_graph = !is_no_grad() && input.requires_grad();

    if !build_graph {
        return Tensor::from_array_no_grad(result);
    }

    let input_clone = input.clone();

    Tensor(Rc::new(RefCell::new(TensorData {
        data: result.into_shared(),
        grad: None,
        parents: vec![input.clone()],
        backward_op: Some(std::rc::Rc::new(move |grad| {
            let g = grad.first().copied().unwrap_or(0.0);
            let input_shape = input_clone.data_ref().shape().to_vec();
            let grad_input = ndarray::ArrayD::from_elem(input_shape, g);
            input_clone.add_grad(grad_input);
        })),
        requires_grad: true,
    })))
}
