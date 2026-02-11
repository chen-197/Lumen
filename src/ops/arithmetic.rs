// src/ops/arithmetic.rs
use crate::autograd::{Tensor, TensorData};
use std::ops::{Add, Sub, Mul};
use std::rc::Rc;
use std::cell::RefCell;
use ndarray::{ArrayD, ArrayViewD, Zip};

fn reduce_gradient(grad: ArrayViewD<'_, f32>, target_shape: &[usize]) -> ArrayD<f32> {
    if grad.shape() == target_shape {
        return grad.to_owned().into_dyn();
    }

    // 将 view materialize 成 owned，然后按旧逻辑做 reduce/broadcast
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
            // 这里继续使用 into_shape()，保持现有行为
            return res.into_shape(target_shape).unwrap();
        }
        panic!("Reduction failed.");
    }

    res
}

impl Add for Tensor {
    type Output = Tensor;
    fn add(self, rhs: Tensor) -> Tensor {
        let data = &*self.data_ref() + &*rhs.data_ref();
        
        let lhs = self.clone();
        let rhs = rhs.clone();

        Tensor(Rc::new(RefCell::new(TensorData {
            data: data.into_shared(),
            grad: None,
            parents: vec![self.clone(), rhs.clone()],
            backward_op: Some(Box::new(move |grad| {
                let l_shape = lhs.data_ref().shape().to_vec();
                let r_shape = rhs.data_ref().shape().to_vec();
                lhs.add_grad(reduce_gradient(grad.view(), &l_shape));
                rhs.add_grad(reduce_gradient(grad.view(), &r_shape));
            })),
            requires_grad: true
        })))
    }
}
impl<'a, 'b> Add<&'b Tensor> for &'a Tensor {
    type Output = Tensor;
    fn add(self, rhs: &'b Tensor) -> Tensor { self.clone() + rhs.clone() }
}

impl Sub for Tensor {
    type Output = Tensor;
    fn sub(self, rhs: Tensor) -> Tensor {
        let data = &*self.data_ref() - &*rhs.data_ref();
        
        let lhs = self.clone();
        let rhs = rhs.clone();

        Tensor(Rc::new(RefCell::new(TensorData {
            data: data.into_shared(),
            grad: None,
            parents: vec![self.clone(), rhs.clone()],
            backward_op: Some(Box::new(move |grad| {
                let l_shape = lhs.data_ref().shape().to_vec();
                let r_shape = rhs.data_ref().shape().to_vec();
                lhs.add_grad(reduce_gradient(grad.view(), &l_shape));
                
                // 并行取反
                let grad_neg = Zip::from(grad).par_map_collect(|&x| -x);
                rhs.add_grad(reduce_gradient(grad_neg.view(), &r_shape));
            })),
            requires_grad: true
        })))
    }
}
impl<'a, 'b> Sub<&'b Tensor> for &'a Tensor {
    type Output = Tensor;
    fn sub(self, rhs: &'b Tensor) -> Tensor { self.clone() - rhs.clone() }
}

impl Mul for Tensor {
    type Output = Tensor;
    fn mul(self, rhs: Tensor) -> Tensor {
        let data = &*self.data_ref() * &*rhs.data_ref();
        
        let lhs = self.clone();
        let rhs = rhs.clone();

        Tensor(Rc::new(RefCell::new(TensorData {
            data: data.into_shared(),
            grad: None,
            parents: vec![self.clone(), rhs.clone()],
            backward_op: Some(Box::new(move |grad| {
                // Backward 这里也需要用 data_ref()，否则求导时也会拷贝
                let a_data = lhs.data_ref();
                let b_data = rhs.data_ref();
                
                // 只有 shape 完全一致才用 Zip，否则广播乘法
                // Zip::from 要求 ArrayBase，Ref<ArrayD> 解引用后即可
                let (g_lhs, g_rhs) = if grad.shape() == a_data.shape() && grad.shape() == b_data.shape() {
                    let gl = Zip::from(grad).and(&*b_data).par_map_collect(|&g, &b| g * b);
                    let gr = Zip::from(grad).and(&*a_data).par_map_collect(|&g, &a| g * a);
                    (gl, gr)
                } else {
                    (grad * &*b_data, grad * &*a_data)
                };

                let l_shape = a_data.shape().to_vec();
                let r_shape = b_data.shape().to_vec();
                lhs.add_grad(reduce_gradient(g_lhs.view(), &l_shape));
                rhs.add_grad(reduce_gradient(g_rhs.view(), &r_shape));
            })),
            requires_grad: true
        })))
    }
}
impl<'a, 'b> Mul<&'b Tensor> for &'a Tensor {
    type Output = Tensor;
    fn mul(self, rhs: &'b Tensor) -> Tensor { self.clone() * rhs.clone() }
}
pub fn sum(input: &Tensor) -> Tensor {
    let input_data = input.data_ref();
    let sum_val = input_data.sum();
    
    // 结果是一个 0 维标量 Tensor
    let result = ndarray::arr0(sum_val).into_dyn();

    let input_clone = input.clone();
    
    Tensor(Rc::new(RefCell::new(TensorData {
        data: result.into_shared(),
        grad: None,
        parents: vec![input.clone()],
        backward_op: Some(Box::new(move |grad| {
            // Grad 是标量 (0-dim)
            let g = grad.first().copied().unwrap_or(0.0);
            
            // Backward: 将标量梯度广播到输入的形状
            // dL/dx = dL/dSum * 1
            let input_shape = input_clone.data_ref().shape().to_vec();
            let grad_input = ndarray::ArrayD::from_elem(input_shape, g);
            
            input_clone.add_grad(grad_input);
        })),
        requires_grad: true
    })))
}