// src/autograd.rs
use ndarray::prelude::*;
use std::cell::{Ref, RefCell, RefMut};
use std::collections::HashSet;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

static NO_GRAD_DEPTH: AtomicUsize = AtomicUsize::new(0);
static INFERENCE_MODE: AtomicBool = AtomicBool::new(false);

pub struct NoGradGuard {
    _priv: (),
}

impl NoGradGuard {
    pub fn enter() -> Self {
        NO_GRAD_DEPTH.fetch_add(1, Ordering::Relaxed);
        Self { _priv: () }
    }
}

impl Drop for NoGradGuard {
    fn drop(&mut self) {
        NO_GRAD_DEPTH.fetch_sub(1, Ordering::Relaxed);
    }
}

/// 开/关 全局推理模式（eval_mode/train_mode 可调用它）
pub fn set_inference_mode(on: bool) {
    INFERENCE_MODE.store(on, Ordering::Relaxed);
}

#[inline]
pub fn is_inference_mode() -> bool {
    INFERENCE_MODE.load(Ordering::Relaxed)
}

/// no_grad 的判定：
/// - 在 NoGradGuard 作用域内为 true
/// - 或者处于 inference_mode 为 true
#[inline]
pub fn is_no_grad() -> bool {
    NO_GRAD_DEPTH.load(Ordering::Relaxed) > 0 || is_inference_mode()
}

/// 便利封装：no_grad(|| { ... })
pub fn no_grad<R>(f: impl FnOnce() -> R) -> R {
    let _g = NoGradGuard::enter();
    f()
}

pub struct TensorData {
    pub data: ArrayD<f32>,
    pub grad: Option<ArrayD<f32>>,
    pub parents: Vec<Tensor>,
    pub backward_op: Option<Box<dyn Fn(&ArrayD<f32>)>>,
    pub requires_grad: bool,
}

#[derive(Clone)]
pub struct Tensor(pub(crate) Rc<RefCell<TensorData>>);

impl Tensor {
    /// 默认构造叶子张量：
    /// - 推理模式/no_grad 下：requires_grad=false
    /// - 否则：requires_grad=true（更适合训练时手工造张量）
    pub fn new(data: ArrayD<f32>) -> Self {
        let req = !is_no_grad();
        Tensor(Rc::new(RefCell::new(TensorData {
            data,
            grad: None,
            parents: Vec::new(),
            backward_op: None,
            requires_grad: req,
        })))
    }

    /// 获取数据的只读引用（零拷贝）
    pub fn data_ref(&self) -> Ref<'_, ArrayD<f32>> {
        let borrow = self.0.borrow();
        Ref::map(borrow, |t| &t.data)
    }

    /// 获取梯度的只读引用（零拷贝）
    pub fn grad_ref(&self) -> Ref<'_, Option<ArrayD<f32>>> {
        let borrow = self.0.borrow();
        Ref::map(borrow, |t| &t.grad)
    }

    /// 获取数据的可变引用
    pub fn data_mut(&self) -> RefMut<'_, ArrayD<f32>> {
        let borrow = self.0.borrow_mut();
        RefMut::map(borrow, |t| &mut t.data)
    }

    /// 获取梯度的可变引用
    pub fn grad_mut(&self) -> RefMut<'_, Option<ArrayD<f32>>> {
        let borrow = self.0.borrow_mut();
        RefMut::map(borrow, |t| &mut t.grad)
    }

    pub fn data(&self) -> ArrayD<f32> {
        self.0.borrow().data.clone()
    }

    pub fn grad(&self) -> Option<ArrayD<f32>> {
        self.0.borrow().grad.clone()
    }

    pub fn sum(&self) -> Tensor {
        crate::ops::arithmetic::sum(self)
    }

    /// 创建叶子张量（显式指定 requires_grad）
    pub fn from_data_with_grad_flag(data: ArrayD<f32>, requires_grad: bool) -> Tensor {
        Tensor(Rc::new(RefCell::new(TensorData {
            data,
            grad: None,
            parents: vec![],
            backward_op: None,
            requires_grad,
        })))
    }

    /// 创建叶子张量：根据 is_no_grad() 自动决定 requires_grad
    pub fn from_data(data: ArrayD<f32>) -> Tensor {
        let req = !is_no_grad();
        Tensor::from_data_with_grad_flag(data, req)
    }

    /// 推理/常量：不需要梯度
    pub fn from_data_no_grad(data: ArrayD<f32>) -> Tensor {
        Tensor::from_data_with_grad_flag(data, false)
    }

    /// 训练参数：需要梯度（叶子）
    pub fn parameter(data: ArrayD<f32>) -> Tensor {
        Tensor::from_data_with_grad_flag(data, true)
    }

    #[inline]
    pub fn requires_grad(&self) -> bool {
        self.0.borrow().requires_grad
    }

    pub fn zero_grad(&self) {
        self.0.borrow_mut().grad = None;
    }

    pub fn reshape(&self, shape: Vec<i32>) -> Tensor {
        crate::ops::shape::reshape(self, shape)
    }

    pub fn permute(&self, axes: Vec<usize>) -> Tensor {
        crate::ops::shape::permute(self, axes)
    }

    pub fn transpose(&self, dim0: usize, dim1: usize) -> Tensor {
        let ndim = self.data_ref().ndim();
        let mut axes: Vec<usize> = (0..ndim).collect();
        axes.swap(dim0, dim1);
        self.permute(axes)
    }

    pub fn add_grad(&self, grad: ArrayD<f32>) {
        let mut inner = self.0.borrow_mut();

        if inner.data.shape() != grad.shape() {
            panic!(
                "CRITICAL: Gradient shape mismatch!\nParameter Shape: {:?}\nGradient Shape: {:?}\nHint: Check ops/arithmetic.rs reduce_gradient logic.",
                inner.data.shape(),
                grad.shape()
            );
        }

        if let Some(existing) = &inner.grad {
            inner.grad = Some(existing + &grad);
        } else {
            inner.grad = Some(grad);
        }
    }

    pub fn backward(&self) {
        let mut topo = Vec::new();
        let mut visited = HashSet::new();

        fn build_topo(
            node: &Tensor,
            topo: &mut Vec<Tensor>,
            visited: &mut HashSet<*const TensorData>,
        ) {
            let ptr = node.0.as_ptr() as *const TensorData;
            if visited.contains(&ptr) {
                return;
            }
            visited.insert(ptr);

            for parent in &node.0.borrow().parents {
                build_topo(parent, topo, visited);
            }
            topo.push(node.clone());
        }

        build_topo(self, &mut topo, &mut visited);

        self.add_grad(ArrayD::ones(self.data().shape()));

        for node in topo.iter().rev() {
            let inner = node.0.borrow();
            if let Some(grad) = &inner.grad {
                if let Some(op) = &inner.backward_op {
                    op(grad);
                }
            }
        }
    }

    pub fn get_raw_data(&self) -> (Vec<usize>, Vec<f32>) {
        let inner = self.0.borrow();
        (
            inner.data.shape().to_vec(),
            inner.data.iter().cloned().collect(),
        )
    }

    pub fn set_raw_data(&self, shape: Vec<usize>, raw_data: Vec<f32>) {
        let new_data = Array::from_shape_vec(shape, raw_data).unwrap().into_dyn();
        self.0.borrow_mut().data = new_data;
    }

    /// detach：返回一个新 Tensor（数据拷贝），requires_grad=false，且无 parents/backward_op
    pub fn detach(&self) -> Tensor {
        let d = self.0.borrow().data.clone();
        Tensor::from_data_with_grad_flag(d, false)
    }
}

/// 切断梯度流（等价于 t.detach()）
pub fn detach(t: &Tensor) -> Tensor {
    t.detach()
}
