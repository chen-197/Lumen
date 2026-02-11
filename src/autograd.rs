// src/autograd.rs
use ndarray::prelude::*;
pub use ndarray::{ArcArray, IxDyn};
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

// 开/关 全局推理模式（eval_mode/train_mode 可调用它）
pub fn set_inference_mode(on: bool) {
    INFERENCE_MODE.store(on, Ordering::Relaxed);
}

#[inline]
pub fn is_inference_mode() -> bool {
    INFERENCE_MODE.load(Ordering::Relaxed)
}

// no_grad 的判定：
// - 在 NoGradGuard 作用域内为 true
// - 或者处于 inference_mode 为 true
#[inline]
pub fn is_no_grad() -> bool {
    NO_GRAD_DEPTH.load(Ordering::Relaxed) > 0 || is_inference_mode()
}

// 便利封装：no_grad(|| { ... })
pub fn no_grad<R>(f: impl FnOnce() -> R) -> R {
    let _g = NoGradGuard::enter();
    f()
}

pub struct TensorData {
    pub data: ArcArray<f32, IxDyn>,
    // 梯度：使用 ArcArray 便于 optimizer 侧 clone 为零拷贝（仅增 refcount）
    pub grad: Option<ArcArray<f32, IxDyn>>,
    pub parents: Vec<Tensor>,
    // backward_op 接收 grad 的 view，避免在反传遍历时额外 to_owned
    pub backward_op: Option<Box<dyn Fn(&ArrayViewD<f32>)>>,
    pub requires_grad: bool,
}

#[derive(Clone)]
pub struct Tensor(pub(crate) Rc<RefCell<TensorData>>);

impl Tensor {
    // 默认构造叶子张量：
    // - 推理模式/no_grad 下：requires_grad=false
    // - 否则：requires_grad=true（更适合训练时手工造张量）
    pub fn new(data: ArrayD<f32>) -> Self {
        let req = !is_no_grad();
        Tensor(Rc::new(RefCell::new(TensorData {
            data: data.into_shared(),
            grad: None,
            parents: Vec::new(),
            backward_op: None,
            requires_grad: req,
        })))
    }

    // 获取数据的只读引用（零拷贝）
    pub fn data_ref(&self) -> Ref<'_, ArcArray<f32, IxDyn>> {
        let borrow = self.0.borrow();
        Ref::map(borrow, |t| &t.data)
    }

    // 获取梯度的只读引用（零拷贝）
    pub fn grad_ref(&self) -> Ref<'_, Option<ArcArray<f32, IxDyn>>> {
        let borrow = self.0.borrow();
        Ref::map(borrow, |t| &t.grad)
    }

    // 获取数据的可变引用
    pub fn data_mut(&self) -> RefMut<'_, ArcArray<f32, IxDyn>> {
        let borrow = self.0.borrow_mut();
        RefMut::map(borrow, |t| &mut t.data)
    }

    // 获取梯度的可变引用
    pub fn grad_mut(&self) -> RefMut<'_, Option<ArcArray<f32, IxDyn>>> {
        let borrow = self.0.borrow_mut();
        RefMut::map(borrow, |t| &mut t.grad)
    }

    pub fn data(&self) -> ArrayD<f32> {
        self.0.borrow().data.to_owned()
    }

    // 快路径：返回共享数据（clone 仅增加引用计数，不复制）
    pub fn data_arc(&self) -> ArcArray<f32, IxDyn> {
        self.0.borrow().data.clone()
    }

    // 慢路径：返回 owned 的 grad（会拷贝）
    pub fn grad(&self) -> Option<ArrayD<f32>> {
        self.0
            .borrow()
            .grad
            .as_ref()
            .map(|g| g.to_owned())
    }

    // 快路径：返回共享 grad（clone 仅增 refcount，不复制）
    pub fn grad_arc(&self) -> Option<ArcArray<f32, IxDyn>> {
        self.0.borrow().grad.clone()
    }

    pub fn sum(&self) -> Tensor {
        crate::ops::arithmetic::sum(self)
    }

    // 创建叶子张量（显式指定 requires_grad）
    pub fn from_data_with_grad_flag(data: ArrayD<f32>, requires_grad: bool) -> Tensor {
        Tensor(Rc::new(RefCell::new(TensorData {
            data: data.into_shared(),
            grad: None,
            parents: vec![],
            backward_op: None,
            requires_grad,
        })))
    }

    // 创建叶子张量：根据 is_no_grad() 自动决定 requires_grad
    pub fn from_data(data: ArrayD<f32>) -> Tensor {
        let req = !is_no_grad();
        Tensor::from_data_with_grad_flag(data, req)
    }

    // 推理/常量：不需要梯度
    pub fn from_data_no_grad(data: ArcArray<f32, IxDyn>) -> Tensor {
        Tensor(Rc::new(RefCell::new(TensorData {
            data,
            grad: None,
            parents: vec![],
            backward_op: None,
            requires_grad: false,
        })))
    }

    // 兼容旧接口：传入 ArrayD 作为常量
    pub fn from_array_no_grad(data: ArrayD<f32>) -> Tensor {
        Tensor::from_data_no_grad(data.into_shared())
    }

    // 训练参数：需要梯度（叶子）
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
            // existing 为共享 ArcArray；累加时会产生一个 owned ArrayD，然后再转回 shared。
            let summed = existing.to_owned() + &grad;
            inner.grad = Some(summed.into_shared());
        } else {
            inner.grad = Some(grad.into_shared());
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

        let shape = self.data_ref().shape().to_vec();
        self.add_grad(ArrayD::ones(shape));

        for node in topo.iter().rev() {
            let inner = node.0.borrow();
            if let Some(grad) = &inner.grad {
                if let Some(op) = &inner.backward_op {
                    let gv = grad.view();
                    op(&gv.into_dyn());
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
        self.0.borrow_mut().data = new_data.into_shared();
    }

    // detach：返回一个新 Tensor（数据拷贝），requires_grad=false，且无 parents/backward_op
    pub fn detach(&self) -> Tensor {
        let d = self.0.borrow().data.to_owned();
        Tensor::from_data_with_grad_flag(d, false)
    }
}

// 切断梯度流（等价于 t.detach()）
pub fn detach(t: &Tensor) -> Tensor {
    t.detach()
}
