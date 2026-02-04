use crate::autograd::{Tensor, TensorData};
use crate::module::Module;
use ndarray::{Array2, Zip};
use std::cell::RefCell;
use std::rc::Rc;

// --- ReLU ---
pub struct ReLU;
impl ReLU {
    pub fn new() -> Self {
        ReLU
    }
}

impl Module for ReLU {
    fn forward(&self, input: Tensor) -> Tensor {
        let input_ref = input.data_ref();

        // Forward: 并行计算 x.max(0.0)
        let data = Zip::from(&*input_ref).par_map_collect(|&x| x.max(0.0));

        let input_clone = input.clone();
        Tensor(Rc::new(RefCell::new(TensorData {
            data,
            grad: None,
            parents: vec![input.clone()],
            backward_op: Some(Box::new(move |grad| {
                let input_d = input_clone.data_ref();
                let mut grad_input = grad.clone();

                // Backward: 原地修改梯度
                Zip::from(&mut grad_input)
                    .and(&*input_d)
                    .par_for_each(|g, &x| {
                        if x <= 0.0 {
                            *g = 0.0;
                        }
                    });
                input_clone.add_grad(grad_input);
            })),
            requires_grad: true,
        })))
    }
    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }
}

// --- Sigmoid ---
pub struct Sigmoid;
impl Sigmoid {
    pub fn new() -> Self {
        Sigmoid
    }
}

impl Module for Sigmoid {
    fn forward(&self, input: Tensor) -> Tensor {
        let input_ref = input.data_ref();

        // Forward: 1 / (1 + exp(-x))
        let data = Zip::from(&*input_ref).par_map_collect(|&x| 1.0 / (1.0 + (-x).exp()));

        let output_data = data.clone();
        let input_clone = input.clone();

        Tensor(Rc::new(RefCell::new(TensorData {
            data,
            grad: None,
            parents: vec![input.clone()],
            backward_op: Some(Box::new(move |grad| {
                let mut grad_input = grad.clone();
                // Backward: grad * y * (1 - y)
                Zip::from(&mut grad_input)
                    .and(&output_data)
                    .par_for_each(|g, &y| {
                        *g = *g * y * (1.0 - y);
                    });

                input_clone.add_grad(grad_input);
            })),
            requires_grad: true,
        })))
    }
    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }
}

// --- Tanh ---
pub struct Tanh;
impl Tanh {
    pub fn new() -> Self {
        Tanh
    }
}

impl Module for Tanh {
    fn forward(&self, input: Tensor) -> Tensor {
        let input_ref = input.data_ref();

        // Forward: tanh(x)
        let data = Zip::from(&*input_ref).par_map_collect(|&x| x.tanh());

        let output_data = data.clone();
        let input_clone = input.clone();

        Tensor(Rc::new(RefCell::new(TensorData {
            data,
            grad: None,
            parents: vec![input.clone()],
            backward_op: Some(Box::new(move |grad| {
                let mut grad_input = grad.clone();
                // Backward: grad * (1 - y^2)
                Zip::from(&mut grad_input)
                    .and(&output_data)
                    .par_for_each(|g, &y| {
                        *g = *g * (1.0 - y * y);
                    });
                input_clone.add_grad(grad_input);
            })),
            requires_grad: true,
        })))
    }
    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }
}

// --- SiLU (Swish) ---
pub struct SiLU;
impl SiLU {
    pub fn new() -> Self {
        SiLU
    }
}

impl Module for SiLU {
    fn forward(&self, input: Tensor) -> Tensor {
        let input_ref = input.data_ref();

        // Forward: x * sigmoid(x)
        let data = Zip::from(&*input_ref).par_map_collect(|&x| {
            let sig = 1.0 / (1.0 + (-x).exp());
            x * sig
        });

        let input_clone = input.clone();

        Tensor(Rc::new(RefCell::new(TensorData {
            data,
            grad: None,
            parents: vec![input.clone()],
            backward_op: Some(Box::new(move |grad| {
                let x_ref = input_clone.data_ref();
                // Backward: dL/dx = grad * (sig + x * sig * (1 - sig))
                let dx = Zip::from(&*x_ref).par_map_collect(|&x| {
                    let sig = 1.0 / (1.0 + (-x).exp());
                    sig + x * sig * (1.0 - sig)
                });

                input_clone.add_grad(&dx * grad);
            })),
            requires_grad: true,
        })))
    }
    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }
}

pub struct Softmax {
    axis: usize,
}

impl Softmax {
    pub fn new(axis: usize) -> Self {
        Softmax { axis }
    }
}

impl Module for Softmax {
    fn forward(&self, input: Tensor) -> Tensor {
        let input_ref = input.data_ref();
        let x = &*input_ref;
        let shape = x.shape();

        let axis = if self.axis == shape.len() - 1 {
            self.axis
        } else {
            self.axis
        };

        let last_dim = shape[axis];
        let outer_dim = x.len() / last_dim;
        let x_cow = x.as_standard_layout();
        let x_2d = x_cow.view().into_shape((outer_dim, last_dim)).unwrap();

        // 2. 预分配输出
        let mut y_flat = Array2::<f32>::zeros((outer_dim, last_dim));

        // 3. 并行计算 Softmax
        Zip::from(y_flat.outer_iter_mut())
            .and(x_2d.outer_iter())
            .par_for_each(|mut y_row, x_row| {
                let max_val = x_row.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let mut sum = 0.0f32;

                for (y_val, &x_val) in y_row.iter_mut().zip(x_row.iter()) {
                    let e = (x_val - max_val).exp();
                    *y_val = e;
                    sum += e;
                }

                let inv_sum = 1.0 / sum;
                for y_val in y_row.iter_mut() {
                    *y_val *= inv_sum;
                }
            });

        let y = y_flat.into_shape(shape).unwrap();
        let output_data = y.clone();

        let input_clone = input.clone();
        let axis_idx = self.axis;

        Tensor(Rc::new(RefCell::new(TensorData {
            data: y,
            grad: None,
            parents: vec![input.clone()],
            backward_op: Some(Box::new(move |grad_output| {
                let grad_shape = grad_output.shape();
                let dim = grad_shape[axis_idx];
                let outer = grad_output.len() / dim;
                let g_cow = grad_output.as_standard_layout();
                let g_2d = g_cow.view().into_shape((outer, dim)).unwrap();

                let y_cow = output_data.as_standard_layout();
                let y_2d = y_cow.view().into_shape((outer, dim)).unwrap();

                let mut d_input_flat = Array2::<f32>::zeros((outer, dim));

                Zip::from(d_input_flat.outer_iter_mut())
                    .and(y_2d.outer_iter())
                    .and(g_2d.outer_iter())
                    .par_for_each(|mut di_row, y_row, g_row| {
                        let dot: f32 = y_row.iter().zip(g_row.iter()).map(|(&y, &g)| y * g).sum();

                        for (di, (&y, &g)) in di_row.iter_mut().zip(y_row.iter().zip(g_row.iter()))
                        {
                            *di = y * (g - dot);
                        }
                    });

                let d_input = d_input_flat.into_shape(grad_shape).unwrap();
                input_clone.add_grad(d_input);
            })),
            requires_grad: true,
        })))
    }
    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }
}

// --- Fast GELU ---
pub struct Gelu;
impl Gelu {
    pub fn new() -> Self {
        Gelu
    }
}

impl Module for Gelu {
    fn forward(&self, input: Tensor) -> Tensor {
        let input_ref = input.data_ref();

        const C: f32 = 0.7978845608;
        const K: f32 = 0.044715;

        let output = Zip::from(&*input_ref).par_map_collect(|&x| {
            let x3 = x * x * x;
            0.5 * x * (1.0 + (C * (x + K * x3)).tanh())
        });

        let input_clone = input.clone();
        Tensor(Rc::new(RefCell::new(TensorData {
            data: output,
            grad: None,
            parents: vec![input.clone()],
            backward_op: Some(Box::new(move |grad| {
                let x_ref = input_clone.data_ref();
                let dx = Zip::from(&*x_ref).par_map_collect(|&x| {
                    let x3 = x * x * x;
                    let inner = C * (x + K * x3);
                    let tanh_i = inner.tanh();
                    let sech2 = 1.0 - tanh_i * tanh_i;
                    0.5 * (1.0 + tanh_i) + 0.5 * x * sech2 * C * (1.0 + 3.0 * K * x * x)
                });
                input_clone.add_grad(&dx * grad);
            })),
            requires_grad: true,
        })))
    }
    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }
}
