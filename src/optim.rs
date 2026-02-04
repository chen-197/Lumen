use crate::autograd::Tensor;
use ndarray::prelude::*;

pub trait Optimizer {
    fn step(&mut self);
    fn zero_grad(&self) {
        for param in self.params() {
            let mut p = param.0.borrow_mut();
            p.grad = None;
        }
    }
    fn params(&self) -> &Vec<Tensor>;
}

pub struct SGD {
    params: Vec<Tensor>,
    lr: f32,
    momentum: f32,
    velocities: Vec<Option<ArrayD<f32>>>,
}

impl SGD {
    pub fn new(params: Vec<Tensor>, lr: f32) -> Self {
        let len = params.len();
        SGD {
            params,
            lr,
            momentum: 0.0, // 默认无动量
            velocities: vec![None; len],
        }
    }

    pub fn with_momentum(mut self, momentum: f32) -> Self {
        self.momentum = momentum;
        self
    }
}

impl Optimizer for SGD {
    fn params(&self) -> &Vec<Tensor> {
        &self.params
    }

    fn step(&mut self) {
        for (i, param) in self.params.iter().enumerate() {
            let mut p_inner = param.0.borrow_mut();
            
            // 只有当参数有梯度时才更新
            if let Some(grad) = &p_inner.grad {
                if self.momentum == 0.0 {
                    // 标准 SGD: w = w - lr * grad
                    p_inner.data = &p_inner.data - &(grad * self.lr);
                } else {
                    // SGD with Momentum
                    // v = m * v + grad (有的实现是 v = m * v + lr * grad，这里参考 PyTorch 默认行为)
                    // w = w - lr * v
                    
                    let v_old = if let Some(v) = &self.velocities[i] {
                        v.clone()
                    } else {
                        ArrayD::zeros(p_inner.data.shape())
                    };

                    let v_new = &(&v_old * self.momentum) + grad;
                    
                    p_inner.data = &p_inner.data - &(&v_new * self.lr);
                    
                    self.velocities[i] = Some(v_new);
                }
            }
        }
    }
}

pub struct Adam {
    params: Vec<Tensor>,
    lr: f32,
    betas: (f32, f32), 
    eps: f32,          
    
    // 状态
    step_count: usize,
    exp_avg: Vec<Option<ArrayD<f32>>>,    // m (一阶矩)
    exp_avg_sq: Vec<Option<ArrayD<f32>>>, // v (二阶矩)
}

impl Adam {
    pub fn new(params: Vec<Tensor>, lr: f32) -> Self {
        let len = params.len();
        Adam {
            params,
            lr,
            betas: (0.9, 0.999), 
            eps: 1e-8,
            step_count: 0,
            exp_avg: vec![None; len],
            exp_avg_sq: vec![None; len],
        }
    }
}

impl Optimizer for Adam {
    fn params(&self) -> &Vec<Tensor> {
        &self.params
    }

    fn step(&mut self) {
        self.step_count += 1;
        let (beta1, beta2) = self.betas;
        
        // 预计算 Bias Correction
        let bias_correction1 = 1.0 - beta1.powi(self.step_count as i32);
        let bias_correction2 = 1.0 - beta2.powi(self.step_count as i32);

        for (i, param) in self.params.iter().enumerate() {
            let mut p_inner = param.0.borrow_mut();

            if let Some(grad) = &p_inner.grad {
                // 初始化状态 (如果第一次运行)
                if self.exp_avg[i].is_none() {
                    self.exp_avg[i] = Some(ArrayD::zeros(p_inner.data.shape()));
                    self.exp_avg_sq[i] = Some(ArrayD::zeros(p_inner.data.shape()));
                }

                let m_prev = self.exp_avg[i].as_ref().unwrap();
                let v_prev = self.exp_avg_sq[i].as_ref().unwrap();

                // 更新一阶矩 m: m_t = beta1 * m_{t-1} + (1 - beta1) * g
                let m_t = &(m_prev * beta1) + &(grad * (1.0 - beta1));

                // 更新二阶矩 v: v_t = beta2 * v_{t-1} + (1 - beta2) * g^2
                // g^2 是逐元素平方
                let grad_sq = grad.mapv(|x| x * x);
                let v_t = &(v_prev * beta2) + &(&grad_sq * (1.0 - beta2));

                // 计算 hat_m 和 hat_v (Bias Correction)
                let m_hat = &m_t / bias_correction1;
                let v_hat = &v_t / bias_correction2;

                // 更新参数: w = w - lr * m_hat / (sqrt(v_hat) + eps)
                let denorm = v_hat.mapv(|x| x.sqrt() + self.eps);
                let step_update = &m_hat / &denorm;

                p_inner.data = &p_inner.data - &(&step_update * self.lr);

                self.exp_avg[i] = Some(m_t);
                self.exp_avg_sq[i] = Some(v_t);
            }
        }
    }
}