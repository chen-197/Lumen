use crate::autograd::Tensor;
use ndarray::prelude::*;
use ndarray::Zip;

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
            // 先把 grad clone 出来（ArcArray clone 仅增 refcount，不复制数据），避免与 data.view_mut() 的可变借用冲突
            let grad = match p_inner.grad.clone() {
                Some(g) => g,
                None => continue,
            };

            {
                if self.momentum == 0.0 {
                    // 标准 SGD（尽量原地更新，避免每步分配）:
                    // w -= lr * grad
                    // 说明：ArcArray 的 DataMut 实现内部用 Arc::make_mut，
                    // 当数据被共享时会触发一次 copy-on-write；未共享时为真·原地。
                    let lr = self.lr;
                    Zip::from(p_inner.data.view_mut())
                        .and(grad.view())
                        .for_each(|w, g| {
                            *w -= lr * *g;
                        });
                } else {
                    // SGD with Momentum
                    // v = m * v + grad (有的实现是 v = m * v + lr * grad，这里参考 PyTorch 默认行为)
                    // w = w - lr * v

                    if self.velocities[i].is_none() {
                        self.velocities[i] = Some(ArrayD::zeros(p_inner.data.shape()));
                    }

                    let m = self.momentum;
                    let lr = self.lr;
                    let v_buf = self.velocities[i].as_mut().unwrap();

                    // v = m * v + grad
                    Zip::from(v_buf.view_mut())
                        .and(grad.view())
                        .for_each(|v, g| {
                            *v = m * (*v) + *g;
                        });

                    // w -= lr * v
                    Zip::from(p_inner.data.view_mut())
                        .and(v_buf)
                        .for_each(|w, vv| {
                            *w -= lr * *vv;
                        });
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

            // 先把 grad clone 出来，避免与 data.view_mut() 的可变借用冲突
            let grad = match p_inner.grad.as_ref() {
                Some(g) => g.clone(),
                None => continue,
            };

            {
                // 初始化状态 (如果第一次运行)
                if self.exp_avg[i].is_none() {
                    self.exp_avg[i] = Some(ArrayD::zeros(p_inner.data.shape()));
                    self.exp_avg_sq[i] = Some(ArrayD::zeros(p_inner.data.shape()));
                }

                // 复用缓冲 + 单次 Zip 完成 m/v 更新与参数更新：
                // m = beta1*m + (1-beta1)*g
                // v = beta2*v + (1-beta2)*g^2
                // w -= lr * (m/bc1) / (sqrt(v/bc2) + eps)
                let lr = self.lr;
                let eps = self.eps;
                let m_buf = self.exp_avg[i].as_mut().unwrap();
                let v_buf = self.exp_avg_sq[i].as_mut().unwrap();

                Zip::from(p_inner.data.view_mut())
                    .and(m_buf.view_mut())
                    .and(v_buf.view_mut())
                    .and(grad.view())
                    .for_each(|w, m, v, g| {
                        *m = beta1 * (*m) + (1.0 - beta1) * g;
                        *v = beta2 * (*v) + (1.0 - beta2) * g * g;
                        let m_hat = *m / bias_correction1;
                        let v_hat = *v / bias_correction2;
                        *w -= lr * (m_hat / (v_hat.sqrt() + eps));
                    });
            }
        }
    }
}