use crate::autograd::Tensor;
use crate::module::Module;
use crate::layers::Linear;
use crate::layers::activation::Tanh;

pub struct RNN {
    w_ih: Linear, // Input to Hidden
    w_hh: Linear, // Hidden to Hidden
    activation: Tanh,
}

impl RNN {
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        RNN {
            w_ih: Linear::new(input_size, hidden_size),
            w_hh: Linear::new(hidden_size, hidden_size),
            activation: Tanh::new(),
        }
    }

    // RNN 的前向传播需要两个输入：当前输入 x 和 上一时刻隐含状态 h_prev
    pub fn forward_step(&self, input: &Tensor, h_prev: &Tensor) -> Tensor {
        // h_t = Tanh( W_ih * x + W_hh * h_{t-1} )
        
        let i_part = self.w_ih.forward(input.clone());
        let h_part = self.w_hh.forward(h_prev.clone());
        let combined = i_part + h_part; 
        
        self.activation.forward(combined)
    }
}

impl Module for RNN {
    fn forward(&self, _input: Tensor) -> Tensor {
        panic!("Use forward_step for RNN!");
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = self.w_ih.parameters();
        params.extend(self.w_hh.parameters());
        params
    }
}