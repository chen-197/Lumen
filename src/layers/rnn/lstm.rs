use crate::autograd::Tensor;
use crate::layers::Linear;
use crate::layers::activation::{Sigmoid, Tanh};
use crate::module::Module;
use crate::ops::shape::slice_last_dim;
use crate::precision::{DType, default_parameter_quantization};
use ndarray::s;

pub struct LSTM {
    hidden_size: usize,
    w_x: Linear,
    w_h: Linear,
    sigmoid: Sigmoid,
    tanh: Tanh,
}

impl LSTM {
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        let w_x = Linear::new(input_size, 4 * hidden_size);
        let w_h = Linear::new(hidden_size, 4 * hidden_size);

        if let Some(bias_tensor) = &w_x.bias {
            {
                let mut bias_view = bias_tensor.data_mut();
                bias_view
                    .slice_mut(s![hidden_size..2 * hidden_size])
                    .mapv_inplace(|_| 1.0);
            }
            let dtype = bias_tensor.dtype();
            if dtype.is_integer() {
                let quantization = default_parameter_quantization();
                if quantization.is_enabled() && quantization.storage_dtype() == Some(dtype) {
                    bias_tensor.quantize_inplace_with_quantization(quantization);
                } else {
                    bias_tensor.cast_inplace(dtype);
                }
            } else {
                bias_tensor.cast_inplace(dtype);
            }
        }

        LSTM {
            hidden_size,
            w_x,
            w_h,
            sigmoid: Sigmoid::new(),
            tanh: Tanh::new(),
        }
    }

    pub fn new_with_dtype(input_size: usize, hidden_size: usize, dtype: DType) -> Self {
        let w_x = Linear::new_with_dtype(input_size, 4 * hidden_size, dtype);
        let w_h = Linear::new_with_dtype(hidden_size, 4 * hidden_size, dtype);
        // 我们的切分顺序是: [Input, Forget, Cell, Output]
        // 所以 Forget Gate 在索引 [hidden_size .. 2*hidden_size]

        if let Some(bias_tensor) = &w_x.bias {
            {
                let mut bias_view = bias_tensor.data_mut();
                bias_view
                    .slice_mut(s![hidden_size..2 * hidden_size])
                    .mapv_inplace(|_| 1.0);
            }
            bias_tensor.cast_inplace(dtype);
        }

        LSTM {
            hidden_size,
            w_x,
            w_h,
            sigmoid: Sigmoid::new(),
            tanh: Tanh::new(),
        }
    }

    pub fn forward_step(&self, x: &Tensor, h_prev: &Tensor, c_prev: &Tensor) -> (Tensor, Tensor) {
        let h_size = self.hidden_size;

        //融合计算
        let gates = self.w_x.forward(x.clone()) + self.w_h.forward(h_prev.clone());

        //切分
        // chunk 0: Input (i)
        // chunk 1: Forget (f) <-- Bias 已经被初始化为 1.0
        // chunk 2: Cell (g)
        // chunk 3: Output (o)
        let i_start = 0;
        let f_start = h_size;
        let g_start = 2 * h_size;
        let o_start = 3 * h_size;
        let gate_end = 4 * h_size;
        let i_raw = slice_last_dim(&gates, i_start, f_start);
        let f_raw = slice_last_dim(&gates, f_start, g_start);
        let g_raw = slice_last_dim(&gates, g_start, o_start);
        let o_raw = slice_last_dim(&gates, o_start, gate_end);

        //激活
        let i = self.sigmoid.forward(i_raw);
        let f = self.sigmoid.forward(f_raw);
        let g = self.tanh.forward(g_raw);
        let o = self.sigmoid.forward(o_raw);

        //Update
        // c_t = f * c_{t-1} + i * g
        // 因为 f 初始值较大，c_{t-1} 容易被保留
        let c_t = (f * c_prev.clone()) + (i * g);

        //Output
        let h_t = o * self.tanh.forward(c_t.clone());

        (h_t, c_t)
    }
}

impl Module for LSTM {
    fn forward(&self, _input: Tensor) -> Tensor {
        panic!("Use forward_step() for LSTM. It returns (h, c).");
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        params.extend(self.w_x.parameters());
        params.extend(self.w_h.parameters());
        params
    }
}

#[cfg(all(test, feature = "cuda"))]
#[path = "../../../tests/unit/layers/rnn/lstm/tests.rs"]
mod tests;
