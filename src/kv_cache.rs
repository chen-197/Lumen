use crate::autograd::Tensor;
use ndarray::{Array4, s};

#[derive(Clone)]
pub struct LlamaKVCache {
    pub k_cache: Array4<f32>,
    pub v_cache: Array4<f32>,
    pub max_seq_len: usize,
    pub dim: usize,
    pub head_num: usize,
}

impl LlamaKVCache {
    pub fn new(config: &crate::models::LlamaConfig) -> Self {
        let max_seq_len = config.max_seq_len;
        let head_num = config.num_key_value_heads;
        let head_dim = config.hidden_size / config.num_attention_heads;

        let k_data = Array4::<f32>::zeros((1, head_num, max_seq_len, head_dim));
        let v_data = Array4::<f32>::zeros((1, head_num, max_seq_len, head_dim));

        Self {
            k_cache: k_data,
            v_cache: v_data,
            max_seq_len,
            dim: head_dim,
            head_num,
        }
    }

    pub fn update(&mut self, k: &Tensor, v: &Tensor, start_pos: usize) {
        let seq_len = k.data_ref().shape()[2];
        let end_pos = start_pos + seq_len;
        if end_pos > self.max_seq_len {
            panic!("KV Cache overflow! Max: {}, Current: {}", self.max_seq_len, end_pos);
        }

        let k_in = k.data_ref();
        let k_in_view = k_in.view().into_dimensionality::<ndarray::Ix4>().unwrap();
        self.k_cache
            .slice_mut(s![.., .., start_pos..end_pos, ..])
            .assign(&k_in_view);

        let v_in = v.data_ref();
        let v_in_view = v_in.view().into_dimensionality::<ndarray::Ix4>().unwrap();
        self.v_cache
            .slice_mut(s![.., .., start_pos..end_pos, ..])
            .assign(&v_in_view);
    }

    pub fn get_view(&self, current_len: usize) -> (Tensor, Tensor) {
        let k_slice = self.k_cache.slice(s![.., .., ..current_len, ..]).to_owned();
        let v_slice = self.v_cache.slice(s![.., .., ..current_len, ..]).to_owned();

        (
            Tensor::from_array_no_grad(k_slice.into_dyn()),
            Tensor::from_array_no_grad(v_slice.into_dyn())
        )
    }
}