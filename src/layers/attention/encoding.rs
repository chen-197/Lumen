use crate::autograd::{is_no_grad, Tensor, TensorData};
use ndarray::{Array, ArrayD, Ix2, Zip, s}; // 引入 Ix2
use std::cell::RefCell;
use std::rc::Rc;

pub struct RotaryEmbedding {
    dim: usize,
    max_seq_len: usize,
    theta: f32,
    // 缓存预计算的 cos/sin
    // Shape: [1, 1, Max_Seq, Dim]
    cos_cache: Tensor,
    sin_cache: Tensor,
}

impl RotaryEmbedding {
    pub fn new(dim: usize, max_seq_len: usize, theta: f32) -> Self {
        let (cos, sin) = Self::precompute_freqs_cis(dim, max_seq_len, theta);

        Self {
            dim,
            max_seq_len,
            theta,
            cos_cache: Tensor::from_array_no_grad(cos),
            sin_cache: Tensor::from_array_no_grad(sin),
        }
    }

    fn precompute_freqs_cis(
        dim: usize,
        max_seq_len: usize,
        theta: f32,
    ) -> (ArrayD<f32>, ArrayD<f32>) {
        let half_d = dim / 2;
        let mut cos_arr = Array::zeros((1, 1, max_seq_len, dim));
        let mut sin_arr = Array::zeros((1, 1, max_seq_len, dim));

        for i in 0..max_seq_len {
            let pos = i as f32;
            for j in 0..half_d {
                let freq = 1.0 / theta.powf((j as f32 * 2.0) / dim as f32);
                let val = pos * freq;

                let c = val.cos();
                let s = val.sin();

                cos_arr[[0, 0, i, j]] = c;
                cos_arr[[0, 0, i, j + half_d]] = c;
                sin_arr[[0, 0, i, j]] = s;
                sin_arr[[0, 0, i, j + half_d]] = s;
            }
        }
        (cos_arr.into_dyn(), sin_arr.into_dyn())
    }

    pub fn forward(&self, x: &Tensor, offset: usize) -> Tensor {
        let x_data = x.data_ref();
        let shape = x_data.shape();
        let (b, h, seq_len, d) = (shape[0], shape[1], shape[2], shape[3]);
        assert_eq!(d, self.dim, "RoPE dimension mismatch");

        let end = offset + seq_len;
        if end > self.max_seq_len {
            panic!(
                "RoPE index out of range: offset {} + len {} > max {}",
                offset, seq_len, self.max_seq_len
            );
        }

        // 1. 获取 Cache 引用
        let cos_cache_ref = self.cos_cache.data_ref();
        let sin_cache_ref = self.sin_cache.data_ref();

        // 2. 切片获取当前窗口，并直接转换为 2D View [Seq, Dim]
        let cos_slice_2d = cos_cache_ref
            .slice(s![0, 0, offset..end, ..]) // 直接切掉前两维 (indices 0, 0)
            .into_dimensionality::<Ix2>()
            .expect("RoPE Cache dimensionality mismatch");

        let sin_slice_2d = sin_cache_ref
            .slice(s![0, 0, offset..end, ..])
            .into_dimensionality::<Ix2>()
            .expect("RoPE Cache dimensionality mismatch");

        // 3. 准备输出
        let mut out = Array::zeros(x_data.dim());

        let x_view = x_data.view().into_dimensionality::<ndarray::Ix4>().unwrap();
        let mut out_view = out
            .view_mut()
            .into_dimensionality::<ndarray::Ix4>()
            .unwrap();

        // 4. 并行计算
        Zip::from(out_view.outer_iter_mut()) // Batch
            .and(x_view.outer_iter())
            .par_for_each(|mut out_b, x_b| {
                Zip::from(out_b.outer_iter_mut()) // Head
                    .and(x_b.outer_iter())
                    .for_each(|mut out_h, x_h| {
                        let half = d / 2;
                        // out_h, x_h 是 2D [Seq, Dim]
                        for ss in 0..seq_len {
                            for j in 0..half {
                                let x1 = x_h[[ss, j]];
                                let x2 = x_h[[ss, j + half]];

                                // 现在这里可以使用 [ss, j] 索引了
                                let c = cos_slice_2d[[ss, j]];
                                let s_val = sin_slice_2d[[ss, j]];

                                // RoPE 旋转
                                out_h[[ss, j]] = x1 * c - x2 * s_val;
                                out_h[[ss, j + half]] = x2 * c + x1 * s_val;
                            }
                        }
                    });
            });

        // 推理/不需要梯度：直接返回常量，不构图
        if is_no_grad() || !x.requires_grad() {
            return Tensor::from_data_no_grad(out.into_dyn().into_shared());
        }

        let x_clone = x.clone();

        // 为 Backward 准备数据：需要拥有所有权的 2D 数组
        let cos_backward = cos_slice_2d.to_owned();
        let sin_backward = sin_slice_2d.to_owned();

        Tensor(Rc::new(RefCell::new(TensorData {
            data: out.into_dyn().into_shared(),
            grad: None,
            parents: vec![x.clone()],
            backward_op: Some(Box::new(move |grad| {
                let grad_view = grad.view().into_dimensionality::<ndarray::Ix4>().unwrap();
                let mut d_x = Array::zeros((b, h, seq_len, d));

                Zip::from(d_x.outer_iter_mut())
                    .and(grad_view.outer_iter())
                    .par_for_each(|mut dx_b, g_b| {
                        Zip::from(dx_b.outer_iter_mut())
                            .and(g_b.outer_iter())
                            .for_each(|mut dx_h, g_h| {
                                let half = d / 2;
                                for ss in 0..seq_len {
                                    for j in 0..half {
                                        let g1 = g_h[[ss, j]];
                                        let g2 = g_h[[ss, j + half]];

                                        let c = cos_backward[[ss, j]];
                                        let s_val = sin_backward[[ss, j]];

                                        // Inverse rotation
                                        dx_h[[ss, j]] = g1 * c + g2 * s_val;
                                        dx_h[[ss, j + half]] = g2 * c - g1 * s_val;
                                    }
                                }
                            });
                    });

                x_clone.add_grad(d_x.into_dyn());
            })),
            requires_grad: true,
        })))
    }

    /// Apply RoPE for a single token at absolute position `pos`.
    ///
    /// Decode (S=1) hot-path helper to avoid allocating intermediate q_rot/k_rot tensors.
    /// `src` and `dst` must both have length == `self.dim`.
    #[inline]
    pub fn rope_1token_copy(&self, src: &[f32], dst: &mut [f32], pos: usize) {
        assert_eq!(src.len(), self.dim, "RoPE src len mismatch");
        assert_eq!(dst.len(), self.dim, "RoPE dst len mismatch");
        if pos >= self.max_seq_len {
            panic!(
                "RoPE index out of range: pos {} >= max {}",
                pos, self.max_seq_len
            );
        }

        // Cache layout: [1,1,Max_Seq,Dim]
        let cos_cache_ref = self.cos_cache.data_ref();
        let sin_cache_ref = self.sin_cache.data_ref();
        let cos_view = cos_cache_ref.slice(s![0, 0, pos, ..]);
        let cos_row = cos_view
            .as_slice()
            .expect("RoPE cos row not contiguous");
        let sin_view = sin_cache_ref.slice(s![0, 0, pos, ..]);
        let sin_row = sin_view
            .as_slice()
            .expect("RoPE sin row not contiguous");

        let half = self.dim / 2;
        for j in 0..half {
            let x1 = src[j];
            let x2 = src[j + half];
            let c = cos_row[j];
            let s_val = sin_row[j];
            dst[j] = x1 * c - x2 * s_val;
            dst[j + half] = x2 * c + x1 * s_val;
        }
    }

    /// Get (cos, sin) row at position `pos` as owned Vecs.
    /// This is useful to pass into rayon-parallel decode kernels without capturing Tensor/Rc.
    pub fn cos_sin_row_vec(&self, pos: usize) -> (Vec<f32>, Vec<f32>) {
        if pos >= self.max_seq_len {
            panic!(
                "RoPE index out of range: pos {} >= max {}",
                pos, self.max_seq_len
            );
        }
        let cos_cache_ref = self.cos_cache.data_ref();
        let sin_cache_ref = self.sin_cache.data_ref();

        let cos_view = cos_cache_ref.slice(s![0, 0, pos, ..]);
        let sin_view = sin_cache_ref.slice(s![0, 0, pos, ..]);

        let cos_row = cos_view
            .as_slice()
            .expect("RoPE cos row not contiguous");
        let sin_row = sin_view
            .as_slice()
            .expect("RoPE sin row not contiguous");

        (cos_row.to_vec(), sin_row.to_vec())
    }

}
