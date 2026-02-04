use crate::autograd::{Tensor, is_no_grad};
use crate::layers::Linear;
use crate::layers::attention::encoding::RotaryEmbedding;
use crate::module::Module;
use crate::ops::fused::fused_softmax;
use crate::ops::matmul::batch_matmul;
use crate::ops::shape::{permute, reshape};

use ndarray::linalg::general_mat_mul;
use ndarray::{Array2, Array4};
use std::cell::RefCell;
use std::rc::Rc;

pub struct KVCacheInner {
    pub k: Array4<f32>, // [B, H_kv, max_seq, D]
    pub v: Array4<f32>, // [B, H_kv, max_seq, D]
    pub len: usize,     // 当前已写入的长度
}

pub type KVCache = Rc<RefCell<KVCacheInner>>;

impl KVCacheInner {
    pub fn new(b: usize, h_kv: usize, max_seq: usize, d: usize) -> Self {
        Self {
            k: Array4::<f32>::zeros((b, h_kv, max_seq, d)),
            v: Array4::<f32>::zeros((b, h_kv, max_seq, d)),
            len: 0,
        }
    }

    pub fn reset(&mut self) {
        self.len = 0;
    }
}

pub struct SelfAttention {
    pub w_q: Linear,
    pub w_k: Linear,
    pub w_v: Linear,
    pub w_o: Linear,
    rope: RotaryEmbedding,
    n_head: usize,
    pub n_kv_head: usize,
    head_dim: usize,
    scale: f32,
    pub causal: bool,
    max_seq: usize, // 为 cache 预分配用
}

impl SelfAttention {
    pub fn new(
        embed_dim: usize,
        n_head: usize,
        n_kv_head: usize,
        max_seq_len: usize,
        rope_theta: f32,
        causal: bool,
    ) -> Self {
        assert_eq!(
            embed_dim % n_head,
            0,
            "Embed dim must be divisible by n_head"
        );

        let head_dim = embed_dim / n_head;
        let kv_dim = n_kv_head * head_dim;

        let rope = RotaryEmbedding::new(head_dim, max_seq_len, rope_theta);

        Self {
            w_q: Linear::new_no_bias(embed_dim, embed_dim),
            w_k: Linear::new_no_bias(embed_dim, kv_dim),
            w_v: Linear::new_no_bias(embed_dim, kv_dim),
            w_o: Linear::new_no_bias(embed_dim, embed_dim),
            rope,
            n_head,
            n_kv_head,
            head_dim,
            scale: (head_dim as f32).sqrt().recip(),
            causal,
            max_seq: max_seq_len, // 存储正确的最大长度
        }
    }

    /// forward：eval 用预分配 cache；train 走原逻辑（cat + repeat_kv）
    pub fn forward(&self, x: Tensor, cache: Option<KVCache>) -> (Tensor, Option<KVCache>) {
        let x_shape = x.data().shape().to_vec();
        let (b, s, _) = (x_shape[0], x_shape[1], x_shape[2]);

        let h = self.n_head;
        let h_kv = self.n_kv_head;
        let d = self.head_dim;
        let n_rep = h / h_kv;

        // 1) QKV projection
        let q = self.w_q.forward(x.clone());
        let k = self.w_k.forward(x.clone());
        let v = self.w_v.forward(x);

        // 2) reshape/permute 到 [B, H, S, D] / [B, H_kv, S, D]
        let q = permute(
            &reshape(&q, vec![b as i32, s as i32, h as i32, d as i32]),
            vec![0, 2, 1, 3],
        );
        let k = permute(
            &reshape(&k, vec![b as i32, s as i32, h_kv as i32, d as i32]),
            vec![0, 2, 1, 3],
        );
        let v = permute(
            &reshape(&v, vec![b as i32, s as i32, h_kv as i32, d as i32]),
            vec![0, 2, 1, 3],
        );

        // eval 路径：预分配 cache + 不 repeat_kv（ndarray 快路径）
        if is_no_grad() {
            // 3) 初始化/取出 cache（预分配）
            let cache_handle: KVCache = match cache {
                Some(c) => c,
                None => Rc::new(RefCell::new(KVCacheInner::new(b, h_kv, self.max_seq, d))),
            };

            // 4) RoPE：offset = past_len
            let past_len = cache_handle.borrow().len;
            let q_rot = self.rope.forward(&q, past_len);
            let k_rot = self.rope.forward(&k, past_len);

            // 5) 写入 cache（不 cat）
            {
                let mut c = cache_handle.borrow_mut();
                let new_len = past_len + s;
                assert!(
                    new_len <= self.max_seq,
                    "KV cache overflow: new_len={} > max_seq={}",
                    new_len,
                    self.max_seq
                );

                let k_src = k_rot
                    .data()
                    .to_owned()
                    .into_dimensionality::<ndarray::Ix4>()
                    .unwrap(); // [B,H_kv,S,D]
                let v_src = v
                    .data()
                    .to_owned()
                    .into_dimensionality::<ndarray::Ix4>()
                    .unwrap();

                c.k.slice_mut(ndarray::s![.., .., past_len..new_len, ..])
                    .assign(&k_src);
                c.v.slice_mut(ndarray::s![.., .., past_len..new_len, ..])
                    .assign(&v_src);
                c.len = new_len;
            }

            // 6) GQA attention（不 repeat_kv）
            let context_bhsd = {
                let c = cache_handle.borrow();
                let total_len = c.len;
                let q4 = q_rot
                    .data()
                    .to_owned()
                    .into_dimensionality::<ndarray::Ix4>()
                    .unwrap(); // [B,H,S,D]
                let k4 = c.k.slice(ndarray::s![.., .., 0..total_len, ..]).to_owned(); // [B,H_kv,L,D]
                let v4 = c.v.slice(ndarray::s![.., .., 0..total_len, ..]).to_owned(); // [B,H_kv,L,D]
                gqa_attention_no_repeat(&q4, &k4, &v4, self.scale, self.causal, n_rep, past_len)
            };

            // 7) 输出投影：context [B,H,S,D] -> [B,S,H*D] -> w_o
            let context = Tensor::from_data_no_grad(context_bhsd.into_dyn());
            let context = permute(&context, vec![0, 2, 1, 3]);
            let context = reshape(&context, vec![b as i32, s as i32, (h * d) as i32]);
            let output = self.w_o.forward(context);

            return (output, Some(cache_handle));
        }

        // train 路径：走原来的逻辑（可导）

        // 希望训练时禁止传 cache：
        if cache.is_some() {
            panic!("Train path does not accept eval KVCache. Use eval_mode + cache for decoding.");
        }

        // 3) RoPE（训练全序列 offset=0）
        let q_rot = self.rope.forward(&q, 0);
        let k_rot = self.rope.forward(&k, 0);

        // 4) Repeat KV heads
        let k_up = if n_rep > 1 {
            repeat_kv(k_rot.clone(), n_rep)
        } else {
            k_rot.clone()
        };

        let v_up = if n_rep > 1 {
            repeat_kv(v.clone(), n_rep)
        } else {
            v.clone()
        };

        // 5) Attention
        let k_t = permute(&k_up, vec![0, 1, 3, 2]);
        let scores = batch_matmul(&q_rot, &k_t);
        let attn_probs = fused_softmax(&scores, self.scale, self.causal);
        let context = batch_matmul(&attn_probs, &v_up);

        // 6) Output
        let context = permute(&context, vec![0, 2, 1, 3]);
        let context = reshape(&context, vec![b as i32, s as i32, (h * d) as i32]);
        let output = self.w_o.forward(context);

        // 训练路径默认不返回 cache
        (output, None)
    }
}

impl Module for SelfAttention {
    fn forward(&self, x: Tensor) -> Tensor {
        let (out, _) = self.forward(x, None);
        out
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut p = self.w_q.parameters();
        p.extend(self.w_k.parameters());
        p.extend(self.w_v.parameters());
        p.extend(self.w_o.parameters());
        p
    }
}

// train 路径：repeat_kv（依赖 Tensor ops，可导）
// x: [B, H_kv, S, D] -> [B, H, S, D]

pub fn repeat_kv(x: Tensor, n_rep: usize) -> Tensor {
    let data_ref = x.data();
    let shape = data_ref.shape();
    let (b, n_kv, s, d) = (shape[0], shape[1], shape[2], shape[3]);

    let contig_data = data_ref.as_standard_layout();

    let expanded = contig_data
        .into_shape((b, n_kv, 1, s, d))
        .expect("Failed to expand KV shape");

    let broadcasted = expanded
        .broadcast((b, n_kv, n_rep, s, d))
        .expect("Failed to broadcast KV");

    let res = broadcasted
        .to_owned()
        .into_shape((b, n_kv * n_rep, s, d))
        .expect("Failed to flatten repeated KV heads");

    Tensor::new(res.into_dyn())
}

// eval 路径核心：GQA attention（不 repeat_kv）
// q: [B, H, S, D]
// k/v: [B, H_kv, L, D]
// 返回 context: [B, H, S, D]
// past_len: cache 写入前的长度（用于 causal mask 的 absolute index）

fn gqa_attention_no_repeat(
    q: &Array4<f32>,
    k: &Array4<f32>,
    v: &Array4<f32>,
    scale: f32,
    causal: bool,
    n_rep: usize,
    past_len: usize,
) -> Array4<f32> {
    let (b, h, s, d) = q.dim();
    let (b2, h_kv, l, d2) = k.dim();
    assert_eq!(b, b2);
    assert_eq!(d, d2);
    assert_eq!(h, h_kv * n_rep);

    let mut out = Array4::<f32>::zeros((b, h, s, d));

    for bb in 0..b {
        for hq in 0..h {
            let hk = hq / n_rep;

            let q_mat = q.slice(ndarray::s![bb, hq, .., ..]); // [S,D]
            let k_mat = k.slice(ndarray::s![bb, hk, .., ..]); // [L,D]
            let v_mat = v.slice(ndarray::s![bb, hk, .., ..]); // [L,D]

            let mut scores = Array2::<f32>::zeros((s, l));
            general_mat_mul(1.0, &q_mat, &k_mat.t(), 0.0, &mut scores);

            // causal mask 用 absolute index：key_j <= past_len + query_i
            softmax_inplace(&mut scores, scale, causal, past_len);

            let mut ctx = Array2::<f32>::zeros((s, d));
            general_mat_mul(1.0, &scores, &v_mat, 0.0, &mut ctx);

            out.slice_mut(ndarray::s![bb, hq, .., ..]).assign(&ctx);
        }
    }

    out
}

fn softmax_inplace(scores: &mut Array2<f32>, scale: f32, causal: bool, past_len: usize) {
    let (s, l) = scores.dim();

    for i in 0..s {
        // query 的 absolute index
        let q_abs = past_len + i;

        // 1) scale + causal mask
        for j in 0..l {
            let mut val = scores[(i, j)] * scale;
            if causal && j > q_abs {
                val = f32::NEG_INFINITY;
            }
            scores[(i, j)] = val;
        }

        // 2) stable softmax
        let mut maxv = f32::NEG_INFINITY;
        for j in 0..l {
            let v = scores[(i, j)];
            if v > maxv {
                maxv = v;
            }
        }
        let mut sum = 0.0f32;
        for j in 0..l {
            let e = (scores[(i, j)] - maxv).exp();
            scores[(i, j)] = e;
            sum += e;
        }
        let inv = 1.0f32 / (sum + 1e-9);
        for j in 0..l {
            scores[(i, j)] *= inv;
        }
    }
}
