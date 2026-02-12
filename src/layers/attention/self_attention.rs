use crate::autograd::{Tensor, is_no_grad};
use crate::layers::Linear;
use crate::layers::attention::encoding::RotaryEmbedding;
use crate::module::Module;
use crate::ops::fused::fused_softmax;
use crate::ops::matmul::batch_matmul;
use crate::ops::shape::{permute, reshape};

use ndarray::linalg::general_mat_mul;
use ndarray::Array4;
use rayon::prelude::*;
use std::cell::RefCell;
use std::rc::Rc;

thread_local! {
    // attention scores buffer: S * L
    static ATT_SCORES_BUF: RefCell<Vec<f32>> = RefCell::new(Vec::new());
    // attention ctx buffer: S * D
    static ATT_CTX_BUF: RefCell<Vec<f32>> = RefCell::new(Vec::new());
    // decode(S=1) q RoPE buffer: D
    static ATT_Q_BUF: RefCell<Vec<f32>> = RefCell::new(Vec::new());
}

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
        let x_shape = x.data_ref().shape().to_vec();
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
        // eval 路径：尽量绕开 Tensor shape ops（不产生中间 Tensor，也不触发 copy）
        if is_no_grad() {
            // ------------- decode(S=1) ultra hot-path -------------
            // Goal:
            // - Only rotate NEW token's Q/K (with offset=past_len)
            // - Write rotated K directly into KV cache (no k_rot tensor)
            // - Fuse RoPE(Q) + online-softmax attention (no scores buffer)
            // - Output BSHD layout to avoid permute copies
            let cache_handle: KVCache = match cache {
                Some(c) => c,
                None => Rc::new(RefCell::new(KVCacheInner::new(b, h_kv, self.max_seq, d))),
            };

            let past_len = cache_handle.borrow().len;
            if s == 1 {
                // Projected shapes are [B,1,H*D] / [B,1,Hkv*D]. For S=1 we can view them as [B,H,D].
                let q3 = q
                    .data_arc()
                    .into_shape((b, h, d))
                    .expect("Q reshape [B,H,D] failed");
                let k3 = k
                    .data_arc()
                    .into_shape((b, h_kv, d))
                    .expect("K reshape [B,H_kv,D] failed");
                let v3 = v
                    .data_arc()
                    .into_shape((b, h_kv, d))
                    .expect("V reshape [B,H_kv,D] failed");

                // 1) Write NEW token into KV cache. Rotate K on-the-fly into destination.
                {
                    let mut c = cache_handle.borrow_mut();
                    let new_len = past_len + 1;
                    assert!(
                        new_len <= self.max_seq,
                        "KV cache overflow: new_len={} > max_seq={}",
                        new_len,
                        self.max_seq
                    );

                    for bb in 0..b {
                        for hk in 0..h_kv {
                            let k_view = k3.slice(ndarray::s![bb, hk, ..]);
                            let src_k = k_view
                                .as_slice()
                                .expect("K src not contiguous");
                            let v_view = v3.slice(ndarray::s![bb, hk, ..]);
                            let src_v = v_view
                                .as_slice()
                                .expect("V src not contiguous");
                            // K: RoPE rotate and write directly
                            {
                                let mut dst_k = c.k.slice_mut(ndarray::s![bb, hk, past_len, ..]);
                                let dst_k_sl = dst_k
                                    .as_slice_mut()
                                    .expect("K dst not contiguous");
                                self.rope.rope_1token_copy(src_k, dst_k_sl, past_len);
                            }
                            // V: direct memcpy
                            {
                                let mut dst_v = c.v.slice_mut(ndarray::s![bb, hk, past_len, ..]);
                                dst_v
                                    .as_slice_mut()
                                    .expect("V dst not contiguous")
                                    .copy_from_slice(src_v);
                            }
                        }
                    }
                    c.len = new_len;
                }

                // 2) Fused attention: RoPE(Q) on-the-fly + online softmax + weighted sum.
                let total_len = past_len + 1;
                let out_vec = {
                    let c = cache_handle.borrow();
                let k_cache = c.k.slice(ndarray::s![.., .., 0..total_len, ..]); // [B,H_kv,L,D]
                let v_cache = c.v.slice(ndarray::s![.., .., 0..total_len, ..]); // [B,H_kv,L,D]

                let mut out_vec = vec![0.0f32; b * h * d]; // [B,H,D] for S=1

                // NOTE: rayon parallel closures must not capture `self` because `SelfAttention` contains `Tensor` (Rc<RefCell<_>>) which is !Sync.
                // Extract only plain data needed for decode kernel.
                let scale = self.scale;
                let causal = self.causal;
                let (cos_row_vec, sin_row_vec) = self.rope.cos_sin_row_vec(past_len);
                let half_d = d / 2;

                // Parallel over (bb, hh) rows: out_vec is [B*H, D] contiguous
                out_vec
                    .par_chunks_mut(d)
                    .enumerate()
                    .for_each(|(row, out_row)| {
                        let bb = row / h;
                        let hh = row % h;
                        let hk = hh / n_rep;

                        // Per-thread buffer for rotated Q (length D)
                        ATT_Q_BUF.with(|qb| {
                            let mut qb = qb.borrow_mut();
                            if qb.len() < d {
                                qb.resize(d, 0.0);
                            }
                            let qbuf = &mut qb[..d];

                            // RoPE rotate Q for this head at position past_len (on-the-fly)
                            let q_view = q3.slice(ndarray::s![bb, hh, ..]);
                            let q_src = q_view
                                .as_slice()
                                .expect("Q src not contiguous");
                            for j in 0..half_d {
                                let x1 = q_src[j];
                                let x2 = q_src[j + half_d];
                                let c = cos_row_vec[j];
                                let s_val = sin_row_vec[j];
                                qbuf[j] = x1 * c - x2 * s_val;
                                qbuf[j + half_d] = x1 * s_val + x2 * c;
                            }

                            // One-pass online softmax (log-sum-exp streaming), no scores buffer.
                            // Avoid per-step slice() overhead by taking contiguous [L,D] blocks.
                            let k_bd = k_cache
                                .slice(ndarray::s![bb, hk, 0..total_len, ..])
                                .into_dimensionality::<ndarray::Ix2>()
                                .unwrap();
                            let v_bd = v_cache
                                .slice(ndarray::s![bb, hk, 0..total_len, ..])
                                .into_dimensionality::<ndarray::Ix2>()
                                .unwrap();
                            let k_sl = k_bd.as_slice().expect("K cache block not contiguous");
                            let v_sl = v_bd.as_slice().expect("V cache block not contiguous");

                            ATT_CTX_BUF.with(|cb| {
                                let mut cb = cb.borrow_mut();
                                if cb.len() < d {
                                    cb.resize(d, 0.0);
                                }
                                let ctx = &mut cb[..d];
                                for i in 0..d {
                                    ctx[i] = 0.0;
                                }

                                let mut m = f32::NEG_INFINITY;
                                let mut l = 0.0f32;

                                for j in 0..total_len {
                                    if causal && j > past_len {
                                        break;
                                    }
                                    let k_row = &k_sl[j * d..(j + 1) * d];
                                    let mut dot = 0.0f32;
                                    for i in 0..d {
                                        dot += qbuf[i] * k_row[i];
                                    }
                                    let score = dot * scale;
                                    if score > m {
                                        let s = (m - score).exp();
                                        for i in 0..d {
                                            ctx[i] *= s;
                                        }
                                        l *= s;
                                        m = score;
                                        let v_row = &v_sl[j * d..(j + 1) * d];
                                        for i in 0..d {
                                            ctx[i] += v_row[i];
                                        }
                                        l += 1.0;
                                    } else {
                                        let w = (score - m).exp();
                                        let v_row = &v_sl[j * d..(j + 1) * d];
                                        for i in 0..d {
                                            ctx[i] += w * v_row[i];
                                        }
                                        l += w;
                                    }
                                }

                                let inv = 1.0f32 / (l + 1e-9);
                                for i in 0..d {
                                    out_row[i] = ctx[i] * inv;
                                }
                            });
                        });
                    });


                    out_vec
                };

                // 3) Output projection: [B,1,H,D] -> [B,1,H*D] (view reshape) -> w_o
                let out_bshd = Array4::from_shape_vec((b, 1, h, d), out_vec).expect("out shape");
                let context = Tensor::from_data_no_grad(out_bshd.into_dyn().into_shared());
                let context = reshape(&context, vec![b as i32, 1, (h * d) as i32]);
                let output = self.w_o.forward(context);

                return (output, Some(cache_handle));
            }

            let q = Tensor::from_data_no_grad(
                q.data_arc()
                    .into_shape((b, s, h, d))
                    .expect("Q reshape failed")
                    .permuted_axes([0, 2, 1, 3])
                    .into_dyn(),
            );
            let k = Tensor::from_data_no_grad(
                k.data_arc()
                    .into_shape((b, s, h_kv, d))
                    .expect("K reshape failed")
                    .permuted_axes([0, 2, 1, 3])
                    .into_dyn(),
            );
            let v = Tensor::from_data_no_grad(
                v.data_arc()
                    .into_shape((b, s, h_kv, d))
                    .expect("V reshape failed")
                    .permuted_axes([0, 2, 1, 3])
                    .into_dyn(),
            );

            // 3) 初始化/取出 cache（预分配）
            // (cache_handle 已在上面创建)

            // 4) RoPE：offset = past_len（S>1 prefill 路径保持原实现）
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

                // 注意：data_ref() 返回 Ref<'_>，不能链式直接拿 view，否则 Ref 会过早 drop（E0716）。
                let k_ref = k_rot.data_ref();
                let k_src = k_ref
                    .view()
                    .into_dimensionality::<ndarray::Ix4>()
                    .unwrap(); // [B,H_kv,S,D] view

                let v_ref = v.data_ref();
                let v_src = v_ref
                    .view()
                    .into_dimensionality::<ndarray::Ix4>()
                    .unwrap();

                // KV cache 写入：
                // - decode_step (S=1) 走更轻量的 slice + copy_from_slice（避免 assign 的逐元素/广播开销）
                // - S>1 保持 assign（依然是 view 写入，不产生额外 cat/copy）
                if s == 1 {
                    let d = k_src.dim().3;
                    let h_kv = k_src.dim().1;
                    for bb in 0..b {
                        for hk in 0..h_kv {
                            let src_k = k_src.slice(ndarray::s![bb, hk, 0, ..]);
                            let src_v = v_src.slice(ndarray::s![bb, hk, 0, ..]);
                            // 避免对 c 的两个可变借用重叠：用两个作用域分开 k/v 写入
                            {
                                let mut dst_k =
                                    c.k.slice_mut(ndarray::s![bb, hk, past_len, ..]);
                                dst_k
                                    .as_slice_mut()
                                    .expect("dst_k not contiguous")
                                    .copy_from_slice(
                                        src_k.as_slice().expect("src_k not contiguous"),
                                    );
                                debug_assert_eq!(dst_k.len(), d);
                            }
                            {
                                let mut dst_v =
                                    c.v.slice_mut(ndarray::s![bb, hk, past_len, ..]);
                                dst_v
                                    .as_slice_mut()
                                    .expect("dst_v not contiguous")
                                    .copy_from_slice(
                                        src_v.as_slice().expect("src_v not contiguous"),
                                    );
                                debug_assert_eq!(dst_v.len(), d);
                            }
                        }
                    }
                } else {
                    c.k.slice_mut(ndarray::s![.., .., past_len..new_len, ..])
                        .assign(&k_src);
                    c.v.slice_mut(ndarray::s![.., .., past_len..new_len, ..])
                        .assign(&v_src);
                }
                c.len = new_len;
            }

            // 6) GQA attention（不 repeat_kv）
            // 6) GQA attention（不 repeat_kv）。为了绕开 eval 的 permute/reshape copy，
            // 这里直接产出 [B,S,H,D]（BSHD）布局，后续 reshape 到 [B,S,H*D] 可视为 view。
            let context_bshd = {
                let c = cache_handle.borrow();
                let total_len = c.len;
                let q_ref = q_rot.data_ref();
                let q4 = q_ref
                    .view()
                    .into_dimensionality::<ndarray::Ix4>()
                    .unwrap(); // [B,H,S,D] view
                let k4 = c.k.slice(ndarray::s![.., .., 0..total_len, ..]); // view [B,H_kv,L,D]
                let v4 = c.v.slice(ndarray::s![.., .., 0..total_len, ..]); // view [B,H_kv,L,D]
                gqa_attention_no_repeat_bshd_view(&q4, &k4, &v4, self.scale, self.causal, n_rep, past_len)
            };

            // 7) 输出投影：context [B,S,H,D] -> [B,S,H*D] -> w_o （全 view，不触发 copy）
            let context = Tensor::from_data_no_grad(context_bshd.into_dyn().into_shared());
            let context = reshape(&context, vec![b as i32, s as i32, (h * d) as i32]);
            let output = self.w_o.forward(context);

            return (output, Some(cache_handle));
        }

        // train 路径：走原来的逻辑（可导）
        let q = permute(&reshape(&q, vec![b as i32, s as i32, h as i32, d as i32]), vec![0, 2, 1, 3]);
        let k = permute(&reshape(&k, vec![b as i32, s as i32, h_kv as i32, d as i32]), vec![0, 2, 1, 3]);
        let v = permute(&reshape(&v, vec![b as i32, s as i32, h_kv as i32, d as i32]), vec![0, 2, 1, 3]);

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
    let data_ref = x.data_ref();
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
// 返回 context: [B, S, H, D]（BSHD，便于后续 reshape 到 [B,S,H*D] 视为 view）
// past_len: cache 写入前的长度（用于 causal mask 的 absolute index）

fn gqa_attention_no_repeat_bshd_view(
    q: &ndarray::ArrayView4<f32>,
    k: &ndarray::ArrayView4<f32>,
    v: &ndarray::ArrayView4<f32>,
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

    let mut out = Array4::<f32>::zeros((b, s, h, d));

    // 将 out 变换为 [H,B,S,D]，对 head 维并行写入（线程间互不重叠）
    let mut out_hbsd = out.view_mut().permuted_axes([2, 0, 1, 3]);
    out_hbsd
        .outer_iter_mut()
        .into_par_iter()
        .enumerate()
        .for_each(|(hq, mut out_for_head)| {
            let hk = hq / n_rep;

            // decode(S=1) 热路径：online softmax + 直接累加 ctx，完全不需要 scores/ctx buffer
            if s == 1 {
                for bb in 0..b {
                    let q_vec = q.slice(ndarray::s![bb, hq, 0, ..]); // [D]
                    let k_mat = k.slice(ndarray::s![bb, hk, .., ..]); // [L,D]
                    let v_mat = v.slice(ndarray::s![bb, hk, .., ..]); // [L,D]

                    let q_abs = past_len; // i == 0

                    // 1) max
                    let mut maxv = f32::NEG_INFINITY;
                    for j in 0..l {
                        // causal mask
                        if causal && j > q_abs {
                            continue;
                        }
                        let kj = k_mat.slice(ndarray::s![j, ..]);
                        let mut dot = 0.0f32;
                        for i in 0..d {
                            dot += q_vec[i] * kj[i];
                        }
                        let score = dot * scale;
                        if score > maxv {
                            maxv = score;
                        }
                    }

                    // 2) sum + ctx
                    let mut sum = 0.0f32;
                    // out_for_head: [B,S,D] and S==1
                    let mut out_row = out_for_head.slice_mut(ndarray::s![bb, 0, ..]);
                    out_row.fill(0.0);

                    for j in 0..l {
                        if causal && j > q_abs {
                            continue;
                        }
                        let kj = k_mat.slice(ndarray::s![j, ..]);
                        let mut dot = 0.0f32;
                        for i in 0..d {
                            dot += q_vec[i] * kj[i];
                        }
                        let w = (dot * scale - maxv).exp();
                        sum += w;

                        for i in 0..d {
                            out_row[i] += w * v_mat[[j, i]];
                        }
                    }

                    let inv = 1.0f32 / (sum + 1e-9);
                    for i in 0..d {
                        out_row[i] *= inv;
                    }
                }
                return;
            }

            // per-thread buffer reuse
            ATT_SCORES_BUF.with(|sb| {
                ATT_CTX_BUF.with(|cb| {
                    let mut scores_buf = sb.borrow_mut();
                    let mut ctx_buf = cb.borrow_mut();

                    if scores_buf.len() != s * l {
                        scores_buf.resize(s * l, 0.0);
                    }
                    if ctx_buf.len() != s * d {
                        ctx_buf.resize(s * d, 0.0);
                    }

                    let mut scores = ndarray::ArrayViewMut2::from_shape((s, l), &mut scores_buf[..])
                        .expect("scores buffer shape mismatch");
                    let mut ctx = ndarray::ArrayViewMut2::from_shape((s, d), &mut ctx_buf[..])
                        .expect("ctx buffer shape mismatch");

                    for bb in 0..b {
                        let q_mat = q.slice(ndarray::s![bb, hq, .., ..]); // [S,D]
                        let k_mat = k.slice(ndarray::s![bb, hk, .., ..]); // [L,D]
                        let v_mat = v.slice(ndarray::s![bb, hk, .., ..]); // [L,D]

                        scores.fill(0.0);
                        general_mat_mul(1.0, &q_mat, &k_mat.t(), 0.0, &mut scores);
                        softmax_inplace_view(&mut scores, scale, causal, past_len);

                        ctx.fill(0.0);
                        general_mat_mul(1.0, &scores, &v_mat, 0.0, &mut ctx);

                        // out_for_head: [B,S,D]
                        out_for_head.slice_mut(ndarray::s![bb, .., ..]).assign(&ctx);
                    }
                })
            });
        });

    out
}

fn softmax_inplace_view(scores: &mut ndarray::ArrayViewMut2<f32>, scale: f32, causal: bool, past_len: usize) {
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
