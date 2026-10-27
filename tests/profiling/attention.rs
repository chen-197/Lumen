//! Opt-in, same-process comparisons; excluded from normal library builds.
use super::*;
use rayon::prelude::*;
use std::{cell::Cell, hint::black_box, sync::Arc, time::Instant};

thread_local! {
    static LEGACY: Cell<bool> = const { Cell::new(false) };
}

pub(super) fn legacy_enabled() -> bool {
    LEGACY.with(Cell::get)
}

fn with_legacy<R>(legacy: bool, f: impl FnOnce() -> R) -> R {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            LEGACY.with(|v| v.set(self.0));
        }
    }
    let _restore = Restore(LEGACY.with(|v| v.replace(legacy)));
    f()
}

// Preserve the former head-only prefill algorithm for controlled comparisons.
pub(super) fn head_only(
    q: ndarray::ArrayView4<'_, f32>,
    k: ndarray::ArrayView4<'_, f32>,
    v: ndarray::ArrayView4<'_, f32>,
    scale: f32,
    causal: bool,
    past_len: usize,
) -> Array4<f32> {
    let (b, h, s, d) = q.dim();
    let l = k.shape()[2];
    let rep = h / k.shape()[1];
    let mut out = Array4::zeros((b, s, h, d));
    let mut heads = out.view_mut().permuted_axes([2, 0, 1, 3]);
    heads
        .outer_iter_mut()
        .into_par_iter()
        .enumerate()
        .for_each(|(hh, mut output)| {
            with_attention_work_buffers(s * l, s * d, |scores, ctx| {
                let mut scores = ndarray::ArrayViewMut2::from_shape((s, l), scores).unwrap();
                let mut ctx = ndarray::ArrayViewMut2::from_shape((s, d), ctx).unwrap();
                for bb in 0..b {
                    let q = q.slice(ndarray::s![bb, hh, .., ..]);
                    let k = k.slice(ndarray::s![bb, hh / rep, .., ..]);
                    let v = v.slice(ndarray::s![bb, hh / rep, .., ..]);
                    scores.fill(0.);
                    general_mat_mul(1., &q, &k.t(), 0., &mut scores);
                    softmax_inplace_view(&mut scores, scale, causal, past_len);
                    ctx.fill(0.);
                    general_mat_mul(1., &scores, &v, 0., &mut ctx);
                    output.slice_mut(ndarray::s![bb, .., ..]).assign(&ctx);
                }
            });
        });
    out
}

fn decode_leaf(row: usize, out: &mut [f32], q: &[f32], k: &[f32], v: &[f32], l: usize, d: usize) {
    ATT_Q_BUF.with(|qb| {
        let mut qb = qb.borrow_mut();
        qb.resize(d, 0.);
        qb.copy_from_slice(&q[row * d..(row + 1) * d]);
        ATT_CTX_BUF.with(|cb| {
            let mut cb = cb.borrow_mut();
            cb.resize(d, 0.);
            cb.fill(0.);
            let mut max = f32::NEG_INFINITY;
            let mut sum = 0.;
            for pos in 0..l {
                let offset = (row / 4 * l + pos) * d;
                let score = dot_unrolled(&qb, &k[offset..offset + d]) / (d as f32).sqrt();
                if score > max {
                    let scale = (max - score).exp();
                    for i in 0..d {
                        cb[i] = cb[i] * scale + v[offset + i];
                    }
                    sum = sum * scale + 1.;
                    max = score;
                } else {
                    let weight = (score - max).exp();
                    for i in 0..d {
                        cb[i] += weight * v[offset + i];
                    }
                    sum += weight;
                }
            }
            for i in 0..d {
                out[i] = cb[i] / (sum + 1e-9);
            }
        });
    });
}

#[test]
#[ignore = "manual CPU profile: set LUMEN_CPU_PROFILE_OUTPUT to a JSON path"]
fn compare_attention_schedulers() {
    let Some(path) = std::env::var_os("LUMEN_CPU_PROFILE_OUTPUT") else {
        return;
    };
    let mut results = Vec::new();
    for threads in [1, 8, 32] {
        let pool = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap(),
        );
        // Run both algorithms inside the same selected pool. In particular,
        // their output allocations must occur on the same class of thread.
        // Running only the legacy function inside install would confound the
        // comparison with allocation/first-touch placement.
        let context = crate::cpu::CpuKernelContext::default();
        for (b, h, s) in [(1, 32, 46), (1, 32, 178), (1, 1, 1024), (4, 1, 1024)] {
            let d = 64;
            let l = s;
            let hk = if h == 1 { 1 } else { 8 };
            let data = |shape, seed| {
                Array4::from_shape_fn(shape, |(bb, hh, ss, dd)| {
                    (((bb * 7 + hh * 13 + ss * 17 + dd * 19 + seed) % 67) as f32 - 33.) / 128.
                })
            };
            let q = data((b, h, s, d), 3);
            let k = data((b, hk, l, d), 11);
            let v = data((b, hk, l, d), 23);
            let reference = pool.install(|| {
                head_only(
                    q.view(),
                    k.view(),
                    v.view(),
                    (d as f32).sqrt().recip(),
                    true,
                    l - s,
                )
            });
            for cycle in 0..3 {
                for (label, legacy) in [
                    ("old", true),
                    ("current", false),
                    ("old-copy", true),
                    ("current", false),
                ] {
                    let run = || {
                        if legacy {
                            pool.install(|| {
                                head_only(
                                    q.view(),
                                    k.view(),
                                    v.view(),
                                    (d as f32).sqrt().recip(),
                                    true,
                                    l - s,
                                )
                            })
                        } else {
                            pool.install(|| {
                                context.scope(|| {
                                    gqa_attention_no_repeat_bshd_view(
                                        &q.view(),
                                        &k.view(),
                                        &v.view(),
                                        (d as f32).sqrt().recip(),
                                        true,
                                        h / hk,
                                        0,
                                    )
                                })
                            })
                        }
                    };
                    for _ in 0..5 {
                        black_box(run());
                    }
                    let iters = if s < 200 { 20 } else { 3 };
                    let mut samples = Vec::new();
                    for _ in 0..7 {
                        let start = Instant::now();
                        for _ in 0..iters {
                            black_box(run());
                        }
                        samples.push(start.elapsed().as_secs_f64() * 1e6 / iters as f64);
                    }
                    let output = run();
                    for (a, e) in output.iter().zip(reference.iter()) {
                        assert!((a - e).abs() < 2e-6);
                    }
                    results.push(serde_json::json!({"kind":"prefill-core","threads":threads,"b":b,"h":h,"s":s,"label":label,"cycle":cycle,"iters":iters,"samples_us":samples}));
                }
            }
        }
        for l in [46usize, 178] {
            let (h, d) = (32, 64);
            let data = |len, seed| {
                (0..len)
                    .map(|i| ((i * 17 + seed) % 67) as f32 / 128. - 0.25)
                    .collect::<Vec<_>>()
            };
            let q = data(h * d, 3);
            let k = data(h / 4 * l * d, 11);
            let v = data(h / 4 * l * d, 23);
            let mut output = vec![0.; h * d];
            for cycle in 0..3 {
                for (label, legacy) in [
                    ("old", true),
                    ("current", false),
                    ("old-copy", true),
                    ("current", false),
                ] {
                    let mut run = || {
                        if legacy {
                            pool.install(|| {
                                output
                                    .par_chunks_mut(d)
                                    .enumerate()
                                    .for_each(|(r, o)| decode_leaf(r, o, &q, &k, &v, l, d))
                            });
                        } else {
                            pool.install(|| {
                                context.scope(|| {
                                    crate::cpu::for_each_chunk_mut(
                                        &mut output,
                                        d,
                                        l * d * 3,
                                        |(r, o)| decode_leaf(r, o, &q, &k, &v, l, d),
                                    )
                                })
                            });
                        }
                    };
                    for _ in 0..5 {
                        run();
                    }
                    let mut samples = Vec::new();
                    for _ in 0..7 {
                        let start = Instant::now();
                        for _ in 0..50 {
                            run();
                        }
                        samples.push(start.elapsed().as_secs_f64() * 1e6 / 50.);
                    }
                    results.push(serde_json::json!({"kind":"decode-scheduler","threads":threads,"l":l,"label":label,"cycle":cycle,"iters":50,"samples_us":samples}));
                }
            }
        }
    }
    std::fs::write(path, serde_json::to_vec_pretty(&results).unwrap()).unwrap();
}

#[test]
#[ignore = "manual model ablation: set LUMEN_CPU_MODEL_PROFILE_OUTPUT and LUMEN_CPU_MODEL_DIR"]
fn compare_model_schedulers() {
    use crate::autograd::no_grad;
    use crate::init::{ParameterInitMode, with_parameter_init_mode};
    use crate::loader::{ModelLoader, WeightLoadOptions};
    use crate::models::{LlamaConfig, LlamaModel};
    use crate::precision::{
        ParameterQuantization, PrecisionConfig, with_parameter_quantization, with_precision_config,
        with_runtime_component_dtypes,
    };
    use crate::tokenizer::LlamaTokenizer;
    let Some(path) = std::env::var_os("LUMEN_CPU_MODEL_PROFILE_OUTPUT") else {
        return;
    };
    let dir = std::path::PathBuf::from(
        std::env::var_os("LUMEN_CPU_MODEL_DIR").expect("model directory required"),
    );
    let _cuda = crate::ops::cuda::set_enabled_scoped(false);
    let tokenizer = LlamaTokenizer::from_file(dir.join("tokenizer.json")).unwrap();
    let config = LlamaConfig {
        max_seq_len: 512,
        ..LlamaConfig::default()
    };
    let tensor = |ids: &[usize]| {
        Tensor::from_array_no_grad(
            Array::from_shape_vec((1, ids.len()), ids.iter().map(|&i| i as f32).collect())
                .unwrap()
                .into_dyn(),
        )
    };
    let mut results = Vec::new();
    for (case, parameter, runtime, prompt) in [
        (
            "f16-short",
            DType::F16,
            DType::F16,
            "Explain how neural networks learn from data in one detailed paragraph.".to_string(),
        ),
        (
            "i8-long",
            DType::I8,
            DType::BF16,
            "Explain how neural networks learn from data. ".repeat(16),
        ),
    ] {
        let quant = if parameter == DType::I8 {
            ParameterQuantization::Int8
        } else {
            ParameterQuantization::Disabled
        };
        let model = with_precision_config(
            PrecisionConfig {
                parameter_dtype: parameter,
                runtime_dtype: runtime,
                allow_parameter_dtype_copies: false,
            },
            || {
                with_runtime_component_dtypes(Some(runtime), Some(runtime), || {
                    with_parameter_quantization(quant, || {
                        let model =
                            with_parameter_init_mode(ParameterInitMode::Placeholder, || {
                                LlamaModel::new(config.clone())
                            });
                        ModelLoader::load_llama_weights_with_options(
                            dir.join("model.safetensors"),
                            &model.named_parameters(),
                            WeightLoadOptions {
                                float_source_quantization: quant,
                                stream_from_disk: false,
                            },
                        )
                        .unwrap();
                        model
                    })
                })
            },
        );
        let prompt = format!(
            "<|system|>\nYou are a helpful assistant.\n</s>\n<|user|>\n{prompt}\n</s>\n<|assistant|>\n"
        );
        let tokens = tokenizer.encode(&prompt, false).unwrap();
        let run = || {
            no_grad(|| {
                let mut caches = model.init_kv_caches(1);
                model.reset_kv_caches(&mut caches);
                let input = tensor(&tokens);
                let start = Instant::now();
                let mut next = model.forward_last_argmax(input, &mut caches, 0);
                let prefill_us = start.elapsed().as_secs_f64() * 1e6;
                let mut ids = Vec::new();
                let mut decode_us = 0.;
                for _ in 0..32 {
                    ids.push(next);
                    let input = tensor(&[next]);
                    let start = Instant::now();
                    next = model.forward_last_argmax(input, &mut caches, 0);
                    decode_us += start.elapsed().as_secs_f64() * 1e6;
                }
                (prefill_us, decode_us, ids)
            })
        };
        let mut reference = None;
        for cycle in 0..2 {
            for (batch, legacy) in [true, false, false, true].into_iter().enumerate() {
                with_legacy(legacy, || {
                    for sample in 0..4 {
                        let (prefill_us, decode_us, ids) = run();
                        if let Some(expected) = &reference {
                            assert_eq!(&ids, expected);
                        } else {
                            reference = Some(ids.clone());
                        }
                        results.push(serde_json::json!({"case":case,"threads":rayon::current_num_threads(),"cycle":cycle,"batch":batch,"legacy":legacy,"warmup":sample==0,"sample":sample,"prompt_tokens":tokens.len(),"prefill_us":prefill_us,"decode_us":decode_us,"ids":ids}));
                    }
                });
                std::fs::write(&path, serde_json::to_vec_pretty(&results).unwrap()).unwrap();
            }
        }
    }
}
