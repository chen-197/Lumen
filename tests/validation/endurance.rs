//! Explicitly selected long checks; no timing assertions or external downloads.
use crate::accuracy_gate::{check, ordered, run, tensor, values};
use crate::autograd::{Device, Tensor, TensorData, no_grad};
use crate::init::{ParameterInitMode, with_parameter_init_mode};
use crate::layers::attention::self_attention::KVCache;
use crate::loss::MSELoss;
use crate::models::{LlamaConfig, LlamaModel};
use crate::module::Module;
use crate::ops::{cuda, matmul::matmul};
use crate::optim::{Adam, Optimizer, SGD};
use crate::precision::DType;
use ndarray::{ArrayD, IxDyn};
use serde_json::json;
use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::{Rc, Weak};

const INPUT: [f32; 12] = [1., 0., 0., 0., 1., 0., 0., 0., 1., 1., 1., 1.];
const TARGET: [f32; 8] = [0.35, -0.125, -0.15, 0.225, 0.15, -0.175, 0.25, -0.025];
const TYPES: [DType; 4] = [DType::F32, DType::F16, DType::BF16, DType::I8];

fn constant(shape: &[usize], data: &[f32], layout: usize, device: Device) -> Tensor {
    let t = Tensor::from_data_with_grad_flag(ordered(shape, data, layout), false);
    t.to_device_inplace(device);
    t
}

struct Pair {
    weight: Tensor,
    bias: Tensor,
}

impl Pair {
    fn new(dtype: DType, layout: usize, device: Device) -> Self {
        Self {
            weight: tensor(
                &[2, 3],
                &[-0.7, 0.6, 0.5, -0.8, 0.4, -0.3],
                dtype,
                layout,
                device,
            ),
            bias: tensor(&[2], &[0.2, -0.2], dtype, layout, device),
        }
    }

    fn optimizer(&self, kind: &str, dtype: DType) -> Box<dyn Optimizer> {
        if kind == "adam" {
            let opt = Adam::new_with_dtype(self.parameters(), 0.02, DType::F32);
            Box::new(if dtype == DType::I8 {
                opt.with_dynamic_i8_master_weights()
            } else {
                opt.with_fp32_master_weights()
            })
        } else {
            let opt = SGD::new_with_dtype(self.parameters(), 0.03, DType::F32)
                .with_momentum(if kind == "momentum" { 0.9 } else { 0. });
            Box::new(if dtype == DType::I8 {
                opt.with_dynamic_i8_master_weights()
            } else {
                opt.with_fp32_master_weights()
            })
        }
    }
}

impl Module for Pair {
    fn parameters(&self) -> Vec<Tensor> {
        vec![self.weight.clone(), self.bias.clone()]
    }

    fn forward(&self, input: Tensor) -> Tensor {
        &matmul(&input, &self.weight) + &self.bias
    }
}

// Direct scalar FP64 equations, independent of Lumen's matmul/loss/backward.
fn reference(model: &Pair) -> (f64, Vec<f64>, Vec<f64>) {
    let w = values(&model.weight);
    let b = values(&model.bias);
    let mut loss = 0.;
    let mut dw = vec![0.; 6];
    let mut db = vec![0.; 2];
    for row in 0..4 {
        for out in 0..2 {
            let y = b[out] as f64
                + (0..3)
                    .map(|k| INPUT[row * 3 + k] as f64 * w[out * 3 + k] as f64)
                    .sum::<f64>();
            let error = y - TARGET[row * 2 + out] as f64;
            loss += error * error / 8.;
            db[out] += error / 4.;
            for k in 0..3 {
                dw[out * 3 + k] += error * INPUT[row * 3 + k] as f64 / 4.;
            }
        }
    }
    (loss, dw, db)
}

fn training(device: Device) {
    for dtype in TYPES {
        for kind in ["sgd", "momentum", "adam"] {
            for layout in 0..3 {
                let model = Pair::new(dtype, layout, device);
                let initial = model.parameters().iter().map(values).collect::<Vec<_>>();
                let input = constant(&[4, 3], &INPUT, layout, device);
                let target = constant(&[4, 2], &TARGET, layout, device);
                let mut opt = model.optimizer(kind, dtype);
                let mut losses = Vec::new();
                for step in 0..=1000 {
                    opt.zero_grad();
                    assert!(model.parameters().iter().all(|p| !p.has_pending_grad()));
                    let loss = MSELoss::apply(&model.forward(input.clone()), &target);
                    let actual = values(&loss)[0];
                    assert!(
                        actual.is_finite(),
                        "{device:?}/{dtype:?}/{kind}/{layout}/{step}"
                    );
                    losses.push(actual);
                    let expected = (step % 100 == 0).then(|| reference(&model));
                    if let Some((expected, _, _)) = &expected {
                        check("long training loss", &[actual], &[*expected], 2e-7, 2e-5);
                    }
                    if step != 1000 {
                        loss.backward();
                        if let Some((_, dw, db)) = expected {
                            for (param, expected) in [(&model.weight, dw), (&model.bias, db)] {
                                let grad = param.grad().expect("trainable gradient");
                                check(
                                    "long training gradient",
                                    &grad.iter().copied().collect::<Vec<_>>(),
                                    &expected,
                                    2e-6,
                                    3e-5,
                                );
                            }
                        }
                        opt.step();
                    }
                    if step % 100 == 0 {
                        for param in model.parameters() {
                            assert_eq!(param.dtype(), dtype);
                            assert!(values(&param).iter().all(|v| v.is_finite()));
                        }
                    }
                }
                let head = losses[..100].iter().map(|x| *x as f64).sum::<f64>() / 100.;
                let tail = losses[901..].iter().map(|x| *x as f64).sum::<f64>() / 100.;
                let slope = losses
                    .iter()
                    .enumerate()
                    .map(|(i, x)| (i as f64 - 500.) * *x as f64)
                    .sum::<f64>();
                assert!(losses[1000] < losses[0] * 0.1 && tail < head && slope < 0.);
                println!(
                    "ENDURANCE_AUDIT={}",
                    json!({"kind":"training","device":format!("{device:?}"),"dtype":format!("{dtype:?}"),"optimizer":kind,"layout":layout,"steps":1000,"initial":initial,"final":model.parameters().iter().map(values).collect::<Vec<_>>(),"losses":losses,"sampled_fp64_loss_checks":11,"sampled_fp64_gradient_checks":20})
                );
            }
        }
    }
}

struct GraphProbes {
    nodes: Vec<Weak<RefCell<TensorData>>>,
    #[cfg(feature = "cuda")]
    buffers: Vec<cuda::BufferDropProbe>,
}

impl GraphProbes {
    fn new(roots: Vec<Tensor>) -> Self {
        let mut probes = Self {
            nodes: Vec::new(),
            #[cfg(feature = "cuda")]
            buffers: Vec::new(),
        };
        let mut stack = roots;
        let mut seen = HashSet::new();
        while let Some(t) = stack.pop() {
            if !seen.insert(Rc::as_ptr(&t.0)) {
                continue;
            }
            probes.nodes.push(Rc::downgrade(&t.0));
            let inner = t.0.borrow();
            #[cfg(feature = "cuda")]
            for buffer in [
                &inner.cuda_f32_data,
                &inner.cuda_f16_data,
                &inner.cuda_bf16_data,
                &inner.cuda_i8_data,
                &inner.cuda_f32_grad,
            ]
            .into_iter()
            .flatten()
            {
                probes.buffers.push(buffer.drop_probe());
            }
            stack.extend(inner.parents.iter().cloned());
        }
        probes
    }

    fn assert_released(&self) {
        assert!(
            self.nodes.iter().all(|node| node.strong_count() == 0),
            "discarded graph retained Tensor owners"
        );
        #[cfg(feature = "cuda")]
        assert!(
            self.buffers.iter().all(cuda::BufferDropProbe::is_released),
            "discarded graph retained CUDA buffer owners"
        );
    }

    fn buffer_count(&self) -> usize {
        #[cfg(feature = "cuda")]
        {
            self.buffers.len()
        }
        #[cfg(not(feature = "cuda"))]
        {
            0
        }
    }
}

fn lifecycle(device: Device) {
    let mut nodes = 0;
    let mut buffers = 0;
    for dtype in TYPES {
        for cycle in 0..64 {
            let probes = {
                let model = Pair::new(dtype, cycle % 3, device);
                let input = constant(&[4, 3], &INPUT, cycle % 3, device);
                let target = constant(&[4, 2], &TARGET, cycle % 3, device);
                let mut opt =
                    model.optimizer(if cycle % 2 == 0 { "adam" } else { "momentum" }, dtype);
                let prediction = model.forward(input);
                let loss = MSELoss::apply(&prediction, &target);
                for _ in 0..3 {
                    loss.backward();
                }
                opt.step();
                let snapshot = model.weight.detach();
                let cpu_copy = model.weight.to_cpu();
                GraphProbes::new(vec![loss, prediction, snapshot, cpu_copy])
            };
            probes.assert_released();
            nodes += probes.nodes.len();
            buffers += probes.buffer_count();
            if device == Device::Cuda && cycle % 16 == 15 {
                cuda::synchronize().unwrap();
                cuda::release_cached_memory().unwrap();
                let a = cuda::upload_f32(&[1., 2., 3., 4.]).unwrap();
                let y = cuda::matmul_f32_no_host(&a, &a, 2, 2, 2).unwrap();
                assert_eq!(cuda::download_f32(&y).unwrap(), [5., 11., 11., 25.]);
            }
        }
    }
    // Model/KV owners are tracked separately from training graphs.
    for _ in 0..16 {
        let probes = no_grad(|| {
            let model = small_model(DType::F32, device);
            let mut caches = model.init_kv_caches(1);
            let output = model.forward_last_logits(ids(&[1, 2, 3], 1, device), &mut caches, 0);
            let mut roots = model.parameters();
            roots.push(output);
            for cache in caches {
                let c = cache.borrow();
                roots.extend([c.k.clone(), c.v.clone()]);
            }
            GraphProbes::new(roots)
        });
        probes.assert_released();
        nodes += probes.nodes.len();
        buffers += probes.buffer_count();
    }
    if device == Device::Cuda {
        assert!(buffers > 0);
        cuda::synchronize().unwrap();
        cuda::release_cached_memory().unwrap();
    }
    println!(
        "ENDURANCE_AUDIT={}",
        json!({"kind":"lifecycle","device":format!("{device:?}"),"training_cycles":256,"model_cycles":16,"tensor_probes":nodes,"buffer_probes":buffers,"all_released":true,"scope":"Rust Tensor and CUDA buffer ownership; not process RSS or driver memory"})
    );
}

fn ids(data: &[usize], batch: usize, device: Device) -> Tensor {
    let input = Tensor::from_array_no_grad(
        ArrayD::from_shape_vec(
            IxDyn(&[batch, data.len() / batch]),
            data.iter().map(|v| *v as f32).collect(),
        )
        .unwrap(),
    );
    input.to_device_inplace(device);
    input
}

fn small_model(dtype: DType, device: Device) -> LlamaModel {
    let runtime = if dtype == DType::I8 {
        DType::F32
    } else {
        dtype
    };
    let model = with_parameter_init_mode(ParameterInitMode::Placeholder, || {
        LlamaModel::new_with_dtypes(
            LlamaConfig {
                vocab_size: 64,
                hidden_size: 32,
                intermediate_size: 64,
                num_hidden_layers: 2,
                num_attention_heads: 4,
                num_key_value_heads: 2,
                rms_norm_eps: 1e-5,
                max_seq_len: 513,
                rope_theta: 10000.,
            },
            dtype,
            runtime,
        )
    });
    for (name, param) in model.named_parameters() {
        let salt = name
            .bytes()
            .fold(0u32, |s, b| s.wrapping_mul(31).wrapping_add(b as u32));
        let a = ArrayD::from_shape_vec(
            IxDyn(&param.shape_vec()),
            (0..param.len())
                .map(|i| {
                    if name.contains("norm") {
                        1.
                    } else {
                        ((i as f32 + (salt % 997) as f32) * 0.17).sin() * 0.05
                    }
                })
                .collect(),
        )
        .unwrap();
        param.set_array_f32_with_dtype(
            a,
            if name == "lm_head.weight" {
                DType::F32
            } else {
                dtype
            },
        );
        param.to_device_inplace(device);
    }
    model
}

fn lengths(caches: &[KVCache], length: usize, device: Device) {
    for cache in caches {
        let c = cache.borrow();
        assert_eq!(c.len, length);
        assert_eq!(c.device(), device);
        assert_eq!(c.k.dtype(), c.dtype);
        assert_eq!(c.v.dtype(), c.dtype);
    }
}

fn logits(model: &LlamaModel, data: &[usize], batch: usize, caches: &mut [KVCache]) -> Vec<f32> {
    #[cfg(feature = "cuda")]
    let before = cuda::test_host_downloads();
    let device = caches[0].borrow().device();
    let output = model.forward_last_logits(ids(data, batch, device), caches, 0);
    #[cfg(feature = "cuda")]
    if output.is_cuda()
        && model.activation_dtype() == DType::F32
        && model.parameters().iter().all(|p| p.dtype() == DType::F32)
    {
        assert_eq!(
            cuda::test_host_downloads(),
            before,
            "F32 model inference downloaded intermediate data"
        );
    }
    assert_eq!(output.shape_vec(), vec![batch, 1, model.config.vocab_size]);
    assert!(!output.requires_grad());
    values(&output)
}

fn compare_logits(label: &str, got: &[f32], expected: &[f32], dtype: DType) -> f64 {
    let (atol, rtol) = match dtype {
        DType::F16 => (8e-4, 3e-3),
        DType::BF16 => (8e-3, 3e-2),
        _ => (2e-4, 2e-4),
    };
    check(
        label,
        got,
        &expected.iter().map(|x| *x as f64).collect::<Vec<_>>(),
        atol,
        rtol,
    );
    got.iter()
        .zip(expected)
        .map(|(a, b)| (*a as f64 - *b as f64).abs())
        .fold(0., f64::max)
}

fn cache_routes(
    model: &LlamaModel,
    data: &[usize],
    batch: usize,
    dtype: DType,
    device: Device,
    real: bool,
) {
    let row_len = 513;
    assert_eq!(data.len(), batch * row_len);
    let slice = |start: usize, end: usize| {
        (0..batch)
            .flat_map(|b| data[b * row_len + start..b * row_len + end].iter().copied())
            .collect::<Vec<_>>()
    };
    let mut chunked = model.init_kv_caches(batch);
    let mut tokenwise = model.init_kv_caches(batch);
    let mut reference_rows = Vec::new();
    let mut start = 0;
    for end in [127, 255, 508, 509, 510, 511, 512, 513] {
        let part = slice(start, end);
        let chunk = logits(model, &part, batch, &mut chunked);
        let mut single = Vec::new();
        for pos in start..end {
            single = logits(model, &slice(pos, pos + 1), batch, &mut tokenwise);
        }
        let mut fresh = model.init_kv_caches(batch);
        let expected = logits(model, &slice(0, end), batch, &mut fresh);
        let chunk_error = compare_logits("chunked/full prefill", &chunk, &expected, dtype);
        let token_error = compare_logits("tokenwise/full prefill", &single, &expected, dtype);
        lengths(&chunked, end, device);
        lengths(&tokenwise, end, device);
        lengths(&fresh, end, device);
        reference_rows.push(json!({"length":end,"chunk_max_absolute_error":chunk_error,"token_max_absolute_error":token_error,"full_logits":if real {Some(expected)} else {None}}));
        start = end;
    }
    // Reset a completely full cache, then use a different, shorter request.
    model.reset_kv_caches(&mut chunked);
    lengths(&chunked, 0, device);
    let other = (0..batch * 17)
        .map(|i| (i * 11 + 7) % model.config.vocab_size)
        .collect::<Vec<_>>();
    let reused = logits(model, &other, batch, &mut chunked);
    let mut fresh = model.init_kv_caches(batch);
    let expected = logits(model, &other, batch, &mut fresh);
    let reset_error = compare_logits("reset/fresh cache", &reused, &expected, dtype);
    lengths(&chunked, 17, device);
    println!(
        "ENDURANCE_AUDIT={}",
        json!({"kind":"real_weight_cache","real_weights":real,"device":format!("{device:?}"),"dtype":format!("{dtype:?}"),"batch":batch,"max_seq_len":513,"input_ids":if real {Some(data)} else {None},"comparisons":reference_rows,"reset_max_absolute_error":reset_error})
    );
}

fn compact_cache(device: Device) {
    no_grad(|| {
        for dtype in TYPES {
            let model = small_model(dtype, device);
            let data = (0..1026).map(|i| (i * 13 + 3) % 64).collect::<Vec<_>>();
            cache_routes(&model, &data, 2, dtype, device, false);
        }
    });
}

fn real_cache(device: Device) {
    use crate::loader::ModelLoader;
    use crate::tokenizer::LlamaTokenizer;
    let dir =
        std::path::PathBuf::from(std::env::var_os("LUMEN_LONG_MODEL_DIR").expect(
            "real-weight test requires LUMEN_LONG_MODEL_DIR; missing assets are not a pass",
        ));
    assert!(
        dir.join("model.safetensors").is_file() && dir.join("tokenizer.json").is_file(),
        "real-weight files missing: {}",
        dir.display()
    );
    no_grad(|| {
        let model = with_parameter_init_mode(ParameterInitMode::Placeholder, || {
            LlamaModel::new_with_dtype(
                LlamaConfig {
                    max_seq_len: 513,
                    ..LlamaConfig::default()
                },
                DType::F32,
            )
        });
        ModelLoader::load_llama_weights(dir.join("model.safetensors"), &model.named_parameters())
            .unwrap();
        for param in model.parameters() {
            param.to_device_inplace(device);
        }
        let tokenizer = LlamaTokenizer::from_file(dir.join("tokenizer.json")).unwrap();
        let text = "A neural network learns by adjusting its parameters using gradients computed from examples. ".repeat(80);
        let mut tokens = tokenizer.encode(&text, false).unwrap();
        assert!(tokens.len() >= 513);
        tokens.truncate(513);
        cache_routes(&model, &tokens, 1, DType::F32, device, true);
    });
    if device == Device::Cuda {
        cuda::synchronize().unwrap();
        cuda::release_cached_memory().unwrap();
    }
}

// Explicit selection fails on unavailable CUDA rather than returning success.
#[test]
#[ignore = "1000-step CPU training across dtypes, layouts and optimizers"]
fn cpu_long_training() {
    run(Device::Cpu, training);
}
#[test]
#[ignore = "CPU graph/model lifetime stress"]
fn cpu_graph_lifecycle() {
    run(Device::Cpu, lifecycle);
}
#[test]
#[ignore = "513-token CPU cache route and reset comparisons"]
fn cpu_long_kv_routes() {
    run(Device::Cpu, compact_cache);
}
#[test]
#[ignore = "requires local TinyLlama weights and tokenizer in LUMEN_LONG_MODEL_DIR"]
fn cpu_real_weight_long_kv() {
    run(Device::Cpu, real_cache);
}
#[cfg(feature = "cuda")]
#[test]
#[ignore = "1000-step strict CUDA training across dtypes, layouts and optimizers"]
fn cuda_long_training() {
    run(Device::Cuda, training);
}
#[cfg(feature = "cuda")]
#[test]
#[ignore = "CUDA graph/model lifetime stress and cache release"]
fn cuda_graph_lifecycle() {
    run(Device::Cuda, lifecycle);
}
#[cfg(feature = "cuda")]
#[test]
#[ignore = "513-token strict CUDA cache route and reset comparisons"]
fn cuda_long_kv_routes() {
    run(Device::Cuda, compact_cache);
}
#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires CUDA and local TinyLlama weights/tokenizer in LUMEN_LONG_MODEL_DIR"]
fn cuda_real_weight_long_kv() {
    run(Device::Cuda, real_cache);
}
