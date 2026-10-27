//! Architecture-independent examples and independent numerical audit records.
use crate::accuracy_gate::{check, ordered, run, tensor, values};
use crate::autograd::{Device, Tensor, no_grad};
use crate::layers::{
    AttentionMask, AttentionOptions, LayerNorm, Linear, scaled_dot_product_attention,
};
use crate::module::Module;
use crate::optim::{Adam, Optimizer, SGD};
use crate::precision::DType;
use ndarray::{ArrayD, IxDyn};
use serde_json::{Value, json};

fn samples(len: usize, seed: usize) -> Vec<f32> {
    (0..len)
        .map(|i| (((i * 7 + seed) % 29) as f32 - 14.0) * 0.09375)
        .collect()
}
fn make(shape: &[usize], dtype: DType, layout: usize, seed: usize, device: Device) -> Tensor {
    tensor(
        shape,
        &samples(shape.iter().product(), seed),
        dtype,
        layout,
        device,
    )
}
fn record(x: &Tensor) -> Value {
    json!({"shape": x.shape_vec(), "values": values(x), "grad": x.grad().map(|g| g.iter().copied().collect::<Vec<_>>())})
}
fn trace(value: Value) {
    if std::env::var_os("LUMEN_OPERATOR_AUDIT").is_some() {
        println!("GENERIC_NN_AUDIT={value}");
    }
}

fn norm_reference(
    x: &[f64],
    w: Option<&[f64]>,
    b: Option<&[f64]>,
    dim: usize,
    eps: f64,
) -> Vec<f64> {
    x.chunks(dim)
        .flat_map(|row| {
            let mean = row.iter().sum::<f64>() / dim as f64;
            let variance = row.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / dim as f64;
            row.iter().enumerate().map(move |(j, &v)| {
                (v - mean) / (variance + eps).sqrt() * w.map_or(1.0, |w| w[j])
                    + b.map_or(0.0, |b| b[j])
            })
        })
        .collect()
}

fn norm_cases(device: Device) {
    let cases: Vec<(Vec<usize>, Vec<usize>, bool, bool)> = vec![
        (vec![2, 3, 5], vec![5], true, true),
        (vec![2, 3, 5], vec![3, 5], true, false),
        (vec![3, 1], vec![1], false, false),
        (vec![2, 7], vec![7], true, true),
        (vec![2, 513], vec![513], true, true),
        (vec![2, 0, 5], vec![5], true, true),
        (vec![2, 3, 7], vec![3, 7], false, false),
        (vec![1, 257], vec![257], true, true),
    ];
    for dtype in [DType::F32, DType::F16, DType::BF16] {
        for layout in 0..3 {
            for (case, (shape, normalized, affine, bias)) in cases.iter().enumerate() {
                let mut data = samples(shape.iter().product(), 3);
                if case == 3 {
                    data.fill(0.25);
                }
                if case == 4 {
                    for v in &mut data {
                        *v = 10000.0 + *v * 0.03125;
                    }
                }
                let x = tensor(shape, &data, dtype, layout, device);
                let norm = LayerNorm::with_options(normalized.clone(), 1e-5, *affine, *bias, dtype);
                for (i, p) in norm.parameters().iter().enumerate() {
                    p.set_array_f32_with_dtype(
                        ordered(normalized, &samples(p.len(), 9 + i), layout),
                        dtype,
                    );
                    p.to_device_inplace(device);
                }
                let y = norm.forward(x.clone());
                assert_eq!(y.dtype(), DType::F32);
                let dy = ordered(shape, &samples(y.len(), 19), layout);
                let xv = values(&x).iter().map(|&v| v as f64).collect::<Vec<_>>();
                let wv = norm
                    .weight
                    .as_ref()
                    .map(|p| values(p).iter().map(|&v| v as f64).collect::<Vec<_>>());
                let bv = norm
                    .bias
                    .as_ref()
                    .map(|p| values(p).iter().map(|&v| v as f64).collect::<Vec<_>>());
                let expected = norm_reference(
                    &xv,
                    wv.as_deref(),
                    bv.as_deref(),
                    normalized.iter().product(),
                    norm.eps() as f64,
                );
                check("generic LayerNorm", &values(&y), &expected, 2e-6, 2e-6);
                y.backward_with_grad(dy.clone());
                let infer = no_grad(|| norm.forward(x.clone()));
                check(
                    "LayerNorm inference",
                    &values(&infer),
                    &expected,
                    2e-6,
                    2e-6,
                );
                trace(
                    json!({"op":"layer_norm", "device":format!("{device:?}"),"dtype":format!("{dtype:?}"),"layout":layout,"case":case,"normalized_shape":normalized,"eps":norm.eps(),"x":record(&x),"weight":norm.weight.as_ref().map(record),"bias":norm.bias.as_ref().map(record),"y":values(&y),"dy":dy.iter().copied().collect::<Vec<_>>(),"inference_y":values(&infer)}),
                );
            }
        }
    }
}

fn broadcast_index(shape: &[usize], coord: [usize; 4]) -> usize {
    shape.iter().enumerate().fold(0, |offset, (axis, &n)| {
        offset * n
            + if n == 1 {
                0
            } else {
                coord[4 - shape.len() + axis]
            }
    })
}

#[allow(clippy::type_complexity)]
fn attention_spec(
    case: usize,
) -> (
    Vec<usize>,
    Vec<usize>,
    Vec<usize>,
    Option<Vec<usize>>,
    bool,
    bool,
    usize,
    Option<f32>,
) {
    match case {
        0 => (
            vec![2, 3, 4, 5],
            vec![2, 3, 4, 5],
            vec![2, 3, 4, 7],
            None,
            false,
            false,
            0,
            None,
        ),
        1 => (
            vec![2, 2, 3, 5],
            vec![2, 2, 7, 5],
            vec![2, 2, 7, 3],
            Some(vec![2, 1, 1, 7]),
            false,
            false,
            0,
            None,
        ),
        2 => (
            vec![1, 4, 3, 5],
            vec![1, 2, 5, 5],
            vec![1, 2, 5, 7],
            Some(vec![3, 5]),
            false,
            true,
            1,
            Some(0.7),
        ),
        3 => (
            vec![2, 3, 1, 7],
            vec![2, 1, 9, 7],
            vec![2, 1, 9, 5],
            None,
            false,
            true,
            8,
            None,
        ),
        4 => (
            vec![1, 2, 5, 3],
            vec![1, 2, 3, 3],
            vec![1, 2, 3, 5],
            None,
            false,
            true,
            0,
            Some(-0.25),
        ),
        5 => (
            vec![2, 2, 3, 7],
            vec![2, 2, 5, 7],
            vec![2, 2, 5, 4],
            Some(vec![1, 1, 3, 5]),
            true,
            false,
            0,
            Some(0.4),
        ),
        6 => (
            vec![1, 4, 3, 5],
            vec![1, 1, 3, 5],
            vec![1, 1, 3, 3],
            None,
            false,
            true,
            0,
            None,
        ),
        7 => (
            vec![0, 2, 3, 5],
            vec![0, 2, 5, 5],
            vec![0, 2, 5, 3],
            Some(vec![3, 5]),
            false,
            true,
            0,
            None,
        ),
        _ => (
            vec![1, 1, 1, 1],
            vec![1, 1, 1, 1],
            vec![1, 1, 1, 1],
            Some(vec![]),
            false,
            false,
            0,
            Some(0.0),
        ),
    }
}

fn attention_reference(
    q: &[f64],
    k: &[f64],
    v: &[f64],
    qs: &[usize],
    ks: &[usize],
    vs: &[usize],
    keep: Option<&ArrayD<bool>>,
    bias: Option<(&[usize], &[f64])>,
    options: AttentionOptions,
) -> Vec<f64> {
    let mut output = vec![0.0; qs[0] * qs[1] * qs[2] * vs[3]];
    for batch in 0..qs[0] {
        for head in 0..qs[1] {
            for i in 0..qs[2] {
                let kvhead = head / (qs[1] / ks[1]);
                let mut row = vec![f64::NEG_INFINITY; ks[2]];
                for (j, s) in row.iter_mut().enumerate() {
                    if (options.causal && j > options.query_offset + i)
                        || keep.is_some_and(|a| {
                            !a.iter()
                                .nth(broadcast_index(a.shape(), [batch, head, i, j]))
                                .unwrap()
                        })
                    {
                        continue;
                    }
                    let dot = (0..qs[3])
                        .map(|d| {
                            q[((batch * qs[1] + head) * qs[2] + i) * qs[3] + d]
                                * k[((batch * ks[1] + kvhead) * ks[2] + j) * ks[3] + d]
                        })
                        .sum::<f64>();
                    *s = dot * options.scale.unwrap_or((qs[3] as f32).sqrt().recip()) as f64
                        + bias.map_or(0.0, |(shape, b)| {
                            b[broadcast_index(shape, [batch, head, i, j])]
                        });
                }
                let max = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                if max == f64::NEG_INFINITY {
                    continue;
                }
                let denominator = row.iter().map(|s| (s - max).exp()).sum::<f64>();
                for d in 0..vs[3] {
                    output[((batch * qs[1] + head) * qs[2] + i) * vs[3] + d] = row
                        .iter()
                        .enumerate()
                        .map(|(j, s)| {
                            (s - max).exp() / denominator
                                * v[((batch * vs[1] + kvhead) * vs[2] + j) * vs[3] + d]
                        })
                        .sum();
                }
            }
        }
    }
    output
}

fn attention_cases(device: Device) {
    for dtype in [DType::F32, DType::F16, DType::BF16] {
        for layout in 0..3 {
            for case in 0..9 {
                let (qs, ks, vs, ms, additive, causal, offset, scale) = attention_spec(case);
                let q = make(&qs, dtype, layout, 3, device);
                let k = make(&ks, dtype, layout, 7, device);
                let v = make(&vs, dtype, layout, 11, device);
                let keep = (!additive)
                    .then(|| {
                        ms.as_ref().map(|s| {
                            ArrayD::from_shape_fn(IxDyn(s), |i| {
                                if case == 8 {
                                    return false;
                                }
                                if case == 1 && i[0] == 1 {
                                    return false;
                                }
                                if case == 2 && i[0] == 1 {
                                    return false;
                                }
                                i[s.len() - 1] % 3 != 1
                            })
                        })
                    })
                    .flatten();
                let bias =
                    additive.then(|| make(ms.as_ref().unwrap(), DType::F32, layout, 5, device));
                let options = AttentionOptions {
                    scale,
                    causal,
                    query_offset: offset,
                };
                let forward = || {
                    scaled_dot_product_attention(
                        &q,
                        &k,
                        &v,
                        if let Some(b) = &bias {
                            Some(AttentionMask::Additive(b))
                        } else {
                            keep.as_ref().map(AttentionMask::Keep)
                        },
                        options,
                    )
                };
                let y = forward();
                let dy = ordered(&y.shape_vec(), &samples(y.len(), 17), layout);
                let to64 = |t: &Tensor| values(t).iter().map(|&v| v as f64).collect::<Vec<_>>();
                let bv = bias.as_ref().map(to64);
                let expected = attention_reference(
                    &to64(&q),
                    &to64(&k),
                    &to64(&v),
                    &qs,
                    &ks,
                    &vs,
                    keep.as_ref(),
                    bv.as_ref()
                        .map(|b| (ms.as_ref().unwrap().as_slice(), b.as_slice())),
                    options,
                );
                check(
                    "generic attention training",
                    &values(&y),
                    &expected,
                    8e-6,
                    8e-6,
                );
                y.backward_with_grad(dy.clone());
                let infer = no_grad(forward);
                // Native lowp inference rounds intermediate scores/probabilities/output.
                let tolerance = match dtype {
                    DType::F32 => 1e-5,
                    DType::F16 => 0.002,
                    DType::BF16 => 0.015,
                    _ => unreachable!(),
                };
                check(
                    "generic attention inference",
                    &values(&infer),
                    &expected,
                    tolerance,
                    tolerance,
                );
                trace(
                    json!({"op":"attention","device":format!("{device:?}"),"dtype":format!("{dtype:?}"),"layout":layout,"case":case,"q":record(&q),"k":record(&k),"v":record(&v),"bias":bias.as_ref().map(record),"keep":keep.as_ref().map(|m|json!({"shape":m.shape(),"values":m.iter().copied().collect::<Vec<_>>()})),"scale":options.scale.unwrap_or((qs[3] as f32).sqrt().recip()),"causal":causal,"offset":offset,"y":values(&y),"dy":dy.iter().copied().collect::<Vec<_>>(),"inference_y":values(&infer),"inference_dtype":format!("{:?}",infer.dtype())}),
                );
            }
        }
    }
}

#[test]
fn cpu_generic_layer_norm_accuracy() {
    run(Device::Cpu, norm_cases);
}

#[test]
fn cpu_kv_head_repetition_preserves_logical_layout_and_dtype() {
    run(Device::Cpu, |device| {
        for dtype in [DType::F32, DType::F16, DType::BF16, DType::I8] {
            for layout in 0..3 {
                let x = make(&[2, 2, 3, 5], dtype, layout, 3, device);
                let source = values(&x);
                let y =
                    no_grad(|| crate::layers::attention::self_attention::repeat_kv(x.clone(), 3));
                assert_eq!(y.dtype(), dtype);
                assert_eq!(y.quantization_scale(), x.quantization_scale());
                let mut expected = Vec::new();
                for b in 0..2 {
                    for h in 0..2 {
                        for _ in 0..3 {
                            expected
                                .extend_from_slice(&source[(b * 2 + h) * 15..(b * 2 + h + 1) * 15]);
                        }
                    }
                }
                assert_eq!(values(&y), expected);
            }
        }
    });
}
#[test]
fn cpu_generic_attention_accuracy() {
    run(Device::Cpu, attention_cases);
}
#[cfg(feature = "cuda")]
#[test]
fn cuda_generic_layer_norm_accuracy() {
    run(Device::Cuda, norm_cases);
}
#[cfg(feature = "cuda")]
#[test]
fn cuda_generic_attention_accuracy() {
    run(Device::Cuda, attention_cases);
}

fn lifecycle(device: Device) {
    for flags in 0..8 {
        let x = make(&[2, 3], DType::F32, 1, 3, device);
        let norm = LayerNorm::new(vec![3], 1e-5);
        norm.to_device(device);
        let mut inputs = vec![x.clone()];
        inputs.extend(norm.parameters());
        for (i, p) in inputs.iter().enumerate() {
            p.0.borrow_mut().requires_grad = flags & (1 << i) != 0;
        }
        let y = norm.forward(x);
        assert_eq!(y.requires_grad(), flags != 0);
        if flags != 0 {
            y.backward();
            let first = inputs.iter().map(Tensor::grad).collect::<Vec<_>>();
            y.backward();
            for (i, p) in inputs.iter().enumerate() {
                assert_eq!(p.grad().is_some(), flags & (1 << i) != 0);
                if let Some(first) = &first[i] {
                    check(
                        "retained norm",
                        &p.grad().unwrap().iter().copied().collect::<Vec<_>>(),
                        &first.iter().map(|&v| 2.0 * v as f64).collect::<Vec<_>>(),
                        2e-5,
                        2e-5,
                    );
                }
            }
        }
        let weak = Rc::downgrade(&y.0);
        drop(y);
        assert!(weak.upgrade().is_none());
        assert!(!no_grad(|| norm.forward(inputs[0].clone())).requires_grad());
    }
    for flags in 0..8 {
        let inputs = [
            make(&[1, 2, 3, 5], DType::F32, 2, 3, device),
            make(&[1, 1, 4, 5], DType::F32, 1, 5, device),
            make(&[1, 1, 4, 3], DType::F32, 2, 7, device),
        ];
        for (i, p) in inputs.iter().enumerate() {
            p.0.borrow_mut().requires_grad = flags & (1 << i) != 0;
        }
        let keep = ArrayD::from_elem(IxDyn(&[3, 4]), true);
        let y = scaled_dot_product_attention(
            &inputs[0],
            &inputs[1],
            &inputs[2],
            Some(AttentionMask::Keep(&keep)),
            AttentionOptions::default(),
        );
        assert_eq!(y.requires_grad(), flags != 0);
        if flags != 0 {
            y.backward();
            for (i, p) in inputs.iter().enumerate() {
                assert_eq!(p.grad().is_some(), flags & (1 << i) != 0);
            }
        }
        let weak = Rc::downgrade(&y.0);
        drop(y);
        assert!(weak.upgrade().is_none());
    }
}
use std::rc::Rc;

#[cfg(feature = "cuda")]
#[test]
fn cuda_generic_forward_backward_keeps_tensors_resident() {
    run(Device::Cuda, |device| {
        use crate::ops::cuda;
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            let x = make(&[2, 513], dtype, 2, 3, device);
            let norm = LayerNorm::with_options(vec![513], 1e-5, true, true, dtype);
            norm.to_device(device);
            let before = cuda::test_host_downloads();
            let y = norm.forward(x);
            y.backward_with_cuda_grad(cuda::upload_f32(&samples(y.len(), 9)).unwrap());
            assert_eq!(
                cuda::test_host_downloads(),
                before,
                "LayerNorm downloaded an intermediate"
            );
            let q = make(&[2, 4, 3, 5], dtype, 1, 3, device);
            let k = make(&[2, 2, 7, 5], dtype, 2, 5, device);
            let v = make(&[2, 2, 7, 3], dtype, 1, 7, device);
            let keep = ArrayD::from_elem(IxDyn(&[2, 1, 1, 7]), true);
            let before = cuda::test_host_downloads();
            let y = scaled_dot_product_attention(
                &q,
                &k,
                &v,
                Some(AttentionMask::Keep(&keep)),
                AttentionOptions::default(),
            );
            y.backward_with_cuda_grad(cuda::upload_f32(&samples(y.len(), 11)).unwrap());
            assert_eq!(
                cuda::test_host_downloads(),
                before,
                "attention downloaded an intermediate"
            );
        }
        // Exceed the row-grid cap to exercise per-block grid-stride reuse.
        let shape = [65537, 3];
        let x = make(&shape, DType::F32, 0, 3, device);
        let norm = LayerNorm::new(vec![3], 1e-5);
        norm.to_device(device);
        let expected = norm_reference(
            &values(&x).iter().map(|&v| v as f64).collect::<Vec<_>>(),
            None,
            None,
            3,
            norm.eps() as f64,
        );
        let y = norm.forward(x.clone());
        check(
            "LayerNorm CUDA grid stride",
            &values(&y),
            &expected,
            2e-6,
            2e-6,
        );
        y.backward();
        assert!(
            x.grad()
                .unwrap()
                .iter()
                .all(|v| v.is_finite() && v.abs() < 1e-5)
        );
        check(
            "LayerNorm grid bias",
            &norm
                .bias
                .as_ref()
                .unwrap()
                .grad()
                .unwrap()
                .iter()
                .copied()
                .collect::<Vec<_>>(),
            &[65537.0; 3],
            0.0,
            0.0,
        );
    });
}
#[test]
fn cpu_generic_frozen_inputs_and_graph_release() {
    run(Device::Cpu, lifecycle);
}
#[cfg(feature = "cuda")]
#[test]
fn cuda_generic_frozen_inputs_and_graph_release() {
    run(Device::Cuda, lifecycle);
}

fn training(device: Device) {
    for cross in [false, true] {
        for adam in [false, true] {
            let norm = LayerNorm::new(vec![5], 1e-5);
            let projection = Linear::new(5, 5);
            let readout = Linear::new(5, 2);
            let mut params = norm.parameters();
            params.extend(projection.parameters());
            params.extend(readout.parameters());
            for (i, p) in params.iter().enumerate() {
                p.set_array_f32_with_dtype(
                    ArrayD::from_shape_vec(
                        IxDyn(&p.shape_vec()),
                        samples(p.len(), i + 2).iter().map(|&v| v * 0.2).collect(),
                    )
                    .unwrap(),
                    DType::F32,
                );
                p.to_device_inplace(device);
            }
            let x = make(&[2, 3, 5], DType::F32, 1, 3, device).detach();
            let memory = make(&[2, 1, 4, 5], DType::F32, 2, 9, device).detach();
            let target = make(&[2, 3, 2], DType::F32, 1, 11, device).detach();
            let mut optimizer: Box<dyn Optimizer> = if adam {
                Box::new(Adam::new(params.clone(), 0.02))
            } else {
                Box::new(SGD::new(params.clone(), 0.1))
            };
            let mut losses = Vec::new();
            for _ in 0..100 {
                optimizer.zero_grad();
                let q = norm
                    .forward(projection.forward(x.clone()))
                    .reshape(vec![2, 1, 3, 5]);
                let (k, v) = if cross {
                    (memory.clone(), memory.clone())
                } else {
                    (q.clone(), q.clone())
                };
                let context =
                    scaled_dot_product_attention(&q, &k, &v, None, AttentionOptions::default())
                        .reshape(vec![2, 3, 5]);
                let output = readout.forward(context);
                let loss = crate::loss::MSELoss::apply(&output, &target);
                let value = values(&loss)[0];
                assert!(value.is_finite());
                losses.push(value);
                loss.backward();
                for p in &params {
                    assert!(p.grad().unwrap().iter().all(|v| v.is_finite()));
                }
                optimizer.step();
            }
            let first = losses[..10].iter().sum::<f32>() / 10.0;
            let last = losses[90..].iter().sum::<f32>() / 10.0;
            assert!(
                last < first && losses[99] < losses[0] * 0.8,
                "generic training did not descend: {losses:?}"
            );
            trace(
                json!({"op":"training","device":format!("{device:?}"),"cross":cross,"adam":adam,"losses":losses}),
            );
        }
    }
}
#[test]
fn cpu_generic_encoder_and_cross_attention_training() {
    run(Device::Cpu, training);
}
#[cfg(feature = "cuda")]
#[test]
fn cuda_generic_encoder_and_cross_attention_training() {
    run(Device::Cuda, training);
}

#[test]
fn generic_validation_rejects_invalid_shapes_and_options() {
    for action in [0, 1, 2, 3, 4, 5] {
        let result = std::panic::catch_unwind(|| match action {
            0 => {
                LayerNorm::new(vec![0], 1e-5);
            }
            1 => {
                LayerNorm::new(vec![3], f32::NAN);
            }
            2 => {
                LayerNorm::new(vec![3], 1e-5)
                    .forward(Tensor::from_array_no_grad(ArrayD::zeros(IxDyn(&[2, 4]))));
            }
            _ => {
                let q = make(&[1, 2, 3, 5], DType::F32, 0, 3, Device::Cpu);
                let k = make(
                    &[1, if action == 3 { 3 } else { 2 }, 4, 5],
                    DType::F32,
                    0,
                    5,
                    Device::Cpu,
                );
                let v = make(
                    &[1, if action == 3 { 3 } else { 2 }, 4, 7],
                    DType::F32,
                    0,
                    7,
                    Device::Cpu,
                );
                let bad = ArrayD::from_elem(IxDyn(&[2, 2]), true);
                scaled_dot_product_attention(
                    &q,
                    &k,
                    &v,
                    if action == 4 {
                        Some(AttentionMask::Keep(&bad))
                    } else {
                        None
                    },
                    AttentionOptions {
                        scale: if action == 5 {
                            Some(f32::INFINITY)
                        } else {
                            None
                        },
                        ..Default::default()
                    },
                );
            }
        });
        assert!(
            result.is_err(),
            "invalid generic configuration {action} was accepted"
        );
    }
}

fn quantized_projection_compatibility(device: Device) {
    use crate::layers::{RotaryEmbedding, SelfAttention};
    use crate::ops::{fused::fused_softmax, matmul::batch_matmul};
    for flags in 0..16 {
        let attention = SelfAttention::new_with_runtime_dtypes(
            8,
            2,
            1,
            8,
            10000.0,
            true,
            DType::I8,
            DType::I8,
            DType::F32,
        );
        let params = attention.parameters();
        assert_eq!(params.len(), 4);
        for (i, p) in params.iter().enumerate() {
            p.set_array_f32_with_dtype(
                ordered(&p.shape_vec(), &samples(p.len(), i + 3), 2),
                DType::I8,
            );
            p.0.borrow_mut().requires_grad = flags & (1 << i) != 0;
        }
        attention.to_device(device);
        let x = make(&[1, 3, 8], DType::I8, 2, 5, device).detach();
        let rope = RotaryEmbedding::new_with_dtype(4, 8, 10000.0, DType::I8);
        rope.to_device(device);
        // Reconstruct the pre-refactor public operator sequence, including each
        // frozen projection's native I8 rounding and its own gradient flag.
        let q = attention
            .w_q
            .forward(x.clone())
            .reshape(vec![1, 3, 2, 4])
            .permute(vec![0, 2, 1, 3]);
        let k = attention
            .w_k
            .forward(x.clone())
            .reshape(vec![1, 3, 1, 4])
            .permute(vec![0, 2, 1, 3]);
        let v = attention
            .w_v
            .forward(x.clone())
            .reshape(vec![1, 3, 1, 4])
            .permute(vec![0, 2, 1, 3]);
        // The old CUDA layer already selected fused QKV when any projection
        // required gradients. Preserve that selection in this compatibility
        // oracle; fused and separate frozen projections round differently.
        #[cfg(feature = "cuda")]
        let (q, k, v) = crate::ops::fused::fused_qkv_train_tensors(
            &x,
            &attention.w_q.weight,
            &attention.w_k.weight,
            &attention.w_v.weight,
            attention.w_q.bias.as_ref(),
            attention.w_k.bias.as_ref(),
            attention.w_v.bias.as_ref(),
            2,
            1,
        )
        .unwrap_or((q, k, v));
        let q = rope.forward(&q, 0);
        let k = crate::layers::attention::self_attention::repeat_kv(rope.forward(&k, 0), 2);
        let v = crate::layers::attention::self_attention::repeat_kv(v, 2);
        let probabilities = fused_softmax(&batch_matmul(&q, &k.transpose(2, 3)), 0.5, true);
        let context = batch_matmul(&probabilities, &v)
            .permute(vec![0, 2, 1, 3])
            .reshape(vec![1, 3, 8]);
        let reference = attention.w_o.forward(context);
        let (actual, cache) = attention.forward(x.clone(), None);
        assert!(cache.is_none());
        assert_eq!(actual.requires_grad(), reference.requires_grad());
        assert_eq!(actual.dtype(), reference.dtype());
        check(
            "quantized self-attention compatibility",
            &values(&actual),
            &values(&reference)
                .iter()
                .map(|&v| v as f64)
                .collect::<Vec<_>>(),
            3e-5,
            3e-5,
        );
        if flags != 0 {
            let upstream = ordered(&actual.shape_vec(), &samples(actual.len(), 17), 1);
            reference.backward_with_grad(upstream.clone());
            let gradients = params.iter().map(Tensor::grad).collect::<Vec<_>>();
            for p in &params {
                p.zero_grad();
            }
            actual.backward_with_grad(upstream);
            for (i, p) in params.iter().enumerate() {
                assert_eq!(p.grad().is_some(), flags & (1 << i) != 0);
                if let Some(g) = &gradients[i] {
                    check(
                        "quantized projection gradient compatibility",
                        &p.grad().unwrap().iter().copied().collect::<Vec<_>>(),
                        &g.iter().map(|&v| v as f64).collect::<Vec<_>>(),
                        6e-5,
                        6e-5,
                    );
                }
            }
        }
        assert!(x.grad().is_none());
    }
}

#[test]
fn cpu_generic_attention_preserves_frozen_i8_projection_routes() {
    run(Device::Cpu, quantized_projection_compatibility);
}
#[cfg(feature = "cuda")]
#[test]
fn cuda_generic_attention_preserves_frozen_i8_projection_routes() {
    run(Device::Cuda, quantized_projection_compatibility);
}
