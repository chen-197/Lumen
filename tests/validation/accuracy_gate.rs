//! Independent logical-index/F64 references, included in ordinary cargo test.
use crate::autograd::{
    Device, Tensor, TensorStorageOwned, no_grad, set_strict_device_execution_scoped,
};
use crate::layers::{activation::Softmax, norm::rms_norm::RMSNorm};
use crate::loss::CrossEntropyLoss;
use crate::module::Module;
use crate::ops::{cuda, shape};
use crate::precision::{DType, PrecisionConfig, with_precision_config};
use ndarray::{ArrayD, IxDyn};

const TYPES: [DType; 4] = [DType::F32, DType::F16, DType::BF16, DType::I8];

pub(crate) fn ordered<T: Clone>(shape: &[usize], data: &[T], order: usize) -> ArrayD<T> {
    let a = ArrayD::from_shape_vec(IxDyn(shape), data.to_vec()).unwrap();
    if shape.is_empty() || a.is_empty() {
        return a;
    }
    match order {
        0 => a,
        1 => a
            .reversed_axes()
            .as_standard_layout()
            .to_owned()
            .reversed_axes(),
        _ => {
            let last = shape.len() - 1;
            let mut padded = shape.to_vec();
            padded[last] *= 2;
            let mut a = ArrayD::from_shape_fn(IxDyn(&padded), |i| {
                let index = shape.iter().enumerate().fold(0, |acc, (axis, n)| {
                    acc * n + if axis == last { i[axis] / 2 } else { i[axis] }
                });
                data[index].clone()
            });
            a.slice_axis_inplace(ndarray::Axis(last), ndarray::Slice::new(0, None, 2));
            a
        }
    }
}

pub(crate) fn tensor(
    shape: &[usize],
    data: &[f32],
    dtype: DType,
    order: usize,
    device: Device,
) -> Tensor {
    let t = Tensor::parameter_with_dtype(ArrayD::zeros(IxDyn(shape)), DType::F32);
    match dtype {
        DType::F32 => t.set_array_f32_with_dtype(ordered(shape, data, order), dtype),
        DType::F16 => t.set_array_f16_with_dtype(
            ordered(
                shape,
                &data
                    .iter()
                    .map(|v| half::f16::from_f32(*v))
                    .collect::<Vec<_>>(),
                order,
            ),
            dtype,
        ),
        DType::BF16 => t.set_array_bf16_with_dtype(
            ordered(
                shape,
                &data
                    .iter()
                    .map(|v| half::bf16::from_f32(*v))
                    .collect::<Vec<_>>(),
                order,
            ),
            dtype,
        ),
        DType::I8 => {
            let max = data.iter().fold(0.0f32, |a, b| a.max(b.abs()));
            let scale = if max == 0.0 { 1.0 } else { max / 127.0 };
            let bytes = data
                .iter()
                .map(|v| (v / scale).round().clamp(-127.0, 127.0) as i8)
                .collect::<Vec<_>>();
            t.set_array_i8_with_dtype(ordered(shape, &bytes, order), scale, dtype);
        }
    }
    if device == Device::Cuda {
        t.to_device_inplace(device);
    }
    t
}

pub(crate) fn values(t: &Tensor) -> Vec<f32> {
    if t.is_cuda() && !t.is_empty() {
        return cuda::download_f32(&t.cloned_cuda_f32_buffer().unwrap()).unwrap();
    }
    match t.native_storage_owned() {
        TensorStorageOwned::F32(a) => a.iter().copied().collect(),
        TensorStorageOwned::F16(a) => a.iter().map(|v| v.to_f32()).collect(),
        TensorStorageOwned::BF16(a) => a.iter().map(|v| v.to_f32()).collect(),
        TensorStorageOwned::I8(a, s) => a.iter().map(|v| *v as f32 * s).collect(),
    }
}

pub(crate) fn check(label: &str, got: &[f32], expected: &[f64], atol: f64, rtol: f64) {
    assert_eq!(got.len(), expected.len(), "{label}: length");
    for (i, (&a, &b)) in got.iter().zip(expected).enumerate() {
        let error = (a as f64 - b).abs();
        assert!(
            a.is_finite() && b.is_finite() && error <= atol + rtol * b.abs(),
            "{label}[{i}]: actual={a} expected={b} error={error} limit={}",
            atol + rtol * b.abs()
        );
    }
}

fn trace(value: serde_json::Value) {
    if std::env::var_os("LUMEN_OPERATOR_AUDIT").is_some() {
        println!("OPERATOR_AUDIT={value}");
    }
}

fn backward(y: &Tensor, grad: ArrayD<f32>) {
    let backward =
        y.0.borrow()
            .backward_op
            .clone()
            .expect("gradient graph missing");
    backward(&grad.view());
}

pub(crate) fn run(device: Device, cases: fn(Device)) {
    if device == Device::Cuda {
        assert!(
            cuda::is_available(),
            "accuracy gate requires a working CUDA runtime"
        );
    }
    let _cuda = cuda::set_enabled_scoped(device == Device::Cuda);
    let _strict = set_strict_device_execution_scoped(device == Device::Cuda);
    with_precision_config(
        PrecisionConfig {
            parameter_dtype: DType::F32,
            runtime_dtype: DType::F32,
            allow_parameter_dtype_copies: true,
        },
        || cases(device),
    );
}

fn reshape_cases(device: Device) {
    let data = [0.25, -0.5, 0.75, -1.0, 1.25, -1.5];
    for dtype in TYPES {
        for order in 0..3 {
            for target in [vec![6], vec![3, 2], vec![1, 2, 3]] {
                for training in [false, true] {
                    let x = tensor(&[2, 3], &data, dtype, order, device);
                    let initial = values(&x);
                    let y = if training {
                        shape::reshape(&x, target.clone())
                    } else {
                        no_grad(|| shape::reshape(&x, target.clone()))
                    };
                    check(
                        "reshape logical order",
                        &values(&y),
                        &initial.iter().map(|v| *v as f64).collect::<Vec<_>>(),
                        1e-6,
                        0.0,
                    );
                    let grad: Vec<_> = (0..6).map(|i| i as f32 * 0.25 - 0.5).collect();
                    if training {
                        let dims = target.iter().map(|v| *v as usize).collect::<Vec<_>>();
                        backward(&y, ordered(&dims, &grad, order));
                        check(
                            "reshape gradient",
                            x.grad()
                                .unwrap()
                                .iter()
                                .copied()
                                .collect::<Vec<_>>()
                                .as_slice(),
                            &grad.iter().map(|v| *v as f64).collect::<Vec<_>>(),
                            0.0,
                            0.0,
                        );
                    }
                    trace(
                        serde_json::json!({"case":"reshape","device":format!("{device:?}"),"dtype":format!("{dtype:?}"),
                "order":order,"training":training,"shape":target,"x":initial,"y":values(&y),
                "dx":if training {x.grad().unwrap().iter().copied().collect::<Vec<_>>()} else {Vec::new()}}),
                    );
                }
            }
        }
    }
}

fn ce_cases(device: Device) {
    for ld in TYPES {
        for td in TYPES {
            for order in 0..3 {
                for data in [
                    vec![1000.0, 0.0, -1000.0, 10000.0, 10000.0, 10000.0],
                    vec![10000.0, 9999.0, 9998.0, -3.0, 2.0, 0.5],
                ] {
                    for training in [false, true] {
                        let x = tensor(&[2, 3], &data, ld, order, device);
                        let t = tensor(&[2, 3], &[0.0, 0.0, 1.0, 1.0, 0.0, 0.0], td, order, device);
                        t.0.borrow_mut().requires_grad = false;
                        let xv = values(&x);
                        let tv = values(&t);
                        let mut expected = 0.0f64;
                        let mut dx = vec![0.0f64; 6];
                        for row in 0..2 {
                            let base = row * 3;
                            let max = xv[base..base + 3]
                                .iter()
                                .map(|v| *v as f64)
                                .fold(f64::NEG_INFINITY, f64::max);
                            let sum = xv[base..base + 3]
                                .iter()
                                .map(|v| (*v as f64 - max).exp())
                                .sum::<f64>();
                            for j in 0..3 {
                                let i = base + j;
                                expected += tv[i] as f64 * ((max - xv[i] as f64) + sum.ln()) / 2.0;
                                dx[i] = ((xv[i] as f64 - max).exp() / sum - tv[i] as f64) / 2.0;
                            }
                        }
                        let loss = if training {
                            CrossEntropyLoss::apply(&x, &t)
                        } else {
                            no_grad(|| CrossEntropyLoss::apply(&x, &t))
                        };
                        check(
                            "cross entropy logsumexp",
                            &values(&loss),
                            &[expected],
                            2e-6,
                            2e-6,
                        );
                        let grad = if training {
                            loss.backward();
                            let g = x.grad().unwrap().iter().copied().collect::<Vec<_>>();
                            check("cross entropy gradient", &g, &dx, 3e-6, 3e-6);
                            g
                        } else {
                            Vec::new()
                        };
                        trace(
                            serde_json::json!({"case":"ce","device":format!("{device:?}"),"dtype":format!("{ld:?}/{td:?}"),
                    "training":training,"order":order,"x":xv,"target":tv,"loss":values(&loss),"dx":grad}),
                        );
                    }
                }
            }
        }
    }
}

fn row_cases(device: Device) {
    let data: Vec<_> = (0..12).map(|i| (i as f32 - 5.0) * 0.125).collect();
    let g: Vec<_> = (0..12).map(|i| ((i * 7) % 13) as f32 * 0.1 - 0.5).collect();
    for dtype in TYPES {
        for wd in [DType::F32, DType::F16, DType::BF16] {
            for order in 0..3 {
                let x = tensor(&[2, 2, 3], &data, dtype, order, device);
                let xv = values(&x);
                let weight = tensor(&[3], &[0.5, 1.0, 1.5], wd, order, device);
                let wv = values(&weight);
                let norm = RMSNorm {
                    weight: weight.clone(),
                    eps: 1e-5,
                };
                let y = norm.forward(x.clone());
                let mut ey = vec![0.0f64; 12];
                let mut dx = ey.clone();
                let mut dw = vec![0.0f64; 3];
                for row in 0..4 {
                    let base = row * 3;
                    let inv = 1.0
                        / (xv[base..base + 3]
                            .iter()
                            .map(|v| (*v as f64).powi(2))
                            .sum::<f64>()
                            / 3.0
                            + 1e-5f32 as f64)
                            .sqrt();
                    let dot = (0..3)
                        .map(|j| g[base + j] as f64 * wv[j] as f64 * xv[base + j] as f64)
                        .sum::<f64>();
                    for j in 0..3 {
                        let i = base + j;
                        ey[i] = xv[i] as f64 * inv * wv[j] as f64;
                        dx[i] = g[i] as f64 * wv[j] as f64 * inv
                            - xv[i] as f64 * inv.powi(3) * dot / 3.0;
                        dw[j] += g[i] as f64 * xv[i] as f64 * inv;
                    }
                }
                check("RMSNorm forward", &values(&y), &ey, 3e-6, 3e-5);
                backward(&y, ordered(&[2, 2, 3], &g, order));
                let actual_dx = x.grad().unwrap().iter().copied().collect::<Vec<_>>();
                let actual_dw = weight.grad().unwrap().iter().copied().collect::<Vec<_>>();
                check("RMSNorm dx", &actual_dx, &dx, 3e-6, 3e-5);
                check("RMSNorm dw", &actual_dw, &dw, 3e-6, 3e-5);
                trace(
                    serde_json::json!({"case":"rms","device":format!("{device:?}"),"dtype":format!("{dtype:?}/{wd:?}"),
            "order":order,"x":xv,"weight":wv,"g":g,"y":values(&y),"dx":actual_dx,"dw":actual_dw}),
                );
                let x = tensor(&[2, 2, 3], &data, dtype, order, device);
                let xv = values(&x);
                let y = Softmax::new(2).forward(x.clone());
                let mut ey = vec![0.0f64; 12];
                let mut dx = ey.clone();
                for row in 0..4 {
                    let base = row * 3;
                    let max = xv[base..base + 3]
                        .iter()
                        .map(|v| *v as f64)
                        .fold(f64::NEG_INFINITY, f64::max);
                    let sum = xv[base..base + 3]
                        .iter()
                        .map(|v| (*v as f64 - max).exp())
                        .sum::<f64>();
                    for j in 0..3 {
                        ey[base + j] = (xv[base + j] as f64 - max).exp() / sum;
                    }
                    let dot = (0..3)
                        .map(|j| g[base + j] as f64 * ey[base + j])
                        .sum::<f64>();
                    for j in 0..3 {
                        dx[base + j] = ey[base + j] * (g[base + j] as f64 - dot);
                    }
                }
                check("softmax forward", &values(&y), &ey, 3e-6, 3e-5);
                backward(&y, ordered(&[2, 2, 3], &g, order));
                let actual_dx = x.grad().unwrap().iter().copied().collect::<Vec<_>>();
                check("softmax dx", &actual_dx, &dx, 3e-6, 3e-5);
                trace(
                    serde_json::json!({"case":"softmax","device":format!("{device:?}"),"dtype":format!("{dtype:?}"),
            "order":order,"x":xv,"g":g,"y":values(&y),"dx":actual_dx}),
                );
            }
        }
    }
}

fn arithmetic_cases(device: Device) {
    for ad in TYPES {
        for bd in TYPES {
            for order in 0..3 {
                for op in ["add", "sub", "mul"] {
                    let a = tensor(
                        &[2, 1, 3],
                        &[0.25, -0.5, 0.75, -1.0, 1.25, -1.5],
                        ad,
                        order,
                        device,
                    );
                    let b = tensor(&[1, 4, 1], &[0.125, -0.25, 0.5, -0.75], bd, order, device);
                    let av = values(&a);
                    let bv = values(&b);
                    let mut expected = vec![0.0f64; 24];
                    let mut da = vec![0.0f64; 6];
                    let mut db = vec![0.0f64; 4];
                    let g: Vec<_> = (0..24)
                        .map(|i| ((i * 11) % 29) as f32 * 0.0625 - 0.5)
                        .collect();
                    for batch in 0..2 {
                        for row in 0..4 {
                            for col in 0..3 {
                                let ai = batch * 3 + col;
                                let i = (batch * 4 + row) * 3 + col;
                                let x = av[ai] as f64;
                                let z = bv[row] as f64;
                                let (value, ga, gb) = match op {
                                    "add" => (x + z, 1.0, 1.0),
                                    "sub" => (x - z, 1.0, -1.0),
                                    _ => (x * z, z, x),
                                };
                                expected[i] = value;
                                da[ai] += g[i] as f64 * ga;
                                db[row] += g[i] as f64 * gb;
                            }
                        }
                    }
                    let y = match op {
                        "add" => &a + &b,
                        "sub" => &a - &b,
                        _ => &a * &b,
                    };
                    assert_eq!(y.shape_vec(), [2, 4, 3]);
                    check(
                        "broadcast arithmetic forward",
                        &values(&y),
                        &expected,
                        3e-6,
                        3e-5,
                    );
                    backward(&y, ordered(&[2, 4, 3], &g, order));
                    let actual_da = a.grad().unwrap().iter().copied().collect::<Vec<_>>();
                    let actual_db = b.grad().unwrap().iter().copied().collect::<Vec<_>>();
                    check("broadcast arithmetic da", &actual_da, &da, 3e-6, 3e-5);
                    check("broadcast arithmetic db", &actual_db, &db, 3e-6, 3e-5);
                    trace(
                        serde_json::json!({"case":"binary","device":format!("{device:?}"),"op":op,"dtype":format!("{ad:?}/{bd:?}"),
            "order":order,"a":av,"b":bv,"g":g,"y":values(&y),"da":actual_da,"db":actual_db}),
                    );
                }
            }
        }
    }
}

fn inference_row_cases(device: Device) {
    let data: Vec<_> = (0..12).map(|i| (i as f32 - 5.0) * 0.125).collect();
    for dtype in TYPES {
        for wd in [DType::F32, DType::F16, DType::BF16] {
            for order in 0..3 {
                let x = tensor(&[2, 2, 3], &data, dtype, order, device);
                let xv = values(&x);
                let weight = tensor(&[3], &[0.5, 1.0, 1.5], wd, order, device);
                let wv = values(&weight);
                let norm = RMSNorm { weight, eps: 1e-5 };
                let y = no_grad(|| norm.forward(x));
                assert_eq!(y.dtype(), dtype);
                let mut expected = [0.0f64; 12];
                for row in 0..4 {
                    let base = row * 3;
                    let inv = 1.0
                        / (xv[base..base + 3]
                            .iter()
                            .map(|v| (*v as f64).powi(2))
                            .sum::<f64>()
                            / 3.0
                            + 1e-5f32 as f64)
                            .sqrt();
                    for j in 0..3 {
                        expected[base + j] = xv[base + j] as f64 * inv * wv[j] as f64;
                    }
                }
                let scale = if dtype == DType::I8 {
                    let max = expected
                        .iter()
                        .map(|v| (*v as f32).abs())
                        .fold(0.0f32, f32::max);
                    let scale = max / 127.0;
                    check(
                        "RMSNorm inference quantization scale",
                        &[y.quantization_scale().unwrap()],
                        &[scale as f64],
                        1e-8,
                        3e-5,
                    );
                    Some(scale)
                } else {
                    None
                };
                let rounded: Vec<_> = expected
                    .iter()
                    .map(|v| {
                        let v = *v as f32;
                        (match dtype {
                            DType::F32 => v,
                            DType::F16 => half::f16::from_f32(v).to_f32(),
                            DType::BF16 => half::bf16::from_f32(v).to_f32(),
                            DType::I8 => {
                                (v / scale.unwrap()).round().clamp(-127.0, 127.0) * scale.unwrap()
                            }
                        }) as f64
                    })
                    .collect();
                check(
                    "RMSNorm inference rounded output",
                    &values(&y),
                    &rounded,
                    3e-6,
                    3e-5,
                );
                trace(
                    serde_json::json!({"case":"rms_inference","device":format!("{device:?}"),
                    "dtype":format!("{dtype:?}"),"weight_dtype":format!("{wd:?}"),"order":order,
                    "x":xv,"weight":wv,"y":values(&y),"scale":scale}),
                );
            }
        }
    }
}

#[test]
fn accuracy_gate_rejects_nonfinite_comparisons() {
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(
            std::panic::catch_unwind(|| check("nonfinite guard", &[value], &[0.0], 1e-6, 1e-6))
                .is_err()
        );
    }
}

#[test]
fn accuracy_gate_empty_reshape() {
    for dtype in TYPES {
        let x = tensor(&[2, 0, 3], &[], dtype, 0, Device::Cpu);
        let y = no_grad(|| shape::reshape(&x, vec![0, 6]));
        assert_eq!(y.shape_vec(), [0, 6]);
        assert_eq!(y.dtype(), dtype);
        assert!(values(&y).is_empty());
    }
}

#[test]
fn accuracy_gate_cpu_inference_rows() {
    run(Device::Cpu, inference_row_cases)
}
#[cfg(feature = "cuda")]
#[test]
fn accuracy_gate_cuda_inference_rows() {
    run(Device::Cuda, inference_row_cases)
}

#[cfg(feature = "cuda")]
#[test]
fn accuracy_gate_cuda_probability_loss_and_invalid_metadata() {
    assert!(cuda::is_available());
    let _enabled = cuda::set_enabled_scoped(true);
    for logits_dtype in TYPES {
        for target_dtype in TYPES {
            let x = tensor(
                &[1, 3],
                &[1000.0, 0.0, -1000.0],
                logits_dtype,
                0,
                Device::Cuda,
            );
            let t = tensor(&[1, 3], &[0.0, 0.0, 1.0], target_dtype, 0, Device::Cuda);
            let storage = |tensor: &Tensor| {
                tensor
                    .cloned_cuda_native_lowp_buffer()
                    .unwrap_or_else(|| (DType::F32, tensor.cloned_cuda_f32_buffer().unwrap(), None))
            };
            let (xd, xb, xs) = storage(&x);
            let (td, tb, ts) = storage(&t);
            // Direct device API: no high-level CPU fallback can mask a failure.
            let (probabilities, _, loss) =
                cuda::cross_entropy_logits_typed(&xb, xd, xs, &tb, td, ts, 1).unwrap();
            assert!(
                (loss - 2000.0).abs() < 0.001,
                "native {logits_dtype:?}/{target_dtype:?} loss={loss}"
            );
            assert_eq!(cuda::download_f32(&probabilities).unwrap(), [1.0, 0.0, 0.0]);
        }
    }
    let p = cuda::upload_f32(&[1.0, 0.0]).unwrap();
    let t = cuda::upload_f32(&[0.0, 1.0]).unwrap();
    let (_, loss) = cuda::cross_entropy_loss_f32(&p, &t, 1).unwrap();
    assert_eq!(loss[0], f32::INFINITY);
    let (_, loss) = cuda::cross_entropy_loss_typed_target(&p, &t, DType::F32, None, 1).unwrap();
    assert_eq!(loss[0], f32::INFINITY);
    assert!(
        cuda::cross_entropy_logits_typed(&p, DType::F32, None, &t, DType::F32, None, 3).is_err()
    );
    let i8buf = cuda::upload_i8_storage(&[1, -1]).unwrap();
    assert!(
        cuda::cross_entropy_logits_typed(
            &i8buf,
            DType::I8,
            Some(f32::NAN),
            &t,
            DType::F32,
            None,
            1
        )
        .is_err()
    );
}

#[test]
fn accuracy_gate_cpu_reshape() {
    run(Device::Cpu, reshape_cases)
}
#[test]
fn accuracy_gate_cpu_cross_entropy() {
    run(Device::Cpu, ce_cases)
}
#[test]
fn accuracy_gate_cpu_rows() {
    run(Device::Cpu, row_cases)
}
#[test]
fn accuracy_gate_cpu_broadcast() {
    run(Device::Cpu, arithmetic_cases)
}
#[cfg(feature = "cuda")]
#[test]
fn accuracy_gate_cuda_reshape() {
    run(Device::Cuda, reshape_cases)
}
#[cfg(feature = "cuda")]
#[test]
fn accuracy_gate_cuda_cross_entropy() {
    run(Device::Cuda, ce_cases)
}
#[cfg(feature = "cuda")]
#[test]
fn accuracy_gate_cuda_rows() {
    run(Device::Cuda, row_cases)
}
#[cfg(feature = "cuda")]
#[test]
fn accuracy_gate_cuda_broadcast() {
    run(Device::Cuda, arithmetic_cases)
}

fn scaled_multiply_cases(device: Device) {
    for dtype in TYPES {
        let magnitudes: &[f32] = if dtype == DType::F16 {
            &[1e-4, 1.0, 1e4]
        } else {
            &[1e-12, 1e-8, 1.0, 1e8, 1e12]
        };
        for &magnitude in magnitudes {
            for order in [0, 1] {
                let av = [0.25, -0.5, 0.75, -1.0, 1.25, -1.5].map(|v| v * magnitude);
                let bv = [0.125, 0.25, -0.375, -0.5, 0.625, 0.75].map(|v| v * magnitude);
                let a = tensor(&[2, 3], &av, dtype, order, device);
                let b = tensor(&[2, 3], &bv, dtype, order, device);
                let av = values(&a);
                let bv = values(&b);
                let expected: Vec<_> = av
                    .iter()
                    .zip(&bv)
                    .map(|(a, b)| *a as f64 * *b as f64)
                    .collect();
                let y = &a * &b;
                assert_eq!(
                    y.dtype(),
                    DType::F32,
                    "training arithmetic must retain F32 output range"
                );
                let output_atol = (magnitude as f64).powi(2) * 1e-7;
                let gradient_atol = magnitude as f64 * 1e-7;
                check(
                    "scaled multiply output",
                    &values(&y),
                    &expected,
                    output_atol,
                    1e-5,
                );
                backward(&y, ordered(&[2, 3], &[1.0; 6], order));
                let da = a.grad().unwrap().iter().copied().collect::<Vec<_>>();
                let db = b.grad().unwrap().iter().copied().collect::<Vec<_>>();
                check(
                    "scaled multiply da",
                    &da,
                    &bv.iter().map(|v| *v as f64).collect::<Vec<_>>(),
                    gradient_atol,
                    1e-5,
                );
                check(
                    "scaled multiply db",
                    &db,
                    &av.iter().map(|v| *v as f64).collect::<Vec<_>>(),
                    gradient_atol,
                    1e-5,
                );
                trace(
                    serde_json::json!({"case":"scaled_mul","device":format!("{device:?}"),
                    "dtype":format!("{dtype:?}"),"order":order,"magnitude":magnitude,
                    "a":av,"b":bv,"y":values(&y),"da":da,"db":db}),
                );
            }
        }
    }
}

#[test]
fn accuracy_gate_cpu_scaled_multiply() {
    run(Device::Cpu, scaled_multiply_cases)
}
#[cfg(feature = "cuda")]
#[test]
fn accuracy_gate_cuda_scaled_multiply() {
    run(Device::Cuda, scaled_multiply_cases)
}
