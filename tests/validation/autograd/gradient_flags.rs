use crate::accuracy_gate::{ordered, run, tensor, values};
use crate::autograd::{Device, Tensor};
use crate::module::Module;
#[cfg(feature = "cuda")]
use crate::ops::fused;
use crate::ops::{convolution, matmul, shape};
use crate::precision::DType;
use ndarray::{ArrayD, IxDyn};

const TYPES: [DType; 4] = [DType::F32, DType::F16, DType::BF16, DType::I8];

fn input(
    shape: &[usize],
    dtype: DType,
    layout: usize,
    seed: usize,
    requested: bool,
    device: Device,
) -> Tensor {
    let len = shape.iter().product();
    let data = (0..len)
        .map(|i| (((i * 5 + seed) % 13) as f32 - 6.0) * 0.125)
        .collect::<Vec<_>>();
    let x = tensor(shape, &data, dtype, layout, Device::Cpu);
    x.0.borrow_mut().requires_grad = requested;
    if device == Device::Cuda {
        x.to_device_inplace(device);
    }
    x
}

fn case(device: Device, op: &str, xd: DType, wd: DType, layout: usize, flags: usize) {
    let (xs, ws): (&[usize], &[usize]) = match op {
        "matmul" => (&[2, 3], &[2, 3]),
        "batch-matmul" => (&[1, 2, 2, 3], &[1, 2, 3, 2]),
        "conv" => (&[1, 1, 3, 3], &[2, 1, 2, 2]),
        "rms" => (&[2, 3], &[3]),
        _ if op.ends_with("row") => (&[2, 3], &[3]),
        _ if op.ends_with("column") => (&[2, 3], &[2, 1]),
        _ if op.ends_with("scalar") => (&[2, 3], &[1]),
        _ => (&[2, 3], &[2, 3]),
    };
    let x = input(xs, xd, layout, 2, flags & 1 != 0, device);
    let w = input(ws, wd, layout, 7, flags & 2 != 0, device);
    let mut inputs = vec![x.clone(), w.clone()];
    let y = match op {
        "matmul" => matmul::matmul(&x, &w),
        "batch-matmul" => matmul::batch_matmul(&x, &w),
        "cat" => shape::cat(&[x.clone(), w.clone()], 1),
        "rms" => crate::layers::norm::rms_norm::RMSNorm {
            weight: w,
            eps: 1e-5,
        }
        .forward(x),
        "conv" => {
            let bias = input(&[2], wd, layout, 4, flags & 4 != 0, device);
            inputs.push(bias.clone());
            convolution::conv2d(&x, &w, Some(&bias), (1, 1), (0, 0))
        }
        _ if op.starts_with("add") => &x + &w,
        _ if op.starts_with("sub") => &x - &w,
        _ => &x * &w,
    };
    verify(device, op, xd, wd, layout, flags, inputs, y);
}

fn verify(
    device: Device,
    op: &str,
    xd: DType,
    wd: DType,
    layout: usize,
    flags: usize,
    inputs: Vec<Tensor>,
    y: Tensor,
) {
    assert_eq!(y.requires_grad(), flags != 0, "{op}: output gradient flag");
    if flags == 0 {
        let inner = y.0.borrow();
        assert!(
            inner.backward_op.is_none() && inner.parents.is_empty(),
            "{op}: constants must not build a graph"
        );
        assert!(inputs.iter().all(|x| !x.has_pending_grad()));
        return;
    }
    assert_eq!(
        y.dtype(),
        DType::F32,
        "{op}: training output must retain F32 compute precision"
    );
    let data = inputs.iter().map(|x| serde_json::json!({
        "shape": x.shape_vec(), "values": values(x), "requires_grad": x.requires_grad(), "dtype": format!("{:?}",x.dtype())
    })).collect::<Vec<_>>();
    let forward = values(&y);
    let upstream = (0..y.len())
        .map(|i| ((i % 7) as f32 - 3.0) * 0.25)
        .collect::<Vec<_>>();
    for pass in 1..=2 {
        y.backward_with_grad(ordered(&y.shape_vec(), &upstream, layout));
        let gradients = inputs
            .iter()
            .enumerate()
            .map(|(i, x)| {
                if !x.requires_grad() {
                    assert!(
                        !x.has_pending_grad(),
                        "{op}: frozen input {i} acquired a gradient, flags={flags}"
                    );
                    None
                } else {
                    if device == Device::Cuda {
                        assert!(
                            x.cloned_cuda_f32_grad().is_some() && !x.has_host_grad(),
                            "{op}: requested gradient must stay CUDA resident"
                        );
                    }
                    let grad = x.grad().expect("requested gradient missing");
                    assert!(
                        grad.iter().all(|v| v.is_finite()),
                        "{op}: non-finite gradient"
                    );
                    Some(grad.iter().copied().collect::<Vec<_>>())
                }
            })
            .collect::<Vec<_>>();
        if op == "conv" {
            for (i, x) in inputs.iter().enumerate() {
                assert_eq!(
                    format!("{:?}", x.dtype()),
                    data[i]["dtype"].as_str().unwrap()
                );
                if x.dtype() != DType::F32 {
                    assert!(
                        !x.0.borrow().has_f32_data,
                        "conv2d materialized a persistent F32 copy while disabled"
                    );
                }
            }
        }
        if std::env::var_os("LUMEN_OPERATOR_AUDIT").is_some() {
            println!(
                "GRADIENT_FLAGS_AUDIT={}",
                serde_json::json!({
                    "device":format!("{device:?}"), "op":op, "xd":format!("{xd:?}"), "wd":format!("{wd:?}"),
                    "layout":layout, "flags":flags, "pass":pass, "inputs":data,
                    "shape":y.shape_vec(), "y":forward, "upstream":upstream, "gradients":gradients,
                    "dtype_copies":crate::precision::allow_parameter_dtype_copies()
                })
            );
        }
    }
}

fn family(device: Device, name: &str) {
    let ops = match name {
        "binary" => vec![
            "add-same",
            "add-row",
            "add-column",
            "add-scalar",
            "sub-same",
            "sub-row",
            "sub-column",
            "sub-scalar",
            "mul-same",
            "mul-row",
            "mul-column",
            "mul-scalar",
        ],
        _ => vec![name],
    };
    for op in ops {
        for xd in TYPES {
            for wd in TYPES {
                for layout in 0..3 {
                    for flags in 0..if op == "conv" { 8 } else { 4 } {
                        if op == "conv" {
                            crate::precision::with_precision_config(
                                crate::precision::PrecisionConfig {
                                    parameter_dtype: DType::F32,
                                    runtime_dtype: DType::F32,
                                    allow_parameter_dtype_copies: false,
                                },
                                || case(device, op, xd, wd, layout, flags),
                            );
                        } else {
                            case(device, op, xd, wd, layout, flags);
                        }
                    }
                }
            }
        }
    }
}

macro_rules! flag_tests {
    ($cpu:ident, $cuda:ident, $family:literal) => {
        #[test]
        fn $cpu() {
            run(Device::Cpu, |d| family(d, $family));
        }
        #[cfg(feature = "cuda")]
        #[test]
        fn $cuda() {
            run(Device::Cuda, |d| family(d, $family));
        }
    };
}
flag_tests!(
    cpu_binary_gradient_flags,
    cuda_binary_gradient_flags,
    "binary"
);
flag_tests!(
    cpu_matmul_gradient_flags,
    cuda_matmul_gradient_flags,
    "matmul"
);
flag_tests!(
    cpu_batch_matmul_gradient_flags,
    cuda_batch_matmul_gradient_flags,
    "batch-matmul"
);
flag_tests!(cpu_cat_gradient_flags, cuda_cat_gradient_flags, "cat");
flag_tests!(cpu_rms_gradient_flags, cuda_rms_gradient_flags, "rms");
flag_tests!(cpu_conv_gradient_flags, cuda_conv_gradient_flags, "conv");

#[cfg(feature = "cuda")]
#[test]
fn cuda_qkv_projection_gradient_metadata() {
    run(Device::Cuda, |device| {
        let x = input(&[1, 2, 4], DType::F32, 0, 2, false, device);
        let q = input(&[4, 4], DType::F32, 0, 7, true, device);
        let frozen = input(&[4, 4], DType::F32, 0, 4, false, device);
        let (q, k, v) =
            fused::fused_qkv_train_tensors(&x, &q, &frozen, &frozen, None, None, None, 2, 2)
                .unwrap();
        assert!(q.requires_grad());
        assert!(
            !k.requires_grad() && !v.requires_grad(),
            "frozen QKV projections incorrectly require gradients"
        );
    });
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_qkv_gradient_flags() {
    run(Device::Cuda, |device| {
        let pairs = [
            (DType::F32, DType::F32),
            (DType::F16, DType::F16),
            (DType::BF16, DType::BF16),
            (DType::I8, DType::I8),
            (DType::BF16, DType::I8),
        ];
        for (xd, wd) in pairs {
            for layout in 0..3 {
                for flags in 0..128 {
                    let x = input(&[1, 2, 4], xd, layout, 2, flags & 1 != 0, device);
                    let mut inputs = vec![x.clone()];
                    for i in 1..7 {
                        inputs.push(input(
                            if i < 4 { &[4, 4] } else { &[4] },
                            wd,
                            layout,
                            i + 3,
                            flags & (1 << i) != 0,
                            device,
                        ));
                    }
                    let outputs = fused::fused_qkv_train_tensors(
                        &x,
                        &inputs[1],
                        &inputs[2],
                        &inputs[3],
                        Some(&inputs[4]),
                        Some(&inputs[5]),
                        Some(&inputs[6]),
                        2,
                        2,
                    );
                    if flags == 0 {
                        assert!(outputs.is_none());
                        continue;
                    }
                    let (q, k, v) = outputs.unwrap();
                    for (i, out) in [&q, &k, &v].iter().enumerate() {
                        let requested = flags & (1 | (1 << (i + 1)) | (1 << (i + 4))) != 0;
                        assert_eq!(
                            out.requires_grad(),
                            requested,
                            "per-projection gradient flag, head {i}, flags={flags}"
                        );
                    }
                    verify(
                        device,
                        "qkv",
                        xd,
                        wd,
                        layout,
                        flags,
                        inputs,
                        shape::cat(&[q, k, v], 1),
                    );
                }
            }
        }
    });
}

fn explicit_frozen_gradient(device: Device) {
    let x = input(&[2, 3], DType::F32, 0, 2, true, device);
    let w = input(&[2, 3], DType::F32, 0, 7, false, device);
    w.add_grad(ArrayD::from_elem(IxDyn(&[2, 3]), 7.0));
    (&x * &w).sum().backward();
    assert!(
        w.grad().unwrap().iter().all(|v| *v == 7.0),
        "backward changed an explicit frozen gradient"
    );
}

#[test]
fn cpu_explicit_frozen_gradient_is_preserved() {
    run(Device::Cpu, explicit_frozen_gradient);
}
#[cfg(feature = "cuda")]
#[test]
fn cuda_explicit_frozen_gradient_is_preserved() {
    run(Device::Cuda, explicit_frozen_gradient);
}

fn frozen_training(device: Device) {
    use crate::loss::MSELoss;
    use crate::optim::{Adam, Optimizer, SGD};
    for dtype in TYPES {
        for layout in 0..3 {
            for optimizer in ["sgd", "adam"] {
                let x = input(&[4, 2], DType::F32, layout, 2, false, device);
                let weight = input(&[2, 2], dtype, layout, 5, true, device);
                let frozen = input(&[2, 2], dtype, layout, 7, false, device);
                let xv = values(&x);
                let initial = values(&weight);
                let frozen_initial = values(&frozen);
                let ideal = [0.35f64, -0.2, 0.15, 0.4];
                let mut target = Vec::new();
                for row in xv.chunks(2) {
                    for o in 0..2 {
                        let mut total = 0.0;
                        for j in 0..2 {
                            let hidden =
                                row[0] as f64 * ideal[j * 2] + row[1] as f64 * ideal[j * 2 + 1];
                            total += hidden * frozen_initial[o * 2 + j] as f64;
                        }
                        target.push(total as f32);
                    }
                }
                let expected = Tensor::from_data_with_grad_flag(
                    ArrayD::from_shape_vec(IxDyn(&[4, 2]), target.clone()).unwrap(),
                    false,
                );
                if device == Device::Cuda {
                    expected.to_device_inplace(device);
                }
                let params = vec![weight.clone(), frozen.clone()];
                let mut opt: Box<dyn Optimizer> = if optimizer == "sgd" {
                    let opt = SGD::new_with_dtype(params, 0.1, DType::F32);
                    Box::new(if dtype == DType::I8 {
                        opt.with_dynamic_i8_master_weights()
                    } else {
                        opt.with_fp32_master_weights()
                    })
                } else {
                    let opt = Adam::new_with_dtype(params, 0.02, DType::F32);
                    Box::new(if dtype == DType::I8 {
                        opt.with_dynamic_i8_master_weights()
                    } else {
                        opt.with_fp32_master_weights()
                    })
                };
                let mut losses = Vec::new();
                for step in 0..=60 {
                    opt.zero_grad();
                    let hidden = matmul::matmul(&x, &weight);
                    let prediction = matmul::matmul(&hidden, &frozen);
                    let loss = MSELoss::apply(&prediction, &expected);
                    losses.push(values(&loss)[0]);
                    if step < 60 {
                        loss.backward();
                        assert!(
                            !frozen.has_pending_grad()
                                && !x.has_pending_grad()
                                && !expected.has_pending_grad()
                        );
                        opt.step();
                        assert_eq!(
                            values(&frozen),
                            frozen_initial,
                            "optimizer changed a frozen parameter"
                        );
                    }
                }
                assert!(losses.iter().all(|v| v.is_finite()));
                assert!(
                    losses[60] < losses[0],
                    "frozen training did not descend: {losses:?}"
                );
                assert!(losses[46..].iter().sum::<f32>() < losses[..15].iter().sum::<f32>());
                assert!(
                    losses
                        .iter()
                        .enumerate()
                        .map(|(i, v)| (i as f64 - 30.0) * (*v as f64))
                        .sum::<f64>()
                        < 0.0
                );
                if std::env::var_os("LUMEN_OPERATOR_AUDIT").is_some() {
                    println!(
                        "FROZEN_TRAINING_AUDIT={}",
                        serde_json::json!({
                            "device":format!("{device:?}"),"dtype":format!("{dtype:?}"),"layout":layout,"optimizer":optimizer,
                            "x":xv,"initial":initial,"frozen_initial":frozen_initial,"frozen_final":values(&frozen),
                            "target":target,"losses":losses,"final":values(&weight),"fp32_master":true,"dynamic_i8":dtype==DType::I8
                        })
                    );
                }
            }
        }
    }
}

#[test]
fn cpu_frozen_training_descent() {
    run(Device::Cpu, frozen_training);
}
#[cfg(feature = "cuda")]
#[test]
fn cuda_frozen_training_descent() {
    run(Device::Cuda, frozen_training);
}
