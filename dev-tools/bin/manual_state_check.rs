//! Public-API manual checks: GPU snapshots, checkpoint files and continued training.
use lumen::autograd::{Device, StoragePreference, Tensor, TensorRawData};
use lumen::loss::MSELoss;
use lumen::module::{CheckpointTensor, ModelCheckpoint, Module};
use lumen::ops::matmul::matmul;
use lumen::optim::{Adam, Optimizer, SGD};
use lumen::precision::{DType, PrecisionConfig, with_precision_config};
use ndarray::{ArrayD, IxDyn};
use serde_json::{Value, json};
use std::path::{Path, PathBuf};

fn array(shape: &[usize], values: &[f32], layout: usize) -> ArrayD<f32> {
    let data = ArrayD::from_shape_vec(IxDyn(shape), values.to_vec()).unwrap();
    if layout == 0 {
        return data;
    }
    if layout == 1 {
        return data
            .reversed_axes()
            .as_standard_layout()
            .to_owned()
            .reversed_axes();
    }
    let mut padded = shape.to_vec();
    let last = shape.len() - 1;
    padded[last] *= 2;
    let mut data = ArrayD::from_shape_fn(IxDyn(&padded), |index| {
        let offset = shape.iter().enumerate().fold(0, |n, (axis, dim)| {
            n * dim
                + if axis == last {
                    index[axis] / 2
                } else {
                    index[axis]
                }
        });
        values[offset]
    });
    data.slice_axis_inplace(ndarray::Axis(last), ndarray::Slice::new(0, None, 2));
    data
}

fn parameter(
    shape: &[usize],
    values: &[f32],
    dtype: DType,
    layout: usize,
    device: Device,
) -> Tensor {
    let data = array(shape, values, layout);
    let t = Tensor::parameter_with_dtype(data.clone(), dtype);
    match dtype {
        DType::F32 => t.set_array_f32_with_dtype(data, dtype),
        DType::F16 => t.set_array_f16_with_dtype(data.mapv(half::f16::from_f32), dtype),
        DType::BF16 => t.set_array_bf16_with_dtype(data.mapv(half::bf16::from_f32), dtype),
        DType::I8 => {}
    }
    t.to_device_inplace(device);
    t
}

fn values(t: &Tensor) -> Vec<f32> {
    match t.export_raw().2 {
        TensorRawData::F32(v) => v,
        TensorRawData::F16(v) => v
            .into_iter()
            .map(|x| half::f16::from_bits(x).to_f32())
            .collect(),
        TensorRawData::BF16(v) => v
            .into_iter()
            .map(|x| half::bf16::from_bits(x).to_f32())
            .collect(),
        TensorRawData::I8 { values, scale } => {
            values.into_iter().map(|x| x as f32 * scale).collect()
        }
    }
}

struct Pair {
    weight: Tensor,
    bias: Tensor,
}
impl Module for Pair {
    fn parameters(&self) -> Vec<Tensor> {
        vec![self.weight.clone(), self.bias.clone()]
    }
    fn forward(&self, input: Tensor) -> Tensor {
        &matmul(&input, &self.weight) + &self.bias
    }
}

fn record(rows: &mut Vec<Value>, case: &Value, phase: &str, t: &Tensor) {
    let (shape, dtype, raw) = t.export_raw();
    rows.push(
        json!({"case":case,"phase":phase,"shape":shape,"dtype":format!("{dtype:?}"),
        "device":format!("{:?}",t.device()),"values":values(t),"raw":raw,
        "requires_grad":t.requires_grad(),"host_f32":t.dev_has_host_f32_data()}),
    );
}

fn snapshots(device: Device, root: &Path, rows: &mut Vec<Value>, files: &mut Vec<Value>) {
    for dtype in [DType::F32, DType::F16, DType::BF16, DType::I8] {
        for layout in 0..3 {
            for copies in [false, true] {
                with_precision_config(
                    PrecisionConfig {
                        parameter_dtype: dtype,
                        runtime_dtype: DType::F32,
                        allow_parameter_dtype_copies: copies,
                    },
                    || {
                        let case =
                            json!({"dtype":format!("{dtype:?}"),"layout":layout,"copies":copies});
                        let x = parameter(
                            &[2, 3],
                            &[1., -2., 0.5, 3., 0.25, -0.75],
                            dtype,
                            layout,
                            device,
                        );
                        record(rows, &case, "initial", &x);
                        let _ =
                            x.with_storage_view_preferring(StoragePreference::F32Compute, |view| {
                                match view {
                                    lumen::autograd::TensorStorageView::F32(v) => v.sum(),
                                    _ => unreachable!(),
                                }
                            });
                        assert_eq!(x.dtype(), dtype);
                        x.add_grad(array(&[2, 3], &[0.5, -1., 2., 0.25, -0.75, 1.5], layout));
                        let mut opt = SGD::new_with_dtype(vec![x.clone()], 0.1, DType::F32);
                        opt.step();
                        record(rows, &case, "updated", &x);
                        let detached = x.detach();
                        let cpu_copy = x.to_cpu();
                        let same_copy = x.to_device(device);
                        assert!(!detached.requires_grad());
                        for (phase, t) in [
                            ("detach", &detached),
                            ("cpu_copy", &cpu_copy),
                            ("same_device_copy", &same_copy),
                        ] {
                            assert_eq!(values(t), values(&x));
                            assert_eq!(t.dtype(), dtype);
                            record(rows, &case, phase, t);
                        }
                        let pair = Pair {
                            weight: x.clone(),
                            bias: parameter(&[2], &[0.125, -0.25], dtype, 0, device),
                        };
                        let tag = format!("{dtype:?}-{layout}-{copies}");
                        let good = root.join(format!("{tag}.bin"));
                        pair.save(good.to_str().unwrap()).unwrap();
                        let restored = Pair {
                            weight: parameter(&[2, 3], &[9.; 6], DType::F32, 0, device),
                            bias: parameter(&[2], &[9.; 2], DType::F32, 0, device),
                        };
                        for t in restored.parameters() {
                            t.add_grad(ArrayD::ones(IxDyn(&t.shape_vec())));
                        }
                        restored.load(good.to_str().unwrap()).unwrap();
                        assert_eq!(values(&restored.weight), values(&x));
                        for t in restored.parameters() {
                            assert!(t.grad().is_none() && !t.dev_has_cuda_f32_grad());
                        }
                        record(rows, &case, "checkpoint_loaded", &restored.weight);
                        let bad = root.join(format!("{tag}-bad.bin"));
                        let checkpoint = ModelCheckpoint {
                            params: vec![
                                CheckpointTensor {
                                    shape: vec![2, 3],
                                    dtype: DType::F32,
                                    raw: TensorRawData::F32(vec![99.; 6]),
                                },
                                CheckpointTensor {
                                    shape: vec![2],
                                    dtype: DType::F16,
                                    raw: TensorRawData::F32(vec![99.; 2]),
                                },
                            ],
                        };
                        std::fs::write(&bad, bincode::serialize(&checkpoint).unwrap()).unwrap();
                        assert!(restored.load(bad.to_str().unwrap()).is_err());
                        assert_eq!(values(&restored.weight), values(&x));
                        record(rows, &case, "bad_checkpoint_rejected", &restored.weight);
                        let truncated = root.join(format!("{tag}-truncated.bin"));
                        let bytes = std::fs::read(&good).unwrap();
                        std::fs::write(&truncated, &bytes[..bytes.len() / 2]).unwrap();
                        assert!(restored.load(truncated.to_str().unwrap()).is_err());
                        assert_eq!(values(&restored.weight), values(&x));
                        files.push(json!({"case":case,"valid":good,"malformed":bad,"truncated":truncated,
                        "malformed_rejected":true,"truncated_rejected":true,"old_gradient_cleared":true}));
                        x.zero_grad();
                        x.add_grad(ArrayD::ones(IxDyn(&[2, 3])));
                        opt.step();
                        record(rows, &case, "updated_again", &x);
                        for (phase, t) in [
                            ("detach_independent", &detached),
                            ("cpu_copy_independent", &cpu_copy),
                            ("same_device_copy_independent", &same_copy),
                        ] {
                            assert_eq!(values(t), values(&restored.weight));
                            record(rows, &case, phase, t);
                        }
                        x.add_grad(ArrayD::ones(IxDyn(&[2, 3])));
                        x.to_cpu_inplace();
                        assert_eq!(x.dtype(), dtype);
                        assert!(!x.dev_has_cuda_f32_grad());
                        record(rows, &case, "inplace_cpu", &x);
                        assert_eq!(
                            x.grad().unwrap().iter().copied().collect::<Vec<_>>(),
                            vec![if device == Device::Cpu { 2. } else { 1. }; 6]
                        );
                        let before = values(&x);
                        let old_grad = x.grad().unwrap();
                        for scale in [0., -1., f32::NAN, f32::INFINITY] {
                            assert!(
                                x.import_raw(
                                    vec![2, 3],
                                    DType::I8,
                                    TensorRawData::I8 {
                                        values: vec![1; 6],
                                        scale
                                    }
                                )
                                .is_err()
                            );
                            assert_eq!(values(&x), before);
                            assert_eq!(x.grad().unwrap(), old_grad);
                        }
                        record(rows, &case, "invalid_import_preserved", &x);
                        rows.last_mut().unwrap()["invalid_scale_rejections"] = json!(4);
                        x.with_storage_view_preferring(StoragePreference::F32Compute, |_| {});
                        let scale = x.quantization_scale();
                        x.with_native_storage_view_mut(|view| match view {
                            lumen::autograd::TensorStorageViewMut::F32(mut data) => {
                                data.mapv_inplace(|v| -v)
                            }
                            lumen::autograd::TensorStorageViewMut::F16(mut data) => {
                                data.mapv_inplace(|v| -v)
                            }
                            lumen::autograd::TensorStorageViewMut::BF16(mut data) => {
                                data.mapv_inplace(|v| -v)
                            }
                            lumen::autograd::TensorStorageViewMut::I8(mut data, _) => {
                                data.mapv_inplace(|v| -v)
                            }
                        });
                        assert_eq!(x.dtype(), dtype);
                        assert_eq!(x.quantization_scale(), scale);
                        record(rows, &case, "native_view_modified", &x);
                        let compute =
                            x.with_storage_view_preferring(StoragePreference::F32Compute, |view| {
                                match view {
                                    lumen::autograd::TensorStorageView::F32(data) => {
                                        data.iter().copied().collect::<Vec<_>>()
                                    }
                                    _ => unreachable!(),
                                }
                            });
                        assert_eq!(compute, before.iter().map(|v| -*v).collect::<Vec<_>>());
                        record(rows, &case, "compute_view_after_write", &x);
                        rows.last_mut().unwrap()["values"] = json!(compute);
                    },
                );
            }
        }
    }
}

fn training(device: Device, root: &Path) -> Vec<Value> {
    let mut curves = Vec::new();
    with_precision_config(
        PrecisionConfig {
            parameter_dtype: DType::F32,
            runtime_dtype: DType::F32,
            allow_parameter_dtype_copies: false,
        },
        || {
            for dtype in [DType::F32, DType::F16, DType::BF16, DType::I8] {
                for kind in ["sgd", "adam"] {
                    let model = Pair {
                        weight: parameter(
                            &[2, 3],
                            &[-0.7, 0.6, 0.5, -0.8, 0.4, -0.3],
                            dtype,
                            0,
                            device,
                        ),
                        bias: parameter(&[2], &[0.2, -0.2], dtype, 0, device),
                    };
                    let input = Tensor::from_data_with_grad_flag(
                        array(
                            &[4, 3],
                            &[1., 0., 0., 0., 1., 0., 0., 0., 1., 1., 1., 1.],
                            0,
                        ),
                        false,
                    );
                    let target = Tensor::from_data_with_grad_flag(
                        array(
                            &[4, 2],
                            &[0.35, -0.125, -0.15, 0.225, 0.15, -0.175, 0.25, -0.025],
                            0,
                        ),
                        false,
                    );
                    input.to_device_inplace(device);
                    target.to_device_inplace(device);
                    let mut opt: Box<dyn Optimizer> = if kind == "sgd" {
                        let opt = SGD::new_with_dtype(model.parameters(), 0.03, DType::F32);
                        Box::new(if dtype == DType::I8 {
                            opt.with_dynamic_i8_master_weights()
                        } else {
                            opt.with_fp32_master_weights()
                        })
                    } else {
                        let opt = Adam::new_with_dtype(model.parameters(), 0.02, DType::F32);
                        Box::new(if dtype == DType::I8 {
                            opt.with_dynamic_i8_master_weights()
                        } else {
                            opt.with_fp32_master_weights()
                        })
                    };
                    let initial = model.parameters().iter().map(values).collect::<Vec<_>>();
                    let mut losses = Vec::new();
                    let mut resume: Option<(Pair, SGD)> = None;
                    let mut resume_losses = Vec::new();
                    for step in 0..=45 {
                        opt.zero_grad();
                        let loss = MSELoss::apply(&model.forward(input.clone()), &target);
                        losses.push(loss.data()[[]]);
                        if let Some((other, opt)) = &mut resume {
                            opt.zero_grad();
                            let other_loss = MSELoss::apply(&other.forward(input.clone()), &target);
                            resume_losses.push(other_loss.data()[[]]);
                            assert!((other_loss.data()[[]] - loss.data()[[]]).abs() < 1e-6);
                            if step < 45 {
                                other_loss.backward();
                                opt.step();
                            }
                        }
                        if step == 20 && dtype == DType::F32 && kind == "sgd" {
                            let path = root.join("resume-f32-sgd.bin");
                            model.save(path.to_str().unwrap()).unwrap();
                            let other = Pair {
                                weight: parameter(&[2, 3], &[0.; 6], dtype, 0, device),
                                bias: parameter(&[2], &[0.; 2], dtype, 0, device),
                            };
                            other.load(path.to_str().unwrap()).unwrap();
                            let mut resumed =
                                SGD::new_with_dtype(other.parameters(), 0.03, DType::F32);
                            let seed = MSELoss::apply(&other.forward(input.clone()), &target);
                            seed.backward();
                            resumed.step();
                            resume = Some((other, resumed));
                        }
                        if step < 45 {
                            loss.backward();
                            opt.step();
                        }
                    }
                    assert!(losses.iter().all(|x| x.is_finite()) && losses[45] < losses[0]);
                    curves.push(json!({"dtype":format!("{dtype:?}"),"optimizer":kind,"initial":initial,"losses":losses,
                    "final":model.parameters().iter().map(values).collect::<Vec<_>>(),"resume_losses":resume_losses,
                    "fp32_master":true,"dynamic_i8":dtype==DType::I8}));
                }
            }
        },
    );
    curves
}

fn main() {
    let mut device = Device::Cpu;
    let mut root = PathBuf::from("target/manual-state-check");
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--device" => {
                device = match args.next().as_deref() {
                    Some("cpu") => Device::Cpu,
                    Some("cuda") => Device::Cuda,
                    _ => panic!("--device cpu|cuda"),
                }
            }
            "--output-directory" => root = PathBuf::from(args.next().expect("output directory")),
            _ => panic!("unknown argument: {arg}"),
        }
    }
    if !root.is_absolute() {
        root = std::env::current_dir().unwrap().join(root);
    }
    std::fs::create_dir_all(&root).unwrap();
    let _cuda = lumen::ops::cuda::set_enabled_scoped(device == Device::Cuda);
    let _strict = lumen::autograd::set_strict_device_execution_scoped(device == Device::Cuda);
    let mut rows = Vec::new();
    let mut files = Vec::new();
    snapshots(device, &root, &mut rows, &mut files);
    let probe = parameter(&[2, 2], &[1., 2., 3., 4.], DType::F32, 0, device);
    probe.data_mut()[[0, 0]] = 7.;
    let view = probe.data_ref();
    let copied = probe.to_device(device);
    let held_view_snapshot = values(&copied);
    assert_eq!(held_view_snapshot, [7., 2., 3., 4.]);
    assert_eq!(view[[0, 0]], 7.);
    let curves = training(device, &root);
    let report = json!({"device":format!("{device:?}"),"snapshots":rows,"checkpoints":files,"training":curves,"held_view_snapshot":held_view_snapshot});
    std::fs::write(
        root.join("results.json"),
        serde_json::to_vec_pretty(&report).unwrap(),
    )
    .unwrap();
    println!(
        "Manual state checks passed: {} snapshots, {} checkpoint sets, {} training curves; {}",
        rows.len(),
        files.len(),
        curves.len(),
        root.display()
    );
}
