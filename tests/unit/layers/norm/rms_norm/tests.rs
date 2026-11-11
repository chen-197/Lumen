use super::*;
use crate::autograd::no_grad;
use crate::precision::{PrecisionConfig, with_precision_config};
use ndarray::{Array, IxDyn};

fn make_tensor(shape: &[usize], data: Vec<f32>, dtype: DType) -> Tensor {
    let t = Tensor::from_array_no_grad(
        Array::from_shape_vec(IxDyn(shape), data)
            .expect("test tensor shape mismatch")
            .into_dyn(),
    );
    t.cast_inplace(dtype);
    t
}

#[test]
fn rms_forward_rows_observe_pool_budget_grain_and_nested_execution() {
    use crate::cpu::CpuKernelContext;
    use std::{num::NonZeroUsize, sync::Arc};
    let workers = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(3)
            .build()
            .unwrap(),
    );
    let caller = std::thread::current().id();
    for budget in [1, 2, 7] {
        let context = CpuKernelContext::with_pool(workers.clone())
            .with_task_limit(NonZeroUsize::new(budget).unwrap());
        context.scope(|| {
            for (rows, dim) in [(0, 513), (2, 129), (65, 513)] {
                let input = Array2::from_shape_fn((rows, dim), |(r, c)| (r * dim + c) as f32);
                let mut output = Array2::zeros((rows, dim));
                let parallel = crate::cpu::should_parallelize(rows, dim * 4);
                let active = std::sync::atomic::AtomicUsize::new(0);
                let peak = std::sync::atomic::AtomicUsize::new(0);
                for_each_forward_row(&mut output, input.view(), |mut out, row| {
                    use std::sync::atomic::Ordering;
                    if parallel {
                        assert!(workers.current_thread_index().is_some());
                    } else {
                        assert_eq!(std::thread::current().id(), caller);
                    }
                    assert_eq!(crate::cpu::parallelism(), 1);
                    let count = active.fetch_add(1, Ordering::SeqCst) + 1;
                    peak.fetch_max(count, Ordering::SeqCst);
                    let thread = std::thread::current().id();
                    let mut inner = Array2::<f32>::zeros((2, 129));
                    for_each_forward_row(
                        &mut inner,
                        input.slice(ndarray::s![..2, ..129]),
                        |_, _| {
                            assert_eq!(std::thread::current().id(), thread);
                        },
                    );
                    out.assign(&row);
                    active.fetch_sub(1, Ordering::SeqCst);
                });
                assert!(peak.load(std::sync::atomic::Ordering::SeqCst) <= budget.min(3));
                assert_eq!(output, input);
            }
        });
    }
}

#[test]
fn rms_forward_policy_preserves_mixed_storage_layouts_and_fp64_accuracy() {
    use crate::cpu::CpuKernelContext;
    use std::{num::NonZeroUsize, sync::Arc};
    let _cpu = cuda::set_enabled_scoped(false);
    let workers = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(3)
            .build()
            .unwrap(),
    );
    fn ordered<T: Clone + Default>(
        array: &ndarray::ArrayD<T>,
        strided: bool,
    ) -> ndarray::ArrayD<T> {
        if !strided {
            array.clone()
        } else if array.ndim() == 1 {
            let values = array.iter().cloned().collect::<Vec<_>>();
            ndarray::Array1::from_shape_fn(values.len() * 2, |i| values[i / 2].clone())
                .slice_move(ndarray::s![..;2])
                .into_dyn()
        } else {
            array
                .view()
                .reversed_axes()
                .as_standard_layout()
                .to_owned()
                .reversed_axes()
        }
    }
    fn tensor(shape: &[usize], values: Vec<f32>, dtype: DType, strided: bool) -> Tensor {
        let t = make_tensor(shape, values, dtype);
        match t.native_storage_owned() {
            TensorStorageOwned::F32(a) => {
                t.set_array_f32_with_dtype(ordered(&a.to_owned(), strided), dtype)
            }
            TensorStorageOwned::F16(a) => {
                t.set_array_f16_with_dtype(ordered(&a.to_owned(), strided), dtype)
            }
            TensorStorageOwned::BF16(a) => {
                t.set_array_bf16_with_dtype(ordered(&a.to_owned(), strided), dtype)
            }
            TensorStorageOwned::I8(a, scale) => {
                t.set_array_i8_with_dtype(ordered(&a.to_owned(), strided), scale, dtype)
            }
        }
        t
    }
    with_precision_config(
        PrecisionConfig {
            parameter_dtype: DType::F32,
            runtime_dtype: DType::F32,
            allow_parameter_dtype_copies: false,
        },
        || {
            for input_dtype in [DType::F32, DType::F16, DType::BF16, DType::I8] {
                for weight_dtype in [DType::F32, DType::F16, DType::BF16, DType::I8] {
                    for strided in [false, true] {
                        for graph in [false, true] {
                            if graph && input_dtype != DType::BF16 {
                                continue;
                            }
                            let (rows, dim) = (43, 513);
                            let input = tensor(
                                &[1, rows, dim],
                                (0..rows * dim)
                                    .map(|i| ((i * 17 + 3) % 251) as f32 * 0.013 - 1.625)
                                    .collect(),
                                input_dtype,
                                strided,
                            );
                            let norm = RMSNorm {
                                weight: tensor(
                                    &[dim],
                                    (0..dim)
                                        .map(|i| ((i * 13 + 11) % 97) as f32 * 0.017 - 0.816)
                                        .collect(),
                                    weight_dtype,
                                    strided,
                                ),
                                eps: 1e-5,
                            };
                            norm.weight.0.borrow_mut().requires_grad = graph;
                            let x = input.detach().data().iter().copied().collect::<Vec<_>>();
                            let w = norm
                                .weight
                                .detach()
                                .data()
                                .iter()
                                .copied()
                                .collect::<Vec<_>>();
                            let mut previous = None;
                            for budget in [1, 3] {
                                let context = CpuKernelContext::with_pool(workers.clone())
                                    .with_task_limit(NonZeroUsize::new(budget).unwrap());
                                let output = context.scope(|| {
                                    if graph {
                                        norm.forward(input.clone())
                                    } else {
                                        no_grad(|| norm.forward(input.clone()))
                                    }
                                });
                                assert_eq!(output.shape_vec(), vec![1, rows, dim]);
                                assert_eq!(
                                    output.dtype(),
                                    if graph { DType::F32 } else { input_dtype }
                                );
                                assert_eq!(input.dtype(), input_dtype);
                                assert_eq!(norm.weight.dtype(), weight_dtype);
                                let raw = bincode::serialize(&output.export_raw()).unwrap();
                                if let Some(old) = &previous {
                                    assert_eq!(&raw, old);
                                }
                                previous = Some(raw);
                                let actual = output.detach().data();
                                let scale = output.quantization_scale().unwrap_or(0.);
                                for r in 0..rows {
                                    let row = &x[r * dim..(r + 1) * dim];
                                    let inv =
                                        (row.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>()
                                            / dim as f64
                                            + f64::from(norm.eps))
                                        .sqrt()
                                        .recip();
                                    for c in 0..dim {
                                        let expected = f64::from(row[c]) * inv * f64::from(w[c]);
                                        let relative = match output.dtype() {
                                            DType::F32 => 2e-5,
                                            DType::F16 => 1. / 1024.,
                                            DType::BF16 => 1. / 128.,
                                            DType::I8 => 0.,
                                        };
                                        let error = (f64::from(actual[[0, r, c]]) - expected).abs();
                                        assert!(
                                            error
                                                <= 2e-5
                                                    + relative * expected.abs()
                                                    + f64::from(scale) * 0.501,
                                            "{input_dtype:?}/{weight_dtype:?}/{strided}/{graph}/{budget}: {error} vs {expected}"
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
            for dtype in [DType::F32, DType::F16, DType::BF16, DType::I8] {
                let input = make_tensor(&[0, 129], vec![], dtype);
                let output = no_grad(|| RMSNorm::new(129, 1e-5).forward(input));
                assert_eq!(output.shape_vec(), vec![0, 129]);
                assert_eq!(output.dtype(), dtype);
            }
        },
    );
}

#[test]
fn rms_norm_training_preserves_cpu_native_input_and_parameter_storage() {
    with_precision_config(
        PrecisionConfig {
            parameter_dtype: DType::F32,
            runtime_dtype: DType::F32,
            allow_parameter_dtype_copies: false,
        },
        || {
            for dtype in [DType::F32, DType::F16, DType::BF16, DType::I8] {
                let norm = RMSNorm::new_with_dtype(129, 1e-5, dtype);
                let scale = norm.weight.quantization_scale();
                let input = make_tensor(
                    &[3, 129],
                    (0..3 * 129)
                        .map(|i| ((i * 7 % 97) as f32 - 48.0) / 17.0)
                        .collect(),
                    DType::BF16,
                );
                let target = make_tensor(&[3, 129], vec![0.5; 3 * 129], DType::F32);
                for _ in 0..2 {
                    let output = norm.forward(input.clone());
                    assert_eq!(input.dtype(), DType::BF16);
                    assert_eq!(norm.weight.dtype(), dtype);
                    crate::loss::MSELoss::apply(&output, &target).backward();
                    assert_eq!(input.dtype(), DType::BF16);
                    assert_eq!(norm.weight.dtype(), dtype);
                    assert_eq!(norm.weight.quantization_scale(), scale);
                    assert!(
                        norm.weight
                            .grad_ref()
                            .as_ref()
                            .unwrap()
                            .iter()
                            .all(|v| v.is_finite())
                    );
                    norm.weight.zero_grad();
                }
            }
        },
    );
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_rms_norm_cooperative_training_loss_decreases_across_parameter_types() {
    use crate::device::{Device, set_strict_device_execution_scoped};
    use crate::loss::MSELoss;
    use crate::optim::{Adam, Optimizer, SGD};
    if !cuda::is_available() {
        return;
    }
    let dim = 257;
    let rows = 3;
    let values: Vec<_> = (0..rows * dim)
        .map(|i| ((i * 37 % 251) as f32 - 125.0) / 43.0)
        .collect();
    let rounded: Vec<_> = values.iter().map(|&v| bf16::from_f32(v).to_f32()).collect();
    let mut targets = Vec::with_capacity(rows * dim);
    for row in rounded.chunks_exact(dim) {
        let inv = (row.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>() / dim as f64
            + f64::from(1e-5f32))
        .sqrt()
        .recip();
        targets.extend(
            row.iter()
                .enumerate()
                .map(|(j, &v)| (f64::from(v) * inv * (0.2 + 0.5 * (j % 17) as f64 / 16.0)) as f32),
        );
    }
    for device in [Device::Cpu, Device::Cuda] {
        let _cuda_guard = cuda::set_enabled_scoped(device == Device::Cuda);
        let _strict_guard = set_strict_device_execution_scoped(device == Device::Cuda);
        for dtype in [DType::F32, DType::F16, DType::BF16, DType::I8] {
            for algorithm in ["sgd_momentum", "adam"] {
                let norm = RMSNorm::new_with_dtype(dim, 1e-5, dtype);
                norm.weight.to_device_inplace(device);
                let input =
                    make_tensor(&[rows, dim], values.clone(), DType::BF16).to_device(device);
                let target =
                    make_tensor(&[rows, dim], targets.clone(), DType::F32).to_device(device);
                let params = vec![norm.weight.clone()];
                let mut optimizer: Box<dyn Optimizer> = if algorithm == "adam" {
                    Box::new(
                        Adam::new_with_dtype(params, 0.03, DType::F32)
                            .with_dynamic_i8_master_weights(),
                    )
                } else {
                    Box::new(
                        SGD::new_with_dtype(params, 10.0, DType::F32)
                            .with_momentum(0.3)
                            .with_dynamic_i8_master_weights(),
                    )
                };
                let mut losses = Vec::new();
                for _ in 0..48 {
                    let output = norm.forward(input.clone());
                    let loss = MSELoss::apply(&output, &target);
                    losses.push(loss.data_ref().iter().copied().next().unwrap());
                    loss.backward();
                    if device == Device::Cuda {
                        assert!(norm.weight.cloned_cuda_f32_grad().is_some());
                        assert!(!norm.weight.has_host_grad());
                    }
                    optimizer.step();
                    optimizer.zero_grad();
                    assert_eq!(norm.weight.dtype(), dtype);
                }
                assert!(losses.iter().all(|v| v.is_finite()));
                let head = losses[..8].iter().sum::<f32>() / 8.0;
                let tail = losses[40..].iter().sum::<f32>() / 8.0;
                assert!(
                    tail < head * 0.1,
                    "{device:?}/{dtype:?}/{algorithm}: {head} -> {tail}"
                );
                assert!(losses[47] < losses[0] * 0.1);
                println!(
                    "rmsnorm_training {}",
                    serde_json::json!({
                        "device": format!("{device:?}"), "dtype": format!("{dtype:?}"),
                        "optimizer": algorithm, "losses": losses,
                    })
                );
            }
        }
    }
}

#[test]
fn rms_norm_no_grad_preserves_bf16_input_dtype() {
    let norm = RMSNorm::new(4, 1e-5);
    norm.weight.set_array_f32_with_dtype(
        Array::from_shape_vec(IxDyn(&[4]), vec![1.0, 0.5, 1.5, 2.0])
            .expect("weight shape mismatch")
            .into_dyn(),
        DType::BF16,
    );

    let input_f32 = make_tensor(&[1, 1, 4], vec![1.0, -2.0, 3.0, -4.0], DType::F32);
    let input_bf16 = make_tensor(&[1, 1, 4], vec![1.0, -2.0, 3.0, -4.0], DType::BF16);

    let ref_out = no_grad(|| norm.forward(input_f32));
    let bf16_out = no_grad(|| norm.forward(input_bf16));

    assert_eq!(bf16_out.dtype(), DType::BF16);
    let ref_vals = ref_out
        .data_ref()
        .iter()
        .map(|&v| bf16::from_f32(v).to_f32())
        .collect::<Vec<_>>();
    bf16_out.with_storage_view(|view| match view {
        TensorStorageView::BF16(view) => {
            let vals = view.iter().map(|v| v.to_f32()).collect::<Vec<_>>();
            assert_eq!(vals, ref_vals);
        }
        TensorStorageView::F16(_) => panic!("bf16 RMSNorm output should stay bf16 in no-grad"),
        TensorStorageView::F32(_) => panic!("bf16 RMSNorm output should stay bf16 in no-grad"),
    });
}

#[test]
fn rms_norm_explicit_dtype_overrides_global_default() {
    with_precision_config(
        PrecisionConfig {
            parameter_dtype: DType::BF16,
            runtime_dtype: DType::F32,
            allow_parameter_dtype_copies: false,
        },
        || {
            let norm = RMSNorm::new_with_dtype(4, 1e-5, DType::F32);
            assert_eq!(norm.weight.dtype(), DType::F32);
        },
    );
}

#[test]
fn rms_norm_no_grad_preserves_i8_input_dtype() {
    let norm = RMSNorm::new(4, 1e-5);
    norm.weight.set_array_f32_with_dtype(
        Array::from_shape_vec(IxDyn(&[4]), vec![1.0, 0.5, 1.5, 2.0])
            .expect("weight shape mismatch")
            .into_dyn(),
        DType::BF16,
    );

    let input_i8 = make_tensor(&[1, 1, 4], vec![1.0, -2.0, 3.0, -4.0], DType::I8);
    let out = no_grad(|| norm.forward(input_i8));
    assert_eq!(out.dtype(), DType::I8);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_rms_norm_matches_cpu_reference_in_strict_mode() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let weight_values = vec![1.0, 0.5, 1.5, 2.0];
    let norm = RMSNorm::new(4, 1e-5);
    norm.weight.set_array_f32_with_dtype(
        Array::from_shape_vec(IxDyn(&[4]), weight_values.clone())
            .expect("weight shape mismatch")
            .into_dyn(),
        DType::BF16,
    );
    norm.weight.to_cuda_inplace();
    let norm_ref = RMSNorm::new(4, 1e-5);
    norm_ref.weight.set_array_f32_with_dtype(
        Array::from_shape_vec(IxDyn(&[4]), weight_values)
            .expect("weight shape mismatch")
            .into_dyn(),
        DType::BF16,
    );

    let input = make_tensor(
        &[1, 2, 4],
        vec![1.0, -2.0, 3.0, -4.0, 0.5, 1.5, -2.5, 4.0],
        DType::BF16,
    )
    .to_cuda();
    assert!(input.0.borrow().cuda_f32_data.is_none());
    assert!(norm.weight.0.borrow().cuda_f32_data.is_none());

    crate::ops::cuda::set_enabled(true);
    crate::device::set_strict_device_execution(true);
    let cuda_out = no_grad(|| norm.forward(input.clone()));
    crate::device::set_strict_device_execution(false);
    crate::ops::cuda::set_enabled(false);
    assert!(input.0.borrow().cuda_f32_data.is_none());
    assert!(norm.weight.0.borrow().cuda_f32_data.is_none());

    let cpu_out = no_grad(|| norm_ref.forward(input.to_cpu()));
    assert!(cuda_out.is_cuda());
    assert_eq!(cuda_out.dtype(), DType::BF16);
    {
        let inner = cuda_out.0.borrow();
        assert!(
            !inner.has_f32_data,
            "CUDA RMSNorm no-grad output should not eagerly materialize host f32 data"
        );
        assert!(
            inner.cuda_f32_data.is_none(),
            "CUDA RMSNorm no-grad output should not keep a resident f32 compute buffer for bf16 output"
        );
        assert!(
            inner.cuda_bf16_data.is_some(),
            "CUDA RMSNorm no-grad output should keep resident bf16 storage"
        );
    }

    for (got, expect) in cuda_out.data_ref().iter().zip(cpu_out.data_ref().iter()) {
        assert!((got - expect).abs() < 2e-2, "got {got}, expect {expect}");
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_empty_rms_norm_stays_on_cuda_in_strict_mode() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let norm = RMSNorm::new(4, 1e-5);
    norm.weight.set_array_f32_with_dtype(
        Array::from_shape_vec(IxDyn(&[4]), vec![1.0, 0.5, 1.5, 2.0])
            .expect("weight shape mismatch")
            .into_dyn(),
        DType::BF16,
    );
    norm.weight.to_cuda_inplace();
    let input = make_tensor(&[0, 4], vec![], DType::BF16).to_cuda();

    crate::ops::cuda::set_enabled(true);
    crate::device::set_strict_device_execution(true);
    let out = no_grad(|| norm.forward(input.clone()));
    crate::device::set_strict_device_execution(false);
    crate::ops::cuda::set_enabled(false);

    assert!(out.is_cuda());
    assert_eq!(out.dtype(), DType::BF16);
    assert_eq!(out.shape_vec(), vec![0, 4]);
    assert_eq!(out.len(), 0);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_rms_norm_backward_matches_cpu_reference_in_strict_mode() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let weight_values = vec![1.0, 0.5, 1.5, 2.0];
    let input_values = vec![1.0, -2.0, 3.0, -4.0, 0.5, 1.5, -2.5, 4.0];
    let coeff_values = vec![0.5, -1.0, 2.0, 0.25, -0.75, 1.5, -0.5, 0.75];

    let norm_cuda = RMSNorm::new(4, 1e-5);
    norm_cuda.weight.set_array_f32_with_dtype(
        Array::from_shape_vec(IxDyn(&[4]), weight_values.clone())
            .expect("weight shape mismatch")
            .into_dyn(),
        DType::F32,
    );
    norm_cuda.weight.to_cuda_inplace();
    let input_cpu = Tensor::from_data_with_grad_flag(
        Array::from_shape_vec(IxDyn(&[1, 2, 4]), input_values.clone())
            .expect("input shape mismatch")
            .into_dyn(),
        true,
    );
    let coeff_cpu = Tensor::from_data_with_grad_flag(
        Array::from_shape_vec(IxDyn(&[1, 2, 4]), coeff_values)
            .expect("coeff shape mismatch")
            .into_dyn(),
        false,
    );
    let input_cuda = input_cpu.to_cuda();
    let coeff_cuda = coeff_cpu.to_cuda();

    crate::ops::cuda::set_enabled(true);
    crate::device::set_strict_device_execution(true);
    let cuda_out = norm_cuda.forward(input_cuda.clone());
    let cuda_loss = crate::ops::arithmetic::sum(&(&cuda_out * &coeff_cuda));
    cuda_loss.backward();
    assert!(input_cuda.cloned_cuda_f32_grad().is_some());
    assert!(norm_cuda.weight.cloned_cuda_f32_grad().is_some());
    assert!(!input_cuda.has_host_grad());
    assert!(!norm_cuda.weight.has_host_grad());
    crate::device::set_strict_device_execution(false);
    crate::ops::cuda::set_enabled(false);

    let norm_cpu = RMSNorm::new(4, 1e-5);
    norm_cpu.weight.set_array_f32_with_dtype(
        Array::from_shape_vec(IxDyn(&[4]), weight_values)
            .expect("weight shape mismatch")
            .into_dyn(),
        DType::F32,
    );
    let cpu_out = norm_cpu.forward(input_cpu.clone());
    let cpu_loss = crate::ops::arithmetic::sum(&(&cpu_out * &coeff_cpu));
    cpu_loss.backward();

    let input_cuda_grad = input_cuda.grad().expect("cuda RMSNorm input grad");
    let input_cpu_grad = input_cpu.grad().expect("cpu RMSNorm input grad");
    let weight_cuda_grad = norm_cuda.weight.grad().expect("cuda RMSNorm weight grad");
    let weight_cpu_grad = norm_cpu.weight.grad().expect("cpu RMSNorm weight grad");
    for (got, expect) in input_cuda_grad.iter().zip(input_cpu_grad.iter()) {
        assert!(
            (got - expect).abs() < 1e-5,
            "input got {got}, expect {expect}"
        );
    }
    for (got, expect) in weight_cuda_grad.iter().zip(weight_cpu_grad.iter()) {
        assert!(
            (got - expect).abs() < 1e-5,
            "weight got {got}, expect {expect}"
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_rms_norm_lowp_backward_uses_resident_inputs_and_f32_grads() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let weight_values = vec![1.0, 0.5, 1.5, 2.0];
    let input_values = vec![1.0, -2.0, 3.0, -4.0, 0.5, 1.5, -2.5, 4.0];
    let coeff_values = vec![0.5, -1.0, 2.0, 0.25, -0.75, 1.5, -0.5, 0.75];

    let norm_cuda = RMSNorm::new(4, 1e-5);
    norm_cuda.weight.set_array_f32_with_dtype(
        Array::from_shape_vec(IxDyn(&[4]), weight_values.clone())
            .expect("weight shape mismatch")
            .into_dyn(),
        DType::BF16,
    );
    norm_cuda.weight.to_cuda_inplace();

    let input_cuda = Tensor::from_data_with_grad_flag(
        Array::from_shape_vec(IxDyn(&[1, 2, 4]), input_values.clone())
            .expect("input shape mismatch")
            .into_dyn(),
        true,
    );
    input_cuda.cast_inplace(DType::BF16);
    let input_cuda = input_cuda.to_cuda();
    let coeff_cuda = Tensor::from_data_with_grad_flag(
        Array::from_shape_vec(IxDyn(&[1, 2, 4]), coeff_values.clone())
            .expect("coeff shape mismatch")
            .into_dyn(),
        false,
    )
    .to_cuda();

    assert!(input_cuda.0.borrow().cuda_f32_data.is_none());
    assert!(norm_cuda.weight.0.borrow().cuda_f32_data.is_none());

    crate::ops::cuda::set_enabled(true);
    crate::device::set_strict_device_execution(true);
    let cuda_out = norm_cuda.forward(input_cuda.clone());
    let cuda_loss = crate::ops::arithmetic::sum(&(&cuda_out * &coeff_cuda));
    cuda_loss.backward();
    crate::device::set_strict_device_execution(false);
    crate::ops::cuda::set_enabled(false);

    assert!(input_cuda.0.borrow().cuda_f32_data.is_none());
    assert!(norm_cuda.weight.0.borrow().cuda_f32_data.is_none());
    assert!(input_cuda.cloned_cuda_f32_grad().is_some());
    assert!(norm_cuda.weight.cloned_cuda_f32_grad().is_some());
    assert!(!input_cuda.has_host_grad());
    assert!(!norm_cuda.weight.has_host_grad());

    let norm_cpu = RMSNorm::new(4, 1e-5);
    norm_cpu.weight.set_array_f32_with_dtype(
        Array::from_shape_vec(IxDyn(&[4]), weight_values)
            .expect("weight shape mismatch")
            .into_dyn(),
        DType::BF16,
    );
    let input_cpu = Tensor::from_data_with_grad_flag(
        Array::from_shape_vec(IxDyn(&[1, 2, 4]), input_values)
            .expect("input shape mismatch")
            .into_dyn(),
        true,
    );
    input_cpu.cast_inplace(DType::BF16);
    let coeff_cpu = Tensor::from_data_with_grad_flag(
        Array::from_shape_vec(IxDyn(&[1, 2, 4]), coeff_values)
            .expect("coeff shape mismatch")
            .into_dyn(),
        false,
    );
    let cpu_out = norm_cpu.forward(input_cpu.clone());
    let cpu_loss = crate::ops::arithmetic::sum(&(&cpu_out * &coeff_cpu));
    cpu_loss.backward();

    let input_cuda_grad = input_cuda.grad().expect("cuda RMSNorm input grad");
    let input_cpu_grad = input_cpu.grad().expect("cpu RMSNorm input grad");
    let weight_cuda_grad = norm_cuda.weight.grad().expect("cuda RMSNorm weight grad");
    let weight_cpu_grad = norm_cpu.weight.grad().expect("cpu RMSNorm weight grad");
    for (got, expect) in input_cuda_grad.iter().zip(input_cpu_grad.iter()) {
        assert!(
            (got - expect).abs() < 3e-2,
            "input got {got}, expect {expect}"
        );
    }
    for (got, expect) in weight_cuda_grad.iter().zip(weight_cpu_grad.iter()) {
        assert!(
            (got - expect).abs() < 3e-2,
            "weight got {got}, expect {expect}"
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_rms_norm_preserves_i8_dtype_in_strict_mode() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let weight_values = vec![1.0, 0.5, 1.5, 2.0];
    let norm = RMSNorm::new(4, 1e-5);
    norm.weight.set_array_f32_with_dtype(
        Array::from_shape_vec(IxDyn(&[4]), weight_values.clone())
            .expect("weight shape mismatch")
            .into_dyn(),
        DType::BF16,
    );
    norm.weight.to_cuda_inplace();
    let norm_ref = RMSNorm::new(4, 1e-5);
    norm_ref.weight.set_array_f32_with_dtype(
        Array::from_shape_vec(IxDyn(&[4]), weight_values)
            .expect("weight shape mismatch")
            .into_dyn(),
        DType::BF16,
    );

    let input = make_tensor(
        &[1, 2, 4],
        vec![1.0, -2.0, 3.0, -4.0, 0.5, 1.5, -2.5, 4.0],
        DType::I8,
    )
    .to_cuda();
    assert!(input.0.borrow().cuda_f32_data.is_none());
    assert!(norm.weight.0.borrow().cuda_f32_data.is_none());

    crate::ops::cuda::set_enabled(true);
    crate::device::set_strict_device_execution(true);
    let cuda_out = no_grad(|| norm.forward(input.clone()));
    crate::device::set_strict_device_execution(false);
    crate::ops::cuda::set_enabled(false);
    assert!(input.0.borrow().cuda_f32_data.is_none());
    assert!(norm.weight.0.borrow().cuda_f32_data.is_none());

    let cpu_out = no_grad(|| norm_ref.forward(input.to_cpu()));
    assert!(cuda_out.is_cuda());
    assert_eq!(cuda_out.dtype(), DType::I8);
    {
        let inner = cuda_out.0.borrow();
        assert!(
            !inner.has_f32_data,
            "CUDA RMSNorm I8 no-grad output should not eagerly materialize host f32 data"
        );
        assert!(
            inner.cuda_f32_data.is_none(),
            "CUDA RMSNorm I8 no-grad output should not keep a resident f32 compute buffer"
        );
        assert!(
            inner.cuda_i8_data.is_some(),
            "CUDA RMSNorm I8 no-grad output should keep resident i8 storage"
        );
        assert!(
            inner.i8_scale.is_some(),
            "CUDA RMSNorm I8 no-grad output should record dynamic quantization scale"
        );
    }
    for (got, expect) in cuda_out.data_ref().iter().zip(cpu_out.data_ref().iter()) {
        assert!((got - expect).abs() < 1e-5, "got {got}, expect {expect}");
    }
}
