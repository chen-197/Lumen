use super::*;
use crate::autograd::{TensorStorageOwned, no_grad};
use crate::cpu::CpuKernelContext;
use crate::precision::{PrecisionConfig, with_precision_config};
use ndarray::{Axis, Dimension, Slice};
use std::{num::NonZeroUsize, sync::Arc};

fn backward_reference(
    input: &[f32],
    grad: &[f32],
    weight: Option<&[f32]>,
    dim: usize,
    eps: f32,
) -> (Vec<u32>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut old_bits = vec![0; input.len()];
    let mut dx = vec![0.0; input.len()];
    let mut dw = vec![0.0; dim];
    let mut db = vec![0.0; dim];
    for (row, src) in input.chunks(dim).enumerate() {
        let offset = row * dim;
        let mean = src.iter().map(|&v| f64::from(v)).sum::<f64>() / dim as f64;
        let var = src
            .iter()
            .map(|&v| (f64::from(v) - mean).powi(2))
            .sum::<f64>()
            / dim as f64;
        let inv = (var + f64::from(eps)).sqrt().recip();
        let normalized = src
            .iter()
            .map(|&v| (f64::from(v) - mean) * inv)
            .collect::<Vec<_>>();
        let mut sum = 0.0;
        let mut dot = 0.0;
        let mut rounded_dot = 0.0;
        for j in 0..dim {
            let g = f64::from(grad[offset + j]) * weight.map_or(1.0, |w| f64::from(w[j]));
            sum += g;
            dot += g * normalized[j];
            rounded_dot += g * f64::from(normalized[j] as f32);
            dw[j] += f64::from(grad[offset + j]) * normalized[j];
            db[j] += f64::from(grad[offset + j]);
        }
        for j in 0..dim {
            let g = f64::from(grad[offset + j]) * weight.map_or(1.0, |w| f64::from(w[j]));
            dx[offset + j] = inv * (g - sum / dim as f64 - normalized[j] * dot / dim as f64);
            let old = f64::from(inv as f32)
                * (g - sum / dim as f64
                    - f64::from(normalized[j] as f32) * rounded_dot / dim as f64);
            old_bits[offset + j] = (old as f32).to_bits();
        }
    }
    (old_bits, dx, dw, db)
}

#[test]
fn cpu_layer_norm_backward_policy_matches_original_rows_and_fp64() {
    let _cpu = crate::ops::cuda::set_enabled_scoped(false);
    let pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(3)
            .build()
            .unwrap(),
    );
    with_precision_config(
        PrecisionConfig {
            parameter_dtype: DType::F32,
            runtime_dtype: DType::F32,
            allow_parameter_dtype_copies: false,
        },
        || {
            for input_dtype in [DType::F32, DType::F16, DType::BF16] {
                for parameter_dtype in [DType::F32, DType::F16, DType::BF16] {
                    for (shape, suffix, layout) in [
                        (vec![2, 3, 5], vec![3, 5], 0),
                        (vec![65, 513], vec![513], 1),
                        (vec![65, 513], vec![513], 2),
                        (vec![2, 0, 17], vec![17], 0),
                        (vec![3, 1], vec![1], 2),
                    ] {
                        let dim = suffix.iter().product::<usize>();
                        let count = shape.iter().product::<usize>();
                        for (affine, bias) in [(true, true), (true, false), (false, false)] {
                            let input = Tensor::new(ArrayD::zeros(IxDyn(&shape)));
                            set_values(
                                &input,
                                &shape,
                                (0..count)
                                    .map(|i| ((i * 17 + 3) % 251) as f32 / 53. - 2.3)
                                    .collect(),
                                input_dtype,
                                layout,
                            );
                            let norm = LayerNorm::with_options(
                                suffix.clone(),
                                1e-5,
                                affine,
                                bias,
                                parameter_dtype,
                            );
                            for (p, seed) in norm.parameters().iter().zip([11, 29]) {
                                set_values(
                                    p,
                                    &suffix,
                                    (0..dim)
                                        .map(|i| ((i * 13 + seed) % 97) as f32 / 59. - 0.8)
                                        .collect(),
                                    parameter_dtype,
                                    layout,
                                );
                            }
                            let x = values(&input);
                            let weight = norm.weight.as_ref().map(values);
                            let b = norm.bias.as_ref().map(values);
                            let seed = ordered(
                                ArrayD::from_shape_fn(IxDyn(&shape), |i| {
                                    let index = i
                                        .slice()
                                        .iter()
                                        .zip(&shape)
                                        .fold(0, |a, (&i, &n)| a * n + i);
                                    ((index * 19 + 7) % 127) as f32 / 127. - 0.4
                                }),
                                layout,
                            );
                            let g = seed.iter().copied().collect::<Vec<_>>();
                            let (old_bits, dx64, dw64, db64) =
                                backward_reference(&x, &g, weight.as_deref(), dim, norm.eps());
                            for budget in [1, 3, 0] {
                                let policy = CpuKernelContext::with_pool(pool.clone());
                                let policy = if let Some(n) = NonZeroUsize::new(budget) {
                                    policy.with_task_limit(n)
                                } else {
                                    policy
                                };
                                input.zero_grad();
                                for p in norm.parameters() {
                                    p.zero_grad();
                                }
                                let output = policy.scope(|| norm.forward(input.clone()));
                                policy.scope(|| output.backward_with_grad(seed.clone()));
                                let dx = input.grad().unwrap().iter().copied().collect::<Vec<_>>();
                                let dw = norm
                                    .weight
                                    .as_ref()
                                    .map(|p| p.grad().unwrap().iter().copied().collect::<Vec<_>>());
                                let db = norm
                                    .bias
                                    .as_ref()
                                    .map(|p| p.grad().unwrap().iter().copied().collect::<Vec<_>>());
                                assert_eq!(
                                    dx.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                                    old_bits
                                );
                                crate::accuracy_gate::check("LayerNorm dx", &dx, &dx64, 3e-6, 3e-5);
                                if let Some(dw) = &dw {
                                    crate::accuracy_gate::check(
                                        "LayerNorm dw",
                                        dw,
                                        &dw64,
                                        3e-6,
                                        3e-5,
                                    );
                                }
                                if let Some(db) = &db {
                                    crate::accuracy_gate::check(
                                        "LayerNorm db",
                                        db,
                                        &db64,
                                        3e-6,
                                        3e-5,
                                    );
                                }
                                assert_eq!(input.dtype(), input_dtype);
                                assert!(
                                    norm.parameters()
                                        .iter()
                                        .all(|p| p.dtype() == parameter_dtype)
                                );
                                if affine
                                    && bias
                                    && budget == 3
                                    && (count == 30
                                        || (input_dtype == DType::F32
                                            && parameter_dtype == DType::F32
                                            && layout == 1))
                                {
                                    println!(
                                        "layernorm_backward_policy {}",
                                        serde_json::json!({"input_dtype":format!("{input_dtype:?}"),"parameter_dtype":format!("{parameter_dtype:?}"),"shape":shape,"suffix":suffix,"input":x,"weight":weight,"bias":b,"seed":g,"dx":dx,"dw":dw,"db":db})
                                    );
                                }
                            }
                        }
                    }
                }
            }
        },
    );
}

#[test]
fn cpu_layer_norm_backward_skips_frozen_gradients_and_handles_aliases() {
    let _cpu = crate::ops::cuda::set_enabled_scoped(false);
    let pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(3)
            .build()
            .unwrap(),
    );
    let policy = CpuKernelContext::with_pool(pool).with_task_limit(NonZeroUsize::new(3).unwrap());
    for (rows, dim) in [(2, 17), (65, 513)] {
        let grad = (0..rows * dim)
            .map(|i| ((i * 7 % 127) as f32 - 60.) / 83.)
            .collect::<Vec<_>>();
        let xhat = (0..rows * dim)
            .map(|i| ((i * 13 % 97) as f32 - 40.) / 59.)
            .collect::<Vec<_>>();
        let inverse = vec![0.7; rows];
        let weight = vec![0.8; dim];
        let reference = CpuKernelContext::default()
            .with_task_limit(NonZeroUsize::new(1).unwrap())
            .scope(|| cpu_backward_rows(&grad, &xhat, &inverse, Some(&weight), dim, [true; 3]));
        for mask in 0..8 {
            let needs = [mask & 1 != 0, mask & 2 != 0, mask & 4 != 0];
            let actual = policy
                .scope(|| cpu_backward_rows(&grad, &xhat, &inverse, Some(&weight), dim, needs));
            assert_eq!(actual.input.is_some(), needs[0]);
            assert_eq!(actual.weight.is_some(), needs[1]);
            assert_eq!(actual.bias.is_some(), needs[2]);
            if let Some(dx) = actual.input {
                assert_eq!(dx, reference.input.as_ref().unwrap().as_slice());
            }
            for (actual, reference) in [
                (actual.weight, &reference.weight),
                (actual.bias, &reference.bias),
            ] {
                if let Some(a) = actual {
                    for (&a, &b) in a.iter().zip(reference.as_ref().unwrap()) {
                        assert!((a - b).abs() <= 1e-12 * (1. + b.abs()));
                    }
                }
            }
            let input =
                Tensor::new(ArrayD::from_shape_vec(IxDyn(&[rows, dim]), xhat.clone()).unwrap());
            let norm = LayerNorm::new(vec![dim], 1e-5);
            // Build a graph first, then change flags to cover backward-time freezing.
            let output = policy.scope(|| norm.forward(input.clone()));
            input.0.borrow_mut().requires_grad = needs[0];
            norm.weight.as_ref().unwrap().0.borrow_mut().requires_grad = needs[1];
            norm.bias.as_ref().unwrap().0.borrow_mut().requires_grad = needs[2];
            policy.scope(|| {
                output.backward_with_grad(
                    ArrayD::from_shape_vec(IxDyn(&[rows, dim]), grad.clone()).unwrap(),
                )
            });
            assert_eq!(input.grad().is_some(), needs[0]);
            assert_eq!(norm.weight.as_ref().unwrap().grad().is_some(), needs[1]);
            assert_eq!(norm.bias.as_ref().unwrap().grad().is_some(), needs[2]);
        }
    }
    for dtype in [DType::F32, DType::F16, DType::BF16] {
        let input = Tensor::new(ArrayD::zeros(IxDyn(&[129])));
        set_values(
            &input,
            &[129],
            (0..129).map(|i| (i % 17) as f32 / 13. - 0.5).collect(),
            dtype,
            2,
        );
        let mut norm = LayerNorm::with_options(vec![129], 1e-5, true, true, dtype);
        norm.weight = Some(input.clone());
        norm.bias = Some(input.clone());
        let x = values(&input);
        let g = vec![0.2; 129];
        let (bits, _, dw, db) = backward_reference(&x, &g, Some(&x), 129, norm.eps());
        let output = policy.scope(|| norm.forward(input.clone()));
        policy.scope(|| output.backward_with_grad(ArrayD::from_elem(IxDyn(&[129]), 0.2)));
        let actual = input.grad().unwrap();
        for (i, &a) in actual.iter().enumerate() {
            let expected = (f32::from_bits(bits[i]) + dw[i] as f32) + db[i] as f32;
            assert!((a - expected).abs() <= 3e-6 * (1. + expected.abs()));
        }
        assert_eq!(input.dtype(), dtype);
    }
}

#[test]
fn cpu_layer_norm_scoped_training_updates_input_and_parameters() {
    use crate::{
        loss::MSELoss,
        optim::{Adam, Optimizer, SGD},
    };
    let _cpu = crate::ops::cuda::set_enabled_scoped(false);
    let pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(3)
            .build()
            .unwrap(),
    );
    let policy = CpuKernelContext::with_pool(pool).with_task_limit(NonZeroUsize::new(3).unwrap());
    with_precision_config(
        PrecisionConfig {
            parameter_dtype: DType::F32,
            runtime_dtype: DType::F32,
            allow_parameter_dtype_copies: false,
        },
        || {
            for dtype in [DType::F32, DType::F16, DType::BF16] {
                for adam in [false, true] {
                    let (rows, dim) = (65, 129);
                    let input = Tensor::new(ArrayD::zeros(IxDyn(&[rows, dim])));
                    set_values(
                        &input,
                        &[rows, dim],
                        (0..rows * dim)
                            .map(|i| ((i * 37 % 251) as f32 - 125.) / 43.)
                            .collect(),
                        dtype,
                        1,
                    );
                    let norm = LayerNorm::with_options(vec![dim], 1e-5, true, true, dtype);
                    let target = Tensor::from_array_no_grad(ArrayD::zeros(IxDyn(&[rows, dim])));
                    let params = std::iter::once(input.clone())
                        .chain(norm.parameters())
                        .collect();
                    let mut optimizer: Box<dyn Optimizer> = if adam {
                        Box::new(Adam::new_with_dtype(params, 0.03, DType::F32))
                    } else {
                        Box::new(SGD::new_with_dtype(params, 5., DType::F32))
                    };
                    let mut losses = Vec::new();
                    for step in 0..=48 {
                        let output = policy.scope(|| norm.forward(input.clone()));
                        if matches!(step, 0 | 24 | 48) {
                            check_reference(&output, &input, &norm);
                        }
                        let loss = MSELoss::apply(&output, &target);
                        losses.push(values(&loss)[0]);
                        if step < 48 {
                            policy.scope(|| {
                                loss.backward();
                                optimizer.step();
                                optimizer.zero_grad();
                            });
                        }
                        assert_eq!(input.dtype(), dtype);
                        assert!(norm.parameters().iter().all(|p| p.dtype() == dtype));
                    }
                    assert!(losses.iter().all(|v| v.is_finite()));
                    assert!(losses[48] < losses[0] * 0.1);
                    assert!(
                        losses[41..].iter().sum::<f32>() < losses[..8].iter().sum::<f32>() * 0.1
                    );
                    println!(
                        "layernorm_backward_training {}",
                        serde_json::json!({"dtype":format!("{dtype:?}"),"adam":adam,"losses":losses})
                    );
                }
            }
        },
    );
}

fn ordered<T: Clone>(a: ArrayD<T>, layout: usize) -> ArrayD<T> {
    match layout {
        0 => a,
        1 => a
            .reversed_axes()
            .as_standard_layout()
            .to_owned()
            .reversed_axes(),
        _ => {
            let last = a.ndim() - 1;
            let mut padded = a.shape().to_vec();
            padded[last] *= 2;
            let mut strided = ArrayD::from_shape_fn(IxDyn(&padded), |i| {
                let mut original = i.slice().to_vec();
                original[last] /= 2;
                a[IxDyn(&original)].clone()
            });
            strided.slice_axis_inplace(Axis(last), Slice::new(0, None, 2));
            strided
        }
    }
}

fn set_values(t: &Tensor, shape: &[usize], values: Vec<f32>, dtype: DType, layout: usize) {
    t.set_array_f32_with_dtype(ArrayD::from_shape_vec(IxDyn(shape), values).unwrap(), dtype);
    match t.native_storage_owned() {
        TensorStorageOwned::F32(a) => {
            t.set_array_f32_with_dtype(ordered(a.to_owned(), layout), dtype)
        }
        TensorStorageOwned::F16(a) => {
            t.set_array_f16_with_dtype(ordered(a.to_owned(), layout), dtype)
        }
        TensorStorageOwned::BF16(a) => {
            t.set_array_bf16_with_dtype(ordered(a.to_owned(), layout), dtype)
        }
        TensorStorageOwned::I8(_, _) => unreachable!(),
    }
}

fn values(t: &Tensor) -> Vec<f32> {
    t.detach().data().iter().copied().collect()
}

fn check_reference(output: &Tensor, input: &Tensor, norm: &LayerNorm) {
    let x = values(input);
    let weight = norm.weight.as_ref().map(values);
    let bias = norm.bias.as_ref().map(values);
    let dim = norm.normalized_shape().iter().product::<usize>();
    let actual = values(output);
    for (row, actual) in x.chunks(dim).zip(actual.chunks(dim)) {
        let mean = row.iter().map(|&v| f64::from(v)).sum::<f64>() / dim as f64;
        let var = row
            .iter()
            .map(|&v| (f64::from(v) - mean).powi(2))
            .sum::<f64>()
            / dim as f64;
        let inv = (var + f64::from(norm.eps())).sqrt().recip();
        for (j, (&x, &y)) in row.iter().zip(actual).enumerate() {
            let expected =
                (f64::from(x) - mean) * inv * weight.as_ref().map_or(1., |w| f64::from(w[j]))
                    + bias.as_ref().map_or(0., |b| f64::from(b[j]));
            assert!(
                y.is_finite() && (f64::from(y) - expected).abs() <= 2e-6 + 2e-6 * expected.abs(),
                "{y} vs {expected}"
            );
        }
    }
}

#[test]
fn cpu_layer_norm_inference_matches_retained_path_and_preserves_storage() {
    let _cpu = crate::ops::cuda::set_enabled_scoped(false);
    let pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(3)
            .build()
            .unwrap(),
    );
    with_precision_config(
        PrecisionConfig {
            parameter_dtype: DType::F32,
            runtime_dtype: DType::F32,
            allow_parameter_dtype_copies: false,
        },
        || {
            let mut cases = Vec::new();
            for (a, input) in [DType::F32, DType::F16, DType::BF16]
                .into_iter()
                .enumerate()
            {
                for (b, parameter) in [DType::F32, DType::F16, DType::BF16]
                    .into_iter()
                    .enumerate()
                {
                    let (shape, normalized) = if a == b {
                        (vec![43, 513], vec![513])
                    } else {
                        (vec![2, 3, 5], vec![3, 5])
                    };
                    cases.push((
                        shape,
                        normalized,
                        input,
                        parameter,
                        (a + b) % 3,
                        true,
                        a != 0 || b != 1,
                        false,
                    ));
                }
            }
            cases.extend([
                (
                    vec![2, 0, 17],
                    vec![17],
                    DType::F16,
                    DType::BF16,
                    2,
                    true,
                    true,
                    false,
                ),
                (
                    vec![4, 1],
                    vec![1],
                    DType::BF16,
                    DType::F32,
                    1,
                    false,
                    false,
                    false,
                ),
                (
                    vec![4, 17],
                    vec![17],
                    DType::F32,
                    DType::BF16,
                    2,
                    true,
                    true,
                    true,
                ),
            ]);
            for (shape, normalized, input_dtype, parameter_dtype, layout, affine, bias, offset) in
                cases
            {
                let input = Tensor::new(ArrayD::zeros(IxDyn(&shape)));
                let data = (0..input.len())
                    .map(|i| {
                        if offset {
                            10000. + (i % 7) as f32 * 0.0009765625
                        } else {
                            ((i * 17 + 3) % 251) as f32 * 0.013 - 1.625
                        }
                    })
                    .collect();
                set_values(&input, &shape, data, input_dtype, layout);
                input.0.borrow_mut().requires_grad = true;
                let norm = LayerNorm::with_options(
                    normalized.clone(),
                    1e-5,
                    affine,
                    bias,
                    parameter_dtype,
                );
                for (i, p) in norm.parameters().iter().enumerate() {
                    set_values(
                        p,
                        &normalized,
                        (0..p.len())
                            .map(|j| ((j * 13 + i * 7) % 97) as f32 * 0.017 - 0.816)
                            .collect(),
                        parameter_dtype,
                        layout,
                    );
                }
                for budget in [1, 3] {
                    let policy = CpuKernelContext::with_pool(pool.clone())
                        .with_task_limit(NonZeroUsize::new(budget).unwrap());
                    let fast = policy.scope(|| no_grad(|| norm.forward(input.clone())));
                    assert_eq!(input.dtype(), input_dtype);
                    assert!(
                        norm.parameters()
                            .iter()
                            .all(|p| p.dtype() == parameter_dtype)
                    );
                    let retained = policy.scope(|| norm.forward(input.clone()));
                    assert!(retained.requires_grad());
                    assert_eq!(input.dtype(), input_dtype);
                    assert!(
                        norm.parameters()
                            .iter()
                            .all(|p| p.dtype() == parameter_dtype)
                    );
                    assert_eq!(fast.dtype(), DType::F32);
                    assert_eq!(fast.shape_vec(), shape);
                    assert_eq!(
                        values(&fast)
                            .iter()
                            .map(|v| v.to_bits())
                            .collect::<Vec<_>>(),
                        values(&retained)
                            .iter()
                            .map(|v| v.to_bits())
                            .collect::<Vec<_>>()
                    );
                    check_reference(&fast, &input, &norm);
                }
            }
            for dtype in [DType::F16, DType::BF16] {
                let input = Tensor::new(ArrayD::zeros(IxDyn(&[17])));
                set_values(
                    &input,
                    &[17],
                    (0..17).map(|i| i as f32 / 11. - 0.7).collect(),
                    dtype,
                    2,
                );
                let mut norm = LayerNorm::with_options(vec![17], 1e-5, true, false, dtype);
                norm.weight = Some(input.clone());
                let fast = no_grad(|| norm.forward(input.clone()));
                assert_eq!(input.dtype(), dtype);
                check_reference(&fast, &input, &norm);
            }
        },
    );
}

#[test]
fn cpu_layer_norm_native_sgd_adam_training_descends_without_storage_promotion() {
    use crate::loss::MSELoss;
    use crate::optim::{Adam, Optimizer, SGD};
    let _cpu = crate::ops::cuda::set_enabled_scoped(false);
    with_precision_config(
        PrecisionConfig {
            parameter_dtype: DType::F32,
            runtime_dtype: DType::F32,
            allow_parameter_dtype_copies: false,
        },
        || {
            for dtype in [DType::F32, DType::F16, DType::BF16] {
                for adam in [false, true] {
                    let input = Tensor::from_array_no_grad(ArrayD::zeros(IxDyn(&[8, 17])));
                    set_values(
                        &input,
                        &[8, 17],
                        (0..136)
                            .map(|i| ((i * 17 + 3) % 251) as f32 * 0.013 - 1.625)
                            .collect(),
                        dtype,
                        1,
                    );
                    let norm = LayerNorm::with_options(vec![17], 1e-5, true, true, dtype);
                    let target = Tensor::from_array_no_grad(ArrayD::zeros(IxDyn(&[8, 17])));
                    let mut optimizer: Box<dyn Optimizer> = if adam {
                        Box::new(Adam::new_with_dtype(norm.parameters(), 0.05, DType::F32))
                    } else {
                        Box::new(SGD::new_with_dtype(norm.parameters(), 1.0, DType::F32))
                    };
                    let mut losses = Vec::new();
                    for step in 0..=48 {
                        let output = norm.forward(input.clone());
                        assert_eq!(input.dtype(), dtype);
                        assert!(norm.parameters().iter().all(|p| p.dtype() == dtype));
                        if matches!(step, 0 | 24 | 48) {
                            check_reference(&output, &input, &norm);
                        }
                        let loss = MSELoss::apply(&output, &target);
                        losses.push(values(&loss)[0]);
                        if step < 48 {
                            loss.backward();
                            optimizer.step();
                            optimizer.zero_grad();
                        }
                        assert!(norm.parameters().iter().all(|p| p.dtype() == dtype));
                    }
                    assert!(losses.iter().all(|v| v.is_finite()));
                    assert!(losses[48] < losses[0] * 0.2);
                    assert!(
                        losses[41..].iter().sum::<f32>() < losses[..8].iter().sum::<f32>() * 0.2
                    );
                    println!(
                        "layernorm_native_training {}",
                        serde_json::json!({"dtype":format!("{dtype:?}"),"adam":adam,"losses":losses})
                    );
                }
            }
        },
    );
}
