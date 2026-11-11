use super::*;
use crate::accuracy_gate::{check, ordered, tensor, values};
use crate::autograd::no_grad;
use crate::device::{Device, set_strict_device_execution_scoped};
use crate::precision::{PrecisionConfig, with_precision_config};

fn check_standard_products(device: Device) {
    let _strict = set_strict_device_execution_scoped(true);
    with_precision_config(
        PrecisionConfig {
            parameter_dtype: DType::F32,
            runtime_dtype: DType::F32,
            allow_parameter_dtype_copies: false,
        },
        || {
            let mut cases = 0;
            for (shape, n) in [
                (vec![5], 3),
                (vec![2, 5], 3),
                (vec![2, 3, 5], 4),
                (vec![0, 5], 3),
                (vec![2, 0], 3),
                (vec![2, 5], 0),
            ] {
                let k = *shape.last().unwrap();
                let m = shape[..shape.len() - 1].iter().product::<usize>();
                let b_shape = [k, n];
                let mut output_shape = shape.clone();
                *output_shape.last_mut().unwrap() = n;
                for ad in [DType::F32, DType::F16, DType::BF16, DType::I8] {
                    for bd in [DType::F32, DType::F16, DType::BF16, DType::I8] {
                        for order in 0..3 {
                            for flags in
                                [(true, true), (true, false), (false, true), (false, false)]
                            {
                                let make_values = |len, seed| {
                                    (0..len)
                                        .map(|i| ((i * 7 + seed) % 23) as f32 / 32. - 0.25)
                                        .collect::<Vec<_>>()
                                };
                                let a = tensor(&shape, &make_values(m * k, 3), ad, order, device);
                                let b =
                                    tensor(&b_shape, &make_values(k * n, 11), bd, order, device);
                                a.0.borrow_mut().requires_grad = flags.0;
                                b.0.borrow_mut().requires_grad = flags.1;
                                let av = values(&a);
                                let bv = values(&b);
                                let before = (
                                    a.dtype(),
                                    b.dtype(),
                                    a.has_host_f32_data(),
                                    b.has_host_f32_data(),
                                );
                                let expected = (0..m * n)
                                    .map(|i| {
                                        (0..k)
                                            .map(|j| {
                                                av[i / n * k + j] as f64 * bv[j * n + i % n] as f64
                                            })
                                            .sum::<f64>()
                                    })
                                    .collect::<Vec<_>>();
                                let g = make_values(m * n, 17);
                                let da = (0..m * k)
                                    .map(|i| {
                                        (0..n)
                                            .map(|j| {
                                                g[i / k * n + j] as f64 * bv[i % k * n + j] as f64
                                            })
                                            .sum::<f64>()
                                    })
                                    .collect::<Vec<_>>();
                                let db = (0..k * n)
                                    .map(|i| {
                                        (0..m)
                                            .map(|j| {
                                                av[j * k + i / n] as f64 * g[j * n + i % n] as f64
                                            })
                                            .sum::<f64>()
                                    })
                                    .collect::<Vec<_>>();
                                let out = matmul(&a, &b);
                                assert_eq!(out.shape_vec(), output_shape);
                                assert_eq!(out.device(), device);
                                assert_eq!(out.requires_grad(), flags.0 || flags.1);
                                let budget = if flags.0 || flags.1 || ad != bd {
                                    2e-5
                                } else {
                                    match ad {
                                        DType::F32 => 2e-5,
                                        DType::F16 => 1e-3,
                                        DType::BF16 => 8e-3,
                                        DType::I8 => 0.02,
                                    }
                                };
                                check(
                                    "standard matmul forward",
                                    &values(&out),
                                    &expected,
                                    budget,
                                    budget,
                                );
                                if flags.0 || flags.1 {
                                    for count in 1..=2 {
                                        out.backward_with_grad(ordered(&output_shape, &g, order));
                                        for (parent, required, reference) in
                                            [(&a, flags.0, &da), (&b, flags.1, &db)]
                                        {
                                            if required {
                                                let grad = parent.grad().unwrap();
                                                assert_eq!(grad.shape(), parent.shape_vec());
                                                check(
                                                    "standard matmul VJP",
                                                    &grad.iter().copied().collect::<Vec<_>>(),
                                                    &reference
                                                        .iter()
                                                        .map(|v| v * count as f64)
                                                        .collect::<Vec<_>>(),
                                                    2e-5,
                                                    2e-5,
                                                );
                                            } else {
                                                assert!(parent.grad().is_none());
                                            }
                                        }
                                    }
                                }
                                let inference = no_grad(|| matmul(&a, &b));
                                assert!(!inference.requires_grad());
                                assert_eq!(
                                    inference.dtype(),
                                    if ad == bd { ad } else { DType::F32 }
                                );
                                let (atol, rtol) = match inference.native_storage_owned() {
                                    TensorStorageOwned::I8(_, scale) => {
                                        (scale as f64 * 0.55 + 2e-5, 0.)
                                    }
                                    _ => (
                                        2e-5,
                                        match inference.dtype() {
                                            DType::F16 => 1e-3,
                                            DType::BF16 => 8e-3,
                                            _ => 2e-5,
                                        },
                                    ),
                                };
                                check(
                                    "standard matmul inference",
                                    &values(&inference),
                                    &expected,
                                    atol,
                                    rtol,
                                );
                                assert_eq!(
                                    (
                                        a.dtype(),
                                        b.dtype(),
                                        a.has_host_f32_data(),
                                        b.has_host_f32_data()
                                    ),
                                    before
                                );
                                cases += 1;
                            }
                        }
                    }
                }
            }
            assert_eq!(cases, 1152);
            println!("STANDARD_MATMUL_CHECK device={device:?} cases={cases} passed");
        },
    );
}

#[test]
fn standard_matmul_cpu_dtype_layout_shape_vjp_and_empty_axes() {
    check_standard_products(Device::Cpu);
}

#[cfg(feature = "cuda")]
#[test]
fn standard_matmul_cuda_dtype_layout_shape_vjp_and_empty_axes() {
    if !cuda::is_available() {
        println!("SKIP standard matmul CUDA: no device");
        return;
    }
    cuda::set_enabled(true);
    check_standard_products(Device::Cuda);
}

#[test]
fn square_matrices_distinguish_standard_and_transposed_rhs_and_accumulate_alias_gradients() {
    let a = Tensor::new(ndarray::arr2(&[[1., 2.], [3., 4.]]).into_dyn());
    let b = Tensor::new(ndarray::arr2(&[[5., 6.], [7., 8.]]).into_dyn());
    assert_eq!(values(&matmul(&a, &b)), vec![19., 22., 43., 50.]);
    assert_eq!(
        values(&matmul_transposed_rhs(&a, &b)),
        vec![17., 23., 39., 53.]
    );
    matmul(&a, &a).sum().backward();
    assert_eq!(
        a.grad().unwrap().iter().copied().collect::<Vec<_>>(),
        vec![7., 11., 9., 13.]
    );
}

#[test]
#[should_panic(expected = "MatMul shape mismatch")]
fn standard_matmul_rejects_weight_layout_without_guessing() {
    matmul(
        &Tensor::new(ndarray::ArrayD::ones(ndarray::IxDyn(&[2, 3]))),
        &Tensor::new(ndarray::ArrayD::ones(ndarray::IxDyn(&[4, 3]))),
    );
}

#[test]
fn standard_native_panels_match_f64_with_mixed_dtypes_strides_and_odd_tails() {
    for (m, k, n) in [(9, 129, 67), (17, 513, 65)] {
        let x = (0..m * k)
            .map(|i| ((i * 37 % 257) as f32 - 128.) * 0.00317)
            .collect::<Vec<_>>();
        let w = (0..k * n)
            .map(|i| ((i * 19 % 251) as f32 - 125.) * 0.00371)
            .collect::<Vec<_>>();
        for ad in [DType::F32, DType::F16, DType::BF16, DType::I8] {
            for bd in [DType::F16, DType::BF16, DType::I8] {
                for order in 0..3 {
                    let a = tensor(&[m, k], &x, ad, order, Device::Cpu);
                    let b = tensor(&[k, n], &w, bd, order, Device::Cpu);
                    let before = (
                        a.dtype(),
                        b.dtype(),
                        a.has_host_f32_data(),
                        b.has_host_f32_data(),
                    );
                    let av = values(&a);
                    let bv = values(&b);
                    let expected = (0..m * n)
                        .map(|i| {
                            (0..k)
                                .map(|kk| av[i / n * k + kk] as f64 * bv[kk * n + i % n] as f64)
                                .sum()
                        })
                        .collect::<Vec<_>>();
                    let available = match bd {
                        DType::F16 => cpu_gemm::available(SliceRef::F16(&[])),
                        DType::BF16 => {
                            cpu_gemm::available(SliceRef::BF16(&[]))
                                || (ad == DType::BF16
                                    && crate::arch::x86_avx512_bf16_kernel_runtime_available())
                        }
                        DType::I8 => cpu_gemm::available(SliceRef::I8(&[], 1.)),
                        _ => unreachable!(),
                    };
                    for tasks in [1, 4] {
                        crate::cpu::CpuKernelContext::default()
                            .with_task_limit(std::num::NonZeroUsize::new(tasks).unwrap())
                            .scope(|| {
                                let fast = standard_cpu::try_forward(&a, &b, m, k, n);
                                assert_eq!(fast.is_some(), available);
                                if let Some(fast) = fast {
                                    check(
                                        "standard native panel",
                                        &fast.iter().copied().collect::<Vec<_>>(),
                                        &expected,
                                        2e-5,
                                        2e-5,
                                    );
                                }
                            });
                    }
                    let out = matmul(&a, &b);
                    check("standard panel graph", &values(&out), &expected, 2e-5, 2e-5);
                    assert_eq!(
                        (
                            a.dtype(),
                            b.dtype(),
                            a.has_host_f32_data(),
                            b.has_host_f32_data()
                        ),
                        before
                    );
                }
            }
        }
    }
}

#[test]
fn matrix_product_backward_omits_frozen_side_and_keeps_parent_shapes() {
    for layout in [layout::RhsLayout::Standard, layout::RhsLayout::Transposed] {
        for (need_a, need_b) in [(true, false), (false, true), (false, false)] {
            let a = Tensor::new(ndarray::ArrayD::ones(ndarray::IxDyn(&[2, 3, 5])));
            let b_shape = if matches!(layout, layout::RhsLayout::Standard) {
                [5, 7]
            } else {
                [7, 5]
            };
            let b = Tensor::new(ndarray::ArrayD::ones(ndarray::IxDyn(&b_shape)));
            a.0.borrow_mut().requires_grad = need_a;
            b.0.borrow_mut().requires_grad = need_b;
            let g = ndarray::ArrayD::ones(ndarray::IxDyn(&[2, 3, 7]));
            let (da, db) = layout.backward_cpu(&a, &b, &g.view(), 6, 5, 7);
            assert_eq!(da.is_some(), need_a);
            assert_eq!(db.is_some(), need_b);
            if let Some(da) = da {
                assert_eq!(da.shape(), a.shape_vec());
                assert!(da.iter().all(|&v| v == 7.));
            }
            if let Some(db) = db {
                assert_eq!(db.shape(), b_shape);
                assert!(db.iter().all(|&v| v == 6.));
            }
        }
    }
}

#[test]
fn small_standard_products_preserve_precision_layouts_vjp_and_task_budgets() {
    for (shape, n) in [
        (vec![129], 67),
        (vec![3, 33], 31),
        (vec![1, 7, 513], 129),
        (vec![2, 7], 9),
    ] {
        let k = *shape.last().unwrap();
        let m = shape[..shape.len() - 1].iter().product::<usize>();
        let x = (0..m * k)
            .map(|i| ((i * 37 % 127) as f32 - 63.) * 0.00317)
            .collect::<Vec<_>>();
        let w = (0..k * n)
            .map(|i| ((i * 19 % 131) as f32 - 65.) * 0.00371)
            .collect::<Vec<_>>();
        let g = (0..m * n)
            .map(|i| ((i * 13 % 29) as f32 - 14.) * 0.0117)
            .collect::<Vec<_>>();
        let mut output_shape = shape.clone();
        *output_shape.last_mut().unwrap() = n;
        for ad in [DType::F32, DType::F16, DType::BF16, DType::I8] {
            for bd in [DType::F32, DType::F16, DType::BF16, DType::I8] {
                for order in 0..3 {
                    let a = tensor(&shape, &x, ad, order, Device::Cpu);
                    let b = tensor(&[k, n], &w, bd, order, Device::Cpu);
                    let before = (
                        a.dtype(),
                        b.dtype(),
                        a.has_host_f32_data(),
                        b.has_host_f32_data(),
                    );
                    let av = values(&a);
                    let bv = values(&b);
                    let y = (0..m * n)
                        .map(|i| {
                            (0..k)
                                .map(|j| av[i / n * k + j] as f64 * bv[j * n + i % n] as f64)
                                .sum()
                        })
                        .collect::<Vec<_>>();
                    let da = (0..m * k)
                        .map(|i| {
                            (0..n)
                                .map(|j| g[i / k * n + j] as f64 * bv[i % k * n + j] as f64)
                                .sum()
                        })
                        .collect::<Vec<_>>();
                    let db = (0..k * n)
                        .map(|i| {
                            (0..m)
                                .map(|j| av[j * k + i / n] as f64 * g[j * n + i % n] as f64)
                                .sum()
                        })
                        .collect::<Vec<_>>();
                    let route = match bd {
                        DType::F32 => SliceRef::F32(&[]),
                        DType::F16 => SliceRef::F16(&[]),
                        DType::BF16 => SliceRef::BF16(&[]),
                        DType::I8 => SliceRef::I8(&[], 1.),
                    };
                    let fast = standard_cpu::try_forward(&a, &b, m, k, n);
                    assert_eq!(
                        fast.is_some(),
                        order == 0
                            && k >= 8
                            && n >= 16
                            && !(bd == DType::F32 && m > 1 && n < 256)
                            && cpu_gemm::small::available(route)
                    );
                    if let Some(fast) = fast {
                        check(
                            "unpacked small product",
                            &fast.iter().copied().collect::<Vec<_>>(),
                            &y,
                            2e-5,
                            2e-5,
                        );
                    }
                    let mut actual = Vec::new();
                    for tasks in [1, 4] {
                        crate::cpu::CpuKernelContext::default()
                            .with_task_limit(std::num::NonZeroUsize::new(tasks).unwrap())
                            .scope(|| {
                                a.zero_grad();
                                b.zero_grad();
                                let out = matmul(&a, &b);
                                assert_eq!(out.shape_vec(), output_shape);
                                actual = values(&out);
                                check("small product forward", &actual, &y, 2e-5, 2e-5);
                                out.backward_with_grad(ordered(&output_shape, &g, order));
                                check(
                                    "small product input gradient",
                                    &a.grad().unwrap().iter().copied().collect::<Vec<_>>(),
                                    &da,
                                    2e-5,
                                    2e-5,
                                );
                                check(
                                    "small product RHS gradient",
                                    &b.grad().unwrap().iter().copied().collect::<Vec<_>>(),
                                    &db,
                                    2e-5,
                                    2e-5,
                                );
                            });
                    }
                    let infer = no_grad(|| matmul(&a, &b));
                    let scale = match infer.native_storage_owned() {
                        TensorStorageOwned::I8(_, s) => Some(s),
                        _ => None,
                    };
                    let infer_values = values(&infer);
                    let (atol, rtol) = match infer.dtype() {
                        DType::F32 => (2e-5, 2e-5),
                        DType::F16 => (2e-5, 1. / 1024.),
                        DType::BF16 => (2e-5, 1. / 128.),
                        DType::I8 => (scale.unwrap() as f64 * 0.5001 + 2e-5, 2e-5),
                    };
                    check("small product inference", &infer_values, &y, atol, rtol);
                    assert_eq!(
                        (
                            a.dtype(),
                            b.dtype(),
                            a.has_host_f32_data(),
                            b.has_host_f32_data()
                        ),
                        before
                    );
                    if shape.len() == 1
                        && order == 0
                        && std::env::var_os("LUMEN_OPERATOR_AUDIT").is_some()
                    {
                        println!(
                            "SMALL_MATMUL_AUDIT={}",
                            serde_json::json!({"input_dtype":format!("{ad:?}"),"rhs_dtype":format!("{bd:?}"),"x":{"shape":shape,"values":av,"grad":a.grad().unwrap().iter().copied().collect::<Vec<_>>()},"w":{"shape":[k,n],"values":bv,"grad":b.grad().unwrap().iter().copied().collect::<Vec<_>>()},"y":actual,"dy":g,"output_shape":output_shape,"infer":infer_values,"infer_dtype":format!("{:?}",infer.dtype()),"infer_scale":scale})
                        );
                    }
                }
            }
        }
    }
}
