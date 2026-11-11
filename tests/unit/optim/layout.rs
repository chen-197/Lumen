use super::*;
use crate::autograd::TensorStorageOwned;
#[cfg(feature = "cuda")]
use crate::device::Device;

fn audit(value: serde_json::Value) {
    if std::env::var_os("LUMEN_LAYOUT_AUDIT").is_some() {
        println!("LAYOUT_AUDIT={value}");
    }
}

fn layout<T: Clone + Default>(values: &[T], order: usize) -> ArrayD<T> {
    match order {
        0 => ArrayD::from_shape_vec(IxDyn(&[2, 3]), values.to_vec()).unwrap(),
        1 => ArrayD::from_shape_fn(IxDyn(&[3, 2]), |i| values[i[1] * 3 + i[0]].clone())
            .reversed_axes(),
        _ => ArrayD::from_shape_fn(IxDyn(&[2, 6]), |i| values[i[0] * 3 + i[1] / 2].clone())
            .slice_move(ndarray::s![.., ..;2])
            .into_dyn(),
    }
}

fn parameter(dtype: DType, order: usize) -> Tensor {
    let t = Tensor::parameter_with_dtype(ArrayD::zeros(IxDyn(&[2, 3])), DType::F32);
    let data = [0.8, -0.4, 0.2, -0.1, 0.6, -0.3];
    match dtype {
        DType::F32 => t.set_array_f32_with_dtype(layout(&data, order), dtype),
        DType::F16 => {
            t.set_array_f16_with_dtype(layout(&data.map(half::f16::from_f32), order), dtype)
        }
        DType::BF16 => {
            t.set_array_bf16_with_dtype(layout(&data.map(half::bf16::from_f32), order), dtype)
        }
        DType::I8 => {
            t.set_array_i8_with_dtype(layout(&[80, -40, 20, -10, 60, -30], order), 0.01, dtype)
        }
    }
    t
}

fn values(t: &Tensor) -> Vec<f32> {
    match t.native_storage_owned() {
        TensorStorageOwned::F32(a) => a.iter().copied().collect(),
        TensorStorageOwned::F16(a) => a.iter().map(|v| v.to_f32()).collect(),
        TensorStorageOwned::BF16(a) => a.iter().map(|v| v.to_f32()).collect(),
        TensorStorageOwned::I8(a, s) => a.iter().map(|v| *v as f32 * s).collect(),
    }
}

fn round(x: f32, dtype: DType) -> f32 {
    match dtype {
        DType::F32 => x,
        DType::F16 => half::f16::from_f32(x).to_f32(),
        DType::BF16 => half::bf16::from_f32(x).to_f32(),
        DType::I8 => (x * 100.0).round().clamp(-127.0, 127.0) * 0.01,
    }
}

#[test]
fn layout_audit_cpu_optimizers_match_logical_coordinates() {
    let _cpu = cuda::set_enabled_scoped(false);
    let mut failures = Vec::new();
    let mut count = 0;
    for dtype in [DType::F32, DType::F16, DType::BF16, DType::I8] {
        for (p_order, g_order) in [(0, 0), (0, 1), (1, 0), (1, 1), (2, 1)] {
            for (kind, state) in [
                ("sgd", DType::F32),
                ("momentum", DType::F32),
                ("adam", DType::F32),
                ("momentum", DType::F16),
                ("adam", DType::F16),
                ("momentum", DType::BF16),
                ("adam", DType::BF16),
            ] {
                count += 1;
                let w = parameter(dtype, p_order);
                let mut expected = values(&w);
                let initial = expected.clone();
                let mut trace = Vec::new();
                let mut m = [0.0; 6];
                let mut v = [0.0; 6];
                let mut opt: Box<dyn Optimizer> = if kind == "adam" {
                    Box::new(Adam::new_with_dtype(vec![w.clone()], 0.01, state))
                } else {
                    Box::new(
                        SGD::new_with_dtype(vec![w.clone()], 0.01, state)
                            .with_momentum(if kind == "momentum" { 0.9 } else { 0.0 }),
                    )
                };
                for step in 1..=5 {
                    opt.zero_grad();
                    let g = [0.3, -0.7, 0.5, -0.2, 0.1, -0.4]
                        .map(|g| if step == 3 { 0.0 } else { g * step as f32 });
                    w.add_grad(layout(&g, g_order));
                    opt.step();
                    for j in 0..6 {
                        let nm = if kind == "adam" {
                            0.9 * m[j] + (1.0 - 0.9) * g[j]
                        } else if kind == "momentum" {
                            0.9 * m[j] + g[j]
                        } else {
                            g[j]
                        };
                        let nv = 0.999 * v[j] + (1.0 - 0.999) * g[j] * g[j];
                        let update = if kind == "adam" {
                            (nm / (1.0 - 0.9f32.powi(step)))
                                / ((nv / (1.0 - 0.999f32.powi(step))).sqrt() + 1e-8)
                        } else {
                            nm
                        };
                        expected[j] = round(expected[j] - 0.01 * update, dtype);
                        m[j] = round(nm, state);
                        v[j] = round(nv, state);
                    }
                    let got = values(&w);
                    trace.push(serde_json::json!({"g":g,"w":got}));
                    if got.iter().zip(&expected).any(|(a, b)| (a - b).abs() > 2e-6)
                        || w.dtype() != dtype
                    {
                        failures.push(format!("{kind} {dtype:?}/{state:?} layouts={p_order}/{g_order} step={step} got={got:?} expected={expected:?} dtype={:?}",w.dtype()));
                        break;
                    }
                }
                audit(
                    serde_json::json!({"case":"optimizer","dtype":format!("{dtype:?}"),
                    "state":format!("{state:?}"),"kind":kind,"orders":[p_order,g_order],
                    "initial":initial,"trace":trace}),
                );
            }
        }
    }
    println!("CPU_LAYOUT_CASES={count} failures={}", failures.len());
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

fn ordered<T: Clone>(shape: &[usize], values: &[T], order: usize) -> ArrayD<T> {
    let a = ArrayD::from_shape_vec(IxDyn(shape), values.to_vec()).unwrap();
    match order {
        0 => a,
        1 => a
            .reversed_axes()
            .as_standard_layout()
            .to_owned()
            .reversed_axes(),
        _ => {
            let mut padded = shape.to_vec();
            let last = shape.len() - 1;
            padded[last] *= 2;
            let mut a = ArrayD::from_shape_fn(IxDyn(&padded), |i| {
                let mut index = 0;
                for axis in 0..shape.len() {
                    index = index * shape[axis] + if axis == last { i[axis] / 2 } else { i[axis] };
                }
                values[index].clone()
            });
            a.slice_axis_inplace(ndarray::Axis(last), ndarray::Slice::new(0, None, 2));
            a
        }
    }
}

fn shaped_parameter(shape: &[usize], data: &[f32], dtype: DType, order: usize) -> Tensor {
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
        DType::I8 => t.set_array_i8_with_dtype(
            ordered(
                shape,
                &data
                    .iter()
                    .map(|v| (*v * 100.0).round() as i8)
                    .collect::<Vec<_>>(),
                order,
            ),
            0.01,
            dtype,
        ),
    }
    t
}

#[test]
fn layout_audit_activation_forward_and_derivatives() {
    use crate::autograd::no_grad;
    use crate::layers::activation::{Gelu, ReLU, SiLU, Sigmoid, Tanh};
    use crate::module::Module;
    let _cpu = cuda::set_enabled_scoped(false);
    type Activation = (&'static str, Box<dyn Module>, fn(f64) -> f64);
    let activations: [Activation; 5] = [
        ("relu", Box::new(ReLU), |x| x.max(0.0)),
        ("sigmoid", Box::new(Sigmoid), |x| 1.0 / (1.0 + (-x).exp())),
        ("tanh", Box::new(Tanh), |x| x.tanh()),
        ("silu", Box::new(SiLU), |x| x / (1.0 + (-x).exp())),
        ("gelu", Box::new(Gelu), |x| {
            0.5 * x
                * (1.0 + ((2.0 / std::f64::consts::PI).sqrt() * (x + 0.044715 * x * x * x)).tanh())
        }),
    ];
    let mut failures = Vec::new();
    for (name, module, f) in activations {
        for order in [0, 1, 2] {
            for g_order in [0, 1, 2] {
                let t = parameter(DType::F32, order);
                let x = values(&t);
                let out = module.forward(t.clone());
                let inference = no_grad(|| module.forward(t.clone()));
                let expected: Vec<_> = x.iter().map(|x| f(*x as f64) as f32).collect();
                let g = [0.3, -0.7, 0.5, -0.2, 0.1, -0.4];
                let grad = layout(&g, g_order);
                let backward = out.0.borrow().backward_op.clone().unwrap();
                backward(&grad.view());
                let dx = t.grad().unwrap().iter().copied().collect::<Vec<_>>();
                let expected_dx: Vec<_> = x
                    .iter()
                    .zip(g)
                    .map(|(x, g)| {
                        let x = *x as f64;
                        (g as f64 * (f(x + 1e-5) - f(x - 1e-5)) / 2e-5) as f32
                    })
                    .collect();
                for (label, got, expected) in [
                    ("training forward", values(&out), &expected),
                    ("inference forward", values(&inference), &expected),
                    ("backward", dx.clone(), &expected_dx),
                ] {
                    if got.iter().zip(expected).any(|(a, b)| (a - b).abs() > 2e-6) {
                        failures.push(format!("{name} {label} layouts={order}/{g_order} got={got:?} expected={expected:?}"));
                    }
                }
                audit(
                    serde_json::json!({"case":"activation","kind":name,"orders":[order,g_order],
                    "x":x,"g":g,"out":values(&out),"inference":values(&inference),"dx":dx}),
                );
            }
        }
    }
    println!("ACTIVATION_LAYOUT_CASES=45 failures={}", failures.len());
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn layout_audit_matmul_forward_and_backward() {
    use crate::ops::matmul::{batch_matmul, matmul_transposed_rhs};
    let _cpu = cuda::set_enabled_scoped(false);
    let mut failures = Vec::new();
    for lhs_dtype in [DType::F32, DType::F16, DType::BF16, DType::I8] {
        for rhs_dtype in [DType::F32, DType::F16, DType::BF16, DType::I8] {
            for (a_order, b_order, g_order) in
                [(0, 0, 0), (1, 0, 1), (0, 1, 0), (1, 1, 1), (2, 1, 2)]
            {
                for batch in [false, true] {
                    let batches = if batch { 2 } else { 1 };
                    let a_shape = if batch { vec![1, 2, 2, 3] } else { vec![2, 3] };
                    let b_shape = if batch { vec![1, 2, 3, 2] } else { vec![2, 3] };
                    let g_shape = if batch { vec![1, 2, 2, 2] } else { vec![2, 2] };
                    let av: Vec<_> = (0..batches * 6)
                        .map(|i| [0.8, -0.4, 0.2, -0.1, 0.6, -0.3][i % 6] + i as f32 * 0.01)
                        .collect();
                    let bv: Vec<_> = (0..batches * 6)
                        .map(|i| [0.25, -0.5, 0.75, -0.375, 0.625, -0.875][i % 6] - i as f32 * 0.01)
                        .collect();
                    let g: Vec<_> = (0..batches * 4)
                        .map(|i| (i + 1) as f32 * if i % 2 == 0 { 0.2 } else { -0.2 })
                        .collect();
                    let a = shaped_parameter(&a_shape, &av, lhs_dtype, a_order);
                    let b = shaped_parameter(&b_shape, &bv, rhs_dtype, b_order);
                    let av = values(&a);
                    let bv = values(&b);
                    let y = if batch {
                        batch_matmul(&a, &b)
                    } else {
                        matmul_transposed_rhs(&a, &b)
                    };
                    let backward = y.0.borrow().backward_op.clone().unwrap();
                    if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        backward(&ordered(&g_shape, &g, g_order).view());
                    }))
                    .is_err()
                    {
                        failures.push(format!("backward panicked {lhs_dtype:?}/{rhs_dtype:?} batch={batch} layouts={a_order}/{b_order}/{g_order}"));
                        continue;
                    }
                    let mut ey = vec![0.0f64; batches * 4];
                    let mut ea = vec![0.0f64; batches * 6];
                    let mut eb = vec![0.0f64; batches * 6];
                    for q in 0..batches {
                        for row in 0..2 {
                            for col in 0..2 {
                                for k in 0..3 {
                                    let ai = q * 6 + row * 3 + k;
                                    let bi = q * 6 + if batch { k * 2 + col } else { col * 3 + k };
                                    let yi = q * 4 + row * 2 + col;
                                    ey[yi] += av[ai] as f64 * bv[bi] as f64;
                                    ea[ai] += g[yi] as f64 * bv[bi] as f64;
                                    eb[bi] += g[yi] as f64 * av[ai] as f64;
                                }
                            }
                        }
                    }
                    let da = a.grad().unwrap().iter().copied().collect::<Vec<_>>();
                    let db = b.grad().unwrap().iter().copied().collect::<Vec<_>>();
                    for (label, got, expected) in [
                        ("forward", values(&y), ey),
                        ("dA", da.clone(), ea),
                        ("dB", db.clone(), eb),
                    ] {
                        if got
                            .iter()
                            .zip(&expected)
                            .any(|(a, b)| (*a as f64 - b).abs() > 2e-5)
                        {
                            failures.push(format!("{label} {lhs_dtype:?}/{rhs_dtype:?} batch={batch} layouts={a_order}/{b_order}/{g_order} got={got:?} expected={expected:?}"));
                        }
                    }
                    audit(
                        serde_json::json!({"case":"matmul_transposed_rhs","batch":batch,"dtype_a":format!("{lhs_dtype:?}"),
                        "dtype_b":format!("{rhs_dtype:?}"),"orders":[a_order,b_order,g_order],
                        "a":av,"b":bv,"g":g,"out":values(&y),"da":da,"db":db}),
                    );
                }
            }
        }
    }
    println!("MATMUL_LAYOUT_CASES=160 failures={}", failures.len());
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn layout_audit_i8_slice_setters_follow_logical_order() {
    let _cpu = cuda::set_enabled_scoped(false);
    let mut failures = Vec::new();
    for order in [0, 1, 2] {
        for source in 0..4 {
            let t = parameter(DType::I8, order);
            let expected = [0.11, -0.22, 0.33, -0.44, 0.55, -0.66];
            let q = ParameterQuantization::Int8.with_scale(0.01);
            match source {
                0 => t.set_f32_slice_with_quantization(&[2, 3], &expected, q),
                1 => t.set_f16_slice_with_quantization(
                    &[2, 3],
                    &expected.map(half::f16::from_f32),
                    q,
                ),
                2 => t.set_bf16_slice_with_quantization(
                    &[2, 3],
                    &expected.map(half::bf16::from_f32),
                    q,
                ),
                _ => t.set_i8_slice_with_dtype(
                    &[2, 3],
                    &[11, -22, 33, -44, 55, -66],
                    0.01,
                    DType::I8,
                ),
            }
            let got = values(&t);
            if got.iter().zip(expected).any(|(a, b)| (a - b).abs() > 1e-6) {
                failures.push(format!("order={order} source={source} got={got:?}"));
            }
        }
    }
    println!("I8_SLICE_CASES=12 failures={}", failures.len());
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[cfg(feature = "cuda")]
#[test]
fn layout_audit_cuda_native_upload_follows_logical_order() {
    assert!(cuda::is_available());
    let _cuda = cuda::set_enabled_scoped(true);
    let mut failures = Vec::new();
    for dtype in [DType::F16, DType::BF16, DType::I8] {
        for order in [0, 1, 2] {
            for lazy in [false, true] {
                let t = parameter(dtype, order);
                let expected = values(&t);
                if lazy {
                    t.set_device_inplace(Device::Cuda);
                } else {
                    t.to_device_inplace(Device::Cuda);
                }
                let native = t.cloned_cuda_native_lowp_buffer();
                let got = native.map(|(_, b, scale)| {
                    if dtype == DType::I8 {
                        cuda::download_i8_storage(&b)
                            .unwrap()
                            .into_iter()
                            .map(|x| x as f32 * scale.unwrap())
                            .collect::<Vec<_>>()
                    } else {
                        cuda::download_u16_storage(&b)
                            .unwrap()
                            .into_iter()
                            .map(|x| {
                                if dtype == DType::F16 {
                                    half::f16::from_bits(x).to_f32()
                                } else {
                                    half::bf16::from_bits(x).to_f32()
                                }
                            })
                            .collect()
                    }
                });
                if got.as_ref() != Some(&expected) {
                    failures.push(format!(
                        "{dtype:?} order={order} lazy={lazy} got={got:?} expected={expected:?}"
                    ));
                }
            }
        }
    }
    println!("CUDA_NATIVE_LAYOUT_CASES=18 failures={}", failures.len());
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn layout_audit_real_backward_training_descends() {
    use crate::layers::activation::SiLU;
    use crate::loss::MSELoss;
    use crate::module::Module;
    use crate::ops::matmul::matmul_transposed_rhs;
    let _cpu = cuda::set_enabled_scoped(false);
    for (dtype, state) in [
        (DType::F32, DType::F32),
        (DType::F32, DType::F16),
        (DType::F32, DType::BF16),
        (DType::F16, DType::F32),
        (DType::BF16, DType::F32),
        (DType::I8, DType::F32),
    ] {
        for adam in [false, true] {
            for order in [0, 1, 2] {
                let x = parameter(dtype, order);
                x.0.borrow_mut().requires_grad = false;
                let w = parameter(dtype, order);
                let initial = values(&w);
                let target = Tensor::from_array_no_grad(
                    ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![0.25, -0.15, 0.15, -0.1]).unwrap(),
                );
                let mut opt: Box<dyn Optimizer> = if adam {
                    let opt = Adam::new_with_dtype(vec![w.clone()], 0.03, state);
                    Box::new(if dtype == DType::F32 {
                        opt
                    } else {
                        opt.with_fp32_master_weights()
                    })
                } else {
                    let opt = SGD::new_with_dtype(vec![w.clone()], 0.05, state).with_momentum(0.8);
                    Box::new(if dtype == DType::F32 {
                        opt
                    } else {
                        opt.with_fp32_master_weights()
                    })
                };
                let mut losses = Vec::new();
                for step in 0..=80 {
                    opt.zero_grad();
                    x.zero_grad();
                    let y = SiLU.forward(matmul_transposed_rhs(&x, &w));
                    let loss = MSELoss::apply(&y, &target);
                    losses.push(values(&loss)[0] as f64);
                    if step < 80 {
                        loss.backward();
                        opt.step();
                    }
                    assert_eq!(w.dtype(), dtype);
                }
                let mean = losses.iter().sum::<f64>() / losses.len() as f64;
                let slope: f64 = losses
                    .iter()
                    .enumerate()
                    .map(|(i, y)| (i as f64 - 40.0) * (y - mean))
                    .sum();
                assert!(
                    losses.iter().all(|x| x.is_finite())
                        && slope < 0.0
                        && losses[80] < losses[0] * 0.4
                        && losses[71..].iter().sum::<f64>() < losses[..10].iter().sum::<f64>(),
                    "{dtype:?}/{state:?} adam={adam} order={order} {losses:?}"
                );
                audit(
                    serde_json::json!({"case":"training","dtype":format!("{dtype:?}"),
                    "state":format!("{state:?}"),"adam":adam,"order":order,
                    "x":values(&x),"initial":initial,"target":values(&target),"losses":losses}),
                );
            }
        }
    }
    println!("LAYOUT_TRAINING_CURVES=36 points=81");
}

#[test]
fn layout_audit_rank3_matmul_flattens_logical_rows() {
    use crate::ops::matmul::matmul_transposed_rhs;
    let _cpu = cuda::set_enabled_scoped(false);
    let mut failures = Vec::new();
    for ad in [DType::F32, DType::F16, DType::BF16, DType::I8] {
        for bd in [DType::F32, DType::F16, DType::BF16, DType::I8] {
            for order in [0, 1, 2] {
                let av: Vec<_> = (0..12).map(|i| (i as f32 - 5.0) * 0.125).collect();
                let bv = [0.25, -0.5, 0.75, -0.375, 0.625, -0.875];
                let g: Vec<_> = (0..8).map(|i| (i as f32 - 3.0) * 0.2).collect();
                let a = shaped_parameter(&[2, 2, 3], &av, ad, order);
                let b = shaped_parameter(&[2, 3], &bv, bd, order);
                let av = values(&a);
                let bv = values(&b);
                let y = matmul_transposed_rhs(&a, &b);
                assert_eq!(y.shape_vec(), [2, 2, 2]);
                let backward = y.0.borrow().backward_op.clone().unwrap();
                backward(&ordered(&[2, 2, 2], &g, order).view());
                let da = a.grad().unwrap().iter().copied().collect::<Vec<_>>();
                let db = b.grad().unwrap().iter().copied().collect::<Vec<_>>();
                let mut ey = vec![0.0f64; 8];
                let mut ea = vec![0.0f64; 12];
                let mut eb = vec![0.0f64; 6];
                for row in 0..4 {
                    for col in 0..2 {
                        for k in 0..3 {
                            let ai = row * 3 + k;
                            let bi = col * 3 + k;
                            let yi = row * 2 + col;
                            ey[yi] += av[ai] as f64 * bv[bi] as f64;
                            ea[ai] += g[yi] as f64 * bv[bi] as f64;
                            eb[bi] += g[yi] as f64 * av[ai] as f64;
                        }
                    }
                }
                for (label, got, expected) in [
                    ("forward", values(&y), ey),
                    ("dA", da.clone(), ea),
                    ("dB", db.clone(), eb),
                ] {
                    if got
                        .iter()
                        .zip(expected)
                        .any(|(a, b)| (*a as f64 - b).abs() > 2e-5)
                    {
                        failures.push(format!(
                            "{label} rank3 {ad:?}/{bd:?} order={order} got={got:?}"
                        ));
                    }
                }
                audit(
                    serde_json::json!({"case":"matmul_transposed_rhs","batch":false,"rank3":true,
                    "dtype_a":format!("{ad:?}"),"dtype_b":format!("{bd:?}"),"order":order,
                    "a":av,"b":bv,"g":g,"out":values(&y),"da":da,"db":db}),
                );
            }
        }
    }
    println!("RANK3_MATMUL_LAYOUT_CASES=48 failures={}", failures.len());
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
