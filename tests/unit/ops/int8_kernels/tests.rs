use super::*;

#[cfg(all(
    feature = "x86-int8-kernels",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[test]
fn vector_quantization_boundary_matches_scalar_across_backends() {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    let mut values = Vec::new();
    for i in -256..=256 {
        let half = i as f32 * 0.5;
        values.extend([
            half,
            f32::from_bits(half.to_bits().wrapping_sub(1)),
            f32::from_bits(half.to_bits().wrapping_add(1)),
        ]);
    }
    let mut bits = 17u32;
    for _ in 0..4096 {
        bits = bits.wrapping_mul(1664525).wrapping_add(1013904223);
        values.push(f32::from_bits(bits));
    }
    values.extend([
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::MAX,
        -f32::MAX,
    ]);
    while values.len() % 16 != 0 {
        values.push(0.0);
    }
    let expected = values
        .iter()
        .map(|v| v.round().clamp(-127.0, 127.0) as i8)
        .collect::<Vec<_>>();
    #[target_feature(enable = "avx2")]
    unsafe fn avx2(values: &[f32], expected: &[i8]) {
        for (input, expected) in values.chunks_exact(8).zip(expected.chunks_exact(8)) {
            let packed = unsafe { requantize_i8_x86_avx2(_mm256_loadu_ps(input.as_ptr())) };
            let mut got = [0i32; 8];
            unsafe { _mm256_storeu_si256(got.as_mut_ptr().cast(), packed) };
            assert!(got.iter().zip(expected).all(|(&a, &b)| a == b as i32));
        }
    }
    #[target_feature(enable = "avx,avx512f,avx512bw")]
    unsafe fn avx512(values: &[f32], expected: &[i8]) {
        for (input, expected) in values.chunks_exact(16).zip(expected.chunks_exact(16)) {
            let packed = unsafe { requantize_i8_x86_avx512(_mm512_loadu_ps(input.as_ptr())) };
            let mut got = [0i32; 16];
            unsafe { _mm512_storeu_si512(got.as_mut_ptr().cast(), packed) };
            assert!(got.iter().zip(expected).all(|(&a, &b)| a == b as i32));
        }
    }
    if arch::x86_i8_kernel_runtime_available() {
        unsafe { avx2(&values, &expected) };
        println!(
            "REQUANTIZATION_BOUNDARY_BACKEND=avx2 values={}",
            values.len()
        );
    }
    if arch::x86_avx512_i8_kernel_runtime_available() {
        unsafe { avx512(&values, &expected) };
        println!(
            "REQUANTIZATION_BOUNDARY_BACKEND=avx512 values={}",
            values.len()
        );
    }
}

#[test]
fn optimizer_quantization_boundary_matches_scalar_rounding_and_saturation() {
    let values = [
        0.5f32,
        -0.5,
        1.5,
        -1.5,
        126.5,
        -126.5,
        1e30,
        -1e30,
        f32::from_bits(0.5f32.to_bits() - 1),
        -f32::from_bits(0.5f32.to_bits() - 1),
    ];
    let grad = (0..67)
        .map(|i| -values[i % values.len()])
        .collect::<Vec<_>>();
    let expected = grad
        .iter()
        .map(|&g| (-g).round().clamp(-127.0, 127.0) as i8)
        .collect::<Vec<_>>();
    let mut weights = vec![0; grad.len()];
    if sgd_update_i8_f32_arch(&mut weights, 1.0, &grad, 1.0) {
        assert_eq!(weights, expected);
        weights.fill(0);
        let mut velocity = vec![0.0; grad.len()];
        assert!(sgd_momentum_update_i8_f32_arch(
            &mut weights,
            1.0,
            &mut velocity,
            &grad,
            1.0,
            0.0
        ));
        assert_eq!(weights, expected);
        assert_eq!(velocity, grad);
        for lr in [0.5, 1.5, 1e30] {
            weights.fill(0);
            let grad = (0..67)
                .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
                .collect::<Vec<_>>();
            let mut m = vec![0.0; 67];
            let mut v = m.clone();
            assert!(adam_update_i8_f32_arch(
                &mut weights,
                1.0,
                &mut m,
                &mut v,
                &grad,
                lr,
                0.0,
                0.0,
                1.0,
                1.0,
                0.0
            ));
            let expected = grad
                .iter()
                .map(|&g| (-lr * g).round().clamp(-127.0, 127.0) as i8)
                .collect::<Vec<_>>();
            assert_eq!(weights, expected);
        }
    }
}

fn patterned_i8(i: i32, mul: i32, add: i32) -> i8 {
    (((i * mul + add) % 255) - 127).clamp(-127, 127) as i8
}

#[test]
fn backend_name_matches_backend_enum() {
    let name = active_int8_backend_name();
    match active_int8_backend() {
        Int8KernelBackend::Portable => assert_eq!(name, "portable"),
        Int8KernelBackend::Arm64Neon => assert_eq!(name, "arm64-neon"),
        Int8KernelBackend::X86Avx512 => assert_eq!(name, "x86-avx512bw"),
        Int8KernelBackend::X86Avx2 => assert_eq!(name, "x86-avx2"),
    }
}

#[test]
fn dynamic_i8_scale_never_underflows_to_zero() {
    assert_eq!(dynamic_i8_scale(0.0), 1.0);
    assert_eq!(dynamic_i8_scale(f32::from_bits(1)), f32::MIN_POSITIVE);
    assert_eq!(dynamic_i8_scale(f32::INFINITY), 1.0);
}

#[test]
fn i8_i8_backend_name_matches_runtime_path() {
    let backend = active_i8_i8_backend();
    let name = active_i8_i8_backend_name();

    if arch::x86_avx512_i8_i8_kernel_runtime_available() {
        assert_eq!(backend, I8I8KernelBackend::X86Avx512);
        assert_eq!(name, "x86-avx512bw");
    } else if arch::x86_i8_kernel_runtime_available() {
        assert_eq!(backend, I8I8KernelBackend::X86Avx2);
        assert_eq!(name, "x86-avx2-i32acc");
    } else {
        assert_eq!(backend, I8I8KernelBackend::Portable);
        assert_eq!(name, "portable");
    }
}

#[test]
fn i8_i8_dispatch_consistency() {
    let x = [3i8, -2, 5, 1, -4, 6, -1, 2];
    let row0 = [3i8, -2, 5, 1, -4, 6, -1, 2];
    let row1 = [-1i8, 4, -3, 2, 5, -2, 7, -6];
    let row2 = [2i8, 1, -2, 3, -5, 4, 1, -3];
    let has_arch = active_i8_i8_backend() != I8I8KernelBackend::Portable;

    assert_eq!(dot_i8_i8_arch(&x, 0.5, &row0, 0.25).is_some(), has_arch);
    assert_eq!(sum_i8_arch(&x, 0.5).is_some(), has_arch);
    assert_eq!(
        dot2_i8_i8_arch(&x, 0.5, &row0, 0.25, &row1, 0.125).is_some(),
        has_arch
    );
    assert_eq!(
        dot3_i8_i8_arch(
            &x,
            0.5,
            [
                I8ScaledRow {
                    values: &row0,
                    scale: 0.25
                },
                I8ScaledRow {
                    values: &row1,
                    scale: 0.125
                },
                I8ScaledRow {
                    values: &row2,
                    scale: 0.0625
                },
            ]
        )
        .is_some(),
        has_arch
    );
}

#[test]
fn i8_sum_arch_matches_scalar_reference() {
    let x = [
        3i8, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16, 17, -18, 19,
    ];
    let scale = 0.125f32;
    if let Some(got) = sum_i8_arch(&x, scale) {
        let expected = x.iter().map(|&v| (v as f32) * scale).sum::<f32>();
        assert!((got - expected).abs() <= 1e-6);
    }
    assert_eq!(sum_i8_arch(&[], scale), Some(0.0));
    assert_eq!(sum_i8_arch(&x, 0.0), None);
}

#[test]
fn i8_mul_grad_arch_matches_scalar_reference() {
    let grad = [
        0.5f32, -1.0, 2.0, 0.25, -0.75, 1.5, -2.5, 3.0, 0.125, -0.5, 0.75, -1.25, 1.75, -2.25,
        2.75, -3.25, 3.75,
    ];
    let x = [
        3i8, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16, 17, -18, 19,
    ];
    let scale = 0.125f32;
    let mut out = [0.0f32; 17];

    if mul_f32_i8_to_f32_arch(&grad, &x, scale, &mut out) {
        for ((&got, &g), &xv) in out.iter().zip(grad.iter()).zip(x.iter()) {
            assert!((got - g * (xv as f32) * scale).abs() <= 1e-6);
        }
    }
    let scalar = -0.375f32;
    if mul_i8_scalar_to_f32_arch(&x, scale, scalar, &mut out) {
        for (&got, &xv) in out.iter().zip(x.iter()) {
            assert!((got - (xv as f32) * scale * scalar).abs() <= 1e-6);
        }
    }
    if add_i8_scalar_to_f32_arch(&x, scale, scalar, &mut out) {
        for (&got, &xv) in out.iter().zip(x.iter()) {
            assert!((got - (scalar + (xv as f32) * scale)).abs() <= 1e-6);
        }
    }
    if sub_i8_scalar_to_f32_arch(&x, scale, scalar, false, &mut out) {
        for (&got, &xv) in out.iter().zip(x.iter()) {
            assert!((got - (scalar - (xv as f32) * scale)).abs() <= 1e-6);
        }
    }
    if sub_i8_scalar_to_f32_arch(&x, scale, scalar, true, &mut out) {
        for (&got, &xv) in out.iter().zip(x.iter()) {
            assert!((got - ((xv as f32) * scale - scalar)).abs() <= 1e-6);
        }
    }
    assert!(!mul_f32_i8_to_f32_arch(&grad, &x, 0.0, &mut out));
    assert!(!mul_i8_scalar_to_f32_arch(&x, 0.0, scalar, &mut out));
    assert!(!mul_i8_scalar_to_f32_arch(&x, scale, f32::NAN, &mut out));
    assert!(!add_i8_scalar_to_f32_arch(&x, 0.0, scalar, &mut out));
    assert!(!sub_i8_scalar_to_f32_arch(
        &x,
        scale,
        f32::NAN,
        true,
        &mut out
    ));
}

#[test]
fn i8_elementwise_arch_matches_dynamic_quantized_reference() {
    let lhs = [3i8, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16, 17];
    let rhs = [
        -2i8, 6, -8, 10, -12, 14, -16, 18, -20, 22, -24, 26, -28, 30, -32,
    ];
    let lhs_scale = 0.125f32;
    let rhs_scale = 0.0625f32;

    let reference = |op: I8ElementwiseOp| -> (Vec<i8>, f32) {
        let values = lhs
            .iter()
            .zip(rhs.iter())
            .map(|(&a, &b)| {
                let av = (a as f32) * lhs_scale;
                let bv = (b as f32) * rhs_scale;
                match op {
                    I8ElementwiseOp::Add => av + bv,
                    I8ElementwiseOp::Sub => av - bv,
                    I8ElementwiseOp::Mul => av * bv,
                }
            })
            .collect::<Vec<_>>();
        let max_abs = values.iter().copied().map(f32::abs).fold(0.0f32, f32::max);
        let scale = dynamic_i8_scale(max_abs);
        let inv_scale = 1.0 / scale;
        (
            values
                .iter()
                .map(|&v| (v * inv_scale).round().clamp(-127.0, 127.0) as i8)
                .collect(),
            scale,
        )
    };

    for (op, run) in [
        (
            I8ElementwiseOp::Add,
            add_i8_i8_arch as fn(&[i8], f32, &[i8], f32, &mut [i8]) -> Option<f32>,
        ),
        (I8ElementwiseOp::Sub, sub_i8_i8_arch),
        (I8ElementwiseOp::Mul, mul_i8_i8_arch),
    ] {
        let mut out = vec![0i8; lhs.len()];
        if let Some(scale) = run(&lhs, lhs_scale, &rhs, rhs_scale, &mut out) {
            let (expected, expected_scale) = reference(op);
            assert!((scale - expected_scale).abs() <= 1e-6);
            assert_eq!(out, expected);
        }
    }
}

#[test]
fn i8_elementwise_to_f32_arch_matches_scalar_reference() {
    let lhs = [3i8, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16, 17];
    let rhs = [
        -2i8, 6, -8, 10, -12, 14, -16, 18, -20, 22, -24, 26, -28, 30, -32,
    ];
    let lhs_scale = 0.125f32;
    let rhs_scale = 0.0625f32;
    let mut out = [0.0f32; 15];

    let reference = |a: i8, b: i8, op: I8ElementwiseOp| {
        let av = (a as f32) * lhs_scale;
        let bv = (b as f32) * rhs_scale;
        match op {
            I8ElementwiseOp::Add => av + bv,
            I8ElementwiseOp::Sub => av - bv,
            I8ElementwiseOp::Mul => av * bv,
        }
    };

    if add_i8_i8_to_f32_arch(&lhs, lhs_scale, &rhs, rhs_scale, &mut out) {
        for ((&got, &a), &b) in out.iter().zip(lhs.iter()).zip(rhs.iter()) {
            assert!((got - reference(a, b, I8ElementwiseOp::Add)).abs() <= 1e-6);
        }
    }
    if sub_i8_i8_to_f32_arch(&lhs, lhs_scale, &rhs, rhs_scale, &mut out) {
        for ((&got, &a), &b) in out.iter().zip(lhs.iter()).zip(rhs.iter()) {
            assert!((got - reference(a, b, I8ElementwiseOp::Sub)).abs() <= 1e-6);
        }
    }
    if mul_i8_i8_to_f32_arch(&lhs, lhs_scale, &rhs, rhs_scale, &mut out) {
        for ((&got, &a), &b) in out.iter().zip(lhs.iter()).zip(rhs.iter()) {
            assert!((got - reference(a, b, I8ElementwiseOp::Mul)).abs() <= 1e-6);
        }
    }

    assert!(!add_i8_i8_to_f32_arch(&lhs, 0.0, &rhs, rhs_scale, &mut out));
}

#[test]
fn f32_i8_elementwise_to_f32_arch_matches_scalar_reference() {
    let lhs = [
        0.25f32, -1.5, 2.0, 0.75, -3.25, 4.5, -0.125, 1.875, 2.25, -2.75, 3.5, -4.25, 0.5, 1.25,
        -1.0,
    ];
    let rhs = [
        -2i8, 6, -8, 10, -12, 14, -16, 18, -20, 22, -24, 26, -28, 30, -32,
    ];
    let scale = 0.0625f32;
    let mut out = [0.0f32; 15];

    if add_f32_i8_to_f32_arch(&lhs, &rhs, scale, &mut out) {
        for ((&got, &a), &b) in out.iter().zip(lhs.iter()).zip(rhs.iter()) {
            let expected = a + (b as f32) * scale;
            assert!((got - expected).abs() <= 1e-6);
        }
    }
    if sub_f32_i8_to_f32_arch(&lhs, &rhs, scale, false, &mut out) {
        for ((&got, &a), &b) in out.iter().zip(lhs.iter()).zip(rhs.iter()) {
            let expected = a - (b as f32) * scale;
            assert!((got - expected).abs() <= 1e-6);
        }
    }
    if sub_f32_i8_to_f32_arch(&lhs, &rhs, scale, true, &mut out) {
        for ((&got, &a), &b) in out.iter().zip(lhs.iter()).zip(rhs.iter()) {
            let expected = (b as f32) * scale - a;
            assert!((got - expected).abs() <= 1e-6);
        }
    }

    assert!(!add_f32_i8_to_f32_arch(
        &lhs[..lhs.len() - 1],
        &rhs,
        scale,
        &mut out
    ));
    assert!(!sub_f32_i8_to_f32_arch(&lhs, &rhs, 0.0, false, &mut out));
}

#[test]
fn i8_row_broadcast_arch_matches_scalar_reference() {
    let matrix = [
        3i8, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16, 17, -18, -19, 20, -21, 22, -23,
        24, -25, 26, -27, 28, -29, 30, -31, 32, -33, 34, 35, -36, 37, -38, 39, -40, 41, -42,
    ];
    let vector = [2i8, -3, 4, -5, 6, -7, 8, -9];
    let matrix_scale = 0.03125f32;
    let vector_scale = 0.0625f32;
    let last_dim = vector.len();

    let value = |a: i8, a_scale: f32, b: i8, b_scale: f32, op: I8ElementwiseOp| {
        let av = (a as f32) * a_scale;
        let bv = (b as f32) * b_scale;
        match op {
            I8ElementwiseOp::Add => av + bv,
            I8ElementwiseOp::Sub => av - bv,
            I8ElementwiseOp::Mul => av * bv,
        }
    };

    let check = |vector_on_rhs: bool, op: I8ElementwiseOp| {
        let args = I8RowBroadcast {
            lhs: if vector_on_rhs {
                I8ScaledRow {
                    values: &matrix,
                    scale: matrix_scale,
                }
            } else {
                I8ScaledRow {
                    values: &vector,
                    scale: vector_scale,
                }
            },
            rhs: if vector_on_rhs {
                I8ScaledRow {
                    values: &vector,
                    scale: vector_scale,
                }
            } else {
                I8ScaledRow {
                    values: &matrix,
                    scale: matrix_scale,
                }
            },
            last_dim,
            vector_on_rhs,
        };

        let values = (0..matrix.len())
            .map(|idx| {
                if vector_on_rhs {
                    value(
                        matrix[idx],
                        matrix_scale,
                        vector[idx % last_dim],
                        vector_scale,
                        op,
                    )
                } else {
                    value(
                        vector[idx % last_dim],
                        vector_scale,
                        matrix[idx],
                        matrix_scale,
                        op,
                    )
                }
            })
            .collect::<Vec<_>>();
        let max_abs = values.iter().copied().map(f32::abs).fold(0.0f32, f32::max);
        let expected_scale = dynamic_i8_scale(max_abs);
        let inv_scale = 1.0 / expected_scale;
        let expected_i8 = values
            .iter()
            .map(|&v| (v * inv_scale).round().clamp(-127.0, 127.0) as i8)
            .collect::<Vec<_>>();

        let mut typed_out = vec![0i8; matrix.len()];
        let got_scale = match op {
            I8ElementwiseOp::Add => add_i8_i8_row_broadcast_arch(args, &mut typed_out),
            I8ElementwiseOp::Sub => sub_i8_i8_row_broadcast_arch(args, &mut typed_out),
            I8ElementwiseOp::Mul => mul_i8_i8_row_broadcast_arch(args, &mut typed_out),
        };
        let Some(got_scale) = got_scale else {
            return;
        };
        assert!((got_scale - expected_scale).abs() <= 1e-6);
        assert_eq!(typed_out, expected_i8);

        let mut f32_out = vec![0.0f32; matrix.len()];
        let handled = match op {
            I8ElementwiseOp::Add => add_i8_i8_row_broadcast_to_f32_arch(args, &mut f32_out),
            I8ElementwiseOp::Sub => sub_i8_i8_row_broadcast_to_f32_arch(args, &mut f32_out),
            I8ElementwiseOp::Mul => mul_i8_i8_row_broadcast_to_f32_arch(args, &mut f32_out),
        };
        assert!(handled);
        for (&got, &expected) in f32_out.iter().zip(values.iter()) {
            assert!((got - expected).abs() <= 1e-6);
        }
    };

    for vector_on_rhs in [true, false] {
        for op in [
            I8ElementwiseOp::Add,
            I8ElementwiseOp::Sub,
            I8ElementwiseOp::Mul,
        ] {
            check(vector_on_rhs, op);
        }
    }

    let bad_args = I8RowBroadcast {
        lhs: I8ScaledRow {
            values: &matrix,
            scale: matrix_scale,
        },
        rhs: I8ScaledRow {
            values: &vector,
            scale: vector_scale,
        },
        last_dim: 0,
        vector_on_rhs: true,
    };
    let mut out = vec![0i8; matrix.len()];
    assert!(add_i8_i8_row_broadcast_arch(bad_args, &mut out).is_none());
}

#[test]
fn i8_i8_avx2_fallback_matches_scalar_reference_when_available() {
    #[cfg(all(
        feature = "x86-int8-kernels",
        any(target_arch = "x86_64", target_arch = "x86")
    ))]
    {
        if !arch::x86_i8_kernel_runtime_available() {
            return;
        }

        let x = [
            3i8, -2, 5, 1, -4, 6, -1, 2, 7, -8, 9, -10, 11, -12, 13, -14, 4, -3, 2,
        ];
        let row0 = [
            -7i8, 3, 12, -5, 9, -11, 15, 1, 8, -4, 6, 10, -13, 14, -2, 5, 2, -3, 7,
        ];
        let row1 = [
            4i8, -9, 6, 2, -8, 7, -5, 3, -1, 10, -6, 11, -12, 13, -4, 8, -2, 1, 5,
        ];
        let row2 = [
            -3i8, 2, -7, 9, -10, 6, 4, -1, 12, -8, 5, -6, 11, -13, 7, -2, 3, -4, 10,
        ];
        let x_scale = 0.0625f32;
        let scale0 = 0.125f32;
        let scale1 = 0.0625f32;
        let scale2 = 0.25f32;
        let scalar = |row: &[i8], scale: f32| -> f32 {
            x.iter()
                .zip(row.iter())
                .map(|(&xv, &rv)| xv as i32 * rv as i32)
                .sum::<i32>() as f32
                * x_scale
                * scale
        };

        let single = unsafe { dot_i8_i8_x86_avx2(&x, x_scale, &row0, scale0) };
        let pair = unsafe { dot2_i8_i8_x86_avx2(&x, x_scale, &row0, scale0, &row1, scale1) };
        let triple = unsafe {
            dot3_i8_i8_x86_avx2(
                &x,
                x_scale,
                [
                    I8ScaledRow {
                        values: &row0,
                        scale: scale0,
                    },
                    I8ScaledRow {
                        values: &row1,
                        scale: scale1,
                    },
                    I8ScaledRow {
                        values: &row2,
                        scale: scale2,
                    },
                ],
            )
        };

        assert!((single - scalar(&row0, scale0)).abs() <= 1e-5);
        assert!((pair.0 - scalar(&row0, scale0)).abs() <= 1e-5);
        assert!((pair.1 - scalar(&row1, scale1)).abs() <= 1e-5);
        assert!((triple.0 - scalar(&row0, scale0)).abs() <= 1e-5);
        assert!((triple.1 - scalar(&row1, scale1)).abs() <= 1e-5);
        assert!((triple.2 - scalar(&row2, scale2)).abs() <= 1e-5);
    }
}

#[test]
fn f32_i8_avx512_path_matches_scalar_reference_when_available() {
    #[cfg(all(
        feature = "x86-int8-kernels",
        any(target_arch = "x86_64", target_arch = "x86")
    ))]
    {
        if !arch::x86_avx512_i8_i8_kernel_runtime_available() {
            return;
        }

        let x: Vec<f32> = (0..67)
            .map(|i| ((i as f32 * 0.03125) - 1.0).sin() * 3.0)
            .collect();
        let row0: Vec<i8> = (0..67).map(|i| patterned_i8(i, 37, 11)).collect();
        let row1: Vec<i8> = (0..67).map(|i| patterned_i8(i, 53, 19)).collect();
        let row2: Vec<i8> = (0..67).map(|i| patterned_i8(i, 71, 23)).collect();
        let scale0 = 0.0078125f32;
        let scale1 = 0.01171875f32;
        let scale2 = 0.015625f32;
        let scalar = |row: &[i8], scale: f32| -> f32 {
            x.iter()
                .zip(row.iter())
                .map(|(&xv, &rv)| xv * rv as f32)
                .sum::<f32>()
                * scale
        };

        let single = unsafe { dot_f32_i8_x86_avx512(&x, &row0, scale0) };
        let pair = unsafe { dot2_f32_i8_x86_avx512(&x, &row0, scale0, &row1, scale1) };
        let triple =
            unsafe { dot3_f32_i8_x86_avx512(&x, &row0, scale0, &row1, scale1, &row2, scale2) };

        assert!((single - scalar(&row0, scale0)).abs() <= 1e-4);
        assert!((pair.0 - scalar(&row0, scale0)).abs() <= 1e-4);
        assert!((pair.1 - scalar(&row1, scale1)).abs() <= 1e-4);
        assert!((triple.0 - scalar(&row0, scale0)).abs() <= 1e-4);
        assert!((triple.1 - scalar(&row1, scale1)).abs() <= 1e-4);
        assert!((triple.2 - scalar(&row2, scale2)).abs() <= 1e-4);
    }
}

#[test]
fn architecture_dispatch_consistency() {
    let x = [0.5f32, -1.0, 2.0, 0.25, -0.75, 1.5, -0.5, 3.0];
    let row0 = [3i8, -2, 5, 1, -4, 6, -1, 2];
    let row1 = [-1i8, 4, -3, 2, 5, -2, 7, -6];
    let row2 = [2i8, 1, -2, 3, -5, 4, 1, -3];
    let single = dot_f32_i8_arch(&x, &row0, 0.125);
    let triple = dot3_f32_i8_arch(&x, &row0, 0.125, &row1, 0.25, &row2, 0.5);

    if active_int8_backend() == Int8KernelBackend::Portable {
        assert!(single.is_none());
        assert!(triple.is_none());
    } else {
        assert!(single.is_some());
        assert!(triple.is_some());
    }
}

#[test]
fn x86_int8_fast_paths_match_scalar_reference() {
    let x = [
        -1.25f32, 0.5, 2.0, -0.75, 1.5, -2.25, 3.0, 0.125, 1.75, -1.0, 0.625, 2.5, -3.5, 4.0,
        -0.875, 1.125, 0.333, -0.666, 1.999,
    ];
    let row0 = [
        -7i8, 3, 12, -5, 9, -11, 15, 1, 8, -4, 6, 10, -13, 14, -2, 5, 2, -3, 7,
    ];
    let row1 = [
        4i8, -9, 6, 2, -8, 7, -5, 3, -1, 10, -6, 11, -12, 13, -4, 8, -2, 1, 5,
    ];
    let row2 = [
        -3i8, 2, -7, 9, -10, 6, 4, -1, 12, -8, 5, -6, 11, -13, 7, -2, 3, -4, 10,
    ];
    let scale0 = 0.125f32;
    let scale1 = 0.0625f32;
    let scale2 = 0.25f32;

    let scalar = |row: &[i8], scale: f32| -> f32 {
        x.iter()
            .zip(row.iter())
            .map(|(&xv, &rv)| xv * rv as f32)
            .sum::<f32>()
            * scale
    };

    if let Some(sum) = dot_f32_i8_arch(&x, &row0, scale0) {
        assert!((sum - scalar(&row0, scale0)).abs() <= 1e-5);
    }
    if let Some((sum0, sum1)) = dot2_f32_i8_arch(&x, &row0, scale0, &row1, scale1) {
        assert!((sum0 - scalar(&row0, scale0)).abs() <= 1e-5);
        assert!((sum1 - scalar(&row1, scale1)).abs() <= 1e-5);
    }
    if let Some((sum0, sum1, sum2)) =
        dot3_f32_i8_arch(&x, &row0, scale0, &row1, scale1, &row2, scale2)
    {
        assert!((sum0 - scalar(&row0, scale0)).abs() <= 1e-5);
        assert!((sum1 - scalar(&row1, scale1)).abs() <= 1e-5);
        assert!((sum2 - scalar(&row2, scale2)).abs() <= 1e-5);
    }

    let x_i8 = [
        3i8, -2, 5, 1, -4, 6, -1, 2, 7, -8, 9, -10, 11, -12, 13, -14, 4, -3, 2,
    ];
    let x_scale = 0.0625f32;
    let scalar_i8 = |row: &[i8], scale: f32| -> f32 {
        x_i8.iter()
            .zip(row.iter())
            .map(|(&xv, &rv)| xv as i32 * rv as i32)
            .sum::<i32>() as f32
            * x_scale
            * scale
    };

    if let Some(sum) = dot_i8_i8_arch(&x_i8, x_scale, &row0, scale0) {
        assert!((sum - scalar_i8(&row0, scale0)).abs() <= 1e-5);
    }
    if let Some((sum0, sum1)) = dot2_i8_i8_arch(&x_i8, x_scale, &row0, scale0, &row1, scale1) {
        assert!((sum0 - scalar_i8(&row0, scale0)).abs() <= 1e-5);
        assert!((sum1 - scalar_i8(&row1, scale1)).abs() <= 1e-5);
    }
    if let Some((sum0, sum1, sum2)) = dot3_i8_i8_arch(
        &x_i8,
        x_scale,
        [
            I8ScaledRow {
                values: &row0,
                scale: scale0,
            },
            I8ScaledRow {
                values: &row1,
                scale: scale1,
            },
            I8ScaledRow {
                values: &row2,
                scale: scale2,
            },
        ],
    ) {
        assert!((sum0 - scalar_i8(&row0, scale0)).abs() <= 1e-5);
        assert!((sum1 - scalar_i8(&row1, scale1)).abs() <= 1e-5);
        assert!((sum2 - scalar_i8(&row2, scale2)).abs() <= 1e-5);
    }
}

#[test]
fn length_mismatch_disables_arch_fast_path() {
    let x = [1.0f32, 2.0, 3.0, 4.0];
    let short = [1i8, 2];

    assert!(dot_f32_i8_arch(&x, &short, 0.5).is_none());
    assert!(dot2_f32_i8_arch(&x, &short, 0.5, &short, 0.25).is_none());
    assert!(dot3_f32_i8_arch(&x, &short, 0.5, &short, 0.25, &short, 0.125).is_none());
    let row = [1i8, 2, 3, 4];
    assert!(dot_i8_i8_arch(&row, 0.5, &short, 0.25).is_none());
    assert!(dot2_i8_i8_arch(&row, 0.5, &short, 0.25, &short, 0.125).is_none());
    assert!(
        dot3_i8_i8_arch(
            &row,
            0.5,
            [
                I8ScaledRow {
                    values: &short,
                    scale: 0.25
                },
                I8ScaledRow {
                    values: &short,
                    scale: 0.125
                },
                I8ScaledRow {
                    values: &short,
                    scale: 0.0625
                },
            ]
        )
        .is_none()
    );
    let mut update = [1i8, 2, 3, 4];
    assert!(!sgd_update_i8_f32_arch(&mut update, 0.5, &x[..2], 0.1));
    assert!(!sgd_update_i8_f32_arch(&mut update, 0.0, &x, 0.1));
    let mut velocity = [0.0f32; 4];
    assert!(!sgd_momentum_update_i8_f32_arch(
        &mut update,
        0.5,
        &mut velocity,
        &x[..2],
        0.1,
        0.9
    ));
    assert!(!sgd_momentum_update_i8_f32_arch(
        &mut update,
        0.0,
        &mut velocity,
        &x,
        0.1,
        0.9
    ));
    let mut exp_avg = [0.0f32; 4];
    let mut exp_avg_sq = [0.0f32; 4];
    assert!(!adam_update_i8_f32_arch(
        &mut update,
        0.5,
        &mut exp_avg,
        &mut exp_avg_sq,
        &x[..2],
        0.1,
        0.9,
        0.999,
        0.1,
        0.001,
        1e-8
    ));
    assert!(!adam_update_i8_f32_arch(
        &mut update,
        0.0,
        &mut exp_avg,
        &mut exp_avg_sq,
        &x,
        0.1,
        0.9,
        0.999,
        0.1,
        0.001,
        1e-8
    ));
}

#[test]
fn i8_sgd_update_fast_path_matches_scalar_reference() {
    let scale = 0.0625f32;
    let lr = 0.03125f32;
    let mut data: Vec<i8> = (0..65).map(|i| patterned_i8(i, 37, 11)).collect();
    let grad: Vec<f32> = (0..data.len())
        .map(|i| ((i as f32 * 0.17).sin() * 3.0) + ((i % 7) as f32 - 3.0) * 0.125)
        .collect();
    let mut expected = data.clone();
    for (w, g) in expected.iter_mut().zip(grad.iter()) {
        let updated = (*w as f32) * scale - lr * *g;
        *w = (updated / scale).round().clamp(-127.0, 127.0) as i8;
    }

    if sgd_update_i8_f32_arch(&mut data, scale, &grad, lr) {
        assert_eq!(data, expected);
    }
}

#[test]
fn i8_adam_update_fast_path_matches_scalar_reference() {
    let scale = 0.0625f32;
    let lr = 0.001f32;
    let beta1 = 0.9f32;
    let beta2 = 0.999f32;
    let bias_correction1 = 0.1f32;
    let bias_correction2 = 0.001f32;
    let eps = 1e-8f32;
    let mut data: Vec<i8> = (0..65).map(|i| patterned_i8(i, 71, 23)).collect();
    let mut exp_avg: Vec<f32> = (0..data.len())
        .map(|i| ((i as f32 * 0.07).sin() * 0.25) - 0.1)
        .collect();
    let mut exp_avg_sq: Vec<f32> = (0..data.len())
        .map(|i| ((i % 11) as f32 + 1.0) * 0.0001)
        .collect();
    let grad: Vec<f32> = (0..data.len())
        .map(|i| ((i as f32 * 0.19).cos() * 0.75) + ((i % 3) as f32 - 1.0) * 0.05)
        .collect();
    let mut expected_data = data.clone();
    let mut expected_m = exp_avg.clone();
    let mut expected_v = exp_avg_sq.clone();
    for (((w, m), v), g) in expected_data
        .iter_mut()
        .zip(expected_m.iter_mut())
        .zip(expected_v.iter_mut())
        .zip(grad.iter())
    {
        *m = beta1 * *m + (1.0 - beta1) * *g;
        *v = beta2 * *v + (1.0 - beta2) * *g * *g;
        let m_hat = *m / bias_correction1;
        let v_hat = *v / bias_correction2;
        let updated = (*w as f32) * scale - lr * (m_hat / (v_hat.sqrt() + eps));
        *w = (updated / scale).round().clamp(-127.0, 127.0) as i8;
    }

    if adam_update_i8_f32_arch(
        &mut data,
        scale,
        &mut exp_avg,
        &mut exp_avg_sq,
        &grad,
        lr,
        beta1,
        beta2,
        bias_correction1,
        bias_correction2,
        eps,
    ) {
        assert_eq!(data, expected_data);
        for (actual, expected) in exp_avg.iter().zip(expected_m.iter()) {
            assert!((actual - expected).abs() <= 1e-6);
        }
        for (actual, expected) in exp_avg_sq.iter().zip(expected_v.iter()) {
            assert!((actual - expected).abs() <= 1e-6);
        }
    }
}

#[test]
fn i8_sgd_momentum_update_fast_path_matches_scalar_reference() {
    let scale = 0.0625f32;
    let lr = 0.03125f32;
    let momentum = 0.875f32;
    let mut data: Vec<i8> = (0..65).map(|i| patterned_i8(i, 53, 19)).collect();
    let mut velocity: Vec<f32> = (0..data.len())
        .map(|i| ((i as f32 * 0.11).cos() * 0.5) - 0.25)
        .collect();
    let grad: Vec<f32> = (0..data.len())
        .map(|i| ((i as f32 * 0.13).sin() * 2.0) + ((i % 5) as f32 - 2.0) * 0.25)
        .collect();
    let mut expected_data = data.clone();
    let mut expected_velocity = velocity.clone();
    for ((w, v), g) in expected_data
        .iter_mut()
        .zip(expected_velocity.iter_mut())
        .zip(grad.iter())
    {
        *v = momentum * *v + *g;
        let updated = (*w as f32) * scale - lr * *v;
        *w = (updated / scale).round().clamp(-127.0, 127.0) as i8;
    }

    if sgd_momentum_update_i8_f32_arch(&mut data, scale, &mut velocity, &grad, lr, momentum) {
        assert_eq!(data, expected_data);
        for (actual, expected) in velocity.iter().zip(expected_velocity.iter()) {
            assert!((actual - expected).abs() <= 1e-6);
        }
    }
}

#[test]
fn i8_update_avx2_fallbacks_match_scalar_reference_when_available() {
    #[cfg(all(
        feature = "x86-int8-kernels",
        any(target_arch = "x86_64", target_arch = "x86")
    ))]
    {
        if !arch::x86_i8_kernel_runtime_available() {
            return;
        }

        let scale = 0.0625f32;
        let lr = 0.03125f32;
        let mut data: Vec<i8> = (0..65).map(|i| patterned_i8(i, 37, 11)).collect();
        let grad: Vec<f32> = (0..data.len())
            .map(|i| ((i as f32 * 0.17).sin() * 3.0) + ((i % 7) as f32 - 3.0) * 0.125)
            .collect();

        let mut expected_sgd = data.clone();
        for (w, g) in expected_sgd.iter_mut().zip(grad.iter()) {
            let updated = (*w as f32) * scale - lr * *g;
            *w = (updated / scale).round().clamp(-127.0, 127.0) as i8;
        }
        unsafe { sgd_update_i8_f32_x86_avx2(&mut data, scale, &grad, lr) };
        assert_eq!(data, expected_sgd);

        let momentum = 0.875f32;
        let mut data: Vec<i8> = (0..65).map(|i| patterned_i8(i, 53, 19)).collect();
        let mut velocity: Vec<f32> = (0..data.len())
            .map(|i| ((i as f32 * 0.11).cos() * 0.5) - 0.25)
            .collect();
        let mut expected_data = data.clone();
        let mut expected_velocity = velocity.clone();
        for ((w, v), g) in expected_data
            .iter_mut()
            .zip(expected_velocity.iter_mut())
            .zip(grad.iter())
        {
            *v = momentum * *v + *g;
            let updated = (*w as f32) * scale - lr * *v;
            *w = (updated / scale).round().clamp(-127.0, 127.0) as i8;
        }
        unsafe {
            sgd_momentum_update_i8_f32_x86_avx2(
                &mut data,
                scale,
                &mut velocity,
                &grad,
                lr,
                momentum,
            )
        };
        assert_eq!(data, expected_data);
        for (actual, expected) in velocity.iter().zip(expected_velocity.iter()) {
            assert!((actual - expected).abs() <= 1e-6);
        }

        let lr = 0.001f32;
        let beta1 = 0.9f32;
        let beta2 = 0.999f32;
        let bias_correction1 = 0.1f32;
        let bias_correction2 = 0.001f32;
        let eps = 1e-8f32;
        let mut data: Vec<i8> = (0..65).map(|i| patterned_i8(i, 71, 23)).collect();
        let mut exp_avg: Vec<f32> = (0..data.len())
            .map(|i| ((i as f32 * 0.07).sin() * 0.25) - 0.1)
            .collect();
        let mut exp_avg_sq: Vec<f32> = (0..data.len())
            .map(|i| ((i % 11) as f32 + 1.0) * 0.0001)
            .collect();
        let mut expected_data = data.clone();
        let mut expected_m = exp_avg.clone();
        let mut expected_v = exp_avg_sq.clone();
        for (((w, m), v), g) in expected_data
            .iter_mut()
            .zip(expected_m.iter_mut())
            .zip(expected_v.iter_mut())
            .zip(grad.iter())
        {
            *m = beta1 * *m + (1.0 - beta1) * *g;
            *v = beta2 * *v + (1.0 - beta2) * *g * *g;
            let m_hat = *m / bias_correction1;
            let v_hat = *v / bias_correction2;
            let updated = (*w as f32) * scale - lr * (m_hat / (v_hat.sqrt() + eps));
            *w = (updated / scale).round().clamp(-127.0, 127.0) as i8;
        }
        unsafe {
            adam_update_i8_f32_x86_avx2(
                &mut data,
                scale,
                &mut exp_avg,
                &mut exp_avg_sq,
                &grad,
                lr,
                beta1,
                beta2,
                bias_correction1,
                bias_correction2,
                eps,
            )
        };
        assert_eq!(data, expected_data);
        for (actual, expected) in exp_avg.iter().zip(expected_m.iter()) {
            assert!((actual - expected).abs() <= 1e-6);
        }
        for (actual, expected) in exp_avg_sq.iter().zip(expected_v.iter()) {
            assert!((actual - expected).abs() <= 1e-6);
        }
    }
}

#[test]
#[ignore = "performance sanity test; run with --ignored --nocapture"]
fn perf_i8_i8_dot2_dot3_arch_paths() {
    let len = 8192usize;
    let iters = 4096usize;
    let make_i8 = |seed: usize| -> Vec<i8> {
        (0..len)
            .map(|i| {
                let v =
                    ((i.wrapping_mul(1103515245usize) ^ seed).rotate_left(7) & 255) as i32 - 128;
                v.clamp(-127, 127) as i8
            })
            .collect()
    };
    let x = make_i8(0x1234);
    let row0 = make_i8(0x2345);
    let row1 = make_i8(0x3456);
    let row2 = make_i8(0x4567);
    let x_f32 = (0..len)
        .map(|i| ((i * 97 % 257) as f32 - 128.0) / 64.0)
        .collect::<Vec<_>>();
    let x_scale = 0.03125f32;
    let scale0 = 0.015625f32;
    let scale1 = 0.0078125f32;
    let scale2 = 0.00390625f32;
    let row_broadcast_last_dim = 256usize;
    let row_broadcast_vector = &row1[..row_broadcast_last_dim];
    let grad = (0..len)
        .map(|i| ((i * 41 % 233) as f32 - 116.0) / 233.0)
        .collect::<Vec<_>>();
    let mut grad_out = vec![0.0f32; len];
    let mut fwd_out = vec![0.0f32; len];
    let row_broadcast_args = I8RowBroadcast {
        lhs: I8ScaledRow {
            values: &x,
            scale: x_scale,
        },
        rhs: I8ScaledRow {
            values: row_broadcast_vector,
            scale: scale1,
        },
        last_dim: row_broadcast_last_dim,
        vector_on_rhs: true,
    };

    if dot2_i8_i8_arch(&x, x_scale, &row0, scale0, &row1, scale1).is_none() {
        eprintln!(
            "perf_i8_i8_dot2_dot3 skipped backend={}",
            active_i8_i8_backend_name()
        );
        return;
    }

    let start = std::time::Instant::now();
    let mut sink = 0.0f32;
    for _ in 0..iters {
        let (a, b) = dot2_i8_i8_arch(&x, x_scale, &row0, scale0, &row1, scale1).expect("dot2 arch");
        sink += a + b;
    }
    let dot2_elapsed = start.elapsed();

    let start = std::time::Instant::now();
    for _ in 0..iters {
        let (a, b, c) = dot3_i8_i8_arch(
            &x,
            x_scale,
            [
                I8ScaledRow {
                    values: &row0,
                    scale: scale0,
                },
                I8ScaledRow {
                    values: &row1,
                    scale: scale1,
                },
                I8ScaledRow {
                    values: &row2,
                    scale: scale2,
                },
            ],
        )
        .expect("dot3 arch");
        sink += a + b + c;
    }
    let dot3_elapsed = start.elapsed();

    let start = std::time::Instant::now();
    for _ in 0..iters {
        let (a, b) =
            dot2_f32_i8_arch(&x_f32, &row0, scale0, &row1, scale1).expect("f32_i8 dot2 arch");
        sink += a + b;
    }
    let f32_i8_dot2_elapsed = start.elapsed();

    let start = std::time::Instant::now();
    for _ in 0..iters {
        let (a, b, c) = dot3_f32_i8_arch(&x_f32, &row0, scale0, &row1, scale1, &row2, scale2)
            .expect("f32_i8 dot3 arch");
        sink += a + b + c;
    }
    let f32_i8_dot3_elapsed = start.elapsed();

    let start = std::time::Instant::now();
    for _ in 0..iters {
        sink += sum_i8_arch(&x, x_scale).expect("sum arch");
    }
    let sum_elapsed = start.elapsed();

    let mut out = vec![0i8; len];
    let start = std::time::Instant::now();
    for _ in 0..iters {
        sink += add_i8_i8_arch(&x, x_scale, &row0, scale0, &mut out).expect("add arch");
    }
    let add_elapsed = start.elapsed();

    let start = std::time::Instant::now();
    for _ in 0..iters {
        sink += mul_i8_i8_arch(&x, x_scale, &row0, scale0, &mut out).expect("mul arch");
    }
    let mul_elapsed = start.elapsed();

    let start = std::time::Instant::now();
    for _ in 0..iters {
        sink += mul_i8_i8_row_broadcast_arch(row_broadcast_args, &mut out)
            .expect("row-broadcast mul arch");
    }
    let row_broadcast_mul_elapsed = start.elapsed();

    let start = std::time::Instant::now();
    for _ in 0..iters {
        assert!(mul_f32_i8_to_f32_arch(&grad, &x, x_scale, &mut grad_out));
        sink += grad_out[0];
    }
    let mul_grad_elapsed = start.elapsed();

    let start = std::time::Instant::now();
    for _ in 0..iters {
        assert!(mul_i8_scalar_to_f32_arch(
            &x,
            x_scale,
            -0.375,
            &mut grad_out
        ));
        sink += grad_out[0];
    }
    let scalar_mul_elapsed = start.elapsed();

    let start = std::time::Instant::now();
    for _ in 0..iters {
        assert!(mul_i8_i8_to_f32_arch(
            &x,
            x_scale,
            &row0,
            scale0,
            &mut fwd_out
        ));
        sink += fwd_out[0];
    }
    let fwd_mul_elapsed = start.elapsed();

    let start = std::time::Instant::now();
    for _ in 0..iters {
        assert!(mul_i8_i8_row_broadcast_to_f32_arch(
            row_broadcast_args,
            &mut fwd_out
        ));
        sink += fwd_out[0];
    }
    let row_broadcast_fwd_mul_elapsed = start.elapsed();

    eprintln!(
        "perf_i8_i8_dot2_dot3 backend={} len={len} iters={iters} dot2={:.3}us dot3={:.3}us f32_i8_dot2={:.3}us f32_i8_dot3={:.3}us sum={:.3}us add={:.3}us mul={:.3}us row_broadcast_mul={:.3}us mul_grad={:.3}us scalar_mul_f32={:.3}us fwd_mul_f32={:.3}us row_broadcast_fwd_mul_f32={:.3}us sink={sink}",
        active_i8_i8_backend_name(),
        dot2_elapsed.as_secs_f64() * 1e6 / iters as f64,
        dot3_elapsed.as_secs_f64() * 1e6 / iters as f64,
        f32_i8_dot2_elapsed.as_secs_f64() * 1e6 / iters as f64,
        f32_i8_dot3_elapsed.as_secs_f64() * 1e6 / iters as f64,
        sum_elapsed.as_secs_f64() * 1e6 / iters as f64,
        add_elapsed.as_secs_f64() * 1e6 / iters as f64,
        mul_elapsed.as_secs_f64() * 1e6 / iters as f64,
        row_broadcast_mul_elapsed.as_secs_f64() * 1e6 / iters as f64,
        mul_grad_elapsed.as_secs_f64() * 1e6 / iters as f64,
        scalar_mul_elapsed.as_secs_f64() * 1e6 / iters as f64,
        fwd_mul_elapsed.as_secs_f64() * 1e6 / iters as f64,
        row_broadcast_fwd_mul_elapsed.as_secs_f64() * 1e6 / iters as f64,
    );
}
