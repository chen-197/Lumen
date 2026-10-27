use super::*;

#[test]
fn backend_name_matches_backend_enum() {
    let name = active_float_backend_name();
    match active_float_backend() {
        FloatKernelBackend::Portable => assert_eq!(name, "portable"),
        FloatKernelBackend::Arm64Neon => assert_eq!(name, "arm64-neon"),
        FloatKernelBackend::X86Avx512 => assert_eq!(name, "x86-avx512"),
        FloatKernelBackend::X86Avx2 => assert_eq!(name, "x86-avx2"),
    }
}

#[test]
fn f16_f16_backend_name_matches_runtime_path() {
    let backend = active_f16_f16_backend();
    let name = active_f16_f16_backend_name();

    if arch::x86_avx512_fp16_kernel_runtime_available() {
        assert_eq!(backend, F16F16KernelBackend::X86Avx512Fp16);
        assert_eq!(name, "x86-avx512fp16-nightly");
    } else if arch::x86_f16c_kernel_runtime_available() {
        assert_eq!(backend, F16F16KernelBackend::X86Avx2F16c);
        assert_eq!(name, "x86-avx2-f16c");
    } else {
        assert_eq!(backend, F16F16KernelBackend::Portable);
        assert_eq!(name, "portable");
    }
}

#[test]
fn bf16_bf16_backend_name_matches_runtime_path() {
    let backend = active_bf16_bf16_backend();
    let name = active_bf16_bf16_backend_name();

    if arch::x86_avx512_bf16_kernel_runtime_available() {
        assert_eq!(backend, Bf16Bf16KernelBackend::X86Avx512Bf16);
        assert_eq!(name, "x86-avx512bf16");
    } else if arch::x86_fp_kernel_runtime_available() {
        assert_eq!(backend, Bf16Bf16KernelBackend::X86Avx2F32Acc);
        assert_eq!(name, "x86-avx2-bf16-f32acc");
    } else {
        assert_eq!(backend, Bf16Bf16KernelBackend::Portable);
        assert_eq!(name, "portable");
    }
}

#[test]
fn bf16_bf16_dispatch_consistency() {
    let x = [
        bf16::from_f32(0.5),
        bf16::from_f32(-1.0),
        bf16::from_f32(2.0),
        bf16::from_f32(0.25),
        bf16::from_f32(-0.75),
        bf16::from_f32(1.5),
        bf16::from_f32(-0.5),
        bf16::from_f32(3.0),
    ];
    let row0 = x.map(|v| bf16::from_f32(v.to_f32() * 0.75 + 0.125));
    let row1 = x.map(|v| bf16::from_f32(v.to_f32() * -0.5 + 0.25));
    let row2 = x.map(|v| bf16::from_f32(v.to_f32() * 1.25 - 0.75));
    let has_arch = active_bf16_bf16_backend() != Bf16Bf16KernelBackend::Portable;

    assert_eq!(dot_bf16_bf16_arch(&x, &row0).is_some(), has_arch);
    assert_eq!(dot2_bf16_bf16_arch(&x, &row0, &row1).is_some(), has_arch);
    assert_eq!(
        dot3_bf16_bf16_arch(&x, &row0, &row1, &row2).is_some(),
        has_arch
    );
}

#[test]
fn f16_f16_dispatch_consistency() {
    let x = [
        f16::from_f32(0.5),
        f16::from_f32(-1.0),
        f16::from_f32(2.0),
        f16::from_f32(0.25),
        f16::from_f32(-0.75),
        f16::from_f32(1.5),
        f16::from_f32(-0.5),
        f16::from_f32(3.0),
    ];
    let row0 = x.map(|v| f16::from_f32(v.to_f32() * 0.75 + 0.125));
    let row1 = x.map(|v| f16::from_f32(v.to_f32() * -0.5 + 0.25));
    let row2 = x.map(|v| f16::from_f32(v.to_f32() * 1.25 - 0.75));
    let has_arch = active_f16_f16_backend() != F16F16KernelBackend::Portable;

    assert_eq!(dot_f16_f16_arch(&x, &row0).is_some(), has_arch);
    assert_eq!(dot2_f16_f16_arch(&x, &row0, &row1).is_some(), has_arch);
    assert_eq!(
        dot3_f16_f16_arch(&x, &row0, &row1, &row2).is_some(),
        has_arch
    );
}

#[test]
fn architecture_dispatch_consistency() {
    let x = [0.5f32, -1.0, 2.0, 0.25, -0.75, 1.5, -0.5, 3.0];
    let row = [3.0f32, -2.0, 5.0, 1.0, -4.0, 6.0, -1.0, 2.0];
    let single = dot_f32_arch(&x, &row);

    if active_float_backend() == FloatKernelBackend::Portable {
        assert!(single.is_none());
    } else {
        assert!(single.is_some());
    }
}

#[test]
fn f16_and_bf16_dispatch_consistency() {
    let x = [0.5f32, -1.0, 2.0, 0.25, -0.75, 1.5, -0.5, 3.0];
    let row_f16 = x.map(f16::from_f32);
    let row_bf16 = x.map(bf16::from_f32);
    let f16_sum = dot_f32_f16_arch(&x, &row_f16);
    let bf16_sum = dot_f32_bf16_arch(&x, &row_bf16);

    match active_float_backend() {
        FloatKernelBackend::Portable => {
            assert!(f16_sum.is_none());
            assert!(bf16_sum.is_none());
        }
        FloatKernelBackend::Arm64Neon => {
            if arch::arm64_fp16_kernel_runtime_available() {
                assert!(f16_sum.is_some());
            } else {
                assert!(f16_sum.is_none());
            }
            assert!(bf16_sum.is_some());
        }
        FloatKernelBackend::X86Avx512 => {
            assert!(f16_sum.is_some());
            assert!(bf16_sum.is_some());
        }
        FloatKernelBackend::X86Avx2 => {
            if arch::x86_fp16_kernel_runtime_available() {
                assert!(f16_sum.is_some());
            } else {
                assert!(f16_sum.is_none());
            }
            assert!(bf16_sum.is_some());
        }
    }
}

#[test]
fn mixed_precision_fast_paths_match_scalar_reference() {
    let x = [
        -1.25f32, 0.5, 2.0, -0.75, 1.5, -2.25, 3.0, 0.125, 1.75, -1.0, 0.625, 2.5, -3.5, 4.0,
        -0.875, 1.125, 0.333, -0.666, 1.999,
    ];
    let row0_f16 = x.map(|v| f16::from_f32(v * 0.75 + 0.125));
    let row1_f16 = x.map(|v| f16::from_f32(v * -0.5 + 0.25));
    let row2_f16 = x.map(|v| f16::from_f32(v * 1.25 - 0.75));
    let row0_bf16 = x.map(|v| bf16::from_f32(v * 0.75 + 0.125));
    let row1_bf16 = x.map(|v| bf16::from_f32(v * -0.5 + 0.25));
    let row2_bf16 = x.map(|v| bf16::from_f32(v * 1.25 - 0.75));
    let x_f16 = x.map(f16::from_f32);
    let x_bf16 = x.map(bf16::from_f32);

    let scalar_f16 = |row: &[f16]| -> f32 {
        x.iter()
            .zip(row.iter())
            .map(|(&xv, &rv)| xv * rv.to_f32())
            .sum()
    };
    let scalar_bf16 = |row: &[bf16]| -> f32 {
        x.iter()
            .zip(row.iter())
            .map(|(&xv, &rv)| xv * rv.to_f32())
            .sum()
    };

    if let Some(sum) = dot_f32_f16_arch(&x, &row0_f16) {
        assert!((sum - scalar_f16(&row0_f16)).abs() <= 1e-3);
    }
    if let Some((sum0, sum1)) = dot2_f32_f16_arch(&x, &row0_f16, &row1_f16) {
        assert!((sum0 - scalar_f16(&row0_f16)).abs() <= 1e-3);
        assert!((sum1 - scalar_f16(&row1_f16)).abs() <= 1e-3);
    }
    if let Some((sum0, sum1, sum2)) = dot3_f32_f16_arch(&x, &row0_f16, &row1_f16, &row2_f16) {
        assert!((sum0 - scalar_f16(&row0_f16)).abs() <= 1e-3);
        assert!((sum1 - scalar_f16(&row1_f16)).abs() <= 1e-3);
        assert!((sum2 - scalar_f16(&row2_f16)).abs() <= 1e-3);
    }

    if let Some(sum) = dot_f32_bf16_arch(&x, &row0_bf16) {
        assert!((sum - scalar_bf16(&row0_bf16)).abs() <= 1e-2);
    }
    if let Some((sum0, sum1)) = dot2_f32_bf16_arch(&x, &row0_bf16, &row1_bf16) {
        assert!((sum0 - scalar_bf16(&row0_bf16)).abs() <= 1e-2);
        assert!((sum1 - scalar_bf16(&row1_bf16)).abs() <= 1e-2);
    }
    if let Some((sum0, sum1, sum2)) = dot3_f32_bf16_arch(&x, &row0_bf16, &row1_bf16, &row2_bf16) {
        assert!((sum0 - scalar_bf16(&row0_bf16)).abs() <= 1e-2);
        assert!((sum1 - scalar_bf16(&row1_bf16)).abs() <= 1e-2);
        assert!((sum2 - scalar_bf16(&row2_bf16)).abs() <= 1e-2);
    }

    let scalar_bf16_bf16 = |row: &[bf16]| -> f32 {
        x_bf16
            .iter()
            .zip(row.iter())
            .map(|(&xv, &rv)| xv.to_f32() * rv.to_f32())
            .sum()
    };
    if let Some(sum) = dot_bf16_bf16_arch(&x_bf16, &row0_bf16) {
        assert!((sum - scalar_bf16_bf16(&row0_bf16)).abs() <= 1e-2);
    }
    if let Some((sum0, sum1)) = dot2_bf16_bf16_arch(&x_bf16, &row0_bf16, &row1_bf16) {
        assert!((sum0 - scalar_bf16_bf16(&row0_bf16)).abs() <= 1e-2);
        assert!((sum1 - scalar_bf16_bf16(&row1_bf16)).abs() <= 1e-2);
    }
    if let Some((sum0, sum1, sum2)) =
        dot3_bf16_bf16_arch(&x_bf16, &row0_bf16, &row1_bf16, &row2_bf16)
    {
        assert!((sum0 - scalar_bf16_bf16(&row0_bf16)).abs() <= 1e-2);
        assert!((sum1 - scalar_bf16_bf16(&row1_bf16)).abs() <= 1e-2);
        assert!((sum2 - scalar_bf16_bf16(&row2_bf16)).abs() <= 1e-2);
    }

    let scalar_f16_f16 = |row: &[f16]| -> f32 {
        x_f16
            .iter()
            .zip(row.iter())
            .map(|(&xv, &rv)| xv.to_f32() * rv.to_f32())
            .sum()
    };
    if let Some(sum) = dot_f16_f16_arch(&x_f16, &row0_f16) {
        assert!((sum - scalar_f16_f16(&row0_f16)).abs() <= 1e-1);
    }
    if let Some((sum0, sum1)) = dot2_f16_f16_arch(&x_f16, &row0_f16, &row1_f16) {
        assert!((sum0 - scalar_f16_f16(&row0_f16)).abs() <= 1e-1);
        assert!((sum1 - scalar_f16_f16(&row1_f16)).abs() <= 1e-1);
    }
    if let Some((sum0, sum1, sum2)) = dot3_f16_f16_arch(&x_f16, &row0_f16, &row1_f16, &row2_f16) {
        assert!((sum0 - scalar_f16_f16(&row0_f16)).abs() <= 1e-1);
        assert!((sum1 - scalar_f16_f16(&row1_f16)).abs() <= 1e-1);
        assert!((sum2 - scalar_f16_f16(&row2_f16)).abs() <= 1e-1);
    }
}

#[test]
fn mixed_bf16_dots_preserve_f32_values_inside_vector_blocks() {
    for k in [
        0, 1, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 511, 512, 513, 1025,
    ] {
        let x = (0..k)
            .map(|i| 1.0 + ((i % 7) + 1) as f32 * 0.0001)
            .collect::<Vec<_>>();
        let rows = [
            vec![bf16::from_f32(1.0); k],
            (0..k)
                .map(|i| bf16::from_f32(if i % 2 == 0 { 1.0 } else { -1.0 }))
                .collect(),
            (0..k)
                .map(|i| bf16::from_f32((i as i32 % 11 - 5) as f32 * 0.09375))
                .collect(),
        ];
        let reference = rows.each_ref().map(|row| {
            x.iter()
                .zip(row)
                .map(|(&a, b)| a as f64 * b.to_f32() as f64)
                .sum::<f64>()
        });
        let budgets = rows.each_ref().map(|row| {
            2e-6 + 2e-7
                * x.iter()
                    .zip(row)
                    .map(|(&a, b)| (a as f64 * b.to_f32() as f64).abs())
                    .sum::<f64>()
        });
        let check = |actual: &[f32], count: usize| {
            for i in 0..count {
                assert!(
                    (actual[i] as f64 - reference[i]).abs() <= budgets[i],
                    "mixed BF16 K={k}, row={i}: {} vs {}",
                    actual[i],
                    reference[i]
                );
            }
        };
        if let Some(value) = dot_f32_bf16_arch(&x, &rows[0]) {
            check(&[value], 1);
            let (a, b) = dot2_f32_bf16_arch(&x, &rows[0], &rows[1]).unwrap();
            check(&[a, b], 2);
            let (a, b, c) = dot3_f32_bf16_arch(&x, &rows[0], &rows[1], &rows[2]).unwrap();
            check(&[a, b, c], 3);
        }
        #[cfg(all(
            feature = "x86-fp-kernels",
            any(target_arch = "x86_64", target_arch = "x86")
        ))]
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            // Check the narrower backend even on an AVX-512 machine.
            unsafe {
                check(&[dot_f32_bf16_x86_avx2(&x, &rows[0])], 1);
                let (a, b) = dot2_f32_bf16_x86_avx2(&x, &rows[0], &rows[1]);
                check(&[a, b], 2);
                let (a, b, c) = dot3_f32_bf16_x86_avx2(&x, &rows[0], &rows[1], &rows[2]);
                check(&[a, b, c], 3);
            }
        }
    }
}

#[test]
fn f16_update_avx2_fallbacks_match_scalar_reference_when_available() {
    #[cfg(all(
        feature = "x86-fp-kernels",
        any(target_arch = "x86_64", target_arch = "x86")
    ))]
    {
        if !arch::x86_f16c_kernel_runtime_available() {
            return;
        }

        let src = [
            1.0f32, -2.0, 0.5, 3.0, -4.5, 0.25, 7.0, -8.0, 9.0, -10.0, 0.125, -0.375, 2.5, -3.5,
            4.5, -5.5, 6.5, -7.5, 8.5,
        ];
        let grad = [
            0.5f32, -1.0, 2.0, 0.25, -0.75, 1.5, -2.5, 3.0, 0.125, -0.5, 0.75, -1.25, 1.75, -2.25,
            2.75, -3.25, 3.75, -4.25, 4.75,
        ];
        let lr = 0.05f32;

        let mut data = src.map(f16::from_f32);
        unsafe { sgd_update_f16_f32_x86_avx2(&mut data, &grad, lr) };
        for ((got, &w), &g) in data.iter().zip(src.iter()).zip(grad.iter()) {
            let expected = f16::from_f32(w - lr * g).to_f32();
            assert!((got.to_f32() - expected).abs() <= 1e-3);
        }

        let momentum = 0.875f32;
        let mut data = src.map(f16::from_f32);
        let mut velocity = src.map(|v| v * 0.01);
        let mut expected_velocity = velocity;
        let mut expected_data = data;
        for (((w, ev), &g), &src_v) in expected_data
            .iter_mut()
            .zip(expected_velocity.iter_mut())
            .zip(grad.iter())
            .zip(src.iter())
        {
            *ev = momentum * *ev + g;
            *w = f16::from_f32(src_v - lr * *ev);
        }
        unsafe {
            sgd_momentum_update_f16_f32_x86_avx2(&mut data, &mut velocity, &grad, lr, momentum)
        };
        for (got, expected) in data.iter().zip(expected_data.iter()) {
            assert!((got.to_f32() - expected.to_f32()).abs() <= 1e-3);
        }
        for (got, expected) in velocity.iter().zip(expected_velocity.iter()) {
            assert!((got - expected).abs() <= 1e-6);
        }

        let beta1 = 0.9f32;
        let beta2 = 0.999f32;
        let bias_correction1 = 0.1f32;
        let bias_correction2 = 0.001f32;
        let eps = 1e-8f32;
        let lr = 0.001f32;
        let mut data = src.map(f16::from_f32);
        let mut exp_avg = src.map(|v| v * 0.01);
        let mut exp_avg_sq = src.map(|v| v.abs() * 0.001 + 0.0001);
        let mut expected_data = data;
        let mut expected_m = exp_avg;
        let mut expected_v = exp_avg_sq;
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
            *w = f16::from_f32(w.to_f32() - lr * (m_hat / (v_hat.sqrt() + eps)));
        }
        unsafe {
            adam_update_f16_f32_x86_avx2(
                &mut data,
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
        for (got, expected) in data.iter().zip(expected_data.iter()) {
            assert!((got.to_f32() - expected.to_f32()).abs() <= 1e-3);
        }
        for (got, expected) in exp_avg.iter().zip(expected_m.iter()) {
            assert!((got - expected).abs() <= 1e-6);
        }
        for (got, expected) in exp_avg_sq.iter().zip(expected_v.iter()) {
            assert!((got - expected).abs() <= 1e-6);
        }
    }
}

#[test]
fn low_precision_elementwise_arch_paths_match_scalar_reference() {
    let f16_lhs = [
        1.0f32, -2.0, 0.5, 3.0, -4.5, 0.25, 7.0, -8.0, 9.0, -10.0, 0.125, -0.375, 2.5, -3.5, 4.5,
        -5.5, 6.5,
    ]
    .map(f16::from_f32);
    let f16_rhs = [
        0.5f32, -1.0, 2.0, 0.25, -0.75, 1.5, -2.5, 3.0, 0.125, -0.5, 0.75, -1.25, 1.75, -2.25,
        2.75, -3.25, 3.75,
    ]
    .map(f16::from_f32);
    let mut f16_out = [f16::from_f32(0.0); 17];

    if add_f16_f16_arch(&f16_lhs, &f16_rhs, &mut f16_out) {
        for ((&got, &a), &b) in f16_out.iter().zip(f16_lhs.iter()).zip(f16_rhs.iter()) {
            let expected = f16::from_f32(a.to_f32() + b.to_f32()).to_f32();
            assert!((got.to_f32() - expected).abs() <= 1e-3);
        }
    }
    if sub_f16_f16_arch(&f16_lhs, &f16_rhs, &mut f16_out) {
        for ((&got, &a), &b) in f16_out.iter().zip(f16_lhs.iter()).zip(f16_rhs.iter()) {
            let expected = f16::from_f32(a.to_f32() - b.to_f32()).to_f32();
            assert!((got.to_f32() - expected).abs() <= 1e-3);
        }
    }
    if mul_f16_f16_arch(&f16_lhs, &f16_rhs, &mut f16_out) {
        for ((&got, &a), &b) in f16_out.iter().zip(f16_lhs.iter()).zip(f16_rhs.iter()) {
            let expected = f16::from_f32(a.to_f32() * b.to_f32()).to_f32();
            assert!((got.to_f32() - expected).abs() <= 1e-3);
        }
    }

    let bf16_lhs = f16_lhs.map(|v| bf16::from_f32(v.to_f32()));
    let bf16_rhs = f16_rhs.map(|v| bf16::from_f32(v.to_f32()));
    let mut bf16_out = [bf16::from_f32(0.0); 17];

    if add_bf16_bf16_arch(&bf16_lhs, &bf16_rhs, &mut bf16_out) {
        for ((&got, &a), &b) in bf16_out.iter().zip(bf16_lhs.iter()).zip(bf16_rhs.iter()) {
            let expected = bf16::from_f32(a.to_f32() + b.to_f32()).to_f32();
            assert!((got.to_f32() - expected).abs() <= 1e-2);
        }
    }
    if sub_bf16_bf16_arch(&bf16_lhs, &bf16_rhs, &mut bf16_out) {
        for ((&got, &a), &b) in bf16_out.iter().zip(bf16_lhs.iter()).zip(bf16_rhs.iter()) {
            let expected = bf16::from_f32(a.to_f32() - b.to_f32()).to_f32();
            assert!((got.to_f32() - expected).abs() <= 1e-2);
        }
    }
    if mul_bf16_bf16_arch(&bf16_lhs, &bf16_rhs, &mut bf16_out) {
        for ((&got, &a), &b) in bf16_out.iter().zip(bf16_lhs.iter()).zip(bf16_rhs.iter()) {
            let expected = bf16::from_f32(a.to_f32() * b.to_f32()).to_f32();
            assert!((got.to_f32() - expected).abs() <= 1e-2);
        }
    }
}

#[test]
fn low_precision_elementwise_to_f32_arch_paths_match_scalar_reference() {
    let f16_lhs = [
        1.0f32, -2.0, 0.5, 3.0, -4.5, 0.25, 7.0, -8.0, 9.0, -10.0, 0.125, -0.375, 2.5, -3.5, 4.5,
        -5.5, 6.5,
    ]
    .map(f16::from_f32);
    let f16_rhs = [
        0.5f32, -1.0, 2.0, 0.25, -0.75, 1.5, -2.5, 3.0, 0.125, -0.5, 0.75, -1.25, 1.75, -2.25,
        2.75, -3.25, 3.75,
    ]
    .map(f16::from_f32);
    let mut out = [0.0f32; 17];

    if add_f16_f16_to_f32_arch(&f16_lhs, &f16_rhs, &mut out) {
        for ((&got, &a), &b) in out.iter().zip(f16_lhs.iter()).zip(f16_rhs.iter()) {
            assert!((got - (a.to_f32() + b.to_f32())).abs() <= 1e-6);
        }
    }
    if sub_f16_f16_to_f32_arch(&f16_lhs, &f16_rhs, &mut out) {
        for ((&got, &a), &b) in out.iter().zip(f16_lhs.iter()).zip(f16_rhs.iter()) {
            assert!((got - (a.to_f32() - b.to_f32())).abs() <= 1e-6);
        }
    }
    if mul_f16_f16_to_f32_arch(&f16_lhs, &f16_rhs, &mut out) {
        for ((&got, &a), &b) in out.iter().zip(f16_lhs.iter()).zip(f16_rhs.iter()) {
            assert!((got - (a.to_f32() * b.to_f32())).abs() <= 1e-6);
        }
    }

    let bf16_lhs = f16_lhs.map(|v| bf16::from_f32(v.to_f32()));
    let bf16_rhs = f16_rhs.map(|v| bf16::from_f32(v.to_f32()));
    if add_bf16_bf16_to_f32_arch(&bf16_lhs, &bf16_rhs, &mut out) {
        for ((&got, &a), &b) in out.iter().zip(bf16_lhs.iter()).zip(bf16_rhs.iter()) {
            assert!((got - (a.to_f32() + b.to_f32())).abs() <= 1e-6);
        }
    }
    if sub_bf16_bf16_to_f32_arch(&bf16_lhs, &bf16_rhs, &mut out) {
        for ((&got, &a), &b) in out.iter().zip(bf16_lhs.iter()).zip(bf16_rhs.iter()) {
            assert!((got - (a.to_f32() - b.to_f32())).abs() <= 1e-6);
        }
    }
    if mul_bf16_bf16_to_f32_arch(&bf16_lhs, &bf16_rhs, &mut out) {
        for ((&got, &a), &b) in out.iter().zip(bf16_lhs.iter()).zip(bf16_rhs.iter()) {
            assert!((got - (a.to_f32() * b.to_f32())).abs() <= 1e-6);
        }
    }
}

#[test]
fn low_precision_sum_arch_paths_match_scalar_reference() {
    let src = [
        1.0f32, -2.0, 0.5, 3.0, -4.5, 0.25, 7.0, -8.0, 9.0, -10.0, 0.125, -0.375, 2.5, -3.5, 4.5,
        -5.5, 6.5,
    ];
    let f16_data = src.map(f16::from_f32);
    if let Some(got) = sum_f16_arch(&f16_data) {
        let expected = f16_data.iter().map(|v| v.to_f32()).sum::<f32>();
        assert!((got - expected).abs() <= 1e-3);
    }

    let bf16_data = src.map(bf16::from_f32);
    if let Some(got) = sum_bf16_arch(&bf16_data) {
        let expected = bf16_data.iter().map(|v| v.to_f32()).sum::<f32>();
        assert!((got - expected).abs() <= 1e-2);
    }

    assert_eq!(sum_f16_arch(&[]), Some(0.0));
    assert_eq!(sum_bf16_arch(&[]), Some(0.0));
}

#[test]
fn low_precision_mul_grad_arch_paths_match_scalar_reference() {
    let grad = [
        0.5f32, -1.0, 2.0, 0.25, -0.75, 1.5, -2.5, 3.0, 0.125, -0.5, 0.75, -1.25, 1.75, -2.25,
        2.75, -3.25, 3.75,
    ];
    let src = [
        1.0f32, -2.0, 0.5, 3.0, -4.5, 0.25, 7.0, -8.0, 9.0, -10.0, 0.125, -0.375, 2.5, -3.5, 4.5,
        -5.5, 6.5,
    ];

    let f16_data = src.map(f16::from_f32);
    let mut f16_out = [0.0f32; 17];
    if mul_f32_f16_to_f32_arch(&grad, &f16_data, &mut f16_out) {
        for ((&got, &g), &x) in f16_out.iter().zip(grad.iter()).zip(f16_data.iter()) {
            assert!((got - g * x.to_f32()).abs() <= 1e-6);
        }
    }
    let scalar = -0.375f32;
    if mul_f16_scalar_to_f32_arch(&f16_data, scalar, &mut f16_out) {
        for ((&got, &x), &src) in f16_out.iter().zip(f16_data.iter()).zip(src.iter()) {
            assert!((got - x.to_f32() * scalar).abs() <= 1e-6);
            assert!((got - f16::from_f32(src).to_f32() * scalar).abs() <= 1e-6);
        }
    }
    if add_f16_scalar_to_f32_arch(&f16_data, scalar, &mut f16_out) {
        for (&got, &x) in f16_out.iter().zip(f16_data.iter()) {
            assert!((got - (scalar + x.to_f32())).abs() <= 1e-6);
        }
    }
    if sub_f16_scalar_to_f32_arch(&f16_data, scalar, false, &mut f16_out) {
        for (&got, &x) in f16_out.iter().zip(f16_data.iter()) {
            assert!((got - (scalar - x.to_f32())).abs() <= 1e-6);
        }
    }
    if sub_f16_scalar_to_f32_arch(&f16_data, scalar, true, &mut f16_out) {
        for (&got, &x) in f16_out.iter().zip(f16_data.iter()) {
            assert!((got - (x.to_f32() - scalar)).abs() <= 1e-6);
        }
    }
    assert!(!mul_f16_scalar_to_f32_arch(
        &f16_data,
        f32::NAN,
        &mut f16_out
    ));
    assert!(!sub_f16_scalar_to_f32_arch(
        &f16_data,
        f32::NAN,
        true,
        &mut f16_out
    ));

    let bf16_data = src.map(bf16::from_f32);
    let mut bf16_out = [0.0f32; 17];
    if mul_f32_bf16_to_f32_arch(&grad, &bf16_data, &mut bf16_out) {
        for ((&got, &g), &x) in bf16_out.iter().zip(grad.iter()).zip(bf16_data.iter()) {
            assert!((got - g * x.to_f32()).abs() <= 1e-6);
        }
    }
    if mul_bf16_scalar_to_f32_arch(&bf16_data, scalar, &mut bf16_out) {
        for ((&got, &x), &src) in bf16_out.iter().zip(bf16_data.iter()).zip(src.iter()) {
            assert!((got - x.to_f32() * scalar).abs() <= 1e-6);
            assert!((got - bf16::from_f32(src).to_f32() * scalar).abs() <= 1e-6);
        }
    }
    if add_bf16_scalar_to_f32_arch(&bf16_data, scalar, &mut bf16_out) {
        for (&got, &x) in bf16_out.iter().zip(bf16_data.iter()) {
            assert!((got - (scalar + x.to_f32())).abs() <= 1e-6);
        }
    }
    if sub_bf16_scalar_to_f32_arch(&bf16_data, scalar, false, &mut bf16_out) {
        for (&got, &x) in bf16_out.iter().zip(bf16_data.iter()) {
            assert!((got - (scalar - x.to_f32())).abs() <= 1e-6);
        }
    }
    if sub_bf16_scalar_to_f32_arch(&bf16_data, scalar, true, &mut bf16_out) {
        for (&got, &x) in bf16_out.iter().zip(bf16_data.iter()) {
            assert!((got - (x.to_f32() - scalar)).abs() <= 1e-6);
        }
    }
    assert!(!mul_bf16_scalar_to_f32_arch(
        &bf16_data,
        f32::NAN,
        &mut bf16_out
    ));
    assert!(!sub_bf16_scalar_to_f32_arch(
        &bf16_data,
        f32::NAN,
        true,
        &mut bf16_out
    ));
}

#[test]
fn bf16_update_avx2_fallbacks_match_scalar_reference_when_available() {
    #[cfg(all(
        feature = "x86-fp-kernels",
        any(target_arch = "x86_64", target_arch = "x86")
    ))]
    {
        if !arch::x86_fp_kernel_runtime_available() {
            return;
        }

        let src = [
            1.0f32, -2.0, 0.5, 3.0, -4.5, 0.25, 7.0, -8.0, 9.0, -10.0, 0.125, -0.375, 2.5, -3.5,
            4.5, -5.5, 6.5, -7.5, 8.5,
        ];
        let grad = [
            0.5f32, -1.0, 2.0, 0.25, -0.75, 1.5, -2.5, 3.0, 0.125, -0.5, 0.75, -1.25, 1.75, -2.25,
            2.75, -3.25, 3.75, -4.25, 4.75,
        ];
        let lr = 0.05f32;

        let mut data = src.map(bf16::from_f32);
        unsafe { sgd_update_bf16_f32_x86_avx2(&mut data, &grad, lr) };
        for ((got, &w), &g) in data.iter().zip(src.iter()).zip(grad.iter()) {
            let expected = bf16::from_f32(w - lr * g).to_f32();
            assert!((got.to_f32() - expected).abs() <= 1e-2);
        }

        let momentum = 0.875f32;
        let mut data = src.map(bf16::from_f32);
        let mut velocity = src.map(|v| v * 0.01);
        let mut expected_velocity = velocity;
        let mut expected_data = data;
        for (((w, ev), &g), &src_v) in expected_data
            .iter_mut()
            .zip(expected_velocity.iter_mut())
            .zip(grad.iter())
            .zip(src.iter())
        {
            *ev = momentum * *ev + g;
            *w = bf16::from_f32(src_v - lr * *ev);
        }
        unsafe {
            sgd_momentum_update_bf16_f32_x86_avx2(&mut data, &mut velocity, &grad, lr, momentum)
        };
        for (got, expected) in data.iter().zip(expected_data.iter()) {
            assert!((got.to_f32() - expected.to_f32()).abs() <= 1e-2);
        }
        for (got, expected) in velocity.iter().zip(expected_velocity.iter()) {
            assert!((got - expected).abs() <= 1e-6);
        }

        let beta1 = 0.9f32;
        let beta2 = 0.999f32;
        let bias_correction1 = 0.1f32;
        let bias_correction2 = 0.001f32;
        let eps = 1e-8f32;
        let lr = 0.001f32;
        let mut data = src.map(bf16::from_f32);
        let mut exp_avg = src.map(|v| v * 0.01);
        let mut exp_avg_sq = src.map(|v| v.abs() * 0.001 + 0.0001);
        let mut expected_data = data;
        let mut expected_m = exp_avg;
        let mut expected_v = exp_avg_sq;
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
            *w = bf16::from_f32(w.to_f32() - lr * (m_hat / (v_hat.sqrt() + eps)));
        }
        unsafe {
            adam_update_bf16_f32_x86_avx2(
                &mut data,
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
        for (got, expected) in data.iter().zip(expected_data.iter()) {
            assert!((got.to_f32() - expected.to_f32()).abs() <= 1e-2);
        }
        for (got, expected) in exp_avg.iter().zip(expected_m.iter()) {
            assert!((got - expected).abs() <= 1e-6);
        }
        for (got, expected) in exp_avg_sq.iter().zip(expected_v.iter()) {
            assert!((got - expected).abs() <= 1e-6);
        }
    }
}

#[test]
#[ignore = "performance sanity test; run with --ignored --nocapture"]
fn perf_f16_f16_dot_dot2_dot3_arch_paths() {
    use std::time::Instant;

    let len = 8192usize;
    let iters = 256usize;
    let x = (0..len)
        .map(|i| f16::from_f32(((i * 17 % 257) as f32 - 128.0) / 257.0))
        .collect::<Vec<_>>();
    let row0 = (0..len)
        .map(|i| f16::from_f32(((i * 29 % 251) as f32 - 125.0) / 251.0))
        .collect::<Vec<_>>();
    let row1 = (0..len)
        .map(|i| f16::from_f32(((i * 31 % 241) as f32 - 120.0) / 241.0))
        .collect::<Vec<_>>();
    let row2 = (0..len)
        .map(|i| f16::from_f32(((i * 37 % 239) as f32 - 119.0) / 239.0))
        .collect::<Vec<_>>();
    let grad = (0..len)
        .map(|i| ((i * 41 % 233) as f32 - 116.0) / 233.0)
        .collect::<Vec<_>>();
    let mut grad_out = vec![0.0f32; len];
    let mut fwd_out = vec![0.0f32; len];

    if dot_f16_f16_arch(&x, &row0).is_none() {
        eprintln!(
            "perf_f16_f16_dot_dot2_dot3 skipped backend={}",
            active_f16_f16_backend_name()
        );
        return;
    }

    let mut sink = 0.0f32;
    let start = Instant::now();
    for _ in 0..iters {
        sink += dot_f16_f16_arch(&x, &row0).expect("f16 dot arch");
    }
    let dot_elapsed = start.elapsed();

    let start = Instant::now();
    for _ in 0..iters {
        let (a, b) = dot2_f16_f16_arch(&x, &row0, &row1).expect("f16 dot2 arch");
        sink += a + b;
    }
    let dot2_elapsed = start.elapsed();

    let start = Instant::now();
    for _ in 0..iters {
        let (a, b, c) = dot3_f16_f16_arch(&x, &row0, &row1, &row2).expect("f16 dot3 arch");
        sink += a + b + c;
    }
    let dot3_elapsed = start.elapsed();

    let start = Instant::now();
    for _ in 0..iters {
        sink += sum_f16_arch(&x).expect("f16 sum arch");
    }
    let sum_elapsed = start.elapsed();

    let start = Instant::now();
    for _ in 0..iters {
        assert!(mul_f32_f16_to_f32_arch(&grad, &x, &mut grad_out));
        sink += grad_out[0];
    }
    let mul_grad_elapsed = start.elapsed();

    let start = Instant::now();
    for _ in 0..iters {
        assert!(mul_f16_f16_to_f32_arch(&x, &row0, &mut fwd_out));
        sink += fwd_out[0];
    }
    let fwd_mul_elapsed = start.elapsed();

    eprintln!(
        "perf_f16_f16_dot_dot2_dot3 backend={} len={len} iters={iters} dot={:.3}us dot2={:.3}us dot3={:.3}us sum={:.3}us mul_grad={:.3}us fwd_mul_f32={:.3}us sink={sink}",
        active_f16_f16_backend_name(),
        dot_elapsed.as_secs_f64() * 1e6 / iters as f64,
        dot2_elapsed.as_secs_f64() * 1e6 / iters as f64,
        dot3_elapsed.as_secs_f64() * 1e6 / iters as f64,
        sum_elapsed.as_secs_f64() * 1e6 / iters as f64,
        mul_grad_elapsed.as_secs_f64() * 1e6 / iters as f64,
        fwd_mul_elapsed.as_secs_f64() * 1e6 / iters as f64,
    );
}

#[test]
#[ignore = "performance sanity test; run with --ignored --nocapture"]
fn perf_bf16_bf16_dot_dot2_dot3_arch_paths() {
    use std::time::Instant;

    let len = 8192usize;
    let iters = 256usize;
    let x = (0..len)
        .map(|i| bf16::from_f32(((i * 17 % 257) as f32 - 128.0) / 257.0))
        .collect::<Vec<_>>();
    let row0 = (0..len)
        .map(|i| bf16::from_f32(((i * 29 % 251) as f32 - 125.0) / 251.0))
        .collect::<Vec<_>>();
    let row1 = (0..len)
        .map(|i| bf16::from_f32(((i * 31 % 241) as f32 - 120.0) / 241.0))
        .collect::<Vec<_>>();
    let row2 = (0..len)
        .map(|i| bf16::from_f32(((i * 37 % 239) as f32 - 119.0) / 239.0))
        .collect::<Vec<_>>();
    let grad = (0..len)
        .map(|i| ((i * 41 % 233) as f32 - 116.0) / 233.0)
        .collect::<Vec<_>>();
    let mut grad_out = vec![0.0f32; len];
    let mut fwd_out = vec![0.0f32; len];

    if dot_bf16_bf16_arch(&x, &row0).is_none() {
        eprintln!(
            "perf_bf16_bf16_dot_dot2_dot3 skipped backend={}",
            active_bf16_bf16_backend_name()
        );
        return;
    }

    let mut sink = 0.0f32;
    let start = Instant::now();
    for _ in 0..iters {
        sink += dot_bf16_bf16_arch(&x, &row0).expect("bf16 dot arch");
    }
    let dot_elapsed = start.elapsed();

    let start = Instant::now();
    for _ in 0..iters {
        let (a, b) = dot2_bf16_bf16_arch(&x, &row0, &row1).expect("bf16 dot2 arch");
        sink += a + b;
    }
    let dot2_elapsed = start.elapsed();

    let start = Instant::now();
    for _ in 0..iters {
        let (a, b, c) = dot3_bf16_bf16_arch(&x, &row0, &row1, &row2).expect("bf16 dot3 arch");
        sink += a + b + c;
    }
    let dot3_elapsed = start.elapsed();

    let start = Instant::now();
    for _ in 0..iters {
        sink += sum_bf16_arch(&x).expect("bf16 sum arch");
    }
    let sum_elapsed = start.elapsed();

    let start = Instant::now();
    for _ in 0..iters {
        assert!(mul_f32_bf16_to_f32_arch(&grad, &x, &mut grad_out));
        sink += grad_out[0];
    }
    let mul_grad_elapsed = start.elapsed();

    let start = Instant::now();
    for _ in 0..iters {
        assert!(mul_bf16_bf16_to_f32_arch(&x, &row0, &mut fwd_out));
        sink += fwd_out[0];
    }
    let fwd_mul_elapsed = start.elapsed();

    eprintln!(
        "perf_bf16_bf16_dot_dot2_dot3 backend={} len={len} iters={iters} dot={:.3}us dot2={:.3}us dot3={:.3}us sum={:.3}us mul_grad={:.3}us fwd_mul_f32={:.3}us sink={sink}",
        active_bf16_bf16_backend_name(),
        dot_elapsed.as_secs_f64() * 1e6 / iters as f64,
        dot2_elapsed.as_secs_f64() * 1e6 / iters as f64,
        dot3_elapsed.as_secs_f64() * 1e6 / iters as f64,
        sum_elapsed.as_secs_f64() * 1e6 / iters as f64,
        mul_grad_elapsed.as_secs_f64() * 1e6 / iters as f64,
        fwd_mul_elapsed.as_secs_f64() * 1e6 / iters as f64,
    );
}

#[test]
fn avx512_fp16_load_store_roundtrip_when_available() {
    #[cfg(all(
        feature = "x86-fp-kernels-nightly",
        any(target_arch = "x86_64", target_arch = "x86")
    ))]
    {
        if !arch::x86_avx512_fp16_kernel_runtime_available() {
            return;
        }

        let mut input = [f16::from_f32(0.0); 32];
        for (i, v) in input.iter_mut().enumerate() {
            *v = f16::from_f32((i as f32 - 11.0) * 0.125);
        }

        let loaded = unsafe { load_f16_x32_x86(input.as_ptr()) };
        let mut stored = [f16::from_f32(0.0); 32];
        unsafe { store_f16_x32_x86(stored.as_mut_ptr(), loaded) };

        for (lhs, rhs) in input.iter().zip(stored.iter()) {
            assert_eq!(lhs.to_bits(), rhs.to_bits());
        }
    }
}

#[test]
fn length_mismatch_disables_arch_fast_path() {
    let x = [1.0f32, 2.0, 3.0, 4.0];
    let short_f32 = [1.0f32, 2.0];
    let short_f16 = [f16::from_f32(1.0), f16::from_f32(2.0)];
    let short_bf16 = [bf16::from_f32(1.0), bf16::from_f32(2.0)];

    assert!(dot_f32_arch(&x, &short_f32).is_none());
    assert!(dot2_f32_arch(&x, &short_f32, &short_f32).is_none());
    assert!(dot3_f32_arch(&x, &short_f32, &short_f32, &short_f32).is_none());
    assert!(dot_f32_f16_arch(&x, &short_f16).is_none());
    assert!(dot2_f32_f16_arch(&x, &short_f16, &short_f16).is_none());
    assert!(dot3_f32_f16_arch(&x, &short_f16, &short_f16, &short_f16).is_none());
    assert!(dot_f32_bf16_arch(&x, &short_bf16).is_none());
    assert!(dot2_f32_bf16_arch(&x, &short_bf16, &short_bf16).is_none());
    assert!(dot3_f32_bf16_arch(&x, &short_bf16, &short_bf16, &short_bf16).is_none());
    assert!(dot_bf16_bf16_arch(&x.map(bf16::from_f32), &short_bf16).is_none());
    assert!(dot2_bf16_bf16_arch(&x.map(bf16::from_f32), &short_bf16, &short_bf16).is_none());
    assert!(
        dot3_bf16_bf16_arch(
            &x.map(bf16::from_f32),
            &short_bf16,
            &short_bf16,
            &short_bf16
        )
        .is_none()
    );
    assert!(dot_f16_f16_arch(&x.map(f16::from_f32), &short_f16).is_none());
    assert!(dot2_f16_f16_arch(&x.map(f16::from_f32), &short_f16, &short_f16).is_none());
    assert!(dot3_f16_f16_arch(&x.map(f16::from_f32), &short_f16, &short_f16, &short_f16).is_none());
    let mut short_update = short_f16;
    assert!(!sgd_update_f16_f32_arch(&mut short_update, &x, 0.1));
    let mut short_update = short_bf16;
    assert!(!sgd_update_bf16_f32_arch(&mut short_update, &x, 0.1));
    let mut velocity = x;
    let mut short_update = short_f16;
    assert!(!sgd_momentum_update_f16_f32_arch(
        &mut short_update,
        &mut velocity,
        &x,
        0.1,
        0.9
    ));
    let mut short_update = short_bf16;
    assert!(!sgd_momentum_update_bf16_f32_arch(
        &mut short_update,
        &mut velocity,
        &x,
        0.1,
        0.9
    ));
    let mut exp_avg = x;
    let mut exp_avg_sq = x;
    let mut short_update = short_f16;
    assert!(!adam_update_f16_f32_arch(
        &mut short_update,
        &mut exp_avg,
        &mut exp_avg_sq,
        &x,
        0.001,
        0.9,
        0.999,
        0.1,
        0.001,
        1e-8
    ));
    let mut short_update = short_bf16;
    assert!(!adam_update_bf16_f32_arch(
        &mut short_update,
        &mut exp_avg,
        &mut exp_avg_sq,
        &x,
        0.001,
        0.9,
        0.999,
        0.1,
        0.001,
        1e-8
    ));
}

#[test]
fn low_precision_sgd_update_fast_paths_match_scalar_reference() {
    let src = [
        1.0f32, -2.0, 0.5, 3.0, -4.5, 0.25, 7.0, -8.0, 9.0, -10.0, 0.125, -0.375, 2.5, -3.5, 4.5,
        -5.5, 6.5, -7.5, 8.5,
    ];
    let grad = [
        0.5f32, -1.0, 2.0, 0.25, -0.75, 1.5, -2.5, 3.0, 0.125, -0.5, 0.75, -1.25, 1.75, -2.25,
        2.75, -3.25, 3.75, -4.25, 4.75,
    ];
    let lr = 0.05f32;

    let mut f16_data = src.map(f16::from_f32);
    if sgd_update_f16_f32_arch(&mut f16_data, &grad, lr) {
        for ((got, &w), &g) in f16_data.iter().zip(src.iter()).zip(grad.iter()) {
            let expected = f16::from_f32(w - lr * g).to_f32();
            assert!((got.to_f32() - expected).abs() <= 1e-3);
        }
    }

    let mut bf16_data = src.map(bf16::from_f32);
    if sgd_update_bf16_f32_arch(&mut bf16_data, &grad, lr) {
        for ((got, &w), &g) in bf16_data.iter().zip(src.iter()).zip(grad.iter()) {
            let expected = bf16::from_f32(w - lr * g).to_f32();
            assert!((got.to_f32() - expected).abs() <= 1e-2);
        }
    }
}

#[test]
fn low_precision_momentum_update_fast_paths_match_scalar_reference() {
    let src = [
        1.0f32, -2.0, 0.5, 3.0, -4.5, 0.25, 7.0, -8.0, 9.0, -10.0, 0.125, -0.375, 2.5, -3.5, 4.5,
        -5.5, 6.5, -7.5, 8.5,
    ];
    let grad = [
        0.5f32, -1.0, 2.0, 0.25, -0.75, 1.5, -2.5, 3.0, 0.125, -0.5, 0.75, -1.25, 1.75, -2.25,
        2.75, -3.25, 3.75, -4.25, 4.75,
    ];
    let velocity_src = [
        0.125f32, -0.25, 0.375, -0.5, 0.625, -0.75, 0.875, -1.0, 1.125, -1.25, 1.375, -1.5, 1.625,
        -1.75, 1.875, -2.0, 2.125, -2.25, 2.375,
    ];
    let lr = 0.05f32;
    let momentum = 0.875f32;

    let mut f16_data = src.map(f16::from_f32);
    let mut f16_velocity = velocity_src;
    let mut expected_f16 = f16_data;
    let mut expected_velocity = velocity_src;
    for ((w, v), g) in expected_f16
        .iter_mut()
        .zip(expected_velocity.iter_mut())
        .zip(grad.iter())
    {
        *v = momentum * *v + *g;
        *w = f16::from_f32(w.to_f32() - lr * *v);
    }
    if sgd_momentum_update_f16_f32_arch(&mut f16_data, &mut f16_velocity, &grad, lr, momentum) {
        for (got, expected) in f16_data.iter().zip(expected_f16.iter()) {
            assert!((got.to_f32() - expected.to_f32()).abs() <= 1e-3);
        }
        for (got, expected) in f16_velocity.iter().zip(expected_velocity.iter()) {
            assert!((got - expected).abs() <= 1e-5);
        }
    }

    let mut bf16_data = src.map(bf16::from_f32);
    let mut bf16_velocity = velocity_src;
    let mut expected_bf16 = bf16_data;
    let mut expected_velocity = velocity_src;
    for ((w, v), g) in expected_bf16
        .iter_mut()
        .zip(expected_velocity.iter_mut())
        .zip(grad.iter())
    {
        *v = momentum * *v + *g;
        *w = bf16::from_f32(w.to_f32() - lr * *v);
    }
    if sgd_momentum_update_bf16_f32_arch(&mut bf16_data, &mut bf16_velocity, &grad, lr, momentum) {
        for (got, expected) in bf16_data.iter().zip(expected_bf16.iter()) {
            assert!((got.to_f32() - expected.to_f32()).abs() <= 1e-2);
        }
        for (got, expected) in bf16_velocity.iter().zip(expected_velocity.iter()) {
            assert!((got - expected).abs() <= 1e-5);
        }
    }
}

#[test]
fn low_precision_adam_update_fast_paths_match_scalar_reference() {
    let src = [
        1.0f32, -2.0, 0.5, 3.0, -4.5, 0.25, 7.0, -8.0, 9.0, -10.0, 0.125, -0.375, 2.5, -3.5, 4.5,
        -5.5, 6.5, -7.5, 8.5,
    ];
    let grad = [
        0.5f32, -1.0, 2.0, 0.25, -0.75, 1.5, -2.5, 3.0, 0.125, -0.5, 0.75, -1.25, 1.75, -2.25,
        2.75, -3.25, 3.75, -4.25, 4.75,
    ];
    let exp_avg_src = [
        0.01f32, -0.02, 0.03, -0.04, 0.05, -0.06, 0.07, -0.08, 0.09, -0.10, 0.11, -0.12, 0.13,
        -0.14, 0.15, -0.16, 0.17, -0.18, 0.19,
    ];
    let exp_avg_sq_src = [
        0.001f32, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.010, 0.011, 0.012,
        0.013, 0.014, 0.015, 0.016, 0.017, 0.018, 0.019,
    ];
    let lr = 0.001f32;
    let beta1 = 0.9f32;
    let beta2 = 0.999f32;
    let bias_correction1 = 0.1f32;
    let bias_correction2 = 0.001f32;
    let eps = 1e-8f32;

    let mut f16_data = src.map(f16::from_f32);
    let mut f16_m = exp_avg_src;
    let mut f16_v = exp_avg_sq_src;
    let mut expected_f16 = f16_data;
    let mut expected_m = exp_avg_src;
    let mut expected_v = exp_avg_sq_src;
    for (((w, m), v), g) in expected_f16
        .iter_mut()
        .zip(expected_m.iter_mut())
        .zip(expected_v.iter_mut())
        .zip(grad.iter())
    {
        *m = beta1 * *m + (1.0 - beta1) * *g;
        *v = beta2 * *v + (1.0 - beta2) * *g * *g;
        let m_hat = *m / bias_correction1;
        let v_hat = *v / bias_correction2;
        *w = f16::from_f32(w.to_f32() - lr * (m_hat / (v_hat.sqrt() + eps)));
    }
    if adam_update_f16_f32_arch(
        &mut f16_data,
        &mut f16_m,
        &mut f16_v,
        &grad,
        lr,
        beta1,
        beta2,
        bias_correction1,
        bias_correction2,
        eps,
    ) {
        for (got, expected) in f16_data.iter().zip(expected_f16.iter()) {
            assert!((got.to_f32() - expected.to_f32()).abs() <= 1e-3);
        }
        for (got, expected) in f16_m.iter().zip(expected_m.iter()) {
            assert!((got - expected).abs() <= 1e-5);
        }
        for (got, expected) in f16_v.iter().zip(expected_v.iter()) {
            assert!((got - expected).abs() <= 1e-5);
        }
    }

    let mut bf16_data = src.map(bf16::from_f32);
    let mut bf16_m = exp_avg_src;
    let mut bf16_v = exp_avg_sq_src;
    let mut expected_bf16 = bf16_data;
    let mut expected_m = exp_avg_src;
    let mut expected_v = exp_avg_sq_src;
    for (((w, m), v), g) in expected_bf16
        .iter_mut()
        .zip(expected_m.iter_mut())
        .zip(expected_v.iter_mut())
        .zip(grad.iter())
    {
        *m = beta1 * *m + (1.0 - beta1) * *g;
        *v = beta2 * *v + (1.0 - beta2) * *g * *g;
        let m_hat = *m / bias_correction1;
        let v_hat = *v / bias_correction2;
        *w = bf16::from_f32(w.to_f32() - lr * (m_hat / (v_hat.sqrt() + eps)));
    }
    if adam_update_bf16_f32_arch(
        &mut bf16_data,
        &mut bf16_m,
        &mut bf16_v,
        &grad,
        lr,
        beta1,
        beta2,
        bias_correction1,
        bias_correction2,
        eps,
    ) {
        for (got, expected) in bf16_data.iter().zip(expected_bf16.iter()) {
            assert!((got.to_f32() - expected.to_f32()).abs() <= 1e-2);
        }
        for (got, expected) in bf16_m.iter().zip(expected_m.iter()) {
            assert!((got - expected).abs() <= 1e-5);
        }
        for (got, expected) in bf16_v.iter().zip(expected_v.iter()) {
            assert!((got - expected).abs() <= 1e-5);
        }
    }
}
