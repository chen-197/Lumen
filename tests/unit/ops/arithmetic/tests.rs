use super::*;
use crate::autograd::no_grad;
use crate::precision::DType;
use ndarray::{Array, IxDyn};

fn logical_values(tensor: &Tensor) -> Vec<f32> {
    match tensor.native_storage_owned() {
        TensorStorageOwned::F32(a) => a.iter().copied().collect(),
        TensorStorageOwned::F16(a) => a.iter().map(|v| v.to_f32()).collect(),
        TensorStorageOwned::BF16(a) => a.iter().map(|v| v.to_f32()).collect(),
        TensorStorageOwned::I8(a, scale) => a.iter().map(|&v| v as f32 * scale).collect(),
    }
}

#[test]
fn cpu_binary_transposed_views_preserve_logical_element_order() {
    for dtype in [DType::F32, DType::F16, DType::BF16, DType::I8] {
        let lhs = no_grad(|| {
            make_tensor(&[2, 3], vec![1., 2., 3., 4., 5., 6.], dtype).permute(vec![1, 0])
        });
        let left = logical_values(&lhs);
        for rhs_dtype in [dtype, DType::F32] {
            for row_broadcast in [false, true] {
                let rhs = if row_broadcast {
                    make_tensor(&[2], vec![10., 20.], rhs_dtype)
                } else {
                    make_tensor(&[3, 2], vec![10., 20., 30., 40., 50., 60.], rhs_dtype)
                };
                let right = logical_values(&rhs);
                for op in [BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul] {
                    let output = no_grad(|| match op {
                        BinaryOp::Add => &lhs + &rhs,
                        BinaryOp::Sub => &lhs - &rhs,
                        BinaryOp::Mul => &lhs * &rhs,
                    });
                    let i8_rounding = match output.native_storage_owned() {
                        TensorStorageOwned::I8(_, scale) => Some(0.51 * scale + 1e-5),
                        _ => None,
                    };
                    for (i, actual) in logical_values(&output).into_iter().enumerate() {
                        let b = right[if row_broadcast { i % 2 } else { i }];
                        let expected = match op {
                            BinaryOp::Add => left[i] + b,
                            BinaryOp::Sub => left[i] - b,
                            BinaryOp::Mul => left[i] * b,
                        };
                        let tolerance = if let Some(rounding) = i8_rounding {
                            rounding
                        } else if dtype == DType::F32 {
                            1e-5
                        } else {
                            0.01 * expected.abs() + 0.05
                        };
                        assert!(
                            (actual - expected).abs() <= tolerance,
                            "{dtype:?}/{rhs_dtype:?} {op:?}, broadcast={row_broadcast}, element={i}: {actual} vs {expected}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn cpu_transposed_mul_backward_uses_logical_axes() {
    let a = Array::from_shape_vec(IxDyn(&[2, 3]), vec![1., 2., 3., 4., 5., 6.])
        .unwrap()
        .reversed_axes();
    let lhs = Tensor::new(a);
    let rhs =
        make_training_tensor_with_dtype(&[3, 2], vec![10., 20., 30., 40., 50., 60.], DType::F32);
    let product = &lhs * &rhs;
    assert_eq!(
        logical_values(&product),
        vec![10., 80., 60., 200., 150., 360.]
    );
    sum(&product).backward();
    assert_eq!(
        lhs.grad().unwrap().iter().copied().collect::<Vec<_>>(),
        vec![10., 20., 30., 40., 50., 60.]
    );
    assert_eq!(
        rhs.grad().unwrap().iter().copied().collect::<Vec<_>>(),
        vec![1., 4., 2., 5., 3., 6.]
    );
}

fn make_tensor(shape: &[usize], data: Vec<f32>, dtype: DType) -> Tensor {
    let t = Tensor::from_array_no_grad(
        Array::from_shape_vec(IxDyn(shape), data)
            .expect("test tensor shape mismatch")
            .into_dyn(),
    );
    t.cast_inplace(dtype);
    t
}

fn make_training_tensor_with_dtype(shape: &[usize], data: Vec<f32>, dtype: DType) -> Tensor {
    let t = Tensor::from_data_with_grad_flag(
        Array::from_shape_vec(IxDyn(shape), data)
            .expect("test tensor shape mismatch")
            .into_dyn(),
        true,
    );
    t.cast_inplace(dtype);
    t
}

#[cfg(feature = "cuda")]
fn make_training_tensor(shape: &[usize], data: Vec<f32>) -> Tensor {
    Tensor::from_data_with_grad_flag(
        Array::from_shape_vec(IxDyn(shape), data)
            .expect("test tensor shape mismatch")
            .into_dyn(),
        true,
    )
}

#[cfg(feature = "cuda")]
fn assert_cuda_lowp_typed_output(out: &Tensor, dtype: DType, shape: &[usize]) {
    assert!(out.is_cuda());
    assert_eq!(out.dtype(), dtype);
    assert_eq!(out.shape_vec(), shape);
    assert!(!out.has_host_f32_data());
    let inner = out.0.borrow();
    assert!(inner.cuda_f32_data.is_none());
    match dtype {
        DType::F16 => assert!(inner.cuda_f16_data.is_some()),
        DType::BF16 => assert!(inner.cuda_bf16_data.is_some()),
        DType::I8 => {
            assert!(inner.cuda_i8_data.is_some());
            assert!(inner.i8_scale.is_some());
        }
        DType::F32 => panic!("expected lowp typed output, got F32"),
    }
}

#[test]
fn bf16_add_no_grad_preserves_dtype_and_inputs() {
    let lhs = make_tensor(&[2], vec![1.0, -2.0], DType::BF16);
    let rhs = make_tensor(&[2], vec![0.5, 3.0], DType::BF16);

    let out = no_grad(|| lhs.clone() + rhs.clone());

    assert_eq!(lhs.dtype(), DType::BF16);
    assert_eq!(rhs.dtype(), DType::BF16);
    assert_eq!(out.dtype(), DType::BF16);
    out.with_storage_view(|view| match view {
        TensorStorageView::BF16(view) => {
            let vals = view.iter().map(|v| v.to_f32()).collect::<Vec<_>>();
            let expected = vec![bf16::from_f32(1.5).to_f32(), bf16::from_f32(1.0).to_f32()];
            assert_eq!(vals, expected);
        }
        TensorStorageView::F16(_) => panic!("bf16 add output should stay bf16 in no-grad"),
        TensorStorageView::F32(_) => panic!("bf16 add output should stay bf16 in no-grad"),
    });
}

#[test]
fn mixed_add_no_grad_promotes_to_f32_without_mutating_bf16_input() {
    let lhs = make_tensor(&[2], vec![1.0, -2.0], DType::BF16);
    let rhs = make_tensor(&[2], vec![0.5, 3.0], DType::F32);

    let out = no_grad(|| lhs.clone() + rhs.clone());

    assert_eq!(lhs.dtype(), DType::BF16);
    assert_eq!(rhs.dtype(), DType::F32);
    assert_eq!(out.dtype(), DType::F32);
    let vals = out.data_ref().iter().copied().collect::<Vec<_>>();
    assert_eq!(vals, vec![1.5, 1.0]);
}

#[test]
fn mixed_add_sub_no_grad_reads_native_low_precision_without_f32_cache() {
    for dtype in [DType::F16, DType::BF16, DType::I8] {
        let lowp = make_tensor(&[2, 3], vec![1.0, -2.0, 0.5, 3.0, -4.0, 2.0], dtype);
        let f32_same = make_tensor(&[2, 3], vec![0.5, 3.0, -1.5, 0.25, -0.75, 2.0], DType::F32);
        let f32_row = make_tensor(&[3], vec![0.5, 3.0, -1.5], DType::F32);
        let f32_row_scalar = make_tensor(&[2, 1], vec![0.5, -1.25], DType::F32);
        let f32_scalar = make_tensor(&[1], vec![-0.75], DType::F32);

        assert!(!lowp.has_host_f32_data());
        let same_add = no_grad(|| f32_same.clone() + lowp.clone());
        let same_sub = no_grad(|| lowp.clone() - f32_same.clone());
        let row_sub = no_grad(|| f32_row.clone() - lowp.clone());
        let row_scalar_add = no_grad(|| lowp.clone() + f32_row_scalar);
        let scalar_sub = no_grad(|| f32_scalar - lowp.clone());
        assert!(!lowp.has_host_f32_data());

        for out in [&same_add, &same_sub, &row_sub, &row_scalar_add, &scalar_sub] {
            assert_eq!(out.dtype(), DType::F32);
            assert_eq!(out.shape_vec(), vec![2, 3]);
        }

        for (got, expected) in same_add
            .data_ref()
            .iter()
            .zip([1.5f32, 1.0, -1.0, 3.25, -4.75, 4.0])
        {
            assert!((got - expected).abs() <= 0.08);
        }
        for (got, expected) in same_sub
            .data_ref()
            .iter()
            .zip([0.5f32, -5.0, 2.0, 2.75, -3.25, 0.0])
        {
            assert!((got - expected).abs() <= 0.08);
        }
        for (got, expected) in row_sub
            .data_ref()
            .iter()
            .zip([-0.5f32, 5.0, -2.0, -2.5, 7.0, -3.5])
        {
            assert!((got - expected).abs() <= 0.08);
        }
        for (got, expected) in row_scalar_add
            .data_ref()
            .iter()
            .zip([1.5f32, -1.5, 1.0, 1.75, -5.25, 0.75])
        {
            assert!((got - expected).abs() <= 0.08);
        }
        for (got, expected) in scalar_sub
            .data_ref()
            .iter()
            .zip([-1.75f32, 1.25, -1.25, -3.75, 3.25, -2.75])
        {
            assert!((got - expected).abs() <= 0.08);
        }
    }
}

#[test]
fn mixed_add_sub_training_forward_reads_native_low_precision_and_outputs_f32() {
    for dtype in [DType::F16, DType::BF16, DType::I8] {
        let lowp =
            make_training_tensor_with_dtype(&[2, 3], vec![1.0, -2.0, 0.5, 3.0, -4.0, 2.0], dtype);
        let f32_same = make_training_tensor_with_dtype(
            &[2, 3],
            vec![0.5, 3.0, -1.5, 0.25, -0.75, 2.0],
            DType::F32,
        );
        let f32_row = make_training_tensor_with_dtype(&[3], vec![0.5, 3.0, -1.5], DType::F32);
        let f32_scalar = make_training_tensor_with_dtype(&[1], vec![-0.75], DType::F32);

        assert!(!lowp.has_host_f32_data());
        let same_add = lowp.clone() + f32_same.clone();
        let same_sub = f32_same - lowp.clone();
        let row_sub = f32_row - lowp.clone();
        let scalar_add = lowp.clone() + f32_scalar;
        assert!(!lowp.has_host_f32_data());

        for out in [&same_add, &same_sub, &row_sub, &scalar_add] {
            assert_eq!(out.dtype(), DType::F32);
            assert!(out.has_host_f32_data());
            assert_eq!(out.shape_vec(), vec![2, 3]);
        }
    }
}

#[test]
fn mixed_add_sub_directional_broadcasts_match_native_reference() {
    fn decoded_lowp(t: &Tensor) -> Vec<f32> {
        match t.native_storage_owned() {
            TensorStorageOwned::F16(data) => data.iter().map(|v| v.to_f32()).collect(),
            TensorStorageOwned::BF16(data) => data.iter().map(|v| v.to_f32()).collect(),
            TensorStorageOwned::I8(data, scale) => {
                data.iter().map(|&v| (v as f32) * scale).collect()
            }
            TensorStorageOwned::F32(_) => panic!("tensor should stay native low precision"),
        }
    }

    fn assert_close(out: &Tensor, expected: &[f32]) {
        assert_eq!(out.dtype(), DType::F32);
        let vals = out.data_ref();
        assert_eq!(vals.len(), expected.len());
        for (&got, &want) in vals.iter().zip(expected) {
            assert!((got - want).abs() <= 1e-6, "got {got}, expected {want}");
        }
    }

    fn matrix_minus_row(matrix: &[f32], row: &[f32]) -> Vec<f32> {
        matrix
            .chunks_exact(row.len())
            .flat_map(|chunk| chunk.iter().zip(row).map(|(&a, &b)| a - b))
            .collect()
    }

    fn row_minus_matrix(row: &[f32], matrix: &[f32]) -> Vec<f32> {
        matrix
            .chunks_exact(row.len())
            .flat_map(|chunk| row.iter().zip(chunk).map(|(&a, &b)| a - b))
            .collect()
    }

    fn matrix_minus_row_scalar(matrix: &[f32], scalars: &[f32], row_len: usize) -> Vec<f32> {
        matrix
            .chunks_exact(row_len)
            .zip(scalars)
            .flat_map(|(row, &scalar)| row.iter().map(move |&v| v - scalar))
            .collect()
    }

    fn row_scalar_minus_matrix(scalars: &[f32], matrix: &[f32], row_len: usize) -> Vec<f32> {
        matrix
            .chunks_exact(row_len)
            .zip(scalars)
            .flat_map(|(row, &scalar)| row.iter().map(move |&v| scalar - v))
            .collect()
    }

    let matrix_vals = vec![1.0f32, -2.0, 0.5, 3.0, -4.0, 2.0];
    let f32_matrix_vals = vec![0.25f32, -1.5, 2.0, 0.75, 3.0, -0.5];
    let row_vals = vec![0.5f32, 3.0, -1.5];
    let f32_row_vals = vec![1.25f32, -0.75, 2.5];
    let row_scalar_vals = vec![0.5f32, -1.25];
    let f32_row_scalar_vals = vec![-0.25f32, 2.0];
    let vector_vals = vec![1.0f32, -2.0, 0.5, 3.0];
    let f32_vector_vals = vec![0.25f32, -1.5, 2.0, 0.75];
    let f32_scalar_val = -0.75f32;

    for dtype in [DType::F16, DType::BF16, DType::I8] {
        let lowp_matrix = make_tensor(&[2, 3], matrix_vals.clone(), dtype);
        let f32_matrix = make_tensor(&[2, 3], f32_matrix_vals.clone(), DType::F32);
        let lowp_matrix_vals = decoded_lowp(&lowp_matrix);
        assert!(!lowp_matrix.has_host_f32_data());
        assert_close(
            &no_grad(|| lowp_matrix.clone() - f32_matrix.clone()),
            &lowp_matrix_vals
                .iter()
                .zip(f32_matrix_vals.iter())
                .map(|(&a, &b)| a - b)
                .collect::<Vec<_>>(),
        );
        assert_close(
            &no_grad(|| f32_matrix.clone() - lowp_matrix.clone()),
            &f32_matrix_vals
                .iter()
                .zip(lowp_matrix_vals.iter())
                .map(|(&a, &b)| a - b)
                .collect::<Vec<_>>(),
        );
        assert!(!lowp_matrix.has_host_f32_data());

        let lowp_row = make_tensor(&[3], row_vals.clone(), dtype);
        let lowp_row_vals = decoded_lowp(&lowp_row);
        assert!(!lowp_row.has_host_f32_data());
        assert_close(
            &no_grad(|| lowp_matrix.clone() - make_tensor(&[3], f32_row_vals.clone(), DType::F32)),
            &matrix_minus_row(&lowp_matrix_vals, &f32_row_vals),
        );
        assert_close(
            &no_grad(|| make_tensor(&[3], f32_row_vals.clone(), DType::F32) - lowp_matrix.clone()),
            &row_minus_matrix(&f32_row_vals, &lowp_matrix_vals),
        );
        assert_close(
            &no_grad(|| f32_matrix.clone() - lowp_row.clone()),
            &matrix_minus_row(&f32_matrix_vals, &lowp_row_vals),
        );
        assert_close(
            &no_grad(|| lowp_row.clone() - f32_matrix.clone()),
            &row_minus_matrix(&lowp_row_vals, &f32_matrix_vals),
        );
        assert!(!lowp_matrix.has_host_f32_data());
        assert!(!lowp_row.has_host_f32_data());

        let lowp_row_scalar = make_tensor(&[2, 1], row_scalar_vals.clone(), dtype);
        let lowp_row_scalar_vals = decoded_lowp(&lowp_row_scalar);
        assert!(!lowp_row_scalar.has_host_f32_data());
        assert_close(
            &no_grad(|| {
                lowp_matrix.clone() - make_tensor(&[2, 1], f32_row_scalar_vals.clone(), DType::F32)
            }),
            &matrix_minus_row_scalar(&lowp_matrix_vals, &f32_row_scalar_vals, 3),
        );
        assert_close(
            &no_grad(|| {
                make_tensor(&[2, 1], f32_row_scalar_vals.clone(), DType::F32) - lowp_matrix.clone()
            }),
            &row_scalar_minus_matrix(&f32_row_scalar_vals, &lowp_matrix_vals, 3),
        );
        assert_close(
            &no_grad(|| f32_matrix.clone() - lowp_row_scalar.clone()),
            &matrix_minus_row_scalar(&f32_matrix_vals, &lowp_row_scalar_vals, 3),
        );
        assert_close(
            &no_grad(|| lowp_row_scalar.clone() - f32_matrix.clone()),
            &row_scalar_minus_matrix(&lowp_row_scalar_vals, &f32_matrix_vals, 3),
        );
        assert!(!lowp_row_scalar.has_host_f32_data());

        let lowp_vector = make_tensor(&[4], vector_vals.clone(), dtype);
        let lowp_scalar = make_tensor(&[1], vec![1.25], dtype);
        let lowp_vector_vals = decoded_lowp(&lowp_vector);
        let lowp_scalar_val = decoded_lowp(&lowp_scalar)[0];
        assert_close(
            &no_grad(|| lowp_vector.clone() - make_tensor(&[1], vec![f32_scalar_val], DType::F32)),
            &lowp_vector_vals
                .iter()
                .map(|&v| v - f32_scalar_val)
                .collect::<Vec<_>>(),
        );
        assert_close(
            &no_grad(|| make_tensor(&[1], vec![f32_scalar_val], DType::F32) - lowp_vector.clone()),
            &lowp_vector_vals
                .iter()
                .map(|&v| f32_scalar_val - v)
                .collect::<Vec<_>>(),
        );
        assert_close(
            &no_grad(|| {
                make_tensor(&[4], f32_vector_vals.clone(), DType::F32) - lowp_scalar.clone()
            }),
            &f32_vector_vals
                .iter()
                .map(|&v| v - lowp_scalar_val)
                .collect::<Vec<_>>(),
        );
        assert_close(
            &no_grad(|| {
                lowp_scalar.clone() - make_tensor(&[4], f32_vector_vals.clone(), DType::F32)
            }),
            &f32_vector_vals
                .iter()
                .map(|&v| lowp_scalar_val - v)
                .collect::<Vec<_>>(),
        );
        assert!(!lowp_vector.has_host_f32_data());
        assert!(!lowp_scalar.has_host_f32_data());
    }
}

#[test]
#[ignore]
fn cpu_mixed_lowp_binary_forward_perf_smoke() {
    fn bench<F>(iters: usize, mut f: F) -> (f64, f32)
    where
        F: FnMut() -> Tensor,
    {
        let mut sink = 0.0f32;
        let start = std::time::Instant::now();
        for _ in 0..iters {
            let out = f();
            let vals = out.data_ref();
            let first = *vals.iter().next().expect("perf output should be non-empty");
            let last = *vals.iter().last().expect("perf output should be non-empty");
            sink += first + last;
        }
        (start.elapsed().as_secs_f64() * 1.0e6 / iters as f64, sink)
    }

    let len = 1 << 20;
    let iters = 24;
    let f32_vals = (0..len)
        .map(|i| (i as f32 % 251.0) / 37.0 - 3.0)
        .collect::<Vec<_>>();
    let lowp_vals = (0..len)
        .map(|i| ((i * 17) as f32 % 241.0) / 41.0 - 2.5)
        .collect::<Vec<_>>();
    let rows = 4096;
    let row_len = len / rows;
    let row_vals = lowp_vals[..row_len].to_vec();

    for dtype in [DType::F16, DType::BF16, DType::I8] {
        let f32_same = make_tensor(&[len], f32_vals.clone(), DType::F32);
        let lowp_same = make_tensor(&[len], lowp_vals.clone(), dtype);
        let f32_matrix = make_tensor(&[rows, row_len], f32_vals.clone(), DType::F32);
        let lowp_row = make_tensor(&[row_len], row_vals.clone(), dtype);
        let lowp_matrix = make_tensor(&[rows, row_len], lowp_vals.clone(), dtype);
        let f32_row = make_tensor(&[row_len], f32_vals[..row_len].to_vec(), DType::F32);
        let f32_row_scalar = make_tensor(
            &[rows, 1],
            (0..rows).map(|i| (i as f32 % 127.0) / 31.0 - 2.0).collect(),
            DType::F32,
        );
        let f32_scalar = make_tensor(&[1], vec![-0.375], DType::F32);

        assert!(!lowp_same.has_host_f32_data());
        assert!(!lowp_row.has_host_f32_data());
        assert!(!lowp_matrix.has_host_f32_data());

        let (same_add_us, same_add_sink) =
            bench(iters, || no_grad(|| f32_same.clone() + lowp_same.clone()));
        let (same_sub_us, same_sub_sink) =
            bench(iters, || no_grad(|| lowp_same.clone() - f32_same.clone()));
        let (same_mul_us, same_mul_sink) =
            bench(iters, || no_grad(|| f32_same.clone() * lowp_same.clone()));
        let (row_add_us, row_add_sink) =
            bench(iters, || no_grad(|| f32_matrix.clone() + lowp_row.clone()));
        let (row_sub_us, row_sub_sink) =
            bench(iters, || no_grad(|| f32_row.clone() - lowp_matrix.clone()));
        let (row_mul_us, row_mul_sink) =
            bench(iters, || no_grad(|| f32_matrix.clone() * lowp_row.clone()));
        let (row_scalar_sub_us, row_scalar_sub_sink) = bench(iters, || {
            no_grad(|| lowp_matrix.clone() - f32_row_scalar.clone())
        });
        let (row_scalar_mul_us, row_scalar_mul_sink) = bench(iters, || {
            no_grad(|| lowp_matrix.clone() * f32_row_scalar.clone())
        });
        let (scalar_add_us, scalar_add_sink) =
            bench(iters, || no_grad(|| f32_scalar.clone() + lowp_same.clone()));
        let (scalar_sub_us, scalar_sub_sink) =
            bench(iters, || no_grad(|| f32_scalar.clone() - lowp_same.clone()));

        assert!(!lowp_same.has_host_f32_data());
        assert!(!lowp_row.has_host_f32_data());
        assert!(!lowp_matrix.has_host_f32_data());

        println!(
            "cpu mixed lowp binary dtype={dtype:?} len={len} rows={rows} row_len={row_len} iters={iters} same_add={same_add_us:.3}us same_sub={same_sub_us:.3}us same_mul={same_mul_us:.3}us row_add={row_add_us:.3}us row_sub={row_sub_us:.3}us row_mul={row_mul_us:.3}us row_scalar_sub={row_scalar_sub_us:.3}us row_scalar_mul={row_scalar_mul_us:.3}us scalar_add={scalar_add_us:.3}us scalar_sub={scalar_sub_us:.3}us sink={:.5}",
            same_add_sink
                + same_sub_sink
                + same_mul_sink
                + row_add_sink
                + row_sub_sink
                + row_mul_sink
                + row_scalar_sub_sink
                + row_scalar_mul_sink
                + scalar_add_sink
                + scalar_sub_sink
        );
    }
}

#[test]
fn mixed_mul_no_grad_reads_native_low_precision_without_f32_cache() {
    for dtype in [DType::F16, DType::BF16, DType::I8] {
        let lowp = make_tensor(&[2, 3], vec![1.0, -2.0, 0.5, 3.0, -4.0, 2.0], dtype);
        let f32_same = make_tensor(&[2, 3], vec![0.5, 3.0, -1.5, 0.25, -0.75, 2.0], DType::F32);
        let f32_row = make_tensor(&[3], vec![0.5, 3.0, -1.5], DType::F32);

        assert!(!lowp.has_host_f32_data());
        let same_out = no_grad(|| lowp.clone() * f32_same.clone());
        let row_out = no_grad(|| f32_row.clone() * lowp.clone());
        assert!(!lowp.has_host_f32_data());

        assert_eq!(same_out.dtype(), DType::F32);
        assert_eq!(row_out.dtype(), DType::F32);
        assert_eq!(same_out.shape_vec(), vec![2, 3]);
        assert_eq!(row_out.shape_vec(), vec![2, 3]);

        let same_vals = same_out.data_ref().iter().copied().collect::<Vec<_>>();
        let row_vals = row_out.data_ref().iter().copied().collect::<Vec<_>>();
        for (got, expected) in same_vals.iter().zip([0.5f32, -6.0, -0.75, 0.75, 3.0, 4.0]) {
            assert!((got - expected).abs() <= 0.08);
        }
        for (got, expected) in row_vals.iter().zip([0.5f32, -6.0, -0.75, 1.5, -12.0, -3.0]) {
            assert!((got - expected).abs() <= 0.08);
        }
    }
}

#[test]
fn mixed_mul_training_forward_reads_native_low_precision_and_outputs_f32() {
    for dtype in [DType::F16, DType::BF16, DType::I8] {
        let lowp =
            make_training_tensor_with_dtype(&[2, 3], vec![1.0, -2.0, 0.5, 3.0, -4.0, 2.0], dtype);
        let f32_same = make_training_tensor_with_dtype(
            &[2, 3],
            vec![0.5, 3.0, -1.5, 0.25, -0.75, 2.0],
            DType::F32,
        );
        let f32_row = make_training_tensor_with_dtype(&[3], vec![0.5, 3.0, -1.5], DType::F32);

        assert!(!lowp.has_host_f32_data());
        let same_out = lowp.clone() * f32_same;
        let row_out = f32_row * lowp.clone();
        assert!(!lowp.has_host_f32_data());

        for out in [&same_out, &row_out] {
            assert_eq!(out.dtype(), DType::F32);
            assert!(out.has_host_f32_data());
            assert_eq!(out.shape_vec(), vec![2, 3]);
        }
    }
}

#[test]
fn mixed_mul_scalar_broadcast_forward_reads_native_low_precision() {
    for dtype in [DType::F16, DType::BF16, DType::I8] {
        let lowp = make_tensor(&[2, 3], vec![1.0, -2.0, 0.5, 3.0, -4.0, 2.0], dtype);
        let f32_row_scalar = make_tensor(&[2, 1], vec![0.5, -1.25], DType::F32);

        assert!(!lowp.has_host_f32_data());
        let row_scalar_out = no_grad(|| lowp.clone() * f32_row_scalar);
        assert!(!lowp.has_host_f32_data());

        assert_eq!(row_scalar_out.dtype(), DType::F32);
        assert_eq!(row_scalar_out.shape_vec(), vec![2, 3]);
        for (got, expected) in row_scalar_out
            .data_ref()
            .iter()
            .zip([0.5f32, -1.0, 0.25, -3.75, 5.0, -2.5])
        {
            assert!((got - expected).abs() <= 0.08);
        }

        let lowp = make_tensor(&[2, 3], vec![1.0, -2.0, 0.5, 3.0, -4.0, 2.0], dtype);
        let f32_scalar = make_tensor(&[1], vec![-0.75], DType::F32);

        assert!(!lowp.has_host_f32_data());
        let scalar_out = no_grad(|| f32_scalar * lowp.clone());
        assert!(!lowp.has_host_f32_data());

        assert_eq!(scalar_out.dtype(), DType::F32);
        assert_eq!(scalar_out.shape_vec(), vec![2, 3]);
        for (got, expected) in scalar_out
            .data_ref()
            .iter()
            .zip([-0.75f32, 1.5, -0.375, -2.25, 3.0, -1.5])
        {
            assert!((got - expected).abs() <= 0.08);
        }
    }
}

#[test]
fn mixed_mul_backward_same_shape_reads_native_low_precision_and_writes_f32_grads() {
    let grad = Array::from_shape_vec(IxDyn(&[2, 3]), vec![1.0f32, -0.5, 0.25, 2.0, -1.0, 0.5])
        .expect("grad shape mismatch")
        .into_dyn();
    let grad_slice = grad.as_slice().unwrap();
    let lowp_vals = vec![1.0f32, -2.0, 0.5, 3.0, -4.0, 2.0];
    let f32_vals = vec![0.25f32, -1.5, 2.0, 0.75, 3.0, -0.5];

    fn decoded_lowp(t: &Tensor) -> Vec<f32> {
        match t.native_storage_owned() {
            TensorStorageOwned::F16(data) => data.iter().map(|v| v.to_f32()).collect(),
            TensorStorageOwned::BF16(data) => data.iter().map(|v| v.to_f32()).collect(),
            TensorStorageOwned::I8(data, scale) => {
                data.iter().map(|&v| (v as f32) * scale).collect()
            }
            TensorStorageOwned::F32(_) => panic!("tensor should stay native low precision"),
        }
    }

    for dtype in [DType::F16, DType::BF16, DType::I8] {
        let lowp = make_tensor(&[2, 3], lowp_vals.clone(), dtype);
        let f32_operand = make_tensor(&[2, 3], f32_vals.clone(), DType::F32);
        assert!(!lowp.has_host_f32_data());
        lowp.0.borrow_mut().requires_grad = true;
        f32_operand.0.borrow_mut().requires_grad = true;
        add_cpu_binary_grads(&lowp, &f32_operand, grad.view(), BinaryOp::Mul);
        assert!(!lowp.has_host_f32_data());

        let lowp_decoded = decoded_lowp(&lowp);
        let lowp_grad = lowp
            .grad_ref()
            .as_ref()
            .expect("lowp grad missing")
            .to_owned();
        let f32_grad = f32_operand
            .grad_ref()
            .as_ref()
            .expect("f32 grad missing")
            .to_owned();
        for (((&lowp_got, &f32_got), &g), (&f32_x, &lowp_x)) in lowp_grad
            .iter()
            .zip(f32_grad.iter())
            .zip(grad_slice.iter())
            .zip(f32_vals.iter().zip(lowp_decoded.iter()))
        {
            assert!((lowp_got - g * f32_x).abs() <= 0.05);
            assert!((f32_got - g * lowp_x).abs() <= 0.05);
        }

        let f32_operand = make_tensor(&[2, 3], f32_vals.clone(), DType::F32);
        let lowp = make_tensor(&[2, 3], lowp_vals.clone(), dtype);
        assert!(!lowp.has_host_f32_data());
        f32_operand.0.borrow_mut().requires_grad = true;
        lowp.0.borrow_mut().requires_grad = true;
        add_cpu_binary_grads(&f32_operand, &lowp, grad.view(), BinaryOp::Mul);
        assert!(!lowp.has_host_f32_data());

        let lowp_decoded = decoded_lowp(&lowp);
        let f32_grad = f32_operand
            .grad_ref()
            .as_ref()
            .expect("f32 grad missing")
            .to_owned();
        let lowp_grad = lowp
            .grad_ref()
            .as_ref()
            .expect("lowp grad missing")
            .to_owned();
        for (((&f32_got, &lowp_got), &g), (&lowp_x, &f32_x)) in f32_grad
            .iter()
            .zip(lowp_grad.iter())
            .zip(grad_slice.iter())
            .zip(lowp_decoded.iter().zip(f32_vals.iter()))
        {
            assert!((f32_got - g * lowp_x).abs() <= 0.05);
            assert!((lowp_got - g * f32_x).abs() <= 0.05);
        }
    }
}

#[test]
fn mixed_mul_scalar_broadcast_backward_reads_native_low_precision_and_writes_f32_grads() {
    let grad = Array::from_shape_vec(IxDyn(&[2, 3]), vec![1.0f32, -0.5, 0.25, 2.0, -1.0, 0.5])
        .expect("grad shape mismatch")
        .into_dyn();
    let grad_slice = grad.as_slice().unwrap();
    let lowp_matrix_vals = vec![1.0f32, -2.0, 0.5, 3.0, -4.0, 2.0];
    let f32_matrix_vals = vec![0.25f32, -1.5, 2.0, 0.75, 3.0, -0.5];
    let f32_row_scalar_vals = vec![0.5f32, -1.25];
    let lowp_row_scalar_vals = vec![1.25f32, -0.75];

    fn decoded_lowp(t: &Tensor) -> Vec<f32> {
        match t.native_storage_owned() {
            TensorStorageOwned::F16(data) => data.iter().map(|v| v.to_f32()).collect(),
            TensorStorageOwned::BF16(data) => data.iter().map(|v| v.to_f32()).collect(),
            TensorStorageOwned::I8(data, scale) => {
                data.iter().map(|&v| (v as f32) * scale).collect()
            }
            TensorStorageOwned::F32(_) => panic!("tensor should stay native low precision"),
        }
    }

    fn assert_matrix_grad_by_row_scalar(got: &Tensor, row_scalars: &[f32], grad_slice: &[f32]) {
        let grad = got
            .grad_ref()
            .as_ref()
            .expect("matrix grad missing")
            .to_owned();
        assert_eq!(grad.shape(), &[2, 3]);
        for (row, &row_scalar) in row_scalars.iter().enumerate().take(2) {
            for col in 0..3 {
                let idx = row * 3 + col;
                let expected = grad_slice[idx] * row_scalar;
                assert!((grad.as_slice().unwrap()[idx] - expected).abs() <= 0.05);
            }
        }
    }

    fn assert_row_scalar_grad(got: &Tensor, matrix_vals: &[f32], grad_slice: &[f32]) {
        let grad = got
            .grad_ref()
            .as_ref()
            .expect("row scalar grad missing")
            .to_owned();
        assert_eq!(grad.shape(), &[2, 1]);
        for row in 0..2 {
            let expected = (0..3)
                .map(|col| {
                    let idx = row * 3 + col;
                    grad_slice[idx] * matrix_vals[idx]
                })
                .sum::<f32>();
            assert!((grad.as_slice().unwrap()[row] - expected).abs() <= 0.05);
        }
    }

    for dtype in [DType::F16, DType::BF16, DType::I8] {
        let lowp_matrix = make_tensor(&[2, 3], lowp_matrix_vals.clone(), dtype);
        let f32_row_scalar = make_tensor(&[2, 1], f32_row_scalar_vals.clone(), DType::F32);
        assert!(!lowp_matrix.has_host_f32_data());
        lowp_matrix.0.borrow_mut().requires_grad = true;
        f32_row_scalar.0.borrow_mut().requires_grad = true;
        add_cpu_binary_grads(&lowp_matrix, &f32_row_scalar, grad.view(), BinaryOp::Mul);
        assert!(!lowp_matrix.has_host_f32_data());
        let lowp_matrix_decoded = decoded_lowp(&lowp_matrix);
        assert_matrix_grad_by_row_scalar(&lowp_matrix, &f32_row_scalar_vals, grad_slice);
        assert_row_scalar_grad(&f32_row_scalar, &lowp_matrix_decoded, grad_slice);

        let f32_matrix = make_tensor(&[2, 3], f32_matrix_vals.clone(), DType::F32);
        let lowp_row_scalar = make_tensor(&[2, 1], lowp_row_scalar_vals.clone(), dtype);
        assert!(!lowp_row_scalar.has_host_f32_data());
        f32_matrix.0.borrow_mut().requires_grad = true;
        lowp_row_scalar.0.borrow_mut().requires_grad = true;
        add_cpu_binary_grads(&f32_matrix, &lowp_row_scalar, grad.view(), BinaryOp::Mul);
        assert!(!lowp_row_scalar.has_host_f32_data());
        let lowp_row_scalar_decoded = decoded_lowp(&lowp_row_scalar);
        assert_matrix_grad_by_row_scalar(&f32_matrix, &lowp_row_scalar_decoded, grad_slice);
        assert_row_scalar_grad(&lowp_row_scalar, &f32_matrix_vals, grad_slice);

        let lowp_vector = make_tensor(&[2, 3], lowp_matrix_vals.clone(), dtype);
        let f32_scalar = make_tensor(&[1], vec![-0.75], DType::F32);
        assert!(!lowp_vector.has_host_f32_data());
        lowp_vector.0.borrow_mut().requires_grad = true;
        f32_scalar.0.borrow_mut().requires_grad = true;
        add_cpu_binary_grads(&lowp_vector, &f32_scalar, grad.view(), BinaryOp::Mul);
        assert!(!lowp_vector.has_host_f32_data());
        let lowp_vector_decoded = decoded_lowp(&lowp_vector);

        let lowp_grad = lowp_vector
            .grad_ref()
            .as_ref()
            .expect("lowp grad missing")
            .to_owned();
        for (&got, &g) in lowp_grad.iter().zip(grad_slice) {
            assert!((got - g * -0.75).abs() <= 0.05);
        }

        let scalar_grad = f32_scalar
            .grad_ref()
            .as_ref()
            .expect("scalar grad missing")
            .to_owned();
        let expected_scalar_grad = grad_slice
            .iter()
            .zip(lowp_vector_decoded.iter())
            .map(|(&g, &x)| g * x)
            .sum::<f32>();
        assert!((scalar_grad.as_slice().unwrap()[0] - expected_scalar_grad).abs() <= 0.05);
    }
}

#[test]
fn bf16_mul_no_grad_preserves_dtype() {
    let lhs = make_tensor(&[2], vec![2.0, -1.5], DType::BF16);
    let rhs = make_tensor(&[2], vec![0.25, 2.0], DType::BF16);

    let out = no_grad(|| lhs * rhs);
    assert_eq!(out.dtype(), DType::BF16);
}

#[test]
fn f16_same_shape_binary_no_grad_preserves_native_storage() {
    let lhs = make_tensor(&[2, 2], vec![1.0, -2.0, 0.5, 3.0], DType::F16);
    let rhs = make_tensor(&[2, 2], vec![0.5, 3.0, -1.5, 0.25], DType::F16);

    let add_out = no_grad(|| lhs.clone() + rhs.clone());
    let sub_out = no_grad(|| lhs.clone() - rhs.clone());
    let mul_out = no_grad(|| lhs * rhs);

    for out in [&add_out, &sub_out, &mul_out] {
        assert_eq!(out.dtype(), DType::F16);
        out.with_storage_view_preferring(StoragePreference::Native, |view| match view {
            TensorStorageView::F16(_) => {}
            TensorStorageView::F32(_) => {
                panic!("same-shape f16 binary should keep native f16 storage")
            }
            TensorStorageView::BF16(_) => {
                panic!("same-shape f16 binary should not produce bf16 storage")
            }
        });
    }

    let add_vals = add_out.data_ref().iter().copied().collect::<Vec<_>>();
    assert_eq!(
        add_vals,
        vec![
            f16::from_f32(1.5).to_f32(),
            f16::from_f32(1.0).to_f32(),
            f16::from_f32(-1.0).to_f32(),
            f16::from_f32(3.25).to_f32(),
        ]
    );
}

#[test]
fn bf16_same_shape_binary_no_grad_preserves_native_storage() {
    let lhs = make_tensor(&[2, 2], vec![1.0, -2.0, 0.5, 3.0], DType::BF16);
    let rhs = make_tensor(&[2, 2], vec![0.5, 3.0, -1.5, 0.25], DType::BF16);

    let add_out = no_grad(|| lhs.clone() + rhs.clone());
    let sub_out = no_grad(|| lhs.clone() - rhs.clone());
    let mul_out = no_grad(|| lhs * rhs);

    for out in [&add_out, &sub_out, &mul_out] {
        assert_eq!(out.dtype(), DType::BF16);
        out.with_storage_view_preferring(StoragePreference::Native, |view| match view {
            TensorStorageView::BF16(_) => {}
            TensorStorageView::F32(_) => {
                panic!("same-shape bf16 binary should keep native bf16 storage")
            }
            TensorStorageView::F16(_) => {
                panic!("same-shape bf16 binary should not produce f16 storage")
            }
        });
    }

    let add_vals = add_out.data_ref().iter().copied().collect::<Vec<_>>();
    assert_eq!(
        add_vals,
        vec![
            bf16::from_f32(1.5).to_f32(),
            bf16::from_f32(1.0).to_f32(),
            bf16::from_f32(-1.0).to_f32(),
            bf16::from_f32(3.25).to_f32(),
        ]
    );
}

#[test]
fn i8_same_shape_binary_no_grad_preserves_native_storage_with_dynamic_scale() {
    let lhs = make_tensor(&[2, 2], vec![1.0, -2.0, 0.5, 3.0], DType::I8);
    let rhs = make_tensor(&[2, 2], vec![0.5, 3.0, -1.5, 0.25], DType::I8);

    let add_out = no_grad(|| lhs.clone() + rhs.clone());
    let sub_out = no_grad(|| lhs.clone() - rhs.clone());
    let mul_out = no_grad(|| lhs.clone() * rhs.clone());

    for out in [&add_out, &sub_out, &mul_out] {
        assert_eq!(out.dtype(), DType::I8);
        assert!(!out.has_host_f32_data());
        match out.native_storage_owned() {
            TensorStorageOwned::I8(data, scale) => {
                assert!(scale.is_finite() && scale > 0.0);
                assert_eq!(data.len(), 4);
            }
            _ => panic!("same-shape i8 binary should keep native i8 storage"),
        }
    }

    let add_vals = add_out.data_ref().iter().copied().collect::<Vec<_>>();
    for (got, expected) in add_vals.iter().zip([1.5f32, 1.0, -1.0, 3.25]) {
        assert!((got - expected).abs() <= 0.03);
    }
}

#[test]
fn row_broadcast_binary_no_grad_preserves_native_low_precision_storage() {
    for dtype in [DType::F16, DType::BF16, DType::I8] {
        let matrix = make_tensor(&[2, 3], vec![1.0, -2.0, 0.5, 3.0, -4.0, 2.0], dtype);
        let row = make_tensor(&[3], vec![0.5, 3.0, -1.5], dtype);

        assert!(!matrix.has_host_f32_data());
        assert!(!row.has_host_f32_data());
        let add_out = no_grad(|| matrix.clone() + row.clone());
        let sub_out = no_grad(|| row.clone() - matrix.clone());
        let mul_out = no_grad(|| matrix.clone() * row.clone());
        assert!(!matrix.has_host_f32_data());
        assert!(!row.has_host_f32_data());

        for out in [&add_out, &sub_out, &mul_out] {
            assert_eq!(out.dtype(), dtype);
            assert_eq!(out.shape_vec(), vec![2, 3]);
            assert!(!out.has_host_f32_data());
        }

        let add_vals = add_out.data_ref().iter().copied().collect::<Vec<_>>();
        let sub_vals = sub_out.data_ref().iter().copied().collect::<Vec<_>>();
        let mul_vals = mul_out.data_ref().iter().copied().collect::<Vec<_>>();
        for (got, expected) in add_vals.iter().zip([1.5f32, 1.0, -1.0, 3.5, -1.0, 0.5]) {
            assert!((got - expected).abs() <= 0.05);
        }
        for (got, expected) in sub_vals.iter().zip([-0.5f32, 5.0, -2.0, -2.5, 7.0, -3.5]) {
            assert!((got - expected).abs() <= 0.08);
        }
        for (got, expected) in mul_vals.iter().zip([0.5f32, -6.0, -0.75, 1.5, -12.0, -3.0]) {
            assert!((got - expected).abs() <= 0.08);
        }
    }
}

#[test]
fn same_shape_binary_training_reads_native_low_precision_and_outputs_f32() {
    for dtype in [DType::F16, DType::BF16, DType::I8] {
        let lhs = make_training_tensor_with_dtype(&[2, 2], vec![1.0, -2.0, 0.5, 3.0], dtype);
        let rhs = make_training_tensor_with_dtype(&[2, 2], vec![0.5, 3.0, -1.5, 0.25], dtype);

        assert!(!lhs.has_host_f32_data());
        assert!(!rhs.has_host_f32_data());
        let add_out = lhs.clone() + rhs.clone();
        let sub_out = lhs.clone() - rhs.clone();
        let mul_out = lhs.clone() * rhs.clone();
        assert!(!lhs.has_host_f32_data());
        assert!(!rhs.has_host_f32_data());

        for out in [&add_out, &sub_out, &mul_out] {
            assert_eq!(out.dtype(), DType::F32);
            assert!(out.has_host_f32_data());
        }

        let add_vals = add_out.data_ref().iter().copied().collect::<Vec<_>>();
        let sub_vals = sub_out.data_ref().iter().copied().collect::<Vec<_>>();
        let mul_vals = mul_out.data_ref().iter().copied().collect::<Vec<_>>();
        for (got, expected) in add_vals.iter().zip([1.5f32, 1.0, -1.0, 3.25]) {
            assert!((got - expected).abs() <= 0.03);
        }
        for (got, expected) in sub_vals.iter().zip([0.5f32, -5.0, 2.0, 2.75]) {
            assert!((got - expected).abs() <= 0.03);
        }
        for (got, expected) in mul_vals.iter().zip([0.5f32, -6.0, -0.75, 0.75]) {
            assert!((got - expected).abs() <= 0.03);
        }
    }
}

#[test]
fn row_broadcast_binary_training_reads_native_low_precision_and_outputs_f32() {
    for dtype in [DType::F16, DType::BF16, DType::I8] {
        let matrix =
            make_training_tensor_with_dtype(&[2, 3], vec![1.0, -2.0, 0.5, 3.0, -4.0, 2.0], dtype);
        let row = make_training_tensor_with_dtype(&[3], vec![0.5, 3.0, -1.5], dtype);

        assert!(!matrix.has_host_f32_data());
        assert!(!row.has_host_f32_data());
        let add_out = matrix.clone() + row.clone();
        let sub_out = row.clone() - matrix.clone();
        let mul_out = matrix.clone() * row.clone();
        assert!(!matrix.has_host_f32_data());
        assert!(!row.has_host_f32_data());

        for out in [&add_out, &sub_out, &mul_out] {
            assert_eq!(out.dtype(), DType::F32);
            assert_eq!(out.shape_vec(), vec![2, 3]);
        }

        let add_vals = add_out.data_ref().iter().copied().collect::<Vec<_>>();
        let sub_vals = sub_out.data_ref().iter().copied().collect::<Vec<_>>();
        let mul_vals = mul_out.data_ref().iter().copied().collect::<Vec<_>>();
        for (got, expected) in add_vals.iter().zip([1.5f32, 1.0, -1.0, 3.5, -1.0, 0.5]) {
            assert!((got - expected).abs() <= 0.03);
        }
        for (got, expected) in sub_vals.iter().zip([-0.5f32, 5.0, -2.0, -2.5, 7.0, -3.5]) {
            assert!((got - expected).abs() <= 0.05);
        }
        for (got, expected) in mul_vals.iter().zip([0.5f32, -6.0, -0.75, 1.5, -12.0, -3.0]) {
            assert!((got - expected).abs() <= 0.05);
        }
    }
}

#[test]
fn mul_backward_same_shape_reads_native_low_precision_and_writes_f32_grads() {
    let grad = Array::from_shape_vec(IxDyn(&[2, 2]), vec![1.0f32, -0.5, 0.25, 2.0])
        .expect("grad shape mismatch")
        .into_dyn();

    for dtype in [DType::F16, DType::BF16, DType::I8] {
        let lhs = make_tensor(&[2, 2], vec![1.0, -2.0, 0.5, 3.0], dtype);
        let rhs = make_tensor(&[2, 2], vec![0.5, 3.0, -1.5, 0.25], dtype);

        assert!(!lhs.has_host_f32_data());
        assert!(!rhs.has_host_f32_data());
        lhs.0.borrow_mut().requires_grad = true;
        rhs.0.borrow_mut().requires_grad = true;
        add_cpu_binary_grads(&lhs, &rhs, grad.view(), BinaryOp::Mul);
        assert!(!lhs.has_host_f32_data());
        assert!(!rhs.has_host_f32_data());

        let rhs_decoded = match rhs.native_storage_owned() {
            TensorStorageOwned::F16(data) => data.iter().map(|v| v.to_f32()).collect::<Vec<_>>(),
            TensorStorageOwned::BF16(data) => data.iter().map(|v| v.to_f32()).collect::<Vec<_>>(),
            TensorStorageOwned::I8(data, scale) => {
                data.iter().map(|&v| (v as f32) * scale).collect::<Vec<_>>()
            }
            TensorStorageOwned::F32(_) => panic!("rhs should stay native low precision"),
        };
        let lhs_decoded = match lhs.native_storage_owned() {
            TensorStorageOwned::F16(data) => data.iter().map(|v| v.to_f32()).collect::<Vec<_>>(),
            TensorStorageOwned::BF16(data) => data.iter().map(|v| v.to_f32()).collect::<Vec<_>>(),
            TensorStorageOwned::I8(data, scale) => {
                data.iter().map(|&v| (v as f32) * scale).collect::<Vec<_>>()
            }
            TensorStorageOwned::F32(_) => panic!("lhs should stay native low precision"),
        };

        let lhs_grad = lhs
            .grad_ref()
            .as_ref()
            .expect("lhs grad missing")
            .to_owned();
        let rhs_grad = rhs
            .grad_ref()
            .as_ref()
            .expect("rhs grad missing")
            .to_owned();
        assert_eq!(lhs_grad.shape(), &[2, 2]);
        assert_eq!(rhs_grad.shape(), &[2, 2]);
        for ((&got, &g), &x) in lhs_grad.iter().zip(grad.iter()).zip(rhs_decoded.iter()) {
            assert!((got - g * x).abs() <= 0.03);
        }
        for ((&got, &g), &x) in rhs_grad.iter().zip(grad.iter()).zip(lhs_decoded.iter()) {
            assert!((got - g * x).abs() <= 0.03);
        }
    }
}

#[test]
fn mul_backward_row_broadcast_reads_native_low_precision_and_writes_f32_grads() {
    let grad = Array::from_shape_vec(IxDyn(&[2, 3]), vec![1.0f32, -0.5, 0.25, 2.0, -1.0, 0.5])
        .expect("grad shape mismatch")
        .into_dyn();

    fn decoded(t: &Tensor) -> Vec<f32> {
        match t.native_storage_owned() {
            TensorStorageOwned::F16(data) => data.iter().map(|v| v.to_f32()).collect(),
            TensorStorageOwned::BF16(data) => data.iter().map(|v| v.to_f32()).collect(),
            TensorStorageOwned::I8(data, scale) => {
                data.iter().map(|&v| (v as f32) * scale).collect()
            }
            TensorStorageOwned::F32(_) => panic!("tensor should stay native low precision"),
        }
    }

    for dtype in [DType::F16, DType::BF16, DType::I8] {
        let matrix = make_tensor(&[2, 3], vec![1.0, -2.0, 0.5, 3.0, -4.0, 2.0], dtype);
        let row = make_tensor(&[3], vec![0.5, 3.0, -1.5], dtype);

        assert!(!matrix.has_host_f32_data());
        assert!(!row.has_host_f32_data());
        matrix.0.borrow_mut().requires_grad = true;
        row.0.borrow_mut().requires_grad = true;
        add_cpu_binary_grads(&matrix, &row, grad.view(), BinaryOp::Mul);
        assert!(!matrix.has_host_f32_data());
        assert!(!row.has_host_f32_data());

        let matrix_vals = decoded(&matrix);
        let row_vals = decoded(&row);
        let matrix_grad = matrix
            .grad_ref()
            .as_ref()
            .expect("matrix grad missing")
            .to_owned();
        let row_grad = row
            .grad_ref()
            .as_ref()
            .expect("row grad missing")
            .to_owned();

        for row_idx in 0..2 {
            for (col, &row_val) in row_vals.iter().enumerate().take(3) {
                let idx = row_idx * 3 + col;
                let expected = grad.as_slice().unwrap()[idx] * row_val;
                assert!((matrix_grad.as_slice().unwrap()[idx] - expected).abs() <= 0.05);
            }
        }
        for col in 0..3 {
            let expected = (0..2)
                .map(|row_idx| {
                    let idx = row_idx * 3 + col;
                    grad.as_slice().unwrap()[idx] * matrix_vals[idx]
                })
                .sum::<f32>();
            assert!((row_grad.as_slice().unwrap()[col] - expected).abs() <= 0.05);
        }

        let row = make_tensor(&[3], vec![0.5, 3.0, -1.5], dtype);
        let matrix = make_tensor(&[2, 3], vec![1.0, -2.0, 0.5, 3.0, -4.0, 2.0], dtype);
        row.0.borrow_mut().requires_grad = true;
        matrix.0.borrow_mut().requires_grad = true;
        add_cpu_binary_grads(&row, &matrix, grad.view(), BinaryOp::Mul);
        assert!(!row.has_host_f32_data());
        assert!(!matrix.has_host_f32_data());

        let row_vals = decoded(&row);
        let matrix_vals = decoded(&matrix);
        let row_grad = row
            .grad_ref()
            .as_ref()
            .expect("row grad missing")
            .to_owned();
        let matrix_grad = matrix
            .grad_ref()
            .as_ref()
            .expect("matrix grad missing")
            .to_owned();
        for col in 0..3 {
            let expected = (0..2)
                .map(|row_idx| {
                    let idx = row_idx * 3 + col;
                    grad.as_slice().unwrap()[idx] * matrix_vals[idx]
                })
                .sum::<f32>();
            assert!((row_grad.as_slice().unwrap()[col] - expected).abs() <= 0.05);
        }
        for row_idx in 0..2 {
            for (col, &row_val) in row_vals.iter().enumerate().take(3) {
                let idx = row_idx * 3 + col;
                let expected = grad.as_slice().unwrap()[idx] * row_val;
                assert!((matrix_grad.as_slice().unwrap()[idx] - expected).abs() <= 0.05);
            }
        }
    }
}

#[test]
fn mixed_mul_backward_row_broadcast_reads_native_low_precision_and_writes_f32_grads() {
    let grad = Array::from_shape_vec(IxDyn(&[2, 3]), vec![1.0f32, -0.5, 0.25, 2.0, -1.0, 0.5])
        .expect("grad shape mismatch")
        .into_dyn();
    let grad_slice = grad.as_slice().unwrap();
    let f32_matrix_vals = vec![0.25f32, -1.5, 2.0, 0.75, 3.0, -0.5];
    let f32_row_vals = vec![0.5f32, 3.0, -1.5];
    let lowp_matrix_vals = vec![1.0f32, -2.0, 0.5, 3.0, -4.0, 2.0];
    let lowp_row_vals = vec![1.25f32, -0.75, 2.5];

    fn decoded_lowp(t: &Tensor) -> Vec<f32> {
        match t.native_storage_owned() {
            TensorStorageOwned::F16(data) => data.iter().map(|v| v.to_f32()).collect(),
            TensorStorageOwned::BF16(data) => data.iter().map(|v| v.to_f32()).collect(),
            TensorStorageOwned::I8(data, scale) => {
                data.iter().map(|&v| (v as f32) * scale).collect()
            }
            TensorStorageOwned::F32(_) => panic!("tensor should stay native low precision"),
        }
    }

    fn assert_matrix_grad(got: &Tensor, row_vals: &[f32], grad_slice: &[f32]) {
        let grad = got
            .grad_ref()
            .as_ref()
            .expect("matrix grad missing")
            .to_owned();
        assert_eq!(grad.shape(), &[2, 3]);
        for row_idx in 0..2 {
            for (col, &row_val) in row_vals.iter().enumerate().take(3) {
                let idx = row_idx * 3 + col;
                let expected = grad_slice[idx] * row_val;
                assert!((grad.as_slice().unwrap()[idx] - expected).abs() <= 0.05);
            }
        }
    }

    fn assert_row_grad(got: &Tensor, matrix_vals: &[f32], grad_slice: &[f32]) {
        let grad = got
            .grad_ref()
            .as_ref()
            .expect("row grad missing")
            .to_owned();
        assert_eq!(grad.shape(), &[3]);
        for col in 0..3 {
            let expected = (0..2)
                .map(|row_idx| {
                    let idx = row_idx * 3 + col;
                    grad_slice[idx] * matrix_vals[idx]
                })
                .sum::<f32>();
            assert!((grad.as_slice().unwrap()[col] - expected).abs() <= 0.05);
        }
    }

    for dtype in [DType::F16, DType::BF16, DType::I8] {
        let lowp_matrix = make_tensor(&[2, 3], lowp_matrix_vals.clone(), dtype);
        let f32_row = make_tensor(&[3], f32_row_vals.clone(), DType::F32);
        assert!(!lowp_matrix.has_host_f32_data());
        lowp_matrix.0.borrow_mut().requires_grad = true;
        f32_row.0.borrow_mut().requires_grad = true;
        add_cpu_binary_grads(&lowp_matrix, &f32_row, grad.view(), BinaryOp::Mul);
        assert!(!lowp_matrix.has_host_f32_data());
        let lowp_matrix_decoded = decoded_lowp(&lowp_matrix);
        assert_matrix_grad(&lowp_matrix, &f32_row_vals, grad_slice);
        assert_row_grad(&f32_row, &lowp_matrix_decoded, grad_slice);

        let f32_row = make_tensor(&[3], f32_row_vals.clone(), DType::F32);
        let lowp_matrix = make_tensor(&[2, 3], lowp_matrix_vals.clone(), dtype);
        assert!(!lowp_matrix.has_host_f32_data());
        f32_row.0.borrow_mut().requires_grad = true;
        lowp_matrix.0.borrow_mut().requires_grad = true;
        add_cpu_binary_grads(&f32_row, &lowp_matrix, grad.view(), BinaryOp::Mul);
        assert!(!lowp_matrix.has_host_f32_data());
        let lowp_matrix_decoded = decoded_lowp(&lowp_matrix);
        assert_row_grad(&f32_row, &lowp_matrix_decoded, grad_slice);
        assert_matrix_grad(&lowp_matrix, &f32_row_vals, grad_slice);

        let f32_matrix = make_tensor(&[2, 3], f32_matrix_vals.clone(), DType::F32);
        let lowp_row = make_tensor(&[3], lowp_row_vals.clone(), dtype);
        assert!(!lowp_row.has_host_f32_data());
        f32_matrix.0.borrow_mut().requires_grad = true;
        lowp_row.0.borrow_mut().requires_grad = true;
        add_cpu_binary_grads(&f32_matrix, &lowp_row, grad.view(), BinaryOp::Mul);
        assert!(!lowp_row.has_host_f32_data());
        let lowp_row_decoded = decoded_lowp(&lowp_row);
        assert_matrix_grad(&f32_matrix, &lowp_row_decoded, grad_slice);
        assert_row_grad(&lowp_row, &f32_matrix_vals, grad_slice);

        let lowp_row = make_tensor(&[3], lowp_row_vals.clone(), dtype);
        let f32_matrix = make_tensor(&[2, 3], f32_matrix_vals.clone(), DType::F32);
        assert!(!lowp_row.has_host_f32_data());
        lowp_row.0.borrow_mut().requires_grad = true;
        f32_matrix.0.borrow_mut().requires_grad = true;
        add_cpu_binary_grads(&lowp_row, &f32_matrix, grad.view(), BinaryOp::Mul);
        assert!(!lowp_row.has_host_f32_data());
        let lowp_row_decoded = decoded_lowp(&lowp_row);
        assert_row_grad(&lowp_row, &f32_matrix_vals, grad_slice);
        assert_matrix_grad(&f32_matrix, &lowp_row_decoded, grad_slice);
    }
}

#[test]
fn bf16_sum_no_grad_keeps_input_dtype() {
    let input = make_tensor(&[2, 2], vec![1.0, 2.0, 3.0, 4.0], DType::BF16);
    let out = no_grad(|| sum(&input));
    assert_eq!(input.dtype(), DType::BF16);
    assert_eq!(out.dtype(), DType::F32);
    assert_eq!(out.data_ref().first().copied(), Some(10.0));
}

#[test]
fn f16_sum_no_grad_reads_native_storage_without_materializing_input_f32() {
    let input = make_tensor(&[2, 2], vec![1.0, 2.0, -3.0, 4.0], DType::F16);
    assert!(!input.has_host_f32_data());

    let out = no_grad(|| sum(&input));

    assert_eq!(input.dtype(), DType::F16);
    assert!(!input.has_host_f32_data());
    assert_eq!(out.dtype(), DType::F32);
    assert_eq!(out.data_ref().first().copied(), Some(4.0));
}

#[test]
fn bf16_sum_no_grad_reads_native_storage_without_materializing_input_f32() {
    let input = make_tensor(&[2, 2], vec![1.0, 2.0, -3.0, 4.0], DType::BF16);
    assert!(!input.has_host_f32_data());

    let out = no_grad(|| sum(&input));

    assert_eq!(input.dtype(), DType::BF16);
    assert!(!input.has_host_f32_data());
    assert_eq!(out.dtype(), DType::F32);
    assert_eq!(out.data_ref().first().copied(), Some(4.0));
}

#[test]
fn i8_sum_no_grad_reads_native_storage_without_materializing_input_f32() {
    let input = make_tensor(&[2, 2], vec![1.0, 2.0, -3.0, 4.0], DType::I8);
    let expected = match input.native_storage_owned() {
        TensorStorageOwned::I8(data, scale) => {
            assert!(!input.has_host_f32_data());
            data.iter().map(|&v| (v as f32) * scale).sum::<f32>()
        }
        _ => panic!("test input should use i8 native storage"),
    };

    let out = no_grad(|| sum(&input));

    assert_eq!(input.dtype(), DType::I8);
    assert!(!input.has_host_f32_data());
    assert_eq!(out.dtype(), DType::F32);
    assert!((out.data_ref().first().copied().unwrap() - expected).abs() <= 1e-6);
}

#[test]
fn i8_add_no_grad_preserves_dtype() {
    let lhs = make_tensor(&[2], vec![1.0, -2.0], DType::I8);
    let rhs = make_tensor(&[2], vec![0.5, 3.0], DType::I8);

    let out = no_grad(|| lhs.clone() + rhs.clone());

    assert_eq!(lhs.dtype(), DType::I8);
    assert_eq!(rhs.dtype(), DType::I8);
    assert_eq!(out.dtype(), DType::I8);
    let vals = out.data_ref().iter().copied().collect::<Vec<_>>();
    assert!((vals[0] - 1.5).abs() <= 0.02);
    assert!((vals[1] - 1.0).abs() <= 0.02);
}

#[test]
fn i8_mul_no_grad_preserves_dtype() {
    let lhs = make_tensor(&[2], vec![2.0, -1.5], DType::I8);
    let rhs = make_tensor(&[2], vec![0.25, 2.0], DType::I8);

    let out = no_grad(|| lhs * rhs);
    assert_eq!(out.dtype(), DType::I8);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_add_matches_cpu_reference() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    crate::ops::cuda::set_enabled(true);
    let lhs = make_tensor(
        &[16384],
        (0..16384).map(|i| i as f32 * 0.25).collect(),
        DType::F32,
    )
    .to_cuda();
    let rhs = make_tensor(
        &[16384],
        (0..16384).map(|i| -(i as f32) * 0.5).collect(),
        DType::F32,
    )
    .to_cuda();

    let out = no_grad(|| lhs.clone() + rhs.clone());
    assert!(out.is_cuda());

    crate::ops::cuda::set_enabled(false);
    let reference = no_grad(|| lhs.to_cpu() + rhs.to_cpu());

    let out_vals = out.data_ref().iter().copied().collect::<Vec<_>>();
    let ref_vals = reference.data_ref().iter().copied().collect::<Vec<_>>();
    for (got, expect) in out_vals.iter().zip(ref_vals.iter()) {
        assert!((got - expect).abs() < 1e-5, "got {got}, expect {expect}");
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_same_shape_lowp_binary_uses_native_forward_buffer_in_strict_mode() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    crate::ops::cuda::set_enabled(true);
    crate::autograd::set_strict_device_execution(true);

    for dtype in [DType::F16, DType::BF16, DType::I8] {
        let lhs = make_tensor(&[2, 2], vec![1.0, -2.0, 0.5, 3.0], dtype).to_cuda();
        let rhs = make_tensor(&[2, 2], vec![0.5, 3.0, -1.5, 0.25], dtype).to_cuda();

        assert!(!lhs.has_host_f32_data());
        assert!(!rhs.has_host_f32_data());
        assert!(lhs.cloned_cuda_native_lowp_buffer().is_some());
        assert!(rhs.cloned_cuda_native_lowp_buffer().is_some());

        let add_out = no_grad(|| lhs.clone() + rhs.clone());
        let sub_out = no_grad(|| lhs.clone() - rhs.clone());
        let mul_out = no_grad(|| lhs.clone() * rhs.clone());

        for out in [&add_out, &sub_out, &mul_out] {
            assert!(out.is_cuda());
            assert_eq!(out.dtype(), dtype);
            assert!(!out.has_host_f32_data());
            let inner = out.0.borrow();
            match dtype {
                DType::F16 => {
                    assert!(inner.cuda_f32_data.is_none());
                    assert!(inner.cuda_f16_data.is_some());
                }
                DType::BF16 => {
                    assert!(inner.cuda_f32_data.is_none());
                    assert!(inner.cuda_bf16_data.is_some());
                }
                DType::I8 => {
                    assert!(inner.cuda_f32_data.is_none());
                    assert!(inner.cuda_i8_data.is_some());
                    assert!(inner.i8_scale.is_some());
                }
                DType::F32 => unreachable!(),
            }
        }

        let add_vals = add_out.data_ref().iter().copied().collect::<Vec<_>>();
        let sub_vals = sub_out.data_ref().iter().copied().collect::<Vec<_>>();
        let mul_vals = mul_out.data_ref().iter().copied().collect::<Vec<_>>();
        for (got, expected) in add_vals.iter().zip([1.5f32, 1.0, -1.0, 3.25]) {
            assert!((got - expected).abs() <= 0.04);
        }
        for (got, expected) in sub_vals.iter().zip([0.5f32, -5.0, 2.0, 2.75]) {
            assert!((got - expected).abs() <= 0.04);
        }
        for (got, expected) in mul_vals.iter().zip([0.5f32, -6.0, -0.75, 0.75]) {
            assert!((got - expected).abs() <= 0.04);
        }
    }

    crate::autograd::set_strict_device_execution(false);
    crate::ops::cuda::set_enabled(false);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_same_shape_lowp_binary_training_forward_outputs_f32_buffer() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    crate::ops::cuda::set_enabled(true);
    crate::autograd::set_strict_device_execution(true);

    for dtype in [DType::F16, DType::BF16, DType::I8] {
        let lhs =
            make_training_tensor_with_dtype(&[2, 2], vec![1.0, -2.0, 0.5, 3.0], dtype).to_cuda();
        let rhs =
            make_training_tensor_with_dtype(&[2, 2], vec![0.5, 3.0, -1.5, 0.25], dtype).to_cuda();
        assert!(lhs.cloned_cuda_native_lowp_buffer().is_some());
        assert!(rhs.cloned_cuda_native_lowp_buffer().is_some());

        let out = lhs.clone() * rhs.clone();
        assert!(out.is_cuda());
        assert_eq!(out.dtype(), DType::F32);
        assert!(!out.has_host_f32_data());
        assert!(out.cloned_cuda_f32_buffer().is_some());

        let loss = sum(&out);
        loss.backward();
        assert!(lhs.cloned_cuda_f32_grad().is_some());
        assert!(rhs.cloned_cuda_f32_grad().is_some());
        assert!(!lhs.has_host_f32_data());
        assert!(!rhs.has_host_f32_data());

        let lhs_grad = lhs.grad().expect("CUDA lowp lhs grad");
        let rhs_grad = rhs.grad().expect("CUDA lowp rhs grad");
        for (got, expected) in lhs_grad.iter().zip([0.5f32, 3.0, -1.5, 0.25]) {
            assert!(
                (got - expected).abs() <= 0.04,
                "{dtype:?} lhs grad got {got}, expected {expected}"
            );
        }
        for (got, expected) in rhs_grad.iter().zip([1.0f32, -2.0, 0.5, 3.0]) {
            assert!(
                (got - expected).abs() <= 0.04,
                "{dtype:?} rhs grad got {got}, expected {expected}"
            );
        }
    }

    crate::autograd::set_strict_device_execution(false);
    crate::ops::cuda::set_enabled(false);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_b1d_1hd_lowp_binary_forward_uses_resident_buffers_in_strict_mode() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    crate::ops::cuda::set_enabled(true);
    crate::autograd::set_strict_device_execution(true);

    let lhs = make_tensor(
        &[2, 1, 3],
        vec![1.0, -2.0, 0.5, 3.0, -4.0, 2.0],
        DType::BF16,
    )
    .to_cuda();
    let rhs = make_tensor(
        &[1, 4, 3],
        vec![
            0.5, 3.0, -1.5, 2.0, 0.25, -0.75, 1.25, -2.5, 4.0, -1.0, 0.0, 1.5,
        ],
        DType::F32,
    )
    .to_cuda();
    assert!(lhs.0.borrow().cuda_f32_data.is_none());
    assert!(lhs.0.borrow().cuda_bf16_data.is_some());

    let add_out = no_grad(|| lhs.clone() + rhs.clone());
    let sub_out = no_grad(|| lhs.clone() - rhs.clone());
    let mul_out = no_grad(|| lhs.clone() * rhs.clone());

    for out in [&add_out, &sub_out, &mul_out] {
        assert!(out.is_cuda());
        assert_eq!(out.dtype(), DType::F32);
        assert!(!out.has_host_f32_data());
        assert!(out.cloned_cuda_f32_buffer().is_some());
    }
    assert!(lhs.0.borrow().cuda_f32_data.is_none());
    assert!(lhs.0.borrow().cuda_bf16_data.is_some());

    crate::autograd::set_strict_device_execution(false);
    crate::ops::cuda::set_enabled(false);

    let add_vals = add_out.data_ref().iter().copied().collect::<Vec<_>>();
    let sub_vals = sub_out.data_ref().iter().copied().collect::<Vec<_>>();
    let mul_vals = mul_out.data_ref().iter().copied().collect::<Vec<_>>();
    let expected_add = [
        1.5f32, 1.0, -1.0, 3.0, -1.75, -0.25, 2.25, -4.5, 4.5, 0.0, -2.0, 2.0, 3.5, -1.0, 0.5, 5.0,
        -3.75, 1.25, 4.25, -6.5, 6.0, 2.0, -4.0, 3.5,
    ];
    let expected_sub = [
        0.5f32, -5.0, 2.0, -1.0, -2.25, 1.25, -0.25, 0.5, -3.5, 2.0, -2.0, -1.0, 2.5, -7.0, 3.5,
        1.0, -4.25, 2.75, 1.75, -1.5, -2.0, 4.0, -4.0, 0.5,
    ];
    let expected_mul = [
        0.5f32, -6.0, -0.75, 2.0, -0.5, -0.375, 1.25, 5.0, 2.0, -1.0, -0.0, 0.75, 1.5, -12.0, -3.0,
        6.0, -1.0, -1.5, 3.75, 10.0, 8.0, -3.0, -0.0, 3.0,
    ];
    for (got, expected) in add_vals.iter().zip(expected_add) {
        assert!(
            (got - expected).abs() <= 0.04,
            "add got {got}, expected {expected}"
        );
    }
    for (got, expected) in sub_vals.iter().zip(expected_sub) {
        assert!(
            (got - expected).abs() <= 0.04,
            "sub got {got}, expected {expected}"
        );
    }
    for (got, expected) in mul_vals.iter().zip(expected_mul) {
        assert!(
            (got - expected).abs() <= 0.04,
            "mul got {got}, expected {expected}"
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_row_scalar_lowp_binary_forward_uses_resident_buffers_in_strict_mode() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    crate::ops::cuda::set_enabled(true);
    crate::autograd::set_strict_device_execution(true);

    let lhs = make_tensor(
        &[2, 2, 3],
        vec![
            1.0, -2.0, 0.5, 3.0, -4.0, 2.0, 0.25, 1.5, -1.0, 2.5, -0.5, 4.0,
        ],
        DType::BF16,
    )
    .to_cuda();
    let rhs = make_tensor(&[2, 2, 1], vec![0.5, -1.5, 2.0, -0.25], DType::F32).to_cuda();
    assert!(lhs.0.borrow().cuda_f32_data.is_none());
    assert!(lhs.0.borrow().cuda_bf16_data.is_some());

    let add_out = no_grad(|| lhs.clone() + rhs.clone());
    let sub_out = no_grad(|| lhs.clone() - rhs.clone());
    let mul_out = no_grad(|| lhs.clone() * rhs.clone());

    for out in [&add_out, &sub_out, &mul_out] {
        assert!(out.is_cuda());
        assert_eq!(out.dtype(), DType::F32);
        assert!(!out.has_host_f32_data());
        assert!(out.cloned_cuda_f32_buffer().is_some());
    }
    assert!(lhs.0.borrow().cuda_f32_data.is_none());
    assert!(lhs.0.borrow().cuda_bf16_data.is_some());

    crate::autograd::set_strict_device_execution(false);
    crate::ops::cuda::set_enabled(false);

    let add_vals = add_out.data_ref().iter().copied().collect::<Vec<_>>();
    let sub_vals = sub_out.data_ref().iter().copied().collect::<Vec<_>>();
    let mul_vals = mul_out.data_ref().iter().copied().collect::<Vec<_>>();
    let expected_add = [
        1.5f32, -1.5, 1.0, 1.5, -5.5, 0.5, 2.25, 3.5, 1.0, 2.25, -0.75, 3.75,
    ];
    let expected_sub = [
        0.5f32, -2.5, 0.0, 4.5, -2.5, 3.5, -1.75, -0.5, -3.0, 2.75, -0.25, 4.25,
    ];
    let expected_mul = [
        0.5f32, -1.0, 0.25, -4.5, 6.0, -3.0, 0.5, 3.0, -2.0, -0.625, 0.125, -1.0,
    ];
    for (got, expected) in add_vals.iter().zip(expected_add) {
        assert!(
            (got - expected).abs() <= 0.04,
            "add got {got}, expected {expected}"
        );
    }
    for (got, expected) in sub_vals.iter().zip(expected_sub) {
        assert!(
            (got - expected).abs() <= 0.04,
            "sub got {got}, expected {expected}"
        );
    }
    for (got, expected) in mul_vals.iter().zip(expected_mul) {
        assert!(
            (got - expected).abs() <= 0.04,
            "mul got {got}, expected {expected}"
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_special_lowp_binary_forward_uses_typed_output_buffers_in_strict_mode() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    crate::ops::cuda::set_enabled(true);
    crate::autograd::set_strict_device_execution(true);

    for dtype in [DType::F16, DType::BF16, DType::I8] {
        let row_lhs = make_tensor(
            &[2, 2, 3],
            vec![
                1.0, -2.0, 0.5, 3.0, -4.0, 2.0, 0.25, 1.5, -1.0, 2.5, -0.5, 4.0,
            ],
            dtype,
        )
        .to_cuda();
        let row_rhs = make_tensor(&[2, 2, 1], vec![0.5, -1.5, 2.0, -0.25], dtype).to_cuda();
        let row_out = no_grad(|| row_lhs.clone() * row_rhs.clone());
        assert_cuda_lowp_typed_output(&row_out, dtype, &[2, 2, 3]);

        let b1d_lhs =
            make_tensor(&[2, 1, 3], vec![1.0, -2.0, 0.5, 3.0, -4.0, 2.0], dtype).to_cuda();
        let h1_rhs = make_tensor(&[1, 4, 1], vec![0.5, 3.0, -1.5, 2.0], dtype).to_cuda();
        let b1d_1h1_out = no_grad(|| b1d_lhs.clone() + h1_rhs.clone());
        assert_cuda_lowp_typed_output(&b1d_1h1_out, dtype, &[2, 4, 3]);

        let hd_rhs = make_tensor(
            &[1, 4, 3],
            vec![
                0.5, 3.0, -1.5, 2.0, 0.25, -0.75, 1.25, -2.5, 4.0, -1.0, 0.0, 1.5,
            ],
            dtype,
        )
        .to_cuda();
        let b1d_1hd_out = no_grad(|| b1d_lhs.clone() - hd_rhs.clone());
        assert_cuda_lowp_typed_output(&b1d_1hd_out, dtype, &[2, 4, 3]);

        let general_lhs =
            make_tensor(&[2, 1, 1, 3], vec![1.0, -2.0, 0.5, 3.0, -4.0, 2.0], dtype).to_cuda();
        let general_rhs = make_tensor(
            &[1, 4, 2, 3],
            vec![
                0.5, 3.0, -1.5, 2.0, 0.25, -0.75, 1.25, -2.5, 4.0, -1.0, 0.0, 1.5, -0.5, 0.75,
                2.25, -3.0, 1.0, -1.25, 0.125, -0.375, 0.875, 1.75, -2.25, 3.5,
            ],
            dtype,
        )
        .to_cuda();
        let general_out = no_grad(|| general_lhs.clone() * general_rhs.clone());
        assert_cuda_lowp_typed_output(&general_out, dtype, &[2, 4, 2, 3]);
    }

    crate::autograd::set_strict_device_execution(false);
    crate::ops::cuda::set_enabled(false);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_row_scalar_mixed_lowp_f32_backward_keeps_lowp_resident() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    crate::ops::cuda::set_enabled(true);
    crate::autograd::set_strict_device_execution(true);

    let lhs = make_training_tensor_with_dtype(
        &[2, 2, 3],
        vec![
            1.0, -2.0, 0.5, 3.0, -4.0, 2.0, 0.25, 1.5, -1.0, 2.5, -0.5, 4.0,
        ],
        DType::BF16,
    )
    .to_cuda();
    let rhs = make_training_tensor_with_dtype(&[2, 2, 1], vec![0.5, -1.5, 2.0, -0.25], DType::F32)
        .to_cuda();
    assert!(lhs.0.borrow().cuda_f32_data.is_none());
    assert!(lhs.0.borrow().cuda_bf16_data.is_some());

    let loss = sum(&(lhs.clone() * rhs.clone()));
    loss.backward();

    assert!(lhs.0.borrow().cuda_f32_data.is_none());
    assert!(lhs.0.borrow().cuda_bf16_data.is_some());
    assert!(lhs.cloned_cuda_f32_grad().is_some());
    assert!(rhs.cloned_cuda_f32_grad().is_some());
    assert!(!lhs.has_host_grad());
    assert!(!rhs.has_host_grad());

    crate::autograd::set_strict_device_execution(false);
    crate::ops::cuda::set_enabled(false);

    let lhs_grad = lhs.grad().expect("CUDA row-scalar mul lhs grad");
    let rhs_grad = rhs.grad().expect("CUDA row-scalar mul rhs grad");
    let expected_lhs = [
        0.5f32, 0.5, 0.5, -1.5, -1.5, -1.5, 2.0, 2.0, 2.0, -0.25, -0.25, -0.25,
    ];
    let expected_rhs = [-0.5f32, 1.0, 0.75, 6.0];
    for (got, expected) in lhs_grad.iter().zip(expected_lhs) {
        assert!(
            (got - expected).abs() <= 0.04,
            "lhs grad got {got}, expected {expected}"
        );
    }
    for (got, expected) in rhs_grad.iter().zip(expected_rhs) {
        assert!(
            (got - expected).abs() <= 0.08,
            "rhs grad got {got}, expected {expected}"
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_row_scalar_mixed_lowp_f32_sub_backward_is_shape_only() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    crate::ops::cuda::set_enabled(true);
    crate::autograd::set_strict_device_execution(true);

    let lhs = make_training_tensor_with_dtype(
        &[2, 2, 3],
        vec![
            1.0, -2.0, 0.5, 3.0, -4.0, 2.0, 0.25, 1.5, -1.0, 2.5, -0.5, 4.0,
        ],
        DType::BF16,
    )
    .to_cuda();
    let rhs = make_training_tensor_with_dtype(&[2, 2, 1], vec![0.5, -1.5, 2.0, -0.25], DType::F32)
        .to_cuda();

    let loss = sum(&(lhs.clone() - rhs.clone()));
    loss.backward();

    assert!(lhs.0.borrow().cuda_f32_data.is_none());
    assert!(lhs.0.borrow().cuda_bf16_data.is_some());
    assert!(lhs.cloned_cuda_f32_grad().is_some());
    assert!(rhs.cloned_cuda_f32_grad().is_some());
    assert!(!lhs.has_host_grad());
    assert!(!rhs.has_host_grad());

    crate::autograd::set_strict_device_execution(false);
    crate::ops::cuda::set_enabled(false);

    let lhs_grad = lhs.grad().expect("CUDA row-scalar sub lhs grad");
    let rhs_grad = rhs.grad().expect("CUDA row-scalar sub rhs grad");
    for got in lhs_grad.iter() {
        assert!((*got - 1.0).abs() <= 1e-6);
    }
    for got in rhs_grad.iter() {
        assert!((*got + 3.0).abs() <= 1e-6);
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_lowp_sum_uses_resident_forward_buffer_in_strict_mode() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    crate::ops::cuda::set_enabled(true);
    crate::autograd::set_strict_device_execution(true);

    for dtype in [DType::F16, DType::BF16, DType::I8] {
        let input = make_tensor(&[2, 2], vec![1.0, -2.0, 0.5, 3.0], dtype).to_cuda();
        assert!(!input.has_host_f32_data());
        assert!(input.cloned_cuda_native_lowp_buffer().is_some());

        let out = no_grad(|| sum(&input));

        assert!(out.is_cuda());
        assert_eq!(out.dtype(), DType::F32);
        assert!(out.cloned_cuda_f32_buffer().is_some());
        assert!(!input.has_host_f32_data());
        let got = out.data_ref().first().copied().unwrap();
        assert!(
            (got - 2.5).abs() <= 0.05,
            "{dtype:?} CUDA sum got {got}, expected 2.5"
        );
    }

    crate::autograd::set_strict_device_execution(false);
    crate::ops::cuda::set_enabled(false);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_lowp_sum_backward_keeps_f32_grad_and_no_operand_f32_materialization() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    crate::ops::cuda::set_enabled(true);
    crate::autograd::set_strict_device_execution(true);

    for dtype in [DType::F16, DType::BF16, DType::I8] {
        let input =
            make_training_tensor_with_dtype(&[2, 2], vec![1.0, -2.0, 0.5, 3.0], dtype).to_cuda();
        assert!(input.cloned_cuda_native_lowp_buffer().is_some());

        let loss = sum(&input);
        loss.backward();

        assert!(input.cloned_cuda_f32_grad().is_some());
        assert!(!input.has_host_f32_data());
        let grad = input.grad().expect("CUDA lowp sum grad");
        for got in grad.iter() {
            assert!((*got - 1.0).abs() <= 1e-6);
        }
    }

    crate::autograd::set_strict_device_execution(false);
    crate::ops::cuda::set_enabled(false);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_resident_i8_sum_matches_quantized_reference() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let len = 1 << 16;
    let scale = 0.05f32;
    let values = (0..len)
        .map(|i| (((i * 31 + 7) % 255) - 127) as i8)
        .collect::<Vec<_>>();
    let expected = values.iter().map(|&v| v as i64).sum::<i64>() as f32 * scale;
    let buffer = crate::ops::cuda::upload_i8_storage(&values).expect("upload i8 sum input");

    let (_, got) = crate::ops::cuda::sum_i8_buffer(&buffer, scale).expect("CUDA resident i8 sum");

    assert!(
        (got[0] - expected).abs() <= 1e-5,
        "resident i8 sum kernel drifted: got {}, expected {}",
        got[0],
        expected
    );
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_row_broadcast_lowp_binary_uses_resident_forward_buffer_in_strict_mode() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    crate::ops::cuda::set_enabled(true);
    crate::autograd::set_strict_device_execution(true);

    for dtype in [DType::F16, DType::BF16, DType::I8] {
        let matrix = make_tensor(&[2, 3], vec![1.0, -2.0, 0.5, 3.0, -4.0, 2.0], dtype).to_cuda();
        let row = make_tensor(&[3], vec![0.5, 3.0, -1.5], dtype).to_cuda();
        assert!(matrix.cloned_cuda_native_lowp_buffer().is_some());
        assert!(row.cloned_cuda_native_lowp_buffer().is_some());

        let add_out = no_grad(|| matrix.clone() + row.clone());
        let sub_out = no_grad(|| row.clone() - matrix.clone());
        let mul_out = no_grad(|| matrix.clone() * row.clone());

        for out in [&add_out, &sub_out, &mul_out] {
            assert!(out.is_cuda());
            assert_eq!(out.dtype(), dtype);
            assert_eq!(out.shape_vec(), vec![2, 3]);
            assert!(!out.has_host_f32_data());
            let inner = out.0.borrow();
            match dtype {
                DType::F16 => {
                    assert!(inner.cuda_f32_data.is_none());
                    assert!(inner.cuda_f16_data.is_some());
                }
                DType::BF16 => {
                    assert!(inner.cuda_f32_data.is_none());
                    assert!(inner.cuda_bf16_data.is_some());
                }
                DType::I8 => {
                    assert!(inner.cuda_f32_data.is_none());
                    assert!(inner.cuda_i8_data.is_some());
                    assert!(inner.i8_scale.is_some());
                }
                DType::F32 => unreachable!(),
            }
        }

        let add_vals = add_out.data_ref().iter().copied().collect::<Vec<_>>();
        let sub_vals = sub_out.data_ref().iter().copied().collect::<Vec<_>>();
        let mul_vals = mul_out.data_ref().iter().copied().collect::<Vec<_>>();
        for (got, expected) in add_vals.iter().zip([1.5f32, 1.0, -1.0, 3.5, -1.0, 0.5]) {
            assert!((got - expected).abs() <= 0.05);
        }
        for (got, expected) in sub_vals.iter().zip([-0.5f32, 5.0, -2.0, -2.5, 7.0, -3.5]) {
            assert!((got - expected).abs() <= 0.08);
        }
        for (got, expected) in mul_vals.iter().zip([0.5f32, -6.0, -0.75, 1.5, -12.0, -3.0]) {
            assert!((got - expected).abs() <= 0.08);
        }
    }

    crate::autograd::set_strict_device_execution(false);
    crate::ops::cuda::set_enabled(false);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_row_broadcast_mixed_lowp_f32_uses_resident_forward_buffer_in_strict_mode() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    crate::ops::cuda::set_enabled(true);
    crate::autograd::set_strict_device_execution(true);

    let matrix = make_tensor(&[2, 3], vec![1.0, -2.0, 0.5, 3.0, -4.0, 2.0], DType::BF16).to_cuda();
    let row = make_tensor(&[3], vec![0.5, 3.0, -1.5], DType::F32).to_cuda();
    assert!(matrix.0.borrow().cuda_f32_data.is_none());
    assert!(matrix.0.borrow().cuda_bf16_data.is_some());

    let add_out = no_grad(|| matrix.clone() + row.clone());
    let mul_out = no_grad(|| matrix.clone() * row.clone());

    assert!(matrix.0.borrow().cuda_f32_data.is_none());
    assert!(matrix.0.borrow().cuda_bf16_data.is_some());
    for out in [&add_out, &mul_out] {
        assert!(out.is_cuda());
        assert_eq!(out.dtype(), DType::F32);
        assert_eq!(out.shape_vec(), vec![2, 3]);
        assert!(out.cloned_cuda_f32_buffer().is_some());
    }

    let add_vals = add_out.data_ref().iter().copied().collect::<Vec<_>>();
    let mul_vals = mul_out.data_ref().iter().copied().collect::<Vec<_>>();
    for (got, expected) in add_vals.iter().zip([1.5f32, 1.0, -1.0, 3.5, -1.0, 0.5]) {
        assert!((got - expected).abs() <= 0.05);
    }
    for (got, expected) in mul_vals.iter().zip([0.5f32, -6.0, -0.75, 1.5, -12.0, -3.0]) {
        assert!((got - expected).abs() <= 0.08);
    }

    crate::autograd::set_strict_device_execution(false);
    crate::ops::cuda::set_enabled(false);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_same_shape_mixed_lowp_f32_mul_backward_keeps_lowp_resident() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    crate::ops::cuda::set_enabled(true);
    crate::autograd::set_strict_device_execution(true);

    let lhs =
        make_training_tensor_with_dtype(&[2, 3], vec![1.0, -2.0, 0.5, 3.0, -4.0, 2.0], DType::BF16)
            .to_cuda();
    let rhs = make_training_tensor_with_dtype(
        &[2, 3],
        vec![0.5, 3.0, -1.5, 2.0, 0.25, -0.75],
        DType::F32,
    )
    .to_cuda();
    assert!(lhs.0.borrow().cuda_f32_data.is_none());
    assert!(lhs.0.borrow().cuda_bf16_data.is_some());

    let loss = sum(&(lhs.clone() * rhs.clone()));
    loss.backward();

    assert!(lhs.0.borrow().cuda_f32_data.is_none());
    assert!(lhs.0.borrow().cuda_bf16_data.is_some());
    assert!(lhs.cloned_cuda_f32_grad().is_some());
    assert!(rhs.cloned_cuda_f32_grad().is_some());
    assert!(!lhs.has_host_grad());
    assert!(!rhs.has_host_grad());

    crate::autograd::set_strict_device_execution(false);
    crate::ops::cuda::set_enabled(false);

    let lhs_grad = lhs.grad().expect("CUDA same-shape mixed mul lhs grad");
    let rhs_grad = rhs.grad().expect("CUDA same-shape mixed mul rhs grad");
    for (got, expected) in lhs_grad.iter().zip([0.5f32, 3.0, -1.5, 2.0, 0.25, -0.75]) {
        assert!((got - expected).abs() <= 0.05);
    }
    for (got, expected) in rhs_grad.iter().zip([1.0f32, -2.0, 0.5, 3.0, -4.0, 2.0]) {
        assert!((got - expected).abs() <= 0.08);
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_row_broadcast_mixed_lowp_f32_mul_backward_keeps_lowp_resident() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    crate::ops::cuda::set_enabled(true);
    crate::autograd::set_strict_device_execution(true);

    let matrix =
        make_training_tensor_with_dtype(&[2, 3], vec![1.0, -2.0, 0.5, 3.0, -4.0, 2.0], DType::BF16)
            .to_cuda();
    let row = make_training_tensor_with_dtype(&[3], vec![0.5, 3.0, -1.5], DType::F32).to_cuda();
    assert!(matrix.0.borrow().cuda_f32_data.is_none());
    assert!(matrix.0.borrow().cuda_bf16_data.is_some());

    let loss = sum(&(matrix.clone() * row.clone()));
    loss.backward();

    assert!(matrix.0.borrow().cuda_f32_data.is_none());
    assert!(matrix.0.borrow().cuda_bf16_data.is_some());
    assert!(matrix.cloned_cuda_f32_grad().is_some());
    assert!(row.cloned_cuda_f32_grad().is_some());
    assert!(!matrix.has_host_grad());
    assert!(!row.has_host_grad());

    crate::autograd::set_strict_device_execution(false);
    crate::ops::cuda::set_enabled(false);

    let matrix_grad = matrix.grad().expect("CUDA mixed row-broadcast matrix grad");
    let row_grad = row.grad().expect("CUDA mixed row-broadcast row grad");
    for (got, expected) in matrix_grad.iter().zip([0.5f32, 3.0, -1.5, 0.5, 3.0, -1.5]) {
        assert!((got - expected).abs() <= 0.05);
    }
    for (got, expected) in row_grad.iter().zip([4.0f32, -6.0, 2.5]) {
        assert!(
            (got - expected).abs() <= 0.08,
            "row grad got {got}, expected {expected}"
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_mixed_lowp_f32_add_sub_backward_is_shape_only() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    crate::ops::cuda::set_enabled(true);
    crate::autograd::set_strict_device_execution(true);

    let lhs =
        make_training_tensor_with_dtype(&[2, 2], vec![1.0, -2.0, 0.5, 3.0], DType::BF16).to_cuda();
    let rhs =
        make_training_tensor_with_dtype(&[2, 2], vec![0.5, 3.0, -1.5, 0.25], DType::F32).to_cuda();
    assert!(lhs.0.borrow().cuda_f32_data.is_none());
    assert!(lhs.0.borrow().cuda_bf16_data.is_some());

    let loss = sum(&(lhs.clone() + rhs.clone()));
    loss.backward();

    assert!(lhs.0.borrow().cuda_f32_data.is_none());
    assert!(lhs.0.borrow().cuda_bf16_data.is_some());
    assert!(lhs.cloned_cuda_f32_grad().is_some());
    assert!(rhs.cloned_cuda_f32_grad().is_some());
    assert!(!lhs.has_host_grad());
    assert!(!rhs.has_host_grad());

    crate::autograd::set_strict_device_execution(false);
    crate::ops::cuda::set_enabled(false);

    let lhs_grad = lhs.grad().expect("CUDA add lhs grad");
    let rhs_grad = rhs.grad().expect("CUDA add rhs grad");
    for got in lhs_grad.iter().chain(rhs_grad.iter()) {
        assert!((*got - 1.0).abs() <= 1e-6);
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_row_broadcast_mixed_lowp_f32_sub_backward_is_shape_only() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    crate::ops::cuda::set_enabled(true);
    crate::autograd::set_strict_device_execution(true);

    let matrix =
        make_training_tensor_with_dtype(&[2, 3], vec![1.0, -2.0, 0.5, 3.0, -4.0, 2.0], DType::BF16)
            .to_cuda();
    let row = make_training_tensor_with_dtype(&[3], vec![0.5, 3.0, -1.5], DType::F32).to_cuda();
    assert!(matrix.0.borrow().cuda_f32_data.is_none());
    assert!(matrix.0.borrow().cuda_bf16_data.is_some());

    let loss = sum(&(matrix.clone() - row.clone()));
    loss.backward();

    assert!(matrix.0.borrow().cuda_f32_data.is_none());
    assert!(matrix.0.borrow().cuda_bf16_data.is_some());
    assert!(matrix.cloned_cuda_f32_grad().is_some());
    assert!(row.cloned_cuda_f32_grad().is_some());
    assert!(!matrix.has_host_grad());
    assert!(!row.has_host_grad());

    crate::autograd::set_strict_device_execution(false);
    crate::ops::cuda::set_enabled(false);

    let matrix_grad = matrix.grad().expect("CUDA row-broadcast sub matrix grad");
    let row_grad = row.grad().expect("CUDA row-broadcast sub row grad");
    for got in matrix_grad.iter() {
        assert!((*got - 1.0).abs() <= 1e-6);
    }
    for got in row_grad.iter() {
        assert!((*got + 2.0).abs() <= 1e-6);
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_general_broadcast_mixed_lowp_f32_sub_backward_is_shape_only() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    crate::ops::cuda::set_enabled(true);
    crate::autograd::set_strict_device_execution(true);

    let lhs = make_training_tensor_with_dtype(
        &[2, 1, 3],
        vec![1.0, -2.0, 0.5, 3.0, -4.0, 2.0],
        DType::BF16,
    )
    .to_cuda();
    let rhs = make_training_tensor_with_dtype(&[1, 4, 1], vec![0.5, 3.0, -1.5, 2.0], DType::F32)
        .to_cuda();
    assert!(lhs.0.borrow().cuda_f32_data.is_none());
    assert!(lhs.0.borrow().cuda_bf16_data.is_some());

    let loss = sum(&(lhs.clone() - rhs.clone()));
    loss.backward();

    assert!(lhs.0.borrow().cuda_f32_data.is_none());
    assert!(lhs.0.borrow().cuda_bf16_data.is_some());
    assert!(lhs.cloned_cuda_f32_grad().is_some());
    assert!(rhs.cloned_cuda_f32_grad().is_some());
    assert!(!lhs.has_host_grad());
    assert!(!rhs.has_host_grad());

    crate::autograd::set_strict_device_execution(false);
    crate::ops::cuda::set_enabled(false);

    let lhs_grad = lhs.grad().expect("CUDA general-broadcast sub lhs grad");
    let rhs_grad = rhs.grad().expect("CUDA general-broadcast sub rhs grad");
    for got in lhs_grad.iter() {
        assert!((*got - 4.0).abs() <= 1e-6);
    }
    for got in rhs_grad.iter() {
        assert!((*got + 6.0).abs() <= 1e-6);
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_general_broadcast_non_special_mixed_lowp_f32_add_backward_is_shape_only() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    crate::ops::cuda::set_enabled(true);
    crate::autograd::set_strict_device_execution(true);

    let lhs = make_training_tensor_with_dtype(
        &[2, 1, 1, 3],
        vec![1.0, -2.0, 0.5, 3.0, -4.0, 2.0],
        DType::BF16,
    )
    .to_cuda();
    let rhs = make_training_tensor_with_dtype(
        &[1, 4, 2, 3],
        vec![
            0.5, 3.0, -1.5, 2.0, 0.25, -0.75, 1.25, -2.5, 4.0, -1.0, 0.0, 1.5, -3.0, 0.75, 2.25,
            1.0, -1.25, 3.5, 0.125, -0.5, 1.75, 2.5, -2.0, 0.625,
        ],
        DType::F32,
    )
    .to_cuda();
    assert!(lhs.0.borrow().cuda_f32_data.is_none());
    assert!(lhs.0.borrow().cuda_bf16_data.is_some());

    let loss = sum(&(lhs.clone() + rhs.clone()));
    loss.backward();

    assert!(lhs.0.borrow().cuda_f32_data.is_none());
    assert!(lhs.0.borrow().cuda_bf16_data.is_some());
    assert!(lhs.cloned_cuda_f32_grad().is_some());
    assert!(rhs.cloned_cuda_f32_grad().is_some());
    assert!(!lhs.has_host_grad());
    assert!(!rhs.has_host_grad());

    crate::autograd::set_strict_device_execution(false);
    crate::ops::cuda::set_enabled(false);

    let lhs_grad = lhs.grad().expect("CUDA non-special broadcast add lhs grad");
    let rhs_grad = rhs.grad().expect("CUDA non-special broadcast add rhs grad");
    for got in lhs_grad.iter() {
        assert!((*got - 8.0).abs() <= 1e-6);
    }
    for got in rhs_grad.iter() {
        assert!((*got - 2.0).abs() <= 1e-6);
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_b1d_1hd_broadcast_mixed_lowp_f32_add_backward_is_shape_only() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    crate::ops::cuda::set_enabled(true);
    crate::autograd::set_strict_device_execution(true);

    let lhs = make_training_tensor_with_dtype(
        &[2, 1, 3],
        vec![1.0, -2.0, 0.5, 3.0, -4.0, 2.0],
        DType::BF16,
    )
    .to_cuda();
    let rhs = make_training_tensor_with_dtype(
        &[1, 4, 3],
        vec![
            0.5, 3.0, -1.5, 2.0, 0.25, -0.75, 1.25, -2.5, 4.0, -1.0, 0.0, 1.5,
        ],
        DType::F32,
    )
    .to_cuda();
    assert!(lhs.0.borrow().cuda_f32_data.is_none());
    assert!(lhs.0.borrow().cuda_bf16_data.is_some());

    let loss = sum(&(lhs.clone() + rhs.clone()));
    loss.backward();

    assert!(lhs.0.borrow().cuda_f32_data.is_none());
    assert!(lhs.0.borrow().cuda_bf16_data.is_some());
    assert!(lhs.cloned_cuda_f32_grad().is_some());
    assert!(rhs.cloned_cuda_f32_grad().is_some());
    assert!(!lhs.has_host_grad());
    assert!(!rhs.has_host_grad());

    crate::autograd::set_strict_device_execution(false);
    crate::ops::cuda::set_enabled(false);

    let lhs_grad = lhs.grad().expect("CUDA b1d/1hd broadcast add lhs grad");
    let rhs_grad = rhs.grad().expect("CUDA b1d/1hd broadcast add rhs grad");
    for got in lhs_grad.iter() {
        assert!((*got - 4.0).abs() <= 1e-6);
    }
    for got in rhs_grad.iter() {
        assert!((*got - 2.0).abs() <= 1e-6);
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_b1d_1hd_mixed_lowp_f32_mul_backward_keeps_lowp_resident() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    crate::ops::cuda::set_enabled(true);
    crate::autograd::set_strict_device_execution(true);

    let lhs = make_training_tensor_with_dtype(
        &[2, 1, 3],
        vec![1.0, -2.0, 0.5, 3.0, -4.0, 2.0],
        DType::BF16,
    )
    .to_cuda();
    let rhs = make_training_tensor_with_dtype(
        &[1, 4, 3],
        vec![
            0.5, 3.0, -1.5, 2.0, 0.25, -0.75, 1.25, -2.5, 4.0, -1.0, 0.0, 1.5,
        ],
        DType::F32,
    )
    .to_cuda();
    assert!(lhs.0.borrow().cuda_f32_data.is_none());
    assert!(lhs.0.borrow().cuda_bf16_data.is_some());

    let loss = sum(&(lhs.clone() * rhs.clone()));
    loss.backward();

    assert!(lhs.0.borrow().cuda_f32_data.is_none());
    assert!(lhs.0.borrow().cuda_bf16_data.is_some());
    assert!(lhs.cloned_cuda_f32_grad().is_some());
    assert!(rhs.cloned_cuda_f32_grad().is_some());
    assert!(!lhs.has_host_grad());
    assert!(!rhs.has_host_grad());

    crate::autograd::set_strict_device_execution(false);
    crate::ops::cuda::set_enabled(false);

    let lhs_grad = lhs.grad().expect("CUDA b1d/1hd mul lhs grad");
    let rhs_grad = rhs.grad().expect("CUDA b1d/1hd mul rhs grad");
    for (got, expected) in lhs_grad.iter().zip([2.75f32, 0.75, 3.25, 2.75, 0.75, 3.25]) {
        assert!(
            (got - expected).abs() <= 0.05,
            "lhs grad got {got}, expected {expected}"
        );
    }
    for (got, expected) in rhs_grad.iter().zip([4.0f32, -6.0, 2.5].into_iter().cycle()) {
        assert!(
            (got - expected).abs() <= 0.08,
            "rhs grad got {got}, expected {expected}"
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_general_broadcast_mixed_lowp_f32_mul_backward_keeps_lowp_resident() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    crate::ops::cuda::set_enabled(true);
    crate::autograd::set_strict_device_execution(true);

    let lhs = make_training_tensor_with_dtype(
        &[2, 1, 1, 3],
        vec![1.0, -2.0, 0.5, 3.0, -4.0, 2.0],
        DType::BF16,
    )
    .to_cuda();
    let rhs = make_training_tensor_with_dtype(
        &[1, 4, 2, 3],
        vec![
            0.5, 3.0, -1.5, 2.0, 0.25, -0.75, 1.25, -2.5, 4.0, -1.0, 0.0, 1.5, 0.75, -1.25, 2.25,
            3.5, -0.5, 1.0, -2.0, 0.5, 0.25, 1.75, -3.0, 2.5,
        ],
        DType::F32,
    )
    .to_cuda();
    assert!(lhs.0.borrow().cuda_f32_data.is_none());
    assert!(lhs.0.borrow().cuda_bf16_data.is_some());

    let loss = sum(&(lhs.clone() * rhs.clone()));
    loss.backward();

    assert!(lhs.0.borrow().cuda_f32_data.is_none());
    assert!(lhs.0.borrow().cuda_bf16_data.is_some());
    assert!(lhs.cloned_cuda_f32_grad().is_some());
    assert!(rhs.cloned_cuda_f32_grad().is_some());
    assert!(!lhs.has_host_grad());
    assert!(!rhs.has_host_grad());

    crate::autograd::set_strict_device_execution(false);
    crate::ops::cuda::set_enabled(false);

    let lhs_grad = lhs.grad().expect("CUDA general-broadcast mul lhs grad");
    let rhs_grad = rhs.grad().expect("CUDA general-broadcast mul rhs grad");
    for (got, expected) in lhs_grad.iter().zip([6.75f32, -3.5, 9.25, 6.75, -3.5, 9.25]) {
        assert!(
            (got - expected).abs() <= 0.08,
            "lhs grad got {got}, expected {expected}"
        );
    }
    for (got, expected) in rhs_grad.iter().zip([4.0f32, -6.0, 2.5].into_iter().cycle()) {
        assert!(
            (got - expected).abs() <= 0.08,
            "rhs grad got {got}, expected {expected}"
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_b1d_1h1_mixed_lowp_f32_mul_backward_keeps_lowp_resident() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    crate::ops::cuda::set_enabled(true);
    crate::autograd::set_strict_device_execution(true);

    let lhs = make_training_tensor_with_dtype(
        &[2, 1, 3],
        vec![1.0, -2.0, 0.5, 3.0, -4.0, 2.0],
        DType::BF16,
    )
    .to_cuda();
    let rhs = make_training_tensor_with_dtype(&[1, 4, 1], vec![0.5, 3.0, -1.5, 2.0], DType::F32)
        .to_cuda();
    assert!(lhs.0.borrow().cuda_f32_data.is_none());
    assert!(lhs.0.borrow().cuda_bf16_data.is_some());

    let loss = sum(&(lhs.clone() * rhs.clone()));
    loss.backward();

    assert!(lhs.0.borrow().cuda_f32_data.is_none());
    assert!(lhs.0.borrow().cuda_bf16_data.is_some());
    assert!(lhs.cloned_cuda_f32_grad().is_some());
    assert!(rhs.cloned_cuda_f32_grad().is_some());
    assert!(!lhs.has_host_grad());
    assert!(!rhs.has_host_grad());

    crate::autograd::set_strict_device_execution(false);
    crate::ops::cuda::set_enabled(false);

    let lhs_grad = lhs.grad().expect("CUDA b1d/1h1 mul lhs grad");
    let rhs_grad = rhs.grad().expect("CUDA b1d/1h1 mul rhs grad");
    for got in lhs_grad.iter() {
        assert!((*got - 4.0).abs() <= 0.05);
    }
    for got in rhs_grad.iter() {
        assert!((*got - 0.5).abs() <= 0.08);
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_scalar_broadcast_mixed_lowp_f32_sub_backward_is_shape_only() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    crate::ops::cuda::set_enabled(true);
    crate::autograd::set_strict_device_execution(true);

    let lhs =
        make_training_tensor_with_dtype(&[2, 2], vec![1.0, -2.0, 0.5, 3.0], DType::BF16).to_cuda();
    let rhs = make_training_tensor_with_dtype(&[1], vec![0.5], DType::F32).to_cuda();
    assert!(lhs.0.borrow().cuda_f32_data.is_none());
    assert!(lhs.0.borrow().cuda_bf16_data.is_some());

    let loss = sum(&(lhs.clone() - rhs.clone()));
    loss.backward();

    assert!(lhs.0.borrow().cuda_f32_data.is_none());
    assert!(lhs.0.borrow().cuda_bf16_data.is_some());
    assert!(lhs.cloned_cuda_f32_grad().is_some());
    assert!(rhs.cloned_cuda_f32_grad().is_some());
    assert!(!lhs.has_host_grad());
    assert!(!rhs.has_host_grad());

    crate::autograd::set_strict_device_execution(false);
    crate::ops::cuda::set_enabled(false);

    let lhs_grad = lhs.grad().expect("CUDA scalar-broadcast sub lhs grad");
    let rhs_grad = rhs.grad().expect("CUDA scalar-broadcast sub rhs grad");
    for got in lhs_grad.iter() {
        assert!((*got - 1.0).abs() <= 1e-6);
    }
    assert!((rhs_grad[IxDyn(&[0])] + 4.0).abs() <= 1e-6);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_scalar_broadcast_mixed_lowp_f32_mul_backward_keeps_lowp_resident() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    crate::ops::cuda::set_enabled(true);
    crate::autograd::set_strict_device_execution(true);

    let lhs =
        make_training_tensor_with_dtype(&[2, 2], vec![1.0, -2.0, 0.5, 3.0], DType::BF16).to_cuda();
    let rhs = make_training_tensor_with_dtype(&[1], vec![0.5], DType::F32).to_cuda();
    assert!(lhs.0.borrow().cuda_f32_data.is_none());
    assert!(lhs.0.borrow().cuda_bf16_data.is_some());

    let loss = sum(&(lhs.clone() * rhs.clone()));
    loss.backward();

    assert!(lhs.0.borrow().cuda_f32_data.is_none());
    assert!(lhs.0.borrow().cuda_bf16_data.is_some());
    assert!(lhs.cloned_cuda_f32_grad().is_some());
    assert!(rhs.cloned_cuda_f32_grad().is_some());
    assert!(!lhs.has_host_grad());
    assert!(!rhs.has_host_grad());

    crate::autograd::set_strict_device_execution(false);
    crate::ops::cuda::set_enabled(false);

    let lhs_grad = lhs.grad().expect("CUDA scalar-broadcast mul lhs grad");
    let rhs_grad = rhs.grad().expect("CUDA scalar-broadcast mul rhs grad");
    for got in lhs_grad.iter() {
        assert!((*got - 0.5).abs() <= 0.05);
    }
    assert!((rhs_grad[IxDyn(&[0])] - 2.5).abs() <= 0.08);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_row_broadcast_lowp_mul_backward_outputs_f32_grads() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    crate::ops::cuda::set_enabled(true);
    crate::autograd::set_strict_device_execution(true);

    for dtype in [DType::F16, DType::BF16, DType::I8] {
        let matrix =
            make_training_tensor_with_dtype(&[2, 3], vec![1.0, -2.0, 0.5, 3.0, -4.0, 2.0], dtype)
                .to_cuda();
        let row = make_training_tensor_with_dtype(&[3], vec![0.5, 3.0, -1.5], dtype).to_cuda();

        let loss = sum(&(matrix.clone() * row.clone()));
        loss.backward();
        assert!(matrix.cloned_cuda_f32_grad().is_some());
        assert!(row.cloned_cuda_f32_grad().is_some());
        assert!(!matrix.has_host_f32_data());
        assert!(!row.has_host_f32_data());

        let matrix_grad = matrix.grad().expect("CUDA row-broadcast matrix grad");
        let row_grad = row.grad().expect("CUDA row-broadcast row grad");
        for (got, expected) in matrix_grad.iter().zip([0.5f32, 3.0, -1.5, 0.5, 3.0, -1.5]) {
            assert!((got - expected).abs() <= 0.05);
        }
        for (got, expected) in row_grad.iter().zip([4.0f32, -6.0, 2.5]) {
            assert!(
                (got - expected).abs() <= 0.08,
                "{dtype:?} row grad got {got}, expected {expected}"
            );
        }
    }

    crate::autograd::set_strict_device_execution(false);
    crate::ops::cuda::set_enabled(false);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_2d_row_broadcast_i8_mul_backward_matches_reference() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    crate::ops::cuda::set_enabled(true);
    crate::autograd::set_strict_device_execution(true);

    let matrix =
        make_training_tensor_with_dtype(&[2, 3], vec![1.0, -2.0, 0.5, 3.0, -4.0, 2.0], DType::I8)
            .to_cuda();
    let row = make_training_tensor_with_dtype(&[1, 3], vec![0.5, 3.0, -1.5], DType::I8).to_cuda();

    let loss = sum(&(matrix.clone() * row.clone()));
    loss.backward();

    assert!(matrix.cloned_cuda_f32_grad().is_some());
    assert!(row.cloned_cuda_f32_grad().is_some());
    assert!(!matrix.has_host_f32_data());
    assert!(!row.has_host_f32_data());

    crate::autograd::set_strict_device_execution(false);
    crate::ops::cuda::set_enabled(false);

    let matrix_grad = matrix.grad().expect("CUDA 2D row-broadcast matrix grad");
    let row_grad = row.grad().expect("CUDA 2D row-broadcast row grad");
    for (got, expected) in matrix_grad.iter().zip([0.5f32, 3.0, -1.5, 0.5, 3.0, -1.5]) {
        assert!((got - expected).abs() <= 0.05);
    }
    for (got, expected) in row_grad.iter().zip([4.0f32, -6.0, 2.5]) {
        assert!(
            (got - expected).abs() <= 0.08,
            "I8 2D row grad got {got}, expected {expected}"
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
#[ignore]
fn cuda_lowp_binary_forward_perf_smoke() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let len = 1 << 20;
    let lhs = (0..len)
        .map(|i| (i as f32 % 251.0) / 37.0 - 3.0)
        .collect::<Vec<_>>();
    let rhs = (0..len)
        .map(|i| ((i * 17) as f32 % 241.0) / 41.0 - 2.5)
        .collect::<Vec<_>>();
    let lhs_f32 = crate::ops::cuda::upload_f32(&lhs).expect("upload lhs f32");
    let rhs_f32 = crate::ops::cuda::upload_f32(&rhs).expect("upload rhs f32");

    let _f32_warmup =
        crate::ops::cuda::binary_f32_buffer(&lhs_f32, &rhs_f32, crate::ops::cuda::BinaryOp::Mul)
            .expect("warm up CUDA f32 binary perf path");
    crate::ops::cuda::synchronize().expect("sync f32 binary warmup");

    let start = std::time::Instant::now();
    let f32_out =
        crate::ops::cuda::binary_f32_buffer(&lhs_f32, &rhs_f32, crate::ops::cuda::BinaryOp::Mul)
            .expect("CUDA f32 binary perf path");
    crate::ops::cuda::synchronize().expect("sync f32 binary");
    let f32_us = start.elapsed().as_secs_f64() * 1.0e6;
    let f32_ref = crate::ops::cuda::download_f32(&f32_out).expect("download f32 binary");

    let lhs_f16 = lhs
        .iter()
        .map(|&v| f16::from_f32(v).to_bits())
        .collect::<Vec<_>>();
    let rhs_f16 = rhs
        .iter()
        .map(|&v| f16::from_f32(v).to_bits())
        .collect::<Vec<_>>();
    let lhs_f16_buf = crate::ops::cuda::upload_u16_storage(&lhs_f16).expect("upload lhs f16");
    let rhs_f16_buf = crate::ops::cuda::upload_u16_storage(&rhs_f16).expect("upload rhs f16");
    let start = std::time::Instant::now();
    let f16_out = crate::ops::cuda::binary_f16_buffer_no_host(
        &lhs_f16_buf,
        &rhs_f16_buf,
        crate::ops::cuda::BinaryOp::Mul,
    )
    .expect("CUDA resident f16 binary perf path");
    crate::ops::cuda::synchronize().expect("sync f16 binary");
    let f16_us = start.elapsed().as_secs_f64() * 1.0e6;
    let f16_vals = crate::ops::cuda::download_f32(&f16_out).expect("download f16 binary");

    let lhs_bf16 = lhs
        .iter()
        .map(|&v| bf16::from_f32(v).to_bits())
        .collect::<Vec<_>>();
    let rhs_bf16 = rhs
        .iter()
        .map(|&v| bf16::from_f32(v).to_bits())
        .collect::<Vec<_>>();
    let lhs_bf16_buf = crate::ops::cuda::upload_u16_storage(&lhs_bf16).expect("upload lhs bf16");
    let rhs_bf16_buf = crate::ops::cuda::upload_u16_storage(&rhs_bf16).expect("upload rhs bf16");
    let start = std::time::Instant::now();
    let bf16_out = crate::ops::cuda::binary_bf16_buffer_no_host(
        &lhs_bf16_buf,
        &rhs_bf16_buf,
        crate::ops::cuda::BinaryOp::Mul,
    )
    .expect("CUDA resident bf16 binary perf path");
    crate::ops::cuda::synchronize().expect("sync bf16 binary");
    let bf16_us = start.elapsed().as_secs_f64() * 1.0e6;
    let bf16_vals = crate::ops::cuda::download_f32(&bf16_out).expect("download bf16 binary");

    let lhs_i8 = lhs
        .iter()
        .map(|&v| (v / 0.05).round().clamp(-127.0, 127.0) as i8)
        .collect::<Vec<_>>();
    let rhs_i8 = rhs
        .iter()
        .map(|&v| (v / 0.05).round().clamp(-127.0, 127.0) as i8)
        .collect::<Vec<_>>();
    let lhs_i8_buf = crate::ops::cuda::upload_i8_storage(&lhs_i8).expect("upload lhs i8");
    let rhs_i8_buf = crate::ops::cuda::upload_i8_storage(&rhs_i8).expect("upload rhs i8");
    let start = std::time::Instant::now();
    let i8_out = crate::ops::cuda::binary_i8_buffer_no_host(
        &lhs_i8_buf,
        0.05,
        &rhs_i8_buf,
        0.05,
        crate::ops::cuda::BinaryOp::Mul,
    )
    .expect("CUDA resident i8 binary perf path");
    crate::ops::cuda::synchronize().expect("sync i8 binary");
    let i8_us = start.elapsed().as_secs_f64() * 1.0e6;
    let i8_vals = crate::ops::cuda::download_f32(&i8_out).expect("download i8 binary");

    let _i8_typed_warmup = crate::ops::cuda::binary_i8_typed_output_buffer_no_host(
        &lhs_i8_buf,
        0.05,
        &rhs_i8_buf,
        0.05,
        crate::ops::cuda::BinaryOp::Mul,
    )
    .expect("warm up CUDA resident typed-output i8 binary");
    crate::ops::cuda::synchronize().expect("sync typed-output i8 binary warmup");
    let start = std::time::Instant::now();
    let (i8_typed_out, i8_typed_scale) = crate::ops::cuda::binary_i8_typed_output_buffer_no_host(
        &lhs_i8_buf,
        0.05,
        &rhs_i8_buf,
        0.05,
        crate::ops::cuda::BinaryOp::Mul,
    )
    .expect("CUDA resident typed-output i8 binary perf path");
    crate::ops::cuda::synchronize().expect("sync typed-output i8 binary");
    let i8_typed_us = start.elapsed().as_secs_f64() * 1.0e6;
    let i8_typed_vals = crate::ops::cuda::download_i8_storage(&i8_typed_out)
        .expect("download typed-output i8 binary")
        .into_iter()
        .map(|value| value as f32 * i8_typed_scale)
        .collect::<Vec<_>>();

    let max_err = |vals: &[f32]| {
        vals.iter()
            .zip(f32_ref.iter())
            .map(|(&got, &expect)| (got - expect).abs())
            .fold(0.0f32, f32::max)
    };

    println!(
        "cuda lowp binary mul len={len}: f32={f32_us:.1}us, f16={f16_us:.1}us max_err={:.5}, bf16={bf16_us:.1}us max_err={:.5}, i8_f32_out={i8_us:.1}us max_err={:.5}, i8_typed_out={i8_typed_us:.1}us kernel_requant_err={:.5} total_err={:.5}",
        max_err(&f16_vals),
        max_err(&bf16_vals),
        max_err(&i8_vals),
        i8_typed_vals
            .iter()
            .zip(i8_vals.iter())
            .map(|(&got, &expect)| (got - expect).abs())
            .fold(0.0f32, f32::max),
        max_err(&i8_typed_vals)
    );

    let start = std::time::Instant::now();
    let (_, f32_sum) = crate::ops::cuda::sum_f32(&lhs_f32).expect("CUDA f32 sum perf path");
    crate::ops::cuda::synchronize().expect("sync f32 sum");
    let f32_sum_us = start.elapsed().as_secs_f64() * 1.0e6;
    let f32_sum = f32_sum[0];

    let start = std::time::Instant::now();
    let (_, f16_sum) =
        crate::ops::cuda::sum_f16_buffer(&lhs_f16_buf).expect("CUDA f16 sum perf path");
    crate::ops::cuda::synchronize().expect("sync f16 sum");
    let f16_sum_us = start.elapsed().as_secs_f64() * 1.0e6;
    let f16_sum = f16_sum[0];

    let start = std::time::Instant::now();
    let (_, bf16_sum) =
        crate::ops::cuda::sum_bf16_buffer(&lhs_bf16_buf).expect("CUDA bf16 sum perf path");
    crate::ops::cuda::synchronize().expect("sync bf16 sum");
    let bf16_sum_us = start.elapsed().as_secs_f64() * 1.0e6;
    let bf16_sum = bf16_sum[0];

    let start = std::time::Instant::now();
    let (_, i8_sum) =
        crate::ops::cuda::sum_i8_buffer(&lhs_i8_buf, 0.05).expect("CUDA i8 sum perf path");
    crate::ops::cuda::synchronize().expect("sync i8 sum");
    let i8_sum_us = start.elapsed().as_secs_f64() * 1.0e6;
    let i8_sum = i8_sum[0];
    let i8_sum_quant_ref = lhs_i8.iter().map(|&v| v as i64).sum::<i64>() as f32 * 0.05;

    println!(
        "cuda lowp sum len={len}: f32={f32_sum_us:.1}us, f16={f16_sum_us:.1}us abs_err={:.5}, bf16={bf16_sum_us:.1}us abs_err={:.5}, i8={i8_sum_us:.1}us kernel_err={:.5} quant_err={:.5}",
        (f16_sum - f32_sum).abs(),
        (bf16_sum - f32_sum).abs(),
        (i8_sum - i8_sum_quant_ref).abs(),
        (i8_sum - f32_sum).abs()
    );

    let grad = (0..len)
        .map(|i| ((i * 7) as f32 % 127.0) / 97.0 - 0.5)
        .collect::<Vec<_>>();
    let grad_buf = crate::ops::cuda::upload_f32(&grad).expect("upload grad f32");

    let start = std::time::Instant::now();
    let (f32_lhs_grad, f32_rhs_grad) = crate::ops::cuda::binary_backward_f32_buffers(
        &lhs_f32,
        &rhs_f32,
        &grad_buf,
        crate::ops::cuda::BinaryOp::Mul,
    )
    .expect("CUDA f32 binary backward perf path");
    crate::ops::cuda::synchronize().expect("sync f32 backward");
    let f32_backward_us = start.elapsed().as_secs_f64() * 1.0e6;
    let f32_lhs_grad =
        crate::ops::cuda::download_f32(&f32_lhs_grad).expect("download f32 lhs grad");
    let f32_rhs_grad =
        crate::ops::cuda::download_f32(&f32_rhs_grad).expect("download f32 rhs grad");

    let start = std::time::Instant::now();
    let f16_lhs_grad = crate::ops::cuda::mul_grad_f16_buffer_no_host(&grad_buf, &rhs_f16_buf)
        .expect("CUDA f16 lhs grad perf path");
    let f16_rhs_grad = crate::ops::cuda::mul_grad_f16_buffer_no_host(&grad_buf, &lhs_f16_buf)
        .expect("CUDA f16 rhs grad perf path");
    crate::ops::cuda::synchronize().expect("sync f16 backward");
    let f16_backward_us = start.elapsed().as_secs_f64() * 1.0e6;
    let f16_lhs_grad =
        crate::ops::cuda::download_f32(&f16_lhs_grad).expect("download f16 lhs grad");
    let f16_rhs_grad =
        crate::ops::cuda::download_f32(&f16_rhs_grad).expect("download f16 rhs grad");

    let start = std::time::Instant::now();
    let bf16_lhs_grad = crate::ops::cuda::mul_grad_bf16_buffer_no_host(&grad_buf, &rhs_bf16_buf)
        .expect("CUDA bf16 lhs grad perf path");
    let bf16_rhs_grad = crate::ops::cuda::mul_grad_bf16_buffer_no_host(&grad_buf, &lhs_bf16_buf)
        .expect("CUDA bf16 rhs grad perf path");
    crate::ops::cuda::synchronize().expect("sync bf16 backward");
    let bf16_backward_us = start.elapsed().as_secs_f64() * 1.0e6;
    let bf16_lhs_grad =
        crate::ops::cuda::download_f32(&bf16_lhs_grad).expect("download bf16 lhs grad");
    let bf16_rhs_grad =
        crate::ops::cuda::download_f32(&bf16_rhs_grad).expect("download bf16 rhs grad");

    let start = std::time::Instant::now();
    let i8_lhs_grad = crate::ops::cuda::mul_grad_i8_buffer_no_host(&grad_buf, &rhs_i8_buf, 0.05)
        .expect("CUDA i8 lhs grad perf path");
    let i8_rhs_grad = crate::ops::cuda::mul_grad_i8_buffer_no_host(&grad_buf, &lhs_i8_buf, 0.05)
        .expect("CUDA i8 rhs grad perf path");
    crate::ops::cuda::synchronize().expect("sync i8 backward");
    let i8_backward_us = start.elapsed().as_secs_f64() * 1.0e6;
    let i8_lhs_grad = crate::ops::cuda::download_f32(&i8_lhs_grad).expect("download i8 lhs grad");
    let i8_rhs_grad = crate::ops::cuda::download_f32(&i8_rhs_grad).expect("download i8 rhs grad");

    let max_err_against = |vals: &[f32], reference: &[f32]| {
        vals.iter()
            .zip(reference.iter())
            .map(|(&got, &expect)| (got - expect).abs())
            .fold(0.0f32, f32::max)
    };
    let max_pair_err = |lhs_vals: &[f32], rhs_vals: &[f32]| {
        max_err_against(lhs_vals, &f32_lhs_grad).max(max_err_against(rhs_vals, &f32_rhs_grad))
    };

    println!(
        "cuda lowp binary mul backward len={len}: f32={f32_backward_us:.1}us, f16={f16_backward_us:.1}us max_err={:.5}, bf16={bf16_backward_us:.1}us max_err={:.5}, i8={i8_backward_us:.1}us max_err={:.5}",
        max_pair_err(&f16_lhs_grad, &f16_rhs_grad),
        max_pair_err(&bf16_lhs_grad, &bf16_rhs_grad),
        max_pair_err(&i8_lhs_grad, &i8_rhs_grad)
    );
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_empty_binary_and_sum_stay_stable_in_strict_mode() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    crate::ops::cuda::set_enabled(true);
    crate::autograd::set_strict_device_execution(true);

    let lhs = make_tensor(&[0, 4], vec![], DType::BF16).to_cuda();
    let rhs = make_tensor(&[0, 4], vec![], DType::BF16).to_cuda();
    let add_out = no_grad(|| lhs.clone() + rhs.clone());
    assert!(add_out.is_cuda());
    assert_eq!(add_out.dtype(), DType::BF16);
    assert_eq!(add_out.shape_vec(), vec![0, 4]);
    assert_eq!(add_out.len(), 0);

    let sum_out = no_grad(|| sum(&add_out));
    assert!(sum_out.is_cuda());
    assert_eq!(sum_out.data_ref().first().copied(), Some(0.0));

    let lhs_train = make_training_tensor(&[0, 4], vec![]).to_cuda();
    let rhs_train = make_training_tensor(&[0, 4], vec![]).to_cuda();
    let train_out = lhs_train.clone() + rhs_train.clone();
    assert!(train_out.is_cuda());
    assert_eq!(train_out.len(), 0);
    let loss = sum(&train_out);
    assert!(loss.is_cuda());
    loss.backward();

    let lhs_grad = lhs_train
        .grad()
        .expect("empty CUDA lhs grad should be recorded");
    let rhs_grad = rhs_train
        .grad()
        .expect("empty CUDA rhs grad should be recorded");
    assert_eq!(lhs_grad.len(), 0);
    assert_eq!(rhs_grad.len(), 0);

    crate::autograd::set_strict_device_execution(false);
    crate::ops::cuda::set_enabled(false);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_sum_backward_matches_cpu_reference_in_strict_mode() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let shape = [128, 128];
    let values = (0..(shape[0] * shape[1]))
        .map(|i| (i as f32 % 67.0) / 23.0 - 1.25)
        .collect::<Vec<_>>();

    crate::ops::cuda::set_enabled(false);
    crate::autograd::set_strict_device_execution(false);
    let cpu_input = make_training_tensor(&shape, values.clone());
    let cpu_out = sum(&cpu_input);
    cpu_out.backward();
    let cpu_grad = cpu_input
        .grad()
        .expect("CPU sum backward should populate input grad");
    let cpu_sum = cpu_out.data_ref().first().copied().unwrap();

    crate::ops::cuda::set_enabled(true);
    crate::autograd::set_strict_device_execution(true);
    let cuda_input = make_training_tensor(&shape, values).to_cuda();
    let cuda_out = sum(&cuda_input);
    assert!(cuda_out.is_cuda());
    cuda_out.backward();
    let cuda_grad = cuda_input
        .grad()
        .expect("CUDA sum backward should populate input grad");
    assert!(cuda_input.cloned_cuda_f32_grad().is_some());
    let cuda_sum = cuda_out.data_ref().first().copied().unwrap();

    crate::autograd::set_strict_device_execution(false);
    crate::ops::cuda::set_enabled(false);

    assert!(
        (cuda_sum - cpu_sum).abs() <= 1e-2,
        "CUDA sum got {cuda_sum}, CPU expected {cpu_sum}"
    );
    for (idx, (got, expect)) in cuda_grad.iter().zip(cpu_grad.iter()).enumerate() {
        assert!(
            (got - expect).abs() <= 1e-6,
            "sum grad mismatch at {idx}: got {got}, expect {expect}"
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_binary_backward_matches_cpu_reference_in_strict_mode() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let shape = [64, 256];
    let lhs_values = (0..(shape[0] * shape[1]))
        .map(|i| (i as f32 % 37.0) / 17.0 - 0.9)
        .collect::<Vec<_>>();
    let rhs_values = (0..(shape[0] * shape[1]))
        .map(|i| ((i * 11) as f32 % 43.0) / 19.0 - 0.7)
        .collect::<Vec<_>>();

    for op in [BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul] {
        crate::ops::cuda::set_enabled(false);
        crate::autograd::set_strict_device_execution(false);
        let cpu_lhs = make_training_tensor(&shape, lhs_values.clone());
        let cpu_rhs = make_training_tensor(&shape, rhs_values.clone());
        let cpu_out = match op {
            BinaryOp::Add => sum(&(cpu_lhs.clone() + cpu_rhs.clone())),
            BinaryOp::Sub => sum(&(cpu_lhs.clone() - cpu_rhs.clone())),
            BinaryOp::Mul => sum(&(cpu_lhs.clone() * cpu_rhs.clone())),
        };
        cpu_out.backward();
        let cpu_lhs_grad = cpu_lhs
            .grad()
            .expect("CPU binary backward should populate lhs grad");
        let cpu_rhs_grad = cpu_rhs
            .grad()
            .expect("CPU binary backward should populate rhs grad");

        crate::ops::cuda::set_enabled(true);
        crate::autograd::set_strict_device_execution(true);
        let cuda_lhs = make_training_tensor(&shape, lhs_values.clone()).to_cuda();
        let cuda_rhs = make_training_tensor(&shape, rhs_values.clone()).to_cuda();
        let binary_out = match op {
            BinaryOp::Add => cuda_lhs.clone() + cuda_rhs.clone(),
            BinaryOp::Sub => cuda_lhs.clone() - cuda_rhs.clone(),
            BinaryOp::Mul => cuda_lhs.clone() * cuda_rhs.clone(),
        };
        assert!(binary_out.is_cuda());
        assert!(!binary_out.has_host_f32_data());
        let cuda_out = match op {
            BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul => sum(&binary_out),
        };
        assert!(cuda_out.is_cuda());
        cuda_out.backward();
        assert!(!cuda_lhs.has_host_grad());
        assert!(!cuda_rhs.has_host_grad());
        assert!(cuda_lhs.cloned_cuda_f32_grad().is_some());
        assert!(cuda_rhs.cloned_cuda_f32_grad().is_some());
        let cuda_lhs_grad = cuda_lhs
            .grad()
            .expect("CUDA binary backward should populate lhs grad");
        let cuda_rhs_grad = cuda_rhs
            .grad()
            .expect("CUDA binary backward should populate rhs grad");

        for (idx, (got, expect)) in cuda_lhs_grad.iter().zip(cpu_lhs_grad.iter()).enumerate() {
            assert!(
                (got - expect).abs() <= 1e-5,
                "{op:?} lhs grad mismatch at {idx}: got {got}, expect {expect}"
            );
        }
        for (idx, (got, expect)) in cuda_rhs_grad.iter().zip(cpu_rhs_grad.iter()).enumerate() {
            assert!(
                (got - expect).abs() <= 1e-5,
                "{op:?} rhs grad mismatch at {idx}: got {got}, expect {expect}"
            );
        }
    }

    crate::autograd::set_strict_device_execution(false);
    crate::ops::cuda::set_enabled(false);
}

#[cfg(feature = "cuda")]
#[test]
fn backward_traverses_node_with_cuda_only_grad() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    crate::ops::cuda::set_enabled(true);
    crate::autograd::set_strict_device_execution(true);

    let shape = [64, 256];
    let values = (0..(shape[0] * shape[1]))
        .map(|i| (i as f32 % 29.0) / 11.0 - 0.75)
        .collect::<Vec<_>>();
    let input = make_training_tensor(&shape, values).to_cuda();
    let out = input.clone() + input.clone();
    assert!(out.is_cuda());
    assert!(!out.has_host_f32_data());

    let grad = vec![1.0f32; out.len()];
    let grad_buffer =
        crate::ops::cuda::upload_f32(&grad).expect("test should upload CUDA-only grad");
    out.add_cuda_grad_buffer_only(grad_buffer);
    out.backward();
    assert!(
        !out.has_host_f32_data(),
        "strict CUDA-only backward should not materialize output data"
    );

    let input_grad = input
        .grad()
        .expect("CUDA-only grad should still propagate to input");
    assert!(input.cloned_cuda_f32_grad().is_some());
    for (idx, got) in input_grad.iter().enumerate() {
        assert!(
            (*got - 2.0).abs() <= 1e-6,
            "CUDA-only backward grad mismatch at {idx}: got {got}"
        );
    }

    crate::autograd::set_strict_device_execution(false);
    crate::ops::cuda::set_enabled(false);
}

#[cfg(feature = "cuda")]
#[test]
fn backward_seeds_no_host_cuda_output_without_host_grad() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    crate::ops::cuda::set_enabled(true);
    crate::autograd::set_strict_device_execution(true);

    let shape = [64, 256];
    let values = (0..(shape[0] * shape[1]))
        .map(|i| (i as f32 % 23.0) / 9.0 - 0.6)
        .collect::<Vec<_>>();
    let input = make_training_tensor(&shape, values).to_cuda();
    let out = input.clone() + input.clone();
    assert!(!out.has_host_f32_data());

    out.backward();
    assert!(
        !out.has_host_f32_data(),
        "strict CUDA backward seed should stay CUDA-only for no-host outputs"
    );

    let input_grad = input.grad().expect("input grad");
    assert!(input.cloned_cuda_f32_grad().is_some());
    for (idx, got) in input_grad.iter().enumerate() {
        assert!(
            (*got - 2.0).abs() <= 1e-6,
            "seeded CUDA backward grad mismatch at {idx}: got {got}"
        );
    }

    crate::autograd::set_strict_device_execution(false);
    crate::ops::cuda::set_enabled(false);
}

#[cfg(feature = "cuda")]
#[test]
fn backward_materializes_cuda_only_grad_for_host_backed_cuda_node() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    crate::ops::cuda::set_enabled(true);
    crate::autograd::set_strict_device_execution(true);

    let shape = [128];
    let values = (0..shape[0])
        .map(|i| i as f32 / 17.0 - 0.5)
        .collect::<Vec<_>>();
    let input = make_training_tensor(&shape, values).to_cuda();
    let out = sum(&input);
    assert!(out.is_cuda());
    assert!(
        out.has_host_f32_data(),
        "sum currently keeps a host scalar result"
    );

    let grad_buffer =
        crate::ops::cuda::upload_f32(&[3.0]).expect("test should upload CUDA-only grad");
    out.backward_with_cuda_grad(grad_buffer);

    let input_grad = input
        .grad()
        .expect("CUDA-only sum grad should still propagate to input");
    assert!(input.cloned_cuda_f32_grad().is_some());
    for (idx, got) in input_grad.iter().enumerate() {
        assert!(
            (*got - 3.0).abs() <= 1e-6,
            "CUDA-only sum backward grad mismatch at {idx}: got {got}"
        );
    }

    crate::autograd::set_strict_device_execution(false);
    crate::ops::cuda::set_enabled(false);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_only_grad_accumulates_after_host_grad() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    crate::ops::cuda::set_enabled(true);
    crate::autograd::set_strict_device_execution(true);

    let input = make_training_tensor(&[3], vec![1.0, 2.0, 3.0]).to_cuda();
    let out = input.clone() + input.clone();
    out.add_grad(ArrayD::from_elem(IxDyn(&[3]), 1.0));
    let cuda_grad = crate::ops::cuda::upload_f32(&[2.0, 2.0, 2.0]).expect("upload grad");
    out.add_cuda_grad_buffer_only(cuda_grad);

    // Accumulated buffers are passed explicitly as the upstream seed.
    // The numerical assertion also checks host/device accumulation above.
    out.backward_with_cuda_grad(out.cloned_cuda_f32_grad().expect("accumulated CUDA seed"));

    let grad = input.grad().expect("input grad");
    assert_eq!(
        grad.iter().copied().collect::<Vec<_>>(),
        vec![6.0, 6.0, 6.0]
    );
    assert!(input.cloned_cuda_f32_grad().is_some());
    crate::autograd::set_strict_device_execution(false);
    crate::ops::cuda::set_enabled(false);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_broadcast_binary_backward_matches_cpu_reference_in_strict_mode() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let lhs_shape = [32, 128];
    let rhs_shape = [128];
    let lhs_values = (0..(lhs_shape[0] * lhs_shape[1]))
        .map(|i| (i as f32 % 31.0) / 13.0 - 0.8)
        .collect::<Vec<_>>();
    let rhs_values = (0..rhs_shape[0])
        .map(|i| (i as f32 % 17.0) / 7.0 - 1.1)
        .collect::<Vec<_>>();

    for op in [BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul] {
        crate::ops::cuda::set_enabled(false);
        crate::autograd::set_strict_device_execution(false);
        let cpu_lhs = make_training_tensor(&lhs_shape, lhs_values.clone());
        let cpu_rhs = make_training_tensor(&rhs_shape, rhs_values.clone());
        let cpu_out = match op {
            BinaryOp::Add => sum(&(cpu_lhs.clone() + cpu_rhs.clone())),
            BinaryOp::Sub => sum(&(cpu_lhs.clone() - cpu_rhs.clone())),
            BinaryOp::Mul => sum(&(cpu_lhs.clone() * cpu_rhs.clone())),
        };
        cpu_out.backward();
        let cpu_lhs_grad = cpu_lhs
            .grad()
            .expect("CPU broadcast binary backward should populate lhs grad");
        let cpu_rhs_grad = cpu_rhs
            .grad()
            .expect("CPU broadcast binary backward should populate rhs grad");

        crate::ops::cuda::set_enabled(true);
        crate::autograd::set_strict_device_execution(true);
        let cuda_lhs = make_training_tensor(&lhs_shape, lhs_values.clone()).to_cuda();
        let cuda_rhs = make_training_tensor(&rhs_shape, rhs_values.clone()).to_cuda();
        let cuda_out = match op {
            BinaryOp::Add => sum(&(cuda_lhs.clone() + cuda_rhs.clone())),
            BinaryOp::Sub => sum(&(cuda_lhs.clone() - cuda_rhs.clone())),
            BinaryOp::Mul => sum(&(cuda_lhs.clone() * cuda_rhs.clone())),
        };
        assert!(cuda_out.is_cuda());
        cuda_out.backward();
        assert!(!cuda_lhs.has_host_grad());
        assert!(!cuda_rhs.has_host_grad());
        assert!(cuda_lhs.cloned_cuda_f32_grad().is_some());
        assert!(cuda_rhs.cloned_cuda_f32_grad().is_some());
        let cuda_lhs_grad = cuda_lhs
            .grad()
            .expect("CUDA broadcast binary backward should populate lhs grad");
        let cuda_rhs_grad = cuda_rhs
            .grad()
            .expect("CUDA broadcast binary backward should populate rhs grad");

        for (idx, (got, expect)) in cuda_lhs_grad.iter().zip(cpu_lhs_grad.iter()).enumerate() {
            assert!(
                (got - expect).abs() <= 1e-5,
                "{op:?} broadcast lhs grad mismatch at {idx}: got {got}, expect {expect}"
            );
        }
        for (idx, (got, expect)) in cuda_rhs_grad.iter().zip(cpu_rhs_grad.iter()).enumerate() {
            assert!(
                (got - expect).abs() <= 1e-4,
                "{op:?} broadcast rhs grad mismatch at {idx}: got {got}, expect {expect}"
            );
        }
    }

    crate::autograd::set_strict_device_execution(false);
    crate::ops::cuda::set_enabled(false);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_async_general_broadcast_chain_matches_cpu_after_metadata_and_buffer_reuse() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    {
        let a = make_tensor(
            &[2, 1, 1, 3],
            vec![1.0, -2.0, 0.5, 3.0, -4.0, 2.0],
            DType::F32,
        );
        let b = make_tensor(
            &[1, 4, 2, 3],
            (0..24).map(|i| (i as f32 % 11.0) * 0.2 - 0.75).collect(),
            DType::F32,
        );
        let c = make_tensor(
            &[2, 4, 1, 1],
            (0..8).map(|i| 0.5 + i as f32 * 0.125).collect(),
            DType::F32,
        );
        let d = make_tensor(
            &[1, 1, 2, 3],
            vec![0.25, -0.5, 1.0, -1.5, 0.75, 0.125],
            DType::F32,
        );
        let cpu_out = no_grad(|| ((a.clone() + b.clone()) * c.clone()) - d.clone());

        crate::ops::cuda::set_enabled(true);
        crate::autograd::set_strict_device_execution(true);
        let cuda_out = no_grad(|| ((a.to_cuda() + b.to_cuda()) * c.to_cuda()) - d.to_cuda());
        assert!(cuda_out.is_cuda());
        assert!(!cuda_out.has_host_f32_data());
        crate::autograd::set_strict_device_execution(false);
        crate::ops::cuda::set_enabled(false);

        for (idx, (got, expected)) in cuda_out
            .data_ref()
            .iter()
            .zip(cpu_out.data_ref().iter())
            .enumerate()
        {
            assert!(
                (got - expected).abs() <= 1e-6,
                "async general broadcast chain mismatch at {idx}: got {got}, expected {expected}"
            );
        }
    }
    crate::ops::cuda::release_cached_memory().expect("release CUDA cached memory");
}

#[cfg(feature = "cuda")]
#[test]
#[should_panic(expected = "same device")]
fn add_panics_on_mixed_devices() {
    if !crate::ops::cuda::is_available() {
        panic!("same device");
    }

    let lhs = make_tensor(&[2], vec![1.0, 2.0], DType::F32).to_cuda();
    let rhs = make_tensor(&[2], vec![3.0, 4.0], DType::F32);
    let _ = no_grad(|| lhs + rhs);
}
