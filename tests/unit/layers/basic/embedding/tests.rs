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
fn embedding_accepts_integer_indices() {
    let emb = Embedding::new(8, 4);
    let indices = make_tensor(&[1, 3], vec![0.0, 2.0, 7.0], DType::F32);
    let out = no_grad(|| emb.forward(&indices));
    assert_eq!(out.shape_vec(), vec![1, 3, 4]);
}

#[test]
#[should_panic(expected = "must be an integer")]
fn embedding_rejects_fractional_indices() {
    let emb = Embedding::new(8, 4);
    let indices = make_tensor(&[1, 1], vec![1.5], DType::F32);
    no_grad(|| {
        let _ = emb.forward(&indices);
    });
}

#[test]
#[should_panic(expected = "must be >= 0")]
fn embedding_rejects_negative_indices() {
    let emb = Embedding::new(8, 4);
    let indices = make_tensor(&[1, 1], vec![-1.0], DType::BF16);
    no_grad(|| {
        let _ = emb.forward(&indices);
    });
}

#[test]
#[should_panic(expected = "must be finite")]
fn embedding_rejects_nan_indices() {
    let emb = Embedding::new(8, 4);
    let indices = make_tensor(&[1, 1], vec![f32::NAN], DType::F32);
    no_grad(|| {
        let _ = emb.forward(&indices);
    });
}

#[test]
fn embedding_no_grad_preserves_bf16_weight_dtype() {
    let emb = Embedding::new(4, 2);
    emb.weight.set_array_f32_with_dtype(
        Array::from_shape_vec(IxDyn(&[4, 2]), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
            .expect("weight shape mismatch")
            .into_dyn(),
        DType::BF16,
    );
    let indices = make_tensor(&[1, 2], vec![1.0, 3.0], DType::F32);

    let out = no_grad(|| emb.forward(&indices));
    assert_eq!(out.dtype(), DType::BF16);
    out.with_storage_view(|view| match view {
        TensorStorageView::BF16(view) => {
            let vals = view.iter().map(|v| v.to_f32()).collect::<Vec<_>>();
            assert_eq!(vals, vec![2.0, 3.0, 6.0, 7.0]);
        }
        TensorStorageView::F16(_) => {
            panic!("bf16 embedding output should stay bf16 in no-grad")
        }
        TensorStorageView::F32(_) => {
            panic!("bf16 embedding output should stay bf16 in no-grad")
        }
    });
}

#[test]
fn embedding_no_grad_preserves_i8_weight_dtype() {
    let emb = Embedding::new(4, 2);
    emb.weight.set_array_f32_with_dtype(
        Array::from_shape_vec(IxDyn(&[4, 2]), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
            .expect("weight shape mismatch")
            .into_dyn(),
        DType::I8,
    );
    let indices = make_tensor(&[1, 2], vec![1.0, 3.0], DType::F32);

    let out = no_grad(|| emb.forward(&indices));
    assert_eq!(out.dtype(), DType::I8);
    let ref_vals = no_grad(|| {
        let ref_emb = Embedding::new(4, 2);
        ref_emb.weight.set_array_f32_with_dtype(
            Array::from_shape_vec(IxDyn(&[4, 2]), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
                .expect("weight shape mismatch")
                .into_dyn(),
            DType::I8,
        );
        ref_emb
            .forward(&indices)
            .data_ref()
            .iter()
            .copied()
            .collect::<Vec<_>>()
    });
    let out_vals = out.data_ref().iter().copied().collect::<Vec<_>>();
    assert_eq!(out_vals, ref_vals);
}

#[test]
fn embedding_explicit_dtype_overrides_global_default() {
    with_precision_config(
        PrecisionConfig {
            parameter_dtype: DType::BF16,
            runtime_dtype: DType::F32,
            allow_parameter_dtype_copies: false,
        },
        || {
            let emb = Embedding::new_with_dtype(8, 4, DType::F32);
            assert_eq!(emb.weight.dtype(), DType::F32);
        },
    );
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_embedding_matches_cpu_reference_in_strict_mode() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let weight_values = vec![
        0.0, 1.0, 2.0, 3.0, 4.0, 5.0, -1.0, -2.0, -3.0, 1.5, 2.5, 3.5, 7.0, 8.0, 9.0, -4.0, -5.0,
        -6.0,
    ];
    let emb = Embedding::new(6, 3);
    emb.weight.set_array_f32_with_dtype(
        Array::from_shape_vec(IxDyn(&[6, 3]), weight_values.clone())
            .expect("weight shape mismatch")
            .into_dyn(),
        DType::BF16,
    );
    emb.weight.to_cuda_inplace();
    let emb_ref = Embedding::new(6, 3);
    emb_ref.weight.set_array_f32_with_dtype(
        Array::from_shape_vec(IxDyn(&[6, 3]), weight_values)
            .expect("weight shape mismatch")
            .into_dyn(),
        DType::BF16,
    );

    let indices = make_tensor(&[2, 2], vec![0.0, 2.0, 5.0, 1.0], DType::F32).to_cuda();
    assert!(emb.weight.0.borrow().cuda_f32_data.is_none());

    crate::ops::cuda::set_enabled(true);
    crate::device::set_strict_device_execution(true);
    let cuda_out = no_grad(|| emb.forward(&indices));
    crate::device::set_strict_device_execution(false);
    crate::ops::cuda::set_enabled(false);
    assert!(emb.weight.0.borrow().cuda_f32_data.is_none());

    let cpu_out = no_grad(|| emb_ref.forward(&indices.to_cpu()));
    assert!(cuda_out.is_cuda());
    assert_eq!(cuda_out.dtype(), DType::BF16);
    {
        let inner = cuda_out.0.borrow();
        assert!(
            !inner.has_f32_data,
            "CUDA embedding no-grad output should not eagerly materialize host f32 data"
        );
        assert!(
            inner.cuda_f32_data.is_none(),
            "CUDA embedding no-grad output should not keep a resident f32 compute buffer for bf16 output"
        );
        assert!(
            inner.cuda_bf16_data.is_some(),
            "CUDA embedding no-grad output should keep resident bf16 storage"
        );
    }

    for (got, expect) in cuda_out.data_ref().iter().zip(cpu_out.data_ref().iter()) {
        assert!((got - expect).abs() < 2e-2, "got {got}, expect {expect}");
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_empty_embedding_no_grad_stays_on_cuda_in_strict_mode() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let emb = Embedding::new(8, 4);
    emb.weight.set_array_f32_with_dtype(
        Array::from_shape_vec(IxDyn(&[8, 4]), (0..32).map(|v| v as f32 / 8.0).collect())
            .expect("weight shape mismatch")
            .into_dyn(),
        DType::BF16,
    );
    emb.weight.to_cuda_inplace();
    let indices = make_tensor(&[0], vec![], DType::F32).to_cuda();

    crate::ops::cuda::set_enabled(true);
    crate::device::set_strict_device_execution(true);
    let out = no_grad(|| emb.forward(&indices));
    crate::device::set_strict_device_execution(false);
    crate::ops::cuda::set_enabled(false);

    assert!(out.is_cuda());
    assert_eq!(out.dtype(), DType::BF16);
    assert_eq!(out.shape_vec(), vec![0, 4]);
    assert_eq!(out.len(), 0);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_embedding_backward_matches_cpu_reference_in_strict_mode() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let weight_values = vec![
        0.0, 1.0, 2.0, 3.0, 4.0, 5.0, -1.0, -2.0, -3.0, 1.5, 2.5, 3.5, 7.0, 8.0, 9.0, -4.0, -5.0,
        -6.0,
    ];
    let indices_cpu = make_tensor(&[2, 3], vec![0.0, 2.0, 2.0, 5.0, 1.0, 0.0], DType::F32);
    let coeff_cpu = make_tensor(
        &[2, 3, 3],
        vec![
            0.5, -1.0, 2.0, 0.25, -0.75, 1.5, 1.25, 0.5, -0.25, -1.5, 0.75, 0.0, 0.1, -0.2, 0.3,
            2.0, -1.0, 0.5,
        ],
        DType::F32,
    );

    let emb_cuda = Embedding::new(6, 3);
    emb_cuda.weight.set_array_f32_with_dtype(
        Array::from_shape_vec(IxDyn(&[6, 3]), weight_values.clone())
            .expect("weight shape mismatch")
            .into_dyn(),
        DType::F32,
    );
    emb_cuda.weight.to_cuda_inplace();
    let indices_cuda = indices_cpu.to_cuda();
    let coeff_cuda = coeff_cpu.to_cuda();

    crate::ops::cuda::set_enabled(true);
    crate::device::set_strict_device_execution(true);
    let cuda_out = emb_cuda.forward(&indices_cuda);
    let cuda_loss = crate::ops::arithmetic::sum(&(&cuda_out * &coeff_cuda));
    cuda_loss.backward();
    assert!(emb_cuda.weight.cloned_cuda_f32_grad().is_some());
    assert!(!emb_cuda.weight.has_host_grad());
    crate::device::set_strict_device_execution(false);
    crate::ops::cuda::set_enabled(false);

    let emb_cpu = Embedding::new(6, 3);
    emb_cpu.weight.set_array_f32_with_dtype(
        Array::from_shape_vec(IxDyn(&[6, 3]), weight_values)
            .expect("weight shape mismatch")
            .into_dyn(),
        DType::F32,
    );
    let cpu_out = emb_cpu.forward(&indices_cpu);
    let cpu_loss = crate::ops::arithmetic::sum(&(&cpu_out * &coeff_cpu));
    cpu_loss.backward();

    let cuda_grad = emb_cuda.weight.grad().expect("cuda embedding weight grad");
    let cpu_grad = emb_cpu.weight.grad().expect("cpu embedding weight grad");
    for (got, expect) in cuda_grad.iter().zip(cpu_grad.iter()) {
        assert!((got - expect).abs() < 1e-5, "got {got}, expect {expect}");
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_embedding_preserves_i8_dtype_in_strict_mode() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let weight_values = vec![
        0.0, 1.0, 2.0, 3.0, 4.0, 5.0, -1.0, -2.0, -3.0, 1.5, 2.5, 3.5,
    ];
    let emb = Embedding::new(4, 3);
    emb.weight.set_array_f32_with_dtype(
        Array::from_shape_vec(IxDyn(&[4, 3]), weight_values.clone())
            .expect("weight shape mismatch")
            .into_dyn(),
        DType::I8,
    );
    emb.weight.to_cuda_inplace();
    let emb_ref = Embedding::new(4, 3);
    emb_ref.weight.set_array_f32_with_dtype(
        Array::from_shape_vec(IxDyn(&[4, 3]), weight_values)
            .expect("weight shape mismatch")
            .into_dyn(),
        DType::I8,
    );

    let indices = make_tensor(&[2, 2], vec![0.0, 2.0, 3.0, 1.0], DType::F32).to_cuda();
    assert!(emb.weight.0.borrow().cuda_f32_data.is_none());

    crate::ops::cuda::set_enabled(true);
    crate::device::set_strict_device_execution(true);
    let cuda_out = no_grad(|| emb.forward(&indices));
    crate::device::set_strict_device_execution(false);
    crate::ops::cuda::set_enabled(false);
    assert!(emb.weight.0.borrow().cuda_f32_data.is_none());

    let cpu_out = no_grad(|| emb_ref.forward(&indices.to_cpu()));
    assert!(cuda_out.is_cuda());
    assert_eq!(cuda_out.dtype(), DType::I8);
    {
        let inner = cuda_out.0.borrow();
        assert!(
            !inner.has_f32_data,
            "I8 CUDA embedding output should not eagerly materialize host f32 data"
        );
        assert!(
            inner.cuda_f32_data.is_none(),
            "I8 CUDA embedding output should not keep a resident f32 compute buffer"
        );
        assert!(
            inner.cuda_i8_data.is_some(),
            "I8 CUDA embedding output should keep resident i8 storage"
        );
    }
    for (got, expect) in cuda_out.data_ref().iter().zip(cpu_out.data_ref().iter()) {
        assert!((got - expect).abs() < 1e-5, "got {got}, expect {expect}");
    }
}
