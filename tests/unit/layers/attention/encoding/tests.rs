use super::*;
use crate::autograd::no_grad;
#[cfg(feature = "cuda")]
use crate::autograd::set_strict_device_execution_scoped;
use crate::precision::{PrecisionConfig, with_precision_config};
use ndarray::{Array, IxDyn};
#[cfg(feature = "cuda")]
use std::sync::{Mutex, MutexGuard, OnceLock};

#[cfg(feature = "cuda")]
fn cuda_strict_test_guard() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .expect("cuda strict test mutex poisoned")
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

#[test]
fn rope_no_grad_preserves_bf16_input_dtype() {
    let rope = RotaryEmbedding::new(4, 8, 10000.0);
    let input_f32 = make_tensor(
        &[1, 1, 2, 4],
        vec![1.0, 2.0, 3.0, 4.0, -1.0, 0.5, 2.5, -3.0],
        DType::F32,
    );
    let input_bf16 = make_tensor(
        &[1, 1, 2, 4],
        vec![1.0, 2.0, 3.0, 4.0, -1.0, 0.5, 2.5, -3.0],
        DType::BF16,
    );

    let ref_out = no_grad(|| rope.forward(&input_f32, 0));
    let bf16_out = no_grad(|| rope.forward(&input_bf16, 0));

    assert_eq!(input_bf16.dtype(), DType::BF16);
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
        TensorStorageView::F16(_) => panic!("bf16 RoPE output should stay bf16 in no-grad"),
        TensorStorageView::F32(_) => panic!("bf16 RoPE output should stay bf16 in no-grad"),
    });
}

#[test]
fn rope_cache_creation_follows_runtime_dtype() {
    with_precision_config(
        PrecisionConfig {
            parameter_dtype: DType::F32,
            runtime_dtype: DType::BF16,
            allow_parameter_dtype_copies: false,
        },
        || {
            let rope = RotaryEmbedding::new(4, 8, 10000.0);
            assert_eq!(rope.cos_cache.dtype(), DType::BF16);
            assert_eq!(rope.sin_cache.dtype(), DType::BF16);
        },
    );
}

#[test]
fn rope_explicit_dtype_overrides_global_default() {
    with_precision_config(
        PrecisionConfig {
            parameter_dtype: DType::F32,
            runtime_dtype: DType::BF16,
            allow_parameter_dtype_copies: false,
        },
        || {
            let rope = RotaryEmbedding::new_with_dtype(4, 8, 10000.0, DType::F32);
            assert_eq!(rope.cos_cache.dtype(), DType::F32);
            assert_eq!(rope.sin_cache.dtype(), DType::F32);
        },
    );
}

#[test]
fn rope_no_grad_preserves_i8_input_dtype() {
    let rope = RotaryEmbedding::new(4, 8, 10000.0);
    let input_i8 = make_tensor(
        &[1, 1, 2, 4],
        vec![1.0, 2.0, 3.0, 4.0, -1.0, 0.5, 2.5, -3.0],
        DType::I8,
    );

    let out = no_grad(|| rope.forward(&input_i8, 0));
    assert_eq!(out.dtype(), DType::I8);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_rope_matches_cpu_reference_in_strict_mode() {
    if !crate::ops::cuda::is_available() {
        return;
    }
    let _test_guard = cuda_strict_test_guard();

    let rope = RotaryEmbedding::new_with_dtype(4, 8, 10000.0, DType::BF16);
    let rope_ref = RotaryEmbedding::new_with_dtype(4, 8, 10000.0, DType::BF16);
    rope.to_cuda();
    let input = make_tensor(
        &[1, 2, 2, 4],
        vec![
            1.0, 2.0, 3.0, 4.0, -1.0, 0.5, 2.5, -3.0, 0.25, -0.75, 1.5, -2.0, 3.0, -1.5, 0.5, 2.25,
        ],
        DType::BF16,
    )
    .to_cuda();
    assert!(input.0.borrow().cuda_f32_data.is_none());
    assert!(rope.cos_cache.0.borrow().cuda_f32_data.is_none());
    assert!(rope.sin_cache.0.borrow().cuda_f32_data.is_none());

    let cuda_out = {
        let _cuda_guard = crate::ops::cuda::set_enabled_scoped(true);
        let _strict_guard = set_strict_device_execution_scoped(true);
        let cuda_out = no_grad(|| rope.forward(&input, 1));
        assert!(input.0.borrow().cuda_f32_data.is_none());
        assert!(rope.cos_cache.0.borrow().cuda_f32_data.is_none());
        assert!(rope.sin_cache.0.borrow().cuda_f32_data.is_none());
        cuda_out
    };

    let cpu_out = no_grad(|| rope_ref.forward(&input.to_cpu(), 1));
    assert!(cuda_out.is_cuda());
    assert_eq!(cuda_out.dtype(), DType::BF16);
    {
        let inner = cuda_out.0.borrow();
        assert!(
            !inner.has_f32_data,
            "CUDA RoPE BF16 no-grad output should not eagerly materialize host f32 data"
        );
        assert!(
            inner.cuda_f32_data.is_none(),
            "CUDA RoPE BF16 no-grad output should not retain only an f32 compute buffer"
        );
        assert!(
            inner.cuda_bf16_data.is_some(),
            "CUDA RoPE BF16 no-grad output should keep resident bf16 storage"
        );
    }

    for (got, expect) in cuda_out.data_ref().iter().zip(cpu_out.data_ref().iter()) {
        assert!((got - expect).abs() < 2e-2, "got {got}, expect {expect}");
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_empty_rope_no_grad_stays_on_cuda_in_strict_mode() {
    if !crate::ops::cuda::is_available() {
        return;
    }
    let _test_guard = cuda_strict_test_guard();

    let rope = RotaryEmbedding::new_with_dtype(4, 8, 10000.0, DType::BF16);
    rope.to_cuda();
    let input = make_tensor(&[1, 2, 0, 4], vec![], DType::BF16).to_cuda();

    let out = {
        let _cuda_guard = crate::ops::cuda::set_enabled_scoped(true);
        let _strict_guard = set_strict_device_execution_scoped(true);
        no_grad(|| rope.forward(&input, 3))
    };

    assert!(out.is_cuda());
    assert_eq!(out.dtype(), DType::BF16);
    assert_eq!(out.shape_vec(), vec![1, 2, 0, 4]);
    assert_eq!(out.len(), 0);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_rope_backward_matches_cpu_reference_in_strict_mode() {
    if !crate::ops::cuda::is_available() {
        return;
    }
    let _test_guard = cuda_strict_test_guard();

    let rope = RotaryEmbedding::new_with_dtype(4, 8, 10000.0, DType::F32);
    let rope_ref = RotaryEmbedding::new_with_dtype(4, 8, 10000.0, DType::F32);
    rope.to_cuda();

    let input_values = vec![
        1.0, 2.0, 3.0, 4.0, -1.0, 0.5, 2.5, -3.0, 0.25, -0.75, 1.5, -2.0, 3.0, -1.5, 0.5, 2.25,
    ];
    let coeff_values = vec![
        0.5, -1.0, 2.0, 0.25, -0.75, 1.5, -0.5, 0.75, 0.1, 0.2, -0.3, 0.4, 1.25, -1.5, 0.6, -0.9,
    ];
    let input_cpu = Tensor::from_data_with_grad_flag(
        Array::from_shape_vec(IxDyn(&[1, 2, 2, 4]), input_values)
            .expect("input shape mismatch")
            .into_dyn(),
        true,
    );
    let coeff_cpu = Tensor::from_data_with_grad_flag(
        Array::from_shape_vec(IxDyn(&[1, 2, 2, 4]), coeff_values)
            .expect("coeff shape mismatch")
            .into_dyn(),
        false,
    );
    let input_cuda = input_cpu.to_cuda();
    let coeff_cuda = coeff_cpu.to_cuda();

    {
        let _cuda_guard = crate::ops::cuda::set_enabled_scoped(true);
        let _strict_guard = set_strict_device_execution_scoped(true);
        let cuda_out = rope.forward(&input_cuda, 1);
        let cuda_loss = crate::ops::arithmetic::sum(&(&cuda_out * &coeff_cuda));
        cuda_loss.backward();
        assert!(input_cuda.cloned_cuda_f32_grad().is_some());
        assert!(!input_cuda.has_host_grad());
    }

    let cpu_out = rope_ref.forward(&input_cpu, 1);
    let cpu_loss = crate::ops::arithmetic::sum(&(&cpu_out * &coeff_cpu));
    cpu_loss.backward();

    let cuda_grad = input_cuda.grad().expect("cuda RoPE input grad");
    let cpu_grad = input_cpu.grad().expect("cpu RoPE input grad");
    for (got, expect) in cuda_grad.iter().zip(cpu_grad.iter()) {
        assert!((got - expect).abs() < 1e-5, "got {got}, expect {expect}");
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_rope_preserves_i8_dtype_in_strict_mode() {
    if !crate::ops::cuda::is_available() {
        return;
    }
    let _test_guard = cuda_strict_test_guard();

    let rope = RotaryEmbedding::new_with_dtype(4, 8, 10000.0, DType::BF16);
    let rope_ref = RotaryEmbedding::new_with_dtype(4, 8, 10000.0, DType::BF16);
    rope.to_cuda();
    let input_values = vec![1.0, 2.0, 3.0, 4.0, -1.0, 0.5, 2.5, -3.0];
    let input = make_tensor(&[1, 1, 2, 4], input_values.clone(), DType::I8).to_cuda();
    assert!(input.0.borrow().cuda_f32_data.is_none());

    let out = {
        let _cuda_guard = crate::ops::cuda::set_enabled_scoped(true);
        let _strict_guard = set_strict_device_execution_scoped(true);
        no_grad(|| rope.forward(&input, 0))
    };

    assert!(out.is_cuda());
    assert_eq!(out.dtype(), DType::I8);
    {
        let inner = out.0.borrow();
        assert!(
            !inner.has_f32_data,
            "CUDA RoPE I8 no-grad output should not eagerly materialize host f32 data"
        );
        assert!(
            inner.cuda_f32_data.is_none(),
            "CUDA RoPE I8 no-grad output should not retain only an f32 compute buffer"
        );
        assert!(
            inner.cuda_i8_data.is_some(),
            "CUDA RoPE I8 no-grad output should keep resident i8 storage"
        );
        assert!(inner.i8_scale.is_some());
    }

    let cpu_input = make_tensor(&[1, 1, 2, 4], input_values, DType::I8);
    let cpu_out = no_grad(|| rope_ref.forward(&cpu_input, 0));
    for (got, expect) in out.data_ref().iter().zip(cpu_out.data_ref().iter()) {
        assert!((got - expect).abs() < 1e-5, "got {got}, expect {expect}");
    }
}
