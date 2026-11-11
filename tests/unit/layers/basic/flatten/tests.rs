use super::*;
use crate::autograd::no_grad;
use crate::precision::DType;
use ndarray::{Array, IxDyn};

#[test]
fn flatten_empty_batches_preserves_shape_and_backward() {
    for (shape, expected) in [
        (vec![0, 3, 4], vec![0, 12]),
        (vec![2, 0, 3], vec![2, 0]),
        (vec![0], vec![0, 1]),
    ] {
        let input = Tensor::from_data_with_grad_flag(Array::zeros(IxDyn(&shape)), true);
        let output = Flatten::new().forward(input.clone());
        assert_eq!(output.shape_vec(), expected);
        output.backward();
        let grad = input.grad().expect("empty input gradient");
        assert_eq!(grad.shape(), shape);
        assert!(grad.is_empty());
        for dtype in [DType::F32, DType::F16, DType::BF16, DType::I8] {
            let input = Tensor::new_with_dtype(Array::zeros(IxDyn(&shape)), dtype);
            let output = no_grad(|| Flatten::new().forward(input));
            assert_eq!(output.shape_vec(), expected);
            assert_eq!(output.dtype(), dtype);
        }
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_flatten_empty_batch_preserves_device_and_gradient_shape() {
    if !crate::ops::cuda::is_available() {
        return;
    }
    let _cuda = crate::ops::cuda::set_enabled_scoped(true);
    let _strict = crate::device::set_strict_device_execution_scoped(true);
    let input = Tensor::from_data_with_grad_flag(Array::zeros(IxDyn(&[0, 3, 4])), true).to_cuda();
    let output = Flatten::new().forward(input.clone());
    assert!(output.is_cuda());
    assert_eq!(output.shape_vec(), vec![0, 12]);
    output.backward();
    assert_eq!(input.grad().unwrap().shape(), &[0, 3, 4]);
    for dtype in [DType::F32, DType::F16, DType::BF16, DType::I8] {
        let input = Tensor::new_with_dtype(Array::zeros(IxDyn(&[0, 3, 4])), dtype).to_cuda();
        let output = no_grad(|| Flatten::new().forward(input));
        assert!(output.is_cuda());
        assert_eq!(output.shape_vec(), vec![0, 12]);
        assert_eq!(output.dtype(), dtype);
    }
}

#[test]
fn flatten_no_grad_preserves_bf16_dtype() {
    let input = Tensor::from_array_no_grad(
        Array::from_shape_vec(IxDyn(&[2, 3, 4]), (0..24).map(|v| v as f32).collect())
            .expect("test tensor shape mismatch")
            .into_dyn(),
    );
    input.cast_inplace(DType::BF16);

    let out = no_grad(|| Flatten::new().forward(input));
    assert_eq!(out.shape_vec(), vec![2, 12]);
    assert_eq!(out.dtype(), DType::BF16);
}
