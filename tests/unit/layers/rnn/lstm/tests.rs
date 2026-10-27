use super::*;
use crate::autograd::{Tensor, set_strict_device_execution};
use ndarray::{Array, IxDyn};

fn grad_tensor(shape: &[usize], data: Vec<f32>) -> Tensor {
    Tensor::from_data_with_grad_flag(
        Array::from_shape_vec(IxDyn(shape), data)
            .expect("tensor shape mismatch")
            .into_dyn(),
        true,
    )
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_lstm_step_backward_runs_in_strict_mode() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let lstm = LSTM::new_with_dtype(3, 4, DType::F32);
    lstm.to_cuda();
    let input = grad_tensor(&[2, 3], (0..6).map(|i| i as f32 * 0.1 - 0.2).collect()).to_cuda();
    let h_prev = grad_tensor(&[2, 4], (0..8).map(|i| i as f32 * 0.05 - 0.1).collect()).to_cuda();
    let c_prev = grad_tensor(&[2, 4], (0..8).map(|i| i as f32 * -0.04 + 0.15).collect()).to_cuda();
    let h_coeff = Tensor::from_data_with_grad_flag(
        Array::from_shape_vec(
            IxDyn(&[2, 4]),
            (0..8).map(|i| i as f32 * 0.03 - 0.2).collect(),
        )
        .expect("h coeff shape mismatch")
        .into_dyn(),
        false,
    )
    .to_cuda();
    let c_coeff = Tensor::from_data_with_grad_flag(
        Array::from_shape_vec(
            IxDyn(&[2, 4]),
            (0..8).map(|i| i as f32 * -0.02 + 0.1).collect(),
        )
        .expect("c coeff shape mismatch")
        .into_dyn(),
        false,
    )
    .to_cuda();

    crate::ops::cuda::set_enabled(true);
    set_strict_device_execution(true);
    let (h_t, c_t) = lstm.forward_step(&input, &h_prev, &c_prev);
    assert!(h_t.is_cuda());
    assert!(c_t.is_cuda());
    assert_eq!(h_t.shape_vec(), vec![2, 4]);
    assert_eq!(c_t.shape_vec(), vec![2, 4]);
    let h_loss = crate::ops::arithmetic::sum(&(&h_t * &h_coeff));
    let c_loss = crate::ops::arithmetic::sum(&(&c_t * &c_coeff));
    let loss = h_loss + c_loss;
    loss.backward();
    assert!(input.cloned_cuda_f32_grad().is_some());
    assert!(h_prev.cloned_cuda_f32_grad().is_some());
    assert!(c_prev.cloned_cuda_f32_grad().is_some());
    assert!(!input.has_host_grad());
    assert!(!h_prev.has_host_grad());
    assert!(!c_prev.has_host_grad());
    for param in lstm.parameters() {
        assert!(param.cloned_cuda_f32_grad().is_some());
        assert!(!param.has_host_grad());
    }
    set_strict_device_execution(false);
    crate::ops::cuda::set_enabled(false);
}
