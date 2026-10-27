//! Resident F16/BF16 optimizer state; parameter/gradient arithmetic remains F32.
use super::{adam_bias_corrections, finalize_cuda_param_update, should_use_cuda_pointer_batch};
use crate::autograd::{Device, Tensor};
use crate::ops::cuda::{self, CudaBuffer, LowpOptimizerConfig};
use crate::precision::DType;
use std::collections::BTreeMap;

pub(super) fn zero_state(param: &Tensor, dtype: DType) -> Tensor {
    let zeros =
        cuda::fill_scalar_f32_buffer(param.len(), 0.0).expect("initialize CUDA optimizer state");
    let native =
        cuda::f32_to_lowp_storage_no_host(&zeros, dtype).expect("allocate lowp CUDA state");
    Tensor::from_cuda_native_lowp_buffer_no_host_with_dtype(
        &param.shape_vec(),
        native,
        Device::Cuda,
        dtype,
        None,
    )
}

fn state_buffer(state: &Tensor, dtype: DType) -> CudaBuffer {
    assert_eq!(state.dtype(), dtype, "optimizer state dtype changed");
    assert!(state.is_cuda(), "optimizer state device changed");
    let (native_dtype, buffer, _) = state
        .cloned_cuda_native_lowp_buffer()
        .expect("resident optimizer state");
    assert_eq!(native_dtype, dtype);
    buffer
}

fn gradient_buffer(param: &Tensor) -> CudaBuffer {
    param.cloned_cuda_f32_grad().unwrap_or_else(|| {
        // Explicitly supplied host gradients need one upload, never a download.
        let grad = param
            .0
            .borrow()
            .grad
            .clone()
            .expect("pending optimizer gradient");
        cuda::upload_f32(&grad.iter().copied().collect::<Vec<_>>())
            .expect("upload supplied gradient")
    })
}

fn update_group(
    params: &[Tensor],
    first: &[Option<Tensor>],
    second: Option<&[Option<Tensor>]>,
    indices: &[usize],
    config: LowpOptimizerConfig,
    updated: &mut [bool],
) -> usize {
    if indices.is_empty() {
        return 0;
    }
    let p = indices
        .iter()
        .map(|&i| params[i].cloned_cuda_f32_buffer().expect("CUDA parameter"))
        .collect::<Vec<_>>();
    let g = indices
        .iter()
        .map(|&i| gradient_buffer(&params[i]))
        .collect::<Vec<_>>();
    let m = indices
        .iter()
        .map(|&i| state_buffer(first[i].as_ref().unwrap(), config.state_dtype))
        .collect::<Vec<_>>();
    let v = second.map(|s| {
        indices
            .iter()
            .map(|&i| state_buffer(s[i].as_ref().unwrap(), config.state_dtype))
            .collect::<Vec<_>>()
    });
    let total = p.iter().fold(0usize, |sum, b| sum.saturating_add(b.len()));
    let batched = should_use_cuda_pointer_batch(indices.len(), total);
    if batched {
        cuda::optimizer_lowp_state_update_no_host(&p, &g, &m, v.as_deref(), config)
            .expect("CUDA lowp-state batch update");
    } else {
        for index in 0..indices.len() {
            cuda::optimizer_lowp_state_update_no_host(
                &p[index..index + 1],
                &g[index..index + 1],
                &m[index..index + 1],
                v.as_ref().map(|s| &s[index..index + 1]),
                config,
            )
            .expect("CUDA lowp-state update");
        }
    }
    for (slot, &index) in indices.iter().enumerate() {
        assert!(
            finalize_cuda_param_update(&params[index], p[slot].clone()),
            "publish CUDA parameter"
        );
        first[index]
            .as_ref()
            .unwrap()
            .replace_cuda_floatlike_state_buffer_no_host_sync(m[slot].clone());
        if let (Some(states), Some(buffers)) = (second, &v) {
            states[index]
                .as_ref()
                .unwrap()
                .replace_cuda_floatlike_state_buffer_no_host_sync(buffers[slot].clone());
        }
        updated[index] = true;
    }
    if batched { indices.len() } else { 0 }
}

pub(super) fn momentum_step(
    params: &[Tensor],
    states: &[Option<Tensor>],
    lr: f32,
    momentum: f32,
    dtype: DType,
) -> (Vec<bool>, usize) {
    let mut updated = vec![false; params.len()];
    let indices = params
        .iter()
        .enumerate()
        .filter_map(|(i, p)| (p.is_cuda() && p.has_pending_grad()).then_some(i))
        .collect::<Vec<_>>();
    let config = LowpOptimizerConfig {
        state_dtype: dtype,
        lr,
        beta1: momentum,
        beta2: 0.0,
        correction1: 1.0,
        correction2: 1.0,
        eps: 0.0,
    };
    let count = update_group(params, states, None, &indices, config, &mut updated);
    (updated, count)
}

pub(super) fn adam_step(
    params: &[Tensor],
    first: &[Option<Tensor>],
    second: &[Option<Tensor>],
    lr: f32,
    beta1: f32,
    beta2: f32,
    steps: &[usize],
    eps: f32,
    dtype: DType,
) -> (Vec<bool>, usize) {
    let mut updated = vec![false; params.len()];
    let mut groups = BTreeMap::<usize, Vec<usize>>::new();
    for (index, param) in params.iter().enumerate() {
        if param.is_cuda() && param.has_pending_grad() {
            groups.entry(steps[index]).or_default().push(index);
        }
    }
    let mut count = 0;
    for (step, indices) in groups {
        let (correction1, correction2) = adam_bias_corrections(beta1, beta2, step);
        let config = LowpOptimizerConfig {
            state_dtype: dtype,
            lr,
            beta1,
            beta2,
            correction1,
            correction2,
            eps,
        };
        count += update_group(params, first, Some(second), &indices, config, &mut updated);
    }
    (updated, count)
}
