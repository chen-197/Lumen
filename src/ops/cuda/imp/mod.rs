//! CUDA argument validation, buffer management, and FFI dispatch.

mod ffi;
use ffi::*;

mod matmul;
pub(super) use matmul::{
    batch_matmul_backward_bf16_i8_no_host, batch_matmul_backward_bf16_no_host,
    batch_matmul_backward_f16_i8_no_host, batch_matmul_backward_f16_no_host,
    batch_matmul_backward_f32_i8_no_host, batch_matmul_backward_f32_no_host,
    batch_matmul_backward_i8_bf16_no_host, batch_matmul_backward_i8_f16_no_host,
    batch_matmul_backward_i8_f32_no_host, batch_matmul_backward_i8_no_host,
    batch_matmul_bf16_buffer_no_host, batch_matmul_bf16_i8_buffer_no_host,
    batch_matmul_bf16_typed_output_buffer_no_host, batch_matmul_f16_buffer_no_host,
    batch_matmul_f16_i8_buffer_no_host, batch_matmul_f16_typed_output_buffer_no_host,
    batch_matmul_f32, batch_matmul_f32_i8_buffer_no_host, batch_matmul_f32_no_host,
    batch_matmul_i8_bf16_buffer_no_host, batch_matmul_i8_buffer_no_host,
    batch_matmul_i8_f16_buffer_no_host, batch_matmul_i8_f32_buffer_no_host,
    batch_matmul_i8_typed_output_buffer_no_host, matmul_backward_bf16_i8_no_host,
    matmul_backward_f16_i8_no_host, matmul_backward_f32_i8_no_host, matmul_backward_f32_no_host,
    matmul_backward_i8_bf16_no_host, matmul_backward_i8_f16_no_host,
    matmul_backward_i8_f32_no_host, matmul_bf16_buffer_no_host, matmul_bf16_host_no_host,
    matmul_bf16_i8_buffer_no_host, matmul_bf16_i8_dynamic_buffer_no_host,
    matmul_bf16_typed_output_buffer_no_host, matmul_f16_buffer_no_host, matmul_f16_host_no_host,
    matmul_f16_i8_buffer_no_host, matmul_f16_i8_dynamic_buffer_no_host,
    matmul_f16_typed_output_buffer_no_host, matmul_f32, matmul_f32_i8_buffer_no_host,
    matmul_f32_no_host, matmul_i8_bf16_buffer_no_host, matmul_i8_buffer_no_host,
    matmul_i8_f16_buffer_no_host, matmul_i8_f32_buffer_no_host, matmul_i8_host_no_host,
    matmul_i8_typed_output_buffer_no_host,
};
mod memory;
pub(super) use memory::{
    add_inplace_f32, alloc_f32, alloc_storage, copy_f32_offset, download_f32, download_f32_offset,
    download_i8_storage, download_u16_storage, f32_to_i8_storage_no_host,
    f32_to_lowp_storage_no_host, fill_scalar_f32, fill_scalar_f32_buffer, free_f32, is_available,
    quantize_f32_storage_no_host, quantize_f32_to_i8_dynamic_no_host, release_cached_memory,
    synchronize, upload_f32, upload_f32_offset, upload_i8_storage, upload_u16_storage,
};
mod selection;
pub(super) use selection::{
    matvec_argmax_bf16_i8, matvec_argmax_f16_i8, matvec_argmax_f32, matvec_argmax_f32_i8,
    matvec_argmax_i8_i8,
};
mod attention;
pub(super) use attention::{
    append_kv_cache_f32, append_kv_cache_pair_f32, decode_attention_f32,
    decode_attention_f32_buffer, decode_rope_q_append_kv_f32_buffer, kv_cache_prefix_f32_buffer,
    kv_cache_prefix_typed_buffer, prefill_attention_f32_buffer, repeat_kv_backward_f32,
    repeat_kv_backward_f32_buffer, repeat_kv_f32, repeat_kv_f32_buffer, repeat_kv_typed_buffer,
    rope_backward_f32, rope_backward_f32_buffer, rope_f32, rope_f32_buffer, rope_typed_buffer,
    rope_typed_i8_dynamic_buffer, round_kv_cache_rows_f32,
};
mod elementwise;
pub(super) use elementwise::{
    add_sub_backward_b1d_1h1_f32_buffers, add_sub_backward_b1d_1hd_f32_buffers,
    add_sub_backward_f32_buffers, add_sub_backward_lastdim_f32_buffers,
    add_sub_backward_row_scalar_f32_buffers, add_sub_backward_scalar_f32_buffers,
    add_sub_broadcast_backward_f32_buffers, binary_backward_f32, binary_backward_f32_buffers,
    binary_bf16_buffer_no_host, binary_bf16_host_no_host, binary_bf16_lastdim_buffer_no_host,
    binary_broadcast_backward_f32, binary_broadcast_backward_f32_buffers, binary_broadcast_f32,
    binary_broadcast_f32_buffer, binary_f16_buffer_no_host, binary_f16_host_no_host,
    binary_f16_lastdim_buffer_no_host, binary_f32, binary_f32_buffer, binary_i8_buffer_no_host,
    binary_i8_host_no_host, binary_i8_lastdim_buffer_no_host,
    binary_i8_typed_b1d_1h1_output_buffer_no_host, binary_i8_typed_b1d_1hd_output_buffer_no_host,
    binary_i8_typed_broadcast_output_buffer_no_host, binary_i8_typed_lastdim_output_buffer_no_host,
    binary_i8_typed_output_buffer_no_host, binary_i8_typed_row_scalar_output_buffer_no_host,
    binary_lowp_typed_b1d_1h1_output_buffer_no_host,
    binary_lowp_typed_b1d_1hd_output_buffer_no_host,
    binary_lowp_typed_broadcast_output_buffer_no_host,
    binary_lowp_typed_lastdim_output_buffer_no_host, binary_lowp_typed_output_buffer_no_host,
    binary_lowp_typed_row_scalar_output_buffer_no_host, binary_typed_b1d_1h1_buffer_no_host,
    binary_typed_b1d_1hd_buffer_no_host, binary_typed_broadcast_buffer_no_host,
    binary_typed_buffer_no_host, binary_typed_lastdim_buffer_no_host,
    binary_typed_row_scalar_buffer_no_host, mul_grad_bf16_buffer_no_host,
    mul_grad_bf16_host_no_host, mul_grad_bf16_lastdim_buffer_no_host, mul_grad_f16_buffer_no_host,
    mul_grad_f16_host_no_host, mul_grad_f16_lastdim_buffer_no_host, mul_grad_i8_buffer_no_host,
    mul_grad_i8_host_no_host, mul_grad_i8_lastdim_buffer_no_host,
    mul_grad_typed_b1d_1h1_buffer_no_host, mul_grad_typed_b1d_1hd_buffer_no_host,
    mul_grad_typed_broadcast_buffer_no_host, mul_grad_typed_buffer_no_host,
    mul_grad_typed_lastdim_buffer_no_host, mul_grad_typed_row_scalar_buffer_no_host,
    mul_grad_typed_scalar_buffer_no_host, unary_backward_bf16_buffer, unary_backward_f16_buffer,
    unary_backward_f32, unary_backward_f32_buffer, unary_backward_i8_buffer, unary_bf16_buffer,
    unary_bf16_typed_output_buffer, unary_f16_buffer, unary_f16_typed_output_buffer, unary_f32,
    unary_f32_buffer, unary_i8_buffer, unary_i8_relu_typed_output_buffer,
};
mod reduction_loss;
pub(super) use reduction_loss::{
    cross_entropy_backward_f32, cross_entropy_backward_f32_buffer,
    cross_entropy_backward_typed_target, cross_entropy_backward_typed_target_buffer,
    cross_entropy_logits_backward_typed, cross_entropy_logits_typed, cross_entropy_loss_f32,
    cross_entropy_loss_typed_target, mse_backward_f32, mse_backward_f32_buffers, mse_forward_typed,
    sum_bf16_buffer, sum_f16_buffer, sum_f32, sum_i8_buffer, sum_lastdim_f32_buffer,
};
mod shape;
pub(super) use shape::{
    bshd_to_bhsd_add_bias_f32_buffer, cat_backward_slice_f32, cat_backward_slice_f32_buffer,
    cat_f32, cat_f32_buffer, cat_typed_buffer, embedding_backward_f32,
    embedding_backward_f32_buffer, embedding_f32, embedding_f32_buffer, embedding_typed,
    embedding_typed_buffer, embedding_typed_same_dtype_buffer, permute_f32, permute_f32_buffer,
    permute_typed_buffer, slice_lastdim_backward_f32, slice_lastdim_backward_f32_buffer,
    slice_lastdim_f32, slice_lastdim_f32_buffer, slice_lastdim_typed_buffer,
};
mod optimizer;
pub(super) use optimizer::{
    adam_update_f32, adam_update_f32_batched_no_host, adam_update_f32_no_host,
    optimizer_lowp_state_update_no_host, sgd_momentum_update_f32,
    sgd_momentum_update_f32_batched_no_host, sgd_momentum_update_f32_no_host, sgd_update_f32,
    sgd_update_f32_batched_no_host, sgd_update_f32_no_host,
};
mod normalization;
pub(super) use normalization::{
    fused_softmax_backward_f32, fused_softmax_backward_f32_buffer, fused_softmax_f32,
    fused_softmax_f32_no_host, fused_softmax_f32_with_past, fused_softmax_f32_with_past_no_host,
    rms_norm_backward_f32, rms_norm_backward_f32_buffers, rms_norm_backward_typed,
    rms_norm_backward_typed_buffers, rms_norm_f32, rms_norm_f32_buffer,
    rms_norm_i8_typed_output_buffer, rms_norm_typed, rms_norm_typed_buffer,
    softmax_lastdim_backward_f32, softmax_lastdim_backward_f32_buffer, softmax_lastdim_f32,
    softmax_lastdim_f32_no_host, softmax_lastdim_typed, softmax_lastdim_typed_no_host,
};
mod projection;
pub(super) use projection::{
    fused_gate_up_silu_f32, fused_gate_up_silu_f32_buffer, fused_gate_up_silu_typed,
    fused_gate_up_silu_typed_buffer, fused_gate_up_silu_typed_output_buffer, fused_qkv_f32,
    fused_qkv_f32_buffer, fused_qkv_typed_buffer, fused_qkv_typed_output_buffer,
    silu_mul_f32_buffer_no_host,
};
mod convolution;
pub(super) use convolution::{
    conv2d_backward_f32, conv2d_backward_f32_buffers, conv2d_f32, conv2d_f32_buffer, conv2d_typed,
    conv2d_typed_buffer, max_pool2d_backward_f32, max_pool2d_backward_f32_buffer,
    max_pool2d_backward_typed_buffer, max_pool2d_f32, max_pool2d_typed,
};

use super::{
    BinaryOp, BroadcastMetadata, CudaAdamHostState, CudaBuffer, CudaConv2dBackwardHostBuffers,
    CudaThreeHostBuffers, CudaTwoHostBuffers, LowpOptimizerConfig, UnaryOp,
};
use crate::precision::DType;
use std::ffi::CStr;
use std::os::raw::{c_char, c_int};

fn last_error_message() -> String {
    unsafe {
        let ptr = lumen_cuda_last_error_message();
        if ptr.is_null() {
            return "unknown CUDA error".to_string();
        }
        CStr::from_ptr(ptr).to_string_lossy().into_owned()
    }
}

fn row_major_strides(shape: &[usize], context: &str) -> Result<Vec<usize>, String> {
    let mut strides = vec![0usize; shape.len()];
    let mut stride = 1usize;
    for i in (0..shape.len()).rev() {
        strides[i] = stride;
        stride = stride
            .checked_mul(shape[i])
            .ok_or_else(|| format!("CUDA {context} stride overflow"))?;
    }
    Ok(strides)
}

fn aligned_broadcast_metadata(
    lhs_shape: &[usize],
    rhs_shape: &[usize],
    out_shape: &[usize],
) -> Result<BroadcastMetadata, String> {
    let ndim = out_shape.len();
    if ndim == 0 {
        return Err("CUDA broadcast expects at least 1 dimension".to_string());
    }
    if lhs_shape.len() > ndim || rhs_shape.len() > ndim {
        return Err(format!(
            "CUDA broadcast rank mismatch: lhs={:?}, rhs={:?}, out={:?}",
            lhs_shape, rhs_shape, out_shape
        ));
    }

    let mut lhs_aligned = vec![1usize; ndim];
    let mut rhs_aligned = vec![1usize; ndim];
    let lhs_offset = ndim - lhs_shape.len();
    let rhs_offset = ndim - rhs_shape.len();
    lhs_aligned[lhs_offset..].copy_from_slice(lhs_shape);
    rhs_aligned[rhs_offset..].copy_from_slice(rhs_shape);

    for i in 0..ndim {
        let lhs_dim = lhs_aligned[i];
        let rhs_dim = rhs_aligned[i];
        let out_dim = out_shape[i];
        let expected = lhs_dim.max(rhs_dim);
        if (lhs_dim != out_dim && lhs_dim != 1)
            || (rhs_dim != out_dim && rhs_dim != 1)
            || expected != out_dim
        {
            return Err(format!(
                "CUDA broadcast shape mismatch: lhs={:?}, rhs={:?}, out={:?}",
                lhs_shape, rhs_shape, out_shape
            ));
        }
    }

    let out_strides = row_major_strides(out_shape, "broadcast output")?;
    let lhs_raw_strides = row_major_strides(&lhs_aligned, "broadcast lhs")?;
    let rhs_raw_strides = row_major_strides(&rhs_aligned, "broadcast rhs")?;
    let lhs_strides = lhs_aligned
        .iter()
        .zip(lhs_raw_strides.iter())
        .map(|(&dim, &stride)| if dim == 1 { 0 } else { stride })
        .collect::<Vec<_>>();
    let rhs_strides = rhs_aligned
        .iter()
        .zip(rhs_raw_strides.iter())
        .map(|(&dim, &stride)| if dim == 1 { 0 } else { stride })
        .collect::<Vec<_>>();
    Ok((
        lhs_aligned,
        rhs_aligned,
        out_strides,
        lhs_strides,
        rhs_strides,
    ))
}

fn dtype_tag(dtype: DType) -> Result<c_int, String> {
    match dtype {
        DType::F32 => Ok(0),
        DType::F16 => Ok(1),
        DType::BF16 => Ok(2),
        DType::I8 => Ok(3),
    }
}

pub(super) fn checked_len(label: &str, factors: &[usize]) -> Result<usize, String> {
    factors.iter().try_fold(1usize, |product, &factor| {
        product
            .checked_mul(factor)
            .ok_or_else(|| format!("{label} overflow"))
    })
}

pub(super) fn range_fits(start: usize, len: usize, total: usize) -> bool {
    start <= total && len <= total - start
}

pub(super) fn conv_output_dim(
    input: usize,
    padding: usize,
    kernel: usize,
    stride: usize,
    label: &str,
) -> Result<usize, String> {
    if stride == 0 {
        return Err(format!("{label} stride must be greater than zero"));
    }
    let padded = padding
        .checked_mul(2)
        .and_then(|value| input.checked_add(value))
        .ok_or_else(|| format!("{label} padded input size overflow"))?;
    if padded < kernel {
        return Err(format!("{label} kernel is larger than the padded input"));
    }
    (padded - kernel)
        .checked_div(stride)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| format!("{label} output size overflow"))
}

pub(super) fn ensure_finite(label: &str, value: f32) -> Result<(), String> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(format!("{label} must be finite"))
    }
}

pub(super) fn ensure_positive_finite(label: &str, value: f32) -> Result<(), String> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(format!("{label} must be finite and > 0"))
    }
}
