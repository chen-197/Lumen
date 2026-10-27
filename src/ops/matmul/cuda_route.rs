use super::*;

pub(super) fn try_cuda_matmul_buffer(
    a: &Tensor,
    b: &Tensor,
    m_dim: usize,
    k_dim: usize,
    n_dim: usize,
) -> Option<cuda::CudaBuffer> {
    if a.device() != crate::autograd::Device::Cuda || b.device() != crate::autograd::Device::Cuda {
        return None;
    }
    if let Some(buffer) = try_cuda_matmul_native_low_precision_buffer(a, b, m_dim, k_dim, n_dim) {
        return Some(buffer);
    }
    let force_cuda = is_strict_device_execution();
    if !force_cuda && !cuda::should_accelerate_matmul(m_dim, n_dim, k_dim) {
        return None;
    }
    let cuda_out = a.with_cuda_f32_buffer(|a_buf| {
        b.with_cuda_f32_buffer(|b_buf| cuda::matmul_f32_no_host(a_buf, b_buf, m_dim, n_dim, k_dim))
    });
    Some(match cuda_out {
        Ok(out) => out,
        Err(err) => {
            if force_cuda {
                panic!("CUDA matmul failed in strict device execution mode: {err}");
            }
            return None;
        }
    })
}

pub(super) fn try_cuda_matmul_native_low_precision_buffer(
    a: &Tensor,
    b: &Tensor,
    m_dim: usize,
    k_dim: usize,
    n_dim: usize,
) -> Option<cuda::CudaBuffer> {
    let a_native = a.cloned_cuda_native_lowp_buffer();
    let b_native = b.cloned_cuda_native_lowp_buffer();

    if let (Some((DType::BF16, a_buffer, _)), Some((DType::I8, b_buffer, Some(b_scale)))) =
        (&a_native, &b_native)
    {
        return if is_no_grad() && crate::precision::allow_dynamic_i8_activations() {
            cuda::matmul_bf16_i8_dynamic_buffer_no_host(
                a_buffer, b_buffer, *b_scale, m_dim, n_dim, k_dim,
            )
        } else {
            cuda::matmul_bf16_i8_buffer_no_host(a_buffer, b_buffer, *b_scale, m_dim, n_dim, k_dim)
        }
        .ok();
    }

    if let (Some((DType::F16, a_buffer, _)), Some((DType::I8, b_buffer, Some(b_scale)))) =
        (&a_native, &b_native)
    {
        return if is_no_grad() && crate::precision::allow_dynamic_i8_activations() {
            cuda::matmul_f16_i8_dynamic_buffer_no_host(
                a_buffer, b_buffer, *b_scale, m_dim, n_dim, k_dim,
            )
        } else {
            cuda::matmul_f16_i8_buffer_no_host(a_buffer, b_buffer, *b_scale, m_dim, n_dim, k_dim)
        }
        .ok();
    }

    if a.dtype() == DType::F32
        && let (Some(a_buffer), Some((DType::I8, b_buffer, Some(b_scale)))) =
            (a.cloned_cuda_f32_buffer(), &b_native)
    {
        return cuda::matmul_f32_i8_buffer_no_host(
            &a_buffer, b_buffer, *b_scale, m_dim, n_dim, k_dim,
        )
        .ok();
    }

    if let (Some((DType::I8, a_buffer, Some(a_scale))), Some((DType::BF16, b_buffer, _))) =
        (&a_native, &b_native)
    {
        return cuda::matmul_i8_bf16_buffer_no_host(
            a_buffer, *a_scale, b_buffer, m_dim, n_dim, k_dim,
        )
        .ok();
    }

    if let (Some((DType::I8, a_buffer, Some(a_scale))), Some((DType::F16, b_buffer, _))) =
        (&a_native, &b_native)
    {
        return cuda::matmul_i8_f16_buffer_no_host(
            a_buffer, *a_scale, b_buffer, m_dim, n_dim, k_dim,
        )
        .ok();
    }

    if b.dtype() == DType::F32
        && let (Some((DType::I8, a_buffer, Some(a_scale))), Some(b_buffer)) =
            (&a_native, b.cloned_cuda_f32_buffer())
    {
        return cuda::matmul_i8_f32_buffer_no_host(
            a_buffer, *a_scale, &b_buffer, m_dim, n_dim, k_dim,
        )
        .ok();
    }

    if let (Some((a_dtype, a_buffer, a_scale)), Some((b_dtype, b_buffer, b_scale))) =
        (&a_native, &b_native)
        && a_dtype == b_dtype
    {
        return match a_dtype {
            DType::BF16 => {
                cuda::matmul_bf16_buffer_no_host(a_buffer, b_buffer, m_dim, n_dim, k_dim).ok()
            }
            DType::F16 => {
                cuda::matmul_f16_buffer_no_host(a_buffer, b_buffer, m_dim, n_dim, k_dim).ok()
            }
            DType::I8 => cuda::matmul_i8_buffer_no_host(
                a_buffer,
                (*a_scale)?,
                b_buffer,
                (*b_scale)?,
                m_dim,
                n_dim,
                k_dim,
            )
            .ok(),
            DType::F32 => None,
        };
    }

    // Only equal low-precision types have a host-storage fallback below. Mixed
    // and F32 pairs use later CUDA paths; probing them must not download inputs.
    if a.dtype() != b.dtype() || a.dtype() == DType::F32 {
        return None;
    }
    match (a.native_storage_owned(), b.native_storage_owned()) {
        (TensorStorageOwned::BF16(a_data), TensorStorageOwned::BF16(b_data)) => {
            let a_bits = a_data.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
            let b_bits = b_data.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
            cuda::matmul_bf16_host_no_host(&a_bits, &b_bits, m_dim, n_dim, k_dim).ok()
        }
        (TensorStorageOwned::F16(a_data), TensorStorageOwned::F16(b_data)) => {
            let a_bits = a_data.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
            let b_bits = b_data.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
            cuda::matmul_f16_host_no_host(&a_bits, &b_bits, m_dim, n_dim, k_dim).ok()
        }
        (TensorStorageOwned::I8(a_data, a_scale), TensorStorageOwned::I8(b_data, b_scale)) => {
            let a_values = a_data.iter().copied().collect::<Vec<_>>();
            let b_values = b_data.iter().copied().collect::<Vec<_>>();
            cuda::matmul_i8_host_no_host(
                &a_values, a_scale, &b_values, b_scale, m_dim, n_dim, k_dim,
            )
            .ok()
        }
        _ => None,
    }
}

pub(super) fn try_cuda_matmul(
    a: &Tensor,
    b: &Tensor,
    m_dim: usize,
    n_dim: usize,
    k_dim: usize,
    output_dtype: DType,
) -> Option<Tensor> {
    let mut out_shape = a.shape_vec();
    let last_idx = out_shape.len() - 1;
    out_shape[last_idx] = n_dim;

    if output_dtype == DType::I8
        && let Some((buffer, scale)) =
            try_cuda_matmul_i8_typed_output_buffer(a, b, m_dim, k_dim, n_dim)
    {
        return Some(Tensor::from_cuda_native_lowp_buffer_no_host_with_dtype(
            &out_shape,
            buffer,
            a.device(),
            output_dtype,
            Some(scale),
        ));
    }

    if matches!(output_dtype, DType::F16 | DType::BF16)
        && let Some(buffer) = try_cuda_matmul_native_low_precision_typed_output_buffer(
            a,
            b,
            m_dim,
            k_dim,
            n_dim,
            output_dtype,
        )
    {
        return Some(Tensor::from_cuda_native_lowp_buffer_no_host_with_dtype(
            &out_shape,
            buffer,
            a.device(),
            output_dtype,
            None,
        ));
    }

    let buffer = try_cuda_matmul_buffer(a, b, m_dim, k_dim, n_dim)?;

    if output_dtype == DType::F32 {
        return Some(Tensor::from_cuda_f32_buffer_no_host(
            &out_shape,
            buffer,
            a.device(),
        ));
    }

    if matches!(output_dtype, DType::F16 | DType::BF16) {
        return Some(Tensor::from_cuda_f32_buffer_no_host_with_dtype(
            &out_shape,
            buffer,
            a.device(),
            output_dtype,
        ));
    }

    if output_dtype == DType::I8 {
        let (i8_buffer, scale) = cuda::quantize_f32_to_i8_dynamic_no_host(&buffer).ok()?;
        return Some(Tensor::from_cuda_native_lowp_buffer_no_host_with_dtype(
            &out_shape,
            i8_buffer,
            a.device(),
            output_dtype,
            Some(scale),
        ));
    }

    None
}

pub(super) fn try_cuda_matmul_i8_typed_output_buffer(
    a: &Tensor,
    b: &Tensor,
    m_dim: usize,
    k_dim: usize,
    n_dim: usize,
) -> Option<(cuda::CudaBuffer, f32)> {
    if a.device() != crate::autograd::Device::Cuda || b.device() != crate::autograd::Device::Cuda {
        return None;
    }
    let (Some((DType::I8, a_buffer, Some(a_scale))), Some((DType::I8, b_buffer, Some(b_scale)))) = (
        a.cloned_cuda_native_lowp_buffer(),
        b.cloned_cuda_native_lowp_buffer(),
    ) else {
        return None;
    };
    cuda::matmul_i8_typed_output_buffer_no_host(
        &a_buffer, a_scale, &b_buffer, b_scale, m_dim, n_dim, k_dim,
    )
    .ok()
}

pub(super) fn try_cuda_matmul_native_low_precision_typed_output_buffer(
    a: &Tensor,
    b: &Tensor,
    m_dim: usize,
    k_dim: usize,
    n_dim: usize,
    output_dtype: DType,
) -> Option<cuda::CudaBuffer> {
    if a.device() != crate::autograd::Device::Cuda || b.device() != crate::autograd::Device::Cuda {
        return None;
    }
    let (Some((a_dtype, a_buffer, _)), Some((b_dtype, b_buffer, _))) = (
        a.cloned_cuda_native_lowp_buffer(),
        b.cloned_cuda_native_lowp_buffer(),
    ) else {
        return None;
    };
    if a_dtype != b_dtype || a_dtype != output_dtype {
        return None;
    }

    match a_dtype {
        DType::BF16 => {
            cuda::matmul_bf16_typed_output_buffer_no_host(&a_buffer, &b_buffer, m_dim, n_dim, k_dim)
                .ok()
        }
        DType::F16 => {
            cuda::matmul_f16_typed_output_buffer_no_host(&a_buffer, &b_buffer, m_dim, n_dim, k_dim)
                .ok()
        }
        DType::F32 | DType::I8 => None,
    }
}

pub(super) fn try_cuda_batch_matmul_buffer(
    lhs: &Tensor,
    rhs: &Tensor,
    b: usize,
    h: usize,
    m: usize,
    k: usize,
    n: usize,
) -> Option<cuda::CudaBuffer> {
    if lhs.device() != crate::autograd::Device::Cuda
        || rhs.device() != crate::autograd::Device::Cuda
    {
        return None;
    }
    let batch_count = b.checked_mul(h)?;
    if let Some(buffer) =
        try_cuda_batch_matmul_native_low_precision_buffer(lhs, rhs, batch_count, m, k, n)
    {
        return Some(buffer);
    }
    let force_cuda = is_strict_device_execution();
    if !force_cuda && !cuda::should_accelerate_batch_matmul(batch_count, m, n, k) {
        return None;
    }
    let cuda_out = lhs.with_cuda_f32_buffer(|lhs_buf| {
        rhs.with_cuda_f32_buffer(|rhs_buf| {
            cuda::batch_matmul_f32_no_host(lhs_buf, rhs_buf, batch_count, m, n, k)
        })
    });
    Some(match cuda_out {
        Ok(out) => out,
        Err(err) => {
            if force_cuda {
                panic!("CUDA batch_matmul failed in strict device execution mode: {err}");
            }
            return None;
        }
    })
}

pub(super) fn try_cuda_batch_matmul_native_low_precision_buffer(
    lhs: &Tensor,
    rhs: &Tensor,
    batch_count: usize,
    m: usize,
    k: usize,
    n: usize,
) -> Option<cuda::CudaBuffer> {
    let rhs_native = rhs.cloned_cuda_native_lowp_buffer();
    if let (Some((DType::BF16, lhs_buffer, _)), Some((DType::I8, rhs_buffer, Some(rhs_scale)))) =
        (lhs.cloned_cuda_native_lowp_buffer(), &rhs_native)
    {
        return cuda::batch_matmul_bf16_i8_buffer_no_host(
            &lhs_buffer,
            rhs_buffer,
            *rhs_scale,
            batch_count,
            m,
            n,
            k,
        )
        .ok();
    }

    if let (Some((DType::F16, lhs_buffer, _)), Some((DType::I8, rhs_buffer, Some(rhs_scale)))) =
        (lhs.cloned_cuda_native_lowp_buffer(), &rhs_native)
    {
        return cuda::batch_matmul_f16_i8_buffer_no_host(
            &lhs_buffer,
            rhs_buffer,
            *rhs_scale,
            batch_count,
            m,
            n,
            k,
        )
        .ok();
    }

    if lhs.dtype() == DType::F32
        && let (Some(lhs_buffer), Some((DType::I8, rhs_buffer, Some(rhs_scale)))) =
            (lhs.cloned_cuda_f32_buffer(), &rhs_native)
    {
        return cuda::batch_matmul_f32_i8_buffer_no_host(
            &lhs_buffer,
            rhs_buffer,
            *rhs_scale,
            batch_count,
            m,
            n,
            k,
        )
        .ok();
    }

    let lhs_native = lhs.cloned_cuda_native_lowp_buffer();
    if let (Some((DType::I8, lhs_buffer, Some(lhs_scale))), Some((DType::BF16, rhs_buffer, _))) =
        (&lhs_native, &rhs_native)
    {
        return cuda::batch_matmul_i8_bf16_buffer_no_host(
            lhs_buffer,
            *lhs_scale,
            rhs_buffer,
            batch_count,
            m,
            n,
            k,
        )
        .ok();
    }

    if let (Some((DType::I8, lhs_buffer, Some(lhs_scale))), Some((DType::F16, rhs_buffer, _))) =
        (&lhs_native, &rhs_native)
    {
        return cuda::batch_matmul_i8_f16_buffer_no_host(
            lhs_buffer,
            *lhs_scale,
            rhs_buffer,
            batch_count,
            m,
            n,
            k,
        )
        .ok();
    }

    if rhs.dtype() == DType::F32
        && let (Some((DType::I8, lhs_buffer, Some(lhs_scale))), Some(rhs_buffer)) =
            (&lhs_native, rhs.cloned_cuda_f32_buffer())
    {
        return cuda::batch_matmul_i8_f32_buffer_no_host(
            lhs_buffer,
            *lhs_scale,
            &rhs_buffer,
            batch_count,
            m,
            n,
            k,
        )
        .ok();
    }

    let (Some((lhs_dtype, lhs_buffer, lhs_scale)), Some((rhs_dtype, rhs_buffer, rhs_scale))) =
        (lhs_native, rhs_native)
    else {
        return None;
    };
    if lhs_dtype != rhs_dtype {
        return None;
    }

    match lhs_dtype {
        DType::BF16 => {
            cuda::batch_matmul_bf16_buffer_no_host(&lhs_buffer, &rhs_buffer, batch_count, m, n, k)
                .ok()
        }
        DType::F16 => {
            cuda::batch_matmul_f16_buffer_no_host(&lhs_buffer, &rhs_buffer, batch_count, m, n, k)
                .ok()
        }
        DType::I8 => cuda::batch_matmul_i8_buffer_no_host(
            &lhs_buffer,
            lhs_scale?,
            &rhs_buffer,
            rhs_scale?,
            batch_count,
            m,
            n,
            k,
        )
        .ok(),
        DType::F32 => None,
    }
}

pub(super) fn try_cuda_batch_matmul(
    lhs: &Tensor,
    rhs: &Tensor,
    dims: BatchMatmulDims,
    output_dtype: DType,
) -> Option<Tensor> {
    let out_shape = dims.out_shape();
    if output_dtype == DType::I8
        && let Some((buffer, scale)) = try_cuda_batch_matmul_i8_typed_output_buffer(lhs, rhs, dims)
    {
        return Some(Tensor::from_cuda_native_lowp_buffer_no_host_with_dtype(
            &out_shape,
            buffer,
            lhs.device(),
            output_dtype,
            Some(scale),
        ));
    }

    if matches!(output_dtype, DType::F16 | DType::BF16)
        && let Some(buffer) = try_cuda_batch_matmul_native_low_precision_typed_output_buffer(
            lhs,
            rhs,
            dims,
            output_dtype,
        )
    {
        return Some(Tensor::from_cuda_native_lowp_buffer_no_host_with_dtype(
            &out_shape,
            buffer,
            lhs.device(),
            output_dtype,
            None,
        ));
    }

    let buffer = try_cuda_batch_matmul_buffer(lhs, rhs, dims.b, dims.h, dims.m, dims.k, dims.n)?;
    if output_dtype == DType::F32 {
        return Some(Tensor::from_cuda_f32_buffer_no_host(
            &out_shape,
            buffer,
            lhs.device(),
        ));
    }

    if matches!(output_dtype, DType::F16 | DType::BF16) {
        return Some(Tensor::from_cuda_f32_buffer_no_host_with_dtype(
            &out_shape,
            buffer,
            lhs.device(),
            output_dtype,
        ));
    }

    if output_dtype == DType::I8 {
        let (i8_buffer, scale) = cuda::quantize_f32_to_i8_dynamic_no_host(&buffer).ok()?;
        return Some(Tensor::from_cuda_native_lowp_buffer_no_host_with_dtype(
            &out_shape,
            i8_buffer,
            lhs.device(),
            output_dtype,
            Some(scale),
        ));
    }

    None
}

pub(super) fn try_cuda_batch_matmul_i8_typed_output_buffer(
    lhs: &Tensor,
    rhs: &Tensor,
    dims: BatchMatmulDims,
) -> Option<(cuda::CudaBuffer, f32)> {
    if lhs.device() != crate::autograd::Device::Cuda
        || rhs.device() != crate::autograd::Device::Cuda
    {
        return None;
    }
    let (
        Some((DType::I8, lhs_buffer, Some(lhs_scale))),
        Some((DType::I8, rhs_buffer, Some(rhs_scale))),
    ) = (
        lhs.cloned_cuda_native_lowp_buffer(),
        rhs.cloned_cuda_native_lowp_buffer(),
    )
    else {
        return None;
    };
    let batch_count = dims.batch_count()?;
    cuda::batch_matmul_i8_typed_output_buffer_no_host(
        &lhs_buffer,
        lhs_scale,
        &rhs_buffer,
        rhs_scale,
        batch_count,
        dims.m,
        dims.n,
        dims.k,
    )
    .ok()
}

pub(super) fn try_cuda_batch_matmul_native_low_precision_typed_output_buffer(
    lhs: &Tensor,
    rhs: &Tensor,
    dims: BatchMatmulDims,
    output_dtype: DType,
) -> Option<cuda::CudaBuffer> {
    if lhs.device() != crate::autograd::Device::Cuda
        || rhs.device() != crate::autograd::Device::Cuda
    {
        return None;
    }
    let batch_count = dims.batch_count()?;
    let (Some((lhs_dtype, lhs_buffer, _)), Some((rhs_dtype, rhs_buffer, _))) = (
        lhs.cloned_cuda_native_lowp_buffer(),
        rhs.cloned_cuda_native_lowp_buffer(),
    ) else {
        return None;
    };
    if lhs_dtype != rhs_dtype || lhs_dtype != output_dtype {
        return None;
    }

    match lhs_dtype {
        DType::BF16 => cuda::batch_matmul_bf16_typed_output_buffer_no_host(
            &lhs_buffer,
            &rhs_buffer,
            batch_count,
            dims.m,
            dims.n,
            dims.k,
        )
        .ok(),
        DType::F16 => cuda::batch_matmul_f16_typed_output_buffer_no_host(
            &lhs_buffer,
            &rhs_buffer,
            batch_count,
            dims.m,
            dims.n,
            dims.k,
        )
        .ok(),
        DType::F32 | DType::I8 => None,
    }
}

pub(super) fn try_cuda_training_matmul_backward(
    grad: &ndarray::ArrayViewD<'_, f32>,
    cuda_grad: Option<cuda::CudaBuffer>,
    a: &Tensor,
    b: &Tensor,
    m_dim: usize,
    k_dim: usize,
    n_dim: usize,
) -> Result<CudaMatmulBackwardHostDevice, String> {
    let (da_buf, db_buf) =
        try_cuda_training_matmul_backward_buffers(grad, cuda_grad, a, b, m_dim, k_dim, n_dim)?;
    let da_host = cuda::download_f32(&da_buf)?;
    let db_host = cuda::download_f32(&db_buf)?;

    let da = Array2::from_shape_vec((m_dim, k_dim), da_host)
        .expect("CUDA matmul backward dA shape build failed")
        .into_shape(a.shape_vec())
        .expect("CUDA matmul backward dA reshape failed")
        .into_dyn();
    let db = Array2::from_shape_vec((n_dim, k_dim), db_host)
        .expect("CUDA matmul backward dB shape build failed")
        .into_dyn();
    Ok(((da, da_buf), (db, db_buf)))
}

pub(super) fn try_cuda_training_matmul_backward_buffers(
    grad: &ndarray::ArrayViewD<'_, f32>,
    cuda_grad: Option<cuda::CudaBuffer>,
    a: &Tensor,
    b: &Tensor,
    m_dim: usize,
    k_dim: usize,
    n_dim: usize,
) -> Result<(cuda::CudaBuffer, cuda::CudaBuffer), String> {
    let grad_buf = match cuda_grad {
        Some(buffer) if buffer.len() == grad.len() => buffer,
        _ => cuda::upload_f32(&grad.iter().copied().collect::<Vec<_>>())?,
    };

    if let (Some((DType::BF16, a_buffer, _)), Some((DType::I8, b_buffer, b_scale))) = (
        a.cloned_cuda_native_lowp_buffer(),
        b.cloned_cuda_native_lowp_buffer(),
    ) {
        return cuda::matmul_backward_bf16_i8_no_host(
            &grad_buf,
            &a_buffer,
            &b_buffer,
            b_scale.ok_or_else(|| "CUDA I8 rhs scale missing".to_string())?,
            m_dim,
            k_dim,
            n_dim,
        );
    }

    if let (Some((DType::F16, a_buffer, _)), Some((DType::I8, b_buffer, b_scale))) = (
        a.cloned_cuda_native_lowp_buffer(),
        b.cloned_cuda_native_lowp_buffer(),
    ) {
        return cuda::matmul_backward_f16_i8_no_host(
            &grad_buf,
            &a_buffer,
            &b_buffer,
            b_scale.ok_or_else(|| "CUDA I8 rhs scale missing".to_string())?,
            m_dim,
            k_dim,
            n_dim,
        );
    }

    if let Some((DType::I8, b_buffer, b_scale)) = b.cloned_cuda_native_lowp_buffer()
        && a.dtype() == DType::F32
    {
        let a_buffer = a
            .cloned_cuda_f32_buffer()
            .ok_or_else(|| "CUDA F32xI8 matmul backward expected lhs buffer".to_string())?;
        return cuda::matmul_backward_f32_i8_no_host(
            &grad_buf,
            &a_buffer,
            &b_buffer,
            b_scale.ok_or_else(|| "CUDA I8 rhs scale missing".to_string())?,
            m_dim,
            k_dim,
            n_dim,
        );
    }

    if let (Some((DType::I8, a_buffer, a_scale)), Some((DType::BF16, b_buffer, _))) = (
        a.cloned_cuda_native_lowp_buffer(),
        b.cloned_cuda_native_lowp_buffer(),
    ) {
        return cuda::matmul_backward_i8_bf16_no_host(
            &grad_buf,
            &a_buffer,
            a_scale.ok_or_else(|| "CUDA I8 lhs scale missing".to_string())?,
            &b_buffer,
            m_dim,
            k_dim,
            n_dim,
        );
    }

    if let (Some((DType::I8, a_buffer, a_scale)), Some((DType::F16, b_buffer, _))) = (
        a.cloned_cuda_native_lowp_buffer(),
        b.cloned_cuda_native_lowp_buffer(),
    ) {
        return cuda::matmul_backward_i8_f16_no_host(
            &grad_buf,
            &a_buffer,
            a_scale.ok_or_else(|| "CUDA I8 lhs scale missing".to_string())?,
            &b_buffer,
            m_dim,
            k_dim,
            n_dim,
        );
    }

    if let Some((DType::I8, a_buffer, a_scale)) = a.cloned_cuda_native_lowp_buffer()
        && b.dtype() == DType::F32
    {
        let b_buffer = b
            .cloned_cuda_f32_buffer()
            .ok_or_else(|| "CUDA I8xF32 matmul backward expected rhs buffer".to_string())?;
        return cuda::matmul_backward_i8_f32_no_host(
            &grad_buf,
            &a_buffer,
            a_scale.ok_or_else(|| "CUDA I8 lhs scale missing".to_string())?,
            &b_buffer,
            m_dim,
            k_dim,
            n_dim,
        );
    }

    let a_buf = a
        .cloned_cuda_f32_buffer()
        .ok_or_else(|| "CUDA matmul backward expected lhs resident buffer".to_string())?;
    let b_buf = b
        .cloned_cuda_f32_buffer()
        .ok_or_else(|| "CUDA matmul backward expected rhs resident buffer".to_string())?;

    cuda::matmul_backward_f32_no_host(&grad_buf, &a_buf, &b_buf, m_dim, k_dim, n_dim)
}

pub(super) fn try_cuda_training_batch_matmul_backward(
    grad: &ndarray::ArrayViewD<'_, f32>,
    cuda_grad: Option<cuda::CudaBuffer>,
    lhs: &Tensor,
    rhs: &Tensor,
    dims: BatchMatmulDims,
) -> Result<CudaMatmulBackwardHostDevice, String> {
    let (d_lhs_buf, d_rhs_buf) =
        try_cuda_training_batch_matmul_backward_buffers(grad, cuda_grad, lhs, rhs, dims)?;
    let d_lhs_host = cuda::download_f32(&d_lhs_buf)?;
    let d_rhs_host = cuda::download_f32(&d_rhs_buf)?;

    let d_lhs = Array4::from_shape_vec((dims.b, dims.h, dims.m, dims.k), d_lhs_host)
        .expect("CUDA batch_matmul backward dLHS shape build failed")
        .into_dyn();
    let d_rhs = Array4::from_shape_vec((dims.b, dims.h, dims.k, dims.n), d_rhs_host)
        .expect("CUDA batch_matmul backward dRHS shape build failed")
        .into_dyn();
    Ok(((d_lhs, d_lhs_buf), (d_rhs, d_rhs_buf)))
}

pub(super) fn try_cuda_training_batch_matmul_backward_buffers(
    grad: &ndarray::ArrayViewD<'_, f32>,
    cuda_grad: Option<cuda::CudaBuffer>,
    lhs: &Tensor,
    rhs: &Tensor,
    dims: BatchMatmulDims,
) -> Result<(cuda::CudaBuffer, cuda::CudaBuffer), String> {
    let batch_count = dims
        .batch_count()
        .ok_or_else(|| "CUDA batch_matmul backward batch count overflow".to_string())?;
    let grad_buf = match cuda_grad {
        Some(buffer) if buffer.len() == grad.len() => buffer,
        _ => cuda::upload_f32(&grad.iter().copied().collect::<Vec<_>>())?,
    };

    if let (Some((DType::BF16, lhs_buffer, _)), Some((DType::I8, rhs_buffer, rhs_scale))) = (
        lhs.cloned_cuda_native_lowp_buffer(),
        rhs.cloned_cuda_native_lowp_buffer(),
    ) {
        return cuda::batch_matmul_backward_bf16_i8_no_host(
            &grad_buf,
            &lhs_buffer,
            &rhs_buffer,
            rhs_scale.ok_or_else(|| "CUDA I8 rhs scale missing".to_string())?,
            batch_count,
            dims.m,
            dims.k,
            dims.n,
        );
    }

    if let (Some((DType::F16, lhs_buffer, _)), Some((DType::I8, rhs_buffer, rhs_scale))) = (
        lhs.cloned_cuda_native_lowp_buffer(),
        rhs.cloned_cuda_native_lowp_buffer(),
    ) {
        return cuda::batch_matmul_backward_f16_i8_no_host(
            &grad_buf,
            &lhs_buffer,
            &rhs_buffer,
            rhs_scale.ok_or_else(|| "CUDA I8 rhs scale missing".to_string())?,
            dims.batch_count()
                .ok_or_else(|| "CUDA batch_matmul backward batch size overflow".to_string())?,
            dims.m,
            dims.k,
            dims.n,
        );
    }

    if let Some((DType::I8, rhs_buffer, rhs_scale)) = rhs.cloned_cuda_native_lowp_buffer()
        && lhs.dtype() == DType::F32
    {
        let lhs_buffer = lhs
            .cloned_cuda_f32_buffer()
            .ok_or_else(|| "CUDA F32xI8 batch_matmul backward expected lhs buffer".to_string())?;
        return cuda::batch_matmul_backward_f32_i8_no_host(
            &grad_buf,
            &lhs_buffer,
            &rhs_buffer,
            rhs_scale.ok_or_else(|| "CUDA I8 rhs scale missing".to_string())?,
            batch_count,
            dims.m,
            dims.k,
            dims.n,
        );
    }

    if let (Some((DType::I8, lhs_buffer, lhs_scale)), Some((DType::BF16, rhs_buffer, _))) = (
        lhs.cloned_cuda_native_lowp_buffer(),
        rhs.cloned_cuda_native_lowp_buffer(),
    ) {
        return cuda::batch_matmul_backward_i8_bf16_no_host(
            &grad_buf,
            &lhs_buffer,
            lhs_scale.ok_or_else(|| "CUDA I8 lhs scale missing".to_string())?,
            &rhs_buffer,
            batch_count,
            dims.m,
            dims.k,
            dims.n,
        );
    }

    if let (Some((DType::I8, lhs_buffer, lhs_scale)), Some((DType::F16, rhs_buffer, _))) = (
        lhs.cloned_cuda_native_lowp_buffer(),
        rhs.cloned_cuda_native_lowp_buffer(),
    ) {
        return cuda::batch_matmul_backward_i8_f16_no_host(
            &grad_buf,
            &lhs_buffer,
            lhs_scale.ok_or_else(|| "CUDA I8 lhs scale missing".to_string())?,
            &rhs_buffer,
            batch_count,
            dims.m,
            dims.k,
            dims.n,
        );
    }

    if let Some((DType::I8, lhs_buffer, lhs_scale)) = lhs.cloned_cuda_native_lowp_buffer()
        && rhs.dtype() == DType::F32
    {
        let rhs_buffer = rhs
            .cloned_cuda_f32_buffer()
            .ok_or_else(|| "CUDA I8xF32 batch_matmul backward expected rhs buffer".to_string())?;
        return cuda::batch_matmul_backward_i8_f32_no_host(
            &grad_buf,
            &lhs_buffer,
            lhs_scale.ok_or_else(|| "CUDA I8 lhs scale missing".to_string())?,
            &rhs_buffer,
            batch_count,
            dims.m,
            dims.k,
            dims.n,
        );
    }

    if let (Some((lhs_dtype, lhs_buffer, lhs_scale)), Some((rhs_dtype, rhs_buffer, rhs_scale))) = (
        lhs.cloned_cuda_native_lowp_buffer(),
        rhs.cloned_cuda_native_lowp_buffer(),
    ) && lhs_dtype == rhs_dtype
        && lhs_dtype != DType::F32
    {
        return match lhs_dtype {
            DType::BF16 => cuda::batch_matmul_backward_bf16_no_host(
                &grad_buf,
                &lhs_buffer,
                &rhs_buffer,
                batch_count,
                dims.m,
                dims.k,
                dims.n,
            ),
            DType::F16 => cuda::batch_matmul_backward_f16_no_host(
                &grad_buf,
                &lhs_buffer,
                &rhs_buffer,
                batch_count,
                dims.m,
                dims.k,
                dims.n,
            ),
            DType::I8 => cuda::batch_matmul_backward_i8_no_host(
                &grad_buf,
                &lhs_buffer,
                lhs_scale.ok_or_else(|| "CUDA I8 lhs scale missing".to_string())?,
                &rhs_buffer,
                rhs_scale.ok_or_else(|| "CUDA I8 rhs scale missing".to_string())?,
                batch_count,
                dims.m,
                dims.k,
                dims.n,
            ),
            DType::F32 => unreachable!("F32 is excluded by the low-precision guard"),
        };
    }
    let lhs_buf = lhs
        .cloned_cuda_f32_buffer()
        .ok_or_else(|| "CUDA batch_matmul backward expected lhs resident buffer".to_string())?;
    let rhs_buf = rhs
        .cloned_cuda_f32_buffer()
        .ok_or_else(|| "CUDA batch_matmul backward expected rhs resident buffer".to_string())?;

    cuda::batch_matmul_backward_f32_no_host(
        &grad_buf,
        &lhs_buf,
        &rhs_buf,
        batch_count,
        dims.m,
        dims.k,
        dims.n,
    )
}
