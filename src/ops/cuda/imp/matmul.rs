use super::memory::validate_i8_scale_product;
use super::*;

pub(super) fn validate_matmul_forward_lengths(
    lhs_len: usize,
    rhs_len: usize,
    m: usize,
    n: usize,
    k: usize,
    label: &str,
) -> Result<usize, String> {
    let expected_lhs = checked_len(&format!("{label} lhs length"), &[m, k])?;
    let expected_rhs = checked_len(&format!("{label} rhs length"), &[n, k])?;
    let output_len = checked_len(&format!("{label} output length"), &[m, n])?;
    if lhs_len != expected_lhs {
        return Err(format!(
            "{label} lhs length mismatch: expected {expected_lhs}, got {lhs_len}"
        ));
    }
    if rhs_len != expected_rhs {
        return Err(format!(
            "{label} rhs length mismatch: expected {expected_rhs}, got {rhs_len}"
        ));
    }
    Ok(output_len)
}

pub(super) fn validate_batch_matmul_forward_lengths(
    lhs_len: usize,
    rhs_len: usize,
    batch_count: usize,
    m: usize,
    n: usize,
    k: usize,
    label: &str,
) -> Result<usize, String> {
    let expected_lhs = checked_len(&format!("{label} lhs length"), &[batch_count, m, k])?;
    let expected_rhs = checked_len(&format!("{label} rhs length"), &[batch_count, k, n])?;
    let output_len = checked_len(&format!("{label} output length"), &[batch_count, m, n])?;
    if lhs_len != expected_lhs {
        return Err(format!(
            "{label} lhs length mismatch: expected {expected_lhs}, got {lhs_len}"
        ));
    }
    if rhs_len != expected_rhs {
        return Err(format!(
            "{label} rhs length mismatch: expected {expected_rhs}, got {rhs_len}"
        ));
    }
    Ok(output_len)
}

pub fn matmul_f32(
    a: &CudaBuffer,
    b: &CudaBuffer,
    m: usize,
    n: usize,
    k: usize,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    let out = matmul_f32_no_host(a, b, m, n, k)?;
    let host = download_f32(&out)?;
    Ok((out, host))
}

pub fn matmul_f32_no_host(
    a: &CudaBuffer,
    b: &CudaBuffer,
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaBuffer, String> {
    let output_len = validate_matmul_forward_lengths(a.len(), b.len(), m, n, k, "CUDA matmul")?;
    let out = alloc_f32(output_len)?;
    let status =
        unsafe { lumen_cuda_matmul_f32_device(a.handle(), b.handle(), out.handle(), m, n, k) };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}

pub fn matmul_bf16_host_no_host(
    a: &[u16],
    b: &[u16],
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaBuffer, String> {
    let output_len = validate_matmul_forward_lengths(a.len(), b.len(), m, n, k, "CUDA matmul")?;
    let out = alloc_f32(output_len)?;
    let status = unsafe {
        lumen_cuda_matmul_bf16_host_device(a.as_ptr(), b.as_ptr(), out.handle(), m, n, k)
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}

pub fn matmul_bf16_buffer_no_host(
    a: &CudaBuffer,
    b: &CudaBuffer,
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaBuffer, String> {
    let output_len = validate_matmul_forward_lengths(a.len(), b.len(), m, n, k, "CUDA matmul")?;
    let out = alloc_f32(output_len)?;
    let status =
        unsafe { lumen_cuda_matmul_bf16_device(a.handle(), b.handle(), out.handle(), m, n, k) };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}

pub fn matmul_bf16_typed_output_buffer_no_host(
    a: &CudaBuffer,
    b: &CudaBuffer,
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaBuffer, String> {
    let output_len = validate_matmul_forward_lengths(a.len(), b.len(), m, n, k, "CUDA matmul")?;
    let out = alloc_storage(output_len)?;
    let status = unsafe {
        lumen_cuda_matmul_bf16_typed_out_device(a.handle(), b.handle(), out.handle(), m, n, k)
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}

pub fn matmul_f16_host_no_host(
    a: &[u16],
    b: &[u16],
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaBuffer, String> {
    let output_len = validate_matmul_forward_lengths(a.len(), b.len(), m, n, k, "CUDA matmul")?;
    let out = alloc_f32(output_len)?;
    let status =
        unsafe { lumen_cuda_matmul_f16_host_device(a.as_ptr(), b.as_ptr(), out.handle(), m, n, k) };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}

pub fn matmul_f16_buffer_no_host(
    a: &CudaBuffer,
    b: &CudaBuffer,
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaBuffer, String> {
    let output_len = validate_matmul_forward_lengths(a.len(), b.len(), m, n, k, "CUDA matmul")?;
    let out = alloc_f32(output_len)?;
    let status =
        unsafe { lumen_cuda_matmul_f16_device(a.handle(), b.handle(), out.handle(), m, n, k) };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}

pub fn matmul_f16_typed_output_buffer_no_host(
    a: &CudaBuffer,
    b: &CudaBuffer,
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaBuffer, String> {
    let output_len = validate_matmul_forward_lengths(a.len(), b.len(), m, n, k, "CUDA matmul")?;
    let out = alloc_storage(output_len)?;
    let status = unsafe {
        lumen_cuda_matmul_f16_typed_out_device(a.handle(), b.handle(), out.handle(), m, n, k)
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}

pub fn matmul_i8_host_no_host(
    a: &[i8],
    a_scale: f32,
    b: &[i8],
    b_scale: f32,
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaBuffer, String> {
    validate_i8_scale_product(a_scale, b_scale, "I8xI8 matmul")?;
    let output_len = validate_matmul_forward_lengths(a.len(), b.len(), m, n, k, "CUDA matmul")?;
    let out = alloc_f32(output_len)?;
    let status = unsafe {
        lumen_cuda_matmul_i8_host_device(
            a.as_ptr(),
            b.as_ptr(),
            a_scale,
            b_scale,
            out.handle(),
            m,
            n,
            k,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}

pub fn matmul_i8_buffer_no_host(
    a: &CudaBuffer,
    a_scale: f32,
    b: &CudaBuffer,
    b_scale: f32,
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaBuffer, String> {
    validate_i8_scale_product(a_scale, b_scale, "I8xI8 matmul")?;
    let output_len = validate_matmul_forward_lengths(a.len(), b.len(), m, n, k, "CUDA matmul")?;
    let out = alloc_f32(output_len)?;
    let status = unsafe {
        lumen_cuda_matmul_i8_device(
            a.handle(),
            a_scale,
            b.handle(),
            b_scale,
            out.handle(),
            m,
            n,
            k,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}

pub fn matmul_bf16_i8_buffer_no_host(
    a: &CudaBuffer,
    b: &CudaBuffer,
    b_scale: f32,
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaBuffer, String> {
    let output_len = validate_matmul_forward_lengths(a.len(), b.len(), m, n, k, "CUDA matmul")?;
    let out = alloc_f32(output_len)?;
    let status = unsafe {
        lumen_cuda_matmul_bf16_i8_device(a.handle(), b.handle(), b_scale, out.handle(), m, n, k)
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}

pub fn matmul_f16_i8_buffer_no_host(
    a: &CudaBuffer,
    b: &CudaBuffer,
    b_scale: f32,
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaBuffer, String> {
    let output_len = validate_matmul_forward_lengths(a.len(), b.len(), m, n, k, "CUDA matmul")?;
    let out = alloc_f32(output_len)?;
    let status = unsafe {
        lumen_cuda_matmul_f16_i8_device(a.handle(), b.handle(), b_scale, out.handle(), m, n, k)
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}

pub fn matmul_bf16_i8_dynamic_buffer_no_host(
    a: &CudaBuffer,
    b: &CudaBuffer,
    b_scale: f32,
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaBuffer, String> {
    let output_len =
        validate_matmul_forward_lengths(a.len(), b.len(), m, n, k, "CUDA dynamic matmul")?;
    let out = alloc_f32(output_len)?;
    let status = unsafe {
        lumen_cuda_matmul_bf16_i8_dynamic_device(
            a.handle(),
            b.handle(),
            b_scale,
            out.handle(),
            m,
            n,
            k,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}

pub fn matmul_f16_i8_dynamic_buffer_no_host(
    a: &CudaBuffer,
    b: &CudaBuffer,
    b_scale: f32,
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaBuffer, String> {
    let output_len =
        validate_matmul_forward_lengths(a.len(), b.len(), m, n, k, "CUDA dynamic matmul")?;
    let out = alloc_f32(output_len)?;
    let status = unsafe {
        lumen_cuda_matmul_f16_i8_dynamic_device(
            a.handle(),
            b.handle(),
            b_scale,
            out.handle(),
            m,
            n,
            k,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}

pub fn matmul_f32_i8_buffer_no_host(
    a: &CudaBuffer,
    b: &CudaBuffer,
    b_scale: f32,
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaBuffer, String> {
    let output_len = validate_matmul_forward_lengths(a.len(), b.len(), m, n, k, "CUDA matmul")?;
    let out = alloc_f32(output_len)?;
    let status = unsafe {
        lumen_cuda_matmul_f32_i8_device(a.handle(), b.handle(), b_scale, out.handle(), m, n, k)
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}

pub fn matmul_i8_bf16_buffer_no_host(
    a: &CudaBuffer,
    a_scale: f32,
    b: &CudaBuffer,
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaBuffer, String> {
    let output_len = validate_matmul_forward_lengths(a.len(), b.len(), m, n, k, "CUDA matmul")?;
    let out = alloc_f32(output_len)?;
    let status = unsafe {
        lumen_cuda_matmul_i8_bf16_device(a.handle(), a_scale, b.handle(), out.handle(), m, n, k)
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}

pub fn matmul_i8_f16_buffer_no_host(
    a: &CudaBuffer,
    a_scale: f32,
    b: &CudaBuffer,
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaBuffer, String> {
    let output_len = validate_matmul_forward_lengths(a.len(), b.len(), m, n, k, "CUDA matmul")?;
    let out = alloc_f32(output_len)?;
    let status = unsafe {
        lumen_cuda_matmul_i8_f16_device(a.handle(), a_scale, b.handle(), out.handle(), m, n, k)
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}

pub fn matmul_i8_f32_buffer_no_host(
    a: &CudaBuffer,
    a_scale: f32,
    b: &CudaBuffer,
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaBuffer, String> {
    let output_len = validate_matmul_forward_lengths(a.len(), b.len(), m, n, k, "CUDA matmul")?;
    let out = alloc_f32(output_len)?;
    let status = unsafe {
        lumen_cuda_matmul_i8_f32_device(a.handle(), a_scale, b.handle(), out.handle(), m, n, k)
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}

pub fn matmul_i8_typed_output_buffer_no_host(
    a: &CudaBuffer,
    a_scale: f32,
    b: &CudaBuffer,
    b_scale: f32,
    m: usize,
    n: usize,
    k: usize,
) -> Result<(CudaBuffer, f32), String> {
    validate_i8_scale_product(a_scale, b_scale, "typed I8xI8 matmul")?;
    let output_len = validate_matmul_forward_lengths(a.len(), b.len(), m, n, k, "CUDA matmul")?;
    let out = alloc_storage(output_len)?;
    let mut out_scale = 1.0f32;
    let status = unsafe {
        lumen_cuda_matmul_i8_typed_out_device(
            a.handle(),
            a_scale,
            b.handle(),
            b_scale,
            out.handle(),
            m,
            n,
            k,
            &mut out_scale as *mut f32,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok((out, out_scale))
}

pub fn batch_matmul_f32(
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    batch_count: usize,
    m: usize,
    n: usize,
    k: usize,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    let out = batch_matmul_f32_no_host(lhs, rhs, batch_count, m, n, k)?;
    let host = download_f32(&out)?;
    Ok((out, host))
}

pub fn batch_matmul_f32_no_host(
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    batch_count: usize,
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaBuffer, String> {
    let output_len = validate_batch_matmul_forward_lengths(
        lhs.len(),
        rhs.len(),
        batch_count,
        m,
        n,
        k,
        "CUDA batch_matmul",
    )?;
    let out = alloc_f32(output_len)?;
    let status = unsafe {
        lumen_cuda_batch_matmul_f32_device(
            lhs.handle(),
            rhs.handle(),
            out.handle(),
            batch_count,
            m,
            n,
            k,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}

pub fn batch_matmul_bf16_buffer_no_host(
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    batch_count: usize,
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaBuffer, String> {
    let output_len = validate_batch_matmul_forward_lengths(
        lhs.len(),
        rhs.len(),
        batch_count,
        m,
        n,
        k,
        "CUDA batch_matmul",
    )?;
    let out = alloc_f32(output_len)?;
    let status = unsafe {
        lumen_cuda_batch_matmul_bf16_device(
            lhs.handle(),
            rhs.handle(),
            out.handle(),
            batch_count,
            m,
            n,
            k,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}

pub fn batch_matmul_bf16_typed_output_buffer_no_host(
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    batch_count: usize,
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaBuffer, String> {
    let output_len = validate_batch_matmul_forward_lengths(
        lhs.len(),
        rhs.len(),
        batch_count,
        m,
        n,
        k,
        "CUDA batch_matmul",
    )?;
    let out = alloc_storage(output_len)?;
    let status = unsafe {
        lumen_cuda_batch_matmul_bf16_typed_out_device(
            lhs.handle(),
            rhs.handle(),
            out.handle(),
            batch_count,
            m,
            n,
            k,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}

pub fn batch_matmul_f16_buffer_no_host(
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    batch_count: usize,
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaBuffer, String> {
    let output_len = validate_batch_matmul_forward_lengths(
        lhs.len(),
        rhs.len(),
        batch_count,
        m,
        n,
        k,
        "CUDA batch_matmul",
    )?;
    let out = alloc_f32(output_len)?;
    let status = unsafe {
        lumen_cuda_batch_matmul_f16_device(
            lhs.handle(),
            rhs.handle(),
            out.handle(),
            batch_count,
            m,
            n,
            k,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}

pub fn batch_matmul_f16_typed_output_buffer_no_host(
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    batch_count: usize,
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaBuffer, String> {
    let output_len = validate_batch_matmul_forward_lengths(
        lhs.len(),
        rhs.len(),
        batch_count,
        m,
        n,
        k,
        "CUDA batch_matmul",
    )?;
    let out = alloc_storage(output_len)?;
    let status = unsafe {
        lumen_cuda_batch_matmul_f16_typed_out_device(
            lhs.handle(),
            rhs.handle(),
            out.handle(),
            batch_count,
            m,
            n,
            k,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}

pub fn batch_matmul_bf16_i8_buffer_no_host(
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    rhs_scale: f32,
    batch_count: usize,
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaBuffer, String> {
    let output_len = validate_batch_matmul_forward_lengths(
        lhs.len(),
        rhs.len(),
        batch_count,
        m,
        n,
        k,
        "CUDA batch_matmul",
    )?;
    let out = alloc_f32(output_len)?;
    let status = unsafe {
        lumen_cuda_batch_matmul_bf16_i8_device(
            lhs.handle(),
            rhs.handle(),
            rhs_scale,
            out.handle(),
            batch_count,
            m,
            n,
            k,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}

pub fn batch_matmul_f16_i8_buffer_no_host(
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    rhs_scale: f32,
    batch_count: usize,
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaBuffer, String> {
    let output_len = validate_batch_matmul_forward_lengths(
        lhs.len(),
        rhs.len(),
        batch_count,
        m,
        n,
        k,
        "CUDA batch_matmul",
    )?;
    let out = alloc_f32(output_len)?;
    let status = unsafe {
        lumen_cuda_batch_matmul_f16_i8_device(
            lhs.handle(),
            rhs.handle(),
            rhs_scale,
            out.handle(),
            batch_count,
            m,
            n,
            k,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}

pub fn batch_matmul_f32_i8_buffer_no_host(
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    rhs_scale: f32,
    batch_count: usize,
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaBuffer, String> {
    let output_len = validate_batch_matmul_forward_lengths(
        lhs.len(),
        rhs.len(),
        batch_count,
        m,
        n,
        k,
        "CUDA batch_matmul",
    )?;
    let out = alloc_f32(output_len)?;
    let status = unsafe {
        lumen_cuda_batch_matmul_f32_i8_device(
            lhs.handle(),
            rhs.handle(),
            rhs_scale,
            out.handle(),
            batch_count,
            m,
            n,
            k,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}

pub fn batch_matmul_i8_bf16_buffer_no_host(
    lhs: &CudaBuffer,
    lhs_scale: f32,
    rhs: &CudaBuffer,
    batch_count: usize,
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaBuffer, String> {
    let output_len = validate_batch_matmul_forward_lengths(
        lhs.len(),
        rhs.len(),
        batch_count,
        m,
        n,
        k,
        "CUDA batch_matmul",
    )?;
    let out = alloc_f32(output_len)?;
    let status = unsafe {
        lumen_cuda_batch_matmul_i8_bf16_device(
            lhs.handle(),
            lhs_scale,
            rhs.handle(),
            out.handle(),
            batch_count,
            m,
            n,
            k,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}

pub fn batch_matmul_i8_f16_buffer_no_host(
    lhs: &CudaBuffer,
    lhs_scale: f32,
    rhs: &CudaBuffer,
    batch_count: usize,
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaBuffer, String> {
    let output_len = validate_batch_matmul_forward_lengths(
        lhs.len(),
        rhs.len(),
        batch_count,
        m,
        n,
        k,
        "CUDA batch_matmul",
    )?;
    let out = alloc_f32(output_len)?;
    let status = unsafe {
        lumen_cuda_batch_matmul_i8_f16_device(
            lhs.handle(),
            lhs_scale,
            rhs.handle(),
            out.handle(),
            batch_count,
            m,
            n,
            k,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}

pub fn batch_matmul_i8_f32_buffer_no_host(
    lhs: &CudaBuffer,
    lhs_scale: f32,
    rhs: &CudaBuffer,
    batch_count: usize,
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaBuffer, String> {
    let output_len = validate_batch_matmul_forward_lengths(
        lhs.len(),
        rhs.len(),
        batch_count,
        m,
        n,
        k,
        "CUDA batch_matmul",
    )?;
    let out = alloc_f32(output_len)?;
    let status = unsafe {
        lumen_cuda_batch_matmul_i8_f32_device(
            lhs.handle(),
            lhs_scale,
            rhs.handle(),
            out.handle(),
            batch_count,
            m,
            n,
            k,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
pub fn batch_matmul_i8_buffer_no_host(
    lhs: &CudaBuffer,
    lhs_scale: f32,
    rhs: &CudaBuffer,
    rhs_scale: f32,
    batch_count: usize,
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaBuffer, String> {
    validate_i8_scale_product(lhs_scale, rhs_scale, "I8xI8 batch_matmul")?;
    let output_len = validate_batch_matmul_forward_lengths(
        lhs.len(),
        rhs.len(),
        batch_count,
        m,
        n,
        k,
        "CUDA batch_matmul",
    )?;
    let out = alloc_f32(output_len)?;
    let status = unsafe {
        lumen_cuda_batch_matmul_i8_device(
            lhs.handle(),
            lhs_scale,
            rhs.handle(),
            rhs_scale,
            out.handle(),
            batch_count,
            m,
            n,
            k,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}

pub fn batch_matmul_i8_typed_output_buffer_no_host(
    lhs: &CudaBuffer,
    lhs_scale: f32,
    rhs: &CudaBuffer,
    rhs_scale: f32,
    batch_count: usize,
    m: usize,
    n: usize,
    k: usize,
) -> Result<(CudaBuffer, f32), String> {
    validate_i8_scale_product(lhs_scale, rhs_scale, "typed I8xI8 batch_matmul")?;
    let output_len = validate_batch_matmul_forward_lengths(
        lhs.len(),
        rhs.len(),
        batch_count,
        m,
        n,
        k,
        "CUDA batch_matmul",
    )?;
    let out = alloc_storage(output_len)?;
    let mut out_scale = 1.0f32;
    let status = unsafe {
        lumen_cuda_batch_matmul_i8_typed_out_device(
            lhs.handle(),
            lhs_scale,
            rhs.handle(),
            rhs_scale,
            out.handle(),
            batch_count,
            m,
            n,
            k,
            &mut out_scale as *mut f32,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok((out, out_scale))
}

pub fn matmul_backward_f32_no_host(
    grad: &CudaBuffer,
    a: &CudaBuffer,
    b: &CudaBuffer,
    m: usize,
    k: usize,
    n: usize,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    let (lhs_len, rhs_len) = validate_matmul_backward_lengths(grad, a, b, m, k, n, "F32")?;
    let da = alloc_f32(lhs_len)?;
    let db = alloc_f32(rhs_len)?;
    let status = unsafe {
        lumen_cuda_matmul_backward_f32_device(
            grad.handle(),
            a.handle(),
            b.handle(),
            da.handle(),
            db.handle(),
            m,
            k,
            n,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok((da, db))
}

pub fn matmul_backward_bf16_i8_no_host(
    grad: &CudaBuffer,
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    rhs_scale: f32,
    m: usize,
    k: usize,
    n: usize,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    let (lhs_len, rhs_len) = validate_matmul_backward_lengths(grad, lhs, rhs, m, k, n, "BF16xI8")?;
    let d_lhs = alloc_f32(lhs_len)?;
    let d_rhs = alloc_f32(rhs_len)?;
    let status = unsafe {
        lumen_cuda_matmul_backward_bf16_i8_device(
            grad.handle(),
            lhs.handle(),
            rhs.handle(),
            rhs_scale,
            d_lhs.handle(),
            d_rhs.handle(),
            m,
            k,
            n,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok((d_lhs, d_rhs))
}

pub fn matmul_backward_f16_i8_no_host(
    grad: &CudaBuffer,
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    rhs_scale: f32,
    m: usize,
    k: usize,
    n: usize,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    let (lhs_len, rhs_len) = validate_matmul_backward_lengths(grad, lhs, rhs, m, k, n, "F16xI8")?;
    let d_lhs = alloc_f32(lhs_len)?;
    let d_rhs = alloc_f32(rhs_len)?;
    let status = unsafe {
        lumen_cuda_matmul_backward_f16_i8_device(
            grad.handle(),
            lhs.handle(),
            rhs.handle(),
            rhs_scale,
            d_lhs.handle(),
            d_rhs.handle(),
            m,
            k,
            n,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok((d_lhs, d_rhs))
}

pub fn matmul_backward_f32_i8_no_host(
    grad: &CudaBuffer,
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    rhs_scale: f32,
    m: usize,
    k: usize,
    n: usize,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    let (lhs_len, rhs_len) = validate_matmul_backward_lengths(grad, lhs, rhs, m, k, n, "F32xI8")?;
    let d_lhs = alloc_f32(lhs_len)?;
    let d_rhs = alloc_f32(rhs_len)?;
    let status = unsafe {
        lumen_cuda_matmul_backward_f32_i8_device(
            grad.handle(),
            lhs.handle(),
            rhs.handle(),
            rhs_scale,
            d_lhs.handle(),
            d_rhs.handle(),
            m,
            k,
            n,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok((d_lhs, d_rhs))
}

pub fn matmul_backward_i8_bf16_no_host(
    grad: &CudaBuffer,
    lhs: &CudaBuffer,
    lhs_scale: f32,
    rhs: &CudaBuffer,
    m: usize,
    k: usize,
    n: usize,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    let (lhs_len, rhs_len) = validate_matmul_backward_lengths(grad, lhs, rhs, m, k, n, "I8xBF16")?;
    let d_lhs = alloc_f32(lhs_len)?;
    let d_rhs = alloc_f32(rhs_len)?;
    let status = unsafe {
        lumen_cuda_matmul_backward_i8_bf16_device(
            grad.handle(),
            lhs.handle(),
            lhs_scale,
            rhs.handle(),
            d_lhs.handle(),
            d_rhs.handle(),
            m,
            k,
            n,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok((d_lhs, d_rhs))
}

pub fn matmul_backward_i8_f16_no_host(
    grad: &CudaBuffer,
    lhs: &CudaBuffer,
    lhs_scale: f32,
    rhs: &CudaBuffer,
    m: usize,
    k: usize,
    n: usize,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    let (lhs_len, rhs_len) = validate_matmul_backward_lengths(grad, lhs, rhs, m, k, n, "I8xF16")?;
    let d_lhs = alloc_f32(lhs_len)?;
    let d_rhs = alloc_f32(rhs_len)?;
    let status = unsafe {
        lumen_cuda_matmul_backward_i8_f16_device(
            grad.handle(),
            lhs.handle(),
            lhs_scale,
            rhs.handle(),
            d_lhs.handle(),
            d_rhs.handle(),
            m,
            k,
            n,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok((d_lhs, d_rhs))
}

pub fn matmul_backward_i8_f32_no_host(
    grad: &CudaBuffer,
    lhs: &CudaBuffer,
    lhs_scale: f32,
    rhs: &CudaBuffer,
    m: usize,
    k: usize,
    n: usize,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    let (lhs_len, rhs_len) = validate_matmul_backward_lengths(grad, lhs, rhs, m, k, n, "I8xF32")?;
    let d_lhs = alloc_f32(lhs_len)?;
    let d_rhs = alloc_f32(rhs_len)?;
    let status = unsafe {
        lumen_cuda_matmul_backward_i8_f32_device(
            grad.handle(),
            lhs.handle(),
            lhs_scale,
            rhs.handle(),
            d_lhs.handle(),
            d_rhs.handle(),
            m,
            k,
            n,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok((d_lhs, d_rhs))
}

pub(super) fn validate_matmul_backward_lengths(
    grad: &CudaBuffer,
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    m: usize,
    k: usize,
    n: usize,
    dtype: &str,
) -> Result<(usize, usize), String> {
    let expected_grad = checked_len(
        &format!("CUDA {dtype} matmul backward grad length"),
        &[m, n],
    )?;
    let expected_lhs = checked_len(&format!("CUDA {dtype} matmul backward lhs length"), &[m, k])?;
    let expected_rhs = checked_len(&format!("CUDA {dtype} matmul backward rhs length"), &[n, k])?;
    if grad.len() != expected_grad {
        return Err(format!(
            "CUDA {dtype} matmul backward grad length mismatch: expected {}, got {}",
            expected_grad,
            grad.len()
        ));
    }
    if lhs.len() != expected_lhs {
        return Err(format!(
            "CUDA {dtype} matmul backward lhs length mismatch: expected {}, got {}",
            expected_lhs,
            lhs.len()
        ));
    }
    if rhs.len() != expected_rhs {
        return Err(format!(
            "CUDA {dtype} matmul backward rhs length mismatch: expected {}, got {}",
            expected_rhs,
            rhs.len()
        ));
    }
    Ok((expected_lhs, expected_rhs))
}

pub fn batch_matmul_backward_f32_no_host(
    grad: &CudaBuffer,
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    batch_count: usize,
    m: usize,
    k: usize,
    n: usize,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    let (lhs_len, rhs_len) =
        validate_batch_matmul_backward_lengths(grad, lhs, rhs, batch_count, m, k, n, "F32")?;
    let d_lhs = alloc_f32(lhs_len)?;
    let d_rhs = alloc_f32(rhs_len)?;
    let status = unsafe {
        lumen_cuda_batch_matmul_backward_f32_device(
            grad.handle(),
            lhs.handle(),
            rhs.handle(),
            d_lhs.handle(),
            d_rhs.handle(),
            batch_count,
            m,
            k,
            n,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok((d_lhs, d_rhs))
}

pub fn batch_matmul_backward_bf16_no_host(
    grad: &CudaBuffer,
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    batch_count: usize,
    m: usize,
    k: usize,
    n: usize,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    let (lhs_len, rhs_len) =
        validate_batch_matmul_backward_lengths(grad, lhs, rhs, batch_count, m, k, n, "BF16")?;
    let d_lhs = alloc_f32(lhs_len)?;
    let d_rhs = alloc_f32(rhs_len)?;
    let status = unsafe {
        lumen_cuda_batch_matmul_backward_bf16_device(
            grad.handle(),
            lhs.handle(),
            rhs.handle(),
            d_lhs.handle(),
            d_rhs.handle(),
            batch_count,
            m,
            k,
            n,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok((d_lhs, d_rhs))
}

pub fn batch_matmul_backward_f16_no_host(
    grad: &CudaBuffer,
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    batch_count: usize,
    m: usize,
    k: usize,
    n: usize,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    let (lhs_len, rhs_len) =
        validate_batch_matmul_backward_lengths(grad, lhs, rhs, batch_count, m, k, n, "F16")?;
    let d_lhs = alloc_f32(lhs_len)?;
    let d_rhs = alloc_f32(rhs_len)?;
    let status = unsafe {
        lumen_cuda_batch_matmul_backward_f16_device(
            grad.handle(),
            lhs.handle(),
            rhs.handle(),
            d_lhs.handle(),
            d_rhs.handle(),
            batch_count,
            m,
            k,
            n,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok((d_lhs, d_rhs))
}

pub fn batch_matmul_backward_bf16_i8_no_host(
    grad: &CudaBuffer,
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    rhs_scale: f32,
    batch_count: usize,
    m: usize,
    k: usize,
    n: usize,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    let (lhs_len, rhs_len) =
        validate_batch_matmul_backward_lengths(grad, lhs, rhs, batch_count, m, k, n, "BF16xI8")?;
    let d_lhs = alloc_f32(lhs_len)?;
    let d_rhs = alloc_f32(rhs_len)?;
    let status = unsafe {
        lumen_cuda_batch_matmul_backward_bf16_i8_device(
            grad.handle(),
            lhs.handle(),
            rhs.handle(),
            rhs_scale,
            d_lhs.handle(),
            d_rhs.handle(),
            batch_count,
            m,
            k,
            n,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok((d_lhs, d_rhs))
}

pub fn batch_matmul_backward_f16_i8_no_host(
    grad: &CudaBuffer,
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    rhs_scale: f32,
    batch_count: usize,
    m: usize,
    k: usize,
    n: usize,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    let (lhs_len, rhs_len) =
        validate_batch_matmul_backward_lengths(grad, lhs, rhs, batch_count, m, k, n, "F16xI8")?;
    let d_lhs = alloc_f32(lhs_len)?;
    let d_rhs = alloc_f32(rhs_len)?;
    let status = unsafe {
        lumen_cuda_batch_matmul_backward_f16_i8_device(
            grad.handle(),
            lhs.handle(),
            rhs.handle(),
            rhs_scale,
            d_lhs.handle(),
            d_rhs.handle(),
            batch_count,
            m,
            k,
            n,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok((d_lhs, d_rhs))
}

pub fn batch_matmul_backward_f32_i8_no_host(
    grad: &CudaBuffer,
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    rhs_scale: f32,
    batch_count: usize,
    m: usize,
    k: usize,
    n: usize,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    let (lhs_len, rhs_len) =
        validate_batch_matmul_backward_lengths(grad, lhs, rhs, batch_count, m, k, n, "F32xI8")?;
    let d_lhs = alloc_f32(lhs_len)?;
    let d_rhs = alloc_f32(rhs_len)?;
    let status = unsafe {
        lumen_cuda_batch_matmul_backward_f32_i8_device(
            grad.handle(),
            lhs.handle(),
            rhs.handle(),
            rhs_scale,
            d_lhs.handle(),
            d_rhs.handle(),
            batch_count,
            m,
            k,
            n,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok((d_lhs, d_rhs))
}

pub fn batch_matmul_backward_i8_bf16_no_host(
    grad: &CudaBuffer,
    lhs: &CudaBuffer,
    lhs_scale: f32,
    rhs: &CudaBuffer,
    batch_count: usize,
    m: usize,
    k: usize,
    n: usize,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    let (lhs_len, rhs_len) =
        validate_batch_matmul_backward_lengths(grad, lhs, rhs, batch_count, m, k, n, "I8xBF16")?;
    let d_lhs = alloc_f32(lhs_len)?;
    let d_rhs = alloc_f32(rhs_len)?;
    let status = unsafe {
        lumen_cuda_batch_matmul_backward_i8_bf16_device(
            grad.handle(),
            lhs.handle(),
            lhs_scale,
            rhs.handle(),
            d_lhs.handle(),
            d_rhs.handle(),
            batch_count,
            m,
            k,
            n,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok((d_lhs, d_rhs))
}

pub fn batch_matmul_backward_i8_f16_no_host(
    grad: &CudaBuffer,
    lhs: &CudaBuffer,
    lhs_scale: f32,
    rhs: &CudaBuffer,
    batch_count: usize,
    m: usize,
    k: usize,
    n: usize,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    let (lhs_len, rhs_len) =
        validate_batch_matmul_backward_lengths(grad, lhs, rhs, batch_count, m, k, n, "I8xF16")?;
    let d_lhs = alloc_f32(lhs_len)?;
    let d_rhs = alloc_f32(rhs_len)?;
    let status = unsafe {
        lumen_cuda_batch_matmul_backward_i8_f16_device(
            grad.handle(),
            lhs.handle(),
            lhs_scale,
            rhs.handle(),
            d_lhs.handle(),
            d_rhs.handle(),
            batch_count,
            m,
            k,
            n,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok((d_lhs, d_rhs))
}

pub fn batch_matmul_backward_i8_f32_no_host(
    grad: &CudaBuffer,
    lhs: &CudaBuffer,
    lhs_scale: f32,
    rhs: &CudaBuffer,
    batch_count: usize,
    m: usize,
    k: usize,
    n: usize,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    let (lhs_len, rhs_len) =
        validate_batch_matmul_backward_lengths(grad, lhs, rhs, batch_count, m, k, n, "I8xF32")?;
    let d_lhs = alloc_f32(lhs_len)?;
    let d_rhs = alloc_f32(rhs_len)?;
    let status = unsafe {
        lumen_cuda_batch_matmul_backward_i8_f32_device(
            grad.handle(),
            lhs.handle(),
            lhs_scale,
            rhs.handle(),
            d_lhs.handle(),
            d_rhs.handle(),
            batch_count,
            m,
            k,
            n,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok((d_lhs, d_rhs))
}

#[allow(clippy::too_many_arguments)]
pub fn batch_matmul_backward_i8_no_host(
    grad: &CudaBuffer,
    lhs: &CudaBuffer,
    lhs_scale: f32,
    rhs: &CudaBuffer,
    rhs_scale: f32,
    batch_count: usize,
    m: usize,
    k: usize,
    n: usize,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    let (lhs_len, rhs_len) =
        validate_batch_matmul_backward_lengths(grad, lhs, rhs, batch_count, m, k, n, "I8")?;
    let d_lhs = alloc_f32(lhs_len)?;
    let d_rhs = alloc_f32(rhs_len)?;
    let status = unsafe {
        lumen_cuda_batch_matmul_backward_i8_device(
            grad.handle(),
            lhs.handle(),
            lhs_scale,
            rhs.handle(),
            rhs_scale,
            d_lhs.handle(),
            d_rhs.handle(),
            batch_count,
            m,
            k,
            n,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok((d_lhs, d_rhs))
}

pub(super) fn validate_batch_matmul_backward_lengths(
    grad: &CudaBuffer,
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    batch_count: usize,
    m: usize,
    k: usize,
    n: usize,
    dtype: &str,
) -> Result<(usize, usize), String> {
    let expected_grad = checked_len(
        &format!("CUDA {dtype} batch_matmul backward grad length"),
        &[batch_count, m, n],
    )?;
    let expected_lhs = checked_len(
        &format!("CUDA {dtype} batch_matmul backward lhs length"),
        &[batch_count, m, k],
    )?;
    let expected_rhs = checked_len(
        &format!("CUDA {dtype} batch_matmul backward rhs length"),
        &[batch_count, k, n],
    )?;
    if grad.len() != expected_grad {
        return Err(format!(
            "CUDA {dtype} batch_matmul backward grad length mismatch: expected {}, got {}",
            expected_grad,
            grad.len()
        ));
    }
    if lhs.len() != expected_lhs {
        return Err(format!(
            "CUDA {dtype} batch_matmul backward lhs length mismatch: expected {}, got {}",
            expected_lhs,
            lhs.len()
        ));
    }
    if rhs.len() != expected_rhs {
        return Err(format!(
            "CUDA {dtype} batch_matmul backward rhs length mismatch: expected {}, got {}",
            expected_rhs,
            rhs.len()
        ));
    }
    Ok((expected_lhs, expected_rhs))
}
