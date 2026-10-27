use super::*;

pub fn unary_f32(input: &CudaBuffer, op: UnaryOp) -> Result<(CudaBuffer, Vec<f32>), String> {
    let out = unary_f32_buffer(input, op)?;
    let host = download_f32(&out)?;
    Ok((out, host))
}

pub fn unary_f32_buffer(input: &CudaBuffer, op: UnaryOp) -> Result<CudaBuffer, String> {
    let out = alloc_f32(input.len())?;
    let status = unsafe {
        lumen_cuda_unary_f32_device(input.handle(), out.handle(), input.len(), op as c_int)
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}

pub fn unary_f16_buffer(input: &CudaBuffer, op: UnaryOp) -> Result<CudaBuffer, String> {
    let out = alloc_f32(input.len())?;
    let status = unsafe {
        lumen_cuda_unary_f16_device(input.handle(), out.handle(), input.len(), op as c_int)
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}

pub fn unary_f16_typed_output_buffer(
    input: &CudaBuffer,
    op: UnaryOp,
) -> Result<CudaBuffer, String> {
    let out = alloc_storage(input.len())?;
    let status = unsafe {
        lumen_cuda_unary_f16_typed_out_device(
            input.handle(),
            out.handle(),
            input.len(),
            op as c_int,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}

pub fn unary_bf16_buffer(input: &CudaBuffer, op: UnaryOp) -> Result<CudaBuffer, String> {
    let out = alloc_f32(input.len())?;
    let status = unsafe {
        lumen_cuda_unary_bf16_device(input.handle(), out.handle(), input.len(), op as c_int)
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}

pub fn unary_bf16_typed_output_buffer(
    input: &CudaBuffer,
    op: UnaryOp,
) -> Result<CudaBuffer, String> {
    let out = alloc_storage(input.len())?;
    let status = unsafe {
        lumen_cuda_unary_bf16_typed_out_device(
            input.handle(),
            out.handle(),
            input.len(),
            op as c_int,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}

pub fn unary_i8_buffer(input: &CudaBuffer, scale: f32, op: UnaryOp) -> Result<CudaBuffer, String> {
    let out = alloc_f32(input.len())?;
    let status = unsafe {
        lumen_cuda_unary_i8_device(
            input.handle(),
            scale,
            out.handle(),
            input.len(),
            op as c_int,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}

pub fn unary_i8_relu_typed_output_buffer(input: &CudaBuffer) -> Result<CudaBuffer, String> {
    let out = alloc_storage(input.len())?;
    let status = unsafe {
        lumen_cuda_unary_i8_relu_typed_out_device(input.handle(), out.handle(), input.len())
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}

pub fn unary_backward_f32(
    input: &CudaBuffer,
    output: &CudaBuffer,
    grad: &CudaBuffer,
    op: UnaryOp,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    let out = unary_backward_f32_buffer(input, output, grad, op)?;
    let host = download_f32(&out)?;
    Ok((out, host))
}

pub fn unary_backward_f32_buffer(
    input: &CudaBuffer,
    output: &CudaBuffer,
    grad: &CudaBuffer,
    op: UnaryOp,
) -> Result<CudaBuffer, String> {
    if input.len() != output.len() || input.len() != grad.len() {
        return Err(format!(
            "CUDA unary backward length mismatch: input={}, output={}, grad={}",
            input.len(),
            output.len(),
            grad.len()
        ));
    }
    let out = alloc_f32(input.len())?;
    let status = unsafe {
        lumen_cuda_unary_backward_f32_device(
            input.handle(),
            output.handle(),
            grad.handle(),
            out.handle(),
            input.len(),
            op as c_int,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}

pub fn unary_backward_f16_buffer(
    input: &CudaBuffer,
    output: &CudaBuffer,
    grad: &CudaBuffer,
    op: UnaryOp,
) -> Result<CudaBuffer, String> {
    if input.len() != output.len() || input.len() != grad.len() {
        return Err(format!(
            "CUDA unary F16 backward length mismatch: input={}, output={}, grad={}",
            input.len(),
            output.len(),
            grad.len()
        ));
    }
    let out = alloc_f32(input.len())?;
    let status = unsafe {
        lumen_cuda_unary_backward_f16_device(
            input.handle(),
            output.handle(),
            grad.handle(),
            out.handle(),
            input.len(),
            op as c_int,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}

pub fn unary_backward_bf16_buffer(
    input: &CudaBuffer,
    output: &CudaBuffer,
    grad: &CudaBuffer,
    op: UnaryOp,
) -> Result<CudaBuffer, String> {
    if input.len() != output.len() || input.len() != grad.len() {
        return Err(format!(
            "CUDA unary BF16 backward length mismatch: input={}, output={}, grad={}",
            input.len(),
            output.len(),
            grad.len()
        ));
    }
    let out = alloc_f32(input.len())?;
    let status = unsafe {
        lumen_cuda_unary_backward_bf16_device(
            input.handle(),
            output.handle(),
            grad.handle(),
            out.handle(),
            input.len(),
            op as c_int,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}

pub fn unary_backward_i8_buffer(
    input: &CudaBuffer,
    scale: f32,
    output: &CudaBuffer,
    grad: &CudaBuffer,
    op: UnaryOp,
) -> Result<CudaBuffer, String> {
    if input.len() != output.len() || input.len() != grad.len() {
        return Err(format!(
            "CUDA unary I8 backward length mismatch: input={}, output={}, grad={}",
            input.len(),
            output.len(),
            grad.len()
        ));
    }
    let out = alloc_f32(input.len())?;
    let status = unsafe {
        lumen_cuda_unary_backward_i8_device(
            input.handle(),
            scale,
            output.handle(),
            grad.handle(),
            out.handle(),
            input.len(),
            op as c_int,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}

pub fn binary_f32(
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    op: BinaryOp,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    if lhs.len() != rhs.len() {
        return Err(format!(
            "CUDA binary op length mismatch: lhs={}, rhs={}",
            lhs.len(),
            rhs.len()
        ));
    }
    let out = alloc_f32(lhs.len())?;
    let status = unsafe {
        lumen_cuda_binary_f32_device(
            lhs.handle(),
            rhs.handle(),
            out.handle(),
            lhs.len(),
            op as c_int,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    let host = download_f32(&out)?;
    Ok((out, host))
}

pub fn binary_f32_buffer(
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    op: BinaryOp,
) -> Result<CudaBuffer, String> {
    if lhs.len() != rhs.len() {
        return Err(format!(
            "CUDA binary op length mismatch: lhs={}, rhs={}",
            lhs.len(),
            rhs.len()
        ));
    }
    let out = alloc_f32(lhs.len())?;
    let status = unsafe {
        lumen_cuda_binary_f32_device(
            lhs.handle(),
            rhs.handle(),
            out.handle(),
            lhs.len(),
            op as c_int,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}

pub fn binary_typed_buffer_no_host(
    lhs: &CudaBuffer,
    lhs_dtype: DType,
    lhs_scale: Option<f32>,
    rhs: &CudaBuffer,
    rhs_dtype: DType,
    rhs_scale: Option<f32>,
    op: BinaryOp,
) -> Result<CudaBuffer, String> {
    if lhs.len() != rhs.len() {
        return Err(format!(
            "CUDA typed mixed binary length mismatch: lhs={}, rhs={}",
            lhs.len(),
            rhs.len()
        ));
    }
    let out = alloc_f32(lhs.len())?;
    let status = unsafe {
        lumen_cuda_binary_typed_device(
            lhs.handle(),
            dtype_tag(lhs_dtype)?,
            lhs_scale.unwrap_or(1.0),
            rhs.handle(),
            dtype_tag(rhs_dtype)?,
            rhs_scale.unwrap_or(1.0),
            out.handle(),
            lhs.len(),
            op as c_int,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}

pub fn binary_lowp_typed_output_buffer_no_host(
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    dtype: DType,
    op: BinaryOp,
) -> Result<CudaBuffer, String> {
    if lhs.len() != rhs.len() {
        return Err(format!(
            "CUDA typed-out lowp binary length mismatch: lhs={}, rhs={}",
            lhs.len(),
            rhs.len()
        ));
    }
    if !matches!(dtype, DType::F16 | DType::BF16) {
        return Err("CUDA typed-out lowp binary supports only F16/BF16".to_string());
    }
    let out = alloc_storage(lhs.len())?;
    let status = unsafe {
        lumen_cuda_binary_lowp_typed_out_device(
            lhs.handle(),
            rhs.handle(),
            out.handle(),
            lhs.len(),
            dtype_tag(dtype)?,
            op as c_int,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
pub fn binary_typed_lastdim_buffer_no_host(
    lhs: &CudaBuffer,
    lhs_dtype: DType,
    lhs_scale: Option<f32>,
    rhs: &CudaBuffer,
    rhs_dtype: DType,
    rhs_scale: Option<f32>,
    out_len: usize,
    last_dim: usize,
    vector_on_rhs: bool,
    op: BinaryOp,
) -> Result<CudaBuffer, String> {
    if last_dim == 0 || out_len == 0 {
        return Err("CUDA typed mixed lastdim binary dimensions must be non-zero".to_string());
    }
    let vector_len = if vector_on_rhs { rhs.len() } else { lhs.len() };
    let full_len = if vector_on_rhs { lhs.len() } else { rhs.len() };
    if vector_len != last_dim || full_len != out_len {
        return Err(format!(
            "CUDA typed mixed lastdim binary length mismatch: lhs={}, rhs={}, out_len={}, last_dim={}, vector_on_rhs={}",
            lhs.len(),
            rhs.len(),
            out_len,
            last_dim,
            vector_on_rhs
        ));
    }
    let out = alloc_f32(out_len)?;
    let status = unsafe {
        lumen_cuda_binary_typed_lastdim_device(
            lhs.handle(),
            dtype_tag(lhs_dtype)?,
            lhs_scale.unwrap_or(1.0),
            rhs.handle(),
            dtype_tag(rhs_dtype)?,
            rhs_scale.unwrap_or(1.0),
            out.handle(),
            out_len,
            last_dim,
            if vector_on_rhs { 1 } else { 0 },
            op as c_int,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}

pub fn binary_lowp_typed_lastdim_output_buffer_no_host(
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    out_len: usize,
    last_dim: usize,
    vector_on_rhs: bool,
    dtype: DType,
    op: BinaryOp,
) -> Result<CudaBuffer, String> {
    if last_dim == 0 || out_len == 0 {
        return Err("CUDA typed-out lowp lastdim binary dimensions must be non-zero".to_string());
    }
    if !matches!(dtype, DType::F16 | DType::BF16) {
        return Err("CUDA typed-out lowp lastdim binary supports only F16/BF16".to_string());
    }
    let expected_lhs = if vector_on_rhs { out_len } else { last_dim };
    let expected_rhs = if vector_on_rhs { last_dim } else { out_len };
    if lhs.len() != expected_lhs || rhs.len() != expected_rhs {
        return Err(format!(
            "CUDA typed-out lowp lastdim binary length mismatch: expected lhs={}, rhs={}, got lhs={}, rhs={}",
            expected_lhs,
            expected_rhs,
            lhs.len(),
            rhs.len()
        ));
    }
    let out = alloc_storage(out_len)?;
    let status = unsafe {
        lumen_cuda_binary_lowp_typed_lastdim_out_device(
            lhs.handle(),
            rhs.handle(),
            out.handle(),
            out_len,
            last_dim,
            if vector_on_rhs { 1 } else { 0 },
            dtype_tag(dtype)?,
            op as c_int,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
pub fn binary_typed_row_scalar_buffer_no_host(
    lhs: &CudaBuffer,
    lhs_dtype: DType,
    lhs_scale: Option<f32>,
    rhs: &CudaBuffer,
    rhs_dtype: DType,
    rhs_scale: Option<f32>,
    rows: usize,
    last_dim: usize,
    scalar_on_rhs: bool,
    op: BinaryOp,
) -> Result<CudaBuffer, String> {
    let out_len = rows
        .checked_mul(last_dim)
        .ok_or_else(|| "CUDA typed mixed row-scalar binary output length overflow".to_string())?;
    let expected_lhs = if scalar_on_rhs { out_len } else { rows };
    let expected_rhs = if scalar_on_rhs { rows } else { out_len };
    if rows == 0 || last_dim == 0 || lhs.len() != expected_lhs || rhs.len() != expected_rhs {
        return Err(format!(
            "CUDA typed mixed row-scalar binary length mismatch: expected lhs={}, rhs={}, got lhs={}, rhs={}",
            expected_lhs,
            expected_rhs,
            lhs.len(),
            rhs.len()
        ));
    }
    let out = alloc_f32(out_len)?;
    let status = unsafe {
        lumen_cuda_binary_typed_row_scalar_device(
            lhs.handle(),
            dtype_tag(lhs_dtype)?,
            lhs_scale.unwrap_or(1.0),
            rhs.handle(),
            dtype_tag(rhs_dtype)?,
            rhs_scale.unwrap_or(1.0),
            out.handle(),
            rows,
            last_dim,
            scalar_on_rhs as c_int,
            op as c_int,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
pub fn binary_lowp_typed_row_scalar_output_buffer_no_host(
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    rows: usize,
    last_dim: usize,
    scalar_on_rhs: bool,
    dtype: DType,
    op: BinaryOp,
) -> Result<CudaBuffer, String> {
    let out_len = rows
        .checked_mul(last_dim)
        .ok_or_else(|| "CUDA typed-out lowp row-scalar output length overflow".to_string())?;
    let expected_lhs = if scalar_on_rhs { out_len } else { rows };
    let expected_rhs = if scalar_on_rhs { rows } else { out_len };
    if rows == 0 || last_dim == 0 || lhs.len() != expected_lhs || rhs.len() != expected_rhs {
        return Err(format!(
            "CUDA typed-out lowp row-scalar length mismatch: expected lhs={}, rhs={}, got lhs={}, rhs={}",
            expected_lhs,
            expected_rhs,
            lhs.len(),
            rhs.len()
        ));
    }
    if !matches!(dtype, DType::F16 | DType::BF16) {
        return Err("CUDA typed-out lowp row-scalar supports only F16/BF16".to_string());
    }
    let out = alloc_storage(out_len)?;
    let status = unsafe {
        lumen_cuda_binary_lowp_typed_row_scalar_out_device(
            lhs.handle(),
            rhs.handle(),
            out.handle(),
            rows,
            last_dim,
            if scalar_on_rhs { 1 } else { 0 },
            dtype_tag(dtype)?,
            op as c_int,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
pub fn binary_i8_typed_row_scalar_output_buffer_no_host(
    lhs: &CudaBuffer,
    lhs_scale: f32,
    rhs: &CudaBuffer,
    rhs_scale: f32,
    rows: usize,
    last_dim: usize,
    scalar_on_rhs: bool,
    op: BinaryOp,
) -> Result<(CudaBuffer, f32), String> {
    let out_len = rows
        .checked_mul(last_dim)
        .ok_or_else(|| "CUDA typed-out I8 row-scalar output length overflow".to_string())?;
    let expected_lhs = if scalar_on_rhs { out_len } else { rows };
    let expected_rhs = if scalar_on_rhs { rows } else { out_len };
    if rows == 0 || last_dim == 0 || lhs.len() != expected_lhs || rhs.len() != expected_rhs {
        return Err(format!(
            "CUDA typed-out I8 row-scalar length mismatch: expected lhs={}, rhs={}, got lhs={}, rhs={}",
            expected_lhs,
            expected_rhs,
            lhs.len(),
            rhs.len()
        ));
    }
    let out = alloc_storage(out_len)?;
    let mut out_scale = 1.0f32;
    let status = unsafe {
        lumen_cuda_binary_i8_typed_row_scalar_out_device(
            lhs.handle(),
            rhs.handle(),
            lhs_scale,
            rhs_scale,
            out.handle(),
            rows,
            last_dim,
            if scalar_on_rhs { 1 } else { 0 },
            op as c_int,
            &mut out_scale as *mut f32,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok((out, out_scale))
}

#[allow(clippy::too_many_arguments)]
pub fn binary_typed_broadcast_buffer_no_host(
    lhs: &CudaBuffer,
    lhs_dtype: DType,
    lhs_scale: Option<f32>,
    rhs: &CudaBuffer,
    rhs_dtype: DType,
    rhs_scale: Option<f32>,
    lhs_shape: &[usize],
    rhs_shape: &[usize],
    out_shape: &[usize],
    op: BinaryOp,
) -> Result<CudaBuffer, String> {
    let len = out_shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| "CUDA typed mixed broadcast output length overflow".to_string())?;
    let lhs_len = lhs_shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| "CUDA typed mixed broadcast lhs length overflow".to_string())?;
    let rhs_len = rhs_shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| "CUDA typed mixed broadcast rhs length overflow".to_string())?;
    if lhs.len() != lhs_len || rhs.len() != rhs_len {
        return Err(format!(
            "CUDA typed mixed broadcast input length mismatch: expected lhs={}, rhs={}, got lhs={}, rhs={}",
            lhs_len,
            rhs_len,
            lhs.len(),
            rhs.len()
        ));
    }
    let (lhs_aligned, rhs_aligned, out_strides, lhs_strides, rhs_strides) =
        aligned_broadcast_metadata(lhs_shape, rhs_shape, out_shape)?;
    let out = alloc_f32(len)?;
    let status = unsafe {
        lumen_cuda_binary_typed_broadcast_device(
            lhs.handle(),
            dtype_tag(lhs_dtype)?,
            lhs_scale.unwrap_or(1.0),
            rhs.handle(),
            dtype_tag(rhs_dtype)?,
            rhs_scale.unwrap_or(1.0),
            out.handle(),
            out_shape.len(),
            out_strides.as_ptr(),
            lhs_aligned.as_ptr(),
            lhs_strides.as_ptr(),
            rhs_aligned.as_ptr(),
            rhs_strides.as_ptr(),
            len,
            op as c_int,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
pub fn binary_lowp_typed_broadcast_output_buffer_no_host(
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    lhs_shape: &[usize],
    rhs_shape: &[usize],
    out_shape: &[usize],
    dtype: DType,
    op: BinaryOp,
) -> Result<CudaBuffer, String> {
    let len = out_shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| "CUDA lowp typed-out broadcast output length overflow".to_string())?;
    let lhs_len = lhs_shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| "CUDA lowp typed-out broadcast lhs length overflow".to_string())?;
    let rhs_len = rhs_shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| "CUDA lowp typed-out broadcast rhs length overflow".to_string())?;
    if lhs.len() != lhs_len || rhs.len() != rhs_len {
        return Err(format!(
            "CUDA lowp typed-out broadcast input length mismatch: expected lhs={}, rhs={}, got lhs={}, rhs={}",
            lhs_len,
            rhs_len,
            lhs.len(),
            rhs.len()
        ));
    }
    if !matches!(dtype, DType::F16 | DType::BF16) {
        return Err("CUDA lowp typed-out broadcast supports only F16/BF16".to_string());
    }
    let (lhs_aligned, rhs_aligned, out_strides, lhs_strides, rhs_strides) =
        aligned_broadcast_metadata(lhs_shape, rhs_shape, out_shape)?;
    let out = alloc_storage(len)?;
    let status = unsafe {
        lumen_cuda_binary_lowp_typed_broadcast_out_device(
            lhs.handle(),
            rhs.handle(),
            out.handle(),
            out_shape.len(),
            out_strides.as_ptr(),
            lhs_aligned.as_ptr(),
            lhs_strides.as_ptr(),
            rhs_aligned.as_ptr(),
            rhs_strides.as_ptr(),
            len,
            dtype_tag(dtype)?,
            op as c_int,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
pub fn binary_i8_typed_broadcast_output_buffer_no_host(
    lhs: &CudaBuffer,
    lhs_scale: f32,
    rhs: &CudaBuffer,
    rhs_scale: f32,
    lhs_shape: &[usize],
    rhs_shape: &[usize],
    out_shape: &[usize],
    op: BinaryOp,
) -> Result<(CudaBuffer, f32), String> {
    let len = out_shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| "CUDA I8 typed-out broadcast output length overflow".to_string())?;
    let lhs_len = lhs_shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| "CUDA I8 typed-out broadcast lhs length overflow".to_string())?;
    let rhs_len = rhs_shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| "CUDA I8 typed-out broadcast rhs length overflow".to_string())?;
    if lhs.len() != lhs_len || rhs.len() != rhs_len {
        return Err(format!(
            "CUDA I8 typed-out broadcast input length mismatch: expected lhs={}, rhs={}, got lhs={}, rhs={}",
            lhs_len,
            rhs_len,
            lhs.len(),
            rhs.len()
        ));
    }
    let (lhs_aligned, rhs_aligned, out_strides, lhs_strides, rhs_strides) =
        aligned_broadcast_metadata(lhs_shape, rhs_shape, out_shape)?;
    let out = alloc_storage(len)?;
    let mut out_scale = 1.0f32;
    let status = unsafe {
        lumen_cuda_binary_i8_typed_broadcast_out_device(
            lhs.handle(),
            rhs.handle(),
            lhs_scale,
            rhs_scale,
            out.handle(),
            out_shape.len(),
            out_strides.as_ptr(),
            lhs_aligned.as_ptr(),
            lhs_strides.as_ptr(),
            rhs_aligned.as_ptr(),
            rhs_strides.as_ptr(),
            len,
            op as c_int,
            &mut out_scale as *mut f32,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok((out, out_scale))
}

#[allow(clippy::too_many_arguments)]
pub fn binary_typed_b1d_1h1_buffer_no_host(
    lhs: &CudaBuffer,
    lhs_dtype: DType,
    lhs_scale: Option<f32>,
    rhs: &CudaBuffer,
    rhs_dtype: DType,
    rhs_scale: Option<f32>,
    batch: usize,
    heads: usize,
    dim: usize,
    b1d_on_lhs: bool,
    op: BinaryOp,
) -> Result<CudaBuffer, String> {
    let b1d_len = batch
        .checked_mul(dim)
        .ok_or_else(|| "CUDA typed mixed b1d/1h1 b1d length overflow".to_string())?;
    let out_len = b1d_len
        .checked_mul(heads)
        .ok_or_else(|| "CUDA typed mixed b1d/1h1 output length overflow".to_string())?;
    let expected_lhs = if b1d_on_lhs { b1d_len } else { heads };
    let expected_rhs = if b1d_on_lhs { heads } else { b1d_len };
    if lhs.len() != expected_lhs || rhs.len() != expected_rhs {
        return Err(format!(
            "CUDA typed mixed b1d/1h1 length mismatch: expected lhs={}, rhs={}, got lhs={}, rhs={}",
            expected_lhs,
            expected_rhs,
            lhs.len(),
            rhs.len()
        ));
    }
    let out = alloc_f32(out_len)?;
    let status = unsafe {
        lumen_cuda_binary_typed_b1d_1h1_device(
            lhs.handle(),
            dtype_tag(lhs_dtype)?,
            lhs_scale.unwrap_or(1.0),
            rhs.handle(),
            dtype_tag(rhs_dtype)?,
            rhs_scale.unwrap_or(1.0),
            out.handle(),
            batch,
            heads,
            dim,
            b1d_on_lhs as c_int,
            op as c_int,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
pub fn binary_lowp_typed_b1d_1h1_output_buffer_no_host(
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    batch: usize,
    heads: usize,
    dim: usize,
    b1d_on_lhs: bool,
    dtype: DType,
    op: BinaryOp,
) -> Result<CudaBuffer, String> {
    let b1d_len = batch
        .checked_mul(dim)
        .ok_or_else(|| "CUDA typed-out lowp b1d/1h1 b1d length overflow".to_string())?;
    let out_len = b1d_len
        .checked_mul(heads)
        .ok_or_else(|| "CUDA typed-out lowp b1d/1h1 output length overflow".to_string())?;
    let expected_lhs = if b1d_on_lhs { b1d_len } else { heads };
    let expected_rhs = if b1d_on_lhs { heads } else { b1d_len };
    if batch == 0
        || heads == 0
        || dim == 0
        || lhs.len() != expected_lhs
        || rhs.len() != expected_rhs
    {
        return Err(format!(
            "CUDA typed-out lowp b1d/1h1 length mismatch: expected lhs={}, rhs={}, got lhs={}, rhs={}",
            expected_lhs,
            expected_rhs,
            lhs.len(),
            rhs.len()
        ));
    }
    if !matches!(dtype, DType::F16 | DType::BF16) {
        return Err("CUDA typed-out lowp b1d/1h1 supports only F16/BF16".to_string());
    }
    let out = alloc_storage(out_len)?;
    let status = unsafe {
        lumen_cuda_binary_lowp_typed_b1d_1h1_out_device(
            lhs.handle(),
            rhs.handle(),
            out.handle(),
            batch,
            heads,
            dim,
            if b1d_on_lhs { 1 } else { 0 },
            dtype_tag(dtype)?,
            op as c_int,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
pub fn binary_i8_typed_b1d_1h1_output_buffer_no_host(
    lhs: &CudaBuffer,
    lhs_scale: f32,
    rhs: &CudaBuffer,
    rhs_scale: f32,
    batch: usize,
    heads: usize,
    dim: usize,
    b1d_on_lhs: bool,
    op: BinaryOp,
) -> Result<(CudaBuffer, f32), String> {
    let b1d_len = batch
        .checked_mul(dim)
        .ok_or_else(|| "CUDA typed-out I8 b1d/1h1 b1d length overflow".to_string())?;
    let out_len = b1d_len
        .checked_mul(heads)
        .ok_or_else(|| "CUDA typed-out I8 b1d/1h1 output length overflow".to_string())?;
    let expected_lhs = if b1d_on_lhs { b1d_len } else { heads };
    let expected_rhs = if b1d_on_lhs { heads } else { b1d_len };
    if batch == 0
        || heads == 0
        || dim == 0
        || lhs.len() != expected_lhs
        || rhs.len() != expected_rhs
    {
        return Err(format!(
            "CUDA typed-out I8 b1d/1h1 length mismatch: expected lhs={}, rhs={}, got lhs={}, rhs={}",
            expected_lhs,
            expected_rhs,
            lhs.len(),
            rhs.len()
        ));
    }
    let out = alloc_storage(out_len)?;
    let mut out_scale = 1.0f32;
    let status = unsafe {
        lumen_cuda_binary_i8_typed_b1d_1h1_out_device(
            lhs.handle(),
            rhs.handle(),
            lhs_scale,
            rhs_scale,
            out.handle(),
            batch,
            heads,
            dim,
            if b1d_on_lhs { 1 } else { 0 },
            op as c_int,
            &mut out_scale as *mut f32,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok((out, out_scale))
}

#[allow(clippy::too_many_arguments)]
pub fn binary_typed_b1d_1hd_buffer_no_host(
    lhs: &CudaBuffer,
    lhs_dtype: DType,
    lhs_scale: Option<f32>,
    rhs: &CudaBuffer,
    rhs_dtype: DType,
    rhs_scale: Option<f32>,
    batch: usize,
    heads: usize,
    dim: usize,
    b1d_on_lhs: bool,
    op: BinaryOp,
) -> Result<CudaBuffer, String> {
    let b1d_len = batch
        .checked_mul(dim)
        .ok_or_else(|| "CUDA typed mixed b1d/1hd b1d length overflow".to_string())?;
    let hd_len = heads
        .checked_mul(dim)
        .ok_or_else(|| "CUDA typed mixed b1d/1hd hd length overflow".to_string())?;
    let out_len = b1d_len
        .checked_mul(heads)
        .ok_or_else(|| "CUDA typed mixed b1d/1hd output length overflow".to_string())?;
    let expected_lhs = if b1d_on_lhs { b1d_len } else { hd_len };
    let expected_rhs = if b1d_on_lhs { hd_len } else { b1d_len };
    if lhs.len() != expected_lhs || rhs.len() != expected_rhs {
        return Err(format!(
            "CUDA typed mixed b1d/1hd length mismatch: expected lhs={}, rhs={}, got lhs={}, rhs={}",
            expected_lhs,
            expected_rhs,
            lhs.len(),
            rhs.len()
        ));
    }
    let out = alloc_f32(out_len)?;
    let status = unsafe {
        lumen_cuda_binary_typed_b1d_1hd_device(
            lhs.handle(),
            dtype_tag(lhs_dtype)?,
            lhs_scale.unwrap_or(1.0),
            rhs.handle(),
            dtype_tag(rhs_dtype)?,
            rhs_scale.unwrap_or(1.0),
            out.handle(),
            batch,
            heads,
            dim,
            b1d_on_lhs as c_int,
            op as c_int,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
pub fn binary_lowp_typed_b1d_1hd_output_buffer_no_host(
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    batch: usize,
    heads: usize,
    dim: usize,
    b1d_on_lhs: bool,
    dtype: DType,
    op: BinaryOp,
) -> Result<CudaBuffer, String> {
    let b1d_len = batch
        .checked_mul(dim)
        .ok_or_else(|| "CUDA typed-out lowp b1d/1hd b1d length overflow".to_string())?;
    let hd_len = heads
        .checked_mul(dim)
        .ok_or_else(|| "CUDA typed-out lowp b1d/1hd hd length overflow".to_string())?;
    let out_len = b1d_len
        .checked_mul(heads)
        .ok_or_else(|| "CUDA typed-out lowp b1d/1hd output length overflow".to_string())?;
    let expected_lhs = if b1d_on_lhs { b1d_len } else { hd_len };
    let expected_rhs = if b1d_on_lhs { hd_len } else { b1d_len };
    if batch == 0
        || heads == 0
        || dim == 0
        || lhs.len() != expected_lhs
        || rhs.len() != expected_rhs
    {
        return Err(format!(
            "CUDA typed-out lowp b1d/1hd length mismatch: expected lhs={}, rhs={}, got lhs={}, rhs={}",
            expected_lhs,
            expected_rhs,
            lhs.len(),
            rhs.len()
        ));
    }
    if !matches!(dtype, DType::F16 | DType::BF16) {
        return Err("CUDA typed-out lowp b1d/1hd supports only F16/BF16".to_string());
    }
    let out = alloc_storage(out_len)?;
    let status = unsafe {
        lumen_cuda_binary_lowp_typed_b1d_1hd_out_device(
            lhs.handle(),
            rhs.handle(),
            out.handle(),
            batch,
            heads,
            dim,
            if b1d_on_lhs { 1 } else { 0 },
            dtype_tag(dtype)?,
            op as c_int,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
pub fn binary_i8_typed_b1d_1hd_output_buffer_no_host(
    lhs: &CudaBuffer,
    lhs_scale: f32,
    rhs: &CudaBuffer,
    rhs_scale: f32,
    batch: usize,
    heads: usize,
    dim: usize,
    b1d_on_lhs: bool,
    op: BinaryOp,
) -> Result<(CudaBuffer, f32), String> {
    let b1d_len = batch
        .checked_mul(dim)
        .ok_or_else(|| "CUDA typed-out I8 b1d/1hd b1d length overflow".to_string())?;
    let hd_len = heads
        .checked_mul(dim)
        .ok_or_else(|| "CUDA typed-out I8 b1d/1hd hd length overflow".to_string())?;
    let out_len = b1d_len
        .checked_mul(heads)
        .ok_or_else(|| "CUDA typed-out I8 b1d/1hd output length overflow".to_string())?;
    let expected_lhs = if b1d_on_lhs { b1d_len } else { hd_len };
    let expected_rhs = if b1d_on_lhs { hd_len } else { b1d_len };
    if batch == 0
        || heads == 0
        || dim == 0
        || lhs.len() != expected_lhs
        || rhs.len() != expected_rhs
    {
        return Err(format!(
            "CUDA typed-out I8 b1d/1hd length mismatch: expected lhs={}, rhs={}, got lhs={}, rhs={}",
            expected_lhs,
            expected_rhs,
            lhs.len(),
            rhs.len()
        ));
    }
    let out = alloc_storage(out_len)?;
    let mut out_scale = 1.0f32;
    let status = unsafe {
        lumen_cuda_binary_i8_typed_b1d_1hd_out_device(
            lhs.handle(),
            rhs.handle(),
            lhs_scale,
            rhs_scale,
            out.handle(),
            batch,
            heads,
            dim,
            if b1d_on_lhs { 1 } else { 0 },
            op as c_int,
            &mut out_scale as *mut f32,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok((out, out_scale))
}

pub fn binary_f16_host_no_host(
    lhs: &[u16],
    rhs: &[u16],
    op: BinaryOp,
) -> Result<CudaBuffer, String> {
    if lhs.len() != rhs.len() {
        return Err(format!(
            "CUDA F16 binary length mismatch: lhs={}, rhs={}",
            lhs.len(),
            rhs.len()
        ));
    }
    let out = alloc_f32(lhs.len())?;
    let status = unsafe {
        lumen_cuda_binary_f16_host_device(
            lhs.as_ptr(),
            rhs.as_ptr(),
            out.handle(),
            lhs.len(),
            op as c_int,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}

pub fn binary_f16_buffer_no_host(
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    op: BinaryOp,
) -> Result<CudaBuffer, String> {
    if lhs.len() != rhs.len() {
        return Err(format!(
            "CUDA resident F16 binary length mismatch: lhs={}, rhs={}",
            lhs.len(),
            rhs.len()
        ));
    }
    let out = alloc_f32(lhs.len())?;
    let status = unsafe {
        lumen_cuda_binary_f16_device(
            lhs.handle(),
            rhs.handle(),
            out.handle(),
            lhs.len(),
            op as c_int,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}

pub fn binary_f16_lastdim_buffer_no_host(
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    out_len: usize,
    last_dim: usize,
    vector_on_rhs: bool,
    op: BinaryOp,
) -> Result<CudaBuffer, String> {
    if last_dim == 0 {
        return Err("CUDA F16 row-broadcast last_dim must be greater than zero".to_string());
    }
    let expected_lhs = if vector_on_rhs { out_len } else { last_dim };
    let expected_rhs = if vector_on_rhs { last_dim } else { out_len };
    if lhs.len() != expected_lhs || rhs.len() != expected_rhs {
        return Err(format!(
            "CUDA F16 row-broadcast length mismatch: expected lhs={}, rhs={}, got lhs={}, rhs={}",
            expected_lhs,
            expected_rhs,
            lhs.len(),
            rhs.len()
        ));
    }
    let out = alloc_f32(out_len)?;
    let status = unsafe {
        lumen_cuda_binary_f16_lastdim_device(
            lhs.handle(),
            rhs.handle(),
            out.handle(),
            out_len,
            last_dim,
            vector_on_rhs as c_int,
            op as c_int,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}

pub fn binary_bf16_host_no_host(
    lhs: &[u16],
    rhs: &[u16],
    op: BinaryOp,
) -> Result<CudaBuffer, String> {
    if lhs.len() != rhs.len() {
        return Err(format!(
            "CUDA BF16 binary length mismatch: lhs={}, rhs={}",
            lhs.len(),
            rhs.len()
        ));
    }
    let out = alloc_f32(lhs.len())?;
    let status = unsafe {
        lumen_cuda_binary_bf16_host_device(
            lhs.as_ptr(),
            rhs.as_ptr(),
            out.handle(),
            lhs.len(),
            op as c_int,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}

pub fn binary_bf16_buffer_no_host(
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    op: BinaryOp,
) -> Result<CudaBuffer, String> {
    if lhs.len() != rhs.len() {
        return Err(format!(
            "CUDA resident BF16 binary length mismatch: lhs={}, rhs={}",
            lhs.len(),
            rhs.len()
        ));
    }
    let out = alloc_f32(lhs.len())?;
    let status = unsafe {
        lumen_cuda_binary_bf16_device(
            lhs.handle(),
            rhs.handle(),
            out.handle(),
            lhs.len(),
            op as c_int,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}

pub fn binary_bf16_lastdim_buffer_no_host(
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    out_len: usize,
    last_dim: usize,
    vector_on_rhs: bool,
    op: BinaryOp,
) -> Result<CudaBuffer, String> {
    if last_dim == 0 {
        return Err("CUDA BF16 row-broadcast last_dim must be greater than zero".to_string());
    }
    let expected_lhs = if vector_on_rhs { out_len } else { last_dim };
    let expected_rhs = if vector_on_rhs { last_dim } else { out_len };
    if lhs.len() != expected_lhs || rhs.len() != expected_rhs {
        return Err(format!(
            "CUDA BF16 row-broadcast length mismatch: expected lhs={}, rhs={}, got lhs={}, rhs={}",
            expected_lhs,
            expected_rhs,
            lhs.len(),
            rhs.len()
        ));
    }
    let out = alloc_f32(out_len)?;
    let status = unsafe {
        lumen_cuda_binary_bf16_lastdim_device(
            lhs.handle(),
            rhs.handle(),
            out.handle(),
            out_len,
            last_dim,
            vector_on_rhs as c_int,
            op as c_int,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}

pub fn binary_i8_host_no_host(
    lhs: &[i8],
    lhs_scale: f32,
    rhs: &[i8],
    rhs_scale: f32,
    op: BinaryOp,
) -> Result<CudaBuffer, String> {
    if lhs.len() != rhs.len() {
        return Err(format!(
            "CUDA I8 binary length mismatch: lhs={}, rhs={}",
            lhs.len(),
            rhs.len()
        ));
    }
    let out = alloc_f32(lhs.len())?;
    let status = unsafe {
        lumen_cuda_binary_i8_host_device(
            lhs.as_ptr(),
            rhs.as_ptr(),
            lhs_scale,
            rhs_scale,
            out.handle(),
            lhs.len(),
            op as c_int,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}

pub fn binary_i8_buffer_no_host(
    lhs: &CudaBuffer,
    lhs_scale: f32,
    rhs: &CudaBuffer,
    rhs_scale: f32,
    op: BinaryOp,
) -> Result<CudaBuffer, String> {
    if lhs.len() != rhs.len() {
        return Err(format!(
            "CUDA resident I8 binary length mismatch: lhs={}, rhs={}",
            lhs.len(),
            rhs.len()
        ));
    }
    let out = alloc_f32(lhs.len())?;
    let status = unsafe {
        lumen_cuda_binary_i8_device(
            lhs.handle(),
            rhs.handle(),
            lhs_scale,
            rhs_scale,
            out.handle(),
            lhs.len(),
            op as c_int,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}

pub fn binary_i8_typed_output_buffer_no_host(
    lhs: &CudaBuffer,
    lhs_scale: f32,
    rhs: &CudaBuffer,
    rhs_scale: f32,
    op: BinaryOp,
) -> Result<(CudaBuffer, f32), String> {
    if lhs.len() != rhs.len() {
        return Err(format!(
            "CUDA resident I8 typed-out binary length mismatch: lhs={}, rhs={}",
            lhs.len(),
            rhs.len()
        ));
    }
    let out = alloc_storage(lhs.len())?;
    let mut out_scale = 1.0f32;
    let status = unsafe {
        lumen_cuda_binary_i8_typed_out_device(
            lhs.handle(),
            rhs.handle(),
            lhs_scale,
            rhs_scale,
            out.handle(),
            lhs.len(),
            op as c_int,
            &mut out_scale as *mut f32,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok((out, out_scale))
}

pub fn binary_i8_lastdim_buffer_no_host(
    lhs: &CudaBuffer,
    lhs_scale: f32,
    rhs: &CudaBuffer,
    rhs_scale: f32,
    out_len: usize,
    last_dim: usize,
    vector_on_rhs: bool,
    op: BinaryOp,
) -> Result<CudaBuffer, String> {
    if last_dim == 0 {
        return Err("CUDA I8 row-broadcast last_dim must be greater than zero".to_string());
    }
    let expected_lhs = if vector_on_rhs { out_len } else { last_dim };
    let expected_rhs = if vector_on_rhs { last_dim } else { out_len };
    if lhs.len() != expected_lhs || rhs.len() != expected_rhs {
        return Err(format!(
            "CUDA I8 row-broadcast length mismatch: expected lhs={}, rhs={}, got lhs={}, rhs={}",
            expected_lhs,
            expected_rhs,
            lhs.len(),
            rhs.len()
        ));
    }
    let out = alloc_f32(out_len)?;
    let status = unsafe {
        lumen_cuda_binary_i8_lastdim_device(
            lhs.handle(),
            rhs.handle(),
            lhs_scale,
            rhs_scale,
            out.handle(),
            out_len,
            last_dim,
            vector_on_rhs as c_int,
            op as c_int,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}

pub fn binary_i8_typed_lastdim_output_buffer_no_host(
    lhs: &CudaBuffer,
    lhs_scale: f32,
    rhs: &CudaBuffer,
    rhs_scale: f32,
    out_len: usize,
    last_dim: usize,
    vector_on_rhs: bool,
    op: BinaryOp,
) -> Result<(CudaBuffer, f32), String> {
    if last_dim == 0 {
        return Err(
            "CUDA I8 typed-out row-broadcast last_dim must be greater than zero".to_string(),
        );
    }
    let expected_lhs = if vector_on_rhs { out_len } else { last_dim };
    let expected_rhs = if vector_on_rhs { last_dim } else { out_len };
    if lhs.len() != expected_lhs || rhs.len() != expected_rhs {
        return Err(format!(
            "CUDA I8 typed-out row-broadcast length mismatch: expected lhs={}, rhs={}, got lhs={}, rhs={}",
            expected_lhs,
            expected_rhs,
            lhs.len(),
            rhs.len()
        ));
    }
    let out = alloc_storage(out_len)?;
    let mut out_scale = 1.0f32;
    let status = unsafe {
        lumen_cuda_binary_i8_typed_lastdim_out_device(
            lhs.handle(),
            rhs.handle(),
            lhs_scale,
            rhs_scale,
            out.handle(),
            out_len,
            last_dim,
            vector_on_rhs as c_int,
            op as c_int,
            &mut out_scale as *mut f32,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok((out, out_scale))
}

pub fn mul_grad_f16_host_no_host(grad: &CudaBuffer, operand: &[u16]) -> Result<CudaBuffer, String> {
    if grad.len() != operand.len() {
        return Err(format!(
            "CUDA F16 mul grad length mismatch: grad={}, operand={}",
            grad.len(),
            operand.len()
        ));
    }
    let out = alloc_f32(operand.len())?;
    let status = unsafe {
        lumen_cuda_mul_grad_f16_host_device(
            grad.handle(),
            operand.as_ptr(),
            out.handle(),
            operand.len(),
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}

pub fn mul_grad_f16_buffer_no_host(
    grad: &CudaBuffer,
    operand: &CudaBuffer,
) -> Result<CudaBuffer, String> {
    if grad.len() != operand.len() {
        return Err(format!(
            "CUDA resident F16 mul grad length mismatch: grad={}, operand={}",
            grad.len(),
            operand.len()
        ));
    }
    let out = alloc_f32(operand.len())?;
    let status = unsafe {
        lumen_cuda_mul_grad_f16_device(grad.handle(), operand.handle(), out.handle(), operand.len())
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}

pub fn mul_grad_f16_lastdim_buffer_no_host(
    grad: &CudaBuffer,
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    out_len: usize,
    last_dim: usize,
    vector_on_rhs: bool,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    let expected_lhs = if vector_on_rhs { out_len } else { last_dim };
    let expected_rhs = if vector_on_rhs { last_dim } else { out_len };
    if grad.len() != out_len || lhs.len() != expected_lhs || rhs.len() != expected_rhs {
        return Err(format!(
            "CUDA F16 row-broadcast mul grad length mismatch: grad={}, lhs={}, rhs={}, expected grad={}, lhs={}, rhs={}",
            grad.len(),
            lhs.len(),
            rhs.len(),
            out_len,
            expected_lhs,
            expected_rhs
        ));
    }
    let grad_lhs = alloc_f32(expected_lhs)?;
    let grad_rhs = alloc_f32(expected_rhs)?;
    let status = unsafe {
        lumen_cuda_mul_grad_f16_lastdim_device(
            grad.handle(),
            lhs.handle(),
            rhs.handle(),
            grad_lhs.handle(),
            grad_rhs.handle(),
            out_len,
            last_dim,
            vector_on_rhs as c_int,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok((grad_lhs, grad_rhs))
}

pub fn mul_grad_bf16_host_no_host(
    grad: &CudaBuffer,
    operand: &[u16],
) -> Result<CudaBuffer, String> {
    if grad.len() != operand.len() {
        return Err(format!(
            "CUDA BF16 mul grad length mismatch: grad={}, operand={}",
            grad.len(),
            operand.len()
        ));
    }
    let out = alloc_f32(operand.len())?;
    let status = unsafe {
        lumen_cuda_mul_grad_bf16_host_device(
            grad.handle(),
            operand.as_ptr(),
            out.handle(),
            operand.len(),
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}

pub fn mul_grad_bf16_buffer_no_host(
    grad: &CudaBuffer,
    operand: &CudaBuffer,
) -> Result<CudaBuffer, String> {
    if grad.len() != operand.len() {
        return Err(format!(
            "CUDA resident BF16 mul grad length mismatch: grad={}, operand={}",
            grad.len(),
            operand.len()
        ));
    }
    let out = alloc_f32(operand.len())?;
    let status = unsafe {
        lumen_cuda_mul_grad_bf16_device(
            grad.handle(),
            operand.handle(),
            out.handle(),
            operand.len(),
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}

pub fn mul_grad_bf16_lastdim_buffer_no_host(
    grad: &CudaBuffer,
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    out_len: usize,
    last_dim: usize,
    vector_on_rhs: bool,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    let expected_lhs = if vector_on_rhs { out_len } else { last_dim };
    let expected_rhs = if vector_on_rhs { last_dim } else { out_len };
    if grad.len() != out_len || lhs.len() != expected_lhs || rhs.len() != expected_rhs {
        return Err(format!(
            "CUDA BF16 row-broadcast mul grad length mismatch: grad={}, lhs={}, rhs={}, expected grad={}, lhs={}, rhs={}",
            grad.len(),
            lhs.len(),
            rhs.len(),
            out_len,
            expected_lhs,
            expected_rhs
        ));
    }
    let grad_lhs = alloc_f32(expected_lhs)?;
    let grad_rhs = alloc_f32(expected_rhs)?;
    let status = unsafe {
        lumen_cuda_mul_grad_bf16_lastdim_device(
            grad.handle(),
            lhs.handle(),
            rhs.handle(),
            grad_lhs.handle(),
            grad_rhs.handle(),
            out_len,
            last_dim,
            vector_on_rhs as c_int,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok((grad_lhs, grad_rhs))
}

pub fn mul_grad_i8_host_no_host(
    grad: &CudaBuffer,
    operand: &[i8],
    scale: f32,
) -> Result<CudaBuffer, String> {
    if grad.len() != operand.len() {
        return Err(format!(
            "CUDA I8 mul grad length mismatch: grad={}, operand={}",
            grad.len(),
            operand.len()
        ));
    }
    let out = alloc_f32(operand.len())?;
    let status = unsafe {
        lumen_cuda_mul_grad_i8_host_device(
            grad.handle(),
            operand.as_ptr(),
            scale,
            out.handle(),
            operand.len(),
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}

pub fn mul_grad_i8_buffer_no_host(
    grad: &CudaBuffer,
    operand: &CudaBuffer,
    scale: f32,
) -> Result<CudaBuffer, String> {
    if grad.len() != operand.len() {
        return Err(format!(
            "CUDA resident I8 mul grad length mismatch: grad={}, operand={}",
            grad.len(),
            operand.len()
        ));
    }
    let out = alloc_f32(operand.len())?;
    let status = unsafe {
        lumen_cuda_mul_grad_i8_device(
            grad.handle(),
            operand.handle(),
            scale,
            out.handle(),
            operand.len(),
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}

pub fn mul_grad_i8_lastdim_buffer_no_host(
    grad: &CudaBuffer,
    lhs: &CudaBuffer,
    lhs_scale: f32,
    rhs: &CudaBuffer,
    rhs_scale: f32,
    out_len: usize,
    last_dim: usize,
    vector_on_rhs: bool,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    let expected_lhs = if vector_on_rhs { out_len } else { last_dim };
    let expected_rhs = if vector_on_rhs { last_dim } else { out_len };
    if grad.len() != out_len || lhs.len() != expected_lhs || rhs.len() != expected_rhs {
        return Err(format!(
            "CUDA I8 row-broadcast mul grad length mismatch: grad={}, lhs={}, rhs={}, expected grad={}, lhs={}, rhs={}",
            grad.len(),
            lhs.len(),
            rhs.len(),
            out_len,
            expected_lhs,
            expected_rhs
        ));
    }
    let grad_lhs = alloc_f32(expected_lhs)?;
    let grad_rhs = alloc_f32(expected_rhs)?;
    let status = unsafe {
        lumen_cuda_mul_grad_i8_lastdim_device(
            grad.handle(),
            lhs.handle(),
            rhs.handle(),
            lhs_scale,
            rhs_scale,
            grad_lhs.handle(),
            grad_rhs.handle(),
            out_len,
            last_dim,
            vector_on_rhs as c_int,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok((grad_lhs, grad_rhs))
}

#[allow(clippy::too_many_arguments)]
pub fn mul_grad_typed_lastdim_buffer_no_host(
    grad: &CudaBuffer,
    lhs: &CudaBuffer,
    lhs_dtype: DType,
    lhs_scale: Option<f32>,
    rhs: &CudaBuffer,
    rhs_dtype: DType,
    rhs_scale: Option<f32>,
    out_len: usize,
    last_dim: usize,
    vector_on_rhs: bool,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    let expected_lhs = if vector_on_rhs { out_len } else { last_dim };
    let expected_rhs = if vector_on_rhs { last_dim } else { out_len };
    if grad.len() != out_len || lhs.len() != expected_lhs || rhs.len() != expected_rhs {
        return Err(format!(
            "CUDA typed mixed row-broadcast mul grad length mismatch: grad={}, lhs={}, rhs={}, expected grad={}, lhs={}, rhs={}",
            grad.len(),
            lhs.len(),
            rhs.len(),
            out_len,
            expected_lhs,
            expected_rhs
        ));
    }
    let grad_lhs = alloc_f32(expected_lhs)?;
    let grad_rhs = alloc_f32(expected_rhs)?;
    let status = unsafe {
        lumen_cuda_mul_grad_typed_lastdim_device(
            grad.handle(),
            lhs.handle(),
            dtype_tag(lhs_dtype)?,
            lhs_scale.unwrap_or(1.0),
            rhs.handle(),
            dtype_tag(rhs_dtype)?,
            rhs_scale.unwrap_or(1.0),
            grad_lhs.handle(),
            grad_rhs.handle(),
            out_len,
            last_dim,
            vector_on_rhs as c_int,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok((grad_lhs, grad_rhs))
}

#[allow(clippy::too_many_arguments)]
pub fn mul_grad_typed_buffer_no_host(
    grad: &CudaBuffer,
    lhs: &CudaBuffer,
    lhs_dtype: DType,
    lhs_scale: Option<f32>,
    rhs: &CudaBuffer,
    rhs_dtype: DType,
    rhs_scale: Option<f32>,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    if grad.len() != lhs.len() || grad.len() != rhs.len() || grad.is_empty() {
        return Err(format!(
            "CUDA typed mixed same-shape mul grad length mismatch: grad={}, lhs={}, rhs={}",
            grad.len(),
            lhs.len(),
            rhs.len()
        ));
    }
    let grad_lhs = alloc_f32(grad.len())?;
    let grad_rhs = alloc_f32(grad.len())?;
    let status = unsafe {
        lumen_cuda_mul_grad_typed_device(
            grad.handle(),
            lhs.handle(),
            dtype_tag(lhs_dtype)?,
            lhs_scale.unwrap_or(1.0),
            rhs.handle(),
            dtype_tag(rhs_dtype)?,
            rhs_scale.unwrap_or(1.0),
            grad_lhs.handle(),
            grad_rhs.handle(),
            grad.len(),
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok((grad_lhs, grad_rhs))
}

#[allow(clippy::too_many_arguments)]
pub fn mul_grad_typed_row_scalar_buffer_no_host(
    grad: &CudaBuffer,
    lhs: &CudaBuffer,
    lhs_dtype: DType,
    lhs_scale: Option<f32>,
    rhs: &CudaBuffer,
    rhs_dtype: DType,
    rhs_scale: Option<f32>,
    rows: usize,
    last_dim: usize,
    scalar_on_rhs: bool,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    let out_len = rows
        .checked_mul(last_dim)
        .ok_or_else(|| "CUDA typed mixed row-scalar mul grad output length overflow".to_string())?;
    let expected_lhs = if scalar_on_rhs { out_len } else { rows };
    let expected_rhs = if scalar_on_rhs { rows } else { out_len };
    if grad.len() != out_len || lhs.len() != expected_lhs || rhs.len() != expected_rhs {
        return Err(format!(
            "CUDA typed mixed row-scalar mul grad length mismatch: expected grad={}, lhs={}, rhs={}, got grad={}, lhs={}, rhs={}",
            out_len,
            expected_lhs,
            expected_rhs,
            grad.len(),
            lhs.len(),
            rhs.len()
        ));
    }
    let grad_lhs = alloc_f32(expected_lhs)?;
    let grad_rhs = alloc_f32(expected_rhs)?;
    let status = unsafe {
        lumen_cuda_mul_grad_typed_row_scalar_device(
            grad.handle(),
            lhs.handle(),
            dtype_tag(lhs_dtype)?,
            lhs_scale.unwrap_or(1.0),
            rhs.handle(),
            dtype_tag(rhs_dtype)?,
            rhs_scale.unwrap_or(1.0),
            grad_lhs.handle(),
            grad_rhs.handle(),
            rows,
            last_dim,
            scalar_on_rhs as c_int,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok((grad_lhs, grad_rhs))
}

#[allow(clippy::too_many_arguments)]
pub fn mul_grad_typed_broadcast_buffer_no_host(
    grad: &CudaBuffer,
    lhs: &CudaBuffer,
    lhs_dtype: DType,
    lhs_scale: Option<f32>,
    rhs: &CudaBuffer,
    rhs_dtype: DType,
    rhs_scale: Option<f32>,
    lhs_shape: &[usize],
    rhs_shape: &[usize],
    out_shape: &[usize],
) -> Result<(CudaBuffer, CudaBuffer), String> {
    let out_len = out_shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| "CUDA typed mixed broadcast mul grad output length overflow".to_string())?;
    let lhs_len = lhs_shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| "CUDA typed mixed broadcast mul grad lhs length overflow".to_string())?;
    let rhs_len = rhs_shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| "CUDA typed mixed broadcast mul grad rhs length overflow".to_string())?;
    if grad.len() != out_len || lhs.len() != lhs_len || rhs.len() != rhs_len {
        return Err(format!(
            "CUDA typed mixed broadcast mul grad length mismatch: expected grad={}, lhs={}, rhs={}, got grad={}, lhs={}, rhs={}",
            out_len,
            lhs_len,
            rhs_len,
            grad.len(),
            lhs.len(),
            rhs.len()
        ));
    }
    let (lhs_aligned, rhs_aligned, out_strides, lhs_strides, rhs_strides) =
        aligned_broadcast_metadata(lhs_shape, rhs_shape, out_shape)?;
    let grad_lhs = alloc_f32(lhs_len)?;
    let grad_rhs = alloc_f32(rhs_len)?;
    let status = unsafe {
        lumen_cuda_mul_grad_typed_broadcast_device(
            grad.handle(),
            lhs.handle(),
            dtype_tag(lhs_dtype)?,
            lhs_scale.unwrap_or(1.0),
            rhs.handle(),
            dtype_tag(rhs_dtype)?,
            rhs_scale.unwrap_or(1.0),
            grad_lhs.handle(),
            grad_rhs.handle(),
            out_shape.len(),
            out_strides.as_ptr(),
            lhs_aligned.as_ptr(),
            lhs_strides.as_ptr(),
            rhs_aligned.as_ptr(),
            rhs_strides.as_ptr(),
            out_len,
            lhs_len,
            rhs_len,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok((grad_lhs, grad_rhs))
}

#[allow(clippy::too_many_arguments)]
pub fn mul_grad_typed_b1d_1h1_buffer_no_host(
    grad: &CudaBuffer,
    lhs: &CudaBuffer,
    lhs_dtype: DType,
    lhs_scale: Option<f32>,
    rhs: &CudaBuffer,
    rhs_dtype: DType,
    rhs_scale: Option<f32>,
    batch: usize,
    heads: usize,
    dim: usize,
    b1d_on_lhs: bool,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    let b1d_len = batch
        .checked_mul(dim)
        .ok_or_else(|| "CUDA typed mixed b1d/1h1 mul grad b1d length overflow".to_string())?;
    let out_len = b1d_len
        .checked_mul(heads)
        .ok_or_else(|| "CUDA typed mixed b1d/1h1 mul grad output length overflow".to_string())?;
    let expected_lhs = if b1d_on_lhs { b1d_len } else { heads };
    let expected_rhs = if b1d_on_lhs { heads } else { b1d_len };
    if grad.len() != out_len || lhs.len() != expected_lhs || rhs.len() != expected_rhs {
        return Err(format!(
            "CUDA typed mixed b1d/1h1 mul grad length mismatch: expected grad={}, lhs={}, rhs={}, got grad={}, lhs={}, rhs={}",
            out_len,
            expected_lhs,
            expected_rhs,
            grad.len(),
            lhs.len(),
            rhs.len()
        ));
    }
    let grad_lhs = alloc_f32(expected_lhs)?;
    let grad_rhs = alloc_f32(expected_rhs)?;
    let status = unsafe {
        lumen_cuda_mul_grad_typed_b1d_1h1_device(
            grad.handle(),
            lhs.handle(),
            dtype_tag(lhs_dtype)?,
            lhs_scale.unwrap_or(1.0),
            rhs.handle(),
            dtype_tag(rhs_dtype)?,
            rhs_scale.unwrap_or(1.0),
            grad_lhs.handle(),
            grad_rhs.handle(),
            batch,
            heads,
            dim,
            b1d_on_lhs as c_int,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok((grad_lhs, grad_rhs))
}

#[allow(clippy::too_many_arguments)]
pub fn mul_grad_typed_b1d_1hd_buffer_no_host(
    grad: &CudaBuffer,
    lhs: &CudaBuffer,
    lhs_dtype: DType,
    lhs_scale: Option<f32>,
    rhs: &CudaBuffer,
    rhs_dtype: DType,
    rhs_scale: Option<f32>,
    batch: usize,
    heads: usize,
    dim: usize,
    b1d_on_lhs: bool,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    let b1d_len = batch
        .checked_mul(dim)
        .ok_or_else(|| "CUDA typed mixed b1d/1hd mul grad b1d length overflow".to_string())?;
    let hd_len = heads
        .checked_mul(dim)
        .ok_or_else(|| "CUDA typed mixed b1d/1hd mul grad hd length overflow".to_string())?;
    let out_len = b1d_len
        .checked_mul(heads)
        .ok_or_else(|| "CUDA typed mixed b1d/1hd mul grad output length overflow".to_string())?;
    let expected_lhs = if b1d_on_lhs { b1d_len } else { hd_len };
    let expected_rhs = if b1d_on_lhs { hd_len } else { b1d_len };
    if grad.len() != out_len || lhs.len() != expected_lhs || rhs.len() != expected_rhs {
        return Err(format!(
            "CUDA typed mixed b1d/1hd mul grad length mismatch: expected grad={}, lhs={}, rhs={}, got grad={}, lhs={}, rhs={}",
            out_len,
            expected_lhs,
            expected_rhs,
            grad.len(),
            lhs.len(),
            rhs.len()
        ));
    }
    let grad_lhs = alloc_f32(expected_lhs)?;
    let grad_rhs = alloc_f32(expected_rhs)?;
    let status = unsafe {
        lumen_cuda_mul_grad_typed_b1d_1hd_device(
            grad.handle(),
            lhs.handle(),
            dtype_tag(lhs_dtype)?,
            lhs_scale.unwrap_or(1.0),
            rhs.handle(),
            dtype_tag(rhs_dtype)?,
            rhs_scale.unwrap_or(1.0),
            grad_lhs.handle(),
            grad_rhs.handle(),
            batch,
            heads,
            dim,
            b1d_on_lhs as c_int,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok((grad_lhs, grad_rhs))
}

#[allow(clippy::too_many_arguments)]
pub fn mul_grad_typed_scalar_buffer_no_host(
    grad: &CudaBuffer,
    lhs: &CudaBuffer,
    lhs_dtype: DType,
    lhs_scale: Option<f32>,
    rhs: &CudaBuffer,
    rhs_dtype: DType,
    rhs_scale: Option<f32>,
    out_len: usize,
    scalar_on_rhs: bool,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    let expected_lhs = if scalar_on_rhs { out_len } else { 1 };
    let expected_rhs = if scalar_on_rhs { 1 } else { out_len };
    if grad.len() != out_len || lhs.len() != expected_lhs || rhs.len() != expected_rhs {
        return Err(format!(
            "CUDA typed mixed scalar-broadcast mul grad length mismatch: expected grad={}, lhs={}, rhs={}, got grad={}, lhs={}, rhs={}",
            out_len,
            expected_lhs,
            expected_rhs,
            grad.len(),
            lhs.len(),
            rhs.len()
        ));
    }
    let grad_lhs = alloc_f32(expected_lhs)?;
    let grad_rhs = alloc_f32(expected_rhs)?;
    let status = unsafe {
        lumen_cuda_mul_grad_typed_scalar_device(
            grad.handle(),
            lhs.handle(),
            dtype_tag(lhs_dtype)?,
            lhs_scale.unwrap_or(1.0),
            rhs.handle(),
            dtype_tag(rhs_dtype)?,
            rhs_scale.unwrap_or(1.0),
            grad_lhs.handle(),
            grad_rhs.handle(),
            out_len,
            scalar_on_rhs as c_int,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok((grad_lhs, grad_rhs))
}

pub fn binary_backward_f32(
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    grad: &CudaBuffer,
    op: BinaryOp,
) -> Result<CudaTwoHostBuffers, String> {
    let (grad_lhs, grad_rhs) = binary_backward_f32_buffers(lhs, rhs, grad, op)?;
    let grad_lhs_host = download_f32(&grad_lhs)?;
    let grad_rhs_host = download_f32(&grad_rhs)?;
    Ok(((grad_lhs, grad_lhs_host), (grad_rhs, grad_rhs_host)))
}

pub fn binary_backward_f32_buffers(
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    grad: &CudaBuffer,
    op: BinaryOp,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    if lhs.len() != rhs.len() || lhs.len() != grad.len() {
        return Err(format!(
            "CUDA binary backward length mismatch: lhs={}, rhs={}, grad={}",
            lhs.len(),
            rhs.len(),
            grad.len()
        ));
    }
    let grad_lhs = alloc_f32(lhs.len())?;
    let grad_rhs = alloc_f32(lhs.len())?;
    let status = unsafe {
        lumen_cuda_binary_backward_f32_device(
            lhs.handle(),
            rhs.handle(),
            grad.handle(),
            grad_lhs.handle(),
            grad_rhs.handle(),
            lhs.len(),
            op as c_int,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok((grad_lhs, grad_rhs))
}

pub fn add_sub_backward_f32_buffers(
    grad: &CudaBuffer,
    len: usize,
    op: BinaryOp,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    if !matches!(op, BinaryOp::Add | BinaryOp::Sub) {
        return Err("CUDA add/sub backward only supports Add/Sub".to_string());
    }
    if grad.len() != len {
        return Err(format!(
            "CUDA add/sub backward length mismatch: grad={}, len={}",
            grad.len(),
            len
        ));
    }
    let grad_lhs = alloc_f32(len)?;
    let grad_rhs = alloc_f32(len)?;
    let status = unsafe {
        lumen_cuda_add_sub_backward_f32_device(
            grad.handle(),
            grad_lhs.handle(),
            grad_rhs.handle(),
            len,
            op as c_int,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok((grad_lhs, grad_rhs))
}

pub fn add_sub_backward_lastdim_f32_buffers(
    grad: &CudaBuffer,
    out_len: usize,
    last_dim: usize,
    vector_on_rhs: bool,
    op: BinaryOp,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    if !matches!(op, BinaryOp::Add | BinaryOp::Sub) {
        return Err("CUDA add/sub row-broadcast backward only supports Add/Sub".to_string());
    }
    if last_dim == 0 {
        return Err(
            "CUDA add/sub row-broadcast backward last_dim must be greater than zero".to_string(),
        );
    }
    if grad.len() != out_len || !out_len.is_multiple_of(last_dim) {
        return Err(format!(
            "CUDA add/sub row-broadcast backward length mismatch: grad={}, out_len={}, last_dim={}",
            grad.len(),
            out_len,
            last_dim
        ));
    }
    let grad_lhs = alloc_f32(if vector_on_rhs { out_len } else { last_dim })?;
    let grad_rhs = alloc_f32(if vector_on_rhs { last_dim } else { out_len })?;
    let status = unsafe {
        lumen_cuda_add_sub_backward_lastdim_f32_device(
            grad.handle(),
            grad_lhs.handle(),
            grad_rhs.handle(),
            out_len,
            last_dim,
            vector_on_rhs as c_int,
            op as c_int,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok((grad_lhs, grad_rhs))
}

pub fn add_sub_backward_scalar_f32_buffers(
    grad: &CudaBuffer,
    out_len: usize,
    scalar_on_rhs: bool,
    op: BinaryOp,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    if !matches!(op, BinaryOp::Add | BinaryOp::Sub) {
        return Err("CUDA add/sub scalar-broadcast backward only supports Add/Sub".to_string());
    }
    if grad.len() != out_len || out_len == 0 {
        return Err(format!(
            "CUDA add/sub scalar-broadcast backward length mismatch: grad={}, out_len={}",
            grad.len(),
            out_len
        ));
    }
    let grad_lhs = alloc_f32(if scalar_on_rhs { out_len } else { 1 })?;
    let grad_rhs = alloc_f32(if scalar_on_rhs { 1 } else { out_len })?;
    let status = unsafe {
        lumen_cuda_add_sub_backward_scalar_f32_device(
            grad.handle(),
            grad_lhs.handle(),
            grad_rhs.handle(),
            out_len,
            scalar_on_rhs as c_int,
            op as c_int,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok((grad_lhs, grad_rhs))
}

pub fn add_sub_backward_row_scalar_f32_buffers(
    grad: &CudaBuffer,
    rows: usize,
    last_dim: usize,
    scalar_on_rhs: bool,
    op: BinaryOp,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    let out_len = rows
        .checked_mul(last_dim)
        .ok_or_else(|| "CUDA add/sub row-scalar backward output length overflow".to_string())?;
    if grad.len() != out_len {
        return Err(format!(
            "CUDA add/sub row-scalar backward length mismatch: expected grad={}, got {}",
            out_len,
            grad.len()
        ));
    }
    let grad_lhs = alloc_f32(if scalar_on_rhs { out_len } else { rows })?;
    let grad_rhs = alloc_f32(if scalar_on_rhs { rows } else { out_len })?;
    let status = unsafe {
        lumen_cuda_add_sub_backward_row_scalar_f32_device(
            grad.handle(),
            grad_lhs.handle(),
            grad_rhs.handle(),
            rows,
            last_dim,
            scalar_on_rhs as c_int,
            op as c_int,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok((grad_lhs, grad_rhs))
}

pub fn add_sub_backward_b1d_1h1_f32_buffers(
    grad: &CudaBuffer,
    batch: usize,
    heads: usize,
    dim: usize,
    b1d_on_lhs: bool,
    op: BinaryOp,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    if !matches!(op, BinaryOp::Add | BinaryOp::Sub) {
        return Err("CUDA add/sub b1d/1h1 backward only supports Add/Sub".to_string());
    }
    let b1d_len = batch
        .checked_mul(dim)
        .ok_or_else(|| "CUDA add/sub b1d/1h1 backward b1d length overflow".to_string())?;
    let out_len = b1d_len
        .checked_mul(heads)
        .ok_or_else(|| "CUDA add/sub b1d/1h1 backward output length overflow".to_string())?;
    if grad.len() != out_len {
        return Err(format!(
            "CUDA add/sub b1d/1h1 backward length mismatch: grad={}, out_len={}",
            grad.len(),
            out_len
        ));
    }
    let grad_lhs = alloc_f32(if b1d_on_lhs { b1d_len } else { heads })?;
    let grad_rhs = alloc_f32(if b1d_on_lhs { heads } else { b1d_len })?;
    let status = unsafe {
        lumen_cuda_add_sub_backward_b1d_1h1_f32_device(
            grad.handle(),
            grad_lhs.handle(),
            grad_rhs.handle(),
            batch,
            heads,
            dim,
            b1d_on_lhs as c_int,
            op as c_int,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok((grad_lhs, grad_rhs))
}

pub fn add_sub_backward_b1d_1hd_f32_buffers(
    grad: &CudaBuffer,
    batch: usize,
    heads: usize,
    dim: usize,
    b1d_on_lhs: bool,
    op: BinaryOp,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    if !matches!(op, BinaryOp::Add | BinaryOp::Sub) {
        return Err("CUDA add/sub b1d/1hd backward only supports Add/Sub".to_string());
    }
    let b1d_len = batch
        .checked_mul(dim)
        .ok_or_else(|| "CUDA add/sub b1d/1hd backward b1d length overflow".to_string())?;
    let hd_len = heads
        .checked_mul(dim)
        .ok_or_else(|| "CUDA add/sub b1d/1hd backward hd length overflow".to_string())?;
    let out_len = b1d_len
        .checked_mul(heads)
        .ok_or_else(|| "CUDA add/sub b1d/1hd backward output length overflow".to_string())?;
    if grad.len() != out_len {
        return Err(format!(
            "CUDA add/sub b1d/1hd backward length mismatch: grad={}, out_len={}",
            grad.len(),
            out_len
        ));
    }
    let grad_lhs = alloc_f32(if b1d_on_lhs { b1d_len } else { hd_len })?;
    let grad_rhs = alloc_f32(if b1d_on_lhs { hd_len } else { b1d_len })?;
    let status = unsafe {
        lumen_cuda_add_sub_backward_b1d_1hd_f32_device(
            grad.handle(),
            grad_lhs.handle(),
            grad_rhs.handle(),
            batch,
            heads,
            dim,
            b1d_on_lhs as c_int,
            op as c_int,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok((grad_lhs, grad_rhs))
}

pub fn add_sub_broadcast_backward_f32_buffers(
    grad: &CudaBuffer,
    lhs_shape: &[usize],
    rhs_shape: &[usize],
    out_shape: &[usize],
    op: BinaryOp,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    if !matches!(op, BinaryOp::Add | BinaryOp::Sub) {
        return Err("CUDA add/sub broadcast backward only supports Add/Sub".to_string());
    }
    let out_len = out_shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| "CUDA add/sub broadcast backward output length overflow".to_string())?;
    let lhs_len = lhs_shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| "CUDA add/sub broadcast backward lhs length overflow".to_string())?;
    let rhs_len = rhs_shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| "CUDA add/sub broadcast backward rhs length overflow".to_string())?;
    if grad.len() != out_len {
        return Err(format!(
            "CUDA add/sub broadcast backward length mismatch: grad={}, out_len={}",
            grad.len(),
            out_len
        ));
    }
    let (lhs_aligned, rhs_aligned, out_strides, lhs_strides, rhs_strides) =
        aligned_broadcast_metadata(lhs_shape, rhs_shape, out_shape)?;
    let grad_lhs = alloc_f32(lhs_len)?;
    let grad_rhs = alloc_f32(rhs_len)?;
    let status = unsafe {
        lumen_cuda_add_sub_broadcast_backward_f32_device(
            grad.handle(),
            grad_lhs.handle(),
            grad_rhs.handle(),
            out_shape.len(),
            out_strides.as_ptr(),
            lhs_aligned.as_ptr(),
            lhs_strides.as_ptr(),
            rhs_aligned.as_ptr(),
            rhs_strides.as_ptr(),
            out_len,
            lhs_len,
            rhs_len,
            op as c_int,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok((grad_lhs, grad_rhs))
}

pub fn binary_broadcast_f32(
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    lhs_shape: &[usize],
    rhs_shape: &[usize],
    out_shape: &[usize],
    op: BinaryOp,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    let out = binary_broadcast_f32_buffer(lhs, rhs, lhs_shape, rhs_shape, out_shape, op)?;
    let host = download_f32(&out)?;
    Ok((out, host))
}

pub fn binary_broadcast_f32_buffer(
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    lhs_shape: &[usize],
    rhs_shape: &[usize],
    out_shape: &[usize],
    op: BinaryOp,
) -> Result<CudaBuffer, String> {
    let len = out_shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| "CUDA binary broadcast output length overflow".to_string())?;
    let lhs_len = lhs_shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| "CUDA binary broadcast lhs length overflow".to_string())?;
    let rhs_len = rhs_shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| "CUDA binary broadcast rhs length overflow".to_string())?;
    if lhs.len() != lhs_len || rhs.len() != rhs_len {
        return Err(format!(
            "CUDA binary broadcast input length mismatch: expected lhs={}, rhs={}, got lhs={}, rhs={}",
            lhs_len,
            rhs_len,
            lhs.len(),
            rhs.len()
        ));
    }
    let (lhs_aligned, rhs_aligned, out_strides, lhs_strides, rhs_strides) =
        aligned_broadcast_metadata(lhs_shape, rhs_shape, out_shape)?;
    let out = alloc_f32(len)?;
    let status = unsafe {
        lumen_cuda_binary_broadcast_f32_device(
            lhs.handle(),
            rhs.handle(),
            out.handle(),
            out_shape.len(),
            out_shape.as_ptr(),
            out_strides.as_ptr(),
            lhs_aligned.as_ptr(),
            lhs_strides.as_ptr(),
            rhs_aligned.as_ptr(),
            rhs_strides.as_ptr(),
            len,
            op as c_int,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}

pub fn binary_broadcast_backward_f32(
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    grad: &CudaBuffer,
    lhs_shape: &[usize],
    rhs_shape: &[usize],
    out_shape: &[usize],
    op: BinaryOp,
) -> Result<CudaTwoHostBuffers, String> {
    let (grad_lhs, grad_rhs) =
        binary_broadcast_backward_f32_buffers(lhs, rhs, grad, lhs_shape, rhs_shape, out_shape, op)?;
    let grad_lhs_host = download_f32(&grad_lhs)?;
    let grad_rhs_host = download_f32(&grad_rhs)?;
    Ok(((grad_lhs, grad_lhs_host), (grad_rhs, grad_rhs_host)))
}

pub fn binary_broadcast_backward_f32_buffers(
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    grad: &CudaBuffer,
    lhs_shape: &[usize],
    rhs_shape: &[usize],
    out_shape: &[usize],
    op: BinaryOp,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    let out_len = out_shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| "CUDA binary broadcast backward output length overflow".to_string())?;
    let lhs_len = lhs_shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| "CUDA binary broadcast backward lhs length overflow".to_string())?;
    let rhs_len = rhs_shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| "CUDA binary broadcast backward rhs length overflow".to_string())?;
    if lhs.len() != lhs_len || rhs.len() != rhs_len || grad.len() != out_len {
        return Err(format!(
            "CUDA binary broadcast backward length mismatch: expected lhs={}, rhs={}, grad={}, got lhs={}, rhs={}, grad={}",
            lhs_len,
            rhs_len,
            out_len,
            lhs.len(),
            rhs.len(),
            grad.len()
        ));
    }
    let (lhs_aligned, rhs_aligned, out_strides, lhs_strides, rhs_strides) =
        aligned_broadcast_metadata(lhs_shape, rhs_shape, out_shape)?;
    let grad_lhs = alloc_f32(lhs_len)?;
    let grad_rhs = alloc_f32(rhs_len)?;
    let status = unsafe {
        lumen_cuda_binary_broadcast_backward_f32_device(
            lhs.handle(),
            rhs.handle(),
            grad.handle(),
            grad_lhs.handle(),
            grad_rhs.handle(),
            out_shape.len(),
            out_shape.as_ptr(),
            out_strides.as_ptr(),
            lhs_aligned.as_ptr(),
            lhs_strides.as_ptr(),
            rhs_aligned.as_ptr(),
            rhs_strides.as_ptr(),
            out_len,
            lhs_len,
            rhs_len,
            op as c_int,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok((grad_lhs, grad_rhs))
}
