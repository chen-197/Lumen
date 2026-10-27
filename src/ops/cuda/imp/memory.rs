use super::*;

pub fn is_available() -> bool {
    unsafe { lumen_cuda_is_available() == 1 }
}

pub fn synchronize() -> Result<(), String> {
    let status = unsafe { lumen_cuda_synchronize() };
    if status == 0 {
        Ok(())
    } else {
        Err(last_error_message())
    }
}

pub fn release_cached_memory() -> Result<(), String> {
    let status = unsafe { lumen_cuda_release_cached_memory() };
    if status == 0 {
        Ok(())
    } else {
        Err(last_error_message())
    }
}

pub fn alloc_f32(len: usize) -> Result<CudaBuffer, String> {
    let mut handle = 0u64;
    let status = unsafe { lumen_cuda_alloc_f32(len, &mut handle as *mut u64) };
    if status == 0 {
        Ok(CudaBuffer::from_raw(handle, len))
    } else {
        Err(last_error_message())
    }
}

pub fn alloc_storage(len: usize) -> Result<CudaBuffer, String> {
    alloc_f32(len)
}

pub fn upload_f32(src: &[f32]) -> Result<CudaBuffer, String> {
    let buffer = alloc_f32(src.len())?;
    let status = unsafe { lumen_cuda_upload_f32(buffer.handle(), src.as_ptr(), src.len()) };
    if status == 0 {
        Ok(buffer)
    } else {
        Err(last_error_message())
    }
}

pub fn upload_u16_storage(src: &[u16]) -> Result<CudaBuffer, String> {
    let buffer = alloc_f32(src.len())?;
    let status = unsafe { lumen_cuda_upload_u16(buffer.handle(), src.as_ptr(), src.len()) };
    if status == 0 {
        Ok(buffer)
    } else {
        Err(last_error_message())
    }
}

pub fn upload_i8_storage(src: &[i8]) -> Result<CudaBuffer, String> {
    let buffer = alloc_f32(src.len())?;
    let status = unsafe { lumen_cuda_upload_i8(buffer.handle(), src.as_ptr(), src.len()) };
    if status == 0 {
        Ok(buffer)
    } else {
        Err(last_error_message())
    }
}

pub fn download_f32(buffer: &CudaBuffer) -> Result<Vec<f32>, String> {
    let mut out = vec![0.0f32; buffer.len()];
    let status =
        unsafe { lumen_cuda_download_f32(buffer.handle(), out.as_mut_ptr(), buffer.len()) };
    if status == 0 {
        Ok(out)
    } else {
        Err(last_error_message())
    }
}

pub fn download_u16_storage(buffer: &CudaBuffer) -> Result<Vec<u16>, String> {
    let mut out = vec![0u16; buffer.len()];
    let status =
        unsafe { lumen_cuda_download_u16(buffer.handle(), out.as_mut_ptr(), buffer.len()) };
    if status == 0 {
        Ok(out)
    } else {
        Err(last_error_message())
    }
}

pub fn download_i8_storage(buffer: &CudaBuffer) -> Result<Vec<i8>, String> {
    let mut out = vec![0i8; buffer.len()];
    let status = unsafe { lumen_cuda_download_i8(buffer.handle(), out.as_mut_ptr(), buffer.len()) };
    if status == 0 {
        Ok(out)
    } else {
        Err(last_error_message())
    }
}

pub fn download_f32_offset(
    buffer: &CudaBuffer,
    offset: usize,
    len: usize,
) -> Result<Vec<f32>, String> {
    if offset > buffer.len() || len > buffer.len().saturating_sub(offset) {
        return Err(format!(
            "CUDA download offset out of bounds: offset={}, len={}, buffer_len={}",
            offset,
            len,
            buffer.len()
        ));
    }
    let mut out = vec![0.0f32; len];
    let status =
        unsafe { lumen_cuda_download_f32_offset(buffer.handle(), out.as_mut_ptr(), offset, len) };
    if status == 0 {
        Ok(out)
    } else {
        Err(last_error_message())
    }
}

pub(super) fn validate_i8_scale_product(
    lhs_scale: f32,
    rhs_scale: f32,
    label: &str,
) -> Result<(), String> {
    if !lhs_scale.is_finite() || lhs_scale <= 0.0 {
        return Err(format!("CUDA {label} lhs scale must be finite and > 0"));
    }
    if !rhs_scale.is_finite() || rhs_scale <= 0.0 {
        return Err(format!("CUDA {label} rhs scale must be finite and > 0"));
    }
    if !(lhs_scale * rhs_scale).is_finite() {
        return Err(format!("CUDA {label} combined scale overflow"));
    }
    Ok(())
}

pub fn upload_f32_offset(buffer: &CudaBuffer, offset: usize, src: &[f32]) -> Result<(), String> {
    if offset > buffer.len() || src.len() > buffer.len() - offset {
        return Err(format!(
            "CUDA upload offset out of bounds: offset={}, len={}, buffer_len={}",
            offset,
            src.len(),
            buffer.len()
        ));
    }
    let status =
        unsafe { lumen_cuda_upload_f32_offset(buffer.handle(), src.as_ptr(), offset, src.len()) };
    if status == 0 {
        Ok(())
    } else {
        Err(last_error_message())
    }
}

pub fn copy_f32_offset(
    dst: &CudaBuffer,
    dst_offset: usize,
    src: &CudaBuffer,
    src_offset: usize,
    len: usize,
) -> Result<(), String> {
    if dst_offset > dst.len() || len > dst.len().saturating_sub(dst_offset) {
        return Err(format!(
            "CUDA copy dst out of bounds: dst_offset={}, len={}, dst_len={}",
            dst_offset,
            len,
            dst.len()
        ));
    }
    if src_offset > src.len() || len > src.len().saturating_sub(src_offset) {
        return Err(format!(
            "CUDA copy src out of bounds: src_offset={}, len={}, src_len={}",
            src_offset,
            len,
            src.len()
        ));
    }
    let status = unsafe {
        lumen_cuda_copy_f32_offset(dst.handle(), dst_offset, src.handle(), src_offset, len)
    };
    if status == 0 {
        Ok(())
    } else {
        Err(last_error_message())
    }
}

pub fn free_f32(handle: u64, len: usize) {
    unsafe { lumen_cuda_free_f32(handle, len) };
}

pub fn fill_scalar_f32(len: usize, value: f32) -> Result<(CudaBuffer, Vec<f32>), String> {
    let out = alloc_f32(len)?;
    let status = unsafe { lumen_cuda_fill_scalar_f32_device(out.handle(), len, value) };
    if status != 0 {
        return Err(last_error_message());
    }
    let host = download_f32(&out)?;
    Ok((out, host))
}

pub fn fill_scalar_f32_buffer(len: usize, value: f32) -> Result<CudaBuffer, String> {
    let out = alloc_f32(len)?;
    let status = unsafe { lumen_cuda_fill_scalar_f32_device(out.handle(), len, value) };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}

pub fn add_inplace_f32(dst: &CudaBuffer, src: &CudaBuffer) -> Result<(), String> {
    if dst.len() != src.len() {
        return Err(format!(
            "CUDA add_inplace length mismatch: dst={}, src={}",
            dst.len(),
            src.len()
        ));
    }
    let status =
        unsafe { lumen_cuda_add_inplace_f32_device(dst.handle(), src.handle(), dst.len()) };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(())
}

pub fn quantize_f32_storage_no_host(
    param: &CudaBuffer,
    dtype: crate::precision::DType,
    scale: Option<f32>,
) -> Result<(), String> {
    let dtype_id = match dtype {
        crate::precision::DType::F32 => return Ok(()),
        crate::precision::DType::F16 => 1,
        crate::precision::DType::BF16 => 2,
        crate::precision::DType::I8 => 3,
    };
    let scale = scale.unwrap_or(1.0);
    let status = unsafe {
        lumen_cuda_quantize_f32_storage_device(param.handle(), param.len(), dtype_id, scale)
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(())
}

pub fn quantize_f32_to_i8_dynamic_no_host(input: &CudaBuffer) -> Result<(CudaBuffer, f32), String> {
    let out = alloc_storage(input.len())?;
    let mut scale = 1.0f32;
    let status = unsafe {
        lumen_cuda_quantize_f32_to_i8_dynamic_device(
            input.handle(),
            out.handle(),
            input.len(),
            &mut scale as *mut f32,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok((out, scale))
}

pub fn f32_to_lowp_storage_no_host(
    input: &CudaBuffer,
    dtype: crate::precision::DType,
) -> Result<CudaBuffer, String> {
    if !matches!(
        dtype,
        crate::precision::DType::F16 | crate::precision::DType::BF16
    ) {
        return Err(format!(
            "CUDA f32 to lowp storage expects F16 or BF16 dtype, got {:?}",
            dtype
        ));
    }
    let out = alloc_storage(input.len())?;
    let status = unsafe {
        lumen_cuda_f32_to_lowp_storage_device(
            input.handle(),
            out.handle(),
            input.len(),
            dtype_tag(dtype)?,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}

pub fn f32_to_i8_storage_no_host(input: &CudaBuffer, scale: f32) -> Result<CudaBuffer, String> {
    if !(scale.is_finite() && scale > 0.0) {
        return Err(format!(
            "CUDA f32 to i8 storage expects finite positive scale, got {scale}"
        ));
    }
    let out = alloc_storage(input.len())?;
    let status = unsafe {
        lumen_cuda_f32_to_i8_storage_device(input.handle(), out.handle(), input.len(), scale)
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}
