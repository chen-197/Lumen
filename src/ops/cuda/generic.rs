//! Resident F32 primitives shared by architecture-independent layers.
use super::{CudaBuffer, alloc_f32};
use std::ffi::{CStr, c_char, c_int};

unsafe extern "C" {
    fn lumen_cuda_last_error_message() -> *const c_char;
    fn lumen_cuda_layer_norm(
        input: u64,
        weight: u64,
        bias: u64,
        output: u64,
        xhat: u64,
        inverse: u64,
        rows: usize,
        dim: usize,
        eps: f32,
    ) -> c_int;
    fn lumen_cuda_layer_norm_backward(
        xhat: u64,
        inverse: u64,
        weight: u64,
        grad: u64,
        dx: u64,
        dw: u64,
        db: u64,
        rows: usize,
        dim: usize,
    ) -> c_int;
    fn lumen_cuda_masked_softmax(input: u64, output: u64, rows: usize, keys: usize) -> c_int;
}

fn check(status: c_int) -> Result<(), String> {
    if status == 0 {
        return Ok(());
    }
    let message = unsafe {
        let ptr = lumen_cuda_last_error_message();
        if ptr.is_null() {
            "CUDA generic operation failed".into()
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    };
    Err(message)
}

pub(crate) fn layer_norm(
    x: &CudaBuffer,
    weight: Option<&CudaBuffer>,
    bias: Option<&CudaBuffer>,
    rows: usize,
    dim: usize,
    eps: f32,
) -> Result<(CudaBuffer, CudaBuffer, CudaBuffer), String> {
    let len = rows.checked_mul(dim).ok_or("LayerNorm length overflow")?;
    if rows == 0
        || dim == 0
        || x.len() != len
        || weight.is_some_and(|b| b.len() != dim)
        || bias.is_some_and(|b| b.len() != dim)
        || !eps.is_finite()
        || eps <= 0.0
    {
        return Err("Invalid CUDA LayerNorm dimensions, buffers or eps".into());
    }
    let output = alloc_f32(len)?;
    let xhat = alloc_f32(len)?;
    let inverse = alloc_f32(rows)?;
    check(unsafe {
        lumen_cuda_layer_norm(
            x.handle(),
            weight.map_or(0, CudaBuffer::handle),
            bias.map_or(0, CudaBuffer::handle),
            output.handle(),
            xhat.handle(),
            inverse.handle(),
            rows,
            dim,
            eps,
        )
    })?;
    Ok((output, xhat, inverse))
}

pub(crate) fn layer_norm_backward(
    xhat: &CudaBuffer,
    inverse: &CudaBuffer,
    weight: Option<&CudaBuffer>,
    grad: &CudaBuffer,
    rows: usize,
    dim: usize,
) -> Result<(CudaBuffer, CudaBuffer, CudaBuffer), String> {
    let len = rows
        .checked_mul(dim)
        .ok_or("LayerNorm backward length overflow")?;
    if rows == 0
        || dim == 0
        || xhat.len() != len
        || grad.len() != len
        || inverse.len() != rows
        || weight.is_some_and(|w| w.len() != dim)
    {
        return Err("Invalid CUDA LayerNorm backward buffers".into());
    }
    let dx = alloc_f32(len)?;
    let dw = alloc_f32(dim)?;
    let db = alloc_f32(dim)?;
    check(unsafe {
        lumen_cuda_layer_norm_backward(
            xhat.handle(),
            inverse.handle(),
            weight.map_or(0, CudaBuffer::handle),
            grad.handle(),
            dx.handle(),
            dw.handle(),
            db.handle(),
            rows,
            dim,
        )
    })?;
    Ok((dx, dw, db))
}

pub(crate) fn masked_softmax(input: &CudaBuffer, keys: usize) -> Result<CudaBuffer, String> {
    if keys == 0 || input.is_empty() || !input.len().is_multiple_of(keys) {
        return Err("Invalid CUDA masked softmax dimensions".into());
    }
    let output = alloc_f32(input.len())?;
    check(unsafe {
        lumen_cuda_masked_softmax(input.handle(), output.handle(), input.len() / keys, keys)
    })?;
    Ok(output)
}
