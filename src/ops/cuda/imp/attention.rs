use super::*;

pub fn round_kv_cache_rows_f32(
    buffer: &CudaBuffer,
    batch_size: usize,
    num_heads: usize,
    rows: usize,
    capacity: usize,
    dim: usize,
    start: usize,
    dtype: crate::precision::DType,
) -> Result<(), String> {
    if [batch_size, num_heads, rows, capacity, dim].contains(&0)
        || start > capacity
        || rows > capacity.saturating_sub(start)
    {
        return Err("Invalid CUDA KV rounding geometry".into());
    }
    let length = batch_size
        .checked_mul(num_heads)
        .and_then(|n| n.checked_mul(capacity))
        .and_then(|n| n.checked_mul(dim))
        .ok_or("CUDA KV rounding length overflow")?;
    if length != buffer.len() {
        return Err("CUDA KV rounding buffer length mismatch".into());
    }
    let dtype = match dtype {
        crate::precision::DType::F32 => return Ok(()),
        crate::precision::DType::F16 => 1,
        crate::precision::DType::BF16 => 2,
        crate::precision::DType::I8 => return Err("KV cache requires a floating dtype".into()),
    };
    let status = unsafe {
        lumen_cuda_round_kv_cache_rows_f32_device(
            buffer.handle(),
            batch_size,
            num_heads,
            rows,
            capacity,
            dim,
            start,
            dtype,
        )
    };
    if status == 0 {
        Ok(())
    } else {
        Err(last_error_message())
    }
}

#[allow(clippy::too_many_arguments)]
pub fn append_kv_cache_f32(
    dst: &CudaBuffer,
    src: &CudaBuffer,
    batch_size: usize,
    num_heads: usize,
    src_seq_len: usize,
    dst_seq_len: usize,
    dim: usize,
    dst_start: usize,
) -> Result<(), String> {
    let src_len = batch_size
        .checked_mul(num_heads)
        .and_then(|v| v.checked_mul(src_seq_len))
        .and_then(|v| v.checked_mul(dim))
        .ok_or_else(|| "CUDA KV cache source length overflow".to_string())?;
    if src.len() != src_len {
        return Err(format!(
            "CUDA KV cache source length mismatch: expected {}, got {}",
            src_len,
            src.len()
        ));
    }

    let dst_len = batch_size
        .checked_mul(num_heads)
        .and_then(|v| v.checked_mul(dst_seq_len))
        .and_then(|v| v.checked_mul(dim))
        .ok_or_else(|| "CUDA KV cache destination length overflow".to_string())?;
    if dst.len() != dst_len {
        return Err(format!(
            "CUDA KV cache destination length mismatch: expected {}, got {}",
            dst_len,
            dst.len()
        ));
    }
    if dst_start > dst_seq_len || src_seq_len > dst_seq_len.saturating_sub(dst_start) {
        return Err(format!(
            "CUDA KV cache append range out of bounds: start={}, src_seq_len={}, dst_seq_len={}",
            dst_start, src_seq_len, dst_seq_len
        ));
    }

    let status = unsafe {
        lumen_cuda_append_kv_cache_f32_device(
            dst.handle(),
            src.handle(),
            batch_size,
            num_heads,
            src_seq_len,
            dst_seq_len,
            dim,
            dst_start,
        )
    };
    if status == 0 {
        Ok(())
    } else {
        Err(last_error_message())
    }
}

#[allow(clippy::too_many_arguments)]
pub fn append_kv_cache_pair_f32(
    k_dst: &CudaBuffer,
    v_dst: &CudaBuffer,
    k_src: &CudaBuffer,
    v_src: &CudaBuffer,
    batch_size: usize,
    num_heads: usize,
    src_seq_len: usize,
    dst_seq_len: usize,
    dim: usize,
    dst_start: usize,
) -> Result<(), String> {
    let src_len = batch_size
        .checked_mul(num_heads)
        .and_then(|v| v.checked_mul(src_seq_len))
        .and_then(|v| v.checked_mul(dim))
        .ok_or_else(|| "CUDA KV cache pair source length overflow".to_string())?;
    if k_src.len() != src_len || v_src.len() != src_len {
        return Err(format!(
            "CUDA KV cache pair source length mismatch: expected {}, got k={}, v={}",
            src_len,
            k_src.len(),
            v_src.len()
        ));
    }

    let dst_len = batch_size
        .checked_mul(num_heads)
        .and_then(|v| v.checked_mul(dst_seq_len))
        .and_then(|v| v.checked_mul(dim))
        .ok_or_else(|| "CUDA KV cache pair destination length overflow".to_string())?;
    if k_dst.len() != dst_len || v_dst.len() != dst_len {
        return Err(format!(
            "CUDA KV cache pair destination length mismatch: expected {}, got k={}, v={}",
            dst_len,
            k_dst.len(),
            v_dst.len()
        ));
    }
    if dst_start > dst_seq_len || src_seq_len > dst_seq_len.saturating_sub(dst_start) {
        return Err(format!(
            "CUDA KV cache pair append range out of bounds: start={}, src_seq_len={}, dst_seq_len={}",
            dst_start, src_seq_len, dst_seq_len
        ));
    }

    let status = unsafe {
        lumen_cuda_append_kv_cache_pair_f32_device(
            k_dst.handle(),
            v_dst.handle(),
            k_src.handle(),
            v_src.handle(),
            batch_size,
            num_heads,
            src_seq_len,
            dst_seq_len,
            dim,
            dst_start,
        )
    };
    if status == 0 {
        Ok(())
    } else {
        Err(last_error_message())
    }
}

#[allow(clippy::too_many_arguments)]
pub fn decode_rope_q_append_kv_f32_buffer(
    q_src: &CudaBuffer,
    k_src: &CudaBuffer,
    v_src: &CudaBuffer,
    cos: &CudaBuffer,
    sin: &CudaBuffer,
    k_cache: &CudaBuffer,
    v_cache: &CudaBuffer,
    batch_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    dim: usize,
    dst_seq_len: usize,
    offset: usize,
    cache_seq_len: usize,
) -> Result<CudaBuffer, String> {
    let q_len = batch_size
        .checked_mul(num_heads)
        .and_then(|v| v.checked_mul(dim))
        .ok_or_else(|| "CUDA decode RoPE Q length overflow".to_string())?;
    let kv_step_len = batch_size
        .checked_mul(num_kv_heads)
        .and_then(|v| v.checked_mul(dim))
        .ok_or_else(|| "CUDA decode RoPE KV step length overflow".to_string())?;
    let cache_len = batch_size
        .checked_mul(num_kv_heads)
        .and_then(|v| v.checked_mul(dst_seq_len))
        .and_then(|v| v.checked_mul(dim))
        .ok_or_else(|| "CUDA decode RoPE KV cache length overflow".to_string())?;
    let rope_cache_len = cache_seq_len
        .checked_mul(dim)
        .ok_or_else(|| "CUDA decode RoPE cache length overflow".to_string())?;
    if q_src.len() != q_len {
        return Err(format!(
            "CUDA decode RoPE Q length mismatch: expected {}, got {}",
            q_len,
            q_src.len()
        ));
    }
    if k_src.len() != kv_step_len || v_src.len() != kv_step_len {
        return Err(format!(
            "CUDA decode RoPE KV step length mismatch: expected {}, got k={}, v={}",
            kv_step_len,
            k_src.len(),
            v_src.len()
        ));
    }
    if k_cache.len() != cache_len || v_cache.len() != cache_len {
        return Err(format!(
            "CUDA decode RoPE KV cache length mismatch: expected {}, got k={}, v={}",
            cache_len,
            k_cache.len(),
            v_cache.len()
        ));
    }
    if cos.len() != rope_cache_len || sin.len() != rope_cache_len {
        return Err(format!(
            "CUDA decode RoPE cache length mismatch: expected {}, got cos={}, sin={}",
            rope_cache_len,
            cos.len(),
            sin.len()
        ));
    }
    if batch_size == 0 || num_heads == 0 || num_kv_heads == 0 || dim == 0 || dst_seq_len == 0 {
        return Err("CUDA decode RoPE dimensions must be greater than zero".to_string());
    }
    if !dim.is_multiple_of(2) {
        return Err(format!(
            "CUDA decode RoPE expects a positive even dimension, got {}",
            dim
        ));
    }
    if offset >= dst_seq_len || offset >= cache_seq_len {
        return Err(format!(
            "CUDA decode RoPE offset out of bounds: offset={}, dst_seq_len={}, cache_seq_len={}",
            offset, dst_seq_len, cache_seq_len
        ));
    }

    let q_out = alloc_f32(q_len)?;
    let status = unsafe {
        lumen_cuda_decode_rope_q_append_kv_f32_device(
            q_src.handle(),
            k_src.handle(),
            v_src.handle(),
            cos.handle(),
            sin.handle(),
            q_out.handle(),
            k_cache.handle(),
            v_cache.handle(),
            batch_size,
            num_heads,
            num_kv_heads,
            dim,
            dst_seq_len,
            offset,
            cache_seq_len,
        )
    };
    if status == 0 {
        Ok(q_out)
    } else {
        Err(last_error_message())
    }
}

pub fn kv_cache_prefix_f32_buffer(
    src: &CudaBuffer,
    batch_size: usize,
    num_heads: usize,
    active_seq_len: usize,
    src_seq_len: usize,
    dim: usize,
) -> Result<CudaBuffer, String> {
    let src_len = batch_size
        .checked_mul(num_heads)
        .and_then(|v| v.checked_mul(src_seq_len))
        .and_then(|v| v.checked_mul(dim))
        .ok_or_else(|| "CUDA KV cache source length overflow".to_string())?;
    if src.len() != src_len {
        return Err(format!(
            "CUDA KV cache source length mismatch: expected {}, got {}",
            src_len,
            src.len()
        ));
    }
    if active_seq_len == 0 || active_seq_len > src_seq_len {
        return Err(format!(
            "CUDA KV cache prefix range out of bounds: active_seq_len={}, src_seq_len={}",
            active_seq_len, src_seq_len
        ));
    }
    let out_len = batch_size
        .checked_mul(num_heads)
        .and_then(|v| v.checked_mul(active_seq_len))
        .and_then(|v| v.checked_mul(dim))
        .ok_or_else(|| "CUDA KV cache prefix output length overflow".to_string())?;
    let out = alloc_f32(out_len)?;
    let status = unsafe {
        lumen_cuda_kv_cache_prefix_f32_device(
            src.handle(),
            out.handle(),
            batch_size,
            num_heads,
            active_seq_len,
            src_seq_len,
            dim,
        )
    };
    if status == 0 {
        Ok(out)
    } else {
        Err(last_error_message())
    }
}

pub fn kv_cache_prefix_typed_buffer(
    src: &CudaBuffer,
    dtype: DType,
    batch_size: usize,
    num_heads: usize,
    active_seq_len: usize,
    src_seq_len: usize,
    dim: usize,
) -> Result<CudaBuffer, String> {
    let src_len = batch_size
        .checked_mul(num_heads)
        .and_then(|v| v.checked_mul(src_seq_len))
        .and_then(|v| v.checked_mul(dim))
        .ok_or_else(|| "CUDA typed KV cache source length overflow".to_string())?;
    if src.len() != src_len {
        return Err(format!(
            "CUDA typed KV cache source length mismatch: expected {}, got {}",
            src_len,
            src.len()
        ));
    }
    if active_seq_len == 0 || active_seq_len > src_seq_len {
        return Err(format!(
            "CUDA typed KV cache prefix range out of bounds: active_seq_len={}, src_seq_len={}",
            active_seq_len, src_seq_len
        ));
    }
    let out_len = batch_size
        .checked_mul(num_heads)
        .and_then(|v| v.checked_mul(active_seq_len))
        .and_then(|v| v.checked_mul(dim))
        .ok_or_else(|| "CUDA typed KV cache prefix output length overflow".to_string())?;
    let out = alloc_storage(out_len)?;
    let status = unsafe {
        lumen_cuda_kv_cache_prefix_typed_device(
            src.handle(),
            dtype_tag(dtype)?,
            out.handle(),
            batch_size,
            num_heads,
            active_seq_len,
            src_seq_len,
            dim,
        )
    };
    if status == 0 {
        Ok(out)
    } else {
        Err(last_error_message())
    }
}

pub fn repeat_kv_f32(
    input: &CudaBuffer,
    batch_size: usize,
    num_kv_heads: usize,
    seq_len: usize,
    dim: usize,
    n_rep: usize,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    let input_len = batch_size
        .checked_mul(num_kv_heads)
        .and_then(|value| value.checked_mul(seq_len))
        .and_then(|value| value.checked_mul(dim))
        .ok_or_else(|| "CUDA repeat_kv input length overflow".to_string())?;
    if input.len() != input_len {
        return Err(format!(
            "CUDA repeat_kv input length mismatch: expected {}, got {}",
            input_len,
            input.len()
        ));
    }
    if batch_size == 0 || num_kv_heads == 0 || seq_len == 0 || dim == 0 || n_rep == 0 {
        return Err("CUDA repeat_kv dimensions must be greater than zero".to_string());
    }
    let out_len = input_len
        .checked_mul(n_rep)
        .ok_or_else(|| "CUDA repeat_kv output length overflow".to_string())?;
    let out = alloc_f32(out_len)?;
    let status = unsafe {
        lumen_cuda_repeat_kv_f32_device(
            input.handle(),
            out.handle(),
            batch_size,
            num_kv_heads,
            seq_len,
            dim,
            n_rep,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    let host = download_f32(&out)?;
    Ok((out, host))
}

pub fn repeat_kv_f32_buffer(
    input: &CudaBuffer,
    batch_size: usize,
    num_kv_heads: usize,
    seq_len: usize,
    dim: usize,
    n_rep: usize,
) -> Result<CudaBuffer, String> {
    let input_len = batch_size
        .checked_mul(num_kv_heads)
        .and_then(|value| value.checked_mul(seq_len))
        .and_then(|value| value.checked_mul(dim))
        .ok_or_else(|| "CUDA repeat_kv input length overflow".to_string())?;
    if input.len() != input_len {
        return Err(format!(
            "CUDA repeat_kv input length mismatch: expected {}, got {}",
            input_len,
            input.len()
        ));
    }
    if batch_size == 0 || num_kv_heads == 0 || seq_len == 0 || dim == 0 || n_rep == 0 {
        return Err("CUDA repeat_kv dimensions must be greater than zero".to_string());
    }
    let out_len = input_len
        .checked_mul(n_rep)
        .ok_or_else(|| "CUDA repeat_kv output length overflow".to_string())?;
    let out = alloc_f32(out_len)?;
    let status = unsafe {
        lumen_cuda_repeat_kv_f32_device(
            input.handle(),
            out.handle(),
            batch_size,
            num_kv_heads,
            seq_len,
            dim,
            n_rep,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}

pub fn repeat_kv_typed_buffer(
    input: &CudaBuffer,
    dtype: DType,
    batch_size: usize,
    num_kv_heads: usize,
    seq_len: usize,
    dim: usize,
    n_rep: usize,
) -> Result<CudaBuffer, String> {
    let input_len = batch_size
        .checked_mul(num_kv_heads)
        .and_then(|value| value.checked_mul(seq_len))
        .and_then(|value| value.checked_mul(dim))
        .ok_or_else(|| "CUDA typed repeat_kv input length overflow".to_string())?;
    if input.len() != input_len {
        return Err(format!(
            "CUDA typed repeat_kv input length mismatch: expected {}, got {}",
            input_len,
            input.len()
        ));
    }
    if batch_size == 0 || num_kv_heads == 0 || seq_len == 0 || dim == 0 || n_rep == 0 {
        return Err("CUDA typed repeat_kv dimensions must be greater than zero".to_string());
    }
    let out_len = batch_size
        .checked_mul(num_kv_heads)
        .and_then(|value| value.checked_mul(n_rep))
        .and_then(|value| value.checked_mul(seq_len))
        .and_then(|value| value.checked_mul(dim))
        .ok_or_else(|| "CUDA typed repeat_kv output length overflow".to_string())?;
    let out = alloc_storage(out_len)?;
    let status = unsafe {
        lumen_cuda_repeat_kv_typed_device(
            input.handle(),
            dtype_tag(dtype)?,
            out.handle(),
            batch_size,
            num_kv_heads,
            seq_len,
            dim,
            n_rep,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}

pub fn repeat_kv_backward_f32(
    grad: &CudaBuffer,
    batch_size: usize,
    num_kv_heads: usize,
    seq_len: usize,
    dim: usize,
    n_rep: usize,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    let out = repeat_kv_backward_f32_buffer(grad, batch_size, num_kv_heads, seq_len, dim, n_rep)?;
    let host = download_f32(&out)?;
    Ok((out, host))
}

pub fn repeat_kv_backward_f32_buffer(
    grad: &CudaBuffer,
    batch_size: usize,
    num_kv_heads: usize,
    seq_len: usize,
    dim: usize,
    n_rep: usize,
) -> Result<CudaBuffer, String> {
    let grad_len = batch_size
        .checked_mul(num_kv_heads)
        .and_then(|value| value.checked_mul(n_rep))
        .and_then(|value| value.checked_mul(seq_len))
        .and_then(|value| value.checked_mul(dim))
        .ok_or_else(|| "CUDA repeat_kv backward grad length overflow".to_string())?;
    if grad.len() != grad_len {
        return Err(format!(
            "CUDA repeat_kv backward grad length mismatch: expected {}, got {}",
            grad_len,
            grad.len()
        ));
    }
    let out_len = batch_size
        .checked_mul(num_kv_heads)
        .and_then(|value| value.checked_mul(seq_len))
        .and_then(|value| value.checked_mul(dim))
        .ok_or_else(|| "CUDA repeat_kv backward output length overflow".to_string())?;
    let out = alloc_f32(out_len)?;
    let status = unsafe {
        lumen_cuda_repeat_kv_backward_f32_device(
            grad.handle(),
            out.handle(),
            batch_size,
            num_kv_heads,
            seq_len,
            dim,
            n_rep,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
pub fn decode_attention_f32(
    q: &CudaBuffer,
    k: &CudaBuffer,
    v: &CudaBuffer,
    batch_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    active_seq_len: usize,
    cache_seq_len: usize,
    dim: usize,
    n_rep: usize,
    scale: f32,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    ensure_finite("CUDA decode attention scale", scale)?;
    let q_len = batch_size
        .checked_mul(num_heads)
        .and_then(|value| value.checked_mul(dim))
        .ok_or_else(|| "CUDA decode attention q length overflow".to_string())?;
    let kv_len = batch_size
        .checked_mul(num_kv_heads)
        .and_then(|value| value.checked_mul(cache_seq_len))
        .and_then(|value| value.checked_mul(dim))
        .ok_or_else(|| "CUDA decode attention kv length overflow".to_string())?;
    if q.len() != q_len {
        return Err(format!(
            "CUDA decode attention q length mismatch: expected {}, got {}",
            q_len,
            q.len()
        ));
    }
    if k.len() != kv_len || v.len() != kv_len {
        return Err(format!(
            "CUDA decode attention kv length mismatch: expected {}, got k={}, v={}",
            kv_len,
            k.len(),
            v.len()
        ));
    }
    if batch_size == 0
        || num_heads == 0
        || num_kv_heads == 0
        || active_seq_len == 0
        || cache_seq_len == 0
        || dim == 0
        || n_rep == 0
    {
        return Err("CUDA decode attention dimensions must be greater than zero".to_string());
    }
    if active_seq_len > cache_seq_len {
        return Err(format!(
            "CUDA decode attention active_seq_len out of bounds: {} > {}",
            active_seq_len, cache_seq_len
        ));
    }
    let out_len = batch_size
        .checked_mul(num_heads)
        .and_then(|value| value.checked_mul(dim))
        .ok_or_else(|| "CUDA decode attention output length overflow".to_string())?;
    let out = alloc_f32(out_len)?;
    let status = unsafe {
        lumen_cuda_decode_attention_f32_device(
            q.handle(),
            k.handle(),
            v.handle(),
            out.handle(),
            batch_size,
            num_heads,
            num_kv_heads,
            active_seq_len,
            cache_seq_len,
            dim,
            n_rep,
            scale,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    let host = download_f32(&out)?;
    Ok((out, host))
}

#[allow(clippy::too_many_arguments)]
pub fn decode_attention_f32_buffer(
    q: &CudaBuffer,
    k: &CudaBuffer,
    v: &CudaBuffer,
    batch_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    active_seq_len: usize,
    cache_seq_len: usize,
    dim: usize,
    n_rep: usize,
    scale: f32,
) -> Result<CudaBuffer, String> {
    ensure_finite("CUDA decode attention scale", scale)?;
    let q_len = batch_size
        .checked_mul(num_heads)
        .and_then(|value| value.checked_mul(dim))
        .ok_or_else(|| "CUDA decode attention q length overflow".to_string())?;
    let kv_len = batch_size
        .checked_mul(num_kv_heads)
        .and_then(|value| value.checked_mul(cache_seq_len))
        .and_then(|value| value.checked_mul(dim))
        .ok_or_else(|| "CUDA decode attention kv length overflow".to_string())?;
    if q.len() != q_len {
        return Err(format!(
            "CUDA decode attention q length mismatch: expected {}, got {}",
            q_len,
            q.len()
        ));
    }
    if k.len() != kv_len || v.len() != kv_len {
        return Err(format!(
            "CUDA decode attention kv length mismatch: expected {}, got k={}, v={}",
            kv_len,
            k.len(),
            v.len()
        ));
    }
    if batch_size == 0
        || num_heads == 0
        || num_kv_heads == 0
        || active_seq_len == 0
        || cache_seq_len == 0
        || dim == 0
        || n_rep == 0
    {
        return Err("CUDA decode attention dimensions must be greater than zero".to_string());
    }
    if active_seq_len > cache_seq_len {
        return Err(format!(
            "CUDA decode attention active_seq_len out of bounds: {} > {}",
            active_seq_len, cache_seq_len
        ));
    }
    let out_len = batch_size
        .checked_mul(num_heads)
        .and_then(|value| value.checked_mul(dim))
        .ok_or_else(|| "CUDA decode attention output length overflow".to_string())?;
    let out = alloc_f32(out_len)?;
    let status = unsafe {
        lumen_cuda_decode_attention_f32_device(
            q.handle(),
            k.handle(),
            v.handle(),
            out.handle(),
            batch_size,
            num_heads,
            num_kv_heads,
            active_seq_len,
            cache_seq_len,
            dim,
            n_rep,
            scale,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
pub fn prefill_attention_f32_buffer(
    q: &CudaBuffer,
    k: &CudaBuffer,
    v: &CudaBuffer,
    batch_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    q_seq_len: usize,
    active_seq_len: usize,
    cache_seq_len: usize,
    dim: usize,
    n_rep: usize,
    past_len: usize,
    scale: f32,
    is_causal: bool,
) -> Result<CudaBuffer, String> {
    ensure_finite("CUDA prefill attention scale", scale)?;
    let q_len = batch_size
        .checked_mul(num_heads)
        .and_then(|value| value.checked_mul(q_seq_len))
        .and_then(|value| value.checked_mul(dim))
        .ok_or_else(|| "CUDA prefill attention q length overflow".to_string())?;
    let kv_len = batch_size
        .checked_mul(num_kv_heads)
        .and_then(|value| value.checked_mul(cache_seq_len))
        .and_then(|value| value.checked_mul(dim))
        .ok_or_else(|| "CUDA prefill attention kv length overflow".to_string())?;
    if q.len() != q_len {
        return Err(format!(
            "CUDA prefill attention q length mismatch: expected {}, got {}",
            q_len,
            q.len()
        ));
    }
    if k.len() != kv_len || v.len() != kv_len {
        return Err(format!(
            "CUDA prefill attention kv length mismatch: expected {}, got k={}, v={}",
            kv_len,
            k.len(),
            v.len()
        ));
    }
    if batch_size == 0
        || num_heads == 0
        || num_kv_heads == 0
        || q_seq_len == 0
        || active_seq_len == 0
        || cache_seq_len == 0
        || dim == 0
        || n_rep == 0
    {
        return Err("CUDA prefill attention dimensions must be greater than zero".to_string());
    }
    if active_seq_len > cache_seq_len
        || past_len > active_seq_len
        || q_seq_len > active_seq_len - past_len
    {
        return Err(format!(
            "CUDA prefill attention sequence range out of bounds: past_len={}, q_seq_len={}, active_seq_len={}, cache_seq_len={}",
            past_len, q_seq_len, active_seq_len, cache_seq_len
        ));
    }
    let out_len = batch_size
        .checked_mul(q_seq_len)
        .and_then(|value| value.checked_mul(num_heads))
        .and_then(|value| value.checked_mul(dim))
        .ok_or_else(|| "CUDA prefill attention output length overflow".to_string())?;
    let out = alloc_f32(out_len)?;
    let status = unsafe {
        lumen_cuda_prefill_attention_f32_device(
            q.handle(),
            k.handle(),
            v.handle(),
            out.handle(),
            batch_size,
            num_heads,
            num_kv_heads,
            q_seq_len,
            active_seq_len,
            cache_seq_len,
            dim,
            n_rep,
            past_len,
            scale,
            if is_causal { 1 } else { 0 },
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}

pub fn rope_f32(
    input: &CudaBuffer,
    cos: &CudaBuffer,
    sin: &CudaBuffer,
    batch_size: usize,
    num_heads: usize,
    seq_len: usize,
    dim: usize,
    offset: usize,
    cache_seq_len: usize,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    let expected_len = batch_size
        .checked_mul(num_heads)
        .and_then(|value| value.checked_mul(seq_len))
        .and_then(|value| value.checked_mul(dim))
        .ok_or_else(|| "CUDA RoPE input length overflow".to_string())?;
    if input.len() != expected_len {
        return Err(format!(
            "CUDA RoPE input length mismatch: expected {}, got {}",
            expected_len,
            input.len()
        ));
    }
    if dim == 0 || !dim.is_multiple_of(2) {
        return Err(format!(
            "CUDA RoPE expects a positive even dimension, got {}",
            dim
        ));
    }
    if !range_fits(offset, seq_len, cache_seq_len) {
        return Err(format!(
            "CUDA RoPE offset out of bounds: offset {} + seq_len {} > cache_seq_len {}",
            offset, seq_len, cache_seq_len
        ));
    }
    let cache_expected_len = cache_seq_len
        .checked_mul(dim)
        .ok_or_else(|| "CUDA RoPE cache length overflow".to_string())?;
    if cos.len() != cache_expected_len || sin.len() != cache_expected_len {
        return Err(format!(
            "CUDA RoPE cache length mismatch: expected {}, got cos={}, sin={}",
            cache_expected_len,
            cos.len(),
            sin.len()
        ));
    }
    let out = alloc_f32(expected_len)?;
    let status = unsafe {
        lumen_cuda_rope_f32_device(
            input.handle(),
            cos.handle(),
            sin.handle(),
            out.handle(),
            batch_size,
            num_heads,
            seq_len,
            dim,
            offset,
            cache_seq_len,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    let host = download_f32(&out)?;
    Ok((out, host))
}

pub fn rope_f32_buffer(
    input: &CudaBuffer,
    cos: &CudaBuffer,
    sin: &CudaBuffer,
    batch_size: usize,
    num_heads: usize,
    seq_len: usize,
    dim: usize,
    offset: usize,
    cache_seq_len: usize,
) -> Result<CudaBuffer, String> {
    let expected_len = batch_size
        .checked_mul(num_heads)
        .and_then(|value| value.checked_mul(seq_len))
        .and_then(|value| value.checked_mul(dim))
        .ok_or_else(|| "CUDA RoPE input length overflow".to_string())?;
    if input.len() != expected_len {
        return Err(format!(
            "CUDA RoPE input length mismatch: expected {}, got {}",
            expected_len,
            input.len()
        ));
    }
    if dim == 0 || !dim.is_multiple_of(2) {
        return Err(format!(
            "CUDA RoPE expects a positive even dimension, got {}",
            dim
        ));
    }
    if !range_fits(offset, seq_len, cache_seq_len) {
        return Err(format!(
            "CUDA RoPE offset out of bounds: offset {} + seq_len {} > cache_seq_len {}",
            offset, seq_len, cache_seq_len
        ));
    }
    let cache_len = cache_seq_len
        .checked_mul(dim)
        .ok_or_else(|| "CUDA RoPE cache length overflow".to_string())?;
    if cos.len() != cache_len || sin.len() != cache_len {
        return Err(format!(
            "CUDA RoPE cache length mismatch: expected {}, got cos={}, sin={}",
            cache_len,
            cos.len(),
            sin.len()
        ));
    }
    let out = alloc_f32(expected_len)?;
    let status = unsafe {
        lumen_cuda_rope_f32_device(
            input.handle(),
            cos.handle(),
            sin.handle(),
            out.handle(),
            batch_size,
            num_heads,
            seq_len,
            dim,
            offset,
            cache_seq_len,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
pub fn rope_typed_buffer(
    input: &CudaBuffer,
    input_dtype: DType,
    input_scale: Option<f32>,
    cos: &CudaBuffer,
    sin: &CudaBuffer,
    cache_dtype: DType,
    cos_scale: Option<f32>,
    sin_scale: Option<f32>,
    batch_size: usize,
    num_heads: usize,
    seq_len: usize,
    dim: usize,
    offset: usize,
    cache_seq_len: usize,
) -> Result<CudaBuffer, String> {
    let expected_len = batch_size
        .checked_mul(num_heads)
        .and_then(|value| value.checked_mul(seq_len))
        .and_then(|value| value.checked_mul(dim))
        .ok_or_else(|| "CUDA typed RoPE input length overflow".to_string())?;
    if input.len() != expected_len {
        return Err(format!(
            "CUDA typed RoPE input length mismatch: expected {}, got {}",
            expected_len,
            input.len()
        ));
    }
    if dim == 0 || !dim.is_multiple_of(2) {
        return Err(format!(
            "CUDA typed RoPE expects a positive even dimension, got {}",
            dim
        ));
    }
    if !range_fits(offset, seq_len, cache_seq_len) {
        return Err(format!(
            "CUDA typed RoPE offset out of bounds: offset {} + seq_len {} > cache_seq_len {}",
            offset, seq_len, cache_seq_len
        ));
    }
    let cache_expected_len = cache_seq_len
        .checked_mul(dim)
        .ok_or_else(|| "CUDA typed RoPE cache length overflow".to_string())?;
    if cos.len() != cache_expected_len || sin.len() != cache_expected_len {
        return Err(format!(
            "CUDA typed RoPE cache length mismatch: expected {}, got cos={}, sin={}",
            cache_expected_len,
            cos.len(),
            sin.len()
        ));
    }
    let out = alloc_f32(expected_len)?;
    let status = unsafe {
        lumen_cuda_rope_typed_device(
            input.handle(),
            dtype_tag(input_dtype)?,
            input_scale.unwrap_or(1.0),
            cos.handle(),
            sin.handle(),
            dtype_tag(cache_dtype)?,
            cos_scale.unwrap_or(1.0),
            sin_scale.unwrap_or(1.0),
            out.handle(),
            batch_size,
            num_heads,
            seq_len,
            dim,
            offset,
            cache_seq_len,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
pub fn rope_typed_i8_dynamic_buffer(
    input: &CudaBuffer,
    input_dtype: DType,
    input_scale: Option<f32>,
    cos: &CudaBuffer,
    sin: &CudaBuffer,
    cache_dtype: DType,
    cos_scale: Option<f32>,
    sin_scale: Option<f32>,
    batch_size: usize,
    num_heads: usize,
    seq_len: usize,
    dim: usize,
    offset: usize,
    cache_seq_len: usize,
) -> Result<(CudaBuffer, f32), String> {
    let expected_len = batch_size
        .checked_mul(num_heads)
        .and_then(|value| value.checked_mul(seq_len))
        .and_then(|value| value.checked_mul(dim))
        .ok_or_else(|| "CUDA typed RoPE dynamic i8 input length overflow".to_string())?;
    if input.len() != expected_len {
        return Err(format!(
            "CUDA typed RoPE dynamic i8 input length mismatch: expected {}, got {}",
            expected_len,
            input.len()
        ));
    }
    if dim == 0 || !dim.is_multiple_of(2) {
        return Err(format!(
            "CUDA typed RoPE dynamic i8 expects a positive even dimension, got {}",
            dim
        ));
    }
    if !range_fits(offset, seq_len, cache_seq_len) {
        return Err(format!(
            "CUDA typed RoPE dynamic i8 offset out of bounds: offset {} + seq_len {} > cache_seq_len {}",
            offset, seq_len, cache_seq_len
        ));
    }
    let cache_expected_len = cache_seq_len
        .checked_mul(dim)
        .ok_or_else(|| "CUDA typed RoPE dynamic i8 cache length overflow".to_string())?;
    if cos.len() != cache_expected_len || sin.len() != cache_expected_len {
        return Err(format!(
            "CUDA typed RoPE dynamic i8 cache length mismatch: expected {}, got cos={}, sin={}",
            cache_expected_len,
            cos.len(),
            sin.len()
        ));
    }
    if input_dtype == DType::I8
        && !input_scale.is_some_and(|scale| scale.is_finite() && scale > 0.0)
    {
        return Err("CUDA typed RoPE dynamic i8 input scale must be finite and > 0".to_string());
    }
    if cache_dtype == DType::I8
        && (!cos_scale.is_some_and(|scale| scale.is_finite() && scale > 0.0)
            || !sin_scale.is_some_and(|scale| scale.is_finite() && scale > 0.0))
    {
        return Err("CUDA typed RoPE dynamic i8 cache scales must be finite and > 0".to_string());
    }

    let out = alloc_storage(expected_len)?;
    let mut scale = 1.0f32;
    let status = unsafe {
        lumen_cuda_rope_typed_i8_dynamic_device(
            input.handle(),
            dtype_tag(input_dtype)?,
            input_scale.unwrap_or(1.0),
            cos.handle(),
            sin.handle(),
            dtype_tag(cache_dtype)?,
            cos_scale.unwrap_or(1.0),
            sin_scale.unwrap_or(1.0),
            out.handle(),
            &mut scale as *mut f32,
            batch_size,
            num_heads,
            seq_len,
            dim,
            offset,
            cache_seq_len,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok((out, scale))
}

#[allow(clippy::too_many_arguments)]
pub fn rope_backward_f32(
    grad: &CudaBuffer,
    cos: &CudaBuffer,
    sin: &CudaBuffer,
    batch_size: usize,
    num_heads: usize,
    seq_len: usize,
    dim: usize,
    offset: usize,
    cache_seq_len: usize,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    let out = rope_backward_f32_buffer(
        grad,
        cos,
        sin,
        batch_size,
        num_heads,
        seq_len,
        dim,
        offset,
        cache_seq_len,
    )?;
    let host = download_f32(&out)?;
    Ok((out, host))
}

#[allow(clippy::too_many_arguments)]
pub fn rope_backward_f32_buffer(
    grad: &CudaBuffer,
    cos: &CudaBuffer,
    sin: &CudaBuffer,
    batch_size: usize,
    num_heads: usize,
    seq_len: usize,
    dim: usize,
    offset: usize,
    cache_seq_len: usize,
) -> Result<CudaBuffer, String> {
    let expected_len = batch_size
        .checked_mul(num_heads)
        .and_then(|value| value.checked_mul(seq_len))
        .and_then(|value| value.checked_mul(dim))
        .ok_or_else(|| "CUDA RoPE backward grad length overflow".to_string())?;
    if grad.len() != expected_len {
        return Err(format!(
            "CUDA RoPE backward grad length mismatch: expected {}, got {}",
            expected_len,
            grad.len()
        ));
    }
    if dim == 0 || !dim.is_multiple_of(2) {
        return Err(format!(
            "CUDA RoPE backward expects a positive even dimension, got {}",
            dim
        ));
    }
    if !range_fits(offset, seq_len, cache_seq_len) {
        return Err(format!(
            "CUDA RoPE backward offset out of bounds: offset {} + seq_len {} > cache_seq_len {}",
            offset, seq_len, cache_seq_len
        ));
    }
    let cache_len = cache_seq_len
        .checked_mul(dim)
        .ok_or_else(|| "CUDA RoPE backward cache length overflow".to_string())?;
    if cos.len() != cache_len || sin.len() != cache_len {
        return Err(format!(
            "CUDA RoPE backward cache length mismatch: expected {}, got cos={}, sin={}",
            cache_len,
            cos.len(),
            sin.len()
        ));
    }
    let out = alloc_f32(expected_len)?;
    let status = unsafe {
        lumen_cuda_rope_backward_f32_device(
            grad.handle(),
            cos.handle(),
            sin.handle(),
            out.handle(),
            batch_size,
            num_heads,
            seq_len,
            dim,
            offset,
            cache_seq_len,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}
