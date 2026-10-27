use super::*;

pub fn fused_gate_up_silu_f32(
    input: &CudaBuffer,
    gate: &CudaBuffer,
    up: &CudaBuffer,
    rows: usize,
    n_dim: usize,
    k_dim: usize,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    let out = fused_gate_up_silu_f32_buffer(input, gate, up, rows, n_dim, k_dim)?;
    let host = download_f32(&out)?;
    Ok((out, host))
}

pub fn fused_gate_up_silu_f32_buffer(
    input: &CudaBuffer,
    gate: &CudaBuffer,
    up: &CudaBuffer,
    rows: usize,
    n_dim: usize,
    k_dim: usize,
) -> Result<CudaBuffer, String> {
    let input_len = rows
        .checked_mul(k_dim)
        .ok_or_else(|| "CUDA fused gate/up input length overflow".to_string())?;
    let weight_len = n_dim
        .checked_mul(k_dim)
        .ok_or_else(|| "CUDA fused gate/up weight length overflow".to_string())?;
    if input.len() != input_len {
        return Err(format!(
            "CUDA fused gate/up input length mismatch: expected {}, got {}",
            input_len,
            input.len()
        ));
    }
    if gate.len() != weight_len || up.len() != weight_len {
        return Err(format!(
            "CUDA fused gate/up weight length mismatch: expected {}, got gate={}, up={}",
            weight_len,
            gate.len(),
            up.len()
        ));
    }
    if rows == 0 || n_dim == 0 || k_dim == 0 {
        return Err("CUDA fused gate/up dimensions must be greater than zero".to_string());
    }
    let out_len = rows
        .checked_mul(n_dim)
        .ok_or_else(|| "CUDA fused gate/up output length overflow".to_string())?;
    let out = alloc_f32(out_len)?;
    let status = unsafe {
        lumen_cuda_fused_gate_up_silu_f32_device(
            input.handle(),
            gate.handle(),
            up.handle(),
            out.handle(),
            rows,
            n_dim,
            k_dim,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}

pub fn silu_mul_f32_buffer_no_host(
    gate: &CudaBuffer,
    up: &CudaBuffer,
) -> Result<CudaBuffer, String> {
    if gate.len() != up.len() {
        return Err(format!(
            "CUDA silu_mul length mismatch: gate={}, up={}",
            gate.len(),
            up.len()
        ));
    }
    if gate.is_empty() {
        return Err("CUDA silu_mul length must be greater than zero".to_string());
    }
    let out = alloc_f32(gate.len())?;
    let status = unsafe {
        lumen_cuda_silu_mul_f32_device(gate.handle(), up.handle(), out.handle(), gate.len())
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
pub fn fused_gate_up_silu_typed(
    input: &CudaBuffer,
    input_dtype: DType,
    input_scale: Option<f32>,
    gate: &CudaBuffer,
    weight_dtype: DType,
    gate_scale: Option<f32>,
    up: &CudaBuffer,
    up_scale: Option<f32>,
    rows: usize,
    n_dim: usize,
    k_dim: usize,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    let out = fused_gate_up_silu_typed_buffer(
        input,
        input_dtype,
        input_scale,
        gate,
        weight_dtype,
        gate_scale,
        up,
        up_scale,
        rows,
        n_dim,
        k_dim,
    )?;
    let host = download_f32(&out)?;
    Ok((out, host))
}

#[allow(clippy::too_many_arguments)]
pub fn fused_gate_up_silu_typed_buffer(
    input: &CudaBuffer,
    input_dtype: DType,
    input_scale: Option<f32>,
    gate: &CudaBuffer,
    weight_dtype: DType,
    gate_scale: Option<f32>,
    up: &CudaBuffer,
    up_scale: Option<f32>,
    rows: usize,
    n_dim: usize,
    k_dim: usize,
) -> Result<CudaBuffer, String> {
    let input_len = rows
        .checked_mul(k_dim)
        .ok_or_else(|| "CUDA typed fused gate/up input length overflow".to_string())?;
    let weight_len = n_dim
        .checked_mul(k_dim)
        .ok_or_else(|| "CUDA typed fused gate/up weight length overflow".to_string())?;
    if input.len() != input_len {
        return Err(format!(
            "CUDA typed fused gate/up input length mismatch: expected {}, got {}",
            input_len,
            input.len()
        ));
    }
    if gate.len() != weight_len || up.len() != weight_len {
        return Err(format!(
            "CUDA typed fused gate/up weight length mismatch: expected {}, got gate={}, up={}",
            weight_len,
            gate.len(),
            up.len()
        ));
    }
    if rows == 0 || n_dim == 0 || k_dim == 0 {
        return Err("CUDA typed fused gate/up dimensions must be greater than zero".to_string());
    }
    let out_len = rows
        .checked_mul(n_dim)
        .ok_or_else(|| "CUDA typed fused gate/up output length overflow".to_string())?;
    let out = alloc_f32(out_len)?;
    let status = unsafe {
        lumen_cuda_fused_gate_up_silu_typed_device(
            input.handle(),
            dtype_tag(input_dtype)?,
            input_scale.unwrap_or(1.0),
            gate.handle(),
            dtype_tag(weight_dtype)?,
            gate_scale.unwrap_or(1.0),
            up.handle(),
            up_scale.unwrap_or(1.0),
            out.handle(),
            i32::from(crate::precision::allow_dynamic_i8_activations()),
            rows,
            n_dim,
            k_dim,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
pub fn fused_gate_up_silu_typed_output_buffer(
    input: &CudaBuffer,
    input_dtype: DType,
    input_scale: Option<f32>,
    gate: &CudaBuffer,
    weight_dtype: DType,
    gate_scale: Option<f32>,
    up: &CudaBuffer,
    up_scale: Option<f32>,
    output_dtype: DType,
    rows: usize,
    n_dim: usize,
    k_dim: usize,
) -> Result<CudaBuffer, String> {
    if !matches!(output_dtype, DType::F16 | DType::BF16) {
        return Err(format!(
            "CUDA typed fused gate/up output only supports f16/bf16, got {output_dtype:?}"
        ));
    }
    let input_len = rows
        .checked_mul(k_dim)
        .ok_or_else(|| "CUDA typed fused gate/up output input length overflow".to_string())?;
    let weight_len = n_dim
        .checked_mul(k_dim)
        .ok_or_else(|| "CUDA typed fused gate/up output weight length overflow".to_string())?;
    if input.len() != input_len {
        return Err(format!(
            "CUDA typed fused gate/up output input length mismatch: expected {}, got {}",
            input_len,
            input.len()
        ));
    }
    if gate.len() != weight_len || up.len() != weight_len {
        return Err(format!(
            "CUDA typed fused gate/up output weight length mismatch: expected {}, got gate={}, up={}",
            weight_len,
            gate.len(),
            up.len()
        ));
    }
    if rows == 0 || n_dim == 0 || k_dim == 0 {
        return Err(
            "CUDA typed fused gate/up output dimensions must be greater than zero".to_string(),
        );
    }
    let out_len = rows
        .checked_mul(n_dim)
        .ok_or_else(|| "CUDA typed fused gate/up output length overflow".to_string())?;
    let out = alloc_storage(out_len)?;
    let status = unsafe {
        lumen_cuda_fused_gate_up_silu_typed_out_device(
            input.handle(),
            dtype_tag(input_dtype)?,
            input_scale.unwrap_or(1.0),
            gate.handle(),
            dtype_tag(weight_dtype)?,
            gate_scale.unwrap_or(1.0),
            up.handle(),
            up_scale.unwrap_or(1.0),
            out.handle(),
            dtype_tag(output_dtype)?,
            rows,
            n_dim,
            k_dim,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}

pub fn fused_qkv_f32(
    input: &CudaBuffer,
    q: &CudaBuffer,
    k: &CudaBuffer,
    v: &CudaBuffer,
    rows: usize,
    q_n: usize,
    k_n: usize,
    k_dim: usize,
) -> Result<CudaThreeHostBuffers, String> {
    let input_len = rows
        .checked_mul(k_dim)
        .ok_or_else(|| "CUDA fused qkv input length overflow".to_string())?;
    if input.len() != input_len {
        return Err(format!(
            "CUDA fused qkv input length mismatch: expected {}, got {}",
            input_len,
            input.len()
        ));
    }
    let q_weight_len = q_n
        .checked_mul(k_dim)
        .ok_or_else(|| "CUDA fused qkv q weight length overflow".to_string())?;
    let kv_weight_len = k_n
        .checked_mul(k_dim)
        .ok_or_else(|| "CUDA fused qkv kv weight length overflow".to_string())?;
    if q.len() != q_weight_len || k.len() != kv_weight_len || v.len() != kv_weight_len {
        return Err(format!(
            "CUDA fused qkv weight length mismatch: expected q={}, k/v={}, got q={}, k={}, v={}",
            q_weight_len,
            kv_weight_len,
            q.len(),
            k.len(),
            v.len()
        ));
    }
    if rows == 0 || q_n == 0 || k_n == 0 || k_dim == 0 {
        return Err("CUDA fused qkv dimensions must be greater than zero".to_string());
    }
    let q_out = alloc_f32(
        rows.checked_mul(q_n)
            .ok_or_else(|| "CUDA fused qkv q output length overflow".to_string())?,
    )?;
    let k_out = alloc_f32(
        rows.checked_mul(k_n)
            .ok_or_else(|| "CUDA fused qkv k output length overflow".to_string())?,
    )?;
    let v_out = alloc_f32(
        rows.checked_mul(k_n)
            .ok_or_else(|| "CUDA fused qkv v output length overflow".to_string())?,
    )?;
    let status = unsafe {
        lumen_cuda_fused_qkv_f32_device(
            input.handle(),
            q.handle(),
            k.handle(),
            v.handle(),
            q_out.handle(),
            k_out.handle(),
            v_out.handle(),
            rows,
            q_n,
            k_n,
            k_dim,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    let q_host = download_f32(&q_out)?;
    let k_host = download_f32(&k_out)?;
    let v_host = download_f32(&v_out)?;
    Ok(((q_out, q_host), (k_out, k_host), (v_out, v_host)))
}

pub fn fused_qkv_f32_buffer(
    input: &CudaBuffer,
    q: &CudaBuffer,
    k: &CudaBuffer,
    v: &CudaBuffer,
    rows: usize,
    q_n: usize,
    k_n: usize,
    k_dim: usize,
) -> Result<(CudaBuffer, CudaBuffer, CudaBuffer), String> {
    let input_len = rows
        .checked_mul(k_dim)
        .ok_or_else(|| "CUDA fused qkv input length overflow".to_string())?;
    if input.len() != input_len {
        return Err(format!(
            "CUDA fused qkv input length mismatch: expected {}, got {}",
            input_len,
            input.len()
        ));
    }
    let q_weight_len = q_n
        .checked_mul(k_dim)
        .ok_or_else(|| "CUDA fused qkv q weight length overflow".to_string())?;
    let kv_weight_len = k_n
        .checked_mul(k_dim)
        .ok_or_else(|| "CUDA fused qkv kv weight length overflow".to_string())?;
    if q.len() != q_weight_len || k.len() != kv_weight_len || v.len() != kv_weight_len {
        return Err(format!(
            "CUDA fused qkv weight length mismatch: expected q={}, k/v={}, got q={}, k={}, v={}",
            q_weight_len,
            kv_weight_len,
            q.len(),
            k.len(),
            v.len()
        ));
    }
    if rows == 0 || q_n == 0 || k_n == 0 || k_dim == 0 {
        return Err("CUDA fused qkv dimensions must be greater than zero".to_string());
    }
    let q_out = alloc_f32(
        rows.checked_mul(q_n)
            .ok_or_else(|| "CUDA fused qkv q output length overflow".to_string())?,
    )?;
    let k_out = alloc_f32(
        rows.checked_mul(k_n)
            .ok_or_else(|| "CUDA fused qkv k output length overflow".to_string())?,
    )?;
    let v_out = alloc_f32(
        rows.checked_mul(k_n)
            .ok_or_else(|| "CUDA fused qkv v output length overflow".to_string())?,
    )?;
    let status = unsafe {
        lumen_cuda_fused_qkv_f32_device(
            input.handle(),
            q.handle(),
            k.handle(),
            v.handle(),
            q_out.handle(),
            k_out.handle(),
            v_out.handle(),
            rows,
            q_n,
            k_n,
            k_dim,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok((q_out, k_out, v_out))
}

#[allow(clippy::too_many_arguments)]
pub fn fused_qkv_typed_buffer(
    input: &CudaBuffer,
    input_dtype: DType,
    input_scale: Option<f32>,
    q: &CudaBuffer,
    k: &CudaBuffer,
    v: &CudaBuffer,
    weight_dtype: DType,
    q_scale: Option<f32>,
    k_scale: Option<f32>,
    v_scale: Option<f32>,
    rows: usize,
    q_n: usize,
    k_n: usize,
    k_dim: usize,
) -> Result<(CudaBuffer, CudaBuffer, CudaBuffer), String> {
    let input_len = rows
        .checked_mul(k_dim)
        .ok_or_else(|| "CUDA typed fused qkv input length overflow".to_string())?;
    if input.len() != input_len {
        return Err(format!(
            "CUDA typed fused qkv input length mismatch: expected {}, got {}",
            input_len,
            input.len()
        ));
    }
    let q_weight_len = q_n
        .checked_mul(k_dim)
        .ok_or_else(|| "CUDA typed fused qkv q weight length overflow".to_string())?;
    let kv_weight_len = k_n
        .checked_mul(k_dim)
        .ok_or_else(|| "CUDA typed fused qkv kv weight length overflow".to_string())?;
    if q.len() != q_weight_len || k.len() != kv_weight_len || v.len() != kv_weight_len {
        return Err(format!(
            "CUDA typed fused qkv weight length mismatch: expected q={}, k/v={}, got q={}, k={}, v={}",
            q_weight_len,
            kv_weight_len,
            q.len(),
            k.len(),
            v.len()
        ));
    }
    if rows == 0 || q_n == 0 || k_n == 0 || k_dim == 0 {
        return Err("CUDA typed fused qkv dimensions must be greater than zero".to_string());
    }
    let q_out = alloc_f32(
        rows.checked_mul(q_n)
            .ok_or_else(|| "CUDA typed fused qkv q output length overflow".to_string())?,
    )?;
    let k_out = alloc_f32(
        rows.checked_mul(k_n)
            .ok_or_else(|| "CUDA typed fused qkv k output length overflow".to_string())?,
    )?;
    let v_out = alloc_f32(
        rows.checked_mul(k_n)
            .ok_or_else(|| "CUDA typed fused qkv v output length overflow".to_string())?,
    )?;
    let status = unsafe {
        lumen_cuda_fused_qkv_typed_device(
            input.handle(),
            dtype_tag(input_dtype)?,
            input_scale.unwrap_or(1.0),
            q.handle(),
            k.handle(),
            v.handle(),
            dtype_tag(weight_dtype)?,
            q_scale.unwrap_or(1.0),
            k_scale.unwrap_or(1.0),
            v_scale.unwrap_or(1.0),
            q_out.handle(),
            k_out.handle(),
            v_out.handle(),
            i32::from(crate::precision::allow_dynamic_i8_activations()),
            rows,
            q_n,
            k_n,
            k_dim,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok((q_out, k_out, v_out))
}

#[allow(clippy::too_many_arguments)]
pub fn fused_qkv_typed_output_buffer(
    input: &CudaBuffer,
    input_dtype: DType,
    input_scale: Option<f32>,
    q: &CudaBuffer,
    k: &CudaBuffer,
    v: &CudaBuffer,
    weight_dtype: DType,
    q_scale: Option<f32>,
    k_scale: Option<f32>,
    v_scale: Option<f32>,
    output_dtype: DType,
    rows: usize,
    q_n: usize,
    k_n: usize,
    k_dim: usize,
) -> Result<(CudaBuffer, CudaBuffer, CudaBuffer), String> {
    if !matches!(output_dtype, DType::F16 | DType::BF16) {
        return Err(format!(
            "CUDA typed fused qkv output only supports f16/bf16, got {output_dtype:?}"
        ));
    }
    let input_len = rows
        .checked_mul(k_dim)
        .ok_or_else(|| "CUDA typed fused qkv output input length overflow".to_string())?;
    if input.len() != input_len {
        return Err(format!(
            "CUDA typed fused qkv output input length mismatch: expected {}, got {}",
            input_len,
            input.len()
        ));
    }
    let q_weight_len = q_n
        .checked_mul(k_dim)
        .ok_or_else(|| "CUDA typed fused qkv output q weight length overflow".to_string())?;
    let kv_weight_len = k_n
        .checked_mul(k_dim)
        .ok_or_else(|| "CUDA typed fused qkv output kv weight length overflow".to_string())?;
    if q.len() != q_weight_len || k.len() != kv_weight_len || v.len() != kv_weight_len {
        return Err(format!(
            "CUDA typed fused qkv output weight length mismatch: expected q={}, k/v={}, got q={}, k={}, v={}",
            q_weight_len,
            kv_weight_len,
            q.len(),
            k.len(),
            v.len()
        ));
    }
    if rows == 0 || q_n == 0 || k_n == 0 || k_dim == 0 {
        return Err("CUDA typed fused qkv output dimensions must be greater than zero".to_string());
    }
    let q_out = alloc_storage(
        rows.checked_mul(q_n)
            .ok_or_else(|| "CUDA typed fused qkv output q length overflow".to_string())?,
    )?;
    let k_out = alloc_storage(
        rows.checked_mul(k_n)
            .ok_or_else(|| "CUDA typed fused qkv output k length overflow".to_string())?,
    )?;
    let v_out = alloc_storage(
        rows.checked_mul(k_n)
            .ok_or_else(|| "CUDA typed fused qkv output v length overflow".to_string())?,
    )?;
    let status = unsafe {
        lumen_cuda_fused_qkv_typed_out_device(
            input.handle(),
            dtype_tag(input_dtype)?,
            input_scale.unwrap_or(1.0),
            q.handle(),
            k.handle(),
            v.handle(),
            dtype_tag(weight_dtype)?,
            q_scale.unwrap_or(1.0),
            k_scale.unwrap_or(1.0),
            v_scale.unwrap_or(1.0),
            q_out.handle(),
            k_out.handle(),
            v_out.handle(),
            dtype_tag(output_dtype)?,
            rows,
            q_n,
            k_n,
            k_dim,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok((q_out, k_out, v_out))
}
