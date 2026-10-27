use super::*;

pub fn bshd_to_bhsd_add_bias_f32_buffer(
    input: &CudaBuffer,
    bias: &CudaBuffer,
    batch: usize,
    seq: usize,
    heads: usize,
    dim: usize,
) -> Result<CudaBuffer, String> {
    let len = batch
        .checked_mul(seq)
        .and_then(|v| v.checked_mul(heads))
        .and_then(|v| v.checked_mul(dim))
        .ok_or_else(|| "CUDA BSHD to BHSD add bias length overflow".to_string())?;
    let bias_len = heads
        .checked_mul(dim)
        .ok_or_else(|| "CUDA BSHD to BHSD add bias bias length overflow".to_string())?;
    if len == 0 || input.len() != len || bias.len() != bias_len {
        return Err(format!(
            "CUDA BSHD to BHSD add bias length mismatch: input={}, bias={}, batch={}, seq={}, heads={}, dim={}",
            input.len(),
            bias.len(),
            batch,
            seq,
            heads,
            dim
        ));
    }
    let out = alloc_f32(len)?;
    let status = unsafe {
        lumen_cuda_bshd_to_bhsd_add_bias_f32_device(
            input.handle(),
            bias.handle(),
            out.handle(),
            batch,
            seq,
            heads,
            dim,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}

pub fn embedding_f32(
    indices: &CudaBuffer,
    weight: &CudaBuffer,
    num_indices: usize,
    vocab_size: usize,
    embed_dim: usize,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    let out = embedding_f32_buffer(indices, weight, num_indices, vocab_size, embed_dim)?;
    let host = download_f32(&out)?;
    Ok((out, host))
}

pub fn embedding_f32_buffer(
    indices: &CudaBuffer,
    weight: &CudaBuffer,
    num_indices: usize,
    vocab_size: usize,
    embed_dim: usize,
) -> Result<CudaBuffer, String> {
    if indices.len() != num_indices {
        return Err(format!(
            "CUDA embedding indices length mismatch: expected {}, got {}",
            num_indices,
            indices.len()
        ));
    }
    let weight_len = checked_len("CUDA embedding weight length", &[vocab_size, embed_dim])?;
    let output_len = checked_len("CUDA embedding output length", &[num_indices, embed_dim])?;
    if weight.len() != weight_len {
        return Err(format!(
            "CUDA embedding weight length mismatch: expected {}, got {}",
            weight_len,
            weight.len()
        ));
    }
    let out = alloc_f32(output_len)?;
    let status = unsafe {
        lumen_cuda_embedding_f32_device(
            indices.handle(),
            weight.handle(),
            out.handle(),
            num_indices,
            vocab_size,
            embed_dim,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}

pub fn embedding_typed(
    indices: &CudaBuffer,
    weight: &CudaBuffer,
    weight_dtype: DType,
    weight_scale: Option<f32>,
    num_indices: usize,
    vocab_size: usize,
    embed_dim: usize,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    let out = embedding_typed_buffer(
        indices,
        weight,
        weight_dtype,
        weight_scale,
        num_indices,
        vocab_size,
        embed_dim,
    )?;
    let host = download_f32(&out)?;
    Ok((out, host))
}

pub fn embedding_typed_buffer(
    indices: &CudaBuffer,
    weight: &CudaBuffer,
    weight_dtype: DType,
    weight_scale: Option<f32>,
    num_indices: usize,
    vocab_size: usize,
    embed_dim: usize,
) -> Result<CudaBuffer, String> {
    if indices.len() != num_indices {
        return Err(format!(
            "CUDA typed embedding indices length mismatch: expected {}, got {}",
            num_indices,
            indices.len()
        ));
    }
    let weight_len = checked_len(
        "CUDA typed embedding weight length",
        &[vocab_size, embed_dim],
    )?;
    let output_len = checked_len(
        "CUDA typed embedding output length",
        &[num_indices, embed_dim],
    )?;
    if weight.len() != weight_len {
        return Err(format!(
            "CUDA typed embedding weight length mismatch: expected {}, got {}",
            weight_len,
            weight.len()
        ));
    }
    let out = alloc_f32(output_len)?;
    let status = unsafe {
        lumen_cuda_embedding_typed_device(
            indices.handle(),
            weight.handle(),
            dtype_tag(weight_dtype)?,
            weight_scale.unwrap_or(1.0),
            out.handle(),
            num_indices,
            vocab_size,
            embed_dim,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}

pub fn embedding_typed_same_dtype_buffer(
    indices: &CudaBuffer,
    weight: &CudaBuffer,
    weight_dtype: DType,
    num_indices: usize,
    vocab_size: usize,
    embed_dim: usize,
) -> Result<CudaBuffer, String> {
    if !matches!(weight_dtype, DType::F16 | DType::BF16 | DType::I8) {
        return Err(format!(
            "CUDA native embedding output only supports f16/bf16/i8, got {weight_dtype:?}"
        ));
    }
    if indices.len() != num_indices {
        return Err(format!(
            "CUDA native embedding indices length mismatch: expected {}, got {}",
            num_indices,
            indices.len()
        ));
    }
    let weight_len = checked_len(
        "CUDA native embedding weight length",
        &[vocab_size, embed_dim],
    )?;
    let output_len = checked_len(
        "CUDA native embedding output length",
        &[num_indices, embed_dim],
    )?;
    if weight.len() != weight_len {
        return Err(format!(
            "CUDA native embedding weight length mismatch: expected {}, got {}",
            weight_len,
            weight.len()
        ));
    }
    let out = alloc_storage(output_len)?;
    let status = unsafe {
        lumen_cuda_embedding_typed_same_dtype_device(
            indices.handle(),
            weight.handle(),
            dtype_tag(weight_dtype)?,
            out.handle(),
            num_indices,
            vocab_size,
            embed_dim,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}

pub fn embedding_backward_f32(
    indices: &CudaBuffer,
    grad: &CudaBuffer,
    num_indices: usize,
    vocab_size: usize,
    embed_dim: usize,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    let grad_weight =
        embedding_backward_f32_buffer(indices, grad, num_indices, vocab_size, embed_dim)?;
    let host = download_f32(&grad_weight)?;
    Ok((grad_weight, host))
}

pub fn embedding_backward_f32_buffer(
    indices: &CudaBuffer,
    grad: &CudaBuffer,
    num_indices: usize,
    vocab_size: usize,
    embed_dim: usize,
) -> Result<CudaBuffer, String> {
    if indices.len() != num_indices {
        return Err(format!(
            "CUDA embedding backward indices length mismatch: expected {}, got {}",
            num_indices,
            indices.len()
        ));
    }
    let grad_len = checked_len(
        "CUDA embedding backward grad length",
        &[num_indices, embed_dim],
    )?;
    let grad_weight_len = checked_len(
        "CUDA embedding backward weight grad length",
        &[vocab_size, embed_dim],
    )?;
    if grad.len() != grad_len {
        return Err(format!(
            "CUDA embedding backward grad length mismatch: expected {}, got {}",
            grad_len,
            grad.len()
        ));
    }
    let grad_weight = alloc_f32(grad_weight_len)?;
    let status = unsafe {
        lumen_cuda_embedding_backward_f32_device(
            indices.handle(),
            grad.handle(),
            grad_weight.handle(),
            num_indices,
            vocab_size,
            embed_dim,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(grad_weight)
}

pub(super) fn permute_metadata(
    input: &CudaBuffer,
    out_shape: &[usize],
    axes: &[usize],
) -> Result<(usize, Vec<usize>, Vec<usize>), String> {
    let ndim = out_shape.len();
    if axes.len() != ndim {
        return Err(format!(
            "CUDA permute axes length mismatch: axes={}, ndim={}",
            axes.len(),
            ndim
        ));
    }
    if ndim == 0 {
        return Err("CUDA permute expects at least 1 dimension".to_string());
    }

    let len = checked_len("CUDA permute output length", out_shape)?;
    if input.len() != len {
        return Err(format!(
            "CUDA permute input length mismatch: expected {}, got {}",
            len,
            input.len()
        ));
    }

    let mut seen = vec![false; ndim];
    for &axis in axes {
        if axis >= ndim || seen[axis] {
            return Err(format!("CUDA permute axes are invalid: {:?}", axes));
        }
        seen[axis] = true;
    }

    let mut input_shape = vec![0usize; ndim];
    for (out_dim, &input_axis) in axes.iter().enumerate() {
        input_shape[input_axis] = out_shape[out_dim];
    }

    let mut input_strides = vec![0usize; ndim];
    let mut stride = 1usize;
    for i in (0..ndim).rev() {
        input_strides[i] = stride;
        stride = stride
            .checked_mul(input_shape[i])
            .ok_or_else(|| "CUDA permute stride overflow".to_string())?;
    }

    let mut out_strides = vec![0usize; ndim];
    stride = 1usize;
    for i in (0..ndim).rev() {
        out_strides[i] = stride;
        stride = stride
            .checked_mul(out_shape[i])
            .ok_or_else(|| "CUDA permute output stride overflow".to_string())?;
    }

    let mapped_input_strides = axes
        .iter()
        .map(|&axis| input_strides[axis])
        .collect::<Vec<_>>();

    Ok((len, out_strides, mapped_input_strides))
}

pub fn permute_f32(
    input: &CudaBuffer,
    out_shape: &[usize],
    axes: &[usize],
) -> Result<(CudaBuffer, Vec<f32>), String> {
    let ndim = out_shape.len();
    if axes.len() != ndim {
        return Err(format!(
            "CUDA permute axes length mismatch: axes={}, ndim={}",
            axes.len(),
            ndim
        ));
    }
    if ndim == 0 {
        return Err("CUDA permute expects at least 1 dimension".to_string());
    }

    let len = checked_len("CUDA permute output length", out_shape)?;
    if input.len() != len {
        return Err(format!(
            "CUDA permute input length mismatch: expected {}, got {}",
            len,
            input.len()
        ));
    }

    let mut seen = vec![false; ndim];
    for &axis in axes {
        if axis >= ndim || seen[axis] {
            return Err(format!("CUDA permute axes are invalid: {:?}", axes));
        }
        seen[axis] = true;
    }

    let mut input_shape = vec![0usize; ndim];
    for (out_dim, &input_axis) in axes.iter().enumerate() {
        input_shape[input_axis] = out_shape[out_dim];
    }

    let mut input_strides = vec![0usize; ndim];
    let mut stride = 1usize;
    for i in (0..ndim).rev() {
        input_strides[i] = stride;
        stride = stride
            .checked_mul(input_shape[i])
            .ok_or_else(|| "CUDA permute stride overflow".to_string())?;
    }

    let mut out_strides = vec![0usize; ndim];
    stride = 1usize;
    for i in (0..ndim).rev() {
        out_strides[i] = stride;
        stride = stride
            .checked_mul(out_shape[i])
            .ok_or_else(|| "CUDA permute output stride overflow".to_string())?;
    }

    let mapped_input_strides = axes
        .iter()
        .map(|&axis| input_strides[axis])
        .collect::<Vec<_>>();
    let out = alloc_f32(len)?;
    let status = unsafe {
        lumen_cuda_permute_f32_device(
            input.handle(),
            out.handle(),
            ndim,
            out_shape.as_ptr(),
            out_strides.as_ptr(),
            mapped_input_strides.as_ptr(),
            len,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    let host = download_f32(&out)?;
    Ok((out, host))
}

pub fn permute_f32_buffer(
    input: &CudaBuffer,
    out_shape: &[usize],
    axes: &[usize],
) -> Result<CudaBuffer, String> {
    let ndim = out_shape.len();
    if axes.len() != ndim {
        return Err(format!(
            "CUDA permute axes length mismatch: axes={}, ndim={}",
            axes.len(),
            ndim
        ));
    }
    if ndim == 0 {
        return Err("CUDA permute expects at least 1 dimension".to_string());
    }

    let len = checked_len("CUDA permute output length", out_shape)?;
    if input.len() != len {
        return Err(format!(
            "CUDA permute input length mismatch: expected {}, got {}",
            len,
            input.len()
        ));
    }

    let mut seen = vec![false; ndim];
    for &axis in axes {
        if axis >= ndim || seen[axis] {
            return Err(format!("CUDA permute axes are invalid: {:?}", axes));
        }
        seen[axis] = true;
    }

    let mut input_shape = vec![0usize; ndim];
    for (out_dim, &input_axis) in axes.iter().enumerate() {
        input_shape[input_axis] = out_shape[out_dim];
    }

    let mut input_strides = vec![0usize; ndim];
    let mut stride = 1usize;
    for i in (0..ndim).rev() {
        input_strides[i] = stride;
        stride = stride
            .checked_mul(input_shape[i])
            .ok_or_else(|| "CUDA permute stride overflow".to_string())?;
    }

    let mut out_strides = vec![0usize; ndim];
    stride = 1usize;
    for i in (0..ndim).rev() {
        out_strides[i] = stride;
        stride = stride
            .checked_mul(out_shape[i])
            .ok_or_else(|| "CUDA permute output stride overflow".to_string())?;
    }

    let mapped_input_strides = axes
        .iter()
        .map(|&axis| input_strides[axis])
        .collect::<Vec<_>>();
    let out = alloc_f32(len)?;
    let status = unsafe {
        lumen_cuda_permute_f32_device(
            input.handle(),
            out.handle(),
            ndim,
            out_shape.as_ptr(),
            out_strides.as_ptr(),
            mapped_input_strides.as_ptr(),
            len,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}

pub fn permute_typed_buffer(
    input: &CudaBuffer,
    dtype: DType,
    out_shape: &[usize],
    axes: &[usize],
) -> Result<CudaBuffer, String> {
    let ndim = out_shape.len();
    let (len, out_strides, mapped_input_strides) = permute_metadata(input, out_shape, axes)?;
    let out = alloc_storage(len)?;
    let status = unsafe {
        lumen_cuda_permute_typed_device(
            input.handle(),
            dtype_tag(dtype)?,
            out.handle(),
            ndim,
            out_shape.as_ptr(),
            out_strides.as_ptr(),
            mapped_input_strides.as_ptr(),
            len,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}

pub fn slice_lastdim_f32(
    input: &CudaBuffer,
    outer: usize,
    input_last_dim: usize,
    start: usize,
    slice_len: usize,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    let expected_len = outer
        .checked_mul(input_last_dim)
        .ok_or_else(|| "CUDA slice length overflow".to_string())?;
    if input.len() != expected_len {
        return Err(format!(
            "CUDA slice input length mismatch: expected {}, got {}",
            expected_len,
            input.len()
        ));
    }
    if !range_fits(start, slice_len, input_last_dim) {
        return Err(format!(
            "CUDA slice range out of bounds: start={}, len={}, input_last_dim={}",
            start, slice_len, input_last_dim
        ));
    }
    let out = alloc_f32(
        outer
            .checked_mul(slice_len)
            .ok_or_else(|| "CUDA slice output length overflow".to_string())?,
    )?;
    let status = unsafe {
        lumen_cuda_slice_lastdim_f32_device(
            input.handle(),
            out.handle(),
            outer,
            input_last_dim,
            start,
            slice_len,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    let host = download_f32(&out)?;
    Ok((out, host))
}

pub fn slice_lastdim_f32_buffer(
    input: &CudaBuffer,
    outer: usize,
    input_last_dim: usize,
    start: usize,
    slice_len: usize,
) -> Result<CudaBuffer, String> {
    let expected_len = outer
        .checked_mul(input_last_dim)
        .ok_or_else(|| "CUDA slice length overflow".to_string())?;
    if input.len() != expected_len {
        return Err(format!(
            "CUDA slice input length mismatch: expected {}, got {}",
            expected_len,
            input.len()
        ));
    }
    if !range_fits(start, slice_len, input_last_dim) {
        return Err(format!(
            "CUDA slice range out of bounds: start={}, len={}, input_last_dim={}",
            start, slice_len, input_last_dim
        ));
    }
    let out = alloc_f32(
        outer
            .checked_mul(slice_len)
            .ok_or_else(|| "CUDA slice output length overflow".to_string())?,
    )?;
    let status = unsafe {
        lumen_cuda_slice_lastdim_f32_device(
            input.handle(),
            out.handle(),
            outer,
            input_last_dim,
            start,
            slice_len,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}

pub fn slice_lastdim_typed_buffer(
    input: &CudaBuffer,
    dtype: DType,
    outer: usize,
    input_last_dim: usize,
    start: usize,
    slice_len: usize,
) -> Result<CudaBuffer, String> {
    let expected = outer
        .checked_mul(input_last_dim)
        .ok_or_else(|| "CUDA typed slice input length overflow".to_string())?;
    if input.len() != expected {
        return Err(format!(
            "CUDA typed slice input length mismatch: expected {}, got {}",
            expected,
            input.len()
        ));
    }
    if outer == 0 || input_last_dim == 0 || slice_len == 0 {
        return Err("CUDA typed slice dimensions must be greater than zero".to_string());
    }
    if !range_fits(start, slice_len, input_last_dim) {
        return Err(format!(
            "CUDA typed slice range out of bounds: start={}, slice_len={}, input_last_dim={}",
            start, slice_len, input_last_dim
        ));
    }
    let out_len = outer
        .checked_mul(slice_len)
        .ok_or_else(|| "CUDA typed slice output length overflow".to_string())?;
    let out = alloc_storage(out_len)?;
    let status = unsafe {
        lumen_cuda_slice_lastdim_typed_device(
            input.handle(),
            dtype_tag(dtype)?,
            out.handle(),
            outer,
            input_last_dim,
            start,
            slice_len,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}

pub fn slice_lastdim_backward_f32(
    grad: &CudaBuffer,
    outer: usize,
    input_last_dim: usize,
    start: usize,
    slice_len: usize,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    let out = slice_lastdim_backward_f32_buffer(grad, outer, input_last_dim, start, slice_len)?;
    let host = download_f32(&out)?;
    Ok((out, host))
}

pub fn slice_lastdim_backward_f32_buffer(
    grad: &CudaBuffer,
    outer: usize,
    input_last_dim: usize,
    start: usize,
    slice_len: usize,
) -> Result<CudaBuffer, String> {
    let grad_len = outer
        .checked_mul(slice_len)
        .ok_or_else(|| "CUDA slice backward grad length overflow".to_string())?;
    if grad.len() != grad_len {
        return Err(format!(
            "CUDA slice backward grad length mismatch: expected {}, got {}",
            grad_len,
            grad.len()
        ));
    }
    if !range_fits(start, slice_len, input_last_dim) {
        return Err(format!(
            "CUDA slice backward range out of bounds: start={}, len={}, input_last_dim={}",
            start, slice_len, input_last_dim
        ));
    }
    let out_len = outer
        .checked_mul(input_last_dim)
        .ok_or_else(|| "CUDA slice backward output length overflow".to_string())?;
    let out = alloc_f32(out_len)?;
    let status = unsafe {
        lumen_cuda_slice_lastdim_backward_f32_device(
            grad.handle(),
            out.handle(),
            outer,
            input_last_dim,
            start,
            slice_len,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}

pub fn cat_f32(
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    out_shape: &[usize],
    axis: usize,
    lhs_axis_len: usize,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    let ndim = out_shape.len();
    if ndim == 0 {
        return Err("CUDA cat expects at least 1 dimension".to_string());
    }
    if axis >= ndim {
        return Err(format!(
            "CUDA cat axis out of bounds: axis={}, ndim={}",
            axis, ndim
        ));
    }
    let len = out_shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| "CUDA cat output length overflow".to_string())?;
    if len == 0 {
        return Err("CUDA cat does not support empty outputs".to_string());
    }
    let out_axis_len = out_shape[axis];
    if lhs_axis_len > out_axis_len {
        return Err(format!(
            "CUDA cat lhs axis length out of bounds: lhs_axis_len={}, out_axis_len={}",
            lhs_axis_len, out_axis_len
        ));
    }
    let rhs_axis_len = out_axis_len - lhs_axis_len;

    let mut lhs_shape = out_shape.to_vec();
    lhs_shape[axis] = lhs_axis_len;
    let mut rhs_shape = out_shape.to_vec();
    rhs_shape[axis] = rhs_axis_len;

    let lhs_expected = lhs_shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| "CUDA cat lhs length overflow".to_string())?;
    let rhs_expected = rhs_shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| "CUDA cat rhs length overflow".to_string())?;
    if lhs.len() != lhs_expected || rhs.len() != rhs_expected {
        return Err(format!(
            "CUDA cat input length mismatch: expected lhs={}, rhs={}, got lhs={}, rhs={}",
            lhs_expected,
            rhs_expected,
            lhs.len(),
            rhs.len()
        ));
    }

    fn row_major_strides(shape: &[usize]) -> Result<Vec<usize>, String> {
        let mut strides = vec![0usize; shape.len()];
        let mut stride = 1usize;
        for i in (0..shape.len()).rev() {
            strides[i] = stride;
            stride = stride
                .checked_mul(shape[i])
                .ok_or_else(|| "CUDA cat stride overflow".to_string())?;
        }
        Ok(strides)
    }

    let out_strides = row_major_strides(out_shape)?;
    let lhs_strides = row_major_strides(&lhs_shape)?;
    let rhs_strides = row_major_strides(&rhs_shape)?;

    let out = alloc_f32(len)?;
    let status = unsafe {
        lumen_cuda_cat_f32_device(
            lhs.handle(),
            rhs.handle(),
            out.handle(),
            ndim,
            out_shape.as_ptr(),
            out_strides.as_ptr(),
            lhs_strides.as_ptr(),
            rhs_strides.as_ptr(),
            axis,
            lhs_axis_len,
            len,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    let host = download_f32(&out)?;
    Ok((out, host))
}

pub fn cat_f32_buffer(
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    out_shape: &[usize],
    axis: usize,
    lhs_axis_len: usize,
) -> Result<CudaBuffer, String> {
    let ndim = out_shape.len();
    if ndim == 0 {
        return Err("CUDA cat expects at least 1 dimension".to_string());
    }
    if axis >= ndim {
        return Err(format!(
            "CUDA cat axis out of bounds: axis={}, ndim={}",
            axis, ndim
        ));
    }
    let len = out_shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| "CUDA cat output length overflow".to_string())?;
    if len == 0 {
        return Err("CUDA cat does not support empty outputs".to_string());
    }
    let out_axis_len = out_shape[axis];
    if lhs_axis_len > out_axis_len {
        return Err(format!(
            "CUDA cat lhs axis length out of bounds: lhs_axis_len={}, out_axis_len={}",
            lhs_axis_len, out_axis_len
        ));
    }
    let rhs_axis_len = out_axis_len - lhs_axis_len;

    let lhs_len = out_shape
        .iter()
        .enumerate()
        .try_fold(1usize, |acc, (idx, &dim)| {
            let actual = if idx == axis { lhs_axis_len } else { dim };
            acc.checked_mul(actual)
        })
        .ok_or_else(|| "CUDA cat lhs length overflow".to_string())?;
    let rhs_len = out_shape
        .iter()
        .enumerate()
        .try_fold(1usize, |acc, (idx, &dim)| {
            let actual = if idx == axis { rhs_axis_len } else { dim };
            acc.checked_mul(actual)
        })
        .ok_or_else(|| "CUDA cat rhs length overflow".to_string())?;
    if lhs.len() != lhs_len || rhs.len() != rhs_len {
        return Err(format!(
            "CUDA cat input length mismatch: expected lhs={}, rhs={}, got lhs={}, rhs={}",
            lhs_len,
            rhs_len,
            lhs.len(),
            rhs.len()
        ));
    }

    let mut out_strides = vec![0usize; ndim];
    let mut stride = 1usize;
    for i in (0..ndim).rev() {
        out_strides[i] = stride;
        stride = stride
            .checked_mul(out_shape[i])
            .ok_or_else(|| "CUDA cat stride overflow".to_string())?;
    }

    let lhs_shape = out_shape
        .iter()
        .enumerate()
        .map(|(idx, &dim)| if idx == axis { lhs_axis_len } else { dim })
        .collect::<Vec<_>>();
    let rhs_shape = out_shape
        .iter()
        .enumerate()
        .map(|(idx, &dim)| if idx == axis { rhs_axis_len } else { dim })
        .collect::<Vec<_>>();

    let mut lhs_strides = vec![0usize; ndim];
    stride = 1usize;
    for i in (0..ndim).rev() {
        lhs_strides[i] = stride;
        stride = stride
            .checked_mul(lhs_shape[i])
            .ok_or_else(|| "CUDA cat lhs stride overflow".to_string())?;
    }

    let mut rhs_strides = vec![0usize; ndim];
    stride = 1usize;
    for i in (0..ndim).rev() {
        rhs_strides[i] = stride;
        stride = stride
            .checked_mul(rhs_shape[i])
            .ok_or_else(|| "CUDA cat rhs stride overflow".to_string())?;
    }

    let out = alloc_f32(len)?;
    let status = unsafe {
        lumen_cuda_cat_f32_device(
            lhs.handle(),
            rhs.handle(),
            out.handle(),
            ndim,
            out_shape.as_ptr(),
            out_strides.as_ptr(),
            lhs_strides.as_ptr(),
            rhs_strides.as_ptr(),
            axis,
            lhs_axis_len,
            len,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}

pub fn cat_typed_buffer(
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    dtype: DType,
    out_shape: &[usize],
    axis: usize,
    lhs_axis_len: usize,
) -> Result<CudaBuffer, String> {
    let ndim = out_shape.len();
    if ndim == 0 {
        return Err("CUDA typed cat expects at least 1 dimension".to_string());
    }
    if axis >= ndim {
        return Err(format!(
            "CUDA typed cat axis out of bounds: axis={}, ndim={}",
            axis, ndim
        ));
    }
    let len = out_shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| "CUDA typed cat output length overflow".to_string())?;
    if len == 0 {
        return Err("CUDA typed cat does not support empty outputs".to_string());
    }
    let out_axis_len = out_shape[axis];
    if lhs_axis_len > out_axis_len {
        return Err(format!(
            "CUDA typed cat lhs axis length out of bounds: lhs_axis_len={}, out_axis_len={}",
            lhs_axis_len, out_axis_len
        ));
    }
    let rhs_axis_len = out_axis_len - lhs_axis_len;

    let lhs_len = out_shape
        .iter()
        .enumerate()
        .try_fold(1usize, |acc, (idx, &dim)| {
            let actual = if idx == axis { lhs_axis_len } else { dim };
            acc.checked_mul(actual)
        })
        .ok_or_else(|| "CUDA typed cat lhs length overflow".to_string())?;
    let rhs_len = out_shape
        .iter()
        .enumerate()
        .try_fold(1usize, |acc, (idx, &dim)| {
            let actual = if idx == axis { rhs_axis_len } else { dim };
            acc.checked_mul(actual)
        })
        .ok_or_else(|| "CUDA typed cat rhs length overflow".to_string())?;
    if lhs.len() != lhs_len || rhs.len() != rhs_len {
        return Err(format!(
            "CUDA typed cat input length mismatch: expected lhs={}, rhs={}, got lhs={}, rhs={}",
            lhs_len,
            rhs_len,
            lhs.len(),
            rhs.len()
        ));
    }

    let mut out_strides = vec![0usize; ndim];
    let mut stride = 1usize;
    for i in (0..ndim).rev() {
        out_strides[i] = stride;
        stride = stride
            .checked_mul(out_shape[i])
            .ok_or_else(|| "CUDA typed cat stride overflow".to_string())?;
    }

    let lhs_shape = out_shape
        .iter()
        .enumerate()
        .map(|(idx, &dim)| if idx == axis { lhs_axis_len } else { dim })
        .collect::<Vec<_>>();
    let rhs_shape = out_shape
        .iter()
        .enumerate()
        .map(|(idx, &dim)| if idx == axis { rhs_axis_len } else { dim })
        .collect::<Vec<_>>();

    let mut lhs_strides = vec![0usize; ndim];
    stride = 1usize;
    for i in (0..ndim).rev() {
        lhs_strides[i] = stride;
        stride = stride
            .checked_mul(lhs_shape[i])
            .ok_or_else(|| "CUDA typed cat lhs stride overflow".to_string())?;
    }

    let mut rhs_strides = vec![0usize; ndim];
    stride = 1usize;
    for i in (0..ndim).rev() {
        rhs_strides[i] = stride;
        stride = stride
            .checked_mul(rhs_shape[i])
            .ok_or_else(|| "CUDA typed cat rhs stride overflow".to_string())?;
    }

    let out = alloc_storage(len)?;
    let status = unsafe {
        lumen_cuda_cat_typed_device(
            lhs.handle(),
            rhs.handle(),
            dtype_tag(dtype)?,
            out.handle(),
            ndim,
            out_shape.as_ptr(),
            out_strides.as_ptr(),
            lhs_strides.as_ptr(),
            rhs_strides.as_ptr(),
            axis,
            lhs_axis_len,
            len,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}

pub fn cat_backward_slice_f32(
    grad: &CudaBuffer,
    input_shape: &[usize],
    out_shape: &[usize],
    axis: usize,
    axis_start: usize,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    let out = cat_backward_slice_f32_buffer(grad, input_shape, out_shape, axis, axis_start)?;
    let host = download_f32(&out)?;
    Ok((out, host))
}

pub fn cat_backward_slice_f32_buffer(
    grad: &CudaBuffer,
    input_shape: &[usize],
    out_shape: &[usize],
    axis: usize,
    axis_start: usize,
) -> Result<CudaBuffer, String> {
    let ndim = out_shape.len();
    if ndim == 0 || input_shape.len() != ndim {
        return Err(format!(
            "CUDA cat backward shape rank mismatch: input_ndim={}, out_ndim={}",
            input_shape.len(),
            ndim
        ));
    }
    if axis >= ndim {
        return Err(format!(
            "CUDA cat backward axis out of bounds: axis={}, ndim={}",
            axis, ndim
        ));
    }
    for (idx, (&in_dim, &out_dim)) in input_shape.iter().zip(out_shape.iter()).enumerate() {
        if idx == axis {
            if !range_fits(axis_start, in_dim, out_dim) {
                return Err(format!(
                    "CUDA cat backward axis range out of bounds: start={}, len={}, out_dim={}",
                    axis_start, in_dim, out_dim
                ));
            }
        } else if in_dim != out_dim {
            return Err(format!(
                "CUDA cat backward non-axis dim mismatch at {}: input={}, output={}",
                idx, in_dim, out_dim
            ));
        }
    }

    let out_len = out_shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| "CUDA cat backward output grad length overflow".to_string())?;
    if grad.len() != out_len {
        return Err(format!(
            "CUDA cat backward grad length mismatch: expected {}, got {}",
            out_len,
            grad.len()
        ));
    }
    let input_len = input_shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| "CUDA cat backward input grad length overflow".to_string())?;

    fn row_major_strides(shape: &[usize]) -> Result<Vec<usize>, String> {
        let mut strides = vec![0usize; shape.len()];
        let mut stride = 1usize;
        for i in (0..shape.len()).rev() {
            strides[i] = stride;
            stride = stride
                .checked_mul(shape[i])
                .ok_or_else(|| "CUDA cat backward stride overflow".to_string())?;
        }
        Ok(strides)
    }

    let input_strides = row_major_strides(input_shape)?;
    let out_strides = row_major_strides(out_shape)?;
    let out = alloc_f32(input_len)?;
    let status = unsafe {
        lumen_cuda_cat_backward_slice_f32_device(
            grad.handle(),
            out.handle(),
            ndim,
            input_shape.as_ptr(),
            input_strides.as_ptr(),
            out_strides.as_ptr(),
            axis,
            axis_start,
            input_len,
        )
    };
    if status != 0 {
        return Err(last_error_message());
    }
    Ok(out)
}
