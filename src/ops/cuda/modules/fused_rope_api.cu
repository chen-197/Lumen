extern "C" int lumen_cuda_fused_gate_up_silu_f32_device(
    uint64_t input_handle,
    uint64_t gate_handle,
    uint64_t up_handle,
    uint64_t out_handle,
    size_t rows,
    size_t n_dim,
    size_t k_dim) {
    if (!validate_handle(input_handle, "CUDA fused gate/up input handle") ||
        !validate_handle(gate_handle, "CUDA fused gate/up gate handle") ||
        !validate_handle(up_handle, "CUDA fused gate/up up handle") ||
        !validate_handle(out_handle, "CUDA fused gate/up output handle")) {
        return 1;
    }
    if (rows == 0 || n_dim == 0 || k_dim == 0) {
        set_error("CUDA fused gate/up dimensions must be greater than zero");
        return 1;
    }
    if (!validate_dims(rows, n_dim, k_dim)) {
        return 1;
    }

    CublasHandle cublas;
    if (!init_cublas(cublas)) {
        return 1;
    }

    const size_t max_size = static_cast<size_t>(-1);
    if (rows > max_size / n_dim || rows * n_dim > max_size / sizeof(float)) {
        set_error("CUDA fused gate/up temporary length overflow");
        return 1;
    }
    size_t len = rows * n_dim;

    thread_local ReusableCudaWorkspace gate_tmp_workspace;
    if (!gate_tmp_workspace.ensure(
            len * sizeof(float),
            "failed to allocate CUDA fused gate temporary buffer")) {
        return 1;
    }
    thread_local ReusableCudaWorkspace up_tmp_workspace;
    if (!up_tmp_workspace.ensure(
            len * sizeof(float),
            "failed to allocate CUDA fused up temporary buffer")) {
        return 1;
    }
    float* gate_tmp = static_cast<float*>(gate_tmp_workspace.ptr);
    float* up_tmp = static_cast<float*>(up_tmp_workspace.ptr);

    const float alpha = 1.0f;
    const float beta = 0.0f;
    const float* input = handle_to_ptr(input_handle);
    const float* gate = handle_to_ptr(gate_handle);
    const float* up = handle_to_ptr(up_handle);
    cublasStatus_t cublas_status = cublasSgemm(
        cublas.handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        static_cast<int>(n_dim),
        static_cast<int>(rows),
        static_cast<int>(k_dim),
        &alpha,
        gate,
        static_cast<int>(k_dim),
        input,
        static_cast<int>(k_dim),
        &beta,
        gate_tmp,
        static_cast<int>(n_dim));
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        set_cublas_error("cuBLAS SGEMM failed for fused gate projection", cublas_status);
        return 1;
    }

    cublas_status = cublasSgemm(
        cublas.handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        static_cast<int>(n_dim),
        static_cast<int>(rows),
        static_cast<int>(k_dim),
        &alpha,
        up,
        static_cast<int>(k_dim),
        input,
        static_cast<int>(k_dim),
        &beta,
        up_tmp,
        static_cast<int>(n_dim));
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        set_cublas_error("cuBLAS SGEMM failed for fused up projection", cublas_status);
        return 1;
    }

    constexpr int block_size = 256;
    const unsigned int grid_size = linear_grid_size(len, block_size);
    silu_mul_kernel<<<grid_size, block_size>>>(gate_tmp, up_tmp, handle_to_ptr(out_handle), len);
    bool ok = sync_cuda("CUDA fused gate/up kernel failed");
    return ok ? 0 : 1;
}
extern "C" int lumen_cuda_silu_mul_f32_device(
    uint64_t gate_handle,
    uint64_t up_handle,
    uint64_t out_handle,
    size_t len) {
    if (!validate_handle(gate_handle, "CUDA silu_mul gate handle") ||
        !validate_handle(up_handle, "CUDA silu_mul up handle") ||
        !validate_handle(out_handle, "CUDA silu_mul output handle")) {
        return 1;
    }
    if (len == 0) {
        set_error("CUDA silu_mul length must be greater than zero");
        return 1;
    }

    constexpr int block_size = 256;
    const unsigned int grid_size = linear_grid_size(len, block_size);
    silu_mul_kernel<<<grid_size, block_size>>>(
        handle_to_ptr(gate_handle),
        handle_to_ptr(up_handle),
        handle_to_ptr(out_handle),
        len);
    bool ok = sync_cuda("CUDA silu_mul kernel failed");
    return ok ? 0 : 1;
}

template <typename OutputT>
int dispatch_fused_gate_up_silu_input_typed_out(
    uint64_t input_handle,
    int input_dtype,
    float input_scale,
    uint64_t gate_handle,
    int weight_dtype,
    float gate_scale,
    uint64_t up_handle,
    float up_scale,
    OutputT* out,
    size_t rows,
    size_t n_dim,
    size_t k_dim) {
    switch (input_dtype) {
        case kDTypeF32:
            return dispatch_fused_gate_up_silu_weight_typed_out(
                handle_to_ptr(input_handle),
                input_scale,
                gate_handle,
                weight_dtype,
                gate_scale,
                up_handle,
                up_scale,
                out,
                rows,
                n_dim,
                k_dim);
        case kDTypeF16:
            return dispatch_fused_gate_up_silu_weight_typed_out(
                reinterpret_cast<const __half*>(handle_to_ptr(input_handle)),
                input_scale,
                gate_handle,
                weight_dtype,
                gate_scale,
                up_handle,
                up_scale,
                out,
                rows,
                n_dim,
                k_dim);
        case kDTypeBF16:
            return dispatch_fused_gate_up_silu_weight_typed_out(
                reinterpret_cast<const __nv_bfloat16*>(handle_to_ptr(input_handle)),
                input_scale,
                gate_handle,
                weight_dtype,
                gate_scale,
                up_handle,
                up_scale,
                out,
                rows,
                n_dim,
                k_dim);
        case kDTypeI8:
            return dispatch_fused_gate_up_silu_weight_typed_out(
                reinterpret_cast<const int8_t*>(handle_to_ptr(input_handle)),
                input_scale,
                gate_handle,
                weight_dtype,
                gate_scale,
                up_handle,
                up_scale,
                out,
                rows,
                n_dim,
                k_dim);
        default:
            set_error("CUDA typed fused gate/up output received unsupported input dtype");
            return 1;
    }
}

template <typename OutputT>
int dispatch_fused_qkv_input_typed_out(
    uint64_t input_handle,
    int input_dtype,
    float input_scale,
    uint64_t q_handle,
    uint64_t k_handle,
    uint64_t v_handle,
    int weight_dtype,
    float q_scale,
    float k_scale,
    float v_scale,
    OutputT* q_out,
    OutputT* k_out,
    OutputT* v_out,
    size_t rows,
    size_t q_n,
    size_t k_n,
    size_t k_dim) {
    switch (input_dtype) {
        case kDTypeF32:
            return dispatch_fused_qkv_weight_typed_out(
                handle_to_ptr(input_handle),
                input_scale,
                q_handle,
                k_handle,
                v_handle,
                weight_dtype,
                q_scale,
                k_scale,
                v_scale,
                q_out,
                k_out,
                v_out,
                rows,
                q_n,
                k_n,
                k_dim);
        case kDTypeF16:
            return dispatch_fused_qkv_weight_typed_out(
                reinterpret_cast<const __half*>(handle_to_ptr(input_handle)),
                input_scale,
                q_handle,
                k_handle,
                v_handle,
                weight_dtype,
                q_scale,
                k_scale,
                v_scale,
                q_out,
                k_out,
                v_out,
                rows,
                q_n,
                k_n,
                k_dim);
        case kDTypeBF16:
            return dispatch_fused_qkv_weight_typed_out(
                reinterpret_cast<const __nv_bfloat16*>(handle_to_ptr(input_handle)),
                input_scale,
                q_handle,
                k_handle,
                v_handle,
                weight_dtype,
                q_scale,
                k_scale,
                v_scale,
                q_out,
                k_out,
                v_out,
                rows,
                q_n,
                k_n,
                k_dim);
        case kDTypeI8:
            return dispatch_fused_qkv_weight_typed_out(
                reinterpret_cast<const int8_t*>(handle_to_ptr(input_handle)),
                input_scale,
                q_handle,
                k_handle,
                v_handle,
                weight_dtype,
                q_scale,
                k_scale,
                v_scale,
                q_out,
                k_out,
                v_out,
                rows,
                q_n,
                k_n,
                k_dim);
        default:
            set_error("CUDA typed fused qkv output received unsupported input dtype");
            return 1;
    }
}

extern "C" int lumen_cuda_fused_gate_up_silu_typed_device(
    uint64_t input_handle,
    int input_dtype,
    float input_scale,
    uint64_t gate_handle,
    int weight_dtype,
    float gate_scale,
    uint64_t up_handle,
    float up_scale,
    uint64_t out_handle,
    size_t rows,
    size_t n_dim,
    size_t k_dim) {
    if (!validate_handle(input_handle, "CUDA typed fused gate/up input handle") ||
        !validate_handle(gate_handle, "CUDA typed fused gate/up gate handle") ||
        !validate_handle(up_handle, "CUDA typed fused gate/up up handle") ||
        !validate_handle(out_handle, "CUDA typed fused gate/up output handle")) {
        return 1;
    }
    if (rows == 0 || n_dim == 0 || k_dim == 0) {
        set_error("CUDA typed fused gate/up dimensions must be greater than zero");
        return 1;
    }

    float* out = handle_to_ptr(out_handle);
    switch (input_dtype) {
        case kDTypeF32:
            return dispatch_fused_gate_up_silu_weight_typed(
                handle_to_ptr(input_handle),
                input_scale,
                gate_handle,
                weight_dtype,
                gate_scale,
                up_handle,
                up_scale,
                out,
                rows,
                n_dim,
                k_dim);
        case kDTypeF16:
            return dispatch_fused_gate_up_silu_weight_typed(
                reinterpret_cast<const __half*>(handle_to_ptr(input_handle)),
                input_scale,
                gate_handle,
                weight_dtype,
                gate_scale,
                up_handle,
                up_scale,
                out,
                rows,
                n_dim,
                k_dim);
        case kDTypeBF16:
            return dispatch_fused_gate_up_silu_weight_typed(
                reinterpret_cast<const __nv_bfloat16*>(handle_to_ptr(input_handle)),
                input_scale,
                gate_handle,
                weight_dtype,
                gate_scale,
                up_handle,
                up_scale,
                out,
                rows,
                n_dim,
                k_dim);
        case kDTypeI8:
            return dispatch_fused_gate_up_silu_weight_typed(
                reinterpret_cast<const int8_t*>(handle_to_ptr(input_handle)),
                input_scale,
                gate_handle,
                weight_dtype,
                gate_scale,
                up_handle,
                up_scale,
                out,
                rows,
                n_dim,
                k_dim);
        default:
            set_error("CUDA typed fused gate/up received unsupported input dtype");
            return 1;
    }
}

extern "C" int lumen_cuda_fused_gate_up_silu_typed_out_device(
    uint64_t input_handle,
    int input_dtype,
    float input_scale,
    uint64_t gate_handle,
    int weight_dtype,
    float gate_scale,
    uint64_t up_handle,
    float up_scale,
    uint64_t out_handle,
    int output_dtype,
    size_t rows,
    size_t n_dim,
    size_t k_dim) {
    if (!validate_handle(input_handle, "CUDA typed fused gate/up output input handle") ||
        !validate_handle(gate_handle, "CUDA typed fused gate/up output gate handle") ||
        !validate_handle(up_handle, "CUDA typed fused gate/up output up handle") ||
        !validate_handle(out_handle, "CUDA typed fused gate/up output handle")) {
        return 1;
    }
    if (rows == 0 || n_dim == 0 || k_dim == 0) {
        set_error("CUDA typed fused gate/up output dimensions must be greater than zero");
        return 1;
    }

    switch (output_dtype) {
        case kDTypeF16:
            return dispatch_fused_gate_up_silu_input_typed_out(
                input_handle,
                input_dtype,
                input_scale,
                gate_handle,
                weight_dtype,
                gate_scale,
                up_handle,
                up_scale,
                reinterpret_cast<__half*>(handle_to_ptr(out_handle)),
                rows,
                n_dim,
                k_dim);
        case kDTypeBF16:
            return dispatch_fused_gate_up_silu_input_typed_out(
                input_handle,
                input_dtype,
                input_scale,
                gate_handle,
                weight_dtype,
                gate_scale,
                up_handle,
                up_scale,
                reinterpret_cast<__nv_bfloat16*>(handle_to_ptr(out_handle)),
                rows,
                n_dim,
                k_dim);
        default:
            set_error("CUDA typed fused gate/up output received unsupported output dtype");
            return 1;
    }
}

extern "C" int lumen_cuda_fused_qkv_f32_device(
    uint64_t input_handle,
    uint64_t q_handle,
    uint64_t k_handle,
    uint64_t v_handle,
    uint64_t q_out_handle,
    uint64_t k_out_handle,
    uint64_t v_out_handle,
    size_t rows,
    size_t q_n,
    size_t k_n,
    size_t k_dim) {
    if (!validate_handle(input_handle, "CUDA fused qkv input handle") ||
        !validate_handle(q_handle, "CUDA fused qkv q handle") ||
        !validate_handle(k_handle, "CUDA fused qkv k handle") ||
        !validate_handle(v_handle, "CUDA fused qkv v handle") ||
        !validate_handle(q_out_handle, "CUDA fused qkv q output handle") ||
        !validate_handle(k_out_handle, "CUDA fused qkv k output handle") ||
        !validate_handle(v_out_handle, "CUDA fused qkv v output handle")) {
        return 1;
    }
    if (rows == 0 || q_n == 0 || k_n == 0 || k_dim == 0) {
        set_error("CUDA fused qkv dimensions must be greater than zero");
        return 1;
    }
    if (!validate_dims(rows, q_n, k_dim) || !validate_dims(rows, k_n, k_dim)) {
        return 1;
    }

    CublasHandle cublas;
    if (!init_cublas(cublas)) {
        return 1;
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    const float* input = handle_to_ptr(input_handle);
    cublasStatus_t cublas_status = cublasSgemm(
        cublas.handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        static_cast<int>(q_n),
        static_cast<int>(rows),
        static_cast<int>(k_dim),
        &alpha,
        handle_to_ptr(q_handle),
        static_cast<int>(k_dim),
        input,
        static_cast<int>(k_dim),
        &beta,
        handle_to_ptr(q_out_handle),
        static_cast<int>(q_n));
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        set_cublas_error("cuBLAS SGEMM failed for fused q projection", cublas_status);
        return 1;
    }

    cublas_status = cublasSgemm(
        cublas.handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        static_cast<int>(k_n),
        static_cast<int>(rows),
        static_cast<int>(k_dim),
        &alpha,
        handle_to_ptr(k_handle),
        static_cast<int>(k_dim),
        input,
        static_cast<int>(k_dim),
        &beta,
        handle_to_ptr(k_out_handle),
        static_cast<int>(k_n));
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        set_cublas_error("cuBLAS SGEMM failed for fused k projection", cublas_status);
        return 1;
    }

    cublas_status = cublasSgemm(
        cublas.handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        static_cast<int>(k_n),
        static_cast<int>(rows),
        static_cast<int>(k_dim),
        &alpha,
        handle_to_ptr(v_handle),
        static_cast<int>(k_dim),
        input,
        static_cast<int>(k_dim),
        &beta,
        handle_to_ptr(v_out_handle),
        static_cast<int>(k_n));
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        set_cublas_error("cuBLAS SGEMM failed for fused v projection", cublas_status);
        return 1;
    }

    return 0;
}

extern "C" int lumen_cuda_fused_qkv_typed_device(
    uint64_t input_handle,
    int input_dtype,
    float input_scale,
    uint64_t q_handle,
    uint64_t k_handle,
    uint64_t v_handle,
    int weight_dtype,
    float q_scale,
    float k_scale,
    float v_scale,
    uint64_t q_out_handle,
    uint64_t k_out_handle,
    uint64_t v_out_handle,
    size_t rows,
    size_t q_n,
    size_t k_n,
    size_t k_dim) {
    if (!validate_handle(input_handle, "CUDA typed fused qkv input handle") ||
        !validate_handle(q_handle, "CUDA typed fused qkv q handle") ||
        !validate_handle(k_handle, "CUDA typed fused qkv k handle") ||
        !validate_handle(v_handle, "CUDA typed fused qkv v handle") ||
        !validate_handle(q_out_handle, "CUDA typed fused qkv q output handle") ||
        !validate_handle(k_out_handle, "CUDA typed fused qkv k output handle") ||
        !validate_handle(v_out_handle, "CUDA typed fused qkv v output handle")) {
        return 1;
    }
    if (rows == 0 || q_n == 0 || k_n == 0 || k_dim == 0) {
        set_error("CUDA typed fused qkv dimensions must be greater than zero");
        return 1;
    }

    float* q_out = handle_to_ptr(q_out_handle);
    float* k_out = handle_to_ptr(k_out_handle);
    float* v_out = handle_to_ptr(v_out_handle);
    switch (input_dtype) {
        case kDTypeF32:
            return dispatch_fused_qkv_weight_typed(
                handle_to_ptr(input_handle),
                input_scale,
                q_handle,
                k_handle,
                v_handle,
                weight_dtype,
                q_scale,
                k_scale,
                v_scale,
                q_out,
                k_out,
                v_out,
                rows,
                q_n,
                k_n,
                k_dim);
        case kDTypeF16:
            return dispatch_fused_qkv_weight_typed(
                reinterpret_cast<const __half*>(handle_to_ptr(input_handle)),
                input_scale,
                q_handle,
                k_handle,
                v_handle,
                weight_dtype,
                q_scale,
                k_scale,
                v_scale,
                q_out,
                k_out,
                v_out,
                rows,
                q_n,
                k_n,
                k_dim);
        case kDTypeBF16:
            return dispatch_fused_qkv_weight_typed(
                reinterpret_cast<const __nv_bfloat16*>(handle_to_ptr(input_handle)),
                input_scale,
                q_handle,
                k_handle,
                v_handle,
                weight_dtype,
                q_scale,
                k_scale,
                v_scale,
                q_out,
                k_out,
                v_out,
                rows,
                q_n,
                k_n,
                k_dim);
        case kDTypeI8:
            return dispatch_fused_qkv_weight_typed(
                reinterpret_cast<const int8_t*>(handle_to_ptr(input_handle)),
                input_scale,
                q_handle,
                k_handle,
                v_handle,
                weight_dtype,
                q_scale,
                k_scale,
                v_scale,
                q_out,
                k_out,
                v_out,
                rows,
                q_n,
                k_n,
                k_dim);
        default:
            set_error("CUDA typed fused qkv received unsupported input dtype");
            return 1;
    }
}

extern "C" int lumen_cuda_fused_qkv_typed_out_device(
    uint64_t input_handle,
    int input_dtype,
    float input_scale,
    uint64_t q_handle,
    uint64_t k_handle,
    uint64_t v_handle,
    int weight_dtype,
    float q_scale,
    float k_scale,
    float v_scale,
    uint64_t q_out_handle,
    uint64_t k_out_handle,
    uint64_t v_out_handle,
    int output_dtype,
    size_t rows,
    size_t q_n,
    size_t k_n,
    size_t k_dim) {
    if (!validate_handle(input_handle, "CUDA typed fused qkv output input handle") ||
        !validate_handle(q_handle, "CUDA typed fused qkv output q handle") ||
        !validate_handle(k_handle, "CUDA typed fused qkv output k handle") ||
        !validate_handle(v_handle, "CUDA typed fused qkv output v handle") ||
        !validate_handle(q_out_handle, "CUDA typed fused qkv output q output handle") ||
        !validate_handle(k_out_handle, "CUDA typed fused qkv output k output handle") ||
        !validate_handle(v_out_handle, "CUDA typed fused qkv output v output handle")) {
        return 1;
    }
    if (rows == 0 || q_n == 0 || k_n == 0 || k_dim == 0) {
        set_error("CUDA typed fused qkv output dimensions must be greater than zero");
        return 1;
    }

    switch (output_dtype) {
        case kDTypeF16:
            return dispatch_fused_qkv_input_typed_out(
                input_handle,
                input_dtype,
                input_scale,
                q_handle,
                k_handle,
                v_handle,
                weight_dtype,
                q_scale,
                k_scale,
                v_scale,
                reinterpret_cast<__half*>(handle_to_ptr(q_out_handle)),
                reinterpret_cast<__half*>(handle_to_ptr(k_out_handle)),
                reinterpret_cast<__half*>(handle_to_ptr(v_out_handle)),
                rows,
                q_n,
                k_n,
                k_dim);
        case kDTypeBF16:
            return dispatch_fused_qkv_input_typed_out(
                input_handle,
                input_dtype,
                input_scale,
                q_handle,
                k_handle,
                v_handle,
                weight_dtype,
                q_scale,
                k_scale,
                v_scale,
                reinterpret_cast<__nv_bfloat16*>(handle_to_ptr(q_out_handle)),
                reinterpret_cast<__nv_bfloat16*>(handle_to_ptr(k_out_handle)),
                reinterpret_cast<__nv_bfloat16*>(handle_to_ptr(v_out_handle)),
                rows,
                q_n,
                k_n,
                k_dim);
        default:
            set_error("CUDA typed fused qkv output received unsupported output dtype");
            return 1;
    }
}

extern "C" int lumen_cuda_rope_f32_device(
    uint64_t input_handle,
    uint64_t cos_handle,
    uint64_t sin_handle,
    uint64_t out_handle,
    size_t batch_size,
    size_t num_heads,
    size_t seq_len,
    size_t dim,
    size_t offset,
    size_t cache_seq_len) {
    if (!validate_handle(input_handle, "CUDA RoPE input handle") ||
        !validate_handle(cos_handle, "CUDA RoPE cos handle") ||
        !validate_handle(sin_handle, "CUDA RoPE sin handle") ||
        !validate_handle(out_handle, "CUDA RoPE output handle")) {
        return 1;
    }
    if (batch_size == 0 || num_heads == 0 || seq_len == 0 || dim == 0) {
        set_error("CUDA RoPE dimensions must be greater than zero");
        return 1;
    }
    if ((dim % 2) != 0) {
        set_error("CUDA RoPE expects an even hidden dimension");
        return 1;
    }
    if (offset > cache_seq_len || seq_len > cache_seq_len - offset) {
        set_error("CUDA RoPE offset exceeds cache sequence length");
        return 1;
    }

    size_t batch_heads = 0;
    size_t half = dim / 2;
    size_t pairs_per_head = 0;
    if (!checked_product(
            "CUDA RoPE batch/head size overflow",
            {batch_size, num_heads},
            &batch_heads) ||
        !checked_product(
            "CUDA RoPE pair count overflow",
            {seq_len, half},
            &pairs_per_head)) {
        return 1;
    }
    constexpr int block_size = 256;
    const unsigned int grid_x =
        std::min(linear_grid_size(pairs_per_head, block_size), 1024u);

    dim3 grid(static_cast<unsigned int>(grid_x), static_cast<unsigned int>(batch_heads), 1);
    rope_kernel<<<grid, block_size>>>(
        handle_to_ptr(input_handle),
        handle_to_ptr(cos_handle),
        handle_to_ptr(sin_handle),
        handle_to_ptr(out_handle),
        seq_len,
        dim,
        offset);
    if (!check_cuda_launch("CUDA RoPE kernel launch failed")) {
        return 1;
    }
    return 0;
}

extern "C" int lumen_cuda_rope_typed_device(
    uint64_t input_handle,
    int input_dtype,
    float input_scale,
    uint64_t cos_handle,
    uint64_t sin_handle,
    int cache_dtype,
    float cos_scale,
    float sin_scale,
    uint64_t out_handle,
    size_t batch_size,
    size_t num_heads,
    size_t seq_len,
    size_t dim,
    size_t offset,
    size_t cache_seq_len) {
    if (!validate_handle(input_handle, "CUDA typed RoPE input handle") ||
        !validate_handle(cos_handle, "CUDA typed RoPE cos handle") ||
        !validate_handle(sin_handle, "CUDA typed RoPE sin handle") ||
        !validate_handle(out_handle, "CUDA typed RoPE output handle")) {
        return 1;
    }
    if (batch_size == 0 || num_heads == 0 || seq_len == 0 || dim == 0) {
        set_error("CUDA typed RoPE dimensions must be greater than zero");
        return 1;
    }
    if ((dim % 2) != 0) {
        set_error("CUDA typed RoPE expects an even hidden dimension");
        return 1;
    }
    if (offset > cache_seq_len || seq_len > cache_seq_len - offset) {
        set_error("CUDA typed RoPE offset exceeds cache sequence length");
        return 1;
    }
    size_t total = 0;
    if (!checked_product(
            "CUDA typed RoPE element count overflow",
            {batch_size, num_heads, seq_len, dim},
            &total)) {
        return 1;
    }

    float* out = handle_to_ptr(out_handle);
    switch (input_dtype) {
        case kDTypeF32:
            return dispatch_rope_cache_typed(
                handle_to_ptr(input_handle),
                input_scale,
                cos_handle,
                sin_handle,
                cache_dtype,
                cos_scale,
                sin_scale,
                out,
                batch_size,
                num_heads,
                seq_len,
                dim,
                offset);
        case kDTypeF16:
            return dispatch_rope_cache_typed(
                reinterpret_cast<const __half*>(handle_to_ptr(input_handle)),
                input_scale,
                cos_handle,
                sin_handle,
                cache_dtype,
                cos_scale,
                sin_scale,
                out,
                batch_size,
                num_heads,
                seq_len,
                dim,
                offset);
        case kDTypeBF16:
            return dispatch_rope_cache_typed(
                reinterpret_cast<const __nv_bfloat16*>(handle_to_ptr(input_handle)),
                input_scale,
                cos_handle,
                sin_handle,
                cache_dtype,
                cos_scale,
                sin_scale,
                out,
                batch_size,
                num_heads,
                seq_len,
                dim,
                offset);
        case kDTypeI8:
            return dispatch_rope_cache_typed(
                reinterpret_cast<const int8_t*>(handle_to_ptr(input_handle)),
                input_scale,
                cos_handle,
                sin_handle,
                cache_dtype,
                cos_scale,
                sin_scale,
                out,
                batch_size,
                num_heads,
                seq_len,
                dim,
                offset);
        default:
            set_error("CUDA typed RoPE received unsupported input dtype");
            return 1;
    }
}

extern "C" int lumen_cuda_rope_typed_i8_dynamic_device(
    uint64_t input_handle,
    int input_dtype,
    float input_scale,
    uint64_t cos_handle,
    uint64_t sin_handle,
    int cache_dtype,
    float cos_scale,
    float sin_scale,
    uint64_t out_handle,
    float* out_scale,
    size_t batch_size,
    size_t num_heads,
    size_t seq_len,
    size_t dim,
    size_t offset,
    size_t cache_seq_len) {
    if (!validate_handle(input_handle, "CUDA typed RoPE dynamic i8 input handle") ||
        !validate_handle(cos_handle, "CUDA typed RoPE dynamic i8 cos handle") ||
        !validate_handle(sin_handle, "CUDA typed RoPE dynamic i8 sin handle") ||
        !validate_handle(out_handle, "CUDA typed RoPE dynamic i8 output handle")) {
        return 1;
    }
    if (out_scale == nullptr) {
        set_error("CUDA typed RoPE dynamic i8 scale output is null");
        return 1;
    }
    if (batch_size == 0 || num_heads == 0 || seq_len == 0 || dim == 0) {
        set_error("CUDA typed RoPE dynamic i8 dimensions must be greater than zero");
        return 1;
    }
    if ((dim % 2) != 0) {
        set_error("CUDA typed RoPE dynamic i8 expects an even hidden dimension");
        return 1;
    }
    if (offset > cache_seq_len || seq_len > cache_seq_len - offset) {
        set_error("CUDA typed RoPE dynamic i8 offset exceeds cache sequence length");
        return 1;
    }
    size_t total = 0;
    if (!checked_product(
            "CUDA typed RoPE dynamic i8 element count overflow",
            {batch_size, num_heads, seq_len, dim},
            &total)) {
        return 1;
    }

    int8_t* out = reinterpret_cast<int8_t*>(handle_to_ptr(out_handle));
    switch (input_dtype) {
        case kDTypeF32:
            return dispatch_rope_cache_typed_i8_dynamic(
                handle_to_ptr(input_handle),
                input_scale,
                cos_handle,
                sin_handle,
                cache_dtype,
                cos_scale,
                sin_scale,
                out,
                out_scale,
                batch_size,
                num_heads,
                seq_len,
                dim,
                offset);
        case kDTypeF16:
            return dispatch_rope_cache_typed_i8_dynamic(
                reinterpret_cast<const __half*>(handle_to_ptr(input_handle)),
                input_scale,
                cos_handle,
                sin_handle,
                cache_dtype,
                cos_scale,
                sin_scale,
                out,
                out_scale,
                batch_size,
                num_heads,
                seq_len,
                dim,
                offset);
        case kDTypeBF16:
            return dispatch_rope_cache_typed_i8_dynamic(
                reinterpret_cast<const __nv_bfloat16*>(handle_to_ptr(input_handle)),
                input_scale,
                cos_handle,
                sin_handle,
                cache_dtype,
                cos_scale,
                sin_scale,
                out,
                out_scale,
                batch_size,
                num_heads,
                seq_len,
                dim,
                offset);
        case kDTypeI8:
            return dispatch_rope_cache_typed_i8_dynamic(
                reinterpret_cast<const int8_t*>(handle_to_ptr(input_handle)),
                input_scale,
                cos_handle,
                sin_handle,
                cache_dtype,
                cos_scale,
                sin_scale,
                out,
                out_scale,
                batch_size,
                num_heads,
                seq_len,
                dim,
                offset);
        default:
            set_error("CUDA typed RoPE dynamic i8 received unsupported input dtype");
            return 1;
    }
}

extern "C" int lumen_cuda_rope_backward_f32_device(
    uint64_t grad_handle,
    uint64_t cos_handle,
    uint64_t sin_handle,
    uint64_t out_handle,
    size_t batch_size,
    size_t num_heads,
    size_t seq_len,
    size_t dim,
    size_t offset,
    size_t cache_seq_len) {
    if (!validate_handle(grad_handle, "CUDA RoPE backward grad handle") ||
        !validate_handle(cos_handle, "CUDA RoPE backward cos handle") ||
        !validate_handle(sin_handle, "CUDA RoPE backward sin handle") ||
        !validate_handle(out_handle, "CUDA RoPE backward output handle")) {
        return 1;
    }
    if (batch_size == 0 || num_heads == 0 || seq_len == 0 || dim == 0) {
        set_error("CUDA RoPE backward dimensions must be greater than zero");
        return 1;
    }
    if (dim % 2 != 0) {
        set_error("CUDA RoPE backward expects an even hidden dimension");
        return 1;
    }
    if (offset > cache_seq_len || seq_len > cache_seq_len - offset) {
        set_error("CUDA RoPE backward offset exceeds cache sequence length");
        return 1;
    }

    size_t batch_heads = 0;
    size_t half = dim / 2;
    size_t pairs_per_head = 0;
    if (!checked_product(
            "CUDA RoPE backward batch/head size overflow",
            {batch_size, num_heads},
            &batch_heads) ||
        !checked_product(
            "CUDA RoPE backward pair count overflow",
            {seq_len, half},
            &pairs_per_head)) {
        return 1;
    }
    constexpr int block_size = 256;
    const unsigned int grid_x =
        std::min(linear_grid_size(pairs_per_head, block_size), 1024u);

    dim3 grid(static_cast<unsigned int>(grid_x), static_cast<unsigned int>(batch_heads), 1);
    rope_backward_kernel<<<grid, block_size>>>(
        handle_to_ptr(grad_handle),
        handle_to_ptr(cos_handle),
        handle_to_ptr(sin_handle),
        handle_to_ptr(out_handle),
        seq_len,
        dim,
        offset);
    if (!check_cuda_launch("CUDA RoPE backward kernel launch failed")) {
        return 1;
    }
    return 0;
}
