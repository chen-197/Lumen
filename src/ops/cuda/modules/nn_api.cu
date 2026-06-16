static int* prepare_embedding_status(const char* context) {
    thread_local ReusableCudaWorkspace status_workspace;
    if (!status_workspace.ensure(sizeof(int), context)) {
        return nullptr;
    }
    cudaError_t status = cudaMemset(status_workspace.ptr, 0, sizeof(int));
    if (status != cudaSuccess) {
        set_cuda_error(context, status);
        return nullptr;
    }
    return static_cast<int*>(status_workspace.ptr);
}
static bool read_embedding_status(
    int* device_status,
    const char* read_context,
    const char* invalid_index_error,
    const char* out_of_bounds_error) {
    int host_status = 0;
    cudaError_t status =
        cudaMemcpy(&host_status, device_status, sizeof(int), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        set_cuda_error(read_context, status);
        return false;
    }
    if (host_status == 1) {
        set_error(invalid_index_error);
        return false;
    }
    if (host_status == 2) {
        set_error(out_of_bounds_error);
        return false;
    }
    return true;
}

extern "C" int lumen_cuda_softmax_lastdim_f32_device(
    uint64_t input_handle,
    uint64_t out_handle,
    size_t outer,
    size_t last_dim) {
    if (!validate_handle(input_handle, "CUDA softmax input handle") ||
        !validate_handle(out_handle, "CUDA softmax output handle")) {
        return 1;
    }
    if (outer == 0 || last_dim == 0) {
        set_error("CUDA softmax dimensions must be greater than zero");
        return 1;
    }

#if LUMEN_HAS_CUDNN
    if (validate_int_dimensions(
            "CUDA softmax dimensions exceed cuDNN int range", {outer, last_dim})) {
        CudnnHandle handle;
        CudnnTensorDescriptor tensor_desc;
        if (!init_cudnn(handle) ||
            !init_tensor_descriptor_4d(
                tensor_desc,
                static_cast<int>(outer),
                static_cast<int>(last_dim),
                1,
                1)) {
            return 1;
        }

        const float alpha = 1.0f;
        const float beta = 0.0f;
        cudnnStatus_t status = cudnnSoftmaxForward(
            handle.handle,
            CUDNN_SOFTMAX_ACCURATE,
            CUDNN_SOFTMAX_MODE_CHANNEL,
            &alpha,
            tensor_desc.desc,
            handle_to_ptr(input_handle),
            &beta,
            tensor_desc.desc,
            handle_to_ptr(out_handle));
        if (status != CUDNN_STATUS_SUCCESS) {
            set_cudnn_error("cuDNN softmax forward failed", status);
            return 1;
        }
        return 0;
    }
    g_last_error.clear();
#endif

    constexpr int block_size = 128;
    const unsigned int grid_size = linear_grid_size(outer, block_size);
    softmax_lastdim_kernel<<<grid_size, block_size>>>(
        handle_to_ptr(input_handle),
        handle_to_ptr(out_handle),
        outer,
        last_dim);
    if (!check_cuda_launch("CUDA softmax kernel launch failed")) {
        return 1;
    }
    return 0;
}

extern "C" int lumen_cuda_softmax_lastdim_typed_device(
    uint64_t input_handle,
    int input_dtype,
    float input_scale,
    uint64_t out_handle,
    size_t outer,
    size_t last_dim) {
    if (!validate_handle(input_handle, "CUDA typed softmax input handle") ||
        !validate_handle(out_handle, "CUDA typed softmax output handle")) {
        return 1;
    }
    if (outer == 0 || last_dim == 0) {
        set_error("CUDA typed softmax dimensions must be greater than zero");
        return 1;
    }

    constexpr int block_size = 128;
    const unsigned int grid_size = linear_grid_size(outer, block_size);
    switch (input_dtype) {
        case kDTypeF32:
            softmax_lastdim_typed_kernel<float><<<grid_size, block_size>>>(
                handle_to_ptr(input_handle), input_scale, handle_to_ptr(out_handle), outer, last_dim);
            break;
        case kDTypeF16:
            softmax_lastdim_typed_kernel<__half><<<grid_size, block_size>>>(
                reinterpret_cast<const __half*>(handle_to_ptr(input_handle)),
                input_scale,
                handle_to_ptr(out_handle),
                outer,
                last_dim);
            break;
        case kDTypeBF16:
            softmax_lastdim_typed_kernel<__nv_bfloat16><<<grid_size, block_size>>>(
                reinterpret_cast<const __nv_bfloat16*>(handle_to_ptr(input_handle)),
                input_scale,
                handle_to_ptr(out_handle),
                outer,
                last_dim);
            break;
        case kDTypeI8:
            softmax_lastdim_typed_kernel<int8_t><<<grid_size, block_size>>>(
                reinterpret_cast<const int8_t*>(handle_to_ptr(input_handle)),
                input_scale,
                handle_to_ptr(out_handle),
                outer,
                last_dim);
            break;
        default:
            set_error("CUDA typed softmax received unsupported input dtype");
            return 1;
    }
    if (!check_cuda_launch("CUDA typed softmax kernel launch failed")) {
        return 1;
    }
    return 0;
}

extern "C" int lumen_cuda_softmax_lastdim_backward_f32_device(
    uint64_t output_handle,
    uint64_t grad_handle,
    uint64_t out_handle,
    size_t outer,
    size_t last_dim) {
    if (!validate_handle(output_handle, "CUDA softmax backward output handle") ||
        !validate_handle(grad_handle, "CUDA softmax backward grad handle") ||
        !validate_handle(out_handle, "CUDA softmax backward result handle")) {
        return 1;
    }
    if (outer == 0 || last_dim == 0) {
        set_error("CUDA softmax backward dimensions must be greater than zero");
        return 1;
    }

#if LUMEN_HAS_CUDNN
    if (validate_int_dimensions(
            "CUDA softmax backward dimensions exceed cuDNN int range", {outer, last_dim})) {
        CudnnHandle handle;
        CudnnTensorDescriptor tensor_desc;
        if (!init_cudnn(handle) ||
            !init_tensor_descriptor_4d(
                tensor_desc,
                static_cast<int>(outer),
                static_cast<int>(last_dim),
                1,
                1)) {
            return 1;
        }

        const float alpha = 1.0f;
        const float beta = 0.0f;
        cudnnStatus_t status = cudnnSoftmaxBackward(
            handle.handle,
            CUDNN_SOFTMAX_ACCURATE,
            CUDNN_SOFTMAX_MODE_CHANNEL,
            &alpha,
            tensor_desc.desc,
            handle_to_ptr(output_handle),
            tensor_desc.desc,
            handle_to_ptr(grad_handle),
            &beta,
            tensor_desc.desc,
            handle_to_ptr(out_handle));
        if (status != CUDNN_STATUS_SUCCESS) {
            set_cudnn_error("cuDNN softmax backward failed", status);
            return 1;
        }
        return 0;
    }
    g_last_error.clear();
#endif

    constexpr int block_size = 128;
    const unsigned int grid_size = linear_grid_size(outer, block_size);
    softmax_lastdim_backward_kernel<<<grid_size, block_size>>>(
        handle_to_ptr(output_handle),
        handle_to_ptr(grad_handle),
        handle_to_ptr(out_handle),
        outer,
        last_dim);
    if (!check_cuda_launch("CUDA softmax backward kernel launch failed")) {
        return 1;
    }
    return 0;
}

extern "C" int lumen_cuda_fused_softmax_f32_device(
    uint64_t input_handle,
    uint64_t out_handle,
    size_t batch_heads,
    size_t q_len,
    size_t k_len,
    float scale,
    int is_causal) {
    if (!validate_handle(input_handle, "CUDA fused_softmax input handle") ||
        !validate_handle(out_handle, "CUDA fused_softmax output handle")) {
        return 1;
    }
    if (batch_heads == 0 || q_len == 0 || k_len == 0) {
        set_error("CUDA fused_softmax dimensions must be greater than zero");
        return 1;
    }
    if (!validate_finite_value(scale, "CUDA fused_softmax scale must be finite")) {
        return 1;
    }

    size_t rows = 0;
    if (!checked_product("CUDA fused_softmax row count overflow", {batch_heads, q_len}, &rows)) {
        return 1;
    }
    constexpr int block_size = 256;
    if (k_len >= 64) {
        const unsigned int grid_size = linear_grid_size(rows, 1);
        fused_softmax_block_kernel<<<grid_size, block_size>>>(
            handle_to_ptr(input_handle),
            handle_to_ptr(out_handle),
            rows,
            q_len,
            k_len,
            scale,
            is_causal);
    } else {
        const unsigned int grid_size = linear_grid_size(rows, block_size);
        fused_softmax_kernel<<<grid_size, block_size>>>(
            handle_to_ptr(input_handle),
            handle_to_ptr(out_handle),
            rows,
            q_len,
            k_len,
            scale,
            is_causal);
    }
    if (!check_cuda_launch("CUDA fused_softmax kernel launch failed")) {
        return 1;
    }
    return 0;
}

extern "C" int lumen_cuda_fused_softmax_backward_f32_device(
    uint64_t output_handle,
    uint64_t grad_handle,
    uint64_t out_handle,
    size_t batch_heads,
    size_t q_len,
    size_t k_len,
    float scale) {
    if (!validate_handle(output_handle, "CUDA fused_softmax backward output handle") ||
        !validate_handle(grad_handle, "CUDA fused_softmax backward grad handle") ||
        !validate_handle(out_handle, "CUDA fused_softmax backward result handle")) {
        return 1;
    }
    if (batch_heads == 0 || q_len == 0 || k_len == 0) {
        set_error("CUDA fused_softmax backward dimensions must be greater than zero");
        return 1;
    }
    if (!validate_finite_value(scale, "CUDA fused_softmax backward scale must be finite")) {
        return 1;
    }

    size_t rows = 0;
    if (!checked_product(
            "CUDA fused_softmax backward row count overflow", {batch_heads, q_len}, &rows)) {
        return 1;
    }
    constexpr int block_size = 256;
    if (k_len >= 64) {
        const unsigned int grid_size = linear_grid_size(rows, 1);
        fused_softmax_backward_block_kernel<<<grid_size, block_size>>>(
            handle_to_ptr(output_handle),
            handle_to_ptr(grad_handle),
            handle_to_ptr(out_handle),
            rows,
            k_len,
            scale);
    } else {
        const unsigned int grid_size = linear_grid_size(rows, block_size);
        fused_softmax_backward_kernel<<<grid_size, block_size>>>(
            handle_to_ptr(output_handle),
            handle_to_ptr(grad_handle),
            handle_to_ptr(out_handle),
            rows,
            k_len,
            scale);
    }
    if (!check_cuda_launch("CUDA fused_softmax backward kernel launch failed")) {
        return 1;
    }
    return 0;
}

extern "C" int lumen_cuda_fused_softmax_f32_with_past_device(
    uint64_t input_handle,
    uint64_t out_handle,
    size_t batch_heads,
    size_t q_len,
    size_t k_len,
    float scale,
    int is_causal,
    size_t past_len) {
    if (!validate_handle(input_handle, "CUDA fused_softmax_with_past input handle") ||
        !validate_handle(out_handle, "CUDA fused_softmax_with_past output handle")) {
        return 1;
    }
    if (batch_heads == 0 || q_len == 0 || k_len == 0) {
        set_error("CUDA fused_softmax_with_past dimensions must be greater than zero");
        return 1;
    }
    if (!validate_finite_value(scale, "CUDA fused_softmax_with_past scale must be finite")) {
        return 1;
    }
    if (past_len > k_len || q_len > k_len - past_len) {
        set_error("CUDA fused_softmax_with_past causal window exceeds key length");
        return 1;
    }

    size_t rows = 0;
    if (!checked_product(
            "CUDA fused_softmax_with_past row count overflow", {batch_heads, q_len}, &rows)) {
        return 1;
    }
    constexpr int block_size = 128;
    const unsigned int grid_size = linear_grid_size(rows, block_size);
    fused_softmax_with_past_kernel<<<grid_size, block_size>>>(
        handle_to_ptr(input_handle),
        handle_to_ptr(out_handle),
        rows,
        q_len,
        k_len,
        scale,
        is_causal,
        past_len);
    if (!check_cuda_launch("CUDA fused_softmax_with_past kernel launch failed")) {
        return 1;
    }
    return 0;
}

extern "C" int lumen_cuda_embedding_f32_device(
    uint64_t indices_handle,
    uint64_t weight_handle,
    uint64_t out_handle,
    size_t num_indices,
    size_t vocab_size,
    size_t embed_dim) {
    if (!validate_handle(indices_handle, "CUDA embedding indices handle") ||
        !validate_handle(weight_handle, "CUDA embedding weight handle") ||
        !validate_handle(out_handle, "CUDA embedding output handle")) {
        return 1;
    }
    if (num_indices == 0 || vocab_size == 0 || embed_dim == 0) {
        set_error("CUDA embedding dimensions must be greater than zero");
        return 1;
    }

    int* d_status = prepare_embedding_status("failed to prepare CUDA embedding status buffer");
    if (d_status == nullptr) {
        return 1;
    }

    constexpr int block_size = 256;
    size_t total = 0;
    if (!checked_product("CUDA embedding output length overflow", {num_indices, embed_dim}, &total)) {
        return 1;
    }
    const unsigned int grid_size = linear_grid_size(total, block_size);
    embedding_kernel<<<grid_size, block_size>>>(
        handle_to_ptr(indices_handle),
        handle_to_ptr(weight_handle),
        handle_to_ptr(out_handle),
        num_indices,
        vocab_size,
        embed_dim,
        d_status);
    if (!check_cuda_launch("CUDA embedding kernel launch failed")) {
        return 1;
    }
    if (!read_embedding_status(
            d_status,
            "failed to read CUDA embedding status buffer",
            "CUDA embedding encountered a non-finite, negative, or fractional index",
            "CUDA embedding encountered an out-of-bounds index")) {
        return 1;
    }
    return 0;
}

extern "C" int lumen_cuda_embedding_typed_device(
    uint64_t indices_handle,
    uint64_t weight_handle,
    int weight_dtype,
    float weight_scale,
    uint64_t out_handle,
    size_t num_indices,
    size_t vocab_size,
    size_t embed_dim) {
    if (!validate_handle(indices_handle, "CUDA typed embedding indices handle") ||
        !validate_handle(weight_handle, "CUDA typed embedding weight handle") ||
        !validate_handle(out_handle, "CUDA typed embedding output handle")) {
        return 1;
    }
    if (num_indices == 0 || vocab_size == 0 || embed_dim == 0) {
        set_error("CUDA typed embedding dimensions must be greater than zero");
        return 1;
    }

    int* d_status =
        prepare_embedding_status("failed to prepare CUDA typed embedding status buffer");
    if (d_status == nullptr) {
        return 1;
    }

    constexpr int block_size = 256;
    size_t total = 0;
    if (!checked_product(
            "CUDA typed embedding output length overflow", {num_indices, embed_dim}, &total)) {
        return 1;
    }
    const unsigned int grid_size = linear_grid_size(total, block_size);
    switch (weight_dtype) {
        case kDTypeF32:
            embedding_typed_kernel<float><<<grid_size, block_size>>>(
                handle_to_ptr(indices_handle),
                handle_to_ptr(weight_handle),
                weight_scale,
                handle_to_ptr(out_handle),
                num_indices,
                vocab_size,
                embed_dim,
                d_status);
            break;
        case kDTypeF16:
            embedding_typed_kernel<__half><<<grid_size, block_size>>>(
                handle_to_ptr(indices_handle),
                reinterpret_cast<const __half*>(handle_to_ptr(weight_handle)),
                weight_scale,
                handle_to_ptr(out_handle),
                num_indices,
                vocab_size,
                embed_dim,
                d_status);
            break;
        case kDTypeBF16:
            embedding_typed_kernel<__nv_bfloat16><<<grid_size, block_size>>>(
                handle_to_ptr(indices_handle),
                reinterpret_cast<const __nv_bfloat16*>(handle_to_ptr(weight_handle)),
                weight_scale,
                handle_to_ptr(out_handle),
                num_indices,
                vocab_size,
                embed_dim,
                d_status);
            break;
        case kDTypeI8:
            embedding_typed_kernel<int8_t><<<grid_size, block_size>>>(
                handle_to_ptr(indices_handle),
                reinterpret_cast<const int8_t*>(handle_to_ptr(weight_handle)),
                weight_scale,
                handle_to_ptr(out_handle),
                num_indices,
                vocab_size,
                embed_dim,
                d_status);
            break;
        default:
            set_error("CUDA typed embedding received unsupported weight dtype");
            return 1;
    }
    if (!check_cuda_launch("CUDA typed embedding kernel launch failed")) {
        return 1;
    }
    if (!read_embedding_status(
            d_status,
            "failed to read CUDA typed embedding status buffer",
            "CUDA typed embedding encountered a non-finite, negative, or fractional index",
            "CUDA typed embedding encountered an out-of-bounds index")) {
        return 1;
    }
    return 0;
}

extern "C" int lumen_cuda_embedding_typed_same_dtype_device(
    uint64_t indices_handle,
    uint64_t weight_handle,
    int weight_dtype,
    uint64_t out_handle,
    size_t num_indices,
    size_t vocab_size,
    size_t embed_dim) {
    if (!validate_handle(indices_handle, "CUDA native embedding indices handle") ||
        !validate_handle(weight_handle, "CUDA native embedding weight handle") ||
        !validate_handle(out_handle, "CUDA native embedding output handle")) {
        return 1;
    }
    if (num_indices == 0 || vocab_size == 0 || embed_dim == 0) {
        set_error("CUDA native embedding dimensions must be greater than zero");
        return 1;
    }

    int* d_status =
        prepare_embedding_status("failed to prepare CUDA native embedding status buffer");
    if (d_status == nullptr) {
        return 1;
    }

    constexpr int block_size = 256;
    size_t total = 0;
    if (!checked_product(
            "CUDA native embedding output length overflow", {num_indices, embed_dim}, &total)) {
        return 1;
    }
    const unsigned int grid_size = linear_grid_size(total, block_size);
    switch (weight_dtype) {
        case kDTypeF16:
            embedding_typed_same_dtype_kernel<__half><<<grid_size, block_size>>>(
                handle_to_ptr(indices_handle),
                reinterpret_cast<const __half*>(handle_to_ptr(weight_handle)),
                reinterpret_cast<__half*>(handle_to_ptr(out_handle)),
                num_indices,
                vocab_size,
                embed_dim,
                d_status);
            break;
        case kDTypeBF16:
            embedding_typed_same_dtype_kernel<__nv_bfloat16><<<grid_size, block_size>>>(
                handle_to_ptr(indices_handle),
                reinterpret_cast<const __nv_bfloat16*>(handle_to_ptr(weight_handle)),
                reinterpret_cast<__nv_bfloat16*>(handle_to_ptr(out_handle)),
                num_indices,
                vocab_size,
                embed_dim,
                d_status);
            break;
        case kDTypeI8:
            embedding_typed_same_dtype_kernel<int8_t><<<grid_size, block_size>>>(
                handle_to_ptr(indices_handle),
                reinterpret_cast<const int8_t*>(handle_to_ptr(weight_handle)),
                reinterpret_cast<int8_t*>(handle_to_ptr(out_handle)),
                num_indices,
                vocab_size,
                embed_dim,
                d_status);
            break;
        default:
            set_error("CUDA native embedding received unsupported weight dtype");
            return 1;
    }
    if (!check_cuda_launch("CUDA native embedding kernel launch failed")) {
        return 1;
    }
    if (!read_embedding_status(
            d_status,
            "failed to read CUDA native embedding status buffer",
            "CUDA native embedding encountered a non-finite, negative, or fractional index",
            "CUDA native embedding encountered an out-of-bounds index")) {
        return 1;
    }
    return 0;
}

extern "C" int lumen_cuda_embedding_backward_f32_device(
    uint64_t indices_handle,
    uint64_t grad_handle,
    uint64_t grad_weight_handle,
    size_t num_indices,
    size_t vocab_size,
    size_t embed_dim) {
    if (!validate_handle(indices_handle, "CUDA embedding backward indices handle") ||
        !validate_handle(grad_handle, "CUDA embedding backward grad handle") ||
        !validate_handle(grad_weight_handle, "CUDA embedding backward weight grad handle")) {
        return 1;
    }
    if (num_indices == 0 || vocab_size == 0 || embed_dim == 0) {
        set_error("CUDA embedding backward dimensions must be greater than zero");
        return 1;
    }

    size_t grad_weight_len = 0;
    size_t grad_weight_bytes = 0;
    size_t total = 0;
    if (!checked_product(
            "CUDA embedding backward weight grad length overflow",
            {vocab_size, embed_dim},
            &grad_weight_len) ||
        !checked_byte_length(
            grad_weight_len,
            sizeof(float),
            "CUDA embedding backward weight grad byte length overflow",
            &grad_weight_bytes) ||
        !checked_product(
            "CUDA embedding backward input grad length overflow",
            {num_indices, embed_dim},
            &total)) {
        return 1;
    }
    cudaError_t memset_status =
        cudaMemset(handle_to_ptr(grad_weight_handle), 0, grad_weight_bytes);
    if (memset_status != cudaSuccess) {
        set_cuda_error("CUDA embedding backward weight grad initialization failed", memset_status);
        return 1;
    }

    int* d_status =
        prepare_embedding_status("failed to prepare CUDA embedding backward status buffer");
    if (d_status == nullptr) {
        return 1;
    }

    constexpr int block_size = 256;
    const unsigned int grid_size = linear_grid_size(total, block_size);
    embedding_backward_kernel<<<grid_size, block_size>>>(
        handle_to_ptr(indices_handle),
        handle_to_ptr(grad_handle),
        handle_to_ptr(grad_weight_handle),
        num_indices,
        vocab_size,
        embed_dim,
        d_status);
    if (!check_cuda_launch("CUDA embedding backward kernel launch failed")) {
        return 1;
    }
    if (!read_embedding_status(
            d_status,
            "failed to read CUDA embedding backward status buffer",
            "CUDA embedding backward encountered a non-finite, negative, or fractional index",
            "CUDA embedding backward encountered an out-of-bounds index")) {
        return 1;
    }
    return 0;
}

extern "C" int lumen_cuda_rms_norm_f32_device(
    uint64_t input_handle,
    uint64_t weight_handle,
    uint64_t out_handle,
    size_t rows,
    size_t dim,
    float eps) {
    if (!validate_handle(input_handle, "CUDA RMSNorm input handle") ||
        !validate_handle(weight_handle, "CUDA RMSNorm weight handle") ||
        !validate_handle(out_handle, "CUDA RMSNorm output handle")) {
        return 1;
    }
    if (rows == 0 || dim == 0) {
        set_error("CUDA RMSNorm dimensions must be greater than zero");
        return 1;
    }
    if (!validate_positive_finite_value(eps, "CUDA RMSNorm epsilon must be finite and > 0")) {
        return 1;
    }

    constexpr int block_size = 128;
    const unsigned int grid_size = linear_grid_size(rows, block_size);
    rms_norm_kernel<<<grid_size, block_size>>>(
        handle_to_ptr(input_handle),
        handle_to_ptr(weight_handle),
        handle_to_ptr(out_handle),
        rows,
        dim,
        eps);
    if (!check_cuda_launch("CUDA RMSNorm kernel launch failed")) {
        return 1;
    }
    return 0;
}

extern "C" int lumen_cuda_rms_norm_typed_device(
    uint64_t input_handle,
    int input_dtype,
    float input_scale,
    uint64_t weight_handle,
    int weight_dtype,
    float weight_scale,
    uint64_t out_handle,
    size_t rows,
    size_t dim,
    float eps) {
    if (!validate_handle(input_handle, "CUDA typed RMSNorm input handle") ||
        !validate_handle(weight_handle, "CUDA typed RMSNorm weight handle") ||
        !validate_handle(out_handle, "CUDA typed RMSNorm output handle")) {
        return 1;
    }
    if (rows == 0 || dim == 0) {
        set_error("CUDA typed RMSNorm dimensions must be greater than zero");
        return 1;
    }
    if (!validate_positive_finite_value(
            eps, "CUDA typed RMSNorm epsilon must be finite and > 0")) {
        return 1;
    }

    float* out = handle_to_ptr(out_handle);
    switch (input_dtype) {
        case kDTypeF32:
            return dispatch_rms_norm_weight_typed(
                handle_to_ptr(input_handle), input_scale, weight_handle, weight_dtype, weight_scale, out, rows, dim, eps);
        case kDTypeF16:
            return dispatch_rms_norm_weight_typed(
                reinterpret_cast<const __half*>(handle_to_ptr(input_handle)),
                input_scale,
                weight_handle,
                weight_dtype,
                weight_scale,
                out,
                rows,
                dim,
                eps);
        case kDTypeBF16:
            return dispatch_rms_norm_weight_typed(
                reinterpret_cast<const __nv_bfloat16*>(handle_to_ptr(input_handle)),
                input_scale,
                weight_handle,
                weight_dtype,
                weight_scale,
                out,
                rows,
                dim,
                eps);
        case kDTypeI8:
            return dispatch_rms_norm_weight_typed(
                reinterpret_cast<const int8_t*>(handle_to_ptr(input_handle)),
                input_scale,
                weight_handle,
                weight_dtype,
                weight_scale,
                out,
                rows,
                dim,
                eps);
        default:
            set_error("CUDA typed RMSNorm received unsupported input dtype");
            return 1;
    }
}

extern "C" int lumen_cuda_rms_norm_i8_typed_out_device(
    uint64_t input_handle,
    int input_dtype,
    float input_scale,
    uint64_t weight_handle,
    int weight_dtype,
    float weight_scale,
    uint64_t out_handle,
    size_t rows,
    size_t dim,
    float eps,
    float* out_scale) {
    if (!validate_handle(input_handle, "CUDA I8 typed-output RMSNorm input handle") ||
        !validate_handle(weight_handle, "CUDA I8 typed-output RMSNorm weight handle") ||
        !validate_handle(out_handle, "CUDA I8 typed-output RMSNorm output handle")) {
        return 1;
    }
    if (out_scale == nullptr) {
        set_error("CUDA I8 typed-output RMSNorm scale output is null");
        return 1;
    }
    if (rows == 0 || dim == 0) {
        set_error("CUDA I8 typed-output RMSNorm dimensions must be greater than zero");
        return 1;
    }
    if (!validate_positive_finite_value(
            eps, "CUDA I8 typed-output RMSNorm epsilon must be finite and > 0")) {
        return 1;
    }

    int8_t* out = reinterpret_cast<int8_t*>(handle_to_ptr(out_handle));
    switch (input_dtype) {
        case kDTypeF32:
            return dispatch_rms_norm_i8_typed_out_weight(
                handle_to_ptr(input_handle), input_scale, weight_handle, weight_dtype, weight_scale, out, rows, dim, eps, out_scale);
        case kDTypeF16:
            return dispatch_rms_norm_i8_typed_out_weight(
                reinterpret_cast<const __half*>(handle_to_ptr(input_handle)),
                input_scale,
                weight_handle,
                weight_dtype,
                weight_scale,
                out,
                rows,
                dim,
                eps,
                out_scale);
        case kDTypeBF16:
            return dispatch_rms_norm_i8_typed_out_weight(
                reinterpret_cast<const __nv_bfloat16*>(handle_to_ptr(input_handle)),
                input_scale,
                weight_handle,
                weight_dtype,
                weight_scale,
                out,
                rows,
                dim,
                eps,
                out_scale);
        case kDTypeI8:
            return dispatch_rms_norm_i8_typed_out_weight(
                reinterpret_cast<const int8_t*>(handle_to_ptr(input_handle)),
                input_scale,
                weight_handle,
                weight_dtype,
                weight_scale,
                out,
                rows,
                dim,
                eps,
                out_scale);
        default:
            set_error("CUDA I8 typed-output RMSNorm received unsupported input dtype");
            return 1;
    }
}

extern "C" int lumen_cuda_rms_norm_backward_f32_device(
    uint64_t input_handle,
    uint64_t weight_handle,
    uint64_t grad_handle,
    uint64_t grad_input_handle,
    uint64_t grad_weight_handle,
    size_t rows,
    size_t dim,
    float eps) {
    if (!validate_handle(input_handle, "CUDA RMSNorm backward input handle") ||
        !validate_handle(weight_handle, "CUDA RMSNorm backward weight handle") ||
        !validate_handle(grad_handle, "CUDA RMSNorm backward grad handle") ||
        !validate_handle(grad_input_handle, "CUDA RMSNorm backward input grad handle") ||
        !validate_handle(grad_weight_handle, "CUDA RMSNorm backward weight grad handle")) {
        return 1;
    }
    if (rows == 0 || dim == 0) {
        set_error("CUDA RMSNorm backward dimensions must be greater than zero");
        return 1;
    }
    if (!validate_positive_finite_value(
            eps, "CUDA RMSNorm backward epsilon must be finite and > 0")) {
        return 1;
    }

    if (!zero_f32_buffer(
            handle_to_ptr(grad_weight_handle),
            dim,
            "CUDA RMSNorm backward weight grad initialization failed")) {
        return 1;
    }

    constexpr int block_size = 128;
    const unsigned int grid_size = linear_grid_size(rows, block_size);
    rms_norm_backward_kernel<<<grid_size, block_size>>>(
        handle_to_ptr(input_handle),
        handle_to_ptr(weight_handle),
        handle_to_ptr(grad_handle),
        handle_to_ptr(grad_input_handle),
        handle_to_ptr(grad_weight_handle),
        rows,
        dim,
        eps);
    if (!check_cuda_launch("CUDA RMSNorm backward kernel launch failed")) {
        return 1;
    }
    return 0;
}

extern "C" int lumen_cuda_rms_norm_backward_typed_device(
    uint64_t input_handle,
    int input_dtype,
    float input_scale,
    uint64_t weight_handle,
    int weight_dtype,
    float weight_scale,
    uint64_t grad_handle,
    uint64_t grad_input_handle,
    uint64_t grad_weight_handle,
    size_t rows,
    size_t dim,
    float eps) {
    if (!validate_handle(input_handle, "CUDA typed RMSNorm backward input handle") ||
        !validate_handle(weight_handle, "CUDA typed RMSNorm backward weight handle") ||
        !validate_handle(grad_handle, "CUDA typed RMSNorm backward grad handle") ||
        !validate_handle(grad_input_handle, "CUDA typed RMSNorm backward input grad handle") ||
        !validate_handle(grad_weight_handle, "CUDA typed RMSNorm backward weight grad handle")) {
        return 1;
    }
    if (rows == 0 || dim == 0) {
        set_error("CUDA typed RMSNorm backward dimensions must be greater than zero");
        return 1;
    }
    if (!validate_positive_finite_value(
            eps, "CUDA typed RMSNorm backward epsilon must be finite and > 0")) {
        return 1;
    }

    const float* grad = handle_to_ptr(grad_handle);
    float* grad_input = handle_to_ptr(grad_input_handle);
    float* grad_weight = handle_to_ptr(grad_weight_handle);
    switch (input_dtype) {
        case kDTypeF32:
            return dispatch_rms_norm_backward_weight_typed(
                handle_to_ptr(input_handle),
                input_scale,
                weight_handle,
                weight_dtype,
                weight_scale,
                grad,
                grad_input,
                grad_weight,
                rows,
                dim,
                eps);
        case kDTypeF16:
            return dispatch_rms_norm_backward_weight_typed(
                reinterpret_cast<const __half*>(handle_to_ptr(input_handle)),
                input_scale,
                weight_handle,
                weight_dtype,
                weight_scale,
                grad,
                grad_input,
                grad_weight,
                rows,
                dim,
                eps);
        case kDTypeBF16:
            return dispatch_rms_norm_backward_weight_typed(
                reinterpret_cast<const __nv_bfloat16*>(handle_to_ptr(input_handle)),
                input_scale,
                weight_handle,
                weight_dtype,
                weight_scale,
                grad,
                grad_input,
                grad_weight,
                rows,
                dim,
                eps);
        case kDTypeI8:
            return dispatch_rms_norm_backward_weight_typed(
                reinterpret_cast<const int8_t*>(handle_to_ptr(input_handle)),
                input_scale,
                weight_handle,
                weight_dtype,
                weight_scale,
                grad,
                grad_input,
                grad_weight,
                rows,
                dim,
                eps);
        default:
            set_error("CUDA typed RMSNorm backward received unsupported input dtype");
            return 1;
    }
}
