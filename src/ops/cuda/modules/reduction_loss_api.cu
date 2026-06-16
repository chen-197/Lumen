extern "C" int lumen_cuda_sum_f32_device(
    uint64_t input_handle,
    uint64_t out_handle,
    size_t len) {
    if (!validate_handle(input_handle, "CUDA sum input handle") ||
        !validate_handle(out_handle, "CUDA sum output handle")) {
        return 1;
    }
    if (len == 0) {
        set_error("CUDA sum length must be greater than zero");
        return 1;
    }

    cudaError_t memset_status = cudaMemset(handle_to_ptr(out_handle), 0, sizeof(float));
    if (memset_status != cudaSuccess) {
        set_cuda_error("CUDA sum output initialization failed", memset_status);
        return 1;
    }

    constexpr int block_size = 256;
    const unsigned int grid_size = std::min(linear_grid_size(len, block_size), 1024u);
    sum_kernel<<<grid_size, block_size>>>(handle_to_ptr(input_handle), handle_to_ptr(out_handle), len);
    if (!check_cuda_launch("CUDA sum kernel launch failed")) {
        return 1;
    }
    return 0;
}
extern "C" int lumen_cuda_sum_f16_device(
    uint64_t input_handle,
    uint64_t out_handle,
    size_t len) {
    if (!validate_handle(input_handle, "CUDA F16 sum input handle") ||
        !validate_handle(out_handle, "CUDA F16 sum output handle")) {
        return 1;
    }
    if (len == 0) {
        set_error("CUDA F16 sum length must be greater than zero");
        return 1;
    }

    cudaError_t memset_status = cudaMemset(handle_to_ptr(out_handle), 0, sizeof(float));
    if (memset_status != cudaSuccess) {
        set_cuda_error("CUDA F16 sum output initialization failed", memset_status);
        return 1;
    }

    constexpr int block_size = 256;
    const unsigned int grid_size = std::min(linear_grid_size(len, block_size), 1024u);
    sum_lowp_kernel<__half><<<grid_size, block_size>>>(
        reinterpret_cast<const __half*>(handle_to_ptr(input_handle)),
        handle_to_ptr(out_handle),
        len);
    return check_cuda_launch("CUDA F16 sum kernel launch failed") ? 0 : 1;
}

extern "C" int lumen_cuda_sum_bf16_device(
    uint64_t input_handle,
    uint64_t out_handle,
    size_t len) {
    if (!validate_handle(input_handle, "CUDA BF16 sum input handle") ||
        !validate_handle(out_handle, "CUDA BF16 sum output handle")) {
        return 1;
    }
    if (len == 0) {
        set_error("CUDA BF16 sum length must be greater than zero");
        return 1;
    }

    cudaError_t memset_status = cudaMemset(handle_to_ptr(out_handle), 0, sizeof(float));
    if (memset_status != cudaSuccess) {
        set_cuda_error("CUDA BF16 sum output initialization failed", memset_status);
        return 1;
    }

    constexpr int block_size = 256;
    const unsigned int grid_size = std::min(linear_grid_size(len, block_size), 1024u);
    sum_lowp_kernel<__nv_bfloat16><<<grid_size, block_size>>>(
        reinterpret_cast<const __nv_bfloat16*>(handle_to_ptr(input_handle)),
        handle_to_ptr(out_handle),
        len);
    return check_cuda_launch("CUDA BF16 sum kernel launch failed") ? 0 : 1;
}

extern "C" int lumen_cuda_sum_i8_device(
    uint64_t input_handle,
    float scale,
    uint64_t out_handle,
    size_t len) {
    if (!validate_handle(input_handle, "CUDA I8 sum input handle") ||
        !validate_handle(out_handle, "CUDA I8 sum output handle")) {
        return 1;
    }
    if (len == 0) {
        set_error("CUDA I8 sum length must be greater than zero");
        return 1;
    }
    if (!std::isfinite(scale) || scale <= 0.0f) {
        set_error("CUDA I8 sum scale must be finite and > 0");
        return 1;
    }

    constexpr int block_size = 256;
    const unsigned int grid_size = std::min(linear_grid_size(len, block_size), 1024u);
    thread_local ReusableCudaWorkspace partial_workspace;
    if (!partial_workspace.ensure(
            static_cast<size_t>(grid_size) * sizeof(long long),
            "failed to prepare CUDA I8 sum partial buffer")) {
        return 1;
    }
    long long* partials = static_cast<long long*>(partial_workspace.ptr);

    sum_i8_partials_kernel<<<grid_size, block_size>>>(
        reinterpret_cast<const int8_t*>(handle_to_ptr(input_handle)),
        partials,
        len);
    if (!check_cuda_launch("CUDA I8 sum partial kernel launch failed")) {
        return 1;
    }

    sum_i8_finalize_kernel<<<1, block_size>>>(
        partials,
        scale,
        handle_to_ptr(out_handle),
        static_cast<size_t>(grid_size));
    return check_cuda_launch("CUDA I8 sum finalize kernel launch failed") ? 0 : 1;
}

extern "C" int lumen_cuda_fill_scalar_f32_device(
    uint64_t out_handle,
    size_t len,
    float value) {
    if (!validate_handle(out_handle, "CUDA fill output handle")) {
        return 1;
    }
    if (len == 0) {
        return 0;
    }

    constexpr int block_size = 256;
    size_t vec_len = len / 4;
    if (vec_len > 0) {
        const unsigned int grid_size = linear_grid_size(vec_len, block_size);
        fill_scalar_vec4_kernel<<<grid_size, block_size>>>(
            reinterpret_cast<float4*>(handle_to_ptr(out_handle)),
            vec_len,
            len,
            value);
    } else {
        const unsigned int grid_size = linear_grid_size(len, block_size);
        fill_scalar_kernel<<<grid_size, block_size>>>(handle_to_ptr(out_handle), len, value);
    }
    if (!check_cuda_launch("CUDA fill kernel launch failed")) {
        return 1;
    }
    return 0;
}

extern "C" int lumen_cuda_add_inplace_f32_device(
    uint64_t dst_handle,
    uint64_t src_handle,
    size_t len) {
    if (!validate_handle(dst_handle, "CUDA add_inplace destination handle") ||
        !validate_handle(src_handle, "CUDA add_inplace source handle")) {
        return 1;
    }
    if (len == 0) {
        return 0;
    }

    constexpr int block_size = 256;
    size_t vec_len = len / 4;
    if (vec_len > 0) {
        const unsigned int grid_size = linear_grid_size(vec_len, block_size);
        add_inplace_vec4_kernel<<<grid_size, block_size>>>(
            reinterpret_cast<float4*>(handle_to_ptr(dst_handle)),
            reinterpret_cast<const float4*>(handle_to_ptr(src_handle)),
            vec_len,
            len);
    } else {
        const unsigned int grid_size = linear_grid_size(len, block_size);
        add_inplace_kernel<<<grid_size, block_size>>>(
            handle_to_ptr(dst_handle),
            handle_to_ptr(src_handle),
            len);
    }
    if (!check_cuda_launch("CUDA add_inplace kernel launch failed")) {
        return 1;
    }
    return 0;
}

extern "C" int lumen_cuda_sum_lastdim_f32_device(
    uint64_t input_handle,
    uint64_t out_handle,
    size_t rows,
    size_t last_dim) {
    if (!validate_handle(input_handle, "CUDA sum_lastdim input handle") ||
        !validate_handle(out_handle, "CUDA sum_lastdim output handle")) {
        return 1;
    }
    if (rows == 0 || last_dim == 0) {
        set_error("CUDA sum_lastdim received invalid dimensions");
        return 1;
    }
    size_t len = 0;
    if (!checked_product("CUDA sum_lastdim length overflow", {rows, last_dim}, &len)) {
        return 1;
    }
    constexpr int block_size = 256;
    if (rows < 64) {
        if (!zero_f32_buffer(
                handle_to_ptr(out_handle),
                last_dim,
                "CUDA sum_lastdim output initialization failed")) {
            return 1;
        }
        const unsigned int grid_size = linear_grid_size(len, block_size);
        sum_lastdim_atomic_kernel<<<grid_size, block_size>>>(
            handle_to_ptr(input_handle),
            handle_to_ptr(out_handle),
            len,
            last_dim);
    } else {
        const unsigned int grid_size = linear_grid_size(last_dim, 1);
        sum_lastdim_kernel<<<grid_size, block_size>>>(
            handle_to_ptr(input_handle),
            handle_to_ptr(out_handle),
            rows,
            last_dim);
    }
    return check_cuda_launch("CUDA sum_lastdim kernel launch failed") ? 0 : 1;
}

extern "C" int lumen_cuda_bshd_to_bhsd_add_bias_f32_device(
    uint64_t input_handle,
    uint64_t bias_handle,
    uint64_t out_handle,
    size_t batch,
    size_t seq,
    size_t heads,
    size_t dim) {
    if (!validate_handle(input_handle, "CUDA BSHD to BHSD input handle") ||
        !validate_handle(bias_handle, "CUDA BSHD to BHSD bias handle") ||
        !validate_handle(out_handle, "CUDA BSHD to BHSD output handle")) {
        return 1;
    }
    if (batch == 0 || seq == 0 || heads == 0 || dim == 0) {
        set_error("CUDA BSHD to BHSD add bias received invalid dimensions");
        return 1;
    }

    size_t len = 0;
    if (!checked_product(
            "CUDA BSHD to BHSD add bias length overflow",
            {batch, seq, heads, dim},
            &len)) {
        return 1;
    }
    constexpr int block_size = 256;
    if (dim % 4 == 0) {
        size_t vec_len = len / 4;
        const unsigned int grid_size = linear_grid_size(vec_len, block_size);
        bshd_to_bhsd_add_bias_vec4_kernel<<<grid_size, block_size>>>(
            reinterpret_cast<const float4*>(handle_to_ptr(input_handle)),
            reinterpret_cast<const float4*>(handle_to_ptr(bias_handle)),
            reinterpret_cast<float4*>(handle_to_ptr(out_handle)),
            seq,
            heads,
            dim,
            vec_len);
    } else {
        const unsigned int grid_size = linear_grid_size(len, block_size);
        bshd_to_bhsd_add_bias_kernel<<<grid_size, block_size>>>(
            handle_to_ptr(input_handle),
            handle_to_ptr(bias_handle),
            handle_to_ptr(out_handle),
            seq,
            heads,
            dim,
            len);
    }
    return check_cuda_launch("CUDA BSHD to BHSD add bias kernel launch failed") ? 0 : 1;
}

template <typename OutputT, typename TargetT>
int launch_mse_forward_typed(
    uint64_t output_handle,
    float output_scale,
    uint64_t target_handle,
    float target_scale,
    uint64_t diff_handle,
    uint64_t loss_handle,
    size_t len) {
    cudaError_t memset_status = cudaMemset(handle_to_ptr(loss_handle), 0, sizeof(float));
    if (memset_status != cudaSuccess) {
        set_cuda_error("CUDA typed MSE loss output initialization failed", memset_status);
        return 1;
    }

    constexpr int block_size = 256;
    const unsigned int grid_size = std::min(linear_grid_size(len, block_size), 1024u);
    mse_forward_typed_kernel<OutputT, TargetT><<<grid_size, block_size>>>(
        reinterpret_cast<const OutputT*>(handle_to_ptr(output_handle)),
        output_scale,
        reinterpret_cast<const TargetT*>(handle_to_ptr(target_handle)),
        target_scale,
        handle_to_ptr(diff_handle),
        handle_to_ptr(loss_handle),
        len,
        1.0f / static_cast<float>(len));
    return check_cuda_launch("CUDA typed MSE forward kernel launch failed") ? 0 : 1;
}

template <typename OutputT>
int dispatch_mse_forward_typed_target(
    uint64_t output_handle,
    float output_scale,
    uint64_t target_handle,
    int target_dtype,
    float target_scale,
    uint64_t diff_handle,
    uint64_t loss_handle,
    size_t len) {
    switch (target_dtype) {
        case kDTypeF32:
            return launch_mse_forward_typed<OutputT, float>(
                output_handle, output_scale, target_handle, target_scale, diff_handle, loss_handle, len);
        case kDTypeF16:
            return launch_mse_forward_typed<OutputT, __half>(
                output_handle, output_scale, target_handle, target_scale, diff_handle, loss_handle, len);
        case kDTypeBF16:
            return launch_mse_forward_typed<OutputT, __nv_bfloat16>(
                output_handle, output_scale, target_handle, target_scale, diff_handle, loss_handle, len);
        case kDTypeI8:
            return launch_mse_forward_typed<OutputT, int8_t>(
                output_handle, output_scale, target_handle, target_scale, diff_handle, loss_handle, len);
        default:
            set_error("unsupported target dtype for CUDA typed MSE forward");
            return 1;
    }
}

extern "C" int lumen_cuda_mse_forward_typed_device(
    uint64_t output_handle,
    int output_dtype,
    float output_scale,
    uint64_t target_handle,
    int target_dtype,
    float target_scale,
    uint64_t diff_handle,
    uint64_t loss_handle,
    size_t len) {
    if (!validate_handle(output_handle, "CUDA typed MSE output handle") ||
        !validate_handle(target_handle, "CUDA typed MSE target handle") ||
        !validate_handle(diff_handle, "CUDA typed MSE diff handle") ||
        !validate_handle(loss_handle, "CUDA typed MSE loss handle")) {
        return 1;
    }
    if (len == 0) {
        set_error("CUDA typed MSE length must be greater than zero");
        return 1;
    }
    if ((output_dtype == kDTypeI8 && (!std::isfinite(output_scale) || output_scale <= 0.0f)) ||
        (target_dtype == kDTypeI8 && (!std::isfinite(target_scale) || target_scale <= 0.0f))) {
        set_error("CUDA typed MSE I8 scales must be finite and > 0");
        return 1;
    }

    switch (output_dtype) {
        case kDTypeF32:
            return dispatch_mse_forward_typed_target<float>(
                output_handle, output_scale, target_handle, target_dtype, target_scale, diff_handle, loss_handle, len);
        case kDTypeF16:
            return dispatch_mse_forward_typed_target<__half>(
                output_handle, output_scale, target_handle, target_dtype, target_scale, diff_handle, loss_handle, len);
        case kDTypeBF16:
            return dispatch_mse_forward_typed_target<__nv_bfloat16>(
                output_handle, output_scale, target_handle, target_dtype, target_scale, diff_handle, loss_handle, len);
        case kDTypeI8:
            return dispatch_mse_forward_typed_target<int8_t>(
                output_handle, output_scale, target_handle, target_dtype, target_scale, diff_handle, loss_handle, len);
        default:
            set_error("unsupported output dtype for CUDA typed MSE forward");
            return 1;
    }
}

extern "C" int lumen_cuda_mse_backward_f32_device(
    uint64_t diff_handle,
    uint64_t grad_output_handle,
    uint64_t grad_target_handle,
    size_t len,
    float factor) {
    if (!validate_handle(diff_handle, "CUDA MSE backward diff handle") ||
        !validate_handle(grad_output_handle, "CUDA MSE backward output grad handle") ||
        !validate_handle(grad_target_handle, "CUDA MSE backward target grad handle")) {
        return 1;
    }

    constexpr int block_size = 256;
    const unsigned int grid_size = linear_grid_size(len, block_size);
    mse_backward_kernel<<<grid_size, block_size>>>(
        handle_to_ptr(diff_handle),
        handle_to_ptr(grad_output_handle),
        handle_to_ptr(grad_target_handle),
        len,
        factor);
    if (!check_cuda_launch("CUDA MSE backward kernel launch failed")) {
        return 1;
    }
    return 0;
}

extern "C" int lumen_cuda_cross_entropy_backward_f32_device(
    uint64_t softmax_handle,
    uint64_t target_handle,
    uint64_t out_handle,
    size_t len,
    float factor) {
    if (!validate_handle(softmax_handle, "CUDA cross_entropy backward softmax handle") ||
        !validate_handle(target_handle, "CUDA cross_entropy backward target handle") ||
        !validate_handle(out_handle, "CUDA cross_entropy backward output handle")) {
        return 1;
    }

    constexpr int block_size = 256;
    const unsigned int grid_size = linear_grid_size(len, block_size);
    cross_entropy_backward_kernel<<<grid_size, block_size>>>(
        handle_to_ptr(softmax_handle),
        handle_to_ptr(target_handle),
        handle_to_ptr(out_handle),
        len,
        factor);
    if (!check_cuda_launch("CUDA cross_entropy backward kernel launch failed")) {
        return 1;
    }
    return 0;
}

extern "C" int lumen_cuda_cross_entropy_loss_f32_device(
    uint64_t softmax_handle,
    uint64_t target_handle,
    uint64_t out_handle,
    size_t len,
    float factor) {
    if (!validate_handle(softmax_handle, "CUDA cross_entropy loss softmax handle") ||
        !validate_handle(target_handle, "CUDA cross_entropy loss target handle") ||
        !validate_handle(out_handle, "CUDA cross_entropy loss output handle")) {
        return 1;
    }
    if (len == 0) {
        set_error("CUDA cross_entropy loss length must be greater than zero");
        return 1;
    }

    cudaError_t memset_status = cudaMemset(handle_to_ptr(out_handle), 0, sizeof(float));
    if (memset_status != cudaSuccess) {
        set_cuda_error("CUDA cross_entropy loss output initialization failed", memset_status);
        return 1;
    }

    constexpr int block_size = 256;
    const unsigned int grid_size = std::min(linear_grid_size(len, block_size), 1024u);
    cross_entropy_loss_kernel<<<grid_size, block_size>>>(
        handle_to_ptr(softmax_handle),
        handle_to_ptr(target_handle),
        handle_to_ptr(out_handle),
        len,
        factor);
    if (!check_cuda_launch("CUDA cross_entropy loss kernel launch failed")) {
        return 1;
    }
    return 0;
}

template <typename TargetT>
int launch_cross_entropy_backward_typed_target(
    uint64_t softmax_handle,
    uint64_t target_handle,
    float target_scale,
    uint64_t out_handle,
    size_t len,
    float factor) {
    constexpr int block_size = 256;
    const unsigned int grid_size = linear_grid_size(len, block_size);
    cross_entropy_backward_typed_target_kernel<TargetT><<<grid_size, block_size>>>(
        handle_to_ptr(softmax_handle),
        reinterpret_cast<const TargetT*>(handle_to_ptr(target_handle)),
        target_scale,
        handle_to_ptr(out_handle),
        len,
        factor);
    return check_cuda_launch("CUDA typed target cross_entropy backward kernel launch failed") ? 0 : 1;
}

extern "C" int lumen_cuda_cross_entropy_backward_typed_target_device(
    uint64_t softmax_handle,
    uint64_t target_handle,
    int target_dtype,
    float target_scale,
    uint64_t out_handle,
    size_t len,
    float factor) {
    if (!validate_handle(softmax_handle, "CUDA typed target cross_entropy backward softmax handle") ||
        !validate_handle(target_handle, "CUDA typed target cross_entropy backward target handle") ||
        !validate_handle(out_handle, "CUDA typed target cross_entropy backward output handle")) {
        return 1;
    }
    if (target_dtype == kDTypeI8 && (!std::isfinite(target_scale) || target_scale <= 0.0f)) {
        set_error("CUDA typed target cross_entropy backward I8 scale must be finite and > 0");
        return 1;
    }

    switch (target_dtype) {
        case kDTypeF32:
            return launch_cross_entropy_backward_typed_target<float>(
                softmax_handle, target_handle, target_scale, out_handle, len, factor);
        case kDTypeF16:
            return launch_cross_entropy_backward_typed_target<__half>(
                softmax_handle, target_handle, target_scale, out_handle, len, factor);
        case kDTypeBF16:
            return launch_cross_entropy_backward_typed_target<__nv_bfloat16>(
                softmax_handle, target_handle, target_scale, out_handle, len, factor);
        case kDTypeI8:
            return launch_cross_entropy_backward_typed_target<int8_t>(
                softmax_handle, target_handle, target_scale, out_handle, len, factor);
        default:
            set_error("unsupported target dtype for CUDA typed target cross_entropy backward");
            return 1;
    }
}

template <typename TargetT>
int launch_cross_entropy_loss_typed_target(
    uint64_t softmax_handle,
    uint64_t target_handle,
    float target_scale,
    uint64_t out_handle,
    size_t len,
    float factor) {
    cudaError_t memset_status = cudaMemset(handle_to_ptr(out_handle), 0, sizeof(float));
    if (memset_status != cudaSuccess) {
        set_cuda_error("CUDA typed target cross_entropy loss output initialization failed", memset_status);
        return 1;
    }

    constexpr int block_size = 256;
    const unsigned int grid_size = std::min(linear_grid_size(len, block_size), 1024u);
    cross_entropy_loss_typed_target_kernel<TargetT><<<grid_size, block_size>>>(
        handle_to_ptr(softmax_handle),
        reinterpret_cast<const TargetT*>(handle_to_ptr(target_handle)),
        target_scale,
        handle_to_ptr(out_handle),
        len,
        factor);
    return check_cuda_launch("CUDA typed target cross_entropy loss kernel launch failed") ? 0 : 1;
}

extern "C" int lumen_cuda_cross_entropy_loss_typed_target_device(
    uint64_t softmax_handle,
    uint64_t target_handle,
    int target_dtype,
    float target_scale,
    uint64_t out_handle,
    size_t len,
    float factor) {
    if (!validate_handle(softmax_handle, "CUDA typed target cross_entropy loss softmax handle") ||
        !validate_handle(target_handle, "CUDA typed target cross_entropy loss target handle") ||
        !validate_handle(out_handle, "CUDA typed target cross_entropy loss output handle")) {
        return 1;
    }
    if (len == 0) {
        set_error("CUDA typed target cross_entropy loss length must be greater than zero");
        return 1;
    }
    if (target_dtype == kDTypeI8 && (!std::isfinite(target_scale) || target_scale <= 0.0f)) {
        set_error("CUDA typed target cross_entropy loss I8 scale must be finite and > 0");
        return 1;
    }

    switch (target_dtype) {
        case kDTypeF32:
            return launch_cross_entropy_loss_typed_target<float>(
                softmax_handle, target_handle, target_scale, out_handle, len, factor);
        case kDTypeF16:
            return launch_cross_entropy_loss_typed_target<__half>(
                softmax_handle, target_handle, target_scale, out_handle, len, factor);
        case kDTypeBF16:
            return launch_cross_entropy_loss_typed_target<__nv_bfloat16>(
                softmax_handle, target_handle, target_scale, out_handle, len, factor);
        case kDTypeI8:
            return launch_cross_entropy_loss_typed_target<int8_t>(
                softmax_handle, target_handle, target_scale, out_handle, len, factor);
        default:
            set_error("unsupported target dtype for CUDA typed target cross_entropy loss");
            return 1;
    }
}
