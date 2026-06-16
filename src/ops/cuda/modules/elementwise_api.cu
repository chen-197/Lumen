bool upload_packed_elementwise_metadata(
    const char* context,
    const size_t* const* host_arrays,
    size_t array_count,
    size_t len,
    size_t** device_arrays) {
    size_t element_count = 0;
    size_t bytes = 0;
    if (!checked_product(context, {array_count, len}, &element_count) ||
        !checked_byte_length(element_count, sizeof(size_t), context, &bytes)) {
        return false;
    }
    thread_local std::vector<size_t> host_metadata;
    host_metadata.resize(element_count);
    for (size_t i = 0; i < array_count; ++i) {
        std::memcpy(host_metadata.data() + i * len, host_arrays[i], len * sizeof(size_t));
    }

    thread_local ReusableCudaWorkspace device_metadata;
    if (!device_metadata.ensure(bytes, context)) {
        return false;
    }
    cudaError_t status = cudaMemcpy(
        device_metadata.ptr,
        host_metadata.data(),
        bytes,
        cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        set_cuda_error(context, status);
        return false;
    }

    size_t* base = static_cast<size_t*>(device_metadata.ptr);
    for (size_t i = 0; i < array_count; ++i) {
        device_arrays[i] = base + i * len;
    }
    return true;
}
extern "C" int lumen_cuda_unary_f32_device(
    uint64_t input_handle,
    uint64_t out_handle,
    size_t len,
    int op) {
    if (!validate_handle(input_handle, "CUDA unary input handle") ||
        !validate_handle(out_handle, "CUDA unary output handle")) {
        return 1;
    }
    if (len == 0) {
        return 0;
    }

#if LUMEN_HAS_CUDNN
    cudnnActivationMode_t mode;
    if (len <= static_cast<size_t>(INT_MAX) && cudnn_activation_mode_for_op(op, mode)) {
        CudnnHandle handle;
        CudnnTensorDescriptor input_desc;
        CudnnActivationDescriptor activation_desc;
        if (!init_cudnn(handle) ||
            !init_tensor_descriptor_4d(input_desc, 1, static_cast<int>(len), 1, 1) ||
            !init_activation_descriptor(activation_desc, mode)) {
            return 1;
        }

        const float alpha = 1.0f;
        const float beta = 0.0f;
        cudnnStatus_t status = cudnnActivationForward(
            handle.handle,
            activation_desc.desc,
            &alpha,
            input_desc.desc,
            handle_to_ptr(input_handle),
            &beta,
            input_desc.desc,
            handle_to_ptr(out_handle));
        if (status != CUDNN_STATUS_SUCCESS) {
            set_cudnn_error("cuDNN activation forward failed", status);
            return 1;
        }
        return 0;
    }
#endif

    constexpr unsigned int block_size = 256;
    size_t vec_len = len / 4;
    if (vec_len > 0) {
        const unsigned int grid_size = linear_grid_size(vec_len, block_size);
        launch_unary_kernel(
            op,
            true,
            grid_size,
            block_size,
            handle_to_ptr(input_handle),
            handle_to_ptr(out_handle),
            vec_len,
            len);
    } else {
        const unsigned int grid_size = linear_grid_size(len, block_size);
        launch_unary_kernel(
            op,
            false,
            grid_size,
            block_size,
            handle_to_ptr(input_handle),
            handle_to_ptr(out_handle),
            len,
            len);
    }
    if (!check_cuda_launch("CUDA unary kernel launch failed")) {
        return 1;
    }
    return 0;
}

extern "C" int lumen_cuda_unary_f16_device(
    uint64_t input_handle,
    uint64_t out_handle,
    size_t len,
    int op) {
    if (!validate_handle(input_handle, "CUDA unary F16 input handle") ||
        !validate_handle(out_handle, "CUDA unary F16 output handle")) {
        return 1;
    }
    if (len == 0) {
        return 0;
    }

    constexpr unsigned int block_size = 256;
    const unsigned int grid_size = linear_grid_size(len, block_size);
    launch_unary_lowp_to_f32_kernel<__half>(
        op,
        grid_size,
        block_size,
        reinterpret_cast<const __half*>(handle_to_ptr(input_handle)),
        handle_to_ptr(out_handle),
        len);
    return check_cuda_launch("CUDA unary F16 kernel launch failed") ? 0 : 1;
}

extern "C" int lumen_cuda_unary_f16_typed_out_device(
    uint64_t input_handle,
    uint64_t out_handle,
    size_t len,
    int op) {
    if (!validate_handle(input_handle, "CUDA unary F16 typed-output input handle") ||
        !validate_handle(out_handle, "CUDA unary F16 typed-output output handle")) {
        return 1;
    }
    if (len == 0) {
        return 0;
    }

    constexpr unsigned int block_size = 256;
    const unsigned int grid_size = linear_grid_size(len, block_size);
    launch_unary_lowp_to_typed_kernel<__half, __half>(
        op,
        grid_size,
        block_size,
        reinterpret_cast<const __half*>(handle_to_ptr(input_handle)),
        reinterpret_cast<__half*>(handle_to_ptr(out_handle)),
        len);
    return check_cuda_launch("CUDA unary F16 typed-output kernel launch failed") ? 0 : 1;
}

extern "C" int lumen_cuda_unary_bf16_device(
    uint64_t input_handle,
    uint64_t out_handle,
    size_t len,
    int op) {
    if (!validate_handle(input_handle, "CUDA unary BF16 input handle") ||
        !validate_handle(out_handle, "CUDA unary BF16 output handle")) {
        return 1;
    }
    if (len == 0) {
        return 0;
    }

    constexpr unsigned int block_size = 256;
    const unsigned int grid_size = linear_grid_size(len, block_size);
    launch_unary_lowp_to_f32_kernel<__nv_bfloat16>(
        op,
        grid_size,
        block_size,
        reinterpret_cast<const __nv_bfloat16*>(handle_to_ptr(input_handle)),
        handle_to_ptr(out_handle),
        len);
    return check_cuda_launch("CUDA unary BF16 kernel launch failed") ? 0 : 1;
}

extern "C" int lumen_cuda_unary_bf16_typed_out_device(
    uint64_t input_handle,
    uint64_t out_handle,
    size_t len,
    int op) {
    if (!validate_handle(input_handle, "CUDA unary BF16 typed-output input handle") ||
        !validate_handle(out_handle, "CUDA unary BF16 typed-output output handle")) {
        return 1;
    }
    if (len == 0) {
        return 0;
    }

    constexpr unsigned int block_size = 256;
    const unsigned int grid_size = linear_grid_size(len, block_size);
    launch_unary_lowp_to_typed_kernel<__nv_bfloat16, __nv_bfloat16>(
        op,
        grid_size,
        block_size,
        reinterpret_cast<const __nv_bfloat16*>(handle_to_ptr(input_handle)),
        reinterpret_cast<__nv_bfloat16*>(handle_to_ptr(out_handle)),
        len);
    return check_cuda_launch("CUDA unary BF16 typed-output kernel launch failed") ? 0 : 1;
}

extern "C" int lumen_cuda_unary_i8_device(
    uint64_t input_handle,
    float scale,
    uint64_t out_handle,
    size_t len,
    int op) {
    if (!validate_handle(input_handle, "CUDA unary I8 input handle") ||
        !validate_handle(out_handle, "CUDA unary I8 output handle")) {
        return 1;
    }
    if (len == 0) {
        return 0;
    }

    constexpr unsigned int block_size = 256;
    const unsigned int grid_size = linear_grid_size(len, block_size);
    launch_unary_i8_to_f32_kernel(
        op,
        grid_size,
        block_size,
        reinterpret_cast<const int8_t*>(handle_to_ptr(input_handle)),
        scale,
        handle_to_ptr(out_handle),
        len);
    return check_cuda_launch("CUDA unary I8 kernel launch failed") ? 0 : 1;
}

extern "C" int lumen_cuda_unary_i8_relu_typed_out_device(
    uint64_t input_handle,
    uint64_t out_handle,
    size_t len) {
    if (!validate_handle(input_handle, "CUDA unary I8 ReLU typed-output input handle") ||
        !validate_handle(out_handle, "CUDA unary I8 ReLU typed-output output handle")) {
        return 1;
    }
    if (len == 0) {
        return 0;
    }

    constexpr unsigned int block_size = 256;
    const unsigned int grid_size = linear_grid_size(len, block_size);
    unary_i8_relu_typed_out_kernel<<<grid_size, block_size>>>(
        reinterpret_cast<const int8_t*>(handle_to_ptr(input_handle)),
        reinterpret_cast<int8_t*>(handle_to_ptr(out_handle)),
        len);
    return check_cuda_launch("CUDA unary I8 ReLU typed-output kernel launch failed") ? 0 : 1;
}

extern "C" int lumen_cuda_unary_backward_f32_device(
    uint64_t input_handle,
    uint64_t output_handle,
    uint64_t grad_handle,
    uint64_t out_handle,
    size_t len,
    int op) {
    if (!validate_handle(input_handle, "CUDA unary backward input handle") ||
        !validate_handle(output_handle, "CUDA unary backward output handle") ||
        !validate_handle(grad_handle, "CUDA unary backward grad handle") ||
        !validate_handle(out_handle, "CUDA unary backward result handle")) {
        return 1;
    }
    if (len == 0) {
        return 0;
    }

#if LUMEN_HAS_CUDNN
    cudnnActivationMode_t mode;
    if (len <= static_cast<size_t>(INT_MAX) && cudnn_activation_mode_for_op(op, mode)) {
        CudnnHandle handle;
        CudnnTensorDescriptor tensor_desc;
        CudnnActivationDescriptor activation_desc;
        if (!init_cudnn(handle) ||
            !init_tensor_descriptor_4d(tensor_desc, 1, static_cast<int>(len), 1, 1) ||
            !init_activation_descriptor(activation_desc, mode)) {
            return 1;
        }

        const float alpha = 1.0f;
        const float beta = 0.0f;
        cudnnStatus_t status = cudnnActivationBackward(
            handle.handle,
            activation_desc.desc,
            &alpha,
            tensor_desc.desc,
            handle_to_ptr(output_handle),
            tensor_desc.desc,
            handle_to_ptr(grad_handle),
            tensor_desc.desc,
            handle_to_ptr(input_handle),
            &beta,
            tensor_desc.desc,
            handle_to_ptr(out_handle));
        if (status != CUDNN_STATUS_SUCCESS) {
            set_cudnn_error("cuDNN activation backward failed", status);
            return 1;
        }
        return 0;
    }
#endif

    constexpr unsigned int block_size = 256;
    size_t vec_len = len / 4;
    if (vec_len > 0) {
        const unsigned int grid_size = linear_grid_size(vec_len, block_size);
        launch_unary_backward_kernel(
            op,
            true,
            grid_size,
            block_size,
            handle_to_ptr(input_handle),
            handle_to_ptr(output_handle),
            handle_to_ptr(grad_handle),
            handle_to_ptr(out_handle),
            vec_len,
            len);
    } else {
        const unsigned int grid_size = linear_grid_size(len, block_size);
        launch_unary_backward_kernel(
            op,
            false,
            grid_size,
            block_size,
            handle_to_ptr(input_handle),
            handle_to_ptr(output_handle),
            handle_to_ptr(grad_handle),
            handle_to_ptr(out_handle),
            len,
            len);
    }
    if (!check_cuda_launch("CUDA unary backward kernel launch failed")) {
        return 1;
    }
    return 0;
}

extern "C" int lumen_cuda_unary_backward_f16_device(
    uint64_t input_handle,
    uint64_t output_handle,
    uint64_t grad_handle,
    uint64_t out_handle,
    size_t len,
    int op) {
    if (!validate_handle(input_handle, "CUDA unary backward F16 input handle") ||
        !validate_handle(output_handle, "CUDA unary backward F16 output handle") ||
        !validate_handle(grad_handle, "CUDA unary backward F16 grad handle") ||
        !validate_handle(out_handle, "CUDA unary backward F16 result handle")) {
        return 1;
    }
    if (len == 0) {
        return 0;
    }

    constexpr unsigned int block_size = 256;
    const unsigned int grid_size = linear_grid_size(len, block_size);
    launch_unary_backward_lowp_to_f32_kernel<__half>(
        op,
        grid_size,
        block_size,
        reinterpret_cast<const __half*>(handle_to_ptr(input_handle)),
        handle_to_ptr(output_handle),
        handle_to_ptr(grad_handle),
        handle_to_ptr(out_handle),
        len);
    return check_cuda_launch("CUDA unary backward F16 kernel launch failed") ? 0 : 1;
}

extern "C" int lumen_cuda_unary_backward_bf16_device(
    uint64_t input_handle,
    uint64_t output_handle,
    uint64_t grad_handle,
    uint64_t out_handle,
    size_t len,
    int op) {
    if (!validate_handle(input_handle, "CUDA unary backward BF16 input handle") ||
        !validate_handle(output_handle, "CUDA unary backward BF16 output handle") ||
        !validate_handle(grad_handle, "CUDA unary backward BF16 grad handle") ||
        !validate_handle(out_handle, "CUDA unary backward BF16 result handle")) {
        return 1;
    }
    if (len == 0) {
        return 0;
    }

    constexpr unsigned int block_size = 256;
    const unsigned int grid_size = linear_grid_size(len, block_size);
    launch_unary_backward_lowp_to_f32_kernel<__nv_bfloat16>(
        op,
        grid_size,
        block_size,
        reinterpret_cast<const __nv_bfloat16*>(handle_to_ptr(input_handle)),
        handle_to_ptr(output_handle),
        handle_to_ptr(grad_handle),
        handle_to_ptr(out_handle),
        len);
    return check_cuda_launch("CUDA unary backward BF16 kernel launch failed") ? 0 : 1;
}

extern "C" int lumen_cuda_unary_backward_i8_device(
    uint64_t input_handle,
    float scale,
    uint64_t output_handle,
    uint64_t grad_handle,
    uint64_t out_handle,
    size_t len,
    int op) {
    if (!validate_handle(input_handle, "CUDA unary backward I8 input handle") ||
        !validate_handle(output_handle, "CUDA unary backward I8 output handle") ||
        !validate_handle(grad_handle, "CUDA unary backward I8 grad handle") ||
        !validate_handle(out_handle, "CUDA unary backward I8 result handle")) {
        return 1;
    }
    if (len == 0) {
        return 0;
    }

    constexpr unsigned int block_size = 256;
    const unsigned int grid_size = linear_grid_size(len, block_size);
    launch_unary_backward_i8_to_f32_kernel(
        op,
        grid_size,
        block_size,
        reinterpret_cast<const int8_t*>(handle_to_ptr(input_handle)),
        scale,
        handle_to_ptr(output_handle),
        handle_to_ptr(grad_handle),
        handle_to_ptr(out_handle),
        len);
    return check_cuda_launch("CUDA unary backward I8 kernel launch failed") ? 0 : 1;
}

extern "C" int lumen_cuda_binary_f32_device(
    uint64_t lhs_handle,
    uint64_t rhs_handle,
    uint64_t out_handle,
    size_t len,
    int op) {
    if (!validate_handle(lhs_handle, "CUDA binary lhs handle") ||
        !validate_handle(rhs_handle, "CUDA binary rhs handle") ||
        !validate_handle(out_handle, "CUDA binary output handle")) {
        return 1;
    }
    if (len == 0) {
        return 0;
    }

    constexpr unsigned int block_size = 256;
    size_t vec_len = len / 4;
    if (vec_len > 0) {
        const unsigned int grid_size = linear_grid_size(vec_len, block_size);
        launch_binary_kernel(
            op,
            true,
            grid_size,
            block_size,
            handle_to_ptr(lhs_handle),
            handle_to_ptr(rhs_handle),
            handle_to_ptr(out_handle),
            vec_len,
            len);
    } else {
        const unsigned int grid_size = linear_grid_size(len, block_size);
        launch_binary_kernel(
            op,
            false,
            grid_size,
            block_size,
            handle_to_ptr(lhs_handle),
            handle_to_ptr(rhs_handle),
            handle_to_ptr(out_handle),
            len,
            len);
    }
    if (!check_cuda_launch("CUDA binary kernel launch failed")) {
        return 1;
    }
    return 0;
}

template <typename LhsT>
int dispatch_binary_typed_rhs(
    uint64_t lhs_handle,
    float lhs_scale,
    uint64_t rhs_handle,
    int rhs_dtype,
    float rhs_scale,
    uint64_t out_handle,
    size_t len,
    int op) {
    switch (rhs_dtype) {
        case kDTypeF32:
            launch_binary_typed_to_f32_kernel<LhsT, float>(
                reinterpret_cast<const LhsT*>(handle_to_ptr(lhs_handle)),
                lhs_scale,
                handle_to_ptr(rhs_handle),
                rhs_scale,
                handle_to_ptr(out_handle),
                len,
                op);
            return check_cuda_launch("CUDA typed mixed binary rhs=f32 launch failed") ? 0 : 1;
        case kDTypeF16:
            launch_binary_typed_to_f32_kernel<LhsT, __half>(
                reinterpret_cast<const LhsT*>(handle_to_ptr(lhs_handle)),
                lhs_scale,
                reinterpret_cast<const __half*>(handle_to_ptr(rhs_handle)),
                rhs_scale,
                handle_to_ptr(out_handle),
                len,
                op);
            return check_cuda_launch("CUDA typed mixed binary rhs=f16 launch failed") ? 0 : 1;
        case kDTypeBF16:
            launch_binary_typed_to_f32_kernel<LhsT, __nv_bfloat16>(
                reinterpret_cast<const LhsT*>(handle_to_ptr(lhs_handle)),
                lhs_scale,
                reinterpret_cast<const __nv_bfloat16*>(handle_to_ptr(rhs_handle)),
                rhs_scale,
                handle_to_ptr(out_handle),
                len,
                op);
            return check_cuda_launch("CUDA typed mixed binary rhs=bf16 launch failed") ? 0 : 1;
        case kDTypeI8:
            launch_binary_typed_to_f32_kernel<LhsT, int8_t>(
                reinterpret_cast<const LhsT*>(handle_to_ptr(lhs_handle)),
                lhs_scale,
                reinterpret_cast<const int8_t*>(handle_to_ptr(rhs_handle)),
                rhs_scale,
                handle_to_ptr(out_handle),
                len,
                op);
            return check_cuda_launch("CUDA typed mixed binary rhs=i8 launch failed") ? 0 : 1;
        default:
            set_error("unsupported rhs dtype for CUDA typed mixed binary");
            return 1;
    }
}

extern "C" int lumen_cuda_binary_typed_device(
    uint64_t lhs_handle,
    int lhs_dtype,
    float lhs_scale,
    uint64_t rhs_handle,
    int rhs_dtype,
    float rhs_scale,
    uint64_t out_handle,
    size_t len,
    int op) {
    if (!validate_handle(lhs_handle, "CUDA typed mixed binary lhs handle") ||
        !validate_handle(rhs_handle, "CUDA typed mixed binary rhs handle") ||
        !validate_handle(out_handle, "CUDA typed mixed binary output handle")) {
        return 1;
    }
    if ((lhs_dtype == kDTypeI8 && (!std::isfinite(lhs_scale) || lhs_scale <= 0.0f)) ||
        (rhs_dtype == kDTypeI8 && (!std::isfinite(rhs_scale) || rhs_scale <= 0.0f))) {
        set_error("CUDA typed mixed binary I8 scales must be finite and > 0");
        return 1;
    }
    if (len == 0) {
        return 0;
    }

    switch (lhs_dtype) {
        case kDTypeF32:
            return dispatch_binary_typed_rhs<float>(
                lhs_handle, lhs_scale, rhs_handle, rhs_dtype, rhs_scale, out_handle, len, op);
        case kDTypeF16:
            return dispatch_binary_typed_rhs<__half>(
                lhs_handle, lhs_scale, rhs_handle, rhs_dtype, rhs_scale, out_handle, len, op);
        case kDTypeBF16:
            return dispatch_binary_typed_rhs<__nv_bfloat16>(
                lhs_handle, lhs_scale, rhs_handle, rhs_dtype, rhs_scale, out_handle, len, op);
        case kDTypeI8:
            return dispatch_binary_typed_rhs<int8_t>(
                lhs_handle, lhs_scale, rhs_handle, rhs_dtype, rhs_scale, out_handle, len, op);
        default:
            set_error("unsupported lhs dtype for CUDA typed mixed binary");
            return 1;
    }
}

extern "C" int lumen_cuda_binary_lowp_typed_out_device(
    uint64_t lhs_handle,
    uint64_t rhs_handle,
    uint64_t out_handle,
    size_t len,
    int dtype,
    int op) {
    if (!validate_handle(lhs_handle, "CUDA lowp typed-out binary lhs handle") ||
        !validate_handle(rhs_handle, "CUDA lowp typed-out binary rhs handle") ||
        !validate_handle(out_handle, "CUDA lowp typed-out binary output handle")) {
        return 1;
    }
    if (len == 0) {
        return 0;
    }

    switch (dtype) {
        case kDTypeF16:
            launch_binary_lowp_to_typed_kernel<__half>(
                reinterpret_cast<const __half*>(handle_to_ptr(lhs_handle)),
                reinterpret_cast<const __half*>(handle_to_ptr(rhs_handle)),
                reinterpret_cast<__half*>(handle_to_ptr(out_handle)),
                len,
                op);
            return check_cuda_launch("CUDA F16 typed-out binary launch failed") ? 0 : 1;
        case kDTypeBF16:
            launch_binary_lowp_to_typed_kernel<__nv_bfloat16>(
                reinterpret_cast<const __nv_bfloat16*>(handle_to_ptr(lhs_handle)),
                reinterpret_cast<const __nv_bfloat16*>(handle_to_ptr(rhs_handle)),
                reinterpret_cast<__nv_bfloat16*>(handle_to_ptr(out_handle)),
                len,
                op);
            return check_cuda_launch("CUDA BF16 typed-out binary launch failed") ? 0 : 1;
        default:
            set_error("CUDA typed-out binary supports only F16/BF16");
            return 1;
    }
}

template <typename LhsT>
int dispatch_binary_typed_lastdim_rhs(
    uint64_t lhs_handle,
    float lhs_scale,
    uint64_t rhs_handle,
    int rhs_dtype,
    float rhs_scale,
    uint64_t out_handle,
    size_t len,
    size_t last_dim,
    int vector_on_rhs,
    int op) {
    switch (rhs_dtype) {
        case kDTypeF32:
            launch_binary_typed_lastdim_to_f32_kernel<LhsT, float>(
                reinterpret_cast<const LhsT*>(handle_to_ptr(lhs_handle)),
                lhs_scale,
                handle_to_ptr(rhs_handle),
                rhs_scale,
                handle_to_ptr(out_handle),
                len,
                last_dim,
                vector_on_rhs != 0,
                op);
            return check_cuda_launch("CUDA typed mixed lastdim binary rhs=f32 launch failed") ? 0 : 1;
        case kDTypeF16:
            launch_binary_typed_lastdim_to_f32_kernel<LhsT, __half>(
                reinterpret_cast<const LhsT*>(handle_to_ptr(lhs_handle)),
                lhs_scale,
                reinterpret_cast<const __half*>(handle_to_ptr(rhs_handle)),
                rhs_scale,
                handle_to_ptr(out_handle),
                len,
                last_dim,
                vector_on_rhs != 0,
                op);
            return check_cuda_launch("CUDA typed mixed lastdim binary rhs=f16 launch failed") ? 0 : 1;
        case kDTypeBF16:
            launch_binary_typed_lastdim_to_f32_kernel<LhsT, __nv_bfloat16>(
                reinterpret_cast<const LhsT*>(handle_to_ptr(lhs_handle)),
                lhs_scale,
                reinterpret_cast<const __nv_bfloat16*>(handle_to_ptr(rhs_handle)),
                rhs_scale,
                handle_to_ptr(out_handle),
                len,
                last_dim,
                vector_on_rhs != 0,
                op);
            return check_cuda_launch("CUDA typed mixed lastdim binary rhs=bf16 launch failed") ? 0 : 1;
        case kDTypeI8:
            launch_binary_typed_lastdim_to_f32_kernel<LhsT, int8_t>(
                reinterpret_cast<const LhsT*>(handle_to_ptr(lhs_handle)),
                lhs_scale,
                reinterpret_cast<const int8_t*>(handle_to_ptr(rhs_handle)),
                rhs_scale,
                handle_to_ptr(out_handle),
                len,
                last_dim,
                vector_on_rhs != 0,
                op);
            return check_cuda_launch("CUDA typed mixed lastdim binary rhs=i8 launch failed") ? 0 : 1;
        default:
            set_error("unsupported rhs dtype for CUDA typed mixed lastdim binary");
            return 1;
    }
}

extern "C" int lumen_cuda_binary_typed_lastdim_device(
    uint64_t lhs_handle,
    int lhs_dtype,
    float lhs_scale,
    uint64_t rhs_handle,
    int rhs_dtype,
    float rhs_scale,
    uint64_t out_handle,
    size_t len,
    size_t last_dim,
    int vector_on_rhs,
    int op) {
    if (!validate_handle(lhs_handle, "CUDA typed mixed lastdim binary lhs handle") ||
        !validate_handle(rhs_handle, "CUDA typed mixed lastdim binary rhs handle") ||
        !validate_handle(out_handle, "CUDA typed mixed lastdim binary output handle")) {
        return 1;
    }
    if (last_dim == 0) {
        set_error("CUDA typed mixed lastdim binary last_dim must be greater than zero");
        return 1;
    }
    if ((lhs_dtype == kDTypeI8 && (!std::isfinite(lhs_scale) || lhs_scale <= 0.0f)) ||
        (rhs_dtype == kDTypeI8 && (!std::isfinite(rhs_scale) || rhs_scale <= 0.0f))) {
        set_error("CUDA typed mixed lastdim binary I8 scales must be finite and > 0");
        return 1;
    }
    if (len == 0) {
        return 0;
    }

    switch (lhs_dtype) {
        case kDTypeF32:
            return dispatch_binary_typed_lastdim_rhs<float>(
                lhs_handle, lhs_scale, rhs_handle, rhs_dtype, rhs_scale, out_handle, len, last_dim, vector_on_rhs, op);
        case kDTypeF16:
            return dispatch_binary_typed_lastdim_rhs<__half>(
                lhs_handle, lhs_scale, rhs_handle, rhs_dtype, rhs_scale, out_handle, len, last_dim, vector_on_rhs, op);
        case kDTypeBF16:
            return dispatch_binary_typed_lastdim_rhs<__nv_bfloat16>(
                lhs_handle, lhs_scale, rhs_handle, rhs_dtype, rhs_scale, out_handle, len, last_dim, vector_on_rhs, op);
        case kDTypeI8:
            return dispatch_binary_typed_lastdim_rhs<int8_t>(
                lhs_handle, lhs_scale, rhs_handle, rhs_dtype, rhs_scale, out_handle, len, last_dim, vector_on_rhs, op);
        default:
            set_error("unsupported lhs dtype for CUDA typed mixed lastdim binary");
            return 1;
    }
}

extern "C" int lumen_cuda_binary_lowp_typed_lastdim_out_device(
    uint64_t lhs_handle,
    uint64_t rhs_handle,
    uint64_t out_handle,
    size_t len,
    size_t last_dim,
    int vector_on_rhs,
    int dtype,
    int op) {
    if (!validate_handle(lhs_handle, "CUDA lowp typed-out lastdim binary lhs handle") ||
        !validate_handle(rhs_handle, "CUDA lowp typed-out lastdim binary rhs handle") ||
        !validate_handle(out_handle, "CUDA lowp typed-out lastdim binary output handle")) {
        return 1;
    }
    if (last_dim == 0) {
        set_error("CUDA lowp typed-out lastdim binary last_dim must be greater than zero");
        return 1;
    }
    if (len == 0) {
        return 0;
    }

    switch (dtype) {
        case kDTypeF16:
            launch_binary_lowp_lastdim_to_typed_kernel<__half>(
                reinterpret_cast<const __half*>(handle_to_ptr(lhs_handle)),
                reinterpret_cast<const __half*>(handle_to_ptr(rhs_handle)),
                reinterpret_cast<__half*>(handle_to_ptr(out_handle)),
                len,
                last_dim,
                vector_on_rhs != 0,
                op);
            return check_cuda_launch("CUDA F16 typed-out lastdim binary launch failed") ? 0 : 1;
        case kDTypeBF16:
            launch_binary_lowp_lastdim_to_typed_kernel<__nv_bfloat16>(
                reinterpret_cast<const __nv_bfloat16*>(handle_to_ptr(lhs_handle)),
                reinterpret_cast<const __nv_bfloat16*>(handle_to_ptr(rhs_handle)),
                reinterpret_cast<__nv_bfloat16*>(handle_to_ptr(out_handle)),
                len,
                last_dim,
                vector_on_rhs != 0,
                op);
            return check_cuda_launch("CUDA BF16 typed-out lastdim binary launch failed") ? 0 : 1;
        default:
            set_error("CUDA typed-out lastdim binary supports only F16/BF16");
            return 1;
    }
}

template <typename LhsT>
int dispatch_binary_typed_row_scalar_rhs(
    uint64_t lhs_handle,
    float lhs_scale,
    uint64_t rhs_handle,
    int rhs_dtype,
    float rhs_scale,
    uint64_t out_handle,
    size_t rows,
    size_t last_dim,
    int scalar_on_rhs,
    int op) {
    switch (rhs_dtype) {
        case kDTypeF32:
            launch_binary_typed_row_scalar_to_f32_kernel<LhsT, float>(
                reinterpret_cast<const LhsT*>(handle_to_ptr(lhs_handle)), lhs_scale,
                handle_to_ptr(rhs_handle), rhs_scale, handle_to_ptr(out_handle), rows, last_dim,
                scalar_on_rhs != 0, op);
            return check_cuda_launch("CUDA typed mixed row-scalar binary rhs=f32 launch failed") ? 0 : 1;
        case kDTypeF16:
            launch_binary_typed_row_scalar_to_f32_kernel<LhsT, __half>(
                reinterpret_cast<const LhsT*>(handle_to_ptr(lhs_handle)), lhs_scale,
                reinterpret_cast<const __half*>(handle_to_ptr(rhs_handle)), rhs_scale,
                handle_to_ptr(out_handle), rows, last_dim, scalar_on_rhs != 0, op);
            return check_cuda_launch("CUDA typed mixed row-scalar binary rhs=f16 launch failed") ? 0 : 1;
        case kDTypeBF16:
            launch_binary_typed_row_scalar_to_f32_kernel<LhsT, __nv_bfloat16>(
                reinterpret_cast<const LhsT*>(handle_to_ptr(lhs_handle)), lhs_scale,
                reinterpret_cast<const __nv_bfloat16*>(handle_to_ptr(rhs_handle)), rhs_scale,
                handle_to_ptr(out_handle), rows, last_dim, scalar_on_rhs != 0, op);
            return check_cuda_launch("CUDA typed mixed row-scalar binary rhs=bf16 launch failed") ? 0 : 1;
        case kDTypeI8:
            launch_binary_typed_row_scalar_to_f32_kernel<LhsT, int8_t>(
                reinterpret_cast<const LhsT*>(handle_to_ptr(lhs_handle)), lhs_scale,
                reinterpret_cast<const int8_t*>(handle_to_ptr(rhs_handle)), rhs_scale,
                handle_to_ptr(out_handle), rows, last_dim, scalar_on_rhs != 0, op);
            return check_cuda_launch("CUDA typed mixed row-scalar binary rhs=i8 launch failed") ? 0 : 1;
        default:
            set_error("unsupported rhs dtype for CUDA typed mixed row-scalar binary");
            return 1;
    }
}

extern "C" int lumen_cuda_binary_typed_row_scalar_device(
    uint64_t lhs_handle,
    int lhs_dtype,
    float lhs_scale,
    uint64_t rhs_handle,
    int rhs_dtype,
    float rhs_scale,
    uint64_t out_handle,
    size_t rows,
    size_t last_dim,
    int scalar_on_rhs,
    int op) {
    if (!validate_handle(lhs_handle, "CUDA typed mixed row-scalar binary lhs handle") ||
        !validate_handle(rhs_handle, "CUDA typed mixed row-scalar binary rhs handle") ||
        !validate_handle(out_handle, "CUDA typed mixed row-scalar binary output handle")) {
        return 1;
    }
    if (rows == 0 || last_dim == 0) {
        set_error("CUDA typed mixed row-scalar binary dimensions must be non-zero");
        return 1;
    }
    if ((lhs_dtype == kDTypeI8 && (!std::isfinite(lhs_scale) || lhs_scale <= 0.0f)) ||
        (rhs_dtype == kDTypeI8 && (!std::isfinite(rhs_scale) || rhs_scale <= 0.0f))) {
        set_error("CUDA typed mixed row-scalar binary I8 scales must be finite and > 0");
        return 1;
    }

    switch (lhs_dtype) {
        case kDTypeF32:
            return dispatch_binary_typed_row_scalar_rhs<float>(
                lhs_handle, lhs_scale, rhs_handle, rhs_dtype, rhs_scale, out_handle, rows, last_dim, scalar_on_rhs, op);
        case kDTypeF16:
            return dispatch_binary_typed_row_scalar_rhs<__half>(
                lhs_handle, lhs_scale, rhs_handle, rhs_dtype, rhs_scale, out_handle, rows, last_dim, scalar_on_rhs, op);
        case kDTypeBF16:
            return dispatch_binary_typed_row_scalar_rhs<__nv_bfloat16>(
                lhs_handle, lhs_scale, rhs_handle, rhs_dtype, rhs_scale, out_handle, rows, last_dim, scalar_on_rhs, op);
        case kDTypeI8:
            return dispatch_binary_typed_row_scalar_rhs<int8_t>(
                lhs_handle, lhs_scale, rhs_handle, rhs_dtype, rhs_scale, out_handle, rows, last_dim, scalar_on_rhs, op);
        default:
            set_error("unsupported lhs dtype for CUDA typed mixed row-scalar binary");
            return 1;
    }
}

extern "C" int lumen_cuda_binary_lowp_typed_row_scalar_out_device(
    uint64_t lhs_handle,
    uint64_t rhs_handle,
    uint64_t out_handle,
    size_t rows,
    size_t last_dim,
    int scalar_on_rhs,
    int dtype,
    int op) {
    if (!validate_handle(lhs_handle, "CUDA lowp typed-out row-scalar binary lhs handle") ||
        !validate_handle(rhs_handle, "CUDA lowp typed-out row-scalar binary rhs handle") ||
        !validate_handle(out_handle, "CUDA lowp typed-out row-scalar binary output handle")) {
        return 1;
    }
    if (rows == 0 || last_dim == 0) {
        set_error("CUDA lowp typed-out row-scalar binary dimensions must be non-zero");
        return 1;
    }

    switch (dtype) {
        case kDTypeF16:
            launch_binary_lowp_row_scalar_to_typed_kernel<__half>(
                reinterpret_cast<const __half*>(handle_to_ptr(lhs_handle)),
                reinterpret_cast<const __half*>(handle_to_ptr(rhs_handle)),
                reinterpret_cast<__half*>(handle_to_ptr(out_handle)),
                rows,
                last_dim,
                scalar_on_rhs != 0,
                op);
            return check_cuda_launch("CUDA F16 typed-out row-scalar binary launch failed") ? 0 : 1;
        case kDTypeBF16:
            launch_binary_lowp_row_scalar_to_typed_kernel<__nv_bfloat16>(
                reinterpret_cast<const __nv_bfloat16*>(handle_to_ptr(lhs_handle)),
                reinterpret_cast<const __nv_bfloat16*>(handle_to_ptr(rhs_handle)),
                reinterpret_cast<__nv_bfloat16*>(handle_to_ptr(out_handle)),
                rows,
                last_dim,
                scalar_on_rhs != 0,
                op);
            return check_cuda_launch("CUDA BF16 typed-out row-scalar binary launch failed") ? 0 : 1;
        default:
            set_error("CUDA typed-out row-scalar binary supports only F16/BF16");
            return 1;
    }
}

extern "C" int lumen_cuda_binary_i8_typed_row_scalar_out_device(
    uint64_t lhs_handle,
    uint64_t rhs_handle,
    float lhs_scale,
    float rhs_scale,
    uint64_t out_handle,
    size_t rows,
    size_t last_dim,
    int scalar_on_rhs,
    int op,
    float* out_scale) {
    if (!validate_handle(lhs_handle, "CUDA I8 typed-out row-scalar binary lhs handle") ||
        !validate_handle(rhs_handle, "CUDA I8 typed-out row-scalar binary rhs handle") ||
        !validate_handle(out_handle, "CUDA I8 typed-out row-scalar binary output handle")) {
        return 1;
    }
    if (out_scale == nullptr) {
        set_error("CUDA I8 typed-out row-scalar binary scale output is null");
        return 1;
    }
    if (rows == 0 || last_dim == 0) {
        set_error("CUDA I8 typed-out row-scalar binary dimensions must be non-zero");
        return 1;
    }
    if (!std::isfinite(lhs_scale) || lhs_scale <= 0.0f ||
        !std::isfinite(rhs_scale) || rhs_scale <= 0.0f) {
        set_error("CUDA I8 typed-out row-scalar binary scales must be finite and > 0");
        return 1;
    }

    size_t len = 0;
    if (!checked_product(
            "CUDA I8 typed-out row-scalar binary length overflow", {rows, last_dim}, &len)) {
        return 1;
    }
    constexpr unsigned int block_size = 256;
    const unsigned int grid_size = linear_grid_size(len, block_size);
    thread_local ReusableCudaWorkspace partial_workspace;
    if (!partial_workspace.ensure(
            static_cast<size_t>(grid_size) * sizeof(float),
            "CUDA I8 typed-out row-scalar absmax allocation failed")) {
        return 1;
    }
    float* partial = static_cast<float*>(partial_workspace.ptr);

    launch_binary_i8_row_scalar_absmax_blocks_kernel(
        reinterpret_cast<const int8_t*>(handle_to_ptr(lhs_handle)),
        reinterpret_cast<const int8_t*>(handle_to_ptr(rhs_handle)),
        lhs_scale,
        rhs_scale,
        partial,
        rows,
        last_dim,
        scalar_on_rhs != 0,
        grid_size,
        block_size,
        op);
    if (!check_cuda_launch("CUDA I8 typed-out row-scalar absmax kernel launch failed")) {
        return 1;
    }

    float max_abs = 0.0f;
    if (!reduce_absmax_partials_to_host(
            partial,
            static_cast<size_t>(grid_size),
            &max_abs,
            "CUDA I8 typed-out row-scalar absmax reduce failed")) {
        return 1;
    }

    *out_scale = max_abs > 0.0f && isfinite(max_abs) ? std::max(max_abs / 127.0f, FLT_MIN) : 1.0f;
    launch_binary_i8_row_scalar_to_i8_kernel(
        reinterpret_cast<const int8_t*>(handle_to_ptr(lhs_handle)),
        reinterpret_cast<const int8_t*>(handle_to_ptr(rhs_handle)),
        lhs_scale,
        rhs_scale,
        reinterpret_cast<int8_t*>(handle_to_ptr(out_handle)),
        rows,
        last_dim,
        scalar_on_rhs != 0,
        *out_scale,
        grid_size,
        block_size,
        op);
    return check_cuda_launch("CUDA I8 typed-out row-scalar quantize kernel launch failed") ? 0 : 1;
}

template <typename LhsT>
int dispatch_binary_typed_broadcast_rhs(
    uint64_t lhs_handle,
    float lhs_scale,
    uint64_t rhs_handle,
    int rhs_dtype,
    float rhs_scale,
    uint64_t out_handle,
    const size_t* d_out_strides,
    const size_t* d_lhs_shape,
    const size_t* d_lhs_strides,
    const size_t* d_rhs_shape,
    const size_t* d_rhs_strides,
    size_t ndim,
    size_t len,
    int op) {
    switch (rhs_dtype) {
        case kDTypeF32:
            launch_binary_typed_broadcast_to_f32_kernel<LhsT, float>(
                reinterpret_cast<const LhsT*>(handle_to_ptr(lhs_handle)),
                lhs_scale,
                handle_to_ptr(rhs_handle),
                rhs_scale,
                handle_to_ptr(out_handle),
                d_out_strides,
                d_lhs_shape,
                d_lhs_strides,
                d_rhs_shape,
                d_rhs_strides,
                ndim,
                len,
                op);
            return check_cuda_launch("CUDA typed mixed broadcast binary rhs=f32 launch failed") ? 0 : 1;
        case kDTypeF16:
            launch_binary_typed_broadcast_to_f32_kernel<LhsT, __half>(
                reinterpret_cast<const LhsT*>(handle_to_ptr(lhs_handle)),
                lhs_scale,
                reinterpret_cast<const __half*>(handle_to_ptr(rhs_handle)),
                rhs_scale,
                handle_to_ptr(out_handle),
                d_out_strides,
                d_lhs_shape,
                d_lhs_strides,
                d_rhs_shape,
                d_rhs_strides,
                ndim,
                len,
                op);
            return check_cuda_launch("CUDA typed mixed broadcast binary rhs=f16 launch failed") ? 0 : 1;
        case kDTypeBF16:
            launch_binary_typed_broadcast_to_f32_kernel<LhsT, __nv_bfloat16>(
                reinterpret_cast<const LhsT*>(handle_to_ptr(lhs_handle)),
                lhs_scale,
                reinterpret_cast<const __nv_bfloat16*>(handle_to_ptr(rhs_handle)),
                rhs_scale,
                handle_to_ptr(out_handle),
                d_out_strides,
                d_lhs_shape,
                d_lhs_strides,
                d_rhs_shape,
                d_rhs_strides,
                ndim,
                len,
                op);
            return check_cuda_launch("CUDA typed mixed broadcast binary rhs=bf16 launch failed") ? 0 : 1;
        case kDTypeI8:
            launch_binary_typed_broadcast_to_f32_kernel<LhsT, int8_t>(
                reinterpret_cast<const LhsT*>(handle_to_ptr(lhs_handle)),
                lhs_scale,
                reinterpret_cast<const int8_t*>(handle_to_ptr(rhs_handle)),
                rhs_scale,
                handle_to_ptr(out_handle),
                d_out_strides,
                d_lhs_shape,
                d_lhs_strides,
                d_rhs_shape,
                d_rhs_strides,
                ndim,
                len,
                op);
            return check_cuda_launch("CUDA typed mixed broadcast binary rhs=i8 launch failed") ? 0 : 1;
        default:
            set_error("unsupported rhs dtype for CUDA typed mixed broadcast binary");
            return 1;
    }
}

extern "C" int lumen_cuda_binary_typed_broadcast_device(
    uint64_t lhs_handle,
    int lhs_dtype,
    float lhs_scale,
    uint64_t rhs_handle,
    int rhs_dtype,
    float rhs_scale,
    uint64_t out_handle,
    size_t ndim,
    const size_t* out_strides,
    const size_t* lhs_shape,
    const size_t* lhs_strides,
    const size_t* rhs_shape,
    const size_t* rhs_strides,
    size_t len,
    int op) {
    if (!validate_handle(lhs_handle, "CUDA typed mixed broadcast binary lhs handle") ||
        !validate_handle(rhs_handle, "CUDA typed mixed broadcast binary rhs handle") ||
        !validate_handle(out_handle, "CUDA typed mixed broadcast binary output handle")) {
        return 1;
    }
    if (ndim == 0 || len == 0 || out_strides == nullptr || lhs_shape == nullptr ||
        lhs_strides == nullptr || rhs_shape == nullptr || rhs_strides == nullptr) {
        set_error("CUDA typed mixed broadcast binary received invalid metadata");
        return 1;
    }
    if ((lhs_dtype == kDTypeI8 && (!std::isfinite(lhs_scale) || lhs_scale <= 0.0f)) ||
        (rhs_dtype == kDTypeI8 && (!std::isfinite(rhs_scale) || rhs_scale <= 0.0f))) {
        set_error("CUDA typed mixed broadcast binary I8 scales must be finite and > 0");
        return 1;
    }

    size_t* d_out_strides = nullptr;
    size_t* d_lhs_shape = nullptr;
    size_t* d_lhs_strides = nullptr;
    size_t* d_rhs_shape = nullptr;
    size_t* d_rhs_strides = nullptr;
    const size_t* host_metadata[] = {
        out_strides, lhs_shape, lhs_strides, rhs_shape, rhs_strides};
    size_t* device_metadata[] = {
        d_out_strides, d_lhs_shape, d_lhs_strides, d_rhs_shape, d_rhs_strides};
    if (!upload_packed_elementwise_metadata(
            "failed to upload CUDA typed mixed broadcast binary metadata",
            host_metadata,
            5,
            ndim,
            device_metadata)) {
        return 1;
    }
    d_out_strides = device_metadata[0];
    d_lhs_shape = device_metadata[1];
    d_lhs_strides = device_metadata[2];
    d_rhs_shape = device_metadata[3];
    d_rhs_strides = device_metadata[4];

    int status = 1;
    switch (lhs_dtype) {
        case kDTypeF32:
            status = dispatch_binary_typed_broadcast_rhs<float>(
                lhs_handle, lhs_scale, rhs_handle, rhs_dtype, rhs_scale, out_handle, d_out_strides,
                d_lhs_shape, d_lhs_strides, d_rhs_shape, d_rhs_strides, ndim, len, op);
            break;
        case kDTypeF16:
            status = dispatch_binary_typed_broadcast_rhs<__half>(
                lhs_handle, lhs_scale, rhs_handle, rhs_dtype, rhs_scale, out_handle, d_out_strides,
                d_lhs_shape, d_lhs_strides, d_rhs_shape, d_rhs_strides, ndim, len, op);
            break;
        case kDTypeBF16:
            status = dispatch_binary_typed_broadcast_rhs<__nv_bfloat16>(
                lhs_handle, lhs_scale, rhs_handle, rhs_dtype, rhs_scale, out_handle, d_out_strides,
                d_lhs_shape, d_lhs_strides, d_rhs_shape, d_rhs_strides, ndim, len, op);
            break;
        case kDTypeI8:
            status = dispatch_binary_typed_broadcast_rhs<int8_t>(
                lhs_handle, lhs_scale, rhs_handle, rhs_dtype, rhs_scale, out_handle, d_out_strides,
                d_lhs_shape, d_lhs_strides, d_rhs_shape, d_rhs_strides, ndim, len, op);
            break;
        default:
            set_error("unsupported lhs dtype for CUDA typed mixed broadcast binary");
            break;
    }
    return status;
}

extern "C" int lumen_cuda_binary_lowp_typed_broadcast_out_device(
    uint64_t lhs_handle,
    uint64_t rhs_handle,
    uint64_t out_handle,
    size_t ndim,
    const size_t* out_strides,
    const size_t* lhs_shape,
    const size_t* lhs_strides,
    const size_t* rhs_shape,
    const size_t* rhs_strides,
    size_t len,
    int dtype,
    int op) {
    if (!validate_handle(lhs_handle, "CUDA lowp typed-out broadcast binary lhs handle") ||
        !validate_handle(rhs_handle, "CUDA lowp typed-out broadcast binary rhs handle") ||
        !validate_handle(out_handle, "CUDA lowp typed-out broadcast binary output handle")) {
        return 1;
    }
    if (ndim == 0 || len == 0 || out_strides == nullptr || lhs_shape == nullptr ||
        lhs_strides == nullptr || rhs_shape == nullptr || rhs_strides == nullptr) {
        set_error("CUDA lowp typed-out broadcast binary received invalid metadata");
        return 1;
    }

    size_t* d_out_strides = nullptr;
    size_t* d_lhs_shape = nullptr;
    size_t* d_lhs_strides = nullptr;
    size_t* d_rhs_shape = nullptr;
    size_t* d_rhs_strides = nullptr;
    const size_t* host_metadata[] = {
        out_strides, lhs_shape, lhs_strides, rhs_shape, rhs_strides};
    size_t* device_metadata[] = {
        d_out_strides, d_lhs_shape, d_lhs_strides, d_rhs_shape, d_rhs_strides};
    if (!upload_packed_elementwise_metadata(
            "failed to upload CUDA lowp typed-out broadcast binary metadata",
            host_metadata,
            5,
            ndim,
            device_metadata)) {
        return 1;
    }
    d_out_strides = device_metadata[0];
    d_lhs_shape = device_metadata[1];
    d_lhs_strides = device_metadata[2];
    d_rhs_shape = device_metadata[3];
    d_rhs_strides = device_metadata[4];

    int status = 1;
    switch (dtype) {
        case kDTypeF16:
            launch_binary_lowp_broadcast_to_typed_kernel<__half>(
                reinterpret_cast<const __half*>(handle_to_ptr(lhs_handle)),
                reinterpret_cast<const __half*>(handle_to_ptr(rhs_handle)),
                reinterpret_cast<__half*>(handle_to_ptr(out_handle)),
                d_out_strides,
                d_lhs_shape,
                d_lhs_strides,
                d_rhs_shape,
                d_rhs_strides,
                ndim,
                len,
                op);
            status = check_cuda_launch("CUDA F16 typed-out broadcast binary launch failed") ? 0 : 1;
            break;
        case kDTypeBF16:
            launch_binary_lowp_broadcast_to_typed_kernel<__nv_bfloat16>(
                reinterpret_cast<const __nv_bfloat16*>(handle_to_ptr(lhs_handle)),
                reinterpret_cast<const __nv_bfloat16*>(handle_to_ptr(rhs_handle)),
                reinterpret_cast<__nv_bfloat16*>(handle_to_ptr(out_handle)),
                d_out_strides,
                d_lhs_shape,
                d_lhs_strides,
                d_rhs_shape,
                d_rhs_strides,
                ndim,
                len,
                op);
            status = check_cuda_launch("CUDA BF16 typed-out broadcast binary launch failed") ? 0 : 1;
            break;
        default:
            set_error("CUDA typed-out broadcast binary supports only F16/BF16");
            break;
    }
    return status;
}

extern "C" int lumen_cuda_binary_i8_typed_broadcast_out_device(
    uint64_t lhs_handle,
    uint64_t rhs_handle,
    float lhs_scale,
    float rhs_scale,
    uint64_t out_handle,
    size_t ndim,
    const size_t* out_strides,
    const size_t* lhs_shape,
    const size_t* lhs_strides,
    const size_t* rhs_shape,
    const size_t* rhs_strides,
    size_t len,
    int op,
    float* out_scale) {
    if (!validate_handle(lhs_handle, "CUDA I8 typed-out broadcast binary lhs handle") ||
        !validate_handle(rhs_handle, "CUDA I8 typed-out broadcast binary rhs handle") ||
        !validate_handle(out_handle, "CUDA I8 typed-out broadcast binary output handle")) {
        return 1;
    }
    if (out_scale == nullptr) {
        set_error("CUDA I8 typed-out broadcast binary scale output is null");
        return 1;
    }
    if (ndim == 0 || len == 0 || out_strides == nullptr || lhs_shape == nullptr ||
        lhs_strides == nullptr || rhs_shape == nullptr || rhs_strides == nullptr) {
        set_error("CUDA I8 typed-out broadcast binary received invalid metadata");
        return 1;
    }
    if (!std::isfinite(lhs_scale) || lhs_scale <= 0.0f ||
        !std::isfinite(rhs_scale) || rhs_scale <= 0.0f) {
        set_error("CUDA I8 typed-out broadcast binary scales must be finite and > 0");
        return 1;
    }

    size_t* d_out_strides = nullptr;
    size_t* d_lhs_shape = nullptr;
    size_t* d_lhs_strides = nullptr;
    size_t* d_rhs_shape = nullptr;
    size_t* d_rhs_strides = nullptr;
    const size_t* host_metadata[] = {
        out_strides, lhs_shape, lhs_strides, rhs_shape, rhs_strides};
    size_t* device_metadata[] = {
        d_out_strides, d_lhs_shape, d_lhs_strides, d_rhs_shape, d_rhs_strides};
    if (!upload_packed_elementwise_metadata(
            "failed to upload CUDA I8 typed-out broadcast binary metadata",
            host_metadata,
            5,
            ndim,
            device_metadata)) {
        return 1;
    }
    d_out_strides = device_metadata[0];
    d_lhs_shape = device_metadata[1];
    d_lhs_strides = device_metadata[2];
    d_rhs_shape = device_metadata[3];
    d_rhs_strides = device_metadata[4];

    constexpr unsigned int block_size = 256;
    const unsigned int grid_size = linear_grid_size(len, block_size);
    thread_local ReusableCudaWorkspace partial_workspace;
    if (!partial_workspace.ensure(
            static_cast<size_t>(grid_size) * sizeof(float),
            "CUDA I8 typed-out broadcast absmax allocation failed")) {
        return 1;
    }
    float* partial = static_cast<float*>(partial_workspace.ptr);

    launch_binary_i8_broadcast_absmax_blocks_kernel(
        reinterpret_cast<const int8_t*>(handle_to_ptr(lhs_handle)),
        reinterpret_cast<const int8_t*>(handle_to_ptr(rhs_handle)),
        lhs_scale,
        rhs_scale,
        partial,
        d_out_strides,
        d_lhs_shape,
        d_lhs_strides,
        d_rhs_shape,
        d_rhs_strides,
        ndim,
        len,
        grid_size,
        block_size,
        op);
    if (!check_cuda_launch("CUDA I8 typed-out broadcast absmax kernel launch failed")) {
        return 1;
    }

    float max_abs = 0.0f;
    if (!reduce_absmax_partials_to_host(
            partial,
            static_cast<size_t>(grid_size),
            &max_abs,
            "CUDA I8 typed-out broadcast absmax reduce failed")) {
        return 1;
    }

    *out_scale = max_abs > 0.0f && isfinite(max_abs) ? std::max(max_abs / 127.0f, FLT_MIN) : 1.0f;
    launch_binary_i8_broadcast_to_i8_kernel(
        reinterpret_cast<const int8_t*>(handle_to_ptr(lhs_handle)),
        reinterpret_cast<const int8_t*>(handle_to_ptr(rhs_handle)),
        lhs_scale,
        rhs_scale,
        reinterpret_cast<int8_t*>(handle_to_ptr(out_handle)),
        d_out_strides,
        d_lhs_shape,
        d_lhs_strides,
        d_rhs_shape,
        d_rhs_strides,
        ndim,
        len,
        *out_scale,
        grid_size,
        block_size,
        op);
    return check_cuda_launch("CUDA I8 typed-out broadcast quantize kernel launch failed") ? 0 : 1;
}

template <typename LhsT>
int dispatch_binary_typed_b1d_1h1_rhs(
    uint64_t lhs_handle,
    float lhs_scale,
    uint64_t rhs_handle,
    int rhs_dtype,
    float rhs_scale,
    uint64_t out_handle,
    size_t batch,
    size_t heads,
    size_t dim,
    int b1d_on_lhs,
    int op) {
    switch (rhs_dtype) {
        case kDTypeF32:
            launch_binary_typed_b1d_1h1_to_f32_kernel<LhsT, float>(
                reinterpret_cast<const LhsT*>(handle_to_ptr(lhs_handle)),
                lhs_scale,
                handle_to_ptr(rhs_handle),
                rhs_scale,
                handle_to_ptr(out_handle),
                batch,
                heads,
                dim,
                b1d_on_lhs != 0,
                op);
            return check_cuda_launch("CUDA typed mixed b1d/1h1 binary rhs=f32 launch failed") ? 0 : 1;
        case kDTypeF16:
            launch_binary_typed_b1d_1h1_to_f32_kernel<LhsT, __half>(
                reinterpret_cast<const LhsT*>(handle_to_ptr(lhs_handle)),
                lhs_scale,
                reinterpret_cast<const __half*>(handle_to_ptr(rhs_handle)),
                rhs_scale,
                handle_to_ptr(out_handle),
                batch,
                heads,
                dim,
                b1d_on_lhs != 0,
                op);
            return check_cuda_launch("CUDA typed mixed b1d/1h1 binary rhs=f16 launch failed") ? 0 : 1;
        case kDTypeBF16:
            launch_binary_typed_b1d_1h1_to_f32_kernel<LhsT, __nv_bfloat16>(
                reinterpret_cast<const LhsT*>(handle_to_ptr(lhs_handle)),
                lhs_scale,
                reinterpret_cast<const __nv_bfloat16*>(handle_to_ptr(rhs_handle)),
                rhs_scale,
                handle_to_ptr(out_handle),
                batch,
                heads,
                dim,
                b1d_on_lhs != 0,
                op);
            return check_cuda_launch("CUDA typed mixed b1d/1h1 binary rhs=bf16 launch failed") ? 0 : 1;
        case kDTypeI8:
            launch_binary_typed_b1d_1h1_to_f32_kernel<LhsT, int8_t>(
                reinterpret_cast<const LhsT*>(handle_to_ptr(lhs_handle)),
                lhs_scale,
                reinterpret_cast<const int8_t*>(handle_to_ptr(rhs_handle)),
                rhs_scale,
                handle_to_ptr(out_handle),
                batch,
                heads,
                dim,
                b1d_on_lhs != 0,
                op);
            return check_cuda_launch("CUDA typed mixed b1d/1h1 binary rhs=i8 launch failed") ? 0 : 1;
        default:
            set_error("unsupported rhs dtype for CUDA typed mixed b1d/1h1 binary");
            return 1;
    }
}

extern "C" int lumen_cuda_binary_typed_b1d_1h1_device(
    uint64_t lhs_handle,
    int lhs_dtype,
    float lhs_scale,
    uint64_t rhs_handle,
    int rhs_dtype,
    float rhs_scale,
    uint64_t out_handle,
    size_t batch,
    size_t heads,
    size_t dim,
    int b1d_on_lhs,
    int op) {
    if (!validate_handle(lhs_handle, "CUDA typed mixed b1d/1h1 binary lhs handle") ||
        !validate_handle(rhs_handle, "CUDA typed mixed b1d/1h1 binary rhs handle") ||
        !validate_handle(out_handle, "CUDA typed mixed b1d/1h1 binary output handle")) {
        return 1;
    }
    if (batch == 0 || heads == 0 || dim == 0) {
        set_error("CUDA typed mixed b1d/1h1 binary dimensions must be non-zero");
        return 1;
    }
    if ((lhs_dtype == kDTypeI8 && (!std::isfinite(lhs_scale) || lhs_scale <= 0.0f)) ||
        (rhs_dtype == kDTypeI8 && (!std::isfinite(rhs_scale) || rhs_scale <= 0.0f))) {
        set_error("CUDA typed mixed b1d/1h1 binary I8 scales must be finite and > 0");
        return 1;
    }

    switch (lhs_dtype) {
        case kDTypeF32:
            return dispatch_binary_typed_b1d_1h1_rhs<float>(
                lhs_handle, lhs_scale, rhs_handle, rhs_dtype, rhs_scale, out_handle,
                batch, heads, dim, b1d_on_lhs, op);
        case kDTypeF16:
            return dispatch_binary_typed_b1d_1h1_rhs<__half>(
                lhs_handle, lhs_scale, rhs_handle, rhs_dtype, rhs_scale, out_handle,
                batch, heads, dim, b1d_on_lhs, op);
        case kDTypeBF16:
            return dispatch_binary_typed_b1d_1h1_rhs<__nv_bfloat16>(
                lhs_handle, lhs_scale, rhs_handle, rhs_dtype, rhs_scale, out_handle,
                batch, heads, dim, b1d_on_lhs, op);
        case kDTypeI8:
            return dispatch_binary_typed_b1d_1h1_rhs<int8_t>(
                lhs_handle, lhs_scale, rhs_handle, rhs_dtype, rhs_scale, out_handle,
                batch, heads, dim, b1d_on_lhs, op);
        default:
            set_error("unsupported lhs dtype for CUDA typed mixed b1d/1h1 binary");
            return 1;
    }
}

extern "C" int lumen_cuda_binary_lowp_typed_b1d_1h1_out_device(
    uint64_t lhs_handle,
    uint64_t rhs_handle,
    uint64_t out_handle,
    size_t batch,
    size_t heads,
    size_t dim,
    int b1d_on_lhs,
    int dtype,
    int op) {
    if (!validate_handle(lhs_handle, "CUDA lowp typed-out b1d/1h1 binary lhs handle") ||
        !validate_handle(rhs_handle, "CUDA lowp typed-out b1d/1h1 binary rhs handle") ||
        !validate_handle(out_handle, "CUDA lowp typed-out b1d/1h1 binary output handle")) {
        return 1;
    }
    if (batch == 0 || heads == 0 || dim == 0) {
        set_error("CUDA lowp typed-out b1d/1h1 binary dimensions must be non-zero");
        return 1;
    }

    switch (dtype) {
        case kDTypeF16:
            launch_binary_lowp_b1d_1h1_to_typed_kernel<__half>(
                reinterpret_cast<const __half*>(handle_to_ptr(lhs_handle)),
                reinterpret_cast<const __half*>(handle_to_ptr(rhs_handle)),
                reinterpret_cast<__half*>(handle_to_ptr(out_handle)),
                batch,
                heads,
                dim,
                b1d_on_lhs != 0,
                op);
            return check_cuda_launch("CUDA F16 typed-out b1d/1h1 binary launch failed") ? 0 : 1;
        case kDTypeBF16:
            launch_binary_lowp_b1d_1h1_to_typed_kernel<__nv_bfloat16>(
                reinterpret_cast<const __nv_bfloat16*>(handle_to_ptr(lhs_handle)),
                reinterpret_cast<const __nv_bfloat16*>(handle_to_ptr(rhs_handle)),
                reinterpret_cast<__nv_bfloat16*>(handle_to_ptr(out_handle)),
                batch,
                heads,
                dim,
                b1d_on_lhs != 0,
                op);
            return check_cuda_launch("CUDA BF16 typed-out b1d/1h1 binary launch failed") ? 0 : 1;
        default:
            set_error("CUDA typed-out b1d/1h1 binary supports only F16/BF16");
            return 1;
    }
}

extern "C" int lumen_cuda_binary_i8_typed_b1d_1h1_out_device(
    uint64_t lhs_handle,
    uint64_t rhs_handle,
    float lhs_scale,
    float rhs_scale,
    uint64_t out_handle,
    size_t batch,
    size_t heads,
    size_t dim,
    int b1d_on_lhs,
    int op,
    float* out_scale) {
    if (!validate_handle(lhs_handle, "CUDA I8 typed-out b1d/1h1 binary lhs handle") ||
        !validate_handle(rhs_handle, "CUDA I8 typed-out b1d/1h1 binary rhs handle") ||
        !validate_handle(out_handle, "CUDA I8 typed-out b1d/1h1 binary output handle")) {
        return 1;
    }
    if (out_scale == nullptr) {
        set_error("CUDA I8 typed-out b1d/1h1 binary scale output is null");
        return 1;
    }
    if (batch == 0 || heads == 0 || dim == 0) {
        set_error("CUDA I8 typed-out b1d/1h1 binary dimensions must be non-zero");
        return 1;
    }
    if (!std::isfinite(lhs_scale) || lhs_scale <= 0.0f ||
        !std::isfinite(rhs_scale) || rhs_scale <= 0.0f) {
        set_error("CUDA I8 typed-out b1d/1h1 binary scales must be finite and > 0");
        return 1;
    }

    size_t len = 0;
    if (!checked_product(
            "CUDA I8 typed-out b1d/1h1 binary length overflow", {batch, heads, dim}, &len)) {
        return 1;
    }
    constexpr unsigned int block_size = 256;
    const unsigned int grid_size = linear_grid_size(len, block_size);
    thread_local ReusableCudaWorkspace partial_workspace;
    if (!partial_workspace.ensure(
            static_cast<size_t>(grid_size) * sizeof(float),
            "CUDA I8 typed-out b1d/1h1 absmax allocation failed")) {
        return 1;
    }
    float* partial = static_cast<float*>(partial_workspace.ptr);

    launch_binary_i8_b1d_1h1_absmax_blocks_kernel(
        reinterpret_cast<const int8_t*>(handle_to_ptr(lhs_handle)),
        reinterpret_cast<const int8_t*>(handle_to_ptr(rhs_handle)),
        lhs_scale,
        rhs_scale,
        partial,
        batch,
        heads,
        dim,
        b1d_on_lhs != 0,
        grid_size,
        block_size,
        op);
    if (!check_cuda_launch("CUDA I8 typed-out b1d/1h1 absmax kernel launch failed")) {
        return 1;
    }

    float max_abs = 0.0f;
    if (!reduce_absmax_partials_to_host(
            partial,
            static_cast<size_t>(grid_size),
            &max_abs,
            "CUDA I8 typed-out b1d/1h1 absmax reduce failed")) {
        return 1;
    }

    *out_scale = max_abs > 0.0f && isfinite(max_abs) ? std::max(max_abs / 127.0f, FLT_MIN) : 1.0f;
    launch_binary_i8_b1d_1h1_to_i8_kernel(
        reinterpret_cast<const int8_t*>(handle_to_ptr(lhs_handle)),
        reinterpret_cast<const int8_t*>(handle_to_ptr(rhs_handle)),
        lhs_scale,
        rhs_scale,
        reinterpret_cast<int8_t*>(handle_to_ptr(out_handle)),
        batch,
        heads,
        dim,
        b1d_on_lhs != 0,
        *out_scale,
        grid_size,
        block_size,
        op);
    return check_cuda_launch("CUDA I8 typed-out b1d/1h1 quantize kernel launch failed") ? 0 : 1;
}

template <typename LhsT>
int dispatch_binary_typed_b1d_1hd_rhs(
    uint64_t lhs_handle,
    float lhs_scale,
    uint64_t rhs_handle,
    int rhs_dtype,
    float rhs_scale,
    uint64_t out_handle,
    size_t batch,
    size_t heads,
    size_t dim,
    int b1d_on_lhs,
    int op) {
    switch (rhs_dtype) {
        case kDTypeF32:
            launch_binary_typed_b1d_1hd_to_f32_kernel<LhsT, float>(
                reinterpret_cast<const LhsT*>(handle_to_ptr(lhs_handle)),
                lhs_scale,
                handle_to_ptr(rhs_handle),
                rhs_scale,
                handle_to_ptr(out_handle),
                batch,
                heads,
                dim,
                b1d_on_lhs != 0,
                op);
            return check_cuda_launch("CUDA typed mixed b1d/1hd binary rhs=f32 launch failed") ? 0 : 1;
        case kDTypeF16:
            launch_binary_typed_b1d_1hd_to_f32_kernel<LhsT, __half>(
                reinterpret_cast<const LhsT*>(handle_to_ptr(lhs_handle)),
                lhs_scale,
                reinterpret_cast<const __half*>(handle_to_ptr(rhs_handle)),
                rhs_scale,
                handle_to_ptr(out_handle),
                batch,
                heads,
                dim,
                b1d_on_lhs != 0,
                op);
            return check_cuda_launch("CUDA typed mixed b1d/1hd binary rhs=f16 launch failed") ? 0 : 1;
        case kDTypeBF16:
            launch_binary_typed_b1d_1hd_to_f32_kernel<LhsT, __nv_bfloat16>(
                reinterpret_cast<const LhsT*>(handle_to_ptr(lhs_handle)),
                lhs_scale,
                reinterpret_cast<const __nv_bfloat16*>(handle_to_ptr(rhs_handle)),
                rhs_scale,
                handle_to_ptr(out_handle),
                batch,
                heads,
                dim,
                b1d_on_lhs != 0,
                op);
            return check_cuda_launch("CUDA typed mixed b1d/1hd binary rhs=bf16 launch failed") ? 0 : 1;
        case kDTypeI8:
            launch_binary_typed_b1d_1hd_to_f32_kernel<LhsT, int8_t>(
                reinterpret_cast<const LhsT*>(handle_to_ptr(lhs_handle)),
                lhs_scale,
                reinterpret_cast<const int8_t*>(handle_to_ptr(rhs_handle)),
                rhs_scale,
                handle_to_ptr(out_handle),
                batch,
                heads,
                dim,
                b1d_on_lhs != 0,
                op);
            return check_cuda_launch("CUDA typed mixed b1d/1hd binary rhs=i8 launch failed") ? 0 : 1;
        default:
            set_error("unsupported rhs dtype for CUDA typed mixed b1d/1hd binary");
            return 1;
    }
}

extern "C" int lumen_cuda_binary_typed_b1d_1hd_device(
    uint64_t lhs_handle,
    int lhs_dtype,
    float lhs_scale,
    uint64_t rhs_handle,
    int rhs_dtype,
    float rhs_scale,
    uint64_t out_handle,
    size_t batch,
    size_t heads,
    size_t dim,
    int b1d_on_lhs,
    int op) {
    if (!validate_handle(lhs_handle, "CUDA typed mixed b1d/1hd binary lhs handle") ||
        !validate_handle(rhs_handle, "CUDA typed mixed b1d/1hd binary rhs handle") ||
        !validate_handle(out_handle, "CUDA typed mixed b1d/1hd binary output handle")) {
        return 1;
    }
    if (batch == 0 || heads == 0 || dim == 0) {
        set_error("CUDA typed mixed b1d/1hd binary dimensions must be non-zero");
        return 1;
    }
    if ((lhs_dtype == kDTypeI8 && (!std::isfinite(lhs_scale) || lhs_scale <= 0.0f)) ||
        (rhs_dtype == kDTypeI8 && (!std::isfinite(rhs_scale) || rhs_scale <= 0.0f))) {
        set_error("CUDA typed mixed b1d/1hd binary I8 scales must be finite and > 0");
        return 1;
    }

    switch (lhs_dtype) {
        case kDTypeF32:
            return dispatch_binary_typed_b1d_1hd_rhs<float>(
                lhs_handle, lhs_scale, rhs_handle, rhs_dtype, rhs_scale, out_handle,
                batch, heads, dim, b1d_on_lhs, op);
        case kDTypeF16:
            return dispatch_binary_typed_b1d_1hd_rhs<__half>(
                lhs_handle, lhs_scale, rhs_handle, rhs_dtype, rhs_scale, out_handle,
                batch, heads, dim, b1d_on_lhs, op);
        case kDTypeBF16:
            return dispatch_binary_typed_b1d_1hd_rhs<__nv_bfloat16>(
                lhs_handle, lhs_scale, rhs_handle, rhs_dtype, rhs_scale, out_handle,
                batch, heads, dim, b1d_on_lhs, op);
        case kDTypeI8:
            return dispatch_binary_typed_b1d_1hd_rhs<int8_t>(
                lhs_handle, lhs_scale, rhs_handle, rhs_dtype, rhs_scale, out_handle,
                batch, heads, dim, b1d_on_lhs, op);
        default:
            set_error("unsupported lhs dtype for CUDA typed mixed b1d/1hd binary");
            return 1;
    }
}

extern "C" int lumen_cuda_binary_lowp_typed_b1d_1hd_out_device(
    uint64_t lhs_handle,
    uint64_t rhs_handle,
    uint64_t out_handle,
    size_t batch,
    size_t heads,
    size_t dim,
    int b1d_on_lhs,
    int dtype,
    int op) {
    if (!validate_handle(lhs_handle, "CUDA lowp typed-out b1d/1hd binary lhs handle") ||
        !validate_handle(rhs_handle, "CUDA lowp typed-out b1d/1hd binary rhs handle") ||
        !validate_handle(out_handle, "CUDA lowp typed-out b1d/1hd binary output handle")) {
        return 1;
    }
    if (batch == 0 || heads == 0 || dim == 0) {
        set_error("CUDA lowp typed-out b1d/1hd binary dimensions must be non-zero");
        return 1;
    }

    switch (dtype) {
        case kDTypeF16:
            launch_binary_lowp_b1d_1hd_to_typed_kernel<__half>(
                reinterpret_cast<const __half*>(handle_to_ptr(lhs_handle)),
                reinterpret_cast<const __half*>(handle_to_ptr(rhs_handle)),
                reinterpret_cast<__half*>(handle_to_ptr(out_handle)),
                batch,
                heads,
                dim,
                b1d_on_lhs != 0,
                op);
            return check_cuda_launch("CUDA F16 typed-out b1d/1hd binary launch failed") ? 0 : 1;
        case kDTypeBF16:
            launch_binary_lowp_b1d_1hd_to_typed_kernel<__nv_bfloat16>(
                reinterpret_cast<const __nv_bfloat16*>(handle_to_ptr(lhs_handle)),
                reinterpret_cast<const __nv_bfloat16*>(handle_to_ptr(rhs_handle)),
                reinterpret_cast<__nv_bfloat16*>(handle_to_ptr(out_handle)),
                batch,
                heads,
                dim,
                b1d_on_lhs != 0,
                op);
            return check_cuda_launch("CUDA BF16 typed-out b1d/1hd binary launch failed") ? 0 : 1;
        default:
            set_error("CUDA typed-out b1d/1hd binary supports only F16/BF16");
            return 1;
    }
}

extern "C" int lumen_cuda_binary_i8_typed_b1d_1hd_out_device(
    uint64_t lhs_handle,
    uint64_t rhs_handle,
    float lhs_scale,
    float rhs_scale,
    uint64_t out_handle,
    size_t batch,
    size_t heads,
    size_t dim,
    int b1d_on_lhs,
    int op,
    float* out_scale) {
    if (!validate_handle(lhs_handle, "CUDA I8 typed-out b1d/1hd binary lhs handle") ||
        !validate_handle(rhs_handle, "CUDA I8 typed-out b1d/1hd binary rhs handle") ||
        !validate_handle(out_handle, "CUDA I8 typed-out b1d/1hd binary output handle")) {
        return 1;
    }
    if (out_scale == nullptr) {
        set_error("CUDA I8 typed-out b1d/1hd binary scale output is null");
        return 1;
    }
    if (batch == 0 || heads == 0 || dim == 0) {
        set_error("CUDA I8 typed-out b1d/1hd binary dimensions must be non-zero");
        return 1;
    }
    if (!std::isfinite(lhs_scale) || lhs_scale <= 0.0f ||
        !std::isfinite(rhs_scale) || rhs_scale <= 0.0f) {
        set_error("CUDA I8 typed-out b1d/1hd binary scales must be finite and > 0");
        return 1;
    }

    size_t len = 0;
    if (!checked_product(
            "CUDA I8 typed-out b1d/1hd binary length overflow", {batch, heads, dim}, &len)) {
        return 1;
    }
    constexpr unsigned int block_size = 256;
    const unsigned int grid_size = linear_grid_size(len, block_size);
    thread_local ReusableCudaWorkspace partial_workspace;
    if (!partial_workspace.ensure(
            static_cast<size_t>(grid_size) * sizeof(float),
            "CUDA I8 typed-out b1d/1hd absmax allocation failed")) {
        return 1;
    }
    float* partial = static_cast<float*>(partial_workspace.ptr);

    launch_binary_i8_b1d_1hd_absmax_blocks_kernel(
        reinterpret_cast<const int8_t*>(handle_to_ptr(lhs_handle)),
        reinterpret_cast<const int8_t*>(handle_to_ptr(rhs_handle)),
        lhs_scale,
        rhs_scale,
        partial,
        batch,
        heads,
        dim,
        b1d_on_lhs != 0,
        grid_size,
        block_size,
        op);
    if (!check_cuda_launch("CUDA I8 typed-out b1d/1hd absmax kernel launch failed")) {
        return 1;
    }

    float max_abs = 0.0f;
    if (!reduce_absmax_partials_to_host(
            partial,
            static_cast<size_t>(grid_size),
            &max_abs,
            "CUDA I8 typed-out b1d/1hd absmax reduce failed")) {
        return 1;
    }

    *out_scale = max_abs > 0.0f && isfinite(max_abs) ? std::max(max_abs / 127.0f, FLT_MIN) : 1.0f;
    launch_binary_i8_b1d_1hd_to_i8_kernel(
        reinterpret_cast<const int8_t*>(handle_to_ptr(lhs_handle)),
        reinterpret_cast<const int8_t*>(handle_to_ptr(rhs_handle)),
        lhs_scale,
        rhs_scale,
        reinterpret_cast<int8_t*>(handle_to_ptr(out_handle)),
        batch,
        heads,
        dim,
        b1d_on_lhs != 0,
        *out_scale,
        grid_size,
        block_size,
        op);
    return check_cuda_launch("CUDA I8 typed-out b1d/1hd quantize kernel launch failed") ? 0 : 1;
}

extern "C" int lumen_cuda_binary_f16_host_device(
    const uint16_t* lhs_host,
    const uint16_t* rhs_host,
    uint64_t out_handle,
    size_t len,
    int op) {
    if (!validate_handle(out_handle, "CUDA F16 binary output handle")) {
        return 1;
    }
    if (len == 0) {
        return 0;
    }

    ScopedDeviceInput<__half> lhs_dev;
    ScopedDeviceInput<__half> rhs_dev;
    if (!upload_typed_input(reinterpret_cast<const __half*>(lhs_host), len, &lhs_dev.ptr, "upload F16 binary lhs") ||
        !upload_typed_input(reinterpret_cast<const __half*>(rhs_host), len, &rhs_dev.ptr, "upload F16 binary rhs")) {
        return 1;
    }

    launch_binary_lowp_to_f32_kernel<__half>(
        lhs_dev.ptr,
        rhs_dev.ptr,
        handle_to_ptr(out_handle),
        len,
        op);
    return check_cuda_launch("CUDA F16 binary kernel launch failed") ? 0 : 1;
}

extern "C" int lumen_cuda_binary_f16_device(
    uint64_t lhs_handle,
    uint64_t rhs_handle,
    uint64_t out_handle,
    size_t len,
    int op) {
    if (!validate_handle(lhs_handle, "CUDA F16 binary lhs handle") ||
        !validate_handle(rhs_handle, "CUDA F16 binary rhs handle") ||
        !validate_handle(out_handle, "CUDA F16 binary output handle")) {
        return 1;
    }
    if (len == 0) {
        return 0;
    }

    launch_binary_lowp_to_f32_kernel<__half>(
        reinterpret_cast<const __half*>(handle_to_ptr(lhs_handle)),
        reinterpret_cast<const __half*>(handle_to_ptr(rhs_handle)),
        handle_to_ptr(out_handle),
        len,
        op);
    return check_cuda_launch("CUDA F16 resident binary kernel launch failed") ? 0 : 1;
}

extern "C" int lumen_cuda_binary_f16_lastdim_device(
    uint64_t lhs_handle,
    uint64_t rhs_handle,
    uint64_t out_handle,
    size_t len,
    size_t last_dim,
    int vector_on_rhs,
    int op) {
    if (!validate_handle(lhs_handle, "CUDA F16 row-broadcast binary lhs handle") ||
        !validate_handle(rhs_handle, "CUDA F16 row-broadcast binary rhs handle") ||
        !validate_handle(out_handle, "CUDA F16 row-broadcast binary output handle")) {
        return 1;
    }
    if (last_dim == 0) {
        set_error("CUDA F16 row-broadcast last_dim must be greater than zero");
        return 1;
    }

    launch_binary_lowp_lastdim_to_f32_kernel<__half>(
        reinterpret_cast<const __half*>(lhs_handle),
        reinterpret_cast<const __half*>(rhs_handle),
        handle_to_ptr(out_handle),
        len,
        last_dim,
        vector_on_rhs != 0,
        op);
    return check_cuda_launch("CUDA F16 row-broadcast binary kernel launch failed") ? 0 : 1;
}

extern "C" int lumen_cuda_binary_bf16_host_device(
    const uint16_t* lhs_host,
    const uint16_t* rhs_host,
    uint64_t out_handle,
    size_t len,
    int op) {
    if (!validate_handle(out_handle, "CUDA BF16 binary output handle")) {
        return 1;
    }
    if (len == 0) {
        return 0;
    }

    ScopedDeviceInput<__nv_bfloat16> lhs_dev;
    ScopedDeviceInput<__nv_bfloat16> rhs_dev;
    if (!upload_typed_input(reinterpret_cast<const __nv_bfloat16*>(lhs_host), len, &lhs_dev.ptr, "upload BF16 binary lhs") ||
        !upload_typed_input(reinterpret_cast<const __nv_bfloat16*>(rhs_host), len, &rhs_dev.ptr, "upload BF16 binary rhs")) {
        return 1;
    }

    launch_binary_lowp_to_f32_kernel<__nv_bfloat16>(
        lhs_dev.ptr,
        rhs_dev.ptr,
        handle_to_ptr(out_handle),
        len,
        op);
    return check_cuda_launch("CUDA BF16 binary kernel launch failed") ? 0 : 1;
}

extern "C" int lumen_cuda_binary_bf16_device(
    uint64_t lhs_handle,
    uint64_t rhs_handle,
    uint64_t out_handle,
    size_t len,
    int op) {
    if (!validate_handle(lhs_handle, "CUDA BF16 binary lhs handle") ||
        !validate_handle(rhs_handle, "CUDA BF16 binary rhs handle") ||
        !validate_handle(out_handle, "CUDA BF16 binary output handle")) {
        return 1;
    }
    if (len == 0) {
        return 0;
    }

    launch_binary_lowp_to_f32_kernel<__nv_bfloat16>(
        reinterpret_cast<const __nv_bfloat16*>(handle_to_ptr(lhs_handle)),
        reinterpret_cast<const __nv_bfloat16*>(handle_to_ptr(rhs_handle)),
        handle_to_ptr(out_handle),
        len,
        op);
    return check_cuda_launch("CUDA BF16 resident binary kernel launch failed") ? 0 : 1;
}

extern "C" int lumen_cuda_binary_bf16_lastdim_device(
    uint64_t lhs_handle,
    uint64_t rhs_handle,
    uint64_t out_handle,
    size_t len,
    size_t last_dim,
    int vector_on_rhs,
    int op) {
    if (!validate_handle(lhs_handle, "CUDA BF16 row-broadcast binary lhs handle") ||
        !validate_handle(rhs_handle, "CUDA BF16 row-broadcast binary rhs handle") ||
        !validate_handle(out_handle, "CUDA BF16 row-broadcast binary output handle")) {
        return 1;
    }
    if (last_dim == 0) {
        set_error("CUDA BF16 row-broadcast last_dim must be greater than zero");
        return 1;
    }

    launch_binary_lowp_lastdim_to_f32_kernel<__nv_bfloat16>(
        reinterpret_cast<const __nv_bfloat16*>(lhs_handle),
        reinterpret_cast<const __nv_bfloat16*>(rhs_handle),
        handle_to_ptr(out_handle),
        len,
        last_dim,
        vector_on_rhs != 0,
        op);
    return check_cuda_launch("CUDA BF16 row-broadcast binary kernel launch failed") ? 0 : 1;
}

extern "C" int lumen_cuda_binary_i8_host_device(
    const int8_t* lhs_host,
    const int8_t* rhs_host,
    float lhs_scale,
    float rhs_scale,
    uint64_t out_handle,
    size_t len,
    int op) {
    if (!validate_handle(out_handle, "CUDA I8 binary output handle")) {
        return 1;
    }
    if (len == 0) {
        return 0;
    }
    if (!std::isfinite(lhs_scale) || lhs_scale <= 0.0f ||
        !std::isfinite(rhs_scale) || rhs_scale <= 0.0f) {
        set_error("CUDA I8 binary scales must be finite and > 0");
        return 1;
    }

    ScopedDeviceInput<int8_t> lhs_dev;
    ScopedDeviceInput<int8_t> rhs_dev;
    if (!upload_typed_input(lhs_host, len, &lhs_dev.ptr, "upload I8 binary lhs") ||
        !upload_typed_input(rhs_host, len, &rhs_dev.ptr, "upload I8 binary rhs")) {
        return 1;
    }

    launch_binary_i8_to_f32_kernel(
        lhs_dev.ptr,
        rhs_dev.ptr,
        lhs_scale,
        rhs_scale,
        handle_to_ptr(out_handle),
        len,
        op);
    return check_cuda_launch("CUDA I8 binary kernel launch failed") ? 0 : 1;
}

extern "C" int lumen_cuda_binary_i8_device(
    uint64_t lhs_handle,
    uint64_t rhs_handle,
    float lhs_scale,
    float rhs_scale,
    uint64_t out_handle,
    size_t len,
    int op) {
    if (!validate_handle(lhs_handle, "CUDA I8 binary lhs handle") ||
        !validate_handle(rhs_handle, "CUDA I8 binary rhs handle") ||
        !validate_handle(out_handle, "CUDA I8 binary output handle")) {
        return 1;
    }
    if (!std::isfinite(lhs_scale) || lhs_scale <= 0.0f ||
        !std::isfinite(rhs_scale) || rhs_scale <= 0.0f) {
        set_error("CUDA I8 binary scales must be finite and > 0");
        return 1;
    }
    if (len == 0) {
        return 0;
    }

    launch_binary_i8_to_f32_kernel(
        reinterpret_cast<const int8_t*>(handle_to_ptr(lhs_handle)),
        reinterpret_cast<const int8_t*>(handle_to_ptr(rhs_handle)),
        lhs_scale,
        rhs_scale,
        handle_to_ptr(out_handle),
        len,
        op);
    return check_cuda_launch("CUDA I8 resident binary kernel launch failed") ? 0 : 1;
}

extern "C" int lumen_cuda_binary_i8_typed_out_device(
    uint64_t lhs_handle,
    uint64_t rhs_handle,
    float lhs_scale,
    float rhs_scale,
    uint64_t out_handle,
    size_t len,
    int op,
    float* out_scale) {
    if (!validate_handle(lhs_handle, "CUDA I8 typed-out binary lhs handle") ||
        !validate_handle(rhs_handle, "CUDA I8 typed-out binary rhs handle") ||
        !validate_handle(out_handle, "CUDA I8 typed-out binary output handle")) {
        return 1;
    }
    if (out_scale == nullptr) {
        set_error("CUDA I8 typed-out binary scale output is null");
        return 1;
    }
    if (!std::isfinite(lhs_scale) || lhs_scale <= 0.0f ||
        !std::isfinite(rhs_scale) || rhs_scale <= 0.0f) {
        set_error("CUDA I8 typed-out binary scales must be finite and > 0");
        return 1;
    }
    if (len == 0) {
        *out_scale = 1.0f;
        return 0;
    }

    constexpr unsigned int block_size = 256;
    const unsigned int grid_size = linear_grid_size(len, block_size);
    thread_local ReusableCudaWorkspace partial_workspace;
    if (!partial_workspace.ensure(
            static_cast<size_t>(grid_size) * sizeof(float),
            "CUDA I8 typed-out binary absmax allocation failed")) {
        return 1;
    }
    float* partial = static_cast<float*>(partial_workspace.ptr);

    launch_binary_i8_absmax_blocks_kernel(
        reinterpret_cast<const int8_t*>(handle_to_ptr(lhs_handle)),
        reinterpret_cast<const int8_t*>(handle_to_ptr(rhs_handle)),
        lhs_scale,
        rhs_scale,
        partial,
        len,
        grid_size,
        block_size,
        op);
    if (!check_cuda_launch("CUDA I8 typed-out binary absmax kernel launch failed")) {
        return 1;
    }

    thread_local ReusableCudaWorkspace device_max_workspace;
    if (!device_max_workspace.ensure(
            sizeof(float),
            "CUDA I8 typed-out binary final absmax allocation failed")) {
        return 1;
    }
    float* device_max = static_cast<float*>(device_max_workspace.ptr);
    f32_absmax_finalize_kernel<<<1, block_size, block_size * sizeof(float)>>>(
        partial,
        device_max,
        static_cast<size_t>(grid_size));
    if (!check_cuda_launch("CUDA I8 typed-out binary absmax reduce failed")) {
        return 1;
    }

    launch_binary_i8_to_i8_device_absmax_kernel(
        reinterpret_cast<const int8_t*>(handle_to_ptr(lhs_handle)),
        reinterpret_cast<const int8_t*>(handle_to_ptr(rhs_handle)),
        lhs_scale,
        rhs_scale,
        reinterpret_cast<int8_t*>(handle_to_ptr(out_handle)),
        len,
        device_max,
        grid_size,
        block_size,
        op);
    if (!check_cuda_launch("CUDA I8 typed-out binary quantize kernel launch failed")) {
        return 1;
    }

    float max_abs = 0.0f;
    cudaError_t status =
        cudaMemcpy(&max_abs, device_max, sizeof(float), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        set_cuda_error("failed to download CUDA I8 typed-out binary absmax", status);
        return 1;
    }

    *out_scale = max_abs > 0.0f && isfinite(max_abs) ? std::max(max_abs / 127.0f, FLT_MIN) : 1.0f;
    return 0;
}

extern "C" int lumen_cuda_binary_i8_lastdim_device(
    uint64_t lhs_handle,
    uint64_t rhs_handle,
    float lhs_scale,
    float rhs_scale,
    uint64_t out_handle,
    size_t len,
    size_t last_dim,
    int vector_on_rhs,
    int op) {
    if (!validate_handle(lhs_handle, "CUDA I8 row-broadcast binary lhs handle") ||
        !validate_handle(rhs_handle, "CUDA I8 row-broadcast binary rhs handle") ||
        !validate_handle(out_handle, "CUDA I8 row-broadcast binary output handle")) {
        return 1;
    }
    if (last_dim == 0) {
        set_error("CUDA I8 row-broadcast last_dim must be greater than zero");
        return 1;
    }
    if (!std::isfinite(lhs_scale) || lhs_scale <= 0.0f ||
        !std::isfinite(rhs_scale) || rhs_scale <= 0.0f) {
        set_error("CUDA I8 row-broadcast binary scales must be finite and > 0");
        return 1;
    }

    launch_binary_i8_lastdim_to_f32_kernel(
        reinterpret_cast<const int8_t*>(handle_to_ptr(lhs_handle)),
        reinterpret_cast<const int8_t*>(handle_to_ptr(rhs_handle)),
        lhs_scale,
        rhs_scale,
        handle_to_ptr(out_handle),
        len,
        last_dim,
        vector_on_rhs != 0,
        op);
    return check_cuda_launch("CUDA I8 row-broadcast binary kernel launch failed") ? 0 : 1;
}

extern "C" int lumen_cuda_binary_i8_typed_lastdim_out_device(
    uint64_t lhs_handle,
    uint64_t rhs_handle,
    float lhs_scale,
    float rhs_scale,
    uint64_t out_handle,
    size_t len,
    size_t last_dim,
    int vector_on_rhs,
    int op,
    float* out_scale) {
    if (!validate_handle(lhs_handle, "CUDA I8 typed-out lastdim binary lhs handle") ||
        !validate_handle(rhs_handle, "CUDA I8 typed-out lastdim binary rhs handle") ||
        !validate_handle(out_handle, "CUDA I8 typed-out lastdim binary output handle")) {
        return 1;
    }
    if (out_scale == nullptr) {
        set_error("CUDA I8 typed-out lastdim binary scale output is null");
        return 1;
    }
    if (last_dim == 0) {
        set_error("CUDA I8 typed-out lastdim binary last_dim must be greater than zero");
        return 1;
    }
    if (!std::isfinite(lhs_scale) || lhs_scale <= 0.0f ||
        !std::isfinite(rhs_scale) || rhs_scale <= 0.0f) {
        set_error("CUDA I8 typed-out lastdim binary scales must be finite and > 0");
        return 1;
    }
    if (len == 0) {
        *out_scale = 1.0f;
        return 0;
    }

    constexpr unsigned int block_size = 256;
    const unsigned int grid_size = linear_grid_size(len, block_size);
    thread_local ReusableCudaWorkspace partial_workspace;
    if (!partial_workspace.ensure(
            static_cast<size_t>(grid_size) * sizeof(float),
            "CUDA I8 typed-out lastdim binary absmax allocation failed")) {
        return 1;
    }
    float* partial = static_cast<float*>(partial_workspace.ptr);

    launch_binary_i8_lastdim_absmax_blocks_kernel(
        reinterpret_cast<const int8_t*>(handle_to_ptr(lhs_handle)),
        reinterpret_cast<const int8_t*>(handle_to_ptr(rhs_handle)),
        lhs_scale,
        rhs_scale,
        partial,
        len,
        last_dim,
        vector_on_rhs != 0,
        grid_size,
        block_size,
        op);
    if (!check_cuda_launch("CUDA I8 typed-out lastdim binary absmax kernel launch failed")) {
        return 1;
    }

    float max_abs = 0.0f;
    if (!reduce_absmax_partials_to_host(
            partial,
            static_cast<size_t>(grid_size),
            &max_abs,
            "CUDA I8 typed-out lastdim binary absmax reduce failed")) {
        return 1;
    }

    *out_scale = max_abs > 0.0f && isfinite(max_abs) ? std::max(max_abs / 127.0f, FLT_MIN) : 1.0f;
    launch_binary_i8_lastdim_to_i8_kernel(
        reinterpret_cast<const int8_t*>(handle_to_ptr(lhs_handle)),
        reinterpret_cast<const int8_t*>(handle_to_ptr(rhs_handle)),
        lhs_scale,
        rhs_scale,
        reinterpret_cast<int8_t*>(handle_to_ptr(out_handle)),
        len,
        last_dim,
        vector_on_rhs != 0,
        *out_scale,
        grid_size,
        block_size,
        op);
    return check_cuda_launch("CUDA I8 typed-out lastdim binary quantize kernel launch failed") ? 0 : 1;
}

extern "C" int lumen_cuda_mul_grad_f16_host_device(
    uint64_t grad_handle,
    const uint16_t* operand_host,
    uint64_t out_handle,
    size_t len) {
    if (!validate_handle(grad_handle, "CUDA F16 mul grad upstream handle") ||
        !validate_handle(out_handle, "CUDA F16 mul grad output handle")) {
        return 1;
    }
    if (len == 0) {
        return 0;
    }

    ScopedDeviceInput<__half> operand_dev;
    if (!upload_typed_input(reinterpret_cast<const __half*>(operand_host), len, &operand_dev.ptr, "upload F16 mul grad operand")) {
        return 1;
    }

    launch_mul_grad_lowp_to_f32_kernel<__half>(
        handle_to_ptr(grad_handle),
        operand_dev.ptr,
        handle_to_ptr(out_handle),
        len);
    return check_cuda_launch("CUDA F16 mul grad kernel launch failed") ? 0 : 1;
}

extern "C" int lumen_cuda_mul_grad_f16_device(
    uint64_t grad_handle,
    uint64_t operand_handle,
    uint64_t out_handle,
    size_t len) {
    if (!validate_handle(grad_handle, "CUDA F16 mul grad upstream handle") ||
        !validate_handle(operand_handle, "CUDA F16 mul grad operand handle") ||
        !validate_handle(out_handle, "CUDA F16 mul grad output handle")) {
        return 1;
    }
    if (len == 0) {
        return 0;
    }

    launch_mul_grad_lowp_to_f32_kernel<__half>(
        handle_to_ptr(grad_handle),
        reinterpret_cast<const __half*>(handle_to_ptr(operand_handle)),
        handle_to_ptr(out_handle),
        len);
    return check_cuda_launch("CUDA F16 resident mul grad kernel launch failed") ? 0 : 1;
}

extern "C" int lumen_cuda_mul_grad_f16_lastdim_device(
    uint64_t grad_handle,
    uint64_t lhs_handle,
    uint64_t rhs_handle,
    uint64_t grad_lhs_handle,
    uint64_t grad_rhs_handle,
    size_t len,
    size_t last_dim,
    int vector_on_rhs) {
    if (!validate_handle(grad_handle, "CUDA F16 row-broadcast mul grad upstream handle") ||
        !validate_handle(lhs_handle, "CUDA F16 row-broadcast mul grad lhs handle") ||
        !validate_handle(rhs_handle, "CUDA F16 row-broadcast mul grad rhs handle") ||
        !validate_handle(grad_lhs_handle, "CUDA F16 row-broadcast mul grad lhs output handle") ||
        !validate_handle(grad_rhs_handle, "CUDA F16 row-broadcast mul grad rhs output handle")) {
        return 1;
    }
    if (last_dim == 0) {
        set_error("CUDA F16 row-broadcast mul grad last_dim must be greater than zero");
        return 1;
    }
    float* reduced_grad = vector_on_rhs != 0 ? handle_to_ptr(grad_rhs_handle) : handle_to_ptr(grad_lhs_handle);
    if (!zero_f32_buffer(
            reduced_grad,
            last_dim,
            "CUDA F16 row-broadcast mul grad vector initialization failed")) {
        return 1;
    }

    launch_mul_grad_lowp_row_broadcast_kernel<__half>(
        handle_to_ptr(grad_handle),
        reinterpret_cast<const __half*>(lhs_handle),
        reinterpret_cast<const __half*>(rhs_handle),
        handle_to_ptr(grad_lhs_handle),
        handle_to_ptr(grad_rhs_handle),
        len,
        last_dim,
        vector_on_rhs != 0);
    return check_cuda_launch("CUDA F16 row-broadcast mul grad kernel launch failed") ? 0 : 1;
}

extern "C" int lumen_cuda_mul_grad_bf16_host_device(
    uint64_t grad_handle,
    const uint16_t* operand_host,
    uint64_t out_handle,
    size_t len) {
    if (!validate_handle(grad_handle, "CUDA BF16 mul grad upstream handle") ||
        !validate_handle(out_handle, "CUDA BF16 mul grad output handle")) {
        return 1;
    }
    if (len == 0) {
        return 0;
    }

    ScopedDeviceInput<__nv_bfloat16> operand_dev;
    if (!upload_typed_input(reinterpret_cast<const __nv_bfloat16*>(operand_host), len, &operand_dev.ptr, "upload BF16 mul grad operand")) {
        return 1;
    }

    launch_mul_grad_lowp_to_f32_kernel<__nv_bfloat16>(
        handle_to_ptr(grad_handle),
        operand_dev.ptr,
        handle_to_ptr(out_handle),
        len);
    return check_cuda_launch("CUDA BF16 mul grad kernel launch failed") ? 0 : 1;
}

extern "C" int lumen_cuda_mul_grad_bf16_device(
    uint64_t grad_handle,
    uint64_t operand_handle,
    uint64_t out_handle,
    size_t len) {
    if (!validate_handle(grad_handle, "CUDA BF16 mul grad upstream handle") ||
        !validate_handle(operand_handle, "CUDA BF16 mul grad operand handle") ||
        !validate_handle(out_handle, "CUDA BF16 mul grad output handle")) {
        return 1;
    }
    if (len == 0) {
        return 0;
    }

    launch_mul_grad_lowp_to_f32_kernel<__nv_bfloat16>(
        handle_to_ptr(grad_handle),
        reinterpret_cast<const __nv_bfloat16*>(handle_to_ptr(operand_handle)),
        handle_to_ptr(out_handle),
        len);
    return check_cuda_launch("CUDA BF16 resident mul grad kernel launch failed") ? 0 : 1;
}

extern "C" int lumen_cuda_mul_grad_bf16_lastdim_device(
    uint64_t grad_handle,
    uint64_t lhs_handle,
    uint64_t rhs_handle,
    uint64_t grad_lhs_handle,
    uint64_t grad_rhs_handle,
    size_t len,
    size_t last_dim,
    int vector_on_rhs) {
    if (!validate_handle(grad_handle, "CUDA BF16 row-broadcast mul grad upstream handle") ||
        !validate_handle(lhs_handle, "CUDA BF16 row-broadcast mul grad lhs handle") ||
        !validate_handle(rhs_handle, "CUDA BF16 row-broadcast mul grad rhs handle") ||
        !validate_handle(grad_lhs_handle, "CUDA BF16 row-broadcast mul grad lhs output handle") ||
        !validate_handle(grad_rhs_handle, "CUDA BF16 row-broadcast mul grad rhs output handle")) {
        return 1;
    }
    if (last_dim == 0) {
        set_error("CUDA BF16 row-broadcast mul grad last_dim must be greater than zero");
        return 1;
    }
    float* reduced_grad = vector_on_rhs != 0 ? handle_to_ptr(grad_rhs_handle) : handle_to_ptr(grad_lhs_handle);
    if (!zero_f32_buffer(
            reduced_grad,
            last_dim,
            "CUDA BF16 row-broadcast mul grad vector initialization failed")) {
        return 1;
    }

    launch_mul_grad_lowp_row_broadcast_kernel<__nv_bfloat16>(
        handle_to_ptr(grad_handle),
        reinterpret_cast<const __nv_bfloat16*>(lhs_handle),
        reinterpret_cast<const __nv_bfloat16*>(rhs_handle),
        handle_to_ptr(grad_lhs_handle),
        handle_to_ptr(grad_rhs_handle),
        len,
        last_dim,
        vector_on_rhs != 0);
    return check_cuda_launch("CUDA BF16 row-broadcast mul grad kernel launch failed") ? 0 : 1;
}

extern "C" int lumen_cuda_mul_grad_i8_host_device(
    uint64_t grad_handle,
    const int8_t* operand_host,
    float scale,
    uint64_t out_handle,
    size_t len) {
    if (!validate_handle(grad_handle, "CUDA I8 mul grad upstream handle") ||
        !validate_handle(out_handle, "CUDA I8 mul grad output handle")) {
        return 1;
    }
    if (len == 0) {
        return 0;
    }
    if (!std::isfinite(scale) || scale <= 0.0f) {
        set_error("CUDA I8 mul grad scale must be finite and > 0");
        return 1;
    }

    ScopedDeviceInput<int8_t> operand_dev;
    if (!upload_typed_input(operand_host, len, &operand_dev.ptr, "upload I8 mul grad operand")) {
        return 1;
    }

    launch_mul_grad_i8_to_f32_kernel(
        handle_to_ptr(grad_handle),
        operand_dev.ptr,
        scale,
        handle_to_ptr(out_handle),
        len);
    return check_cuda_launch("CUDA I8 mul grad kernel launch failed") ? 0 : 1;
}

extern "C" int lumen_cuda_mul_grad_i8_device(
    uint64_t grad_handle,
    uint64_t operand_handle,
    float scale,
    uint64_t out_handle,
    size_t len) {
    if (!validate_handle(grad_handle, "CUDA I8 mul grad upstream handle") ||
        !validate_handle(operand_handle, "CUDA I8 mul grad operand handle") ||
        !validate_handle(out_handle, "CUDA I8 mul grad output handle")) {
        return 1;
    }
    if (!std::isfinite(scale) || scale <= 0.0f) {
        set_error("CUDA I8 mul grad scale must be finite and > 0");
        return 1;
    }
    if (len == 0) {
        return 0;
    }

    launch_mul_grad_i8_to_f32_kernel(
        handle_to_ptr(grad_handle),
        reinterpret_cast<const int8_t*>(handle_to_ptr(operand_handle)),
        scale,
        handle_to_ptr(out_handle),
        len);
    return check_cuda_launch("CUDA I8 resident mul grad kernel launch failed") ? 0 : 1;
}

extern "C" int lumen_cuda_mul_grad_i8_lastdim_device(
    uint64_t grad_handle,
    uint64_t lhs_handle,
    uint64_t rhs_handle,
    float lhs_scale,
    float rhs_scale,
    uint64_t grad_lhs_handle,
    uint64_t grad_rhs_handle,
    size_t len,
    size_t last_dim,
    int vector_on_rhs) {
    if (!validate_handle(grad_handle, "CUDA I8 row-broadcast mul grad upstream handle") ||
        !validate_handle(lhs_handle, "CUDA I8 row-broadcast mul grad lhs handle") ||
        !validate_handle(rhs_handle, "CUDA I8 row-broadcast mul grad rhs handle") ||
        !validate_handle(grad_lhs_handle, "CUDA I8 row-broadcast mul grad lhs output handle") ||
        !validate_handle(grad_rhs_handle, "CUDA I8 row-broadcast mul grad rhs output handle")) {
        return 1;
    }
    if (last_dim == 0) {
        set_error("CUDA I8 row-broadcast mul grad last_dim must be greater than zero");
        return 1;
    }
    if (!std::isfinite(lhs_scale) || lhs_scale <= 0.0f ||
        !std::isfinite(rhs_scale) || rhs_scale <= 0.0f) {
        set_error("CUDA I8 row-broadcast mul grad scales must be finite and > 0");
        return 1;
    }
    float* reduced_grad = vector_on_rhs != 0 ? handle_to_ptr(grad_rhs_handle) : handle_to_ptr(grad_lhs_handle);
    if (!zero_f32_buffer(
            reduced_grad,
            last_dim,
            "CUDA I8 row-broadcast mul grad vector initialization failed")) {
        return 1;
    }

    launch_mul_grad_i8_row_broadcast_kernel(
        handle_to_ptr(grad_handle),
        reinterpret_cast<const int8_t*>(lhs_handle),
        reinterpret_cast<const int8_t*>(rhs_handle),
        lhs_scale,
        rhs_scale,
        handle_to_ptr(grad_lhs_handle),
        handle_to_ptr(grad_rhs_handle),
        len,
        last_dim,
        vector_on_rhs != 0);
    return check_cuda_launch("CUDA I8 row-broadcast mul grad kernel launch failed") ? 0 : 1;
}

template <typename LhsT>
int dispatch_mul_grad_typed_rhs(
    uint64_t grad_handle,
    uint64_t lhs_handle,
    float lhs_scale,
    uint64_t rhs_handle,
    int rhs_dtype,
    float rhs_scale,
    uint64_t grad_lhs_handle,
    uint64_t grad_rhs_handle,
    size_t len) {
    switch (rhs_dtype) {
        case kDTypeF32:
            launch_mul_grad_typed_same_shape_kernel<LhsT, float>(
                handle_to_ptr(grad_handle),
                reinterpret_cast<const LhsT*>(handle_to_ptr(lhs_handle)),
                lhs_scale,
                handle_to_ptr(rhs_handle),
                rhs_scale,
                handle_to_ptr(grad_lhs_handle),
                handle_to_ptr(grad_rhs_handle),
                len);
            return 0;
        case kDTypeF16:
            launch_mul_grad_typed_same_shape_kernel<LhsT, __half>(
                handle_to_ptr(grad_handle),
                reinterpret_cast<const LhsT*>(handle_to_ptr(lhs_handle)),
                lhs_scale,
                reinterpret_cast<const __half*>(handle_to_ptr(rhs_handle)),
                rhs_scale,
                handle_to_ptr(grad_lhs_handle),
                handle_to_ptr(grad_rhs_handle),
                len);
            return 0;
        case kDTypeBF16:
            launch_mul_grad_typed_same_shape_kernel<LhsT, __nv_bfloat16>(
                handle_to_ptr(grad_handle),
                reinterpret_cast<const LhsT*>(handle_to_ptr(lhs_handle)),
                lhs_scale,
                reinterpret_cast<const __nv_bfloat16*>(handle_to_ptr(rhs_handle)),
                rhs_scale,
                handle_to_ptr(grad_lhs_handle),
                handle_to_ptr(grad_rhs_handle),
                len);
            return 0;
        case kDTypeI8:
            launch_mul_grad_typed_same_shape_kernel<LhsT, int8_t>(
                handle_to_ptr(grad_handle),
                reinterpret_cast<const LhsT*>(handle_to_ptr(lhs_handle)),
                lhs_scale,
                reinterpret_cast<const int8_t*>(handle_to_ptr(rhs_handle)),
                rhs_scale,
                handle_to_ptr(grad_lhs_handle),
                handle_to_ptr(grad_rhs_handle),
                len);
            return 0;
        default:
            set_error("unsupported rhs dtype for CUDA typed mixed same-shape mul grad");
            return 1;
    }
}

extern "C" int lumen_cuda_mul_grad_typed_device(
    uint64_t grad_handle,
    uint64_t lhs_handle,
    int lhs_dtype,
    float lhs_scale,
    uint64_t rhs_handle,
    int rhs_dtype,
    float rhs_scale,
    uint64_t grad_lhs_handle,
    uint64_t grad_rhs_handle,
    size_t len) {
    if (!validate_handle(grad_handle, "CUDA typed mixed same-shape mul grad upstream handle") ||
        !validate_handle(lhs_handle, "CUDA typed mixed same-shape mul grad lhs handle") ||
        !validate_handle(rhs_handle, "CUDA typed mixed same-shape mul grad rhs handle") ||
        !validate_handle(grad_lhs_handle, "CUDA typed mixed same-shape mul grad lhs output handle") ||
        !validate_handle(grad_rhs_handle, "CUDA typed mixed same-shape mul grad rhs output handle")) {
        return 1;
    }
    if (len == 0) {
        set_error("CUDA typed mixed same-shape mul grad length must be non-zero");
        return 1;
    }
    if ((lhs_dtype == kDTypeI8 && (!std::isfinite(lhs_scale) || lhs_scale <= 0.0f)) ||
        (rhs_dtype == kDTypeI8 && (!std::isfinite(rhs_scale) || rhs_scale <= 0.0f))) {
        set_error("CUDA typed mixed same-shape mul grad I8 scales must be finite and > 0");
        return 1;
    }

    int dispatch_status = 1;
    switch (lhs_dtype) {
        case kDTypeF32:
            dispatch_status = dispatch_mul_grad_typed_rhs<float>(
                grad_handle, lhs_handle, lhs_scale, rhs_handle, rhs_dtype, rhs_scale,
                grad_lhs_handle, grad_rhs_handle, len);
            break;
        case kDTypeF16:
            dispatch_status = dispatch_mul_grad_typed_rhs<__half>(
                grad_handle, lhs_handle, lhs_scale, rhs_handle, rhs_dtype, rhs_scale,
                grad_lhs_handle, grad_rhs_handle, len);
            break;
        case kDTypeBF16:
            dispatch_status = dispatch_mul_grad_typed_rhs<__nv_bfloat16>(
                grad_handle, lhs_handle, lhs_scale, rhs_handle, rhs_dtype, rhs_scale,
                grad_lhs_handle, grad_rhs_handle, len);
            break;
        case kDTypeI8:
            dispatch_status = dispatch_mul_grad_typed_rhs<int8_t>(
                grad_handle, lhs_handle, lhs_scale, rhs_handle, rhs_dtype, rhs_scale,
                grad_lhs_handle, grad_rhs_handle, len);
            break;
        default:
            set_error("unsupported lhs dtype for CUDA typed mixed same-shape mul grad");
            return 1;
    }
    if (dispatch_status != 0) {
        return 1;
    }
    return check_cuda_launch("CUDA typed mixed same-shape mul grad kernel launch failed") ? 0 : 1;
}

template <typename LhsT>
int dispatch_mul_grad_typed_lastdim_rhs(
    uint64_t grad_handle,
    uint64_t lhs_handle,
    float lhs_scale,
    uint64_t rhs_handle,
    int rhs_dtype,
    float rhs_scale,
    uint64_t grad_lhs_handle,
    uint64_t grad_rhs_handle,
    size_t len,
    size_t last_dim,
    int vector_on_rhs) {
    switch (rhs_dtype) {
        case kDTypeF32:
            launch_mul_grad_typed_row_broadcast_kernel<LhsT, float>(
                handle_to_ptr(grad_handle),
                reinterpret_cast<const LhsT*>(handle_to_ptr(lhs_handle)),
                lhs_scale,
                handle_to_ptr(rhs_handle),
                rhs_scale,
                handle_to_ptr(grad_lhs_handle),
                handle_to_ptr(grad_rhs_handle),
                len,
                last_dim,
                vector_on_rhs != 0);
            return check_cuda_launch("CUDA typed mixed row-broadcast mul grad rhs=f32 launch failed") ? 0 : 1;
        case kDTypeF16:
            launch_mul_grad_typed_row_broadcast_kernel<LhsT, __half>(
                handle_to_ptr(grad_handle),
                reinterpret_cast<const LhsT*>(handle_to_ptr(lhs_handle)),
                lhs_scale,
                reinterpret_cast<const __half*>(handle_to_ptr(rhs_handle)),
                rhs_scale,
                handle_to_ptr(grad_lhs_handle),
                handle_to_ptr(grad_rhs_handle),
                len,
                last_dim,
                vector_on_rhs != 0);
            return check_cuda_launch("CUDA typed mixed row-broadcast mul grad rhs=f16 launch failed") ? 0 : 1;
        case kDTypeBF16:
            launch_mul_grad_typed_row_broadcast_kernel<LhsT, __nv_bfloat16>(
                handle_to_ptr(grad_handle),
                reinterpret_cast<const LhsT*>(handle_to_ptr(lhs_handle)),
                lhs_scale,
                reinterpret_cast<const __nv_bfloat16*>(handle_to_ptr(rhs_handle)),
                rhs_scale,
                handle_to_ptr(grad_lhs_handle),
                handle_to_ptr(grad_rhs_handle),
                len,
                last_dim,
                vector_on_rhs != 0);
            return check_cuda_launch("CUDA typed mixed row-broadcast mul grad rhs=bf16 launch failed") ? 0 : 1;
        case kDTypeI8:
            launch_mul_grad_typed_row_broadcast_kernel<LhsT, int8_t>(
                handle_to_ptr(grad_handle),
                reinterpret_cast<const LhsT*>(handle_to_ptr(lhs_handle)),
                lhs_scale,
                reinterpret_cast<const int8_t*>(handle_to_ptr(rhs_handle)),
                rhs_scale,
                handle_to_ptr(grad_lhs_handle),
                handle_to_ptr(grad_rhs_handle),
                len,
                last_dim,
                vector_on_rhs != 0);
            return check_cuda_launch("CUDA typed mixed row-broadcast mul grad rhs=i8 launch failed") ? 0 : 1;
        default:
            set_error("unsupported rhs dtype for CUDA typed mixed row-broadcast mul grad");
            return 1;
    }
}

extern "C" int lumen_cuda_mul_grad_typed_lastdim_device(
    uint64_t grad_handle,
    uint64_t lhs_handle,
    int lhs_dtype,
    float lhs_scale,
    uint64_t rhs_handle,
    int rhs_dtype,
    float rhs_scale,
    uint64_t grad_lhs_handle,
    uint64_t grad_rhs_handle,
    size_t len,
    size_t last_dim,
    int vector_on_rhs) {
    if (!validate_handle(grad_handle, "CUDA typed mixed row-broadcast mul grad upstream handle") ||
        !validate_handle(lhs_handle, "CUDA typed mixed row-broadcast mul grad lhs handle") ||
        !validate_handle(rhs_handle, "CUDA typed mixed row-broadcast mul grad rhs handle") ||
        !validate_handle(grad_lhs_handle, "CUDA typed mixed row-broadcast mul grad lhs output handle") ||
        !validate_handle(grad_rhs_handle, "CUDA typed mixed row-broadcast mul grad rhs output handle")) {
        return 1;
    }
    if (last_dim == 0) {
        set_error("CUDA typed mixed row-broadcast mul grad last_dim must be greater than zero");
        return 1;
    }
    if ((lhs_dtype == kDTypeI8 && (!std::isfinite(lhs_scale) || lhs_scale <= 0.0f)) ||
        (rhs_dtype == kDTypeI8 && (!std::isfinite(rhs_scale) || rhs_scale <= 0.0f))) {
        set_error("CUDA typed mixed row-broadcast mul grad I8 scales must be finite and > 0");
        return 1;
    }
    float* reduced_grad = vector_on_rhs != 0 ? handle_to_ptr(grad_rhs_handle) : handle_to_ptr(grad_lhs_handle);
    if (!zero_f32_buffer(
            reduced_grad,
            last_dim,
            "CUDA typed mixed row-broadcast mul grad vector initialization failed")) {
        return 1;
    }

    switch (lhs_dtype) {
        case kDTypeF32:
            return dispatch_mul_grad_typed_lastdim_rhs<float>(
                grad_handle, lhs_handle, lhs_scale, rhs_handle, rhs_dtype, rhs_scale,
                grad_lhs_handle, grad_rhs_handle, len, last_dim, vector_on_rhs);
        case kDTypeF16:
            return dispatch_mul_grad_typed_lastdim_rhs<__half>(
                grad_handle, lhs_handle, lhs_scale, rhs_handle, rhs_dtype, rhs_scale,
                grad_lhs_handle, grad_rhs_handle, len, last_dim, vector_on_rhs);
        case kDTypeBF16:
            return dispatch_mul_grad_typed_lastdim_rhs<__nv_bfloat16>(
                grad_handle, lhs_handle, lhs_scale, rhs_handle, rhs_dtype, rhs_scale,
                grad_lhs_handle, grad_rhs_handle, len, last_dim, vector_on_rhs);
        case kDTypeI8:
            return dispatch_mul_grad_typed_lastdim_rhs<int8_t>(
                grad_handle, lhs_handle, lhs_scale, rhs_handle, rhs_dtype, rhs_scale,
                grad_lhs_handle, grad_rhs_handle, len, last_dim, vector_on_rhs);
        default:
            set_error("unsupported lhs dtype for CUDA typed mixed row-broadcast mul grad");
            return 1;
    }
}

template <typename LhsT>
int dispatch_mul_grad_typed_row_scalar_rhs(
    uint64_t grad_handle,
    uint64_t lhs_handle,
    float lhs_scale,
    uint64_t rhs_handle,
    int rhs_dtype,
    float rhs_scale,
    uint64_t grad_lhs_handle,
    uint64_t grad_rhs_handle,
    size_t rows,
    size_t last_dim,
    int scalar_on_rhs) {
    switch (rhs_dtype) {
        case kDTypeF32:
            launch_mul_grad_typed_row_scalar_kernel<LhsT, float>(
                handle_to_ptr(grad_handle), reinterpret_cast<const LhsT*>(handle_to_ptr(lhs_handle)),
                lhs_scale, handle_to_ptr(rhs_handle), rhs_scale, handle_to_ptr(grad_lhs_handle),
                handle_to_ptr(grad_rhs_handle), rows, last_dim, scalar_on_rhs != 0);
            return 0;
        case kDTypeF16:
            launch_mul_grad_typed_row_scalar_kernel<LhsT, __half>(
                handle_to_ptr(grad_handle), reinterpret_cast<const LhsT*>(handle_to_ptr(lhs_handle)),
                lhs_scale, reinterpret_cast<const __half*>(handle_to_ptr(rhs_handle)), rhs_scale,
                handle_to_ptr(grad_lhs_handle), handle_to_ptr(grad_rhs_handle), rows, last_dim,
                scalar_on_rhs != 0);
            return 0;
        case kDTypeBF16:
            launch_mul_grad_typed_row_scalar_kernel<LhsT, __nv_bfloat16>(
                handle_to_ptr(grad_handle), reinterpret_cast<const LhsT*>(handle_to_ptr(lhs_handle)),
                lhs_scale, reinterpret_cast<const __nv_bfloat16*>(handle_to_ptr(rhs_handle)), rhs_scale,
                handle_to_ptr(grad_lhs_handle), handle_to_ptr(grad_rhs_handle), rows, last_dim,
                scalar_on_rhs != 0);
            return 0;
        case kDTypeI8:
            launch_mul_grad_typed_row_scalar_kernel<LhsT, int8_t>(
                handle_to_ptr(grad_handle), reinterpret_cast<const LhsT*>(handle_to_ptr(lhs_handle)),
                lhs_scale, reinterpret_cast<const int8_t*>(handle_to_ptr(rhs_handle)), rhs_scale,
                handle_to_ptr(grad_lhs_handle), handle_to_ptr(grad_rhs_handle), rows, last_dim,
                scalar_on_rhs != 0);
            return 0;
        default:
            set_error("unsupported rhs dtype for CUDA typed mixed row-scalar mul grad");
            return 1;
    }
}

extern "C" int lumen_cuda_mul_grad_typed_row_scalar_device(
    uint64_t grad_handle,
    uint64_t lhs_handle,
    int lhs_dtype,
    float lhs_scale,
    uint64_t rhs_handle,
    int rhs_dtype,
    float rhs_scale,
    uint64_t grad_lhs_handle,
    uint64_t grad_rhs_handle,
    size_t rows,
    size_t last_dim,
    int scalar_on_rhs) {
    if (!validate_handle(grad_handle, "CUDA typed mixed row-scalar mul grad upstream handle") ||
        !validate_handle(lhs_handle, "CUDA typed mixed row-scalar mul grad lhs handle") ||
        !validate_handle(rhs_handle, "CUDA typed mixed row-scalar mul grad rhs handle") ||
        !validate_handle(grad_lhs_handle, "CUDA typed mixed row-scalar mul grad lhs output handle") ||
        !validate_handle(grad_rhs_handle, "CUDA typed mixed row-scalar mul grad rhs output handle")) {
        return 1;
    }
    if (rows == 0 || last_dim == 0) {
        set_error("CUDA typed mixed row-scalar mul grad dimensions must be non-zero");
        return 1;
    }
    if ((lhs_dtype == kDTypeI8 && (!std::isfinite(lhs_scale) || lhs_scale <= 0.0f)) ||
        (rhs_dtype == kDTypeI8 && (!std::isfinite(rhs_scale) || rhs_scale <= 0.0f))) {
        set_error("CUDA typed mixed row-scalar mul grad I8 scales must be finite and > 0");
        return 1;
    }

    int dispatch_status = 1;
    switch (lhs_dtype) {
        case kDTypeF32:
            dispatch_status = dispatch_mul_grad_typed_row_scalar_rhs<float>(
                grad_handle, lhs_handle, lhs_scale, rhs_handle, rhs_dtype, rhs_scale,
                grad_lhs_handle, grad_rhs_handle, rows, last_dim, scalar_on_rhs);
            break;
        case kDTypeF16:
            dispatch_status = dispatch_mul_grad_typed_row_scalar_rhs<__half>(
                grad_handle, lhs_handle, lhs_scale, rhs_handle, rhs_dtype, rhs_scale,
                grad_lhs_handle, grad_rhs_handle, rows, last_dim, scalar_on_rhs);
            break;
        case kDTypeBF16:
            dispatch_status = dispatch_mul_grad_typed_row_scalar_rhs<__nv_bfloat16>(
                grad_handle, lhs_handle, lhs_scale, rhs_handle, rhs_dtype, rhs_scale,
                grad_lhs_handle, grad_rhs_handle, rows, last_dim, scalar_on_rhs);
            break;
        case kDTypeI8:
            dispatch_status = dispatch_mul_grad_typed_row_scalar_rhs<int8_t>(
                grad_handle, lhs_handle, lhs_scale, rhs_handle, rhs_dtype, rhs_scale,
                grad_lhs_handle, grad_rhs_handle, rows, last_dim, scalar_on_rhs);
            break;
        default:
            set_error("unsupported lhs dtype for CUDA typed mixed row-scalar mul grad");
            return 1;
    }
    if (dispatch_status != 0) {
        return 1;
    }
    return check_cuda_launch("CUDA typed mixed row-scalar mul grad kernel launch failed") ? 0 : 1;
}

template <typename LhsT>
int dispatch_mul_grad_typed_b1d_1h1_rhs(
    uint64_t grad_handle,
    uint64_t lhs_handle,
    float lhs_scale,
    uint64_t rhs_handle,
    int rhs_dtype,
    float rhs_scale,
    uint64_t grad_lhs_handle,
    uint64_t grad_rhs_handle,
    size_t batch,
    size_t heads,
    size_t dim,
    bool b1d_on_lhs) {
    switch (rhs_dtype) {
        case kDTypeF32:
            launch_mul_grad_typed_b1d_1h1_kernel<LhsT, float>(
                handle_to_ptr(grad_handle), reinterpret_cast<const LhsT*>(handle_to_ptr(lhs_handle)),
                lhs_scale, handle_to_ptr(rhs_handle), rhs_scale, handle_to_ptr(grad_lhs_handle),
                handle_to_ptr(grad_rhs_handle), batch, heads, dim, b1d_on_lhs);
            return 0;
        case kDTypeF16:
            launch_mul_grad_typed_b1d_1h1_kernel<LhsT, __half>(
                handle_to_ptr(grad_handle), reinterpret_cast<const LhsT*>(handle_to_ptr(lhs_handle)),
                lhs_scale, reinterpret_cast<const __half*>(handle_to_ptr(rhs_handle)), rhs_scale,
                handle_to_ptr(grad_lhs_handle), handle_to_ptr(grad_rhs_handle), batch, heads, dim,
                b1d_on_lhs);
            return 0;
        case kDTypeBF16:
            launch_mul_grad_typed_b1d_1h1_kernel<LhsT, __nv_bfloat16>(
                handle_to_ptr(grad_handle), reinterpret_cast<const LhsT*>(handle_to_ptr(lhs_handle)),
                lhs_scale, reinterpret_cast<const __nv_bfloat16*>(handle_to_ptr(rhs_handle)), rhs_scale,
                handle_to_ptr(grad_lhs_handle), handle_to_ptr(grad_rhs_handle), batch, heads, dim,
                b1d_on_lhs);
            return 0;
        case kDTypeI8:
            launch_mul_grad_typed_b1d_1h1_kernel<LhsT, int8_t>(
                handle_to_ptr(grad_handle), reinterpret_cast<const LhsT*>(handle_to_ptr(lhs_handle)),
                lhs_scale, reinterpret_cast<const int8_t*>(handle_to_ptr(rhs_handle)), rhs_scale,
                handle_to_ptr(grad_lhs_handle), handle_to_ptr(grad_rhs_handle), batch, heads, dim,
                b1d_on_lhs);
            return 0;
        default:
            set_error("unsupported rhs dtype for CUDA typed mixed b1d/1h1 mul grad");
            return 1;
    }
}

extern "C" int lumen_cuda_mul_grad_typed_b1d_1h1_device(
    uint64_t grad_handle,
    uint64_t lhs_handle,
    int lhs_dtype,
    float lhs_scale,
    uint64_t rhs_handle,
    int rhs_dtype,
    float rhs_scale,
    uint64_t grad_lhs_handle,
    uint64_t grad_rhs_handle,
    size_t batch,
    size_t heads,
    size_t dim,
    int b1d_on_lhs) {
    if (!validate_handle(grad_handle, "CUDA typed mixed b1d/1h1 mul grad upstream handle") ||
        !validate_handle(lhs_handle, "CUDA typed mixed b1d/1h1 mul grad lhs handle") ||
        !validate_handle(rhs_handle, "CUDA typed mixed b1d/1h1 mul grad rhs handle") ||
        !validate_handle(grad_lhs_handle, "CUDA typed mixed b1d/1h1 mul grad lhs output handle") ||
        !validate_handle(grad_rhs_handle, "CUDA typed mixed b1d/1h1 mul grad rhs output handle")) {
        return 1;
    }
    if (batch == 0 || heads == 0 || dim == 0) {
        set_error("CUDA typed mixed b1d/1h1 mul grad dimensions must be non-zero");
        return 1;
    }
    if ((lhs_dtype == kDTypeI8 && (!std::isfinite(lhs_scale) || lhs_scale <= 0.0f)) ||
        (rhs_dtype == kDTypeI8 && (!std::isfinite(rhs_scale) || rhs_scale <= 0.0f))) {
        set_error("CUDA typed mixed b1d/1h1 mul grad I8 scales must be finite and > 0");
        return 1;
    }

    int dispatch_status = 1;
    switch (lhs_dtype) {
        case kDTypeF32:
            dispatch_status = dispatch_mul_grad_typed_b1d_1h1_rhs<float>(
                grad_handle, lhs_handle, lhs_scale, rhs_handle, rhs_dtype, rhs_scale,
                grad_lhs_handle, grad_rhs_handle, batch, heads, dim, b1d_on_lhs != 0);
            break;
        case kDTypeF16:
            dispatch_status = dispatch_mul_grad_typed_b1d_1h1_rhs<__half>(
                grad_handle, lhs_handle, lhs_scale, rhs_handle, rhs_dtype, rhs_scale,
                grad_lhs_handle, grad_rhs_handle, batch, heads, dim, b1d_on_lhs != 0);
            break;
        case kDTypeBF16:
            dispatch_status = dispatch_mul_grad_typed_b1d_1h1_rhs<__nv_bfloat16>(
                grad_handle, lhs_handle, lhs_scale, rhs_handle, rhs_dtype, rhs_scale,
                grad_lhs_handle, grad_rhs_handle, batch, heads, dim, b1d_on_lhs != 0);
            break;
        case kDTypeI8:
            dispatch_status = dispatch_mul_grad_typed_b1d_1h1_rhs<int8_t>(
                grad_handle, lhs_handle, lhs_scale, rhs_handle, rhs_dtype, rhs_scale,
                grad_lhs_handle, grad_rhs_handle, batch, heads, dim, b1d_on_lhs != 0);
            break;
        default:
            set_error("unsupported lhs dtype for CUDA typed mixed b1d/1h1 mul grad");
            return 1;
    }
    if (dispatch_status != 0) {
        return 1;
    }
    return check_cuda_launch("CUDA typed mixed b1d/1h1 mul grad kernel launch failed") ? 0 : 1;
}

template <typename LhsT>
int dispatch_mul_grad_typed_b1d_1hd_rhs(
    uint64_t grad_handle,
    uint64_t lhs_handle,
    float lhs_scale,
    uint64_t rhs_handle,
    int rhs_dtype,
    float rhs_scale,
    uint64_t grad_lhs_handle,
    uint64_t grad_rhs_handle,
    size_t batch,
    size_t heads,
    size_t dim,
    bool b1d_on_lhs) {
    switch (rhs_dtype) {
        case kDTypeF32:
            launch_mul_grad_typed_b1d_1hd_kernel<LhsT, float>(
                handle_to_ptr(grad_handle), reinterpret_cast<const LhsT*>(handle_to_ptr(lhs_handle)),
                lhs_scale, handle_to_ptr(rhs_handle), rhs_scale, handle_to_ptr(grad_lhs_handle),
                handle_to_ptr(grad_rhs_handle), batch, heads, dim, b1d_on_lhs);
            return 0;
        case kDTypeF16:
            launch_mul_grad_typed_b1d_1hd_kernel<LhsT, __half>(
                handle_to_ptr(grad_handle), reinterpret_cast<const LhsT*>(handle_to_ptr(lhs_handle)),
                lhs_scale, reinterpret_cast<const __half*>(handle_to_ptr(rhs_handle)), rhs_scale,
                handle_to_ptr(grad_lhs_handle), handle_to_ptr(grad_rhs_handle), batch, heads, dim,
                b1d_on_lhs);
            return 0;
        case kDTypeBF16:
            launch_mul_grad_typed_b1d_1hd_kernel<LhsT, __nv_bfloat16>(
                handle_to_ptr(grad_handle), reinterpret_cast<const LhsT*>(handle_to_ptr(lhs_handle)),
                lhs_scale, reinterpret_cast<const __nv_bfloat16*>(handle_to_ptr(rhs_handle)), rhs_scale,
                handle_to_ptr(grad_lhs_handle), handle_to_ptr(grad_rhs_handle), batch, heads, dim,
                b1d_on_lhs);
            return 0;
        case kDTypeI8:
            launch_mul_grad_typed_b1d_1hd_kernel<LhsT, int8_t>(
                handle_to_ptr(grad_handle), reinterpret_cast<const LhsT*>(handle_to_ptr(lhs_handle)),
                lhs_scale, reinterpret_cast<const int8_t*>(handle_to_ptr(rhs_handle)), rhs_scale,
                handle_to_ptr(grad_lhs_handle), handle_to_ptr(grad_rhs_handle), batch, heads, dim,
                b1d_on_lhs);
            return 0;
        default:
            set_error("unsupported rhs dtype for CUDA typed mixed b1d/1hd mul grad");
            return 1;
    }
}

extern "C" int lumen_cuda_mul_grad_typed_b1d_1hd_device(
    uint64_t grad_handle,
    uint64_t lhs_handle,
    int lhs_dtype,
    float lhs_scale,
    uint64_t rhs_handle,
    int rhs_dtype,
    float rhs_scale,
    uint64_t grad_lhs_handle,
    uint64_t grad_rhs_handle,
    size_t batch,
    size_t heads,
    size_t dim,
    int b1d_on_lhs) {
    if (!validate_handle(grad_handle, "CUDA typed mixed b1d/1hd mul grad upstream handle") ||
        !validate_handle(lhs_handle, "CUDA typed mixed b1d/1hd mul grad lhs handle") ||
        !validate_handle(rhs_handle, "CUDA typed mixed b1d/1hd mul grad rhs handle") ||
        !validate_handle(grad_lhs_handle, "CUDA typed mixed b1d/1hd mul grad lhs output handle") ||
        !validate_handle(grad_rhs_handle, "CUDA typed mixed b1d/1hd mul grad rhs output handle")) {
        return 1;
    }
    if (batch == 0 || heads == 0 || dim == 0) {
        set_error("CUDA typed mixed b1d/1hd mul grad dimensions must be non-zero");
        return 1;
    }
    if ((lhs_dtype == kDTypeI8 && (!std::isfinite(lhs_scale) || lhs_scale <= 0.0f)) ||
        (rhs_dtype == kDTypeI8 && (!std::isfinite(rhs_scale) || rhs_scale <= 0.0f))) {
        set_error("CUDA typed mixed b1d/1hd mul grad I8 scales must be finite and > 0");
        return 1;
    }

    int dispatch_status = 1;
    switch (lhs_dtype) {
        case kDTypeF32:
            dispatch_status = dispatch_mul_grad_typed_b1d_1hd_rhs<float>(
                grad_handle, lhs_handle, lhs_scale, rhs_handle, rhs_dtype, rhs_scale,
                grad_lhs_handle, grad_rhs_handle, batch, heads, dim, b1d_on_lhs != 0);
            break;
        case kDTypeF16:
            dispatch_status = dispatch_mul_grad_typed_b1d_1hd_rhs<__half>(
                grad_handle, lhs_handle, lhs_scale, rhs_handle, rhs_dtype, rhs_scale,
                grad_lhs_handle, grad_rhs_handle, batch, heads, dim, b1d_on_lhs != 0);
            break;
        case kDTypeBF16:
            dispatch_status = dispatch_mul_grad_typed_b1d_1hd_rhs<__nv_bfloat16>(
                grad_handle, lhs_handle, lhs_scale, rhs_handle, rhs_dtype, rhs_scale,
                grad_lhs_handle, grad_rhs_handle, batch, heads, dim, b1d_on_lhs != 0);
            break;
        case kDTypeI8:
            dispatch_status = dispatch_mul_grad_typed_b1d_1hd_rhs<int8_t>(
                grad_handle, lhs_handle, lhs_scale, rhs_handle, rhs_dtype, rhs_scale,
                grad_lhs_handle, grad_rhs_handle, batch, heads, dim, b1d_on_lhs != 0);
            break;
        default:
            set_error("unsupported lhs dtype for CUDA typed mixed b1d/1hd mul grad");
            return 1;
    }
    if (dispatch_status != 0) {
        return 1;
    }
    return check_cuda_launch("CUDA typed mixed b1d/1hd mul grad kernel launch failed") ? 0 : 1;
}

template <typename LhsT>
int dispatch_mul_grad_typed_scalar_rhs(
    uint64_t grad_handle,
    uint64_t lhs_handle,
    float lhs_scale,
    uint64_t rhs_handle,
    int rhs_dtype,
    float rhs_scale,
    uint64_t grad_lhs_handle,
    uint64_t grad_rhs_handle,
    size_t len,
    bool scalar_on_rhs) {
    switch (rhs_dtype) {
        case kDTypeF32:
            launch_mul_grad_typed_scalar_broadcast_kernel<LhsT, float>(
                handle_to_ptr(grad_handle), reinterpret_cast<const LhsT*>(handle_to_ptr(lhs_handle)),
                lhs_scale, handle_to_ptr(rhs_handle), rhs_scale, handle_to_ptr(grad_lhs_handle),
                handle_to_ptr(grad_rhs_handle), len, scalar_on_rhs);
            return 0;
        case kDTypeF16:
            launch_mul_grad_typed_scalar_broadcast_kernel<LhsT, __half>(
                handle_to_ptr(grad_handle), reinterpret_cast<const LhsT*>(handle_to_ptr(lhs_handle)),
                lhs_scale, reinterpret_cast<const __half*>(handle_to_ptr(rhs_handle)), rhs_scale,
                handle_to_ptr(grad_lhs_handle), handle_to_ptr(grad_rhs_handle), len, scalar_on_rhs);
            return 0;
        case kDTypeBF16:
            launch_mul_grad_typed_scalar_broadcast_kernel<LhsT, __nv_bfloat16>(
                handle_to_ptr(grad_handle), reinterpret_cast<const LhsT*>(handle_to_ptr(lhs_handle)),
                lhs_scale, reinterpret_cast<const __nv_bfloat16*>(handle_to_ptr(rhs_handle)), rhs_scale,
                handle_to_ptr(grad_lhs_handle), handle_to_ptr(grad_rhs_handle), len, scalar_on_rhs);
            return 0;
        case kDTypeI8:
            launch_mul_grad_typed_scalar_broadcast_kernel<LhsT, int8_t>(
                handle_to_ptr(grad_handle), reinterpret_cast<const LhsT*>(handle_to_ptr(lhs_handle)),
                lhs_scale, reinterpret_cast<const int8_t*>(handle_to_ptr(rhs_handle)), rhs_scale,
                handle_to_ptr(grad_lhs_handle), handle_to_ptr(grad_rhs_handle), len, scalar_on_rhs);
            return 0;
        default:
            set_error("unsupported rhs dtype for CUDA typed mixed scalar-broadcast mul grad");
            return 1;
    }
}

extern "C" int lumen_cuda_mul_grad_typed_scalar_device(
    uint64_t grad_handle,
    uint64_t lhs_handle,
    int lhs_dtype,
    float lhs_scale,
    uint64_t rhs_handle,
    int rhs_dtype,
    float rhs_scale,
    uint64_t grad_lhs_handle,
    uint64_t grad_rhs_handle,
    size_t len,
    int scalar_on_rhs) {
    if (!validate_handle(grad_handle, "CUDA typed mixed scalar-broadcast mul grad upstream handle") ||
        !validate_handle(lhs_handle, "CUDA typed mixed scalar-broadcast mul grad lhs handle") ||
        !validate_handle(rhs_handle, "CUDA typed mixed scalar-broadcast mul grad rhs handle") ||
        !validate_handle(grad_lhs_handle, "CUDA typed mixed scalar-broadcast mul grad lhs output handle") ||
        !validate_handle(grad_rhs_handle, "CUDA typed mixed scalar-broadcast mul grad rhs output handle")) {
        return 1;
    }
    if (len == 0) {
        set_error("CUDA typed mixed scalar-broadcast mul grad length must be non-zero");
        return 1;
    }
    if ((lhs_dtype == kDTypeI8 && (!std::isfinite(lhs_scale) || lhs_scale <= 0.0f)) ||
        (rhs_dtype == kDTypeI8 && (!std::isfinite(rhs_scale) || rhs_scale <= 0.0f))) {
        set_error("CUDA typed mixed scalar-broadcast mul grad I8 scales must be finite and > 0");
        return 1;
    }

    float* scalar_grad = scalar_on_rhs != 0 ? handle_to_ptr(grad_rhs_handle) : handle_to_ptr(grad_lhs_handle);
    cudaError_t status = cudaMemset(scalar_grad, 0, sizeof(float));
    if (status != cudaSuccess) {
        set_cuda_error("CUDA typed mixed scalar-broadcast mul grad scalar initialization failed", status);
        return 1;
    }

    int dispatch_status = 1;
    switch (lhs_dtype) {
        case kDTypeF32:
            dispatch_status = dispatch_mul_grad_typed_scalar_rhs<float>(
                grad_handle, lhs_handle, lhs_scale, rhs_handle, rhs_dtype, rhs_scale,
                grad_lhs_handle, grad_rhs_handle, len, scalar_on_rhs != 0);
            break;
        case kDTypeF16:
            dispatch_status = dispatch_mul_grad_typed_scalar_rhs<__half>(
                grad_handle, lhs_handle, lhs_scale, rhs_handle, rhs_dtype, rhs_scale,
                grad_lhs_handle, grad_rhs_handle, len, scalar_on_rhs != 0);
            break;
        case kDTypeBF16:
            dispatch_status = dispatch_mul_grad_typed_scalar_rhs<__nv_bfloat16>(
                grad_handle, lhs_handle, lhs_scale, rhs_handle, rhs_dtype, rhs_scale,
                grad_lhs_handle, grad_rhs_handle, len, scalar_on_rhs != 0);
            break;
        case kDTypeI8:
            dispatch_status = dispatch_mul_grad_typed_scalar_rhs<int8_t>(
                grad_handle, lhs_handle, lhs_scale, rhs_handle, rhs_dtype, rhs_scale,
                grad_lhs_handle, grad_rhs_handle, len, scalar_on_rhs != 0);
            break;
        default:
            set_error("unsupported lhs dtype for CUDA typed mixed scalar-broadcast mul grad");
            return 1;
    }
    if (dispatch_status != 0) {
        return 1;
    }
    return check_cuda_launch("CUDA typed mixed scalar-broadcast mul grad kernel launch failed") ? 0 : 1;
}

template <typename LhsT>
int dispatch_mul_grad_typed_broadcast_rhs(
    uint64_t grad_handle,
    uint64_t lhs_handle,
    float lhs_scale,
    uint64_t rhs_handle,
    int rhs_dtype,
    float rhs_scale,
    uint64_t grad_lhs_handle,
    uint64_t grad_rhs_handle,
    const size_t* d_out_strides,
    const size_t* d_lhs_shape,
    const size_t* d_lhs_strides,
    const size_t* d_rhs_shape,
    const size_t* d_rhs_strides,
    size_t ndim,
    size_t out_len) {
    switch (rhs_dtype) {
        case kDTypeF32:
            launch_mul_grad_typed_broadcast_kernel<LhsT, float>(
                handle_to_ptr(grad_handle),
                reinterpret_cast<const LhsT*>(handle_to_ptr(lhs_handle)),
                lhs_scale,
                handle_to_ptr(rhs_handle),
                rhs_scale,
                handle_to_ptr(grad_lhs_handle),
                handle_to_ptr(grad_rhs_handle),
                d_out_strides,
                d_lhs_shape,
                d_lhs_strides,
                d_rhs_shape,
                d_rhs_strides,
                ndim,
                out_len);
            return 0;
        case kDTypeF16:
            launch_mul_grad_typed_broadcast_kernel<LhsT, __half>(
                handle_to_ptr(grad_handle),
                reinterpret_cast<const LhsT*>(handle_to_ptr(lhs_handle)),
                lhs_scale,
                reinterpret_cast<const __half*>(handle_to_ptr(rhs_handle)),
                rhs_scale,
                handle_to_ptr(grad_lhs_handle),
                handle_to_ptr(grad_rhs_handle),
                d_out_strides,
                d_lhs_shape,
                d_lhs_strides,
                d_rhs_shape,
                d_rhs_strides,
                ndim,
                out_len);
            return 0;
        case kDTypeBF16:
            launch_mul_grad_typed_broadcast_kernel<LhsT, __nv_bfloat16>(
                handle_to_ptr(grad_handle),
                reinterpret_cast<const LhsT*>(handle_to_ptr(lhs_handle)),
                lhs_scale,
                reinterpret_cast<const __nv_bfloat16*>(handle_to_ptr(rhs_handle)),
                rhs_scale,
                handle_to_ptr(grad_lhs_handle),
                handle_to_ptr(grad_rhs_handle),
                d_out_strides,
                d_lhs_shape,
                d_lhs_strides,
                d_rhs_shape,
                d_rhs_strides,
                ndim,
                out_len);
            return 0;
        case kDTypeI8:
            launch_mul_grad_typed_broadcast_kernel<LhsT, int8_t>(
                handle_to_ptr(grad_handle),
                reinterpret_cast<const LhsT*>(handle_to_ptr(lhs_handle)),
                lhs_scale,
                reinterpret_cast<const int8_t*>(handle_to_ptr(rhs_handle)),
                rhs_scale,
                handle_to_ptr(grad_lhs_handle),
                handle_to_ptr(grad_rhs_handle),
                d_out_strides,
                d_lhs_shape,
                d_lhs_strides,
                d_rhs_shape,
                d_rhs_strides,
                ndim,
                out_len);
            return 0;
        default:
            set_error("unsupported rhs dtype for CUDA typed mixed broadcast mul grad");
            return 1;
    }
}

extern "C" int lumen_cuda_mul_grad_typed_broadcast_device(
    uint64_t grad_handle,
    uint64_t lhs_handle,
    int lhs_dtype,
    float lhs_scale,
    uint64_t rhs_handle,
    int rhs_dtype,
    float rhs_scale,
    uint64_t grad_lhs_handle,
    uint64_t grad_rhs_handle,
    size_t ndim,
    const size_t* out_strides,
    const size_t* lhs_shape,
    const size_t* lhs_strides,
    const size_t* rhs_shape,
    const size_t* rhs_strides,
    size_t out_len,
    size_t lhs_len,
    size_t rhs_len) {
    if (!validate_handle(grad_handle, "CUDA typed mixed broadcast mul grad upstream handle") ||
        !validate_handle(lhs_handle, "CUDA typed mixed broadcast mul grad lhs handle") ||
        !validate_handle(rhs_handle, "CUDA typed mixed broadcast mul grad rhs handle") ||
        !validate_handle(grad_lhs_handle, "CUDA typed mixed broadcast mul grad lhs output handle") ||
        !validate_handle(grad_rhs_handle, "CUDA typed mixed broadcast mul grad rhs output handle")) {
        return 1;
    }
    if (ndim == 0 || out_len == 0 || lhs_len == 0 || rhs_len == 0 ||
        out_strides == nullptr || lhs_shape == nullptr || lhs_strides == nullptr ||
        rhs_shape == nullptr || rhs_strides == nullptr) {
        set_error("CUDA typed mixed broadcast mul grad received invalid metadata");
        return 1;
    }
    if ((lhs_dtype == kDTypeI8 && (!std::isfinite(lhs_scale) || lhs_scale <= 0.0f)) ||
        (rhs_dtype == kDTypeI8 && (!std::isfinite(rhs_scale) || rhs_scale <= 0.0f))) {
        set_error("CUDA typed mixed broadcast mul grad I8 scales must be finite and > 0");
        return 1;
    }

    if (!zero_f32_buffer(
            handle_to_ptr(grad_lhs_handle),
            lhs_len,
            "CUDA typed mixed broadcast mul grad lhs initialization failed")) {
        return 1;
    }
    if (!zero_f32_buffer(
            handle_to_ptr(grad_rhs_handle),
            rhs_len,
            "CUDA typed mixed broadcast mul grad rhs initialization failed")) {
        return 1;
    }

    size_t* d_out_strides = nullptr;
    size_t* d_lhs_shape = nullptr;
    size_t* d_lhs_strides = nullptr;
    size_t* d_rhs_shape = nullptr;
    size_t* d_rhs_strides = nullptr;
    const size_t* host_metadata[] = {
        out_strides, lhs_shape, lhs_strides, rhs_shape, rhs_strides};
    size_t* device_metadata[] = {
        d_out_strides, d_lhs_shape, d_lhs_strides, d_rhs_shape, d_rhs_strides};
    if (!upload_packed_elementwise_metadata(
            "failed to upload CUDA typed mixed broadcast mul grad metadata",
            host_metadata,
            5,
            ndim,
            device_metadata)) {
        return 1;
    }
    d_out_strides = device_metadata[0];
    d_lhs_shape = device_metadata[1];
    d_lhs_strides = device_metadata[2];
    d_rhs_shape = device_metadata[3];
    d_rhs_strides = device_metadata[4];

    int dispatch_status = 1;
    switch (lhs_dtype) {
        case kDTypeF32:
            dispatch_status = dispatch_mul_grad_typed_broadcast_rhs<float>(
                grad_handle, lhs_handle, lhs_scale, rhs_handle, rhs_dtype, rhs_scale,
                grad_lhs_handle, grad_rhs_handle, d_out_strides, d_lhs_shape, d_lhs_strides,
                d_rhs_shape, d_rhs_strides, ndim, out_len);
            break;
        case kDTypeF16:
            dispatch_status = dispatch_mul_grad_typed_broadcast_rhs<__half>(
                grad_handle, lhs_handle, lhs_scale, rhs_handle, rhs_dtype, rhs_scale,
                grad_lhs_handle, grad_rhs_handle, d_out_strides, d_lhs_shape, d_lhs_strides,
                d_rhs_shape, d_rhs_strides, ndim, out_len);
            break;
        case kDTypeBF16:
            dispatch_status = dispatch_mul_grad_typed_broadcast_rhs<__nv_bfloat16>(
                grad_handle, lhs_handle, lhs_scale, rhs_handle, rhs_dtype, rhs_scale,
                grad_lhs_handle, grad_rhs_handle, d_out_strides, d_lhs_shape, d_lhs_strides,
                d_rhs_shape, d_rhs_strides, ndim, out_len);
            break;
        case kDTypeI8:
            dispatch_status = dispatch_mul_grad_typed_broadcast_rhs<int8_t>(
                grad_handle, lhs_handle, lhs_scale, rhs_handle, rhs_dtype, rhs_scale,
                grad_lhs_handle, grad_rhs_handle, d_out_strides, d_lhs_shape, d_lhs_strides,
                d_rhs_shape, d_rhs_strides, ndim, out_len);
            break;
        default:
            set_error("unsupported lhs dtype for CUDA typed mixed broadcast mul grad");
            return 1;
    }
    if (dispatch_status != 0) {
        return 1;
    }
    return check_cuda_launch("CUDA typed mixed broadcast mul grad kernel launch failed") ? 0 : 1;
}

extern "C" int lumen_cuda_binary_backward_f32_device(
    uint64_t lhs_handle,
    uint64_t rhs_handle,
    uint64_t grad_handle,
    uint64_t grad_lhs_handle,
    uint64_t grad_rhs_handle,
    size_t len,
    int op) {
    if (!validate_handle(lhs_handle, "CUDA binary backward lhs handle") ||
        !validate_handle(rhs_handle, "CUDA binary backward rhs handle") ||
        !validate_handle(grad_handle, "CUDA binary backward grad handle") ||
        !validate_handle(grad_lhs_handle, "CUDA binary backward lhs grad handle") ||
        !validate_handle(grad_rhs_handle, "CUDA binary backward rhs grad handle")) {
        return 1;
    }
    if (len == 0) {
        return 0;
    }

    constexpr unsigned int block_size = 256;
    size_t vec_len = len / 4;
    if (vec_len > 0) {
        const unsigned int grid_size = linear_grid_size(vec_len, block_size);
        launch_binary_backward_kernel(
            op,
            true,
            grid_size,
            block_size,
            handle_to_ptr(lhs_handle),
            handle_to_ptr(rhs_handle),
            handle_to_ptr(grad_handle),
            handle_to_ptr(grad_lhs_handle),
            handle_to_ptr(grad_rhs_handle),
            vec_len,
            len);
    } else {
        const unsigned int grid_size = linear_grid_size(len, block_size);
        launch_binary_backward_kernel(
            op,
            false,
            grid_size,
            block_size,
            handle_to_ptr(lhs_handle),
            handle_to_ptr(rhs_handle),
            handle_to_ptr(grad_handle),
            handle_to_ptr(grad_lhs_handle),
            handle_to_ptr(grad_rhs_handle),
            len,
            len);
    }
    if (!check_cuda_launch("CUDA binary backward kernel launch failed")) {
        return 1;
    }
    return 0;
}

extern "C" int lumen_cuda_add_sub_backward_f32_device(
    uint64_t grad_handle,
    uint64_t grad_lhs_handle,
    uint64_t grad_rhs_handle,
    size_t len,
    int op) {
    if (!validate_handle(grad_handle, "CUDA add/sub backward grad handle") ||
        !validate_handle(grad_lhs_handle, "CUDA add/sub backward lhs grad handle") ||
        !validate_handle(grad_rhs_handle, "CUDA add/sub backward rhs grad handle")) {
        return 1;
    }
    if (len == 0) {
        set_error("CUDA add/sub backward length must be greater than zero");
        return 1;
    }
    if (op != kBinaryAdd && op != kBinarySub) {
        set_error("CUDA add/sub backward received unsupported op");
        return 1;
    }
    launch_add_sub_same_shape_backward_kernel(
        handle_to_ptr(grad_handle),
        handle_to_ptr(grad_lhs_handle),
        handle_to_ptr(grad_rhs_handle),
        len,
        op == kBinarySub);
    return check_cuda_launch("CUDA add/sub backward kernel launch failed") ? 0 : 1;
}

extern "C" int lumen_cuda_add_sub_backward_lastdim_f32_device(
    uint64_t grad_handle,
    uint64_t grad_lhs_handle,
    uint64_t grad_rhs_handle,
    size_t len,
    size_t last_dim,
    int vector_on_rhs,
    int op) {
    if (!validate_handle(grad_handle, "CUDA add/sub row-broadcast backward grad handle") ||
        !validate_handle(grad_lhs_handle, "CUDA add/sub row-broadcast backward lhs grad handle") ||
        !validate_handle(grad_rhs_handle, "CUDA add/sub row-broadcast backward rhs grad handle")) {
        return 1;
    }
    if (len == 0 || last_dim == 0 || len % last_dim != 0) {
        set_error("CUDA add/sub row-broadcast backward received invalid dimensions");
        return 1;
    }
    if (op != kBinaryAdd && op != kBinarySub) {
        set_error("CUDA add/sub row-broadcast backward received unsupported op");
        return 1;
    }
    if (!launch_add_sub_row_broadcast_backward_kernel(
            handle_to_ptr(grad_handle),
            handle_to_ptr(grad_lhs_handle),
            handle_to_ptr(grad_rhs_handle),
            len,
            last_dim,
            vector_on_rhs != 0,
            op == kBinarySub)) {
        return 1;
    }
    return check_cuda_launch("CUDA add/sub row-broadcast backward kernel launch failed") ? 0 : 1;
}

extern "C" int lumen_cuda_add_sub_backward_scalar_f32_device(
    uint64_t grad_handle,
    uint64_t grad_lhs_handle,
    uint64_t grad_rhs_handle,
    size_t len,
    int scalar_on_rhs,
    int op) {
    if (!validate_handle(grad_handle, "CUDA add/sub scalar-broadcast backward grad handle") ||
        !validate_handle(grad_lhs_handle, "CUDA add/sub scalar-broadcast backward lhs grad handle") ||
        !validate_handle(grad_rhs_handle, "CUDA add/sub scalar-broadcast backward rhs grad handle")) {
        return 1;
    }
    if (len == 0) {
        set_error("CUDA add/sub scalar-broadcast backward length must be greater than zero");
        return 1;
    }
    if (op != kBinaryAdd && op != kBinarySub) {
        set_error("CUDA add/sub scalar-broadcast backward received unsupported op");
        return 1;
    }
    void* scalar_ptr = scalar_on_rhs != 0 ? handle_to_ptr(grad_rhs_handle) : handle_to_ptr(grad_lhs_handle);
    cudaError_t status = cudaMemset(scalar_ptr, 0, sizeof(float));
    if (status != cudaSuccess) {
        set_cuda_error("CUDA add/sub scalar-broadcast grad initialization failed", status);
        return 1;
    }
    launch_add_sub_scalar_broadcast_backward_kernel(
        handle_to_ptr(grad_handle),
        handle_to_ptr(grad_lhs_handle),
        handle_to_ptr(grad_rhs_handle),
        len,
        scalar_on_rhs != 0,
        op == kBinarySub);
    return check_cuda_launch("CUDA add/sub scalar-broadcast backward kernel launch failed") ? 0 : 1;
}

extern "C" int lumen_cuda_add_sub_backward_row_scalar_f32_device(
    uint64_t grad_handle,
    uint64_t grad_lhs_handle,
    uint64_t grad_rhs_handle,
    size_t rows,
    size_t last_dim,
    int scalar_on_rhs,
    int op) {
    if (!validate_handle(grad_handle, "CUDA add/sub row-scalar backward grad handle") ||
        !validate_handle(grad_lhs_handle, "CUDA add/sub row-scalar backward lhs grad handle") ||
        !validate_handle(grad_rhs_handle, "CUDA add/sub row-scalar backward rhs grad handle")) {
        return 1;
    }
    if (rows == 0 || last_dim == 0) {
        set_error("CUDA add/sub row-scalar backward dimensions must be non-zero");
        return 1;
    }
    if (op != kBinaryAdd && op != kBinarySub) {
        set_error("CUDA add/sub row-scalar backward received unsupported op");
        return 1;
    }
    launch_add_sub_row_scalar_backward_kernel(
        handle_to_ptr(grad_handle),
        handle_to_ptr(grad_lhs_handle),
        handle_to_ptr(grad_rhs_handle),
        rows,
        last_dim,
        scalar_on_rhs != 0,
        op == kBinarySub);
    return check_cuda_launch("CUDA add/sub row-scalar backward kernel launch failed") ? 0 : 1;
}

extern "C" int lumen_cuda_add_sub_backward_b1d_1h1_f32_device(
    uint64_t grad_handle,
    uint64_t grad_lhs_handle,
    uint64_t grad_rhs_handle,
    size_t batch,
    size_t heads,
    size_t dim,
    int b1d_on_lhs,
    int op) {
    if (!validate_handle(grad_handle, "CUDA add/sub b1d/1h1 backward grad handle") ||
        !validate_handle(grad_lhs_handle, "CUDA add/sub b1d/1h1 backward lhs grad handle") ||
        !validate_handle(grad_rhs_handle, "CUDA add/sub b1d/1h1 backward rhs grad handle")) {
        return 1;
    }
    if (batch == 0 || heads == 0 || dim == 0) {
        set_error("CUDA add/sub b1d/1h1 backward dimensions must be non-zero");
        return 1;
    }
    if (op != kBinaryAdd && op != kBinarySub) {
        set_error("CUDA add/sub b1d/1h1 backward received unsupported op");
        return 1;
    }
    launch_add_sub_b1d_1h1_backward_kernel(
        handle_to_ptr(grad_handle),
        handle_to_ptr(grad_lhs_handle),
        handle_to_ptr(grad_rhs_handle),
        batch,
        heads,
        dim,
        b1d_on_lhs != 0,
        op == kBinarySub);
    return check_cuda_launch("CUDA add/sub b1d/1h1 backward kernel launch failed") ? 0 : 1;
}

extern "C" int lumen_cuda_add_sub_backward_b1d_1hd_f32_device(
    uint64_t grad_handle,
    uint64_t grad_lhs_handle,
    uint64_t grad_rhs_handle,
    size_t batch,
    size_t heads,
    size_t dim,
    int b1d_on_lhs,
    int op) {
    if (!validate_handle(grad_handle, "CUDA add/sub b1d/1hd backward grad handle") ||
        !validate_handle(grad_lhs_handle, "CUDA add/sub b1d/1hd backward lhs grad handle") ||
        !validate_handle(grad_rhs_handle, "CUDA add/sub b1d/1hd backward rhs grad handle")) {
        return 1;
    }
    if (batch == 0 || heads == 0 || dim == 0) {
        set_error("CUDA add/sub b1d/1hd backward dimensions must be non-zero");
        return 1;
    }
    if (op != kBinaryAdd && op != kBinarySub) {
        set_error("CUDA add/sub b1d/1hd backward received unsupported op");
        return 1;
    }
    launch_add_sub_b1d_1hd_backward_kernel(
        handle_to_ptr(grad_handle),
        handle_to_ptr(grad_lhs_handle),
        handle_to_ptr(grad_rhs_handle),
        batch,
        heads,
        dim,
        b1d_on_lhs != 0,
        op == kBinarySub);
    return check_cuda_launch("CUDA add/sub b1d/1hd backward kernel launch failed") ? 0 : 1;
}

extern "C" int lumen_cuda_add_sub_broadcast_backward_f32_device(
    uint64_t grad_handle,
    uint64_t grad_lhs_handle,
    uint64_t grad_rhs_handle,
    size_t ndim,
    const size_t* out_strides,
    const size_t* lhs_shape,
    const size_t* lhs_strides,
    const size_t* rhs_shape,
    const size_t* rhs_strides,
    size_t out_len,
    size_t lhs_len,
    size_t rhs_len,
    int op) {
    if (!validate_handle(grad_handle, "CUDA add/sub broadcast backward grad handle") ||
        !validate_handle(grad_lhs_handle, "CUDA add/sub broadcast backward lhs grad handle") ||
        !validate_handle(grad_rhs_handle, "CUDA add/sub broadcast backward rhs grad handle")) {
        return 1;
    }
    if (ndim == 0 || out_len == 0 || lhs_len == 0 || rhs_len == 0 ||
        out_strides == nullptr || lhs_shape == nullptr || lhs_strides == nullptr ||
        rhs_shape == nullptr || rhs_strides == nullptr) {
        set_error("CUDA add/sub broadcast backward received invalid metadata");
        return 1;
    }
    if (op != kBinaryAdd && op != kBinarySub) {
        set_error("CUDA add/sub broadcast backward received unsupported op");
        return 1;
    }

    if (!zero_f32_buffer(
            handle_to_ptr(grad_lhs_handle),
            lhs_len,
            "CUDA add/sub broadcast lhs grad initialization failed")) {
        return 1;
    }
    if (!zero_f32_buffer(
            handle_to_ptr(grad_rhs_handle),
            rhs_len,
            "CUDA add/sub broadcast rhs grad initialization failed")) {
        return 1;
    }

    size_t* d_out_strides = nullptr;
    size_t* d_lhs_shape = nullptr;
    size_t* d_lhs_strides = nullptr;
    size_t* d_rhs_shape = nullptr;
    size_t* d_rhs_strides = nullptr;
    const size_t* host_metadata[] = {
        out_strides, lhs_shape, lhs_strides, rhs_shape, rhs_strides};
    size_t* device_metadata[] = {
        d_out_strides, d_lhs_shape, d_lhs_strides, d_rhs_shape, d_rhs_strides};
    if (!upload_packed_elementwise_metadata(
            "failed to upload CUDA add/sub broadcast backward metadata",
            host_metadata,
            5,
            ndim,
            device_metadata)) {
        return 1;
    }
    d_out_strides = device_metadata[0];
    d_lhs_shape = device_metadata[1];
    d_lhs_strides = device_metadata[2];
    d_rhs_shape = device_metadata[3];
    d_rhs_strides = device_metadata[4];

    constexpr unsigned int block_size = 256;
    const unsigned int grid_size = linear_grid_size(out_len, block_size);
    add_sub_broadcast_backward_kernel<<<grid_size, block_size>>>(
        handle_to_ptr(grad_handle),
        handle_to_ptr(grad_lhs_handle),
        handle_to_ptr(grad_rhs_handle),
        d_out_strides,
        d_lhs_shape,
        d_lhs_strides,
        d_rhs_shape,
        d_rhs_strides,
        ndim,
        out_len,
        op == kBinarySub ? -1.0f : 1.0f);
    return check_cuda_launch("CUDA add/sub broadcast backward kernel launch failed") ? 0 : 1;
}

extern "C" int lumen_cuda_binary_broadcast_f32_device(
    uint64_t lhs_handle,
    uint64_t rhs_handle,
    uint64_t out_handle,
    size_t ndim,
    const size_t* out_shape,
    const size_t* out_strides,
    const size_t* lhs_shape,
    const size_t* lhs_strides,
    const size_t* rhs_shape,
    const size_t* rhs_strides,
    size_t len,
    int op) {
    if (!validate_handle(lhs_handle, "CUDA binary broadcast lhs handle") ||
        !validate_handle(rhs_handle, "CUDA binary broadcast rhs handle") ||
        !validate_handle(out_handle, "CUDA binary broadcast output handle")) {
        return 1;
    }
    if (ndim == 0 || len == 0 || out_shape == nullptr || out_strides == nullptr ||
        lhs_shape == nullptr || lhs_strides == nullptr || rhs_shape == nullptr || rhs_strides == nullptr) {
        set_error("CUDA binary broadcast received invalid metadata");
        return 1;
    }
    size_t lhs_len = 0;
    size_t rhs_len = 0;
    if (!checked_shape_numel("CUDA binary broadcast lhs shape", lhs_shape, ndim, &lhs_len) ||
        !checked_shape_numel("CUDA binary broadcast rhs shape", rhs_shape, ndim, &rhs_len)) {
        return 1;
    }
    constexpr unsigned int block_size = 256;
    if ((rhs_len == 1 && lhs_len == len) || (lhs_len == 1 && rhs_len == len)) {
        size_t vec_len = len / 4;
        const unsigned int grid_size = linear_grid_size(std::max<size_t>(vec_len, 1), block_size);
        if (rhs_len == 1 && lhs_len == len) {
            launch_binary_scalar_rhs_vec4_kernel(
                op,
                grid_size,
                block_size,
                reinterpret_cast<const float4*>(handle_to_ptr(lhs_handle)),
                handle_to_ptr(rhs_handle),
                reinterpret_cast<float4*>(handle_to_ptr(out_handle)),
                vec_len,
                len);
        } else {
            launch_binary_scalar_lhs_vec4_kernel(
                op,
                grid_size,
                block_size,
                handle_to_ptr(lhs_handle),
                reinterpret_cast<const float4*>(handle_to_ptr(rhs_handle)),
                reinterpret_cast<float4*>(handle_to_ptr(out_handle)),
                vec_len,
                len);
        }
        return check_cuda_launch("CUDA scalar broadcast binary kernel launch failed") ? 0 : 1;
    }

    if (len >= kLastdimBroadcastFastMinLen &&
        is_lastdim_rhs_broadcast(ndim, out_shape, lhs_shape, rhs_shape, lhs_len, rhs_len, len)) {
        size_t last_dim = rhs_len;
        bool vec4 = len % 4 == 0 && last_dim % 4 == 0;
        size_t launch_len = vec4 ? len / 4 : len;
        const unsigned int grid_size = linear_grid_size(std::max<size_t>(launch_len, 1), block_size);
        launch_binary_lastdim_rhs_kernel(
            op,
            vec4,
            grid_size,
            block_size,
            handle_to_ptr(lhs_handle),
            handle_to_ptr(rhs_handle),
            handle_to_ptr(out_handle),
            len,
            last_dim);
        return check_cuda_launch("CUDA last-dim rhs broadcast binary kernel launch failed") ? 0 : 1;
    }

    size_t* d_out_shape = nullptr;
    size_t* d_out_strides = nullptr;
    size_t* d_lhs_shape = nullptr;
    size_t* d_lhs_strides = nullptr;
    size_t* d_rhs_shape = nullptr;
    size_t* d_rhs_strides = nullptr;
    const size_t* host_metadata[] = {
        out_shape, out_strides, lhs_shape, lhs_strides, rhs_shape, rhs_strides};
    size_t* device_metadata[] = {
        d_out_shape, d_out_strides, d_lhs_shape, d_lhs_strides, d_rhs_shape, d_rhs_strides};
    if (!upload_packed_elementwise_metadata(
            "failed to upload CUDA binary broadcast metadata",
            host_metadata,
            6,
            ndim,
            device_metadata)) {
        return 1;
    }
    d_out_shape = device_metadata[0];
    d_out_strides = device_metadata[1];
    d_lhs_shape = device_metadata[2];
    d_lhs_strides = device_metadata[3];
    d_rhs_shape = device_metadata[4];
    d_rhs_strides = device_metadata[5];

    const unsigned int grid_size = linear_grid_size(len, block_size);
    launch_binary_broadcast_kernel(
        op,
        grid_size,
        block_size,
        handle_to_ptr(lhs_handle),
        handle_to_ptr(rhs_handle),
        handle_to_ptr(out_handle),
        d_out_shape,
        d_out_strides,
        d_lhs_shape,
        d_lhs_strides,
        d_rhs_shape,
        d_rhs_strides,
        ndim,
        len);
    return check_cuda_launch("CUDA binary broadcast kernel launch failed") ? 0 : 1;
}

extern "C" int lumen_cuda_binary_broadcast_backward_f32_device(
    uint64_t lhs_handle,
    uint64_t rhs_handle,
    uint64_t grad_handle,
    uint64_t grad_lhs_handle,
    uint64_t grad_rhs_handle,
    size_t ndim,
    const size_t* out_shape,
    const size_t* out_strides,
    const size_t* lhs_shape,
    const size_t* lhs_strides,
    const size_t* rhs_shape,
    const size_t* rhs_strides,
    size_t out_len,
    size_t lhs_len,
    size_t rhs_len,
    int op) {
    if (!validate_handle(lhs_handle, "CUDA binary broadcast backward lhs handle") ||
        !validate_handle(rhs_handle, "CUDA binary broadcast backward rhs handle") ||
        !validate_handle(grad_handle, "CUDA binary broadcast backward grad handle") ||
        !validate_handle(grad_lhs_handle, "CUDA binary broadcast backward lhs grad handle") ||
        !validate_handle(grad_rhs_handle, "CUDA binary broadcast backward rhs grad handle")) {
        return 1;
    }
    if (ndim == 0 || out_len == 0 || out_shape == nullptr || out_strides == nullptr ||
        lhs_shape == nullptr || lhs_strides == nullptr || rhs_shape == nullptr || rhs_strides == nullptr) {
        set_error("CUDA binary broadcast backward received invalid metadata");
        return 1;
    }

    constexpr unsigned int block_size = 256;
    if ((rhs_len == 1 && lhs_len == out_len) || (lhs_len == 1 && rhs_len == out_len)) {
        cudaError_t status;
        const unsigned int grid_size = linear_grid_size(out_len, block_size);
        if (rhs_len == 1 && lhs_len == out_len) {
            status = cudaMemset(handle_to_ptr(grad_rhs_handle), 0, sizeof(float));
            if (status != cudaSuccess) {
                set_cuda_error("CUDA scalar rhs broadcast grad initialization failed", status);
                return 1;
            }
            launch_binary_scalar_rhs_backward_kernel(
                op,
                grid_size,
                block_size,
                handle_to_ptr(lhs_handle),
                handle_to_ptr(rhs_handle),
                handle_to_ptr(grad_handle),
                handle_to_ptr(grad_lhs_handle),
                handle_to_ptr(grad_rhs_handle),
                out_len);
        } else {
            status = cudaMemset(handle_to_ptr(grad_lhs_handle), 0, sizeof(float));
            if (status != cudaSuccess) {
                set_cuda_error("CUDA scalar lhs broadcast grad initialization failed", status);
                return 1;
            }
            launch_binary_scalar_lhs_backward_kernel(
                op,
                grid_size,
                block_size,
                handle_to_ptr(lhs_handle),
                handle_to_ptr(rhs_handle),
                handle_to_ptr(grad_handle),
                handle_to_ptr(grad_lhs_handle),
                handle_to_ptr(grad_rhs_handle),
                out_len);
        }
        return check_cuda_launch("CUDA scalar broadcast binary backward kernel launch failed") ? 0 : 1;
    }

    if (out_len >= kLastdimBroadcastFastMinLen &&
        is_lastdim_rhs_broadcast(ndim, out_shape, lhs_shape, rhs_shape, lhs_len, rhs_len, out_len)) {
        size_t rows = out_len / rhs_len;
        bool use_atomic = rows < 64;
        if (use_atomic) {
            if (!zero_f32_buffer(
                    handle_to_ptr(grad_rhs_handle),
                    rhs_len,
                    "CUDA last-dim rhs broadcast grad initialization failed")) {
                return 1;
            }
        }
        const unsigned int grid_size = use_atomic
            ? linear_grid_size(out_len, block_size)
            : linear_grid_size(rhs_len, 1);
        launch_binary_lastdim_rhs_backward_kernel(
            op,
            use_atomic,
            grid_size,
            block_size,
            handle_to_ptr(lhs_handle),
            handle_to_ptr(rhs_handle),
            handle_to_ptr(grad_handle),
            handle_to_ptr(grad_lhs_handle),
            handle_to_ptr(grad_rhs_handle),
            rows,
            rhs_len);
        return check_cuda_launch("CUDA last-dim rhs broadcast binary backward kernel launch failed") ? 0 : 1;
    }

    if (!zero_f32_buffer(
            handle_to_ptr(grad_lhs_handle),
            lhs_len,
            "CUDA binary broadcast lhs grad initialization failed")) {
        return 1;
    }
    if (!zero_f32_buffer(
            handle_to_ptr(grad_rhs_handle),
            rhs_len,
            "CUDA binary broadcast rhs grad initialization failed")) {
        return 1;
    }

    size_t* d_out_shape = nullptr;
    size_t* d_out_strides = nullptr;
    size_t* d_lhs_shape = nullptr;
    size_t* d_lhs_strides = nullptr;
    size_t* d_rhs_shape = nullptr;
    size_t* d_rhs_strides = nullptr;
    const size_t* host_metadata[] = {
        out_shape, out_strides, lhs_shape, lhs_strides, rhs_shape, rhs_strides};
    size_t* device_metadata[] = {
        d_out_shape, d_out_strides, d_lhs_shape, d_lhs_strides, d_rhs_shape, d_rhs_strides};
    if (!upload_packed_elementwise_metadata(
            "failed to upload CUDA binary broadcast backward metadata",
            host_metadata,
            6,
            ndim,
            device_metadata)) {
        return 1;
    }
    d_out_shape = device_metadata[0];
    d_out_strides = device_metadata[1];
    d_lhs_shape = device_metadata[2];
    d_lhs_strides = device_metadata[3];
    d_rhs_shape = device_metadata[4];
    d_rhs_strides = device_metadata[5];

    const unsigned int grid_size = linear_grid_size(out_len, block_size);
    launch_binary_broadcast_backward_kernel(
        op,
        grid_size,
        block_size,
        handle_to_ptr(lhs_handle),
        handle_to_ptr(rhs_handle),
        handle_to_ptr(grad_handle),
        handle_to_ptr(grad_lhs_handle),
        handle_to_ptr(grad_rhs_handle),
        d_out_shape,
        d_out_strides,
        d_lhs_shape,
        d_lhs_strides,
        d_rhs_shape,
        d_rhs_strides,
        ndim,
        out_len);
    return check_cuda_launch("CUDA binary broadcast backward kernel launch failed") ? 0 : 1;
}
