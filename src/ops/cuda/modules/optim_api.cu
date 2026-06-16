bool append_optimizer_metadata_block(
    size_t count,
    size_t element_size,
    size_t alignment,
    size_t* cursor,
    size_t* offset,
    const char* context) {
    size_t bytes = 0;
    if (!checked_align_up(context, *cursor, alignment, offset) ||
        !checked_product(context, {count, element_size}, &bytes) ||
        !checked_add(context, *offset, bytes, cursor)) {
        return false;
    }
    return true;
}

extern "C" int lumen_cuda_sgd_update_f32_device(
    uint64_t param_handle,
    uint64_t grad_handle,
    size_t len,
    float lr) {
    if (!validate_handle(param_handle, "CUDA SGD param handle") ||
        !validate_handle(grad_handle, "CUDA SGD grad handle")) {
        return 1;
    }
    if (len == 0) {
        return 0;
    }

    constexpr unsigned int block_size = 256;
    const unsigned int grid_size = linear_grid_size(len, block_size);
    sgd_update_kernel<<<grid_size, block_size>>>(
        handle_to_ptr(param_handle),
        handle_to_ptr(grad_handle),
        len,
        lr);
    cudaError_t launch_status = cudaGetLastError();
    if (launch_status != cudaSuccess) {
        set_cuda_error("CUDA SGD update kernel launch failed", launch_status);
        return 1;
    }
    return 0;
}
extern "C" int lumen_cuda_sgd_update_f32_batched_device(
    const uint64_t* param_handles,
    const uint64_t* grad_handles,
    const size_t* lens,
    size_t count,
    float lr) {
    if (param_handles == nullptr || grad_handles == nullptr || lens == nullptr) {
        set_error("CUDA batched SGD received null metadata pointer");
        return 1;
    }
    if (count == 0) {
        set_error("CUDA batched SGD count must be greater than zero");
        return 1;
    }
    if (!validate_grid_yz_dimension("CUDA batched SGD count exceeds grid.y range", count)) {
        return 1;
    }

    thread_local std::vector<float*> param_ptrs;
    thread_local std::vector<const float*> grad_ptrs;
    param_ptrs.resize(count);
    grad_ptrs.resize(count);
    size_t max_len = 0;
    for (size_t i = 0; i < count; ++i) {
        if (!validate_handle(param_handles[i], "CUDA batched SGD param handle") ||
            !validate_handle(grad_handles[i], "CUDA batched SGD grad handle")) {
            return 1;
        }
        if (lens[i] == 0) {
            set_error("CUDA batched SGD tensor length must be greater than zero");
            return 1;
        }
        param_ptrs[i] = handle_to_ptr(param_handles[i]);
        grad_ptrs[i] = handle_to_ptr(grad_handles[i]);
        max_len = std::max(max_len, lens[i]);
    }

    size_t metadata_bytes = 0;
    size_t params_offset = 0;
    size_t grads_offset = 0;
    size_t lens_offset = 0;
    if (!append_optimizer_metadata_block(
            count,
            sizeof(float*),
            alignof(float*),
            &metadata_bytes,
            &params_offset,
            "CUDA batched SGD metadata length overflow") ||
        !append_optimizer_metadata_block(
            count,
            sizeof(const float*),
            alignof(const float*),
            &metadata_bytes,
            &grads_offset,
            "CUDA batched SGD metadata length overflow") ||
        !append_optimizer_metadata_block(
            count,
            sizeof(size_t),
            alignof(size_t),
            &metadata_bytes,
            &lens_offset,
            "CUDA batched SGD metadata length overflow")) {
        return 1;
    }
    thread_local std::vector<unsigned char> metadata;
    metadata.resize(metadata_bytes);
    std::memcpy(metadata.data() + params_offset, param_ptrs.data(), count * sizeof(float*));
    std::memcpy(metadata.data() + grads_offset, grad_ptrs.data(), count * sizeof(const float*));
    std::memcpy(metadata.data() + lens_offset, lens, count * sizeof(size_t));

    thread_local ReusableCudaWorkspace d_metadata;
    if (!d_metadata.ensure(metadata_bytes, "CUDA batched SGD metadata allocation failed")) {
        return 1;
    }
    cudaError_t status = cudaMemcpy(d_metadata.ptr, metadata.data(), metadata_bytes, cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        set_cuda_error("CUDA batched SGD metadata upload failed", status);
        return 1;
    }

    char* d_base = reinterpret_cast<char*>(d_metadata.ptr);
    constexpr int block_size = 256;
    const unsigned int grid_x = std::min(linear_grid_size(max_len, block_size), 1024u);
    dim3 grid(grid_x, static_cast<unsigned int>(count), 1);
    sgd_update_batched_kernel<<<grid, block_size>>>(
        reinterpret_cast<float**>(d_base + params_offset),
        reinterpret_cast<const float**>(d_base + grads_offset),
        reinterpret_cast<const size_t*>(d_base + lens_offset),
        count,
        lr);
    return check_cuda_launch("CUDA batched SGD update kernel launch failed") ? 0 : 1;
}

extern "C" int lumen_cuda_quantize_f32_storage_device(
    uint64_t param_handle,
    size_t len,
    int dtype,
    float scale) {
    if (!validate_handle(param_handle, "CUDA quantize param handle")) {
        return 1;
    }
    if (dtype < 1 || dtype > 3) {
        set_error("CUDA quantize dtype must be F16, BF16, or I8");
        return 1;
    }
    if (dtype == 3 && !(scale > 0.0f && isfinite(scale))) {
        set_error("CUDA quantize I8 scale must be finite and > 0");
        return 1;
    }
    if (len == 0) {
        return 0;
    }

    constexpr unsigned int block_size = 256;
    const unsigned int grid_size = linear_grid_size(len, block_size);
    quantize_f32_storage_kernel<<<grid_size, block_size>>>(
        handle_to_ptr(param_handle),
        len,
        dtype,
        scale);
    return check_cuda_launch("CUDA quantize f32 storage kernel launch failed") ? 0 : 1;
}

extern "C" int lumen_cuda_quantize_f32_to_i8_dynamic_device(
    uint64_t input_handle,
    uint64_t out_handle,
    size_t len,
    float* out_scale) {
    if (!validate_handle(input_handle, "CUDA dynamic i8 quantize input handle") ||
        !validate_handle(out_handle, "CUDA dynamic i8 quantize output handle")) {
        return 1;
    }
    if (out_scale == nullptr) {
        set_error("CUDA dynamic i8 quantize scale output is null");
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
            "failed to allocate CUDA dynamic i8 quantize reduction buffer")) {
        return 1;
    }
    float* partial = static_cast<float*>(partial_workspace.ptr);

    f32_absmax_blocks_kernel<<<grid_size, block_size, block_size * sizeof(float)>>>(
        handle_to_ptr(input_handle),
        partial,
        len);
    if (!check_cuda_launch("CUDA dynamic i8 quantize absmax kernel launch failed")) {
        return 1;
    }

    thread_local ReusableCudaWorkspace device_max_workspace;
    if (!device_max_workspace.ensure(
            sizeof(float),
            "failed to allocate CUDA dynamic i8 quantize final reduction buffer")) {
        return 1;
    }
    float* device_max = static_cast<float*>(device_max_workspace.ptr);
    f32_absmax_finalize_kernel<<<1, block_size, block_size * sizeof(float)>>>(
        partial,
        device_max,
        static_cast<size_t>(grid_size));
    if (!check_cuda_launch("CUDA dynamic i8 quantize final absmax reduction kernel launch failed")) {
        return 1;
    }

    quantize_f32_to_i8_device_absmax_kernel<<<grid_size, block_size>>>(
        handle_to_ptr(input_handle),
        reinterpret_cast<int8_t*>(handle_to_ptr(out_handle)),
        len,
        device_max);
    if (!check_cuda_launch("CUDA dynamic i8 quantize kernel launch failed")) {
        return 1;
    }

    float max_abs = 0.0f;
    cudaError_t status =
        cudaMemcpy(&max_abs, device_max, sizeof(float), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        set_cuda_error("failed to download CUDA dynamic i8 quantize absmax", status);
        return 1;
    }

    float scale =
        max_abs > 0.0f && isfinite(max_abs) ? std::max(max_abs / 127.0f, FLT_MIN) : 1.0f;
    *out_scale = scale;
    return 0;
}

extern "C" int lumen_cuda_f32_to_lowp_storage_device(
    uint64_t input_handle,
    uint64_t out_handle,
    size_t len,
    int dtype) {
    if (!validate_handle(input_handle, "CUDA f32 to lowp input handle") ||
        !validate_handle(out_handle, "CUDA f32 to lowp output handle")) {
        return 1;
    }
    if (dtype != kDTypeF16 && dtype != kDTypeBF16) {
        set_error("CUDA f32 to lowp dtype must be F16 or BF16");
        return 1;
    }
    if (len == 0) {
        return 0;
    }

    constexpr unsigned int block_size = 256;
    const unsigned int grid_size = linear_grid_size(len, block_size);
    f32_to_lowp_storage_kernel<<<grid_size, block_size>>>(
        handle_to_ptr(input_handle),
        reinterpret_cast<uint16_t*>(handle_to_ptr(out_handle)),
        len,
        dtype);
    return check_cuda_launch("CUDA f32 to lowp storage kernel launch failed") ? 0 : 1;
}

extern "C" int lumen_cuda_sgd_momentum_update_f32_device(
    uint64_t param_handle,
    uint64_t grad_handle,
    uint64_t velocity_handle,
    size_t len,
    float lr,
    float momentum) {
    if (!validate_handle(param_handle, "CUDA SGD momentum param handle") ||
        !validate_handle(grad_handle, "CUDA SGD momentum grad handle") ||
        !validate_handle(velocity_handle, "CUDA SGD momentum velocity handle")) {
        return 1;
    }
    if (len == 0) {
        return 0;
    }

    constexpr unsigned int block_size = 256;
    const unsigned int grid_size = linear_grid_size(len, block_size);
    sgd_momentum_update_kernel<<<grid_size, block_size>>>(
        handle_to_ptr(param_handle),
        handle_to_ptr(grad_handle),
        handle_to_ptr(velocity_handle),
        len,
        lr,
        momentum);
    cudaError_t launch_status = cudaGetLastError();
    if (launch_status != cudaSuccess) {
        set_cuda_error("CUDA SGD momentum update kernel launch failed", launch_status);
        return 1;
    }
    return 0;
}

extern "C" int lumen_cuda_sgd_momentum_update_f32_batched_device(
    const uint64_t* param_handles,
    const uint64_t* grad_handles,
    const uint64_t* velocity_handles,
    const size_t* lens,
    size_t count,
    float lr,
    float momentum) {
    if (param_handles == nullptr || grad_handles == nullptr || velocity_handles == nullptr || lens == nullptr) {
        set_error("CUDA batched SGD momentum received null metadata pointer");
        return 1;
    }
    if (count == 0) {
        set_error("CUDA batched SGD momentum count must be greater than zero");
        return 1;
    }
    if (!validate_grid_yz_dimension(
            "CUDA batched SGD momentum count exceeds grid.y range", count)) {
        return 1;
    }

    thread_local std::vector<float*> param_ptrs;
    thread_local std::vector<const float*> grad_ptrs;
    thread_local std::vector<float*> velocity_ptrs;
    param_ptrs.resize(count);
    grad_ptrs.resize(count);
    velocity_ptrs.resize(count);
    size_t max_len = 0;
    for (size_t i = 0; i < count; ++i) {
        if (!validate_handle(param_handles[i], "CUDA batched SGD momentum param handle") ||
            !validate_handle(grad_handles[i], "CUDA batched SGD momentum grad handle") ||
            !validate_handle(velocity_handles[i], "CUDA batched SGD momentum velocity handle")) {
            return 1;
        }
        if (lens[i] == 0) {
            set_error("CUDA batched SGD momentum tensor length must be greater than zero");
            return 1;
        }
        param_ptrs[i] = handle_to_ptr(param_handles[i]);
        grad_ptrs[i] = handle_to_ptr(grad_handles[i]);
        velocity_ptrs[i] = handle_to_ptr(velocity_handles[i]);
        max_len = std::max(max_len, lens[i]);
    }

    size_t metadata_bytes = 0;
    size_t params_offset = 0;
    size_t grads_offset = 0;
    size_t velocities_offset = 0;
    size_t lens_offset = 0;
    if (!append_optimizer_metadata_block(
            count,
            sizeof(float*),
            alignof(float*),
            &metadata_bytes,
            &params_offset,
            "CUDA batched SGD momentum metadata length overflow") ||
        !append_optimizer_metadata_block(
            count,
            sizeof(const float*),
            alignof(const float*),
            &metadata_bytes,
            &grads_offset,
            "CUDA batched SGD momentum metadata length overflow") ||
        !append_optimizer_metadata_block(
            count,
            sizeof(float*),
            alignof(float*),
            &metadata_bytes,
            &velocities_offset,
            "CUDA batched SGD momentum metadata length overflow") ||
        !append_optimizer_metadata_block(
            count,
            sizeof(size_t),
            alignof(size_t),
            &metadata_bytes,
            &lens_offset,
            "CUDA batched SGD momentum metadata length overflow")) {
        return 1;
    }
    thread_local std::vector<unsigned char> metadata;
    metadata.resize(metadata_bytes);
    std::memcpy(metadata.data() + params_offset, param_ptrs.data(), count * sizeof(float*));
    std::memcpy(metadata.data() + grads_offset, grad_ptrs.data(), count * sizeof(const float*));
    std::memcpy(metadata.data() + velocities_offset, velocity_ptrs.data(), count * sizeof(float*));
    std::memcpy(metadata.data() + lens_offset, lens, count * sizeof(size_t));

    thread_local ReusableCudaWorkspace d_metadata;
    if (!d_metadata.ensure(metadata_bytes, "CUDA batched SGD momentum metadata allocation failed")) {
        return 1;
    }
    cudaError_t status = cudaMemcpy(d_metadata.ptr, metadata.data(), metadata_bytes, cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        set_cuda_error("CUDA batched SGD momentum metadata upload failed", status);
        return 1;
    }

    char* d_base = reinterpret_cast<char*>(d_metadata.ptr);
    constexpr int block_size = 256;
    const unsigned int grid_x = std::min(linear_grid_size(max_len, block_size), 1024u);
    dim3 grid(grid_x, static_cast<unsigned int>(count), 1);
    sgd_momentum_update_batched_kernel<<<grid, block_size>>>(
        reinterpret_cast<float**>(d_base + params_offset),
        reinterpret_cast<const float**>(d_base + grads_offset),
        reinterpret_cast<float**>(d_base + velocities_offset),
        reinterpret_cast<const size_t*>(d_base + lens_offset),
        count,
        lr,
        momentum);
    return check_cuda_launch("CUDA batched SGD momentum update kernel launch failed") ? 0 : 1;
}

extern "C" int lumen_cuda_adam_update_f32_device(
    uint64_t param_handle,
    uint64_t grad_handle,
    uint64_t exp_avg_handle,
    uint64_t exp_avg_sq_handle,
    size_t len,
    float lr,
    float beta1,
    float beta2,
    float bias_correction1,
    float bias_correction2,
    float eps) {
    if (!validate_handle(param_handle, "CUDA Adam param handle") ||
        !validate_handle(grad_handle, "CUDA Adam grad handle") ||
        !validate_handle(exp_avg_handle, "CUDA Adam exp_avg handle") ||
        !validate_handle(exp_avg_sq_handle, "CUDA Adam exp_avg_sq handle")) {
        return 1;
    }
    if (len == 0) {
        return 0;
    }

    constexpr unsigned int block_size = 256;
    size_t vec_len = len / 4;
    if (vec_len > 0) {
        const unsigned int grid_size = linear_grid_size(vec_len, block_size);
        adam_update_vec4_kernel<<<grid_size, block_size>>>(
            reinterpret_cast<float4*>(handle_to_ptr(param_handle)),
            reinterpret_cast<const float4*>(handle_to_ptr(grad_handle)),
            reinterpret_cast<float4*>(handle_to_ptr(exp_avg_handle)),
            reinterpret_cast<float4*>(handle_to_ptr(exp_avg_sq_handle)),
            vec_len,
            len,
            lr,
            beta1,
            beta2,
            bias_correction1,
            bias_correction2,
            eps);
    } else {
        const unsigned int grid_size = linear_grid_size(len, block_size);
        adam_update_kernel<<<grid_size, block_size>>>(
            handle_to_ptr(param_handle),
            handle_to_ptr(grad_handle),
            handle_to_ptr(exp_avg_handle),
            handle_to_ptr(exp_avg_sq_handle),
            len,
            lr,
            beta1,
            beta2,
            bias_correction1,
            bias_correction2,
            eps);
    }
    cudaError_t launch_status = cudaGetLastError();
    if (launch_status != cudaSuccess) {
        set_cuda_error("CUDA Adam update kernel launch failed", launch_status);
        return 1;
    }
    return 0;
}

extern "C" int lumen_cuda_adam_update_f32_batched_device(
    const uint64_t* param_handles,
    const uint64_t* grad_handles,
    const uint64_t* exp_avg_handles,
    const uint64_t* exp_avg_sq_handles,
    const size_t* lens,
    size_t count,
    float lr,
    float beta1,
    float beta2,
    float bias_correction1,
    float bias_correction2,
    float eps) {
    if (param_handles == nullptr || grad_handles == nullptr || exp_avg_handles == nullptr ||
        exp_avg_sq_handles == nullptr || lens == nullptr) {
        set_error("CUDA batched Adam received null metadata pointer");
        return 1;
    }
    if (count == 0) {
        set_error("CUDA batched Adam count must be greater than zero");
        return 1;
    }
    if (!validate_grid_yz_dimension("CUDA batched Adam count exceeds grid.y range", count)) {
        return 1;
    }

    thread_local std::vector<float*> param_ptrs;
    thread_local std::vector<const float*> grad_ptrs;
    thread_local std::vector<float*> exp_avg_ptrs;
    thread_local std::vector<float*> exp_avg_sq_ptrs;
    param_ptrs.resize(count);
    grad_ptrs.resize(count);
    exp_avg_ptrs.resize(count);
    exp_avg_sq_ptrs.resize(count);
    size_t max_len = 0;
    for (size_t i = 0; i < count; ++i) {
        if (!validate_handle(param_handles[i], "CUDA batched Adam param handle") ||
            !validate_handle(grad_handles[i], "CUDA batched Adam grad handle") ||
            !validate_handle(exp_avg_handles[i], "CUDA batched Adam exp_avg handle") ||
            !validate_handle(exp_avg_sq_handles[i], "CUDA batched Adam exp_avg_sq handle")) {
            return 1;
        }
        if (lens[i] == 0) {
            set_error("CUDA batched Adam tensor length must be greater than zero");
            return 1;
        }
        param_ptrs[i] = handle_to_ptr(param_handles[i]);
        grad_ptrs[i] = handle_to_ptr(grad_handles[i]);
        exp_avg_ptrs[i] = handle_to_ptr(exp_avg_handles[i]);
        exp_avg_sq_ptrs[i] = handle_to_ptr(exp_avg_sq_handles[i]);
        max_len = std::max(max_len, lens[i]);
    }

    size_t metadata_bytes = 0;
    size_t params_offset = 0;
    size_t grads_offset = 0;
    size_t exp_avgs_offset = 0;
    size_t exp_avg_sqs_offset = 0;
    size_t lens_offset = 0;
    if (!append_optimizer_metadata_block(
            count,
            sizeof(float*),
            alignof(float*),
            &metadata_bytes,
            &params_offset,
            "CUDA batched Adam metadata length overflow") ||
        !append_optimizer_metadata_block(
            count,
            sizeof(const float*),
            alignof(const float*),
            &metadata_bytes,
            &grads_offset,
            "CUDA batched Adam metadata length overflow") ||
        !append_optimizer_metadata_block(
            count,
            sizeof(float*),
            alignof(float*),
            &metadata_bytes,
            &exp_avgs_offset,
            "CUDA batched Adam metadata length overflow") ||
        !append_optimizer_metadata_block(
            count,
            sizeof(float*),
            alignof(float*),
            &metadata_bytes,
            &exp_avg_sqs_offset,
            "CUDA batched Adam metadata length overflow") ||
        !append_optimizer_metadata_block(
            count,
            sizeof(size_t),
            alignof(size_t),
            &metadata_bytes,
            &lens_offset,
            "CUDA batched Adam metadata length overflow")) {
        return 1;
    }
    thread_local std::vector<unsigned char> metadata;
    metadata.resize(metadata_bytes);
    std::memcpy(metadata.data() + params_offset, param_ptrs.data(), count * sizeof(float*));
    std::memcpy(metadata.data() + grads_offset, grad_ptrs.data(), count * sizeof(const float*));
    std::memcpy(metadata.data() + exp_avgs_offset, exp_avg_ptrs.data(), count * sizeof(float*));
    std::memcpy(metadata.data() + exp_avg_sqs_offset, exp_avg_sq_ptrs.data(), count * sizeof(float*));
    std::memcpy(metadata.data() + lens_offset, lens, count * sizeof(size_t));

    thread_local ReusableCudaWorkspace d_metadata;
    if (!d_metadata.ensure(metadata_bytes, "CUDA batched Adam metadata allocation failed")) {
        return 1;
    }
    cudaError_t status = cudaMemcpy(d_metadata.ptr, metadata.data(), metadata_bytes, cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        set_cuda_error("CUDA batched Adam metadata upload failed", status);
        return 1;
    }

    char* d_base = reinterpret_cast<char*>(d_metadata.ptr);
    constexpr int block_size = 256;
    const unsigned int grid_x = std::min(linear_grid_size(max_len, block_size), 1024u);
    dim3 grid(grid_x, static_cast<unsigned int>(count), 1);
    adam_update_batched_kernel<<<grid, block_size>>>(
        reinterpret_cast<float**>(d_base + params_offset),
        reinterpret_cast<const float**>(d_base + grads_offset),
        reinterpret_cast<float**>(d_base + exp_avgs_offset),
        reinterpret_cast<float**>(d_base + exp_avg_sqs_offset),
        reinterpret_cast<const size_t*>(d_base + lens_offset),
        count,
        lr,
        beta1,
        beta2,
        bias_correction1,
        bias_correction2,
        eps);
    return check_cuda_launch("CUDA batched Adam update kernel launch failed") ? 0 : 1;
}
