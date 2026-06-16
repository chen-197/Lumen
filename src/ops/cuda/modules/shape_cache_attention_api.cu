bool upload_packed_shape_metadata(
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
extern "C" int lumen_cuda_permute_f32_device(
    uint64_t input_handle,
    uint64_t out_handle,
    size_t ndim,
    const size_t* out_shape,
    const size_t* out_strides,
    const size_t* mapped_input_strides,
    size_t len) {
    if (!validate_handle(input_handle, "CUDA permute input handle") ||
        !validate_handle(out_handle, "CUDA permute output handle")) {
        return 1;
    }
    if (ndim == 0 || out_shape == nullptr || out_strides == nullptr || mapped_input_strides == nullptr) {
        set_error("CUDA permute received invalid metadata");
        return 1;
    }
    if (len == 0) {
        return 0;
    }

    if (ndim == 4) {
        constexpr int block_size = 256;
        if (out_strides[3] == 1 &&
            mapped_input_strides[3] == 1 &&
            out_shape[3] % 4 == 0 &&
            mapped_input_strides[0] % 4 == 0 &&
            mapped_input_strides[1] % 4 == 0 &&
            mapped_input_strides[2] % 4 == 0) {
            size_t vec_len = len / 4;
            const unsigned int grid_size = linear_grid_size(vec_len, block_size);
            permute4d_lastdim_vec4_kernel<<<grid_size, block_size>>>(
                reinterpret_cast<const float4*>(handle_to_ptr(input_handle)),
                reinterpret_cast<float4*>(handle_to_ptr(out_handle)),
                out_shape[0],
                out_shape[1],
                out_shape[2],
                out_shape[3],
                out_strides[0],
                out_strides[1],
                out_strides[2],
                mapped_input_strides[0],
                mapped_input_strides[1],
                mapped_input_strides[2],
                vec_len);
            return check_cuda_launch("CUDA 4D vec4 permute kernel launch failed") ? 0 : 1;
        }

        const unsigned int grid_size = linear_grid_size(len, block_size);
        permute4d_kernel<<<grid_size, block_size>>>(
            handle_to_ptr(input_handle),
            handle_to_ptr(out_handle),
            out_shape[0],
            out_shape[1],
            out_shape[2],
            out_shape[3],
            out_strides[0],
            out_strides[1],
            out_strides[2],
            out_strides[3],
            mapped_input_strides[0],
            mapped_input_strides[1],
            mapped_input_strides[2],
            mapped_input_strides[3],
            len);
        return check_cuda_launch("CUDA 4D permute kernel launch failed") ? 0 : 1;
    }

    size_t* d_out_shape = nullptr;
    size_t* d_out_strides = nullptr;
    size_t* d_mapped_input_strides = nullptr;
    const size_t* host_metadata[] = {out_shape, out_strides, mapped_input_strides};
    size_t* device_metadata[] = {nullptr, nullptr, nullptr};
    if (!upload_packed_shape_metadata(
            "CUDA permute metadata upload failed",
            host_metadata,
            3,
            ndim,
            device_metadata)) {
        return 1;
    }
    d_out_shape = device_metadata[0];
    d_out_strides = device_metadata[1];
    d_mapped_input_strides = device_metadata[2];

    constexpr int block_size = 256;
    const unsigned int grid_size = linear_grid_size(len, block_size);
    permute_kernel<<<grid_size, block_size>>>(
        handle_to_ptr(input_handle),
        handle_to_ptr(out_handle),
        ndim,
        d_out_shape,
        d_out_strides,
        d_mapped_input_strides,
        len);
    bool ok = check_cuda_launch("CUDA permute kernel launch failed");
    return ok ? 0 : 1;
}

template <typename T>
int launch_permute_typed(
    uint64_t input_handle,
    uint64_t out_handle,
    size_t ndim,
    const size_t* out_shape,
    const size_t* out_strides,
    const size_t* mapped_input_strides,
    size_t len) {
    if (ndim == 4) {
        constexpr int block_size = 256;
        const unsigned int grid_size = linear_grid_size(len, block_size);
        permute4d_typed_kernel<T><<<grid_size, block_size>>>(
            reinterpret_cast<const T*>(handle_to_ptr(input_handle)),
            reinterpret_cast<T*>(handle_to_ptr(out_handle)),
            out_shape[0],
            out_shape[1],
            out_shape[2],
            out_shape[3],
            out_strides[0],
            out_strides[1],
            out_strides[2],
            out_strides[3],
            mapped_input_strides[0],
            mapped_input_strides[1],
            mapped_input_strides[2],
            mapped_input_strides[3],
            len);
        return check_cuda_launch("CUDA typed 4D permute kernel launch failed") ? 0 : 1;
    }

    size_t* d_out_shape = nullptr;
    size_t* d_out_strides = nullptr;
    size_t* d_mapped_input_strides = nullptr;
    const size_t* host_metadata[] = {out_shape, out_strides, mapped_input_strides};
    size_t* device_metadata[] = {nullptr, nullptr, nullptr};
    if (!upload_packed_shape_metadata(
            "CUDA typed permute metadata upload failed",
            host_metadata,
            3,
            ndim,
            device_metadata)) {
        return 1;
    }
    d_out_shape = device_metadata[0];
    d_out_strides = device_metadata[1];
    d_mapped_input_strides = device_metadata[2];

    constexpr int block_size = 256;
    const unsigned int grid_size = linear_grid_size(len, block_size);
    permute_typed_kernel<T><<<grid_size, block_size>>>(
        reinterpret_cast<const T*>(handle_to_ptr(input_handle)),
        reinterpret_cast<T*>(handle_to_ptr(out_handle)),
        ndim,
        d_out_shape,
        d_out_strides,
        d_mapped_input_strides,
        len);
    bool ok = check_cuda_launch("CUDA typed permute kernel launch failed");
    return ok ? 0 : 1;
}

extern "C" int lumen_cuda_permute_typed_device(
    uint64_t input_handle,
    int dtype,
    uint64_t out_handle,
    size_t ndim,
    const size_t* out_shape,
    const size_t* out_strides,
    const size_t* mapped_input_strides,
    size_t len) {
    if (!validate_handle(input_handle, "CUDA typed permute input handle") ||
        !validate_handle(out_handle, "CUDA typed permute output handle")) {
        return 1;
    }
    if (ndim == 0 || out_shape == nullptr || out_strides == nullptr || mapped_input_strides == nullptr) {
        set_error("CUDA typed permute received invalid metadata");
        return 1;
    }
    if (len == 0) {
        return 0;
    }

    switch (dtype) {
        case 0:
            return launch_permute_typed<float>(
                input_handle,
                out_handle,
                ndim,
                out_shape,
                out_strides,
                mapped_input_strides,
                len);
        case 1:
        case 2:
            return launch_permute_typed<uint16_t>(
                input_handle,
                out_handle,
                ndim,
                out_shape,
                out_strides,
                mapped_input_strides,
                len);
        case 3:
            return launch_permute_typed<int8_t>(
                input_handle,
                out_handle,
                ndim,
                out_shape,
                out_strides,
                mapped_input_strides,
                len);
        default:
            set_error("unsupported dtype for CUDA typed permute");
            return 1;
    }
}

extern "C" int lumen_cuda_slice_lastdim_f32_device(
    uint64_t input_handle,
    uint64_t out_handle,
    size_t outer,
    size_t input_last_dim,
    size_t start,
    size_t slice_len) {
    if (!validate_handle(input_handle, "CUDA slice input handle") ||
        !validate_handle(out_handle, "CUDA slice output handle")) {
        return 1;
    }
    if (outer == 0 || input_last_dim == 0 || slice_len == 0) {
        set_error("CUDA slice dimensions must be greater than zero");
        return 1;
    }
    if (start > input_last_dim || slice_len > input_last_dim - start) {
        set_error("CUDA slice range is out of bounds");
        return 1;
    }

    constexpr int block_size = 256;
    size_t total = 0;
    if (!checked_product("CUDA slice output length overflow", {outer, slice_len}, &total)) {
        return 1;
    }
    const unsigned int grid_size = linear_grid_size(total, block_size);
    slice_lastdim_kernel<<<grid_size, block_size>>>(
        handle_to_ptr(input_handle),
        handle_to_ptr(out_handle),
        outer,
        input_last_dim,
        start,
        slice_len);
    if (!check_cuda_launch("CUDA slice kernel launch failed")) {
        return 1;
    }
    return 0;
}

extern "C" int lumen_cuda_slice_lastdim_typed_device(
    uint64_t input_handle,
    int dtype,
    uint64_t out_handle,
    size_t outer,
    size_t input_last_dim,
    size_t start,
    size_t slice_len) {
    if (!validate_handle(input_handle, "CUDA typed slice input handle") ||
        !validate_handle(out_handle, "CUDA typed slice output handle")) {
        return 1;
    }
    if (outer == 0 || input_last_dim == 0 || slice_len == 0) {
        set_error("CUDA typed slice dimensions must be greater than zero");
        return 1;
    }
    if (start > input_last_dim || slice_len > input_last_dim - start) {
        set_error("CUDA typed slice range is out of bounds");
        return 1;
    }

    constexpr int block_size = 256;
    size_t total = 0;
    if (!checked_product("CUDA typed slice output length overflow", {outer, slice_len}, &total)) {
        return 1;
    }
    const unsigned int grid_size = linear_grid_size(total, block_size);
    switch (dtype) {
        case 0:
            slice_lastdim_typed_kernel<float><<<grid_size, block_size>>>(
                handle_to_ptr(input_handle),
                handle_to_ptr(out_handle),
                outer,
                input_last_dim,
                start,
                slice_len);
            break;
        case 1:
        case 2:
            slice_lastdim_typed_kernel<uint16_t><<<grid_size, block_size>>>(
                reinterpret_cast<const uint16_t*>(handle_to_ptr(input_handle)),
                reinterpret_cast<uint16_t*>(handle_to_ptr(out_handle)),
                outer,
                input_last_dim,
                start,
                slice_len);
            break;
        case 3:
            slice_lastdim_typed_kernel<int8_t><<<grid_size, block_size>>>(
                reinterpret_cast<const int8_t*>(handle_to_ptr(input_handle)),
                reinterpret_cast<int8_t*>(handle_to_ptr(out_handle)),
                outer,
                input_last_dim,
                start,
                slice_len);
            break;
        default:
            set_error("unsupported dtype for CUDA typed slice_lastdim");
            return 1;
    }
    if (!check_cuda_launch("CUDA typed slice kernel launch failed")) {
        return 1;
    }
    return 0;
}

extern "C" int lumen_cuda_slice_lastdim_backward_f32_device(
    uint64_t grad_handle,
    uint64_t out_handle,
    size_t outer,
    size_t input_last_dim,
    size_t start,
    size_t slice_len) {
    if (!validate_handle(grad_handle, "CUDA slice_lastdim backward grad handle") ||
        !validate_handle(out_handle, "CUDA slice_lastdim backward output handle")) {
        return 1;
    }
    if (outer == 0 || input_last_dim == 0 || slice_len == 0 ||
        start > input_last_dim || slice_len > input_last_dim - start) {
        set_error("CUDA slice_lastdim backward received invalid dimensions");
        return 1;
    }

    size_t output_len = 0;
    size_t output_bytes = 0;
    size_t total = 0;
    if (!checked_product(
            "CUDA slice_lastdim backward output length overflow",
            {outer, input_last_dim},
            &output_len) ||
        !checked_byte_length(
            output_len,
            sizeof(float),
            "CUDA slice_lastdim backward output byte length overflow",
            &output_bytes) ||
        !checked_product(
            "CUDA slice_lastdim backward grad length overflow", {outer, slice_len}, &total)) {
        return 1;
    }
    cudaError_t memset_status = cudaMemset(handle_to_ptr(out_handle), 0, output_bytes);
    if (memset_status != cudaSuccess) {
        set_cuda_error("CUDA slice_lastdim backward output initialization failed", memset_status);
        return 1;
    }

    constexpr int block_size = 256;
    const unsigned int grid_size = linear_grid_size(total, block_size);
    slice_lastdim_backward_kernel<<<grid_size, block_size>>>(
        handle_to_ptr(grad_handle),
        handle_to_ptr(out_handle),
        outer,
        input_last_dim,
        start,
        slice_len);
    if (!check_cuda_launch("CUDA slice_lastdim backward kernel launch failed")) {
        return 1;
    }
    return 0;
}

extern "C" int lumen_cuda_cat_f32_device(
    uint64_t lhs_handle,
    uint64_t rhs_handle,
    uint64_t out_handle,
    size_t ndim,
    const size_t* out_shape,
    const size_t* out_strides,
    const size_t* lhs_strides,
    const size_t* rhs_strides,
    size_t axis,
    size_t lhs_axis_len,
    size_t len) {
    if (!validate_handle(lhs_handle, "CUDA cat lhs handle") ||
        !validate_handle(rhs_handle, "CUDA cat rhs handle") ||
        !validate_handle(out_handle, "CUDA cat output handle")) {
        return 1;
    }
    if (ndim == 0 || len == 0) {
        set_error("CUDA cat received invalid metadata");
        return 1;
    }
    if (axis >= ndim) {
        set_error("CUDA cat axis is out of bounds");
        return 1;
    }
    if (out_shape == nullptr || out_strides == nullptr || lhs_strides == nullptr || rhs_strides == nullptr) {
        set_error("CUDA cat metadata pointers must not be null");
        return 1;
    }

    size_t* d_out_shape = nullptr;
    size_t* d_out_strides = nullptr;
    size_t* d_lhs_strides = nullptr;
    size_t* d_rhs_strides = nullptr;
    const size_t* host_metadata[] = {out_shape, out_strides, lhs_strides, rhs_strides};
    size_t* device_metadata[] = {nullptr, nullptr, nullptr, nullptr};
    if (!upload_packed_shape_metadata(
            "CUDA cat metadata upload failed",
            host_metadata,
            4,
            ndim,
            device_metadata)) {
        return 1;
    }
    d_out_shape = device_metadata[0];
    d_out_strides = device_metadata[1];
    d_lhs_strides = device_metadata[2];
    d_rhs_strides = device_metadata[3];

    constexpr int block_size = 256;
    const unsigned int grid_size = linear_grid_size(len, block_size);
    cat_kernel<<<grid_size, block_size>>>(
        handle_to_ptr(lhs_handle),
        handle_to_ptr(rhs_handle),
        handle_to_ptr(out_handle),
        d_out_shape,
        d_out_strides,
        d_lhs_strides,
        d_rhs_strides,
        ndim,
        axis,
        lhs_axis_len,
        len);

    bool ok = check_cuda_launch("CUDA cat kernel launch failed");
    return ok ? 0 : 1;
}

template <typename T>
int launch_cat_typed(
    uint64_t lhs_handle,
    uint64_t rhs_handle,
    uint64_t out_handle,
    size_t ndim,
    const size_t* out_shape,
    const size_t* out_strides,
    const size_t* lhs_strides,
    const size_t* rhs_strides,
    size_t axis,
    size_t lhs_axis_len,
    size_t len) {
    size_t* d_out_shape = nullptr;
    size_t* d_out_strides = nullptr;
    size_t* d_lhs_strides = nullptr;
    size_t* d_rhs_strides = nullptr;
    const size_t* host_metadata[] = {out_shape, out_strides, lhs_strides, rhs_strides};
    size_t* device_metadata[] = {nullptr, nullptr, nullptr, nullptr};
    if (!upload_packed_shape_metadata(
            "CUDA typed cat metadata upload failed",
            host_metadata,
            4,
            ndim,
            device_metadata)) {
        return 1;
    }
    d_out_shape = device_metadata[0];
    d_out_strides = device_metadata[1];
    d_lhs_strides = device_metadata[2];
    d_rhs_strides = device_metadata[3];

    constexpr int block_size = 256;
    const unsigned int grid_size = linear_grid_size(len, block_size);
    cat_typed_kernel<T><<<grid_size, block_size>>>(
        reinterpret_cast<const T*>(handle_to_ptr(lhs_handle)),
        reinterpret_cast<const T*>(handle_to_ptr(rhs_handle)),
        reinterpret_cast<T*>(handle_to_ptr(out_handle)),
        d_out_shape,
        d_out_strides,
        d_lhs_strides,
        d_rhs_strides,
        ndim,
        axis,
        lhs_axis_len,
        len);

    bool ok = check_cuda_launch("CUDA typed cat kernel launch failed");
    return ok ? 0 : 1;
}

extern "C" int lumen_cuda_cat_typed_device(
    uint64_t lhs_handle,
    uint64_t rhs_handle,
    int dtype,
    uint64_t out_handle,
    size_t ndim,
    const size_t* out_shape,
    const size_t* out_strides,
    const size_t* lhs_strides,
    const size_t* rhs_strides,
    size_t axis,
    size_t lhs_axis_len,
    size_t len) {
    if (!validate_handle(lhs_handle, "CUDA typed cat lhs handle") ||
        !validate_handle(rhs_handle, "CUDA typed cat rhs handle") ||
        !validate_handle(out_handle, "CUDA typed cat output handle")) {
        return 1;
    }
    if (ndim == 0 || len == 0) {
        set_error("CUDA typed cat received invalid metadata");
        return 1;
    }
    if (axis >= ndim) {
        set_error("CUDA typed cat axis is out of bounds");
        return 1;
    }
    if (out_shape == nullptr || out_strides == nullptr || lhs_strides == nullptr || rhs_strides == nullptr) {
        set_error("CUDA typed cat metadata pointers must not be null");
        return 1;
    }

    switch (dtype) {
        case 0:
            return launch_cat_typed<float>(
                lhs_handle,
                rhs_handle,
                out_handle,
                ndim,
                out_shape,
                out_strides,
                lhs_strides,
                rhs_strides,
                axis,
                lhs_axis_len,
                len);
        case 1:
        case 2:
            return launch_cat_typed<uint16_t>(
                lhs_handle,
                rhs_handle,
                out_handle,
                ndim,
                out_shape,
                out_strides,
                lhs_strides,
                rhs_strides,
                axis,
                lhs_axis_len,
                len);
        case 3:
            return launch_cat_typed<int8_t>(
                lhs_handle,
                rhs_handle,
                out_handle,
                ndim,
                out_shape,
                out_strides,
                lhs_strides,
                rhs_strides,
                axis,
                lhs_axis_len,
                len);
        default:
            set_error("unsupported dtype for CUDA typed cat");
            return 1;
    }
}

extern "C" int lumen_cuda_cat_backward_slice_f32_device(
    uint64_t grad_handle,
    uint64_t out_handle,
    size_t ndim,
    const size_t* input_shape,
    const size_t* input_strides,
    const size_t* out_strides,
    size_t axis,
    size_t axis_start,
    size_t len) {
    if (!validate_handle(grad_handle, "CUDA cat backward grad handle") ||
        !validate_handle(out_handle, "CUDA cat backward output handle")) {
        return 1;
    }
    if (ndim == 0 || len == 0) {
        set_error("CUDA cat backward received invalid metadata");
        return 1;
    }
    if (axis >= ndim) {
        set_error("CUDA cat backward axis is out of bounds");
        return 1;
    }
    if (input_shape == nullptr || input_strides == nullptr || out_strides == nullptr) {
        set_error("CUDA cat backward metadata pointers must not be null");
        return 1;
    }

    size_t* d_input_shape = nullptr;
    size_t* d_input_strides = nullptr;
    size_t* d_out_strides = nullptr;
    const size_t* host_metadata[] = {input_shape, input_strides, out_strides};
    size_t* device_metadata[] = {nullptr, nullptr, nullptr};
    if (!upload_packed_shape_metadata(
            "CUDA cat backward metadata upload failed",
            host_metadata,
            3,
            ndim,
            device_metadata)) {
        return 1;
    }
    d_input_shape = device_metadata[0];
    d_input_strides = device_metadata[1];
    d_out_strides = device_metadata[2];

    constexpr int block_size = 256;
    const unsigned int grid_size = linear_grid_size(len, block_size);
    cat_backward_slice_kernel<<<grid_size, block_size>>>(
        handle_to_ptr(grad_handle),
        handle_to_ptr(out_handle),
        d_input_shape,
        d_input_strides,
        d_out_strides,
        ndim,
        axis,
        axis_start,
        len);

    bool ok = check_cuda_launch("CUDA cat backward kernel launch failed");
    return ok ? 0 : 1;
}

extern "C" int lumen_cuda_repeat_kv_f32_device(
    uint64_t input_handle,
    uint64_t out_handle,
    size_t batch_size,
    size_t num_kv_heads,
    size_t seq_len,
    size_t dim,
    size_t n_rep) {
    if (!validate_handle(input_handle, "CUDA repeat_kv input handle") ||
        !validate_handle(out_handle, "CUDA repeat_kv output handle")) {
        return 1;
    }
    if (batch_size == 0 || num_kv_heads == 0 || seq_len == 0 || dim == 0 || n_rep == 0) {
        set_error("CUDA repeat_kv dimensions must be greater than zero");
        return 1;
    }

    size_t num_heads = 0;
    size_t len = 0;
    if (!checked_product(
            "CUDA repeat_kv head count overflow",
            {batch_size, num_kv_heads, n_rep},
            &num_heads) ||
        !checked_product("CUDA repeat_kv output length overflow", {num_heads, seq_len, dim}, &len)) {
        return 1;
    }
    constexpr int block_size = 256;
    const unsigned int grid_size = linear_grid_size(len, block_size);
    repeat_kv_kernel<<<grid_size, block_size>>>(
        handle_to_ptr(input_handle),
        handle_to_ptr(out_handle),
        num_heads,
        seq_len,
        dim,
        n_rep);
    if (!check_cuda_launch("CUDA repeat_kv kernel launch failed")) {
        return 1;
    }
    return 0;
}

extern "C" int lumen_cuda_repeat_kv_typed_device(
    uint64_t input_handle,
    int dtype,
    uint64_t out_handle,
    size_t batch_size,
    size_t num_kv_heads,
    size_t seq_len,
    size_t dim,
    size_t n_rep) {
    if (!validate_handle(input_handle, "CUDA typed repeat_kv input handle") ||
        !validate_handle(out_handle, "CUDA typed repeat_kv output handle")) {
        return 1;
    }
    if (batch_size == 0 || num_kv_heads == 0 || seq_len == 0 || dim == 0 || n_rep == 0) {
        set_error("CUDA typed repeat_kv dimensions must be greater than zero");
        return 1;
    }

    size_t num_heads = 0;
    size_t len = 0;
    if (!checked_product(
            "CUDA typed repeat_kv head count overflow",
            {batch_size, num_kv_heads, n_rep},
            &num_heads) ||
        !checked_product(
            "CUDA typed repeat_kv output length overflow", {num_heads, seq_len, dim}, &len)) {
        return 1;
    }
    constexpr int block_size = 256;
    const unsigned int grid_size = linear_grid_size(len, block_size);
    switch (dtype) {
        case kDTypeF32:
            repeat_kv_typed_kernel<float><<<grid_size, block_size>>>(
                handle_to_ptr(input_handle),
                handle_to_ptr(out_handle),
                num_heads,
                seq_len,
                dim,
                n_rep);
            break;
        case kDTypeF16:
        case kDTypeBF16:
            repeat_kv_typed_kernel<uint16_t><<<grid_size, block_size>>>(
                reinterpret_cast<const uint16_t*>(handle_to_ptr(input_handle)),
                reinterpret_cast<uint16_t*>(handle_to_ptr(out_handle)),
                num_heads,
                seq_len,
                dim,
                n_rep);
            break;
        case kDTypeI8:
            repeat_kv_typed_kernel<int8_t><<<grid_size, block_size>>>(
                reinterpret_cast<const int8_t*>(handle_to_ptr(input_handle)),
                reinterpret_cast<int8_t*>(handle_to_ptr(out_handle)),
                num_heads,
                seq_len,
                dim,
                n_rep);
            break;
        default:
            set_error("CUDA typed repeat_kv received unsupported dtype");
            return 1;
    }
    if (!check_cuda_launch("CUDA typed repeat_kv kernel launch failed")) {
        return 1;
    }
    return 0;
}

extern "C" int lumen_cuda_append_kv_cache_pair_f32_device(
    uint64_t k_dst_handle,
    uint64_t v_dst_handle,
    uint64_t k_src_handle,
    uint64_t v_src_handle,
    size_t batch_size,
    size_t num_heads,
    size_t src_seq_len,
    size_t dst_seq_len,
    size_t dim,
    size_t dst_start) {
    if (!validate_handle(k_dst_handle, "CUDA KV cache K destination handle") ||
        !validate_handle(v_dst_handle, "CUDA KV cache V destination handle") ||
        !validate_handle(k_src_handle, "CUDA KV cache K source handle") ||
        !validate_handle(v_src_handle, "CUDA KV cache V source handle")) {
        return 1;
    }
    if (batch_size == 0 || num_heads == 0 || src_seq_len == 0 || dst_seq_len == 0 || dim == 0) {
        set_error("CUDA KV cache pair append dimensions must be greater than zero");
        return 1;
    }
    if (dst_start > dst_seq_len || src_seq_len > dst_seq_len - dst_start) {
        set_error("CUDA KV cache pair append range is out of bounds");
        return 1;
    }

    constexpr int block_size = 256;
    size_t total = 0;
    if (!checked_product(
            "CUDA KV cache pair append length overflow",
            {batch_size, num_heads, src_seq_len, dim},
            &total)) {
        return 1;
    }
    const unsigned int grid_size = linear_grid_size(total, block_size);
    append_kv_cache_pair_kernel<<<grid_size, block_size>>>(
        handle_to_ptr(k_dst_handle),
        handle_to_ptr(v_dst_handle),
        handle_to_ptr(k_src_handle),
        handle_to_ptr(v_src_handle),
        batch_size,
        num_heads,
        src_seq_len,
        dst_seq_len,
        dim,
        dst_start);
    if (!check_cuda_launch("CUDA KV cache pair append kernel launch failed")) {
        return 1;
    }
    return 0;
}

extern "C" int lumen_cuda_decode_rope_q_append_kv_f32_device(
    uint64_t q_src_handle,
    uint64_t k_src_handle,
    uint64_t v_src_handle,
    uint64_t cos_handle,
    uint64_t sin_handle,
    uint64_t q_out_handle,
    uint64_t k_cache_handle,
    uint64_t v_cache_handle,
    size_t batch_size,
    size_t num_heads,
    size_t num_kv_heads,
    size_t dim,
    size_t dst_seq_len,
    size_t offset,
    size_t cache_seq_len) {
    if (!validate_handle(q_src_handle, "CUDA decode RoPE Q source handle") ||
        !validate_handle(k_src_handle, "CUDA decode RoPE K source handle") ||
        !validate_handle(v_src_handle, "CUDA decode RoPE V source handle") ||
        !validate_handle(cos_handle, "CUDA decode RoPE cos handle") ||
        !validate_handle(sin_handle, "CUDA decode RoPE sin handle") ||
        !validate_handle(q_out_handle, "CUDA decode RoPE Q output handle") ||
        !validate_handle(k_cache_handle, "CUDA decode RoPE K cache handle") ||
        !validate_handle(v_cache_handle, "CUDA decode RoPE V cache handle")) {
        return 1;
    }
    if (batch_size == 0 || num_heads == 0 || num_kv_heads == 0 || dim == 0 || dst_seq_len == 0 ||
        cache_seq_len == 0) {
        set_error("CUDA decode RoPE append dimensions must be greater than zero");
        return 1;
    }
    if ((dim % 2) != 0) {
        set_error("CUDA decode RoPE append expects an even hidden dimension");
        return 1;
    }
    if (offset >= dst_seq_len || offset >= cache_seq_len) {
        set_error("CUDA decode RoPE append offset is out of bounds");
        return 1;
    }

    constexpr int block_size = 256;
    size_t q_len = 0;
    size_t kv_len = 0;
    if (!checked_product(
            "CUDA decode RoPE Q length overflow", {batch_size, num_heads, dim}, &q_len) ||
        !checked_product(
            "CUDA decode RoPE KV length overflow", {batch_size, num_kv_heads, dim}, &kv_len)) {
        return 1;
    }
    size_t total = q_len > kv_len ? q_len : kv_len;
    const unsigned int grid_size = linear_grid_size(total, block_size);
    decode_rope_q_append_kv_kernel<<<grid_size, block_size>>>(
        handle_to_ptr(q_src_handle),
        handle_to_ptr(k_src_handle),
        handle_to_ptr(v_src_handle),
        handle_to_ptr(cos_handle),
        handle_to_ptr(sin_handle),
        handle_to_ptr(q_out_handle),
        handle_to_ptr(k_cache_handle),
        handle_to_ptr(v_cache_handle),
        batch_size,
        num_heads,
        num_kv_heads,
        dim,
        dst_seq_len,
        offset);
    if (!check_cuda_launch("CUDA decode RoPE append kernel launch failed")) {
        return 1;
    }
    return 0;
}

extern "C" int lumen_cuda_repeat_kv_backward_f32_device(
    uint64_t grad_handle,
    uint64_t out_handle,
    size_t batch_size,
    size_t num_kv_heads,
    size_t seq_len,
    size_t dim,
    size_t n_rep) {
    if (!validate_handle(grad_handle, "CUDA repeat_kv backward grad handle") ||
        !validate_handle(out_handle, "CUDA repeat_kv backward output handle")) {
        return 1;
    }
    if (batch_size == 0 || num_kv_heads == 0 || seq_len == 0 || dim == 0 || n_rep == 0) {
        set_error("CUDA repeat_kv backward dimensions must be greater than zero");
        return 1;
    }

    size_t input_len = 0;
    if (!checked_product(
            "CUDA repeat_kv backward input length overflow",
            {batch_size, num_kv_heads, seq_len, dim},
            &input_len)) {
        return 1;
    }
    constexpr int block_size = 256;
    const unsigned int grid_size = linear_grid_size(input_len, block_size);
    repeat_kv_backward_kernel<<<grid_size, block_size>>>(
        handle_to_ptr(grad_handle),
        handle_to_ptr(out_handle),
        batch_size,
        num_kv_heads,
        seq_len,
        dim,
        n_rep);
    if (!check_cuda_launch("CUDA repeat_kv backward kernel launch failed")) {
        return 1;
    }
    return 0;
}

extern "C" int lumen_cuda_decode_attention_f32_device(
    uint64_t q_handle,
    uint64_t k_handle,
    uint64_t v_handle,
    uint64_t out_handle,
    size_t batch_size,
    size_t num_heads,
    size_t num_kv_heads,
    size_t active_seq_len,
    size_t cache_seq_len,
    size_t dim,
    size_t n_rep,
    float scale) {
    if (!validate_handle(q_handle, "CUDA decode attention q handle") ||
        !validate_handle(k_handle, "CUDA decode attention k handle") ||
        !validate_handle(v_handle, "CUDA decode attention v handle") ||
        !validate_handle(out_handle, "CUDA decode attention out handle")) {
        return 1;
    }
    if (batch_size == 0 || num_heads == 0 || num_kv_heads == 0 || active_seq_len == 0 ||
        cache_seq_len == 0 || dim == 0 || n_rep == 0) {
        set_error("CUDA decode attention dimensions must be greater than zero");
        return 1;
    }
    if (active_seq_len > cache_seq_len) {
        set_error("CUDA decode attention active sequence length exceeds cache length");
        return 1;
    }
    if (!validate_finite_value(scale, "CUDA decode attention scale must be finite")) {
        return 1;
    }

    constexpr int block_size = 256;
    size_t rows = 0;
    size_t shared_bytes = 0;
    if (!checked_product(
            "CUDA decode attention row count overflow", {batch_size, num_heads}, &rows)) {
        return 1;
    }
    if (dim > (static_cast<size_t>(-1) - block_size - 4) / 2) {
        set_error("CUDA decode attention shared memory length overflow");
        return 1;
    }
    if (!checked_byte_length(
            block_size + 2 * dim + 4,
            sizeof(float),
            "CUDA decode attention shared memory length overflow",
            &shared_bytes)) {
        return 1;
    }
    decode_attention_kernel<<<linear_grid_size(rows, 1), block_size, shared_bytes>>>(
        handle_to_ptr(q_handle),
        handle_to_ptr(k_handle),
        handle_to_ptr(v_handle),
        handle_to_ptr(out_handle),
        num_heads,
        num_kv_heads,
        active_seq_len,
        cache_seq_len,
        dim,
        n_rep,
        rows,
        scale);
    if (!check_cuda_launch("CUDA decode attention kernel launch failed")) {
        return 1;
    }
    return 0;
}

extern "C" int lumen_cuda_prefill_attention_f32_device(
    uint64_t q_handle,
    uint64_t k_handle,
    uint64_t v_handle,
    uint64_t out_handle,
    size_t batch_size,
    size_t num_heads,
    size_t num_kv_heads,
    size_t q_seq_len,
    size_t active_seq_len,
    size_t cache_seq_len,
    size_t dim,
    size_t n_rep,
    size_t past_len,
    float scale,
    int is_causal) {
    if (!validate_handle(q_handle, "CUDA prefill attention q handle") ||
        !validate_handle(k_handle, "CUDA prefill attention k handle") ||
        !validate_handle(v_handle, "CUDA prefill attention v handle") ||
        !validate_handle(out_handle, "CUDA prefill attention out handle")) {
        return 1;
    }
    if (batch_size == 0 || num_heads == 0 || num_kv_heads == 0 || q_seq_len == 0 ||
        active_seq_len == 0 || cache_seq_len == 0 || dim == 0 || n_rep == 0) {
        set_error("CUDA prefill attention dimensions must be greater than zero");
        return 1;
    }
    if (active_seq_len > cache_seq_len ||
        past_len > active_seq_len ||
        q_seq_len > active_seq_len - past_len) {
        set_error("CUDA prefill attention sequence range is out of bounds");
        return 1;
    }
    if (!validate_finite_value(scale, "CUDA prefill attention scale must be finite")) {
        return 1;
    }

    constexpr int block_size = 256;
    size_t rows = 0;
    size_t shared_bytes = 0;
    if (!checked_product(
            "CUDA prefill attention row count overflow",
            {batch_size, num_heads, q_seq_len},
            &rows)) {
        return 1;
    }
    if (dim > (static_cast<size_t>(-1) - block_size - 4) / 2) {
        set_error("CUDA prefill attention shared memory length overflow");
        return 1;
    }
    if (!checked_byte_length(
            block_size + 2 * dim + 4,
            sizeof(float),
            "CUDA prefill attention shared memory length overflow",
            &shared_bytes)) {
        return 1;
    }
    prefill_attention_kernel<<<linear_grid_size(rows, 1), block_size, shared_bytes>>>(
        handle_to_ptr(q_handle),
        handle_to_ptr(k_handle),
        handle_to_ptr(v_handle),
        handle_to_ptr(out_handle),
        num_heads,
        num_kv_heads,
        q_seq_len,
        active_seq_len,
        cache_seq_len,
        dim,
        n_rep,
        past_len,
        rows,
        scale,
        is_causal);
    if (!check_cuda_launch("CUDA prefill attention kernel launch failed")) {
        return 1;
    }
    return 0;
}
