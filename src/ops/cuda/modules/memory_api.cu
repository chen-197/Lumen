extern "C" int lumen_cuda_is_available() {
    int device_count = 0;
    cudaError_t status = cudaGetDeviceCount(&device_count);
    if (status != cudaSuccess) {
        set_cuda_error("failed to query CUDA devices", status);
        return 0;
    }
    return device_count > 0 ? 1 : 0;
}
extern "C" const char* lumen_cuda_last_error_message() {
    return g_last_error.c_str();
}

extern "C" int lumen_cuda_alloc_f32(size_t len, uint64_t* out_handle) {
    if (out_handle == nullptr) {
        set_error("CUDA alloc received a null output handle");
        return 1;
    }
    if (len > static_cast<size_t>(-1) / sizeof(float)) {
        set_error("CUDA alloc length overflow");
        return 1;
    }
    size_t bytes = len * sizeof(float);
    float* ptr = nullptr;
    if (try_take_pooled_cuda_buffer(bytes, reinterpret_cast<void**>(&ptr))) {
        *out_handle = reinterpret_cast<uint64_t>(ptr);
        return 0;
    }

    cudaError_t status = cudaMalloc(reinterpret_cast<void**>(&ptr), bytes);
    if (status != cudaSuccess) {
        if (!clear_cuda_buffer_pool("failed to release CUDA buffer pool after allocation failure")) {
            return 1;
        }
        status = cudaMalloc(reinterpret_cast<void**>(&ptr), bytes);
    }
    if (status != cudaSuccess) {
        set_cuda_error("failed to allocate CUDA buffer", status);
        return 1;
    }
    *out_handle = reinterpret_cast<uint64_t>(ptr);
    return 0;
}

extern "C" int lumen_cuda_upload_f32(uint64_t handle, const float* src, size_t len) {
    if (!validate_handle(handle, "CUDA upload handle")) {
        return 1;
    }
    if (src == nullptr) {
        set_error("CUDA upload source is null");
        return 1;
    }
    size_t bytes = 0;
    if (!checked_byte_length(len, sizeof(float), "CUDA upload length overflow", &bytes)) {
        return 1;
    }
    cudaError_t status = cudaMemcpy(handle_to_ptr(handle), src, bytes, cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        set_cuda_error("failed to upload CUDA buffer", status);
        return 1;
    }
    return 0;
}

extern "C" int lumen_cuda_upload_u16(uint64_t handle, const uint16_t* src, size_t len) {
    if (!validate_handle(handle, "CUDA upload u16 handle")) {
        return 1;
    }
    if (src == nullptr) {
        set_error("CUDA upload u16 source is null");
        return 1;
    }
    size_t bytes = 0;
    if (!checked_byte_length(len, sizeof(uint16_t), "CUDA upload u16 length overflow", &bytes)) {
        return 1;
    }
    cudaError_t status =
        cudaMemcpy(reinterpret_cast<void*>(handle), src, bytes, cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        set_cuda_error("failed to upload u16 data to CUDA", status);
        return 1;
    }
    return 0;
}

extern "C" int lumen_cuda_upload_i8(uint64_t handle, const int8_t* src, size_t len) {
    if (!validate_handle(handle, "CUDA upload i8 handle")) {
        return 1;
    }
    if (src == nullptr) {
        set_error("CUDA upload i8 source is null");
        return 1;
    }
    size_t bytes = 0;
    if (!checked_byte_length(len, sizeof(int8_t), "CUDA upload i8 length overflow", &bytes)) {
        return 1;
    }
    cudaError_t status =
        cudaMemcpy(reinterpret_cast<void*>(handle), src, bytes, cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        set_cuda_error("failed to upload i8 data to CUDA", status);
        return 1;
    }
    return 0;
}

extern "C" int lumen_cuda_upload_f32_offset(
    uint64_t handle,
    const float* src,
    size_t offset,
    size_t len) {
    if (!validate_handle(handle, "CUDA upload handle")) {
        return 1;
    }
    if (src == nullptr) {
        set_error("CUDA upload source is null");
        return 1;
    }
    if (offset > static_cast<size_t>(-1) / sizeof(float)) {
        set_error("CUDA upload slice offset overflow");
        return 1;
    }
    size_t bytes = 0;
    if (!checked_byte_length(len, sizeof(float), "CUDA upload slice length overflow", &bytes)) {
        return 1;
    }
    cudaError_t status = cudaMemcpy(
        handle_to_ptr(handle) + offset,
        src,
        bytes,
        cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        set_cuda_error("failed to upload CUDA buffer slice", status);
        return 1;
    }
    return 0;
}

extern "C" int lumen_cuda_copy_f32_offset(
    uint64_t dst_handle,
    size_t dst_offset,
    uint64_t src_handle,
    size_t src_offset,
    size_t len) {
    if (!validate_handle(dst_handle, "CUDA copy destination handle") ||
        !validate_handle(src_handle, "CUDA copy source handle")) {
        return 1;
    }
    if (dst_offset > static_cast<size_t>(-1) / sizeof(float) ||
        src_offset > static_cast<size_t>(-1) / sizeof(float)) {
        set_error("CUDA copy slice offset overflow");
        return 1;
    }
    size_t bytes = 0;
    if (!checked_byte_length(len, sizeof(float), "CUDA copy slice length overflow", &bytes)) {
        return 1;
    }
    cudaError_t status = cudaMemcpy(
        handle_to_ptr(dst_handle) + dst_offset,
        handle_to_ptr(src_handle) + src_offset,
        bytes,
        cudaMemcpyDeviceToDevice);
    if (status != cudaSuccess) {
        set_cuda_error("failed to copy CUDA tensor slice", status);
        return 1;
    }
    return 0;
}

extern "C" int lumen_cuda_append_kv_cache_f32_device(
    uint64_t dst_handle,
    uint64_t src_handle,
    size_t batch_size,
    size_t num_heads,
    size_t src_seq_len,
    size_t dst_seq_len,
    size_t dim,
    size_t dst_start) {
    if (!validate_handle(dst_handle, "CUDA KV cache destination handle") ||
        !validate_handle(src_handle, "CUDA KV cache source handle")) {
        return 1;
    }
    if (batch_size == 0 || num_heads == 0 || src_seq_len == 0 || dst_seq_len == 0 || dim == 0) {
        set_error("CUDA KV cache append dimensions must be greater than zero");
        return 1;
    }
    if (dst_start > dst_seq_len || src_seq_len > dst_seq_len - dst_start) {
        set_error("CUDA KV cache append range is out of bounds");
        return 1;
    }

    size_t total = 0;
    if (!checked_product(
            "CUDA KV cache append length overflow",
            {batch_size, num_heads, src_seq_len, dim},
            &total)) {
        return 1;
    }
    constexpr int block_size = 256;
    const unsigned int grid_size = linear_grid_size(total, block_size);
    append_kv_cache_kernel<<<grid_size, block_size>>>(
        handle_to_ptr(dst_handle),
        handle_to_ptr(src_handle),
        batch_size,
        num_heads,
        src_seq_len,
        dst_seq_len,
        dim,
        dst_start);
    if (!check_cuda_launch("CUDA KV cache append kernel launch failed")) {
        return 1;
    }
    return 0;
}

extern "C" int lumen_cuda_kv_cache_prefix_f32_device(
    uint64_t src_handle,
    uint64_t out_handle,
    size_t batch_size,
    size_t num_heads,
    size_t active_seq_len,
    size_t src_seq_len,
    size_t dim) {
    if (!validate_handle(src_handle, "CUDA KV cache source handle") ||
        !validate_handle(out_handle, "CUDA KV cache prefix output handle")) {
        return 1;
    }
    if (batch_size == 0 || num_heads == 0 || active_seq_len == 0 || src_seq_len == 0 ||
        dim == 0) {
        set_error("CUDA KV cache prefix dimensions must be greater than zero");
        return 1;
    }
    if (active_seq_len > src_seq_len) {
        set_error("CUDA KV cache prefix range is out of bounds");
        return 1;
    }

    size_t total = 0;
    if (!checked_product(
            "CUDA KV cache prefix length overflow",
            {batch_size, num_heads, active_seq_len, dim},
            &total)) {
        return 1;
    }
    constexpr int block_size = 256;
    const unsigned int grid_size = linear_grid_size(total, block_size);
    kv_cache_prefix_kernel<<<grid_size, block_size>>>(
        handle_to_ptr(src_handle),
        handle_to_ptr(out_handle),
        batch_size,
        num_heads,
        active_seq_len,
        src_seq_len,
        dim);
    if (!check_cuda_launch("CUDA KV cache prefix kernel launch failed")) {
        return 1;
    }
    return 0;
}

extern "C" int lumen_cuda_kv_cache_prefix_typed_device(
    uint64_t src_handle,
    int dtype,
    uint64_t out_handle,
    size_t batch_size,
    size_t num_heads,
    size_t active_seq_len,
    size_t src_seq_len,
    size_t dim) {
    if (!validate_handle(src_handle, "CUDA typed KV cache source handle") ||
        !validate_handle(out_handle, "CUDA typed KV cache prefix output handle")) {
        return 1;
    }
    if (batch_size == 0 || num_heads == 0 || active_seq_len == 0 || src_seq_len == 0 ||
        dim == 0) {
        set_error("CUDA typed KV cache prefix dimensions must be greater than zero");
        return 1;
    }
    if (active_seq_len > src_seq_len) {
        set_error("CUDA typed KV cache prefix range is out of bounds");
        return 1;
    }

    size_t total = 0;
    if (!checked_product(
            "CUDA typed KV cache prefix length overflow",
            {batch_size, num_heads, active_seq_len, dim},
            &total)) {
        return 1;
    }
    constexpr int block_size = 256;
    const unsigned int grid_size = linear_grid_size(total, block_size);
    switch (dtype) {
        case 0:
            kv_cache_prefix_typed_kernel<float><<<grid_size, block_size>>>(
                handle_to_ptr(src_handle),
                handle_to_ptr(out_handle),
                batch_size,
                num_heads,
                active_seq_len,
                src_seq_len,
                dim);
            break;
        case 1:
        case 2:
            kv_cache_prefix_typed_kernel<uint16_t><<<grid_size, block_size>>>(
                reinterpret_cast<const uint16_t*>(handle_to_ptr(src_handle)),
                reinterpret_cast<uint16_t*>(handle_to_ptr(out_handle)),
                batch_size,
                num_heads,
                active_seq_len,
                src_seq_len,
                dim);
            break;
        case 3:
            kv_cache_prefix_typed_kernel<int8_t><<<grid_size, block_size>>>(
                reinterpret_cast<const int8_t*>(handle_to_ptr(src_handle)),
                reinterpret_cast<int8_t*>(handle_to_ptr(out_handle)),
                batch_size,
                num_heads,
                active_seq_len,
                src_seq_len,
                dim);
            break;
        default:
            set_error("unsupported dtype for CUDA typed KV cache prefix");
            return 1;
    }
    if (!check_cuda_launch("CUDA typed KV cache prefix kernel launch failed")) {
        return 1;
    }
    return 0;
}

extern "C" int lumen_cuda_download_f32(uint64_t handle, float* dst, size_t len) {
    if (!validate_handle(handle, "CUDA download handle")) {
        return 1;
    }
    if (dst == nullptr) {
        set_error("CUDA download destination is null");
        return 1;
    }
    size_t bytes = 0;
    if (!checked_byte_length(len, sizeof(float), "CUDA download length overflow", &bytes)) {
        return 1;
    }
    cudaError_t status = cudaMemcpy(dst, handle_to_ptr(handle), bytes, cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        set_cuda_error("failed to download CUDA buffer", status);
        return 1;
    }
    return 0;
}

extern "C" int lumen_cuda_download_u16(uint64_t handle, uint16_t* dst, size_t len) {
    if (!validate_handle(handle, "CUDA u16 download handle")) {
        return 1;
    }
    if (dst == nullptr) {
        set_error("CUDA u16 download destination is null");
        return 1;
    }
    size_t bytes = 0;
    if (!checked_byte_length(len, sizeof(uint16_t), "CUDA u16 download length overflow", &bytes)) {
        return 1;
    }
    cudaError_t status = cudaMemcpy(dst, handle_to_ptr(handle), bytes, cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        set_cuda_error("failed to download CUDA u16 storage", status);
        return 1;
    }
    return 0;
}

extern "C" int lumen_cuda_download_i8(uint64_t handle, int8_t* dst, size_t len) {
    if (!validate_handle(handle, "CUDA i8 download handle")) {
        return 1;
    }
    if (dst == nullptr) {
        set_error("CUDA i8 download destination is null");
        return 1;
    }
    size_t bytes = 0;
    if (!checked_byte_length(len, sizeof(int8_t), "CUDA i8 download length overflow", &bytes)) {
        return 1;
    }
    cudaError_t status = cudaMemcpy(dst, handle_to_ptr(handle), bytes, cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        set_cuda_error("failed to download CUDA i8 storage", status);
        return 1;
    }
    return 0;
}

extern "C" int lumen_cuda_download_f32_offset(
    uint64_t handle,
    float* dst,
    size_t offset,
    size_t len) {
    if (!validate_handle(handle, "CUDA download handle")) {
        return 1;
    }
    if (dst == nullptr) {
        set_error("CUDA download destination is null");
        return 1;
    }
    if (offset > static_cast<size_t>(-1) / sizeof(float)) {
        set_error("CUDA download slice offset overflow");
        return 1;
    }
    size_t bytes = 0;
    if (!checked_byte_length(len, sizeof(float), "CUDA download slice length overflow", &bytes)) {
        return 1;
    }
    cudaError_t status = cudaMemcpy(
        dst,
        handle_to_ptr(handle) + offset,
        bytes,
        cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        set_cuda_error("failed to download CUDA buffer slice", status);
        return 1;
    }
    return 0;
}

extern "C" void lumen_cuda_free_f32(uint64_t handle, size_t len) {
    release_cuda_buffer(handle, len);
}

extern "C" int lumen_cuda_synchronize() {
    return sync_cuda("CUDA synchronize failed") ? 0 : 1;
}

extern "C" int lumen_cuda_release_cached_memory() {
    if (!sync_cuda("CUDA cached memory release synchronization failed")) {
        return 1;
    }
    if (!thread_cublas_handle().release("CUDA cuBLAS cache release failed")) {
        return 1;
    }
#if LUMEN_HAS_CUDNN
    if (!thread_cudnn_handle().release("CUDA cuDNN cache release failed")) {
        return 1;
    }
#endif
    if (!release_thread_cuda_workspaces("CUDA workspace cache release failed")) {
        return 1;
    }
    return clear_cuda_buffer_pool("CUDA buffer pool release failed") ? 0 : 1;
}
