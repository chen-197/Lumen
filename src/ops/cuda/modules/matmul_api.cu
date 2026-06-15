extern "C" int lumen_cuda_matvec_argmax_f32_device(
    uint64_t input_handle,
    uint64_t weight_handle,
    size_t* out_indices,
    size_t batch_size,
    size_t vocab_size,
    size_t hidden_size) {
    if (!validate_handle(input_handle, "CUDA matvec argmax input handle") ||
        !validate_handle(weight_handle, "CUDA matvec argmax weight handle")) {
        return 1;
    }
    if (out_indices == nullptr) {
        set_error("CUDA matvec argmax output is null");
        return 1;
    }
    if (batch_size == 0 || vocab_size == 0 || hidden_size == 0) {
        set_error("CUDA matvec argmax dimensions must be greater than zero");
        return 1;
    }
    if (batch_size > static_cast<size_t>(INT_MAX) ||
        vocab_size > static_cast<size_t>(INT_MAX) ||
        hidden_size > static_cast<size_t>(INT_MAX)) {
        set_error("CUDA matvec argmax dimensions exceed cuBLAS int range");
        return 1;
    }

    CublasHandle cublas;
    if (!init_cublas(cublas)) {
        return 1;
    }

    const size_t max_size = static_cast<size_t>(-1);
    if (batch_size > max_size / vocab_size ||
        batch_size * vocab_size > max_size / sizeof(float)) {
        set_error("CUDA matvec argmax logits length overflow");
        return 1;
    }
    if (batch_size > max_size / sizeof(size_t)) {
        set_error("CUDA matvec argmax output length overflow");
        return 1;
    }

    thread_local ReusableCudaWorkspace logits_tmp;
    if (!logits_tmp.ensure(
            batch_size * vocab_size * sizeof(float),
            "failed to allocate CUDA matvec argmax logits")) {
        return 1;
    }
    thread_local ReusableCudaWorkspace device_out_tmp;
    if (!device_out_tmp.ensure(
            batch_size * sizeof(size_t),
            "failed to allocate CUDA matvec argmax output")) {
        return 1;
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
    if (batch_size == 1) {
        cublas_status = cublasSgemv(
            cublas.handle,
            CUBLAS_OP_T,
            static_cast<int>(hidden_size),
            static_cast<int>(vocab_size),
            &alpha,
            handle_to_ptr(weight_handle),
            static_cast<int>(hidden_size),
            handle_to_ptr(input_handle),
            1,
            &beta,
            static_cast<float*>(logits_tmp.ptr),
            1);
    } else {
        cublas_status = cublasSgemm(
            cublas.handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            static_cast<int>(vocab_size),
            static_cast<int>(batch_size),
            static_cast<int>(hidden_size),
            &alpha,
            handle_to_ptr(weight_handle),
            static_cast<int>(hidden_size),
            handle_to_ptr(input_handle),
            static_cast<int>(hidden_size),
            &beta,
            static_cast<float*>(logits_tmp.ptr),
            static_cast<int>(vocab_size));
    }
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        set_cublas_error("cuBLAS failed for matvec argmax logits", cublas_status);
        return 1;
    }

    constexpr int block_size = 256;
    argmax_rows_kernel<<<linear_grid_size(batch_size, 1), block_size>>>(
        static_cast<float*>(logits_tmp.ptr),
        static_cast<size_t*>(device_out_tmp.ptr),
        batch_size,
        vocab_size);
    if (!sync_cuda("CUDA matvec argmax kernel failed")) {
        return 1;
    }

    cudaError_t status = cudaMemcpy(
        out_indices,
        static_cast<size_t*>(device_out_tmp.ptr),
        batch_size * sizeof(size_t),
        cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        set_cuda_error("failed to download CUDA matvec argmax output", status);
        return 1;
    }
    return 0;
}

extern "C" int lumen_cuda_matmul_f32_device(
    uint64_t a_handle,
    uint64_t b_handle,
    uint64_t out_handle,
    size_t m,
    size_t n,
    size_t k) {
    if (!validate_handle(a_handle, "CUDA matmul A handle") ||
        !validate_handle(b_handle, "CUDA matmul B handle") ||
        !validate_handle(out_handle, "CUDA matmul output handle")) {
        return 1;
    }
    if (!validate_dims(m, n, k)) {
        return 1;
    }

    CublasHandle handle;
    if (!init_cublas(handle)) {
        return 1;
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasStatus_t cublas_status = cublasSgemm(
        handle.handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        static_cast<int>(n),
        static_cast<int>(m),
        static_cast<int>(k),
        &alpha,
        handle_to_ptr(b_handle),
        static_cast<int>(k),
        handle_to_ptr(a_handle),
        static_cast<int>(k),
        &beta,
        handle_to_ptr(out_handle),
        static_cast<int>(n));
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        set_cublas_error("cuBLAS SGEMM failed for matmul", cublas_status);
        return 1;
    }
    return 0;
}

extern "C" int lumen_cuda_matmul_bf16_host_device(
    const uint16_t* a_host,
    const uint16_t* b_host,
    uint64_t out_handle,
    size_t m,
    size_t n,
    size_t k) {
    if (!validate_handle(out_handle, "CUDA BF16 matmul output handle") || !validate_dims(m, n, k)) {
        return 1;
    }

    ScopedDeviceInput<__nv_bfloat16> a_dev;
    ScopedDeviceInput<__nv_bfloat16> b_dev;
    if (!upload_typed_input(reinterpret_cast<const __nv_bfloat16*>(a_host), m * k, &a_dev.ptr, "upload BF16 matmul A") ||
        !upload_typed_input(reinterpret_cast<const __nv_bfloat16*>(b_host), n * k, &b_dev.ptr, "upload BF16 matmul B")) {
        return 1;
    }

    CublasHandle handle;
    if (!init_cublas(handle)) {
        return 1;
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasStatus_t cublas_status = cublasGemmEx(
        handle.handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        static_cast<int>(n),
        static_cast<int>(m),
        static_cast<int>(k),
        &alpha,
        b_dev.ptr,
        CUDA_R_16BF,
        static_cast<int>(k),
        a_dev.ptr,
        CUDA_R_16BF,
        static_cast<int>(k),
        &beta,
        handle_to_ptr(out_handle),
        CUDA_R_32F,
        static_cast<int>(n),
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        set_cublas_error("cuBLAS BF16 GEMM failed for matmul", cublas_status);
        return 1;
    }
    return 0;
}

extern "C" int lumen_cuda_matmul_bf16_device(
    uint64_t a_handle,
    uint64_t b_handle,
    uint64_t out_handle,
    size_t m,
    size_t n,
    size_t k) {
    if (!validate_handle(a_handle, "CUDA BF16 matmul A handle") ||
        !validate_handle(b_handle, "CUDA BF16 matmul B handle") ||
        !validate_handle(out_handle, "CUDA BF16 matmul output handle") ||
        !validate_dims(m, n, k)) {
        return 1;
    }

    CublasHandle handle;
    if (!init_cublas(handle)) {
        return 1;
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasStatus_t cublas_status = cublasGemmEx(
        handle.handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        static_cast<int>(n),
        static_cast<int>(m),
        static_cast<int>(k),
        &alpha,
        reinterpret_cast<const __nv_bfloat16*>(handle_to_ptr(b_handle)),
        CUDA_R_16BF,
        static_cast<int>(k),
        reinterpret_cast<const __nv_bfloat16*>(handle_to_ptr(a_handle)),
        CUDA_R_16BF,
        static_cast<int>(k),
        &beta,
        handle_to_ptr(out_handle),
        CUDA_R_32F,
        static_cast<int>(n),
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        set_cublas_error("cuBLAS BF16 resident GEMM failed for matmul", cublas_status);
        return 1;
    }
    return 0;
}

extern "C" int lumen_cuda_matmul_bf16_typed_out_device(
    uint64_t a_handle,
    uint64_t b_handle,
    uint64_t out_handle,
    size_t m,
    size_t n,
    size_t k) {
    if (!validate_handle(a_handle, "CUDA BF16 typed-out matmul A handle") ||
        !validate_handle(b_handle, "CUDA BF16 typed-out matmul B handle") ||
        !validate_handle(out_handle, "CUDA BF16 typed-out matmul output handle") ||
        !validate_dims(m, n, k)) {
        return 1;
    }

    CublasHandle handle;
    if (!init_cublas(handle)) {
        return 1;
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasStatus_t cublas_status = cublasGemmEx(
        handle.handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        static_cast<int>(n),
        static_cast<int>(m),
        static_cast<int>(k),
        &alpha,
        reinterpret_cast<const __nv_bfloat16*>(handle_to_ptr(b_handle)),
        CUDA_R_16BF,
        static_cast<int>(k),
        reinterpret_cast<const __nv_bfloat16*>(handle_to_ptr(a_handle)),
        CUDA_R_16BF,
        static_cast<int>(k),
        &beta,
        reinterpret_cast<__nv_bfloat16*>(handle_to_ptr(out_handle)),
        CUDA_R_16BF,
        static_cast<int>(n),
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        set_cublas_error("cuBLAS BF16 typed-output GEMM failed for matmul", cublas_status);
        return 1;
    }
    return 0;
}

extern "C" int lumen_cuda_matmul_f16_host_device(
    const uint16_t* a_host,
    const uint16_t* b_host,
    uint64_t out_handle,
    size_t m,
    size_t n,
    size_t k) {
    if (!validate_handle(out_handle, "CUDA F16 matmul output handle") || !validate_dims(m, n, k)) {
        return 1;
    }

    ScopedDeviceInput<__half> a_dev;
    ScopedDeviceInput<__half> b_dev;
    if (!upload_typed_input(reinterpret_cast<const __half*>(a_host), m * k, &a_dev.ptr, "upload F16 matmul A") ||
        !upload_typed_input(reinterpret_cast<const __half*>(b_host), n * k, &b_dev.ptr, "upload F16 matmul B")) {
        return 1;
    }

    CublasHandle handle;
    if (!init_cublas(handle)) {
        return 1;
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasStatus_t cublas_status = cublasGemmEx(
        handle.handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        static_cast<int>(n),
        static_cast<int>(m),
        static_cast<int>(k),
        &alpha,
        b_dev.ptr,
        CUDA_R_16F,
        static_cast<int>(k),
        a_dev.ptr,
        CUDA_R_16F,
        static_cast<int>(k),
        &beta,
        handle_to_ptr(out_handle),
        CUDA_R_32F,
        static_cast<int>(n),
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        set_cublas_error("cuBLAS F16 GEMM failed for matmul", cublas_status);
        return 1;
    }
    return 0;
}

extern "C" int lumen_cuda_matmul_f16_device(
    uint64_t a_handle,
    uint64_t b_handle,
    uint64_t out_handle,
    size_t m,
    size_t n,
    size_t k) {
    if (!validate_handle(a_handle, "CUDA F16 matmul A handle") ||
        !validate_handle(b_handle, "CUDA F16 matmul B handle") ||
        !validate_handle(out_handle, "CUDA F16 matmul output handle") ||
        !validate_dims(m, n, k)) {
        return 1;
    }

    CublasHandle handle;
    if (!init_cublas(handle)) {
        return 1;
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasStatus_t cublas_status = cublasGemmEx(
        handle.handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        static_cast<int>(n),
        static_cast<int>(m),
        static_cast<int>(k),
        &alpha,
        reinterpret_cast<const __half*>(handle_to_ptr(b_handle)),
        CUDA_R_16F,
        static_cast<int>(k),
        reinterpret_cast<const __half*>(handle_to_ptr(a_handle)),
        CUDA_R_16F,
        static_cast<int>(k),
        &beta,
        handle_to_ptr(out_handle),
        CUDA_R_32F,
        static_cast<int>(n),
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        set_cublas_error("cuBLAS F16 resident GEMM failed for matmul", cublas_status);
        return 1;
    }
    return 0;
}

extern "C" int lumen_cuda_matmul_f16_typed_out_device(
    uint64_t a_handle,
    uint64_t b_handle,
    uint64_t out_handle,
    size_t m,
    size_t n,
    size_t k) {
    if (!validate_handle(a_handle, "CUDA F16 typed-out matmul A handle") ||
        !validate_handle(b_handle, "CUDA F16 typed-out matmul B handle") ||
        !validate_handle(out_handle, "CUDA F16 typed-out matmul output handle") ||
        !validate_dims(m, n, k)) {
        return 1;
    }

    CublasHandle handle;
    if (!init_cublas(handle)) {
        return 1;
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasStatus_t cublas_status = cublasGemmEx(
        handle.handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        static_cast<int>(n),
        static_cast<int>(m),
        static_cast<int>(k),
        &alpha,
        reinterpret_cast<const __half*>(handle_to_ptr(b_handle)),
        CUDA_R_16F,
        static_cast<int>(k),
        reinterpret_cast<const __half*>(handle_to_ptr(a_handle)),
        CUDA_R_16F,
        static_cast<int>(k),
        &beta,
        reinterpret_cast<__half*>(handle_to_ptr(out_handle)),
        CUDA_R_16F,
        static_cast<int>(n),
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        set_cublas_error("cuBLAS F16 typed-output GEMM failed for matmul", cublas_status);
        return 1;
    }
    return 0;
}

__global__ void matmul_i8_kernel(
    const int8_t* a,
    const int8_t* b,
    float* out,
    size_t total,
    size_t m,
    size_t n,
    size_t k,
    float scale) {
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t idx = start; idx < total; idx += stride) {
        size_t row = idx / n;
        size_t col = idx - row * n;
        int acc = 0;
        for (size_t kk = 0; kk < k; ++kk) {
            acc += static_cast<int>(a[row * k + kk]) * static_cast<int>(b[col * k + kk]);
        }
        out[idx] = static_cast<float>(acc) * scale;
    }
}
static bool can_launch_matmul_tiled(size_t m, size_t n) {
    constexpr size_t max_grid_x = 2147483647;
    constexpr size_t max_grid_y = 65535;
    constexpr size_t tile_m = 8;
    constexpr size_t tile_n = 16;
    return m <= max_grid_y * tile_m && n <= max_grid_x * tile_n;
}

__global__ void matmul_i8_tiled_kernel(
    const int8_t* a,
    const int8_t* b,
    float* out,
    size_t m,
    size_t n,
    size_t k,
    float scale) {
    constexpr int tile_m = 8;
    constexpr int tile_n = 16;
    constexpr int tile_k = 32;
    __shared__ int8_t a_tile[tile_m][tile_k];
    __shared__ int8_t b_tile[tile_n][tile_k];

    const size_t row = blockIdx.y * tile_m + threadIdx.y;
    const size_t col = blockIdx.x * tile_n + threadIdx.x;
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int block_threads = blockDim.x * blockDim.y;
    int acc = 0;

    for (size_t kt = 0; kt < k; kt += tile_k) {
        for (int idx = tid; idx < tile_m * tile_k; idx += block_threads) {
            const int local_row = idx / tile_k;
            const int local_k = idx - local_row * tile_k;
            const size_t global_row = blockIdx.y * tile_m + local_row;
            const size_t global_k = kt + local_k;
            a_tile[local_row][local_k] =
                (global_row < m && global_k < k) ? a[global_row * k + global_k] : 0;
        }
        for (int idx = tid; idx < tile_n * tile_k; idx += block_threads) {
            const int local_col = idx / tile_k;
            const int local_k = idx - local_col * tile_k;
            const size_t global_col = blockIdx.x * tile_n + local_col;
            const size_t global_k = kt + local_k;
            b_tile[local_col][local_k] =
                (global_col < n && global_k < k) ? b[global_col * k + global_k] : 0;
        }
        __syncthreads();

        if (row < m && col < n) {
            for (int local_k = 0; local_k < tile_k; ++local_k) {
                acc += static_cast<int>(a_tile[threadIdx.y][local_k]) *
                       static_cast<int>(b_tile[threadIdx.x][local_k]);
            }
        }
        __syncthreads();
    }

    if (row < m && col < n) {
        out[row * n + col] = static_cast<float>(acc) * scale;
    }
}

static bool launch_matmul_i8_compute(
    const int8_t* a,
    const int8_t* b,
    float* out,
    size_t m,
    size_t n,
    size_t k,
    float scale,
    const char* error_message) {
    const size_t max_size = static_cast<size_t>(-1);
    if (m > max_size / n) {
        set_error("CUDA I8 matmul output length overflow");
        return false;
    }
    if (can_launch_matmul_tiled(m, n)) {
        dim3 block(16, 8);
        dim3 grid(
            static_cast<unsigned int>((n + 15) / 16),
            static_cast<unsigned int>((m + 7) / 8));
        matmul_i8_tiled_kernel<<<grid, block>>>(a, b, out, m, n, k, scale);
    } else {
        const size_t total = m * n;
        constexpr unsigned int block_size = 256;
        const unsigned int grid = linear_grid_size(total, block_size);
        matmul_i8_kernel<<<grid, block_size>>>(a, b, out, total, m, n, k, scale);
    }
    return check_cuda_launch(error_message);
}

__device__ inline float load_floatlike_value(const float* ptr, size_t idx) {
    return ptr[idx];
}

__device__ inline float load_floatlike_value(const __half* ptr, size_t idx) {
    return __half2float(ptr[idx]);
}

__device__ inline float load_floatlike_value(const __nv_bfloat16* ptr, size_t idx) {
    return __bfloat162float(ptr[idx]);
}

template <typename LhsT>
__global__ void matmul_floatlike_i8_kernel(
    const LhsT* a,
    const int8_t* b,
    float* out,
    size_t total,
    size_t m,
    size_t n,
    size_t k,
    float b_scale) {
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t idx = start; idx < total; idx += stride) {
        size_t row = idx / n;
        size_t col = idx - row * n;
        float acc = 0.0f;
        for (size_t kk = 0; kk < k; ++kk) {
            const float lhs = load_floatlike_value(a, row * k + kk);
            const float rhs = static_cast<float>(b[col * k + kk]) * b_scale;
            acc += lhs * rhs;
        }
        out[idx] = acc;
    }
}

template <typename RhsT>
__global__ void matmul_i8_floatlike_kernel(
    const int8_t* a,
    const RhsT* b,
    float* out,
    size_t total,
    size_t m,
    size_t n,
    size_t k,
    float a_scale) {
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t idx = start; idx < total; idx += stride) {
        size_t row = idx / n;
        size_t col = idx - row * n;
        float acc = 0.0f;
        for (size_t kk = 0; kk < k; ++kk) {
            const float lhs = static_cast<float>(a[row * k + kk]) * a_scale;
            const float rhs = load_floatlike_value(b, col * k + kk);
            acc += lhs * rhs;
        }
        out[idx] = acc;
    }
}

__global__ void matvec_bf16_i8_kernel(
    const __nv_bfloat16* a,
    const int8_t* b,
    float* out,
    size_t n,
    size_t k,
    float b_scale) {
    __shared__ float partials[256];
    size_t tid = threadIdx.x;
    for (size_t col = blockIdx.x; col < n; col += gridDim.x) {
        float acc = 0.0f;
        for (size_t kk = tid; kk < k; kk += blockDim.x) {
            const float lhs = __bfloat162float(a[kk]);
            const float rhs = static_cast<float>(b[col * k + kk]) * b_scale;
            acc += lhs * rhs;
        }
        partials[tid] = acc;
        __syncthreads();

        for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                partials[tid] += partials[tid + stride];
            }
            __syncthreads();
        }

        if (tid == 0) {
            out[col] = partials[0];
        }
        __syncthreads();
    }
}

__global__ void matvec_f16_i8_kernel(
    const __half* a,
    const int8_t* b,
    float* out,
    size_t n,
    size_t k,
    float b_scale) {
    __shared__ float partials[256];
    size_t tid = threadIdx.x;
    for (size_t col = blockIdx.x; col < n; col += gridDim.x) {
        float acc = 0.0f;
        for (size_t kk = tid; kk < k; kk += blockDim.x) {
            const float lhs = __half2float(a[kk]);
            const float rhs = static_cast<float>(b[col * k + kk]) * b_scale;
            acc += lhs * rhs;
        }
        partials[tid] = acc;
        __syncthreads();

        for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                partials[tid] += partials[tid + stride];
            }
            __syncthreads();
        }

        if (tid == 0) {
            out[col] = partials[0];
        }
        __syncthreads();
    }
}

__global__ void matvec_f32_i8_kernel(
    const float* a,
    const int8_t* b,
    float* out,
    size_t n,
    size_t k,
    float b_scale) {
    __shared__ float partials[256];
    size_t tid = threadIdx.x;
    for (size_t col = blockIdx.x; col < n; col += gridDim.x) {
        float acc = 0.0f;
        for (size_t kk = tid; kk < k; kk += blockDim.x) {
            const float lhs = a[kk];
            const float rhs = static_cast<float>(b[col * k + kk]) * b_scale;
            acc += lhs * rhs;
        }
        partials[tid] = acc;
        __syncthreads();

        for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                partials[tid] += partials[tid + stride];
            }
            __syncthreads();
        }

        if (tid == 0) {
            out[col] = partials[0];
        }
        __syncthreads();
    }
}

template <typename RhsT>
__global__ void matvec_i8_floatlike_kernel(
    const int8_t* a,
    const RhsT* b,
    float* out,
    size_t n,
    size_t k,
    float a_scale) {
    __shared__ float partials[256];
    size_t tid = threadIdx.x;
    for (size_t col = blockIdx.x; col < n; col += gridDim.x) {
        float acc = 0.0f;
        for (size_t kk = tid; kk < k; kk += blockDim.x) {
            const float lhs = static_cast<float>(a[kk]) * a_scale;
            const float rhs = load_floatlike_value(b, col * k + kk);
            acc += lhs * rhs;
        }
        partials[tid] = acc;
        __syncthreads();

        for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                partials[tid] += partials[tid + stride];
            }
            __syncthreads();
        }

        if (tid == 0) {
            out[col] = partials[0];
        }
        __syncthreads();
    }
}

__global__ void matvec_i8_i8_kernel(
    const int8_t* a,
    const int8_t* b,
    float* out,
    size_t n,
    size_t k,
    float scale) {
    __shared__ int partials[256];
    size_t tid = threadIdx.x;
    for (size_t col = blockIdx.x; col < n; col += gridDim.x) {
        int acc = 0;
        for (size_t kk = tid; kk < k; kk += blockDim.x) {
            acc += static_cast<int>(a[kk]) * static_cast<int>(b[col * k + kk]);
        }
        partials[tid] = acc;
        __syncthreads();

        for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                partials[tid] += partials[tid + stride];
            }
            __syncthreads();
        }

        if (tid == 0) {
            out[col] = static_cast<float>(partials[0]) * scale;
        }
        __syncthreads();
    }
}

__global__ void argmax_pairs_kernel(
    const float* values,
    const size_t* indices,
    size_t* out_indices,
    size_t count) {
    constexpr int block_size = 256;
    __shared__ float best_values[block_size];
    __shared__ size_t best_indices[block_size];

    float best_value = -FLT_MAX;
    size_t best_index = 0;
    for (size_t i = threadIdx.x; i < count; i += blockDim.x) {
        const float value = values[i];
        const size_t index = indices[i];
        if (value > best_value || (value == best_value && index < best_index)) {
            best_value = value;
            best_index = index;
        }
    }

    best_values[threadIdx.x] = best_value;
    best_indices[threadIdx.x] = best_index;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            const float other_value = best_values[threadIdx.x + stride];
            const size_t other_index = best_indices[threadIdx.x + stride];
            if (other_value > best_values[threadIdx.x] ||
                (other_value == best_values[threadIdx.x] &&
                 other_index < best_indices[threadIdx.x])) {
                best_values[threadIdx.x] = other_value;
                best_indices[threadIdx.x] = other_index;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        out_indices[0] = best_indices[0];
    }
}

__global__ void matvec_argmax_bf16_i8_block_best_kernel(
    const __nv_bfloat16* a,
    const int8_t* b,
    float* best_values,
    size_t* best_indices,
    size_t n,
    size_t k,
    float b_scale) {
    __shared__ float partials[256];

    const size_t tid = threadIdx.x;
    float best_value = -FLT_MAX;
    size_t best_index = 0;
    for (size_t col = blockIdx.x; col < n; col += gridDim.x) {
        float acc = 0.0f;
        for (size_t kk = tid; kk < k; kk += blockDim.x) {
            const float lhs = __bfloat162float(a[kk]);
            const float rhs = static_cast<float>(b[col * k + kk]) * b_scale;
            acc += lhs * rhs;
        }

        partials[tid] = acc;
        __syncthreads();
        for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                partials[tid] += partials[tid + stride];
            }
            __syncthreads();
        }

        if (tid == 0) {
            const float value = partials[0];
            if (value > best_value || (value == best_value && col < best_index)) {
                best_value = value;
                best_index = col;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        best_values[blockIdx.x] = best_value;
        best_indices[blockIdx.x] = best_index;
    }
}

__global__ void matvec_argmax_f16_i8_block_best_kernel(
    const __half* a,
    const int8_t* b,
    float* best_values,
    size_t* best_indices,
    size_t n,
    size_t k,
    float b_scale) {
    __shared__ float partials[256];

    const size_t tid = threadIdx.x;
    float best_value = -FLT_MAX;
    size_t best_index = 0;
    for (size_t col = blockIdx.x; col < n; col += gridDim.x) {
        float acc = 0.0f;
        for (size_t kk = tid; kk < k; kk += blockDim.x) {
            const float lhs = __half2float(a[kk]);
            const float rhs = static_cast<float>(b[col * k + kk]) * b_scale;
            acc += lhs * rhs;
        }

        partials[tid] = acc;
        __syncthreads();
        for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                partials[tid] += partials[tid + stride];
            }
            __syncthreads();
        }

        if (tid == 0) {
            const float value = partials[0];
            if (value > best_value || (value == best_value && col < best_index)) {
                best_value = value;
                best_index = col;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        best_values[blockIdx.x] = best_value;
        best_indices[blockIdx.x] = best_index;
    }
}

__global__ void matvec_argmax_f32_i8_block_best_kernel(
    const float* a,
    const int8_t* b,
    float* best_values,
    size_t* best_indices,
    size_t n,
    size_t k,
    float b_scale) {
    __shared__ float partials[256];

    const size_t tid = threadIdx.x;
    float best_value = -FLT_MAX;
    size_t best_index = 0;
    for (size_t col = blockIdx.x; col < n; col += gridDim.x) {
        float acc = 0.0f;
        for (size_t kk = tid; kk < k; kk += blockDim.x) {
            const float lhs = a[kk];
            const float rhs = static_cast<float>(b[col * k + kk]) * b_scale;
            acc += lhs * rhs;
        }

        partials[tid] = acc;
        __syncthreads();
        for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                partials[tid] += partials[tid + stride];
            }
            __syncthreads();
        }

        if (tid == 0) {
            const float value = partials[0];
            if (value > best_value || (value == best_value && col < best_index)) {
                best_value = value;
                best_index = col;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        best_values[blockIdx.x] = best_value;
        best_indices[blockIdx.x] = best_index;
    }
}

__global__ void matvec_argmax_i8_i8_block_best_kernel(
    const int8_t* a,
    const int8_t* b,
    float* best_values,
    size_t* best_indices,
    size_t n,
    size_t k,
    float scale) {
    __shared__ int partials[256];

    const size_t tid = threadIdx.x;
    int best_acc = INT_MIN;
    size_t best_index = 0;
    for (size_t col = blockIdx.x; col < n; col += gridDim.x) {
        int acc = 0;
        for (size_t kk = tid; kk < k; kk += blockDim.x) {
            acc += static_cast<int>(a[kk]) * static_cast<int>(b[col * k + kk]);
        }

        partials[tid] = acc;
        __syncthreads();
        for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                partials[tid] += partials[tid + stride];
            }
            __syncthreads();
        }

        if (tid == 0) {
            const int value = partials[0];
            if (value > best_acc || (value == best_acc && col < best_index)) {
                best_acc = value;
                best_index = col;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        best_values[blockIdx.x] = static_cast<float>(best_acc) * scale;
        best_indices[blockIdx.x] = best_index;
    }
}

template <typename LhsT>
__global__ void matmul_floatlike_i8_tiled_kernel(
    const LhsT* a,
    const int8_t* b,
    float* out,
    size_t m,
    size_t n,
    size_t k,
    float b_scale) {
    constexpr int tile_m = 8;
    constexpr int tile_n = 16;
    constexpr int tile_k = 32;
    __shared__ float a_tile[tile_m][tile_k];
    __shared__ float b_tile[tile_n][tile_k];

    size_t row = blockIdx.y * tile_m + threadIdx.y;
    size_t col = blockIdx.x * tile_n + threadIdx.x;
    float acc = 0.0f;
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int block_threads = blockDim.x * blockDim.y;

    for (size_t kt = 0; kt < k; kt += tile_k) {
        for (int idx = tid; idx < tile_m * tile_k; idx += block_threads) {
            const int local_row = idx / tile_k;
            const int local_k = idx - local_row * tile_k;
            const size_t global_row = blockIdx.y * tile_m + local_row;
            const size_t global_k = kt + local_k;
            a_tile[local_row][local_k] =
                (global_row < m && global_k < k)
                    ? load_floatlike_value(a, global_row * k + global_k)
                    : 0.0f;
        }
        for (int idx = tid; idx < tile_n * tile_k; idx += block_threads) {
            const int local_col = idx / tile_k;
            const int local_k = idx - local_col * tile_k;
            const size_t global_col = blockIdx.x * tile_n + local_col;
            const size_t global_k = kt + local_k;
            b_tile[local_col][local_k] =
                (global_col < n && global_k < k)
                    ? static_cast<float>(b[global_col * k + global_k]) * b_scale
                    : 0.0f;
        }
        __syncthreads();

        if (row < m && col < n) {
            for (int local_k = 0; local_k < tile_k; ++local_k) {
                acc += a_tile[threadIdx.y][local_k] * b_tile[threadIdx.x][local_k];
            }
        }
        __syncthreads();
    }

    if (row < m && col < n) {
        out[row * n + col] = acc;
    }
}

template <typename RhsT>
__global__ void matmul_i8_floatlike_tiled_kernel(
    const int8_t* a,
    const RhsT* b,
    float* out,
    size_t m,
    size_t n,
    size_t k,
    float a_scale) {
    constexpr int tile_m = 8;
    constexpr int tile_n = 16;
    constexpr int tile_k = 32;
    __shared__ float a_tile[tile_m][tile_k];
    __shared__ float b_tile[tile_n][tile_k];

    size_t row = blockIdx.y * tile_m + threadIdx.y;
    size_t col = blockIdx.x * tile_n + threadIdx.x;
    float acc = 0.0f;
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int block_threads = blockDim.x * blockDim.y;

    for (size_t kt = 0; kt < k; kt += tile_k) {
        for (int idx = tid; idx < tile_m * tile_k; idx += block_threads) {
            const int local_row = idx / tile_k;
            const int local_k = idx - local_row * tile_k;
            const size_t global_row = blockIdx.y * tile_m + local_row;
            const size_t global_k = kt + local_k;
            a_tile[local_row][local_k] =
                (global_row < m && global_k < k)
                    ? static_cast<float>(a[global_row * k + global_k]) * a_scale
                    : 0.0f;
        }
        for (int idx = tid; idx < tile_n * tile_k; idx += block_threads) {
            const int local_col = idx / tile_k;
            const int local_k = idx - local_col * tile_k;
            const size_t global_col = blockIdx.x * tile_n + local_col;
            const size_t global_k = kt + local_k;
            b_tile[local_col][local_k] =
                (global_col < n && global_k < k)
                    ? load_floatlike_value(b, global_col * k + global_k)
                    : 0.0f;
        }
        __syncthreads();

        if (row < m && col < n) {
            for (int local_k = 0; local_k < tile_k; ++local_k) {
                acc += a_tile[threadIdx.y][local_k] * b_tile[threadIdx.x][local_k];
            }
        }
        __syncthreads();
    }

    if (row < m && col < n) {
        out[row * n + col] = acc;
    }
}

template <typename LhsT>
static bool launch_matmul_floatlike_i8_compute(
    const LhsT* a,
    const int8_t* b,
    float* out,
    size_t m,
    size_t n,
    size_t k,
    float b_scale,
    const char* error_message) {
    if (can_launch_matmul_tiled(m, n)) {
        dim3 block(16, 8);
        dim3 grid(
            static_cast<unsigned int>((n + 15) / 16),
            static_cast<unsigned int>((m + 7) / 8));
        matmul_floatlike_i8_tiled_kernel<<<grid, block>>>(a, b, out, m, n, k, b_scale);
    } else {
        const size_t max_size = static_cast<size_t>(-1);
        if (m > max_size / n) {
            set_error("CUDA float-likexI8 matmul output length overflow");
            return false;
        }
        const size_t total = m * n;
        constexpr unsigned int block_size = 256;
        matmul_floatlike_i8_kernel<<<linear_grid_size(total, block_size), block_size>>>(
            a, b, out, total, m, n, k, b_scale);
    }
    return check_cuda_launch(error_message);
}

template <typename RhsT>
static bool launch_matmul_i8_floatlike_compute(
    const int8_t* a,
    const RhsT* b,
    float* out,
    size_t m,
    size_t n,
    size_t k,
    float a_scale,
    const char* error_message) {
    if (can_launch_matmul_tiled(m, n)) {
        dim3 block(16, 8);
        dim3 grid(
            static_cast<unsigned int>((n + 15) / 16),
            static_cast<unsigned int>((m + 7) / 8));
        matmul_i8_floatlike_tiled_kernel<<<grid, block>>>(a, b, out, m, n, k, a_scale);
    } else {
        const size_t max_size = static_cast<size_t>(-1);
        if (m > max_size / n) {
            set_error("CUDA I8xfloat-like matmul output length overflow");
            return false;
        }
        const size_t total = m * n;
        constexpr unsigned int block_size = 256;
        matmul_i8_floatlike_kernel<<<linear_grid_size(total, block_size), block_size>>>(
            a, b, out, total, m, n, k, a_scale);
    }
    return check_cuda_launch(error_message);
}

extern "C" int lumen_cuda_matvec_argmax_bf16_i8_device(
    uint64_t input_handle,
    uint64_t weight_handle,
    float weight_scale,
    size_t* out_indices,
    size_t batch_size,
    size_t vocab_size,
    size_t hidden_size) {
    if (!validate_handle(input_handle, "CUDA BF16xI8 argmax input handle") ||
        !validate_handle(weight_handle, "CUDA BF16xI8 argmax weight handle")) {
        return 1;
    }
    if (out_indices == nullptr) {
        set_error("CUDA BF16xI8 argmax output is null");
        return 1;
    }
    if (batch_size == 0 || vocab_size == 0 || hidden_size == 0) {
        set_error("CUDA BF16xI8 argmax dimensions must be greater than zero");
        return 1;
    }
    if (!std::isfinite(weight_scale) || weight_scale <= 0.0f) {
        set_error("CUDA BF16xI8 argmax weight scale must be finite and > 0");
        return 1;
    }

    const size_t max_size = static_cast<size_t>(-1);
    if (batch_size > max_size / vocab_size ||
        batch_size * vocab_size > max_size / sizeof(float)) {
        set_error("CUDA BF16xI8 argmax logits length overflow");
        return 1;
    }
    if (batch_size > max_size / sizeof(size_t)) {
        set_error("CUDA BF16xI8 argmax output length overflow");
        return 1;
    }

    thread_local ReusableCudaWorkspace device_out_tmp;
    if (!device_out_tmp.ensure(
            batch_size * sizeof(size_t),
            "failed to allocate CUDA BF16xI8 argmax output")) {
        return 1;
    }

    constexpr int block_size = 256;
    if (batch_size == 1 && vocab_size <= 4096) {
        const size_t partial_count = vocab_size < 1024 ? vocab_size : 1024;
        thread_local ReusableCudaWorkspace best_values_tmp;
        if (!best_values_tmp.ensure(
                partial_count * sizeof(float),
                "failed to allocate CUDA BF16xI8 argmax partial values")) {
            return 1;
        }
        thread_local ReusableCudaWorkspace best_indices_tmp;
        if (!best_indices_tmp.ensure(
                partial_count * sizeof(size_t),
                "failed to allocate CUDA BF16xI8 argmax partial indices")) {
            return 1;
        }
        matvec_argmax_bf16_i8_block_best_kernel<<<
            static_cast<unsigned int>(partial_count),
            block_size>>>(
            reinterpret_cast<const __nv_bfloat16*>(handle_to_ptr(input_handle)),
            reinterpret_cast<const int8_t*>(handle_to_ptr(weight_handle)),
            static_cast<float*>(best_values_tmp.ptr),
            static_cast<size_t*>(best_indices_tmp.ptr),
            vocab_size,
            hidden_size,
            weight_scale);
        argmax_pairs_kernel<<<1, block_size>>>(
            static_cast<float*>(best_values_tmp.ptr),
            static_cast<size_t*>(best_indices_tmp.ptr),
            static_cast<size_t*>(device_out_tmp.ptr),
            partial_count);
    } else {
        thread_local ReusableCudaWorkspace logits_tmp;
        if (!logits_tmp.ensure(
                batch_size * vocab_size * sizeof(float),
                "failed to allocate CUDA BF16xI8 argmax logits")) {
            return 1;
        }
        if (batch_size == 1) {
            matvec_bf16_i8_kernel<<<linear_grid_size(vocab_size, 1), block_size>>>(
                reinterpret_cast<const __nv_bfloat16*>(handle_to_ptr(input_handle)),
                reinterpret_cast<const int8_t*>(handle_to_ptr(weight_handle)),
                static_cast<float*>(logits_tmp.ptr),
                vocab_size,
                hidden_size,
                weight_scale);
        } else {
            dim3 block(16, 8);
            dim3 grid(
                static_cast<unsigned int>((vocab_size + 15) / 16),
                static_cast<unsigned int>((batch_size + 7) / 8));
            matmul_floatlike_i8_tiled_kernel<<<grid, block>>>(
                reinterpret_cast<const __nv_bfloat16*>(handle_to_ptr(input_handle)),
                reinterpret_cast<const int8_t*>(handle_to_ptr(weight_handle)),
                static_cast<float*>(logits_tmp.ptr),
                batch_size,
                vocab_size,
                hidden_size,
                weight_scale);
        }
        argmax_rows_kernel<<<linear_grid_size(batch_size, 1), block_size>>>(
            static_cast<float*>(logits_tmp.ptr),
            static_cast<size_t*>(device_out_tmp.ptr),
            batch_size,
            vocab_size);
    }
    if (!sync_cuda("CUDA BF16xI8 matvec argmax kernel failed")) {
        return 1;
    }

    cudaError_t status = cudaMemcpy(
        out_indices,
        static_cast<size_t*>(device_out_tmp.ptr),
        batch_size * sizeof(size_t),
        cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        set_cuda_error("failed to download CUDA BF16xI8 argmax output", status);
        return 1;
    }
    return 0;
}

extern "C" int lumen_cuda_matvec_argmax_f16_i8_device(
    uint64_t input_handle,
    uint64_t weight_handle,
    float weight_scale,
    size_t* out_indices,
    size_t batch_size,
    size_t vocab_size,
    size_t hidden_size) {
    if (!validate_handle(input_handle, "CUDA F16xI8 argmax input handle") ||
        !validate_handle(weight_handle, "CUDA F16xI8 argmax weight handle")) {
        return 1;
    }
    if (out_indices == nullptr) {
        set_error("CUDA F16xI8 argmax output is null");
        return 1;
    }
    if (batch_size == 0 || vocab_size == 0 || hidden_size == 0) {
        set_error("CUDA F16xI8 argmax dimensions must be greater than zero");
        return 1;
    }
    if (!std::isfinite(weight_scale) || weight_scale <= 0.0f) {
        set_error("CUDA F16xI8 argmax weight scale must be finite and > 0");
        return 1;
    }

    const size_t max_size = static_cast<size_t>(-1);
    if (batch_size > max_size / vocab_size ||
        batch_size * vocab_size > max_size / sizeof(float)) {
        set_error("CUDA F16xI8 argmax logits length overflow");
        return 1;
    }
    if (batch_size > max_size / sizeof(size_t)) {
        set_error("CUDA F16xI8 argmax output length overflow");
        return 1;
    }

    thread_local ReusableCudaWorkspace device_out_tmp;
    if (!device_out_tmp.ensure(
            batch_size * sizeof(size_t),
            "failed to allocate CUDA F16xI8 argmax output")) {
        return 1;
    }

    constexpr int block_size = 256;
    if (batch_size == 1 && vocab_size <= 4096) {
        const size_t partial_count = vocab_size < 1024 ? vocab_size : 1024;
        thread_local ReusableCudaWorkspace best_values_tmp;
        if (!best_values_tmp.ensure(
                partial_count * sizeof(float),
                "failed to allocate CUDA F16xI8 argmax partial values")) {
            return 1;
        }
        thread_local ReusableCudaWorkspace best_indices_tmp;
        if (!best_indices_tmp.ensure(
                partial_count * sizeof(size_t),
                "failed to allocate CUDA F16xI8 argmax partial indices")) {
            return 1;
        }
        matvec_argmax_f16_i8_block_best_kernel<<<
            static_cast<unsigned int>(partial_count),
            block_size>>>(
            reinterpret_cast<const __half*>(handle_to_ptr(input_handle)),
            reinterpret_cast<const int8_t*>(handle_to_ptr(weight_handle)),
            static_cast<float*>(best_values_tmp.ptr),
            static_cast<size_t*>(best_indices_tmp.ptr),
            vocab_size,
            hidden_size,
            weight_scale);
        argmax_pairs_kernel<<<1, block_size>>>(
            static_cast<float*>(best_values_tmp.ptr),
            static_cast<size_t*>(best_indices_tmp.ptr),
            static_cast<size_t*>(device_out_tmp.ptr),
            partial_count);
    } else {
        thread_local ReusableCudaWorkspace logits_tmp;
        if (!logits_tmp.ensure(
                batch_size * vocab_size * sizeof(float),
                "failed to allocate CUDA F16xI8 argmax logits")) {
            return 1;
        }
        if (batch_size == 1) {
            matvec_f16_i8_kernel<<<linear_grid_size(vocab_size, 1), block_size>>>(
                reinterpret_cast<const __half*>(handle_to_ptr(input_handle)),
                reinterpret_cast<const int8_t*>(handle_to_ptr(weight_handle)),
                static_cast<float*>(logits_tmp.ptr),
                vocab_size,
                hidden_size,
                weight_scale);
        } else {
            dim3 block(16, 8);
            dim3 grid(
                static_cast<unsigned int>((vocab_size + 15) / 16),
                static_cast<unsigned int>((batch_size + 7) / 8));
            matmul_floatlike_i8_tiled_kernel<<<grid, block>>>(
                reinterpret_cast<const __half*>(handle_to_ptr(input_handle)),
                reinterpret_cast<const int8_t*>(handle_to_ptr(weight_handle)),
                static_cast<float*>(logits_tmp.ptr),
                batch_size,
                vocab_size,
                hidden_size,
                weight_scale);
        }
        argmax_rows_kernel<<<linear_grid_size(batch_size, 1), block_size>>>(
            static_cast<float*>(logits_tmp.ptr),
            static_cast<size_t*>(device_out_tmp.ptr),
            batch_size,
            vocab_size);
    }
    if (!sync_cuda("CUDA F16xI8 matvec argmax kernel failed")) {
        return 1;
    }

    cudaError_t status = cudaMemcpy(
        out_indices,
        static_cast<size_t*>(device_out_tmp.ptr),
        batch_size * sizeof(size_t),
        cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        set_cuda_error("failed to download CUDA F16xI8 argmax output", status);
        return 1;
    }
    return 0;
}

extern "C" int lumen_cuda_matvec_argmax_f32_i8_device(
    uint64_t input_handle,
    uint64_t weight_handle,
    float weight_scale,
    size_t* out_indices,
    size_t batch_size,
    size_t vocab_size,
    size_t hidden_size) {
    if (!validate_handle(input_handle, "CUDA F32xI8 argmax input handle") ||
        !validate_handle(weight_handle, "CUDA F32xI8 argmax weight handle")) {
        return 1;
    }
    if (out_indices == nullptr) {
        set_error("CUDA F32xI8 argmax output is null");
        return 1;
    }
    if (batch_size == 0 || vocab_size == 0 || hidden_size == 0) {
        set_error("CUDA F32xI8 argmax dimensions must be greater than zero");
        return 1;
    }
    if (!std::isfinite(weight_scale) || weight_scale <= 0.0f) {
        set_error("CUDA F32xI8 argmax weight scale must be finite and > 0");
        return 1;
    }

    const size_t max_size = static_cast<size_t>(-1);
    if (batch_size > max_size / vocab_size ||
        batch_size * vocab_size > max_size / sizeof(float)) {
        set_error("CUDA F32xI8 argmax logits length overflow");
        return 1;
    }
    if (batch_size > max_size / sizeof(size_t)) {
        set_error("CUDA F32xI8 argmax output length overflow");
        return 1;
    }

    thread_local ReusableCudaWorkspace device_out_tmp;
    if (!device_out_tmp.ensure(
            batch_size * sizeof(size_t),
            "failed to allocate CUDA F32xI8 argmax output")) {
        return 1;
    }

    constexpr int block_size = 256;
    if (batch_size == 1 && vocab_size <= 4096) {
        const size_t partial_count = vocab_size < 1024 ? vocab_size : 1024;
        thread_local ReusableCudaWorkspace best_values_tmp;
        if (!best_values_tmp.ensure(
                partial_count * sizeof(float),
                "failed to allocate CUDA F32xI8 argmax partial values")) {
            return 1;
        }
        thread_local ReusableCudaWorkspace best_indices_tmp;
        if (!best_indices_tmp.ensure(
                partial_count * sizeof(size_t),
                "failed to allocate CUDA F32xI8 argmax partial indices")) {
            return 1;
        }
        matvec_argmax_f32_i8_block_best_kernel<<<
            static_cast<unsigned int>(partial_count),
            block_size>>>(
            handle_to_ptr(input_handle),
            reinterpret_cast<const int8_t*>(handle_to_ptr(weight_handle)),
            static_cast<float*>(best_values_tmp.ptr),
            static_cast<size_t*>(best_indices_tmp.ptr),
            vocab_size,
            hidden_size,
            weight_scale);
        argmax_pairs_kernel<<<1, block_size>>>(
            static_cast<float*>(best_values_tmp.ptr),
            static_cast<size_t*>(best_indices_tmp.ptr),
            static_cast<size_t*>(device_out_tmp.ptr),
            partial_count);
    } else {
        thread_local ReusableCudaWorkspace logits_tmp;
        if (!logits_tmp.ensure(
                batch_size * vocab_size * sizeof(float),
                "failed to allocate CUDA F32xI8 argmax logits")) {
            return 1;
        }
        if (batch_size == 1) {
            matvec_f32_i8_kernel<<<linear_grid_size(vocab_size, 1), block_size>>>(
                handle_to_ptr(input_handle),
                reinterpret_cast<const int8_t*>(handle_to_ptr(weight_handle)),
                static_cast<float*>(logits_tmp.ptr),
                vocab_size,
                hidden_size,
                weight_scale);
        } else {
            dim3 block(16, 8);
            dim3 grid(
                static_cast<unsigned int>((vocab_size + 15) / 16),
                static_cast<unsigned int>((batch_size + 7) / 8));
            matmul_floatlike_i8_tiled_kernel<<<grid, block>>>(
                handle_to_ptr(input_handle),
                reinterpret_cast<const int8_t*>(handle_to_ptr(weight_handle)),
                static_cast<float*>(logits_tmp.ptr),
                batch_size,
                vocab_size,
                hidden_size,
                weight_scale);
        }
        argmax_rows_kernel<<<linear_grid_size(batch_size, 1), block_size>>>(
            static_cast<float*>(logits_tmp.ptr),
            static_cast<size_t*>(device_out_tmp.ptr),
            batch_size,
            vocab_size);
    }
    if (!sync_cuda("CUDA F32xI8 matvec argmax kernel failed")) {
        return 1;
    }

    cudaError_t status = cudaMemcpy(
        out_indices,
        static_cast<size_t*>(device_out_tmp.ptr),
        batch_size * sizeof(size_t),
        cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        set_cuda_error("failed to download CUDA F32xI8 argmax output", status);
        return 1;
    }
    return 0;
}

extern "C" int lumen_cuda_matvec_argmax_i8_i8_device(
    uint64_t input_handle,
    float input_scale,
    uint64_t weight_handle,
    float weight_scale,
    size_t* out_indices,
    size_t batch_size,
    size_t vocab_size,
    size_t hidden_size) {
    if (!validate_handle(input_handle, "CUDA I8xI8 argmax input handle") ||
        !validate_handle(weight_handle, "CUDA I8xI8 argmax weight handle")) {
        return 1;
    }
    if (out_indices == nullptr) {
        set_error("CUDA I8xI8 argmax output is null");
        return 1;
    }
    if (batch_size == 0 || vocab_size == 0 || hidden_size == 0) {
        set_error("CUDA I8xI8 argmax dimensions must be greater than zero");
        return 1;
    }
    if (!std::isfinite(input_scale) || input_scale <= 0.0f ||
        !std::isfinite(weight_scale) || weight_scale <= 0.0f) {
        set_error("CUDA I8xI8 argmax scales must be finite and > 0");
        return 1;
    }

    const size_t max_size = static_cast<size_t>(-1);
    if (batch_size > max_size / vocab_size ||
        batch_size * vocab_size > max_size / sizeof(float)) {
        set_error("CUDA I8xI8 argmax logits length overflow");
        return 1;
    }
    if (batch_size > max_size / sizeof(size_t)) {
        set_error("CUDA I8xI8 argmax output length overflow");
        return 1;
    }

    thread_local ReusableCudaWorkspace device_out_tmp;
    if (!device_out_tmp.ensure(
            batch_size * sizeof(size_t),
            "failed to allocate CUDA I8xI8 argmax output")) {
        return 1;
    }

    constexpr int block_size = 256;
    const float scale = input_scale * weight_scale;
    if (batch_size == 1 && vocab_size <= 4096) {
        const size_t partial_count = vocab_size < 1024 ? vocab_size : 1024;
        thread_local ReusableCudaWorkspace best_values_tmp;
        if (!best_values_tmp.ensure(
                partial_count * sizeof(float),
                "failed to allocate CUDA I8xI8 argmax partial values")) {
            return 1;
        }
        thread_local ReusableCudaWorkspace best_indices_tmp;
        if (!best_indices_tmp.ensure(
                partial_count * sizeof(size_t),
                "failed to allocate CUDA I8xI8 argmax partial indices")) {
            return 1;
        }
        matvec_argmax_i8_i8_block_best_kernel<<<
            static_cast<unsigned int>(partial_count),
            block_size>>>(
            reinterpret_cast<const int8_t*>(handle_to_ptr(input_handle)),
            reinterpret_cast<const int8_t*>(handle_to_ptr(weight_handle)),
            static_cast<float*>(best_values_tmp.ptr),
            static_cast<size_t*>(best_indices_tmp.ptr),
            vocab_size,
            hidden_size,
            scale);
        argmax_pairs_kernel<<<1, block_size>>>(
            static_cast<float*>(best_values_tmp.ptr),
            static_cast<size_t*>(best_indices_tmp.ptr),
            static_cast<size_t*>(device_out_tmp.ptr),
            partial_count);
    } else {
        thread_local ReusableCudaWorkspace logits_tmp;
        if (!logits_tmp.ensure(
                batch_size * vocab_size * sizeof(float),
                "failed to allocate CUDA I8xI8 argmax logits")) {
            return 1;
        }
        if (batch_size == 1) {
            matvec_i8_i8_kernel<<<linear_grid_size(vocab_size, 1), block_size>>>(
                reinterpret_cast<const int8_t*>(handle_to_ptr(input_handle)),
                reinterpret_cast<const int8_t*>(handle_to_ptr(weight_handle)),
                static_cast<float*>(logits_tmp.ptr),
                vocab_size,
                hidden_size,
                scale);
        } else {
            if (!launch_matmul_i8_compute(
                reinterpret_cast<const int8_t*>(handle_to_ptr(input_handle)),
                reinterpret_cast<const int8_t*>(handle_to_ptr(weight_handle)),
                static_cast<float*>(logits_tmp.ptr),
                batch_size,
                vocab_size,
                hidden_size,
                scale,
                "CUDA I8xI8 batched argmax logits kernel launch failed")) {
                return 1;
            }
        }
        argmax_rows_kernel<<<linear_grid_size(batch_size, 1), block_size>>>(
            static_cast<float*>(logits_tmp.ptr),
            static_cast<size_t*>(device_out_tmp.ptr),
            batch_size,
            vocab_size);
    }
    if (!sync_cuda("CUDA I8xI8 matvec argmax kernel failed")) {
        return 1;
    }

    cudaError_t status = cudaMemcpy(
        out_indices,
        static_cast<size_t*>(device_out_tmp.ptr),
        batch_size * sizeof(size_t),
        cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        set_cuda_error("failed to download CUDA I8xI8 argmax output", status);
        return 1;
    }
    return 0;
}

static bool can_launch_batch_matmul_tiled(size_t batch_count, size_t m, size_t n) {
    constexpr size_t max_grid_x = 2147483647;
    constexpr size_t max_grid_yz = 65535;
    constexpr size_t tile_m = 8;
    constexpr size_t tile_n = 16;
    return batch_count <= max_grid_yz && m <= max_grid_yz * tile_m &&
           n <= max_grid_x * tile_n;
}

__global__ void batch_matmul_i8_kernel(
    const int8_t* lhs,
    const int8_t* rhs,
    float* out,
    size_t total,
    size_t m,
    size_t n,
    size_t k,
    float scale) {
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    const size_t batch_area = m * n;
    for (size_t idx = start; idx < total; idx += stride) {
        size_t batch = idx / batch_area;
        size_t offset = idx - batch * batch_area;
        size_t row = offset / n;
        size_t col = offset - row * n;
        const int8_t* lhs_batch = lhs + batch * m * k;
        const int8_t* rhs_batch = rhs + batch * k * n;
        int acc = 0;
        for (size_t kk = 0; kk < k; ++kk) {
            acc += static_cast<int>(lhs_batch[row * k + kk]) *
                   static_cast<int>(rhs_batch[kk * n + col]);
        }
        out[idx] = static_cast<float>(acc) * scale;
    }
}

__global__ void batch_matmul_i8_tiled_kernel(
    const int8_t* lhs,
    const int8_t* rhs,
    float* out,
    size_t m,
    size_t n,
    size_t k,
    float scale) {
    constexpr int tile_m = 8;
    constexpr int tile_n = 16;
    constexpr int tile_k = 32;
    __shared__ int8_t lhs_tile[tile_m][tile_k];
    __shared__ int8_t rhs_tile[tile_k][tile_n];

    const size_t batch = blockIdx.z;
    const int8_t* lhs_batch = lhs + batch * m * k;
    const int8_t* rhs_batch = rhs + batch * k * n;
    float* out_batch = out + batch * m * n;
    const size_t row = blockIdx.y * tile_m + threadIdx.y;
    const size_t col = blockIdx.x * tile_n + threadIdx.x;
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int block_threads = blockDim.x * blockDim.y;
    int acc = 0;

    for (size_t kt = 0; kt < k; kt += tile_k) {
        for (int idx = tid; idx < tile_m * tile_k; idx += block_threads) {
            const int local_row = idx / tile_k;
            const int local_k = idx - local_row * tile_k;
            const size_t global_row = blockIdx.y * tile_m + local_row;
            const size_t global_k = kt + local_k;
            lhs_tile[local_row][local_k] =
                (global_row < m && global_k < k) ? lhs_batch[global_row * k + global_k] : 0;
        }
        for (int idx = tid; idx < tile_k * tile_n; idx += block_threads) {
            const int local_k = idx / tile_n;
            const int local_col = idx - local_k * tile_n;
            const size_t global_k = kt + local_k;
            const size_t global_col = blockIdx.x * tile_n + local_col;
            rhs_tile[local_k][local_col] =
                (global_k < k && global_col < n) ? rhs_batch[global_k * n + global_col] : 0;
        }
        __syncthreads();

        if (row < m && col < n) {
            for (int local_k = 0; local_k < tile_k; ++local_k) {
                acc += static_cast<int>(lhs_tile[threadIdx.y][local_k]) *
                       static_cast<int>(rhs_tile[local_k][threadIdx.x]);
            }
        }
        __syncthreads();
    }

    if (row < m && col < n) {
        out_batch[row * n + col] = static_cast<float>(acc) * scale;
    }
}

static bool launch_batch_matmul_i8_compute(
    const int8_t* lhs,
    const int8_t* rhs,
    float* out,
    size_t batch_count,
    size_t m,
    size_t n,
    size_t k,
    float scale,
    const char* error_message) {
    if (can_launch_batch_matmul_tiled(batch_count, m, n)) {
        dim3 block(16, 8);
        dim3 grid(
            static_cast<unsigned int>((n + 15) / 16),
            static_cast<unsigned int>((m + 7) / 8),
            static_cast<unsigned int>(batch_count));
        batch_matmul_i8_tiled_kernel<<<grid, block>>>(lhs, rhs, out, m, n, k, scale);
    } else {
        const size_t total = batch_count * m * n;
        constexpr unsigned int block_size = 256;
        const unsigned int grid = linear_grid_size(total, block_size);
        batch_matmul_i8_kernel<<<grid, block_size>>>(lhs, rhs, out, total, m, n, k, scale);
    }
    return check_cuda_launch(error_message);
}

template <typename LhsT>
__global__ void batch_matmul_floatlike_i8_kernel(
    const LhsT* lhs,
    const int8_t* rhs,
    float* out,
    size_t total,
    size_t m,
    size_t n,
    size_t k,
    float rhs_scale) {
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    const size_t batch_area = m * n;
    for (size_t idx = start; idx < total; idx += stride) {
        size_t batch = idx / batch_area;
        size_t offset = idx - batch * batch_area;
        size_t row = offset / n;
        size_t col = offset - row * n;
        const LhsT* lhs_batch = lhs + batch * m * k;
        const int8_t* rhs_batch = rhs + batch * k * n;
        float acc = 0.0f;
        for (size_t kk = 0; kk < k; ++kk) {
            const float lhs_value = load_floatlike_value(lhs_batch, row * k + kk);
            const float rhs_value = static_cast<float>(rhs_batch[kk * n + col]) * rhs_scale;
            acc += lhs_value * rhs_value;
        }
        out[idx] = acc;
    }
}

template <typename LhsT>
__global__ void batch_matmul_floatlike_i8_tiled_kernel(
    const LhsT* lhs,
    const int8_t* rhs,
    float* out,
    size_t m,
    size_t n,
    size_t k,
    float rhs_scale) {
    constexpr int tile_m = 8;
    constexpr int tile_n = 16;
    constexpr int tile_k = 32;
    __shared__ float lhs_tile[tile_m][tile_k];
    __shared__ float rhs_tile[tile_k][tile_n];

    const size_t batch = blockIdx.z;
    const LhsT* lhs_batch = lhs + batch * m * k;
    const int8_t* rhs_batch = rhs + batch * k * n;
    float* out_batch = out + batch * m * n;
    const size_t row = blockIdx.y * tile_m + threadIdx.y;
    const size_t col = blockIdx.x * tile_n + threadIdx.x;
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int block_threads = blockDim.x * blockDim.y;
    float acc = 0.0f;

    for (size_t kt = 0; kt < k; kt += tile_k) {
        for (int idx = tid; idx < tile_m * tile_k; idx += block_threads) {
            const int local_row = idx / tile_k;
            const int local_k = idx - local_row * tile_k;
            const size_t global_row = blockIdx.y * tile_m + local_row;
            const size_t global_k = kt + local_k;
            lhs_tile[local_row][local_k] =
                (global_row < m && global_k < k)
                    ? load_floatlike_value(lhs_batch, global_row * k + global_k)
                    : 0.0f;
        }
        for (int idx = tid; idx < tile_k * tile_n; idx += block_threads) {
            const int local_k = idx / tile_n;
            const int local_col = idx - local_k * tile_n;
            const size_t global_k = kt + local_k;
            const size_t global_col = blockIdx.x * tile_n + local_col;
            rhs_tile[local_k][local_col] =
                (global_k < k && global_col < n)
                    ? static_cast<float>(rhs_batch[global_k * n + global_col]) * rhs_scale
                    : 0.0f;
        }
        __syncthreads();

        if (row < m && col < n) {
            for (int local_k = 0; local_k < tile_k; ++local_k) {
                acc += lhs_tile[threadIdx.y][local_k] * rhs_tile[local_k][threadIdx.x];
            }
        }
        __syncthreads();
    }

    if (row < m && col < n) {
        out_batch[row * n + col] = acc;
    }
}

template <typename RhsT>
__global__ void batch_matmul_i8_floatlike_kernel(
    const int8_t* lhs,
    const RhsT* rhs,
    float* out,
    size_t total,
    size_t m,
    size_t n,
    size_t k,
    float lhs_scale) {
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    const size_t batch_area = m * n;
    for (size_t idx = start; idx < total; idx += stride) {
        size_t batch = idx / batch_area;
        size_t offset = idx - batch * batch_area;
        size_t row = offset / n;
        size_t col = offset - row * n;
        const int8_t* lhs_batch = lhs + batch * m * k;
        const RhsT* rhs_batch = rhs + batch * k * n;
        float acc = 0.0f;
        for (size_t kk = 0; kk < k; ++kk) {
            const float lhs_value = static_cast<float>(lhs_batch[row * k + kk]) * lhs_scale;
            const float rhs_value = load_floatlike_value(rhs_batch, kk * n + col);
            acc += lhs_value * rhs_value;
        }
        out[idx] = acc;
    }
}

template <typename RhsT>
__global__ void batch_matmul_i8_floatlike_tiled_kernel(
    const int8_t* lhs,
    const RhsT* rhs,
    float* out,
    size_t m,
    size_t n,
    size_t k,
    float lhs_scale) {
    constexpr int tile_m = 8;
    constexpr int tile_n = 16;
    constexpr int tile_k = 32;
    __shared__ float lhs_tile[tile_m][tile_k];
    __shared__ float rhs_tile[tile_k][tile_n];

    const size_t batch = blockIdx.z;
    const int8_t* lhs_batch = lhs + batch * m * k;
    const RhsT* rhs_batch = rhs + batch * k * n;
    float* out_batch = out + batch * m * n;
    const size_t row = blockIdx.y * tile_m + threadIdx.y;
    const size_t col = blockIdx.x * tile_n + threadIdx.x;
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int block_threads = blockDim.x * blockDim.y;
    float acc = 0.0f;

    for (size_t kt = 0; kt < k; kt += tile_k) {
        for (int idx = tid; idx < tile_m * tile_k; idx += block_threads) {
            const int local_row = idx / tile_k;
            const int local_k = idx - local_row * tile_k;
            const size_t global_row = blockIdx.y * tile_m + local_row;
            const size_t global_k = kt + local_k;
            lhs_tile[local_row][local_k] =
                (global_row < m && global_k < k)
                    ? static_cast<float>(lhs_batch[global_row * k + global_k]) * lhs_scale
                    : 0.0f;
        }
        for (int idx = tid; idx < tile_k * tile_n; idx += block_threads) {
            const int local_k = idx / tile_n;
            const int local_col = idx - local_k * tile_n;
            const size_t global_k = kt + local_k;
            const size_t global_col = blockIdx.x * tile_n + local_col;
            rhs_tile[local_k][local_col] =
                (global_k < k && global_col < n)
                    ? load_floatlike_value(rhs_batch, global_k * n + global_col)
                    : 0.0f;
        }
        __syncthreads();

        if (row < m && col < n) {
            for (int local_k = 0; local_k < tile_k; ++local_k) {
                acc += lhs_tile[threadIdx.y][local_k] * rhs_tile[local_k][threadIdx.x];
            }
        }
        __syncthreads();
    }

    if (row < m && col < n) {
        out_batch[row * n + col] = acc;
    }
}

template <typename LhsT>
static bool launch_batch_matmul_floatlike_i8_compute(
    const LhsT* lhs,
    const int8_t* rhs,
    float* out,
    size_t batch_count,
    size_t m,
    size_t n,
    size_t k,
    float rhs_scale,
    const char* error_message) {
    if (can_launch_batch_matmul_tiled(batch_count, m, n)) {
        dim3 block(16, 8);
        dim3 grid(
            static_cast<unsigned int>((n + 15) / 16),
            static_cast<unsigned int>((m + 7) / 8),
            static_cast<unsigned int>(batch_count));
        batch_matmul_floatlike_i8_tiled_kernel<<<grid, block>>>(
            lhs, rhs, out, m, n, k, rhs_scale);
    } else {
        const size_t total = batch_count * m * n;
        constexpr unsigned int block_size = 256;
        batch_matmul_floatlike_i8_kernel<<<linear_grid_size(total, block_size), block_size>>>(
            lhs, rhs, out, total, m, n, k, rhs_scale);
    }
    return sync_cuda(error_message);
}

template <typename RhsT>
static bool launch_batch_matmul_i8_floatlike_compute(
    const int8_t* lhs,
    const RhsT* rhs,
    float* out,
    size_t batch_count,
    size_t m,
    size_t n,
    size_t k,
    float lhs_scale,
    const char* error_message) {
    if (can_launch_batch_matmul_tiled(batch_count, m, n)) {
        dim3 block(16, 8);
        dim3 grid(
            static_cast<unsigned int>((n + 15) / 16),
            static_cast<unsigned int>((m + 7) / 8),
            static_cast<unsigned int>(batch_count));
        batch_matmul_i8_floatlike_tiled_kernel<<<grid, block>>>(
            lhs, rhs, out, m, n, k, lhs_scale);
    } else {
        const size_t total = batch_count * m * n;
        constexpr unsigned int block_size = 256;
        batch_matmul_i8_floatlike_kernel<<<linear_grid_size(total, block_size), block_size>>>(
            lhs, rhs, out, total, m, n, k, lhs_scale);
    }
    return sync_cuda(error_message);
}

__global__ void batch_matmul_i8_backward_kernel(
    const float* grad,
    const int8_t* lhs,
    float lhs_scale,
    const int8_t* rhs,
    float rhs_scale,
    float* d_lhs,
    float* d_rhs,
    size_t lhs_total,
    size_t rhs_total,
    size_t m,
    size_t k,
    size_t n) {
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t lhs_idx = start; lhs_idx < lhs_total; lhs_idx += stride) {
        size_t batch_area = m * k;
        size_t batch = lhs_idx / batch_area;
        size_t offset = lhs_idx - batch * batch_area;
        size_t row = offset / k;
        size_t kk = offset - row * k;
        const float* grad_batch = grad + batch * m * n;
        const int8_t* rhs_batch = rhs + batch * k * n;
        float acc = 0.0f;
        for (size_t col = 0; col < n; ++col) {
            acc += grad_batch[row * n + col] *
                   static_cast<float>(rhs_batch[kk * n + col]) * rhs_scale;
        }
        d_lhs[lhs_idx] = acc;
    }

    for (size_t rhs_idx = start; rhs_idx < rhs_total; rhs_idx += stride) {
        size_t batch_area = k * n;
        size_t batch = rhs_idx / batch_area;
        size_t offset = rhs_idx - batch * batch_area;
        size_t kk = offset / n;
        size_t col = offset - kk * n;
        const float* grad_batch = grad + batch * m * n;
        const int8_t* lhs_batch = lhs + batch * m * k;
        float acc = 0.0f;
        for (size_t row = 0; row < m; ++row) {
            acc += static_cast<float>(lhs_batch[row * k + kk]) * lhs_scale *
                   grad_batch[row * n + col];
        }
        d_rhs[rhs_idx] = acc;
    }
}

template <typename T>
__global__ void batch_matmul_lowp_backward_kernel(
    const float* grad,
    const T* lhs,
    const T* rhs,
    float* d_lhs,
    float* d_rhs,
    size_t lhs_total,
    size_t rhs_total,
    size_t m,
    size_t k,
    size_t n) {
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t lhs_idx = start; lhs_idx < lhs_total; lhs_idx += stride) {
        size_t batch_area = m * k;
        size_t batch = lhs_idx / batch_area;
        size_t offset = lhs_idx - batch * batch_area;
        size_t row = offset / k;
        size_t kk = offset - row * k;
        const float* grad_batch = grad + batch * m * n;
        const T* rhs_batch = rhs + batch * k * n;
        float acc = 0.0f;
        for (size_t col = 0; col < n; ++col) {
            acc += grad_batch[row * n + col] * lowp_to_float(rhs_batch[kk * n + col]);
        }
        d_lhs[lhs_idx] = acc;
    }

    for (size_t rhs_idx = start; rhs_idx < rhs_total; rhs_idx += stride) {
        size_t batch_area = k * n;
        size_t batch = rhs_idx / batch_area;
        size_t offset = rhs_idx - batch * batch_area;
        size_t kk = offset / n;
        size_t col = offset - kk * n;
        const float* grad_batch = grad + batch * m * n;
        const T* lhs_batch = lhs + batch * m * k;
        float acc = 0.0f;
        for (size_t row = 0; row < m; ++row) {
            acc += lowp_to_float(lhs_batch[row * k + kk]) * grad_batch[row * n + col];
        }
        d_rhs[rhs_idx] = acc;
    }
}

__device__ inline float matmul_backward_lhs_to_float(float value) {
    return value;
}

__device__ inline float matmul_backward_lhs_to_float(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

__device__ inline float matmul_backward_lhs_to_float(__half value) {
    return __half2float(value);
}

template <typename LhsT>
__global__ void batch_matmul_lhs_i8_backward_kernel(
    const float* grad,
    const LhsT* lhs,
    const int8_t* rhs,
    float rhs_scale,
    float* d_lhs,
    float* d_rhs,
    size_t lhs_total,
    size_t rhs_total,
    size_t m,
    size_t k,
    size_t n) {
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t lhs_idx = start; lhs_idx < lhs_total; lhs_idx += stride) {
        size_t batch_area = m * k;
        size_t batch = lhs_idx / batch_area;
        size_t offset = lhs_idx - batch * batch_area;
        size_t row = offset / k;
        size_t kk = offset - row * k;
        const float* grad_batch = grad + batch * m * n;
        const int8_t* rhs_batch = rhs + batch * k * n;
        float acc = 0.0f;
        for (size_t col = 0; col < n; ++col) {
            acc += grad_batch[row * n + col] *
                   static_cast<float>(rhs_batch[kk * n + col]) * rhs_scale;
        }
        d_lhs[lhs_idx] = acc;
    }

    for (size_t rhs_idx = start; rhs_idx < rhs_total; rhs_idx += stride) {
        size_t batch_area = k * n;
        size_t batch = rhs_idx / batch_area;
        size_t offset = rhs_idx - batch * batch_area;
        size_t kk = offset / n;
        size_t col = offset - kk * n;
        const float* grad_batch = grad + batch * m * n;
        const LhsT* lhs_batch = lhs + batch * m * k;
        float acc = 0.0f;
        for (size_t row = 0; row < m; ++row) {
            acc += matmul_backward_lhs_to_float(lhs_batch[row * k + kk]) *
                   grad_batch[row * n + col];
        }
        d_rhs[rhs_idx] = acc;
    }
}

template <typename LhsT>
__global__ void matmul_lhs_i8_backward_kernel(
    const float* grad,
    const LhsT* lhs,
    const int8_t* rhs,
    float rhs_scale,
    float* d_lhs,
    float* d_rhs,
    size_t lhs_total,
    size_t rhs_total,
    size_t m,
    size_t k,
    size_t n) {
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t lhs_idx = start; lhs_idx < lhs_total; lhs_idx += stride) {
        size_t row = lhs_idx / k;
        size_t kk = lhs_idx - row * k;
        float acc = 0.0f;
        for (size_t col = 0; col < n; ++col) {
            acc += grad[row * n + col] * static_cast<float>(rhs[col * k + kk]) * rhs_scale;
        }
        d_lhs[lhs_idx] = acc;
    }

    for (size_t rhs_idx = start; rhs_idx < rhs_total; rhs_idx += stride) {
        size_t col = rhs_idx / k;
        size_t kk = rhs_idx - col * k;
        float acc = 0.0f;
        for (size_t row = 0; row < m; ++row) {
            acc += matmul_backward_lhs_to_float(lhs[row * k + kk]) * grad[row * n + col];
        }
        d_rhs[rhs_idx] = acc;
    }
}

template <typename RhsT>
__global__ void matmul_i8_rhs_floatlike_backward_kernel(
    const float* grad,
    const int8_t* lhs,
    float lhs_scale,
    const RhsT* rhs,
    float* d_lhs,
    float* d_rhs,
    size_t lhs_total,
    size_t rhs_total,
    size_t m,
    size_t k,
    size_t n) {
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t lhs_idx = start; lhs_idx < lhs_total; lhs_idx += stride) {
        size_t row = lhs_idx / k;
        size_t kk = lhs_idx - row * k;
        float acc = 0.0f;
        for (size_t col = 0; col < n; ++col) {
            acc += grad[row * n + col] * load_floatlike_value(rhs, col * k + kk);
        }
        d_lhs[lhs_idx] = acc;
    }

    for (size_t rhs_idx = start; rhs_idx < rhs_total; rhs_idx += stride) {
        size_t col = rhs_idx / k;
        size_t kk = rhs_idx - col * k;
        float acc = 0.0f;
        for (size_t row = 0; row < m; ++row) {
            acc += static_cast<float>(lhs[row * k + kk]) * lhs_scale * grad[row * n + col];
        }
        d_rhs[rhs_idx] = acc;
    }
}

template <typename RhsT>
__global__ void batch_matmul_i8_rhs_floatlike_backward_kernel(
    const float* grad,
    const int8_t* lhs,
    float lhs_scale,
    const RhsT* rhs,
    float* d_lhs,
    float* d_rhs,
    size_t lhs_total,
    size_t rhs_total,
    size_t m,
    size_t k,
    size_t n) {
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t lhs_idx = start; lhs_idx < lhs_total; lhs_idx += stride) {
        size_t batch_area = m * k;
        size_t batch = lhs_idx / batch_area;
        size_t offset = lhs_idx - batch * batch_area;
        size_t row = offset / k;
        size_t kk = offset - row * k;
        const float* grad_batch = grad + batch * m * n;
        const RhsT* rhs_batch = rhs + batch * k * n;
        float acc = 0.0f;
        for (size_t col = 0; col < n; ++col) {
            acc += grad_batch[row * n + col] * load_floatlike_value(rhs_batch, kk * n + col);
        }
        d_lhs[lhs_idx] = acc;
    }

    for (size_t rhs_idx = start; rhs_idx < rhs_total; rhs_idx += stride) {
        size_t batch_area = k * n;
        size_t batch = rhs_idx / batch_area;
        size_t offset = rhs_idx - batch * batch_area;
        size_t kk = offset / n;
        size_t col = offset - kk * n;
        const float* grad_batch = grad + batch * m * n;
        const int8_t* lhs_batch = lhs + batch * m * k;
        float acc = 0.0f;
        for (size_t row = 0; row < m; ++row) {
            acc += static_cast<float>(lhs_batch[row * k + kk]) * lhs_scale *
                   grad_batch[row * n + col];
        }
        d_rhs[rhs_idx] = acc;
    }
}

extern "C" int lumen_cuda_matmul_i8_host_device(
    const int8_t* a_host,
    const int8_t* b_host,
    float a_scale,
    float b_scale,
    uint64_t out_handle,
    size_t m,
    size_t n,
    size_t k) {
    if (!validate_handle(out_handle, "CUDA I8 matmul output handle") || !validate_dims(m, n, k)) {
        return 1;
    }
    if (!std::isfinite(a_scale) || a_scale <= 0.0f ||
        !std::isfinite(b_scale) || b_scale <= 0.0f) {
        set_error("CUDA I8 matmul scales must be finite and > 0");
        return 1;
    }

    ScopedDeviceInput<int8_t> a_dev;
    ScopedDeviceInput<int8_t> b_dev;
    if (!upload_typed_input(a_host, m * k, &a_dev.ptr, "upload I8 matmul A") ||
        !upload_typed_input(b_host, n * k, &b_dev.ptr, "upload I8 matmul B")) {
        return 1;
    }

    if (!launch_matmul_i8_compute(
        a_dev.ptr,
        b_dev.ptr,
        handle_to_ptr(out_handle),
        m,
        n,
        k,
        a_scale * b_scale,
        "CUDA I8 matmul kernel launch failed")) {
        return 1;
    }
    if (!sync_cuda("CUDA I8 matmul kernel failed")) {
        return 1;
    }
    return 0;
}

extern "C" int lumen_cuda_matmul_i8_device(
    uint64_t a_handle,
    float a_scale,
    uint64_t b_handle,
    float b_scale,
    uint64_t out_handle,
    size_t m,
    size_t n,
    size_t k) {
    if (!validate_handle(a_handle, "CUDA I8 matmul A handle") ||
        !validate_handle(b_handle, "CUDA I8 matmul B handle") ||
        !validate_handle(out_handle, "CUDA I8 matmul output handle") ||
        !validate_dims(m, n, k)) {
        return 1;
    }
    if (!std::isfinite(a_scale) || a_scale <= 0.0f ||
        !std::isfinite(b_scale) || b_scale <= 0.0f) {
        set_error("CUDA I8 resident matmul scales must be finite and > 0");
        return 1;
    }

    if (!launch_matmul_i8_compute(
        reinterpret_cast<const int8_t*>(handle_to_ptr(a_handle)),
        reinterpret_cast<const int8_t*>(handle_to_ptr(b_handle)),
        handle_to_ptr(out_handle),
        m,
        n,
        k,
        a_scale * b_scale,
        "CUDA I8 resident matmul kernel launch failed")) {
        return 1;
    }
    if (!sync_cuda("CUDA I8 resident matmul kernel failed")) {
        return 1;
    }
    return 0;
}

extern "C" int lumen_cuda_matmul_bf16_i8_device(
    uint64_t a_handle,
    uint64_t b_handle,
    float b_scale,
    uint64_t out_handle,
    size_t m,
    size_t n,
    size_t k) {
    if (!validate_handle(a_handle, "CUDA BF16xI8 matmul A handle") ||
        !validate_handle(b_handle, "CUDA BF16xI8 matmul B handle") ||
        !validate_handle(out_handle, "CUDA BF16xI8 matmul output handle") ||
        !validate_dims(m, n, k)) {
        return 1;
    }
    if (!std::isfinite(b_scale) || b_scale <= 0.0f) {
        set_error("CUDA BF16xI8 matmul scale must be finite and > 0");
        return 1;
    }

    constexpr int block_size = 256;
    if (m == 1) {
        matvec_bf16_i8_kernel<<<linear_grid_size(n, 1), block_size>>>(
            reinterpret_cast<const __nv_bfloat16*>(handle_to_ptr(a_handle)),
            reinterpret_cast<const int8_t*>(handle_to_ptr(b_handle)),
            handle_to_ptr(out_handle),
            n,
            k,
            b_scale);
    } else {
        if (!launch_matmul_floatlike_i8_compute(
            reinterpret_cast<const __nv_bfloat16*>(handle_to_ptr(a_handle)),
            reinterpret_cast<const int8_t*>(handle_to_ptr(b_handle)),
            handle_to_ptr(out_handle),
            m,
            n,
            k,
            b_scale,
            "CUDA BF16xI8 resident matmul kernel launch failed")) {
            return 1;
        }
    }
    if (!sync_cuda("CUDA BF16xI8 resident matmul kernel failed")) {
        return 1;
    }
    return 0;
}

extern "C" int lumen_cuda_matmul_f16_i8_device(
    uint64_t a_handle,
    uint64_t b_handle,
    float b_scale,
    uint64_t out_handle,
    size_t m,
    size_t n,
    size_t k) {
    if (!validate_handle(a_handle, "CUDA F16xI8 matmul A handle") ||
        !validate_handle(b_handle, "CUDA F16xI8 matmul B handle") ||
        !validate_handle(out_handle, "CUDA F16xI8 matmul output handle") ||
        !validate_dims(m, n, k)) {
        return 1;
    }
    if (!std::isfinite(b_scale) || b_scale <= 0.0f) {
        set_error("CUDA F16xI8 matmul scale must be finite and > 0");
        return 1;
    }

    constexpr int block_size = 256;
    if (m == 1) {
        matvec_f16_i8_kernel<<<linear_grid_size(n, 1), block_size>>>(
            reinterpret_cast<const __half*>(handle_to_ptr(a_handle)),
            reinterpret_cast<const int8_t*>(handle_to_ptr(b_handle)),
            handle_to_ptr(out_handle),
            n,
            k,
            b_scale);
    } else {
        if (!launch_matmul_floatlike_i8_compute(
            reinterpret_cast<const __half*>(handle_to_ptr(a_handle)),
            reinterpret_cast<const int8_t*>(handle_to_ptr(b_handle)),
            handle_to_ptr(out_handle),
            m,
            n,
            k,
            b_scale,
            "CUDA F16xI8 resident matmul kernel launch failed")) {
            return 1;
        }
    }
    if (!sync_cuda("CUDA F16xI8 resident matmul kernel failed")) {
        return 1;
    }
    return 0;
}

extern "C" int lumen_cuda_matmul_f32_i8_device(
    uint64_t a_handle,
    uint64_t b_handle,
    float b_scale,
    uint64_t out_handle,
    size_t m,
    size_t n,
    size_t k) {
    if (!validate_handle(a_handle, "CUDA F32xI8 matmul A handle") ||
        !validate_handle(b_handle, "CUDA F32xI8 matmul B handle") ||
        !validate_handle(out_handle, "CUDA F32xI8 matmul output handle") ||
        !validate_dims(m, n, k)) {
        return 1;
    }
    if (!std::isfinite(b_scale) || b_scale <= 0.0f) {
        set_error("CUDA F32xI8 matmul scale must be finite and > 0");
        return 1;
    }

    constexpr int block_size = 256;
    if (m == 1) {
        matvec_f32_i8_kernel<<<linear_grid_size(n, 1), block_size>>>(
            handle_to_ptr(a_handle),
            reinterpret_cast<const int8_t*>(handle_to_ptr(b_handle)),
            handle_to_ptr(out_handle),
            n,
            k,
            b_scale);
    } else {
        if (!launch_matmul_floatlike_i8_compute(
            handle_to_ptr(a_handle),
            reinterpret_cast<const int8_t*>(handle_to_ptr(b_handle)),
            handle_to_ptr(out_handle),
            m,
            n,
            k,
            b_scale,
            "CUDA F32xI8 resident matmul kernel launch failed")) {
            return 1;
        }
    }
    if (!sync_cuda("CUDA F32xI8 resident matmul kernel failed")) {
        return 1;
    }
    return 0;
}

extern "C" int lumen_cuda_matmul_i8_bf16_device(
    uint64_t a_handle,
    float a_scale,
    uint64_t b_handle,
    uint64_t out_handle,
    size_t m,
    size_t n,
    size_t k) {
    if (!validate_handle(a_handle, "CUDA I8xBF16 matmul A handle") ||
        !validate_handle(b_handle, "CUDA I8xBF16 matmul B handle") ||
        !validate_handle(out_handle, "CUDA I8xBF16 matmul output handle") ||
        !validate_dims(m, n, k)) {
        return 1;
    }
    if (!std::isfinite(a_scale) || a_scale <= 0.0f) {
        set_error("CUDA I8xBF16 matmul scale must be finite and > 0");
        return 1;
    }

    constexpr int block_size = 256;
    if (m == 1) {
        matvec_i8_floatlike_kernel<<<linear_grid_size(n, 1), block_size>>>(
            reinterpret_cast<const int8_t*>(handle_to_ptr(a_handle)),
            reinterpret_cast<const __nv_bfloat16*>(handle_to_ptr(b_handle)),
            handle_to_ptr(out_handle),
            n,
            k,
            a_scale);
    } else {
        if (!launch_matmul_i8_floatlike_compute(
            reinterpret_cast<const int8_t*>(handle_to_ptr(a_handle)),
            reinterpret_cast<const __nv_bfloat16*>(handle_to_ptr(b_handle)),
            handle_to_ptr(out_handle),
            m,
            n,
            k,
            a_scale,
            "CUDA I8xBF16 resident matmul kernel launch failed")) {
            return 1;
        }
    }
    if (!sync_cuda("CUDA I8xBF16 resident matmul kernel failed")) {
        return 1;
    }
    return 0;
}

extern "C" int lumen_cuda_matmul_i8_f16_device(
    uint64_t a_handle,
    float a_scale,
    uint64_t b_handle,
    uint64_t out_handle,
    size_t m,
    size_t n,
    size_t k) {
    if (!validate_handle(a_handle, "CUDA I8xF16 matmul A handle") ||
        !validate_handle(b_handle, "CUDA I8xF16 matmul B handle") ||
        !validate_handle(out_handle, "CUDA I8xF16 matmul output handle") ||
        !validate_dims(m, n, k)) {
        return 1;
    }
    if (!std::isfinite(a_scale) || a_scale <= 0.0f) {
        set_error("CUDA I8xF16 matmul scale must be finite and > 0");
        return 1;
    }

    constexpr int block_size = 256;
    if (m == 1) {
        matvec_i8_floatlike_kernel<<<linear_grid_size(n, 1), block_size>>>(
            reinterpret_cast<const int8_t*>(handle_to_ptr(a_handle)),
            reinterpret_cast<const __half*>(handle_to_ptr(b_handle)),
            handle_to_ptr(out_handle),
            n,
            k,
            a_scale);
    } else {
        if (!launch_matmul_i8_floatlike_compute(
            reinterpret_cast<const int8_t*>(handle_to_ptr(a_handle)),
            reinterpret_cast<const __half*>(handle_to_ptr(b_handle)),
            handle_to_ptr(out_handle),
            m,
            n,
            k,
            a_scale,
            "CUDA I8xF16 resident matmul kernel launch failed")) {
            return 1;
        }
    }
    if (!sync_cuda("CUDA I8xF16 resident matmul kernel failed")) {
        return 1;
    }
    return 0;
}

extern "C" int lumen_cuda_matmul_i8_f32_device(
    uint64_t a_handle,
    float a_scale,
    uint64_t b_handle,
    uint64_t out_handle,
    size_t m,
    size_t n,
    size_t k) {
    if (!validate_handle(a_handle, "CUDA I8xF32 matmul A handle") ||
        !validate_handle(b_handle, "CUDA I8xF32 matmul B handle") ||
        !validate_handle(out_handle, "CUDA I8xF32 matmul output handle") ||
        !validate_dims(m, n, k)) {
        return 1;
    }
    if (!std::isfinite(a_scale) || a_scale <= 0.0f) {
        set_error("CUDA I8xF32 matmul scale must be finite and > 0");
        return 1;
    }

    constexpr int block_size = 256;
    if (m == 1) {
        matvec_i8_floatlike_kernel<<<linear_grid_size(n, 1), block_size>>>(
            reinterpret_cast<const int8_t*>(handle_to_ptr(a_handle)),
            handle_to_ptr(b_handle),
            handle_to_ptr(out_handle),
            n,
            k,
            a_scale);
    } else {
        if (!launch_matmul_i8_floatlike_compute(
            reinterpret_cast<const int8_t*>(handle_to_ptr(a_handle)),
            handle_to_ptr(b_handle),
            handle_to_ptr(out_handle),
            m,
            n,
            k,
            a_scale,
            "CUDA I8xF32 resident matmul kernel launch failed")) {
            return 1;
        }
    }
    if (!sync_cuda("CUDA I8xF32 resident matmul kernel failed")) {
        return 1;
    }
    return 0;
}

static bool quantize_f32_output_to_i8(
    const float* input,
    int8_t* output,
    size_t total,
    float* out_scale,
    const char* allocation_error,
    const char* absmax_launch_error,
    const char* absmax_finalize_error,
    const char* quantize_launch_error) {
    constexpr unsigned int block_size = 256;
    const unsigned int grid = linear_grid_size(total, block_size);
    thread_local ReusableCudaWorkspace partial_tmp;
    if (!partial_tmp.ensure(static_cast<size_t>(grid) * sizeof(float), allocation_error)) {
        return false;
    }
    float* partial = static_cast<float*>(partial_tmp.ptr);
    f32_absmax_blocks_kernel<<<grid, block_size, block_size * sizeof(float)>>>(input, partial, total);
    if (!check_cuda_launch(absmax_launch_error)) {
        return false;
    }

    thread_local ReusableCudaWorkspace device_max_tmp;
    if (!device_max_tmp.ensure(sizeof(float), allocation_error)) {
        return false;
    }
    float* device_max = static_cast<float*>(device_max_tmp.ptr);
    f32_absmax_finalize_kernel<<<1, block_size, block_size * sizeof(float)>>>(partial, device_max, grid);
    if (!check_cuda_launch(absmax_finalize_error)) {
        return false;
    }

    quantize_f32_to_i8_device_absmax_kernel<<<grid, block_size>>>(
        input,
        output,
        total,
        device_max);
    if (!check_cuda_launch(quantize_launch_error)) {
        return false;
    }

    float max_abs = 0.0f;
    cudaError_t status = cudaMemcpy(&max_abs, device_max, sizeof(float), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        set_cuda_error("failed to download CUDA I8 typed-output absmax", status);
        return false;
    }

    const float scale =
        max_abs > 0.0f && isfinite(max_abs) ? std::max(max_abs / 127.0f, FLT_MIN) : 1.0f;
    *out_scale = scale;
    return true;
}

extern "C" int lumen_cuda_matmul_i8_typed_out_device(
    uint64_t a_handle,
    float a_scale,
    uint64_t b_handle,
    float b_scale,
    uint64_t out_handle,
    size_t m,
    size_t n,
    size_t k,
    float* out_scale) {
    if (!validate_handle(a_handle, "CUDA I8 typed-output matmul A handle") ||
        !validate_handle(b_handle, "CUDA I8 typed-output matmul B handle") ||
        !validate_handle(out_handle, "CUDA I8 typed-output matmul output handle") ||
        !validate_dims(m, n, k)) {
        return 1;
    }
    if (out_scale == nullptr) {
        set_error("CUDA I8 typed-output matmul scale output is null");
        return 1;
    }
    if (!std::isfinite(a_scale) || a_scale <= 0.0f ||
        !std::isfinite(b_scale) || b_scale <= 0.0f) {
        set_error("CUDA I8 typed-output matmul scales must be finite and > 0");
        return 1;
    }

    const size_t max_size = static_cast<size_t>(-1);
    if (m > max_size / n || m * n > max_size / sizeof(float)) {
        set_error("CUDA I8 typed-output matmul temporary output length overflow");
        return 1;
    }
    const size_t total = m * n;
    thread_local ReusableCudaWorkspace f32_output_tmp;
    if (!f32_output_tmp.ensure(
            total * sizeof(float),
            "failed to allocate CUDA I8 typed-output matmul f32 output buffer")) {
        return 1;
    }
    float* f32_output = static_cast<float*>(f32_output_tmp.ptr);
    if (!launch_matmul_i8_compute(
        reinterpret_cast<const int8_t*>(handle_to_ptr(a_handle)),
        reinterpret_cast<const int8_t*>(handle_to_ptr(b_handle)),
        f32_output,
        m,
        n,
        k,
        a_scale * b_scale,
        "CUDA I8 typed-output matmul f32 kernel launch failed")) {
        return 1;
    }

    return quantize_f32_output_to_i8(
        f32_output,
        reinterpret_cast<int8_t*>(handle_to_ptr(out_handle)),
        total,
        out_scale,
        "failed to allocate CUDA I8 typed-output matmul reduction buffer",
        "CUDA I8 typed-output matmul absmax kernel launch failed",
        "CUDA I8 typed-output matmul final absmax reduction kernel launch failed",
        "CUDA I8 typed-output matmul kernel launch failed")
        ? 0
        : 1;
}

extern "C" int lumen_cuda_batch_matmul_f32_device(
    uint64_t lhs_handle,
    uint64_t rhs_handle,
    uint64_t out_handle,
    size_t batch_count,
    size_t m,
    size_t n,
    size_t k) {
    if (!validate_handle(lhs_handle, "CUDA batch_matmul lhs handle") ||
        !validate_handle(rhs_handle, "CUDA batch_matmul rhs handle") ||
        !validate_handle(out_handle, "CUDA batch_matmul output handle")) {
        return 1;
    }
    if (!validate_cublas_batch_count(batch_count) || !validate_dims(m, n, k)) {
        return 1;
    }

    CublasHandle handle;
    if (!init_cublas(handle)) {
        return 1;
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasStatus_t cublas_status = cublasSgemmStridedBatched(
        handle.handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        static_cast<int>(n),
        static_cast<int>(m),
        static_cast<int>(k),
        &alpha,
        handle_to_ptr(rhs_handle),
        static_cast<int>(n),
        static_cast<long long>(n * k),
        handle_to_ptr(lhs_handle),
        static_cast<int>(k),
        static_cast<long long>(m * k),
        &beta,
        handle_to_ptr(out_handle),
        static_cast<int>(n),
        static_cast<long long>(m * n),
        static_cast<int>(batch_count));
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        set_cublas_error("cuBLAS strided batched SGEMM failed", cublas_status);
        return 1;
    }
    return 0;
}

extern "C" int lumen_cuda_batch_matmul_bf16_device(
    uint64_t lhs_handle,
    uint64_t rhs_handle,
    uint64_t out_handle,
    size_t batch_count,
    size_t m,
    size_t n,
    size_t k) {
    if (!validate_handle(lhs_handle, "CUDA BF16 batch_matmul lhs handle") ||
        !validate_handle(rhs_handle, "CUDA BF16 batch_matmul rhs handle") ||
        !validate_handle(out_handle, "CUDA BF16 batch_matmul output handle")) {
        return 1;
    }
    if (!validate_cublas_batch_count(batch_count) || !validate_dims(m, n, k)) {
        return 1;
    }

    CublasHandle handle;
    if (!init_cublas(handle)) {
        return 1;
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasStatus_t cublas_status = cublasGemmStridedBatchedEx(
        handle.handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        static_cast<int>(n),
        static_cast<int>(m),
        static_cast<int>(k),
        &alpha,
        reinterpret_cast<const __nv_bfloat16*>(handle_to_ptr(rhs_handle)),
        CUDA_R_16BF,
        static_cast<int>(n),
        static_cast<long long>(n * k),
        reinterpret_cast<const __nv_bfloat16*>(handle_to_ptr(lhs_handle)),
        CUDA_R_16BF,
        static_cast<int>(k),
        static_cast<long long>(m * k),
        &beta,
        handle_to_ptr(out_handle),
        CUDA_R_32F,
        static_cast<int>(n),
        static_cast<long long>(m * n),
        static_cast<int>(batch_count),
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        set_cublas_error("cuBLAS BF16 strided batched GEMM failed", cublas_status);
        return 1;
    }
    return 0;
}

extern "C" int lumen_cuda_batch_matmul_bf16_typed_out_device(
    uint64_t lhs_handle,
    uint64_t rhs_handle,
    uint64_t out_handle,
    size_t batch_count,
    size_t m,
    size_t n,
    size_t k) {
    if (!validate_handle(lhs_handle, "CUDA BF16 typed-out batch_matmul lhs handle") ||
        !validate_handle(rhs_handle, "CUDA BF16 typed-out batch_matmul rhs handle") ||
        !validate_handle(out_handle, "CUDA BF16 typed-out batch_matmul output handle")) {
        return 1;
    }
    if (!validate_cublas_batch_count(batch_count) || !validate_dims(m, n, k)) {
        return 1;
    }

    CublasHandle handle;
    if (!init_cublas(handle)) {
        return 1;
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasStatus_t cublas_status = cublasGemmStridedBatchedEx(
        handle.handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        static_cast<int>(n),
        static_cast<int>(m),
        static_cast<int>(k),
        &alpha,
        reinterpret_cast<const __nv_bfloat16*>(handle_to_ptr(rhs_handle)),
        CUDA_R_16BF,
        static_cast<int>(n),
        static_cast<long long>(n * k),
        reinterpret_cast<const __nv_bfloat16*>(handle_to_ptr(lhs_handle)),
        CUDA_R_16BF,
        static_cast<int>(k),
        static_cast<long long>(m * k),
        &beta,
        reinterpret_cast<__nv_bfloat16*>(handle_to_ptr(out_handle)),
        CUDA_R_16BF,
        static_cast<int>(n),
        static_cast<long long>(m * n),
        static_cast<int>(batch_count),
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        set_cublas_error("cuBLAS BF16 typed-output strided batched GEMM failed", cublas_status);
        return 1;
    }
    return 0;
}

extern "C" int lumen_cuda_batch_matmul_f16_device(
    uint64_t lhs_handle,
    uint64_t rhs_handle,
    uint64_t out_handle,
    size_t batch_count,
    size_t m,
    size_t n,
    size_t k) {
    if (!validate_handle(lhs_handle, "CUDA F16 batch_matmul lhs handle") ||
        !validate_handle(rhs_handle, "CUDA F16 batch_matmul rhs handle") ||
        !validate_handle(out_handle, "CUDA F16 batch_matmul output handle")) {
        return 1;
    }
    if (!validate_cublas_batch_count(batch_count) || !validate_dims(m, n, k)) {
        return 1;
    }

    CublasHandle handle;
    if (!init_cublas(handle)) {
        return 1;
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasStatus_t cublas_status = cublasGemmStridedBatchedEx(
        handle.handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        static_cast<int>(n),
        static_cast<int>(m),
        static_cast<int>(k),
        &alpha,
        reinterpret_cast<const __half*>(handle_to_ptr(rhs_handle)),
        CUDA_R_16F,
        static_cast<int>(n),
        static_cast<long long>(n * k),
        reinterpret_cast<const __half*>(handle_to_ptr(lhs_handle)),
        CUDA_R_16F,
        static_cast<int>(k),
        static_cast<long long>(m * k),
        &beta,
        handle_to_ptr(out_handle),
        CUDA_R_32F,
        static_cast<int>(n),
        static_cast<long long>(m * n),
        static_cast<int>(batch_count),
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        set_cublas_error("cuBLAS F16 strided batched GEMM failed", cublas_status);
        return 1;
    }
    return 0;
}

extern "C" int lumen_cuda_batch_matmul_f16_typed_out_device(
    uint64_t lhs_handle,
    uint64_t rhs_handle,
    uint64_t out_handle,
    size_t batch_count,
    size_t m,
    size_t n,
    size_t k) {
    if (!validate_handle(lhs_handle, "CUDA F16 typed-out batch_matmul lhs handle") ||
        !validate_handle(rhs_handle, "CUDA F16 typed-out batch_matmul rhs handle") ||
        !validate_handle(out_handle, "CUDA F16 typed-out batch_matmul output handle")) {
        return 1;
    }
    if (!validate_cublas_batch_count(batch_count) || !validate_dims(m, n, k)) {
        return 1;
    }

    CublasHandle handle;
    if (!init_cublas(handle)) {
        return 1;
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasStatus_t cublas_status = cublasGemmStridedBatchedEx(
        handle.handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        static_cast<int>(n),
        static_cast<int>(m),
        static_cast<int>(k),
        &alpha,
        reinterpret_cast<const __half*>(handle_to_ptr(rhs_handle)),
        CUDA_R_16F,
        static_cast<int>(n),
        static_cast<long long>(n * k),
        reinterpret_cast<const __half*>(handle_to_ptr(lhs_handle)),
        CUDA_R_16F,
        static_cast<int>(k),
        static_cast<long long>(m * k),
        &beta,
        reinterpret_cast<__half*>(handle_to_ptr(out_handle)),
        CUDA_R_16F,
        static_cast<int>(n),
        static_cast<long long>(m * n),
        static_cast<int>(batch_count),
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        set_cublas_error("cuBLAS F16 typed-output strided batched GEMM failed", cublas_status);
        return 1;
    }
    return 0;
}

extern "C" int lumen_cuda_batch_matmul_bf16_i8_device(
    uint64_t lhs_handle,
    uint64_t rhs_handle,
    float rhs_scale,
    uint64_t out_handle,
    size_t batch_count,
    size_t m,
    size_t n,
    size_t k) {
    if (!validate_handle(lhs_handle, "CUDA BF16xI8 batch_matmul lhs handle") ||
        !validate_handle(rhs_handle, "CUDA BF16xI8 batch_matmul rhs handle") ||
        !validate_handle(out_handle, "CUDA BF16xI8 batch_matmul output handle")) {
        return 1;
    }
    if (batch_count == 0) {
        set_error("CUDA BF16xI8 batch_matmul batch_count must be greater than zero");
        return 1;
    }
    if (!validate_dims(m, n, k)) {
        return 1;
    }
    if (!std::isfinite(rhs_scale) || rhs_scale <= 0.0f) {
        set_error("CUDA BF16xI8 batch_matmul scale must be finite and > 0");
        return 1;
    }
    const size_t max_size = static_cast<size_t>(-1);
    if (batch_count > max_size / m || batch_count * m > max_size / n) {
        set_error("CUDA BF16xI8 batch_matmul output length overflow");
        return 1;
    }

    return launch_batch_matmul_floatlike_i8_compute(
               reinterpret_cast<const __nv_bfloat16*>(handle_to_ptr(lhs_handle)),
               reinterpret_cast<const int8_t*>(handle_to_ptr(rhs_handle)),
               handle_to_ptr(out_handle),
               batch_count,
               m,
               n,
               k,
               rhs_scale,
               "CUDA BF16xI8 batch_matmul kernel failed")
               ? 0
               : 1;
}

extern "C" int lumen_cuda_batch_matmul_f16_i8_device(
    uint64_t lhs_handle,
    uint64_t rhs_handle,
    float rhs_scale,
    uint64_t out_handle,
    size_t batch_count,
    size_t m,
    size_t n,
    size_t k) {
    if (!validate_handle(lhs_handle, "CUDA F16xI8 batch_matmul lhs handle") ||
        !validate_handle(rhs_handle, "CUDA F16xI8 batch_matmul rhs handle") ||
        !validate_handle(out_handle, "CUDA F16xI8 batch_matmul output handle")) {
        return 1;
    }
    if (batch_count == 0) {
        set_error("CUDA F16xI8 batch_matmul batch_count must be greater than zero");
        return 1;
    }
    if (!validate_dims(m, n, k)) {
        return 1;
    }
    if (!std::isfinite(rhs_scale) || rhs_scale <= 0.0f) {
        set_error("CUDA F16xI8 batch_matmul scale must be finite and > 0");
        return 1;
    }
    const size_t max_size = static_cast<size_t>(-1);
    if (batch_count > max_size / m || batch_count * m > max_size / n) {
        set_error("CUDA F16xI8 batch_matmul output length overflow");
        return 1;
    }

    return launch_batch_matmul_floatlike_i8_compute(
               reinterpret_cast<const __half*>(handle_to_ptr(lhs_handle)),
               reinterpret_cast<const int8_t*>(handle_to_ptr(rhs_handle)),
               handle_to_ptr(out_handle),
               batch_count,
               m,
               n,
               k,
               rhs_scale,
               "CUDA F16xI8 batch_matmul kernel failed")
               ? 0
               : 1;
}

extern "C" int lumen_cuda_batch_matmul_f32_i8_device(
    uint64_t lhs_handle,
    uint64_t rhs_handle,
    float rhs_scale,
    uint64_t out_handle,
    size_t batch_count,
    size_t m,
    size_t n,
    size_t k) {
    if (!validate_handle(lhs_handle, "CUDA F32xI8 batch_matmul lhs handle") ||
        !validate_handle(rhs_handle, "CUDA F32xI8 batch_matmul rhs handle") ||
        !validate_handle(out_handle, "CUDA F32xI8 batch_matmul output handle")) {
        return 1;
    }
    if (batch_count == 0) {
        set_error("CUDA F32xI8 batch_matmul batch_count must be greater than zero");
        return 1;
    }
    if (!validate_dims(m, n, k)) {
        return 1;
    }
    if (!std::isfinite(rhs_scale) || rhs_scale <= 0.0f) {
        set_error("CUDA F32xI8 batch_matmul scale must be finite and > 0");
        return 1;
    }
    const size_t max_size = static_cast<size_t>(-1);
    if (batch_count > max_size / m || batch_count * m > max_size / n) {
        set_error("CUDA F32xI8 batch_matmul output length overflow");
        return 1;
    }

    return launch_batch_matmul_floatlike_i8_compute(
               handle_to_ptr(lhs_handle),
               reinterpret_cast<const int8_t*>(handle_to_ptr(rhs_handle)),
               handle_to_ptr(out_handle),
               batch_count,
               m,
               n,
               k,
               rhs_scale,
               "CUDA F32xI8 batch_matmul kernel failed")
               ? 0
               : 1;
}

extern "C" int lumen_cuda_batch_matmul_i8_bf16_device(
    uint64_t lhs_handle,
    float lhs_scale,
    uint64_t rhs_handle,
    uint64_t out_handle,
    size_t batch_count,
    size_t m,
    size_t n,
    size_t k) {
    if (!validate_handle(lhs_handle, "CUDA I8xBF16 batch_matmul lhs handle") ||
        !validate_handle(rhs_handle, "CUDA I8xBF16 batch_matmul rhs handle") ||
        !validate_handle(out_handle, "CUDA I8xBF16 batch_matmul output handle")) {
        return 1;
    }
    if (batch_count == 0) {
        set_error("CUDA I8xBF16 batch_matmul batch_count must be greater than zero");
        return 1;
    }
    if (!validate_dims(m, n, k)) {
        return 1;
    }
    if (!std::isfinite(lhs_scale) || lhs_scale <= 0.0f) {
        set_error("CUDA I8xBF16 batch_matmul scale must be finite and > 0");
        return 1;
    }
    const size_t max_size = static_cast<size_t>(-1);
    if (batch_count > max_size / m || batch_count * m > max_size / n) {
        set_error("CUDA I8xBF16 batch_matmul output length overflow");
        return 1;
    }

    return launch_batch_matmul_i8_floatlike_compute(
               reinterpret_cast<const int8_t*>(handle_to_ptr(lhs_handle)),
               reinterpret_cast<const __nv_bfloat16*>(handle_to_ptr(rhs_handle)),
               handle_to_ptr(out_handle),
               batch_count,
               m,
               n,
               k,
               lhs_scale,
               "CUDA I8xBF16 batch_matmul kernel failed")
               ? 0
               : 1;
}

extern "C" int lumen_cuda_batch_matmul_i8_f16_device(
    uint64_t lhs_handle,
    float lhs_scale,
    uint64_t rhs_handle,
    uint64_t out_handle,
    size_t batch_count,
    size_t m,
    size_t n,
    size_t k) {
    if (!validate_handle(lhs_handle, "CUDA I8xF16 batch_matmul lhs handle") ||
        !validate_handle(rhs_handle, "CUDA I8xF16 batch_matmul rhs handle") ||
        !validate_handle(out_handle, "CUDA I8xF16 batch_matmul output handle")) {
        return 1;
    }
    if (batch_count == 0) {
        set_error("CUDA I8xF16 batch_matmul batch_count must be greater than zero");
        return 1;
    }
    if (!validate_dims(m, n, k)) {
        return 1;
    }
    if (!std::isfinite(lhs_scale) || lhs_scale <= 0.0f) {
        set_error("CUDA I8xF16 batch_matmul scale must be finite and > 0");
        return 1;
    }
    const size_t max_size = static_cast<size_t>(-1);
    if (batch_count > max_size / m || batch_count * m > max_size / n) {
        set_error("CUDA I8xF16 batch_matmul output length overflow");
        return 1;
    }

    return launch_batch_matmul_i8_floatlike_compute(
               reinterpret_cast<const int8_t*>(handle_to_ptr(lhs_handle)),
               reinterpret_cast<const __half*>(handle_to_ptr(rhs_handle)),
               handle_to_ptr(out_handle),
               batch_count,
               m,
               n,
               k,
               lhs_scale,
               "CUDA I8xF16 batch_matmul kernel failed")
               ? 0
               : 1;
}

extern "C" int lumen_cuda_batch_matmul_i8_f32_device(
    uint64_t lhs_handle,
    float lhs_scale,
    uint64_t rhs_handle,
    uint64_t out_handle,
    size_t batch_count,
    size_t m,
    size_t n,
    size_t k) {
    if (!validate_handle(lhs_handle, "CUDA I8xF32 batch_matmul lhs handle") ||
        !validate_handle(rhs_handle, "CUDA I8xF32 batch_matmul rhs handle") ||
        !validate_handle(out_handle, "CUDA I8xF32 batch_matmul output handle")) {
        return 1;
    }
    if (batch_count == 0) {
        set_error("CUDA I8xF32 batch_matmul batch_count must be greater than zero");
        return 1;
    }
    if (!validate_dims(m, n, k)) {
        return 1;
    }
    if (!std::isfinite(lhs_scale) || lhs_scale <= 0.0f) {
        set_error("CUDA I8xF32 batch_matmul scale must be finite and > 0");
        return 1;
    }
    const size_t max_size = static_cast<size_t>(-1);
    if (batch_count > max_size / m || batch_count * m > max_size / n) {
        set_error("CUDA I8xF32 batch_matmul output length overflow");
        return 1;
    }

    return launch_batch_matmul_i8_floatlike_compute(
               reinterpret_cast<const int8_t*>(handle_to_ptr(lhs_handle)),
               handle_to_ptr(rhs_handle),
               handle_to_ptr(out_handle),
               batch_count,
               m,
               n,
               k,
               lhs_scale,
               "CUDA I8xF32 batch_matmul kernel failed")
               ? 0
               : 1;
}

extern "C" int lumen_cuda_batch_matmul_i8_device(
    uint64_t lhs_handle,
    float lhs_scale,
    uint64_t rhs_handle,
    float rhs_scale,
    uint64_t out_handle,
    size_t batch_count,
    size_t m,
    size_t n,
    size_t k) {
    if (!validate_handle(lhs_handle, "CUDA I8 batch_matmul lhs handle") ||
        !validate_handle(rhs_handle, "CUDA I8 batch_matmul rhs handle") ||
        !validate_handle(out_handle, "CUDA I8 batch_matmul output handle")) {
        return 1;
    }
    if (batch_count == 0) {
        set_error("CUDA I8 batch_matmul batch_count must be greater than zero");
        return 1;
    }
    if (!validate_dims(m, n, k)) {
        return 1;
    }
    if (!std::isfinite(lhs_scale) || lhs_scale <= 0.0f ||
        !std::isfinite(rhs_scale) || rhs_scale <= 0.0f) {
        set_error("CUDA I8 batch_matmul scales must be finite and > 0");
        return 1;
    }
    const size_t max_size = static_cast<size_t>(-1);
    if (batch_count > max_size / m || batch_count * m > max_size / n) {
        set_error("CUDA I8 batch_matmul output length overflow");
        return 1;
    }

    return launch_batch_matmul_i8_compute(
               reinterpret_cast<const int8_t*>(handle_to_ptr(lhs_handle)),
               reinterpret_cast<const int8_t*>(handle_to_ptr(rhs_handle)),
               handle_to_ptr(out_handle),
               batch_count,
               m,
               n,
               k,
               lhs_scale * rhs_scale,
               "CUDA I8 batch_matmul kernel launch failed")
               ? 0
               : 1;
}

extern "C" int lumen_cuda_batch_matmul_i8_typed_out_device(
    uint64_t lhs_handle,
    float lhs_scale,
    uint64_t rhs_handle,
    float rhs_scale,
    uint64_t out_handle,
    size_t batch_count,
    size_t m,
    size_t n,
    size_t k,
    float* out_scale) {
    if (!validate_handle(lhs_handle, "CUDA I8 typed-output batch_matmul lhs handle") ||
        !validate_handle(rhs_handle, "CUDA I8 typed-output batch_matmul rhs handle") ||
        !validate_handle(out_handle, "CUDA I8 typed-output batch_matmul output handle")) {
        return 1;
    }
    if (out_scale == nullptr) {
        set_error("CUDA I8 typed-output batch_matmul scale output is null");
        return 1;
    }
    if (batch_count == 0) {
        set_error("CUDA I8 typed-output batch_matmul batch_count must be greater than zero");
        return 1;
    }
    if (!validate_dims(m, n, k)) {
        return 1;
    }
    if (!std::isfinite(lhs_scale) || lhs_scale <= 0.0f ||
        !std::isfinite(rhs_scale) || rhs_scale <= 0.0f) {
        set_error("CUDA I8 typed-output batch_matmul scales must be finite and > 0");
        return 1;
    }
    const size_t max_size = static_cast<size_t>(-1);
    if (batch_count > max_size / m || batch_count * m > max_size / n) {
        set_error("CUDA I8 typed-output batch_matmul output length overflow");
        return 1;
    }

    if (batch_count * m * n > max_size / sizeof(float)) {
        set_error("CUDA I8 typed-output batch_matmul temporary output length overflow");
        return 1;
    }
    const size_t total = batch_count * m * n;
    thread_local ReusableCudaWorkspace f32_output_tmp;
    if (!f32_output_tmp.ensure(
            total * sizeof(float),
            "failed to allocate CUDA I8 typed-output batch_matmul f32 output buffer")) {
        return 1;
    }
    const int8_t* lhs = reinterpret_cast<const int8_t*>(handle_to_ptr(lhs_handle));
    const int8_t* rhs = reinterpret_cast<const int8_t*>(handle_to_ptr(rhs_handle));
    float* f32_output = static_cast<float*>(f32_output_tmp.ptr);
    if (!launch_batch_matmul_i8_compute(
            lhs,
            rhs,
            f32_output,
            batch_count,
            m,
            n,
            k,
            lhs_scale * rhs_scale,
            "CUDA I8 typed-output batch_matmul f32 kernel launch failed")) {
        return 1;
    }

    return quantize_f32_output_to_i8(
        f32_output,
        reinterpret_cast<int8_t*>(handle_to_ptr(out_handle)),
        total,
        out_scale,
        "failed to allocate CUDA I8 typed-output batch_matmul reduction buffer",
        "CUDA I8 typed-output batch_matmul absmax kernel launch failed",
        "CUDA I8 typed-output batch_matmul final absmax reduction kernel launch failed",
        "CUDA I8 typed-output batch_matmul kernel launch failed")
        ? 0
        : 1;
}

extern "C" int lumen_cuda_matmul_backward_f32_device(
    uint64_t grad_handle,
    uint64_t a_handle,
    uint64_t b_handle,
    uint64_t da_handle,
    uint64_t db_handle,
    size_t m,
    size_t k,
    size_t n) {
    if (!validate_handle(grad_handle, "CUDA matmul backward grad handle") ||
        !validate_handle(a_handle, "CUDA matmul backward A handle") ||
        !validate_handle(b_handle, "CUDA matmul backward B handle") ||
        !validate_handle(da_handle, "CUDA matmul backward dA handle") ||
        !validate_handle(db_handle, "CUDA matmul backward dB handle")) {
        return 1;
    }
    if (!validate_dims(m, n, k)) {
        return 1;
    }

    CublasHandle handle;
    if (!init_cublas(handle)) {
        return 1;
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    // Row-major dA[m,k] = grad[m,n] * B[n,k].
    // As column-major, dA^T[k,m] = B^T[k,n] * grad^T[n,m].
    cublasStatus_t cublas_status = cublasSgemm(
        handle.handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        static_cast<int>(k),
        static_cast<int>(m),
        static_cast<int>(n),
        &alpha,
        handle_to_ptr(b_handle),
        static_cast<int>(k),
        handle_to_ptr(grad_handle),
        static_cast<int>(n),
        &beta,
        handle_to_ptr(da_handle),
        static_cast<int>(k));
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        set_cublas_error("cuBLAS SGEMM failed for matmul backward dA", cublas_status);
        return 1;
    }

    // Row-major dB[n,k] = grad^T[n,m] * A[m,k].
    // As column-major, dB^T[k,n] = A^T[k,m] * grad[m,n].
    cublas_status = cublasSgemm(
        handle.handle,
        CUBLAS_OP_N,
        CUBLAS_OP_T,
        static_cast<int>(k),
        static_cast<int>(n),
        static_cast<int>(m),
        &alpha,
        handle_to_ptr(a_handle),
        static_cast<int>(k),
        handle_to_ptr(grad_handle),
        static_cast<int>(n),
        &beta,
        handle_to_ptr(db_handle),
        static_cast<int>(k));
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        set_cublas_error("cuBLAS SGEMM failed for matmul backward dB", cublas_status);
        return 1;
    }
    return 0;
}

extern "C" int lumen_cuda_matmul_backward_bf16_i8_device(
    uint64_t grad_handle,
    uint64_t lhs_handle,
    uint64_t rhs_handle,
    float rhs_scale,
    uint64_t d_lhs_handle,
    uint64_t d_rhs_handle,
    size_t m,
    size_t k,
    size_t n) {
    if (!validate_handle(grad_handle, "CUDA BF16xI8 matmul backward grad handle") ||
        !validate_handle(lhs_handle, "CUDA BF16xI8 matmul backward lhs handle") ||
        !validate_handle(rhs_handle, "CUDA BF16xI8 matmul backward rhs handle") ||
        !validate_handle(d_lhs_handle, "CUDA BF16xI8 matmul backward d_lhs handle") ||
        !validate_handle(d_rhs_handle, "CUDA BF16xI8 matmul backward d_rhs handle")) {
        return 1;
    }
    if (!validate_dims(m, n, k)) {
        return 1;
    }
    if (!std::isfinite(rhs_scale) || rhs_scale <= 0.0f) {
        set_error("CUDA BF16xI8 matmul backward scale must be finite and > 0");
        return 1;
    }

    size_t lhs_total = m * k;
    size_t rhs_total = n * k;
    constexpr int block_size = 256;
    const size_t total = lhs_total > rhs_total ? lhs_total : rhs_total;
    const unsigned int grid = linear_grid_size(total, block_size);
    matmul_lhs_i8_backward_kernel<__nv_bfloat16><<<grid, block_size>>>(
        handle_to_ptr(grad_handle),
        reinterpret_cast<const __nv_bfloat16*>(handle_to_ptr(lhs_handle)),
        reinterpret_cast<const int8_t*>(handle_to_ptr(rhs_handle)),
        rhs_scale,
        handle_to_ptr(d_lhs_handle),
        handle_to_ptr(d_rhs_handle),
        lhs_total,
        rhs_total,
        m,
        k,
        n);
    return check_cuda_launch("CUDA BF16xI8 matmul backward kernel launch failed") ? 0 : 1;
}

extern "C" int lumen_cuda_matmul_backward_f16_i8_device(
    uint64_t grad_handle,
    uint64_t lhs_handle,
    uint64_t rhs_handle,
    float rhs_scale,
    uint64_t d_lhs_handle,
    uint64_t d_rhs_handle,
    size_t m,
    size_t k,
    size_t n) {
    if (!validate_handle(grad_handle, "CUDA F16xI8 matmul backward grad handle") ||
        !validate_handle(lhs_handle, "CUDA F16xI8 matmul backward lhs handle") ||
        !validate_handle(rhs_handle, "CUDA F16xI8 matmul backward rhs handle") ||
        !validate_handle(d_lhs_handle, "CUDA F16xI8 matmul backward d_lhs handle") ||
        !validate_handle(d_rhs_handle, "CUDA F16xI8 matmul backward d_rhs handle")) {
        return 1;
    }
    if (!validate_dims(m, n, k)) {
        return 1;
    }
    if (!std::isfinite(rhs_scale) || rhs_scale <= 0.0f) {
        set_error("CUDA F16xI8 matmul backward scale must be finite and > 0");
        return 1;
    }

    size_t lhs_total = m * k;
    size_t rhs_total = n * k;
    constexpr int block_size = 256;
    const size_t total = lhs_total > rhs_total ? lhs_total : rhs_total;
    const unsigned int grid = linear_grid_size(total, block_size);
    matmul_lhs_i8_backward_kernel<__half><<<grid, block_size>>>(
        handle_to_ptr(grad_handle),
        reinterpret_cast<const __half*>(handle_to_ptr(lhs_handle)),
        reinterpret_cast<const int8_t*>(handle_to_ptr(rhs_handle)),
        rhs_scale,
        handle_to_ptr(d_lhs_handle),
        handle_to_ptr(d_rhs_handle),
        lhs_total,
        rhs_total,
        m,
        k,
        n);
    return check_cuda_launch("CUDA F16xI8 matmul backward kernel launch failed") ? 0 : 1;
}

extern "C" int lumen_cuda_matmul_backward_f32_i8_device(
    uint64_t grad_handle,
    uint64_t lhs_handle,
    uint64_t rhs_handle,
    float rhs_scale,
    uint64_t d_lhs_handle,
    uint64_t d_rhs_handle,
    size_t m,
    size_t k,
    size_t n) {
    if (!validate_handle(grad_handle, "CUDA F32xI8 matmul backward grad handle") ||
        !validate_handle(lhs_handle, "CUDA F32xI8 matmul backward lhs handle") ||
        !validate_handle(rhs_handle, "CUDA F32xI8 matmul backward rhs handle") ||
        !validate_handle(d_lhs_handle, "CUDA F32xI8 matmul backward d_lhs handle") ||
        !validate_handle(d_rhs_handle, "CUDA F32xI8 matmul backward d_rhs handle")) {
        return 1;
    }
    if (!validate_dims(m, n, k)) {
        return 1;
    }
    if (!std::isfinite(rhs_scale) || rhs_scale <= 0.0f) {
        set_error("CUDA F32xI8 matmul backward scale must be finite and > 0");
        return 1;
    }

    size_t lhs_total = m * k;
    size_t rhs_total = n * k;
    constexpr int block_size = 256;
    const size_t total = lhs_total > rhs_total ? lhs_total : rhs_total;
    const unsigned int grid = linear_grid_size(total, block_size);
    matmul_lhs_i8_backward_kernel<float><<<grid, block_size>>>(
        handle_to_ptr(grad_handle),
        handle_to_ptr(lhs_handle),
        reinterpret_cast<const int8_t*>(handle_to_ptr(rhs_handle)),
        rhs_scale,
        handle_to_ptr(d_lhs_handle),
        handle_to_ptr(d_rhs_handle),
        lhs_total,
        rhs_total,
        m,
        k,
        n);
    return check_cuda_launch("CUDA F32xI8 matmul backward kernel launch failed") ? 0 : 1;
}

extern "C" int lumen_cuda_matmul_backward_i8_bf16_device(
    uint64_t grad_handle,
    uint64_t lhs_handle,
    float lhs_scale,
    uint64_t rhs_handle,
    uint64_t d_lhs_handle,
    uint64_t d_rhs_handle,
    size_t m,
    size_t k,
    size_t n) {
    if (!validate_handle(grad_handle, "CUDA I8xBF16 matmul backward grad handle") ||
        !validate_handle(lhs_handle, "CUDA I8xBF16 matmul backward lhs handle") ||
        !validate_handle(rhs_handle, "CUDA I8xBF16 matmul backward rhs handle") ||
        !validate_handle(d_lhs_handle, "CUDA I8xBF16 matmul backward d_lhs handle") ||
        !validate_handle(d_rhs_handle, "CUDA I8xBF16 matmul backward d_rhs handle")) {
        return 1;
    }
    if (!validate_dims(m, n, k)) {
        return 1;
    }
    if (!std::isfinite(lhs_scale) || lhs_scale <= 0.0f) {
        set_error("CUDA I8xBF16 matmul backward scale must be finite and > 0");
        return 1;
    }

    size_t lhs_total = m * k;
    size_t rhs_total = n * k;
    constexpr int block_size = 256;
    const size_t total = lhs_total > rhs_total ? lhs_total : rhs_total;
    const unsigned int grid = linear_grid_size(total, block_size);
    matmul_i8_rhs_floatlike_backward_kernel<__nv_bfloat16><<<grid, block_size>>>(
        handle_to_ptr(grad_handle),
        reinterpret_cast<const int8_t*>(handle_to_ptr(lhs_handle)),
        lhs_scale,
        reinterpret_cast<const __nv_bfloat16*>(handle_to_ptr(rhs_handle)),
        handle_to_ptr(d_lhs_handle),
        handle_to_ptr(d_rhs_handle),
        lhs_total,
        rhs_total,
        m,
        k,
        n);
    return check_cuda_launch("CUDA I8xBF16 matmul backward kernel launch failed") ? 0 : 1;
}

extern "C" int lumen_cuda_matmul_backward_i8_f16_device(
    uint64_t grad_handle,
    uint64_t lhs_handle,
    float lhs_scale,
    uint64_t rhs_handle,
    uint64_t d_lhs_handle,
    uint64_t d_rhs_handle,
    size_t m,
    size_t k,
    size_t n) {
    if (!validate_handle(grad_handle, "CUDA I8xF16 matmul backward grad handle") ||
        !validate_handle(lhs_handle, "CUDA I8xF16 matmul backward lhs handle") ||
        !validate_handle(rhs_handle, "CUDA I8xF16 matmul backward rhs handle") ||
        !validate_handle(d_lhs_handle, "CUDA I8xF16 matmul backward d_lhs handle") ||
        !validate_handle(d_rhs_handle, "CUDA I8xF16 matmul backward d_rhs handle")) {
        return 1;
    }
    if (!validate_dims(m, n, k)) {
        return 1;
    }
    if (!std::isfinite(lhs_scale) || lhs_scale <= 0.0f) {
        set_error("CUDA I8xF16 matmul backward scale must be finite and > 0");
        return 1;
    }

    size_t lhs_total = m * k;
    size_t rhs_total = n * k;
    constexpr int block_size = 256;
    const size_t total = lhs_total > rhs_total ? lhs_total : rhs_total;
    const unsigned int grid = linear_grid_size(total, block_size);
    matmul_i8_rhs_floatlike_backward_kernel<__half><<<grid, block_size>>>(
        handle_to_ptr(grad_handle),
        reinterpret_cast<const int8_t*>(handle_to_ptr(lhs_handle)),
        lhs_scale,
        reinterpret_cast<const __half*>(handle_to_ptr(rhs_handle)),
        handle_to_ptr(d_lhs_handle),
        handle_to_ptr(d_rhs_handle),
        lhs_total,
        rhs_total,
        m,
        k,
        n);
    return check_cuda_launch("CUDA I8xF16 matmul backward kernel launch failed") ? 0 : 1;
}

extern "C" int lumen_cuda_matmul_backward_i8_f32_device(
    uint64_t grad_handle,
    uint64_t lhs_handle,
    float lhs_scale,
    uint64_t rhs_handle,
    uint64_t d_lhs_handle,
    uint64_t d_rhs_handle,
    size_t m,
    size_t k,
    size_t n) {
    if (!validate_handle(grad_handle, "CUDA I8xF32 matmul backward grad handle") ||
        !validate_handle(lhs_handle, "CUDA I8xF32 matmul backward lhs handle") ||
        !validate_handle(rhs_handle, "CUDA I8xF32 matmul backward rhs handle") ||
        !validate_handle(d_lhs_handle, "CUDA I8xF32 matmul backward d_lhs handle") ||
        !validate_handle(d_rhs_handle, "CUDA I8xF32 matmul backward d_rhs handle")) {
        return 1;
    }
    if (!validate_dims(m, n, k)) {
        return 1;
    }
    if (!std::isfinite(lhs_scale) || lhs_scale <= 0.0f) {
        set_error("CUDA I8xF32 matmul backward scale must be finite and > 0");
        return 1;
    }

    size_t lhs_total = m * k;
    size_t rhs_total = n * k;
    constexpr int block_size = 256;
    const size_t total = lhs_total > rhs_total ? lhs_total : rhs_total;
    const unsigned int grid = linear_grid_size(total, block_size);
    matmul_i8_rhs_floatlike_backward_kernel<float><<<grid, block_size>>>(
        handle_to_ptr(grad_handle),
        reinterpret_cast<const int8_t*>(handle_to_ptr(lhs_handle)),
        lhs_scale,
        handle_to_ptr(rhs_handle),
        handle_to_ptr(d_lhs_handle),
        handle_to_ptr(d_rhs_handle),
        lhs_total,
        rhs_total,
        m,
        k,
        n);
    return check_cuda_launch("CUDA I8xF32 matmul backward kernel launch failed") ? 0 : 1;
}

static bool validate_batch_matmul_backward_sizes(
    size_t batch_count,
    size_t m,
    size_t k,
    size_t n) {
    constexpr size_t max_cublas_int = 2147483647;
    const size_t max_size = static_cast<size_t>(-1);
    if (batch_count > max_cublas_int) {
        set_error("CUDA batch_matmul backward batch_count exceeds supported range");
        return false;
    }
    if (batch_count > max_size / m || batch_count * m > max_size / k ||
        batch_count * m > max_size / n || batch_count > max_size / k ||
        batch_count * k > max_size / n) {
        set_error("CUDA batch_matmul backward tensor length overflow");
        return false;
    }
    return true;
}

extern "C" int lumen_cuda_batch_matmul_backward_f32_device(
    uint64_t grad_handle,
    uint64_t lhs_handle,
    uint64_t rhs_handle,
    uint64_t d_lhs_handle,
    uint64_t d_rhs_handle,
    size_t batch_count,
    size_t m,
    size_t k,
    size_t n) {
    if (!validate_handle(grad_handle, "CUDA batch_matmul backward grad handle") ||
        !validate_handle(lhs_handle, "CUDA batch_matmul backward lhs handle") ||
        !validate_handle(rhs_handle, "CUDA batch_matmul backward rhs handle") ||
        !validate_handle(d_lhs_handle, "CUDA batch_matmul backward d_lhs handle") ||
        !validate_handle(d_rhs_handle, "CUDA batch_matmul backward d_rhs handle")) {
        return 1;
    }
    if (!validate_cublas_batch_count(batch_count) || !validate_dims(m, n, k)) {
        return 1;
    }

    if (!validate_batch_matmul_backward_sizes(batch_count, m, k, n)) {
        return 1;
    }

    CublasHandle handle;
    if (!init_cublas(handle)) {
        return 1;
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasStatus_t cublas_status = cublasSgemmStridedBatched(
        handle.handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        static_cast<int>(k),
        static_cast<int>(m),
        static_cast<int>(n),
        &alpha,
        handle_to_ptr(rhs_handle),
        static_cast<int>(n),
        static_cast<long long>(k * n),
        handle_to_ptr(grad_handle),
        static_cast<int>(n),
        static_cast<long long>(m * n),
        &beta,
        handle_to_ptr(d_lhs_handle),
        static_cast<int>(k),
        static_cast<long long>(m * k),
        static_cast<int>(batch_count));
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        set_cublas_error("cuBLAS strided batched SGEMM failed for batch_matmul backward d_lhs", cublas_status);
        return 1;
    }

    cublas_status = cublasSgemmStridedBatched(
        handle.handle,
        CUBLAS_OP_N,
        CUBLAS_OP_T,
        static_cast<int>(n),
        static_cast<int>(k),
        static_cast<int>(m),
        &alpha,
        handle_to_ptr(grad_handle),
        static_cast<int>(n),
        static_cast<long long>(m * n),
        handle_to_ptr(lhs_handle),
        static_cast<int>(k),
        static_cast<long long>(m * k),
        &beta,
        handle_to_ptr(d_rhs_handle),
        static_cast<int>(n),
        static_cast<long long>(k * n),
        static_cast<int>(batch_count));
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        set_cublas_error("cuBLAS strided batched SGEMM failed for batch_matmul backward d_rhs", cublas_status);
        return 1;
    }
    return 0;
}

extern "C" int lumen_cuda_batch_matmul_backward_bf16_device(
    uint64_t grad_handle,
    uint64_t lhs_handle,
    uint64_t rhs_handle,
    uint64_t d_lhs_handle,
    uint64_t d_rhs_handle,
    size_t batch_count,
    size_t m,
    size_t k,
    size_t n) {
    if (!validate_handle(grad_handle, "CUDA BF16 batch_matmul backward grad handle") ||
        !validate_handle(lhs_handle, "CUDA BF16 batch_matmul backward lhs handle") ||
        !validate_handle(rhs_handle, "CUDA BF16 batch_matmul backward rhs handle") ||
        !validate_handle(d_lhs_handle, "CUDA BF16 batch_matmul backward d_lhs handle") ||
        !validate_handle(d_rhs_handle, "CUDA BF16 batch_matmul backward d_rhs handle")) {
        return 1;
    }
    if (batch_count == 0) {
        set_error("CUDA BF16 batch_matmul backward batch_count must be greater than zero");
        return 1;
    }
    if (!validate_dims(m, n, k)) {
        return 1;
    }

    if (!validate_batch_matmul_backward_sizes(batch_count, m, k, n)) {
        return 1;
    }

    {
        size_t lhs_total = batch_count * m * k;
        size_t rhs_total = batch_count * k * n;
        constexpr int block_size = 256;
        const size_t total = lhs_total > rhs_total ? lhs_total : rhs_total;
        const unsigned int grid = linear_grid_size(total, block_size);
        batch_matmul_lowp_backward_kernel<__nv_bfloat16><<<grid, block_size>>>(
            handle_to_ptr(grad_handle),
            reinterpret_cast<const __nv_bfloat16*>(handle_to_ptr(lhs_handle)),
            reinterpret_cast<const __nv_bfloat16*>(handle_to_ptr(rhs_handle)),
            handle_to_ptr(d_lhs_handle),
            handle_to_ptr(d_rhs_handle),
            lhs_total,
            rhs_total,
            m,
            k,
            n);
        return check_cuda_launch("CUDA BF16 batch_matmul backward kernel launch failed") ? 0 : 1;
    }
}

extern "C" int lumen_cuda_batch_matmul_backward_f16_device(
    uint64_t grad_handle,
    uint64_t lhs_handle,
    uint64_t rhs_handle,
    uint64_t d_lhs_handle,
    uint64_t d_rhs_handle,
    size_t batch_count,
    size_t m,
    size_t k,
    size_t n) {
    if (!validate_handle(grad_handle, "CUDA F16 batch_matmul backward grad handle") ||
        !validate_handle(lhs_handle, "CUDA F16 batch_matmul backward lhs handle") ||
        !validate_handle(rhs_handle, "CUDA F16 batch_matmul backward rhs handle") ||
        !validate_handle(d_lhs_handle, "CUDA F16 batch_matmul backward d_lhs handle") ||
        !validate_handle(d_rhs_handle, "CUDA F16 batch_matmul backward d_rhs handle")) {
        return 1;
    }
    if (batch_count == 0) {
        set_error("CUDA F16 batch_matmul backward batch_count must be greater than zero");
        return 1;
    }
    if (!validate_dims(m, n, k)) {
        return 1;
    }

    if (!validate_batch_matmul_backward_sizes(batch_count, m, k, n)) {
        return 1;
    }

    {
        size_t lhs_total = batch_count * m * k;
        size_t rhs_total = batch_count * k * n;
        constexpr int block_size = 256;
        const size_t total = lhs_total > rhs_total ? lhs_total : rhs_total;
        const unsigned int grid = linear_grid_size(total, block_size);
        batch_matmul_lowp_backward_kernel<__half><<<grid, block_size>>>(
            handle_to_ptr(grad_handle),
            reinterpret_cast<const __half*>(handle_to_ptr(lhs_handle)),
            reinterpret_cast<const __half*>(handle_to_ptr(rhs_handle)),
            handle_to_ptr(d_lhs_handle),
            handle_to_ptr(d_rhs_handle),
            lhs_total,
            rhs_total,
            m,
            k,
            n);
        return check_cuda_launch("CUDA F16 batch_matmul backward kernel launch failed") ? 0 : 1;
    }
}

extern "C" int lumen_cuda_batch_matmul_backward_bf16_i8_device(
    uint64_t grad_handle,
    uint64_t lhs_handle,
    uint64_t rhs_handle,
    float rhs_scale,
    uint64_t d_lhs_handle,
    uint64_t d_rhs_handle,
    size_t batch_count,
    size_t m,
    size_t k,
    size_t n) {
    if (!validate_handle(grad_handle, "CUDA BF16xI8 batch_matmul backward grad handle") ||
        !validate_handle(lhs_handle, "CUDA BF16xI8 batch_matmul backward lhs handle") ||
        !validate_handle(rhs_handle, "CUDA BF16xI8 batch_matmul backward rhs handle") ||
        !validate_handle(d_lhs_handle, "CUDA BF16xI8 batch_matmul backward d_lhs handle") ||
        !validate_handle(d_rhs_handle, "CUDA BF16xI8 batch_matmul backward d_rhs handle")) {
        return 1;
    }
    if (batch_count == 0) {
        set_error("CUDA BF16xI8 batch_matmul backward batch_count must be greater than zero");
        return 1;
    }
    if (!validate_dims(m, n, k)) {
        return 1;
    }
    if (!validate_batch_matmul_backward_sizes(batch_count, m, k, n)) {
        return 1;
    }
    if (!std::isfinite(rhs_scale) || rhs_scale <= 0.0f) {
        set_error("CUDA BF16xI8 batch_matmul backward scale must be finite and > 0");
        return 1;
    }

    size_t lhs_total = batch_count * m * k;
    size_t rhs_total = batch_count * k * n;
    constexpr int block_size = 256;
    const size_t total = lhs_total > rhs_total ? lhs_total : rhs_total;
    const unsigned int grid = linear_grid_size(total, block_size);
    batch_matmul_lhs_i8_backward_kernel<__nv_bfloat16><<<grid, block_size>>>(
        handle_to_ptr(grad_handle),
        reinterpret_cast<const __nv_bfloat16*>(handle_to_ptr(lhs_handle)),
        reinterpret_cast<const int8_t*>(handle_to_ptr(rhs_handle)),
        rhs_scale,
        handle_to_ptr(d_lhs_handle),
        handle_to_ptr(d_rhs_handle),
        lhs_total,
        rhs_total,
        m,
        k,
        n);
    return check_cuda_launch("CUDA BF16xI8 batch_matmul backward kernel launch failed") ? 0 : 1;
}

extern "C" int lumen_cuda_batch_matmul_backward_f16_i8_device(
    uint64_t grad_handle,
    uint64_t lhs_handle,
    uint64_t rhs_handle,
    float rhs_scale,
    uint64_t d_lhs_handle,
    uint64_t d_rhs_handle,
    size_t batch_count,
    size_t m,
    size_t k,
    size_t n) {
    if (!validate_handle(grad_handle, "CUDA F16xI8 batch_matmul backward grad handle") ||
        !validate_handle(lhs_handle, "CUDA F16xI8 batch_matmul backward lhs handle") ||
        !validate_handle(rhs_handle, "CUDA F16xI8 batch_matmul backward rhs handle") ||
        !validate_handle(d_lhs_handle, "CUDA F16xI8 batch_matmul backward d_lhs handle") ||
        !validate_handle(d_rhs_handle, "CUDA F16xI8 batch_matmul backward d_rhs handle")) {
        return 1;
    }
    if (batch_count == 0) {
        set_error("CUDA F16xI8 batch_matmul backward batch_count must be greater than zero");
        return 1;
    }
    if (!validate_dims(m, n, k)) {
        return 1;
    }
    if (!validate_batch_matmul_backward_sizes(batch_count, m, k, n)) {
        return 1;
    }
    if (!std::isfinite(rhs_scale) || rhs_scale <= 0.0f) {
        set_error("CUDA F16xI8 batch_matmul backward scale must be finite and > 0");
        return 1;
    }

    size_t lhs_total = batch_count * m * k;
    size_t rhs_total = batch_count * k * n;
    constexpr int block_size = 256;
    const size_t total = lhs_total > rhs_total ? lhs_total : rhs_total;
    const unsigned int grid = linear_grid_size(total, block_size);
    batch_matmul_lhs_i8_backward_kernel<__half><<<grid, block_size>>>(
        handle_to_ptr(grad_handle),
        reinterpret_cast<const __half*>(handle_to_ptr(lhs_handle)),
        reinterpret_cast<const int8_t*>(handle_to_ptr(rhs_handle)),
        rhs_scale,
        handle_to_ptr(d_lhs_handle),
        handle_to_ptr(d_rhs_handle),
        lhs_total,
        rhs_total,
        m,
        k,
        n);
    return check_cuda_launch("CUDA F16xI8 batch_matmul backward kernel launch failed") ? 0 : 1;
}

extern "C" int lumen_cuda_batch_matmul_backward_f32_i8_device(
    uint64_t grad_handle,
    uint64_t lhs_handle,
    uint64_t rhs_handle,
    float rhs_scale,
    uint64_t d_lhs_handle,
    uint64_t d_rhs_handle,
    size_t batch_count,
    size_t m,
    size_t k,
    size_t n) {
    if (!validate_handle(grad_handle, "CUDA F32xI8 batch_matmul backward grad handle") ||
        !validate_handle(lhs_handle, "CUDA F32xI8 batch_matmul backward lhs handle") ||
        !validate_handle(rhs_handle, "CUDA F32xI8 batch_matmul backward rhs handle") ||
        !validate_handle(d_lhs_handle, "CUDA F32xI8 batch_matmul backward d_lhs handle") ||
        !validate_handle(d_rhs_handle, "CUDA F32xI8 batch_matmul backward d_rhs handle")) {
        return 1;
    }
    if (batch_count == 0) {
        set_error("CUDA F32xI8 batch_matmul backward batch_count must be greater than zero");
        return 1;
    }
    if (!validate_dims(m, n, k)) {
        return 1;
    }
    if (!validate_batch_matmul_backward_sizes(batch_count, m, k, n)) {
        return 1;
    }
    if (!std::isfinite(rhs_scale) || rhs_scale <= 0.0f) {
        set_error("CUDA F32xI8 batch_matmul backward scale must be finite and > 0");
        return 1;
    }

    size_t lhs_total = batch_count * m * k;
    size_t rhs_total = batch_count * k * n;
    constexpr int block_size = 256;
    const size_t total = lhs_total > rhs_total ? lhs_total : rhs_total;
    const unsigned int grid = linear_grid_size(total, block_size);
    batch_matmul_lhs_i8_backward_kernel<float><<<grid, block_size>>>(
        handle_to_ptr(grad_handle),
        handle_to_ptr(lhs_handle),
        reinterpret_cast<const int8_t*>(handle_to_ptr(rhs_handle)),
        rhs_scale,
        handle_to_ptr(d_lhs_handle),
        handle_to_ptr(d_rhs_handle),
        lhs_total,
        rhs_total,
        m,
        k,
        n);
    return check_cuda_launch("CUDA F32xI8 batch_matmul backward kernel launch failed") ? 0 : 1;
}

extern "C" int lumen_cuda_batch_matmul_backward_i8_bf16_device(
    uint64_t grad_handle,
    uint64_t lhs_handle,
    float lhs_scale,
    uint64_t rhs_handle,
    uint64_t d_lhs_handle,
    uint64_t d_rhs_handle,
    size_t batch_count,
    size_t m,
    size_t k,
    size_t n) {
    if (!validate_handle(grad_handle, "CUDA I8xBF16 batch_matmul backward grad handle") ||
        !validate_handle(lhs_handle, "CUDA I8xBF16 batch_matmul backward lhs handle") ||
        !validate_handle(rhs_handle, "CUDA I8xBF16 batch_matmul backward rhs handle") ||
        !validate_handle(d_lhs_handle, "CUDA I8xBF16 batch_matmul backward d_lhs handle") ||
        !validate_handle(d_rhs_handle, "CUDA I8xBF16 batch_matmul backward d_rhs handle")) {
        return 1;
    }
    if (batch_count == 0) {
        set_error("CUDA I8xBF16 batch_matmul backward batch_count must be greater than zero");
        return 1;
    }
    if (!validate_dims(m, n, k)) {
        return 1;
    }
    if (!validate_batch_matmul_backward_sizes(batch_count, m, k, n)) {
        return 1;
    }
    if (!std::isfinite(lhs_scale) || lhs_scale <= 0.0f) {
        set_error("CUDA I8xBF16 batch_matmul backward scale must be finite and > 0");
        return 1;
    }

    size_t lhs_total = batch_count * m * k;
    size_t rhs_total = batch_count * k * n;
    constexpr int block_size = 256;
    const size_t total = lhs_total > rhs_total ? lhs_total : rhs_total;
    const unsigned int grid = linear_grid_size(total, block_size);
    batch_matmul_i8_rhs_floatlike_backward_kernel<__nv_bfloat16><<<grid, block_size>>>(
        handle_to_ptr(grad_handle),
        reinterpret_cast<const int8_t*>(handle_to_ptr(lhs_handle)),
        lhs_scale,
        reinterpret_cast<const __nv_bfloat16*>(handle_to_ptr(rhs_handle)),
        handle_to_ptr(d_lhs_handle),
        handle_to_ptr(d_rhs_handle),
        lhs_total,
        rhs_total,
        m,
        k,
        n);
    return check_cuda_launch("CUDA I8xBF16 batch_matmul backward kernel launch failed") ? 0 : 1;
}

extern "C" int lumen_cuda_batch_matmul_backward_i8_f16_device(
    uint64_t grad_handle,
    uint64_t lhs_handle,
    float lhs_scale,
    uint64_t rhs_handle,
    uint64_t d_lhs_handle,
    uint64_t d_rhs_handle,
    size_t batch_count,
    size_t m,
    size_t k,
    size_t n) {
    if (!validate_handle(grad_handle, "CUDA I8xF16 batch_matmul backward grad handle") ||
        !validate_handle(lhs_handle, "CUDA I8xF16 batch_matmul backward lhs handle") ||
        !validate_handle(rhs_handle, "CUDA I8xF16 batch_matmul backward rhs handle") ||
        !validate_handle(d_lhs_handle, "CUDA I8xF16 batch_matmul backward d_lhs handle") ||
        !validate_handle(d_rhs_handle, "CUDA I8xF16 batch_matmul backward d_rhs handle")) {
        return 1;
    }
    if (batch_count == 0) {
        set_error("CUDA I8xF16 batch_matmul backward batch_count must be greater than zero");
        return 1;
    }
    if (!validate_dims(m, n, k)) {
        return 1;
    }
    if (!validate_batch_matmul_backward_sizes(batch_count, m, k, n)) {
        return 1;
    }
    if (!std::isfinite(lhs_scale) || lhs_scale <= 0.0f) {
        set_error("CUDA I8xF16 batch_matmul backward scale must be finite and > 0");
        return 1;
    }

    size_t lhs_total = batch_count * m * k;
    size_t rhs_total = batch_count * k * n;
    constexpr int block_size = 256;
    const size_t total = lhs_total > rhs_total ? lhs_total : rhs_total;
    const unsigned int grid = linear_grid_size(total, block_size);
    batch_matmul_i8_rhs_floatlike_backward_kernel<__half><<<grid, block_size>>>(
        handle_to_ptr(grad_handle),
        reinterpret_cast<const int8_t*>(handle_to_ptr(lhs_handle)),
        lhs_scale,
        reinterpret_cast<const __half*>(handle_to_ptr(rhs_handle)),
        handle_to_ptr(d_lhs_handle),
        handle_to_ptr(d_rhs_handle),
        lhs_total,
        rhs_total,
        m,
        k,
        n);
    return check_cuda_launch("CUDA I8xF16 batch_matmul backward kernel launch failed") ? 0 : 1;
}

extern "C" int lumen_cuda_batch_matmul_backward_i8_f32_device(
    uint64_t grad_handle,
    uint64_t lhs_handle,
    float lhs_scale,
    uint64_t rhs_handle,
    uint64_t d_lhs_handle,
    uint64_t d_rhs_handle,
    size_t batch_count,
    size_t m,
    size_t k,
    size_t n) {
    if (!validate_handle(grad_handle, "CUDA I8xF32 batch_matmul backward grad handle") ||
        !validate_handle(lhs_handle, "CUDA I8xF32 batch_matmul backward lhs handle") ||
        !validate_handle(rhs_handle, "CUDA I8xF32 batch_matmul backward rhs handle") ||
        !validate_handle(d_lhs_handle, "CUDA I8xF32 batch_matmul backward d_lhs handle") ||
        !validate_handle(d_rhs_handle, "CUDA I8xF32 batch_matmul backward d_rhs handle")) {
        return 1;
    }
    if (batch_count == 0) {
        set_error("CUDA I8xF32 batch_matmul backward batch_count must be greater than zero");
        return 1;
    }
    if (!validate_dims(m, n, k)) {
        return 1;
    }
    if (!validate_batch_matmul_backward_sizes(batch_count, m, k, n)) {
        return 1;
    }
    if (!std::isfinite(lhs_scale) || lhs_scale <= 0.0f) {
        set_error("CUDA I8xF32 batch_matmul backward scale must be finite and > 0");
        return 1;
    }

    size_t lhs_total = batch_count * m * k;
    size_t rhs_total = batch_count * k * n;
    constexpr int block_size = 256;
    const size_t total = lhs_total > rhs_total ? lhs_total : rhs_total;
    const unsigned int grid = linear_grid_size(total, block_size);
    batch_matmul_i8_rhs_floatlike_backward_kernel<float><<<grid, block_size>>>(
        handle_to_ptr(grad_handle),
        reinterpret_cast<const int8_t*>(handle_to_ptr(lhs_handle)),
        lhs_scale,
        handle_to_ptr(rhs_handle),
        handle_to_ptr(d_lhs_handle),
        handle_to_ptr(d_rhs_handle),
        lhs_total,
        rhs_total,
        m,
        k,
        n);
    return check_cuda_launch("CUDA I8xF32 batch_matmul backward kernel launch failed") ? 0 : 1;
}

extern "C" int lumen_cuda_batch_matmul_backward_i8_device(
    uint64_t grad_handle,
    uint64_t lhs_handle,
    float lhs_scale,
    uint64_t rhs_handle,
    float rhs_scale,
    uint64_t d_lhs_handle,
    uint64_t d_rhs_handle,
    size_t batch_count,
    size_t m,
    size_t k,
    size_t n) {
    if (!validate_handle(grad_handle, "CUDA I8 batch_matmul backward grad handle") ||
        !validate_handle(lhs_handle, "CUDA I8 batch_matmul backward lhs handle") ||
        !validate_handle(rhs_handle, "CUDA I8 batch_matmul backward rhs handle") ||
        !validate_handle(d_lhs_handle, "CUDA I8 batch_matmul backward d_lhs handle") ||
        !validate_handle(d_rhs_handle, "CUDA I8 batch_matmul backward d_rhs handle")) {
        return 1;
    }
    if (batch_count == 0) {
        set_error("CUDA I8 batch_matmul backward batch_count must be greater than zero");
        return 1;
    }
    if (!validate_dims(m, n, k)) {
        return 1;
    }
    if (!validate_batch_matmul_backward_sizes(batch_count, m, k, n)) {
        return 1;
    }
    if (!std::isfinite(lhs_scale) || lhs_scale <= 0.0f ||
        !std::isfinite(rhs_scale) || rhs_scale <= 0.0f) {
        set_error("CUDA I8 batch_matmul backward scales must be finite and > 0");
        return 1;
    }

    size_t lhs_total = batch_count * m * k;
    size_t rhs_total = batch_count * k * n;
    constexpr int block_size = 256;
    const size_t total = lhs_total > rhs_total ? lhs_total : rhs_total;
    const unsigned int grid = linear_grid_size(total, block_size);
    batch_matmul_i8_backward_kernel<<<grid, block_size>>>(
        handle_to_ptr(grad_handle),
        reinterpret_cast<const int8_t*>(handle_to_ptr(lhs_handle)),
        lhs_scale,
        reinterpret_cast<const int8_t*>(handle_to_ptr(rhs_handle)),
        rhs_scale,
        handle_to_ptr(d_lhs_handle),
        handle_to_ptr(d_rhs_handle),
        lhs_total,
        rhs_total,
        m,
        k,
        n);
    return check_cuda_launch("CUDA I8 batch_matmul backward kernel launch failed") ? 0 : 1;
}
