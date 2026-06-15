__global__ void sum_kernel(const float* input, float* out, size_t len) {
    __shared__ float shared[256];
    unsigned int tid = threadIdx.x;
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;

    float local = 0.0f;
    for (size_t i = idx; i < len; i += stride) {
        local += input[i];
    }

    shared[tid] = local;
    __syncthreads();

    for (unsigned int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            shared[tid] += shared[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(out, shared[0]);
    }
}

template <typename T>
__global__ void sum_lowp_kernel(const T* input, float* out, size_t len) {
    __shared__ float shared[256];
    unsigned int tid = threadIdx.x;
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;

    float local = 0.0f;
    for (size_t i = idx; i < len; i += stride) {
        local += lowp_to_float(input[i]);
    }

    shared[tid] = local;
    __syncthreads();

    for (unsigned int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            shared[tid] += shared[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(out, shared[0]);
    }
}

__global__ void sum_i8_partials_kernel(const int8_t* input, long long* partials, size_t len) {
    __shared__ long long shared[256];
    unsigned int tid = threadIdx.x;
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;

    long long local = 0;
    for (size_t i = idx; i < len; i += stride) {
        local += static_cast<long long>(input[i]);
    }

    shared[tid] = local;
    __syncthreads();

    for (unsigned int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            shared[tid] += shared[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partials[blockIdx.x] = shared[0];
    }
}

__global__ void sum_i8_finalize_kernel(const long long* partials, float scale, float* out, size_t len) {
    __shared__ long long shared[256];
    unsigned int tid = threadIdx.x;
    long long local = 0;
    for (size_t i = tid; i < len; i += blockDim.x) {
        local += partials[i];
    }

    shared[tid] = local;
    __syncthreads();

    for (unsigned int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            shared[tid] += shared[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        *out = static_cast<float>(shared[0]) * scale;
    }
}

__global__ void fill_scalar_kernel(float* out, size_t len, float value) {
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < len; idx += stride) {
        out[idx] = value;
    }
}
__global__ void fill_scalar_vec4_kernel(float4* out, size_t vec_len, size_t len, float value) {
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < vec_len; idx += stride) {
        out[idx] = make_float4(value, value, value, value);
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        float* out_tail = reinterpret_cast<float*>(out);
        for (size_t tail = vec_len * 4; tail < len; ++tail) {
            out_tail[tail] = value;
        }
    }
}

__global__ void add_inplace_kernel(float* dst, const float* src, size_t len) {
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < len; idx += stride) {
        dst[idx] += src[idx];
    }
}

__global__ void add_inplace_vec4_kernel(float4* dst, const float4* src, size_t vec_len, size_t len) {
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < vec_len; idx += stride) {
        float4 a = dst[idx];
        float4 b = src[idx];
        dst[idx] = make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        float* dst_tail = reinterpret_cast<float*>(dst);
        const float* src_tail = reinterpret_cast<const float*>(src);
        for (size_t tail = vec_len * 4; tail < len; ++tail) {
            dst_tail[tail] += src_tail[tail];
        }
    }
}

__global__ void sum_lastdim_kernel(const float* input, float* out, size_t rows, size_t last_dim) {
    __shared__ float partials[256];
    unsigned int tid = threadIdx.x;
    for (size_t col = blockIdx.x; col < last_dim; col += gridDim.x) {
        float acc = 0.0f;
        for (size_t row = tid; row < rows; row += blockDim.x) {
            acc += input[row * last_dim + col];
        }

        partials[tid] = acc;
        __syncthreads();
        for (unsigned int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
            if (tid < offset) {
                partials[tid] += partials[tid + offset];
            }
            __syncthreads();
        }
        if (tid == 0) {
            out[col] = partials[0];
        }
        __syncthreads();
    }
}

__global__ void sum_lastdim_atomic_kernel(const float* input, float* out, size_t len, size_t last_dim) {
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < len; idx += stride) {
        atomicAdd(out + (idx % last_dim), input[idx]);
    }
}

template <typename OutputT, typename TargetT>
__global__ void mse_forward_typed_kernel(
    const OutputT* output,
    float output_scale,
    const TargetT* target,
    float target_scale,
    float* diff,
    float* loss,
    size_t len,
    float factor) {
    __shared__ float shared[256];
    unsigned int tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + tid;
    size_t stride = blockDim.x * gridDim.x;

    float local = 0.0f;
    while (idx < len) {
        float d = typed_value_to_float(output, output_scale, idx) -
                  typed_value_to_float(target, target_scale, idx);
        diff[idx] = d;
        local += d * d;
        idx += stride;
    }

    shared[tid] = local;
    __syncthreads();
    for (unsigned int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            shared[tid] += shared[tid + offset];
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(loss, shared[0] * factor);
    }
}

__global__ void mse_backward_kernel(
    const float* diff,
    float* grad_output,
    float* grad_target,
    size_t len,
    float factor) {
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < len; idx += stride) {
        float grad = diff[idx] * factor;
        grad_output[idx] = grad;
        grad_target[idx] = -grad;
    }
}

__global__ void cross_entropy_backward_kernel(
    const float* softmax,
    const float* target,
    float* out,
    size_t len,
    float factor) {
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < len; idx += stride) {
        out[idx] = (softmax[idx] - target[idx]) * factor;
    }
}

__global__ void cross_entropy_loss_kernel(
    const float* softmax,
    const float* target,
    float* out,
    size_t len,
    float factor) {
    __shared__ float shared[256];
    unsigned int tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + tid;
    size_t stride = blockDim.x * gridDim.x;

    float local = 0.0f;
    constexpr float epsilon = 1.0e-9f;
    while (idx < len) {
        float t = target[idx];
        if (t > 0.0f) {
            local += -t * logf(softmax[idx] + epsilon);
        }
        idx += stride;
    }

    shared[tid] = local;
    __syncthreads();

    for (unsigned int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            shared[tid] += shared[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(out, shared[0] * factor);
    }
}

template <typename TargetT>
__global__ void cross_entropy_backward_typed_target_kernel(
    const float* softmax,
    const TargetT* target,
    float target_scale,
    float* out,
    size_t len,
    float factor) {
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < len; idx += stride) {
        out[idx] = (softmax[idx] - typed_value_to_float(target, target_scale, idx)) * factor;
    }
}

template <typename TargetT>
__global__ void cross_entropy_loss_typed_target_kernel(
    const float* softmax,
    const TargetT* target,
    float target_scale,
    float* out,
    size_t len,
    float factor) {
    __shared__ float shared[256];
    unsigned int tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + tid;
    size_t stride = blockDim.x * gridDim.x;

    float local = 0.0f;
    constexpr float epsilon = 1.0e-9f;
    while (idx < len) {
        float t = typed_value_to_float(target, target_scale, idx);
        if (t > 0.0f) {
            local += -t * logf(softmax[idx] + epsilon);
        }
        idx += stride;
    }

    shared[tid] = local;
    __syncthreads();

    for (unsigned int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            shared[tid] += shared[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(out, shared[0] * factor);
    }
}

__global__ void sgd_update_kernel(float* param, const float* grad, size_t len, float lr) {
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t idx = start; idx < len; idx += stride) {
        param[idx] -= lr * grad[idx];
    }
}

__global__ void sgd_update_batched_kernel(
    float** params,
    const float** grads,
    const size_t* lens,
    size_t count,
    float lr) {
    size_t tensor_idx = blockIdx.y;
    if (tensor_idx >= count) {
        return;
    }
    float* param = params[tensor_idx];
    const float* grad = grads[tensor_idx];
    size_t len = lens[tensor_idx];
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    while (idx < len) {
        param[idx] -= lr * grad[idx];
        idx += stride;
    }
}

__global__ void quantize_f32_storage_kernel(float* param, size_t len, int dtype, float scale) {
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t idx = start; idx < len; idx += stride) {
        float value = param[idx];
        if (dtype == 1) {
            param[idx] = __half2float(__float2half_rn(value));
        } else if (dtype == 2) {
            param[idx] = __bfloat162float(__float2bfloat16_rn(value));
        } else if (dtype == 3) {
            if (scale > 0.0f && isfinite(scale)) {
                float q = nearbyintf(value / scale);
                q = fminf(127.0f, fmaxf(-127.0f, q));
                param[idx] = q * scale;
            }
        }
    }
}

__global__ void f32_absmax_blocks_kernel(const float* input, float* partial, size_t len) {
    extern __shared__ float shared[];
    size_t tid = threadIdx.x;
    float value = 0.0f;
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + tid;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t idx = start; idx < len; idx += stride) {
        value = fmaxf(value, fabsf(input[idx]));
    }
    shared[tid] = value;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial[blockIdx.x] = shared[0];
    }
}

__global__ void f32_absmax_finalize_kernel(const float* partial, float* out, size_t len) {
    extern __shared__ float shared[];
    size_t tid = threadIdx.x;
    float value = 0.0f;
    for (size_t idx = tid; idx < len; idx += blockDim.x) {
        value = fmaxf(value, partial[idx]);
    }
    shared[tid] = value;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[0] = shared[0];
    }
}

bool reduce_absmax_partials_to_host(
    const float* partial,
    size_t partial_len,
    float* host_max,
    const char* context) {
    if (host_max == nullptr) {
        set_error("CUDA absmax reduction host output is null");
        return false;
    }
    if (partial_len == 0) {
        *host_max = 0.0f;
        return true;
    }

    constexpr size_t kHostAbsmaxPartialThreshold = 1024;
    if (partial_len <= kHostAbsmaxPartialThreshold) {
        std::vector<float> host_partial(partial_len);
        cudaError_t status = cudaMemcpy(
            host_partial.data(),
            partial,
            partial_len * sizeof(float),
            cudaMemcpyDeviceToHost);
        if (status != cudaSuccess) {
            set_cuda_error("failed to download CUDA absmax partial reduction", status);
            return false;
        }
        float max_abs = 0.0f;
        for (float value : host_partial) {
            max_abs = fmaxf(max_abs, value);
        }
        *host_max = max_abs;
        return true;
    }

    thread_local ReusableCudaWorkspace device_max_workspace;
    if (!device_max_workspace.ensure(
            sizeof(float),
            "failed to prepare CUDA absmax final reduction buffer")) {
        return false;
    }
    float* device_max = static_cast<float*>(device_max_workspace.ptr);

    constexpr int block_size = 256;
    f32_absmax_finalize_kernel<<<1, block_size, block_size * sizeof(float)>>>(
        partial,
        device_max,
        partial_len);
    if (!check_cuda_launch(context)) {
        return false;
    }

    cudaError_t status = cudaMemcpy(host_max, device_max, sizeof(float), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        set_cuda_error("failed to download CUDA absmax final reduction", status);
        return false;
    }
    return true;
}

template <int Op>
void launch_binary_i8_absmax_blocks_kernel(
    const int8_t* lhs,
    const int8_t* rhs,
    float lhs_scale,
    float rhs_scale,
    float* partial,
    size_t len,
    unsigned int grid_size,
    unsigned int block_size) {
    binary_i8_absmax_blocks_kernel<Op><<<grid_size, block_size, block_size * sizeof(float)>>>(
        lhs,
        rhs,
        lhs_scale,
        rhs_scale,
        partial,
        len);
}

void launch_binary_i8_absmax_blocks_kernel(
    const int8_t* lhs,
    const int8_t* rhs,
    float lhs_scale,
    float rhs_scale,
    float* partial,
    size_t len,
    unsigned int grid_size,
    unsigned int block_size,
    int op) {
    switch (op) {
        case kBinaryAdd:
            launch_binary_i8_absmax_blocks_kernel<kBinaryAdd>(
                lhs, rhs, lhs_scale, rhs_scale, partial, len, grid_size, block_size);
            break;
        case kBinarySub:
            launch_binary_i8_absmax_blocks_kernel<kBinarySub>(
                lhs, rhs, lhs_scale, rhs_scale, partial, len, grid_size, block_size);
            break;
        case kBinaryMul:
            launch_binary_i8_absmax_blocks_kernel<kBinaryMul>(
                lhs, rhs, lhs_scale, rhs_scale, partial, len, grid_size, block_size);
            break;
        default:
            binary_i8_absmax_blocks_kernel<-1><<<grid_size, block_size, block_size * sizeof(float)>>>(
                lhs, rhs, lhs_scale, rhs_scale, partial, len);
            break;
    }
}

template <int Op>
void launch_binary_i8_to_i8_device_absmax_kernel(
    const int8_t* lhs,
    const int8_t* rhs,
    float lhs_scale,
    float rhs_scale,
    int8_t* out,
    size_t len,
    const float* device_max_abs,
    unsigned int grid_size,
    unsigned int block_size) {
    binary_i8_to_i8_device_absmax_kernel<Op><<<grid_size, block_size>>>(
        lhs,
        rhs,
        lhs_scale,
        rhs_scale,
        out,
        len,
        device_max_abs);
}

void launch_binary_i8_to_i8_device_absmax_kernel(
    const int8_t* lhs,
    const int8_t* rhs,
    float lhs_scale,
    float rhs_scale,
    int8_t* out,
    size_t len,
    const float* device_max_abs,
    unsigned int grid_size,
    unsigned int block_size,
    int op) {
    switch (op) {
        case kBinaryAdd:
            launch_binary_i8_to_i8_device_absmax_kernel<kBinaryAdd>(
                lhs, rhs, lhs_scale, rhs_scale, out, len, device_max_abs, grid_size, block_size);
            break;
        case kBinarySub:
            launch_binary_i8_to_i8_device_absmax_kernel<kBinarySub>(
                lhs, rhs, lhs_scale, rhs_scale, out, len, device_max_abs, grid_size, block_size);
            break;
        case kBinaryMul:
            launch_binary_i8_to_i8_device_absmax_kernel<kBinaryMul>(
                lhs, rhs, lhs_scale, rhs_scale, out, len, device_max_abs, grid_size, block_size);
            break;
        default:
            binary_i8_to_i8_device_absmax_kernel<-1><<<grid_size, block_size>>>(
                lhs, rhs, lhs_scale, rhs_scale, out, len, device_max_abs);
            break;
    }
}

template <int Op>
void launch_binary_i8_lastdim_absmax_blocks_kernel(
    const int8_t* lhs,
    const int8_t* rhs,
    float lhs_scale,
    float rhs_scale,
    float* partial,
    size_t len,
    size_t last_dim,
    bool vector_on_rhs,
    unsigned int grid_size,
    unsigned int block_size) {
    binary_i8_lastdim_absmax_blocks_kernel<Op><<<grid_size, block_size, block_size * sizeof(float)>>>(
        lhs,
        rhs,
        lhs_scale,
        rhs_scale,
        partial,
        len,
        last_dim,
        vector_on_rhs);
}

void launch_binary_i8_lastdim_absmax_blocks_kernel(
    const int8_t* lhs,
    const int8_t* rhs,
    float lhs_scale,
    float rhs_scale,
    float* partial,
    size_t len,
    size_t last_dim,
    bool vector_on_rhs,
    unsigned int grid_size,
    unsigned int block_size,
    int op) {
    switch (op) {
        case kBinaryAdd:
            launch_binary_i8_lastdim_absmax_blocks_kernel<kBinaryAdd>(
                lhs, rhs, lhs_scale, rhs_scale, partial, len, last_dim, vector_on_rhs, grid_size, block_size);
            break;
        case kBinarySub:
            launch_binary_i8_lastdim_absmax_blocks_kernel<kBinarySub>(
                lhs, rhs, lhs_scale, rhs_scale, partial, len, last_dim, vector_on_rhs, grid_size, block_size);
            break;
        case kBinaryMul:
            launch_binary_i8_lastdim_absmax_blocks_kernel<kBinaryMul>(
                lhs, rhs, lhs_scale, rhs_scale, partial, len, last_dim, vector_on_rhs, grid_size, block_size);
            break;
        default:
            binary_i8_lastdim_absmax_blocks_kernel<-1><<<grid_size, block_size, block_size * sizeof(float)>>>(
                lhs, rhs, lhs_scale, rhs_scale, partial, len, last_dim, vector_on_rhs);
            break;
    }
}

template <int Op>
void launch_binary_i8_lastdim_to_i8_kernel(
    const int8_t* lhs,
    const int8_t* rhs,
    float lhs_scale,
    float rhs_scale,
    int8_t* out,
    size_t len,
    size_t last_dim,
    bool vector_on_rhs,
    float out_scale,
    unsigned int grid_size,
    unsigned int block_size) {
    binary_i8_lastdim_to_i8_kernel<Op><<<grid_size, block_size>>>(
        lhs,
        rhs,
        lhs_scale,
        rhs_scale,
        out,
        len,
        last_dim,
        vector_on_rhs,
        out_scale);
}

void launch_binary_i8_lastdim_to_i8_kernel(
    const int8_t* lhs,
    const int8_t* rhs,
    float lhs_scale,
    float rhs_scale,
    int8_t* out,
    size_t len,
    size_t last_dim,
    bool vector_on_rhs,
    float out_scale,
    unsigned int grid_size,
    unsigned int block_size,
    int op) {
    switch (op) {
        case kBinaryAdd:
            launch_binary_i8_lastdim_to_i8_kernel<kBinaryAdd>(
                lhs, rhs, lhs_scale, rhs_scale, out, len, last_dim, vector_on_rhs, out_scale, grid_size, block_size);
            break;
        case kBinarySub:
            launch_binary_i8_lastdim_to_i8_kernel<kBinarySub>(
                lhs, rhs, lhs_scale, rhs_scale, out, len, last_dim, vector_on_rhs, out_scale, grid_size, block_size);
            break;
        case kBinaryMul:
            launch_binary_i8_lastdim_to_i8_kernel<kBinaryMul>(
                lhs, rhs, lhs_scale, rhs_scale, out, len, last_dim, vector_on_rhs, out_scale, grid_size, block_size);
            break;
        default:
            binary_i8_lastdim_to_i8_kernel<-1><<<grid_size, block_size>>>(
                lhs, rhs, lhs_scale, rhs_scale, out, len, last_dim, vector_on_rhs, out_scale);
            break;
    }
}

template <int Op>
void launch_binary_i8_row_scalar_absmax_blocks_kernel(
    const int8_t* lhs,
    const int8_t* rhs,
    float lhs_scale,
    float rhs_scale,
    float* partial,
    size_t rows,
    size_t last_dim,
    bool scalar_on_rhs,
    unsigned int grid_size,
    unsigned int block_size) {
    binary_i8_row_scalar_absmax_blocks_kernel<Op><<<grid_size, block_size, block_size * sizeof(float)>>>(
        lhs, rhs, lhs_scale, rhs_scale, partial, rows, last_dim, scalar_on_rhs);
}

void launch_binary_i8_row_scalar_absmax_blocks_kernel(
    const int8_t* lhs,
    const int8_t* rhs,
    float lhs_scale,
    float rhs_scale,
    float* partial,
    size_t rows,
    size_t last_dim,
    bool scalar_on_rhs,
    unsigned int grid_size,
    unsigned int block_size,
    int op) {
    switch (op) {
        case kBinaryAdd:
            launch_binary_i8_row_scalar_absmax_blocks_kernel<kBinaryAdd>(
                lhs, rhs, lhs_scale, rhs_scale, partial, rows, last_dim, scalar_on_rhs, grid_size, block_size);
            break;
        case kBinarySub:
            launch_binary_i8_row_scalar_absmax_blocks_kernel<kBinarySub>(
                lhs, rhs, lhs_scale, rhs_scale, partial, rows, last_dim, scalar_on_rhs, grid_size, block_size);
            break;
        case kBinaryMul:
            launch_binary_i8_row_scalar_absmax_blocks_kernel<kBinaryMul>(
                lhs, rhs, lhs_scale, rhs_scale, partial, rows, last_dim, scalar_on_rhs, grid_size, block_size);
            break;
        default:
            binary_i8_row_scalar_absmax_blocks_kernel<-1><<<grid_size, block_size, block_size * sizeof(float)>>>(
                lhs, rhs, lhs_scale, rhs_scale, partial, rows, last_dim, scalar_on_rhs);
            break;
    }
}

template <int Op>
void launch_binary_i8_row_scalar_to_i8_kernel(
    const int8_t* lhs,
    const int8_t* rhs,
    float lhs_scale,
    float rhs_scale,
    int8_t* out,
    size_t rows,
    size_t last_dim,
    bool scalar_on_rhs,
    float out_scale,
    unsigned int grid_size,
    unsigned int block_size) {
    binary_i8_row_scalar_to_i8_kernel<Op><<<grid_size, block_size>>>(
        lhs, rhs, lhs_scale, rhs_scale, out, rows, last_dim, scalar_on_rhs, out_scale);
}

void launch_binary_i8_row_scalar_to_i8_kernel(
    const int8_t* lhs,
    const int8_t* rhs,
    float lhs_scale,
    float rhs_scale,
    int8_t* out,
    size_t rows,
    size_t last_dim,
    bool scalar_on_rhs,
    float out_scale,
    unsigned int grid_size,
    unsigned int block_size,
    int op) {
    switch (op) {
        case kBinaryAdd:
            launch_binary_i8_row_scalar_to_i8_kernel<kBinaryAdd>(
                lhs, rhs, lhs_scale, rhs_scale, out, rows, last_dim, scalar_on_rhs, out_scale, grid_size, block_size);
            break;
        case kBinarySub:
            launch_binary_i8_row_scalar_to_i8_kernel<kBinarySub>(
                lhs, rhs, lhs_scale, rhs_scale, out, rows, last_dim, scalar_on_rhs, out_scale, grid_size, block_size);
            break;
        case kBinaryMul:
            launch_binary_i8_row_scalar_to_i8_kernel<kBinaryMul>(
                lhs, rhs, lhs_scale, rhs_scale, out, rows, last_dim, scalar_on_rhs, out_scale, grid_size, block_size);
            break;
        default:
            binary_i8_row_scalar_to_i8_kernel<-1><<<grid_size, block_size>>>(
                lhs, rhs, lhs_scale, rhs_scale, out, rows, last_dim, scalar_on_rhs, out_scale);
            break;
    }
}

template <int Op>
void launch_binary_i8_b1d_1h1_absmax_blocks_kernel(
    const int8_t* lhs,
    const int8_t* rhs,
    float lhs_scale,
    float rhs_scale,
    float* partial,
    size_t batch,
    size_t heads,
    size_t dim,
    bool b1d_on_lhs,
    unsigned int grid_size,
    unsigned int block_size) {
    binary_i8_b1d_1h1_absmax_blocks_kernel<Op><<<grid_size, block_size, block_size * sizeof(float)>>>(
        lhs, rhs, lhs_scale, rhs_scale, partial, batch, heads, dim, b1d_on_lhs);
}

void launch_binary_i8_b1d_1h1_absmax_blocks_kernel(
    const int8_t* lhs,
    const int8_t* rhs,
    float lhs_scale,
    float rhs_scale,
    float* partial,
    size_t batch,
    size_t heads,
    size_t dim,
    bool b1d_on_lhs,
    unsigned int grid_size,
    unsigned int block_size,
    int op) {
    switch (op) {
        case kBinaryAdd:
            launch_binary_i8_b1d_1h1_absmax_blocks_kernel<kBinaryAdd>(
                lhs, rhs, lhs_scale, rhs_scale, partial, batch, heads, dim, b1d_on_lhs, grid_size, block_size);
            break;
        case kBinarySub:
            launch_binary_i8_b1d_1h1_absmax_blocks_kernel<kBinarySub>(
                lhs, rhs, lhs_scale, rhs_scale, partial, batch, heads, dim, b1d_on_lhs, grid_size, block_size);
            break;
        case kBinaryMul:
            launch_binary_i8_b1d_1h1_absmax_blocks_kernel<kBinaryMul>(
                lhs, rhs, lhs_scale, rhs_scale, partial, batch, heads, dim, b1d_on_lhs, grid_size, block_size);
            break;
        default:
            binary_i8_b1d_1h1_absmax_blocks_kernel<-1><<<grid_size, block_size, block_size * sizeof(float)>>>(
                lhs, rhs, lhs_scale, rhs_scale, partial, batch, heads, dim, b1d_on_lhs);
            break;
    }
}

template <int Op>
void launch_binary_i8_b1d_1h1_to_i8_kernel(
    const int8_t* lhs,
    const int8_t* rhs,
    float lhs_scale,
    float rhs_scale,
    int8_t* out,
    size_t batch,
    size_t heads,
    size_t dim,
    bool b1d_on_lhs,
    float out_scale,
    unsigned int grid_size,
    unsigned int block_size) {
    binary_i8_b1d_1h1_to_i8_kernel<Op><<<grid_size, block_size>>>(
        lhs, rhs, lhs_scale, rhs_scale, out, batch, heads, dim, b1d_on_lhs, out_scale);
}

void launch_binary_i8_b1d_1h1_to_i8_kernel(
    const int8_t* lhs,
    const int8_t* rhs,
    float lhs_scale,
    float rhs_scale,
    int8_t* out,
    size_t batch,
    size_t heads,
    size_t dim,
    bool b1d_on_lhs,
    float out_scale,
    unsigned int grid_size,
    unsigned int block_size,
    int op) {
    switch (op) {
        case kBinaryAdd:
            launch_binary_i8_b1d_1h1_to_i8_kernel<kBinaryAdd>(
                lhs, rhs, lhs_scale, rhs_scale, out, batch, heads, dim, b1d_on_lhs, out_scale, grid_size, block_size);
            break;
        case kBinarySub:
            launch_binary_i8_b1d_1h1_to_i8_kernel<kBinarySub>(
                lhs, rhs, lhs_scale, rhs_scale, out, batch, heads, dim, b1d_on_lhs, out_scale, grid_size, block_size);
            break;
        case kBinaryMul:
            launch_binary_i8_b1d_1h1_to_i8_kernel<kBinaryMul>(
                lhs, rhs, lhs_scale, rhs_scale, out, batch, heads, dim, b1d_on_lhs, out_scale, grid_size, block_size);
            break;
        default:
            binary_i8_b1d_1h1_to_i8_kernel<-1><<<grid_size, block_size>>>(
                lhs, rhs, lhs_scale, rhs_scale, out, batch, heads, dim, b1d_on_lhs, out_scale);
            break;
    }
}

template <int Op>
void launch_binary_i8_b1d_1hd_absmax_blocks_kernel(
    const int8_t* lhs,
    const int8_t* rhs,
    float lhs_scale,
    float rhs_scale,
    float* partial,
    size_t batch,
    size_t heads,
    size_t dim,
    bool b1d_on_lhs,
    unsigned int grid_size,
    unsigned int block_size) {
    binary_i8_b1d_1hd_absmax_blocks_kernel<Op><<<grid_size, block_size, block_size * sizeof(float)>>>(
        lhs, rhs, lhs_scale, rhs_scale, partial, batch, heads, dim, b1d_on_lhs);
}

void launch_binary_i8_b1d_1hd_absmax_blocks_kernel(
    const int8_t* lhs,
    const int8_t* rhs,
    float lhs_scale,
    float rhs_scale,
    float* partial,
    size_t batch,
    size_t heads,
    size_t dim,
    bool b1d_on_lhs,
    unsigned int grid_size,
    unsigned int block_size,
    int op) {
    switch (op) {
        case kBinaryAdd:
            launch_binary_i8_b1d_1hd_absmax_blocks_kernel<kBinaryAdd>(
                lhs, rhs, lhs_scale, rhs_scale, partial, batch, heads, dim, b1d_on_lhs, grid_size, block_size);
            break;
        case kBinarySub:
            launch_binary_i8_b1d_1hd_absmax_blocks_kernel<kBinarySub>(
                lhs, rhs, lhs_scale, rhs_scale, partial, batch, heads, dim, b1d_on_lhs, grid_size, block_size);
            break;
        case kBinaryMul:
            launch_binary_i8_b1d_1hd_absmax_blocks_kernel<kBinaryMul>(
                lhs, rhs, lhs_scale, rhs_scale, partial, batch, heads, dim, b1d_on_lhs, grid_size, block_size);
            break;
        default:
            binary_i8_b1d_1hd_absmax_blocks_kernel<-1><<<grid_size, block_size, block_size * sizeof(float)>>>(
                lhs, rhs, lhs_scale, rhs_scale, partial, batch, heads, dim, b1d_on_lhs);
            break;
    }
}

template <int Op>
void launch_binary_i8_b1d_1hd_to_i8_kernel(
    const int8_t* lhs,
    const int8_t* rhs,
    float lhs_scale,
    float rhs_scale,
    int8_t* out,
    size_t batch,
    size_t heads,
    size_t dim,
    bool b1d_on_lhs,
    float out_scale,
    unsigned int grid_size,
    unsigned int block_size) {
    binary_i8_b1d_1hd_to_i8_kernel<Op><<<grid_size, block_size>>>(
        lhs, rhs, lhs_scale, rhs_scale, out, batch, heads, dim, b1d_on_lhs, out_scale);
}

void launch_binary_i8_b1d_1hd_to_i8_kernel(
    const int8_t* lhs,
    const int8_t* rhs,
    float lhs_scale,
    float rhs_scale,
    int8_t* out,
    size_t batch,
    size_t heads,
    size_t dim,
    bool b1d_on_lhs,
    float out_scale,
    unsigned int grid_size,
    unsigned int block_size,
    int op) {
    switch (op) {
        case kBinaryAdd:
            launch_binary_i8_b1d_1hd_to_i8_kernel<kBinaryAdd>(
                lhs, rhs, lhs_scale, rhs_scale, out, batch, heads, dim, b1d_on_lhs, out_scale, grid_size, block_size);
            break;
        case kBinarySub:
            launch_binary_i8_b1d_1hd_to_i8_kernel<kBinarySub>(
                lhs, rhs, lhs_scale, rhs_scale, out, batch, heads, dim, b1d_on_lhs, out_scale, grid_size, block_size);
            break;
        case kBinaryMul:
            launch_binary_i8_b1d_1hd_to_i8_kernel<kBinaryMul>(
                lhs, rhs, lhs_scale, rhs_scale, out, batch, heads, dim, b1d_on_lhs, out_scale, grid_size, block_size);
            break;
        default:
            binary_i8_b1d_1hd_to_i8_kernel<-1><<<grid_size, block_size>>>(
                lhs, rhs, lhs_scale, rhs_scale, out, batch, heads, dim, b1d_on_lhs, out_scale);
            break;
    }
}

template <int Op>
void launch_binary_i8_broadcast_absmax_blocks_kernel(
    const int8_t* lhs,
    const int8_t* rhs,
    float lhs_scale,
    float rhs_scale,
    float* partial,
    const size_t* d_out_strides,
    const size_t* d_lhs_shape,
    const size_t* d_lhs_strides,
    const size_t* d_rhs_shape,
    const size_t* d_rhs_strides,
    size_t ndim,
    size_t len,
    int grid_size,
    int block_size) {
    binary_i8_broadcast_absmax_blocks_kernel<Op><<<grid_size, block_size, block_size * sizeof(float)>>>(
        lhs,
        rhs,
        lhs_scale,
        rhs_scale,
        partial,
        d_out_strides,
        d_lhs_shape,
        d_lhs_strides,
        d_rhs_shape,
        d_rhs_strides,
        ndim,
        len);
}

void launch_binary_i8_broadcast_absmax_blocks_kernel(
    const int8_t* lhs,
    const int8_t* rhs,
    float lhs_scale,
    float rhs_scale,
    float* partial,
    const size_t* d_out_strides,
    const size_t* d_lhs_shape,
    const size_t* d_lhs_strides,
    const size_t* d_rhs_shape,
    const size_t* d_rhs_strides,
    size_t ndim,
    size_t len,
    int grid_size,
    int block_size,
    int op) {
    switch (op) {
        case kBinaryAdd:
            launch_binary_i8_broadcast_absmax_blocks_kernel<kBinaryAdd>(
                lhs, rhs, lhs_scale, rhs_scale, partial, d_out_strides, d_lhs_shape, d_lhs_strides, d_rhs_shape, d_rhs_strides, ndim, len, grid_size, block_size);
            break;
        case kBinarySub:
            launch_binary_i8_broadcast_absmax_blocks_kernel<kBinarySub>(
                lhs, rhs, lhs_scale, rhs_scale, partial, d_out_strides, d_lhs_shape, d_lhs_strides, d_rhs_shape, d_rhs_strides, ndim, len, grid_size, block_size);
            break;
        case kBinaryMul:
            launch_binary_i8_broadcast_absmax_blocks_kernel<kBinaryMul>(
                lhs, rhs, lhs_scale, rhs_scale, partial, d_out_strides, d_lhs_shape, d_lhs_strides, d_rhs_shape, d_rhs_strides, ndim, len, grid_size, block_size);
            break;
        default:
            binary_i8_broadcast_absmax_blocks_kernel<-1><<<grid_size, block_size, block_size * sizeof(float)>>>(
                lhs, rhs, lhs_scale, rhs_scale, partial, d_out_strides, d_lhs_shape, d_lhs_strides, d_rhs_shape, d_rhs_strides, ndim, len);
            break;
    }
}

template <int Op>
void launch_binary_i8_broadcast_to_i8_kernel(
    const int8_t* lhs,
    const int8_t* rhs,
    float lhs_scale,
    float rhs_scale,
    int8_t* out,
    const size_t* d_out_strides,
    const size_t* d_lhs_shape,
    const size_t* d_lhs_strides,
    const size_t* d_rhs_shape,
    const size_t* d_rhs_strides,
    size_t ndim,
    size_t len,
    float out_scale,
    int grid_size,
    int block_size) {
    binary_i8_broadcast_to_i8_kernel<Op><<<grid_size, block_size>>>(
        lhs,
        rhs,
        lhs_scale,
        rhs_scale,
        out,
        d_out_strides,
        d_lhs_shape,
        d_lhs_strides,
        d_rhs_shape,
        d_rhs_strides,
        ndim,
        len,
        out_scale);
}

void launch_binary_i8_broadcast_to_i8_kernel(
    const int8_t* lhs,
    const int8_t* rhs,
    float lhs_scale,
    float rhs_scale,
    int8_t* out,
    const size_t* d_out_strides,
    const size_t* d_lhs_shape,
    const size_t* d_lhs_strides,
    const size_t* d_rhs_shape,
    const size_t* d_rhs_strides,
    size_t ndim,
    size_t len,
    float out_scale,
    int grid_size,
    int block_size,
    int op) {
    switch (op) {
        case kBinaryAdd:
            launch_binary_i8_broadcast_to_i8_kernel<kBinaryAdd>(
                lhs, rhs, lhs_scale, rhs_scale, out, d_out_strides, d_lhs_shape, d_lhs_strides, d_rhs_shape, d_rhs_strides, ndim, len, out_scale, grid_size, block_size);
            break;
        case kBinarySub:
            launch_binary_i8_broadcast_to_i8_kernel<kBinarySub>(
                lhs, rhs, lhs_scale, rhs_scale, out, d_out_strides, d_lhs_shape, d_lhs_strides, d_rhs_shape, d_rhs_strides, ndim, len, out_scale, grid_size, block_size);
            break;
        case kBinaryMul:
            launch_binary_i8_broadcast_to_i8_kernel<kBinaryMul>(
                lhs, rhs, lhs_scale, rhs_scale, out, d_out_strides, d_lhs_shape, d_lhs_strides, d_rhs_shape, d_rhs_strides, ndim, len, out_scale, grid_size, block_size);
            break;
        default:
            binary_i8_broadcast_to_i8_kernel<-1><<<grid_size, block_size>>>(
                lhs, rhs, lhs_scale, rhs_scale, out, d_out_strides, d_lhs_shape, d_lhs_strides, d_rhs_shape, d_rhs_strides, ndim, len, out_scale);
            break;
    }
}

__global__ void quantize_f32_to_i8_kernel(
    const float* input,
    int8_t* output,
    size_t len,
    float scale) {
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t idx = start; idx < len; idx += stride) {
        float q = nearbyintf(input[idx] / scale);
        q = fminf(127.0f, fmaxf(-127.0f, q));
        output[idx] = static_cast<int8_t>(q);
    }
}

__global__ void quantize_f32_to_i8_device_absmax_kernel(
    const float* input,
    int8_t* output,
    size_t len,
    const float* device_max_abs) {
    const float max_abs = device_max_abs[0];
    const float scale =
        max_abs > 0.0f && isfinite(max_abs) ? fmaxf(max_abs / 127.0f, FLT_MIN) : 1.0f;
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t idx = start; idx < len; idx += stride) {
        float q = nearbyintf(input[idx] / scale);
        q = fminf(127.0f, fmaxf(-127.0f, q));
        output[idx] = static_cast<int8_t>(q);
    }
}

__global__ void f32_to_lowp_storage_kernel(
    const float* input,
    uint16_t* output,
    size_t len,
    int dtype) {
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t idx = start; idx < len; idx += stride) {
        float value = input[idx];
        if (dtype == kDTypeF16) {
            __half h = __float2half_rn(value);
            output[idx] = *reinterpret_cast<uint16_t*>(&h);
        } else if (dtype == kDTypeBF16) {
            __nv_bfloat16 b = __float2bfloat16_rn(value);
            output[idx] = *reinterpret_cast<uint16_t*>(&b);
        }
    }
}

__global__ void sgd_momentum_update_kernel(
    float* param,
    const float* grad,
    float* velocity,
    size_t len,
    float lr,
    float momentum) {
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t idx = start; idx < len; idx += stride) {
        float v = momentum * velocity[idx] + grad[idx];
        velocity[idx] = v;
        param[idx] -= lr * v;
    }
}

__global__ void sgd_momentum_update_batched_kernel(
    float** params,
    const float** grads,
    float** velocities,
    const size_t* lens,
    size_t count,
    float lr,
    float momentum) {
    size_t tensor_idx = blockIdx.y;
    if (tensor_idx >= count) {
        return;
    }
    float* param = params[tensor_idx];
    const float* grad = grads[tensor_idx];
    float* velocity = velocities[tensor_idx];
    size_t len = lens[tensor_idx];
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    while (idx < len) {
        float v = momentum * velocity[idx] + grad[idx];
        velocity[idx] = v;
        param[idx] -= lr * v;
        idx += stride;
    }
}

__global__ void adam_update_kernel(
    float* param,
    const float* grad,
    float* exp_avg,
    float* exp_avg_sq,
    size_t len,
    float lr,
    float beta1,
    float beta2,
    float bias_correction1,
    float bias_correction2,
    float eps) {
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t idx = start; idx < len; idx += stride) {
        float g = grad[idx];
        float m = beta1 * exp_avg[idx] + (1.0f - beta1) * g;
        float v = beta2 * exp_avg_sq[idx] + (1.0f - beta2) * g * g;
        exp_avg[idx] = m;
        exp_avg_sq[idx] = v;

        float m_hat = m / bias_correction1;
        float v_hat = v / bias_correction2;
        param[idx] -= lr * (m_hat / (sqrtf(v_hat) + eps));
    }
}

__device__ inline void adam_update_one(
    float& param,
    float grad,
    float& exp_avg,
    float& exp_avg_sq,
    float lr,
    float beta1,
    float beta2,
    float bias_correction1,
    float bias_correction2,
    float eps) {
    float m = beta1 * exp_avg + (1.0f - beta1) * grad;
    float v = beta2 * exp_avg_sq + (1.0f - beta2) * grad * grad;
    exp_avg = m;
    exp_avg_sq = v;

    float m_hat = m / bias_correction1;
    float v_hat = v / bias_correction2;
    param -= lr * (m_hat / (sqrtf(v_hat) + eps));
}

__global__ void adam_update_vec4_kernel(
    float4* param,
    const float4* grad,
    float4* exp_avg,
    float4* exp_avg_sq,
    size_t vec_len,
    size_t len,
    float lr,
    float beta1,
    float beta2,
    float bias_correction1,
    float bias_correction2,
    float eps) {
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t idx = start; idx < vec_len; idx += stride) {
        float4 p = param[idx];
        float4 g = grad[idx];
        float4 m = exp_avg[idx];
        float4 v = exp_avg_sq[idx];
        adam_update_one(p.x, g.x, m.x, v.x, lr, beta1, beta2, bias_correction1, bias_correction2, eps);
        adam_update_one(p.y, g.y, m.y, v.y, lr, beta1, beta2, bias_correction1, bias_correction2, eps);
        adam_update_one(p.z, g.z, m.z, v.z, lr, beta1, beta2, bias_correction1, bias_correction2, eps);
        adam_update_one(p.w, g.w, m.w, v.w, lr, beta1, beta2, bias_correction1, bias_correction2, eps);
        param[idx] = p;
        exp_avg[idx] = m;
        exp_avg_sq[idx] = v;
    }
    if (start == 0) {
        float* param_tail = reinterpret_cast<float*>(param);
        const float* grad_tail = reinterpret_cast<const float*>(grad);
        float* exp_avg_tail = reinterpret_cast<float*>(exp_avg);
        float* exp_avg_sq_tail = reinterpret_cast<float*>(exp_avg_sq);
        for (size_t tail = vec_len * 4; tail < len; ++tail) {
            adam_update_one(
                param_tail[tail],
                grad_tail[tail],
                exp_avg_tail[tail],
                exp_avg_sq_tail[tail],
                lr,
                beta1,
                beta2,
                bias_correction1,
                bias_correction2,
                eps);
        }
    }
}

__global__ void adam_update_batched_kernel(
    float** params,
    const float** grads,
    float** exp_avgs,
    float** exp_avg_sqs,
    const size_t* lens,
    size_t count,
    float lr,
    float beta1,
    float beta2,
    float bias_correction1,
    float bias_correction2,
    float eps) {
    size_t tensor_idx = blockIdx.y;
    if (tensor_idx >= count) {
        return;
    }
    float* param = params[tensor_idx];
    const float* grad = grads[tensor_idx];
    float* exp_avg = exp_avgs[tensor_idx];
    float* exp_avg_sq = exp_avg_sqs[tensor_idx];
    size_t len = lens[tensor_idx];
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    while (idx < len) {
        float g = grad[idx];
        float m = beta1 * exp_avg[idx] + (1.0f - beta1) * g;
        float v = beta2 * exp_avg_sq[idx] + (1.0f - beta2) * g * g;
        exp_avg[idx] = m;
        exp_avg_sq[idx] = v;

        float m_hat = m / bias_correction1;
        float v_hat = v / bias_correction2;
        param[idx] -= lr * (m_hat / (sqrtf(v_hat) + eps));
        idx += stride;
    }
}
