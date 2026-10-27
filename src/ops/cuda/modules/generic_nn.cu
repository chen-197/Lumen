// Architecture-independent normalization and masked attention primitives.
// A block owns one row. Reductions are deterministic and do not use atomics.
__device__ double generic_block_sum(double value, double *shared) {
    shared[threadIdx.x] = value;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride)
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        __syncthreads();
    }
    return shared[0];
}

__global__ void generic_layer_norm_kernel(const float *x, const float *w, const float *b, float *y,
                                          float *xhat, float *inverse, size_t rows, size_t dim,
                                          float eps) {
    __shared__ double shared[256];
    for (size_t row = blockIdx.x; row < rows; row += gridDim.x) {
        size_t start = row * dim;
        double sum = 0.0;
        for (size_t j = threadIdx.x; j < dim; j += blockDim.x)
            sum += x[start + j];
        double mean = generic_block_sum(sum, shared) / static_cast<double>(dim);
        __syncthreads();
        double variance = 0.0;
        for (size_t j = threadIdx.x; j < dim; j += blockDim.x) {
            double d = static_cast<double>(x[start + j]) - mean;
            variance += d * d;
        }
        double inv = 1.0 / sqrt(generic_block_sum(variance, shared) / static_cast<double>(dim) +
                                static_cast<double>(eps));
        if (threadIdx.x == 0)
            inverse[row] = static_cast<float>(inv);
        for (size_t j = threadIdx.x; j < dim; j += blockDim.x) {
            float normalized = static_cast<float>((static_cast<double>(x[start + j]) - mean) * inv);
            xhat[start + j] = normalized;
            y[start + j] = normalized * (w ? w[j] : 1.0f) + (b ? b[j] : 0.0f);
        }
        __syncthreads();
    }
}

__global__ void generic_layer_norm_dx_kernel(const float *xhat, const float *inverse,
                                             const float *w, const float *grad, float *dx,
                                             size_t rows, size_t dim) {
    __shared__ double shared[256];
    for (size_t row = blockIdx.x; row < rows; row += gridDim.x) {
        size_t start = row * dim;
        double sum = 0.0, dot = 0.0;
        for (size_t j = threadIdx.x; j < dim; j += blockDim.x) {
            double g = static_cast<double>(grad[start + j]) * (w ? w[j] : 1.0f);
            sum += g;
            dot += g * xhat[start + j];
        }
        double mean = generic_block_sum(sum, shared) / static_cast<double>(dim);
        __syncthreads();
        double mean_dot = generic_block_sum(dot, shared) / static_cast<double>(dim);
        for (size_t j = threadIdx.x; j < dim; j += blockDim.x) {
            double g = static_cast<double>(grad[start + j]) * (w ? w[j] : 1.0f);
            dx[start + j] = static_cast<float>(
                inverse[row] * (g - mean - static_cast<double>(xhat[start + j]) * mean_dot));
        }
        __syncthreads();
    }
}

__global__ void generic_layer_norm_affine_grad_kernel(const float *xhat, const float *grad,
                                                      float *dw, float *db, size_t rows,
                                                      size_t dim) {
    for (size_t j = blockIdx.x * blockDim.x + threadIdx.x; j < dim;
         j += static_cast<size_t>(gridDim.x) * blockDim.x) {
        double sum_w = 0.0, sum_b = 0.0;
        for (size_t row = 0; row < rows; ++row) {
            size_t i = row * dim + j;
            sum_w += static_cast<double>(grad[i]) * xhat[i];
            sum_b += grad[i];
        }
        dw[j] = static_cast<float>(sum_w);
        db[j] = static_cast<float>(sum_b);
    }
}

__global__ void generic_masked_softmax_kernel(const float *x, float *y, size_t rows, size_t keys) {
    __shared__ float shared[256];
    for (size_t row = blockIdx.x; row < rows; row += gridDim.x) {
        size_t start = row * keys;
        float max_value = -INFINITY;
        for (size_t j = threadIdx.x; j < keys; j += blockDim.x)
            max_value = fmaxf(max_value, x[start + j]);
        shared[threadIdx.x] = max_value;
        __syncthreads();
        for (unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2) {
            if (threadIdx.x < stride)
                shared[threadIdx.x] = fmaxf(shared[threadIdx.x], shared[threadIdx.x + stride]);
            __syncthreads();
        }
        max_value = shared[0];
        __syncthreads();
        float sum = 0.0f;
        for (size_t j = threadIdx.x; j < keys; j += blockDim.x) {
            float e = max_value == -INFINITY ? 0.0f : expf(x[start + j] - max_value);
            y[start + j] = e;
            sum += e;
        }
        shared[threadIdx.x] = sum;
        __syncthreads();
        for (unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2) {
            if (threadIdx.x < stride)
                shared[threadIdx.x] += shared[threadIdx.x + stride];
            __syncthreads();
        }
        sum = shared[0];
        for (size_t j = threadIdx.x; j < keys; j += blockDim.x)
            y[start + j] = sum == 0.0f ? 0.0f : y[start + j] / sum;
        __syncthreads();
    }
}

extern "C" int lumen_cuda_layer_norm(uint64_t x, uint64_t w, uint64_t b, uint64_t y, uint64_t xhat,
                                     uint64_t inverse, size_t rows, size_t dim, float eps) {
    if (!validate_handle(x, "LayerNorm input") || !validate_handle(y, "LayerNorm output") ||
        !validate_handle(xhat, "LayerNorm normalized") ||
        !validate_handle(inverse, "LayerNorm inverse") || rows == 0 || dim == 0 ||
        !std::isfinite(eps) || eps <= 0.0f) {
        set_error("Invalid CUDA LayerNorm arguments");
        return 1;
    }
    generic_layer_norm_kernel<<<static_cast<unsigned int>(std::min(rows, size_t(65535))), 256>>>(
        handle_to_ptr(x), w ? handle_to_ptr(w) : nullptr, b ? handle_to_ptr(b) : nullptr,
        handle_to_ptr(y), handle_to_ptr(xhat), handle_to_ptr(inverse), rows, dim, eps);
    return check_cuda_launch("LayerNorm forward launch") ? 0 : 1;
}

extern "C" int lumen_cuda_layer_norm_backward(uint64_t xhat, uint64_t inverse, uint64_t w,
                                              uint64_t grad, uint64_t dx, uint64_t dw, uint64_t db,
                                              size_t rows, size_t dim) {
    if (!validate_handle(xhat, "LayerNorm normalized") ||
        !validate_handle(inverse, "LayerNorm inverse") ||
        !validate_handle(grad, "LayerNorm gradient") || !validate_handle(dx, "LayerNorm dx") ||
        !validate_handle(dw, "LayerNorm dw") || !validate_handle(db, "LayerNorm db") || rows == 0 ||
        dim == 0)
        return 1;
    generic_layer_norm_dx_kernel<<<static_cast<unsigned int>(std::min(rows, size_t(65535))), 256>>>(
        handle_to_ptr(xhat), handle_to_ptr(inverse), w ? handle_to_ptr(w) : nullptr,
        handle_to_ptr(grad), handle_to_ptr(dx), rows, dim);
    if (!check_cuda_launch("LayerNorm input backward launch"))
        return 1;
    generic_layer_norm_affine_grad_kernel<<<linear_grid_size(dim, 256), 256>>>(
        handle_to_ptr(xhat), handle_to_ptr(grad), handle_to_ptr(dw), handle_to_ptr(db), rows, dim);
    return check_cuda_launch("LayerNorm affine backward launch") ? 0 : 1;
}

extern "C" int lumen_cuda_masked_softmax(uint64_t input, uint64_t output, size_t rows,
                                         size_t keys) {
    if (!validate_handle(input, "masked softmax input") ||
        !validate_handle(output, "masked softmax output") || rows == 0 || keys == 0)
        return 1;
    generic_masked_softmax_kernel<<<static_cast<unsigned int>(std::min(rows, size_t(65535))),
                                    256>>>(handle_to_ptr(input), handle_to_ptr(output), rows, keys);
    return check_cuda_launch("masked softmax launch") ? 0 : 1;
}
