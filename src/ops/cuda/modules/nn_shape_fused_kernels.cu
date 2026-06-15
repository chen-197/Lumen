__global__ void softmax_lastdim_kernel(const float* input, float* out, size_t outer, size_t last_dim) {
    for (size_t row = blockIdx.x * blockDim.x + threadIdx.x;
         row < outer;
         row += static_cast<size_t>(blockDim.x) * gridDim.x) {
        const float* row_in = input + row * last_dim;
        float* row_out = out + row * last_dim;

        float max_val = -FLT_MAX;
        for (size_t j = 0; j < last_dim; ++j) {
            max_val = fmaxf(max_val, row_in[j]);
        }

        float sum_exp = 0.0f;
        for (size_t j = 0; j < last_dim; ++j) {
            float e = expf(row_in[j] - max_val);
            row_out[j] = e;
            sum_exp += e;
        }

        float inv_sum = 1.0f / sum_exp;
        for (size_t j = 0; j < last_dim; ++j) {
            row_out[j] *= inv_sum;
        }
    }
}

template <typename T>
__global__ void softmax_lastdim_typed_kernel(
    const T* input,
    float input_scale,
    float* out,
    size_t outer,
    size_t last_dim) {
    for (size_t row = blockIdx.x * blockDim.x + threadIdx.x;
         row < outer;
         row += static_cast<size_t>(blockDim.x) * gridDim.x) {
        size_t row_offset = row * last_dim;
        float* row_out = out + row_offset;

        float max_val = -FLT_MAX;
        for (size_t j = 0; j < last_dim; ++j) {
            max_val = fmaxf(max_val, typed_value_to_float(input, input_scale, row_offset + j));
        }

        float sum_exp = 0.0f;
        for (size_t j = 0; j < last_dim; ++j) {
            float e = expf(typed_value_to_float(input, input_scale, row_offset + j) - max_val);
            row_out[j] = e;
            sum_exp += e;
        }

        float inv_sum = 1.0f / sum_exp;
        for (size_t j = 0; j < last_dim; ++j) {
            row_out[j] *= inv_sum;
        }
    }
}

__global__ void softmax_lastdim_backward_kernel(
    const float* output,
    const float* grad,
    float* out,
    size_t outer,
    size_t last_dim) {
    for (size_t row = blockIdx.x * blockDim.x + threadIdx.x;
         row < outer;
         row += static_cast<size_t>(blockDim.x) * gridDim.x) {
        const float* row_y = output + row * last_dim;
        const float* row_grad = grad + row * last_dim;
        float* row_out = out + row * last_dim;

        float dot = 0.0f;
        for (size_t j = 0; j < last_dim; ++j) {
            dot += row_y[j] * row_grad[j];
        }

        for (size_t j = 0; j < last_dim; ++j) {
            row_out[j] = row_y[j] * (row_grad[j] - dot);
        }
    }
}

__global__ void fused_softmax_kernel(
    const float* input,
    float* out,
    size_t rows,
    size_t q_len,
    size_t k_len,
    float scale,
    int is_causal) {
    for (size_t row = blockIdx.x * blockDim.x + threadIdx.x;
         row < rows;
         row += static_cast<size_t>(blockDim.x) * gridDim.x) {
        size_t q_idx = row % q_len;
        const float* row_in = input + row * k_len;
        float* row_out = out + row * k_len;

        float max_val = -FLT_MAX;
        for (size_t j = 0; j < k_len; ++j) {
            bool masked = is_causal != 0 && q_len > 1 && j > q_idx;
            if (masked) {
                continue;
            }
            float value = row_in[j] * scale;
            max_val = fmaxf(max_val, value);
        }

        float sum_exp = 0.0f;
        for (size_t j = 0; j < k_len; ++j) {
            bool masked = is_causal != 0 && q_len > 1 && j > q_idx;
            if (masked) {
                row_out[j] = 0.0f;
                continue;
            }
            float value = expf(row_in[j] * scale - max_val);
            row_out[j] = value;
            sum_exp += value;
        }

        float inv_sum = 1.0f / (sum_exp + 1.0e-10f);
        for (size_t j = 0; j < k_len; ++j) {
            row_out[j] *= inv_sum;
        }
    }
}

__global__ void fused_softmax_block_kernel(
    const float* input,
    float* out,
    size_t rows,
    size_t q_len,
    size_t k_len,
    float scale,
    int is_causal) {
    __shared__ float partials[256];
    unsigned int tid = threadIdx.x;
    for (size_t row = blockIdx.x; row < rows; row += gridDim.x) {
        size_t q_idx = row % q_len;
        const float* row_in = input + row * k_len;
        float* row_out = out + row * k_len;

        float local_max = -FLT_MAX;
        for (size_t j = tid; j < k_len; j += blockDim.x) {
            bool masked = is_causal != 0 && q_len > 1 && j > q_idx;
            if (!masked) {
                local_max = fmaxf(local_max, row_in[j] * scale);
            }
        }
        partials[tid] = local_max;
        __syncthreads();
        for (unsigned int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
            if (tid < offset) {
                partials[tid] = fmaxf(partials[tid], partials[tid + offset]);
            }
            __syncthreads();
        }
        float max_val = partials[0];

        float local_sum = 0.0f;
        for (size_t j = tid; j < k_len; j += blockDim.x) {
            bool masked = is_causal != 0 && q_len > 1 && j > q_idx;
            if (masked) {
                row_out[j] = 0.0f;
            } else {
                float value = expf(row_in[j] * scale - max_val);
                row_out[j] = value;
                local_sum += value;
            }
        }
        partials[tid] = local_sum;
        __syncthreads();
        for (unsigned int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
            if (tid < offset) {
                partials[tid] += partials[tid + offset];
            }
            __syncthreads();
        }
        float inv_sum = 1.0f / (partials[0] + 1.0e-10f);

        for (size_t j = tid; j < k_len; j += blockDim.x) {
            row_out[j] *= inv_sum;
        }
        __syncthreads();
    }
}

__global__ void fused_softmax_with_past_kernel(
    const float* input,
    float* out,
    size_t rows,
    size_t q_len,
    size_t k_len,
    float scale,
    int is_causal,
    size_t past_len) {
    for (size_t row = blockIdx.x * blockDim.x + threadIdx.x;
         row < rows;
         row += static_cast<size_t>(blockDim.x) * gridDim.x) {
        size_t q_idx = row % q_len;
        size_t causal_limit = past_len + q_idx;
        const float* row_in = input + row * k_len;
        float* row_out = out + row * k_len;

        float max_val = -FLT_MAX;
        for (size_t j = 0; j < k_len; ++j) {
            bool masked = is_causal != 0 && q_len > 1 && j > causal_limit;
            if (masked) {
                continue;
            }
            float value = row_in[j] * scale;
            max_val = fmaxf(max_val, value);
        }

        float sum_exp = 0.0f;
        for (size_t j = 0; j < k_len; ++j) {
            bool masked = is_causal != 0 && q_len > 1 && j > causal_limit;
            if (masked) {
                row_out[j] = 0.0f;
                continue;
            }
            float value = expf(row_in[j] * scale - max_val);
            row_out[j] = value;
            sum_exp += value;
        }

        float inv_sum = 1.0f / (sum_exp + 1.0e-10f);
        for (size_t j = 0; j < k_len; ++j) {
            row_out[j] *= inv_sum;
        }
    }
}

__global__ void fused_softmax_backward_kernel(
    const float* output,
    const float* grad,
    float* out,
    size_t rows,
    size_t k_len,
    float scale) {
    for (size_t row = blockIdx.x * blockDim.x + threadIdx.x;
         row < rows;
         row += static_cast<size_t>(blockDim.x) * gridDim.x) {
        const float* row_y = output + row * k_len;
        const float* row_grad = grad + row * k_len;
        float* row_out = out + row * k_len;

        float dot = 0.0f;
        for (size_t j = 0; j < k_len; ++j) {
            dot += row_y[j] * row_grad[j];
        }

        for (size_t j = 0; j < k_len; ++j) {
            row_out[j] = scale * row_y[j] * (row_grad[j] - dot);
        }
    }
}

__global__ void fused_softmax_backward_block_kernel(
    const float* output,
    const float* grad,
    float* out,
    size_t rows,
    size_t k_len,
    float scale) {
    __shared__ float partials[256];
    unsigned int tid = threadIdx.x;
    for (size_t row = blockIdx.x; row < rows; row += gridDim.x) {
        const float* row_y = output + row * k_len;
        const float* row_grad = grad + row * k_len;
        float* row_out = out + row * k_len;

        float local_dot = 0.0f;
        for (size_t j = tid; j < k_len; j += blockDim.x) {
            local_dot += row_y[j] * row_grad[j];
        }
        partials[tid] = local_dot;
        __syncthreads();
        for (unsigned int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
            if (tid < offset) {
                partials[tid] += partials[tid + offset];
            }
            __syncthreads();
        }
        float dot = partials[0];

        for (size_t j = tid; j < k_len; j += blockDim.x) {
            row_out[j] = scale * row_y[j] * (row_grad[j] - dot);
        }
        __syncthreads();
    }
}

__global__ void embedding_kernel(
    const float* indices,
    const float* weight,
    float* out,
    size_t num_indices,
    size_t vocab_size,
    size_t embed_dim,
    int* status) {
    size_t total = num_indices * embed_dim;
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += static_cast<size_t>(blockDim.x) * gridDim.x) {
        size_t token_idx = idx / embed_dim;
        size_t col = idx % embed_dim;
        float raw_index = indices[token_idx];
        if (!isfinite(raw_index) || raw_index < 0.0f || floorf(raw_index) != raw_index) {
            atomicCAS(status, 0, 1);
            out[idx] = 0.0f;
            continue;
        }

        size_t row = static_cast<size_t>(raw_index);
        if (row >= vocab_size) {
            atomicCAS(status, 0, 2);
            out[idx] = 0.0f;
            continue;
        }

        out[idx] = weight[row * embed_dim + col];
    }
}

template <typename WeightT>
__global__ void embedding_typed_kernel(
    const float* indices,
    const WeightT* weight,
    float weight_scale,
    float* out,
    size_t num_indices,
    size_t vocab_size,
    size_t embed_dim,
    int* status) {
    size_t total = num_indices * embed_dim;
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += static_cast<size_t>(blockDim.x) * gridDim.x) {
        size_t token_idx = idx / embed_dim;
        size_t col = idx % embed_dim;
        float raw_index = indices[token_idx];
        if (!isfinite(raw_index) || raw_index < 0.0f || floorf(raw_index) != raw_index) {
            atomicCAS(status, 0, 1);
            out[idx] = 0.0f;
            continue;
        }

        size_t row = static_cast<size_t>(raw_index);
        if (row >= vocab_size) {
            atomicCAS(status, 0, 2);
            out[idx] = 0.0f;
            continue;
        }

        out[idx] = typed_value_to_float(weight, weight_scale, row * embed_dim + col);
    }
}

template <typename WeightT>
__global__ void embedding_typed_same_dtype_kernel(
    const float* indices,
    const WeightT* weight,
    WeightT* out,
    size_t num_indices,
    size_t vocab_size,
    size_t embed_dim,
    int* status) {
    size_t total = num_indices * embed_dim;
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += static_cast<size_t>(blockDim.x) * gridDim.x) {
        size_t token_idx = idx / embed_dim;
        size_t col = idx % embed_dim;
        float raw_index = indices[token_idx];
        if (!isfinite(raw_index) || raw_index < 0.0f || floorf(raw_index) != raw_index) {
            atomicCAS(status, 0, 1);
            out[idx] = WeightT{};
            continue;
        }

        size_t row = static_cast<size_t>(raw_index);
        if (row >= vocab_size) {
            atomicCAS(status, 0, 2);
            out[idx] = WeightT{};
            continue;
        }

        out[idx] = weight[row * embed_dim + col];
    }
}

__global__ void embedding_backward_kernel(
    const float* indices,
    const float* grad,
    float* grad_weight,
    size_t num_indices,
    size_t vocab_size,
    size_t embed_dim,
    int* status) {
    size_t total = num_indices * embed_dim;
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += static_cast<size_t>(blockDim.x) * gridDim.x) {
        size_t token_idx = idx / embed_dim;
        size_t col = idx % embed_dim;
        float raw_index = indices[token_idx];
        if (!isfinite(raw_index) || raw_index < 0.0f || floorf(raw_index) != raw_index) {
            atomicCAS(status, 0, 1);
            continue;
        }

        size_t row = static_cast<size_t>(raw_index);
        if (row >= vocab_size) {
            atomicCAS(status, 0, 2);
            continue;
        }

        atomicAdd(grad_weight + row * embed_dim + col, grad[idx]);
    }
}

__global__ void rms_norm_kernel(
    const float* input,
    const float* weight,
    float* out,
    size_t rows,
    size_t dim,
    float eps) {
    for (size_t row = blockIdx.x * blockDim.x + threadIdx.x;
         row < rows;
         row += static_cast<size_t>(blockDim.x) * gridDim.x) {
        const float* row_in = input + row * dim;
        float* row_out = out + row * dim;

        float sum_sq = 0.0f;
        for (size_t j = 0; j < dim; ++j) {
            float value = row_in[j];
            sum_sq += value * value;
        }

        float inv_rms = rsqrtf(sum_sq / static_cast<float>(dim) + eps);
        for (size_t j = 0; j < dim; ++j) {
            row_out[j] = row_in[j] * inv_rms * weight[j];
        }
    }
}

__global__ void rms_norm_backward_kernel(
    const float* input,
    const float* weight,
    const float* grad,
    float* grad_input,
    float* grad_weight,
    size_t rows,
    size_t dim,
    float eps) {
    for (size_t row = blockIdx.x * blockDim.x + threadIdx.x;
         row < rows;
         row += static_cast<size_t>(blockDim.x) * gridDim.x) {
        const float* row_x = input + row * dim;
        const float* row_g = grad + row * dim;
        float* row_dx = grad_input + row * dim;

        float sum_sq = 0.0f;
        for (size_t j = 0; j < dim; ++j) {
            float x = row_x[j];
            sum_sq += x * x;
        }
        float inv_rms = rsqrtf(sum_sq / static_cast<float>(dim) + eps);

        float dot = 0.0f;
        for (size_t j = 0; j < dim; ++j) {
            dot += (row_g[j] * weight[j]) * (row_x[j] * inv_rms);
        }
        float mean_dot = dot / static_cast<float>(dim);

        for (size_t j = 0; j < dim; ++j) {
            float x_norm = row_x[j] * inv_rms;
            row_dx[j] = inv_rms * (row_g[j] * weight[j] - x_norm * mean_dot);
            atomicAdd(grad_weight + j, row_g[j] * row_x[j] * inv_rms);
        }
    }
}

template <typename InputT, typename WeightT>
__global__ void rms_norm_typed_kernel(
    const InputT* input,
    float input_scale,
    const WeightT* weight,
    float weight_scale,
    float* out,
    size_t rows,
    size_t dim,
    float eps) {
    for (size_t row = blockIdx.x * blockDim.x + threadIdx.x;
         row < rows;
         row += static_cast<size_t>(blockDim.x) * gridDim.x) {
        size_t row_offset = row * dim;
        float sum_sq = 0.0f;
        for (size_t j = 0; j < dim; ++j) {
            float value = typed_value_to_float(input, input_scale, row_offset + j);
            sum_sq += value * value;
        }

        float inv_rms = rsqrtf(sum_sq / static_cast<float>(dim) + eps);
        for (size_t j = 0; j < dim; ++j) {
            float x = typed_value_to_float(input, input_scale, row_offset + j);
            float w = typed_value_to_float(weight, weight_scale, j);
            out[row_offset + j] = x * inv_rms * w;
        }
    }
}

template <typename InputT, typename WeightT>
__global__ void rms_norm_i8_absmax_blocks_kernel(
    const InputT* input,
    float input_scale,
    const WeightT* weight,
    float weight_scale,
    float* partial,
    size_t rows,
    size_t dim,
    float eps) {
    extern __shared__ float shared[];
    size_t tid = threadIdx.x;
    float max_abs = 0.0f;
    for (size_t row = blockIdx.x * blockDim.x + tid;
         row < rows;
         row += static_cast<size_t>(blockDim.x) * gridDim.x) {
        size_t row_offset = row * dim;
        float sum_sq = 0.0f;
        for (size_t j = 0; j < dim; ++j) {
            float value = typed_value_to_float(input, input_scale, row_offset + j);
            sum_sq += value * value;
        }

        float inv_rms = rsqrtf(sum_sq / static_cast<float>(dim) + eps);
        for (size_t j = 0; j < dim; ++j) {
            float x = typed_value_to_float(input, input_scale, row_offset + j);
            float w = typed_value_to_float(weight, weight_scale, j);
            max_abs = fmaxf(max_abs, fabsf(x * inv_rms * w));
        }
    }
    shared[tid] = max_abs;
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

template <typename InputT, typename WeightT>
__global__ void rms_norm_i8_typed_out_kernel(
    const InputT* input,
    float input_scale,
    const WeightT* weight,
    float weight_scale,
    int8_t* out,
    size_t rows,
    size_t dim,
    float eps,
    float out_scale) {
    for (size_t row = blockIdx.x * blockDim.x + threadIdx.x;
         row < rows;
         row += static_cast<size_t>(blockDim.x) * gridDim.x) {
        size_t row_offset = row * dim;
        float sum_sq = 0.0f;
        for (size_t j = 0; j < dim; ++j) {
            float value = typed_value_to_float(input, input_scale, row_offset + j);
            sum_sq += value * value;
        }

        float inv_rms = rsqrtf(sum_sq / static_cast<float>(dim) + eps);
        for (size_t j = 0; j < dim; ++j) {
            float x = typed_value_to_float(input, input_scale, row_offset + j);
            float w = typed_value_to_float(weight, weight_scale, j);
            float q = nearbyintf((x * inv_rms * w) / out_scale);
            q = fminf(127.0f, fmaxf(-127.0f, q));
            out[row_offset + j] = static_cast<int8_t>(q);
        }
    }
}

template <typename InputT, typename WeightT>
__global__ void rms_norm_backward_typed_kernel(
    const InputT* input,
    float input_scale,
    const WeightT* weight,
    float weight_scale,
    const float* grad,
    float* grad_input,
    float* grad_weight,
    size_t rows,
    size_t dim,
    float eps) {
    for (size_t row = blockIdx.x * blockDim.x + threadIdx.x;
         row < rows;
         row += static_cast<size_t>(blockDim.x) * gridDim.x) {
        size_t row_offset = row * dim;
        float sum_sq = 0.0f;
        for (size_t j = 0; j < dim; ++j) {
            float x = typed_value_to_float(input, input_scale, row_offset + j);
            sum_sq += x * x;
        }
        float inv_rms = rsqrtf(sum_sq / static_cast<float>(dim) + eps);

        float dot = 0.0f;
        for (size_t j = 0; j < dim; ++j) {
            float x_norm = typed_value_to_float(input, input_scale, row_offset + j) * inv_rms;
            float w = typed_value_to_float(weight, weight_scale, j);
            dot += (grad[row_offset + j] * w) * x_norm;
        }
        float mean_dot = dot / static_cast<float>(dim);

        for (size_t j = 0; j < dim; ++j) {
            float x = typed_value_to_float(input, input_scale, row_offset + j);
            float x_norm = x * inv_rms;
            float w = typed_value_to_float(weight, weight_scale, j);
            grad_input[row_offset + j] =
                inv_rms * (grad[row_offset + j] * w - x_norm * mean_dot);
            atomicAdd(grad_weight + j, grad[row_offset + j] * x_norm);
        }
    }
}

template <typename InputT, typename WeightT>
int launch_rms_norm_typed(
    const InputT* input,
    float input_scale,
    const WeightT* weight,
    float weight_scale,
    float* out,
    size_t rows,
    size_t dim,
    float eps) {
    constexpr int block_size = 128;
    const unsigned int grid_size = linear_grid_size(rows, block_size);
    rms_norm_typed_kernel<InputT, WeightT><<<grid_size, block_size>>>(
        input,
        input_scale,
        weight,
        weight_scale,
        out,
        rows,
        dim,
        eps);
    return check_cuda_launch("CUDA typed RMSNorm kernel launch failed") ? 0 : 1;
}

template <typename InputT, typename WeightT>
int launch_rms_norm_i8_typed_out(
    const InputT* input,
    float input_scale,
    const WeightT* weight,
    float weight_scale,
    int8_t* out,
    size_t rows,
    size_t dim,
    float eps,
    float* out_scale) {
    if (out_scale == nullptr) {
        set_error("CUDA I8 typed-output RMSNorm scale output is null");
        return 1;
    }

    constexpr int block_size = 128;
    const unsigned int grid_size = linear_grid_size(rows, block_size);
    thread_local ReusableCudaWorkspace partial_workspace;
    if (!partial_workspace.ensure(
            static_cast<size_t>(grid_size) * sizeof(float),
            "failed to prepare CUDA I8 typed-output RMSNorm reduction buffer")) {
        return 1;
    }
    float* partial = static_cast<float*>(partial_workspace.ptr);

    rms_norm_i8_absmax_blocks_kernel<InputT, WeightT><<<
        grid_size,
        block_size,
        block_size * sizeof(float)>>>(
        input,
        input_scale,
        weight,
        weight_scale,
        partial,
        rows,
        dim,
        eps);
    if (!check_cuda_launch("CUDA I8 typed-output RMSNorm absmax kernel launch failed")) {
        return 1;
    }

    float max_abs = 0.0f;
    bool reduced = reduce_absmax_partials_to_host(
        partial,
        static_cast<size_t>(grid_size),
        &max_abs,
        "CUDA I8 typed-output RMSNorm final absmax reduction kernel launch failed");
    if (!reduced) {
        return 1;
    }

    float scale = max_abs > 0.0f && isfinite(max_abs) ? fmaxf(max_abs / 127.0f, FLT_MIN) : 1.0f;
    *out_scale = scale;
    rms_norm_i8_typed_out_kernel<InputT, WeightT><<<grid_size, block_size>>>(
        input,
        input_scale,
        weight,
        weight_scale,
        out,
        rows,
        dim,
        eps,
        scale);
    return check_cuda_launch("CUDA I8 typed-output RMSNorm kernel launch failed") ? 0 : 1;
}

template <typename InputT, typename WeightT>
int launch_rms_norm_backward_typed(
    const InputT* input,
    float input_scale,
    const WeightT* weight,
    float weight_scale,
    const float* grad,
    float* grad_input,
    float* grad_weight,
    size_t rows,
    size_t dim,
    float eps) {
    cudaError_t status = cudaMemset(grad_weight, 0, dim * sizeof(float));
    if (status != cudaSuccess) {
        set_cuda_error("CUDA typed RMSNorm backward weight grad initialization failed", status);
        return 1;
    }

    constexpr int block_size = 128;
    const unsigned int grid_size = linear_grid_size(rows, block_size);
    rms_norm_backward_typed_kernel<InputT, WeightT><<<grid_size, block_size>>>(
        input,
        input_scale,
        weight,
        weight_scale,
        grad,
        grad_input,
        grad_weight,
        rows,
        dim,
        eps);
    return check_cuda_launch("CUDA typed RMSNorm backward kernel launch failed") ? 0 : 1;
}

template <typename InputT>
int dispatch_rms_norm_weight_typed(
    const InputT* input,
    float input_scale,
    uint64_t weight_handle,
    int weight_dtype,
    float weight_scale,
    float* out,
    size_t rows,
    size_t dim,
    float eps) {
    switch (weight_dtype) {
        case kDTypeF32:
            return launch_rms_norm_typed(
                input, input_scale, handle_to_ptr(weight_handle), weight_scale, out, rows, dim, eps);
        case kDTypeF16:
            return launch_rms_norm_typed(
                input,
                input_scale,
                reinterpret_cast<const __half*>(handle_to_ptr(weight_handle)),
                weight_scale,
                out,
                rows,
                dim,
                eps);
        case kDTypeBF16:
            return launch_rms_norm_typed(
                input,
                input_scale,
                reinterpret_cast<const __nv_bfloat16*>(handle_to_ptr(weight_handle)),
                weight_scale,
                out,
                rows,
                dim,
                eps);
        case kDTypeI8:
            return launch_rms_norm_typed(
                input,
                input_scale,
                reinterpret_cast<const int8_t*>(handle_to_ptr(weight_handle)),
                weight_scale,
                out,
                rows,
                dim,
                eps);
        default:
            set_error("CUDA typed RMSNorm received unsupported weight dtype");
            return 1;
    }
}

template <typename InputT>
int dispatch_rms_norm_i8_typed_out_weight(
    const InputT* input,
    float input_scale,
    uint64_t weight_handle,
    int weight_dtype,
    float weight_scale,
    int8_t* out,
    size_t rows,
    size_t dim,
    float eps,
    float* out_scale) {
    switch (weight_dtype) {
        case kDTypeF32:
            return launch_rms_norm_i8_typed_out(
                input,
                input_scale,
                handle_to_ptr(weight_handle),
                weight_scale,
                out,
                rows,
                dim,
                eps,
                out_scale);
        case kDTypeF16:
            return launch_rms_norm_i8_typed_out(
                input,
                input_scale,
                reinterpret_cast<const __half*>(handle_to_ptr(weight_handle)),
                weight_scale,
                out,
                rows,
                dim,
                eps,
                out_scale);
        case kDTypeBF16:
            return launch_rms_norm_i8_typed_out(
                input,
                input_scale,
                reinterpret_cast<const __nv_bfloat16*>(handle_to_ptr(weight_handle)),
                weight_scale,
                out,
                rows,
                dim,
                eps,
                out_scale);
        case kDTypeI8:
            return launch_rms_norm_i8_typed_out(
                input,
                input_scale,
                reinterpret_cast<const int8_t*>(handle_to_ptr(weight_handle)),
                weight_scale,
                out,
                rows,
                dim,
                eps,
                out_scale);
        default:
            set_error("CUDA I8 typed-output RMSNorm received unsupported weight dtype");
            return 1;
    }
}

template <typename InputT>
int dispatch_rms_norm_backward_weight_typed(
    const InputT* input,
    float input_scale,
    uint64_t weight_handle,
    int weight_dtype,
    float weight_scale,
    const float* grad,
    float* grad_input,
    float* grad_weight,
    size_t rows,
    size_t dim,
    float eps) {
    switch (weight_dtype) {
        case kDTypeF32:
            return launch_rms_norm_backward_typed(
                input,
                input_scale,
                handle_to_ptr(weight_handle),
                weight_scale,
                grad,
                grad_input,
                grad_weight,
                rows,
                dim,
                eps);
        case kDTypeF16:
            return launch_rms_norm_backward_typed(
                input,
                input_scale,
                reinterpret_cast<const __half*>(handle_to_ptr(weight_handle)),
                weight_scale,
                grad,
                grad_input,
                grad_weight,
                rows,
                dim,
                eps);
        case kDTypeBF16:
            return launch_rms_norm_backward_typed(
                input,
                input_scale,
                reinterpret_cast<const __nv_bfloat16*>(handle_to_ptr(weight_handle)),
                weight_scale,
                grad,
                grad_input,
                grad_weight,
                rows,
                dim,
                eps);
        case kDTypeI8:
            return launch_rms_norm_backward_typed(
                input,
                input_scale,
                reinterpret_cast<const int8_t*>(handle_to_ptr(weight_handle)),
                weight_scale,
                grad,
                grad_input,
                grad_weight,
                rows,
                dim,
                eps);
        default:
            set_error("CUDA typed RMSNorm backward received unsupported weight dtype");
            return 1;
    }
}

__global__ void permute_kernel(
    const float* input,
    float* out,
    size_t ndim,
    const size_t* out_shape,
    const size_t* out_strides,
    const size_t* mapped_input_strides,
    size_t len) {
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t out_idx = start; out_idx < len; out_idx += stride) {
        size_t remaining = out_idx;
        size_t input_idx = 0;
        for (size_t i = 0; i < ndim; ++i) {
            size_t coord = 0;
            if (out_shape[i] > 0) {
                coord = remaining / out_strides[i];
                remaining %= out_strides[i];
            }
            input_idx += coord * mapped_input_strides[i];
        }
        out[out_idx] = input[input_idx];
    }
}

template <typename T>
__global__ void permute_typed_kernel(
    const T* input,
    T* out,
    size_t ndim,
    const size_t* out_shape,
    const size_t* out_strides,
    const size_t* mapped_input_strides,
    size_t len) {
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t out_idx = start; out_idx < len; out_idx += stride) {
        size_t remaining = out_idx;
        size_t input_idx = 0;
        for (size_t i = 0; i < ndim; ++i) {
            size_t coord = 0;
            if (out_shape[i] > 0) {
                coord = remaining / out_strides[i];
                remaining %= out_strides[i];
            }
            input_idx += coord * mapped_input_strides[i];
        }
        out[out_idx] = input[input_idx];
    }
}

__global__ void permute4d_kernel(
    const float* input,
    float* out,
    size_t o0,
    size_t o1,
    size_t o2,
    size_t o3,
    size_t os0,
    size_t os1,
    size_t os2,
    size_t os3,
    size_t is0,
    size_t is1,
    size_t is2,
    size_t is3,
    size_t len) {
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t out_idx = start; out_idx < len; out_idx += stride) {
        size_t remaining = out_idx;
        size_t c0 = remaining / os0;
        remaining %= os0;
        size_t c1 = remaining / os1;
        remaining %= os1;
        size_t c2 = remaining / os2;
        remaining %= os2;
        size_t c3 = remaining / os3;
        if (c0 < o0 && c1 < o1 && c2 < o2 && c3 < o3) {
            out[out_idx] = input[c0 * is0 + c1 * is1 + c2 * is2 + c3 * is3];
        }
    }
}

template <typename T>
__global__ void permute4d_typed_kernel(
    const T* input,
    T* out,
    size_t o0,
    size_t o1,
    size_t o2,
    size_t o3,
    size_t os0,
    size_t os1,
    size_t os2,
    size_t os3,
    size_t is0,
    size_t is1,
    size_t is2,
    size_t is3,
    size_t len) {
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t out_idx = start; out_idx < len; out_idx += stride) {
        size_t remaining = out_idx;
        size_t c0 = remaining / os0;
        remaining %= os0;
        size_t c1 = remaining / os1;
        remaining %= os1;
        size_t c2 = remaining / os2;
        remaining %= os2;
        size_t c3 = remaining / os3;
        if (c0 < o0 && c1 < o1 && c2 < o2 && c3 < o3) {
            out[out_idx] = input[c0 * is0 + c1 * is1 + c2 * is2 + c3 * is3];
        }
    }
}

__global__ void permute4d_lastdim_vec4_kernel(
    const float4* input,
    float4* out,
    size_t o0,
    size_t o1,
    size_t o2,
    size_t o3,
    size_t os0,
    size_t os1,
    size_t os2,
    size_t is0,
    size_t is1,
    size_t is2,
    size_t vec_len) {
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t vec_out_idx = start; vec_out_idx < vec_len; vec_out_idx += stride) {
        size_t out_idx = vec_out_idx * 4;
        size_t remaining = out_idx;
        size_t c0 = remaining / os0;
        remaining %= os0;
        size_t c1 = remaining / os1;
        remaining %= os1;
        size_t c2 = remaining / os2;
        remaining %= os2;
        size_t c3 = remaining;
        if (c0 < o0 && c1 < o1 && c2 < o2 && c3 < o3) {
            size_t input_idx = c0 * is0 + c1 * is1 + c2 * is2 + c3;
            out[vec_out_idx] = input[input_idx / 4];
        }
    }
}

__global__ void bshd_to_bhsd_add_bias_kernel(
    const float* input,
    const float* bias,
    float* out,
    size_t seq,
    size_t heads,
    size_t dim,
    size_t len) {
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;
         out_idx < len;
         out_idx += stride) {
        size_t d = out_idx % dim;
        size_t tmp = out_idx / dim;
        size_t s = tmp % seq;
        tmp /= seq;
        size_t h = tmp % heads;
        size_t b = tmp / heads;

        size_t input_idx = ((b * seq + s) * heads + h) * dim + d;
        out[out_idx] = input[input_idx] + bias[h * dim + d];
    }
}

__global__ void bshd_to_bhsd_add_bias_vec4_kernel(
    const float4* input,
    const float4* bias,
    float4* out,
    size_t seq,
    size_t heads,
    size_t dim,
    size_t vec_len) {
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t vec_out_idx = blockIdx.x * blockDim.x + threadIdx.x;
         vec_out_idx < vec_len;
         vec_out_idx += stride) {
        size_t out_idx = vec_out_idx * 4;
        size_t d = out_idx % dim;
        size_t tmp = out_idx / dim;
        size_t s = tmp % seq;
        tmp /= seq;
        size_t h = tmp % heads;
        size_t b = tmp / heads;

        size_t input_idx = ((b * seq + s) * heads + h) * dim + d;
        float4 a = input[input_idx / 4];
        float4 bias_vec = bias[(h * dim + d) / 4];
        out[vec_out_idx] = make_float4(
            a.x + bias_vec.x,
            a.y + bias_vec.y,
            a.z + bias_vec.z,
            a.w + bias_vec.w);
    }
}

__global__ void slice_lastdim_kernel(
    const float* input,
    float* out,
    size_t outer,
    size_t input_last_dim,
    size_t start,
    size_t slice_len) {
    size_t total = outer * slice_len;
    const size_t begin = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t idx = begin; idx < total; idx += stride) {
        size_t row = idx / slice_len;
        size_t col = idx % slice_len;
        out[idx] = input[row * input_last_dim + start + col];
    }
}

template <typename T>
__global__ void slice_lastdim_typed_kernel(
    const T* input,
    T* out,
    size_t outer,
    size_t input_last_dim,
    size_t start,
    size_t slice_len) {
    size_t total = outer * slice_len;
    const size_t begin = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t idx = begin; idx < total; idx += stride) {
        size_t row = idx / slice_len;
        size_t col = idx % slice_len;
        out[idx] = input[row * input_last_dim + start + col];
    }
}

__global__ void slice_lastdim_backward_kernel(
    const float* grad,
    float* out,
    size_t outer,
    size_t input_last_dim,
    size_t start,
    size_t slice_len) {
    size_t total = outer * slice_len;
    const size_t begin = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t idx = begin; idx < total; idx += stride) {
        size_t row = idx / slice_len;
        size_t col = idx % slice_len;
        out[row * input_last_dim + start + col] = grad[idx];
    }
}

__global__ void append_kv_cache_kernel(
    float* dst,
    const float* src,
    size_t batch_size,
    size_t num_heads,
    size_t src_seq_len,
    size_t dst_seq_len,
    size_t dim,
    size_t dst_start) {
    size_t total = batch_size * num_heads * src_seq_len * dim;
    const size_t begin = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t idx = begin; idx < total; idx += stride) {
        size_t dd = idx % dim;
        size_t tmp = idx / dim;
        size_t ss = tmp % src_seq_len;
        tmp /= src_seq_len;
        size_t hh = tmp % num_heads;
        size_t bb = tmp / num_heads;

        size_t src_idx = (((bb * num_heads + hh) * src_seq_len + ss) * dim) + dd;
        size_t dst_idx = (((bb * num_heads + hh) * dst_seq_len + (dst_start + ss)) * dim) + dd;
        dst[dst_idx] = src[src_idx];
    }
}

__global__ void append_kv_cache_pair_kernel(
    float* k_dst,
    float* v_dst,
    const float* k_src,
    const float* v_src,
    size_t batch_size,
    size_t num_heads,
    size_t src_seq_len,
    size_t dst_seq_len,
    size_t dim,
    size_t dst_start) {
    size_t total = batch_size * num_heads * src_seq_len * dim;
    const size_t begin = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t idx = begin; idx < total; idx += stride) {
        size_t dd = idx % dim;
        size_t tmp = idx / dim;
        size_t ss = tmp % src_seq_len;
        tmp /= src_seq_len;
        size_t hh = tmp % num_heads;
        size_t bb = tmp / num_heads;

        size_t src_idx = (((bb * num_heads + hh) * src_seq_len + ss) * dim) + dd;
        size_t dst_idx = (((bb * num_heads + hh) * dst_seq_len + (dst_start + ss)) * dim) + dd;
        k_dst[dst_idx] = k_src[src_idx];
        v_dst[dst_idx] = v_src[src_idx];
    }
}

__global__ void decode_rope_q_append_kv_kernel(
    const float* q_src,
    const float* k_src,
    const float* v_src,
    const float* cos,
    const float* sin,
    float* q_out,
    float* k_cache,
    float* v_cache,
    size_t batch_size,
    size_t num_heads,
    size_t num_kv_heads,
    size_t dim,
    size_t dst_seq_len,
    size_t offset) {
    size_t half = dim / 2;
    size_t q_len = batch_size * num_heads * dim;
    size_t kv_len = batch_size * num_kv_heads * dim;
    size_t total = q_len > kv_len ? q_len : kv_len;
    const float* cos_row = cos + offset * dim;
    const float* sin_row = sin + offset * dim;
    const size_t begin = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t idx = begin; idx < total; idx += stride) {
        if (idx < q_len) {
            size_t dd = idx % dim;
            size_t pair = dd % half;
            size_t base = idx - dd;
            float x1 = q_src[base + pair];
            float x2 = q_src[base + pair + half];
            float c = cos_row[pair];
            float s = sin_row[pair];
            q_out[idx] = dd < half ? x1 * c - x2 * s : x1 * s + x2 * c;
        }

        if (idx < kv_len) {
            size_t dd = idx % dim;
            size_t pair = dd % half;
            size_t tmp = idx / dim;
            size_t hk = tmp % num_kv_heads;
            size_t bb = tmp / num_kv_heads;
            size_t base = idx - dd;
            float x1 = k_src[base + pair];
            float x2 = k_src[base + pair + half];
            float c = cos_row[pair];
            float s = sin_row[pair];
            size_t cache_idx = ((bb * num_kv_heads + hk) * dst_seq_len + offset) * dim + dd;
            k_cache[cache_idx] = dd < half ? x1 * c - x2 * s : x1 * s + x2 * c;
            v_cache[cache_idx] = v_src[idx];
        }
    }
}

__global__ void kv_cache_prefix_kernel(
    const float* src,
    float* out,
    size_t batch_size,
    size_t num_heads,
    size_t active_seq_len,
    size_t src_seq_len,
    size_t dim) {
    size_t total = batch_size * num_heads * active_seq_len * dim;
    const size_t begin = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t idx = begin; idx < total; idx += stride) {
        size_t dd = idx % dim;
        size_t tmp = idx / dim;
        size_t ss = tmp % active_seq_len;
        tmp /= active_seq_len;
        size_t hh = tmp % num_heads;
        size_t bb = tmp / num_heads;

        size_t src_idx = (((bb * num_heads + hh) * src_seq_len + ss) * dim) + dd;
        out[idx] = src[src_idx];
    }
}

template <typename T>
__global__ void kv_cache_prefix_typed_kernel(
    const T* src,
    T* out,
    size_t batch_size,
    size_t num_heads,
    size_t active_seq_len,
    size_t src_seq_len,
    size_t dim) {
    size_t total = batch_size * num_heads * active_seq_len * dim;
    const size_t begin = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t idx = begin; idx < total; idx += stride) {
        size_t dd = idx % dim;
        size_t tmp = idx / dim;
        size_t ss = tmp % active_seq_len;
        tmp /= active_seq_len;
        size_t hh = tmp % num_heads;
        size_t bb = tmp / num_heads;

        size_t src_idx = (((bb * num_heads + hh) * src_seq_len + ss) * dim) + dd;
        out[idx] = src[src_idx];
    }
}

__global__ void cat_kernel(
    const float* lhs,
    const float* rhs,
    float* out,
    const size_t* out_shape,
    const size_t* out_strides,
    const size_t* lhs_strides,
    const size_t* rhs_strides,
    size_t ndim,
    size_t axis,
    size_t lhs_axis_len,
    size_t len) {
    const size_t begin = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t out_idx = begin; out_idx < len; out_idx += stride) {
        size_t remaining = out_idx;
        size_t lhs_idx = 0;
        size_t rhs_idx = 0;
        bool use_rhs = false;

        for (size_t i = 0; i < ndim; ++i) {
            size_t coord = 0;
            if (out_shape[i] > 0) {
                coord = remaining / out_strides[i];
                remaining %= out_strides[i];
            }

            if (i == axis) {
                if (coord < lhs_axis_len) {
                    lhs_idx += coord * lhs_strides[i];
                } else {
                    use_rhs = true;
                    rhs_idx += (coord - lhs_axis_len) * rhs_strides[i];
                }
            } else {
                lhs_idx += coord * lhs_strides[i];
                rhs_idx += coord * rhs_strides[i];
            }
        }

        out[out_idx] = use_rhs ? rhs[rhs_idx] : lhs[lhs_idx];
    }
}

template <typename T>
__global__ void cat_typed_kernel(
    const T* lhs,
    const T* rhs,
    T* out,
    const size_t* out_shape,
    const size_t* out_strides,
    const size_t* lhs_strides,
    const size_t* rhs_strides,
    size_t ndim,
    size_t axis,
    size_t lhs_axis_len,
    size_t len) {
    const size_t begin = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t out_idx = begin; out_idx < len; out_idx += stride) {
        size_t remaining = out_idx;
        size_t lhs_idx = 0;
        size_t rhs_idx = 0;
        bool use_rhs = false;

        for (size_t i = 0; i < ndim; ++i) {
            size_t coord = 0;
            if (out_shape[i] > 0) {
                coord = remaining / out_strides[i];
                remaining %= out_strides[i];
            }

            if (i == axis) {
                if (coord < lhs_axis_len) {
                    lhs_idx += coord * lhs_strides[i];
                } else {
                    use_rhs = true;
                    rhs_idx += (coord - lhs_axis_len) * rhs_strides[i];
                }
            } else {
                lhs_idx += coord * lhs_strides[i];
                rhs_idx += coord * rhs_strides[i];
            }
        }

        out[out_idx] = use_rhs ? rhs[rhs_idx] : lhs[lhs_idx];
    }
}

__global__ void cat_backward_slice_kernel(
    const float* grad,
    float* out,
    const size_t* input_shape,
    const size_t* input_strides,
    const size_t* out_strides,
    size_t ndim,
    size_t axis,
    size_t axis_start,
    size_t len) {
    const size_t begin = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t input_idx = begin; input_idx < len; input_idx += stride) {
        size_t remaining = input_idx;
        size_t grad_idx = 0;
        for (size_t i = 0; i < ndim; ++i) {
            size_t coord = 0;
            if (input_shape[i] > 0) {
                coord = remaining / input_strides[i];
                remaining %= input_strides[i];
            }
            if (i == axis) {
                coord += axis_start;
            }
            grad_idx += coord * out_strides[i];
        }
        out[input_idx] = grad[grad_idx];
    }
}

__global__ void repeat_kv_kernel(
    const float* input,
    float* out,
    size_t num_heads,
    size_t seq_len,
    size_t dim,
    size_t n_rep) {
    size_t head_stride = seq_len * dim;
    size_t len = num_heads * head_stride;
    const size_t begin = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t out_idx = begin; out_idx < len; out_idx += stride) {
        size_t out_head_idx = out_idx / head_stride;
        size_t within_head = out_idx % head_stride;
        size_t kv_head_idx = out_head_idx / n_rep;
        size_t input_idx = kv_head_idx * head_stride + within_head;
        out[out_idx] = input[input_idx];
    }
}

template <typename T>
__global__ void repeat_kv_typed_kernel(
    const T* input,
    T* out,
    size_t num_heads,
    size_t seq_len,
    size_t dim,
    size_t n_rep) {
    size_t head_stride = seq_len * dim;
    size_t len = num_heads * head_stride;
    const size_t begin = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t out_idx = begin; out_idx < len; out_idx += stride) {
        size_t out_head_idx = out_idx / head_stride;
        size_t within_head = out_idx % head_stride;
        size_t kv_head_idx = out_head_idx / n_rep;
        size_t input_idx = kv_head_idx * head_stride + within_head;
        out[out_idx] = input[input_idx];
    }
}

__global__ void repeat_kv_backward_kernel(
    const float* grad,
    float* out,
    size_t batch_size,
    size_t num_kv_heads,
    size_t seq_len,
    size_t dim,
    size_t n_rep) {
    size_t input_len = batch_size * num_kv_heads * seq_len * dim;
    const size_t begin = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t idx = begin; idx < input_len; idx += stride) {
        size_t within_dim = idx % dim;
        size_t tmp = idx / dim;
        size_t seq_idx = tmp % seq_len;
        tmp /= seq_len;
        size_t kv_head = tmp % num_kv_heads;
        size_t batch = tmp / num_kv_heads;

        size_t out_heads = num_kv_heads * n_rep;
        float acc = 0.0f;
        for (size_t rep = 0; rep < n_rep; ++rep) {
            size_t out_head = kv_head * n_rep + rep;
            size_t grad_idx = ((batch * out_heads + out_head) * seq_len + seq_idx) * dim + within_dim;
            acc += grad[grad_idx];
        }
        out[idx] = acc;
    }
}

__global__ void decode_attention_kernel(
    const float* q,
    const float* k,
    const float* v,
    float* out,
    size_t num_heads,
    size_t num_kv_heads,
    size_t active_seq_len,
    size_t cache_seq_len,
    size_t dim,
    size_t n_rep,
    size_t rows,
    float scale) {
    size_t tid = threadIdx.x;

    extern __shared__ float shared[];
    float* reduce = shared;
    float* q_shared = reduce + blockDim.x;
    float* ctx_shared = q_shared + dim;
    float* scalars = ctx_shared + dim;

    for (size_t row = blockIdx.x; row < rows; row += gridDim.x) {
        size_t hh = row % num_heads;
        size_t hk = hh / n_rep;
        const float* q_row = q + row * dim;
        float* out_row = out + row * dim;
        for (size_t i = tid; i < dim; i += blockDim.x) {
            q_shared[i] = q_row[i];
            ctx_shared[i] = 0.0f;
        }
        if (tid == 0) {
            scalars[0] = -FLT_MAX;
            scalars[1] = 0.0f;
            scalars[2] = 0.0f;
            scalars[3] = 0.0f;
        }
        __syncthreads();

        size_t batch_idx = row / num_heads;
        const float* k_base = k + (batch_idx * num_kv_heads + hk) * cache_seq_len * dim;
        const float* v_base = v + (batch_idx * num_kv_heads + hk) * cache_seq_len * dim;

        for (size_t pos = 0; pos < active_seq_len; ++pos) {
            const float* k_row = k_base + pos * dim;
            const float* v_row = v_base + pos * dim;

            float partial = 0.0f;
            for (size_t i = tid; i < dim; i += blockDim.x) {
                partial += q_shared[i] * k_row[i];
            }
            reduce[tid] = partial;
            __syncthreads();

            for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    reduce[tid] += reduce[tid + stride];
                }
                __syncthreads();
            }

            if (tid == 0) {
                float score = reduce[0] * scale;
                float m = scalars[0];
                float l = scalars[1];
                float prev_scale;
                float weight;
                if (score > m) {
                    prev_scale = l > 0.0f ? expf(m - score) : 0.0f;
                    weight = 1.0f;
                    l = l * prev_scale + 1.0f;
                    m = score;
                } else {
                    prev_scale = 1.0f;
                    weight = expf(score - m);
                    l += weight;
                }
                scalars[0] = m;
                scalars[1] = l;
                scalars[2] = prev_scale;
                scalars[3] = weight;
            }
            __syncthreads();

            float prev_scale = scalars[2];
            float weight = scalars[3];
            for (size_t i = tid; i < dim; i += blockDim.x) {
                ctx_shared[i] = ctx_shared[i] * prev_scale + weight * v_row[i];
            }
            __syncthreads();
        }

        float inv_l = 1.0f / (scalars[1] + 1e-9f);
        for (size_t i = tid; i < dim; i += blockDim.x) {
            out_row[i] = ctx_shared[i] * inv_l;
        }
        __syncthreads();
    }
}

__global__ void prefill_attention_kernel(
    const float* q,
    const float* k,
    const float* v,
    float* out,
    size_t num_heads,
    size_t num_kv_heads,
    size_t q_seq_len,
    size_t active_seq_len,
    size_t cache_seq_len,
    size_t dim,
    size_t n_rep,
    size_t past_len,
    size_t rows,
    float scale,
    int is_causal) {
    size_t tid = threadIdx.x;

    extern __shared__ float shared[];
    float* reduce = shared;
    float* q_shared = reduce + blockDim.x;
    float* ctx_shared = q_shared + dim;
    float* scalars = ctx_shared + dim;

    for (size_t row = blockIdx.x; row < rows; row += gridDim.x) {
        size_t sq = row % q_seq_len;
        size_t tmp = row / q_seq_len;
        size_t hh = tmp % num_heads;
        size_t batch_idx = tmp / num_heads;
        size_t hk = hh / n_rep;

        const float* q_row = q + ((batch_idx * num_heads + hh) * q_seq_len + sq) * dim;
        float* out_row = out + ((batch_idx * q_seq_len + sq) * num_heads + hh) * dim;
        for (size_t i = tid; i < dim; i += blockDim.x) {
            q_shared[i] = q_row[i];
            ctx_shared[i] = 0.0f;
        }
        if (tid == 0) {
            scalars[0] = -FLT_MAX;
            scalars[1] = 0.0f;
            scalars[2] = 0.0f;
            scalars[3] = 0.0f;
        }
        __syncthreads();

        const float* k_base = k + (batch_idx * num_kv_heads + hk) * cache_seq_len * dim;
        const float* v_base = v + (batch_idx * num_kv_heads + hk) * cache_seq_len * dim;
        size_t query_abs = past_len + sq;

        for (size_t pos = 0; pos < active_seq_len; ++pos) {
            if (is_causal != 0 && pos > query_abs) {
                break;
            }
            const float* k_row = k_base + pos * dim;
            const float* v_row = v_base + pos * dim;

            float partial = 0.0f;
            for (size_t i = tid; i < dim; i += blockDim.x) {
                partial += q_shared[i] * k_row[i];
            }
            reduce[tid] = partial;
            __syncthreads();

            for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    reduce[tid] += reduce[tid + stride];
                }
                __syncthreads();
            }

            if (tid == 0) {
                float score = reduce[0] * scale;
                float m = scalars[0];
                float l = scalars[1];
                float prev_scale;
                float weight;
                if (score > m) {
                    prev_scale = l > 0.0f ? expf(m - score) : 0.0f;
                    weight = 1.0f;
                    l = l * prev_scale + 1.0f;
                    m = score;
                } else {
                    prev_scale = 1.0f;
                    weight = expf(score - m);
                    l += weight;
                }
                scalars[0] = m;
                scalars[1] = l;
                scalars[2] = prev_scale;
                scalars[3] = weight;
            }
            __syncthreads();

            float prev_scale = scalars[2];
            float weight = scalars[3];
            for (size_t i = tid; i < dim; i += blockDim.x) {
                ctx_shared[i] = ctx_shared[i] * prev_scale + weight * v_row[i];
            }
            __syncthreads();
        }

        float inv_l = 1.0f / (scalars[1] + 1e-9f);
        for (size_t i = tid; i < dim; i += blockDim.x) {
            out_row[i] = ctx_shared[i] * inv_l;
        }
        __syncthreads();
    }
}

__global__ void silu_mul_kernel(const float* gate, const float* up, float* out, size_t len) {
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < len;
         idx += static_cast<size_t>(blockDim.x) * gridDim.x) {
        float g = gate[idx];
        float sig = 1.0f / (1.0f + expf(-g));
        out[idx] = (g * sig) * up[idx];
    }
}

template <typename InputT, typename WeightT>
__global__ void fused_gate_up_silu_typed_kernel(
    const InputT* input,
    float input_scale,
    const WeightT* gate,
    float gate_scale,
    const WeightT* up,
    float up_scale,
    float* out,
    size_t rows,
    size_t n_dim,
    size_t k_dim) {
    size_t len = rows * n_dim;
    for (size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < len;
         idx += static_cast<size_t>(blockDim.x) * gridDim.x) {
        size_t row = idx / n_dim;
        size_t col = idx % n_dim;
        size_t input_offset = row * k_dim;
        size_t weight_offset = col * k_dim;

        float gate_acc = 0.0f;
        float up_acc = 0.0f;
        for (size_t k = 0; k < k_dim; ++k) {
            float x = typed_value_to_float(input, input_scale, input_offset + k);
            gate_acc += x * typed_value_to_float(gate, gate_scale, weight_offset + k);
            up_acc += x * typed_value_to_float(up, up_scale, weight_offset + k);
        }

        float sig = 1.0f / (1.0f + expf(-gate_acc));
        out[idx] = (gate_acc * sig) * up_acc;
    }
}

template <typename InputT, typename WeightT>
__global__ void fused_gate_up_silu_matvec_typed_kernel(
    const InputT* input,
    float input_scale,
    const WeightT* gate,
    float gate_scale,
    const WeightT* up,
    float up_scale,
    float* out,
    size_t n_dim,
    size_t k_dim) {
    __shared__ float gate_partials[256];
    __shared__ float up_partials[256];
    size_t tid = threadIdx.x;
    for (size_t col = blockIdx.x; col < n_dim; col += gridDim.x) {
        float gate_acc = 0.0f;
        float up_acc = 0.0f;
        size_t weight_offset = col * k_dim;

        for (size_t k = tid; k < k_dim; k += blockDim.x) {
            float x = typed_value_to_float(input, input_scale, k);
            gate_acc += x * typed_value_to_float(gate, gate_scale, weight_offset + k);
            up_acc += x * typed_value_to_float(up, up_scale, weight_offset + k);
        }

        gate_partials[tid] = gate_acc;
        up_partials[tid] = up_acc;
        __syncthreads();

        for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                gate_partials[tid] += gate_partials[tid + stride];
                up_partials[tid] += up_partials[tid + stride];
            }
            __syncthreads();
        }

        if (tid == 0) {
            float gate_value = gate_partials[0];
            float sig = 1.0f / (1.0f + expf(-gate_value));
            out[col] = (gate_value * sig) * up_partials[0];
        }
        __syncthreads();
    }
}

template <typename InputT, typename WeightT>
__global__ void fused_qkv_matvec_typed_kernel(
    const InputT* input,
    float input_scale,
    const WeightT* q,
    float q_scale,
    const WeightT* k_weight,
    float k_scale,
    const WeightT* v,
    float v_scale,
    float* q_out,
    float* k_out,
    float* v_out,
    size_t q_n,
    size_t k_n,
    size_t k_dim) {
    __shared__ float partials[256];
    size_t tid = threadIdx.x;
    size_t total_n = q_n + k_n + k_n;
    for (size_t out_col = blockIdx.x; out_col < total_n; out_col += gridDim.x) {
        size_t local_col = out_col;
        const WeightT* weight = q;
        float current_weight_scale = q_scale;
        float* out = q_out;

        if (out_col >= q_n) {
            local_col = out_col - q_n;
            if (local_col < k_n) {
                weight = k_weight;
                current_weight_scale = k_scale;
                out = k_out;
            } else {
                local_col -= k_n;
                weight = v;
                current_weight_scale = v_scale;
                out = v_out;
            }
        }

        float acc = 0.0f;
        size_t weight_offset = local_col * k_dim;
        for (size_t kk = tid; kk < k_dim; kk += blockDim.x) {
            float x = typed_value_to_float(input, input_scale, kk);
            float w = typed_value_to_float(weight, current_weight_scale, weight_offset + kk);
            acc += x * w;
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
            out[local_col] = partials[0];
        }
        __syncthreads();
    }
}

template <typename OutputT>
__device__ inline void write_float_output(OutputT* out, size_t idx, float value) {
    out[idx] = static_cast<OutputT>(value);
}

template <>
__device__ inline void write_float_output<__half>(__half* out, size_t idx, float value) {
    out[idx] = __float2half(value);
}

template <>
__device__ inline void write_float_output<__nv_bfloat16>(
    __nv_bfloat16* out,
    size_t idx,
    float value) {
    out[idx] = __float2bfloat16(value);
}

template <typename InputT, typename WeightT, typename OutputT>
__global__ void fused_gate_up_silu_typed_out_kernel(
    const InputT* input,
    float input_scale,
    const WeightT* gate,
    float gate_scale,
    const WeightT* up,
    float up_scale,
    OutputT* out,
    size_t rows,
    size_t n_dim,
    size_t k_dim) {
    size_t len = rows * n_dim;
    for (size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < len;
         idx += static_cast<size_t>(blockDim.x) * gridDim.x) {
        size_t row = idx / n_dim;
        size_t col = idx % n_dim;
        size_t input_offset = row * k_dim;
        size_t weight_offset = col * k_dim;

        float gate_acc = 0.0f;
        float up_acc = 0.0f;
        for (size_t k = 0; k < k_dim; ++k) {
            float x = typed_value_to_float(input, input_scale, input_offset + k);
            gate_acc += x * typed_value_to_float(gate, gate_scale, weight_offset + k);
            up_acc += x * typed_value_to_float(up, up_scale, weight_offset + k);
        }

        float sig = 1.0f / (1.0f + expf(-gate_acc));
        write_float_output(out, idx, (gate_acc * sig) * up_acc);
    }
}

template <typename InputT, typename WeightT>
int launch_fused_gate_up_silu_typed(
    const InputT* input,
    float input_scale,
    const WeightT* gate,
    float gate_scale,
    const WeightT* up,
    float up_scale,
    float* out,
    size_t rows,
    size_t n_dim,
    size_t k_dim) {
    constexpr int block_size = 256;
    size_t len = rows * n_dim;
    if (rows == 1) {
        fused_gate_up_silu_matvec_typed_kernel<InputT, WeightT>
            <<<linear_grid_size(n_dim, 1), block_size>>>(
                input,
                input_scale,
                gate,
                gate_scale,
                up,
                up_scale,
                out,
                n_dim,
                k_dim);
        return check_cuda_launch("CUDA typed fused gate/up matvec kernel launch failed") ? 0 : 1;
    }
    const unsigned int grid_size = linear_grid_size(len, block_size);
    fused_gate_up_silu_typed_kernel<InputT, WeightT><<<grid_size, block_size>>>(
        input,
        input_scale,
        gate,
        gate_scale,
        up,
        up_scale,
        out,
        rows,
        n_dim,
        k_dim);
    return check_cuda_launch("CUDA typed fused gate/up kernel launch failed") ? 0 : 1;
}

template <typename InputT, typename WeightT, typename OutputT>
int launch_fused_gate_up_silu_typed_out(
    const InputT* input,
    float input_scale,
    const WeightT* gate,
    float gate_scale,
    const WeightT* up,
    float up_scale,
    OutputT* out,
    size_t rows,
    size_t n_dim,
    size_t k_dim) {
    constexpr int block_size = 256;
    size_t len = rows * n_dim;
    const unsigned int grid_size = linear_grid_size(len, block_size);
    fused_gate_up_silu_typed_out_kernel<InputT, WeightT, OutputT><<<grid_size, block_size>>>(
        input,
        input_scale,
        gate,
        gate_scale,
        up,
        up_scale,
        out,
        rows,
        n_dim,
        k_dim);
    return check_cuda_launch("CUDA typed fused gate/up output kernel launch failed") ? 0 : 1;
}

template <typename InputT, typename WeightT>
__global__ void projection_typed_kernel(
    const InputT* input,
    float input_scale,
    const WeightT* weight,
    float weight_scale,
    float* out,
    size_t rows,
    size_t n_dim,
    size_t k_dim) {
    size_t len = rows * n_dim;
    for (size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < len;
         idx += static_cast<size_t>(blockDim.x) * gridDim.x) {
        size_t row = idx / n_dim;
        size_t col = idx % n_dim;
        size_t input_offset = row * k_dim;
        size_t weight_offset = col * k_dim;
        float acc = 0.0f;
        for (size_t k = 0; k < k_dim; ++k) {
            float x = typed_value_to_float(input, input_scale, input_offset + k);
            float w = typed_value_to_float(weight, weight_scale, weight_offset + k);
            acc += x * w;
        }
        out[idx] = acc;
    }
}

template <typename InputT, typename WeightT, typename OutputT>
__global__ void projection_typed_out_kernel(
    const InputT* input,
    float input_scale,
    const WeightT* weight,
    float weight_scale,
    OutputT* out,
    size_t rows,
    size_t n_dim,
    size_t k_dim) {
    size_t len = rows * n_dim;
    for (size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < len;
         idx += static_cast<size_t>(blockDim.x) * gridDim.x) {
        size_t row = idx / n_dim;
        size_t col = idx % n_dim;
        size_t input_offset = row * k_dim;
        size_t weight_offset = col * k_dim;
        float acc = 0.0f;
        for (size_t k = 0; k < k_dim; ++k) {
            float x = typed_value_to_float(input, input_scale, input_offset + k);
            float w = typed_value_to_float(weight, weight_scale, weight_offset + k);
            acc += x * w;
        }
        write_float_output(out, idx, acc);
    }
}

template <typename InputT, typename WeightT>
int launch_projection_typed(
    const InputT* input,
    float input_scale,
    const WeightT* weight,
    float weight_scale,
    float* out,
    size_t rows,
    size_t n_dim,
    size_t k_dim,
    const char* error_prefix) {
    constexpr int block_size = 256;
    size_t len = rows * n_dim;
    const unsigned int grid_size = linear_grid_size(len, block_size);
    projection_typed_kernel<InputT, WeightT><<<grid_size, block_size>>>(
        input,
        input_scale,
        weight,
        weight_scale,
        out,
        rows,
        n_dim,
        k_dim);
    return check_cuda_launch(error_prefix) ? 0 : 1;
}

template <typename InputT, typename WeightT>
int launch_fused_qkv_matvec_typed(
    const InputT* input,
    float input_scale,
    const WeightT* q,
    float q_scale,
    const WeightT* k,
    float k_scale,
    const WeightT* v,
    float v_scale,
    float* q_out,
    float* k_out,
    float* v_out,
    size_t q_n,
    size_t k_n,
    size_t k_dim) {
    constexpr int block_size = 256;
    size_t total_n = q_n + k_n + k_n;
    fused_qkv_matvec_typed_kernel<InputT, WeightT>
        <<<linear_grid_size(total_n, 1), block_size>>>(
            input,
            input_scale,
            q,
            q_scale,
            k,
            k_scale,
            v,
            v_scale,
            q_out,
            k_out,
            v_out,
            q_n,
            k_n,
            k_dim);
    return check_cuda_launch("CUDA typed fused qkv matvec kernel launch failed") ? 0 : 1;
}

template <typename InputT, typename WeightT, typename OutputT>
int launch_projection_typed_out(
    const InputT* input,
    float input_scale,
    const WeightT* weight,
    float weight_scale,
    OutputT* out,
    size_t rows,
    size_t n_dim,
    size_t k_dim,
    const char* error_prefix) {
    constexpr int block_size = 256;
    size_t len = rows * n_dim;
    const unsigned int grid_size = linear_grid_size(len, block_size);
    projection_typed_out_kernel<InputT, WeightT, OutputT><<<grid_size, block_size>>>(
        input,
        input_scale,
        weight,
        weight_scale,
        out,
        rows,
        n_dim,
        k_dim);
    return check_cuda_launch(error_prefix) ? 0 : 1;
}

template <typename InputT>
int dispatch_projection_weight_typed(
    const InputT* input,
    float input_scale,
    uint64_t weight_handle,
    int weight_dtype,
    float weight_scale,
    float* out,
    size_t rows,
    size_t n_dim,
    size_t k_dim,
    const char* error_prefix) {
    switch (weight_dtype) {
        case kDTypeF32:
            return launch_projection_typed(
                input,
                input_scale,
                handle_to_ptr(weight_handle),
                weight_scale,
                out,
                rows,
                n_dim,
                k_dim,
                error_prefix);
        case kDTypeF16:
            return launch_projection_typed(
                input,
                input_scale,
                reinterpret_cast<const __half*>(handle_to_ptr(weight_handle)),
                weight_scale,
                out,
                rows,
                n_dim,
                k_dim,
                error_prefix);
        case kDTypeBF16:
            return launch_projection_typed(
                input,
                input_scale,
                reinterpret_cast<const __nv_bfloat16*>(handle_to_ptr(weight_handle)),
                weight_scale,
                out,
                rows,
                n_dim,
                k_dim,
                error_prefix);
        case kDTypeI8:
            return launch_projection_typed(
                input,
                input_scale,
                reinterpret_cast<const int8_t*>(handle_to_ptr(weight_handle)),
                weight_scale,
                out,
                rows,
                n_dim,
                k_dim,
                error_prefix);
        default:
            set_error("CUDA typed projection received unsupported weight dtype");
            return 1;
    }
}

template <typename InputT>
int dispatch_fused_qkv_weight_typed(
    const InputT* input,
    float input_scale,
    uint64_t q_handle,
    uint64_t k_handle,
    uint64_t v_handle,
    int weight_dtype,
    float q_scale,
    float k_scale,
    float v_scale,
    float* q_out,
    float* k_out,
    float* v_out,
    size_t rows,
    size_t q_n,
    size_t k_n,
    size_t k_dim) {
    if (rows == 1) {
        switch (weight_dtype) {
            case kDTypeF32:
                return launch_fused_qkv_matvec_typed(
                    input,
                    input_scale,
                    handle_to_ptr(q_handle),
                    q_scale,
                    handle_to_ptr(k_handle),
                    k_scale,
                    handle_to_ptr(v_handle),
                    v_scale,
                    q_out,
                    k_out,
                    v_out,
                    q_n,
                    k_n,
                    k_dim);
            case kDTypeF16:
                return launch_fused_qkv_matvec_typed(
                    input,
                    input_scale,
                    reinterpret_cast<const __half*>(handle_to_ptr(q_handle)),
                    q_scale,
                    reinterpret_cast<const __half*>(handle_to_ptr(k_handle)),
                    k_scale,
                    reinterpret_cast<const __half*>(handle_to_ptr(v_handle)),
                    v_scale,
                    q_out,
                    k_out,
                    v_out,
                    q_n,
                    k_n,
                    k_dim);
            case kDTypeBF16:
                return launch_fused_qkv_matvec_typed(
                    input,
                    input_scale,
                    reinterpret_cast<const __nv_bfloat16*>(handle_to_ptr(q_handle)),
                    q_scale,
                    reinterpret_cast<const __nv_bfloat16*>(handle_to_ptr(k_handle)),
                    k_scale,
                    reinterpret_cast<const __nv_bfloat16*>(handle_to_ptr(v_handle)),
                    v_scale,
                    q_out,
                    k_out,
                    v_out,
                    q_n,
                    k_n,
                    k_dim);
            case kDTypeI8:
                return launch_fused_qkv_matvec_typed(
                    input,
                    input_scale,
                    reinterpret_cast<const int8_t*>(handle_to_ptr(q_handle)),
                    q_scale,
                    reinterpret_cast<const int8_t*>(handle_to_ptr(k_handle)),
                    k_scale,
                    reinterpret_cast<const int8_t*>(handle_to_ptr(v_handle)),
                    v_scale,
                    q_out,
                    k_out,
                    v_out,
                    q_n,
                    k_n,
                    k_dim);
            default:
                set_error("CUDA typed fused qkv matvec received unsupported weight dtype");
                return 1;
        }
    }
    if (dispatch_projection_weight_typed(
            input,
            input_scale,
            q_handle,
            weight_dtype,
            q_scale,
            q_out,
            rows,
            q_n,
            k_dim,
            "CUDA typed fused q projection launch failed") != 0) {
        return 1;
    }
    if (dispatch_projection_weight_typed(
            input,
            input_scale,
            k_handle,
            weight_dtype,
            k_scale,
            k_out,
            rows,
            k_n,
            k_dim,
            "CUDA typed fused k projection launch failed") != 0) {
        return 1;
    }
    if (dispatch_projection_weight_typed(
            input,
            input_scale,
            v_handle,
            weight_dtype,
            v_scale,
            v_out,
            rows,
            k_n,
            k_dim,
            "CUDA typed fused v projection launch failed") != 0) {
        return 1;
    }
    return check_cuda_launch("CUDA typed fused qkv kernels launch failed") ? 0 : 1;
}

template <typename InputT, typename OutputT>
int dispatch_projection_weight_typed_out(
    const InputT* input,
    float input_scale,
    uint64_t weight_handle,
    int weight_dtype,
    float weight_scale,
    OutputT* out,
    size_t rows,
    size_t n_dim,
    size_t k_dim,
    const char* error_prefix) {
    switch (weight_dtype) {
        case kDTypeF32:
            return launch_projection_typed_out(
                input,
                input_scale,
                handle_to_ptr(weight_handle),
                weight_scale,
                out,
                rows,
                n_dim,
                k_dim,
                error_prefix);
        case kDTypeF16:
            return launch_projection_typed_out(
                input,
                input_scale,
                reinterpret_cast<const __half*>(handle_to_ptr(weight_handle)),
                weight_scale,
                out,
                rows,
                n_dim,
                k_dim,
                error_prefix);
        case kDTypeBF16:
            return launch_projection_typed_out(
                input,
                input_scale,
                reinterpret_cast<const __nv_bfloat16*>(handle_to_ptr(weight_handle)),
                weight_scale,
                out,
                rows,
                n_dim,
                k_dim,
                error_prefix);
        case kDTypeI8:
            return launch_projection_typed_out(
                input,
                input_scale,
                reinterpret_cast<const int8_t*>(handle_to_ptr(weight_handle)),
                weight_scale,
                out,
                rows,
                n_dim,
                k_dim,
                error_prefix);
        default:
            set_error("CUDA typed projection output received unsupported weight dtype");
            return 1;
    }
}

template <typename InputT, typename OutputT>
int dispatch_fused_qkv_weight_typed_out(
    const InputT* input,
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
    if (dispatch_projection_weight_typed_out(
            input,
            input_scale,
            q_handle,
            weight_dtype,
            q_scale,
            q_out,
            rows,
            q_n,
            k_dim,
            "CUDA typed fused q projection output launch failed") != 0) {
        return 1;
    }
    if (dispatch_projection_weight_typed_out(
            input,
            input_scale,
            k_handle,
            weight_dtype,
            k_scale,
            k_out,
            rows,
            k_n,
            k_dim,
            "CUDA typed fused k projection output launch failed") != 0) {
        return 1;
    }
    if (dispatch_projection_weight_typed_out(
            input,
            input_scale,
            v_handle,
            weight_dtype,
            v_scale,
            v_out,
            rows,
            k_n,
            k_dim,
            "CUDA typed fused v projection output launch failed") != 0) {
        return 1;
    }
    return check_cuda_launch("CUDA typed fused qkv output kernels launch failed") ? 0 : 1;
}

template <typename InputT>
int dispatch_fused_gate_up_silu_weight_typed(
    const InputT* input,
    float input_scale,
    uint64_t gate_handle,
    int weight_dtype,
    float gate_scale,
    uint64_t up_handle,
    float up_scale,
    float* out,
    size_t rows,
    size_t n_dim,
    size_t k_dim) {
    switch (weight_dtype) {
        case kDTypeF32:
            return launch_fused_gate_up_silu_typed(
                input,
                input_scale,
                handle_to_ptr(gate_handle),
                gate_scale,
                handle_to_ptr(up_handle),
                up_scale,
                out,
                rows,
                n_dim,
                k_dim);
        case kDTypeF16:
            return launch_fused_gate_up_silu_typed(
                input,
                input_scale,
                reinterpret_cast<const __half*>(handle_to_ptr(gate_handle)),
                gate_scale,
                reinterpret_cast<const __half*>(handle_to_ptr(up_handle)),
                up_scale,
                out,
                rows,
                n_dim,
                k_dim);
        case kDTypeBF16:
            return launch_fused_gate_up_silu_typed(
                input,
                input_scale,
                reinterpret_cast<const __nv_bfloat16*>(handle_to_ptr(gate_handle)),
                gate_scale,
                reinterpret_cast<const __nv_bfloat16*>(handle_to_ptr(up_handle)),
                up_scale,
                out,
                rows,
                n_dim,
                k_dim);
        case kDTypeI8:
            return launch_fused_gate_up_silu_typed(
                input,
                input_scale,
                reinterpret_cast<const int8_t*>(handle_to_ptr(gate_handle)),
                gate_scale,
                reinterpret_cast<const int8_t*>(handle_to_ptr(up_handle)),
                up_scale,
                out,
                rows,
                n_dim,
                k_dim);
        default:
            set_error("CUDA typed fused gate/up received unsupported weight dtype");
            return 1;
    }
}

template <typename InputT, typename OutputT>
int dispatch_fused_gate_up_silu_weight_typed_out(
    const InputT* input,
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
    switch (weight_dtype) {
        case kDTypeF32:
            return launch_fused_gate_up_silu_typed_out(
                input,
                input_scale,
                handle_to_ptr(gate_handle),
                gate_scale,
                handle_to_ptr(up_handle),
                up_scale,
                out,
                rows,
                n_dim,
                k_dim);
        case kDTypeF16:
            return launch_fused_gate_up_silu_typed_out(
                input,
                input_scale,
                reinterpret_cast<const __half*>(handle_to_ptr(gate_handle)),
                gate_scale,
                reinterpret_cast<const __half*>(handle_to_ptr(up_handle)),
                up_scale,
                out,
                rows,
                n_dim,
                k_dim);
        case kDTypeBF16:
            return launch_fused_gate_up_silu_typed_out(
                input,
                input_scale,
                reinterpret_cast<const __nv_bfloat16*>(handle_to_ptr(gate_handle)),
                gate_scale,
                reinterpret_cast<const __nv_bfloat16*>(handle_to_ptr(up_handle)),
                up_scale,
                out,
                rows,
                n_dim,
                k_dim);
        case kDTypeI8:
            return launch_fused_gate_up_silu_typed_out(
                input,
                input_scale,
                reinterpret_cast<const int8_t*>(handle_to_ptr(gate_handle)),
                gate_scale,
                reinterpret_cast<const int8_t*>(handle_to_ptr(up_handle)),
                up_scale,
                out,
                rows,
                n_dim,
                k_dim);
        default:
            set_error("CUDA typed fused gate/up output received unsupported weight dtype");
            return 1;
    }
}

__global__ void rope_kernel(
    const float* input,
    const float* cos,
    const float* sin,
    float* out,
    size_t seq_len,
    size_t dim,
    size_t offset) {
    size_t half = dim / 2;
    size_t elements = seq_len * half;
    size_t batch_head = blockIdx.y;
    const size_t grid_stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t pair_idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         pair_idx < elements;
         pair_idx += grid_stride) {
        size_t seq_idx = pair_idx / half;
        size_t j = pair_idx % half;
        size_t base = (batch_head * seq_len + seq_idx) * dim;
        size_t cache_base = (offset + seq_idx) * dim;

        float x1 = input[base + j];
        float x2 = input[base + j + half];
        float c = cos[cache_base + j];
        float s_val = sin[cache_base + j];

        out[base + j] = x1 * c - x2 * s_val;
        out[base + j + half] = x2 * c + x1 * s_val;
    }
}

template <typename InputT, typename CacheT>
__global__ void rope_typed_kernel(
    const InputT* input,
    float input_scale,
    const CacheT* cos,
    float cos_scale,
    const CacheT* sin,
    float sin_scale,
    float* out,
    size_t seq_len,
    size_t dim,
    size_t offset) {
    size_t half = dim / 2;
    size_t elements = seq_len * half;
    size_t batch_head = blockIdx.y;
    const size_t grid_stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t pair_idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         pair_idx < elements;
         pair_idx += grid_stride) {
        size_t seq_idx = pair_idx / half;
        size_t j = pair_idx % half;
        size_t base = (batch_head * seq_len + seq_idx) * dim;
        size_t cache_base = (offset + seq_idx) * dim;

        float x1 = typed_value_to_float(input, input_scale, base + j);
        float x2 = typed_value_to_float(input, input_scale, base + j + half);
        float c = typed_value_to_float(cos, cos_scale, cache_base + j);
        float s_val = typed_value_to_float(sin, sin_scale, cache_base + j);

        out[base + j] = x1 * c - x2 * s_val;
        out[base + j + half] = x2 * c + x1 * s_val;
    }
}

template <typename InputT, typename CacheT>
__device__ inline void compute_rope_pair(
    const InputT* input,
    float input_scale,
    const CacheT* cos,
    float cos_scale,
    const CacheT* sin,
    float sin_scale,
    size_t base,
    size_t cache_base,
    size_t j,
    size_t half,
    float& out1,
    float& out2) {
    float x1 = typed_value_to_float(input, input_scale, base + j);
    float x2 = typed_value_to_float(input, input_scale, base + j + half);
    float c = typed_value_to_float(cos, cos_scale, cache_base + j);
    float s_val = typed_value_to_float(sin, sin_scale, cache_base + j);

    out1 = x1 * c - x2 * s_val;
    out2 = x2 * c + x1 * s_val;
}

template <typename InputT, typename CacheT>
__global__ void rope_typed_absmax_blocks_kernel(
    const InputT* input,
    float input_scale,
    const CacheT* cos,
    float cos_scale,
    const CacheT* sin,
    float sin_scale,
    float* partial,
    size_t seq_len,
    size_t dim,
    size_t offset) {
    extern __shared__ float shared[];
    size_t tid = threadIdx.x;
    size_t half = dim / 2;
    size_t elements = seq_len * half;
    size_t batch_head = blockIdx.y;
    float value = 0.0f;
    const size_t grid_stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t pair_idx = static_cast<size_t>(blockIdx.x) * blockDim.x + tid;
         pair_idx < elements;
         pair_idx += grid_stride) {
        size_t seq_idx = pair_idx / half;
        size_t j = pair_idx % half;
        size_t base = (batch_head * seq_len + seq_idx) * dim;
        size_t cache_base = (offset + seq_idx) * dim;
        float out1 = 0.0f;
        float out2 = 0.0f;
        compute_rope_pair(
            input,
            input_scale,
            cos,
            cos_scale,
            sin,
            sin_scale,
            base,
            cache_base,
            j,
            half,
            out1,
            out2);
        value = fmaxf(value, fmaxf(fabsf(out1), fabsf(out2)));
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
        partial[static_cast<size_t>(blockIdx.y) * gridDim.x + blockIdx.x] = shared[0];
    }
}

template <typename InputT, typename CacheT>
__global__ void rope_typed_quantize_i8_kernel(
    const InputT* input,
    float input_scale,
    const CacheT* cos,
    float cos_scale,
    const CacheT* sin,
    float sin_scale,
    int8_t* out,
    float output_scale,
    size_t seq_len,
    size_t dim,
    size_t offset) {
    size_t half = dim / 2;
    size_t elements = seq_len * half;
    size_t batch_head = blockIdx.y;
    const size_t grid_stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t pair_idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         pair_idx < elements;
         pair_idx += grid_stride) {
        size_t seq_idx = pair_idx / half;
        size_t j = pair_idx % half;
        size_t base = (batch_head * seq_len + seq_idx) * dim;
        size_t cache_base = (offset + seq_idx) * dim;
        float out1 = 0.0f;
        float out2 = 0.0f;
        compute_rope_pair(
            input,
            input_scale,
            cos,
            cos_scale,
            sin,
            sin_scale,
            base,
            cache_base,
            j,
            half,
            out1,
            out2);

        float q1 = nearbyintf(out1 / output_scale);
        float q2 = nearbyintf(out2 / output_scale);
        q1 = fminf(127.0f, fmaxf(-127.0f, q1));
        q2 = fminf(127.0f, fmaxf(-127.0f, q2));
        out[base + j] = static_cast<int8_t>(q1);
        out[base + j + half] = static_cast<int8_t>(q2);
    }
}

template <typename InputT, typename CacheT>
int launch_rope_typed(
    const InputT* input,
    float input_scale,
    const CacheT* cos,
    float cos_scale,
    const CacheT* sin,
    float sin_scale,
    float* out,
    size_t batch_size,
    size_t num_heads,
    size_t seq_len,
    size_t dim,
    size_t offset) {
    size_t batch_heads = batch_size * num_heads;
    constexpr int block_size = 256;
    size_t half = dim / 2;
    size_t elements = seq_len * half;
    dim3 grid(std::min(linear_grid_size(elements, block_size), 1024u), batch_heads);
    rope_typed_kernel<InputT, CacheT><<<grid, block_size>>>(
        input,
        input_scale,
        cos,
        cos_scale,
        sin,
        sin_scale,
        out,
        seq_len,
        dim,
        offset);
    return check_cuda_launch("CUDA typed RoPE kernel launch failed") ? 0 : 1;
}

template <typename InputT, typename CacheT>
int launch_rope_typed_i8_dynamic(
    const InputT* input,
    float input_scale,
    const CacheT* cos,
    float cos_scale,
    const CacheT* sin,
    float sin_scale,
    int8_t* out,
    float* out_scale,
    size_t batch_size,
    size_t num_heads,
    size_t seq_len,
    size_t dim,
    size_t offset) {
    size_t batch_heads = batch_size * num_heads;
    constexpr int block_size = 256;
    size_t half = dim / 2;
    size_t pairs_per_head = seq_len * half;
    const unsigned int grid_x =
        std::min(linear_grid_size(pairs_per_head, block_size), 1024u);
    size_t partial_len = grid_x * batch_heads;
    thread_local ReusableCudaWorkspace partial_workspace;
    if (!partial_workspace.ensure(
            partial_len * sizeof(float),
            "failed to prepare CUDA RoPE dynamic i8 reduction buffer")) {
        return 1;
    }
    float* partial = static_cast<float*>(partial_workspace.ptr);

    dim3 grid(static_cast<unsigned int>(grid_x), static_cast<unsigned int>(batch_heads), 1);
    rope_typed_absmax_blocks_kernel<InputT, CacheT><<<
        grid,
        block_size,
        block_size * sizeof(float)>>>(
        input,
        input_scale,
        cos,
        cos_scale,
        sin,
        sin_scale,
        partial,
        seq_len,
        dim,
        offset);
    if (!check_cuda_launch("CUDA typed RoPE dynamic i8 absmax kernel launch failed")) {
        return 1;
    }

    float max_abs = 0.0f;
    bool reduced = reduce_absmax_partials_to_host(
        partial,
        partial_len,
        &max_abs,
        "CUDA typed RoPE dynamic i8 final absmax reduction kernel launch failed");
    if (!reduced) {
        return 1;
    }

    float scale = max_abs > 0.0f && isfinite(max_abs) ? fmaxf(max_abs / 127.0f, FLT_MIN) : 1.0f;
    *out_scale = scale;

    rope_typed_quantize_i8_kernel<InputT, CacheT><<<grid, block_size>>>(
        input,
        input_scale,
        cos,
        cos_scale,
        sin,
        sin_scale,
        out,
        scale,
        seq_len,
        dim,
        offset);
    return check_cuda_launch("CUDA typed RoPE dynamic i8 quantize kernel launch failed") ? 0 : 1;
}

template <typename InputT>
int dispatch_rope_cache_typed(
    const InputT* input,
    float input_scale,
    uint64_t cos_handle,
    uint64_t sin_handle,
    int cache_dtype,
    float cos_scale,
    float sin_scale,
    float* out,
    size_t batch_size,
    size_t num_heads,
    size_t seq_len,
    size_t dim,
    size_t offset) {
    switch (cache_dtype) {
        case kDTypeF32:
            return launch_rope_typed(
                input,
                input_scale,
                handle_to_ptr(cos_handle),
                cos_scale,
                handle_to_ptr(sin_handle),
                sin_scale,
                out,
                batch_size,
                num_heads,
                seq_len,
                dim,
                offset);
        case kDTypeF16:
            return launch_rope_typed(
                input,
                input_scale,
                reinterpret_cast<const __half*>(handle_to_ptr(cos_handle)),
                cos_scale,
                reinterpret_cast<const __half*>(handle_to_ptr(sin_handle)),
                sin_scale,
                out,
                batch_size,
                num_heads,
                seq_len,
                dim,
                offset);
        case kDTypeBF16:
            return launch_rope_typed(
                input,
                input_scale,
                reinterpret_cast<const __nv_bfloat16*>(handle_to_ptr(cos_handle)),
                cos_scale,
                reinterpret_cast<const __nv_bfloat16*>(handle_to_ptr(sin_handle)),
                sin_scale,
                out,
                batch_size,
                num_heads,
                seq_len,
                dim,
                offset);
        case kDTypeI8:
            return launch_rope_typed(
                input,
                input_scale,
                reinterpret_cast<const int8_t*>(handle_to_ptr(cos_handle)),
                cos_scale,
                reinterpret_cast<const int8_t*>(handle_to_ptr(sin_handle)),
                sin_scale,
                out,
                batch_size,
                num_heads,
                seq_len,
                dim,
                offset);
        default:
            set_error("CUDA typed RoPE received unsupported cache dtype");
            return 1;
    }
}

template <typename InputT>
int dispatch_rope_cache_typed_i8_dynamic(
    const InputT* input,
    float input_scale,
    uint64_t cos_handle,
    uint64_t sin_handle,
    int cache_dtype,
    float cos_scale,
    float sin_scale,
    int8_t* out,
    float* out_scale,
    size_t batch_size,
    size_t num_heads,
    size_t seq_len,
    size_t dim,
    size_t offset) {
    switch (cache_dtype) {
        case kDTypeF32:
            return launch_rope_typed_i8_dynamic(
                input,
                input_scale,
                handle_to_ptr(cos_handle),
                cos_scale,
                handle_to_ptr(sin_handle),
                sin_scale,
                out,
                out_scale,
                batch_size,
                num_heads,
                seq_len,
                dim,
                offset);
        case kDTypeF16:
            return launch_rope_typed_i8_dynamic(
                input,
                input_scale,
                reinterpret_cast<const __half*>(handle_to_ptr(cos_handle)),
                cos_scale,
                reinterpret_cast<const __half*>(handle_to_ptr(sin_handle)),
                sin_scale,
                out,
                out_scale,
                batch_size,
                num_heads,
                seq_len,
                dim,
                offset);
        case kDTypeBF16:
            return launch_rope_typed_i8_dynamic(
                input,
                input_scale,
                reinterpret_cast<const __nv_bfloat16*>(handle_to_ptr(cos_handle)),
                cos_scale,
                reinterpret_cast<const __nv_bfloat16*>(handle_to_ptr(sin_handle)),
                sin_scale,
                out,
                out_scale,
                batch_size,
                num_heads,
                seq_len,
                dim,
                offset);
        case kDTypeI8:
            return launch_rope_typed_i8_dynamic(
                input,
                input_scale,
                reinterpret_cast<const int8_t*>(handle_to_ptr(cos_handle)),
                cos_scale,
                reinterpret_cast<const int8_t*>(handle_to_ptr(sin_handle)),
                sin_scale,
                out,
                out_scale,
                batch_size,
                num_heads,
                seq_len,
                dim,
                offset);
        default:
            set_error("CUDA typed RoPE dynamic i8 received unsupported cache dtype");
            return 1;
    }
}

__global__ void rope_backward_kernel(
    const float* grad,
    const float* cos,
    const float* sin,
    float* out,
    size_t seq_len,
    size_t dim,
    size_t offset) {
    size_t half = dim / 2;
    size_t elements = seq_len * half;
    size_t batch_head = blockIdx.y;
    const size_t grid_stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t pair_idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         pair_idx < elements;
         pair_idx += grid_stride) {
        size_t seq_idx = pair_idx / half;
        size_t j = pair_idx % half;
        size_t base = (batch_head * seq_len + seq_idx) * dim;
        size_t cache_base = (offset + seq_idx) * dim;

        float g1 = grad[base + j];
        float g2 = grad[base + j + half];
        float c = cos[cache_base + j];
        float s_val = sin[cache_base + j];

        out[base + j] = g1 * c + g2 * s_val;
        out[base + j + half] = g2 * c - g1 * s_val;
    }
}

bool validate_handle(uint64_t handle, const char* name) {
    if (handle == 0) {
        set_error(std::string(name) + " is null");
        return false;
    }
    return true;
}

}  // namespace
