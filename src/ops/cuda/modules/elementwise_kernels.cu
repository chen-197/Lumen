__device__ float gelu_approx(float x) {
    constexpr float c = 0.7978845608f;
    constexpr float k = 0.044715f;
    float x3 = x * x * x;
    return 0.5f * x * (1.0f + tanhf(c * (x + k * x3)));
}

__device__ float gelu_approx_grad(float x) {
    constexpr float c = 0.7978845608f;
    constexpr float k = 0.044715f;
    float x2 = x * x;
    float x3 = x2 * x;
    float inner = c * (x + k * x3);
    float tanh_i = tanhf(inner);
    float sech2 = 1.0f - tanh_i * tanh_i;
    return 0.5f * (1.0f + tanh_i) + 0.5f * x * sech2 * c * (1.0f + 3.0f * k * x2);
}

__global__ void argmax_rows_kernel(
    const float* input,
    size_t* out_indices,
    size_t rows,
    size_t cols) {
    constexpr int block_size = 256;
    __shared__ float best_values[block_size];
    __shared__ size_t best_indices[block_size];

    for (size_t row = blockIdx.x; row < rows; row += gridDim.x) {
        float best_value = -FLT_MAX;
        size_t best_index = 0;
        const float* row_ptr = input + row * cols;
        for (size_t col = threadIdx.x; col < cols; col += blockDim.x) {
            float value = row_ptr[col];
            if (value > best_value || (value == best_value && col < best_index)) {
                best_value = value;
                best_index = col;
            }
        }

        best_values[threadIdx.x] = best_value;
        best_indices[threadIdx.x] = best_index;
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                float other_value = best_values[threadIdx.x + stride];
                size_t other_index = best_indices[threadIdx.x + stride];
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
            out_indices[row] = best_indices[0];
        }
        __syncthreads();
    }
}

template <int Op>
__device__ inline float apply_unary_op_static(float x) {
    if constexpr (Op == kUnaryRelu) {
        return x > 0.0f ? x : 0.0f;
    } else if constexpr (Op == kUnarySigmoid) {
        return 1.0f / (1.0f + expf(-x));
    } else if constexpr (Op == kUnaryTanh) {
        return tanhf(x);
    } else if constexpr (Op == kUnarySilu) {
        float sig = 1.0f / (1.0f + expf(-x));
        return x * sig;
    } else if constexpr (Op == kUnaryGelu) {
        return gelu_approx(x);
    } else {
        return x;
    }
}

template <int Op>
__device__ inline float apply_unary_backward_op_static(float x, float y, float g) {
    if constexpr (Op == kUnaryRelu) {
        return x > 0.0f ? g : 0.0f;
    } else if constexpr (Op == kUnarySigmoid) {
        return g * y * (1.0f - y);
    } else if constexpr (Op == kUnaryTanh) {
        return g * (1.0f - y * y);
    } else if constexpr (Op == kUnarySilu) {
        float sig = 1.0f / (1.0f + expf(-x));
        return g * (sig + x * sig * (1.0f - sig));
    } else if constexpr (Op == kUnaryGelu) {
        return g * gelu_approx_grad(x);
    } else {
        return g;
    }
}

template <int Op>
__device__ inline float apply_binary_op_static(float lhs, float rhs) {
    if constexpr (Op == kBinaryAdd) {
        return lhs + rhs;
    } else if constexpr (Op == kBinarySub) {
        return lhs - rhs;
    } else if constexpr (Op == kBinaryMul) {
        return lhs * rhs;
    } else {
        return lhs;
    }
}

template <int Op>
__device__ inline void apply_binary_backward_op_static(
    float lhs,
    float rhs,
    float grad,
    float& grad_lhs,
    float& grad_rhs) {
    if constexpr (Op == kBinaryAdd) {
        grad_lhs = grad;
        grad_rhs = grad;
    } else if constexpr (Op == kBinarySub) {
        grad_lhs = grad;
        grad_rhs = -grad;
    } else if constexpr (Op == kBinaryMul) {
        grad_lhs = grad * rhs;
        grad_rhs = grad * lhs;
    } else {
        grad_lhs = grad;
        grad_rhs = 0.0f;
    }
}

template <int Op>
__global__ void unary_kernel(const float* input, float* out, size_t len) {
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t idx = start; idx < len; idx += stride) {
        out[idx] = apply_unary_op_static<Op>(input[idx]);
    }
}

template <int Op>
__global__ void unary_vec4_kernel(const float4* input, float4* out, size_t vec_len, size_t len) {
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t idx = start; idx < vec_len; idx += stride) {
        float4 x = input[idx];
        out[idx] = make_float4(
            apply_unary_op_static<Op>(x.x),
            apply_unary_op_static<Op>(x.y),
            apply_unary_op_static<Op>(x.z),
            apply_unary_op_static<Op>(x.w));
    }
    if (start == 0) {
        const float* input_tail = reinterpret_cast<const float*>(input);
        float* out_tail = reinterpret_cast<float*>(out);
        for (size_t tail = vec_len * 4; tail < len; ++tail) {
            out_tail[tail] = apply_unary_op_static<Op>(input_tail[tail]);
        }
    }
}

template <int Op>
__global__ void unary_backward_kernel(
    const float* input,
    const float* output,
    const float* grad,
    float* out,
    size_t len) {
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t idx = start; idx < len; idx += stride) {
        out[idx] = apply_unary_backward_op_static<Op>(input[idx], output[idx], grad[idx]);
    }
}

template <int Op>
__global__ void unary_backward_vec4_kernel(
    const float4* input,
    const float4* output,
    const float4* grad,
    float4* out,
    size_t vec_len,
    size_t len) {
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t idx = start; idx < vec_len; idx += stride) {
        float4 x = input[idx];
        float4 y = output[idx];
        float4 g = grad[idx];
        out[idx] = make_float4(
            apply_unary_backward_op_static<Op>(x.x, y.x, g.x),
            apply_unary_backward_op_static<Op>(x.y, y.y, g.y),
            apply_unary_backward_op_static<Op>(x.z, y.z, g.z),
            apply_unary_backward_op_static<Op>(x.w, y.w, g.w));
    }
    if (start == 0) {
        const float* input_tail = reinterpret_cast<const float*>(input);
        const float* output_tail = reinterpret_cast<const float*>(output);
        const float* grad_tail = reinterpret_cast<const float*>(grad);
        float* out_tail = reinterpret_cast<float*>(out);
        for (size_t tail = vec_len * 4; tail < len; ++tail) {
            out_tail[tail] = apply_unary_backward_op_static<Op>(
                input_tail[tail],
                output_tail[tail],
                grad_tail[tail]);
        }
    }
}

template <int Op>
__global__ void binary_kernel(const float* lhs, const float* rhs, float* out, size_t len) {
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t idx = start; idx < len; idx += stride) {
        out[idx] = apply_binary_op_static<Op>(lhs[idx], rhs[idx]);
    }
}

__device__ inline float lowp_to_float(__half value) {
    return __half2float(value);
}

__device__ inline float lowp_to_float(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

__device__ inline void store_unary_lowp_output(__half* out, size_t idx, float value) {
    out[idx] = __float2half(value);
}

__device__ inline void store_unary_lowp_output(__nv_bfloat16* out, size_t idx, float value) {
    out[idx] = __float2bfloat16(value);
}

template <typename T, int Op>
__device__ inline T apply_lowp_binary_op_static(T lhs, T rhs) {
    if constexpr (Op == kBinaryAdd) {
        return __hadd(lhs, rhs);
    } else if constexpr (Op == kBinarySub) {
        return __hsub(lhs, rhs);
    } else if constexpr (Op == kBinaryMul) {
        return __hmul(lhs, rhs);
    } else {
        return lhs;
    }
}

__device__ inline float typed_value_to_float(const float* data, float, size_t idx) {
    return data[idx];
}

__device__ inline float typed_value_to_float(const __half* data, float, size_t idx) {
    return __half2float(data[idx]);
}

__device__ inline float typed_value_to_float(const __nv_bfloat16* data, float, size_t idx) {
    return __bfloat162float(data[idx]);
}

__device__ inline float typed_value_to_float(const int8_t* data, float scale, size_t idx) {
    return static_cast<float>(data[idx]) * scale;
}

template <typename LhsT, typename RhsT, int Op>
__global__ void binary_typed_to_f32_kernel(
    const LhsT* lhs,
    float lhs_scale,
    const RhsT* rhs,
    float rhs_scale,
    float* out,
    size_t len) {
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t idx = start; idx < len; idx += stride) {
        float lhs_value = typed_value_to_float(lhs, lhs_scale, idx);
        float rhs_value = typed_value_to_float(rhs, rhs_scale, idx);
        out[idx] = apply_binary_op_static<Op>(lhs_value, rhs_value);
    }
}

template <typename LhsT, typename RhsT, int Op>
__global__ void binary_typed_lastdim_rhs_to_f32_kernel(
    const LhsT* lhs,
    float lhs_scale,
    const RhsT* rhs,
    float rhs_scale,
    float* out,
    size_t len,
    size_t last_dim) {
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t idx = start; idx < len; idx += stride) {
        float lhs_value = typed_value_to_float(lhs, lhs_scale, idx);
        float rhs_value = typed_value_to_float(rhs, rhs_scale, idx % last_dim);
        out[idx] = apply_binary_op_static<Op>(lhs_value, rhs_value);
    }
}

template <typename LhsT, typename RhsT, int Op>
__global__ void binary_typed_lastdim_lhs_to_f32_kernel(
    const LhsT* lhs,
    float lhs_scale,
    const RhsT* rhs,
    float rhs_scale,
    float* out,
    size_t len,
    size_t last_dim) {
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t idx = start; idx < len; idx += stride) {
        float lhs_value = typed_value_to_float(lhs, lhs_scale, idx % last_dim);
        float rhs_value = typed_value_to_float(rhs, rhs_scale, idx);
        out[idx] = apply_binary_op_static<Op>(lhs_value, rhs_value);
    }
}

template <typename T, int Op>
__global__ void binary_lowp_to_typed_kernel(
    const T* lhs,
    const T* rhs,
    T* out,
    size_t len) {
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t idx = start; idx < len; idx += stride) {
        out[idx] = apply_lowp_binary_op_static<T, Op>(lhs[idx], rhs[idx]);
    }
}

template <typename T, int Op>
__global__ void binary_lowp_lastdim_rhs_to_typed_kernel(
    const T* lhs,
    const T* rhs,
    T* out,
    size_t len,
    size_t last_dim) {
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t idx = start; idx < len; idx += stride) {
        out[idx] = apply_lowp_binary_op_static<T, Op>(lhs[idx], rhs[idx % last_dim]);
    }
}

template <typename T, int Op>
__global__ void binary_lowp_lastdim_lhs_to_typed_kernel(
    const T* lhs,
    const T* rhs,
    T* out,
    size_t len,
    size_t last_dim) {
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t idx = start; idx < len; idx += stride) {
        out[idx] = apply_lowp_binary_op_static<T, Op>(lhs[idx % last_dim], rhs[idx]);
    }
}

template <typename LhsT, typename RhsT, int Op>
__global__ void binary_typed_row_scalar_to_f32_kernel(
    const LhsT* lhs,
    float lhs_scale,
    const RhsT* rhs,
    float rhs_scale,
    float* out,
    size_t rows,
    size_t last_dim,
    bool scalar_on_rhs) {
    const size_t len = rows * last_dim;
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t idx = start; idx < len; idx += stride) {
        size_t row = idx / last_dim;
        size_t lhs_idx = scalar_on_rhs ? idx : row;
        size_t rhs_idx = scalar_on_rhs ? row : idx;
        float lhs_value = typed_value_to_float(lhs, lhs_scale, lhs_idx);
        float rhs_value = typed_value_to_float(rhs, rhs_scale, rhs_idx);
        out[idx] = apply_binary_op_static<Op>(lhs_value, rhs_value);
    }
}

template <typename T, int Op>
__global__ void binary_lowp_row_scalar_to_typed_kernel(
    const T* lhs,
    const T* rhs,
    T* out,
    size_t rows,
    size_t last_dim,
    bool scalar_on_rhs) {
    const size_t len = rows * last_dim;
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t idx = start; idx < len; idx += stride) {
        size_t row = idx / last_dim;
        size_t lhs_idx = scalar_on_rhs ? idx : row;
        size_t rhs_idx = scalar_on_rhs ? row : idx;
        out[idx] = apply_lowp_binary_op_static<T, Op>(lhs[lhs_idx], rhs[rhs_idx]);
    }
}

template <typename LhsT, typename RhsT, int Op>
__global__ void binary_typed_broadcast_to_f32_kernel(
    const LhsT* lhs,
    float lhs_scale,
    const RhsT* rhs,
    float rhs_scale,
    float* out,
    const size_t* out_strides,
    const size_t* lhs_shape,
    const size_t* lhs_strides,
    const size_t* rhs_shape,
    const size_t* rhs_strides,
    size_t ndim,
    size_t len) {
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t out_idx = start; out_idx < len; out_idx += stride) {
        size_t remaining = out_idx;
        size_t lhs_idx = 0;
        size_t rhs_idx = 0;
        for (size_t i = 0; i < ndim; ++i) {
            size_t coord = remaining / out_strides[i];
            remaining %= out_strides[i];
            if (lhs_shape[i] != 1) {
                lhs_idx += coord * lhs_strides[i];
            }
            if (rhs_shape[i] != 1) {
                rhs_idx += coord * rhs_strides[i];
            }
        }

        float lhs_value = typed_value_to_float(lhs, lhs_scale, lhs_idx);
        float rhs_value = typed_value_to_float(rhs, rhs_scale, rhs_idx);
        out[out_idx] = apply_binary_op_static<Op>(lhs_value, rhs_value);
    }
}

template <typename T, int Op>
__global__ void binary_lowp_broadcast_to_typed_kernel(
    const T* lhs,
    const T* rhs,
    T* out,
    const size_t* out_strides,
    const size_t* lhs_shape,
    const size_t* lhs_strides,
    const size_t* rhs_shape,
    const size_t* rhs_strides,
    size_t ndim,
    size_t len) {
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t out_idx = start; out_idx < len; out_idx += stride) {
        size_t remaining = out_idx;
        size_t lhs_idx = 0;
        size_t rhs_idx = 0;
        for (size_t i = 0; i < ndim; ++i) {
            size_t coord = remaining / out_strides[i];
            remaining %= out_strides[i];
            if (lhs_shape[i] != 1) {
                lhs_idx += coord * lhs_strides[i];
            }
            if (rhs_shape[i] != 1) {
                rhs_idx += coord * rhs_strides[i];
            }
        }
        out[out_idx] = apply_lowp_binary_op_static<T, Op>(lhs[lhs_idx], rhs[rhs_idx]);
    }
}

template <typename LhsT, typename RhsT, int Op>
__global__ void binary_typed_b1d_1h1_to_f32_kernel(
    const LhsT* lhs,
    float lhs_scale,
    const RhsT* rhs,
    float rhs_scale,
    float* out,
    size_t batch,
    size_t heads,
    size_t dim,
    bool b1d_on_lhs) {
    const size_t len = batch * heads * dim;
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t idx = start; idx < len; idx += stride) {
        size_t d = idx % dim;
        size_t h = (idx / dim) % heads;
        size_t b = idx / (heads * dim);
        size_t b1d_idx = b * dim + d;
        size_t h_idx = h;
        size_t lhs_idx = b1d_on_lhs ? b1d_idx : h_idx;
        size_t rhs_idx = b1d_on_lhs ? h_idx : b1d_idx;
        float lhs_value = typed_value_to_float(lhs, lhs_scale, lhs_idx);
        float rhs_value = typed_value_to_float(rhs, rhs_scale, rhs_idx);
        out[idx] = apply_binary_op_static<Op>(lhs_value, rhs_value);
    }
}

template <typename LhsT, typename RhsT, int Op>
__global__ void binary_typed_b1d_1hd_to_f32_kernel(
    const LhsT* lhs,
    float lhs_scale,
    const RhsT* rhs,
    float rhs_scale,
    float* out,
    size_t batch,
    size_t heads,
    size_t dim,
    bool b1d_on_lhs) {
    const size_t len = batch * heads * dim;
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t idx = start; idx < len; idx += stride) {
        size_t d = idx % dim;
        size_t h = (idx / dim) % heads;
        size_t b = idx / (heads * dim);
        size_t b1d_idx = b * dim + d;
        size_t hd_idx = h * dim + d;
        size_t lhs_idx = b1d_on_lhs ? b1d_idx : hd_idx;
        size_t rhs_idx = b1d_on_lhs ? hd_idx : b1d_idx;
        float lhs_value = typed_value_to_float(lhs, lhs_scale, lhs_idx);
        float rhs_value = typed_value_to_float(rhs, rhs_scale, rhs_idx);
        out[idx] = apply_binary_op_static<Op>(lhs_value, rhs_value);
    }
}

template <typename T, int Op>
__global__ void binary_lowp_b1d_1h1_to_typed_kernel(
    const T* lhs,
    const T* rhs,
    T* out,
    size_t batch,
    size_t heads,
    size_t dim,
    bool b1d_on_lhs) {
    const size_t len = batch * heads * dim;
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t idx = start; idx < len; idx += stride) {
        size_t d = idx % dim;
        size_t h = (idx / dim) % heads;
        size_t b = idx / (heads * dim);
        size_t b1d_idx = b * dim + d;
        size_t h_idx = h;
        size_t lhs_idx = b1d_on_lhs ? b1d_idx : h_idx;
        size_t rhs_idx = b1d_on_lhs ? h_idx : b1d_idx;
        out[idx] = apply_lowp_binary_op_static<T, Op>(lhs[lhs_idx], rhs[rhs_idx]);
    }
}

template <typename T, int Op>
__global__ void binary_lowp_b1d_1hd_to_typed_kernel(
    const T* lhs,
    const T* rhs,
    T* out,
    size_t batch,
    size_t heads,
    size_t dim,
    bool b1d_on_lhs) {
    const size_t len = batch * heads * dim;
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t idx = start; idx < len; idx += stride) {
        size_t d = idx % dim;
        size_t h = (idx / dim) % heads;
        size_t b = idx / (heads * dim);
        size_t b1d_idx = b * dim + d;
        size_t hd_idx = h * dim + d;
        size_t lhs_idx = b1d_on_lhs ? b1d_idx : hd_idx;
        size_t rhs_idx = b1d_on_lhs ? hd_idx : b1d_idx;
        out[idx] = apply_lowp_binary_op_static<T, Op>(lhs[lhs_idx], rhs[rhs_idx]);
    }
}

template <typename T, int Op>
__global__ void unary_lowp_to_f32_kernel(const T* input, float* out, size_t len) {
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t idx = start; idx < len; idx += stride) {
        out[idx] = apply_unary_op_static<Op>(lowp_to_float(input[idx]));
    }
}

template <typename T, typename OutputT, int Op>
__global__ void unary_lowp_to_typed_kernel(const T* input, OutputT* out, size_t len) {
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t idx = start; idx < len; idx += stride) {
        store_unary_lowp_output(out, idx, apply_unary_op_static<Op>(lowp_to_float(input[idx])));
    }
}

template <int Op>
__global__ void unary_i8_to_f32_kernel(const int8_t* input, float scale, float* out, size_t len) {
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t idx = start; idx < len; idx += stride) {
        out[idx] = apply_unary_op_static<Op>(static_cast<float>(input[idx]) * scale);
    }
}
__global__ void unary_i8_relu_typed_out_kernel(const int8_t* input, int8_t* out, size_t len) {
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t idx = start; idx < len; idx += stride) {
        int8_t value = input[idx];
        out[idx] = value < 0 ? 0 : value;
    }
}

template <typename T, int Op>
__global__ void binary_lowp_to_f32_kernel(const T* lhs, const T* rhs, float* out, size_t len) {
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t idx = start; idx < len; idx += stride) {
        float a = lowp_to_float(lhs[idx]);
        float b = lowp_to_float(rhs[idx]);
        out[idx] = apply_binary_op_static<Op>(a, b);
    }
}

template <int Op>
__global__ void binary_i8_to_f32_kernel(
    const int8_t* lhs,
    const int8_t* rhs,
    float lhs_scale,
    float rhs_scale,
    float* out,
    size_t len) {
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t idx = start; idx < len; idx += stride) {
        float a = static_cast<float>(lhs[idx]) * lhs_scale;
        float b = static_cast<float>(rhs[idx]) * rhs_scale;
        out[idx] = apply_binary_op_static<Op>(a, b);
    }
}

template <typename T, int Op>
__global__ void binary_lowp_lastdim_rhs_to_f32_kernel(
    const T* lhs,
    const T* rhs,
    float* out,
    size_t len,
    size_t last_dim) {
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t idx = start; idx < len; idx += stride) {
        float a = lowp_to_float(lhs[idx]);
        float b = lowp_to_float(rhs[idx % last_dim]);
        out[idx] = apply_binary_op_static<Op>(a, b);
    }
}

template <typename T, int Op>
__global__ void binary_lowp_lastdim_lhs_to_f32_kernel(
    const T* lhs,
    const T* rhs,
    float* out,
    size_t len,
    size_t last_dim) {
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t idx = start; idx < len; idx += stride) {
        float a = lowp_to_float(lhs[idx % last_dim]);
        float b = lowp_to_float(rhs[idx]);
        out[idx] = apply_binary_op_static<Op>(a, b);
    }
}

template <int Op>
__global__ void binary_i8_lastdim_rhs_to_f32_kernel(
    const int8_t* lhs,
    const int8_t* rhs,
    float lhs_scale,
    float rhs_scale,
    float* out,
    size_t len,
    size_t last_dim) {
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t idx = start; idx < len; idx += stride) {
        float a = static_cast<float>(lhs[idx]) * lhs_scale;
        float b = static_cast<float>(rhs[idx % last_dim]) * rhs_scale;
        out[idx] = apply_binary_op_static<Op>(a, b);
    }
}

template <int Op>
__global__ void binary_i8_lastdim_lhs_to_f32_kernel(
    const int8_t* lhs,
    const int8_t* rhs,
    float lhs_scale,
    float rhs_scale,
    float* out,
    size_t len,
    size_t last_dim) {
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t idx = start; idx < len; idx += stride) {
        float a = static_cast<float>(lhs[idx % last_dim]) * lhs_scale;
        float b = static_cast<float>(rhs[idx]) * rhs_scale;
        out[idx] = apply_binary_op_static<Op>(a, b);
    }
}

template <int Op>
__device__ float apply_i8_binary_value(int8_t a, float a_scale, int8_t b, float b_scale) {
    float lhs_v = static_cast<float>(a) * a_scale;
    float rhs_v = static_cast<float>(b) * b_scale;
    return apply_binary_op_static<Op>(lhs_v, rhs_v);
}

template <int Op>
__global__ void binary_i8_absmax_blocks_kernel(
    const int8_t* lhs,
    const int8_t* rhs,
    float lhs_scale,
    float rhs_scale,
    float* partial,
    size_t len) {
    extern __shared__ float shared[];
    size_t tid = threadIdx.x;
    float value = 0.0f;
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + tid;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t idx = start; idx < len; idx += stride) {
        value = fmaxf(
            value,
            fabsf(apply_i8_binary_value<Op>(lhs[idx], lhs_scale, rhs[idx], rhs_scale)));
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

template <int Op>
__global__ void binary_i8_to_i8_device_absmax_kernel(
    const int8_t* lhs,
    const int8_t* rhs,
    float lhs_scale,
    float rhs_scale,
    int8_t* out,
    size_t len,
    const float* device_max_abs) {
    const float max_abs = device_max_abs[0];
    const float out_scale =
        max_abs > 0.0f && isfinite(max_abs) ? fmaxf(max_abs / 127.0f, FLT_MIN) : 1.0f;
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t idx = start; idx < len; idx += stride) {
        float value = apply_i8_binary_value<Op>(lhs[idx], lhs_scale, rhs[idx], rhs_scale);
        float q = nearbyintf(value / out_scale);
        q = fminf(127.0f, fmaxf(-127.0f, q));
        out[idx] = static_cast<int8_t>(q);
    }
}

template <int Op>
__global__ void binary_i8_lastdim_absmax_blocks_kernel(
    const int8_t* lhs,
    const int8_t* rhs,
    float lhs_scale,
    float rhs_scale,
    float* partial,
    size_t len,
    size_t last_dim,
    bool vector_on_rhs) {
    extern __shared__ float shared[];
    size_t tid = threadIdx.x;
    float value = 0.0f;
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + tid;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t idx = start; idx < len; idx += stride) {
        int8_t a = vector_on_rhs ? lhs[idx] : lhs[idx % last_dim];
        int8_t b = vector_on_rhs ? rhs[idx % last_dim] : rhs[idx];
        value = fmaxf(value, fabsf(apply_i8_binary_value<Op>(a, lhs_scale, b, rhs_scale)));
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

template <int Op>
__global__ void binary_i8_lastdim_to_i8_kernel(
    const int8_t* lhs,
    const int8_t* rhs,
    float lhs_scale,
    float rhs_scale,
    int8_t* out,
    size_t len,
    size_t last_dim,
    bool vector_on_rhs,
    float out_scale) {
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t idx = start; idx < len; idx += stride) {
        int8_t a = vector_on_rhs ? lhs[idx] : lhs[idx % last_dim];
        int8_t b = vector_on_rhs ? rhs[idx % last_dim] : rhs[idx];
        float value = apply_i8_binary_value<Op>(a, lhs_scale, b, rhs_scale);
        float q = nearbyintf(value / out_scale);
        q = fminf(127.0f, fmaxf(-127.0f, q));
        out[idx] = static_cast<int8_t>(q);
    }
}

template <int Op>
__global__ void binary_i8_row_scalar_absmax_blocks_kernel(
    const int8_t* lhs,
    const int8_t* rhs,
    float lhs_scale,
    float rhs_scale,
    float* partial,
    size_t rows,
    size_t last_dim,
    bool scalar_on_rhs) {
    extern __shared__ float shared[];
    size_t tid = threadIdx.x;
    const size_t len = rows * last_dim;
    float value = 0.0f;
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + tid;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t idx = start; idx < len; idx += stride) {
        size_t row = idx / last_dim;
        int8_t a = scalar_on_rhs ? lhs[idx] : lhs[row];
        int8_t b = scalar_on_rhs ? rhs[row] : rhs[idx];
        value = fmaxf(value, fabsf(apply_i8_binary_value<Op>(a, lhs_scale, b, rhs_scale)));
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

template <int Op>
__global__ void binary_i8_row_scalar_to_i8_kernel(
    const int8_t* lhs,
    const int8_t* rhs,
    float lhs_scale,
    float rhs_scale,
    int8_t* out,
    size_t rows,
    size_t last_dim,
    bool scalar_on_rhs,
    float out_scale) {
    const size_t len = rows * last_dim;
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t idx = start; idx < len; idx += stride) {
        size_t row = idx / last_dim;
        int8_t a = scalar_on_rhs ? lhs[idx] : lhs[row];
        int8_t b = scalar_on_rhs ? rhs[row] : rhs[idx];
        float value = apply_i8_binary_value<Op>(a, lhs_scale, b, rhs_scale);
        float q = nearbyintf(value / out_scale);
        q = fminf(127.0f, fmaxf(-127.0f, q));
        out[idx] = static_cast<int8_t>(q);
    }
}

template <int Op>
__global__ void binary_i8_b1d_1h1_absmax_blocks_kernel(
    const int8_t* lhs,
    const int8_t* rhs,
    float lhs_scale,
    float rhs_scale,
    float* partial,
    size_t batch,
    size_t heads,
    size_t dim,
    bool b1d_on_lhs) {
    extern __shared__ float shared[];
    size_t tid = threadIdx.x;
    const size_t len = batch * heads * dim;
    float value = 0.0f;
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + tid;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t idx = start; idx < len; idx += stride) {
        size_t d = idx % dim;
        size_t h = (idx / dim) % heads;
        size_t b = idx / (heads * dim);
        size_t b1d_idx = b * dim + d;
        size_t h_idx = h;
        int8_t a = b1d_on_lhs ? lhs[b1d_idx] : lhs[h_idx];
        int8_t r = b1d_on_lhs ? rhs[h_idx] : rhs[b1d_idx];
        value = fmaxf(value, fabsf(apply_i8_binary_value<Op>(a, lhs_scale, r, rhs_scale)));
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

template <int Op>
__global__ void binary_i8_b1d_1h1_to_i8_kernel(
    const int8_t* lhs,
    const int8_t* rhs,
    float lhs_scale,
    float rhs_scale,
    int8_t* out,
    size_t batch,
    size_t heads,
    size_t dim,
    bool b1d_on_lhs,
    float out_scale) {
    const size_t len = batch * heads * dim;
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t idx = start; idx < len; idx += stride) {
        size_t d = idx % dim;
        size_t h = (idx / dim) % heads;
        size_t b = idx / (heads * dim);
        size_t b1d_idx = b * dim + d;
        size_t h_idx = h;
        int8_t a = b1d_on_lhs ? lhs[b1d_idx] : lhs[h_idx];
        int8_t r = b1d_on_lhs ? rhs[h_idx] : rhs[b1d_idx];
        float value = apply_i8_binary_value<Op>(a, lhs_scale, r, rhs_scale);
        float q = nearbyintf(value / out_scale);
        q = fminf(127.0f, fmaxf(-127.0f, q));
        out[idx] = static_cast<int8_t>(q);
    }
}

template <int Op>
__global__ void binary_i8_b1d_1hd_absmax_blocks_kernel(
    const int8_t* lhs,
    const int8_t* rhs,
    float lhs_scale,
    float rhs_scale,
    float* partial,
    size_t batch,
    size_t heads,
    size_t dim,
    bool b1d_on_lhs) {
    extern __shared__ float shared[];
    size_t tid = threadIdx.x;
    const size_t len = batch * heads * dim;
    float value = 0.0f;
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + tid;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t idx = start; idx < len; idx += stride) {
        size_t d = idx % dim;
        size_t h = (idx / dim) % heads;
        size_t b = idx / (heads * dim);
        size_t b1d_idx = b * dim + d;
        size_t hd_idx = h * dim + d;
        int8_t a = b1d_on_lhs ? lhs[b1d_idx] : lhs[hd_idx];
        int8_t r = b1d_on_lhs ? rhs[hd_idx] : rhs[b1d_idx];
        value = fmaxf(value, fabsf(apply_i8_binary_value<Op>(a, lhs_scale, r, rhs_scale)));
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

template <int Op>
__global__ void binary_i8_b1d_1hd_to_i8_kernel(
    const int8_t* lhs,
    const int8_t* rhs,
    float lhs_scale,
    float rhs_scale,
    int8_t* out,
    size_t batch,
    size_t heads,
    size_t dim,
    bool b1d_on_lhs,
    float out_scale) {
    const size_t len = batch * heads * dim;
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t idx = start; idx < len; idx += stride) {
        size_t d = idx % dim;
        size_t h = (idx / dim) % heads;
        size_t b = idx / (heads * dim);
        size_t b1d_idx = b * dim + d;
        size_t hd_idx = h * dim + d;
        int8_t a = b1d_on_lhs ? lhs[b1d_idx] : lhs[hd_idx];
        int8_t r = b1d_on_lhs ? rhs[hd_idx] : rhs[b1d_idx];
        float value = apply_i8_binary_value<Op>(a, lhs_scale, r, rhs_scale);
        float q = nearbyintf(value / out_scale);
        q = fminf(127.0f, fmaxf(-127.0f, q));
        out[idx] = static_cast<int8_t>(q);
    }
}

template <int Op>
__global__ void binary_i8_broadcast_absmax_blocks_kernel(
    const int8_t* lhs,
    const int8_t* rhs,
    float lhs_scale,
    float rhs_scale,
    float* partial,
    const size_t* out_strides,
    const size_t* lhs_shape,
    const size_t* lhs_strides,
    const size_t* rhs_shape,
    const size_t* rhs_strides,
    size_t ndim,
    size_t len) {
    extern __shared__ float shared[];
    size_t tid = threadIdx.x;
    float value = 0.0f;
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + tid;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t out_idx = start; out_idx < len; out_idx += stride) {
        size_t remaining = out_idx;
        size_t lhs_idx = 0;
        size_t rhs_idx = 0;
        for (size_t i = 0; i < ndim; ++i) {
            size_t coord = remaining / out_strides[i];
            remaining %= out_strides[i];
            if (lhs_shape[i] != 1) {
                lhs_idx += coord * lhs_strides[i];
            }
            if (rhs_shape[i] != 1) {
                rhs_idx += coord * rhs_strides[i];
            }
        }
        value = fmaxf(
            value,
            fabsf(apply_i8_binary_value<Op>(lhs[lhs_idx], lhs_scale, rhs[rhs_idx], rhs_scale)));
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

template <int Op>
__global__ void binary_i8_broadcast_to_i8_kernel(
    const int8_t* lhs,
    const int8_t* rhs,
    float lhs_scale,
    float rhs_scale,
    int8_t* out,
    const size_t* out_strides,
    const size_t* lhs_shape,
    const size_t* lhs_strides,
    const size_t* rhs_shape,
    const size_t* rhs_strides,
    size_t ndim,
    size_t len,
    float out_scale) {
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t out_idx = start; out_idx < len; out_idx += stride) {
        size_t remaining = out_idx;
        size_t lhs_idx = 0;
        size_t rhs_idx = 0;
        for (size_t i = 0; i < ndim; ++i) {
            size_t coord = remaining / out_strides[i];
            remaining %= out_strides[i];
            if (lhs_shape[i] != 1) {
                lhs_idx += coord * lhs_strides[i];
            }
            if (rhs_shape[i] != 1) {
                rhs_idx += coord * rhs_strides[i];
            }
        }
        float value = apply_i8_binary_value<Op>(lhs[lhs_idx], lhs_scale, rhs[rhs_idx], rhs_scale);
        float q = nearbyintf(value / out_scale);
        q = fminf(127.0f, fmaxf(-127.0f, q));
        out[out_idx] = static_cast<int8_t>(q);
    }
}

template <typename T>
__global__ void mul_grad_lowp_to_f32_kernel(
    const float* grad,
    const T* operand,
    float* out,
    size_t len) {
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t idx = start; idx < len; idx += stride) {
        out[idx] = grad[idx] * lowp_to_float(operand[idx]);
    }
}

__global__ void mul_grad_i8_to_f32_kernel(
    const float* grad,
    const int8_t* operand,
    float scale,
    float* out,
    size_t len) {
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t idx = start; idx < len; idx += stride) {
        out[idx] = grad[idx] * static_cast<float>(operand[idx]) * scale;
    }
}

template <typename LhsT, typename RhsT>
__global__ void mul_grad_typed_same_shape_kernel(
    const float* grad,
    const LhsT* lhs,
    float lhs_scale,
    const RhsT* rhs,
    float rhs_scale,
    float* grad_lhs,
    float* grad_rhs,
    size_t len) {
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t idx = start; idx < len; idx += stride) {
        float g = grad[idx];
        float lhs_value = typed_value_to_float(lhs, lhs_scale, idx);
        float rhs_value = typed_value_to_float(rhs, rhs_scale, idx);
        grad_lhs[idx] = g * rhs_value;
        grad_rhs[idx] = g * lhs_value;
    }
}

template <typename T, int Op>
__global__ void unary_backward_lowp_to_f32_kernel(
    const T* input,
    const float* output,
    const float* grad,
    float* out,
    size_t len) {
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t idx = start; idx < len; idx += stride) {
        float x = lowp_to_float(input[idx]);
        out[idx] = apply_unary_backward_op_static<Op>(x, output[idx], grad[idx]);
    }
}

template <int Op>
__global__ void unary_backward_i8_to_f32_kernel(
    const int8_t* input,
    float scale,
    const float* output,
    const float* grad,
    float* out,
    size_t len) {
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t idx = start; idx < len; idx += stride) {
        float x = static_cast<float>(input[idx]) * scale;
        out[idx] = apply_unary_backward_op_static<Op>(x, output[idx], grad[idx]);
    }
}

template <typename T>
__global__ void mul_grad_lowp_row_broadcast_kernel(
    const float* grad,
    const T* lhs,
    const T* rhs,
    float* grad_lhs,
    float* grad_rhs,
    size_t len,
    size_t last_dim,
    bool vector_on_rhs) {
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t idx = start; idx < len; idx += stride) {
        size_t col = idx % last_dim;
        float g = grad[idx];
        if (vector_on_rhs) {
            float lhs_value = lowp_to_float(lhs[idx]);
            float rhs_value = lowp_to_float(rhs[col]);
            grad_lhs[idx] = g * rhs_value;
            atomicAdd(&grad_rhs[col], g * lhs_value);
        } else {
            float lhs_value = lowp_to_float(lhs[col]);
            float rhs_value = lowp_to_float(rhs[idx]);
            atomicAdd(&grad_lhs[col], g * rhs_value);
            grad_rhs[idx] = g * lhs_value;
        }
    }
}

__global__ void mul_grad_i8_row_broadcast_kernel(
    const float* grad,
    const int8_t* lhs,
    const int8_t* rhs,
    float lhs_scale,
    float rhs_scale,
    float* grad_lhs,
    float* grad_rhs,
    size_t len,
    size_t last_dim,
    bool vector_on_rhs) {
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t idx = start; idx < len; idx += stride) {
        size_t col = idx % last_dim;
        float g = grad[idx];
        if (vector_on_rhs) {
            float lhs_value = static_cast<float>(lhs[idx]) * lhs_scale;
            float rhs_value = static_cast<float>(rhs[col]) * rhs_scale;
            grad_lhs[idx] = g * rhs_value;
            atomicAdd(&grad_rhs[col], g * lhs_value);
        } else {
            float lhs_value = static_cast<float>(lhs[col]) * lhs_scale;
            float rhs_value = static_cast<float>(rhs[idx]) * rhs_scale;
            atomicAdd(&grad_lhs[col], g * rhs_value);
            grad_rhs[idx] = g * lhs_value;
        }
    }
}

template <typename LhsT, typename RhsT>
__global__ void mul_grad_typed_row_broadcast_kernel(
    const float* grad,
    const LhsT* lhs,
    float lhs_scale,
    const RhsT* rhs,
    float rhs_scale,
    float* grad_lhs,
    float* grad_rhs,
    size_t len,
    size_t last_dim,
    bool vector_on_rhs) {
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t idx = start; idx < len; idx += stride) {
        size_t col = idx % last_dim;
        float g = grad[idx];
        if (vector_on_rhs) {
            float lhs_value = typed_value_to_float(lhs, lhs_scale, idx);
            float rhs_value = typed_value_to_float(rhs, rhs_scale, col);
            grad_lhs[idx] = g * rhs_value;
            atomicAdd(&grad_rhs[col], g * lhs_value);
        } else {
            float lhs_value = typed_value_to_float(lhs, lhs_scale, col);
            float rhs_value = typed_value_to_float(rhs, rhs_scale, idx);
            atomicAdd(&grad_lhs[col], g * rhs_value);
            grad_rhs[idx] = g * lhs_value;
        }
    }
}

template <typename LhsT, typename RhsT>
__global__ void mul_grad_typed_row_scalar_kernel(
    const float* grad,
    const LhsT* lhs,
    float lhs_scale,
    const RhsT* rhs,
    float rhs_scale,
    float* grad_lhs,
    float* grad_rhs,
    size_t rows,
    size_t last_dim,
    bool scalar_on_rhs) {
    __shared__ float partials[256];
    unsigned int tid = threadIdx.x;
    for (size_t row = blockIdx.x; row < rows; row += gridDim.x) {
        float acc = 0.0f;
        for (size_t col = tid; col < last_dim; col += blockDim.x) {
            size_t idx = row * last_dim + col;
            float g = grad[idx];
            if (scalar_on_rhs) {
                float lhs_value = typed_value_to_float(lhs, lhs_scale, idx);
                float rhs_value = typed_value_to_float(rhs, rhs_scale, row);
                grad_lhs[idx] = g * rhs_value;
                acc += g * lhs_value;
            } else {
                float lhs_value = typed_value_to_float(lhs, lhs_scale, row);
                float rhs_value = typed_value_to_float(rhs, rhs_scale, idx);
                acc += g * rhs_value;
                grad_rhs[idx] = g * lhs_value;
            }
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
            if (scalar_on_rhs) {
                grad_rhs[row] = partials[0];
            } else {
                grad_lhs[row] = partials[0];
            }
        }
        __syncthreads();
    }
}

template <typename LhsT, typename RhsT>
__global__ void mul_grad_typed_broadcast_kernel(
    const float* grad,
    const LhsT* lhs,
    float lhs_scale,
    const RhsT* rhs,
    float rhs_scale,
    float* grad_lhs,
    float* grad_rhs,
    const size_t* out_strides,
    const size_t* lhs_shape,
    const size_t* lhs_strides,
    const size_t* rhs_shape,
    const size_t* rhs_strides,
    size_t ndim,
    size_t out_len) {
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t idx = start; idx < out_len; idx += stride) {
        size_t remainder = idx;
        size_t lhs_idx = 0;
        size_t rhs_idx = 0;
        for (size_t dim = 0; dim < ndim; ++dim) {
            size_t coord = remainder / out_strides[dim];
            remainder %= out_strides[dim];
            if (lhs_shape[dim] != 1) {
                lhs_idx += coord * lhs_strides[dim];
            }
            if (rhs_shape[dim] != 1) {
                rhs_idx += coord * rhs_strides[dim];
            }
        }

        float g = grad[idx];
        float lhs_value = typed_value_to_float(lhs, lhs_scale, lhs_idx);
        float rhs_value = typed_value_to_float(rhs, rhs_scale, rhs_idx);
        atomicAdd(grad_lhs + lhs_idx, g * rhs_value);
        atomicAdd(grad_rhs + rhs_idx, g * lhs_value);
    }
}

template <typename LhsT, typename RhsT>
__global__ void mul_grad_typed_scalar_broadcast_kernel(
    const float* grad,
    const LhsT* lhs,
    float lhs_scale,
    const RhsT* rhs,
    float rhs_scale,
    float* grad_lhs,
    float* grad_rhs,
    size_t len,
    bool scalar_on_rhs) {
    __shared__ float partials[256];
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    unsigned int tid = threadIdx.x;
    float scalar_acc = 0.0f;
    for (size_t idx = start; idx < len; idx += stride) {
        float g = grad[idx];
        if (scalar_on_rhs) {
            float lhs_value = typed_value_to_float(lhs, lhs_scale, idx);
            float rhs_value = typed_value_to_float(rhs, rhs_scale, 0);
            grad_lhs[idx] = g * rhs_value;
            scalar_acc += g * lhs_value;
        } else {
            float lhs_value = typed_value_to_float(lhs, lhs_scale, 0);
            float rhs_value = typed_value_to_float(rhs, rhs_scale, idx);
            scalar_acc += g * rhs_value;
            grad_rhs[idx] = g * lhs_value;
        }
    }
    partials[tid] = scalar_acc;
    __syncthreads();
    for (unsigned int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            partials[tid] += partials[tid + offset];
        }
        __syncthreads();
    }
    if (tid == 0) {
        if (scalar_on_rhs) {
            atomicAdd(grad_rhs, partials[0]);
        } else {
            atomicAdd(grad_lhs, partials[0]);
        }
    }
}

template <typename LhsT, typename RhsT>
__global__ void mul_grad_typed_b1d_1h1_combined_backward_kernel(
    const float* grad,
    const LhsT* lhs,
    float lhs_scale,
    const RhsT* rhs,
    float rhs_scale,
    float* grad_lhs,
    float* grad_rhs,
    size_t batch,
    size_t heads,
    size_t dim,
    bool b1d_on_lhs) {
    __shared__ float partials[256];
    size_t b1d_blocks = batch * dim;
    unsigned int tid = threadIdx.x;
    size_t total_blocks = b1d_blocks + heads;
    for (size_t block = blockIdx.x; block < total_blocks; block += gridDim.x) {
        float acc = 0.0f;
        if (block < b1d_blocks) {
            size_t b = block / dim;
            size_t d = block % dim;
            for (size_t h = tid; h < heads; h += blockDim.x) {
                size_t out_idx = (b * heads + h) * dim + d;
                float other = b1d_on_lhs
                    ? typed_value_to_float(rhs, rhs_scale, h)
                    : typed_value_to_float(lhs, lhs_scale, h);
                acc += grad[out_idx] * other;
            }
        } else {
            size_t h = block - b1d_blocks;
            size_t reduce_len = batch * dim;
            for (size_t linear = tid; linear < reduce_len; linear += blockDim.x) {
                size_t b = linear / dim;
                size_t d = linear % dim;
                size_t out_idx = (b * heads + h) * dim + d;
                float other = b1d_on_lhs
                    ? typed_value_to_float(lhs, lhs_scale, linear)
                    : typed_value_to_float(rhs, rhs_scale, linear);
                acc += grad[out_idx] * other;
            }
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
            if (block < b1d_blocks) {
                if (b1d_on_lhs) {
                    grad_lhs[block] = partials[0];
                } else {
                    grad_rhs[block] = partials[0];
                }
            } else {
                size_t h = block - b1d_blocks;
                if (b1d_on_lhs) {
                    grad_rhs[h] = partials[0];
                } else {
                    grad_lhs[h] = partials[0];
                }
            }
        }
        __syncthreads();
    }
}

template <typename LhsT, typename RhsT>
__global__ void mul_grad_typed_b1d_1hd_combined_backward_kernel(
    const float* grad,
    const LhsT* lhs,
    float lhs_scale,
    const RhsT* rhs,
    float rhs_scale,
    float* grad_lhs,
    float* grad_rhs,
    size_t batch,
    size_t heads,
    size_t dim,
    bool b1d_on_lhs) {
    __shared__ float partials[256];
    size_t b1d_blocks = batch * dim;
    size_t hd_blocks = heads * dim;
    unsigned int tid = threadIdx.x;
    size_t total_blocks = b1d_blocks + hd_blocks;
    for (size_t block = blockIdx.x; block < total_blocks; block += gridDim.x) {
        float acc = 0.0f;
        if (block < b1d_blocks) {
            size_t b = block / dim;
            size_t d = block % dim;
            for (size_t h = tid; h < heads; h += blockDim.x) {
                size_t out_idx = (b * heads + h) * dim + d;
                size_t hd_idx = h * dim + d;
                float other = b1d_on_lhs
                    ? typed_value_to_float(rhs, rhs_scale, hd_idx)
                    : typed_value_to_float(lhs, lhs_scale, hd_idx);
                acc += grad[out_idx] * other;
            }
        } else {
            size_t hd_idx = block - b1d_blocks;
            size_t h = hd_idx / dim;
            size_t d = hd_idx % dim;
            for (size_t b = tid; b < batch; b += blockDim.x) {
                size_t out_idx = (b * heads + h) * dim + d;
                size_t b1d_idx = b * dim + d;
                float other = b1d_on_lhs
                    ? typed_value_to_float(lhs, lhs_scale, b1d_idx)
                    : typed_value_to_float(rhs, rhs_scale, b1d_idx);
                acc += grad[out_idx] * other;
            }
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
            if (block < b1d_blocks) {
                if (b1d_on_lhs) {
                    grad_lhs[block] = partials[0];
                } else {
                    grad_rhs[block] = partials[0];
                }
            } else {
                size_t hd_idx = block - b1d_blocks;
                if (b1d_on_lhs) {
                    grad_rhs[hd_idx] = partials[0];
                } else {
                    grad_lhs[hd_idx] = partials[0];
                }
            }
        }
        __syncthreads();
    }
}

template <int Op>
__global__ void binary_vec4_kernel(
    const float4* lhs,
    const float4* rhs,
    float4* out,
    size_t vec_len,
    size_t len) {
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t idx = start; idx < vec_len; idx += stride) {
        float4 a = lhs[idx];
        float4 b = rhs[idx];
        out[idx] = make_float4(
            apply_binary_op_static<Op>(a.x, b.x),
            apply_binary_op_static<Op>(a.y, b.y),
            apply_binary_op_static<Op>(a.z, b.z),
            apply_binary_op_static<Op>(a.w, b.w));
    }
    if (start == 0) {
        const float* lhs_tail = reinterpret_cast<const float*>(lhs);
        const float* rhs_tail = reinterpret_cast<const float*>(rhs);
        float* out_tail = reinterpret_cast<float*>(out);
        for (size_t tail = vec_len * 4; tail < len; ++tail) {
            out_tail[tail] = apply_binary_op_static<Op>(lhs_tail[tail], rhs_tail[tail]);
        }
    }
}

template <int Op>
__global__ void binary_scalar_rhs_vec4_kernel(
    const float4* lhs,
    const float* rhs,
    float4* out,
    size_t vec_len,
    size_t len) {
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    float scalar = rhs[0];
    for (size_t idx = start; idx < vec_len; idx += stride) {
        float4 a = lhs[idx];
        out[idx] = make_float4(
            apply_binary_op_static<Op>(a.x, scalar),
            apply_binary_op_static<Op>(a.y, scalar),
            apply_binary_op_static<Op>(a.z, scalar),
            apply_binary_op_static<Op>(a.w, scalar));
    }
    if (start == 0) {
        const float* lhs_tail = reinterpret_cast<const float*>(lhs);
        float* out_tail = reinterpret_cast<float*>(out);
        for (size_t tail = vec_len * 4; tail < len; ++tail) {
            out_tail[tail] = apply_binary_op_static<Op>(lhs_tail[tail], scalar);
        }
    }
}

template <int Op>
__global__ void binary_scalar_lhs_vec4_kernel(
    const float* lhs,
    const float4* rhs,
    float4* out,
    size_t vec_len,
    size_t len) {
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    float scalar = lhs[0];
    for (size_t idx = start; idx < vec_len; idx += stride) {
        float4 b = rhs[idx];
        out[idx] = make_float4(
            apply_binary_op_static<Op>(scalar, b.x),
            apply_binary_op_static<Op>(scalar, b.y),
            apply_binary_op_static<Op>(scalar, b.z),
            apply_binary_op_static<Op>(scalar, b.w));
    }
    if (start == 0) {
        const float* rhs_tail = reinterpret_cast<const float*>(rhs);
        float* out_tail = reinterpret_cast<float*>(out);
        for (size_t tail = vec_len * 4; tail < len; ++tail) {
            out_tail[tail] = apply_binary_op_static<Op>(scalar, rhs_tail[tail]);
        }
    }
}

template <int Op>
__global__ void binary_lastdim_rhs_kernel(
    const float* lhs,
    const float* rhs,
    float* out,
    size_t len,
    size_t last_dim) {
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t idx = start; idx < len; idx += stride) {
        out[idx] = apply_binary_op_static<Op>(lhs[idx], rhs[idx % last_dim]);
    }
}

template <int Op>
__global__ void binary_lastdim_rhs_vec4_kernel(
    const float4* lhs,
    const float4* rhs,
    float4* out,
    size_t vec_len,
    size_t last_dim) {
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t idx = start; idx < vec_len; idx += stride) {
        size_t rhs_vec_idx = (idx * 4 % last_dim) / 4;
        float4 a = lhs[idx];
        float4 b = rhs[rhs_vec_idx];
        out[idx] = make_float4(
            apply_binary_op_static<Op>(a.x, b.x),
            apply_binary_op_static<Op>(a.y, b.y),
            apply_binary_op_static<Op>(a.z, b.z),
            apply_binary_op_static<Op>(a.w, b.w));
    }
}

template <int Op>
__global__ void binary_backward_kernel(
    const float* lhs,
    const float* rhs,
    const float* grad,
    float* grad_lhs,
    float* grad_rhs,
    size_t len) {
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t idx = start; idx < len; idx += stride) {
        float gl;
        float gr;
        apply_binary_backward_op_static<Op>(lhs[idx], rhs[idx], grad[idx], gl, gr);
        grad_lhs[idx] = gl;
        grad_rhs[idx] = gr;
    }
}

template <int Op>
__global__ void binary_backward_vec4_kernel(
    const float4* lhs,
    const float4* rhs,
    const float4* grad,
    float4* grad_lhs,
    float4* grad_rhs,
    size_t vec_len,
    size_t len) {
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t idx = start; idx < vec_len; idx += stride) {
        float4 a = lhs[idx];
        float4 b = rhs[idx];
        float4 g = grad[idx];
        float4 gl;
        float4 gr;
        apply_binary_backward_op_static<Op>(a.x, b.x, g.x, gl.x, gr.x);
        apply_binary_backward_op_static<Op>(a.y, b.y, g.y, gl.y, gr.y);
        apply_binary_backward_op_static<Op>(a.z, b.z, g.z, gl.z, gr.z);
        apply_binary_backward_op_static<Op>(a.w, b.w, g.w, gl.w, gr.w);
        grad_lhs[idx] = gl;
        grad_rhs[idx] = gr;
    }
    if (start == 0) {
        const float* lhs_tail = reinterpret_cast<const float*>(lhs);
        const float* rhs_tail = reinterpret_cast<const float*>(rhs);
        const float* grad_tail = reinterpret_cast<const float*>(grad);
        float* grad_lhs_tail = reinterpret_cast<float*>(grad_lhs);
        float* grad_rhs_tail = reinterpret_cast<float*>(grad_rhs);
        for (size_t tail = vec_len * 4; tail < len; ++tail) {
            float gl;
            float gr;
            apply_binary_backward_op_static<Op>(
                lhs_tail[tail],
                rhs_tail[tail],
                grad_tail[tail],
                gl,
                gr);
            grad_lhs_tail[tail] = gl;
            grad_rhs_tail[tail] = gr;
        }
    }
}

__global__ void add_sub_same_shape_backward_kernel(
    const float* grad,
    float* grad_lhs,
    float* grad_rhs,
    size_t len,
    float rhs_sign) {
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t idx = start; idx < len; idx += stride) {
        float g = grad[idx];
        grad_lhs[idx] = g;
        grad_rhs[idx] = rhs_sign * g;
    }
}

__global__ void add_sub_same_shape_backward_vec4_kernel(
    const float4* grad,
    float4* grad_lhs,
    float4* grad_rhs,
    size_t vec_len,
    size_t len,
    float rhs_sign) {
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t idx = start; idx < vec_len; idx += stride) {
        float4 g = grad[idx];
        grad_lhs[idx] = g;
        grad_rhs[idx] = make_float4(rhs_sign * g.x, rhs_sign * g.y, rhs_sign * g.z, rhs_sign * g.w);
    }
    if (start == 0) {
        const float* grad_tail = reinterpret_cast<const float*>(grad);
        float* grad_lhs_tail = reinterpret_cast<float*>(grad_lhs);
        float* grad_rhs_tail = reinterpret_cast<float*>(grad_rhs);
        for (size_t tail = vec_len * 4; tail < len; ++tail) {
            float g = grad_tail[tail];
            grad_lhs_tail[tail] = g;
            grad_rhs_tail[tail] = rhs_sign * g;
        }
    }
}

__global__ void add_sub_row_broadcast_copy_full_kernel(
    const float* grad,
    float* full_grad,
    size_t len,
    float sign) {
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t idx = start; idx < len; idx += stride) {
        full_grad[idx] = sign * grad[idx];
    }
}

__global__ void add_sub_row_broadcast_reduce_vector_kernel(
    const float* grad,
    float* vector_grad,
    size_t rows,
    size_t last_dim,
    float sign) {
    __shared__ float partials[256];
    unsigned int tid = threadIdx.x;
    for (size_t col = blockIdx.x; col < last_dim; col += gridDim.x) {
        float acc = 0.0f;
        for (size_t row = tid; row < rows; row += blockDim.x) {
            acc += grad[row * last_dim + col];
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
            vector_grad[col] = sign * partials[0];
        }
        __syncthreads();
    }
}

__global__ void add_sub_row_broadcast_reduce_vector_atomic_kernel(
    const float* grad,
    float* vector_grad,
    size_t len,
    size_t last_dim,
    float sign) {
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t idx = start; idx < len; idx += stride) {
        size_t col = idx % last_dim;
        atomicAdd(vector_grad + col, sign * grad[idx]);
    }
}

__global__ void add_sub_row_broadcast_backward_atomic_combined_kernel(
    const float* grad,
    float* full_grad,
    float* vector_grad,
    size_t len,
    size_t last_dim,
    float full_sign,
    float vector_sign) {
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t idx = start; idx < len; idx += stride) {
        float g = grad[idx];
        full_grad[idx] = full_sign * g;
        size_t col = idx % last_dim;
        atomicAdd(vector_grad + col, vector_sign * g);
    }
}

__global__ void add_sub_scalar_broadcast_backward_kernel(
    const float* grad,
    float* full_grad,
    float* scalar_grad,
    size_t len,
    float full_sign,
    float scalar_sign) {
    __shared__ float partials[256];
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    unsigned int tid = threadIdx.x;
    float scalar_acc = 0.0f;
    for (size_t idx = start; idx < len; idx += stride) {
        float g = grad[idx];
        full_grad[idx] = full_sign * g;
        scalar_acc += scalar_sign * g;
    }
    partials[tid] = scalar_acc;
    __syncthreads();
    for (unsigned int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            partials[tid] += partials[tid + offset];
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(scalar_grad, partials[0]);
    }
}

__global__ void add_sub_row_scalar_backward_kernel(
    const float* grad,
    float* grad_lhs,
    float* grad_rhs,
    size_t rows,
    size_t last_dim,
    bool scalar_on_rhs,
    float rhs_sign) {
    __shared__ float partials[256];
    unsigned int tid = threadIdx.x;
    for (size_t row = blockIdx.x; row < rows; row += gridDim.x) {
        float acc = 0.0f;
        for (size_t col = tid; col < last_dim; col += blockDim.x) {
            size_t idx = row * last_dim + col;
            float g = grad[idx];
            if (scalar_on_rhs) {
                grad_lhs[idx] = g;
                acc += rhs_sign * g;
            } else {
                acc += g;
                grad_rhs[idx] = rhs_sign * g;
            }
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
            if (scalar_on_rhs) {
                grad_rhs[row] = partials[0];
            } else {
                grad_lhs[row] = partials[0];
            }
        }
        __syncthreads();
    }
}

__global__ void add_sub_1h1_backward_kernel(
    const float* grad,
    float* h_grad,
    size_t batch,
    size_t heads,
    size_t dim,
    float sign) {
    __shared__ float partials[256];
    size_t h = blockIdx.x;
    if (h >= heads) {
        return;
    }
    unsigned int tid = threadIdx.x;
    size_t reduce_len = batch * dim;
    float acc = 0.0f;
    for (size_t linear = tid; linear < reduce_len; linear += blockDim.x) {
        size_t b = linear / dim;
        size_t d = linear % dim;
        acc += grad[(b * heads + h) * dim + d];
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
        h_grad[h] = sign * partials[0];
    }
}

__global__ void add_sub_b1d_1h1_combined_backward_kernel(
    const float* grad,
    float* grad_lhs,
    float* grad_rhs,
    size_t batch,
    size_t heads,
    size_t dim,
    bool b1d_on_lhs,
    float rhs_sign) {
    __shared__ float partials[256];
    size_t b1d_blocks = batch * dim;
    unsigned int tid = threadIdx.x;
    size_t total_blocks = b1d_blocks + heads;
    for (size_t block = blockIdx.x; block < total_blocks; block += gridDim.x) {
        float acc = 0.0f;
        if (block < b1d_blocks) {
            size_t b = block / dim;
            size_t d = block % dim;
            for (size_t h = tid; h < heads; h += blockDim.x) {
                acc += grad[(b * heads + h) * dim + d];
            }
        } else {
            size_t h = block - b1d_blocks;
            size_t reduce_len = batch * dim;
            for (size_t linear = tid; linear < reduce_len; linear += blockDim.x) {
                size_t b = linear / dim;
                size_t d = linear % dim;
                acc += grad[(b * heads + h) * dim + d];
            }
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
            if (block < b1d_blocks) {
                if (b1d_on_lhs) {
                    grad_lhs[block] = partials[0];
                } else {
                    grad_rhs[block] = rhs_sign * partials[0];
                }
            } else {
                size_t h = block - b1d_blocks;
                if (b1d_on_lhs) {
                    grad_rhs[h] = rhs_sign * partials[0];
                } else {
                    grad_lhs[h] = partials[0];
                }
            }
        }
        __syncthreads();
    }
}

__global__ void add_sub_b1d_1hd_combined_backward_kernel(
    const float* grad,
    float* grad_lhs,
    float* grad_rhs,
    size_t batch,
    size_t heads,
    size_t dim,
    bool b1d_on_lhs,
    float rhs_sign) {
    __shared__ float partials[256];
    size_t b1d_blocks = batch * dim;
    unsigned int tid = threadIdx.x;
    size_t hd_blocks = heads * dim;
    size_t total_blocks = b1d_blocks + hd_blocks;
    for (size_t block = blockIdx.x; block < total_blocks; block += gridDim.x) {
        float acc = 0.0f;
        if (block < b1d_blocks) {
            size_t b = block / dim;
            size_t d = block % dim;
            for (size_t h = tid; h < heads; h += blockDim.x) {
                acc += grad[(b * heads + h) * dim + d];
            }
        } else {
            size_t hd_idx = block - b1d_blocks;
            size_t h = hd_idx / dim;
            size_t d = hd_idx % dim;
            for (size_t b = tid; b < batch; b += blockDim.x) {
                acc += grad[(b * heads + h) * dim + d];
            }
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
            if (block < b1d_blocks) {
                if (b1d_on_lhs) {
                    grad_lhs[block] = partials[0];
                } else {
                    grad_rhs[block] = rhs_sign * partials[0];
                }
            } else {
                size_t hd_idx = block - b1d_blocks;
                if (b1d_on_lhs) {
                    grad_rhs[hd_idx] = rhs_sign * partials[0];
                } else {
                    grad_lhs[hd_idx] = partials[0];
                }
            }
        }
        __syncthreads();
    }
}

template <int Op>
__global__ void binary_scalar_rhs_backward_kernel(
    const float* lhs,
    const float* rhs,
    const float* grad,
    float* grad_lhs,
    float* grad_rhs,
    size_t len) {
    __shared__ float partials[256];
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    unsigned int tid = threadIdx.x;
    float rhs_scalar = rhs[0];
    float scalar_grad = 0.0f;
    for (size_t idx = start; idx < len; idx += stride) {
        float gl;
        float gr;
        apply_binary_backward_op_static<Op>(lhs[idx], rhs_scalar, grad[idx], gl, gr);
        grad_lhs[idx] = gl;
        scalar_grad += gr;
    }

    partials[tid] = scalar_grad;
    __syncthreads();
    for (unsigned int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            partials[tid] += partials[tid + offset];
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(grad_rhs, partials[0]);
    }
}

template <int Op>
__global__ void binary_scalar_lhs_backward_kernel(
    const float* lhs,
    const float* rhs,
    const float* grad,
    float* grad_lhs,
    float* grad_rhs,
    size_t len) {
    __shared__ float partials[256];
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    unsigned int tid = threadIdx.x;
    float lhs_scalar = lhs[0];
    float scalar_grad = 0.0f;
    for (size_t idx = start; idx < len; idx += stride) {
        float gl;
        float gr;
        apply_binary_backward_op_static<Op>(lhs_scalar, rhs[idx], grad[idx], gl, gr);
        grad_rhs[idx] = gr;
        scalar_grad += gl;
    }

    partials[tid] = scalar_grad;
    __syncthreads();
    for (unsigned int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            partials[tid] += partials[tid + offset];
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(grad_lhs, partials[0]);
    }
}

template <int Op>
__global__ void binary_lastdim_rhs_backward_kernel(
    const float* lhs,
    const float* rhs,
    const float* grad,
    float* grad_lhs,
    float* grad_rhs,
    size_t rows,
    size_t last_dim) {
    __shared__ float partials[256];
    unsigned int tid = threadIdx.x;
    for (size_t col = blockIdx.x; col < last_dim; col += gridDim.x) {
        float rhs_value = rhs[col];
        float rhs_grad = 0.0f;
        for (size_t row = tid; row < rows; row += blockDim.x) {
            size_t idx = row * last_dim + col;
            float gl;
            float gr;
            apply_binary_backward_op_static<Op>(lhs[idx], rhs_value, grad[idx], gl, gr);
            grad_lhs[idx] = gl;
            rhs_grad += gr;
        }
        partials[tid] = rhs_grad;
        __syncthreads();
        for (unsigned int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
            if (tid < offset) {
                partials[tid] += partials[tid + offset];
            }
            __syncthreads();
        }
        if (tid == 0) {
            grad_rhs[col] = partials[0];
        }
        __syncthreads();
    }
}

template <int Op>
__global__ void binary_lastdim_rhs_backward_atomic_kernel(
    const float* lhs,
    const float* rhs,
    const float* grad,
    float* grad_lhs,
    float* grad_rhs,
    size_t len,
    size_t last_dim) {
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t idx = start; idx < len; idx += stride) {
        size_t col = idx % last_dim;
        float gl;
        float gr;
        apply_binary_backward_op_static<Op>(lhs[idx], rhs[col], grad[idx], gl, gr);
        grad_lhs[idx] = gl;
        atomicAdd(grad_rhs + col, gr);
    }
}

template <int Op>
__global__ void binary_broadcast_kernel(
    const float* lhs,
    const float* rhs,
    float* out,
    const size_t* out_shape,
    const size_t* out_strides,
    const size_t* lhs_shape,
    const size_t* lhs_strides,
    const size_t* rhs_shape,
    const size_t* rhs_strides,
    size_t ndim,
    size_t len) {
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t out_idx = start; out_idx < len; out_idx += stride) {
        size_t remaining = out_idx;
        size_t lhs_idx = 0;
        size_t rhs_idx = 0;
        for (size_t i = 0; i < ndim; ++i) {
            size_t coord = remaining / out_strides[i];
            remaining %= out_strides[i];
            if (lhs_shape[i] != 1) {
                lhs_idx += coord * lhs_strides[i];
            }
            if (rhs_shape[i] != 1) {
                rhs_idx += coord * rhs_strides[i];
            }
        }
        out[out_idx] = apply_binary_op_static<Op>(lhs[lhs_idx], rhs[rhs_idx]);
    }
}

template <int Op>
__global__ void binary_broadcast_backward_kernel(
    const float* lhs,
    const float* rhs,
    const float* grad,
    float* grad_lhs,
    float* grad_rhs,
    const size_t* out_shape,
    const size_t* out_strides,
    const size_t* lhs_shape,
    const size_t* lhs_strides,
    const size_t* rhs_shape,
    const size_t* rhs_strides,
    size_t ndim,
    size_t len) {
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t out_idx = start; out_idx < len; out_idx += stride) {
        size_t remaining = out_idx;
        size_t lhs_idx = 0;
        size_t rhs_idx = 0;
        for (size_t i = 0; i < ndim; ++i) {
            size_t coord = remaining / out_strides[i];
            remaining %= out_strides[i];
            if (lhs_shape[i] != 1) {
                lhs_idx += coord * lhs_strides[i];
            }
            if (rhs_shape[i] != 1) {
                rhs_idx += coord * rhs_strides[i];
            }
        }

        float gl;
        float gr;
        apply_binary_backward_op_static<Op>(lhs[lhs_idx], rhs[rhs_idx], grad[out_idx], gl, gr);
        atomicAdd(grad_lhs + lhs_idx, gl);
        atomicAdd(grad_rhs + rhs_idx, gr);
    }
}

__global__ void add_sub_broadcast_backward_kernel(
    const float* grad,
    float* grad_lhs,
    float* grad_rhs,
    const size_t* out_strides,
    const size_t* lhs_shape,
    const size_t* lhs_strides,
    const size_t* rhs_shape,
    const size_t* rhs_strides,
    size_t ndim,
    size_t len,
    float rhs_sign) {
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t out_idx = start; out_idx < len; out_idx += stride) {
        size_t remaining = out_idx;
        size_t lhs_idx = 0;
        size_t rhs_idx = 0;
        for (size_t i = 0; i < ndim; ++i) {
            size_t coord = remaining / out_strides[i];
            remaining %= out_strides[i];
            if (lhs_shape[i] != 1) {
                lhs_idx += coord * lhs_strides[i];
            }
            if (rhs_shape[i] != 1) {
                rhs_idx += coord * rhs_strides[i];
            }
        }
        float g = grad[out_idx];
        atomicAdd(grad_lhs + lhs_idx, g);
        atomicAdd(grad_rhs + rhs_idx, rhs_sign * g);
    }
}

void launch_unary_kernel(
    int op,
    bool vec4,
    int grid_size,
    int block_size,
    const float* input,
    float* out,
    size_t vec_len,
    size_t len) {
    if (vec4) {
        const auto* input4 = reinterpret_cast<const float4*>(input);
        auto* out4 = reinterpret_cast<float4*>(out);
        switch (op) {
            case kUnaryRelu:
                unary_vec4_kernel<kUnaryRelu><<<grid_size, block_size>>>(input4, out4, vec_len, len);
                break;
            case kUnarySigmoid:
                unary_vec4_kernel<kUnarySigmoid><<<grid_size, block_size>>>(input4, out4, vec_len, len);
                break;
            case kUnaryTanh:
                unary_vec4_kernel<kUnaryTanh><<<grid_size, block_size>>>(input4, out4, vec_len, len);
                break;
            case kUnarySilu:
                unary_vec4_kernel<kUnarySilu><<<grid_size, block_size>>>(input4, out4, vec_len, len);
                break;
            case kUnaryGelu:
                unary_vec4_kernel<kUnaryGelu><<<grid_size, block_size>>>(input4, out4, vec_len, len);
                break;
            default:
                unary_vec4_kernel<-1><<<grid_size, block_size>>>(input4, out4, vec_len, len);
                break;
        }
    } else {
        switch (op) {
            case kUnaryRelu:
                unary_kernel<kUnaryRelu><<<grid_size, block_size>>>(input, out, len);
                break;
            case kUnarySigmoid:
                unary_kernel<kUnarySigmoid><<<grid_size, block_size>>>(input, out, len);
                break;
            case kUnaryTanh:
                unary_kernel<kUnaryTanh><<<grid_size, block_size>>>(input, out, len);
                break;
            case kUnarySilu:
                unary_kernel<kUnarySilu><<<grid_size, block_size>>>(input, out, len);
                break;
            case kUnaryGelu:
                unary_kernel<kUnaryGelu><<<grid_size, block_size>>>(input, out, len);
                break;
            default:
                unary_kernel<-1><<<grid_size, block_size>>>(input, out, len);
                break;
        }
    }
}

template <typename T>
void launch_unary_lowp_to_f32_kernel(
    int op,
    int grid_size,
    int block_size,
    const T* input,
    float* out,
    size_t len) {
    switch (op) {
        case kUnaryRelu:
            unary_lowp_to_f32_kernel<T, kUnaryRelu><<<grid_size, block_size>>>(input, out, len);
            break;
        case kUnarySigmoid:
            unary_lowp_to_f32_kernel<T, kUnarySigmoid><<<grid_size, block_size>>>(input, out, len);
            break;
        case kUnaryTanh:
            unary_lowp_to_f32_kernel<T, kUnaryTanh><<<grid_size, block_size>>>(input, out, len);
            break;
        case kUnarySilu:
            unary_lowp_to_f32_kernel<T, kUnarySilu><<<grid_size, block_size>>>(input, out, len);
            break;
        case kUnaryGelu:
            unary_lowp_to_f32_kernel<T, kUnaryGelu><<<grid_size, block_size>>>(input, out, len);
            break;
        default:
            unary_lowp_to_f32_kernel<T, -1><<<grid_size, block_size>>>(input, out, len);
            break;
    }
}

template <typename T, typename OutputT>
void launch_unary_lowp_to_typed_kernel(
    int op,
    int grid_size,
    int block_size,
    const T* input,
    OutputT* out,
    size_t len) {
    switch (op) {
        case kUnaryRelu:
            unary_lowp_to_typed_kernel<T, OutputT, kUnaryRelu><<<grid_size, block_size>>>(input, out, len);
            break;
        case kUnarySigmoid:
            unary_lowp_to_typed_kernel<T, OutputT, kUnarySigmoid><<<grid_size, block_size>>>(input, out, len);
            break;
        case kUnaryTanh:
            unary_lowp_to_typed_kernel<T, OutputT, kUnaryTanh><<<grid_size, block_size>>>(input, out, len);
            break;
        case kUnarySilu:
            unary_lowp_to_typed_kernel<T, OutputT, kUnarySilu><<<grid_size, block_size>>>(input, out, len);
            break;
        case kUnaryGelu:
            unary_lowp_to_typed_kernel<T, OutputT, kUnaryGelu><<<grid_size, block_size>>>(input, out, len);
            break;
        default:
            unary_lowp_to_typed_kernel<T, OutputT, -1><<<grid_size, block_size>>>(input, out, len);
            break;
    }
}

void launch_unary_i8_to_f32_kernel(
    int op,
    int grid_size,
    int block_size,
    const int8_t* input,
    float scale,
    float* out,
    size_t len) {
    switch (op) {
        case kUnaryRelu:
            unary_i8_to_f32_kernel<kUnaryRelu><<<grid_size, block_size>>>(input, scale, out, len);
            break;
        case kUnarySigmoid:
            unary_i8_to_f32_kernel<kUnarySigmoid><<<grid_size, block_size>>>(input, scale, out, len);
            break;
        case kUnaryTanh:
            unary_i8_to_f32_kernel<kUnaryTanh><<<grid_size, block_size>>>(input, scale, out, len);
            break;
        case kUnarySilu:
            unary_i8_to_f32_kernel<kUnarySilu><<<grid_size, block_size>>>(input, scale, out, len);
            break;
        case kUnaryGelu:
            unary_i8_to_f32_kernel<kUnaryGelu><<<grid_size, block_size>>>(input, scale, out, len);
            break;
        default:
            unary_i8_to_f32_kernel<-1><<<grid_size, block_size>>>(input, scale, out, len);
            break;
    }
}

void launch_unary_backward_kernel(
    int op,
    bool vec4,
    int grid_size,
    int block_size,
    const float* input,
    const float* output,
    const float* grad,
    float* out,
    size_t vec_len,
    size_t len) {
    if (vec4) {
        const auto* input4 = reinterpret_cast<const float4*>(input);
        const auto* output4 = reinterpret_cast<const float4*>(output);
        const auto* grad4 = reinterpret_cast<const float4*>(grad);
        auto* out4 = reinterpret_cast<float4*>(out);
        switch (op) {
            case kUnaryRelu:
                unary_backward_vec4_kernel<kUnaryRelu><<<grid_size, block_size>>>(
                    input4, output4, grad4, out4, vec_len, len);
                break;
            case kUnarySigmoid:
                unary_backward_vec4_kernel<kUnarySigmoid><<<grid_size, block_size>>>(
                    input4, output4, grad4, out4, vec_len, len);
                break;
            case kUnaryTanh:
                unary_backward_vec4_kernel<kUnaryTanh><<<grid_size, block_size>>>(
                    input4, output4, grad4, out4, vec_len, len);
                break;
            case kUnarySilu:
                unary_backward_vec4_kernel<kUnarySilu><<<grid_size, block_size>>>(
                    input4, output4, grad4, out4, vec_len, len);
                break;
            case kUnaryGelu:
                unary_backward_vec4_kernel<kUnaryGelu><<<grid_size, block_size>>>(
                    input4, output4, grad4, out4, vec_len, len);
                break;
            default:
                unary_backward_vec4_kernel<-1><<<grid_size, block_size>>>(
                    input4, output4, grad4, out4, vec_len, len);
                break;
        }
    } else {
        switch (op) {
            case kUnaryRelu:
                unary_backward_kernel<kUnaryRelu><<<grid_size, block_size>>>(input, output, grad, out, len);
                break;
            case kUnarySigmoid:
                unary_backward_kernel<kUnarySigmoid><<<grid_size, block_size>>>(input, output, grad, out, len);
                break;
            case kUnaryTanh:
                unary_backward_kernel<kUnaryTanh><<<grid_size, block_size>>>(input, output, grad, out, len);
                break;
            case kUnarySilu:
                unary_backward_kernel<kUnarySilu><<<grid_size, block_size>>>(input, output, grad, out, len);
                break;
            case kUnaryGelu:
                unary_backward_kernel<kUnaryGelu><<<grid_size, block_size>>>(input, output, grad, out, len);
                break;
            default:
                unary_backward_kernel<-1><<<grid_size, block_size>>>(input, output, grad, out, len);
                break;
        }
    }
}

template <typename T>
void launch_unary_backward_lowp_to_f32_kernel(
    int op,
    int grid_size,
    int block_size,
    const T* input,
    const float* output,
    const float* grad,
    float* out,
    size_t len) {
    switch (op) {
        case kUnaryRelu:
            unary_backward_lowp_to_f32_kernel<T, kUnaryRelu><<<grid_size, block_size>>>(
                input, output, grad, out, len);
            break;
        case kUnarySigmoid:
            unary_backward_lowp_to_f32_kernel<T, kUnarySigmoid><<<grid_size, block_size>>>(
                input, output, grad, out, len);
            break;
        case kUnaryTanh:
            unary_backward_lowp_to_f32_kernel<T, kUnaryTanh><<<grid_size, block_size>>>(
                input, output, grad, out, len);
            break;
        case kUnarySilu:
            unary_backward_lowp_to_f32_kernel<T, kUnarySilu><<<grid_size, block_size>>>(
                input, output, grad, out, len);
            break;
        case kUnaryGelu:
            unary_backward_lowp_to_f32_kernel<T, kUnaryGelu><<<grid_size, block_size>>>(
                input, output, grad, out, len);
            break;
        default:
            unary_backward_lowp_to_f32_kernel<T, -1><<<grid_size, block_size>>>(
                input, output, grad, out, len);
            break;
    }
}

void launch_unary_backward_i8_to_f32_kernel(
    int op,
    int grid_size,
    int block_size,
    const int8_t* input,
    float scale,
    const float* output,
    const float* grad,
    float* out,
    size_t len) {
    switch (op) {
        case kUnaryRelu:
            unary_backward_i8_to_f32_kernel<kUnaryRelu><<<grid_size, block_size>>>(
                input, scale, output, grad, out, len);
            break;
        case kUnarySigmoid:
            unary_backward_i8_to_f32_kernel<kUnarySigmoid><<<grid_size, block_size>>>(
                input, scale, output, grad, out, len);
            break;
        case kUnaryTanh:
            unary_backward_i8_to_f32_kernel<kUnaryTanh><<<grid_size, block_size>>>(
                input, scale, output, grad, out, len);
            break;
        case kUnarySilu:
            unary_backward_i8_to_f32_kernel<kUnarySilu><<<grid_size, block_size>>>(
                input, scale, output, grad, out, len);
            break;
        case kUnaryGelu:
            unary_backward_i8_to_f32_kernel<kUnaryGelu><<<grid_size, block_size>>>(
                input, scale, output, grad, out, len);
            break;
        default:
            unary_backward_i8_to_f32_kernel<-1><<<grid_size, block_size>>>(
                input, scale, output, grad, out, len);
            break;
    }
}

void launch_binary_kernel(
    int op,
    bool vec4,
    int grid_size,
    int block_size,
    const float* lhs,
    const float* rhs,
    float* out,
    size_t vec_len,
    size_t len) {
    if (vec4) {
        const auto* lhs4 = reinterpret_cast<const float4*>(lhs);
        const auto* rhs4 = reinterpret_cast<const float4*>(rhs);
        auto* out4 = reinterpret_cast<float4*>(out);
        switch (op) {
            case kBinaryAdd:
                binary_vec4_kernel<kBinaryAdd><<<grid_size, block_size>>>(lhs4, rhs4, out4, vec_len, len);
                break;
            case kBinarySub:
                binary_vec4_kernel<kBinarySub><<<grid_size, block_size>>>(lhs4, rhs4, out4, vec_len, len);
                break;
            case kBinaryMul:
                binary_vec4_kernel<kBinaryMul><<<grid_size, block_size>>>(lhs4, rhs4, out4, vec_len, len);
                break;
            default:
                binary_vec4_kernel<-1><<<grid_size, block_size>>>(lhs4, rhs4, out4, vec_len, len);
                break;
        }
    } else {
        switch (op) {
            case kBinaryAdd:
                binary_kernel<kBinaryAdd><<<grid_size, block_size>>>(lhs, rhs, out, len);
                break;
            case kBinarySub:
                binary_kernel<kBinarySub><<<grid_size, block_size>>>(lhs, rhs, out, len);
                break;
            case kBinaryMul:
                binary_kernel<kBinaryMul><<<grid_size, block_size>>>(lhs, rhs, out, len);
                break;
            default:
                binary_kernel<-1><<<grid_size, block_size>>>(lhs, rhs, out, len);
                break;
        }
    }
}

template <typename T>
void launch_binary_lowp_to_f32_kernel(
    const T* lhs,
    const T* rhs,
    float* out,
    size_t len,
    int op) {
    constexpr unsigned int block_size = 256;
    const unsigned int grid_size = linear_grid_size(len, block_size);
    switch (op) {
        case kBinaryAdd:
            binary_lowp_to_f32_kernel<T, kBinaryAdd><<<grid_size, block_size>>>(lhs, rhs, out, len);
            break;
        case kBinarySub:
            binary_lowp_to_f32_kernel<T, kBinarySub><<<grid_size, block_size>>>(lhs, rhs, out, len);
            break;
        case kBinaryMul:
            binary_lowp_to_f32_kernel<T, kBinaryMul><<<grid_size, block_size>>>(lhs, rhs, out, len);
            break;
        default:
            binary_lowp_to_f32_kernel<T, -1><<<grid_size, block_size>>>(lhs, rhs, out, len);
            break;
    }
}

void launch_binary_i8_to_f32_kernel(
    const int8_t* lhs,
    const int8_t* rhs,
    float lhs_scale,
    float rhs_scale,
    float* out,
    size_t len,
    int op) {
    constexpr unsigned int block_size = 256;
    const unsigned int grid_size = linear_grid_size(len, block_size);
    switch (op) {
        case kBinaryAdd:
            binary_i8_to_f32_kernel<kBinaryAdd><<<grid_size, block_size>>>(
                lhs, rhs, lhs_scale, rhs_scale, out, len);
            break;
        case kBinarySub:
            binary_i8_to_f32_kernel<kBinarySub><<<grid_size, block_size>>>(
                lhs, rhs, lhs_scale, rhs_scale, out, len);
            break;
        case kBinaryMul:
            binary_i8_to_f32_kernel<kBinaryMul><<<grid_size, block_size>>>(
                lhs, rhs, lhs_scale, rhs_scale, out, len);
            break;
        default:
            binary_i8_to_f32_kernel<-1><<<grid_size, block_size>>>(
                lhs, rhs, lhs_scale, rhs_scale, out, len);
            break;
    }
}

template <typename LhsT, typename RhsT>
void launch_binary_typed_to_f32_kernel(
    const LhsT* lhs,
    float lhs_scale,
    const RhsT* rhs,
    float rhs_scale,
    float* out,
    size_t len,
    int op) {
    constexpr unsigned int block_size = 256;
    const unsigned int grid_size = linear_grid_size(len, block_size);
    switch (op) {
        case kBinaryAdd:
            binary_typed_to_f32_kernel<LhsT, RhsT, kBinaryAdd><<<grid_size, block_size>>>(
                lhs, lhs_scale, rhs, rhs_scale, out, len);
            break;
        case kBinarySub:
            binary_typed_to_f32_kernel<LhsT, RhsT, kBinarySub><<<grid_size, block_size>>>(
                lhs, lhs_scale, rhs, rhs_scale, out, len);
            break;
        case kBinaryMul:
            binary_typed_to_f32_kernel<LhsT, RhsT, kBinaryMul><<<grid_size, block_size>>>(
                lhs, lhs_scale, rhs, rhs_scale, out, len);
            break;
        default:
            binary_typed_to_f32_kernel<LhsT, RhsT, -1><<<grid_size, block_size>>>(
                lhs, lhs_scale, rhs, rhs_scale, out, len);
            break;
    }
}

template <typename LhsT, typename RhsT>
void launch_binary_typed_lastdim_to_f32_kernel(
    const LhsT* lhs,
    float lhs_scale,
    const RhsT* rhs,
    float rhs_scale,
    float* out,
    size_t len,
    size_t last_dim,
    bool vector_on_rhs,
    int op) {
    constexpr unsigned int block_size = 256;
    const unsigned int grid_size = linear_grid_size(len, block_size);
    if (vector_on_rhs) {
        switch (op) {
            case kBinaryAdd:
                binary_typed_lastdim_rhs_to_f32_kernel<LhsT, RhsT, kBinaryAdd><<<grid_size, block_size>>>(
                    lhs, lhs_scale, rhs, rhs_scale, out, len, last_dim);
                break;
            case kBinarySub:
                binary_typed_lastdim_rhs_to_f32_kernel<LhsT, RhsT, kBinarySub><<<grid_size, block_size>>>(
                    lhs, lhs_scale, rhs, rhs_scale, out, len, last_dim);
                break;
            case kBinaryMul:
                binary_typed_lastdim_rhs_to_f32_kernel<LhsT, RhsT, kBinaryMul><<<grid_size, block_size>>>(
                    lhs, lhs_scale, rhs, rhs_scale, out, len, last_dim);
                break;
            default:
                binary_typed_lastdim_rhs_to_f32_kernel<LhsT, RhsT, -1><<<grid_size, block_size>>>(
                    lhs, lhs_scale, rhs, rhs_scale, out, len, last_dim);
                break;
        }
    } else {
        switch (op) {
            case kBinaryAdd:
                binary_typed_lastdim_lhs_to_f32_kernel<LhsT, RhsT, kBinaryAdd><<<grid_size, block_size>>>(
                    lhs, lhs_scale, rhs, rhs_scale, out, len, last_dim);
                break;
            case kBinarySub:
                binary_typed_lastdim_lhs_to_f32_kernel<LhsT, RhsT, kBinarySub><<<grid_size, block_size>>>(
                    lhs, lhs_scale, rhs, rhs_scale, out, len, last_dim);
                break;
            case kBinaryMul:
                binary_typed_lastdim_lhs_to_f32_kernel<LhsT, RhsT, kBinaryMul><<<grid_size, block_size>>>(
                    lhs, lhs_scale, rhs, rhs_scale, out, len, last_dim);
                break;
            default:
                binary_typed_lastdim_lhs_to_f32_kernel<LhsT, RhsT, -1><<<grid_size, block_size>>>(
                    lhs, lhs_scale, rhs, rhs_scale, out, len, last_dim);
                break;
        }
    }
}

template <typename T>
void launch_binary_lowp_to_typed_kernel(
    const T* lhs,
    const T* rhs,
    T* out,
    size_t len,
    int op) {
    constexpr unsigned int block_size = 256;
    const unsigned int grid_size = linear_grid_size(len, block_size);
    switch (op) {
        case kBinaryAdd:
            binary_lowp_to_typed_kernel<T, kBinaryAdd><<<grid_size, block_size>>>(
                lhs, rhs, out, len);
            break;
        case kBinarySub:
            binary_lowp_to_typed_kernel<T, kBinarySub><<<grid_size, block_size>>>(
                lhs, rhs, out, len);
            break;
        case kBinaryMul:
            binary_lowp_to_typed_kernel<T, kBinaryMul><<<grid_size, block_size>>>(
                lhs, rhs, out, len);
            break;
        default:
            binary_lowp_to_typed_kernel<T, -1><<<grid_size, block_size>>>(
                lhs, rhs, out, len);
            break;
    }
}

template <typename T>
void launch_binary_lowp_lastdim_to_typed_kernel(
    const T* lhs,
    const T* rhs,
    T* out,
    size_t len,
    size_t last_dim,
    bool vector_on_rhs,
    int op) {
    constexpr unsigned int block_size = 256;
    const unsigned int grid_size = linear_grid_size(len, block_size);
    if (vector_on_rhs) {
        switch (op) {
            case kBinaryAdd:
                binary_lowp_lastdim_rhs_to_typed_kernel<T, kBinaryAdd><<<grid_size, block_size>>>(
                    lhs, rhs, out, len, last_dim);
                break;
            case kBinarySub:
                binary_lowp_lastdim_rhs_to_typed_kernel<T, kBinarySub><<<grid_size, block_size>>>(
                    lhs, rhs, out, len, last_dim);
                break;
            case kBinaryMul:
                binary_lowp_lastdim_rhs_to_typed_kernel<T, kBinaryMul><<<grid_size, block_size>>>(
                    lhs, rhs, out, len, last_dim);
                break;
            default:
                binary_lowp_lastdim_rhs_to_typed_kernel<T, -1><<<grid_size, block_size>>>(
                    lhs, rhs, out, len, last_dim);
                break;
        }
    } else {
        switch (op) {
            case kBinaryAdd:
                binary_lowp_lastdim_lhs_to_typed_kernel<T, kBinaryAdd><<<grid_size, block_size>>>(
                    lhs, rhs, out, len, last_dim);
                break;
            case kBinarySub:
                binary_lowp_lastdim_lhs_to_typed_kernel<T, kBinarySub><<<grid_size, block_size>>>(
                    lhs, rhs, out, len, last_dim);
                break;
            case kBinaryMul:
                binary_lowp_lastdim_lhs_to_typed_kernel<T, kBinaryMul><<<grid_size, block_size>>>(
                    lhs, rhs, out, len, last_dim);
                break;
            default:
                binary_lowp_lastdim_lhs_to_typed_kernel<T, -1><<<grid_size, block_size>>>(
                    lhs, rhs, out, len, last_dim);
                break;
        }
    }
}

template <typename LhsT, typename RhsT>
void launch_binary_typed_row_scalar_to_f32_kernel(
    const LhsT* lhs,
    float lhs_scale,
    const RhsT* rhs,
    float rhs_scale,
    float* out,
    size_t rows,
    size_t last_dim,
    bool scalar_on_rhs,
    int op) {
    constexpr unsigned int block_size = 256;
    const size_t len = rows * last_dim;
    const unsigned int grid_size = linear_grid_size(len, block_size);
    switch (op) {
        case kBinaryAdd:
            binary_typed_row_scalar_to_f32_kernel<LhsT, RhsT, kBinaryAdd><<<grid_size, block_size>>>(
                lhs, lhs_scale, rhs, rhs_scale, out, rows, last_dim, scalar_on_rhs);
            break;
        case kBinarySub:
            binary_typed_row_scalar_to_f32_kernel<LhsT, RhsT, kBinarySub><<<grid_size, block_size>>>(
                lhs, lhs_scale, rhs, rhs_scale, out, rows, last_dim, scalar_on_rhs);
            break;
        case kBinaryMul:
            binary_typed_row_scalar_to_f32_kernel<LhsT, RhsT, kBinaryMul><<<grid_size, block_size>>>(
                lhs, lhs_scale, rhs, rhs_scale, out, rows, last_dim, scalar_on_rhs);
            break;
        default:
            binary_typed_row_scalar_to_f32_kernel<LhsT, RhsT, -1><<<grid_size, block_size>>>(
                lhs, lhs_scale, rhs, rhs_scale, out, rows, last_dim, scalar_on_rhs);
            break;
    }
}

template <typename T>
void launch_binary_lowp_row_scalar_to_typed_kernel(
    const T* lhs,
    const T* rhs,
    T* out,
    size_t rows,
    size_t last_dim,
    bool scalar_on_rhs,
    int op) {
    constexpr unsigned int block_size = 256;
    const size_t len = rows * last_dim;
    const unsigned int grid_size = linear_grid_size(len, block_size);
    switch (op) {
        case kBinaryAdd:
            binary_lowp_row_scalar_to_typed_kernel<T, kBinaryAdd><<<grid_size, block_size>>>(
                lhs, rhs, out, rows, last_dim, scalar_on_rhs);
            break;
        case kBinarySub:
            binary_lowp_row_scalar_to_typed_kernel<T, kBinarySub><<<grid_size, block_size>>>(
                lhs, rhs, out, rows, last_dim, scalar_on_rhs);
            break;
        case kBinaryMul:
            binary_lowp_row_scalar_to_typed_kernel<T, kBinaryMul><<<grid_size, block_size>>>(
                lhs, rhs, out, rows, last_dim, scalar_on_rhs);
            break;
        default:
            binary_lowp_row_scalar_to_typed_kernel<T, -1><<<grid_size, block_size>>>(
                lhs, rhs, out, rows, last_dim, scalar_on_rhs);
            break;
    }
}

template <typename T>
void launch_binary_lowp_lastdim_to_f32_kernel(
    const T* lhs,
    const T* rhs,
    float* out,
    size_t len,
    size_t last_dim,
    bool vector_on_rhs,
    int op) {
    constexpr unsigned int block_size = 256;
    const unsigned int grid_size = linear_grid_size(len, block_size);
    if (vector_on_rhs) {
        switch (op) {
            case kBinaryAdd:
                binary_lowp_lastdim_rhs_to_f32_kernel<T, kBinaryAdd><<<grid_size, block_size>>>(
                    lhs, rhs, out, len, last_dim);
                break;
            case kBinarySub:
                binary_lowp_lastdim_rhs_to_f32_kernel<T, kBinarySub><<<grid_size, block_size>>>(
                    lhs, rhs, out, len, last_dim);
                break;
            case kBinaryMul:
                binary_lowp_lastdim_rhs_to_f32_kernel<T, kBinaryMul><<<grid_size, block_size>>>(
                    lhs, rhs, out, len, last_dim);
                break;
            default:
                binary_lowp_lastdim_rhs_to_f32_kernel<T, -1><<<grid_size, block_size>>>(
                    lhs, rhs, out, len, last_dim);
                break;
        }
    } else {
        switch (op) {
            case kBinaryAdd:
                binary_lowp_lastdim_lhs_to_f32_kernel<T, kBinaryAdd><<<grid_size, block_size>>>(
                    lhs, rhs, out, len, last_dim);
                break;
            case kBinarySub:
                binary_lowp_lastdim_lhs_to_f32_kernel<T, kBinarySub><<<grid_size, block_size>>>(
                    lhs, rhs, out, len, last_dim);
                break;
            case kBinaryMul:
                binary_lowp_lastdim_lhs_to_f32_kernel<T, kBinaryMul><<<grid_size, block_size>>>(
                    lhs, rhs, out, len, last_dim);
                break;
            default:
                binary_lowp_lastdim_lhs_to_f32_kernel<T, -1><<<grid_size, block_size>>>(
                    lhs, rhs, out, len, last_dim);
                break;
        }
    }
}

void launch_binary_i8_lastdim_to_f32_kernel(
    const int8_t* lhs,
    const int8_t* rhs,
    float lhs_scale,
    float rhs_scale,
    float* out,
    size_t len,
    size_t last_dim,
    bool vector_on_rhs,
    int op) {
    constexpr unsigned int block_size = 256;
    const unsigned int grid_size = linear_grid_size(len, block_size);
    if (vector_on_rhs) {
        switch (op) {
            case kBinaryAdd:
                binary_i8_lastdim_rhs_to_f32_kernel<kBinaryAdd><<<grid_size, block_size>>>(
                    lhs, rhs, lhs_scale, rhs_scale, out, len, last_dim);
                break;
            case kBinarySub:
                binary_i8_lastdim_rhs_to_f32_kernel<kBinarySub><<<grid_size, block_size>>>(
                    lhs, rhs, lhs_scale, rhs_scale, out, len, last_dim);
                break;
            case kBinaryMul:
                binary_i8_lastdim_rhs_to_f32_kernel<kBinaryMul><<<grid_size, block_size>>>(
                    lhs, rhs, lhs_scale, rhs_scale, out, len, last_dim);
                break;
            default:
                binary_i8_lastdim_rhs_to_f32_kernel<-1><<<grid_size, block_size>>>(
                    lhs, rhs, lhs_scale, rhs_scale, out, len, last_dim);
                break;
        }
    } else {
        switch (op) {
            case kBinaryAdd:
                binary_i8_lastdim_lhs_to_f32_kernel<kBinaryAdd><<<grid_size, block_size>>>(
                    lhs, rhs, lhs_scale, rhs_scale, out, len, last_dim);
                break;
            case kBinarySub:
                binary_i8_lastdim_lhs_to_f32_kernel<kBinarySub><<<grid_size, block_size>>>(
                    lhs, rhs, lhs_scale, rhs_scale, out, len, last_dim);
                break;
            case kBinaryMul:
                binary_i8_lastdim_lhs_to_f32_kernel<kBinaryMul><<<grid_size, block_size>>>(
                    lhs, rhs, lhs_scale, rhs_scale, out, len, last_dim);
                break;
            default:
                binary_i8_lastdim_lhs_to_f32_kernel<-1><<<grid_size, block_size>>>(
                    lhs, rhs, lhs_scale, rhs_scale, out, len, last_dim);
                break;
        }
    }
}

template <typename T>
void launch_binary_lowp_b1d_1h1_to_typed_kernel(
    const T* lhs,
    const T* rhs,
    T* out,
    size_t batch,
    size_t heads,
    size_t dim,
    bool b1d_on_lhs,
    int op) {
    constexpr unsigned int block_size = 256;
    const size_t len = batch * heads * dim;
    const unsigned int grid_size = linear_grid_size(len, block_size);
    switch (op) {
        case kBinaryAdd:
            binary_lowp_b1d_1h1_to_typed_kernel<T, kBinaryAdd><<<grid_size, block_size>>>(
                lhs, rhs, out, batch, heads, dim, b1d_on_lhs);
            break;
        case kBinarySub:
            binary_lowp_b1d_1h1_to_typed_kernel<T, kBinarySub><<<grid_size, block_size>>>(
                lhs, rhs, out, batch, heads, dim, b1d_on_lhs);
            break;
        case kBinaryMul:
            binary_lowp_b1d_1h1_to_typed_kernel<T, kBinaryMul><<<grid_size, block_size>>>(
                lhs, rhs, out, batch, heads, dim, b1d_on_lhs);
            break;
        default:
            binary_lowp_b1d_1h1_to_typed_kernel<T, -1><<<grid_size, block_size>>>(
                lhs, rhs, out, batch, heads, dim, b1d_on_lhs);
            break;
    }
}

template <typename T>
void launch_binary_lowp_b1d_1hd_to_typed_kernel(
    const T* lhs,
    const T* rhs,
    T* out,
    size_t batch,
    size_t heads,
    size_t dim,
    bool b1d_on_lhs,
    int op) {
    constexpr unsigned int block_size = 256;
    const size_t len = batch * heads * dim;
    const unsigned int grid_size = linear_grid_size(len, block_size);
    switch (op) {
        case kBinaryAdd:
            binary_lowp_b1d_1hd_to_typed_kernel<T, kBinaryAdd><<<grid_size, block_size>>>(
                lhs, rhs, out, batch, heads, dim, b1d_on_lhs);
            break;
        case kBinarySub:
            binary_lowp_b1d_1hd_to_typed_kernel<T, kBinarySub><<<grid_size, block_size>>>(
                lhs, rhs, out, batch, heads, dim, b1d_on_lhs);
            break;
        case kBinaryMul:
            binary_lowp_b1d_1hd_to_typed_kernel<T, kBinaryMul><<<grid_size, block_size>>>(
                lhs, rhs, out, batch, heads, dim, b1d_on_lhs);
            break;
        default:
            binary_lowp_b1d_1hd_to_typed_kernel<T, -1><<<grid_size, block_size>>>(
                lhs, rhs, out, batch, heads, dim, b1d_on_lhs);
            break;
    }
}

template <typename LhsT, typename RhsT>
void launch_binary_typed_broadcast_to_f32_kernel(
    const LhsT* lhs,
    float lhs_scale,
    const RhsT* rhs,
    float rhs_scale,
    float* out,
    const size_t* out_strides,
    const size_t* lhs_shape,
    const size_t* lhs_strides,
    const size_t* rhs_shape,
    const size_t* rhs_strides,
    size_t ndim,
    size_t len,
    int op) {
    constexpr unsigned int block_size = 256;
    const unsigned int grid_size = linear_grid_size(len, block_size);
    switch (op) {
        case kBinaryAdd:
            binary_typed_broadcast_to_f32_kernel<LhsT, RhsT, kBinaryAdd><<<grid_size, block_size>>>(
                lhs, lhs_scale, rhs, rhs_scale, out, out_strides, lhs_shape, lhs_strides,
                rhs_shape, rhs_strides, ndim, len);
            break;
        case kBinarySub:
            binary_typed_broadcast_to_f32_kernel<LhsT, RhsT, kBinarySub><<<grid_size, block_size>>>(
                lhs, lhs_scale, rhs, rhs_scale, out, out_strides, lhs_shape, lhs_strides,
                rhs_shape, rhs_strides, ndim, len);
            break;
        case kBinaryMul:
            binary_typed_broadcast_to_f32_kernel<LhsT, RhsT, kBinaryMul><<<grid_size, block_size>>>(
                lhs, lhs_scale, rhs, rhs_scale, out, out_strides, lhs_shape, lhs_strides,
                rhs_shape, rhs_strides, ndim, len);
            break;
        default:
            binary_typed_broadcast_to_f32_kernel<LhsT, RhsT, -1><<<grid_size, block_size>>>(
                lhs, lhs_scale, rhs, rhs_scale, out, out_strides, lhs_shape, lhs_strides,
                rhs_shape, rhs_strides, ndim, len);
            break;
    }
}

template <typename T>
void launch_binary_lowp_broadcast_to_typed_kernel(
    const T* lhs,
    const T* rhs,
    T* out,
    const size_t* out_strides,
    const size_t* lhs_shape,
    const size_t* lhs_strides,
    const size_t* rhs_shape,
    const size_t* rhs_strides,
    size_t ndim,
    size_t len,
    int op) {
    constexpr unsigned int block_size = 256;
    const unsigned int grid_size = linear_grid_size(len, block_size);
    switch (op) {
        case kBinaryAdd:
            binary_lowp_broadcast_to_typed_kernel<T, kBinaryAdd><<<grid_size, block_size>>>(
                lhs, rhs, out, out_strides, lhs_shape, lhs_strides, rhs_shape, rhs_strides, ndim, len);
            break;
        case kBinarySub:
            binary_lowp_broadcast_to_typed_kernel<T, kBinarySub><<<grid_size, block_size>>>(
                lhs, rhs, out, out_strides, lhs_shape, lhs_strides, rhs_shape, rhs_strides, ndim, len);
            break;
        case kBinaryMul:
            binary_lowp_broadcast_to_typed_kernel<T, kBinaryMul><<<grid_size, block_size>>>(
                lhs, rhs, out, out_strides, lhs_shape, lhs_strides, rhs_shape, rhs_strides, ndim, len);
            break;
        default:
            binary_lowp_broadcast_to_typed_kernel<T, -1><<<grid_size, block_size>>>(
                lhs, rhs, out, out_strides, lhs_shape, lhs_strides, rhs_shape, rhs_strides, ndim, len);
            break;
    }
}

template <typename LhsT, typename RhsT>
void launch_binary_typed_b1d_1h1_to_f32_kernel(
    const LhsT* lhs,
    float lhs_scale,
    const RhsT* rhs,
    float rhs_scale,
    float* out,
    size_t batch,
    size_t heads,
    size_t dim,
    bool b1d_on_lhs,
    int op) {
    constexpr unsigned int block_size = 256;
    const size_t len = batch * heads * dim;
    const unsigned int grid_size = linear_grid_size(len, block_size);
    switch (op) {
        case kBinaryAdd:
            binary_typed_b1d_1h1_to_f32_kernel<LhsT, RhsT, kBinaryAdd><<<grid_size, block_size>>>(
                lhs, lhs_scale, rhs, rhs_scale, out, batch, heads, dim, b1d_on_lhs);
            break;
        case kBinarySub:
            binary_typed_b1d_1h1_to_f32_kernel<LhsT, RhsT, kBinarySub><<<grid_size, block_size>>>(
                lhs, lhs_scale, rhs, rhs_scale, out, batch, heads, dim, b1d_on_lhs);
            break;
        case kBinaryMul:
            binary_typed_b1d_1h1_to_f32_kernel<LhsT, RhsT, kBinaryMul><<<grid_size, block_size>>>(
                lhs, lhs_scale, rhs, rhs_scale, out, batch, heads, dim, b1d_on_lhs);
            break;
        default:
            binary_typed_b1d_1h1_to_f32_kernel<LhsT, RhsT, -1><<<grid_size, block_size>>>(
                lhs, lhs_scale, rhs, rhs_scale, out, batch, heads, dim, b1d_on_lhs);
            break;
    }
}

template <typename LhsT, typename RhsT>
void launch_binary_typed_b1d_1hd_to_f32_kernel(
    const LhsT* lhs,
    float lhs_scale,
    const RhsT* rhs,
    float rhs_scale,
    float* out,
    size_t batch,
    size_t heads,
    size_t dim,
    bool b1d_on_lhs,
    int op) {
    constexpr unsigned int block_size = 256;
    const size_t len = batch * heads * dim;
    const unsigned int grid_size = linear_grid_size(len, block_size);
    switch (op) {
        case kBinaryAdd:
            binary_typed_b1d_1hd_to_f32_kernel<LhsT, RhsT, kBinaryAdd><<<grid_size, block_size>>>(
                lhs, lhs_scale, rhs, rhs_scale, out, batch, heads, dim, b1d_on_lhs);
            break;
        case kBinarySub:
            binary_typed_b1d_1hd_to_f32_kernel<LhsT, RhsT, kBinarySub><<<grid_size, block_size>>>(
                lhs, lhs_scale, rhs, rhs_scale, out, batch, heads, dim, b1d_on_lhs);
            break;
        case kBinaryMul:
            binary_typed_b1d_1hd_to_f32_kernel<LhsT, RhsT, kBinaryMul><<<grid_size, block_size>>>(
                lhs, lhs_scale, rhs, rhs_scale, out, batch, heads, dim, b1d_on_lhs);
            break;
        default:
            binary_typed_b1d_1hd_to_f32_kernel<LhsT, RhsT, -1><<<grid_size, block_size>>>(
                lhs, lhs_scale, rhs, rhs_scale, out, batch, heads, dim, b1d_on_lhs);
            break;
    }
}

template <typename T>
void launch_mul_grad_lowp_to_f32_kernel(
    const float* grad,
    const T* operand,
    float* out,
    size_t len) {
    constexpr unsigned int block_size = 256;
    const unsigned int grid_size = linear_grid_size(len, block_size);
    mul_grad_lowp_to_f32_kernel<T><<<grid_size, block_size>>>(grad, operand, out, len);
}

void launch_mul_grad_i8_to_f32_kernel(
    const float* grad,
    const int8_t* operand,
    float scale,
    float* out,
    size_t len) {
    constexpr unsigned int block_size = 256;
    const unsigned int grid_size = linear_grid_size(len, block_size);
    mul_grad_i8_to_f32_kernel<<<grid_size, block_size>>>(grad, operand, scale, out, len);
}

template <typename LhsT, typename RhsT>
void launch_mul_grad_typed_same_shape_kernel(
    const float* grad,
    const LhsT* lhs,
    float lhs_scale,
    const RhsT* rhs,
    float rhs_scale,
    float* grad_lhs,
    float* grad_rhs,
    size_t len) {
    constexpr unsigned int block_size = 256;
    const unsigned int grid_size = linear_grid_size(len, block_size);
    mul_grad_typed_same_shape_kernel<LhsT, RhsT><<<grid_size, block_size>>>(
        grad, lhs, lhs_scale, rhs, rhs_scale, grad_lhs, grad_rhs, len);
}

template <typename T>
void launch_mul_grad_lowp_row_broadcast_kernel(
    const float* grad,
    const T* lhs,
    const T* rhs,
    float* grad_lhs,
    float* grad_rhs,
    size_t len,
    size_t last_dim,
    bool vector_on_rhs) {
    constexpr unsigned int block_size = 256;
    const unsigned int grid_size = linear_grid_size(len, block_size);
    mul_grad_lowp_row_broadcast_kernel<T><<<grid_size, block_size>>>(
        grad, lhs, rhs, grad_lhs, grad_rhs, len, last_dim, vector_on_rhs);
}

void launch_mul_grad_i8_row_broadcast_kernel(
    const float* grad,
    const int8_t* lhs,
    const int8_t* rhs,
    float lhs_scale,
    float rhs_scale,
    float* grad_lhs,
    float* grad_rhs,
    size_t len,
    size_t last_dim,
    bool vector_on_rhs) {
    constexpr unsigned int block_size = 256;
    const unsigned int grid_size = linear_grid_size(len, block_size);
    mul_grad_i8_row_broadcast_kernel<<<grid_size, block_size>>>(
        grad, lhs, rhs, lhs_scale, rhs_scale, grad_lhs, grad_rhs, len, last_dim, vector_on_rhs);
}

template <typename LhsT, typename RhsT>
void launch_mul_grad_typed_row_broadcast_kernel(
    const float* grad,
    const LhsT* lhs,
    float lhs_scale,
    const RhsT* rhs,
    float rhs_scale,
    float* grad_lhs,
    float* grad_rhs,
    size_t len,
    size_t last_dim,
    bool vector_on_rhs) {
    constexpr unsigned int block_size = 256;
    const unsigned int grid_size = linear_grid_size(len, block_size);
    mul_grad_typed_row_broadcast_kernel<LhsT, RhsT><<<grid_size, block_size>>>(
        grad, lhs, lhs_scale, rhs, rhs_scale, grad_lhs, grad_rhs, len, last_dim, vector_on_rhs);
}

template <typename LhsT, typename RhsT>
void launch_mul_grad_typed_row_scalar_kernel(
    const float* grad,
    const LhsT* lhs,
    float lhs_scale,
    const RhsT* rhs,
    float rhs_scale,
    float* grad_lhs,
    float* grad_rhs,
    size_t rows,
    size_t last_dim,
    bool scalar_on_rhs) {
    constexpr unsigned int block_size = 256;
    mul_grad_typed_row_scalar_kernel<LhsT, RhsT><<<linear_grid_size(rows, 1), block_size>>>(
        grad, lhs, lhs_scale, rhs, rhs_scale, grad_lhs, grad_rhs, rows, last_dim, scalar_on_rhs);
}

template <typename LhsT, typename RhsT>
void launch_mul_grad_typed_broadcast_kernel(
    const float* grad,
    const LhsT* lhs,
    float lhs_scale,
    const RhsT* rhs,
    float rhs_scale,
    float* grad_lhs,
    float* grad_rhs,
    const size_t* out_strides,
    const size_t* lhs_shape,
    const size_t* lhs_strides,
    const size_t* rhs_shape,
    const size_t* rhs_strides,
    size_t ndim,
    size_t out_len) {
    constexpr unsigned int block_size = 256;
    const unsigned int grid_size = linear_grid_size(out_len, block_size);
    mul_grad_typed_broadcast_kernel<LhsT, RhsT><<<grid_size, block_size>>>(
        grad,
        lhs,
        lhs_scale,
        rhs,
        rhs_scale,
        grad_lhs,
        grad_rhs,
        out_strides,
        lhs_shape,
        lhs_strides,
        rhs_shape,
        rhs_strides,
        ndim,
        out_len);
}

template <typename LhsT, typename RhsT>
void launch_mul_grad_typed_scalar_broadcast_kernel(
    const float* grad,
    const LhsT* lhs,
    float lhs_scale,
    const RhsT* rhs,
    float rhs_scale,
    float* grad_lhs,
    float* grad_rhs,
    size_t len,
    bool scalar_on_rhs) {
    constexpr unsigned int block_size = 256;
    const unsigned int grid_size = linear_grid_size(len, block_size);
    mul_grad_typed_scalar_broadcast_kernel<LhsT, RhsT><<<grid_size, block_size>>>(
        grad,
        lhs,
        lhs_scale,
        rhs,
        rhs_scale,
        grad_lhs,
        grad_rhs,
        len,
        scalar_on_rhs);
}

template <typename LhsT, typename RhsT>
void launch_mul_grad_typed_b1d_1h1_kernel(
    const float* grad,
    const LhsT* lhs,
    float lhs_scale,
    const RhsT* rhs,
    float rhs_scale,
    float* grad_lhs,
    float* grad_rhs,
    size_t batch,
    size_t heads,
    size_t dim,
    bool b1d_on_lhs) {
    constexpr int block_size = 256;
    const unsigned int grid_size = linear_grid_size(batch * dim + heads, 1);
    mul_grad_typed_b1d_1h1_combined_backward_kernel<LhsT, RhsT><<<grid_size, block_size>>>(
        grad,
        lhs,
        lhs_scale,
        rhs,
        rhs_scale,
        grad_lhs,
        grad_rhs,
        batch,
        heads,
        dim,
        b1d_on_lhs);
}

template <typename LhsT, typename RhsT>
void launch_mul_grad_typed_b1d_1hd_kernel(
    const float* grad,
    const LhsT* lhs,
    float lhs_scale,
    const RhsT* rhs,
    float rhs_scale,
    float* grad_lhs,
    float* grad_rhs,
    size_t batch,
    size_t heads,
    size_t dim,
    bool b1d_on_lhs) {
    constexpr int block_size = 256;
    const unsigned int grid_size = linear_grid_size(batch * dim + heads * dim, 1);
    mul_grad_typed_b1d_1hd_combined_backward_kernel<LhsT, RhsT><<<grid_size, block_size>>>(
        grad,
        lhs,
        lhs_scale,
        rhs,
        rhs_scale,
        grad_lhs,
        grad_rhs,
        batch,
        heads,
        dim,
        b1d_on_lhs);
}

void launch_binary_backward_kernel(
    int op,
    bool vec4,
    unsigned int grid_size,
    unsigned int block_size,
    const float* lhs,
    const float* rhs,
    const float* grad,
    float* grad_lhs,
    float* grad_rhs,
    size_t vec_len,
    size_t len) {
    if (vec4) {
        const auto* lhs4 = reinterpret_cast<const float4*>(lhs);
        const auto* rhs4 = reinterpret_cast<const float4*>(rhs);
        const auto* grad4 = reinterpret_cast<const float4*>(grad);
        auto* grad_lhs4 = reinterpret_cast<float4*>(grad_lhs);
        auto* grad_rhs4 = reinterpret_cast<float4*>(grad_rhs);
        switch (op) {
            case kBinaryAdd:
                binary_backward_vec4_kernel<kBinaryAdd><<<grid_size, block_size>>>(
                    lhs4, rhs4, grad4, grad_lhs4, grad_rhs4, vec_len, len);
                break;
            case kBinarySub:
                binary_backward_vec4_kernel<kBinarySub><<<grid_size, block_size>>>(
                    lhs4, rhs4, grad4, grad_lhs4, grad_rhs4, vec_len, len);
                break;
            case kBinaryMul:
                binary_backward_vec4_kernel<kBinaryMul><<<grid_size, block_size>>>(
                    lhs4, rhs4, grad4, grad_lhs4, grad_rhs4, vec_len, len);
                break;
            default:
                binary_backward_vec4_kernel<-1><<<grid_size, block_size>>>(
                    lhs4, rhs4, grad4, grad_lhs4, grad_rhs4, vec_len, len);
                break;
        }
    } else {
        switch (op) {
            case kBinaryAdd:
                binary_backward_kernel<kBinaryAdd><<<grid_size, block_size>>>(
                    lhs, rhs, grad, grad_lhs, grad_rhs, len);
                break;
            case kBinarySub:
                binary_backward_kernel<kBinarySub><<<grid_size, block_size>>>(
                    lhs, rhs, grad, grad_lhs, grad_rhs, len);
                break;
            case kBinaryMul:
                binary_backward_kernel<kBinaryMul><<<grid_size, block_size>>>(
                    lhs, rhs, grad, grad_lhs, grad_rhs, len);
                break;
            default:
                binary_backward_kernel<-1><<<grid_size, block_size>>>(
                    lhs, rhs, grad, grad_lhs, grad_rhs, len);
                break;
        }
    }
}

void launch_add_sub_same_shape_backward_kernel(
    const float* grad,
    float* grad_lhs,
    float* grad_rhs,
    size_t len,
    bool is_sub) {
    constexpr unsigned int block_size = 256;
    float rhs_sign = is_sub ? -1.0f : 1.0f;
    size_t vec_len = len / 4;
    if (vec_len > 0) {
        const unsigned int grid_size = linear_grid_size(vec_len, block_size);
        add_sub_same_shape_backward_vec4_kernel<<<grid_size, block_size>>>(
            reinterpret_cast<const float4*>(grad),
            reinterpret_cast<float4*>(grad_lhs),
            reinterpret_cast<float4*>(grad_rhs),
            vec_len,
            len,
            rhs_sign);
    } else {
        const unsigned int grid_size = linear_grid_size(len, block_size);
        add_sub_same_shape_backward_kernel<<<grid_size, block_size>>>(
            grad, grad_lhs, grad_rhs, len, rhs_sign);
    }
}

bool launch_add_sub_row_broadcast_backward_kernel(
    const float* grad,
    float* grad_lhs,
    float* grad_rhs,
    size_t len,
    size_t last_dim,
    bool vector_on_rhs,
    bool is_sub) {
    constexpr unsigned int block_size = 256;
    float rhs_sign = is_sub ? -1.0f : 1.0f;
    size_t rows = len / last_dim;
    const unsigned int full_grid = linear_grid_size(len, block_size);
    bool use_atomic = rows < 64;
    if (vector_on_rhs) {
        if (use_atomic) {
            if (!zero_f32_buffer(
                    grad_rhs,
                    last_dim,
                    "CUDA add/sub row-broadcast rhs grad initialization failed")) {
                return false;
            }
            add_sub_row_broadcast_backward_atomic_combined_kernel<<<full_grid, block_size>>>(
                grad, grad_lhs, grad_rhs, len, last_dim, 1.0f, rhs_sign);
        } else {
            add_sub_row_broadcast_copy_full_kernel<<<full_grid, block_size>>>(grad, grad_lhs, len, 1.0f);
            add_sub_row_broadcast_reduce_vector_kernel<<<linear_grid_size(last_dim, 1), block_size>>>(
                grad, grad_rhs, rows, last_dim, rhs_sign);
        }
    } else {
        if (use_atomic) {
            if (!zero_f32_buffer(
                    grad_lhs,
                    last_dim,
                    "CUDA add/sub row-broadcast lhs grad initialization failed")) {
                return false;
            }
            add_sub_row_broadcast_backward_atomic_combined_kernel<<<full_grid, block_size>>>(
                grad, grad_rhs, grad_lhs, len, last_dim, rhs_sign, 1.0f);
        } else {
            add_sub_row_broadcast_copy_full_kernel<<<full_grid, block_size>>>(grad, grad_rhs, len, rhs_sign);
            add_sub_row_broadcast_reduce_vector_kernel<<<linear_grid_size(last_dim, 1), block_size>>>(
                grad, grad_lhs, rows, last_dim, 1.0f);
        }
    }
    return true;
}

void launch_add_sub_scalar_broadcast_backward_kernel(
    const float* grad,
    float* grad_lhs,
    float* grad_rhs,
    size_t len,
    bool scalar_on_rhs,
    bool is_sub) {
    constexpr unsigned int block_size = 256;
    const unsigned int grid_size = linear_grid_size(len, block_size);
    float rhs_sign = is_sub ? -1.0f : 1.0f;
    if (scalar_on_rhs) {
        add_sub_scalar_broadcast_backward_kernel<<<grid_size, block_size>>>(
            grad, grad_lhs, grad_rhs, len, 1.0f, rhs_sign);
    } else {
        add_sub_scalar_broadcast_backward_kernel<<<grid_size, block_size>>>(
            grad, grad_rhs, grad_lhs, len, rhs_sign, 1.0f);
    }
}

void launch_add_sub_row_scalar_backward_kernel(
    const float* grad,
    float* grad_lhs,
    float* grad_rhs,
    size_t rows,
    size_t last_dim,
    bool scalar_on_rhs,
    bool is_sub) {
    constexpr unsigned int block_size = 256;
    float rhs_sign = is_sub ? -1.0f : 1.0f;
    add_sub_row_scalar_backward_kernel<<<linear_grid_size(rows, 1), block_size>>>(
        grad, grad_lhs, grad_rhs, rows, last_dim, scalar_on_rhs, rhs_sign);
}

void launch_add_sub_b1d_1h1_backward_kernel(
    const float* grad,
    float* grad_lhs,
    float* grad_rhs,
    size_t batch,
    size_t heads,
    size_t dim,
    bool b1d_on_lhs,
    bool is_sub) {
    constexpr int block_size = 256;
    float rhs_sign = is_sub ? -1.0f : 1.0f;
    add_sub_b1d_1h1_combined_backward_kernel<<<linear_grid_size(batch * dim + heads, 1), block_size>>>(
        grad, grad_lhs, grad_rhs, batch, heads, dim, b1d_on_lhs, rhs_sign);
}

void launch_add_sub_b1d_1hd_backward_kernel(
    const float* grad,
    float* grad_lhs,
    float* grad_rhs,
    size_t batch,
    size_t heads,
    size_t dim,
    bool b1d_on_lhs,
    bool is_sub) {
    constexpr int block_size = 256;
    float rhs_sign = is_sub ? -1.0f : 1.0f;
    add_sub_b1d_1hd_combined_backward_kernel<<<linear_grid_size(batch * dim + heads * dim, 1), block_size>>>(
        grad, grad_lhs, grad_rhs, batch, heads, dim, b1d_on_lhs, rhs_sign);
}

void launch_binary_scalar_rhs_vec4_kernel(
    int op,
    unsigned int grid_size,
    unsigned int block_size,
    const float4* lhs,
    const float* rhs,
    float4* out,
    size_t vec_len,
    size_t len) {
    switch (op) {
        case kBinaryAdd:
            binary_scalar_rhs_vec4_kernel<kBinaryAdd><<<grid_size, block_size>>>(lhs, rhs, out, vec_len, len);
            break;
        case kBinarySub:
            binary_scalar_rhs_vec4_kernel<kBinarySub><<<grid_size, block_size>>>(lhs, rhs, out, vec_len, len);
            break;
        case kBinaryMul:
            binary_scalar_rhs_vec4_kernel<kBinaryMul><<<grid_size, block_size>>>(lhs, rhs, out, vec_len, len);
            break;
        default:
            binary_scalar_rhs_vec4_kernel<-1><<<grid_size, block_size>>>(lhs, rhs, out, vec_len, len);
            break;
    }
}

void launch_binary_scalar_lhs_vec4_kernel(
    int op,
    unsigned int grid_size,
    unsigned int block_size,
    const float* lhs,
    const float4* rhs,
    float4* out,
    size_t vec_len,
    size_t len) {
    switch (op) {
        case kBinaryAdd:
            binary_scalar_lhs_vec4_kernel<kBinaryAdd><<<grid_size, block_size>>>(lhs, rhs, out, vec_len, len);
            break;
        case kBinarySub:
            binary_scalar_lhs_vec4_kernel<kBinarySub><<<grid_size, block_size>>>(lhs, rhs, out, vec_len, len);
            break;
        case kBinaryMul:
            binary_scalar_lhs_vec4_kernel<kBinaryMul><<<grid_size, block_size>>>(lhs, rhs, out, vec_len, len);
            break;
        default:
            binary_scalar_lhs_vec4_kernel<-1><<<grid_size, block_size>>>(lhs, rhs, out, vec_len, len);
            break;
    }
}

void launch_binary_lastdim_rhs_kernel(
    int op,
    bool vec4,
    unsigned int grid_size,
    unsigned int block_size,
    const float* lhs,
    const float* rhs,
    float* out,
    size_t len,
    size_t last_dim) {
    if (vec4) {
        const auto* lhs4 = reinterpret_cast<const float4*>(lhs);
        const auto* rhs4 = reinterpret_cast<const float4*>(rhs);
        auto* out4 = reinterpret_cast<float4*>(out);
        size_t vec_len = len / 4;
        switch (op) {
            case kBinaryAdd:
                binary_lastdim_rhs_vec4_kernel<kBinaryAdd><<<grid_size, block_size>>>(
                    lhs4, rhs4, out4, vec_len, last_dim);
                break;
            case kBinarySub:
                binary_lastdim_rhs_vec4_kernel<kBinarySub><<<grid_size, block_size>>>(
                    lhs4, rhs4, out4, vec_len, last_dim);
                break;
            case kBinaryMul:
                binary_lastdim_rhs_vec4_kernel<kBinaryMul><<<grid_size, block_size>>>(
                    lhs4, rhs4, out4, vec_len, last_dim);
                break;
            default:
                binary_lastdim_rhs_vec4_kernel<-1><<<grid_size, block_size>>>(
                    lhs4, rhs4, out4, vec_len, last_dim);
                break;
        }
    } else {
        switch (op) {
            case kBinaryAdd:
                binary_lastdim_rhs_kernel<kBinaryAdd><<<grid_size, block_size>>>(
                    lhs, rhs, out, len, last_dim);
                break;
            case kBinarySub:
                binary_lastdim_rhs_kernel<kBinarySub><<<grid_size, block_size>>>(
                    lhs, rhs, out, len, last_dim);
                break;
            case kBinaryMul:
                binary_lastdim_rhs_kernel<kBinaryMul><<<grid_size, block_size>>>(
                    lhs, rhs, out, len, last_dim);
                break;
            default:
                binary_lastdim_rhs_kernel<-1><<<grid_size, block_size>>>(
                    lhs, rhs, out, len, last_dim);
                break;
        }
    }
}

void launch_binary_scalar_rhs_backward_kernel(
    int op,
    unsigned int grid_size,
    unsigned int block_size,
    const float* lhs,
    const float* rhs,
    const float* grad,
    float* grad_lhs,
    float* grad_rhs,
    size_t len) {
    switch (op) {
        case kBinaryAdd:
            binary_scalar_rhs_backward_kernel<kBinaryAdd><<<grid_size, block_size>>>(
                lhs, rhs, grad, grad_lhs, grad_rhs, len);
            break;
        case kBinarySub:
            binary_scalar_rhs_backward_kernel<kBinarySub><<<grid_size, block_size>>>(
                lhs, rhs, grad, grad_lhs, grad_rhs, len);
            break;
        case kBinaryMul:
            binary_scalar_rhs_backward_kernel<kBinaryMul><<<grid_size, block_size>>>(
                lhs, rhs, grad, grad_lhs, grad_rhs, len);
            break;
        default:
            binary_scalar_rhs_backward_kernel<-1><<<grid_size, block_size>>>(
                lhs, rhs, grad, grad_lhs, grad_rhs, len);
            break;
    }
}

void launch_binary_scalar_lhs_backward_kernel(
    int op,
    unsigned int grid_size,
    unsigned int block_size,
    const float* lhs,
    const float* rhs,
    const float* grad,
    float* grad_lhs,
    float* grad_rhs,
    size_t len) {
    switch (op) {
        case kBinaryAdd:
            binary_scalar_lhs_backward_kernel<kBinaryAdd><<<grid_size, block_size>>>(
                lhs, rhs, grad, grad_lhs, grad_rhs, len);
            break;
        case kBinarySub:
            binary_scalar_lhs_backward_kernel<kBinarySub><<<grid_size, block_size>>>(
                lhs, rhs, grad, grad_lhs, grad_rhs, len);
            break;
        case kBinaryMul:
            binary_scalar_lhs_backward_kernel<kBinaryMul><<<grid_size, block_size>>>(
                lhs, rhs, grad, grad_lhs, grad_rhs, len);
            break;
        default:
            binary_scalar_lhs_backward_kernel<-1><<<grid_size, block_size>>>(
                lhs, rhs, grad, grad_lhs, grad_rhs, len);
            break;
    }
}

void launch_binary_lastdim_rhs_backward_kernel(
    int op,
    bool atomic,
    unsigned int grid_size,
    unsigned int block_size,
    const float* lhs,
    const float* rhs,
    const float* grad,
    float* grad_lhs,
    float* grad_rhs,
    size_t rows,
    size_t last_dim) {
    if (atomic) {
        size_t len = rows * last_dim;
        switch (op) {
            case kBinaryAdd:
                binary_lastdim_rhs_backward_atomic_kernel<kBinaryAdd><<<grid_size, block_size>>>(
                    lhs, rhs, grad, grad_lhs, grad_rhs, len, last_dim);
                break;
            case kBinarySub:
                binary_lastdim_rhs_backward_atomic_kernel<kBinarySub><<<grid_size, block_size>>>(
                    lhs, rhs, grad, grad_lhs, grad_rhs, len, last_dim);
                break;
            case kBinaryMul:
                binary_lastdim_rhs_backward_atomic_kernel<kBinaryMul><<<grid_size, block_size>>>(
                    lhs, rhs, grad, grad_lhs, grad_rhs, len, last_dim);
                break;
            default:
                binary_lastdim_rhs_backward_atomic_kernel<-1><<<grid_size, block_size>>>(
                    lhs, rhs, grad, grad_lhs, grad_rhs, len, last_dim);
                break;
        }
    } else {
        switch (op) {
            case kBinaryAdd:
                binary_lastdim_rhs_backward_kernel<kBinaryAdd><<<grid_size, block_size>>>(
                    lhs, rhs, grad, grad_lhs, grad_rhs, rows, last_dim);
                break;
            case kBinarySub:
                binary_lastdim_rhs_backward_kernel<kBinarySub><<<grid_size, block_size>>>(
                    lhs, rhs, grad, grad_lhs, grad_rhs, rows, last_dim);
                break;
            case kBinaryMul:
                binary_lastdim_rhs_backward_kernel<kBinaryMul><<<grid_size, block_size>>>(
                    lhs, rhs, grad, grad_lhs, grad_rhs, rows, last_dim);
                break;
            default:
                binary_lastdim_rhs_backward_kernel<-1><<<grid_size, block_size>>>(
                    lhs, rhs, grad, grad_lhs, grad_rhs, rows, last_dim);
                break;
        }
    }
}

void launch_binary_broadcast_kernel(
    int op,
    unsigned int grid_size,
    unsigned int block_size,
    const float* lhs,
    const float* rhs,
    float* out,
    const size_t* out_shape,
    const size_t* out_strides,
    const size_t* lhs_shape,
    const size_t* lhs_strides,
    const size_t* rhs_shape,
    const size_t* rhs_strides,
    size_t ndim,
    size_t len) {
    switch (op) {
        case kBinaryAdd:
            binary_broadcast_kernel<kBinaryAdd><<<grid_size, block_size>>>(
                lhs, rhs, out, out_shape, out_strides, lhs_shape, lhs_strides, rhs_shape, rhs_strides, ndim, len);
            break;
        case kBinarySub:
            binary_broadcast_kernel<kBinarySub><<<grid_size, block_size>>>(
                lhs, rhs, out, out_shape, out_strides, lhs_shape, lhs_strides, rhs_shape, rhs_strides, ndim, len);
            break;
        case kBinaryMul:
            binary_broadcast_kernel<kBinaryMul><<<grid_size, block_size>>>(
                lhs, rhs, out, out_shape, out_strides, lhs_shape, lhs_strides, rhs_shape, rhs_strides, ndim, len);
            break;
        default:
            binary_broadcast_kernel<-1><<<grid_size, block_size>>>(
                lhs, rhs, out, out_shape, out_strides, lhs_shape, lhs_strides, rhs_shape, rhs_strides, ndim, len);
            break;
    }
}

void launch_binary_broadcast_backward_kernel(
    int op,
    unsigned int grid_size,
    unsigned int block_size,
    const float* lhs,
    const float* rhs,
    const float* grad,
    float* grad_lhs,
    float* grad_rhs,
    const size_t* out_shape,
    const size_t* out_strides,
    const size_t* lhs_shape,
    const size_t* lhs_strides,
    const size_t* rhs_shape,
    const size_t* rhs_strides,
    size_t ndim,
    size_t len) {
    switch (op) {
        case kBinaryAdd:
            binary_broadcast_backward_kernel<kBinaryAdd><<<grid_size, block_size>>>(
                lhs, rhs, grad, grad_lhs, grad_rhs, out_shape, out_strides, lhs_shape, lhs_strides,
                rhs_shape, rhs_strides, ndim, len);
            break;
        case kBinarySub:
            binary_broadcast_backward_kernel<kBinarySub><<<grid_size, block_size>>>(
                lhs, rhs, grad, grad_lhs, grad_rhs, out_shape, out_strides, lhs_shape, lhs_strides,
                rhs_shape, rhs_strides, ndim, len);
            break;
        case kBinaryMul:
            binary_broadcast_backward_kernel<kBinaryMul><<<grid_size, block_size>>>(
                lhs, rhs, grad, grad_lhs, grad_rhs, out_shape, out_strides, lhs_shape, lhs_strides,
                rhs_shape, rhs_strides, ndim, len);
            break;
        default:
            binary_broadcast_backward_kernel<-1><<<grid_size, block_size>>>(
                lhs, rhs, grad, grad_lhs, grad_rhs, out_shape, out_strides, lhs_shape, lhs_strides,
                rhs_shape, rhs_strides, ndim, len);
            break;
    }
}

bool is_lastdim_rhs_broadcast(
    size_t ndim,
    const size_t* out_shape,
    const size_t* lhs_shape,
    const size_t* rhs_shape,
    size_t lhs_len,
    size_t rhs_len,
    size_t out_len) {
    if (ndim == 0 || out_shape == nullptr || lhs_shape == nullptr || rhs_shape == nullptr) {
        return false;
    }
    size_t last_dim = out_shape[ndim - 1];
    if (last_dim == 0 || rhs_len != last_dim || lhs_len != out_len) {
        return false;
    }
    for (size_t i = 0; i < ndim; ++i) {
        if (lhs_shape[i] != out_shape[i]) {
            return false;
        }
        if (i + 1 == ndim) {
            if (rhs_shape[i] != last_dim) {
                return false;
            }
        } else if (rhs_shape[i] != 1) {
            return false;
        }
    }
    return true;
}
