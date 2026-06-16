#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#if LUMEN_HAS_CUDNN
#include <cudnn.h>
#endif

#include <algorithm>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

thread_local std::string g_last_error;

constexpr int kUnaryRelu = 0;
constexpr int kUnarySigmoid = 1;
constexpr int kUnaryTanh = 2;
constexpr int kUnarySilu = 3;
constexpr int kUnaryGelu = 4;

constexpr int kBinaryAdd = 0;
constexpr int kBinarySub = 1;
constexpr int kBinaryMul = 2;
constexpr size_t kLastdimBroadcastFastMinLen = 1u << 15;
constexpr int kDTypeF32 = 0;
constexpr int kDTypeF16 = 1;
constexpr int kDTypeBF16 = 2;
constexpr int kDTypeI8 = 3;

float* handle_to_ptr(uint64_t handle) {
    return reinterpret_cast<float*>(handle);
}

void set_error(const std::string& message) {
    g_last_error = message;
}

void set_cuda_error(const char* prefix, cudaError_t status) {
    std::ostringstream oss;
    oss << prefix << ": " << cudaGetErrorString(status);
    set_error(oss.str());
}

void set_cublas_error(const char* prefix, cublasStatus_t status) {
    std::ostringstream oss;
    oss << prefix << ": cuBLAS status " << static_cast<int>(status);
    set_error(oss.str());
}

#if LUMEN_HAS_CUDNN
void set_cudnn_error(const char* prefix, cudnnStatus_t status) {
    std::ostringstream oss;
    oss << prefix << ": " << cudnnGetErrorString(status);
    set_error(oss.str());
}
#endif

struct CublasHandle {
    cublasHandle_t handle = nullptr;
    bool owns = true;
    int device = -1;

    ~CublasHandle() {
        // CUDA's Windows hybrid runtime can tear down per-thread state before C++
        // thread_local destructors run. Do not call CUDA APIs from TLS teardown;
        // cached handles are reclaimed with the CUDA context.
    }

    bool release(const char* context) {
        if (!owns || handle == nullptr) {
            return true;
        }
        int current_device = -1;
        cudaError_t device_status = cudaGetDevice(&current_device);
        if (device_status != cudaSuccess) {
            set_cuda_error(context, device_status);
            return false;
        }
        const bool restore = device >= 0 && current_device != device;
        if (restore) {
            cudaError_t switch_status = cudaSetDevice(device);
            if (switch_status != cudaSuccess) {
                set_cuda_error(context, switch_status);
                return false;
            }
        }
        cublasStatus_t destroy_status = cublasDestroy(handle);
        cudaError_t restore_status = restore ? cudaSetDevice(current_device) : cudaSuccess;
        if (destroy_status != CUBLAS_STATUS_SUCCESS) {
            set_cublas_error(context, destroy_status);
            return false;
        }
        handle = nullptr;
        device = -1;
        if (restore_status != cudaSuccess) {
            set_cuda_error(context, restore_status);
            return false;
        }
        return true;
    }
};

CublasHandle& thread_cublas_handle() {
    thread_local CublasHandle cached;
    return cached;
}

#if LUMEN_HAS_CUDNN
struct CudnnHandle {
    cudnnHandle_t handle = nullptr;
    bool owns = true;
    int device = -1;

    ~CudnnHandle() {
        // See CublasHandle: cached TLS resources must not call CUDA APIs while
        // the runtime is tearing down thread state.
    }

    bool release(const char* context) {
        if (!owns || handle == nullptr) {
            return true;
        }
        int current_device = -1;
        cudaError_t device_status = cudaGetDevice(&current_device);
        if (device_status != cudaSuccess) {
            set_cuda_error(context, device_status);
            return false;
        }
        const bool restore = device >= 0 && current_device != device;
        if (restore) {
            cudaError_t switch_status = cudaSetDevice(device);
            if (switch_status != cudaSuccess) {
                set_cuda_error(context, switch_status);
                return false;
            }
        }
        cudnnStatus_t destroy_status = cudnnDestroy(handle);
        cudaError_t restore_status = restore ? cudaSetDevice(current_device) : cudaSuccess;
        if (destroy_status != CUDNN_STATUS_SUCCESS) {
            set_cudnn_error(context, destroy_status);
            return false;
        }
        handle = nullptr;
        device = -1;
        if (restore_status != cudaSuccess) {
            set_cuda_error(context, restore_status);
            return false;
        }
        return true;
    }
};

CudnnHandle& thread_cudnn_handle() {
    thread_local CudnnHandle cached;
    return cached;
}

struct CudnnTensorDescriptor {
    cudnnTensorDescriptor_t desc = nullptr;

    ~CudnnTensorDescriptor() {
        if (desc != nullptr) {
            cudnnDestroyTensorDescriptor(desc);
        }
    }
};

struct CudnnActivationDescriptor {
    cudnnActivationDescriptor_t desc = nullptr;

    ~CudnnActivationDescriptor() {
        if (desc != nullptr) {
            cudnnDestroyActivationDescriptor(desc);
        }
    }
};

struct CudnnFilterDescriptor {
    cudnnFilterDescriptor_t desc = nullptr;

    ~CudnnFilterDescriptor() {
        if (desc != nullptr) {
            cudnnDestroyFilterDescriptor(desc);
        }
    }
};

struct CudnnConvolutionDescriptor {
    cudnnConvolutionDescriptor_t desc = nullptr;

    ~CudnnConvolutionDescriptor() {
        if (desc != nullptr) {
            cudnnDestroyConvolutionDescriptor(desc);
        }
    }
};
#endif

struct ReusableCudaWorkspace;

std::vector<ReusableCudaWorkspace*>& thread_cuda_workspaces() {
    // Keep the registry itself alive through TLS teardown. Workspace objects
    // unregistering or freeing CUDA memory from TLS destructors is unsafe with
    // CUDA's Windows hybrid runtime teardown order.
    thread_local auto* workspaces = new std::vector<ReusableCudaWorkspace*>();
    return *workspaces;
}

struct ReusableCudaWorkspace {
    void* ptr = nullptr;
    size_t capacity = 0;
    int device = -1;

    ReusableCudaWorkspace() {
        thread_cuda_workspaces().push_back(this);
    }

    ~ReusableCudaWorkspace() {
        // See CublasHandle. Runtime paths still release on resize/device switch;
        // callers can explicitly release final allocations before TLS teardown.
    }

    bool release(const char* context) {
        if (ptr == nullptr) {
            capacity = 0;
            device = -1;
            return true;
        }

        int current_device = -1;
        cudaError_t device_status = cudaGetDevice(&current_device);
        if (device_status != cudaSuccess) {
            if (context != nullptr) {
                set_cuda_error(context, device_status);
            }
            return false;
        }
        const bool restore = device >= 0 && current_device != device;
        if (restore) {
            cudaError_t status = cudaSetDevice(device);
            if (status != cudaSuccess) {
                if (context != nullptr) {
                    set_cuda_error(context, status);
                }
                return false;
            }
        }
        cudaError_t free_status = cudaFree(ptr);
        cudaError_t restore_status = restore ? cudaSetDevice(current_device) : cudaSuccess;
        if (free_status != cudaSuccess) {
            if (context != nullptr) {
                set_cuda_error(context, free_status);
            }
            return false;
        }
        ptr = nullptr;
        capacity = 0;
        device = -1;
        if (restore_status != cudaSuccess) {
            if (context != nullptr) {
                set_cuda_error(context, restore_status);
            }
            return false;
        }
        return true;
    }

    bool ensure(size_t bytes, const char* context) {
        int current_device = -1;
        cudaError_t status = cudaGetDevice(&current_device);
        if (status != cudaSuccess) {
            set_cuda_error(context, status);
            return false;
        }
        if (ptr != nullptr && device != current_device) {
            if (!release(context)) {
                return false;
            }
        }
        if (bytes <= capacity && (ptr != nullptr || bytes == 0)) {
            return true;
        }
        if (ptr != nullptr) {
            if (!release(context)) {
                return false;
            }
        }
        if (bytes == 0) {
            return true;
        }
        status = cudaMalloc(&ptr, bytes);
        if (status != cudaSuccess) {
            set_cuda_error(context, status);
            return false;
        }
        capacity = bytes;
        device = current_device;
        return true;
    }
};

bool release_thread_cuda_workspaces(const char* context) {
    bool ok = true;
    for (ReusableCudaWorkspace* workspace : thread_cuda_workspaces()) {
        if (workspace != nullptr && !workspace->release(context)) {
            ok = false;
        }
    }
    return ok;
}

static unsigned int linear_grid_size(size_t total, unsigned int block_size) {
    constexpr size_t max_grid_x = 2147483647;
    const size_t blocks = total / block_size + (total % block_size != 0);
    return static_cast<unsigned int>(blocks < max_grid_x ? blocks : max_grid_x);
}
struct CudaBufferPool {
    // NVCC is built with --default-stream legacy. Its cross-thread ordering
    // keeps an asynchronously released pointer safe before shared-pool reuse.
    // Per-thread streams require event-gated reuse instead.
    std::mutex mutex;
    std::unordered_map<int, std::unordered_map<size_t, std::vector<void*>>> free_lists_by_device;
    std::unordered_map<int, size_t> cached_bytes_by_device;
};

constexpr size_t kMaxCudaBufferPoolBytes = 256ull * 1024ull * 1024ull;
constexpr size_t kMaxPooledCudaBufferBytes = 64ull * 1024ull * 1024ull;

CudaBufferPool& cuda_buffer_pool() {
    static CudaBufferPool* pool = new CudaBufferPool();
    return *pool;
}

bool is_poolable_cuda_buffer(size_t bytes) {
    return bytes > 0 && bytes <= kMaxPooledCudaBufferBytes;
}

bool current_cuda_device(int& device) {
    cudaError_t status = cudaGetDevice(&device);
    return status == cudaSuccess;
}

bool cuda_pointer_device(void* ptr, int& device) {
    cudaPointerAttributes attrs;
    cudaError_t status = cudaPointerGetAttributes(&attrs, ptr);
    if (status != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    device = attrs.device;
    return true;
}

bool free_cuda_ptr_on_device(void* ptr, int device, const char* context) {
    int current = 0;
    cudaError_t status = cudaGetDevice(&current);
    if (status != cudaSuccess) {
        if (context != nullptr) {
            set_cuda_error(context, status);
        }
        return false;
    }
    bool restore = current != device;
    if (restore) {
        status = cudaSetDevice(device);
        if (status != cudaSuccess) {
            if (context != nullptr) {
                set_cuda_error(context, status);
            }
            return false;
        }
    }
    cudaError_t free_status = cudaFree(ptr);
    cudaError_t restore_status = restore ? cudaSetDevice(current) : cudaSuccess;
    if (free_status != cudaSuccess) {
        if (context != nullptr) {
            set_cuda_error(context, free_status);
        }
        return false;
    }
    if (restore_status != cudaSuccess) {
        if (context != nullptr) {
            set_cuda_error(context, restore_status);
        }
        return false;
    }
    return true;
}

bool try_take_pooled_cuda_buffer(size_t bytes, void** out) {
    if (!is_poolable_cuda_buffer(bytes)) {
        return false;
    }

    int device = 0;
    if (!current_cuda_device(device)) {
        return false;
    }

    CudaBufferPool& pool = cuda_buffer_pool();
    std::lock_guard<std::mutex> lock(pool.mutex);
    auto device_it = pool.free_lists_by_device.find(device);
    if (device_it == pool.free_lists_by_device.end()) {
        return false;
    }
    auto size_it = device_it->second.find(bytes);
    if (size_it == device_it->second.end() || size_it->second.empty()) {
        return false;
    }

    *out = size_it->second.back();
    size_it->second.pop_back();
    auto cached_it = pool.cached_bytes_by_device.find(device);
    if (cached_it != pool.cached_bytes_by_device.end() && cached_it->second >= bytes) {
        cached_it->second -= bytes;
    }
    return true;
}

void release_cuda_buffer(uint64_t handle, size_t len) {
    if (handle == 0) {
        return;
    }

    if (len > static_cast<size_t>(-1) / sizeof(float)) {
        cudaFree(handle_to_ptr(handle));
        return;
    }
    size_t bytes = len * sizeof(float);
    if (is_poolable_cuda_buffer(bytes)) {
        int device = 0;
        void* ptr = handle_to_ptr(handle);
        if (!cuda_pointer_device(ptr, device)) {
            cudaFree(ptr);
            return;
        }

        CudaBufferPool& pool = cuda_buffer_pool();
        std::lock_guard<std::mutex> lock(pool.mutex);
        size_t cached_bytes = pool.cached_bytes_by_device[device];
        if (cached_bytes <= kMaxCudaBufferPoolBytes &&
            bytes <= kMaxCudaBufferPoolBytes - cached_bytes) {
            pool.free_lists_by_device[device][bytes].push_back(ptr);
            pool.cached_bytes_by_device[device] += bytes;
            return;
        }
    }

    cudaFree(handle_to_ptr(handle));
}

bool clear_cuda_buffer_pool(const char* context) {
    std::vector<std::pair<int, void*>> to_free;
    CudaBufferPool& pool = cuda_buffer_pool();
    {
        std::lock_guard<std::mutex> lock(pool.mutex);
        for (auto& device_entry : pool.free_lists_by_device) {
            int device = device_entry.first;
            for (auto& size_entry : device_entry.second) {
                for (void* ptr : size_entry.second) {
                    to_free.push_back({device, ptr});
                }
            }
        }
        pool.free_lists_by_device.clear();
        pool.cached_bytes_by_device.clear();
    }

    bool ok = true;
    for (auto& entry : to_free) {
        if (!free_cuda_ptr_on_device(entry.second, entry.first, context)) {
            ok = false;
        }
    }
    return ok;
}

bool validate_dims(size_t m, size_t n, size_t k) {
    if (m == 0 || n == 0 || k == 0) {
        set_error("CUDA matmul dimensions must be greater than zero");
        return false;
    }
    constexpr size_t max_cublas_int = static_cast<size_t>(INT_MAX);
    if (m > max_cublas_int || n > max_cublas_int || k > max_cublas_int) {
        set_error("CUDA matmul dimensions exceed cuBLAS int range");
        return false;
    }
    return true;
}

bool validate_cublas_batch_count(size_t batch_count) {
    if (batch_count == 0) {
        set_error("CUDA batch_count must be greater than zero");
        return false;
    }
    if (batch_count > static_cast<size_t>(INT_MAX)) {
        set_error("CUDA batch_count exceeds cuBLAS int range");
        return false;
    }
    return true;
}

bool validate_grid_yz_dimension(const char* context, size_t dimension) {
    constexpr size_t max_grid_yz = 65535;
    if (dimension > max_grid_yz) {
        set_error(context);
        return false;
    }
    return true;
}

bool validate_int_dimensions(const char* context, std::initializer_list<size_t> dimensions) {
    for (size_t dimension : dimensions) {
        if (dimension > static_cast<size_t>(INT_MAX)) {
            set_error(context);
            return false;
        }
    }
    return true;
}

bool validate_finite_value(float value, const char* context) {
    if (!std::isfinite(value)) {
        set_error(context);
        return false;
    }
    return true;
}

bool validate_positive_finite_value(float value, const char* context) {
    if (!std::isfinite(value) || value <= 0.0f) {
        set_error(context);
        return false;
    }
    return true;
}

bool checked_scale_product(
    const char* context,
    float lhs_scale,
    float rhs_scale,
    float* out) {
    if (!std::isfinite(lhs_scale) || lhs_scale <= 0.0f ||
        !std::isfinite(rhs_scale) || rhs_scale <= 0.0f) {
        set_error(std::string(context) + ": scales must be finite and > 0");
        return false;
    }
    const float product = lhs_scale * rhs_scale;
    if (!std::isfinite(product)) {
        set_error(std::string(context) + ": combined scale overflow");
        return false;
    }
    *out = product;
    return true;
}

bool checked_product(
    const char* context,
    std::initializer_list<size_t> factors,
    size_t* out) {
    size_t product = 1;
    for (size_t factor : factors) {
        if (factor != 0 && product > static_cast<size_t>(-1) / factor) {
            set_error(context);
            return false;
        }
        product *= factor;
    }
    *out = product;
    return true;
}

bool checked_add(const char* context, size_t lhs, size_t rhs, size_t* out) {
    if (lhs > static_cast<size_t>(-1) - rhs) {
        set_error(context);
        return false;
    }
    *out = lhs + rhs;
    return true;
}

bool checked_align_up(const char* context, size_t value, size_t alignment, size_t* out) {
    if (alignment == 0) {
        set_error(context);
        return false;
    }
    size_t padded = 0;
    if (!checked_add(context, value, alignment - 1, &padded)) {
        return false;
    }
    *out = padded / alignment * alignment;
    return true;
}

bool checked_byte_length(size_t len, size_t element_size, const char* context, size_t* out) {
    return checked_product(context, {len, element_size}, out);
}

bool zero_f32_buffer(float* ptr, size_t len, const char* context) {
    size_t bytes = 0;
    if (!checked_byte_length(len, sizeof(float), context, &bytes)) {
        return false;
    }
    cudaError_t status = cudaMemset(ptr, 0, bytes);
    if (status != cudaSuccess) {
        set_cuda_error(context, status);
        return false;
    }
    return true;
}

template <typename T>
bool upload_typed_input(const T* host, size_t len, T** device, const char* context) {
    *device = nullptr;
    if (host == nullptr) {
        set_error(std::string(context) + ": null host input");
        return false;
    }
    size_t bytes = 0;
    const std::string overflow_context = std::string(context) + ": byte length overflow";
    if (!checked_byte_length(len, sizeof(T), overflow_context.c_str(), &bytes)) {
        return false;
    }
    cudaError_t status = cudaMalloc(reinterpret_cast<void**>(device), bytes);
    if (status != cudaSuccess) {
        set_cuda_error(context, status);
        return false;
    }
    status = cudaMemcpy(*device, host, bytes, cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        cudaFree(*device);
        *device = nullptr;
        set_cuda_error(context, status);
        return false;
    }
    return true;
}

template <typename T>
struct ScopedDeviceInput {
    T* ptr = nullptr;
    ~ScopedDeviceInput() {
        if (ptr != nullptr) {
            cudaFree(ptr);
        }
    }
};

bool init_cublas(CublasHandle& handle) {
    CublasHandle& cached = thread_cublas_handle();
    int current_device = -1;
    cudaError_t device_status = cudaGetDevice(&current_device);
    if (device_status != cudaSuccess) {
        set_cuda_error("failed to query CUDA device for cuBLAS handle", device_status);
        return false;
    }
    if (cached.handle != nullptr && cached.device != current_device) {
        cudaError_t switch_status = cudaSetDevice(cached.device);
        if (switch_status != cudaSuccess) {
            set_cuda_error("failed to switch CUDA device for stale cuBLAS handle", switch_status);
            return false;
        }
        cublasStatus_t destroy_status = cublasDestroy(cached.handle);
        if (destroy_status == CUBLAS_STATUS_SUCCESS) {
            cached.handle = nullptr;
            cached.device = -1;
        }
        switch_status = cudaSetDevice(current_device);
        if (destroy_status != CUBLAS_STATUS_SUCCESS) {
            set_cublas_error("failed to destroy stale cuBLAS handle", destroy_status);
            return false;
        }
        if (switch_status != cudaSuccess) {
            set_cuda_error("failed to restore CUDA device after cuBLAS handle reset", switch_status);
            return false;
        }
    }
    if (cached.handle == nullptr) {
        cublasStatus_t status = cublasCreate(&cached.handle);
        if (status != CUBLAS_STATUS_SUCCESS) {
            set_cublas_error("failed to create cuBLAS handle", status);
            return false;
        }
        cached.device = current_device;
    }
    handle.handle = cached.handle;
    handle.owns = false;
    handle.device = current_device;
    return true;
}

#if LUMEN_HAS_CUDNN
bool init_cudnn(CudnnHandle& handle) {
    CudnnHandle& cached = thread_cudnn_handle();
    int current_device = -1;
    cudaError_t device_status = cudaGetDevice(&current_device);
    if (device_status != cudaSuccess) {
        set_cuda_error("failed to query CUDA device for cuDNN handle", device_status);
        return false;
    }
    if (cached.handle != nullptr && cached.device != current_device) {
        cudaError_t switch_status = cudaSetDevice(cached.device);
        if (switch_status != cudaSuccess) {
            set_cuda_error("failed to switch CUDA device for stale cuDNN handle", switch_status);
            return false;
        }
        cudnnStatus_t destroy_status = cudnnDestroy(cached.handle);
        if (destroy_status == CUDNN_STATUS_SUCCESS) {
            cached.handle = nullptr;
            cached.device = -1;
        }
        switch_status = cudaSetDevice(current_device);
        if (destroy_status != CUDNN_STATUS_SUCCESS) {
            set_cudnn_error("failed to destroy stale cuDNN handle", destroy_status);
            return false;
        }
        if (switch_status != cudaSuccess) {
            set_cuda_error("failed to restore CUDA device after cuDNN handle reset", switch_status);
            return false;
        }
    }
    if (cached.handle == nullptr) {
        cudnnStatus_t status = cudnnCreate(&cached.handle);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_cudnn_error("failed to create cuDNN handle", status);
            return false;
        }
        cached.device = current_device;
    }
    handle.handle = cached.handle;
    handle.owns = false;
    handle.device = current_device;
    return true;
}

bool init_tensor_descriptor_4d(
    CudnnTensorDescriptor& desc,
    int n,
    int c,
    int h,
    int w) {
    cudnnStatus_t status = cudnnCreateTensorDescriptor(&desc.desc);
    if (status != CUDNN_STATUS_SUCCESS) {
        set_cudnn_error("failed to create cuDNN tensor descriptor", status);
        return false;
    }
    status = cudnnSetTensor4dDescriptor(desc.desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);
    if (status != CUDNN_STATUS_SUCCESS) {
        set_cudnn_error("failed to initialize cuDNN tensor descriptor", status);
        return false;
    }
    return true;
}

bool init_activation_descriptor(
    CudnnActivationDescriptor& desc,
    cudnnActivationMode_t mode) {
    cudnnStatus_t status = cudnnCreateActivationDescriptor(&desc.desc);
    if (status != CUDNN_STATUS_SUCCESS) {
        set_cudnn_error("failed to create cuDNN activation descriptor", status);
        return false;
    }
    status = cudnnSetActivationDescriptor(desc.desc, mode, CUDNN_PROPAGATE_NAN, 0.0);
    if (status != CUDNN_STATUS_SUCCESS) {
        set_cudnn_error("failed to initialize cuDNN activation descriptor", status);
        return false;
    }
    return true;
}

bool init_filter_descriptor_4d(
    CudnnFilterDescriptor& desc,
    int out_channels,
    int in_channels,
    int kernel_h,
    int kernel_w) {
    cudnnStatus_t status = cudnnCreateFilterDescriptor(&desc.desc);
    if (status != CUDNN_STATUS_SUCCESS) {
        set_cudnn_error("failed to create cuDNN filter descriptor", status);
        return false;
    }
    status = cudnnSetFilter4dDescriptor(
        desc.desc,
        CUDNN_DATA_FLOAT,
        CUDNN_TENSOR_NCHW,
        out_channels,
        in_channels,
        kernel_h,
        kernel_w);
    if (status != CUDNN_STATUS_SUCCESS) {
        set_cudnn_error("failed to initialize cuDNN filter descriptor", status);
        return false;
    }
    return true;
}

bool init_convolution_descriptor_2d(
    CudnnConvolutionDescriptor& desc,
    int pad_h,
    int pad_w,
    int stride_h,
    int stride_w) {
    cudnnStatus_t status = cudnnCreateConvolutionDescriptor(&desc.desc);
    if (status != CUDNN_STATUS_SUCCESS) {
        set_cudnn_error("failed to create cuDNN convolution descriptor", status);
        return false;
    }
    status = cudnnSetConvolution2dDescriptor(
        desc.desc,
        pad_h,
        pad_w,
        stride_h,
        stride_w,
        1,
        1,
        CUDNN_CROSS_CORRELATION,
        CUDNN_DATA_FLOAT);
    if (status != CUDNN_STATUS_SUCCESS) {
        set_cudnn_error("failed to initialize cuDNN convolution descriptor", status);
        return false;
    }
    return true;
}

constexpr size_t kMaxCudnnConvWorkspaceBytes = 256ull * 1024ull * 1024ull;

bool workspace_fits(size_t bytes) {
    return bytes <= kMaxCudnnConvWorkspaceBytes;
}

bool select_cudnn_fwd_algo(
    cudnnHandle_t handle,
    cudnnTensorDescriptor_t input_desc,
    cudnnFilterDescriptor_t filter_desc,
    cudnnConvolutionDescriptor_t conv_desc,
    cudnnTensorDescriptor_t output_desc,
    cudnnConvolutionFwdAlgo_t& algo,
    size_t& workspace_bytes) {
    algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    workspace_bytes = 0;

    int max_count = 0;
    cudnnStatus_t status = cudnnGetConvolutionForwardAlgorithmMaxCount(handle, &max_count);
    if (status == CUDNN_STATUS_SUCCESS && max_count > 0) {
        std::vector<cudnnConvolutionFwdAlgoPerf_t> results(static_cast<size_t>(max_count));
        int returned = 0;
        status = cudnnGetConvolutionForwardAlgorithm_v7(
            handle,
            input_desc,
            filter_desc,
            conv_desc,
            output_desc,
            max_count,
            &returned,
            results.data());
        if (status == CUDNN_STATUS_SUCCESS) {
            for (int i = 0; i < returned; ++i) {
                if (results[static_cast<size_t>(i)].status != CUDNN_STATUS_SUCCESS ||
                    !workspace_fits(results[static_cast<size_t>(i)].memory)) {
                    continue;
                }
                size_t bytes = 0;
                cudnnStatus_t workspace_status = cudnnGetConvolutionForwardWorkspaceSize(
                    handle,
                    input_desc,
                    filter_desc,
                    conv_desc,
                    output_desc,
                    results[static_cast<size_t>(i)].algo,
                    &bytes);
                if (workspace_status == CUDNN_STATUS_SUCCESS && workspace_fits(bytes)) {
                    algo = results[static_cast<size_t>(i)].algo;
                    workspace_bytes = bytes;
                    return true;
                }
            }
        }
    }

    status = cudnnGetConvolutionForwardWorkspaceSize(
        handle,
        input_desc,
        filter_desc,
        conv_desc,
        output_desc,
        algo,
        &workspace_bytes);
    if (status != CUDNN_STATUS_SUCCESS) {
        set_cudnn_error("failed to query cuDNN conv2d forward workspace", status);
        return false;
    }
    if (!workspace_fits(workspace_bytes)) {
        set_error("cuDNN conv2d forward workspace exceeds the configured limit");
        return false;
    }
    return true;
}

bool select_cudnn_bwd_data_algo(
    cudnnHandle_t handle,
    cudnnFilterDescriptor_t filter_desc,
    cudnnTensorDescriptor_t grad_output_desc,
    cudnnConvolutionDescriptor_t conv_desc,
    cudnnTensorDescriptor_t grad_input_desc,
    cudnnConvolutionBwdDataAlgo_t& algo,
    size_t& workspace_bytes) {
    algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
    workspace_bytes = 0;

    int max_count = 0;
    cudnnStatus_t status = cudnnGetConvolutionBackwardDataAlgorithmMaxCount(handle, &max_count);
    if (status == CUDNN_STATUS_SUCCESS && max_count > 0) {
        std::vector<cudnnConvolutionBwdDataAlgoPerf_t> results(static_cast<size_t>(max_count));
        int returned = 0;
        status = cudnnGetConvolutionBackwardDataAlgorithm_v7(
            handle,
            filter_desc,
            grad_output_desc,
            conv_desc,
            grad_input_desc,
            max_count,
            &returned,
            results.data());
        if (status == CUDNN_STATUS_SUCCESS) {
            for (int i = 0; i < returned; ++i) {
                if (results[static_cast<size_t>(i)].status != CUDNN_STATUS_SUCCESS ||
                    !workspace_fits(results[static_cast<size_t>(i)].memory)) {
                    continue;
                }
                size_t bytes = 0;
                cudnnStatus_t workspace_status = cudnnGetConvolutionBackwardDataWorkspaceSize(
                    handle,
                    filter_desc,
                    grad_output_desc,
                    conv_desc,
                    grad_input_desc,
                    results[static_cast<size_t>(i)].algo,
                    &bytes);
                if (workspace_status == CUDNN_STATUS_SUCCESS && workspace_fits(bytes)) {
                    algo = results[static_cast<size_t>(i)].algo;
                    workspace_bytes = bytes;
                    return true;
                }
            }
        }
    }

    status = cudnnGetConvolutionBackwardDataWorkspaceSize(
        handle,
        filter_desc,
        grad_output_desc,
        conv_desc,
        grad_input_desc,
        algo,
        &workspace_bytes);
    if (status != CUDNN_STATUS_SUCCESS) {
        set_cudnn_error("failed to query cuDNN conv2d backward data workspace", status);
        return false;
    }
    if (!workspace_fits(workspace_bytes)) {
        set_error("cuDNN conv2d backward data workspace exceeds the configured limit");
        return false;
    }
    return true;
}

bool select_cudnn_bwd_filter_algo(
    cudnnHandle_t handle,
    cudnnTensorDescriptor_t input_desc,
    cudnnTensorDescriptor_t grad_output_desc,
    cudnnConvolutionDescriptor_t conv_desc,
    cudnnFilterDescriptor_t grad_weight_desc,
    cudnnConvolutionBwdFilterAlgo_t& algo,
    size_t& workspace_bytes) {
    algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
    workspace_bytes = 0;

    int max_count = 0;
    cudnnStatus_t status = cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(handle, &max_count);
    if (status == CUDNN_STATUS_SUCCESS && max_count > 0) {
        std::vector<cudnnConvolutionBwdFilterAlgoPerf_t> results(static_cast<size_t>(max_count));
        int returned = 0;
        status = cudnnGetConvolutionBackwardFilterAlgorithm_v7(
            handle,
            input_desc,
            grad_output_desc,
            conv_desc,
            grad_weight_desc,
            max_count,
            &returned,
            results.data());
        if (status == CUDNN_STATUS_SUCCESS) {
            for (int i = 0; i < returned; ++i) {
                if (results[static_cast<size_t>(i)].status != CUDNN_STATUS_SUCCESS ||
                    !workspace_fits(results[static_cast<size_t>(i)].memory)) {
                    continue;
                }
                size_t bytes = 0;
                cudnnStatus_t workspace_status = cudnnGetConvolutionBackwardFilterWorkspaceSize(
                    handle,
                    input_desc,
                    grad_output_desc,
                    conv_desc,
                    grad_weight_desc,
                    results[static_cast<size_t>(i)].algo,
                    &bytes);
                if (workspace_status == CUDNN_STATUS_SUCCESS && workspace_fits(bytes)) {
                    algo = results[static_cast<size_t>(i)].algo;
                    workspace_bytes = bytes;
                    return true;
                }
            }
        }
    }

    status = cudnnGetConvolutionBackwardFilterWorkspaceSize(
        handle,
        input_desc,
        grad_output_desc,
        conv_desc,
        grad_weight_desc,
        algo,
        &workspace_bytes);
    if (status != CUDNN_STATUS_SUCCESS) {
        set_cudnn_error("failed to query cuDNN conv2d backward filter workspace", status);
        return false;
    }
    if (!workspace_fits(workspace_bytes)) {
        set_error("cuDNN conv2d backward filter workspace exceeds the configured limit");
        return false;
    }
    return true;
}

bool cudnn_activation_mode_for_op(int op, cudnnActivationMode_t& mode) {
    switch (op) {
        case kUnaryRelu:
            mode = CUDNN_ACTIVATION_RELU;
            return true;
        case kUnarySigmoid:
            mode = CUDNN_ACTIVATION_SIGMOID;
            return true;
        case kUnaryTanh:
            mode = CUDNN_ACTIVATION_TANH;
            return true;
#ifdef CUDNN_ACTIVATION_SWISH
        case kUnarySilu:
            mode = CUDNN_ACTIVATION_SWISH;
            return true;
#endif
        default:
            return false;
    }
}
#endif

bool sync_cuda(const char* context) {
    cudaError_t status = cudaDeviceSynchronize();
    if (status != cudaSuccess) {
        set_cuda_error(context, status);
        return false;
    }
    return true;
}

bool check_cuda_launch(const char* context) {
    // Device-only operations run on the default stream and remain ordered without
    // blocking the host. Synchronize only at explicit host-observation boundaries.
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        set_cuda_error(context, status);
        return false;
    }
    return true;
}

bool checked_shape_numel(
    const char* context,
    const size_t* shape,
    size_t ndim,
    size_t* out) {
    if (shape == nullptr || out == nullptr || ndim == 0) {
        set_error(std::string(context) + ": invalid shape metadata");
        return false;
    }
    size_t len = 1;
    for (size_t i = 0; i < ndim; ++i) {
        if (shape[i] == 0) {
            set_error(std::string(context) + ": shape dimensions must be greater than zero");
            return false;
        }
        if (len > static_cast<size_t>(-1) / shape[i]) {
            set_error(std::string(context) + ": shape element count overflow");
            return false;
        }
        len *= shape[i];
    }
    *out = len;
    return true;
}
