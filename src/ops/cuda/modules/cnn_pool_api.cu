extern "C" int lumen_cuda_conv2d_f32_device(
    uint64_t input_handle,
    uint64_t weight_handle,
    uint64_t bias_handle,
    uint64_t out_handle,
    size_t batch_size,
    size_t in_channels,
    size_t in_h,
    size_t in_w,
    size_t out_channels,
    size_t k_h,
    size_t k_w,
    size_t pad_h,
    size_t pad_w,
    size_t stride_h,
    size_t stride_w,
    size_t out_h,
    size_t out_w) {
    if (!validate_handle(input_handle, "CUDA conv2d input handle") ||
        !validate_handle(weight_handle, "CUDA conv2d weight handle") ||
        !validate_handle(out_handle, "CUDA conv2d output handle")) {
        return 1;
    }
    if (batch_size == 0 || in_channels == 0 || in_h == 0 || in_w == 0 || out_channels == 0 ||
        k_h == 0 || k_w == 0 || stride_h == 0 || stride_w == 0 || out_h == 0 || out_w == 0) {
        set_error("CUDA conv2d dimensions must be greater than zero");
        return 1;
    }
    if (!validate_int_dimensions(
            "CUDA conv2d dimensions exceed cuDNN int range",
            {batch_size, in_channels, in_h, in_w, out_channels, k_h, k_w, pad_h, pad_w, stride_h,
             stride_w, out_h, out_w})) {
        return 1;
    }

#if !LUMEN_HAS_CUDNN
    set_error("CUDA conv2d requires cuDNN support");
    return 1;
#else
    CudnnHandle handle;
    CudnnTensorDescriptor input_desc;
    CudnnTensorDescriptor output_desc;
    CudnnTensorDescriptor bias_desc;
    CudnnFilterDescriptor filter_desc;
    CudnnConvolutionDescriptor conv_desc;
    if (!init_cudnn(handle) ||
        !init_tensor_descriptor_4d(
            input_desc,
            static_cast<int>(batch_size),
            static_cast<int>(in_channels),
            static_cast<int>(in_h),
            static_cast<int>(in_w)) ||
        !init_tensor_descriptor_4d(
            output_desc,
            static_cast<int>(batch_size),
            static_cast<int>(out_channels),
            static_cast<int>(out_h),
            static_cast<int>(out_w)) ||
        !init_filter_descriptor_4d(
            filter_desc,
            static_cast<int>(out_channels),
            static_cast<int>(in_channels),
            static_cast<int>(k_h),
            static_cast<int>(k_w)) ||
        !init_convolution_descriptor_2d(
            conv_desc,
            static_cast<int>(pad_h),
            static_cast<int>(pad_w),
            static_cast<int>(stride_h),
            static_cast<int>(stride_w))) {
        return 1;
    }

    int cudnn_n = 0;
    int cudnn_c = 0;
    int cudnn_h = 0;
    int cudnn_w = 0;
    cudnnStatus_t status = cudnnGetConvolution2dForwardOutputDim(
        conv_desc.desc,
        input_desc.desc,
        filter_desc.desc,
        &cudnn_n,
        &cudnn_c,
        &cudnn_h,
        &cudnn_w);
    if (status != CUDNN_STATUS_SUCCESS) {
        set_cudnn_error("failed to query cuDNN conv2d output shape", status);
        return 1;
    }
    if (cudnn_n != static_cast<int>(batch_size) || cudnn_c != static_cast<int>(out_channels) ||
        cudnn_h != static_cast<int>(out_h) || cudnn_w != static_cast<int>(out_w)) {
        set_error("cuDNN conv2d output shape does not match the expected dimensions");
        return 1;
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    cudnnConvolutionFwdAlgo_t fwd_algo;
    size_t fwd_workspace_bytes = 0;
    if (!select_cudnn_fwd_algo(
            handle.handle,
            input_desc.desc,
            filter_desc.desc,
            conv_desc.desc,
            output_desc.desc,
            fwd_algo,
            fwd_workspace_bytes)) {
        return 1;
    }
    thread_local ReusableCudaWorkspace workspace;
    if (!workspace.ensure(fwd_workspace_bytes, "failed to allocate cuDNN conv2d forward workspace")) {
        fwd_algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
        fwd_workspace_bytes = 0;
    }
    status = cudnnConvolutionForward(
        handle.handle,
        &alpha,
        input_desc.desc,
        handle_to_ptr(input_handle),
        filter_desc.desc,
        handle_to_ptr(weight_handle),
        conv_desc.desc,
        fwd_algo,
        workspace.ptr,
        fwd_workspace_bytes,
        &beta,
        output_desc.desc,
        handle_to_ptr(out_handle));
    if (status != CUDNN_STATUS_SUCCESS && fwd_algo != CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM) {
        fwd_workspace_bytes = 0;
        status = cudnnConvolutionForward(
            handle.handle,
            &alpha,
            input_desc.desc,
            handle_to_ptr(input_handle),
            filter_desc.desc,
            handle_to_ptr(weight_handle),
            conv_desc.desc,
            CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
            nullptr,
            0,
            &beta,
            output_desc.desc,
            handle_to_ptr(out_handle));
    }
    if (status != CUDNN_STATUS_SUCCESS) {
        set_cudnn_error("cuDNN conv2d forward failed", status);
        return 1;
    }

    if (bias_handle != 0) {
        if (!init_tensor_descriptor_4d(
                bias_desc, 1, static_cast<int>(out_channels), 1, 1)) {
            return 1;
        }
        const float add_alpha = 1.0f;
        const float add_beta = 1.0f;
        status = cudnnAddTensor(
            handle.handle,
            &add_alpha,
            bias_desc.desc,
            handle_to_ptr(bias_handle),
            &add_beta,
            output_desc.desc,
            handle_to_ptr(out_handle));
        if (status != CUDNN_STATUS_SUCCESS) {
            set_cudnn_error("cuDNN conv2d bias add failed", status);
            return 1;
        }
    }

    if (!check_cuda_launch("CUDA conv2d launch failed")) {
        return 1;
    }
    return 0;
#endif
}

template <typename InputT, typename WeightT, typename BiasT>
__global__ void conv2d_typed_forward_kernel(
    const InputT* input,
    float input_scale,
    const WeightT* weight,
    float weight_scale,
    const BiasT* bias,
    float bias_scale,
    bool has_bias,
    float* out,
    size_t total,
    size_t batch_size,
    size_t in_channels,
    size_t in_h,
    size_t in_w,
    size_t out_channels,
    size_t k_h,
    size_t k_w,
    size_t pad_h,
    size_t pad_w,
    size_t stride_h,
    size_t stride_w,
    size_t out_h,
    size_t out_w) {
    const size_t grid_stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += grid_stride) {
        size_t ow = idx % out_w;
        size_t oh = (idx / out_w) % out_h;
        size_t oc = (idx / (out_w * out_h)) % out_channels;
        size_t batch = idx / (out_channels * out_h * out_w);

        float acc = has_bias ? typed_value_to_float(bias, bias_scale, oc) : 0.0f;
        for (size_t ic = 0; ic < in_channels; ++ic) {
            for (size_t ky = 0; ky < k_h; ++ky) {
                const size_t padded_h = oh * stride_h + ky;
                if (padded_h < pad_h) {
                    continue;
                }
                const size_t ih = padded_h - pad_h;
                if (ih >= in_h) {
                    continue;
                }
                for (size_t kx = 0; kx < k_w; ++kx) {
                    const size_t padded_w = ow * stride_w + kx;
                    if (padded_w < pad_w) {
                        continue;
                    }
                    const size_t iw = padded_w - pad_w;
                    if (iw >= in_w) {
                        continue;
                    }
                    size_t input_idx =
                        ((batch * in_channels + ic) * in_h + ih) * in_w + iw;
                    size_t weight_idx = ((oc * in_channels + ic) * k_h + ky) * k_w + kx;
                    acc += typed_value_to_float(input, input_scale, input_idx) *
                           typed_value_to_float(weight, weight_scale, weight_idx);
                }
            }
        }
        out[idx] = acc;
    }
}

template <typename InputT, typename WeightT, typename BiasT>
int launch_conv2d_typed_bias(
    uint64_t input_handle,
    float input_scale,
    uint64_t weight_handle,
    float weight_scale,
    uint64_t bias_handle,
    float bias_scale,
    uint64_t out_handle,
    size_t batch_size,
    size_t in_channels,
    size_t in_h,
    size_t in_w,
    size_t out_channels,
    size_t k_h,
    size_t k_w,
    size_t pad_h,
    size_t pad_w,
    size_t stride_h,
    size_t stride_w,
    size_t out_h,
    size_t out_w) {
    size_t total = 0;
    if (!checked_product(
            "CUDA typed conv2d output length overflow",
            {batch_size, out_channels, out_h, out_w},
            &total)) {
        return 1;
    }
    constexpr int block_size = 256;
    const unsigned int grid_size = linear_grid_size(total, block_size);
    conv2d_typed_forward_kernel<InputT, WeightT, BiasT><<<grid_size, block_size>>>(
        reinterpret_cast<const InputT*>(handle_to_ptr(input_handle)),
        input_scale,
        reinterpret_cast<const WeightT*>(handle_to_ptr(weight_handle)),
        weight_scale,
        bias_handle == 0 ? nullptr : reinterpret_cast<const BiasT*>(handle_to_ptr(bias_handle)),
        bias_scale,
        bias_handle != 0,
        handle_to_ptr(out_handle),
        total,
        batch_size,
        in_channels,
        in_h,
        in_w,
        out_channels,
        k_h,
        k_w,
        pad_h,
        pad_w,
        stride_h,
        stride_w,
        out_h,
        out_w);
    return check_cuda_launch("CUDA typed conv2d forward kernel launch failed") ? 0 : 1;
}

template <typename InputT, typename WeightT>
int dispatch_conv2d_typed_bias(
    uint64_t input_handle,
    float input_scale,
    uint64_t weight_handle,
    float weight_scale,
    uint64_t bias_handle,
    int bias_dtype,
    float bias_scale,
    uint64_t out_handle,
    size_t batch_size,
    size_t in_channels,
    size_t in_h,
    size_t in_w,
    size_t out_channels,
    size_t k_h,
    size_t k_w,
    size_t pad_h,
    size_t pad_w,
    size_t stride_h,
    size_t stride_w,
    size_t out_h,
    size_t out_w) {
    if (bias_handle == 0) {
        return launch_conv2d_typed_bias<InputT, WeightT, float>(
            input_handle, input_scale, weight_handle, weight_scale, 0, 1.0f, out_handle,
            batch_size, in_channels, in_h, in_w, out_channels, k_h, k_w, pad_h, pad_w,
            stride_h, stride_w, out_h, out_w);
    }
    switch (bias_dtype) {
        case kDTypeF32:
            return launch_conv2d_typed_bias<InputT, WeightT, float>(
                input_handle, input_scale, weight_handle, weight_scale, bias_handle, bias_scale,
                out_handle, batch_size, in_channels, in_h, in_w, out_channels, k_h, k_w, pad_h,
                pad_w, stride_h, stride_w, out_h, out_w);
        case kDTypeF16:
            return launch_conv2d_typed_bias<InputT, WeightT, __half>(
                input_handle, input_scale, weight_handle, weight_scale, bias_handle, bias_scale,
                out_handle, batch_size, in_channels, in_h, in_w, out_channels, k_h, k_w, pad_h,
                pad_w, stride_h, stride_w, out_h, out_w);
        case kDTypeBF16:
            return launch_conv2d_typed_bias<InputT, WeightT, __nv_bfloat16>(
                input_handle, input_scale, weight_handle, weight_scale, bias_handle, bias_scale,
                out_handle, batch_size, in_channels, in_h, in_w, out_channels, k_h, k_w, pad_h,
                pad_w, stride_h, stride_w, out_h, out_w);
        case kDTypeI8:
            return launch_conv2d_typed_bias<InputT, WeightT, int8_t>(
                input_handle, input_scale, weight_handle, weight_scale, bias_handle, bias_scale,
                out_handle, batch_size, in_channels, in_h, in_w, out_channels, k_h, k_w, pad_h,
                pad_w, stride_h, stride_w, out_h, out_w);
        default:
            set_error("unsupported bias dtype for CUDA typed conv2d");
            return 1;
    }
}

template <typename InputT>
int dispatch_conv2d_typed_weight(
    uint64_t input_handle,
    float input_scale,
    uint64_t weight_handle,
    int weight_dtype,
    float weight_scale,
    uint64_t bias_handle,
    int bias_dtype,
    float bias_scale,
    uint64_t out_handle,
    size_t batch_size,
    size_t in_channels,
    size_t in_h,
    size_t in_w,
    size_t out_channels,
    size_t k_h,
    size_t k_w,
    size_t pad_h,
    size_t pad_w,
    size_t stride_h,
    size_t stride_w,
    size_t out_h,
    size_t out_w) {
    switch (weight_dtype) {
        case kDTypeF32:
            return dispatch_conv2d_typed_bias<InputT, float>(
                input_handle, input_scale, weight_handle, weight_scale, bias_handle, bias_dtype,
                bias_scale, out_handle, batch_size, in_channels, in_h, in_w, out_channels, k_h,
                k_w, pad_h, pad_w, stride_h, stride_w, out_h, out_w);
        case kDTypeF16:
            return dispatch_conv2d_typed_bias<InputT, __half>(
                input_handle, input_scale, weight_handle, weight_scale, bias_handle, bias_dtype,
                bias_scale, out_handle, batch_size, in_channels, in_h, in_w, out_channels, k_h,
                k_w, pad_h, pad_w, stride_h, stride_w, out_h, out_w);
        case kDTypeBF16:
            return dispatch_conv2d_typed_bias<InputT, __nv_bfloat16>(
                input_handle, input_scale, weight_handle, weight_scale, bias_handle, bias_dtype,
                bias_scale, out_handle, batch_size, in_channels, in_h, in_w, out_channels, k_h,
                k_w, pad_h, pad_w, stride_h, stride_w, out_h, out_w);
        case kDTypeI8:
            return dispatch_conv2d_typed_bias<InputT, int8_t>(
                input_handle, input_scale, weight_handle, weight_scale, bias_handle, bias_dtype,
                bias_scale, out_handle, batch_size, in_channels, in_h, in_w, out_channels, k_h,
                k_w, pad_h, pad_w, stride_h, stride_w, out_h, out_w);
        default:
            set_error("unsupported weight dtype for CUDA typed conv2d");
            return 1;
    }
}

extern "C" int lumen_cuda_conv2d_typed_device(
    uint64_t input_handle,
    int input_dtype,
    float input_scale,
    uint64_t weight_handle,
    int weight_dtype,
    float weight_scale,
    uint64_t bias_handle,
    int bias_dtype,
    float bias_scale,
    uint64_t out_handle,
    size_t batch_size,
    size_t in_channels,
    size_t in_h,
    size_t in_w,
    size_t out_channels,
    size_t k_h,
    size_t k_w,
    size_t pad_h,
    size_t pad_w,
    size_t stride_h,
    size_t stride_w,
    size_t out_h,
    size_t out_w) {
    if (!validate_handle(input_handle, "CUDA typed conv2d input handle") ||
        !validate_handle(weight_handle, "CUDA typed conv2d weight handle") ||
        !validate_handle(out_handle, "CUDA typed conv2d output handle")) {
        return 1;
    }
    if (bias_handle != 0 && !validate_handle(bias_handle, "CUDA typed conv2d bias handle")) {
        return 1;
    }
    if (batch_size == 0 || in_channels == 0 || in_h == 0 || in_w == 0 || out_channels == 0 ||
        k_h == 0 || k_w == 0 || stride_h == 0 || stride_w == 0 || out_h == 0 || out_w == 0) {
        set_error("CUDA typed conv2d dimensions must be greater than zero");
        return 1;
    }
    if ((input_dtype == kDTypeI8 && (!std::isfinite(input_scale) || input_scale <= 0.0f)) ||
        (weight_dtype == kDTypeI8 && (!std::isfinite(weight_scale) || weight_scale <= 0.0f)) ||
        (bias_handle != 0 && bias_dtype == kDTypeI8 &&
            (!std::isfinite(bias_scale) || bias_scale <= 0.0f))) {
        set_error("CUDA typed conv2d I8 scales must be finite and > 0");
        return 1;
    }

    switch (input_dtype) {
        case kDTypeF32:
            return dispatch_conv2d_typed_weight<float>(
                input_handle, input_scale, weight_handle, weight_dtype, weight_scale, bias_handle,
                bias_dtype, bias_scale, out_handle, batch_size, in_channels, in_h, in_w,
                out_channels, k_h, k_w, pad_h, pad_w, stride_h, stride_w, out_h, out_w);
        case kDTypeF16:
            return dispatch_conv2d_typed_weight<__half>(
                input_handle, input_scale, weight_handle, weight_dtype, weight_scale, bias_handle,
                bias_dtype, bias_scale, out_handle, batch_size, in_channels, in_h, in_w,
                out_channels, k_h, k_w, pad_h, pad_w, stride_h, stride_w, out_h, out_w);
        case kDTypeBF16:
            return dispatch_conv2d_typed_weight<__nv_bfloat16>(
                input_handle, input_scale, weight_handle, weight_dtype, weight_scale, bias_handle,
                bias_dtype, bias_scale, out_handle, batch_size, in_channels, in_h, in_w,
                out_channels, k_h, k_w, pad_h, pad_w, stride_h, stride_w, out_h, out_w);
        case kDTypeI8:
            return dispatch_conv2d_typed_weight<int8_t>(
                input_handle, input_scale, weight_handle, weight_dtype, weight_scale, bias_handle,
                bias_dtype, bias_scale, out_handle, batch_size, in_channels, in_h, in_w,
                out_channels, k_h, k_w, pad_h, pad_w, stride_h, stride_w, out_h, out_w);
        default:
            set_error("unsupported input dtype for CUDA typed conv2d");
            return 1;
    }
}

extern "C" int lumen_cuda_conv2d_backward_f32_device(
    uint64_t input_handle,
    uint64_t weight_handle,
    uint64_t grad_output_handle,
    uint64_t grad_input_handle,
    uint64_t grad_weight_handle,
    uint64_t grad_bias_handle,
    size_t batch_size,
    size_t in_channels,
    size_t in_h,
    size_t in_w,
    size_t out_channels,
    size_t k_h,
    size_t k_w,
    size_t pad_h,
    size_t pad_w,
    size_t stride_h,
    size_t stride_w,
    size_t out_h,
    size_t out_w) {
    if (!validate_handle(input_handle, "CUDA conv2d backward input handle") ||
        !validate_handle(weight_handle, "CUDA conv2d backward weight handle") ||
        !validate_handle(grad_output_handle, "CUDA conv2d backward grad output handle") ||
        !validate_handle(grad_input_handle, "CUDA conv2d backward grad input handle") ||
        !validate_handle(grad_weight_handle, "CUDA conv2d backward grad weight handle")) {
        return 1;
    }
    if (batch_size == 0 || in_channels == 0 || in_h == 0 || in_w == 0 || out_channels == 0 ||
        k_h == 0 || k_w == 0 || stride_h == 0 || stride_w == 0 || out_h == 0 || out_w == 0) {
        set_error("CUDA conv2d backward dimensions must be greater than zero");
        return 1;
    }
    if (!validate_int_dimensions(
            "CUDA conv2d backward dimensions exceed cuDNN int range",
            {batch_size, in_channels, in_h, in_w, out_channels, k_h, k_w, pad_h, pad_w, stride_h,
             stride_w, out_h, out_w})) {
        return 1;
    }

#if !LUMEN_HAS_CUDNN
    set_error("CUDA conv2d backward requires cuDNN support");
    return 1;
#else
    CudnnHandle handle;
    CudnnTensorDescriptor input_desc;
    CudnnTensorDescriptor grad_output_desc;
    CudnnTensorDescriptor bias_desc;
    CudnnFilterDescriptor filter_desc;
    CudnnConvolutionDescriptor conv_desc;
    if (!init_cudnn(handle) ||
        !init_tensor_descriptor_4d(
            input_desc,
            static_cast<int>(batch_size),
            static_cast<int>(in_channels),
            static_cast<int>(in_h),
            static_cast<int>(in_w)) ||
        !init_tensor_descriptor_4d(
            grad_output_desc,
            static_cast<int>(batch_size),
            static_cast<int>(out_channels),
            static_cast<int>(out_h),
            static_cast<int>(out_w)) ||
        !init_filter_descriptor_4d(
            filter_desc,
            static_cast<int>(out_channels),
            static_cast<int>(in_channels),
            static_cast<int>(k_h),
            static_cast<int>(k_w)) ||
        !init_convolution_descriptor_2d(
            conv_desc,
            static_cast<int>(pad_h),
            static_cast<int>(pad_w),
            static_cast<int>(stride_h),
            static_cast<int>(stride_w))) {
        return 1;
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    cudnnConvolutionBwdDataAlgo_t bwd_data_algo;
    cudnnConvolutionBwdFilterAlgo_t bwd_filter_algo;
    size_t bwd_data_workspace_bytes = 0;
    size_t bwd_filter_workspace_bytes = 0;
    if (!select_cudnn_bwd_data_algo(
            handle.handle,
            filter_desc.desc,
            grad_output_desc.desc,
            conv_desc.desc,
            input_desc.desc,
            bwd_data_algo,
            bwd_data_workspace_bytes) ||
        !select_cudnn_bwd_filter_algo(
            handle.handle,
            input_desc.desc,
            grad_output_desc.desc,
            conv_desc.desc,
            filter_desc.desc,
            bwd_filter_algo,
            bwd_filter_workspace_bytes)) {
        return 1;
    }
    size_t workspace_bytes =
        bwd_data_workspace_bytes > bwd_filter_workspace_bytes
            ? bwd_data_workspace_bytes
            : bwd_filter_workspace_bytes;
    thread_local ReusableCudaWorkspace workspace;
    if (!workspace.ensure(workspace_bytes, "failed to allocate cuDNN conv2d backward workspace")) {
        bwd_data_algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
        bwd_filter_algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
        bwd_data_workspace_bytes = 0;
        bwd_filter_workspace_bytes = 0;
    }
    cudnnStatus_t status = cudnnConvolutionBackwardData(
        handle.handle,
        &alpha,
        filter_desc.desc,
        handle_to_ptr(weight_handle),
        grad_output_desc.desc,
        handle_to_ptr(grad_output_handle),
        conv_desc.desc,
        bwd_data_algo,
        workspace.ptr,
        bwd_data_workspace_bytes,
        &beta,
        input_desc.desc,
        handle_to_ptr(grad_input_handle));
    if (status != CUDNN_STATUS_SUCCESS && bwd_data_algo != CUDNN_CONVOLUTION_BWD_DATA_ALGO_0) {
        bwd_data_workspace_bytes = 0;
        status = cudnnConvolutionBackwardData(
            handle.handle,
            &alpha,
            filter_desc.desc,
            handle_to_ptr(weight_handle),
            grad_output_desc.desc,
            handle_to_ptr(grad_output_handle),
            conv_desc.desc,
            CUDNN_CONVOLUTION_BWD_DATA_ALGO_0,
            nullptr,
            0,
            &beta,
            input_desc.desc,
            handle_to_ptr(grad_input_handle));
    }
    if (status != CUDNN_STATUS_SUCCESS) {
        set_cudnn_error("cuDNN conv2d backward data failed", status);
        return 1;
    }

    status = cudnnConvolutionBackwardFilter(
        handle.handle,
        &alpha,
        input_desc.desc,
        handle_to_ptr(input_handle),
        grad_output_desc.desc,
        handle_to_ptr(grad_output_handle),
        conv_desc.desc,
        bwd_filter_algo,
        workspace.ptr,
        bwd_filter_workspace_bytes,
        &beta,
        filter_desc.desc,
        handle_to_ptr(grad_weight_handle));
    if (status != CUDNN_STATUS_SUCCESS && bwd_filter_algo != CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0) {
        bwd_filter_workspace_bytes = 0;
        status = cudnnConvolutionBackwardFilter(
            handle.handle,
            &alpha,
            input_desc.desc,
            handle_to_ptr(input_handle),
            grad_output_desc.desc,
            handle_to_ptr(grad_output_handle),
            conv_desc.desc,
            CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,
            nullptr,
            0,
            &beta,
            filter_desc.desc,
            handle_to_ptr(grad_weight_handle));
    }
    if (status != CUDNN_STATUS_SUCCESS) {
        set_cudnn_error("cuDNN conv2d backward filter failed", status);
        return 1;
    }

    if (grad_bias_handle != 0) {
        if (!init_tensor_descriptor_4d(
                bias_desc, 1, static_cast<int>(out_channels), 1, 1)) {
            return 1;
        }
        status = cudnnConvolutionBackwardBias(
            handle.handle,
            &alpha,
            grad_output_desc.desc,
            handle_to_ptr(grad_output_handle),
            &beta,
            bias_desc.desc,
            handle_to_ptr(grad_bias_handle));
        if (status != CUDNN_STATUS_SUCCESS) {
            set_cudnn_error("cuDNN conv2d backward bias failed", status);
            return 1;
        }
    }

    if (!check_cuda_launch("CUDA conv2d backward launch failed")) {
        return 1;
    }
    return 0;
#endif
}

__global__ void max_pool2d_forward_kernel(
    const float* input,
    float* out,
    size_t total,
    size_t channels,
    size_t in_h,
    size_t in_w,
    size_t kernel_h,
    size_t kernel_w,
    size_t stride_h,
    size_t stride_w,
    size_t out_h,
    size_t out_w) {
    const size_t grid_stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += grid_stride) {
        size_t ow = idx % out_w;
        size_t oh = (idx / out_w) % out_h;
        size_t channel = (idx / (out_w * out_h)) % channels;
        size_t batch = idx / (channels * out_h * out_w);

        size_t h_start = oh * stride_h;
        size_t w_start = ow * stride_w;
        float max_val = -3.4028234663852886e+38F;
        for (size_t ky = 0; ky < kernel_h; ++ky) {
            for (size_t kx = 0; kx < kernel_w; ++kx) {
                size_t input_idx =
                    ((batch * channels + channel) * in_h + h_start + ky) * in_w + w_start + kx;
                float value = input[input_idx];
                if (value > max_val) {
                    max_val = value;
                }
            }
        }
        out[idx] = max_val;
    }
}

template <typename T>
__global__ void max_pool2d_forward_typed_kernel(
    const T* input,
    float input_scale,
    float* out,
    size_t total,
    size_t channels,
    size_t in_h,
    size_t in_w,
    size_t kernel_h,
    size_t kernel_w,
    size_t stride_h,
    size_t stride_w,
    size_t out_h,
    size_t out_w) {
    const size_t grid_stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += grid_stride) {
        size_t ow = idx % out_w;
        size_t oh = (idx / out_w) % out_h;
        size_t channel = (idx / (out_w * out_h)) % channels;
        size_t batch = idx / (channels * out_h * out_w);

        size_t h_start = oh * stride_h;
        size_t w_start = ow * stride_w;
        float max_val = -3.4028234663852886e+38F;
        for (size_t ky = 0; ky < kernel_h; ++ky) {
            for (size_t kx = 0; kx < kernel_w; ++kx) {
                size_t input_idx =
                    ((batch * channels + channel) * in_h + h_start + ky) * in_w + w_start + kx;
                float value = typed_value_to_float(input, input_scale, input_idx);
                if (value > max_val) {
                    max_val = value;
                }
            }
        }
        out[idx] = max_val;
    }
}

__global__ void max_pool2d_backward_kernel(
    const float* input,
    const float* grad_output,
    float* grad_input,
    size_t total,
    size_t channels,
    size_t in_h,
    size_t in_w,
    size_t kernel_h,
    size_t kernel_w,
    size_t stride_h,
    size_t stride_w,
    size_t out_h,
    size_t out_w) {
    const size_t grid_stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += grid_stride) {
        size_t ow = idx % out_w;
        size_t oh = (idx / out_w) % out_h;
        size_t channel = (idx / (out_w * out_h)) % channels;
        size_t batch = idx / (channels * out_h * out_w);

        size_t h_start = oh * stride_h;
        size_t w_start = ow * stride_w;
        float max_val = -3.4028234663852886e+38F;
        size_t max_idx = 0;
        for (size_t ky = 0; ky < kernel_h; ++ky) {
            for (size_t kx = 0; kx < kernel_w; ++kx) {
                size_t input_idx =
                    ((batch * channels + channel) * in_h + h_start + ky) * in_w + w_start + kx;
                float value = input[input_idx];
                if (value > max_val) {
                    max_val = value;
                    max_idx = input_idx;
                }
            }
        }
        atomicAdd(grad_input + max_idx, grad_output[idx]);
    }
}

template <typename T>
__global__ void max_pool2d_backward_typed_kernel(
    const T* input,
    float input_scale,
    const float* grad_output,
    float* grad_input,
    size_t total,
    size_t channels,
    size_t in_h,
    size_t in_w,
    size_t kernel_h,
    size_t kernel_w,
    size_t stride_h,
    size_t stride_w,
    size_t out_h,
    size_t out_w) {
    const size_t grid_stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += grid_stride) {
        size_t ow = idx % out_w;
        size_t oh = (idx / out_w) % out_h;
        size_t channel = (idx / (out_w * out_h)) % channels;
        size_t batch = idx / (channels * out_h * out_w);

        size_t h_start = oh * stride_h;
        size_t w_start = ow * stride_w;
        float max_val = -3.4028234663852886e+38F;
        size_t max_idx = 0;
        for (size_t ky = 0; ky < kernel_h; ++ky) {
            for (size_t kx = 0; kx < kernel_w; ++kx) {
                size_t input_idx =
                    ((batch * channels + channel) * in_h + h_start + ky) * in_w + w_start + kx;
                float value = typed_value_to_float(input, input_scale, input_idx);
                if (value > max_val) {
                    max_val = value;
                    max_idx = input_idx;
                }
            }
        }
        atomicAdd(grad_input + max_idx, grad_output[idx]);
    }
}

extern "C" int lumen_cuda_max_pool2d_f32_device(
    uint64_t input_handle,
    uint64_t out_handle,
    size_t batch_size,
    size_t channels,
    size_t in_h,
    size_t in_w,
    size_t kernel_h,
    size_t kernel_w,
    size_t stride_h,
    size_t stride_w,
    size_t out_h,
    size_t out_w) {
    if (!validate_handle(input_handle, "CUDA max_pool2d input handle") ||
        !validate_handle(out_handle, "CUDA max_pool2d output handle")) {
        return 1;
    }
    if (batch_size == 0 || channels == 0 || in_h == 0 || in_w == 0 || kernel_h == 0 ||
        kernel_w == 0 || stride_h == 0 || stride_w == 0 || out_h == 0 || out_w == 0) {
        set_error("CUDA max_pool2d dimensions must be greater than zero");
        return 1;
    }

    size_t total = 0;
    if (!checked_product(
            "CUDA max_pool2d output length overflow",
            {batch_size, channels, out_h, out_w},
            &total)) {
        return 1;
    }
    constexpr int block_size = 256;
    const unsigned int grid_size = linear_grid_size(total, block_size);
    max_pool2d_forward_kernel<<<grid_size, block_size>>>(
        handle_to_ptr(input_handle),
        handle_to_ptr(out_handle),
        total,
        channels,
        in_h,
        in_w,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        out_h,
        out_w);
    if (!check_cuda_launch("CUDA max_pool2d forward kernel launch failed")) {
        return 1;
    }
    return 0;
}

template <typename T>
int launch_max_pool2d_typed(
    uint64_t input_handle,
    float input_scale,
    uint64_t out_handle,
    size_t batch_size,
    size_t channels,
    size_t in_h,
    size_t in_w,
    size_t kernel_h,
    size_t kernel_w,
    size_t stride_h,
    size_t stride_w,
    size_t out_h,
    size_t out_w) {
    size_t total = 0;
    if (!checked_product(
            "CUDA typed max_pool2d output length overflow",
            {batch_size, channels, out_h, out_w},
            &total)) {
        return 1;
    }
    constexpr int block_size = 256;
    const unsigned int grid_size = linear_grid_size(total, block_size);
    max_pool2d_forward_typed_kernel<T><<<grid_size, block_size>>>(
        reinterpret_cast<const T*>(handle_to_ptr(input_handle)),
        input_scale,
        handle_to_ptr(out_handle),
        total,
        channels,
        in_h,
        in_w,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        out_h,
        out_w);
    return check_cuda_launch("CUDA typed max_pool2d forward kernel launch failed") ? 0 : 1;
}

extern "C" int lumen_cuda_max_pool2d_typed_device(
    uint64_t input_handle,
    int input_dtype,
    float input_scale,
    uint64_t out_handle,
    size_t batch_size,
    size_t channels,
    size_t in_h,
    size_t in_w,
    size_t kernel_h,
    size_t kernel_w,
    size_t stride_h,
    size_t stride_w,
    size_t out_h,
    size_t out_w) {
    if (!validate_handle(input_handle, "CUDA typed max_pool2d input handle") ||
        !validate_handle(out_handle, "CUDA typed max_pool2d output handle")) {
        return 1;
    }
    if (batch_size == 0 || channels == 0 || in_h == 0 || in_w == 0 || kernel_h == 0 ||
        kernel_w == 0 || stride_h == 0 || stride_w == 0 || out_h == 0 || out_w == 0) {
        set_error("CUDA typed max_pool2d dimensions must be greater than zero");
        return 1;
    }
    if (input_dtype == kDTypeI8 && (!std::isfinite(input_scale) || input_scale <= 0.0f)) {
        set_error("CUDA typed max_pool2d I8 scale must be finite and > 0");
        return 1;
    }

    switch (input_dtype) {
        case kDTypeF32:
            return launch_max_pool2d_typed<float>(
                input_handle,
                input_scale,
                out_handle,
                batch_size,
                channels,
                in_h,
                in_w,
                kernel_h,
                kernel_w,
                stride_h,
                stride_w,
                out_h,
                out_w);
        case kDTypeF16:
            return launch_max_pool2d_typed<__half>(
                input_handle,
                input_scale,
                out_handle,
                batch_size,
                channels,
                in_h,
                in_w,
                kernel_h,
                kernel_w,
                stride_h,
                stride_w,
                out_h,
                out_w);
        case kDTypeBF16:
            return launch_max_pool2d_typed<__nv_bfloat16>(
                input_handle,
                input_scale,
                out_handle,
                batch_size,
                channels,
                in_h,
                in_w,
                kernel_h,
                kernel_w,
                stride_h,
                stride_w,
                out_h,
                out_w);
        case kDTypeI8:
            return launch_max_pool2d_typed<int8_t>(
                input_handle,
                input_scale,
                out_handle,
                batch_size,
                channels,
                in_h,
                in_w,
                kernel_h,
                kernel_w,
                stride_h,
                stride_w,
                out_h,
                out_w);
        default:
            set_error("unsupported dtype for CUDA typed max_pool2d");
            return 1;
    }
}

extern "C" int lumen_cuda_max_pool2d_backward_f32_device(
    uint64_t input_handle,
    uint64_t grad_output_handle,
    uint64_t grad_input_handle,
    size_t batch_size,
    size_t channels,
    size_t in_h,
    size_t in_w,
    size_t kernel_h,
    size_t kernel_w,
    size_t stride_h,
    size_t stride_w,
    size_t out_h,
    size_t out_w) {
    if (!validate_handle(input_handle, "CUDA max_pool2d backward input handle") ||
        !validate_handle(grad_output_handle, "CUDA max_pool2d backward grad output handle") ||
        !validate_handle(grad_input_handle, "CUDA max_pool2d backward grad input handle")) {
        return 1;
    }
    if (batch_size == 0 || channels == 0 || in_h == 0 || in_w == 0 || kernel_h == 0 ||
        kernel_w == 0 || stride_h == 0 || stride_w == 0 || out_h == 0 || out_w == 0) {
        set_error("CUDA max_pool2d backward dimensions must be greater than zero");
        return 1;
    }

    size_t input_len = 0;
    size_t total = 0;
    if (!checked_product(
            "CUDA max_pool2d backward input length overflow",
            {batch_size, channels, in_h, in_w},
            &input_len) ||
        !checked_product(
            "CUDA max_pool2d backward output length overflow",
            {batch_size, channels, out_h, out_w},
            &total)) {
        return 1;
    }
    if (!zero_f32_buffer(
            handle_to_ptr(grad_input_handle),
            input_len,
            "CUDA max_pool2d backward grad input initialization failed")) {
        return 1;
    }

    constexpr int block_size = 256;
    const unsigned int grid_size = linear_grid_size(total, block_size);
    max_pool2d_backward_kernel<<<grid_size, block_size>>>(
        handle_to_ptr(input_handle),
        handle_to_ptr(grad_output_handle),
        handle_to_ptr(grad_input_handle),
        total,
        channels,
        in_h,
        in_w,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        out_h,
        out_w);
    if (!check_cuda_launch("CUDA max_pool2d backward kernel launch failed")) {
        return 1;
    }
    return 0;
}

template <typename T>
int launch_max_pool2d_backward_typed(
    uint64_t input_handle,
    float input_scale,
    uint64_t grad_output_handle,
    uint64_t grad_input_handle,
    size_t batch_size,
    size_t channels,
    size_t in_h,
    size_t in_w,
    size_t kernel_h,
    size_t kernel_w,
    size_t stride_h,
    size_t stride_w,
    size_t out_h,
    size_t out_w) {
    size_t input_total = 0;
    size_t total = 0;
    if (!checked_product(
            "CUDA typed max_pool2d backward input length overflow",
            {batch_size, channels, in_h, in_w},
            &input_total) ||
        !checked_product(
            "CUDA typed max_pool2d backward output length overflow",
            {batch_size, channels, out_h, out_w},
            &total)) {
        return 1;
    }
    if (!zero_f32_buffer(
            handle_to_ptr(grad_input_handle),
            input_total,
            "CUDA typed max_pool2d backward grad initialization failed")) {
        return 1;
    }
    constexpr int block_size = 256;
    const unsigned int grid_size = linear_grid_size(total, block_size);
    max_pool2d_backward_typed_kernel<T><<<grid_size, block_size>>>(
        reinterpret_cast<const T*>(handle_to_ptr(input_handle)),
        input_scale,
        handle_to_ptr(grad_output_handle),
        handle_to_ptr(grad_input_handle),
        total,
        channels,
        in_h,
        in_w,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        out_h,
        out_w);
    return check_cuda_launch("CUDA typed max_pool2d backward kernel launch failed") ? 0 : 1;
}

extern "C" int lumen_cuda_max_pool2d_backward_typed_device(
    uint64_t input_handle,
    int input_dtype,
    float input_scale,
    uint64_t grad_output_handle,
    uint64_t grad_input_handle,
    size_t batch_size,
    size_t channels,
    size_t in_h,
    size_t in_w,
    size_t kernel_h,
    size_t kernel_w,
    size_t stride_h,
    size_t stride_w,
    size_t out_h,
    size_t out_w) {
    if (!validate_handle(input_handle, "CUDA typed max_pool2d backward input handle") ||
        !validate_handle(grad_output_handle, "CUDA typed max_pool2d backward grad output handle") ||
        !validate_handle(grad_input_handle, "CUDA typed max_pool2d backward grad input handle")) {
        return 1;
    }
    if (batch_size == 0 || channels == 0 || in_h == 0 || in_w == 0 || kernel_h == 0 ||
        kernel_w == 0 || stride_h == 0 || stride_w == 0 || out_h == 0 || out_w == 0) {
        set_error("CUDA typed max_pool2d backward dimensions must be greater than zero");
        return 1;
    }
    if (input_dtype == kDTypeI8 && (!std::isfinite(input_scale) || input_scale <= 0.0f)) {
        set_error("CUDA typed max_pool2d backward I8 scale must be finite and > 0");
        return 1;
    }

    switch (input_dtype) {
        case kDTypeF32:
            return launch_max_pool2d_backward_typed<float>(
                input_handle,
                input_scale,
                grad_output_handle,
                grad_input_handle,
                batch_size,
                channels,
                in_h,
                in_w,
                kernel_h,
                kernel_w,
                stride_h,
                stride_w,
                out_h,
                out_w);
        case kDTypeF16:
            return launch_max_pool2d_backward_typed<__half>(
                input_handle,
                input_scale,
                grad_output_handle,
                grad_input_handle,
                batch_size,
                channels,
                in_h,
                in_w,
                kernel_h,
                kernel_w,
                stride_h,
                stride_w,
                out_h,
                out_w);
        case kDTypeBF16:
            return launch_max_pool2d_backward_typed<__nv_bfloat16>(
                input_handle,
                input_scale,
                grad_output_handle,
                grad_input_handle,
                batch_size,
                channels,
                in_h,
                in_w,
                kernel_h,
                kernel_w,
                stride_h,
                stride_w,
                out_h,
                out_w);
        case kDTypeI8:
            return launch_max_pool2d_backward_typed<int8_t>(
                input_handle,
                input_scale,
                grad_output_handle,
                grad_input_handle,
                batch_size,
                channels,
                in_h,
                in_w,
                kernel_h,
                kernel_w,
                stride_h,
                stride_w,
                out_h,
                out_w);
        default:
            set_error("unsupported dtype for CUDA typed max_pool2d backward");
            return 1;
    }
}
