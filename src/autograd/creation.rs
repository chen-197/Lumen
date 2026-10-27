use super::*;

impl Tensor {
    #[inline]
    pub(super) fn empty_tensor_data_for_shape(
        shape: &[usize],
        dtype: DType,
        requires_grad: bool,
        is_parameter: bool,
        i8_scale: Option<f32>,
    ) -> TensorData {
        let shape_dyn = IxDyn(shape);
        match dtype {
            DType::F32 => TensorData {
                data: ArrayD::<f32>::zeros(shape_dyn).into_shared(),
                f16_data: None,
                bf16_data: None,
                i8_data: None,
                cuda_f32_data: None,
                cuda_f16_data: None,
                cuda_bf16_data: None,
                cuda_i8_data: None,
                i8_scale: None,
                has_f32_data: true,
                storage_dtype: DType::F32,
                cache_dirty: false,
                is_parameter,
                grad: None,
                cuda_f32_grad: None,
                parents: Vec::new(),
                backward_op: None,
                requires_grad,
                device: Device::Cpu,
            },
            DType::F16 => TensorData {
                data: Self::empty_f32_storage(),
                f16_data: Some(
                    ArrayD::<f16>::from_elem(shape_dyn, f16::from_f32(0.0)).into_shared(),
                ),
                bf16_data: None,
                i8_data: None,
                cuda_f32_data: None,
                cuda_f16_data: None,
                cuda_bf16_data: None,
                cuda_i8_data: None,
                i8_scale: None,
                has_f32_data: false,
                storage_dtype: DType::F16,
                cache_dirty: false,
                is_parameter,
                grad: None,
                cuda_f32_grad: None,
                parents: Vec::new(),
                backward_op: None,
                requires_grad,
                device: Device::Cpu,
            },
            DType::BF16 => TensorData {
                data: Self::empty_f32_storage(),
                bf16_data: Some(
                    ArrayD::<bf16>::from_elem(shape_dyn, bf16::from_f32(0.0)).into_shared(),
                ),
                f16_data: None,
                i8_data: None,
                cuda_f32_data: None,
                cuda_f16_data: None,
                cuda_bf16_data: None,
                cuda_i8_data: None,
                i8_scale: None,
                has_f32_data: false,
                storage_dtype: DType::BF16,
                cache_dirty: false,
                is_parameter,
                grad: None,
                cuda_f32_grad: None,
                parents: Vec::new(),
                backward_op: None,
                requires_grad,
                device: Device::Cpu,
            },
            DType::I8 => TensorData {
                data: Self::empty_f32_storage(),
                f16_data: None,
                bf16_data: None,
                i8_data: Some(ArrayD::<i8>::zeros(shape_dyn).into_shared()),
                cuda_f32_data: None,
                cuda_f16_data: None,
                cuda_bf16_data: None,
                cuda_i8_data: None,
                i8_scale: Some(i8_scale.unwrap_or(1.0)),
                has_f32_data: false,
                storage_dtype: DType::I8,
                cache_dirty: false,
                is_parameter,
                grad: None,
                cuda_f32_grad: None,
                parents: Vec::new(),
                backward_op: None,
                requires_grad,
                device: Device::Cpu,
            },
        }
    }

    pub(super) fn build_tensor_data(
        dtype: DType,
        f32_data: ArrayD<f32>,
        requires_grad: bool,
        is_parameter: bool,
    ) -> TensorData {
        match dtype {
            DType::F32 => TensorData {
                data: f32_data.into_shared(),
                f16_data: None,
                bf16_data: None,
                i8_data: None,
                cuda_f32_data: None,
                cuda_f16_data: None,
                cuda_bf16_data: None,
                cuda_i8_data: None,
                i8_scale: None,
                has_f32_data: true,
                storage_dtype: DType::F32,
                cache_dirty: false,
                is_parameter,
                grad: None,
                cuda_f32_grad: None,
                parents: Vec::new(),
                backward_op: None,
                requires_grad,
                device: Device::Cpu,
            },
            DType::F16 => TensorData {
                data: Self::empty_f32_storage(),
                f16_data: Some(Self::f32_array_to_f16(&f32_data)),
                bf16_data: None,
                i8_data: None,
                cuda_f32_data: None,
                cuda_f16_data: None,
                cuda_bf16_data: None,
                cuda_i8_data: None,
                i8_scale: None,
                has_f32_data: false,
                storage_dtype: DType::F16,
                cache_dirty: false,
                is_parameter,
                grad: None,
                cuda_f32_grad: None,
                parents: Vec::new(),
                backward_op: None,
                requires_grad,
                device: Device::Cpu,
            },
            DType::BF16 => TensorData {
                data: Self::empty_f32_storage(),
                f16_data: None,
                bf16_data: Some(Self::f32_array_to_bf16(&f32_data)),
                i8_data: None,
                cuda_f32_data: None,
                cuda_f16_data: None,
                cuda_bf16_data: None,
                cuda_i8_data: None,
                i8_scale: None,
                has_f32_data: false,
                storage_dtype: DType::BF16,
                cache_dirty: false,
                is_parameter,
                grad: None,
                cuda_f32_grad: None,
                parents: Vec::new(),
                backward_op: None,
                requires_grad,
                device: Device::Cpu,
            },
            DType::I8 => {
                let (i8_data, scale) = Self::quantize_f32_to_dtype(&f32_data, DType::I8, None);
                TensorData {
                    data: Self::empty_f32_storage(),
                    f16_data: None,
                    bf16_data: None,
                    i8_data: Some(i8_data),
                    cuda_f32_data: None,
                    cuda_f16_data: None,
                    cuda_bf16_data: None,
                    cuda_i8_data: None,
                    i8_scale: Some(scale),
                    has_f32_data: false,
                    storage_dtype: DType::I8,
                    cache_dirty: false,
                    is_parameter,
                    grad: None,
                    cuda_f32_grad: None,
                    parents: Vec::new(),
                    backward_op: None,
                    requires_grad,
                    device: Device::Cpu,
                }
            }
        }
    }

    pub fn op_data(
        data: ArcArray<f32, IxDyn>,
        parents: Vec<Tensor>,
        backward_op: Option<BackwardOp>,
        requires_grad: bool,
    ) -> TensorData {
        TensorData {
            data,
            f16_data: None,
            bf16_data: None,
            i8_data: None,
            cuda_f32_data: None,
            cuda_f16_data: None,
            cuda_bf16_data: None,
            cuda_i8_data: None,
            i8_scale: None,
            has_f32_data: true,
            storage_dtype: DType::F32,
            cache_dirty: false,
            is_parameter: false,
            grad: None,
            cuda_f32_grad: None,
            parents,
            backward_op,
            requires_grad,
            device: Device::Cpu,
        }
    }

    // 默认构造叶子张量：
    // - 推理模式/no_grad 下：requires_grad=false
    // - 否则：requires_grad=true（更适合训练时手工造张量）
    pub fn new(data: ArrayD<f32>) -> Self {
        let req = !is_no_grad();
        Tensor(Rc::new(RefCell::new(Self::build_tensor_data(
            DType::F32,
            data,
            req,
            false,
        ))))
    }

    pub fn new_with_dtype(data: ArrayD<f32>, dtype: DType) -> Self {
        let req = !is_no_grad();
        Tensor(Rc::new(RefCell::new(Self::build_tensor_data(
            dtype, data, req, false,
        ))))
    }

    // 创建叶子张量（显式指定 requires_grad）
    pub fn from_data_with_grad_flag(data: ArrayD<f32>, requires_grad: bool) -> Tensor {
        Tensor(Rc::new(RefCell::new(Self::build_tensor_data(
            DType::F32,
            data,
            requires_grad,
            false,
        ))))
    }

    // 创建叶子张量：根据 is_no_grad() 自动决定 requires_grad
    pub fn from_data(data: ArrayD<f32>) -> Tensor {
        let req = !is_no_grad();
        Tensor::from_data_with_grad_flag(data, req)
    }

    // 推理/常量：不需要梯度
    pub fn from_data_no_grad(data: ArcArray<f32, IxDyn>) -> Tensor {
        Tensor(Rc::new(RefCell::new(TensorData {
            data,
            f16_data: None,
            bf16_data: None,
            i8_data: None,
            cuda_f32_data: None,
            cuda_f16_data: None,
            cuda_bf16_data: None,
            cuda_i8_data: None,
            i8_scale: None,
            has_f32_data: true,
            storage_dtype: DType::F32,
            cache_dirty: false,
            is_parameter: false,
            grad: None,
            cuda_f32_grad: None,
            parents: vec![],
            backward_op: None,
            requires_grad: false,
            device: Device::Cpu,
        })))
    }

    pub(crate) fn from_f16_data_no_grad(data: ArcArray<f16, IxDyn>) -> Tensor {
        Tensor(Rc::new(RefCell::new(TensorData {
            data: Self::empty_f32_storage(),
            f16_data: Some(data),
            bf16_data: None,
            i8_data: None,
            cuda_f32_data: None,
            cuda_f16_data: None,
            cuda_bf16_data: None,
            cuda_i8_data: None,
            i8_scale: None,
            has_f32_data: false,
            storage_dtype: DType::F16,
            cache_dirty: false,
            is_parameter: false,
            grad: None,
            cuda_f32_grad: None,
            parents: vec![],
            backward_op: None,
            requires_grad: false,
            device: Device::Cpu,
        })))
    }

    pub(crate) fn from_bf16_data_no_grad(data: ArcArray<bf16, IxDyn>) -> Tensor {
        Tensor(Rc::new(RefCell::new(TensorData {
            data: Self::empty_f32_storage(),
            f16_data: None,
            bf16_data: Some(data),
            i8_data: None,
            cuda_f32_data: None,
            cuda_f16_data: None,
            cuda_bf16_data: None,
            cuda_i8_data: None,
            i8_scale: None,
            has_f32_data: false,
            storage_dtype: DType::BF16,
            cache_dirty: false,
            is_parameter: false,
            grad: None,
            cuda_f32_grad: None,
            parents: vec![],
            backward_op: None,
            requires_grad: false,
            device: Device::Cpu,
        })))
    }

    pub(crate) fn from_i8_data_no_grad(data: ArcArray<i8, IxDyn>, scale: f32) -> Tensor {
        Tensor(Rc::new(RefCell::new(TensorData {
            data: Self::empty_f32_storage(),
            f16_data: None,
            bf16_data: None,
            i8_data: Some(data),
            cuda_f32_data: None,
            cuda_f16_data: None,
            cuda_bf16_data: None,
            cuda_i8_data: None,
            i8_scale: Some(scale),
            has_f32_data: false,
            storage_dtype: DType::I8,
            cache_dirty: false,
            is_parameter: false,
            grad: None,
            cuda_f32_grad: None,
            parents: vec![],
            backward_op: None,
            requires_grad: false,
            device: Device::Cpu,
        })))
    }

    pub(crate) fn from_f32_data_no_grad_with_dtype(data: ArrayD<f32>, dtype: DType) -> Tensor {
        match dtype {
            DType::F32 => Tensor::from_array_no_grad(data),
            DType::F16 => Tensor::from_f16_data_no_grad(data.mapv(f16::from_f32).into_shared()),
            DType::BF16 => Tensor::from_bf16_data_no_grad(data.mapv(bf16::from_f32).into_shared()),
            DType::I8 => {
                let (i8_data, scale) = Self::quantize_f32_to_dtype(&data, DType::I8, None);
                Tensor::from_i8_data_no_grad(i8_data, scale)
            }
        }
    }

    pub(crate) fn from_f32_data_no_grad_with_device_dtype(
        data: ArrayD<f32>,
        dtype: DType,
        device: Device,
    ) -> Tensor {
        Self::from_f32_data_no_grad_with_device_dtype_and_cuda_buffer(data, dtype, device, None)
    }

    pub(crate) fn from_f32_data_no_grad_with_device_dtype_and_cuda_buffer(
        data: ArrayD<f32>,
        dtype: DType,
        device: Device,
        cuda_f32_data: Option<crate::ops::cuda::CudaBuffer>,
    ) -> Tensor {
        let tensor = Self::from_f32_data_no_grad_with_dtype(data, dtype);
        match (device, cuda_f32_data) {
            (Device::Cpu, _) => tensor.to_cpu_inplace(),
            (Device::Cuda, Some(buffer)) => tensor.set_cuda_f32_buffer_inplace(buffer),
            (Device::Cuda, None) => tensor.to_cuda_inplace(),
        }
        tensor
    }

    pub(crate) fn from_cuda_f32_buffer_no_host(
        shape: &[usize],
        buffer: crate::ops::cuda::CudaBuffer,
        device: Device,
    ) -> Tensor {
        assert_eq!(
            device,
            Device::Cuda,
            "from_cuda_f32_buffer_no_host currently expects CUDA device"
        );
        let len = Self::checked_shape_len(shape, "CUDA f32 tensor");
        assert_eq!(
            buffer.len(),
            len,
            "CUDA f32 buffer length does not match tensor shape"
        );
        Tensor(Rc::new(RefCell::new(TensorData {
            data: ArrayD::<f32>::zeros(IxDyn(shape)).into_shared(),
            f16_data: None,
            bf16_data: None,
            i8_data: None,
            cuda_f32_data: Some(buffer),
            cuda_f16_data: None,
            cuda_bf16_data: None,
            cuda_i8_data: None,
            i8_scale: None,
            has_f32_data: false,
            storage_dtype: DType::F32,
            cache_dirty: false,
            is_parameter: false,
            grad: None,
            cuda_f32_grad: None,
            parents: vec![],
            backward_op: None,
            requires_grad: false,
            device,
        })))
    }

    pub(crate) fn from_cuda_f32_buffer_no_host_with_dtype(
        shape: &[usize],
        buffer: crate::ops::cuda::CudaBuffer,
        device: Device,
        dtype: DType,
    ) -> Tensor {
        if dtype == DType::F32 {
            return Self::from_cuda_f32_buffer_no_host(shape, buffer, device);
        }
        assert_eq!(
            device,
            Device::Cuda,
            "from_cuda_f32_buffer_no_host_with_dtype currently expects CUDA device"
        );
        let len = Self::checked_shape_len(shape, "CUDA typed tensor");
        assert_eq!(
            buffer.len(),
            len,
            "CUDA f32 buffer length does not match typed tensor shape"
        );
        let (cuda_f16_data, cuda_bf16_data, cuda_i8_data, i8_scale) = match dtype {
            DType::F16 => (
                Some(
                    crate::ops::cuda::f32_to_lowp_storage_no_host(&buffer, DType::F16)
                        .unwrap_or_else(|err| {
                            panic!("Failed to convert CUDA f32 buffer to F16 storage: {err}")
                        }),
                ),
                None,
                None,
                None,
            ),
            DType::BF16 => (
                None,
                Some(
                    crate::ops::cuda::f32_to_lowp_storage_no_host(&buffer, DType::BF16)
                        .unwrap_or_else(|err| {
                            panic!("Failed to convert CUDA f32 buffer to BF16 storage: {err}")
                        }),
                ),
                None,
                None,
            ),
            DType::I8 => {
                let (i8_buffer, scale) =
                    crate::ops::cuda::quantize_f32_to_i8_dynamic_no_host(&buffer).unwrap_or_else(
                        |err| panic!("Failed to quantize CUDA f32 buffer to I8 storage: {err}"),
                    );
                (None, None, Some(i8_buffer), Some(scale))
            }
            DType::F32 => unreachable!("F32 dtype returned earlier"),
        };
        Tensor(Rc::new(RefCell::new(TensorData {
            data: ArrayD::<f32>::zeros(IxDyn(shape)).into_shared(),
            f16_data: None,
            bf16_data: None,
            i8_data: None,
            cuda_f32_data: None,
            cuda_f16_data,
            cuda_bf16_data,
            cuda_i8_data,
            i8_scale,
            has_f32_data: false,
            storage_dtype: dtype,
            cache_dirty: false,
            is_parameter: false,
            grad: None,
            cuda_f32_grad: None,
            parents: vec![],
            backward_op: None,
            requires_grad: false,
            device,
        })))
    }

    pub(crate) fn from_cuda_native_lowp_buffer_no_host_with_dtype(
        shape: &[usize],
        buffer: crate::ops::cuda::CudaBuffer,
        device: Device,
        dtype: DType,
        i8_scale: Option<f32>,
    ) -> Tensor {
        assert_eq!(
            device,
            Device::Cuda,
            "from_cuda_native_lowp_buffer_no_host_with_dtype expects CUDA device"
        );
        assert!(
            dtype != DType::F32,
            "from_cuda_native_lowp_buffer_no_host_with_dtype expects a low precision dtype"
        );
        if dtype == DType::I8 {
            assert!(
                i8_scale.is_some_and(|scale| scale.is_finite() && scale > 0.0),
                "I8 CUDA native tensor requires a finite positive quantization scale"
            );
        }
        let len = Self::checked_shape_len(shape, "CUDA native low precision tensor");
        assert_eq!(
            buffer.len(),
            len,
            "CUDA native low precision buffer length does not match tensor shape"
        );

        let (cuda_f16_data, cuda_bf16_data, cuda_i8_data) = match dtype {
            DType::F16 => (Some(buffer), None, None),
            DType::BF16 => (None, Some(buffer), None),
            DType::I8 => (None, None, Some(buffer)),
            DType::F32 => unreachable!("checked above"),
        };
        Tensor(Rc::new(RefCell::new(TensorData {
            data: ArrayD::<f32>::zeros(IxDyn(shape)).into_shared(),
            f16_data: None,
            bf16_data: None,
            i8_data: None,
            cuda_f32_data: None,
            cuda_f16_data,
            cuda_bf16_data,
            cuda_i8_data,
            i8_scale,
            has_f32_data: false,
            storage_dtype: dtype,
            cache_dirty: false,
            is_parameter: false,
            grad: None,
            cuda_f32_grad: None,
            parents: vec![],
            backward_op: None,
            requires_grad: false,
            device,
        })))
    }

    pub(crate) fn from_shared_f32_no_grad_with_device(
        data: ArcArray<f32, IxDyn>,
        device: Device,
    ) -> Tensor {
        let tensor = Self::from_data_no_grad(data);
        tensor.to_device_inplace(device);
        tensor
    }

    pub(crate) fn from_shared_f16_no_grad_with_device(
        data: ArcArray<f16, IxDyn>,
        device: Device,
    ) -> Tensor {
        let tensor = Self::from_f16_data_no_grad(data);
        tensor.to_device_inplace(device);
        tensor
    }

    pub(crate) fn from_shared_bf16_no_grad_with_device(
        data: ArcArray<bf16, IxDyn>,
        device: Device,
    ) -> Tensor {
        let tensor = Self::from_bf16_data_no_grad(data);
        tensor.to_device_inplace(device);
        tensor
    }

    pub(crate) fn from_shared_i8_no_grad_with_device(
        data: ArcArray<i8, IxDyn>,
        scale: f32,
        device: Device,
    ) -> Tensor {
        let tensor = Self::from_i8_data_no_grad(data, scale);
        tensor.to_device_inplace(device);
        tensor
    }

    // 兼容旧接口：传入 ArrayD 作为常量
    pub fn from_array_no_grad(data: ArrayD<f32>) -> Tensor {
        Tensor::from_data_no_grad(data.into_shared())
    }

    // 训练参数：需要梯度（叶子）
    pub fn parameter(data: ArrayD<f32>) -> Tensor {
        let quantization = default_parameter_quantization();
        if quantization.is_enabled() {
            return Self::parameter_with_quantization(data, quantization);
        }

        let dtype = default_parameter_dtype();
        Self::parameter_with_dtype(data, dtype)
    }

    // 训练参数：显式指定 dtype，优先级高于全局默认参数 dtype
    pub fn parameter_with_dtype(data: ArrayD<f32>, dtype: DType) -> Tensor {
        Tensor(Rc::new(RefCell::new(Self::build_tensor_data(
            dtype, data, true, true,
        ))))
    }

    pub fn parameter_placeholder_with_dtype(shape: &[usize], dtype: DType) -> Tensor {
        Tensor(Rc::new(RefCell::new(Self::empty_tensor_data_for_shape(
            shape, dtype, true, true, None,
        ))))
    }

    pub fn parameter_with_quantization(
        data: ArrayD<f32>,
        quantization: ParameterQuantization,
    ) -> Tensor {
        if !quantization.is_enabled() {
            return Self::parameter_with_dtype(data, default_parameter_dtype());
        }

        let dtype = quantization
            .storage_dtype()
            .expect("enabled quantization must provide storage dtype");
        match dtype {
            DType::I8 => {
                let (i8_data, scale) =
                    Self::quantize_f32_to_dtype(&data, dtype, quantization.scale_override());
                Tensor(Rc::new(RefCell::new(TensorData {
                    data: Self::empty_f32_storage(),
                    f16_data: None,
                    bf16_data: None,
                    i8_data: Some(i8_data),
                    cuda_f32_data: None,
                    cuda_f16_data: None,
                    cuda_bf16_data: None,
                    cuda_i8_data: None,
                    i8_scale: Some(scale),
                    has_f32_data: false,
                    storage_dtype: dtype,
                    cache_dirty: false,
                    is_parameter: true,
                    grad: None,
                    cuda_f32_grad: None,
                    parents: Vec::new(),
                    backward_op: None,
                    requires_grad: true,
                    device: Device::Cpu,
                })))
            }
            other => {
                panic!(
                    "quantized storage dtype {:?} is not implemented yet; currently only I8 is supported",
                    other
                );
            }
        }
    }

    pub fn parameter_placeholder_with_quantization(
        shape: &[usize],
        quantization: ParameterQuantization,
    ) -> Tensor {
        if !quantization.is_enabled() {
            return Self::parameter_placeholder_with_dtype(shape, default_parameter_dtype());
        }

        let dtype = quantization
            .storage_dtype()
            .expect("enabled quantization must provide storage dtype");
        Tensor(Rc::new(RefCell::new(Self::empty_tensor_data_for_shape(
            shape,
            dtype,
            true,
            true,
            quantization.scale_override(),
        ))))
    }
}
