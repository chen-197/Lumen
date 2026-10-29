use super::*;

impl Tensor {
    pub(super) fn ensure_cuda_f32_data(&self) {
        assert!(
            self.device() == Device::Cuda,
            "ensure_cuda_f32_data expects a CUDA tensor"
        );
        if self.0.borrow().cuda_f32_data.is_some() {
            return;
        }
        if self.is_empty() {
            let mut inner = self.0.borrow_mut();
            inner.device = Device::Cuda;
            inner.cuda_f32_data = None;
            return;
        }

        let native_cuda_source = {
            let inner = self.0.borrow();
            if !inner.has_f32_data && inner.cuda_f32_data.is_none() {
                match inner.storage_dtype {
                    DType::F16 if inner.f16_data.is_none() => inner
                        .cuda_f16_data
                        .clone()
                        .map(|buffer| (DType::F16, buffer)),
                    DType::BF16 if inner.bf16_data.is_none() => inner
                        .cuda_bf16_data
                        .clone()
                        .map(|buffer| (DType::BF16, buffer)),
                    DType::I8 if inner.i8_data.is_none() => {
                        inner.cuda_i8_data.clone().map(|buffer| (DType::I8, buffer))
                    }
                    _ => None,
                }
            } else {
                None
            }
        };
        if let Some((dtype, native_buffer)) = native_cuda_source {
            let shape = self.shape_vec();
            let host_data = match dtype {
                DType::F16 => crate::ops::cuda::download_u16_storage(&native_buffer)
                    .unwrap_or_else(|err| {
                        panic!("Failed to download CUDA F16 tensor for f32 upload: {}", err)
                    })
                    .into_iter()
                    .map(|bits| f16::from_bits(bits).to_f32())
                    .collect::<Vec<_>>(),
                DType::BF16 => crate::ops::cuda::download_u16_storage(&native_buffer)
                    .unwrap_or_else(|err| {
                        panic!(
                            "Failed to download CUDA BF16 tensor for f32 upload: {}",
                            err
                        )
                    })
                    .into_iter()
                    .map(|bits| bf16::from_bits(bits).to_f32())
                    .collect::<Vec<_>>(),
                DType::I8 => {
                    let scale = self
                        .0
                        .borrow()
                        .i8_scale
                        .expect("I8 tensor missing quantization scale");
                    crate::ops::cuda::download_i8_storage(&native_buffer)
                        .unwrap_or_else(|err| {
                            panic!("Failed to download CUDA I8 tensor for f32 upload: {}", err)
                        })
                        .into_iter()
                        .map(|value| (value as f32) * scale)
                        .collect::<Vec<_>>()
                }
                DType::F32 => unreachable!("native CUDA f32 upload excludes f32"),
            };
            let buffer = crate::ops::cuda::upload_f32(&host_data)
                .unwrap_or_else(|err| panic!("Failed to upload tensor to CUDA: {}", err));
            let mut inner = self.0.borrow_mut();
            inner.data = Array::from_shape_vec(IxDyn(&shape), host_data)
                .expect("Failed to materialize host f32 storage from CUDA native buffer")
                .into_shared();
            inner.has_f32_data = true;
            inner.cuda_f32_data = Some(buffer);
            return;
        }

        let host_data = self.storage_as_f32_vec();
        let buffer = crate::ops::cuda::upload_f32(&host_data)
            .unwrap_or_else(|err| panic!("Failed to upload tensor to CUDA: {}", err));

        let mut inner = self.0.borrow_mut();
        inner.device = Device::Cuda;
        inner.cuda_f32_data = Some(buffer);
    }

    pub(super) fn ensure_cuda_native_lowp_data(&self) {
        assert!(
            self.device() == Device::Cuda,
            "ensure_cuda_native_lowp_data expects a CUDA tensor"
        );
        if self.is_empty() {
            return;
        }

        let should_upload = {
            let inner = self.0.borrow();
            if inner.cache_dirty {
                return;
            }
            match inner.storage_dtype {
                DType::F16 => inner.cuda_f16_data.is_none() && inner.f16_data.is_some(),
                DType::BF16 => inner.cuda_bf16_data.is_none() && inner.bf16_data.is_some(),
                DType::I8 => inner.cuda_i8_data.is_none() && inner.i8_data.is_some(),
                DType::F32 => false,
            }
        };
        if !should_upload {
            return;
        }

        let owned = self.native_storage_owned();
        let mut inner = self.0.borrow_mut();
        match owned {
            TensorStorageOwned::F16(data) if inner.storage_dtype == DType::F16 => {
                let data = data.as_standard_layout();
                if let Some(slice) = data.as_slice() {
                    let bits = slice.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
                    inner.cuda_f16_data = Some(
                        crate::ops::cuda::upload_u16_storage(&bits).unwrap_or_else(|err| {
                            panic!(
                                "Failed to upload F16 tensor to CUDA native storage: {}",
                                err
                            )
                        }),
                    );
                }
            }
            TensorStorageOwned::BF16(data) if inner.storage_dtype == DType::BF16 => {
                let data = data.as_standard_layout();
                if let Some(slice) = data.as_slice() {
                    let bits = slice.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
                    inner.cuda_bf16_data = Some(
                        crate::ops::cuda::upload_u16_storage(&bits).unwrap_or_else(|err| {
                            panic!(
                                "Failed to upload BF16 tensor to CUDA native storage: {}",
                                err
                            )
                        }),
                    );
                }
            }
            TensorStorageOwned::I8(data, _) if inner.storage_dtype == DType::I8 => {
                let data = data.as_standard_layout();
                if let Some(slice) = data.as_slice() {
                    inner.cuda_i8_data = Some(
                        crate::ops::cuda::upload_i8_storage(slice).unwrap_or_else(|err| {
                            panic!("Failed to upload I8 tensor to CUDA native storage: {}", err)
                        }),
                    );
                }
            }
            _ => {}
        }
    }

    pub(crate) fn cloned_cuda_native_lowp_buffer(
        &self,
    ) -> Option<(DType, crate::ops::cuda::CudaBuffer, Option<f32>)> {
        if self.device() != Device::Cuda {
            return None;
        }
        {
            let inner = self.0.borrow();
            if inner.cache_dirty {
                return None;
            }
            match inner.storage_dtype {
                DType::F16 => {
                    if let Some(buffer) = inner.cuda_f16_data.clone() {
                        return Some((DType::F16, buffer, None));
                    }
                }
                DType::BF16 => {
                    if let Some(buffer) = inner.cuda_bf16_data.clone() {
                        return Some((DType::BF16, buffer, None));
                    }
                }
                DType::I8 => {
                    if let Some(buffer) = inner.cuda_i8_data.clone() {
                        return Some((DType::I8, buffer, inner.i8_scale));
                    }
                }
                DType::F32 => return None,
            }
        }

        self.ensure_cuda_native_lowp_data();
        let inner = self.0.borrow();
        match inner.storage_dtype {
            DType::F16 => inner
                .cuda_f16_data
                .clone()
                .map(|buffer| (DType::F16, buffer, None)),
            DType::BF16 => inner
                .cuda_bf16_data
                .clone()
                .map(|buffer| (DType::BF16, buffer, None)),
            DType::I8 => inner
                .cuda_i8_data
                .clone()
                .map(|buffer| (DType::I8, buffer, inner.i8_scale)),
            DType::F32 => None,
        }
    }

    pub(crate) fn with_cuda_f32_buffer<R>(
        &self,
        f: impl FnOnce(&crate::ops::cuda::CudaBuffer) -> R,
    ) -> R {
        self.ensure_cuda_f32_data();
        let inner = self.0.borrow();
        let buffer = inner
            .cuda_f32_data
            .as_ref()
            .expect("CUDA tensor missing resident f32 buffer");
        f(buffer)
    }

    pub(crate) fn cloned_cuda_f32_buffer(&self) -> Option<crate::ops::cuda::CudaBuffer> {
        if self.device() != Device::Cuda {
            return None;
        }
        self.ensure_cuda_f32_data();
        self.0.borrow().cuda_f32_data.clone()
    }

    /// Invalidate host/native mirrors after a kernel writes the resident F32
    /// buffer. Logical shape/dtype are retained; snapshots rebuild native values
    /// lazily. This does not allocate or copy the full tensor on every update.
    pub(crate) fn mark_cuda_f32_buffer_modified(&self) {
        let mut inner = self.0.borrow_mut();
        assert_eq!(inner.device, Device::Cuda);
        assert!(
            inner.cuda_f32_data.is_some(),
            "Missing modified CUDA F32 buffer"
        );
        inner.has_f32_data = false;
        // Keep stale host arrays for their logical shape. cache_dirty makes
        // native views rebuild them from the authoritative device buffer.
        inner.cuda_f16_data = None;
        inner.cuda_bf16_data = None;
        inner.cuda_i8_data = None;
        inner.cache_dirty = inner.storage_dtype != DType::F32;
        inner.grad = None;
        inner.cuda_f32_grad = None;
    }

    pub(crate) fn set_cuda_f32_buffer_inplace(&self, buffer: crate::ops::cuda::CudaBuffer) {
        let mut inner = self.0.borrow_mut();
        inner.device = Device::Cuda;
        inner.cuda_f32_data = Some(buffer);
        inner.cuda_f16_data = None;
        inner.cuda_bf16_data = None;
        inner.cuda_i8_data = None;
    }

    pub(crate) fn replace_cuda_floatlike_state_buffer_no_host_sync(
        &self,
        buffer: crate::ops::cuda::CudaBuffer,
    ) {
        let shape = self.shape_vec();
        assert_eq!(
            buffer.len(),
            Self::checked_shape_len(&shape, "CUDA optimizer state")
        );
        let mut inner = self.0.borrow_mut();
        let dtype = inner.storage_dtype;
        assert!(matches!(dtype, DType::F16 | DType::BF16));
        assert_eq!(inner.device, Device::Cuda);
        if inner.has_f32_data {
            inner.data = ArrayD::<f32>::zeros(IxDyn(&shape)).into_shared();
        }
        Self::clear_non_f32_storage(&mut inner);
        Self::clear_cuda_storage(&mut inner);
        match dtype {
            DType::F16 => inner.cuda_f16_data = Some(buffer),
            DType::BF16 => inner.cuda_bf16_data = Some(buffer),
            _ => unreachable!(),
        }
        inner.has_f32_data = false;
        inner.cache_dirty = false;
        inner.grad = None;
        inner.cuda_f32_grad = None;
    }

    pub(crate) fn replace_cuda_i8_and_f32_buffers_no_host_sync(
        &self,
        buffer: crate::ops::cuda::CudaBuffer,
        rounded_f32: crate::ops::cuda::CudaBuffer,
        scale: f32,
    ) {
        assert_eq!(
            self.dtype(),
            DType::I8,
            "I8 buffer replacement expects I8 storage"
        );
        assert!(scale.is_finite() && scale > 0.0, "invalid I8 scale");
        let shape = self.shape_vec();
        assert_eq!(
            buffer.len(),
            Self::checked_shape_len(&shape, "CUDA I8 replacement")
        );
        assert_eq!(
            rounded_f32.len(),
            buffer.len(),
            "CUDA I8/F32 shape mismatch"
        );
        let mut inner = self.0.borrow_mut();
        Self::clear_non_f32_storage(&mut inner);
        Self::clear_cuda_storage(&mut inner);
        // The placeholder retains logical shape without exposing stale host values.
        inner.data = ArrayD::<f32>::zeros(IxDyn(&shape)).into_shared();
        inner.has_f32_data = false;
        inner.cache_dirty = false;
        inner.i8_scale = Some(scale);
        inner.cuda_i8_data = Some(buffer);
        inner.cuda_f32_data = Some(rounded_f32);
        inner.device = Device::Cuda;
        inner.grad = None;
        inner.cuda_f32_grad = None;
    }

    pub(crate) fn replace_cuda_f32_buffer_no_host_sync(
        &self,
        buffer: crate::ops::cuda::CudaBuffer,
    ) {
        let shape = {
            let inner = self.0.borrow();
            Self::logical_shape(&inner).to_vec()
        };
        let len = Self::checked_shape_len(&shape, "CUDA tensor buffer replacement");
        assert_eq!(
            buffer.len(),
            len,
            "CUDA tensor buffer length mismatch: expected {}, got {}",
            len,
            buffer.len()
        );

        let mut inner = self.0.borrow_mut();
        let storage_dtype = inner.storage_dtype;
        let i8_scale = inner.i8_scale;
        Self::clear_non_f32_storage(&mut inner);
        let native_lowp_buffer = match storage_dtype {
            DType::F16 | DType::BF16 => {
                crate::ops::cuda::f32_to_lowp_storage_no_host(&buffer, storage_dtype).ok()
            }
            DType::I8 => i8_scale
                .and_then(|scale| crate::ops::cuda::f32_to_i8_storage_no_host(&buffer, scale).ok()),
            DType::F32 => None,
        };
        inner.data = ArrayD::<f32>::zeros(IxDyn(&shape)).into_shared();
        inner.has_f32_data = false;
        inner.storage_dtype = storage_dtype;
        inner.i8_scale = if storage_dtype == DType::I8 {
            i8_scale
        } else {
            None
        };
        inner.cache_dirty = storage_dtype != DType::F32 && native_lowp_buffer.is_none();
        inner.device = Device::Cuda;
        inner.cuda_f32_data = Some(buffer);
        inner.cuda_f16_data = if storage_dtype == DType::F16 {
            native_lowp_buffer.clone()
        } else {
            None
        };
        inner.cuda_bf16_data = if storage_dtype == DType::BF16 {
            native_lowp_buffer.clone()
        } else {
            None
        };
        inner.cuda_i8_data = if storage_dtype == DType::I8 {
            native_lowp_buffer
        } else {
            None
        };
        inner.grad = None;
        inner.cuda_f32_grad = None;
    }

    pub(super) fn write_f32_row_4d_impl(
        &self,
        bb: usize,
        hk: usize,
        pos: usize,
        src: &[f32],
        cuda_src: Option<(&crate::ops::cuda::CudaBuffer, usize)>,
    ) {
        let cuda_buffer = {
            let inner = self.0.borrow();
            if inner.device == Device::Cuda {
                inner.cuda_f32_data.clone()
            } else {
                None
            }
        };

        let (offset, can_use_cuda_src) = {
            let mut inner = self.0.borrow_mut();
            match inner.storage_dtype {
                DType::F32 => {
                    let shape = inner.data.shape().to_vec();
                    assert_eq!(shape.len(), 4, "write_f32_row_4d_inplace expects [B,H,S,D]");
                    let (b_dim, h_dim, s_dim, d_dim) = (shape[0], shape[1], shape[2], shape[3]);
                    assert!(
                        bb < b_dim && hk < h_dim && pos < s_dim,
                        "cache row index out of bounds"
                    );
                    assert_eq!(src.len(), d_dim, "cache row width mismatch");
                    let mut view4 = inner
                        .data
                        .view_mut()
                        .into_dimensionality::<ndarray::Ix4>()
                        .expect("KV cache must be [B,H,S,D]");
                    let mut row = view4.slice_mut(ndarray::s![bb, hk, pos, ..]);
                    row.as_slice_mut()
                        .expect("KV cache row must be contiguous")
                        .copy_from_slice(src);
                    ((((bb * h_dim) + hk) * s_dim + pos) * d_dim, true)
                }
                DType::F16 => {
                    let data = inner
                        .f16_data
                        .as_mut()
                        .expect("f16 storage missing for cache row write");
                    let shape = data.shape().to_vec();
                    assert_eq!(shape.len(), 4, "write_f32_row_4d_inplace expects [B,H,S,D]");
                    let (b_dim, h_dim, s_dim, d_dim) = (shape[0], shape[1], shape[2], shape[3]);
                    assert!(
                        bb < b_dim && hk < h_dim && pos < s_dim,
                        "cache row index out of bounds"
                    );
                    assert_eq!(src.len(), d_dim, "cache row width mismatch");
                    let mut view4 = data
                        .view_mut()
                        .into_dimensionality::<ndarray::Ix4>()
                        .expect("KV cache must be [B,H,S,D]");
                    let mut row = view4.slice_mut(ndarray::s![bb, hk, pos, ..]);
                    let row_slice = row.as_slice_mut().expect("KV cache row must be contiguous");
                    for (dst, &value) in row_slice.iter_mut().zip(src.iter()) {
                        *dst = f16::from_f32(value);
                    }
                    ((((bb * h_dim) + hk) * s_dim + pos) * d_dim, false)
                }
                DType::BF16 => {
                    let data = inner
                        .bf16_data
                        .as_mut()
                        .expect("bf16 storage missing for cache row write");
                    let shape = data.shape().to_vec();
                    assert_eq!(shape.len(), 4, "write_f32_row_4d_inplace expects [B,H,S,D]");
                    let (b_dim, h_dim, s_dim, d_dim) = (shape[0], shape[1], shape[2], shape[3]);
                    assert!(
                        bb < b_dim && hk < h_dim && pos < s_dim,
                        "cache row index out of bounds"
                    );
                    assert_eq!(src.len(), d_dim, "cache row width mismatch");
                    let mut view4 = data
                        .view_mut()
                        .into_dimensionality::<ndarray::Ix4>()
                        .expect("KV cache must be [B,H,S,D]");
                    let mut row = view4.slice_mut(ndarray::s![bb, hk, pos, ..]);
                    let row_slice = row.as_slice_mut().expect("KV cache row must be contiguous");
                    for (dst, &value) in row_slice.iter_mut().zip(src.iter()) {
                        *dst = bf16::from_f32(value);
                    }
                    ((((bb * h_dim) + hk) * s_dim + pos) * d_dim, false)
                }
                DType::I8 => {
                    panic!("write_f32_row_4d_inplace does not support i8 storage");
                }
            }
        };

        if let Some(buffer) = cuda_buffer {
            if can_use_cuda_src {
                if let Some((src_buffer, src_offset)) = cuda_src {
                    crate::ops::cuda::copy_f32_offset(
                        &buffer,
                        offset,
                        src_buffer,
                        src_offset,
                        src.len(),
                    )
                    .unwrap_or_else(|err| panic!("Failed to copy CUDA tensor row: {}", err));
                } else {
                    crate::ops::cuda::upload_f32_offset(&buffer, offset, src)
                        .unwrap_or_else(|err| panic!("Failed to update CUDA tensor row: {}", err));
                }
            } else {
                crate::ops::cuda::upload_f32_offset(&buffer, offset, src)
                    .unwrap_or_else(|err| panic!("Failed to update CUDA tensor row: {}", err));
            }
        }
    }

    pub(crate) fn write_f32_row_4d_inplace(&self, bb: usize, hk: usize, pos: usize, src: &[f32]) {
        self.write_f32_row_4d_impl(bb, hk, pos, src, None);
    }

    pub(crate) fn write_f32_row_4d_from_cuda_source_inplace(
        &self,
        bb: usize,
        hk: usize,
        pos: usize,
        src: &[f32],
        src_buffer: &crate::ops::cuda::CudaBuffer,
        src_offset: usize,
    ) {
        self.write_f32_row_4d_impl(bb, hk, pos, src, Some((src_buffer, src_offset)));
    }

    pub(crate) fn write_f32_row_4d_from_cuda_buffer_inplace(
        &self,
        bb: usize,
        hk: usize,
        pos: usize,
        src_buffer: &crate::ops::cuda::CudaBuffer,
        src_offset: usize,
        len: usize,
    ) {
        let mut host = crate::ops::cuda::download_f32_offset(src_buffer, src_offset, len)
            .unwrap_or_else(|err| panic!("Failed to download CUDA tensor row: {}", err));
        match self.dtype() {
            DType::F32 => {}
            DType::F16 => {
                for value in host.iter_mut() {
                    *value = f16::from_f32(*value).to_f32();
                }
            }
            DType::BF16 => {
                for value in host.iter_mut() {
                    *value = bf16::from_f32(*value).to_f32();
                }
            }
            DType::I8 => {
                panic!("write_f32_row_4d_from_cuda_buffer_inplace does not support i8 storage");
            }
        }
        self.write_f32_row_4d_impl(bb, hk, pos, &host, Some((src_buffer, src_offset)));
    }

    #[inline]
    pub fn device(&self) -> Device {
        self.0.borrow().device
    }

    #[inline]
    pub fn is_cuda(&self) -> bool {
        self.device() == Device::Cuda
    }

    #[inline]
    pub fn is_cpu(&self) -> bool {
        self.device() == Device::Cpu
    }

    pub(crate) fn set_device_inplace(&self, device: Device) {
        let mut inner = self.0.borrow_mut();
        inner.device = device;
        if device == Device::Cpu {
            Self::clear_cuda_storage(&mut inner);
        }
    }

    pub fn to_device_inplace(&self, device: Device) {
        match device {
            Device::Cpu => {
                if self.is_cpu() {
                    return;
                }
                self.ensure_native_host_storage();
                if self.0.borrow().grad.is_none() {
                    self.materialize_cuda_grad_to_host();
                }
                let mut inner = self.0.borrow_mut();
                inner.device = Device::Cpu;
                Self::clear_cuda_storage(&mut inner);
                inner.cuda_f32_grad = None;
            }
            Device::Cuda => {
                assert!(
                    crate::ops::cuda::is_available(),
                    "CUDA is not available. Rebuild with `--features cuda` and ensure the NVIDIA runtime is installed."
                );
                if self.is_cuda() {
                    let inner = self.0.borrow();
                    if inner.cuda_f32_data.is_some()
                        || inner.cuda_f16_data.is_some()
                        || inner.cuda_bf16_data.is_some()
                        || inner.cuda_i8_data.is_some()
                    {
                        return;
                    }
                }
                if self.is_empty() {
                    let mut inner = self.0.borrow_mut();
                    inner.device = Device::Cuda;
                    inner.cuda_f32_data = None;
                    return;
                }

                let lowp_upload =
                    {
                        let (storage_dtype, cache_dirty) = {
                            let inner = self.0.borrow();
                            (inner.storage_dtype, inner.cache_dirty)
                        };
                        if cache_dirty {
                            None
                        } else {
                            match self.native_storage_owned() {
                                TensorStorageOwned::F16(data) if storage_dtype == DType::F16 => {
                                    let data = data.as_standard_layout();
                                    data.as_slice().map(|slice| {
                                        let bits =
                                            slice.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
                                        (
                                            DType::F16,
                                            crate::ops::cuda::upload_u16_storage(&bits)
                                                .unwrap_or_else(|err| {
                                                    panic!(
                                                        "Failed to upload F16 tensor to CUDA: {}",
                                                        err
                                                    )
                                                }),
                                        )
                                    })
                                }
                                TensorStorageOwned::BF16(data) if storage_dtype == DType::BF16 => {
                                    let data = data.as_standard_layout();
                                    data.as_slice().map(|slice| {
                                        let bits =
                                            slice.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
                                        (
                                            DType::BF16,
                                            crate::ops::cuda::upload_u16_storage(&bits)
                                                .unwrap_or_else(|err| {
                                                    panic!(
                                                        "Failed to upload BF16 tensor to CUDA: {}",
                                                        err
                                                    )
                                                }),
                                        )
                                    })
                                }
                                TensorStorageOwned::I8(data, _) if storage_dtype == DType::I8 => {
                                    let data = data.as_standard_layout();
                                    data.as_slice().map(|slice| {
                                        (
                                            DType::I8,
                                            crate::ops::cuda::upload_i8_storage(slice)
                                                .unwrap_or_else(|err| {
                                                    panic!(
                                                        "Failed to upload I8 tensor to CUDA: {}",
                                                        err
                                                    )
                                                }),
                                        )
                                    })
                                }
                                _ => None,
                            }
                        }
                    };
                if let Some((dtype, buffer)) = lowp_upload {
                    let mut inner = self.0.borrow_mut();
                    inner.device = Device::Cuda;
                    match dtype {
                        DType::F16 => inner.cuda_f16_data = Some(buffer),
                        DType::BF16 => inner.cuda_bf16_data = Some(buffer),
                        DType::I8 => inner.cuda_i8_data = Some(buffer),
                        DType::F32 => {}
                    }
                    return;
                }

                let host_data = self.storage_as_f32_vec();
                let buffer = crate::ops::cuda::upload_f32(&host_data)
                    .unwrap_or_else(|err| panic!("Failed to upload tensor to CUDA: {}", err));
                let mut inner = self.0.borrow_mut();
                inner.device = Device::Cuda;
                inner.cuda_f32_data = Some(buffer);
            }
        }
    }

    pub fn to_cpu_inplace(&self) {
        self.to_device_inplace(Device::Cpu);
    }

    pub fn to_cuda_inplace(&self) {
        self.to_device_inplace(Device::Cuda);
    }

    pub(super) fn clone_storage_with_device(&self, device: Device, requires_grad: bool) -> Tensor {
        // Snapshot access never populates the source's device cache: callers can
        // retain shared views while asking for an independent copy.
        let resident = {
            let inner = self.0.borrow();
            if device == Device::Cuda && inner.device == Device::Cuda {
                match inner.storage_dtype {
                    DType::F32 => inner.cuda_f32_data.clone().map(|b| (DType::F32, b, None)),
                    DType::F16 if !inner.cache_dirty => {
                        inner.cuda_f16_data.clone().map(|b| (DType::F16, b, None))
                    }
                    DType::BF16 if !inner.cache_dirty => {
                        inner.cuda_bf16_data.clone().map(|b| (DType::BF16, b, None))
                    }
                    DType::I8 if !inner.cache_dirty => inner
                        .cuda_i8_data
                        .clone()
                        .map(|b| (DType::I8, b, inner.i8_scale)),
                    _ => None,
                }
            } else {
                None
            }
        };
        let result = if let Some((dtype, buffer, scale)) = resident {
            if dtype == DType::F32 {
                let copy =
                    crate::ops::cuda::alloc_f32(buffer.len()).expect("allocate CUDA tensor copy");
                crate::ops::cuda::copy_f32_offset(&copy, 0, &buffer, 0, buffer.len())
                    .expect("copy CUDA tensor storage");
                Self::from_cuda_f32_buffer_no_host(&self.shape_vec(), copy, device)
            } else {
                let copy = crate::ops::cuda::slice_lastdim_typed_buffer(
                    &buffer,
                    dtype,
                    1,
                    buffer.len(),
                    0,
                    buffer.len(),
                )
                .expect("copy native CUDA tensor storage");
                Self::from_cuda_native_lowp_buffer_no_host_with_dtype(
                    &self.shape_vec(),
                    copy,
                    device,
                    dtype,
                    scale,
                )
            }
        } else {
            let result = match self.native_storage_owned() {
                TensorStorageOwned::F32(data) => Self::from_data_no_grad(data),
                TensorStorageOwned::F16(data) => Self::from_f16_data_no_grad(data),
                TensorStorageOwned::BF16(data) => Self::from_bf16_data_no_grad(data),
                TensorStorageOwned::I8(data, scale) => Self::from_i8_data_no_grad(data, scale),
            };
            result.to_device_inplace(device);
            result
        };
        let mut target = result.0.borrow_mut();
        target.is_parameter = self.0.borrow().is_parameter;
        target.requires_grad = requires_grad;
        drop(target);
        result
    }

    pub fn to_device(&self, device: Device) -> Tensor {
        self.clone_storage_with_device(device, self.requires_grad())
    }

    pub fn to_cpu(&self) -> Tensor {
        self.to_device(Device::Cpu)
    }

    pub fn to_cuda(&self) -> Tensor {
        self.to_device(Device::Cuda)
    }

    pub fn cpu(&self) -> Tensor {
        self.to_cpu()
    }

    pub fn cuda(&self) -> Tensor {
        self.to_cuda()
    }

    // detach：返回一个新 Tensor（数据拷贝），requires_grad=false，且无 parents/backward_op
    pub fn detach(&self) -> Tensor {
        self.clone_storage_with_device(self.device(), false)
    }
}
