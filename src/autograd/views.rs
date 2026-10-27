use super::*;

impl Tensor {
    pub(super) fn with_bf16_compute_view<R>(
        data: &ArcArray<bf16, IxDyn>,
        f: impl FnOnce(ArrayViewD<'_, f32>) -> R,
    ) -> R {
        let shape = data.shape().to_vec();
        let len = data.len();
        F32_COMPUTE_SCRATCH.with(|scratch| {
            if let Ok(mut scratch) = scratch.try_borrow_mut() {
                if scratch.len() < len {
                    scratch.resize(len, 0.0);
                }
                if let Some(slice) = data.as_slice() {
                    slice.convert_to_f32_slice(&mut scratch[..len]);
                } else {
                    for (dst, src) in scratch[..len].iter_mut().zip(data.iter()) {
                        *dst = src.to_f32();
                    }
                }
                let view = ArrayViewD::from_shape(IxDyn(&shape), &scratch[..len])
                    .expect("Failed to build temporary f32 compute view");
                return f(view);
            }

            let mut fallback = vec![0.0f32; len];
            if let Some(slice) = data.as_slice() {
                slice.convert_to_f32_slice(&mut fallback);
            } else {
                for (dst, src) in fallback.iter_mut().zip(data.iter()) {
                    *dst = src.to_f32();
                }
            }
            let view = ArrayViewD::from_shape(IxDyn(&shape), &fallback)
                .expect("Failed to build temporary f32 compute view");
            f(view)
        })
    }

    pub(super) fn with_f16_compute_view<R>(
        data: &ArcArray<f16, IxDyn>,
        f: impl FnOnce(ArrayViewD<'_, f32>) -> R,
    ) -> R {
        let shape = data.shape().to_vec();
        let len = data.len();
        F32_COMPUTE_SCRATCH.with(|scratch| {
            if let Ok(mut scratch) = scratch.try_borrow_mut() {
                if scratch.len() < len {
                    scratch.resize(len, 0.0);
                }
                if let Some(slice) = data.as_slice() {
                    slice.convert_to_f32_slice(&mut scratch[..len]);
                } else {
                    for (dst, src) in scratch[..len].iter_mut().zip(data.iter()) {
                        *dst = src.to_f32();
                    }
                }
                let view = ArrayViewD::from_shape(IxDyn(&shape), &scratch[..len])
                    .expect("Failed to build temporary f32 compute view");
                return f(view);
            }

            let mut fallback = vec![0.0f32; len];
            if let Some(slice) = data.as_slice() {
                slice.convert_to_f32_slice(&mut fallback);
            } else {
                for (dst, src) in fallback.iter_mut().zip(data.iter()) {
                    *dst = src.to_f32();
                }
            }
            let view = ArrayViewD::from_shape(IxDyn(&shape), &fallback)
                .expect("Failed to build temporary f32 compute view");
            f(view)
        })
    }

    pub(super) fn with_i8_compute_view<R>(
        data: &ArcArray<i8, IxDyn>,
        scale: f32,
        f: impl FnOnce(ArrayViewD<'_, f32>) -> R,
    ) -> R {
        let shape = data.shape().to_vec();
        let len = data.len();
        F32_COMPUTE_SCRATCH.with(|scratch| {
            if let Ok(mut scratch) = scratch.try_borrow_mut() {
                if scratch.len() < len {
                    scratch.resize(len, 0.0);
                }
                for (dst, src) in scratch[..len].iter_mut().zip(data.iter()) {
                    *dst = (*src as f32) * scale;
                }
                let view = ArrayViewD::from_shape(IxDyn(&shape), &scratch[..len])
                    .expect("Failed to build temporary f32 compute view");
                return f(view);
            }

            let mut fallback = vec![0.0f32; len];
            for (dst, src) in fallback.iter_mut().zip(data.iter()) {
                *dst = (*src as f32) * scale;
            }
            let view = ArrayViewD::from_shape(IxDyn(&shape), &fallback)
                .expect("Failed to build temporary f32 compute view");
            f(view)
        })
    }

    pub fn with_storage_view_preferring<R>(
        &self,
        preference: StoragePreference,
        f: impl FnOnce(TensorStorageView<'_>) -> R,
    ) -> R {
        self.ensure_native_host_storage();
        if matches!(preference, StoragePreference::F32Compute) {
            {
                let inner = self.0.borrow();
                if inner.cuda_f32_data.is_some()
                    && !inner.has_f32_data
                    && (inner.storage_dtype == DType::F32 || inner.cache_dirty)
                {
                    drop(inner);
                    self.ensure_f32_data(false);
                }
            }

            let should_materialize_parameter_cache = {
                let inner = self.0.borrow();
                inner.is_parameter
                    && allow_parameter_dtype_copies()
                    && inner.storage_dtype != DType::F32
                    && !inner.has_f32_data
            };

            if should_materialize_parameter_cache {
                self.ensure_f32_data(false);
            }

            let inner = self.0.borrow();
            if inner.has_f32_data {
                return f(TensorStorageView::F32(inner.data.view()));
            }
            if let Some(f16_data) = inner.f16_data.clone() {
                drop(inner);
                return Self::with_f16_compute_view(&f16_data, |view| {
                    f(TensorStorageView::F32(view))
                });
            }
            if let Some(bf16_data) = inner.bf16_data.clone() {
                drop(inner);
                return Self::with_bf16_compute_view(&bf16_data, |view| {
                    f(TensorStorageView::F32(view))
                });
            }
            if let Some(i8_data) = inner.i8_data.clone() {
                let scale = inner
                    .i8_scale
                    .expect("I8 tensor missing quantization scale");
                drop(inner);
                return Self::with_i8_compute_view(&i8_data, scale, |view| {
                    f(TensorStorageView::F32(view))
                });
            }
            return f(TensorStorageView::F32(inner.data.view()));
        }

        let should_ensure_f32 = {
            let inner = self.0.borrow();
            match preference {
                StoragePreference::Auto => {
                    (inner.is_parameter
                        && allow_parameter_dtype_copies()
                        && inner.storage_dtype != DType::F32
                        && !inner.has_f32_data)
                        || (inner.storage_dtype == DType::F32 && !inner.has_f32_data)
                        || (inner.cache_dirty
                            && inner.cuda_f32_data.is_some()
                            && !inner.has_f32_data)
                        || (inner.cuda_f32_data.is_some()
                            && ((inner.storage_dtype == DType::F16 && inner.f16_data.is_none())
                                || (inner.storage_dtype == DType::BF16
                                    && inner.bf16_data.is_none())))
                }
                StoragePreference::Native => {
                    (inner.storage_dtype == DType::F32 && !inner.has_f32_data)
                        || (inner.cache_dirty
                            && inner.cuda_f32_data.is_some()
                            && !inner.has_f32_data)
                        || (inner.cuda_f32_data.is_some()
                            && ((inner.storage_dtype == DType::F16 && inner.f16_data.is_none())
                                || (inner.storage_dtype == DType::BF16
                                    && inner.bf16_data.is_none())))
                }
                StoragePreference::F32Compute => unreachable!("handled above"),
            }
        };

        if should_ensure_f32 {
            self.ensure_f32_data(false);
        }

        let inner = self.0.borrow();
        if matches!(preference, StoragePreference::Auto)
            && inner.is_parameter
            && allow_parameter_dtype_copies()
            && inner.has_f32_data
        {
            return f(TensorStorageView::F32(inner.data.view()));
        }

        if inner.storage_dtype == DType::F32 {
            return f(TensorStorageView::F32(inner.data.view()));
        }

        if inner.cache_dirty {
            return f(TensorStorageView::F32(inner.data.view()));
        }

        if let Some(f16_data) = inner.f16_data.as_ref() {
            return f(TensorStorageView::F16(f16_data.view()));
        }

        if let Some(bf16_data) = inner.bf16_data.as_ref() {
            return f(TensorStorageView::BF16(bf16_data.view()));
        }

        if let Some(i8_data) = inner.i8_data.clone() {
            let scale = inner
                .i8_scale
                .expect("I8 tensor missing quantization scale");
            drop(inner);
            Self::with_i8_compute_view(&i8_data, scale, |view| f(TensorStorageView::F32(view)))
        } else {
            f(TensorStorageView::F32(inner.data.view()))
        }
    }

    pub fn with_storage_view<R>(&self, f: impl FnOnce(TensorStorageView<'_>) -> R) -> R {
        self.with_storage_view_preferring(StoragePreference::Auto, f)
    }

    pub fn with_storage_view_for_input_dtype<R>(
        &self,
        input_dtype: DType,
        f: impl FnOnce(TensorStorageView<'_>) -> R,
    ) -> R {
        self.with_storage_view_for_input_dtype_and_route(
            input_dtype,
            KernelRouteClass::DecodeKernel,
            f,
        )
    }

    pub fn with_storage_view_for_input_dtype_and_route<R>(
        &self,
        input_dtype: DType,
        route: KernelRouteClass,
        f: impl FnOnce(TensorStorageView<'_>) -> R,
    ) -> R {
        let preference = preferred_parameter_storage_for_route(input_dtype, self.dtype(), route);
        self.with_storage_view_preferring(preference, f)
    }

    pub fn with_native_storage_view_mut<R>(
        &self,
        f: impl FnOnce(TensorStorageViewMut<'_>) -> R,
    ) -> R {
        self.ensure_native_host_storage();

        let mut inner = self.0.borrow_mut();
        Self::clear_cuda_storage(&mut inner);
        if inner.storage_dtype != DType::F32 {
            // Native writes invalidate any F32 compute cache as well as device copies.
            inner.data = Self::empty_f32_storage();
            inner.has_f32_data = false;
            inner.cache_dirty = false;
        }
        match inner.storage_dtype {
            DType::F32 => f(TensorStorageViewMut::F32(inner.data.view_mut())),
            DType::F16 => {
                let view = inner
                    .f16_data
                    .as_mut()
                    .expect("F16 tensor missing f16 storage")
                    .view_mut();
                f(TensorStorageViewMut::F16(view))
            }
            DType::BF16 => {
                let view = inner
                    .bf16_data
                    .as_mut()
                    .expect("BF16 tensor missing bf16 storage")
                    .view_mut();
                f(TensorStorageViewMut::BF16(view))
            }
            DType::I8 => {
                let scale = inner
                    .i8_scale
                    .expect("I8 tensor missing quantization scale");
                let view = inner
                    .i8_data
                    .as_mut()
                    .expect("I8 tensor missing i8 storage")
                    .view_mut();
                f(TensorStorageViewMut::I8(view, scale))
            }
        }
    }

    // 获取数据的只读引用（零拷贝）
    pub fn data_ref(&self) -> Ref<'_, ArcArray<f32, IxDyn>> {
        self.ensure_f32_data(false);
        let borrow = self.0.borrow();
        Ref::map(borrow, |t| &t.data)
    }

    // 获取数据的可变引用
    pub fn data_mut(&self) -> RefMut<'_, ArcArray<f32, IxDyn>> {
        self.ensure_f32_data(true);
        let borrow = self.0.borrow_mut();
        RefMut::map(borrow, |t| &mut t.data)
    }

    pub fn data(&self) -> ArrayD<f32> {
        self.data_ref().to_owned()
    }

    // 快路径：返回共享数据（clone 仅增加引用计数，不复制）
    pub fn data_arc(&self) -> ArcArray<f32, IxDyn> {
        self.data_ref().clone()
    }
}
