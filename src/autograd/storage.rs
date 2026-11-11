use super::*;

impl Tensor {
    #[inline]
    pub(super) fn empty_f32_storage() -> ArcArray<f32, IxDyn> {
        ArrayD::<f32>::zeros(IxDyn(&[0])).into_shared()
    }

    pub(crate) fn clear_non_f32_storage(inner: &mut TensorData) {
        inner.f16_data = None;
        inner.bf16_data = None;
        inner.i8_data = None;
        inner.i8_scale = None;
    }

    pub(crate) fn clear_cuda_storage(inner: &mut TensorData) {
        inner.cuda_f32_data = None;
        inner.cuda_f16_data = None;
        inner.cuda_bf16_data = None;
        inner.cuda_i8_data = None;
    }

    pub(crate) fn logical_shape(inner: &TensorData) -> &[usize] {
        if inner.storage_dtype == DType::F16
            && let Some(f16_data) = inner.f16_data.as_ref()
        {
            return f16_data.shape();
        }
        if inner.storage_dtype == DType::BF16
            && let Some(bf16_data) = inner.bf16_data.as_ref()
        {
            return bf16_data.shape();
        }
        if inner.storage_dtype == DType::I8
            && let Some(i8_data) = inner.i8_data.as_ref()
        {
            return i8_data.shape();
        }

        if inner.has_f32_data {
            inner.data.shape()
        } else if let Some(bf16_data) = inner.bf16_data.as_ref() {
            bf16_data.shape()
        } else if let Some(i8_data) = inner.i8_data.as_ref() {
            i8_data.shape()
        } else {
            inner.data.shape()
        }
    }

    pub(crate) fn checked_shape_len(shape: &[usize], context: &str) -> usize {
        shape
            .iter()
            .try_fold(1usize, |len, &dim| len.checked_mul(dim))
            .unwrap_or_else(|| panic!("{context} shape element count overflow"))
    }

    pub(super) fn ensure_f32_data(&self, mutable: bool) {
        // A ready read-only view must not request a mutable borrow: aliased
        // operands and nested data_ref() calls can already hold shared views.
        if !mutable && self.0.borrow().has_f32_data {
            return;
        }
        let mut inner = self.0.borrow_mut();

        if inner.has_f32_data {
            if mutable {
                Self::clear_cuda_storage(&mut inner);
            }
            if mutable && inner.storage_dtype != DType::F32 {
                inner.storage_dtype = DType::F32;
                if inner.is_parameter
                    && allow_parameter_dtype_copies()
                    && (inner.f16_data.is_some()
                        || inner.bf16_data.is_some()
                        || inner.i8_data.is_some())
                {
                    inner.cache_dirty = true;
                } else {
                    Self::clear_non_f32_storage(&mut inner);
                    inner.cache_dirty = false;
                }
            }
            return;
        }

        let should_download_cuda_f32 = if inner.cuda_f32_data.is_some() {
            inner.storage_dtype == DType::F32
                || inner.cache_dirty
                || match inner.storage_dtype {
                    DType::F32 => false,
                    DType::F16 => inner.f16_data.is_none(),
                    DType::BF16 => inner.bf16_data.is_none(),
                    DType::I8 => inner.i8_data.is_none(),
                }
        } else {
            false
        };

        if should_download_cuda_f32 {
            let buffer = inner
                .cuda_f32_data
                .as_ref()
                .expect("CUDA download requested without resident buffer");
            let host_data = crate::ops::cuda::download_f32(buffer)
                .unwrap_or_else(|err| panic!("Failed to download CUDA tensor to host: {}", err));
            let shape = Self::logical_shape(&inner).to_vec();
            let keep_parameter_dtype = !mutable
                && inner.is_parameter
                && inner.storage_dtype != DType::F32
                && allow_parameter_dtype_copies();
            inner.data = Array::from_shape_vec(IxDyn(&shape), host_data)
                .expect("Failed to materialize host f32 storage from CUDA buffer")
                .into_shared();
            inner.has_f32_data = true;
            if mutable {
                Self::clear_cuda_storage(&mut inner);
            }
            if keep_parameter_dtype {
                inner.cache_dirty = true;
            } else if mutable || !inner.is_parameter || !allow_parameter_dtype_copies() {
                inner.storage_dtype = DType::F32;
                Self::clear_non_f32_storage(&mut inner);
                inner.cache_dirty = false;
            } else {
                inner.cache_dirty = false;
            }
            return;
        }

        let cuda_native_buffer = match inner.storage_dtype {
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
        };
        if let Some((dtype, buffer)) = cuda_native_buffer {
            let shape = Self::logical_shape(&inner).to_vec();
            match dtype {
                DType::F16 => {
                    let bits =
                        crate::ops::cuda::download_u16_storage(&buffer).unwrap_or_else(|err| {
                            panic!("Failed to download CUDA F16 tensor to host: {}", err)
                        });
                    let values = bits.into_iter().map(f16::from_bits).collect::<Vec<_>>();
                    let host = Array::from_shape_vec(IxDyn(&shape), values)
                        .expect("Failed to materialize host f16 storage from CUDA buffer")
                        .into_shared();
                    inner.data = Self::f16_arc_to_f32(&host);
                    inner.f16_data = Some(host);
                }
                DType::BF16 => {
                    let bits =
                        crate::ops::cuda::download_u16_storage(&buffer).unwrap_or_else(|err| {
                            panic!("Failed to download CUDA BF16 tensor to host: {}", err)
                        });
                    let values = bits.into_iter().map(bf16::from_bits).collect::<Vec<_>>();
                    let host = Array::from_shape_vec(IxDyn(&shape), values)
                        .expect("Failed to materialize host bf16 storage from CUDA buffer")
                        .into_shared();
                    inner.data = Self::bf16_arc_to_f32(&host);
                    inner.bf16_data = Some(host);
                }
                DType::I8 => {
                    let values =
                        crate::ops::cuda::download_i8_storage(&buffer).unwrap_or_else(|err| {
                            panic!("Failed to download CUDA I8 tensor to host: {}", err)
                        });
                    let scale = inner
                        .i8_scale
                        .expect("I8 tensor missing quantization scale");
                    let host = Array::from_shape_vec(IxDyn(&shape), values)
                        .expect("Failed to materialize host i8 storage from CUDA buffer")
                        .into_shared();
                    inner.data = Self::i8_arc_to_f32(&host, scale);
                    inner.i8_data = Some(host);
                }
                DType::F32 => unreachable!("native CUDA lowp materialization excludes f32"),
            }
            inner.has_f32_data = true;
            if mutable {
                Self::clear_cuda_storage(&mut inner);
            }
            if mutable || !inner.is_parameter || !allow_parameter_dtype_copies() {
                inner.storage_dtype = DType::F32;
                if inner.is_parameter && allow_parameter_dtype_copies() && mutable {
                    inner.cache_dirty = true;
                } else {
                    Self::clear_non_f32_storage(&mut inner);
                    inner.cache_dirty = false;
                }
            }
            return;
        }

        if mutable {
            Self::clear_cuda_storage(&mut inner);
        }

        if let Some(f16_data) = inner.f16_data.clone() {
            inner.data = Self::f16_arc_to_f32(&f16_data);
        } else if let Some(bf16_data) = inner.bf16_data.clone() {
            inner.data = Self::bf16_arc_to_f32(&bf16_data);
        } else if let Some(i8_data) = inner.i8_data.clone() {
            let scale = inner
                .i8_scale
                .expect("I8 tensor missing quantization scale");
            inner.data = Self::i8_arc_to_f32(&i8_data, scale);
        } else {
            panic!("Tensor has no materializable storage");
        }
        inner.has_f32_data = true;

        if mutable || !inner.is_parameter || !allow_parameter_dtype_copies() {
            inner.storage_dtype = DType::F32;
            if inner.is_parameter && allow_parameter_dtype_copies() && mutable {
                inner.cache_dirty = true;
            } else {
                Self::clear_non_f32_storage(&mut inner);
                inner.cache_dirty = false;
            }
        }
    }

    pub(crate) fn storage_as_f32_vec(&self) -> Vec<f32> {
        let inner = self.0.borrow();
        if inner.has_f32_data {
            return inner.data.iter().copied().collect();
        }

        if let Some(f16_data) = inner.f16_data.as_ref() {
            if let Some(slice) = f16_data.as_slice() {
                let mut out = vec![0.0f32; slice.len()];
                slice.convert_to_f32_slice(&mut out);
                return out;
            }
            return f16_data.iter().map(|&v| v.to_f32()).collect();
        }

        if let Some(bf16_data) = inner.bf16_data.as_ref() {
            if let Some(slice) = bf16_data.as_slice() {
                let mut out = vec![0.0f32; slice.len()];
                slice.convert_to_f32_slice(&mut out);
                return out;
            }
            return bf16_data.iter().map(|&v| v.to_f32()).collect();
        }

        if let Some(i8_data) = inner.i8_data.as_ref() {
            let scale = inner
                .i8_scale
                .expect("I8 tensor missing quantization scale");
            return i8_data.iter().map(|&v| (v as f32) * scale).collect();
        }

        inner.data.iter().copied().collect()
    }

    #[allow(dead_code)]
    pub(crate) fn has_host_f32_data(&self) -> bool {
        self.0.borrow().has_f32_data
    }

    pub fn shape_vec(&self) -> Vec<usize> {
        let inner = self.0.borrow();
        Self::logical_shape(&inner).to_vec()
    }

    #[inline]
    pub fn len(&self) -> usize {
        let inner = self.0.borrow();
        if inner.storage_dtype == DType::F16
            && let Some(f16_data) = inner.f16_data.as_ref()
        {
            return f16_data.len();
        }
        if inner.storage_dtype == DType::BF16
            && let Some(bf16_data) = inner.bf16_data.as_ref()
        {
            return bf16_data.len();
        }
        if inner.storage_dtype == DType::I8
            && let Some(i8_data) = inner.i8_data.as_ref()
        {
            return i8_data.len();
        }
        if inner.has_f32_data {
            inner.data.len()
        } else if let Some(f16_data) = inner.f16_data.as_ref() {
            f16_data.len()
        } else if let Some(bf16_data) = inner.bf16_data.as_ref() {
            bf16_data.len()
        } else if let Some(i8_data) = inner.i8_data.as_ref() {
            i8_data.len()
        } else {
            inner.data.len()
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub fn ndim(&self) -> usize {
        self.shape_vec().len()
    }

    #[inline]
    pub fn dtype(&self) -> DType {
        self.0.borrow().storage_dtype
    }

    #[inline]
    pub fn quantization_scale(&self) -> Option<f32> {
        let inner = self.0.borrow();
        if inner.storage_dtype == DType::I8 {
            inner.i8_scale
        } else {
            None
        }
    }

    pub fn cast_inplace(&self, dtype: DType) {
        match dtype {
            DType::F32 => {
                self.ensure_f32_data(false);
                let mut inner = self.0.borrow_mut();
                Self::clear_cuda_storage(&mut inner);
                inner.storage_dtype = DType::F32;
                Self::clear_non_f32_storage(&mut inner);
                inner.cache_dirty = false;
            }
            DType::F16 => {
                self.ensure_f32_data(false);
                let mut inner = self.0.borrow_mut();
                Self::clear_cuda_storage(&mut inner);
                inner.f16_data = Some(Self::f32_arc_to_f16(&inner.data));
                inner.bf16_data = None;
                inner.i8_data = None;
                inner.i8_scale = None;
                inner.storage_dtype = DType::F16;
                inner.data = Self::empty_f32_storage();
                inner.has_f32_data = false;
                inner.cache_dirty = false;
            }
            DType::BF16 => {
                self.ensure_f32_data(false);
                let mut inner = self.0.borrow_mut();
                Self::clear_cuda_storage(&mut inner);
                inner.f16_data = None;
                inner.bf16_data = Some(Self::f32_arc_to_bf16(&inner.data));
                inner.i8_data = None;
                inner.i8_scale = None;
                inner.storage_dtype = DType::BF16;
                inner.data = Self::empty_f32_storage();
                inner.has_f32_data = false;
                inner.cache_dirty = false;
            }
            DType::I8 => {
                self.ensure_f32_data(false);
                let mut inner = self.0.borrow_mut();
                Self::clear_cuda_storage(&mut inner);
                inner.f16_data = None;
                let (i8_data, scale) =
                    Self::quantize_f32_to_dtype(&inner.data.to_owned(), DType::I8, None);
                inner.bf16_data = None;
                inner.i8_data = Some(i8_data);
                inner.i8_scale = Some(scale);
                inner.storage_dtype = DType::I8;
                inner.data = Self::empty_f32_storage();
                inner.has_f32_data = false;
                inner.cache_dirty = false;
            }
        }
    }

    pub fn set_array_f32_with_dtype(&self, data: ArrayD<f32>, dtype: DType) {
        let mut inner = self.0.borrow_mut();
        Self::clear_cuda_storage(&mut inner);
        match dtype {
            DType::F32 => {
                inner.data = data.into_shared();
                Self::clear_non_f32_storage(&mut inner);
                inner.has_f32_data = true;
                inner.storage_dtype = DType::F32;
                inner.cache_dirty = false;
            }
            DType::F16 => {
                inner.data = Self::empty_f32_storage();
                inner.f16_data = Some(Self::f32_array_to_f16(&data));
                inner.bf16_data = None;
                inner.i8_data = None;
                inner.i8_scale = None;
                inner.has_f32_data = false;
                inner.storage_dtype = DType::F16;
                inner.cache_dirty = false;
            }
            DType::BF16 => {
                inner.data = Self::empty_f32_storage();
                inner.f16_data = None;
                inner.bf16_data = Some(Self::f32_array_to_bf16(&data));
                inner.i8_data = None;
                inner.i8_scale = None;
                inner.has_f32_data = false;
                inner.storage_dtype = DType::BF16;
                inner.cache_dirty = false;
            }
            DType::I8 => {
                let (i8_data, scale) = Self::quantize_f32_to_dtype(&data, DType::I8, None);
                inner.data = Self::empty_f32_storage();
                inner.f16_data = None;
                inner.bf16_data = None;
                inner.i8_data = Some(i8_data);
                inner.i8_scale = Some(scale);
                inner.has_f32_data = false;
                inner.storage_dtype = DType::I8;
                inner.cache_dirty = false;
            }
        }
        inner.grad = None;
        inner.cuda_f32_grad = None;
    }

    pub fn set_array_f32_with_quantization(
        &self,
        data: ArrayD<f32>,
        quantization: ParameterQuantization,
    ) {
        if let Some(dtype) = quantization.storage_dtype() {
            let mut inner = self.0.borrow_mut();
            Self::clear_cuda_storage(&mut inner);
            match dtype {
                DType::I8 => {
                    let (i8_data, scale) =
                        Self::quantize_f32_to_dtype(&data, dtype, quantization.scale_override());
                    inner.data = Self::empty_f32_storage();
                    inner.f16_data = None;
                    inner.bf16_data = None;
                    inner.i8_data = Some(i8_data);
                    inner.i8_scale = Some(scale);
                    inner.has_f32_data = false;
                    inner.storage_dtype = dtype;
                    inner.cache_dirty = false;
                    inner.grad = None;
                    inner.cuda_f32_grad = None;
                }
                other => {
                    panic!(
                        "quantized storage dtype {:?} is not implemented yet; currently only I8 is supported",
                        other
                    );
                }
            }
            return;
        }

        self.set_array_f32_with_dtype(data, DType::F32);
    }

    pub(super) fn set_i8_storage(&self, i8_data: ArcArray<i8, IxDyn>, scale: f32) {
        let mut inner = self.0.borrow_mut();
        Self::clear_cuda_storage(&mut inner);
        inner.data = Self::empty_f32_storage();
        inner.f16_data = None;
        inner.bf16_data = None;
        inner.i8_data = Some(i8_data);
        inner.i8_scale = Some(scale);
        inner.has_f32_data = false;
        inner.storage_dtype = DType::I8;
        inner.cache_dirty = false;
        inner.grad = None;
        inner.cuda_f32_grad = None;
    }

    pub(super) fn try_quantize_into_existing_i8_storage_f32(
        &self,
        shape: &[usize],
        data: &[f32],
        quantization: ParameterQuantization,
    ) -> bool {
        let mut inner = self.0.borrow_mut();
        Self::clear_cuda_storage(&mut inner);
        if inner.storage_dtype != DType::I8 {
            return false;
        }
        let Some(existing) = inner.i8_data.as_mut() else {
            return false;
        };
        if existing.shape() != shape {
            return false;
        }
        let Some(dst) = existing.as_slice_mut() else {
            return false;
        };
        let scale =
            Self::quantize_f32_values_into_i8_slice(data, quantization.scale_override(), dst);
        inner.i8_scale = Some(scale);
        inner.cache_dirty = false;
        inner.grad = None;
        inner.cuda_f32_grad = None;
        true
    }

    pub(super) fn try_quantize_into_existing_i8_storage_f16(
        &self,
        shape: &[usize],
        data: &[f16],
        quantization: ParameterQuantization,
    ) -> bool {
        let mut inner = self.0.borrow_mut();
        Self::clear_cuda_storage(&mut inner);
        if inner.storage_dtype != DType::I8 {
            return false;
        }
        let Some(existing) = inner.i8_data.as_mut() else {
            return false;
        };
        if existing.shape() != shape {
            return false;
        }
        let Some(dst) = existing.as_slice_mut() else {
            return false;
        };
        let scale =
            Self::quantize_f16_values_into_i8_slice(data, quantization.scale_override(), dst);
        inner.i8_scale = Some(scale);
        inner.cache_dirty = false;
        inner.grad = None;
        inner.cuda_f32_grad = None;
        true
    }

    pub(super) fn try_quantize_into_existing_i8_storage_bf16(
        &self,
        shape: &[usize],
        data: &[bf16],
        quantization: ParameterQuantization,
    ) -> bool {
        let mut inner = self.0.borrow_mut();
        Self::clear_cuda_storage(&mut inner);
        if inner.storage_dtype != DType::I8 {
            return false;
        }
        let Some(existing) = inner.i8_data.as_mut() else {
            return false;
        };
        if existing.shape() != shape {
            return false;
        }
        let Some(dst) = existing.as_slice_mut() else {
            return false;
        };
        let scale =
            Self::quantize_bf16_values_into_i8_slice(data, quantization.scale_override(), dst);
        inner.i8_scale = Some(scale);
        inner.cache_dirty = false;
        inner.grad = None;
        inner.cuda_f32_grad = None;
        true
    }

    pub(super) fn try_copy_into_existing_i8_storage_i8(
        &self,
        shape: &[usize],
        data: &[i8],
        scale: f32,
    ) -> bool {
        let mut inner = self.0.borrow_mut();
        Self::clear_cuda_storage(&mut inner);
        if inner.storage_dtype != DType::I8 {
            return false;
        }
        let Some(existing) = inner.i8_data.as_mut() else {
            return false;
        };
        if existing.shape() != shape {
            return false;
        }
        let Some(dst) = existing.as_slice_mut() else {
            return false;
        };
        dst.copy_from_slice(data);
        inner.i8_scale = Some(scale);
        inner.cache_dirty = false;
        inner.grad = None;
        inner.cuda_f32_grad = None;
        true
    }

    pub fn set_f32_slice_with_quantization(
        &self,
        shape: &[usize],
        data: &[f32],
        quantization: ParameterQuantization,
    ) {
        if self.try_quantize_into_existing_i8_storage_f32(shape, data, quantization) {
            return;
        }
        let dtype = quantization
            .storage_dtype()
            .expect("enabled quantization must provide storage dtype");
        match dtype {
            DType::I8 => {
                let (i8_data, scale) =
                    Self::quantize_f32_slice_to_i8(shape, data, quantization.scale_override());
                self.set_i8_storage(i8_data, scale);
            }
            other => panic!(
                "quantized storage dtype {:?} is not implemented yet; currently only I8 is supported",
                other
            ),
        }
    }

    pub fn set_f16_slice_with_quantization(
        &self,
        shape: &[usize],
        data: &[f16],
        quantization: ParameterQuantization,
    ) {
        if self.try_quantize_into_existing_i8_storage_f16(shape, data, quantization) {
            return;
        }
        let dtype = quantization
            .storage_dtype()
            .expect("enabled quantization must provide storage dtype");
        match dtype {
            DType::I8 => {
                let (i8_data, scale) =
                    Self::quantize_f16_slice_to_i8(shape, data, quantization.scale_override());
                self.set_i8_storage(i8_data, scale);
            }
            other => panic!(
                "quantized storage dtype {:?} is not implemented yet; currently only I8 is supported",
                other
            ),
        }
    }

    pub fn set_bf16_slice_with_quantization(
        &self,
        shape: &[usize],
        data: &[bf16],
        quantization: ParameterQuantization,
    ) {
        if self.try_quantize_into_existing_i8_storage_bf16(shape, data, quantization) {
            return;
        }
        let dtype = quantization
            .storage_dtype()
            .expect("enabled quantization must provide storage dtype");
        match dtype {
            DType::I8 => {
                let (i8_data, scale) =
                    Self::quantize_bf16_slice_to_i8(shape, data, quantization.scale_override());
                self.set_i8_storage(i8_data, scale);
            }
            other => panic!(
                "quantized storage dtype {:?} is not implemented yet; currently only I8 is supported",
                other
            ),
        }
    }

    pub fn set_i8_slice_with_dtype(&self, shape: &[usize], data: &[i8], scale: f32, dtype: DType) {
        match dtype {
            DType::F32 => {
                let mut inner = self.0.borrow_mut();
                Self::clear_cuda_storage(&mut inner);
                inner.data = Self::i8_slice_to_f32_shared(shape, data, scale);
                Self::clear_non_f32_storage(&mut inner);
                inner.has_f32_data = true;
                inner.storage_dtype = DType::F32;
                inner.cache_dirty = false;
                inner.grad = None;
                inner.cuda_f32_grad = None;
            }
            DType::F16 => {
                let mut inner = self.0.borrow_mut();
                Self::clear_cuda_storage(&mut inner);
                inner.data = Self::empty_f32_storage();
                inner.f16_data = Some(Self::i8_slice_to_f16_shared(shape, data, scale));
                inner.bf16_data = None;
                inner.i8_data = None;
                inner.i8_scale = None;
                inner.has_f32_data = false;
                inner.storage_dtype = DType::F16;
                inner.cache_dirty = false;
                inner.grad = None;
                inner.cuda_f32_grad = None;
            }
            DType::BF16 => {
                let mut inner = self.0.borrow_mut();
                Self::clear_cuda_storage(&mut inner);
                inner.data = Self::empty_f32_storage();
                inner.f16_data = None;
                inner.bf16_data = Some(Self::i8_slice_to_bf16_shared(shape, data, scale));
                inner.i8_data = None;
                inner.i8_scale = None;
                inner.has_f32_data = false;
                inner.storage_dtype = DType::BF16;
                inner.cache_dirty = false;
                inner.grad = None;
                inner.cuda_f32_grad = None;
            }
            DType::I8 => {
                if self.try_copy_into_existing_i8_storage_i8(shape, data, scale) {
                    return;
                }
                self.set_i8_storage(Self::i8_slice_to_shared(shape, data), scale);
            }
        }
    }

    pub fn set_array_f16_with_dtype(&self, data: ArrayD<f16>, dtype: DType) {
        let mut inner = self.0.borrow_mut();
        Self::clear_cuda_storage(&mut inner);
        match dtype {
            DType::F32 => {
                inner.data = Self::f16_arc_to_f32(&data.into_shared());
                Self::clear_non_f32_storage(&mut inner);
                inner.has_f32_data = true;
                inner.storage_dtype = DType::F32;
                inner.cache_dirty = false;
            }
            DType::F16 => {
                inner.data = Self::empty_f32_storage();
                inner.f16_data = Some(data.into_shared());
                inner.bf16_data = None;
                inner.i8_data = None;
                inner.i8_scale = None;
                inner.has_f32_data = false;
                inner.storage_dtype = DType::F16;
                inner.cache_dirty = false;
            }
            DType::BF16 => {
                let shared = data.into_shared();
                let f32_data = Self::f16_arc_to_f32(&shared);
                inner.data = Self::empty_f32_storage();
                inner.f16_data = None;
                inner.bf16_data = Some(Self::f32_arc_to_bf16(&f32_data));
                inner.i8_data = None;
                inner.i8_scale = None;
                inner.has_f32_data = false;
                inner.storage_dtype = DType::BF16;
                inner.cache_dirty = false;
            }
            DType::I8 => {
                let shared = data.into_shared();
                let f32_data = Self::f16_arc_to_f32(&shared).to_owned();
                let (i8_data, scale) = Self::quantize_f32_to_dtype(&f32_data, DType::I8, None);
                inner.data = Self::empty_f32_storage();
                inner.f16_data = None;
                inner.bf16_data = None;
                inner.i8_data = Some(i8_data);
                inner.i8_scale = Some(scale);
                inner.has_f32_data = false;
                inner.storage_dtype = DType::I8;
                inner.cache_dirty = false;
            }
        }
        inner.grad = None;
        inner.cuda_f32_grad = None;
    }

    pub fn set_array_bf16_with_dtype(&self, data: ArrayD<bf16>, dtype: DType) {
        let mut inner = self.0.borrow_mut();
        Self::clear_cuda_storage(&mut inner);
        match dtype {
            DType::F32 => {
                inner.data = Self::bf16_arc_to_f32(&data.into_shared());
                Self::clear_non_f32_storage(&mut inner);
                inner.has_f32_data = true;
                inner.storage_dtype = DType::F32;
                inner.cache_dirty = false;
            }
            DType::F16 => {
                let shared = data.into_shared();
                let f32_data = Self::bf16_arc_to_f32(&shared);
                inner.data = Self::empty_f32_storage();
                inner.f16_data = Some(Self::f32_arc_to_f16(&f32_data));
                inner.bf16_data = None;
                inner.i8_data = None;
                inner.i8_scale = None;
                inner.has_f32_data = false;
                inner.storage_dtype = DType::F16;
                inner.cache_dirty = false;
            }
            DType::BF16 => {
                inner.data = Self::empty_f32_storage();
                inner.f16_data = None;
                inner.bf16_data = Some(data.into_shared());
                inner.i8_data = None;
                inner.i8_scale = None;
                inner.has_f32_data = false;
                inner.storage_dtype = DType::BF16;
                inner.cache_dirty = false;
            }
            DType::I8 => {
                let shared = data.into_shared();
                let f32_data = Self::bf16_arc_to_f32(&shared).to_owned();
                let (i8_data, scale) = Self::quantize_f32_to_dtype(&f32_data, DType::I8, None);
                inner.data = Self::empty_f32_storage();
                inner.f16_data = None;
                inner.bf16_data = None;
                inner.i8_data = Some(i8_data);
                inner.i8_scale = Some(scale);
                inner.has_f32_data = false;
                inner.storage_dtype = DType::I8;
                inner.cache_dirty = false;
            }
        }
        inner.grad = None;
        inner.cuda_f32_grad = None;
    }

    pub fn set_array_i8_with_dtype(&self, data: ArrayD<i8>, scale: f32, dtype: DType) {
        let mut inner = self.0.borrow_mut();
        Self::clear_cuda_storage(&mut inner);
        match dtype {
            DType::F32 => {
                inner.data = Self::i8_arc_to_f32(&data.into_shared(), scale);
                Self::clear_non_f32_storage(&mut inner);
                inner.has_f32_data = true;
                inner.storage_dtype = DType::F32;
                inner.cache_dirty = false;
            }
            DType::F16 => {
                let shared = data.into_shared();
                inner.data = Self::empty_f32_storage();
                inner.f16_data = Some(Self::f32_arc_to_f16(&Self::i8_arc_to_f32(&shared, scale)));
                inner.bf16_data = None;
                inner.i8_data = None;
                inner.i8_scale = None;
                inner.has_f32_data = false;
                inner.storage_dtype = DType::F16;
                inner.cache_dirty = false;
            }
            DType::BF16 => {
                let shared = data.into_shared();
                inner.data = Self::empty_f32_storage();
                inner.f16_data = None;
                inner.bf16_data = Some(Self::f32_arc_to_bf16(&Self::i8_arc_to_f32(&shared, scale)));
                inner.i8_data = None;
                inner.i8_scale = None;
                inner.has_f32_data = false;
                inner.storage_dtype = DType::BF16;
                inner.cache_dirty = false;
            }
            DType::I8 => {
                inner.data = Self::empty_f32_storage();
                inner.f16_data = None;
                inner.bf16_data = None;
                inner.i8_data = Some(data.into_shared());
                inner.i8_scale = Some(scale);
                inner.has_f32_data = false;
                inner.storage_dtype = DType::I8;
                inner.cache_dirty = false;
            }
        }
        inner.grad = None;
        inner.cuda_f32_grad = None;
    }

    pub(crate) fn ensure_native_host_storage(&self) {
        let ready = {
            let inner = self.0.borrow();
            match inner.storage_dtype {
                DType::F32 => inner.has_f32_data,
                DType::F16 => inner.f16_data.is_some() && !inner.cache_dirty,
                DType::BF16 => inner.bf16_data.is_some() && !inner.cache_dirty,
                DType::I8 => inner.i8_data.is_some() && !inner.cache_dirty,
            }
        };
        if ready {
            return;
        }
        let snapshot = self.native_storage_owned();
        let mut inner = self.0.borrow_mut();
        match snapshot {
            TensorStorageOwned::F32(data) => {
                inner.data = data;
                inner.has_f32_data = true;
            }
            TensorStorageOwned::F16(data) => inner.f16_data = Some(data),
            TensorStorageOwned::BF16(data) => inner.bf16_data = Some(data),
            TensorStorageOwned::I8(data, scale) => {
                inner.i8_data = Some(data);
                inner.i8_scale = Some(scale);
            }
        }
        inner.cache_dirty = false;
    }

    pub(crate) fn native_storage_owned(&self) -> TensorStorageOwned {
        let inner = self.0.borrow();
        let shape = Self::logical_shape(&inner);
        // Snapshot the authoritative storage without changing dtype or graph metadata.
        if !inner.cache_dirty {
            match inner.storage_dtype {
                DType::F16 => {
                    if let Some(data) = &inner.f16_data {
                        return TensorStorageOwned::F16(data.clone());
                    }
                    if let Some(buffer) = &inner.cuda_f16_data {
                        let bits = crate::ops::cuda::download_u16_storage(buffer)
                            .expect("download native F16 snapshot");
                        return TensorStorageOwned::F16(
                            ArrayD::from_shape_vec(
                                IxDyn(shape),
                                bits.into_iter().map(f16::from_bits).collect(),
                            )
                            .expect("F16 snapshot shape")
                            .into_shared(),
                        );
                    }
                }
                DType::BF16 => {
                    if let Some(data) = &inner.bf16_data {
                        return TensorStorageOwned::BF16(data.clone());
                    }
                    if let Some(buffer) = &inner.cuda_bf16_data {
                        let bits = crate::ops::cuda::download_u16_storage(buffer)
                            .expect("download native BF16 snapshot");
                        return TensorStorageOwned::BF16(
                            ArrayD::from_shape_vec(
                                IxDyn(shape),
                                bits.into_iter().map(bf16::from_bits).collect(),
                            )
                            .expect("BF16 snapshot shape")
                            .into_shared(),
                        );
                    }
                }
                DType::I8 => {
                    let scale = inner.i8_scale.expect("I8 snapshot scale");
                    if let Some(data) = &inner.i8_data {
                        return TensorStorageOwned::I8(data.clone(), scale);
                    }
                    if let Some(buffer) = &inner.cuda_i8_data {
                        let bytes = crate::ops::cuda::download_i8_storage(buffer)
                            .expect("download native I8 snapshot");
                        return TensorStorageOwned::I8(
                            ArrayD::from_shape_vec(IxDyn(shape), bytes)
                                .expect("I8 snapshot shape")
                                .into_shared(),
                            scale,
                        );
                    }
                }
                DType::F32 => {}
            }
        }
        let data = if inner.has_f32_data {
            inner.data.clone()
        } else if let Some(buffer) = &inner.cuda_f32_data {
            let values = crate::ops::cuda::download_f32(buffer).expect("download F32 snapshot");
            ArrayD::from_shape_vec(IxDyn(shape), values)
                .expect("F32 snapshot shape")
                .into_shared()
        } else {
            assert!(
                inner.data.is_empty(),
                "Tensor has no materializable storage"
            );
            inner.data.clone()
        };
        match inner.storage_dtype {
            DType::F32 => TensorStorageOwned::F32(data),
            DType::F16 => TensorStorageOwned::F16(Self::f32_arc_to_f16(&data)),
            DType::BF16 => TensorStorageOwned::BF16(Self::f32_arc_to_bf16(&data)),
            DType::I8 => {
                let scale = inner.i8_scale.expect("I8 snapshot scale");
                let (data, _) =
                    Self::quantize_f32_to_dtype(&data.to_owned(), DType::I8, Some(scale));
                TensorStorageOwned::I8(data, scale)
            }
        }
    }
}
