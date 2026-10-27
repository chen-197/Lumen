use super::*;

impl Tensor {
    pub fn export_raw(&self) -> (Vec<usize>, DType, TensorRawData) {
        match self.native_storage_owned() {
            TensorStorageOwned::F32(data) => (
                data.shape().to_vec(),
                DType::F32,
                TensorRawData::F32(data.iter().copied().collect()),
            ),
            TensorStorageOwned::F16(data) => (
                data.shape().to_vec(),
                DType::F16,
                TensorRawData::F16(data.iter().map(|v| v.to_bits()).collect()),
            ),
            TensorStorageOwned::BF16(data) => (
                data.shape().to_vec(),
                DType::BF16,
                TensorRawData::BF16(data.iter().map(|v| v.to_bits()).collect()),
            ),
            TensorStorageOwned::I8(data, scale) => (
                data.shape().to_vec(),
                DType::I8,
                TensorRawData::I8 {
                    values: data.iter().copied().collect(),
                    scale,
                },
            ),
        }
    }

    pub(crate) fn validate_raw(
        shape: &[usize],
        dtype: DType,
        raw: &TensorRawData,
    ) -> Result<(), String> {
        let expected = shape
            .iter()
            .try_fold(1usize, |n, &dim| n.checked_mul(dim))
            .ok_or_else(|| "Checkpoint shape element count overflow".to_string())?;
        let (actual_dtype, len) = match raw {
            TensorRawData::F32(v) => (DType::F32, v.len()),
            TensorRawData::F16(v) => (DType::F16, v.len()),
            TensorRawData::BF16(v) => (DType::BF16, v.len()),
            TensorRawData::I8 { values, scale } => {
                if !scale.is_finite() || *scale <= 0.0 {
                    return Err(
                        "Checkpoint I8 scale must be finite and greater than zero".to_string()
                    );
                }
                (DType::I8, values.len())
            }
        };
        if actual_dtype != dtype {
            let payload = match actual_dtype {
                DType::F32 => "f32",
                DType::F16 => "f16",
                DType::BF16 => "bf16",
                DType::I8 => "i8",
            };
            return Err(format!(
                "Raw checkpoint payload mismatch: dtype {dtype:?} with {payload} data"
            ));
        }
        if expected != len {
            return Err(format!(
                "Checkpoint shape mismatch: expected {expected} values, got {len}"
            ));
        }
        Ok(())
    }

    pub fn import_raw(
        &self,
        shape: Vec<usize>,
        dtype: DType,
        raw: TensorRawData,
    ) -> Result<(), String> {
        Self::validate_raw(&shape, dtype, &raw)?;
        match raw {
            TensorRawData::F32(values) => {
                if dtype != DType::F32 {
                    return Err(format!(
                        "Raw checkpoint payload mismatch: dtype {:?} with f32 data",
                        dtype
                    ));
                }
                let data = Array::from_shape_vec(IxDyn(&shape), values)
                    .map_err(|e| format!("Checkpoint shape mismatch for f32 payload: {}", e))?;
                self.set_array_f32_with_dtype(data, DType::F32);
            }
            TensorRawData::F16(values) => {
                if dtype != DType::F16 {
                    return Err(format!(
                        "Raw checkpoint payload mismatch: dtype {:?} with f16 data",
                        dtype
                    ));
                }
                let f16_values = values.into_iter().map(f16::from_bits).collect::<Vec<_>>();
                let data = Array::from_shape_vec(IxDyn(&shape), f16_values)
                    .map_err(|e| format!("Checkpoint shape mismatch for f16 payload: {}", e))?;
                self.set_array_f16_with_dtype(data, DType::F16);
            }
            TensorRawData::BF16(values) => {
                if dtype != DType::BF16 {
                    return Err(format!(
                        "Raw checkpoint payload mismatch: dtype {:?} with bf16 data",
                        dtype
                    ));
                }
                let bf16_values = values.into_iter().map(bf16::from_bits).collect::<Vec<_>>();
                let data = Array::from_shape_vec(IxDyn(&shape), bf16_values)
                    .map_err(|e| format!("Checkpoint shape mismatch for bf16 payload: {}", e))?;
                self.set_array_bf16_with_dtype(data, DType::BF16);
            }
            TensorRawData::I8 { values, scale } => {
                if dtype != DType::I8 {
                    return Err(format!(
                        "Raw checkpoint payload mismatch: dtype {:?} with i8 data",
                        dtype
                    ));
                }
                let data = Array::from_shape_vec(IxDyn(&shape), values)
                    .map_err(|e| format!("Checkpoint shape mismatch for i8 payload: {}", e))?;
                self.set_array_i8_with_dtype(data, scale, DType::I8);
            }
        }
        Ok(())
    }

    pub fn get_raw_data(&self) -> (Vec<usize>, Vec<f32>) {
        let data = self.data();
        (data.shape().to_vec(), data.iter().copied().collect())
    }

    pub fn take_raw_data(&self) -> (Vec<usize>, Vec<f32>) {
        let (shape, raw_data) = self.get_raw_data();
        let mut inner = self.0.borrow_mut();
        inner.data = Self::empty_f32_storage();
        Self::clear_non_f32_storage(&mut inner);
        Self::clear_cuda_storage(&mut inner);
        inner.has_f32_data = true;
        inner.storage_dtype = DType::F32;
        inner.cache_dirty = false;
        inner.grad = None;
        inner.cuda_f32_grad = None;
        (shape, raw_data)
    }

    pub fn set_raw_data(&self, shape: Vec<usize>, raw_data: Vec<f32>) {
        let new_data = Array::from_shape_vec(shape, raw_data).unwrap().into_dyn();
        self.set_array_f32_with_dtype(new_data, DType::F32);
    }
}
