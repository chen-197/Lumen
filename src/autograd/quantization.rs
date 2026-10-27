use super::*;
use crate::precision::I8Requantizer;

impl Tensor {
    pub(super) fn quantize_f32_to_i8(
        data: &ArrayD<f32>,
        scale_override: Option<f32>,
    ) -> (ArcArray<i8, IxDyn>, f32) {
        let shape = data.shape().to_vec();
        let max_abs = Self::finite_max_abs(data.iter().copied());
        let scale = if let Some(scale) = scale_override {
            assert!(
                scale.is_finite() && scale > 0.0,
                "quantization scale must be finite and > 0, got {}",
                scale
            );
            scale
        } else {
            if max_abs > 0.0 {
                (max_abs / 127.0).max(f32::MIN_POSITIVE)
            } else {
                1.0
            }
        };
        let quantizer = I8Requantizer::new(scale);
        let raw = data
            .iter()
            .map(|&v| quantizer.quantize(v))
            .collect::<Vec<_>>();
        (
            Array::from_shape_vec(IxDyn(&shape), raw)
                .expect("Failed to convert f32 array to i8 storage")
                .into_shared(),
            scale,
        )
    }

    pub(super) fn quantize_f32_values_into_i8_slice(
        data: &[f32],
        scale_override: Option<f32>,
        dst: &mut [i8],
    ) -> f32 {
        assert_eq!(data.len(), dst.len(), "quantized dst len mismatch");
        let max_abs = Self::finite_max_abs(data.iter().copied());
        let scale = if let Some(scale) = scale_override {
            assert!(
                scale.is_finite() && scale > 0.0,
                "quantization scale must be finite and > 0, got {}",
                scale
            );
            scale
        } else {
            if max_abs > 0.0 {
                (max_abs / 127.0).max(f32::MIN_POSITIVE)
            } else {
                1.0
            }
        };
        let quantizer = I8Requantizer::new(scale);
        for (src, dst) in data.iter().zip(dst.iter_mut()) {
            *dst = quantizer.quantize(*src);
        }
        scale
    }

    pub(super) fn quantize_f16_values_into_i8_slice(
        data: &[f16],
        scale_override: Option<f32>,
        dst: &mut [i8],
    ) -> f32 {
        assert_eq!(data.len(), dst.len(), "quantized dst len mismatch");
        let max_abs = Self::finite_max_abs(data.iter().map(|&v| v.to_f32()));
        let scale = if let Some(scale) = scale_override {
            assert!(
                scale.is_finite() && scale > 0.0,
                "quantization scale must be finite and > 0, got {}",
                scale
            );
            scale
        } else {
            if max_abs > 0.0 {
                (max_abs / 127.0).max(f32::MIN_POSITIVE)
            } else {
                1.0
            }
        };
        let quantizer = I8Requantizer::new(scale);
        for (src, dst) in data.iter().zip(dst.iter_mut()) {
            *dst = quantizer.quantize(src.to_f32());
        }
        scale
    }

    pub(super) fn quantize_bf16_values_into_i8_slice(
        data: &[bf16],
        scale_override: Option<f32>,
        dst: &mut [i8],
    ) -> f32 {
        assert_eq!(data.len(), dst.len(), "quantized dst len mismatch");
        let max_abs = Self::finite_max_abs(data.iter().map(|&v| v.to_f32()));
        let scale = if let Some(scale) = scale_override {
            assert!(
                scale.is_finite() && scale > 0.0,
                "quantization scale must be finite and > 0, got {}",
                scale
            );
            scale
        } else {
            if max_abs > 0.0 {
                (max_abs / 127.0).max(f32::MIN_POSITIVE)
            } else {
                1.0
            }
        };
        let quantizer = I8Requantizer::new(scale);
        for (src, dst) in data.iter().zip(dst.iter_mut()) {
            *dst = quantizer.quantize(src.to_f32());
        }
        scale
    }

    pub(super) fn finite_max_abs(values: impl Iterator<Item = f32>) -> f32 {
        let mut max_abs = 0.0f32;
        for value in values {
            assert!(
                value.is_finite(),
                "cannot quantize non-finite value to I8: {value}"
            );
            max_abs = max_abs.max(value.abs());
        }
        max_abs
    }

    pub(super) fn quantize_f32_slice_to_i8(
        shape: &[usize],
        data: &[f32],
        scale_override: Option<f32>,
    ) -> (ArcArray<i8, IxDyn>, f32) {
        let mut raw = vec![0i8; data.len()];
        let scale = Self::quantize_f32_values_into_i8_slice(data, scale_override, &mut raw);
        (
            Array::from_shape_vec(IxDyn(shape), raw)
                .expect("Failed to convert f32 slice to i8 storage")
                .into_shared(),
            scale,
        )
    }

    pub(super) fn quantize_f16_slice_to_i8(
        shape: &[usize],
        data: &[f16],
        scale_override: Option<f32>,
    ) -> (ArcArray<i8, IxDyn>, f32) {
        let mut raw = vec![0i8; data.len()];
        let scale = Self::quantize_f16_values_into_i8_slice(data, scale_override, &mut raw);
        (
            Array::from_shape_vec(IxDyn(shape), raw)
                .expect("Failed to convert f16 slice to i8 storage")
                .into_shared(),
            scale,
        )
    }

    pub(super) fn quantize_bf16_slice_to_i8(
        shape: &[usize],
        data: &[bf16],
        scale_override: Option<f32>,
    ) -> (ArcArray<i8, IxDyn>, f32) {
        let mut raw = vec![0i8; data.len()];
        let scale = Self::quantize_bf16_values_into_i8_slice(data, scale_override, &mut raw);
        (
            Array::from_shape_vec(IxDyn(shape), raw)
                .expect("Failed to convert bf16 slice to i8 storage")
                .into_shared(),
            scale,
        )
    }

    pub(super) fn i8_slice_to_shared(shape: &[usize], data: &[i8]) -> ArcArray<i8, IxDyn> {
        Array::from_shape_vec(IxDyn(shape), data.to_vec())
            .expect("Failed to convert i8 slice to i8 storage")
            .into_shared()
    }

    pub(super) fn i8_slice_to_f32_shared(
        shape: &[usize],
        data: &[i8],
        scale: f32,
    ) -> ArcArray<f32, IxDyn> {
        let raw = data.iter().map(|&v| (v as f32) * scale).collect::<Vec<_>>();
        Array::from_shape_vec(IxDyn(shape), raw)
            .expect("Failed to convert i8 slice to f32 storage")
            .into_shared()
    }

    pub(super) fn i8_slice_to_f16_shared(
        shape: &[usize],
        data: &[i8],
        scale: f32,
    ) -> ArcArray<f16, IxDyn> {
        let raw = data
            .iter()
            .map(|&v| f16::from_f32((v as f32) * scale))
            .collect::<Vec<_>>();
        Array::from_shape_vec(IxDyn(shape), raw)
            .expect("Failed to convert i8 slice to f16 storage")
            .into_shared()
    }

    pub(super) fn i8_slice_to_bf16_shared(
        shape: &[usize],
        data: &[i8],
        scale: f32,
    ) -> ArcArray<bf16, IxDyn> {
        let raw = data
            .iter()
            .map(|&v| bf16::from_f32((v as f32) * scale))
            .collect::<Vec<_>>();
        Array::from_shape_vec(IxDyn(shape), raw)
            .expect("Failed to convert i8 slice to bf16 storage")
            .into_shared()
    }

    pub(super) fn quantize_f32_to_dtype(
        data: &ArrayD<f32>,
        dtype: DType,
        scale_override: Option<f32>,
    ) -> (ArcArray<i8, IxDyn>, f32) {
        match dtype {
            DType::I8 => Self::quantize_f32_to_i8(data, scale_override),
            other => {
                panic!(
                    "quantized storage dtype {:?} is not implemented yet; currently only I8 is supported",
                    other
                )
            }
        }
    }

    pub(super) fn f32_array_to_bf16(data: &ArrayD<f32>) -> ArcArray<bf16, IxDyn> {
        let shape = data.shape().to_vec();
        let raw = if let Some(slice) = data.as_slice() {
            let mut raw = vec![bf16::from_bits(0); slice.len()];
            raw.convert_from_f32_slice(slice);
            raw
        } else {
            data.iter().map(|&v| bf16::from_f32(v)).collect::<Vec<_>>()
        };
        Array::from_shape_vec(IxDyn(&shape), raw)
            .expect("Failed to convert f32 array to bf16 storage")
            .into_shared()
    }

    pub(super) fn f32_array_to_f16(data: &ArrayD<f32>) -> ArcArray<f16, IxDyn> {
        let shape = data.shape().to_vec();
        let raw = if let Some(slice) = data.as_slice() {
            let mut raw = vec![f16::from_bits(0); slice.len()];
            raw.convert_from_f32_slice(slice);
            raw
        } else {
            data.iter().map(|&v| f16::from_f32(v)).collect::<Vec<_>>()
        };
        Array::from_shape_vec(IxDyn(&shape), raw)
            .expect("Failed to convert f32 array to f16 storage")
            .into_shared()
    }

    pub(super) fn f32_arc_to_f16(data: &ArcArray<f32, IxDyn>) -> ArcArray<f16, IxDyn> {
        let shape = data.shape().to_vec();
        let raw = if let Some(slice) = data.as_slice() {
            let mut raw = vec![f16::from_bits(0); slice.len()];
            raw.convert_from_f32_slice(slice);
            raw
        } else {
            data.iter().map(|&v| f16::from_f32(v)).collect::<Vec<_>>()
        };
        Array::from_shape_vec(IxDyn(&shape), raw)
            .expect("Failed to convert f32 array to f16 storage")
            .into_shared()
    }

    pub(super) fn f32_arc_to_bf16(data: &ArcArray<f32, IxDyn>) -> ArcArray<bf16, IxDyn> {
        let shape = data.shape().to_vec();
        let raw = if let Some(slice) = data.as_slice() {
            let mut raw = vec![bf16::from_bits(0); slice.len()];
            raw.convert_from_f32_slice(slice);
            raw
        } else {
            data.iter().map(|&v| bf16::from_f32(v)).collect::<Vec<_>>()
        };
        Array::from_shape_vec(IxDyn(&shape), raw)
            .expect("Failed to convert f32 array to bf16 storage")
            .into_shared()
    }

    pub(super) fn bf16_arc_to_f32(data: &ArcArray<bf16, IxDyn>) -> ArcArray<f32, IxDyn> {
        let shape = data.shape().to_vec();
        let raw = if let Some(slice) = data.as_slice() {
            let mut raw = vec![0.0f32; slice.len()];
            slice.convert_to_f32_slice(&mut raw);
            raw
        } else {
            data.iter().map(|&v| v.to_f32()).collect::<Vec<_>>()
        };
        Array::from_shape_vec(IxDyn(&shape), raw)
            .expect("Failed to convert bf16 storage to f32")
            .into_shared()
    }

    pub(super) fn f16_arc_to_f32(data: &ArcArray<f16, IxDyn>) -> ArcArray<f32, IxDyn> {
        let shape = data.shape().to_vec();
        let raw = if let Some(slice) = data.as_slice() {
            let mut raw = vec![0.0f32; slice.len()];
            slice.convert_to_f32_slice(&mut raw);
            raw
        } else {
            data.iter().map(|&v| v.to_f32()).collect::<Vec<_>>()
        };
        Array::from_shape_vec(IxDyn(&shape), raw)
            .expect("Failed to convert f16 storage to f32")
            .into_shared()
    }

    pub(super) fn i8_arc_to_f32(data: &ArcArray<i8, IxDyn>, scale: f32) -> ArcArray<f32, IxDyn> {
        let shape = data.shape().to_vec();
        let raw = data.iter().map(|&v| (v as f32) * scale).collect::<Vec<_>>();
        Array::from_shape_vec(IxDyn(&shape), raw)
            .expect("Failed to convert i8 storage to f32")
            .into_shared()
    }

    pub fn quantize_inplace(&self, dtype: DType) {
        assert!(
            dtype.is_integer(),
            "quantize_inplace currently expects integer dtype, got {:?}",
            dtype
        );
        self.cast_inplace(dtype);
    }

    pub fn quantize_inplace_with_quantization(&self, quantization: ParameterQuantization) {
        if !quantization.is_enabled() {
            return;
        }
        let dtype = quantization
            .storage_dtype()
            .expect("enabled quantization must provide storage dtype");
        assert!(
            dtype.is_integer(),
            "quantize_inplace_with_quantization currently expects integer dtype, got {:?}",
            dtype
        );
        self.ensure_f32_data(false);
        let data = self.data_ref().to_owned();
        self.set_array_f32_with_quantization(data, quantization);
    }

    pub fn dequantize_inplace(&self, dtype: DType) {
        assert!(
            dtype.is_float(),
            "dequantize_inplace currently expects floating dtype, got {:?}",
            dtype
        );
        self.cast_inplace(dtype);
    }
}
