use half::bf16;
use rayon::prelude::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeightDType {
    F32,
    BF16,
}

#[derive(Clone, Debug)]
pub enum InferenceWeightStorage {
    F32 { shape: Vec<usize>, data: Vec<f32> },
    BF16 { shape: Vec<usize>, data: Vec<u16> },
}

impl InferenceWeightStorage {
    pub fn from_f32_data(shape: Vec<usize>, data: Vec<f32>, dtype: WeightDType) -> Self {
        match dtype {
            WeightDType::F32 => Self::from_f32_data_f32(shape, data),
            WeightDType::BF16 => Self::from_f32_data_bf16(shape, data),
        }
    }

    #[inline]
    pub fn from_f32_data_f32(shape: Vec<usize>, data: Vec<f32>) -> Self {
        Self::F32 { shape, data }
    }

    #[inline]
    pub fn from_f32_data_bf16(shape: Vec<usize>, data: Vec<f32>) -> Self {
        Self::BF16 {
            shape,
            data: data.into_par_iter().map(|x| bf16::from_f32(x).to_bits()).collect(),
        }
    }

    #[inline]
    pub fn shape(&self) -> &[usize] {
        match self {
            Self::F32 { shape, .. } | Self::BF16 { shape, .. } => shape.as_slice(),
        }
    }

    #[inline]
    pub fn rows_cols(&self) -> (usize, usize) {
        let shape = self.shape();
        assert_eq!(shape.len(), 2, "temporary inference weight must be 2D");
        (shape[0], shape[1])
    }

    #[inline]
    pub fn as_f32(&self) -> Option<&[f32]> {
        match self {
            Self::F32 { data, .. } => Some(data.as_slice()),
            _ => None,
        }
    }

    #[inline]
    pub fn as_bf16(&self) -> Option<&[bf16]> {
        match self {
            Self::BF16 { data, .. } => Some(unsafe {
                std::slice::from_raw_parts(data.as_ptr() as *const bf16, data.len())
            }),
            _ => None,
        }
    }

    #[inline]
    pub fn as_bf16_bits(&self) -> Option<&[u16]> {
        match self {
            Self::BF16 { data, .. } => Some(data.as_slice()),
            _ => None,
        }
    }
}

