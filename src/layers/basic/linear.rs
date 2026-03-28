// src/layers/linear.rs
use crate::autograd::{Tensor, is_no_grad};
use crate::inference::{InferenceWeightStorage, WeightDType};
use crate::init::{tensor_init, InitType};
use crate::module::Module;
use crate::ops::matmul::{
    matmul,
    matmul_prefill_rowmajor_bf16,
    matvec_rowmajor_parallel,
    matvec_rowmajor_parallel_bf16,
    matvec_rowmajor_serial,
    matvec_rowmajor_serial_bf16,
};
use ndarray::{Array2, ArrayD, Axis, Ix1, IxDyn};
use rayon::prelude::*;

pub struct Linear {
    pub weight: Tensor,       // shape: [out_features, in_features]
    pub bias: Option<Tensor>, // shape: [out_features]
    pub in_features: usize,
    pub out_features: usize,
    // 临时：推理模式专用压缩权重。
    pub infer_weight: Option<InferenceWeightStorage>,
    // 临时：原训练权重已迁移，训练路径禁用。
    pub infer_weight_migrated: bool,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        let weight = tensor_init(vec![out_features, in_features], InitType::KaimingNormal);
        let bias = tensor_init(vec![out_features], InitType::Zeros);

        Linear {
            weight,
            bias: Some(bias),
            in_features,
            out_features,
            infer_weight: None,
            infer_weight_migrated: false,
        }
    }

    pub fn new_no_bias(in_features: usize, out_features: usize) -> Self {
        let weight = tensor_init(vec![out_features, in_features], InitType::KaimingNormal);

        Linear {
            weight,
            bias: None,
            in_features,
            out_features,
            infer_weight: None,
            infer_weight_migrated: false,
        }
    }

    // 临时：推理压缩权重准备接口，后续由全链路 dtype 替代。
    pub fn prepare_infer_weight_tmp(&mut self, dtype: WeightDType, migrate_primary_weight: bool) {
        let (shape, data) = if migrate_primary_weight {
            self.weight.take_raw_data()
        } else {
            self.weight.get_raw_data()
        };
        self.infer_weight = Some(InferenceWeightStorage::from_f32_data(shape, data, dtype));
        self.infer_weight_migrated = migrate_primary_weight;
        if migrate_primary_weight {
            self.weight.0.borrow_mut().requires_grad = false;
        }
    }

    #[inline]
    pub fn load_infer_weight_bf16_direct(&mut self, shape: Vec<usize>, data: Vec<u16>) {
        assert_eq!(
            shape.as_slice(),
            &[self.out_features, self.in_features],
            "Linear BF16 infer weight shape mismatch"
        );

        self.infer_weight = Some(InferenceWeightStorage::BF16 { shape, data });
        self.infer_weight_migrated = true;

        let _ = self.weight.take_raw_data();
        self.weight.0.borrow_mut().requires_grad = false;
    }

    #[inline]
    pub fn has_migrated_infer_weight_tmp(&self) -> bool {
        self.infer_weight_migrated
    }

    #[inline]
    pub fn has_infer_weight_tmp(&self) -> bool {
        self.infer_weight.is_some()
    }

    #[inline]
    fn bias_slice_owned(&self) -> Option<Vec<f32>> {
        self.bias.as_ref().map(|bias| {
            let bias_guard = bias.data_ref();
            let bias_1d = bias_guard
                .view()
                .into_dimensionality::<Ix1>()
                .expect("Linear bias must be 1D [out]");
            if let Some(s) = bias_1d.as_slice() {
                s.to_vec()
            } else {
                bias_1d.iter().copied().collect()
            }
        })
    }

    fn forward_infer_tmp_f32(&self, input: Tensor, weight: &[f32]) -> Tensor {
        assert!(is_no_grad(), "forward_infer_tmp_f32 is inference-only");

        let x_data = input.data_arc();
        let x_shape = x_data.shape().to_vec();
        let k_dim = *x_shape.last().expect("input must have last dim");
        assert_eq!(k_dim, self.in_features, "input width mismatch");
        let m_dim = x_data.len() / k_dim;
        let x_2d = x_data
            .view()
            .into_shape((m_dim, k_dim))
            .expect("input reshape failed");

        let mut out = Array2::<f32>::zeros((m_dim, self.out_features));
        let bias_vec = self.bias_slice_owned();

        if m_dim == 1 {
            let x_row = x_2d.row(0);
            let x_owned;
            let x_slice: &[f32] = if let Some(s) = x_row.as_slice() {
                s
            } else {
                x_owned = x_row.iter().copied().collect::<Vec<f32>>();
                x_owned.as_slice()
            };
            let mut out_row = out.row_mut(0);
            let out_slice = out_row.as_slice_mut().expect("output row should be contiguous");
            matvec_rowmajor_parallel(x_slice, weight, self.out_features, self.in_features, out_slice);
            if let Some(bias) = &bias_vec {
                for (dst, &b) in out_slice.iter_mut().zip(bias.iter()) {
                    *dst += b;
                }
            }
        } else {
            let x_owned_storage;
            let x_flat: &[f32] = if let Some(s) = x_2d.as_slice() {
                s
            } else {
                x_owned_storage = x_2d.iter().copied().collect::<Vec<f32>>();
                x_owned_storage.as_slice()
            };
            let out_flat = out.as_slice_mut().expect("batched linear output should be contiguous");
            let out_features = self.out_features;
            let in_features = self.in_features;
            let bias_slice = bias_vec.as_deref();

            out_flat
                .par_chunks_mut(out_features)
                .zip(x_flat.par_chunks(k_dim))
                .for_each(|(out_row, x_slice)| {
                    matvec_rowmajor_serial(x_slice, weight, out_features, in_features, out_row);
                    if let Some(bias) = bias_slice {
                        for (dst, &b) in out_row.iter_mut().zip(bias.iter()) {
                            *dst += b;
                        }
                    }
                });
        }

        let mut out_shape = x_shape;
        let last = out_shape.len() - 1;
        out_shape[last] = self.out_features;
        Tensor::from_array_no_grad(out.into_shape(out_shape).unwrap().into_dyn())
    }

    fn forward_infer_tmp_bf16(&self, input: Tensor, weight: &[half::bf16]) -> Tensor {
        assert!(is_no_grad(), "forward_infer_tmp_bf16 is inference-only");

        let x_data = input.data_arc();
        let x_shape = x_data.shape().to_vec();
        let k_dim = *x_shape.last().expect("input must have last dim");
        assert_eq!(k_dim, self.in_features, "input width mismatch");
        let m_dim = x_data.len() / k_dim;
        let x_2d = x_data
            .view()
            .into_shape((m_dim, k_dim))
            .expect("input reshape failed");

        let mut out = Array2::<f32>::zeros((m_dim, self.out_features));
        let bias_vec = self.bias_slice_owned();

        if m_dim == 1 {
            let x_row = x_2d.row(0);
            let x_owned;
            let x_slice: &[f32] = if let Some(s) = x_row.as_slice() {
                s
            } else {
                x_owned = x_row.iter().copied().collect::<Vec<f32>>();
                x_owned.as_slice()
            };
            let mut out_row = out.row_mut(0);
            let out_slice = out_row.as_slice_mut().expect("output row should be contiguous");
            matvec_rowmajor_parallel_bf16(x_slice, weight, self.out_features, self.in_features, out_slice);
            if let Some(bias) = &bias_vec {
                for (dst, &b) in out_slice.iter_mut().zip(bias.iter()) {
                    *dst += b;
                }
            }
        } else {
            let x_owned_storage;
            let x_flat: &[f32] = if let Some(s) = x_2d.as_slice() {
                s
            } else {
                x_owned_storage = x_2d.iter().copied().collect::<Vec<f32>>();
                x_owned_storage.as_slice()
            };
            let out_flat = out.as_slice_mut().expect("batched linear output should be contiguous");
            matmul_prefill_rowmajor_bf16(
                x_flat,
                weight,
                m_dim,
                self.out_features,
                self.in_features,
                out_flat,
                bias_vec.as_deref(),
            );
        }

        let mut out_shape = x_shape;
        let last = out_shape.len() - 1;
        out_shape[last] = self.out_features;
        Tensor::from_array_no_grad(out.into_shape(out_shape).unwrap().into_dyn())
    }

    #[inline]
    pub fn forward_decode_tmp_f32_into(&self, input: &[f32], weight: &[f32], out: &mut [f32]) {
        assert!(is_no_grad(), "forward_decode_tmp_f32_into is inference-only");
        assert!(self.bias.is_none(), "forward_decode_tmp_f32_into currently expects no bias");
        assert_eq!(input.len(), self.in_features, "input width mismatch");
        assert_eq!(out.len(), self.out_features, "output width mismatch");

        matvec_rowmajor_parallel(input, weight, self.out_features, self.in_features, out);
    }

    #[inline]
    pub fn forward_decode_tmp_bf16_into(&self, input: &[f32], weight: &[half::bf16], out: &mut [f32]) {
        assert!(is_no_grad(), "forward_decode_tmp_bf16_into is inference-only");
        assert!(self.bias.is_none(), "forward_decode_tmp_bf16_into currently expects no bias");
        assert_eq!(input.len(), self.in_features, "input width mismatch");
        assert_eq!(out.len(), self.out_features, "output width mismatch");

        matvec_rowmajor_parallel_bf16(input, weight, self.out_features, self.in_features, out);
    }

    #[inline]
    pub fn forward_decode_slice_no_bias_into(&self, input: &[f32], out: &mut [f32]) {
        assert!(is_no_grad(), "forward_decode_slice_no_bias_into is inference-only");
        assert!(self.bias.is_none(), "forward_decode_slice_no_bias_into currently expects no bias");
        assert_eq!(input.len(), self.in_features, "input width mismatch");
        assert_eq!(out.len(), self.out_features, "output width mismatch");

        if let Some(infer) = self.infer_weight.as_ref() {
            match infer {
                InferenceWeightStorage::F32 { data, .. } => self.forward_decode_tmp_f32_into(input, data.as_slice(), out),
                InferenceWeightStorage::BF16 { data, .. } => {
                    let weight = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const half::bf16, data.len()) };
                    self.forward_decode_tmp_bf16_into(input, weight, out)
                }
            }
            return;
        }

        let weight_guard = self.weight.data_ref();
        let weight2 = weight_guard
            .view()
            .into_dimensionality::<ndarray::Ix2>()
            .expect("Linear weight must be 2D [out,in]");

        let weight_owned;
        let weight_slice: &[f32] = if let Some(s) = weight2.as_slice() {
            s
        } else {
            weight_owned = weight2.iter().copied().collect::<Vec<f32>>();
            weight_owned.as_slice()
        };

        matvec_rowmajor_parallel(input, weight_slice, self.out_features, self.in_features, out);
    }

    pub fn forward_decode_slice_no_bias(&self, input: &[f32]) -> Tensor {
        assert!(is_no_grad(), "forward_decode_slice_no_bias is inference-only");
        assert!(self.bias.is_none(), "forward_decode_slice_no_bias currently expects no bias");
        assert_eq!(input.len(), self.in_features, "input width mismatch");

        let mut data = ArrayD::<f32>::zeros(IxDyn(&[1, 1, self.out_features])).into_shared();
        let out_slice = data
            .as_slice_mut()
            .expect("decode linear output should be contiguous");
        self.forward_decode_slice_no_bias_into(input, out_slice);
        Tensor::from_data_no_grad(data)
    }
}

impl Module for Linear {
    fn forward(&self, input: Tensor) -> Tensor {
        if !is_no_grad() && self.infer_weight_migrated {
            panic!("temporary migrated infer_weight disables linear training path");
        }
        if is_no_grad() {
            if let Some(infer) = self.infer_weight.as_ref() {
                return match infer {
                    InferenceWeightStorage::F32 { data, .. } => self.forward_infer_tmp_f32(input, data.as_slice()),
                    InferenceWeightStorage::BF16 { data, .. } => {
                        let weight = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const half::bf16, data.len()) };
                        self.forward_infer_tmp_bf16(input, weight)
                    }
                };
            }
        }

        let y = matmul(&input, &self.weight);

        if let Some(bias) = &self.bias {
            if is_no_grad() {
                let bias_guard = bias.data_ref();
                let bias_1d = bias_guard
                    .view()
                    .into_dimensionality::<Ix1>()
                    .expect("Linear bias must be 1D [out]");
                let bias_owned;
                let bias_slice: &[f32] = if let Some(s) = bias_1d.as_slice() {
                    s
                } else {
                    bias_owned = bias_1d.iter().copied().collect::<Vec<f32>>();
                    bias_owned.as_slice()
                };

                {
                    let mut y_data = y.data_mut();
                    let last_axis = Axis(y_data.ndim() - 1);
                    for mut lane in y_data.lanes_mut(last_axis) {
                        for (dst, &b) in lane.iter_mut().zip(bias_slice.iter()) {
                            *dst += b;
                        }
                    }
                }
                y
            } else {
                y + bias.clone()
            }
        } else {
            y
        }
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = vec![self.weight.clone()];
        if let Some(b) = &self.bias {
            params.push(b.clone());
        }
        params
    }

    fn has_temporary_infer_migration(&self) -> bool {
        self.infer_weight_migrated
    }
}
