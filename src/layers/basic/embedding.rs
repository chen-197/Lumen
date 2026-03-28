use crate::autograd::{is_no_grad, Tensor, TensorData};
use crate::inference::{InferenceWeightStorage, WeightDType};
use crate::init::{tensor_init, InitType};
use crate::module::Module;
use ndarray::{Array, Zip};
use half::bf16;
use std::cell::RefCell;
use std::ops::AddAssign;
use std::rc::Rc;

pub struct Embedding {
    pub weight: Tensor,
    pub vocab_size: usize,
    pub embed_dim: usize,
    // 临时：推理模式专用压缩权重。
    pub infer_weight: Option<InferenceWeightStorage>,
    // 临时：原训练权重已迁移，训练路径禁用。
    pub infer_weight_migrated: bool,
}

impl Embedding {
    pub fn new(vocab_size: usize, embed_dim: usize) -> Self {
        let weight = tensor_init(vec![vocab_size, embed_dim], InitType::KaimingNormal);
        Self {
            weight,
            vocab_size,
            embed_dim,
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
            &[self.vocab_size, self.embed_dim],
            "Embedding BF16 infer weight shape mismatch"
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

    fn forward_infer_tmp_f32(&self, indices: &Tensor, weight: &[f32]) -> Tensor {
        assert!(is_no_grad(), "forward_infer_tmp_f32 is inference-only");
        let idx_data = indices.data_arc();
        let e_dim = self.embed_dim;
        let v_size = self.vocab_size;

        let mut out_shape = idx_data.shape().to_vec();
        out_shape.push(e_dim);
        let mut out = Array::zeros(out_shape);

        let num_elements = idx_data.len();
        let idx_flat = idx_data
            .view()
            .into_shape(num_elements)
            .expect("Flatten indices failed");
        let mut out_flat = out
            .view_mut()
            .into_shape((num_elements, e_dim))
            .expect("Flatten output failed");

        Zip::from(out_flat.outer_iter_mut())
            .and(&idx_flat)
            .par_for_each(|mut out_row, &idx_f32| {
                let idx = idx_f32 as usize;
                if idx >= v_size {
                    panic!("Embedding index out of bounds: {} >= {}", idx, v_size);
                }
                let row = &weight[idx * e_dim..(idx + 1) * e_dim];
                for (dst, &src) in out_row.iter_mut().zip(row.iter()) {
                    *dst = src;
                }
            });

        Tensor::from_array_no_grad(out.into_dyn())
    }

    fn forward_infer_tmp_bf16(&self, indices: &Tensor, weight: &[bf16]) -> Tensor {
        assert!(is_no_grad(), "forward_infer_tmp_bf16 is inference-only");
        let idx_data = indices.data_arc();
        let e_dim = self.embed_dim;
        let v_size = self.vocab_size;

        let mut out_shape = idx_data.shape().to_vec();
        out_shape.push(e_dim);
        let mut out = Array::zeros(out_shape);

        let num_elements = idx_data.len();
        let idx_flat = idx_data
            .view()
            .into_shape(num_elements)
            .expect("Flatten indices failed");
        let mut out_flat = out
            .view_mut()
            .into_shape((num_elements, e_dim))
            .expect("Flatten output failed");

        Zip::from(out_flat.outer_iter_mut())
            .and(&idx_flat)
            .par_for_each(|mut out_row, &idx_f32| {
                let idx = idx_f32 as usize;
                if idx >= v_size {
                    panic!("Embedding index out of bounds: {} >= {}", idx, v_size);
                }
                let row = &weight[idx * e_dim..(idx + 1) * e_dim];
                let row_bits: &[u16] = unsafe {
                    std::slice::from_raw_parts(row.as_ptr() as *const u16, row.len())
                };
                for (dst, &bits) in out_row.iter_mut().zip(row_bits.iter()) {
                    *dst = bf16::from_bits(bits).to_f32();
                }
            });

        Tensor::from_array_no_grad(out.into_dyn())
    }

    pub fn forward(&self, indices: &Tensor) -> Tensor {
        if !is_no_grad() && self.infer_weight_migrated {
            panic!("temporary migrated infer_weight disables embedding training path");
        }
        if is_no_grad() {
            if let Some(infer) = self.infer_weight.as_ref() {
                return match infer {
                    InferenceWeightStorage::F32 { data, .. } => self.forward_infer_tmp_f32(indices, data.as_slice()),
                    InferenceWeightStorage::BF16 { data, .. } => {
                        let weight = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const bf16, data.len()) };
                        self.forward_infer_tmp_bf16(indices, weight)
                    }
                };
            }
        }

        let w_data = self.weight.data_arc();
        let idx_data = indices.data_arc();
        let e_dim = self.embed_dim;
        let v_size = self.vocab_size;

        let mut out_shape = idx_data.shape().to_vec();
        out_shape.push(e_dim);
        let mut out = Array::zeros(out_shape);

        let num_elements = idx_data.len();
        let idx_flat = idx_data
            .view()
            .into_shape(num_elements)
            .expect("Flatten indices failed");
        let mut out_flat = out
            .view_mut()
            .into_shape((num_elements, e_dim))
            .expect("Flatten output failed");

        let w_2d = w_data
            .into_dimensionality::<ndarray::Ix2>()
            .expect("Embedding weight must be 2D");

        Zip::from(out_flat.outer_iter_mut())
            .and(&idx_flat)
            .par_for_each(|mut out_row, &idx_f32| {
                let idx = idx_f32 as usize;
                if idx < v_size {
                    let w_row = w_2d.slice(ndarray::s![idx, ..]);
                    out_row.assign(&w_row);
                } else {
                    panic!("Embedding index out of bounds: {} >= {}", idx, v_size);
                }
            });

        let out_dyn = out.into_dyn();
        let build_graph = !is_no_grad() && self.weight.requires_grad();

        if !build_graph {
            return Tensor::from_array_no_grad(out_dyn);
        }

        let indices_clone = indices.clone();
        let w_clone = self.weight.clone();
        let v_snap = v_size;
        let e_snap = e_dim;

        Tensor(Rc::new(RefCell::new(TensorData {
            data: out_dyn.into_shared(),
            grad: None,
            parents: vec![indices.clone(), self.weight.clone()],
            backward_op: Some(std::rc::Rc::new(move |grad| {
                let binding = indices_clone.data_ref();
                let idx_flat = binding.view().into_shape(num_elements).unwrap();
                let grad_2d = grad.view().into_shape((num_elements, e_snap)).unwrap();

                let mut d_w = Array::zeros((v_snap, e_snap));
                for (i, &idx_f32) in idx_flat.iter().enumerate() {
                    let idx = idx_f32 as usize;
                    if idx < v_snap {
                        d_w.slice_mut(ndarray::s![idx, ..])
                            .add_assign(&grad_2d.slice(ndarray::s![i, ..]));
                    }
                }
                w_clone.add_grad(d_w.into_dyn());
            })),
            requires_grad: true,
        })))
    }
}

impl Module for Embedding {
    fn forward(&self, x: Tensor) -> Tensor {
        self.forward(&x)
    }
    fn parameters(&self) -> Vec<Tensor> {
        vec![self.weight.clone()]
    }
    fn has_temporary_infer_migration(&self) -> bool {
        self.infer_weight_migrated
    }
}
