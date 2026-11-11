//! Layer normalization over an arbitrary nonempty suffix of the input shape.
use crate::autograd::{StoragePreference, Tensor, TensorStorageView, is_no_grad};
use crate::module::Module;
use crate::precision::DType;
use ndarray::{ArrayD, IxDyn};
use rayon::prelude::*;
use std::cell::RefCell;
use std::rc::Rc;

/// Population-variance LayerNorm, with optional elementwise affine parameters.
///
/// Inputs and parameters may be F32/F16/BF16. Centered statistics use F64
/// reductions; normalized values, outputs and gradients are F32, including
/// during inference. Reading inputs and parameters preserves their storage dtype.
/// Integer quantization is not supported here.
pub struct LayerNorm {
    pub weight: Option<Tensor>,
    pub bias: Option<Tensor>,
    normalized_shape: Vec<usize>,
    eps: f32,
}

impl LayerNorm {
    pub fn new(normalized_shape: impl Into<Vec<usize>>, eps: f32) -> Self {
        Self::with_options(normalized_shape, eps, true, true, DType::F32)
    }

    /// `bias` requires `affine`. Parameter dtype does not change reduction dtype.
    pub fn with_options(
        normalized_shape: impl Into<Vec<usize>>,
        eps: f32,
        affine: bool,
        bias: bool,
        dtype: DType,
    ) -> Self {
        let normalized_shape = normalized_shape.into();
        assert!(
            !normalized_shape.is_empty() && normalized_shape.iter().all(|&n| n > 0),
            "LayerNorm normalized_shape must contain positive dimensions"
        );
        assert!(
            eps.is_finite() && eps > 0.0,
            "LayerNorm eps must be finite and positive"
        );
        assert!(
            dtype.is_float(),
            "LayerNorm parameters must be floating point"
        );
        assert!(affine || !bias, "LayerNorm bias requires affine parameters");
        normalized_shape
            .iter()
            .try_fold(1usize, |a, &b| a.checked_mul(b))
            .expect("LayerNorm normalized_shape overflow");
        let weight = affine
            .then(|| Tensor::parameter_with_dtype(ArrayD::ones(IxDyn(&normalized_shape)), dtype));
        let bias = bias
            .then(|| Tensor::parameter_with_dtype(ArrayD::zeros(IxDyn(&normalized_shape)), dtype));
        Self {
            weight,
            bias,
            normalized_shape,
            eps,
        }
    }

    pub fn normalized_shape(&self) -> &[usize] {
        &self.normalized_shape
    }
    pub fn eps(&self) -> f32 {
        self.eps
    }
}

fn cpu_values(tensor: &Tensor) -> Vec<f32> {
    tensor.with_storage_view_preferring(StoragePreference::F32Compute, |view| {
        let TensorStorageView::F32(values) = view else {
            unreachable!("LayerNorm requested an F32 compute view")
        };
        values.iter().copied().collect()
    })
}

fn cpu_inference_rows(
    input: ndarray::ArrayViewD<'_, f32>,
    weight: Option<&[f32]>,
    bias: Option<&[f32]>,
    dim: usize,
    eps: f32,
) -> Vec<f32> {
    let input = input.as_standard_layout();
    let input = input.as_slice().expect("standard-layout LayerNorm input");
    let mut output = vec![0.0; input.len()];
    // Keep the centered F64 statistics and F32 affine order. Each row needs
    // only local statistics; inference does not retain backward intermediates.
    crate::cpu::for_each_chunk_mut(&mut output, dim, dim.saturating_mul(4), |(row, dst)| {
        let src = &input[row * dim..(row + 1) * dim];
        let mean = src.iter().map(|&v| v as f64).sum::<f64>() / dim as f64;
        let var = src.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / dim as f64;
        let inv = (var + eps as f64).sqrt().recip();
        for (j, value) in dst.iter_mut().enumerate() {
            let normalized = ((src[j] as f64 - mean) * inv) as f32;
            *value = normalized * weight.map_or(1.0, |w| w[j]) + bias.map_or(0.0, |b| b[j]);
        }
    });
    output
}

struct CpuGradients {
    input: Option<Vec<f32>>,
    weight: Option<Vec<f64>>,
    bias: Option<Vec<f64>>,
}

fn cpu_backward_rows(
    grad: &[f32],
    xhat: &[f32],
    inverse: &[f32],
    weight: Option<&[f32]>,
    dim: usize,
    needs: [bool; 3],
) -> CpuGradients {
    match needs {
        [false, false, false] => CpuGradients {
            input: None,
            weight: None,
            bias: None,
        },
        [true, false, false] => cpu_backward_rows_impl::<1>(grad, xhat, inverse, weight, dim),
        [false, true, false] => cpu_backward_rows_impl::<2>(grad, xhat, inverse, weight, dim),
        [true, true, false] => cpu_backward_rows_impl::<3>(grad, xhat, inverse, weight, dim),
        [false, false, true] => cpu_backward_rows_impl::<4>(grad, xhat, inverse, weight, dim),
        [true, false, true] => cpu_backward_rows_impl::<5>(grad, xhat, inverse, weight, dim),
        [false, true, true] => cpu_backward_rows_impl::<6>(grad, xhat, inverse, weight, dim),
        [true, true, true] => cpu_backward_rows_impl::<7>(grad, xhat, inverse, weight, dim),
    }
}

// Specialize gradient requirements before entering numerical loops. Small work
// retains the fused traversal; larger work uses disjoint rows and F64 folds.
fn cpu_backward_rows_impl<const NEED: u8>(
    grad: &[f32],
    xhat: &[f32],
    inverse: &[f32],
    weight: Option<&[f32]>,
    dim: usize,
) -> CpuGradients {
    let mut dx = vec![0.0; if NEED & 1 != 0 { grad.len() } else { 0 }];
    if !crate::cpu::should_parallelize(inverse.len(), dim.saturating_mul(8)) {
        let mut dw = vec![0.0f64; if NEED & 2 != 0 { dim } else { 0 }];
        let mut db = vec![0.0f64; if NEED & 4 != 0 { dim } else { 0 }];
        for (row, &inv) in inverse.iter().enumerate() {
            let start = row * dim;
            let mut sum = 0.0f64;
            let mut dot = 0.0f64;
            for j in 0..dim {
                let i = start + j;
                if NEED & 1 != 0 {
                    let g = grad[i] as f64 * weight.map_or(1.0, |w| w[j]) as f64;
                    sum += g;
                    dot += g * xhat[i] as f64;
                }
                if NEED & 2 != 0 {
                    dw[j] += grad[i] as f64 * xhat[i] as f64;
                }
                if NEED & 4 != 0 {
                    db[j] += grad[i] as f64;
                }
            }
            if NEED & 1 != 0 {
                for j in 0..dim {
                    let i = start + j;
                    let g = grad[i] as f64 * weight.map_or(1.0, |w| w[j]) as f64;
                    dx[i] = (inv as f64
                        * (g - sum / dim as f64 - xhat[i] as f64 * dot / dim as f64))
                        as f32;
                }
            }
        }
        return CpuGradients {
            input: (NEED & 1 != 0).then_some(dx),
            weight: (NEED & 2 != 0).then_some(dw),
            bias: (NEED & 4 != 0).then_some(db),
        };
    }
    if NEED & 1 != 0 {
        let (dw, db) = crate::cpu::fold_reduce_indexed(
            dx.par_chunks_mut(dim),
            dim.saturating_mul(8),
            || {
                (
                    vec![0.0f64; if NEED & 2 != 0 { dim } else { 0 }],
                    vec![0.0f64; if NEED & 4 != 0 { dim } else { 0 }],
                )
            },
            |(mut dw, mut db), (row, dx)| {
                let start = row * dim;
                let mut sum = 0.0f64;
                let mut dot = 0.0f64;
                for j in 0..dim {
                    let i = start + j;
                    let g = grad[i] as f64 * weight.map_or(1.0, |w| w[j]) as f64;
                    sum += g;
                    dot += g * xhat[i] as f64;
                    if NEED & 2 != 0 {
                        dw[j] += grad[i] as f64 * xhat[i] as f64;
                    }
                    if NEED & 4 != 0 {
                        db[j] += grad[i] as f64;
                    }
                }
                for (j, dx) in dx.iter_mut().enumerate() {
                    let i = start + j;
                    let g = grad[i] as f64 * weight.map_or(1.0, |w| w[j]) as f64;
                    *dx = (inverse[row] as f64
                        * (g - sum / dim as f64 - xhat[i] as f64 * dot / dim as f64))
                        as f32;
                }
                (dw, db)
            },
            |(mut dw, mut db), (w, b)| {
                for (v, w) in dw.iter_mut().zip(w) {
                    *v += w;
                }
                for (v, b) in db.iter_mut().zip(b) {
                    *v += b;
                }
                (dw, db)
            },
        );
        return CpuGradients {
            input: Some(dx),
            weight: (NEED & 2 != 0).then_some(dw),
            bias: (NEED & 4 != 0).then_some(db),
        };
    }
    let (dw, db) = if NEED & 6 != 0 {
        crate::cpu::fold_reduce(
            inverse.len(),
            dim.saturating_mul(2),
            || {
                (
                    vec![0.0f64; if NEED & 2 != 0 { dim } else { 0 }],
                    vec![0.0f64; if NEED & 4 != 0 { dim } else { 0 }],
                )
            },
            |(mut dw, mut db), row| {
                let start = row * dim;
                for (j, v) in dw.iter_mut().enumerate() {
                    *v += grad[start + j] as f64 * xhat[start + j] as f64;
                }
                for (j, v) in db.iter_mut().enumerate() {
                    *v += grad[start + j] as f64;
                }
                (dw, db)
            },
            |(mut dw, mut db), (w, b)| {
                for (v, w) in dw.iter_mut().zip(w) {
                    *v += w;
                }
                for (v, b) in db.iter_mut().zip(b) {
                    *v += b;
                }
                (dw, db)
            },
        )
    } else {
        (Vec::new(), Vec::new())
    };
    CpuGradients {
        input: (NEED & 1 != 0).then_some(dx),
        weight: (NEED & 2 != 0).then_some(dw),
        bias: (NEED & 4 != 0).then_some(db),
    }
}

impl Module for LayerNorm {
    fn forward(&self, input: Tensor) -> Tensor {
        let shape = input.shape_vec();
        assert!(
            input.dtype().is_float(),
            "LayerNorm input must be floating point"
        );
        assert!(
            shape.ends_with(&self.normalized_shape),
            "LayerNorm input suffix does not match normalized_shape"
        );
        for parameter in self.weight.iter().chain(self.bias.iter()) {
            assert_eq!(
                parameter.shape_vec(),
                self.normalized_shape,
                "LayerNorm parameter shape mismatch"
            );
            assert_eq!(
                parameter.device(),
                input.device(),
                "LayerNorm device mismatch"
            );
            assert!(
                parameter.dtype().is_float(),
                "LayerNorm parameters must be floating point"
            );
        }
        let dim = self.normalized_shape.iter().product::<usize>();
        let rows = input.len() / dim;
        let parents = std::iter::once(input.clone())
            .chain(self.parameters())
            .collect::<Vec<_>>();
        let build_graph = !is_no_grad() && parents.iter().any(Tensor::requires_grad);
        #[cfg(feature = "cuda")]
        if input.is_cuda() && rows > 0 {
            return cuda_forward(
                &input,
                self.weight.as_ref(),
                self.bias.as_ref(),
                rows,
                dim,
                self.eps,
                parents,
                build_graph,
            );
        }
        let weight = self.weight.as_ref().map(cpu_values);
        let bias = self.bias.as_ref().map(cpu_values);
        if !build_graph {
            let output =
                input.with_storage_view_preferring(StoragePreference::F32Compute, |view| {
                    let TensorStorageView::F32(values) = view else {
                        unreachable!("LayerNorm requested an F32 compute view")
                    };
                    cpu_inference_rows(values, weight.as_deref(), bias.as_deref(), dim, self.eps)
                });
            return Tensor::from_f32_data_no_grad_with_device_dtype(
                ArrayD::from_shape_vec(IxDyn(&shape), output).unwrap(),
                DType::F32,
                input.device(),
            );
        }
        let x = cpu_values(&input);
        let mut xhat = vec![0.0; x.len()];
        let mut statistics = vec![(0.0f64, 0.0f64); rows];
        let mut output = vec![0.0; x.len()];
        let eps = self.eps;
        // Two-pass, centered variance avoids subtracting large squared means.
        crate::cpu::for_each_chunk_mut(&mut statistics, 1, dim.saturating_mul(4), |(row, dst)| {
            let src = &x[row * dim..(row + 1) * dim];
            let mean = src.iter().map(|&v| v as f64).sum::<f64>() / dim as f64;
            let var = src.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / dim as f64;
            dst[0] = (mean, (var + eps as f64).sqrt().recip());
        });
        crate::cpu::for_each_chunk2_mut(
            &mut xhat,
            &mut output,
            dim,
            dim.saturating_mul(3),
            |(row, (normalized, output))| {
                let (mean, inv) = statistics[row];
                for j in 0..dim {
                    normalized[j] = ((x[row * dim + j] as f64 - mean) * inv) as f32;
                    output[j] = normalized[j] * weight.as_ref().map_or(1.0, |w| w[j])
                        + bias.as_ref().map_or(0.0, |b| b[j]);
                }
            },
        );
        let inverse = statistics
            .iter()
            .map(|&(_, inv)| inv as f32)
            .collect::<Vec<_>>();
        let output = ArrayD::from_shape_vec(IxDyn(&shape), output).unwrap();
        let input_clone = input.clone();
        let w = self.weight.clone();
        let b = self.bias.clone();
        let parameter_shape = self.normalized_shape.clone();
        let backward = Rc::new(move |grad: &ndarray::ArrayViewD<f32>| {
            let needs = [
                input_clone.requires_grad(),
                w.as_ref().is_some_and(Tensor::requires_grad),
                b.as_ref().is_some_and(Tensor::requires_grad),
            ];
            if !needs.iter().any(|&need| need) {
                return;
            }
            let grad = grad.iter().copied().collect::<Vec<_>>();
            let gradients =
                cpu_backward_rows(&grad, &xhat, &inverse, weight.as_deref(), dim, needs);
            if let Some(dx) = gradients.input {
                input_clone.add_backward_grad(ArrayD::from_shape_vec(IxDyn(&shape), dx).unwrap());
            }
            if let (Some(w), Some(dw)) = (&w, gradients.weight) {
                w.add_backward_grad(
                    ArrayD::from_shape_vec(
                        IxDyn(&parameter_shape),
                        dw.into_iter().map(|v| v as f32).collect(),
                    )
                    .unwrap(),
                );
            }
            if let (Some(b), Some(db)) = (&b, gradients.bias) {
                b.add_backward_grad(
                    ArrayD::from_shape_vec(
                        IxDyn(&parameter_shape),
                        db.into_iter().map(|v| v as f32).collect(),
                    )
                    .unwrap(),
                );
            }
        });
        let mut data = Tensor::op_data(output.into_shared(), parents, Some(backward), true);
        data.device = input.device();
        Tensor(Rc::new(RefCell::new(data)))
    }

    fn parameters(&self) -> Vec<Tensor> {
        self.weight
            .iter()
            .chain(self.bias.iter())
            .cloned()
            .collect()
    }
}

#[cfg(feature = "cuda")]
fn cuda_forward(
    input: &Tensor,
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    rows: usize,
    dim: usize,
    eps: f32,
    parents: Vec<Tensor>,
    build_graph: bool,
) -> Tensor {
    use crate::ops::cuda;
    let x = input.with_cuda_f32_buffer(Clone::clone);
    let w = weight.map(|p| p.with_cuda_f32_buffer(Clone::clone));
    let b = bias.map(|p| p.with_cuda_f32_buffer(Clone::clone));
    let (out, xhat, inverse) =
        cuda::generic::layer_norm(&x, w.as_ref(), b.as_ref(), rows, dim, eps)
            .expect("LayerNorm CUDA forward");
    let shape = input.shape_vec();
    let result = Tensor::from_cuda_f32_buffer_no_host(&shape, out, input.device());
    if !build_graph {
        return result;
    }
    let weak = Rc::downgrade(&result.0);
    let input = input.clone();
    let weight = weight.cloned();
    let bias = bias.cloned();
    let mut data = result.0.borrow_mut();
    data.requires_grad = true;
    data.parents = parents;
    data.backward_op = Some(Rc::new(move |grad| {
        let upstream = weak
            .upgrade()
            .map(Tensor)
            .and_then(|t| t.cloned_cuda_f32_grad())
            .unwrap_or_else(|| {
                cuda::upload_f32(&grad.iter().copied().collect::<Vec<_>>())
                    .expect("LayerNorm CUDA upstream")
            });
        let (dx, dw, db) =
            cuda::generic::layer_norm_backward(&xhat, &inverse, w.as_ref(), &upstream, rows, dim)
                .expect("LayerNorm CUDA backward");
        input.add_backward_cuda_grad_buffer_only(dx);
        if let Some(w) = &weight {
            w.add_backward_cuda_grad_buffer_only(dw);
        }
        if let Some(b) = &bias {
            b.add_backward_cuda_grad_buffer_only(db);
        }
    }));
    drop(data);
    // Weak output ownership avoids retaining the entire graph through its callback.
    result
}

#[cfg(test)]
#[path = "../../../tests/unit/layers/norm/layer_norm/tests.rs"]
mod tests;
