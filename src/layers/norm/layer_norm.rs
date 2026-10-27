//! Layer normalization over an arbitrary nonempty suffix of the input shape.
use crate::autograd::{Tensor, is_no_grad};
use crate::module::Module;
use crate::precision::DType;
use ndarray::{ArrayD, IxDyn};
use std::cell::RefCell;
use std::rc::Rc;

/// Population-variance LayerNorm, with optional elementwise affine parameters.
///
/// Inputs and parameters may be F32/F16/BF16. Centered statistics use F64
/// reductions; normalized values, outputs and gradients are F32, including
/// during inference. Integer quantization is not supported here.
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
        let x = input.data_ref().iter().copied().collect::<Vec<_>>();
        let weight = self
            .weight
            .as_ref()
            .map(|p| p.data_ref().iter().copied().collect::<Vec<_>>());
        let bias = self
            .bias
            .as_ref()
            .map(|p| p.data_ref().iter().copied().collect::<Vec<_>>());
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
        if !build_graph {
            return Tensor::from_f32_data_no_grad_with_device_dtype(
                output,
                DType::F32,
                input.device(),
            );
        }
        let input_clone = input.clone();
        let w = self.weight.clone();
        let b = self.bias.clone();
        let parameter_shape = self.normalized_shape.clone();
        let backward = Rc::new(move |grad: &ndarray::ArrayViewD<f32>| {
            let grad = grad.iter().copied().collect::<Vec<_>>();
            let mut dx = vec![0.0; grad.len()];
            let mut dw = vec![0.0f64; dim];
            let mut db = vec![0.0f64; dim];
            for (row, &inv) in inverse.iter().enumerate() {
                let start = row * dim;
                let mut sum = 0.0f64;
                let mut dot = 0.0f64;
                for j in 0..dim {
                    let i = start + j;
                    let g = grad[i] as f64 * weight.as_ref().map_or(1.0, |w| w[j]) as f64;
                    sum += g;
                    dot += g * xhat[i] as f64;
                    dw[j] += grad[i] as f64 * xhat[i] as f64;
                    db[j] += grad[i] as f64;
                }
                for j in 0..dim {
                    let i = start + j;
                    let g = grad[i] as f64 * weight.as_ref().map_or(1.0, |w| w[j]) as f64;
                    dx[i] = (inv as f64
                        * (g - sum / dim as f64 - xhat[i] as f64 * dot / dim as f64))
                        as f32;
                }
            }
            input_clone.add_backward_grad(ArrayD::from_shape_vec(IxDyn(&shape), dx).unwrap());
            if let Some(w) = &w {
                w.add_backward_grad(
                    ArrayD::from_shape_vec(
                        IxDyn(&parameter_shape),
                        dw.into_iter().map(|v| v as f32).collect(),
                    )
                    .unwrap(),
                );
            }
            if let Some(b) = &b {
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
