use super::*;

#[derive(Clone, Copy)]
pub(super) enum RhsLayout {
    Standard,
    Transposed,
}

impl RhsLayout {
    pub(super) fn name(self) -> &'static str {
        match self {
            Self::Standard => "matmul",
            Self::Transposed => "matmul_transposed_rhs",
        }
    }

    #[cfg(feature = "cpu-profiling")]
    pub(super) fn profile_total(self) -> &'static str {
        match self {
            Self::Standard => "matmul.total",
            Self::Transposed => "matmul_transposed_rhs.total",
        }
    }

    #[cfg(feature = "cpu-profiling")]
    pub(super) fn profile_kernel(self) -> &'static str {
        match self {
            Self::Standard => "matmul.kernel",
            Self::Transposed => "matmul_transposed_rhs.kernel",
        }
    }

    fn dims(m: usize, k: usize, n: usize) -> BatchMatmulDims {
        BatchMatmulDims {
            b: 1,
            h: 1,
            m,
            k,
            n,
        }
    }

    pub(super) fn forward_cpu(
        self,
        a: &Tensor,
        b: &Tensor,
        m: usize,
        k: usize,
        n: usize,
    ) -> Array2<f32> {
        if matches!(self, Self::Transposed) {
            return tensor::matmul_forward_cpu_native(a, b, m, k, n);
        }
        if let Some(output) = standard_cpu::try_forward(a, b, m, k, n) {
            return output;
        }
        conversion::with_native_f32_compute_view(a, |av| {
            conversion::with_native_f32_compute_view(b, |bv| {
                let av = av.as_standard_layout();
                let av = av.into_shape((m, k)).expect("matmul input shape");
                let bv = bv.into_dimensionality::<Ix2>().expect("matmul RHS shape");
                let mut output = Array2::zeros((m, n));
                cpu_gemm::f32_matmul(av.view(), bv, output.view_mut());
                output
            })
        })
    }

    pub(super) fn backward_cpu(
        self,
        a: &Tensor,
        b: &Tensor,
        grad: &ndarray::ArrayViewD<'_, f32>,
        m: usize,
        k: usize,
        n: usize,
    ) -> (Option<ndarray::ArrayD<f32>>, Option<ndarray::ArrayD<f32>>) {
        let need_a = a.requires_grad();
        let need_b = b.requires_grad();
        if matches!(self, Self::Transposed) && need_a && need_b {
            let (da, db) = backward::matmul_backward_cpu_f32(a, b, grad, m, k, n);
            return (Some(da), Some(db));
        }
        let grad = grad.as_standard_layout();
        let grad = grad.into_shape((m, n)).expect("matmul upstream shape");
        let da = need_a.then(|| {
            conversion::with_native_f32_compute_view(b, |bv| {
                let bv = bv.into_dimensionality::<Ix2>().expect("matmul RHS shape");
                let rhs = match self {
                    Self::Standard => bv.t(),
                    Self::Transposed => bv.view(),
                };
                let mut da = Array2::zeros((m, k));
                cpu_gemm::f32_matmul(grad.view(), rhs, da.view_mut());
                da.into_shape(a.shape_vec()).unwrap().into_dyn()
            })
        });
        let db = need_b.then(|| {
            conversion::with_native_f32_compute_view(a, |av| {
                let av = av.as_standard_layout();
                let av = av.into_shape((m, k)).expect("matmul input shape");
                match self {
                    Self::Standard => {
                        let mut db = Array2::zeros((k, n));
                        cpu_gemm::f32_matmul(av.t(), grad.view(), db.view_mut());
                        db.into_dyn()
                    }
                    Self::Transposed => {
                        let mut db = Array2::zeros((n, k));
                        cpu_gemm::f32_matmul(grad.t(), av.view(), db.view_mut());
                        db.into_dyn()
                    }
                }
            })
        });
        (da, db)
    }

    pub(super) fn forward_cuda_buffer(
        self,
        a: &Tensor,
        b: &Tensor,
        m: usize,
        k: usize,
        n: usize,
    ) -> Option<cuda::CudaBuffer> {
        match self {
            Self::Transposed => cuda_route::try_cuda_matmul_buffer(a, b, m, k, n),
            Self::Standard => cuda_route::try_cuda_batch_matmul_buffer(a, b, 1, 1, m, k, n),
        }
    }

    pub(super) fn forward_cuda(
        self,
        a: &Tensor,
        b: &Tensor,
        m: usize,
        k: usize,
        n: usize,
        dtype: DType,
        shape: &[usize],
    ) -> Option<Tensor> {
        match self {
            Self::Transposed => cuda_route::try_cuda_matmul(a, b, m, n, k, dtype),
            Self::Standard => cuda_route::try_cuda_batch_matmul_with_shape(
                a,
                b,
                Self::dims(m, k, n),
                dtype,
                shape,
            ),
        }
    }

    pub(super) fn backward_cuda_buffers(
        self,
        grad: &ndarray::ArrayViewD<'_, f32>,
        cuda_grad: Option<cuda::CudaBuffer>,
        a: &Tensor,
        b: &Tensor,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<(cuda::CudaBuffer, cuda::CudaBuffer), String> {
        match self {
            Self::Transposed => cuda_route::try_cuda_training_matmul_backward_buffers(
                grad, cuda_grad, a, b, m, k, n,
            ),
            Self::Standard => cuda_route::try_cuda_training_batch_matmul_backward_buffers(
                grad,
                cuda_grad,
                a,
                b,
                Self::dims(m, k, n),
            ),
        }
    }

    pub(super) fn backward_cuda(
        self,
        grad: &ndarray::ArrayViewD<'_, f32>,
        cuda_grad: Option<cuda::CudaBuffer>,
        a: &Tensor,
        b: &Tensor,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<CudaMatmulBackwardHostDevice, String> {
        match self {
            Self::Transposed => {
                cuda_route::try_cuda_training_matmul_backward(grad, cuda_grad, a, b, m, k, n)
            }
            Self::Standard => {
                let ((da, da_buf), (db, db_buf)) =
                    cuda_route::try_cuda_training_batch_matmul_backward(
                        grad,
                        cuda_grad,
                        a,
                        b,
                        Self::dims(m, k, n),
                    )?;
                Ok((
                    (da.into_shape(a.shape_vec()).unwrap(), da_buf),
                    (db.into_shape(b.shape_vec()).unwrap(), db_buf),
                ))
            }
        }
    }
}
