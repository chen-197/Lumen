use super::backward::batch_matmul_backward_cpu_f32;
use super::conversion::with_native_f32_compute_view;
use super::cuda_route::{
    try_cuda_batch_matmul, try_cuda_batch_matmul_buffer, try_cuda_training_batch_matmul_backward,
    try_cuda_training_batch_matmul_backward_buffers,
};
use super::prefill::try_prefill_gemm;
use super::rows::{
    matmul_rows_bf16_bf16, matmul_rows_bf16_f32, matmul_rows_bf16_i8, matmul_rows_bf16_slice,
    matmul_rows_f16_f16, matmul_rows_f16_f32, matmul_rows_f16_slice, matmul_rows_f32_bf16,
    matmul_rows_f32_f16, matmul_rows_f32_i8, matmul_rows_i8_slice,
};
use super::*;

pub(super) fn matmul_forward_cpu_native(
    a: &Tensor,
    b: &Tensor,
    m_dim: usize,
    k_dim: usize,
    n_dim: usize,
) -> Array2<f32> {
    if let Some(output) = try_prefill_gemm(a, b, m_dim, k_dim, n_dim) {
        return output;
    }
    let input_dtype = a.dtype();
    if b.dtype() == DType::I8 {
        return match b.native_storage_owned() {
            TensorStorageOwned::I8(b_data, scale) => {
                let b_2d = b_data
                    .view()
                    .into_dimensionality::<Ix2>()
                    .expect("matmul RHS must be 2D [N, K]");
                let b_owned;
                let b_slice: &[i8] = if let Some(s) = b_2d.as_slice() {
                    s
                } else {
                    b_owned = b_2d.as_standard_layout().to_owned();
                    b_owned
                        .as_slice()
                        .expect("standard-layout matmul RHS should be contiguous")
                };

                if input_dtype == DType::I8 {
                    match a.native_storage_owned() {
                        TensorStorageOwned::I8(a_data, a_scale) => {
                            if m_dim == 1 {
                                let a_owned;
                                let a_vec: &[i8] = if let Some(s) = a_data.as_slice() {
                                    s
                                } else {
                                    a_owned = a_data.iter().copied().collect::<Vec<i8>>();
                                    a_owned.as_slice()
                                };

                                let mut out_vec = vec![0.0f32; n_dim];
                                matvec_rowmajor_parallel_i8_i8(
                                    a_vec,
                                    a_scale,
                                    b_slice,
                                    scale,
                                    n_dim,
                                    k_dim,
                                    &mut out_vec,
                                );
                                Array2::from_shape_vec((1, n_dim), out_vec)
                                    .expect("decode matvec shape build failed")
                            } else {
                                matmul_rows_i8_slice(
                                    a_data.view().into_dyn(),
                                    a_scale,
                                    SliceRef::I8(b_slice, scale),
                                    m_dim,
                                    k_dim,
                                    n_dim,
                                )
                            }
                        }
                        TensorStorageOwned::F32(_)
                        | TensorStorageOwned::F16(_)
                        | TensorStorageOwned::BF16(_) => {
                            unreachable!("checked i8 lhs above")
                        }
                    }
                } else {
                    a.with_storage_view_preferring(StoragePreference::Native, |a_view| {
                        if m_dim == 1 {
                            match a_view {
                                TensorStorageView::F32(a_view) => {
                                    let a_owned;
                                    let a_vec: &[f32] = if let Some(s) = a_view.as_slice() {
                                        s
                                    } else {
                                        a_owned = a_view.iter().copied().collect::<Vec<f32>>();
                                        a_owned.as_slice()
                                    };

                                    let mut out_vec = vec![0.0f32; n_dim];
                                    matvec_rowmajor_parallel_f32_i8_matmul(
                                        a_vec,
                                        b_slice,
                                        scale,
                                        n_dim,
                                        k_dim,
                                        &mut out_vec,
                                    );
                                    Array2::from_shape_vec((1, n_dim), out_vec)
                                        .expect("decode matvec shape build failed")
                                }
                                TensorStorageView::F16(a_view) => {
                                    let a_owned;
                                    let a_vec: &[f16] = if let Some(s) = a_view.as_slice() {
                                        s
                                    } else {
                                        a_owned = a_view.iter().copied().collect::<Vec<f16>>();
                                        a_owned.as_slice()
                                    };

                                    let mut out_vec = vec![0.0f32; n_dim];
                                    with_f16_input_as_f32(a_vec, |a_f32| {
                                        matvec_rowmajor_parallel_f32_i8_matmul(
                                            a_f32,
                                            b_slice,
                                            scale,
                                            n_dim,
                                            k_dim,
                                            &mut out_vec,
                                        );
                                    });
                                    Array2::from_shape_vec((1, n_dim), out_vec)
                                        .expect("decode matvec shape build failed")
                                }
                                TensorStorageView::BF16(a_view) => {
                                    let a_owned;
                                    let a_vec: &[bf16] = if let Some(s) = a_view.as_slice() {
                                        s
                                    } else {
                                        a_owned = a_view.iter().copied().collect::<Vec<bf16>>();
                                        a_owned.as_slice()
                                    };

                                    let mut out_vec = vec![0.0f32; n_dim];
                                    with_bf16_input_as_f32(a_vec, |a_f32| {
                                        matvec_rowmajor_parallel_f32_i8_matmul(
                                            a_f32,
                                            b_slice,
                                            scale,
                                            n_dim,
                                            k_dim,
                                            &mut out_vec,
                                        );
                                    });
                                    Array2::from_shape_vec((1, n_dim), out_vec)
                                        .expect("decode matvec shape build failed")
                                }
                            }
                        } else {
                            let b_view = b_data.view().into_dyn();
                            match a_view {
                                TensorStorageView::F32(a_view) => {
                                    matmul_rows_f32_i8(a_view, b_view, scale, m_dim, k_dim, n_dim)
                                }
                                TensorStorageView::F16(a_view) => matmul_rows_f16_slice(
                                    a_view,
                                    SliceRef::I8(b_slice, scale),
                                    m_dim,
                                    k_dim,
                                    n_dim,
                                ),
                                TensorStorageView::BF16(a_view) => {
                                    matmul_rows_bf16_i8(a_view, b_view, scale, m_dim, k_dim, n_dim)
                                }
                            }
                        }
                    })
                }
            }
            TensorStorageOwned::F32(_)
            | TensorStorageOwned::F16(_)
            | TensorStorageOwned::BF16(_) => unreachable!("checked i8 RHS above"),
        };
    }

    a.with_storage_view_preferring(StoragePreference::Native, |a_view| {
        b.with_storage_view_preferring(StoragePreference::Native, |b_view| match (a_view, b_view) {
            (TensorStorageView::F32(a_view), TensorStorageView::F32(b_view)) => {
                if m_dim == 1 {
                    let a_owned;
                    let a_vec: &[f32] = if let Some(s) = a_view.as_slice() {
                        s
                    } else {
                        a_owned = a_view.iter().copied().collect::<Vec<f32>>();
                        a_owned.as_slice()
                    };

                    let b_2d = b_view.into_dimensionality::<Ix2>().unwrap();
                    let b_owned;
                    let b_slice: &[f32] = if let Some(s) = b_2d.as_slice() {
                        s
                    } else {
                        b_owned = b_2d.as_standard_layout().to_owned();
                        b_owned
                            .as_slice()
                            .expect("standard-layout matmul RHS should be contiguous")
                    };

                    let mut out_vec = vec![0.0f32; n_dim];
                    matvec_rowmajor_parallel(a_vec, b_slice, n_dim, k_dim, &mut out_vec);
                    Array2::from_shape_vec((1, n_dim), out_vec)
                        .expect("decode matvec shape build failed")
                } else {
                    let b_2d = b_view.into_dimensionality::<Ix2>().unwrap();
                    let mut res = Array2::<f32>::zeros((m_dim, n_dim));
                    if let Ok(a_2d_view) = a_view.as_standard_layout().into_shape((m_dim, k_dim)) {
                        super::cpu_gemm::f32_matmul(a_2d_view.view(), b_2d.t(), res.view_mut());
                    } else {
                        let a_2d_owned = a_view
                            .to_owned()
                            .into_shape((m_dim, k_dim))
                            .expect("Reshape A failed");
                        super::cpu_gemm::f32_matmul(a_2d_owned.view(), b_2d.t(), res.view_mut());
                    }
                    res
                }
            }
            (TensorStorageView::F32(a_view), TensorStorageView::F16(b_view)) => {
                matmul_rows_f32_f16(a_view, b_view, m_dim, k_dim, n_dim)
            }
            (TensorStorageView::F32(a_view), TensorStorageView::BF16(b_view)) => {
                matmul_rows_f32_bf16(a_view, b_view, m_dim, k_dim, n_dim)
            }
            (TensorStorageView::F16(a_view), TensorStorageView::F32(b_view)) => {
                matmul_rows_f16_f32(a_view, b_view, m_dim, k_dim, n_dim)
            }
            (TensorStorageView::F16(a_view), TensorStorageView::F16(b_view)) => {
                matmul_rows_f16_f16(a_view, b_view, m_dim, k_dim, n_dim)
            }
            (TensorStorageView::F16(a_view), TensorStorageView::BF16(b_view)) => {
                let b_2d = b_view.into_dimensionality::<Ix2>().unwrap();
                let b_2d = b_2d.as_standard_layout();
                matmul_rows_f16_slice(
                    a_view,
                    SliceRef::BF16(
                        b_2d.as_slice()
                            .expect("standard-layout matmul RHS should be contiguous"),
                    ),
                    m_dim,
                    k_dim,
                    n_dim,
                )
            }
            (TensorStorageView::BF16(a_view), TensorStorageView::F32(b_view)) => {
                matmul_rows_bf16_f32(a_view, b_view, m_dim, k_dim, n_dim)
            }
            (TensorStorageView::BF16(a_view), TensorStorageView::F16(b_view)) => {
                let b_2d = b_view.into_dimensionality::<Ix2>().unwrap();
                let b_2d = b_2d.as_standard_layout();
                matmul_rows_bf16_slice(
                    a_view,
                    SliceRef::F16(
                        b_2d.as_slice()
                            .expect("standard-layout matmul RHS should be contiguous"),
                    ),
                    m_dim,
                    k_dim,
                    n_dim,
                )
            }
            (TensorStorageView::BF16(a_view), TensorStorageView::BF16(b_view)) => {
                matmul_rows_bf16_bf16(a_view, b_view, m_dim, k_dim, n_dim)
            }
        })
    })
}

// An empty output or reduction has a zero VJP for both operands. In particular,
// empty CUDA tensors have no allocation and must not enter a numerical kernel.
pub(super) fn zero_matmul_output(
    a: &Tensor,
    b: &Tensor,
    shape: &[usize],
    build_graph: bool,
) -> Tensor {
    let device = a.device();
    let dtype = if build_graph || a.dtype() != b.dtype() {
        DType::F32
    } else {
        a.dtype()
    };
    let output = if build_graph && a.is_cuda() && !shape.contains(&0) {
        let len = shape
            .iter()
            .try_fold(1usize, |n, &d| n.checked_mul(d))
            .expect("MatMul output shape overflow");
        let buffer = cuda::fill_scalar_f32_buffer(len, 0.).expect("CUDA zero MatMul output");
        Tensor::from_cuda_f32_buffer_no_host(shape, buffer, device)
    } else {
        Tensor::from_f32_data_no_grad_with_device_dtype(
            ndarray::ArrayD::zeros(ndarray::IxDyn(shape)),
            dtype,
            device,
        )
    };
    if build_graph {
        let parents = vec![a.clone(), b.clone()];
        let backward_parents = parents.clone();
        let mut data = output.0.borrow_mut();
        data.requires_grad = true;
        data.parents = parents;
        data.backward_op = Some(Rc::new(move |_: &ndarray::ArrayViewD<f32>| {
            for parent in &backward_parents {
                if !parent.requires_grad() {
                    continue;
                }
                if parent.is_cuda() && !parent.is_empty() {
                    let buffer = cuda::fill_scalar_f32_buffer(parent.len(), 0.)
                        .expect("CUDA zero MatMul gradient");
                    parent.add_backward_cuda_grad_buffer_only(buffer);
                } else {
                    parent.add_backward_grad(ndarray::ArrayD::zeros(ndarray::IxDyn(
                        &parent.shape_vec(),
                    )));
                }
            }
        }));
    }
    output
}

/// Multiply `A[..., K]` by `B[N, K]` transposed, producing `[..., N]`.
/// Empty leading/output axes are supported. A zero-length reduction returns
/// zeros; its operand gradients are shaped zeros as well.
pub fn matmul_transposed_rhs(a: &Tensor, weights: &Tensor) -> Tensor {
    matrix_product(a, weights, super::layout::RhsLayout::Transposed)
}

/// Standard matrix product: `A[..., K] @ B[K, N] -> [..., N]`.
///
/// The RHS must be a 2D matrix; leading LHS dimensions are preserved, and a
/// vector LHS produces a vector output. For output-major linear weights use
/// [`matmul_transposed_rhs`]. For paired 4D batches use [`batch_matmul`].
/// Empty axes and zero reductions have the same semantics as those APIs.
/// Earlier versions used a transposed RHS for `matmul`; migrate those weight-layout
/// calls to `matmul_transposed_rhs`.
pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor {
    matrix_product(a, b, super::layout::RhsLayout::Standard)
}

fn matrix_product(a: &Tensor, b: &Tensor, layout: super::layout::RhsLayout) -> Tensor {
    #[cfg(feature = "cpu-profiling")]
    let _profile = crate::cpu::profiling::region(layout.profile_total());
    let output_device = assert_same_device(a, b, layout.name());
    let build_graph = !is_no_grad() && (a.requires_grad() || b.requires_grad());
    let cuda_native_supported = output_device == crate::device::Device::Cuda;
    assert_native_device_support(output_device, layout.name(), cuda_native_supported);

    let a_shape = a.shape_vec();
    let b_shape = b.shape_vec();

    assert!(
        !a_shape.is_empty(),
        "MatMul LHS must have at least one dimension"
    );

    if b_shape.len() != 2 {
        panic!("MatMul RHS must be 2D, got {:?}", b_shape);
    }

    let k_dim_a = a_shape[a_shape.len() - 1];
    let (k_dim_b, n_dim) = match layout {
        super::layout::RhsLayout::Standard => (b_shape[0], b_shape[1]),
        super::layout::RhsLayout::Transposed => (b_shape[1], b_shape[0]),
    };

    if k_dim_a != k_dim_b {
        panic!(
            "MatMul shape mismatch: a {:?} (K={}) vs b {:?} (K={})",
            a_shape, k_dim_a, b_shape, k_dim_b
        );
    }

    // The row count cannot be inferred from len / K when K is zero.
    let leading = &a_shape[..a_shape.len() - 1];
    let m_dim = if leading.contains(&0) {
        0
    } else {
        leading
            .iter()
            .try_fold(1usize, |m, &d| m.checked_mul(d))
            .expect("MatMul leading shape overflow")
    };
    if m_dim == 0 || k_dim_a == 0 || n_dim == 0 {
        let mut out_shape = a_shape.clone();
        *out_shape.last_mut().unwrap() = n_dim;
        return zero_matmul_output(a, b, &out_shape, build_graph);
    }

    if build_graph
        && output_device == crate::device::Device::Cuda
        && let Some(buffer) = layout.forward_cuda_buffer(a, b, m_dim, k_dim_a, n_dim)
    {
        let mut out_shape = a_shape.clone();
        let last_idx = out_shape.len() - 1;
        out_shape[last_idx] = n_dim;

        let a_clone = a.clone();
        let b_clone = b.clone();
        let output_self = Rc::new(RefCell::new(std::rc::Weak::<RefCell<TensorData>>::new()));
        let output_self_for_backward = output_self.clone();
        let tensor = Tensor(Rc::new(RefCell::new(TensorData {
            data: ndarray::ArrayD::<f32>::zeros(ndarray::IxDyn(&out_shape)).into_shared(),
            f16_data: None,
            bf16_data: None,
            i8_data: None,
            cuda_f32_data: Some(buffer),
            cuda_f16_data: None,
            cuda_bf16_data: None,
            cuda_i8_data: None,
            i8_scale: None,
            has_f32_data: false,
            storage_dtype: crate::precision::DType::F32,
            cache_dirty: false,
            is_parameter: false,
            grad: None,
            cuda_f32_grad: None,
            parents: vec![a_clone.clone(), b_clone.clone()],
            requires_grad: true,
            backward_op: Some(std::rc::Rc::new(move |grad: &ndarray::ArrayViewD<f32>| {
                let cuda_grad = output_self_for_backward
                    .borrow()
                    .upgrade()
                    .map(Tensor)
                    .and_then(|output| output.cloned_cuda_f32_grad())
                    .filter(|buffer| buffer.len() == grad.len());
                if is_strict_device_execution() {
                    match layout.backward_cuda_buffers(
                        grad,
                        cuda_grad.clone(),
                        &a_clone,
                        &b_clone,
                        m_dim,
                        k_dim_a,
                        n_dim,
                    ) {
                        Ok((da_buf, db_buf)) => {
                            a_clone.add_backward_cuda_grad_buffer_only(da_buf);
                            b_clone.add_backward_cuda_grad_buffer_only(db_buf);
                            return;
                        }
                        Err(err) => {
                            panic!(
                                "CUDA matmul backward failed in strict device execution mode: {err}"
                            );
                        }
                    }
                }
                let cuda_result = layout
                    .backward_cuda(grad, cuda_grad, &a_clone, &b_clone, m_dim, k_dim_a, n_dim);
                match cuda_result {
                    Ok(((da, da_buf), (db, db_buf))) => {
                        a_clone.add_backward_grad_with_cuda_buffer(da, Some(da_buf));
                        b_clone.add_backward_grad_with_cuda_buffer(db, Some(db_buf));
                    }
                    Err(err) => {
                        if is_strict_device_execution() {
                            panic!(
                                "CUDA matmul backward failed in strict device execution mode: {err}"
                            );
                        }
                        let (da, db) =
                            layout.backward_cpu(&a_clone, &b_clone, grad, m_dim, k_dim_a, n_dim);
                        if let Some(da) = da {
                            a_clone.add_backward_grad(da);
                        }
                        if let Some(db) = db {
                            b_clone.add_backward_grad(db);
                        }
                    }
                }
            })),
            device: output_device,
        })));
        *output_self.borrow_mut() = Rc::downgrade(&tensor.0);
        return tensor;
    }

    if !build_graph {
        let output_dtype = if a.dtype() == b.dtype() {
            a.dtype()
        } else {
            DType::F32
        };
        let mut out_shape = a_shape.clone();
        *out_shape.last_mut().unwrap() = n_dim;
        if let Some(cuda_out) =
            layout.forward_cuda(a, b, m_dim, k_dim_a, n_dim, output_dtype, &out_shape)
        {
            return cuda_out;
        }
        let res_2d = {
            #[cfg(feature = "cpu-profiling")]
            let _profile = crate::cpu::profiling::region(layout.profile_kernel());
            layout.forward_cpu(a, b, m_dim, k_dim_a, n_dim)
        };
        return Tensor::from_f32_data_no_grad_with_device_dtype(
            res_2d.into_shape(out_shape).unwrap().into_dyn(),
            output_dtype,
            output_device,
        );
    }

    let res_2d = layout.forward_cpu(a, b, m_dim, k_dim_a, n_dim);

    let mut out_shape = a_shape.clone();
    let last_idx = out_shape.len() - 1;
    out_shape[last_idx] = n_dim;

    let result = res_2d.into_shape(out_shape).unwrap().into_dyn();

    let a_clone = a.clone();
    let b_clone = b.clone();

    Tensor(Rc::new(RefCell::new(TensorData {
        data: result.into_shared(),
        f16_data: None,
        bf16_data: None,
        i8_data: None,
        cuda_f32_data: None,
        cuda_f16_data: None,
        cuda_bf16_data: None,
        cuda_i8_data: None,
        i8_scale: None,
        has_f32_data: true,
        storage_dtype: crate::precision::DType::F32,
        cache_dirty: false,
        is_parameter: false,
        grad: None,
        cuda_f32_grad: None,
        parents: vec![a_clone.clone(), b_clone.clone()],
        requires_grad: true,
        backward_op: Some(std::rc::Rc::new(move |grad: &ndarray::ArrayViewD<f32>| {
            let (da, db) = layout.backward_cpu(&a_clone, &b_clone, grad, m_dim, k_dim_a, n_dim);
            if let Some(da) = da {
                a_clone.add_backward_grad(da);
            }
            if let Some(db) = db {
                b_clone.add_backward_grad(db);
            }
        })),
        device: output_device,
    })))
}

pub(super) fn batch_matmul_forward_cpu_f32(
    lhs: &Tensor,
    rhs: &Tensor,
    b: usize,
    h: usize,
    m: usize,
    n: usize,
) -> ndarray::ArrayD<f32> {
    with_native_f32_compute_view(lhs, |lhs_view| {
        with_native_f32_compute_view(rhs, |rhs_view| {
            let lhs_view = lhs_view.into_dimensionality::<Ix4>().unwrap();
            let rhs_view = rhs_view.into_dimensionality::<Ix4>().unwrap();
            let mut output = Array4::<f32>::zeros((b, h, m, n));

            crate::cpu_profile!(
                "batch_matmul.gemm",
                super::cpu_gemm::f32_batched_matmul(lhs_view, rhs_view, output.view_mut())
            );

            output.into_dyn()
        })
    })
}

// lhs: [B, H, M, K]
// rhs: [B, H, K, N]
// out: [B, H, M, N]
pub fn batch_matmul(lhs: &Tensor, rhs: &Tensor) -> Tensor {
    let output_device = assert_same_device(lhs, rhs, "batch_matmul");
    let build_graph = !is_no_grad() && (lhs.requires_grad() || rhs.requires_grad());
    let cuda_native_supported = output_device == crate::device::Device::Cuda;
    assert_native_device_support(output_device, "batch_matmul", cuda_native_supported);

    let lhs_shape = lhs.shape_vec();
    let rhs_shape = rhs.shape_vec();
    assert_eq!(lhs_shape.len(), 4, "batch_matmul lhs must be [B,H,M,K]");
    assert_eq!(rhs_shape.len(), 4, "batch_matmul rhs must be [B,H,K,N]");

    let (b, h, m, k) = (lhs_shape[0], lhs_shape[1], lhs_shape[2], lhs_shape[3]);
    let (b2, h2, k2, n) = (rhs_shape[0], rhs_shape[1], rhs_shape[2], rhs_shape[3]);

    assert_eq!(b, b2, "batch dim mismatch");
    assert_eq!(h, h2, "head dim mismatch");
    assert_eq!(k, k2, "k dim mismatch");
    let dims = BatchMatmulDims { b, h, m, k, n };

    if [b, h, m, k, n].contains(&0) {
        return zero_matmul_output(lhs, rhs, &[b, h, m, n], build_graph);
    }

    if build_graph
        && output_device == crate::device::Device::Cuda
        && let Some(buffer) = try_cuda_batch_matmul_buffer(lhs, rhs, b, h, m, k, n)
    {
        let lhs_clone = lhs.clone();
        let rhs_clone = rhs.clone();
        let out_shape = vec![b, h, m, n];
        let output_self = Rc::new(RefCell::new(std::rc::Weak::<RefCell<TensorData>>::new()));
        let output_self_for_backward = output_self.clone();
        let tensor = Tensor(Rc::new(RefCell::new(TensorData {
            data: ndarray::ArrayD::<f32>::zeros(ndarray::IxDyn(&out_shape)).into_shared(),
            f16_data: None,
            bf16_data: None,
            i8_data: None,
            cuda_f32_data: Some(buffer),
            cuda_f16_data: None,
            cuda_bf16_data: None,
            cuda_i8_data: None,
            i8_scale: None,
            has_f32_data: false,
            storage_dtype: crate::precision::DType::F32,
            cache_dirty: false,
            is_parameter: false,
            grad: None,
            cuda_f32_grad: None,
            parents: vec![lhs_clone.clone(), rhs_clone.clone()],
            backward_op: Some(std::rc::Rc::new(move |grad: &ndarray::ArrayViewD<f32>| {
                let cuda_grad = output_self_for_backward
                    .borrow()
                    .upgrade()
                    .map(Tensor)
                    .and_then(|output| output.cloned_cuda_f32_grad())
                    .filter(|buffer| buffer.len() == grad.len());
                if is_strict_device_execution() {
                    match try_cuda_training_batch_matmul_backward_buffers(
                        grad,
                        cuda_grad.clone(),
                        &lhs_clone,
                        &rhs_clone,
                        dims,
                    ) {
                        Ok((d_lhs_buf, d_rhs_buf)) => {
                            lhs_clone.add_backward_cuda_grad_buffer_only(d_lhs_buf);
                            rhs_clone.add_backward_cuda_grad_buffer_only(d_rhs_buf);
                            return;
                        }
                        Err(err) => {
                            panic!(
                                "CUDA batch_matmul backward failed in strict device execution mode: {err}"
                            );
                        }
                    }
                }
                let cuda_result = try_cuda_training_batch_matmul_backward(
                    grad, cuda_grad, &lhs_clone, &rhs_clone, dims,
                );
                match cuda_result {
                    Ok(((d_lhs, d_lhs_buf), (d_rhs, d_rhs_buf))) => {
                        lhs_clone.add_backward_grad_with_cuda_buffer(d_lhs, Some(d_lhs_buf));
                        rhs_clone.add_backward_grad_with_cuda_buffer(d_rhs, Some(d_rhs_buf));
                    }
                    Err(err) => {
                        if is_strict_device_execution() {
                            panic!(
                                "CUDA batch_matmul backward failed in strict device execution mode: {err}"
                            );
                        }
                        let (d_lhs, d_rhs) = batch_matmul_backward_cpu_f32(
                            &lhs_clone, &rhs_clone, grad, b, h, m, k, n,
                        );
                        lhs_clone.add_backward_grad(d_lhs);
                        rhs_clone.add_backward_grad(d_rhs);
                    }
                }
            })),
            requires_grad: true,
            device: output_device,
        })));
        *output_self.borrow_mut() = Rc::downgrade(&tensor.0);
        return tensor;
    }

    if !build_graph {
        let output_dtype = if lhs.dtype() == rhs.dtype() {
            lhs.dtype()
        } else {
            DType::F32
        };
        if let Some(cuda_out) = try_cuda_batch_matmul(lhs, rhs, dims, output_dtype) {
            return cuda_out;
        }
        let output_dyn = batch_matmul_forward_cpu_f32(lhs, rhs, b, h, m, n);

        return crate::cpu_profile!(
            "batch_matmul.output",
            Tensor::from_f32_data_no_grad_with_device_dtype(
                output_dyn,
                output_dtype,
                output_device,
            )
        );
    }

    let output_dyn = batch_matmul_forward_cpu_f32(lhs, rhs, b, h, m, n);

    let lhs_clone = lhs.clone();
    let rhs_clone = rhs.clone();

    Tensor(Rc::new(RefCell::new(TensorData {
        data: output_dyn.into_shared(),
        f16_data: None,
        bf16_data: None,
        i8_data: None,
        cuda_f32_data: None,
        cuda_f16_data: None,
        cuda_bf16_data: None,
        cuda_i8_data: None,
        i8_scale: None,
        has_f32_data: true,
        storage_dtype: crate::precision::DType::F32,
        cache_dirty: false,
        is_parameter: false,
        grad: None,
        cuda_f32_grad: None,
        parents: vec![lhs_clone.clone(), rhs_clone.clone()],
        backward_op: Some(std::rc::Rc::new(move |grad: &ndarray::ArrayViewD<f32>| {
            let (d_lhs, d_rhs) =
                batch_matmul_backward_cpu_f32(&lhs_clone, &rhs_clone, grad, b, h, m, k, n);
            lhs_clone.add_backward_grad(d_lhs);
            rhs_clone.add_backward_grad(d_rhs);
        })),
        requires_grad: true,
        device: output_device,
    })))
}
