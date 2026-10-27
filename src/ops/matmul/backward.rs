use super::conversion::with_native_f32_compute_view;
use super::*;

pub(super) fn matmul_backward_f16_slices(
    a: &[f16],
    b: &[f16],
    grad: &[f32],
    m_dim: usize,
    k_dim: usize,
    n_dim: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut da = vec![0.0f32; m_dim * k_dim];
    crate::cpu::for_each_chunk_mut(
        &mut da,
        k_dim.max(1),
        k_dim.saturating_mul(n_dim),
        |(m, da_row)| {
            let grad_row = &grad[m * n_dim..(m + 1) * n_dim];
            for n in 0..n_dim {
                let g = grad_row[n];
                let b_row = &b[n * k_dim..(n + 1) * k_dim];
                for (dst, &bv) in da_row.iter_mut().zip(b_row.iter()) {
                    *dst += g * bv.to_f32();
                }
            }
        },
    );

    let mut db = vec![0.0f32; n_dim * k_dim];
    crate::cpu::for_each_chunk_mut(
        &mut db,
        k_dim.max(1),
        k_dim.saturating_mul(m_dim),
        |(n, db_row)| {
            for m in 0..m_dim {
                let g = grad[m * n_dim + n];
                let a_row = &a[m * k_dim..(m + 1) * k_dim];
                for (dst, &av) in db_row.iter_mut().zip(a_row.iter()) {
                    *dst += g * av.to_f32();
                }
            }
        },
    );

    (da, db)
}

pub(super) fn matmul_backward_bf16_slices(
    a: &[bf16],
    b: &[bf16],
    grad: &[f32],
    m_dim: usize,
    k_dim: usize,
    n_dim: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut da = vec![0.0f32; m_dim * k_dim];
    crate::cpu::for_each_chunk_mut(
        &mut da,
        k_dim.max(1),
        k_dim.saturating_mul(n_dim),
        |(m, da_row)| {
            let grad_row = &grad[m * n_dim..(m + 1) * n_dim];
            for n in 0..n_dim {
                let g = grad_row[n];
                let b_row = &b[n * k_dim..(n + 1) * k_dim];
                for (dst, &bv) in da_row.iter_mut().zip(b_row.iter()) {
                    *dst += g * bv.to_f32();
                }
            }
        },
    );

    let mut db = vec![0.0f32; n_dim * k_dim];
    crate::cpu::for_each_chunk_mut(
        &mut db,
        k_dim.max(1),
        k_dim.saturating_mul(m_dim),
        |(n, db_row)| {
            for m in 0..m_dim {
                let g = grad[m * n_dim + n];
                let a_row = &a[m * k_dim..(m + 1) * k_dim];
                for (dst, &av) in db_row.iter_mut().zip(a_row.iter()) {
                    *dst += g * av.to_f32();
                }
            }
        },
    );

    (da, db)
}

pub(super) fn matmul_backward_i8_slices(
    a: &[i8],
    a_scale: f32,
    b: &[i8],
    b_scale: f32,
    grad: &[f32],
    m_dim: usize,
    k_dim: usize,
    n_dim: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut da = vec![0.0f32; m_dim * k_dim];
    crate::cpu::for_each_chunk_mut(
        &mut da,
        k_dim.max(1),
        k_dim.saturating_mul(n_dim),
        |(m, da_row)| {
            let grad_row = &grad[m * n_dim..(m + 1) * n_dim];
            for n in 0..n_dim {
                let g = grad_row[n] * b_scale;
                let b_row = &b[n * k_dim..(n + 1) * k_dim];
                for (dst, &bv) in da_row.iter_mut().zip(b_row.iter()) {
                    *dst += g * (bv as f32);
                }
            }
        },
    );

    let mut db = vec![0.0f32; n_dim * k_dim];
    crate::cpu::for_each_chunk_mut(
        &mut db,
        k_dim.max(1),
        k_dim.saturating_mul(m_dim),
        |(n, db_row)| {
            for m in 0..m_dim {
                let g = grad[m * n_dim + n] * a_scale;
                let a_row = &a[m * k_dim..(m + 1) * k_dim];
                for (dst, &av) in db_row.iter_mut().zip(a_row.iter()) {
                    *dst += g * (av as f32);
                }
            }
        },
    );

    (da, db)
}

pub(super) fn matmul_backward_rhs_i8_slices<T: DotElem>(
    a: &[T],
    b: &[i8],
    b_scale: f32,
    grad: &[f32],
    m_dim: usize,
    k_dim: usize,
    n_dim: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut da = vec![0.0f32; m_dim * k_dim];
    crate::cpu::for_each_chunk_mut(
        &mut da,
        k_dim.max(1),
        k_dim.saturating_mul(n_dim),
        |(m, da_row)| {
            let grad_row = &grad[m * n_dim..(m + 1) * n_dim];
            for n in 0..n_dim {
                let g = grad_row[n] * b_scale;
                let b_row = &b[n * k_dim..(n + 1) * k_dim];
                for (dst, &bv) in da_row.iter_mut().zip(b_row.iter()) {
                    *dst += g * (bv as f32);
                }
            }
        },
    );

    let mut db = vec![0.0f32; n_dim * k_dim];
    crate::cpu::for_each_chunk_mut(
        &mut db,
        k_dim.max(1),
        k_dim.saturating_mul(m_dim),
        |(n, db_row)| {
            for m in 0..m_dim {
                let g = grad[m * n_dim + n];
                let a_row = &a[m * k_dim..(m + 1) * k_dim];
                for (dst, &av) in db_row.iter_mut().zip(a_row.iter()) {
                    *dst += g * av.to_f32();
                }
            }
        },
    );

    (da, db)
}

pub(super) fn matmul_backward_lhs_i8_slices<T: DotElem>(
    a: &[i8],
    a_scale: f32,
    b: &[T],
    grad: &[f32],
    m_dim: usize,
    k_dim: usize,
    n_dim: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut da = vec![0.0f32; m_dim * k_dim];
    crate::cpu::for_each_chunk_mut(
        &mut da,
        k_dim.max(1),
        k_dim.saturating_mul(n_dim),
        |(m, da_row)| {
            let grad_row = &grad[m * n_dim..(m + 1) * n_dim];
            for n in 0..n_dim {
                let g = grad_row[n];
                let b_row = &b[n * k_dim..(n + 1) * k_dim];
                for (dst, &bv) in da_row.iter_mut().zip(b_row.iter()) {
                    *dst += g * bv.to_f32();
                }
            }
        },
    );

    let mut db = vec![0.0f32; n_dim * k_dim];
    crate::cpu::for_each_chunk_mut(
        &mut db,
        k_dim.max(1),
        k_dim.saturating_mul(m_dim),
        |(n, db_row)| {
            for m in 0..m_dim {
                let g = grad[m * n_dim + n] * a_scale;
                let a_row = &a[m * k_dim..(m + 1) * k_dim];
                for (dst, &av) in db_row.iter_mut().zip(a_row.iter()) {
                    *dst += g * (av as f32);
                }
            }
        },
    );

    (da, db)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn batch_matmul_backward_typed_slices<T: DotElem>(
    lhs: &[T],
    rhs: &[T],
    grad: &[f32],
    batch_count: usize,
    m_dim: usize,
    k_dim: usize,
    n_dim: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut d_lhs = vec![0.0f32; batch_count * m_dim * k_dim];
    crate::cpu::for_each_chunk_mut(
        &mut d_lhs,
        k_dim.max(1),
        k_dim.saturating_mul(n_dim),
        |(batch_row, d_lhs_row)| {
            let batch = batch_row / m_dim;
            let row = batch_row - batch * m_dim;
            let grad_batch = &grad[batch * m_dim * n_dim..(batch + 1) * m_dim * n_dim];
            let rhs_batch = &rhs[batch * k_dim * n_dim..(batch + 1) * k_dim * n_dim];
            let grad_row = &grad_batch[row * n_dim..(row + 1) * n_dim];
            for kk in 0..k_dim {
                let mut acc = 0.0f32;
                for col in 0..n_dim {
                    acc += grad_row[col] * rhs_batch[kk * n_dim + col].to_f32();
                }
                d_lhs_row[kk] = acc;
            }
        },
    );

    let mut d_rhs = vec![0.0f32; batch_count * k_dim * n_dim];
    crate::cpu::for_each_chunk_mut(
        &mut d_rhs,
        n_dim.max(1),
        n_dim.saturating_mul(m_dim),
        |(batch_kk, d_rhs_row)| {
            let batch = batch_kk / k_dim;
            let kk = batch_kk - batch * k_dim;
            let lhs_batch = &lhs[batch * m_dim * k_dim..(batch + 1) * m_dim * k_dim];
            let grad_batch = &grad[batch * m_dim * n_dim..(batch + 1) * m_dim * n_dim];
            // Keep each output's ascending-row reduction order, while updating
            // contiguous columns together and converting the lhs value once.
            for row in 0..m_dim {
                let value = lhs_batch[row * k_dim + kk].to_f32();
                let grad_row = &grad_batch[row * n_dim..(row + 1) * n_dim];
                for (dst, &g) in d_rhs_row.iter_mut().zip(grad_row) {
                    *dst += value * g;
                }
            }
        },
    );

    (d_lhs, d_rhs)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn batch_matmul_backward_i8_slices(
    lhs: &[i8],
    lhs_scale: f32,
    rhs: &[i8],
    rhs_scale: f32,
    grad: &[f32],
    batch_count: usize,
    m_dim: usize,
    k_dim: usize,
    n_dim: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut d_lhs = vec![0.0f32; batch_count * m_dim * k_dim];
    crate::cpu::for_each_chunk_mut(
        &mut d_lhs,
        k_dim.max(1),
        k_dim.saturating_mul(n_dim),
        |(batch_row, d_lhs_row)| {
            let batch = batch_row / m_dim;
            let row = batch_row - batch * m_dim;
            let grad_batch = &grad[batch * m_dim * n_dim..(batch + 1) * m_dim * n_dim];
            let rhs_batch = &rhs[batch * k_dim * n_dim..(batch + 1) * k_dim * n_dim];
            let grad_row = &grad_batch[row * n_dim..(row + 1) * n_dim];
            for kk in 0..k_dim {
                let mut acc = 0.0f32;
                for col in 0..n_dim {
                    acc += grad_row[col] * (rhs_batch[kk * n_dim + col] as f32) * rhs_scale;
                }
                d_lhs_row[kk] = acc;
            }
        },
    );

    let mut d_rhs = vec![0.0f32; batch_count * k_dim * n_dim];
    crate::cpu::for_each_chunk_mut(
        &mut d_rhs,
        n_dim.max(1),
        n_dim.saturating_mul(m_dim),
        |(batch_kk, d_rhs_row)| {
            let batch = batch_kk / k_dim;
            let kk = batch_kk - batch * k_dim;
            let lhs_batch = &lhs[batch * m_dim * k_dim..(batch + 1) * m_dim * k_dim];
            let grad_batch = &grad[batch * m_dim * n_dim..(batch + 1) * m_dim * n_dim];
            for row in 0..m_dim {
                // Retain the original (integer * scale) * upstream ordering.
                let value = (lhs_batch[row * k_dim + kk] as f32) * lhs_scale;
                let grad_row = &grad_batch[row * n_dim..(row + 1) * n_dim];
                for (dst, &g) in d_rhs_row.iter_mut().zip(grad_row) {
                    *dst += value * g;
                }
            }
        },
    );

    (d_lhs, d_rhs)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn batch_matmul_backward_rhs_i8_slices<T: DotElem>(
    lhs: &[T],
    rhs: &[i8],
    rhs_scale: f32,
    grad: &[f32],
    batch_count: usize,
    m_dim: usize,
    k_dim: usize,
    n_dim: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut d_lhs = vec![0.0f32; batch_count * m_dim * k_dim];
    crate::cpu::for_each_chunk_mut(
        &mut d_lhs,
        k_dim.max(1),
        k_dim.saturating_mul(n_dim),
        |(batch_row, d_lhs_row)| {
            let batch = batch_row / m_dim;
            let row = batch_row - batch * m_dim;
            let grad_batch = &grad[batch * m_dim * n_dim..(batch + 1) * m_dim * n_dim];
            let rhs_batch = &rhs[batch * k_dim * n_dim..(batch + 1) * k_dim * n_dim];
            let grad_row = &grad_batch[row * n_dim..(row + 1) * n_dim];
            for kk in 0..k_dim {
                let mut acc = 0.0f32;
                for col in 0..n_dim {
                    acc += grad_row[col] * (rhs_batch[kk * n_dim + col] as f32) * rhs_scale;
                }
                d_lhs_row[kk] = acc;
            }
        },
    );

    let mut d_rhs = vec![0.0f32; batch_count * k_dim * n_dim];
    crate::cpu::for_each_chunk_mut(
        &mut d_rhs,
        n_dim.max(1),
        n_dim.saturating_mul(m_dim),
        |(batch_kk, d_rhs_row)| {
            let batch = batch_kk / k_dim;
            let kk = batch_kk - batch * k_dim;
            let lhs_batch = &lhs[batch * m_dim * k_dim..(batch + 1) * m_dim * k_dim];
            let grad_batch = &grad[batch * m_dim * n_dim..(batch + 1) * m_dim * n_dim];
            // Keep each output's ascending-row reduction order, while updating
            // contiguous columns together and converting the lhs value once.
            for row in 0..m_dim {
                let value = lhs_batch[row * k_dim + kk].to_f32();
                let grad_row = &grad_batch[row * n_dim..(row + 1) * n_dim];
                for (dst, &g) in d_rhs_row.iter_mut().zip(grad_row) {
                    *dst += value * g;
                }
            }
        },
    );

    (d_lhs, d_rhs)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn batch_matmul_backward_lhs_i8_slices<T: DotElem>(
    lhs: &[i8],
    lhs_scale: f32,
    rhs: &[T],
    grad: &[f32],
    batch_count: usize,
    m_dim: usize,
    k_dim: usize,
    n_dim: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut d_lhs = vec![0.0f32; batch_count * m_dim * k_dim];
    crate::cpu::for_each_chunk_mut(
        &mut d_lhs,
        k_dim.max(1),
        k_dim.saturating_mul(n_dim),
        |(batch_row, d_lhs_row)| {
            let batch = batch_row / m_dim;
            let row = batch_row - batch * m_dim;
            let grad_batch = &grad[batch * m_dim * n_dim..(batch + 1) * m_dim * n_dim];
            let rhs_batch = &rhs[batch * k_dim * n_dim..(batch + 1) * k_dim * n_dim];
            let grad_row = &grad_batch[row * n_dim..(row + 1) * n_dim];
            for kk in 0..k_dim {
                let mut acc = 0.0f32;
                for col in 0..n_dim {
                    acc += grad_row[col] * rhs_batch[kk * n_dim + col].to_f32();
                }
                d_lhs_row[kk] = acc;
            }
        },
    );

    let mut d_rhs = vec![0.0f32; batch_count * k_dim * n_dim];
    crate::cpu::for_each_chunk_mut(
        &mut d_rhs,
        n_dim.max(1),
        n_dim.saturating_mul(m_dim),
        |(batch_kk, d_rhs_row)| {
            let batch = batch_kk / k_dim;
            let kk = batch_kk - batch * k_dim;
            let lhs_batch = &lhs[batch * m_dim * k_dim..(batch + 1) * m_dim * k_dim];
            let grad_batch = &grad[batch * m_dim * n_dim..(batch + 1) * m_dim * n_dim];
            for row in 0..m_dim {
                // Retain the original (integer * scale) * upstream ordering.
                let value = (lhs_batch[row * k_dim + kk] as f32) * lhs_scale;
                let grad_row = &grad_batch[row * n_dim..(row + 1) * n_dim];
                for (dst, &g) in d_rhs_row.iter_mut().zip(grad_row) {
                    *dst += value * g;
                }
            }
        },
    );

    (d_lhs, d_rhs)
}

pub(super) fn try_matmul_backward_cpu_native_lowp(
    a: &Tensor,
    b: &Tensor,
    grad: &ndarray::ArrayViewD<'_, f32>,
    m_dim: usize,
    k_dim: usize,
    n_dim: usize,
) -> Option<(ndarray::ArrayD<f32>, ndarray::ArrayD<f32>)> {
    if grad.len() != m_dim * n_dim {
        return None;
    }

    macro_rules! slice_or_owned {
        ($array:expr, $owned:ident) => {
            if let Some(slice) = $array.as_slice() {
                slice
            } else {
                $owned = $array.iter().copied().collect::<Vec<_>>();
                $owned.as_slice()
            }
        };
    }

    let grad_owned;
    let grad_slice = if let Some(slice) = grad.as_slice() {
        slice
    } else {
        grad_owned = grad.iter().copied().collect::<Vec<_>>();
        grad_owned.as_slice()
    };
    let a_shape = a.shape_vec();

    let (da, db) = match (a.native_storage_owned(), b.native_storage_owned()) {
        (TensorStorageOwned::F16(a_data), TensorStorageOwned::F16(b_data)) => {
            let a_owned;
            let b_owned;
            matmul_backward_f16_slices(
                slice_or_owned!(a_data, a_owned),
                slice_or_owned!(b_data, b_owned),
                grad_slice,
                m_dim,
                k_dim,
                n_dim,
            )
        }
        (TensorStorageOwned::BF16(a_data), TensorStorageOwned::BF16(b_data)) => {
            let a_owned;
            let b_owned;
            matmul_backward_bf16_slices(
                slice_or_owned!(a_data, a_owned),
                slice_or_owned!(b_data, b_owned),
                grad_slice,
                m_dim,
                k_dim,
                n_dim,
            )
        }
        (TensorStorageOwned::I8(a_data, a_scale), TensorStorageOwned::I8(b_data, b_scale)) => {
            let a_owned;
            let b_owned;
            matmul_backward_i8_slices(
                slice_or_owned!(a_data, a_owned),
                a_scale,
                slice_or_owned!(b_data, b_owned),
                b_scale,
                grad_slice,
                m_dim,
                k_dim,
                n_dim,
            )
        }
        (TensorStorageOwned::F32(a_data), TensorStorageOwned::I8(b_data, b_scale)) => {
            let a_owned;
            let b_owned;
            matmul_backward_rhs_i8_slices(
                slice_or_owned!(a_data, a_owned),
                slice_or_owned!(b_data, b_owned),
                b_scale,
                grad_slice,
                m_dim,
                k_dim,
                n_dim,
            )
        }
        (TensorStorageOwned::F16(a_data), TensorStorageOwned::I8(b_data, b_scale)) => {
            let a_owned;
            let b_owned;
            matmul_backward_rhs_i8_slices(
                slice_or_owned!(a_data, a_owned),
                slice_or_owned!(b_data, b_owned),
                b_scale,
                grad_slice,
                m_dim,
                k_dim,
                n_dim,
            )
        }
        (TensorStorageOwned::BF16(a_data), TensorStorageOwned::I8(b_data, b_scale)) => {
            let a_owned;
            let b_owned;
            matmul_backward_rhs_i8_slices(
                slice_or_owned!(a_data, a_owned),
                slice_or_owned!(b_data, b_owned),
                b_scale,
                grad_slice,
                m_dim,
                k_dim,
                n_dim,
            )
        }
        (TensorStorageOwned::I8(a_data, a_scale), TensorStorageOwned::F32(b_data)) => {
            let a_owned;
            let b_owned;
            matmul_backward_lhs_i8_slices(
                slice_or_owned!(a_data, a_owned),
                a_scale,
                slice_or_owned!(b_data, b_owned),
                grad_slice,
                m_dim,
                k_dim,
                n_dim,
            )
        }
        (TensorStorageOwned::I8(a_data, a_scale), TensorStorageOwned::F16(b_data)) => {
            let a_owned;
            let b_owned;
            matmul_backward_lhs_i8_slices(
                slice_or_owned!(a_data, a_owned),
                a_scale,
                slice_or_owned!(b_data, b_owned),
                grad_slice,
                m_dim,
                k_dim,
                n_dim,
            )
        }
        (TensorStorageOwned::I8(a_data, a_scale), TensorStorageOwned::BF16(b_data)) => {
            let a_owned;
            let b_owned;
            matmul_backward_lhs_i8_slices(
                slice_or_owned!(a_data, a_owned),
                a_scale,
                slice_or_owned!(b_data, b_owned),
                grad_slice,
                m_dim,
                k_dim,
                n_dim,
            )
        }
        _ => return None,
    };

    Some((
        Array2::from_shape_vec((m_dim, k_dim), da)
            .ok()?
            .into_shape(a_shape)
            .ok()?
            .into_dyn(),
        Array2::from_shape_vec((n_dim, k_dim), db).ok()?.into_dyn(),
    ))
}

pub(super) fn matmul_backward_cpu_f32(
    a: &Tensor,
    b: &Tensor,
    grad: &ndarray::ArrayViewD<'_, f32>,
    m_dim: usize,
    k_dim: usize,
    n_dim: usize,
) -> (ndarray::ArrayD<f32>, ndarray::ArrayD<f32>) {
    if let Some(grads) = try_matmul_backward_cpu_native_lowp(a, b, grad, m_dim, k_dim, n_dim) {
        return grads;
    }

    let grad_2d = grad
        .as_standard_layout()
        .into_shape((m_dim, n_dim))
        .expect("Grad reshape failed: non-contiguous gradient?");
    let a_shape = a.shape_vec();

    with_native_f32_compute_view(a, |a_view| {
        with_native_f32_compute_view(b, |b_view| {
            let b_2d = b_view.into_dimensionality::<Ix2>().unwrap();

            let mut da_2d = Array2::<f32>::zeros((m_dim, k_dim));
            super::cpu_gemm::f32_matmul(grad_2d.view(), b_2d.view(), da_2d.view_mut());

            let mut db_2d = Array2::<f32>::zeros((n_dim, k_dim));
            if let Ok(a_2d) = a_view.as_standard_layout().into_shape((m_dim, k_dim)) {
                super::cpu_gemm::f32_matmul(grad_2d.t(), a_2d.view(), db_2d.view_mut());
            } else {
                let a_2d_owned = a_view.to_owned().into_shape((m_dim, k_dim)).unwrap();
                super::cpu_gemm::f32_matmul(grad_2d.t(), a_2d_owned.view(), db_2d.view_mut());
            }

            (
                da_2d.into_shape(a_shape).unwrap().into_dyn(),
                db_2d.into_dyn(),
            )
        })
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn try_batch_matmul_backward_cpu_native_lowp(
    lhs: &Tensor,
    rhs: &Tensor,
    grad: &ndarray::ArrayViewD<'_, f32>,
    b: usize,
    h: usize,
    m: usize,
    k: usize,
    n: usize,
) -> Option<(ndarray::ArrayD<f32>, ndarray::ArrayD<f32>)> {
    if grad.len() != b * h * m * n {
        return None;
    }

    macro_rules! slice_or_owned {
        ($array:expr, $owned:ident) => {
            if let Some(slice) = $array.as_slice() {
                slice
            } else {
                $owned = $array.iter().copied().collect::<Vec<_>>();
                $owned.as_slice()
            }
        };
    }

    let grad_owned;
    let grad_slice = if let Some(slice) = grad.as_slice() {
        slice
    } else {
        grad_owned = grad.iter().copied().collect::<Vec<_>>();
        grad_owned.as_slice()
    };
    let batch_count = b.checked_mul(h)?;

    let (d_lhs, d_rhs) = match (lhs.native_storage_owned(), rhs.native_storage_owned()) {
        (TensorStorageOwned::F16(lhs_data), TensorStorageOwned::F16(rhs_data)) => {
            let lhs_owned;
            let rhs_owned;
            batch_matmul_backward_typed_slices(
                slice_or_owned!(lhs_data, lhs_owned),
                slice_or_owned!(rhs_data, rhs_owned),
                grad_slice,
                batch_count,
                m,
                k,
                n,
            )
        }
        (TensorStorageOwned::BF16(lhs_data), TensorStorageOwned::BF16(rhs_data)) => {
            let lhs_owned;
            let rhs_owned;
            batch_matmul_backward_typed_slices(
                slice_or_owned!(lhs_data, lhs_owned),
                slice_or_owned!(rhs_data, rhs_owned),
                grad_slice,
                batch_count,
                m,
                k,
                n,
            )
        }
        (
            TensorStorageOwned::I8(lhs_data, lhs_scale),
            TensorStorageOwned::I8(rhs_data, rhs_scale),
        ) => {
            let lhs_owned;
            let rhs_owned;
            batch_matmul_backward_i8_slices(
                slice_or_owned!(lhs_data, lhs_owned),
                lhs_scale,
                slice_or_owned!(rhs_data, rhs_owned),
                rhs_scale,
                grad_slice,
                batch_count,
                m,
                k,
                n,
            )
        }
        (TensorStorageOwned::F32(lhs_data), TensorStorageOwned::I8(rhs_data, rhs_scale)) => {
            let lhs_owned;
            let rhs_owned;
            batch_matmul_backward_rhs_i8_slices(
                slice_or_owned!(lhs_data, lhs_owned),
                slice_or_owned!(rhs_data, rhs_owned),
                rhs_scale,
                grad_slice,
                batch_count,
                m,
                k,
                n,
            )
        }
        (TensorStorageOwned::F16(lhs_data), TensorStorageOwned::I8(rhs_data, rhs_scale)) => {
            let lhs_owned;
            let rhs_owned;
            batch_matmul_backward_rhs_i8_slices(
                slice_or_owned!(lhs_data, lhs_owned),
                slice_or_owned!(rhs_data, rhs_owned),
                rhs_scale,
                grad_slice,
                batch_count,
                m,
                k,
                n,
            )
        }
        (TensorStorageOwned::BF16(lhs_data), TensorStorageOwned::I8(rhs_data, rhs_scale)) => {
            let lhs_owned;
            let rhs_owned;
            batch_matmul_backward_rhs_i8_slices(
                slice_or_owned!(lhs_data, lhs_owned),
                slice_or_owned!(rhs_data, rhs_owned),
                rhs_scale,
                grad_slice,
                batch_count,
                m,
                k,
                n,
            )
        }
        (TensorStorageOwned::I8(lhs_data, lhs_scale), TensorStorageOwned::F32(rhs_data)) => {
            let lhs_owned;
            let rhs_owned;
            batch_matmul_backward_lhs_i8_slices(
                slice_or_owned!(lhs_data, lhs_owned),
                lhs_scale,
                slice_or_owned!(rhs_data, rhs_owned),
                grad_slice,
                batch_count,
                m,
                k,
                n,
            )
        }
        (TensorStorageOwned::I8(lhs_data, lhs_scale), TensorStorageOwned::F16(rhs_data)) => {
            let lhs_owned;
            let rhs_owned;
            batch_matmul_backward_lhs_i8_slices(
                slice_or_owned!(lhs_data, lhs_owned),
                lhs_scale,
                slice_or_owned!(rhs_data, rhs_owned),
                grad_slice,
                batch_count,
                m,
                k,
                n,
            )
        }
        (TensorStorageOwned::I8(lhs_data, lhs_scale), TensorStorageOwned::BF16(rhs_data)) => {
            let lhs_owned;
            let rhs_owned;
            batch_matmul_backward_lhs_i8_slices(
                slice_or_owned!(lhs_data, lhs_owned),
                lhs_scale,
                slice_or_owned!(rhs_data, rhs_owned),
                grad_slice,
                batch_count,
                m,
                k,
                n,
            )
        }
        _ => return None,
    };

    Some((
        Array4::from_shape_vec((b, h, m, k), d_lhs).ok()?.into_dyn(),
        Array4::from_shape_vec((b, h, k, n), d_rhs).ok()?.into_dyn(),
    ))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn batch_matmul_backward_cpu_f32(
    lhs: &Tensor,
    rhs: &Tensor,
    grad: &ndarray::ArrayViewD<'_, f32>,
    b: usize,
    h: usize,
    m: usize,
    k: usize,
    n: usize,
) -> (ndarray::ArrayD<f32>, ndarray::ArrayD<f32>) {
    if let Some(grads) = try_batch_matmul_backward_cpu_native_lowp(lhs, rhs, grad, b, h, m, k, n) {
        return grads;
    }

    let grad_view = grad.view().into_dimensionality::<Ix4>().unwrap();
    with_native_f32_compute_view(lhs, |lhs_view| {
        with_native_f32_compute_view(rhs, |rhs_view| {
            let lhs_view = lhs_view.into_dimensionality::<Ix4>().unwrap();
            let rhs_view = rhs_view.into_dimensionality::<Ix4>().unwrap();

            let mut d_lhs = Array4::<f32>::zeros((b, h, m, k));
            super::cpu_gemm::f32_batched_matmul(
                grad_view,
                rhs_view.permuted_axes([0, 1, 3, 2]),
                d_lhs.view_mut(),
            );

            let mut d_rhs = Array4::<f32>::zeros((b, h, k, n));
            super::cpu_gemm::f32_batched_matmul(
                lhs_view.permuted_axes([0, 1, 3, 2]),
                grad_view,
                d_rhs.view_mut(),
            );

            (d_lhs.into_dyn(), d_rhs.into_dyn())
        })
    })
}
