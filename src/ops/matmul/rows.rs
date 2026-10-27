use super::matvec::{
    matvec_rowmajor_serial, matvec_rowmajor_serial_f32_bf16, matvec_rowmajor_serial_f32_f16,
    matvec_rowmajor_serial_f32_i8,
};
use super::*;

// Keep the tensor/precision state on the caller. Workers receive only numeric
// views, and the existing scheduler marks each row group as a kernel region so
// any MatVec called by a row cannot recursively oversubscribe the pool.
pub(super) fn matmul_row_fallback<A: Copy + Send + Sync>(
    input: ndarray::ArrayViewD<'_, A>,
    m: usize,
    k: usize,
    n: usize,
    compute: impl Fn(&[A], &mut [f32]) + Send + Sync,
) -> Array2<f32> {
    let mut output = Array2::zeros((m, n));
    if m == 0 || n == 0 {
        return output;
    }
    let input = input.as_standard_layout();
    let input = input.into_shape((m, k)).expect("matmul row input shape");
    if m == 1 {
        // No row parallelism is available. Keep the caller's context visible
        // so native MatVec dispatch can use its own column parallelism.
        compute(input.as_slice().unwrap(), output.as_slice_mut().unwrap());
    } else {
        crate::cpu::for_each_indexed(
            output.outer_iter_mut(),
            k.saturating_mul(n),
            |(row, mut output)| {
                let input = input.row(row);
                compute(input.as_slice().unwrap(), output.as_slice_mut().unwrap());
            },
        );
    }
    output
}

pub(super) fn matmul_rows_f32_bf16(
    a_view: ndarray::ArrayViewD<'_, f32>,
    b_view: ndarray::ArrayViewD<'_, bf16>,
    m_dim: usize,
    k_dim: usize,
    n_dim: usize,
) -> Array2<f32> {
    let b_2d = b_view.into_dimensionality::<Ix2>().unwrap();
    let b_owned;
    let b_slice: &[bf16] = if let Some(s) = b_2d.as_slice() {
        s
    } else {
        b_owned = b_2d.as_standard_layout().to_owned();
        b_owned
            .as_slice()
            .expect("standard-layout matmul RHS should be contiguous")
    };

    if m_dim == 1 {
        let a_flat = a_view
            .as_standard_layout()
            .into_shape(k_dim)
            .expect("single-row lhs reshape failed");
        let a_owned;
        let a_slice: &[f32] = if let Some(s) = a_flat.as_slice() {
            s
        } else {
            a_owned = a_flat.iter().copied().collect::<Vec<f32>>();
            a_owned.as_slice()
        };
        let mut out_vec = vec![0.0f32; n_dim];
        matvec_rowmajor_parallel_f32_bf16(a_slice, b_slice, n_dim, k_dim, &mut out_vec);
        return Array2::from_shape_vec((1, n_dim), out_vec)
            .expect("decode matvec shape build failed");
    }

    matmul_row_fallback(a_view, m_dim, k_dim, n_dim, |a_slice, out_slice| {
        matvec_rowmajor_serial_f32_bf16(a_slice, b_slice, n_dim, k_dim, out_slice);
    })
}

pub(super) fn matmul_rows_f32_f16(
    a_view: ndarray::ArrayViewD<'_, f32>,
    b_view: ndarray::ArrayViewD<'_, f16>,
    m_dim: usize,
    k_dim: usize,
    n_dim: usize,
) -> Array2<f32> {
    let b_2d = b_view.into_dimensionality::<Ix2>().unwrap();
    let b_owned;
    let b_slice: &[f16] = if let Some(s) = b_2d.as_slice() {
        s
    } else {
        b_owned = b_2d.as_standard_layout().to_owned();
        b_owned
            .as_slice()
            .expect("standard-layout matmul RHS should be contiguous")
    };

    #[cfg(all(feature = "arm64-fp-kernels", target_arch = "aarch64"))]
    if m_dim == 1 {
        let a_flat = a_view
            .as_standard_layout()
            .into_shape(k_dim)
            .expect("single-row lhs reshape failed");
        let a_owned;
        let a_slice: &[f32] = if let Some(s) = a_flat.as_slice() {
            s
        } else {
            a_owned = a_flat.iter().copied().collect::<Vec<f32>>();
            a_owned.as_slice()
        };
        let mut out_vec = vec![0.0f32; n_dim];
        matvec_rowmajor_parallel_f32_f16(a_slice, b_slice, n_dim, k_dim, &mut out_vec);
        return Array2::from_shape_vec((1, n_dim), out_vec)
            .expect("decode matvec shape build failed");
    }

    matmul_row_fallback(a_view, m_dim, k_dim, n_dim, |a_slice, out_slice| {
        matvec_rowmajor_serial_f32_f16(a_slice, b_slice, n_dim, k_dim, out_slice);
    })
}

pub(super) fn matmul_rows_f16_f32(
    a_view: ndarray::ArrayViewD<'_, f16>,
    b_view: ndarray::ArrayViewD<'_, f32>,
    m_dim: usize,
    k_dim: usize,
    n_dim: usize,
) -> Array2<f32> {
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

    #[cfg(all(feature = "arm64-fp-kernels", target_arch = "aarch64"))]
    if m_dim == 1 {
        let a_flat = a_view
            .as_standard_layout()
            .into_shape(k_dim)
            .expect("single-row lhs reshape failed");
        let a_owned;
        let a_slice: &[f16] = if let Some(s) = a_flat.as_slice() {
            s
        } else {
            a_owned = a_flat.iter().copied().collect::<Vec<f16>>();
            a_owned.as_slice()
        };
        let mut out_vec = vec![0.0f32; n_dim];
        with_f16_input_as_f32(a_slice, |a_f32| {
            matvec_rowmajor_parallel(a_f32, b_slice, n_dim, k_dim, &mut out_vec);
        });
        return Array2::from_shape_vec((1, n_dim), out_vec)
            .expect("decode matvec shape build failed");
    }

    matmul_row_fallback(a_view, m_dim, k_dim, n_dim, |a_slice, out_slice| {
        with_f16_input_as_f32(a_slice, |a_f32| {
            matvec_rowmajor_serial(a_f32, b_slice, n_dim, k_dim, out_slice);
        });
    })
}

pub(super) fn matmul_rows_f16_f16(
    a_view: ndarray::ArrayViewD<'_, f16>,
    b_view: ndarray::ArrayViewD<'_, f16>,
    m_dim: usize,
    k_dim: usize,
    n_dim: usize,
) -> Array2<f32> {
    let b_2d = b_view.into_dimensionality::<Ix2>().unwrap();
    let b_owned;
    let b_slice: &[f16] = if let Some(s) = b_2d.as_slice() {
        s
    } else {
        b_owned = b_2d.as_standard_layout().to_owned();
        b_owned
            .as_slice()
            .expect("standard-layout matmul RHS should be contiguous")
    };

    #[cfg(all(feature = "arm64-fp-kernels", target_arch = "aarch64"))]
    if m_dim == 1 {
        let a_flat = a_view
            .as_standard_layout()
            .into_shape(k_dim)
            .expect("single-row lhs reshape failed");
        let a_owned;
        let a_slice: &[f16] = if let Some(s) = a_flat.as_slice() {
            s
        } else {
            a_owned = a_flat.iter().copied().collect::<Vec<f16>>();
            a_owned.as_slice()
        };
        let mut out_vec = vec![0.0f32; n_dim];
        matvec_rowmajor_parallel_f16_f16(a_slice, b_slice, n_dim, k_dim, &mut out_vec);
        return Array2::from_shape_vec((1, n_dim), out_vec)
            .expect("decode matvec shape build failed");
    }

    matmul_row_fallback(a_view, m_dim, k_dim, n_dim, |a_slice, out_slice| {
        if crate::arch::x86_avx512_fp16_kernel_runtime_available() {
            matvec_rowmajor_parallel_f16_f16(a_slice, b_slice, n_dim, k_dim, out_slice);
        } else {
            with_f16_input_as_f32(a_slice, |a_f32| {
                matvec_rowmajor_serial_f32_f16(a_f32, b_slice, n_dim, k_dim, out_slice);
            });
        }
    })
}

pub(super) fn matmul_rows_f16_slice(
    a_view: ndarray::ArrayViewD<'_, f16>,
    b_rowmajor: SliceRef<'_>,
    m_dim: usize,
    k_dim: usize,
    n_dim: usize,
) -> Array2<f32> {
    #[cfg(all(feature = "arm64-fp-kernels", target_arch = "aarch64"))]
    if m_dim == 1 {
        let a_flat = a_view
            .as_standard_layout()
            .into_shape(k_dim)
            .expect("single-row lhs reshape failed");
        let a_owned;
        let a_slice: &[f16] = if let Some(s) = a_flat.as_slice() {
            s
        } else {
            a_owned = a_flat.iter().copied().collect::<Vec<f16>>();
            a_owned.as_slice()
        };
        let mut out_vec = vec![0.0f32; n_dim];
        matvec_rowmajor_parallel_mixed(
            SliceRef::F16(a_slice),
            b_rowmajor,
            n_dim,
            k_dim,
            &mut out_vec,
        );
        return Array2::from_shape_vec((1, n_dim), out_vec)
            .expect("decode matvec shape build failed");
    }

    matmul_row_fallback(a_view, m_dim, k_dim, n_dim, |a_slice, out_slice| {
        matvec_rowmajor_parallel_mixed(SliceRef::F16(a_slice), b_rowmajor, n_dim, k_dim, out_slice);
    })
}

pub(super) fn matmul_rows_i8_slice(
    a_view: ndarray::ArrayViewD<'_, i8>,
    a_scale: f32,
    b_rowmajor: SliceRef<'_>,
    m_dim: usize,
    k_dim: usize,
    n_dim: usize,
) -> Array2<f32> {
    matmul_row_fallback(a_view, m_dim, k_dim, n_dim, |a_slice, out_slice| {
        matvec_rowmajor_parallel_mixed(
            SliceRef::I8(a_slice, a_scale),
            b_rowmajor,
            n_dim,
            k_dim,
            out_slice,
        );
    })
}

pub(super) fn matmul_rows_bf16_slice(
    a_view: ndarray::ArrayViewD<'_, bf16>,
    b_rowmajor: SliceRef<'_>,
    m_dim: usize,
    k_dim: usize,
    n_dim: usize,
) -> Array2<f32> {
    #[cfg(all(feature = "arm64-fp-kernels", target_arch = "aarch64"))]
    if m_dim == 1 {
        let a_flat = a_view
            .as_standard_layout()
            .into_shape(k_dim)
            .expect("single-row lhs reshape failed");
        let a_owned;
        let a_slice: &[bf16] = if let Some(s) = a_flat.as_slice() {
            s
        } else {
            a_owned = a_flat.iter().copied().collect::<Vec<bf16>>();
            a_owned.as_slice()
        };
        let mut out_vec = vec![0.0f32; n_dim];
        matvec_rowmajor_parallel_mixed(
            SliceRef::BF16(a_slice),
            b_rowmajor,
            n_dim,
            k_dim,
            &mut out_vec,
        );
        return Array2::from_shape_vec((1, n_dim), out_vec)
            .expect("decode matvec shape build failed");
    }

    matmul_row_fallback(a_view, m_dim, k_dim, n_dim, |a_slice, out_slice| {
        matvec_rowmajor_parallel_mixed(
            SliceRef::BF16(a_slice),
            b_rowmajor,
            n_dim,
            k_dim,
            out_slice,
        );
    })
}

pub(super) fn matmul_rows_bf16_f32(
    a_view: ndarray::ArrayViewD<'_, bf16>,
    b_view: ndarray::ArrayViewD<'_, f32>,
    m_dim: usize,
    k_dim: usize,
    n_dim: usize,
) -> Array2<f32> {
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

    #[cfg(all(feature = "arm64-fp-kernels", target_arch = "aarch64"))]
    if m_dim == 1 {
        let a_flat = a_view
            .as_standard_layout()
            .into_shape(k_dim)
            .expect("single-row lhs reshape failed");
        let a_owned;
        let a_slice: &[bf16] = if let Some(s) = a_flat.as_slice() {
            s
        } else {
            a_owned = a_flat.iter().copied().collect::<Vec<bf16>>();
            a_owned.as_slice()
        };
        let mut out_vec = vec![0.0f32; n_dim];
        with_bf16_input_as_f32(a_slice, |a_f32| {
            matvec_rowmajor_parallel(a_f32, b_slice, n_dim, k_dim, &mut out_vec);
        });
        return Array2::from_shape_vec((1, n_dim), out_vec)
            .expect("decode matvec shape build failed");
    }

    matmul_row_fallback(a_view, m_dim, k_dim, n_dim, |a_slice, out_slice| {
        with_bf16_input_as_f32(a_slice, |a_f32| {
            matvec_rowmajor_serial(a_f32, b_slice, n_dim, k_dim, out_slice);
        });
    })
}

pub(super) fn matmul_rows_bf16_bf16(
    a_view: ndarray::ArrayViewD<'_, bf16>,
    b_view: ndarray::ArrayViewD<'_, bf16>,
    m_dim: usize,
    k_dim: usize,
    n_dim: usize,
) -> Array2<f32> {
    let b_2d = b_view.into_dimensionality::<Ix2>().unwrap();
    let b_owned;
    let b_slice: &[bf16] = if let Some(s) = b_2d.as_slice() {
        s
    } else {
        b_owned = b_2d.as_standard_layout().to_owned();
        b_owned
            .as_slice()
            .expect("standard-layout matmul RHS should be contiguous")
    };

    #[cfg(all(feature = "arm64-fp-kernels", target_arch = "aarch64"))]
    if m_dim == 1 {
        let a_flat = a_view
            .as_standard_layout()
            .into_shape(k_dim)
            .expect("single-row lhs reshape failed");
        let a_owned;
        let a_slice: &[bf16] = if let Some(s) = a_flat.as_slice() {
            s
        } else {
            a_owned = a_flat.iter().copied().collect::<Vec<bf16>>();
            a_owned.as_slice()
        };
        let mut out_vec = vec![0.0f32; n_dim];
        with_bf16_input_as_f32(a_slice, |a_f32| {
            matvec_rowmajor_parallel_f32_bf16(a_f32, b_slice, n_dim, k_dim, &mut out_vec);
        });
        return Array2::from_shape_vec((1, n_dim), out_vec)
            .expect("decode matvec shape build failed");
    }

    matmul_row_fallback(a_view, m_dim, k_dim, n_dim, |a_slice, out_slice| {
        matvec_rowmajor_parallel_bf16_bf16(a_slice, b_slice, n_dim, k_dim, out_slice);
    })
}

pub(super) fn matmul_rows_f32_i8(
    a_view: ndarray::ArrayViewD<'_, f32>,
    b_view: ndarray::ArrayViewD<'_, i8>,
    scale: f32,
    m_dim: usize,
    k_dim: usize,
    n_dim: usize,
) -> Array2<f32> {
    let b_2d = b_view.into_dimensionality::<Ix2>().unwrap();
    let b_owned;
    let b_slice: &[i8] = if let Some(s) = b_2d.as_slice() {
        s
    } else {
        b_owned = b_2d.as_standard_layout().to_owned();
        b_owned
            .as_slice()
            .expect("standard-layout matmul RHS should be contiguous")
    };

    matmul_row_fallback(a_view, m_dim, k_dim, n_dim, |a_slice, out_slice| {
        matvec_rowmajor_serial_f32_i8(a_slice, b_slice, scale, n_dim, k_dim, out_slice);
    })
}

pub(super) fn matmul_rows_bf16_i8(
    a_view: ndarray::ArrayViewD<'_, bf16>,
    b_view: ndarray::ArrayViewD<'_, i8>,
    scale: f32,
    m_dim: usize,
    k_dim: usize,
    n_dim: usize,
) -> Array2<f32> {
    let b_2d = b_view.into_dimensionality::<Ix2>().unwrap();
    let b_owned;
    let b_slice: &[i8] = if let Some(s) = b_2d.as_slice() {
        s
    } else {
        b_owned = b_2d.as_standard_layout().to_owned();
        b_owned
            .as_slice()
            .expect("standard-layout matmul RHS should be contiguous")
    };

    matmul_row_fallback(a_view, m_dim, k_dim, n_dim, |a_slice, out_slice| {
        with_bf16_input_as_f32(a_slice, |a_f32| {
            matvec_rowmajor_serial_f32_i8(a_f32, b_slice, scale, n_dim, k_dim, out_slice);
        });
    })
}
