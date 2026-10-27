use super::conversion::with_lowp_f32_compute_view;
use super::rows::matmul_rows_f32_bf16;
use super::*;

// Pack a small output-channel panel once and reuse it across all prompt tokens.
// Keeping panels local avoids retaining a full F32 copy of low-precision weights.
pub(super) fn prefill_gemm_panels<A: DotElem, B: DotElem>(
    a: ndarray::ArrayViewD<'_, A>,
    b: ndarray::ArrayViewD<'_, B>,
    scales: (f32, f32),
    dims: (usize, usize, usize),
) -> Array2<f32> {
    let (m, k, n) = dims;
    let input: Vec<f32> = a.iter().map(|x| x.to_f32() * scales.0).collect();
    let input = ndarray::ArrayView2::from_shape((m, k), &input).unwrap();
    let weights_owned;
    let weights = if let Some(slice) = b.as_slice() {
        slice
    } else {
        weights_owned = b.iter().copied().collect::<Vec<_>>();
        &weights_owned
    };
    const PANEL_ROWS: usize = 64;
    let mut transposed = vec![0.0f32; n * m];
    crate::cpu::for_each_chunk_mut(
        &mut transposed,
        PANEL_ROWS * m,
        k.saturating_mul(PANEL_ROWS).saturating_mul(m),
        |(panel, output)| {
            let start = panel * PANEL_ROWS;
            let rows = output.len() / m;
            let packed: Vec<f32> = weights[start * k..(start + rows) * k]
                .iter()
                .map(|x| x.to_f32() * scales.1)
                .collect();
            let packed = ndarray::ArrayView2::from_shape((rows, k), &packed).unwrap();
            let mut output = ndarray::ArrayViewMut2::from_shape((rows, m), output).unwrap();
            general_mat_mul(1.0, &packed, &input.t(), 0.0, &mut output);
        },
    );
    Array2::from_shape_vec((n, m), transposed)
        .unwrap()
        .reversed_axes()
        .as_standard_layout()
        .to_owned()
}

pub(super) fn prefill_bf16_rows<A: DotElem>(
    a: ndarray::ArrayViewD<'_, A>,
    b: ndarray::ArrayViewD<'_, bf16>,
    dims: (usize, usize, usize),
) -> Array2<f32> {
    let (m, k, n) = dims;
    let input = a
        .iter()
        .map(|x| bf16::from_f32(x.to_f32()))
        .collect::<Vec<_>>();
    let weights = b.as_standard_layout();
    let weights = weights.as_slice().unwrap();
    let mut transposed = vec![0.0f32; n * m];
    crate::cpu::for_each_chunk_mut(
        &mut transposed,
        16 * m,
        k.saturating_mul(16).saturating_mul(m),
        |(block, output)| {
            for (row, output) in output.chunks_mut(m).enumerate() {
                let weight = &weights[(block * 16 + row) * k..(block * 16 + row + 1) * k];
                let mut token = 0;
                while token + 3 <= m {
                    let (a, b, c) = dot3_bf16_bf16_arch(
                        weight,
                        &input[token * k..(token + 1) * k],
                        &input[(token + 1) * k..(token + 2) * k],
                        &input[(token + 2) * k..(token + 3) * k],
                    )
                    .unwrap();
                    output[token..token + 3].copy_from_slice(&[a, b, c]);
                    token += 3;
                }
                for (token, value) in output.iter_mut().enumerate().skip(token) {
                    *value =
                        dot_bf16_bf16_arch(weight, &input[token * k..(token + 1) * k]).unwrap();
                }
            }
        },
    );
    Array2::from_shape_vec((n, m), transposed)
        .unwrap()
        .reversed_axes()
        .as_standard_layout()
        .to_owned()
}

pub(super) fn prefill_lowp_panels<A: DotElem>(
    a: ndarray::ArrayViewD<'_, A>,
    weights: SliceRef<'_>,
    dims: (usize, usize, usize),
) -> Array2<f32> {
    let (m, k, n) = dims;
    let mut input = vec![0.; m * k];
    if let Some(values) = a.as_slice() {
        A::copy_to_f32(values, &mut input);
    } else {
        for (dst, src) in input.iter_mut().zip(a.iter()) {
            *dst = src.to_f32();
        }
    }
    const CHANNELS: usize = super::cpu_gemm::CHANNELS;
    let mut output = vec![0.; n.div_ceil(CHANNELS) * CHANNELS * m];
    crate::cpu::for_each_chunk_mut(
        &mut output,
        CHANNELS * m,
        k.saturating_mul(CHANNELS).saturating_mul(m),
        |(panel, output)| {
            super::cpu_gemm::panel(&input, weights, k, n, panel * CHANNELS, output);
        },
    );
    let mut result = Array2::zeros((m, n));
    let destination = result.as_slice_mut().unwrap();
    for (panel, source) in output.chunks(CHANNELS * m).enumerate() {
        let first = panel * CHANNELS;
        let count = (n - first).min(CHANNELS);
        for (token, row) in destination.chunks_mut(n).enumerate() {
            row[first..first + count]
                .copy_from_slice(&source[token * CHANNELS..token * CHANNELS + count]);
        }
    }
    result
}

pub(super) fn prefill_mixed_bf16_rows<A: DotElem>(
    input: ndarray::ArrayViewD<'_, A>,
    weights: ndarray::ArrayViewD<'_, bf16>,
    dims: (usize, usize, usize),
) -> Array2<f32> {
    let (m, k, n) = dims;
    with_lowp_f32_compute_view(input, |input| matmul_rows_f32_bf16(input, weights, m, k, n))
}

pub(super) fn try_prefill_gemm(
    a: &Tensor,
    b: &Tensor,
    m: usize,
    k: usize,
    n: usize,
) -> Option<Array2<f32>> {
    if !is_no_grad()
        || !a.is_cpu()
        || !b.is_cpu()
        || !(8..=256).contains(&m)
        || k < 512
        || n < 512
        || a.dtype() == DType::I8
    {
        return None;
    }
    if a.dtype() == DType::F32 && b.dtype() == DType::F32 {
        // Retain the prefill dispatcher contract without packing a complete
        // F32 weight copy. GEMM consumes the original weight strides directly.
        let input = a.data_ref();
        let input = input.as_standard_layout();
        let input = input.into_shape((m, k)).expect("F32 prefill input shape");
        let weights = b.data_ref();
        let weights = weights
            .view()
            .into_dimensionality::<Ix2>()
            .expect("F32 prefill weights shape");
        let mut output = Array2::zeros((m, n));
        super::cpu_gemm::f32_matmul(input.view(), weights.t(), output.view_mut());
        return Some(output);
    }
    if matches!(b.dtype(), DType::F16 | DType::I8) {
        // Packing is amortized by long reductions and multiple input rows.
        // Preserve the existing short-K and small-batch paths.
        if k < 2048 || n < 2048 || m < 32 {
            return None;
        }
        return match b.native_storage_owned() {
            TensorStorageOwned::F16(weights) => {
                let weights = weights.as_standard_layout();
                let weights = SliceRef::F16(weights.as_slice().unwrap());
                if !super::cpu_gemm::available(weights) {
                    return None;
                }
                a.with_storage_view_preferring(StoragePreference::Native, |input| {
                    Some(match input {
                        TensorStorageView::F32(input) => {
                            prefill_lowp_panels(input, weights, (m, k, n))
                        }
                        TensorStorageView::F16(input) => {
                            prefill_lowp_panels(input, weights, (m, k, n))
                        }
                        TensorStorageView::BF16(input) => {
                            prefill_lowp_panels(input, weights, (m, k, n))
                        }
                    })
                })
            }
            TensorStorageOwned::I8(weights, scale) => {
                let weights = weights.as_standard_layout();
                let weights = SliceRef::I8(weights.as_slice().unwrap(), scale);
                if !super::cpu_gemm::available(weights) {
                    return None;
                }
                a.with_storage_view_preferring(StoragePreference::Native, |input| {
                    Some(match input {
                        TensorStorageView::F32(input) => {
                            prefill_lowp_panels(input, weights, (m, k, n))
                        }
                        TensorStorageView::F16(input) => {
                            prefill_lowp_panels(input, weights, (m, k, n))
                        }
                        TensorStorageView::BF16(input) => {
                            prefill_lowp_panels(input, weights, (m, k, n))
                        }
                    })
                })
            }
            _ => unreachable!(),
        };
    }
    a.with_storage_view_preferring(StoragePreference::Native, |a_view| {
        macro_rules! run_input {
            ($input:expr) => {
                b.with_storage_view_preferring(StoragePreference::Native, |weights| match weights {
                    TensorStorageView::F32(weights) => {
                        Some(prefill_gemm_panels($input, weights, (1.0, 1.0), (m, k, n)))
                    }
                    TensorStorageView::BF16(weights)
                        if crate::arch::x86_avx512_bf16_kernel_runtime_available()
                            && a.dtype() == DType::BF16 =>
                    {
                        Some(prefill_bf16_rows($input, weights, (m, k, n)))
                    }
                    TensorStorageView::BF16(weights) => {
                        // F32/F16 inputs must not be rounded again to BF16.
                        // Small batches do not amortize F32 panel conversion.
                        // This crossover is a portable grain heuristic.
                        Some(if m < 64 {
                            prefill_mixed_bf16_rows($input, weights, (m, k, n))
                        } else {
                            prefill_gemm_panels($input, weights, (1.0, 1.0), (m, k, n))
                        })
                    }
                    _ => None,
                })
            };
        }
        match a_view {
            TensorStorageView::F32(input) => run_input!(input),
            TensorStorageView::F16(input) => run_input!(input),
            TensorStorageView::BF16(input) => run_input!(input),
        }
    })
}
