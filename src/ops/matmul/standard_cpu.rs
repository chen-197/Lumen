//! Native RHS panels for standard A[..., K] @ B[K, N] products.
use super::*;

pub(super) fn try_forward(
    a: &Tensor,
    b: &Tensor,
    m: usize,
    k: usize,
    n: usize,
) -> Option<Array2<f32>> {
    if !a.is_cpu() || !b.is_cpu() {
        return None;
    }
    if (1..8).contains(&m) && k >= 8 && n >= 16 {
        // F32 has no conversion to avoid; let GEMM reuse weights across rows
        // when the output is narrow. Streaming favors vectors/wider outputs.
        if m > 1 && n < 256 && b.dtype() == DType::F32 {
            return None;
        }
        return try_small(a, b, m, k, n);
    }
    // Multiple rows and a substantial reduction amortize packing and dispatch.
    // Keep small products on the general GEMM path, independent of CPU model.
    if m < 8 || k < 128 || n < 64 {
        return None;
    }
    let pair_bf16 = a.dtype() == DType::BF16
        && b.dtype() == DType::BF16
        && crate::arch::x86_avx512_bf16_kernel_runtime_available();
    let available = match b.dtype() {
        DType::F16 => cpu_gemm::available(SliceRef::F16(&[])),
        DType::BF16 => pair_bf16 || cpu_gemm::available(SliceRef::BF16(&[])),
        DType::I8 => cpu_gemm::available(SliceRef::I8(&[], 1.)),
        DType::F32 => false,
    };
    if !available {
        return None;
    }
    if pair_bf16 {
        return a.with_storage_view_preferring(StoragePreference::Native, |input| {
            b.with_storage_view_preferring(StoragePreference::Native, |weights| {
                let TensorStorageView::BF16(input) = input else {
                    return None;
                };
                let TensorStorageView::BF16(weights) = weights else {
                    return None;
                };
                let packed = prefill::pack_bf16_inputs(input, m, k);
                let weights = weights.into_dimensionality::<Ix2>().unwrap();
                Some(cpu_gemm::bf16_standard_matmul(&packed, weights))
            })
        });
    }
    // I8 native_storage_owned is an Arc clone, retaining its quantization scale.
    // Other dtypes borrow their original strides; no whole-RHS conversion occurs.
    if b.dtype() == DType::I8 {
        let TensorStorageOwned::I8(weights, scale) = b.native_storage_owned() else {
            return None;
        };
        return Some(lowp(
            a,
            cpu_gemm::NativeRhsView::I8(
                weights.view().into_dimensionality::<Ix2>().unwrap(),
                scale,
            ),
            (m, k, n),
        ));
    }
    b.with_storage_view_preferring(StoragePreference::Native, |weights| match weights {
        TensorStorageView::F16(weights) => Some(lowp(
            a,
            cpu_gemm::NativeRhsView::F16(weights.into_dimensionality::<Ix2>().unwrap()),
            (m, k, n),
        )),
        TensorStorageView::BF16(weights) => Some(lowp(
            a,
            cpu_gemm::NativeRhsView::BF16(weights.into_dimensionality::<Ix2>().unwrap()),
            (m, k, n),
        )),
        _ => None,
    })
}

// Keep the dtype variants out of the general matrix-product dispatcher.
#[inline(never)]
fn try_small(a: &Tensor, b: &Tensor, m: usize, k: usize, n: usize) -> Option<Array2<f32>> {
    fn run(weights: SliceRef<'_>, a: &Tensor, m: usize, k: usize, n: usize) -> Option<Array2<f32>> {
        if !cpu_gemm::small::available(weights) {
            return None;
        }
        conversion::with_native_f32_compute_view(a, |input| {
            let input = input.as_standard_layout();
            cpu_gemm::small::matmul(input.as_slice().unwrap(), weights, m, k, n)
        })
    }
    if b.dtype() == DType::I8 {
        let TensorStorageOwned::I8(weights, scale) = b.native_storage_owned() else {
            return None;
        };
        return weights
            .as_slice()
            .and_then(|w| run(SliceRef::I8(w, scale), a, m, k, n));
    }
    b.with_storage_view_preferring(StoragePreference::Native, |weights| match weights {
        TensorStorageView::F32(w) => w.as_slice().and_then(|w| run(SliceRef::F32(w), a, m, k, n)),
        TensorStorageView::F16(w) => w.as_slice().and_then(|w| run(SliceRef::F16(w), a, m, k, n)),
        TensorStorageView::BF16(w) => w
            .as_slice()
            .and_then(|w| run(SliceRef::BF16(w), a, m, k, n)),
    })
}

fn lowp(
    a: &Tensor,
    weights: cpu_gemm::NativeRhsView<'_>,
    (m, k, n): (usize, usize, usize),
) -> Array2<f32> {
    conversion::with_native_f32_compute_view(a, |input| {
        let input = input.as_standard_layout();
        let input = input.as_slice().unwrap();
        panels(m, k, n, |first, output| {
            cpu_gemm::standard_panel(input, weights, first, output);
        })
    })
}

fn panels(
    m: usize,
    k: usize,
    n: usize,
    compute: impl Fn(usize, &mut [f32]) + Send + Sync,
) -> Array2<f32> {
    const C: usize = cpu_gemm::CHANNELS;
    let mut output = vec![0.; n.div_ceil(C) * C * m];
    crate::cpu::for_each_chunk_mut(
        &mut output,
        C * m,
        k.saturating_mul(C).saturating_mul(m),
        |(panel, output)| compute(panel * C, output),
    );
    let mut result = Array2::zeros((m, n));
    for (panel, source) in output.chunks(C * m).enumerate() {
        let first = panel * C;
        let count = (n - first).min(C);
        for (row, destination) in result.as_slice_mut().unwrap().chunks_mut(n).enumerate() {
            destination[first..first + count].copy_from_slice(&source[row * C..row * C + count]);
        }
    }
    result
}
