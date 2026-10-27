use super::*;

#[inline]
pub(crate) fn with_f16_input_as_f32<R>(x: &[f16], f: impl FnOnce(&[f32]) -> R) -> R {
    F16_TO_F32_SCRATCH.with(|scratch| {
        if let Ok(mut scratch) = scratch.try_borrow_mut() {
            if scratch.len() < x.len() {
                scratch.resize(x.len(), 0.0);
            }
            x.convert_to_f32_slice(&mut scratch[..x.len()]);
            return f(&scratch[..x.len()]);
        }

        let mut fallback = vec![0.0f32; x.len()];
        x.convert_to_f32_slice(&mut fallback);
        f(&fallback)
    })
}

#[inline]
pub(crate) fn with_bf16_input_as_f32<R>(x: &[bf16], f: impl FnOnce(&[f32]) -> R) -> R {
    BF16_TO_F32_SCRATCH.with(|scratch| {
        if let Ok(mut scratch) = scratch.try_borrow_mut() {
            if scratch.len() < x.len() {
                scratch.resize(x.len(), 0.0);
            }
            x.convert_to_f32_slice(&mut scratch[..x.len()]);
            return f(&scratch[..x.len()]);
        }

        let mut fallback = vec![0.0f32; x.len()];
        x.convert_to_f32_slice(&mut fallback);
        f(&fallback)
    })
}

#[inline]
pub(crate) fn with_i8_input_as_f32<R>(x: &[i8], scale: f32, f: impl FnOnce(&[f32]) -> R) -> R {
    I8_TO_F32_SCRATCH.with(|scratch| {
        if let Ok(mut scratch) = scratch.try_borrow_mut() {
            if scratch.len() < x.len() {
                scratch.resize(x.len(), 0.0);
            }
            for (dst, &src) in scratch[..x.len()].iter_mut().zip(x.iter()) {
                *dst = src as f32 * scale;
            }
            return f(&scratch[..x.len()]);
        }

        let mut fallback = vec![0.0f32; x.len()];
        for (dst, &src) in fallback.iter_mut().zip(x.iter()) {
            *dst = src as f32 * scale;
        }
        f(&fallback)
    })
}

pub(super) fn with_lowp_f32_compute_view<A: DotElem, R>(
    view: ndarray::ArrayViewD<'_, A>,
    f: impl FnOnce(ndarray::ArrayViewD<'_, f32>) -> R,
) -> R {
    COMPUTE_F32_SCRATCH.with(|slots| {
        let mut retained = if view.len() <= MAX_COMPUTE_SCRATCH_ELEMENTS {
            slots.iter().find_map(|slot| slot.try_borrow_mut().ok())
        } else {
            None
        };
        let mut temporary = Vec::new();
        let raw = retained.as_deref_mut().unwrap_or(&mut temporary);
        if raw.len() < view.len() {
            // Avoid geometric growth beyond the retained storage bound.
            raw.reserve_exact(view.len() - raw.len());
            raw.resize(view.len(), 0.0);
        }
        let raw = &mut raw[..view.len()];
        crate::cpu_profile!("matmul.native_to_f32", {
            if let Some(slice) = view.as_slice() {
                A::copy_to_f32(slice, raw);
            } else {
                for (dst, src) in raw.iter_mut().zip(view.iter()) {
                    *dst = src.to_f32();
                }
            }
        });
        f(ndarray::ArrayViewD::from_shape(view.raw_dim(), &*raw)
            .expect("native F32 compute view shape"))
    })
}

pub(super) fn with_native_f32_compute_view<R>(
    tensor: &Tensor,
    f: impl FnOnce(ndarray::ArrayViewD<'_, f32>) -> R,
) -> R {
    tensor.with_storage_view_preferring(StoragePreference::Native, |view| match view {
        TensorStorageView::F32(view) => f(view),
        TensorStorageView::F16(view) => with_lowp_f32_compute_view(view, f),
        TensorStorageView::BF16(view) => with_lowp_f32_compute_view(view, f),
    })
}
