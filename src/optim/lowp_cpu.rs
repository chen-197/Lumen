//! Fused CPU updates: native state storage, F32 arithmetic, one rounding on store.
use super::*;
use crate::autograd::{Device, TensorStorageViewMut};
use half::{bf16, f16, slice::HalfFloatSliceExt};

// Keep small tensors on the caller; amortize scheduling over sizable chunks.
const CHUNK: usize = 16384;
const TILE: usize = 1024;

trait Element: Copy + Send + Sync {
    const BULK_STATE: bool = false;
    fn load(self, scale: f32) -> f32;
    fn store(value: f32, scale: f32) -> Self;
    fn load_slice(src: &[Self], dst: &mut [f32], scale: f32) {
        for (src, dst) in src.iter().zip(dst) {
            *dst = src.load(scale);
        }
    }
    fn store_slice(src: &[f32], dst: &mut [Self], scale: f32) {
        for (src, dst) in src.iter().zip(dst) {
            *dst = Self::store(*src, scale);
        }
    }
    fn with_f32_mut(values: &mut [Self], scale: f32, update: impl FnOnce(&mut [f32])) {
        let mut scratch = [0.0; TILE];
        let scratch = &mut scratch[..values.len()];
        Self::load_slice(values, scratch, scale);
        update(scratch);
        Self::store_slice(scratch, values, scale);
    }
}

impl Element for f32 {
    #[inline]
    fn load(self, _: f32) -> f32 {
        self
    }
    #[inline]
    fn store(value: f32, _: f32) -> Self {
        value
    }
    fn with_f32_mut(values: &mut [Self], _: f32, update: impl FnOnce(&mut [f32])) {
        update(values);
    }
}

macro_rules! float_element {
    ($ty:ty, $bulk:expr) => {
        impl Element for $ty {
            const BULK_STATE: bool = $bulk;
            #[inline]
            fn load(self, _: f32) -> f32 {
                self.to_f32()
            }
            #[inline]
            fn store(value: f32, _: f32) -> Self {
                Self::from_f32(value)
            }
            fn load_slice(src: &[Self], dst: &mut [f32], _: f32) {
                src.convert_to_f32_slice(dst);
            }
            fn store_slice(src: &[f32], dst: &mut [Self], _: f32) {
                dst.convert_from_f32_slice(src);
            }
        }
    };
}
// Bulk half conversion can use the half crate's runtime SIMD dispatch. Scalar
// conversion in each arithmetic iteration prevents efficient F16 vectorization.
float_element!(f16, true);
float_element!(bf16, false);

impl Element for i8 {
    #[inline]
    fn load(self, scale: f32) -> f32 {
        self as f32 * scale
    }
    #[inline]
    fn store(value: f32, scale: f32) -> Self {
        crate::precision::I8Requantizer::new(scale).quantize(value)
    }
}

fn supported(param: &Tensor, states: &[&Tensor]) -> bool {
    param.device() == Device::Cpu
        && !param.0.borrow().cache_dirty
        && states.iter().all(|s| {
            s.device() == Device::Cpu
                && matches!(s.dtype(), DType::F16 | DType::BF16)
                && !s.0.borrow().cache_dirty
                && s.shape_vec() == param.shape_vec()
                && !std::rc::Rc::ptr_eq(&s.0, &param.0)
        })
        && (states.len() < 2 || !std::rc::Rc::ptr_eq(&states[0].0, &states[1].0))
}

// Invalidate derived host caches before mutation, including when a callback panics.
// This is deliberately limited to clean CPU storage; dirty/device storage uses
// the existing fallback, which first resolves its authoritative representation.
fn native_mut<R>(tensor: &Tensor, f: impl FnOnce(TensorStorageViewMut<'_>) -> R) -> R {
    {
        let mut inner = tensor.0.borrow_mut();
        if inner.storage_dtype != DType::F32 && inner.has_f32_data {
            inner.data = ArrayD::zeros(IxDyn(&[0])).into_shared();
            inner.has_f32_data = false;
        } else if inner.storage_dtype == DType::F32 {
            inner.f16_data = None;
            inner.bf16_data = None;
            inner.i8_data = None;
            inner.i8_scale = None;
        }
    }
    tensor.with_native_storage_view_mut(f)
}

pub(super) fn momentum(param: &Tensor, state: &Tensor, lr: f32, momentum: f32) -> bool {
    if !supported(param, &[state]) {
        return false;
    }
    let Some(grad) = param.grad_arc() else {
        return false;
    };
    assert_eq!(param.shape_vec(), grad.shape());
    native_mut(state, |view| match view {
        TensorStorageViewMut::F16(v) => momentum_param(param, v, grad.view(), lr, momentum),
        TensorStorageViewMut::BF16(v) => momentum_param(param, v, grad.view(), lr, momentum),
        _ => unreachable!("checked low-precision state"),
    });
    true
}

fn momentum_param<S: Element>(
    param: &Tensor,
    state: ArrayViewMutD<'_, S>,
    grad: ArrayViewD<'_, f32>,
    lr: f32,
    momentum: f32,
) {
    native_mut(param, |view| match view {
        TensorStorageViewMut::F32(w) => momentum_values(w, state, grad, 1.0, lr, momentum),
        TensorStorageViewMut::F16(w) => momentum_values(w, state, grad, 1.0, lr, momentum),
        TensorStorageViewMut::BF16(w) => momentum_values(w, state, grad, 1.0, lr, momentum),
        TensorStorageViewMut::I8(w, scale) => momentum_values(w, state, grad, scale, lr, momentum),
    });
}

fn momentum_values<W: Element, S: Element>(
    mut w: ArrayViewMutD<'_, W>,
    mut v: ArrayViewMutD<'_, S>,
    g: ArrayViewD<'_, f32>,
    scale: f32,
    lr: f32,
    momentum: f32,
) {
    let update = |w: &mut W, v: &mut S, g: &f32| {
        let next = momentum * v.load(1.0) + *g;
        *w = W::store(w.load(scale) - lr * next, scale);
        *v = S::store(next, 1.0);
    };
    // Flat loops require matching logical order, not just contiguous memory.
    if let (Some(w), Some(v), Some(g)) = (w.as_slice_mut(), v.as_slice_mut(), g.as_slice()) {
        crate::cpu::for_each_chunk2_mut(w, v, CHUNK, CHUNK * 4, |(index, (w, v))| {
            let g = &g[index * CHUNK..index * CHUNK + w.len()];
            if S::BULK_STATE {
                let mut scratch = [0.0; TILE];
                for ((w, v), g) in w
                    .chunks_mut(TILE)
                    .zip(v.chunks_mut(TILE))
                    .zip(g.chunks(TILE))
                {
                    let next = &mut scratch[..w.len()];
                    S::load_slice(v, next, 1.0);
                    W::with_f32_mut(w, scale, |w| {
                        for ((w, v), g) in w.iter_mut().zip(next.iter_mut()).zip(g) {
                            *v = momentum * *v + *g;
                            *w -= lr * *v;
                        }
                    });
                    S::store_slice(next, v, 1.0);
                }
            } else {
                for ((w, v), g) in w.iter_mut().zip(v).zip(g) {
                    update(w, v, g);
                }
            }
        });
    } else {
        Zip::from(w).and(v).and(g).for_each(update);
    }
}

#[derive(Clone, Copy)]
pub(super) struct AdamConfig {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub correction1: f32,
    pub correction2: f32,
    pub eps: f32,
}

pub(super) fn adam(param: &Tensor, m: &Tensor, v: &Tensor, config: AdamConfig) -> bool {
    if !supported(param, &[m, v]) || m.dtype() != v.dtype() {
        return false;
    }
    let Some(grad) = param.grad_arc() else {
        return false;
    };
    assert_eq!(param.shape_vec(), grad.shape());
    native_mut(m, |m| {
        native_mut(v, |v| match (m, v) {
            (TensorStorageViewMut::F16(m), TensorStorageViewMut::F16(v)) => {
                adam_param(param, m, v, grad.view(), config)
            }
            (TensorStorageViewMut::BF16(m), TensorStorageViewMut::BF16(v)) => {
                adam_param(param, m, v, grad.view(), config)
            }
            _ => unreachable!("checked matching low-precision states"),
        })
    });
    true
}

fn adam_param<S: Element>(
    param: &Tensor,
    m: ArrayViewMutD<'_, S>,
    v: ArrayViewMutD<'_, S>,
    grad: ArrayViewD<'_, f32>,
    config: AdamConfig,
) {
    native_mut(param, |view| match view {
        TensorStorageViewMut::F32(w) => adam_values(w, m, v, grad, 1.0, config),
        TensorStorageViewMut::F16(w) => adam_values(w, m, v, grad, 1.0, config),
        TensorStorageViewMut::BF16(w) => adam_values(w, m, v, grad, 1.0, config),
        TensorStorageViewMut::I8(w, scale) => adam_values(w, m, v, grad, scale, config),
    });
}

fn adam_values<W: Element, S: Element>(
    mut w: ArrayViewMutD<'_, W>,
    mut m: ArrayViewMutD<'_, S>,
    mut v: ArrayViewMutD<'_, S>,
    g: ArrayViewD<'_, f32>,
    scale: f32,
    c: AdamConfig,
) {
    let update = |w: &mut W, m: &mut S, v: &mut S, g: &f32| {
        let next_m = c.beta1 * m.load(1.0) + (1.0 - c.beta1) * *g;
        let next_v = c.beta2 * v.load(1.0) + (1.0 - c.beta2) * *g * *g;
        let m_hat = next_m / c.correction1;
        let v_hat = next_v / c.correction2;
        *w = W::store(
            w.load(scale) - c.lr * (m_hat / (v_hat.sqrt() + c.eps)),
            scale,
        );
        // This update uses unrounded moments; only saved state is rounded.
        *m = S::store(next_m, 1.0);
        *v = S::store(next_v, 1.0);
    };
    if let (Some(w), Some(m), Some(v), Some(g)) = (
        w.as_slice_mut(),
        m.as_slice_mut(),
        v.as_slice_mut(),
        g.as_slice(),
    ) {
        crate::cpu::for_each_chunk3_mut(w, m, v, CHUNK, CHUNK * 16, |(index, ((w, m), v))| {
            let g = &g[index * CHUNK..index * CHUNK + w.len()];
            if S::BULK_STATE {
                let mut first = [0.0; TILE];
                let mut second = [0.0; TILE];
                for (((w, m), v), g) in w
                    .chunks_mut(TILE)
                    .zip(m.chunks_mut(TILE))
                    .zip(v.chunks_mut(TILE))
                    .zip(g.chunks(TILE))
                {
                    let nm = &mut first[..w.len()];
                    let nv = &mut second[..w.len()];
                    S::load_slice(m, nm, 1.0);
                    S::load_slice(v, nv, 1.0);
                    W::with_f32_mut(w, scale, |w| {
                        for (((w, m), v), g) in
                            w.iter_mut().zip(nm.iter_mut()).zip(nv.iter_mut()).zip(g)
                        {
                            *m = c.beta1 * *m + (1.0 - c.beta1) * *g;
                            *v = c.beta2 * *v + (1.0 - c.beta2) * *g * *g;
                            let m_hat = *m / c.correction1;
                            let v_hat = *v / c.correction2;
                            *w -= c.lr * (m_hat / (v_hat.sqrt() + c.eps));
                        }
                    });
                    S::store_slice(nm, m, 1.0);
                    S::store_slice(nv, v, 1.0);
                }
            } else {
                for (((w, m), v), g) in w.iter_mut().zip(m).zip(v).zip(g) {
                    update(w, m, v, g);
                }
            }
        });
    } else {
        Zip::from(w).and(m).and(v).and(g).for_each(update);
    }
}
