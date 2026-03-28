use std::sync::atomic::{AtomicBool, Ordering};

static PURE_BF16_INFER_LOADING: AtomicBool = AtomicBool::new(false);

pub fn set_pure_bf16_infer_loading(on: bool) {
    PURE_BF16_INFER_LOADING.store(on, Ordering::Relaxed);
}

#[inline]
pub fn is_pure_bf16_infer_loading() -> bool {
    PURE_BF16_INFER_LOADING.load(Ordering::Relaxed)
}
