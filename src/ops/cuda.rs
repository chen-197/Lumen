use std::cell::Cell;
use std::sync::Arc;

const CUDA_MATMUL_MIN_WORK: usize = 1 << 18;
const CUDA_BATCH_MATMUL_MIN_WORK: usize = 1 << 18;
const CUDA_ELEMENTWISE_MIN_WORK: usize = 1 << 14;
const CUDA_SOFTMAX_MIN_WORK: usize = 1 << 13;

#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnaryOp {
    Relu = 0,
    Sigmoid = 1,
    Tanh = 2,
    Silu = 3,
    Gelu = 4,
}

#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinaryOp {
    Add = 0,
    Sub = 1,
    Mul = 2,
}

struct CudaBufferInner {
    handle: u64,
    len: usize,
}

#[derive(Clone)]
pub struct CudaBuffer(Arc<CudaBufferInner>);

impl CudaBuffer {
    #[cfg(feature = "cuda")]
    pub(crate) fn from_raw(handle: u64, len: usize) -> Self {
        Self(Arc::new(CudaBufferInner { handle, len }))
    }

    pub fn handle(&self) -> u64 {
        self.0.handle
    }

    pub fn len(&self) -> usize {
        self.0.len
    }

    pub fn is_empty(&self) -> bool {
        self.0.len == 0
    }
}

pub type CudaHostBuffer = (CudaBuffer, Vec<f32>);
pub type CudaTwoHostBuffers = (CudaHostBuffer, CudaHostBuffer);
pub type CudaThreeHostBuffers = (CudaHostBuffer, CudaHostBuffer, CudaHostBuffer);
pub type CudaAdamHostState = (Vec<f32>, Vec<f32>, Vec<f32>);
pub type CudaConv2dBackwardHostBuffers = (
    CudaBuffer,
    Vec<f32>,
    CudaBuffer,
    Vec<f32>,
    Option<CudaHostBuffer>,
);

type BroadcastMetadata = (Vec<usize>, Vec<usize>, Vec<usize>, Vec<usize>, Vec<usize>);

impl Drop for CudaBufferInner {
    fn drop(&mut self) {
        if self.handle != 0 {
            imp::free_f32(self.handle, self.len);
        }
    }
}

fn env_enabled() -> bool {
    std::env::var("LUMEN_CUDA")
        .ok()
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

thread_local! {
    static CUDA_ENABLED: Cell<bool> = Cell::new(env_enabled());
}

pub fn set_enabled(enabled: bool) {
    CUDA_ENABLED.with(|flag| flag.set(enabled));
}

fn is_enabled_requested() -> bool {
    CUDA_ENABLED.with(|flag| flag.get())
}

pub fn is_enabled() -> bool {
    is_enabled_requested() && is_available()
}

pub struct CudaEnabledGuard {
    previous: bool,
}

pub fn set_enabled_scoped(enabled: bool) -> CudaEnabledGuard {
    let previous = is_enabled_requested();
    set_enabled(enabled);
    CudaEnabledGuard { previous }
}

impl Drop for CudaEnabledGuard {
    fn drop(&mut self) {
        set_enabled(self.previous);
    }
}

pub fn should_accelerate_matmul(m: usize, n: usize, k: usize) -> bool {
    is_enabled()
        && m.checked_mul(n)
            .and_then(|value| value.checked_mul(k))
            .is_some_and(|work| work >= CUDA_MATMUL_MIN_WORK)
}

pub fn should_accelerate_batch_matmul(batch_count: usize, m: usize, n: usize, k: usize) -> bool {
    is_enabled()
        && batch_count
            .checked_mul(m)
            .and_then(|value| value.checked_mul(n))
            .and_then(|value| value.checked_mul(k))
            .is_some_and(|work| work >= CUDA_BATCH_MATMUL_MIN_WORK)
}

pub fn should_accelerate_elementwise(len: usize) -> bool {
    is_enabled() && len >= CUDA_ELEMENTWISE_MIN_WORK
}

pub fn should_accelerate_softmax(outer: usize, last_dim: usize) -> bool {
    is_enabled()
        && outer
            .checked_mul(last_dim)
            .is_some_and(|work| work >= CUDA_SOFTMAX_MIN_WORK)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuda_enabled_flag_is_thread_local() {
        set_enabled(false);

        let (tx, rx) = std::sync::mpsc::channel();
        let handle = std::thread::spawn(move || {
            set_enabled(true);
            tx.send(is_enabled_requested())
                .expect("send thread-local CUDA enabled state");
            std::thread::sleep(std::time::Duration::from_millis(25));
            tx.send(is_enabled_requested())
                .expect("send thread-local CUDA enabled state");
            set_enabled(false);
        });

        assert!(
            rx.recv().expect("receive spawned thread state"),
            "spawned thread should observe its own CUDA enabled flag"
        );
        assert!(
            !is_enabled_requested(),
            "main thread should not inherit spawned thread's CUDA enabled flag"
        );
        assert!(
            rx.recv().expect("receive spawned thread state"),
            "spawned thread should keep CUDA enabled until it resets"
        );
        assert!(
            !is_enabled_requested(),
            "main thread should still keep its own CUDA disabled flag"
        );

        handle.join().expect("join CUDA enabled thread");
        set_enabled(false);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_shape_helpers_reject_overflow_and_zero_stride() {
        assert!(imp::checked_len("test length", &[usize::MAX, 2]).is_err());
        assert!(imp::conv_output_dim(4, 0, 3, 0, "test conv").is_err());
        assert!(imp::conv_output_dim(4, usize::MAX, 3, 1, "test conv").is_err());
        assert!(!imp::range_fits(usize::MAX, 1, usize::MAX));
        assert!(imp::range_fits(3, 2, 5));
        assert!(imp::ensure_finite("test scale", f32::NAN).is_err());
        assert!(imp::ensure_positive_finite("test epsilon", 0.0).is_err());
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_dynamic_i8_quantize_matches_reference_and_handles_zero_absmax() {
        if !is_available() {
            return;
        }

        for values in [
            (0..4099)
                .map(|i| ((i * 37 % 1009) as f32 - 504.0) / 31.0)
                .collect::<Vec<_>>(),
            vec![0.0; 4099],
            vec![f32::from_bits(1); 4099],
        ] {
            let input = upload_f32(&values).expect("upload dynamic i8 quantize input");
            let (output, scale) =
                quantize_f32_to_i8_dynamic_no_host(&input).expect("dynamic i8 quantize");
            let got = download_i8_storage(&output).expect("download dynamic i8 quantize output");

            let max_abs = values.iter().copied().map(f32::abs).fold(0.0f32, f32::max);
            let expected_scale = if max_abs > 0.0 {
                (max_abs / 127.0).max(f32::MIN_POSITIVE)
            } else {
                1.0
            };
            let expected = values
                .iter()
                .map(|value| (value / expected_scale).round().clamp(-127.0, 127.0) as i8)
                .collect::<Vec<_>>();

            assert!((scale - expected_scale).abs() <= 1e-6);
            assert_eq!(got, expected);
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_i8_typed_binary_zero_absmax_uses_unit_scale() {
        if !is_available() {
            return;
        }

        let len = 4099;
        let lhs = upload_i8_storage(&vec![0; len]).expect("upload zero i8 binary lhs");
        let rhs = upload_i8_storage(&vec![0; len]).expect("upload zero i8 binary rhs");
        let (output, scale) =
            binary_i8_typed_output_buffer_no_host(&lhs, 0.03125, &rhs, 0.046875, BinaryOp::Mul)
                .expect("zero typed-output i8 binary");
        let got = download_i8_storage(&output).expect("download zero typed-output i8 binary");

        assert_eq!(scale, 1.0);
        assert_eq!(got, vec![0; len]);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_i8_typed_binary_overflow_keeps_scale_finite() {
        if !is_available() {
            return;
        }

        let lhs = upload_i8_storage(&[1, -1]).expect("upload overflow i8 binary lhs");
        let rhs = upload_i8_storage(&[1, 1]).expect("upload overflow i8 binary rhs");
        let (output, scale) =
            binary_i8_typed_output_buffer_no_host(&lhs, f32::MAX, &rhs, f32::MAX, BinaryOp::Mul)
                .expect("overflow typed-output i8 binary");
        let got = download_i8_storage(&output).expect("download overflow typed-output i8 binary");

        assert_eq!(scale, 1.0);
        assert_eq!(got, vec![127, -127]);
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "performance sanity test; run with --ignored --nocapture"]
    fn cuda_dynamic_i8_quantize_perf_smoke() {
        if !is_available() {
            return;
        }

        let len = 1 << 20;
        let values = (0..len)
            .map(|i| ((i * 37 % 1009) as f32 - 504.0) / 31.0)
            .collect::<Vec<_>>();
        let input = upload_f32(&values).expect("upload dynamic i8 quantize perf input");

        let mut samples = Vec::with_capacity(9);
        let _ = quantize_f32_to_i8_dynamic_no_host(&input).expect("warm up dynamic i8 quantize");
        synchronize().expect("sync dynamic i8 quantize warmup");
        for _ in 0..9 {
            let start = std::time::Instant::now();
            let _ = quantize_f32_to_i8_dynamic_no_host(&input)
                .expect("dynamic i8 quantize performance sample");
            synchronize().expect("sync dynamic i8 quantize performance sample");
            samples.push(start.elapsed().as_secs_f64() * 1.0e6);
        }
        samples.sort_by(f64::total_cmp);
        println!(
            "cuda dynamic i8 quantize len={len}: median={:.1}us",
            samples[samples.len() / 2]
        );
    }
}

#[cfg(feature = "cuda")]
mod imp {
    use super::{
        BinaryOp, BroadcastMetadata, CudaAdamHostState, CudaBuffer, CudaConv2dBackwardHostBuffers,
        CudaThreeHostBuffers, CudaTwoHostBuffers, UnaryOp,
    };
    use crate::precision::DType;
    use std::ffi::CStr;
    use std::os::raw::{c_char, c_int};

    unsafe extern "C" {
        fn lumen_cuda_is_available() -> c_int;
        fn lumen_cuda_alloc_f32(len: usize, out_handle: *mut u64) -> c_int;
        fn lumen_cuda_upload_f32(handle: u64, src: *const f32, len: usize) -> c_int;
        fn lumen_cuda_upload_u16(handle: u64, src: *const u16, len: usize) -> c_int;
        fn lumen_cuda_upload_i8(handle: u64, src: *const i8, len: usize) -> c_int;
        fn lumen_cuda_upload_f32_offset(
            handle: u64,
            src: *const f32,
            offset: usize,
            len: usize,
        ) -> c_int;
        fn lumen_cuda_copy_f32_offset(
            dst_handle: u64,
            dst_offset: usize,
            src_handle: u64,
            src_offset: usize,
            len: usize,
        ) -> c_int;
        fn lumen_cuda_append_kv_cache_f32_device(
            dst_handle: u64,
            src_handle: u64,
            batch_size: usize,
            num_heads: usize,
            src_seq_len: usize,
            dst_seq_len: usize,
            dim: usize,
            dst_start: usize,
        ) -> c_int;
        fn lumen_cuda_append_kv_cache_pair_f32_device(
            k_dst_handle: u64,
            v_dst_handle: u64,
            k_src_handle: u64,
            v_src_handle: u64,
            batch_size: usize,
            num_heads: usize,
            src_seq_len: usize,
            dst_seq_len: usize,
            dim: usize,
            dst_start: usize,
        ) -> c_int;
        fn lumen_cuda_decode_rope_q_append_kv_f32_device(
            q_src_handle: u64,
            k_src_handle: u64,
            v_src_handle: u64,
            cos_handle: u64,
            sin_handle: u64,
            q_out_handle: u64,
            k_cache_handle: u64,
            v_cache_handle: u64,
            batch_size: usize,
            num_heads: usize,
            num_kv_heads: usize,
            dim: usize,
            dst_seq_len: usize,
            offset: usize,
            cache_seq_len: usize,
        ) -> c_int;
        fn lumen_cuda_kv_cache_prefix_f32_device(
            src_handle: u64,
            out_handle: u64,
            batch_size: usize,
            num_heads: usize,
            active_seq_len: usize,
            src_seq_len: usize,
            dim: usize,
        ) -> c_int;
        fn lumen_cuda_kv_cache_prefix_typed_device(
            src_handle: u64,
            dtype: c_int,
            out_handle: u64,
            batch_size: usize,
            num_heads: usize,
            active_seq_len: usize,
            src_seq_len: usize,
            dim: usize,
        ) -> c_int;
        fn lumen_cuda_download_f32(handle: u64, dst: *mut f32, len: usize) -> c_int;
        fn lumen_cuda_download_u16(handle: u64, dst: *mut u16, len: usize) -> c_int;
        fn lumen_cuda_download_i8(handle: u64, dst: *mut i8, len: usize) -> c_int;
        fn lumen_cuda_download_f32_offset(
            handle: u64,
            dst: *mut f32,
            offset: usize,
            len: usize,
        ) -> c_int;
        fn lumen_cuda_free_f32(handle: u64, len: usize);
        fn lumen_cuda_synchronize() -> c_int;
        fn lumen_cuda_matvec_argmax_f32_device(
            input_handle: u64,
            weight_handle: u64,
            out_indices: *mut usize,
            batch_size: usize,
            vocab_size: usize,
            hidden_size: usize,
        ) -> c_int;
        fn lumen_cuda_matvec_argmax_bf16_i8_device(
            input_handle: u64,
            weight_handle: u64,
            weight_scale: f32,
            out_indices: *mut usize,
            batch_size: usize,
            vocab_size: usize,
            hidden_size: usize,
        ) -> c_int;
        fn lumen_cuda_matvec_argmax_f16_i8_device(
            input_handle: u64,
            weight_handle: u64,
            weight_scale: f32,
            out_indices: *mut usize,
            batch_size: usize,
            vocab_size: usize,
            hidden_size: usize,
        ) -> c_int;
        fn lumen_cuda_matvec_argmax_f32_i8_device(
            input_handle: u64,
            weight_handle: u64,
            weight_scale: f32,
            out_indices: *mut usize,
            batch_size: usize,
            vocab_size: usize,
            hidden_size: usize,
        ) -> c_int;
        fn lumen_cuda_matvec_argmax_i8_i8_device(
            input_handle: u64,
            input_scale: f32,
            weight_handle: u64,
            weight_scale: f32,
            out_indices: *mut usize,
            batch_size: usize,
            vocab_size: usize,
            hidden_size: usize,
        ) -> c_int;
        fn lumen_cuda_matmul_f32_device(
            a_handle: u64,
            b_handle: u64,
            out_handle: u64,
            m: usize,
            n: usize,
            k: usize,
        ) -> c_int;
        fn lumen_cuda_matmul_bf16_host_device(
            a_host: *const u16,
            b_host: *const u16,
            out_handle: u64,
            m: usize,
            n: usize,
            k: usize,
        ) -> c_int;
        fn lumen_cuda_matmul_bf16_device(
            a_handle: u64,
            b_handle: u64,
            out_handle: u64,
            m: usize,
            n: usize,
            k: usize,
        ) -> c_int;
        fn lumen_cuda_matmul_bf16_typed_out_device(
            a_handle: u64,
            b_handle: u64,
            out_handle: u64,
            m: usize,
            n: usize,
            k: usize,
        ) -> c_int;
        fn lumen_cuda_matmul_f16_host_device(
            a_host: *const u16,
            b_host: *const u16,
            out_handle: u64,
            m: usize,
            n: usize,
            k: usize,
        ) -> c_int;
        fn lumen_cuda_matmul_f16_typed_out_device(
            a_handle: u64,
            b_handle: u64,
            out_handle: u64,
            m: usize,
            n: usize,
            k: usize,
        ) -> c_int;
        fn lumen_cuda_matmul_f16_device(
            a_handle: u64,
            b_handle: u64,
            out_handle: u64,
            m: usize,
            n: usize,
            k: usize,
        ) -> c_int;
        fn lumen_cuda_matmul_i8_host_device(
            a_host: *const i8,
            b_host: *const i8,
            a_scale: f32,
            b_scale: f32,
            out_handle: u64,
            m: usize,
            n: usize,
            k: usize,
        ) -> c_int;
        fn lumen_cuda_matmul_i8_device(
            a_handle: u64,
            a_scale: f32,
            b_handle: u64,
            b_scale: f32,
            out_handle: u64,
            m: usize,
            n: usize,
            k: usize,
        ) -> c_int;
        fn lumen_cuda_matmul_bf16_i8_device(
            a_handle: u64,
            b_handle: u64,
            b_scale: f32,
            out_handle: u64,
            m: usize,
            n: usize,
            k: usize,
        ) -> c_int;
        fn lumen_cuda_matmul_f16_i8_device(
            a_handle: u64,
            b_handle: u64,
            b_scale: f32,
            out_handle: u64,
            m: usize,
            n: usize,
            k: usize,
        ) -> c_int;
        fn lumen_cuda_matmul_f32_i8_device(
            a_handle: u64,
            b_handle: u64,
            b_scale: f32,
            out_handle: u64,
            m: usize,
            n: usize,
            k: usize,
        ) -> c_int;
        fn lumen_cuda_matmul_i8_bf16_device(
            a_handle: u64,
            a_scale: f32,
            b_handle: u64,
            out_handle: u64,
            m: usize,
            n: usize,
            k: usize,
        ) -> c_int;
        fn lumen_cuda_matmul_i8_f16_device(
            a_handle: u64,
            a_scale: f32,
            b_handle: u64,
            out_handle: u64,
            m: usize,
            n: usize,
            k: usize,
        ) -> c_int;
        fn lumen_cuda_matmul_i8_f32_device(
            a_handle: u64,
            a_scale: f32,
            b_handle: u64,
            out_handle: u64,
            m: usize,
            n: usize,
            k: usize,
        ) -> c_int;
        fn lumen_cuda_matmul_i8_typed_out_device(
            a_handle: u64,
            a_scale: f32,
            b_handle: u64,
            b_scale: f32,
            out_handle: u64,
            m: usize,
            n: usize,
            k: usize,
            out_scale: *mut f32,
        ) -> c_int;
        fn lumen_cuda_batch_matmul_f32_device(
            lhs_handle: u64,
            rhs_handle: u64,
            out_handle: u64,
            batch_count: usize,
            m: usize,
            n: usize,
            k: usize,
        ) -> c_int;
        fn lumen_cuda_batch_matmul_bf16_device(
            lhs_handle: u64,
            rhs_handle: u64,
            out_handle: u64,
            batch_count: usize,
            m: usize,
            n: usize,
            k: usize,
        ) -> c_int;
        fn lumen_cuda_batch_matmul_bf16_typed_out_device(
            lhs_handle: u64,
            rhs_handle: u64,
            out_handle: u64,
            batch_count: usize,
            m: usize,
            n: usize,
            k: usize,
        ) -> c_int;
        fn lumen_cuda_batch_matmul_f16_device(
            lhs_handle: u64,
            rhs_handle: u64,
            out_handle: u64,
            batch_count: usize,
            m: usize,
            n: usize,
            k: usize,
        ) -> c_int;
        fn lumen_cuda_batch_matmul_f16_typed_out_device(
            lhs_handle: u64,
            rhs_handle: u64,
            out_handle: u64,
            batch_count: usize,
            m: usize,
            n: usize,
            k: usize,
        ) -> c_int;
        fn lumen_cuda_batch_matmul_bf16_i8_device(
            lhs_handle: u64,
            rhs_handle: u64,
            rhs_scale: f32,
            out_handle: u64,
            batch_count: usize,
            m: usize,
            n: usize,
            k: usize,
        ) -> c_int;
        fn lumen_cuda_batch_matmul_f16_i8_device(
            lhs_handle: u64,
            rhs_handle: u64,
            rhs_scale: f32,
            out_handle: u64,
            batch_count: usize,
            m: usize,
            n: usize,
            k: usize,
        ) -> c_int;
        fn lumen_cuda_batch_matmul_f32_i8_device(
            lhs_handle: u64,
            rhs_handle: u64,
            rhs_scale: f32,
            out_handle: u64,
            batch_count: usize,
            m: usize,
            n: usize,
            k: usize,
        ) -> c_int;
        fn lumen_cuda_batch_matmul_i8_bf16_device(
            lhs_handle: u64,
            lhs_scale: f32,
            rhs_handle: u64,
            out_handle: u64,
            batch_count: usize,
            m: usize,
            n: usize,
            k: usize,
        ) -> c_int;
        fn lumen_cuda_batch_matmul_i8_f16_device(
            lhs_handle: u64,
            lhs_scale: f32,
            rhs_handle: u64,
            out_handle: u64,
            batch_count: usize,
            m: usize,
            n: usize,
            k: usize,
        ) -> c_int;
        fn lumen_cuda_batch_matmul_i8_f32_device(
            lhs_handle: u64,
            lhs_scale: f32,
            rhs_handle: u64,
            out_handle: u64,
            batch_count: usize,
            m: usize,
            n: usize,
            k: usize,
        ) -> c_int;
        fn lumen_cuda_batch_matmul_i8_device(
            lhs_handle: u64,
            lhs_scale: f32,
            rhs_handle: u64,
            rhs_scale: f32,
            out_handle: u64,
            batch_count: usize,
            m: usize,
            n: usize,
            k: usize,
        ) -> c_int;
        fn lumen_cuda_batch_matmul_i8_typed_out_device(
            lhs_handle: u64,
            lhs_scale: f32,
            rhs_handle: u64,
            rhs_scale: f32,
            out_handle: u64,
            batch_count: usize,
            m: usize,
            n: usize,
            k: usize,
            out_scale: *mut f32,
        ) -> c_int;
        fn lumen_cuda_matmul_backward_f32_device(
            grad_handle: u64,
            a_handle: u64,
            b_handle: u64,
            da_handle: u64,
            db_handle: u64,
            m: usize,
            k: usize,
            n: usize,
        ) -> c_int;
        fn lumen_cuda_matmul_backward_bf16_i8_device(
            grad_handle: u64,
            lhs_handle: u64,
            rhs_handle: u64,
            rhs_scale: f32,
            d_lhs_handle: u64,
            d_rhs_handle: u64,
            m: usize,
            k: usize,
            n: usize,
        ) -> c_int;
        fn lumen_cuda_matmul_backward_f16_i8_device(
            grad_handle: u64,
            lhs_handle: u64,
            rhs_handle: u64,
            rhs_scale: f32,
            d_lhs_handle: u64,
            d_rhs_handle: u64,
            m: usize,
            k: usize,
            n: usize,
        ) -> c_int;
        fn lumen_cuda_matmul_backward_f32_i8_device(
            grad_handle: u64,
            lhs_handle: u64,
            rhs_handle: u64,
            rhs_scale: f32,
            d_lhs_handle: u64,
            d_rhs_handle: u64,
            m: usize,
            k: usize,
            n: usize,
        ) -> c_int;
        fn lumen_cuda_matmul_backward_i8_bf16_device(
            grad_handle: u64,
            lhs_handle: u64,
            lhs_scale: f32,
            rhs_handle: u64,
            d_lhs_handle: u64,
            d_rhs_handle: u64,
            m: usize,
            k: usize,
            n: usize,
        ) -> c_int;
        fn lumen_cuda_matmul_backward_i8_f16_device(
            grad_handle: u64,
            lhs_handle: u64,
            lhs_scale: f32,
            rhs_handle: u64,
            d_lhs_handle: u64,
            d_rhs_handle: u64,
            m: usize,
            k: usize,
            n: usize,
        ) -> c_int;
        fn lumen_cuda_matmul_backward_i8_f32_device(
            grad_handle: u64,
            lhs_handle: u64,
            lhs_scale: f32,
            rhs_handle: u64,
            d_lhs_handle: u64,
            d_rhs_handle: u64,
            m: usize,
            k: usize,
            n: usize,
        ) -> c_int;
        fn lumen_cuda_batch_matmul_backward_f32_device(
            grad_handle: u64,
            lhs_handle: u64,
            rhs_handle: u64,
            d_lhs_handle: u64,
            d_rhs_handle: u64,
            batch_count: usize,
            m: usize,
            k: usize,
            n: usize,
        ) -> c_int;
        fn lumen_cuda_batch_matmul_backward_bf16_device(
            grad_handle: u64,
            lhs_handle: u64,
            rhs_handle: u64,
            d_lhs_handle: u64,
            d_rhs_handle: u64,
            batch_count: usize,
            m: usize,
            k: usize,
            n: usize,
        ) -> c_int;
        fn lumen_cuda_batch_matmul_backward_f16_device(
            grad_handle: u64,
            lhs_handle: u64,
            rhs_handle: u64,
            d_lhs_handle: u64,
            d_rhs_handle: u64,
            batch_count: usize,
            m: usize,
            k: usize,
            n: usize,
        ) -> c_int;
        fn lumen_cuda_batch_matmul_backward_bf16_i8_device(
            grad_handle: u64,
            lhs_handle: u64,
            rhs_handle: u64,
            rhs_scale: f32,
            d_lhs_handle: u64,
            d_rhs_handle: u64,
            batch_count: usize,
            m: usize,
            k: usize,
            n: usize,
        ) -> c_int;
        fn lumen_cuda_batch_matmul_backward_f16_i8_device(
            grad_handle: u64,
            lhs_handle: u64,
            rhs_handle: u64,
            rhs_scale: f32,
            d_lhs_handle: u64,
            d_rhs_handle: u64,
            batch_count: usize,
            m: usize,
            k: usize,
            n: usize,
        ) -> c_int;
        fn lumen_cuda_batch_matmul_backward_f32_i8_device(
            grad_handle: u64,
            lhs_handle: u64,
            rhs_handle: u64,
            rhs_scale: f32,
            d_lhs_handle: u64,
            d_rhs_handle: u64,
            batch_count: usize,
            m: usize,
            k: usize,
            n: usize,
        ) -> c_int;
        fn lumen_cuda_batch_matmul_backward_i8_bf16_device(
            grad_handle: u64,
            lhs_handle: u64,
            lhs_scale: f32,
            rhs_handle: u64,
            d_lhs_handle: u64,
            d_rhs_handle: u64,
            batch_count: usize,
            m: usize,
            k: usize,
            n: usize,
        ) -> c_int;
        fn lumen_cuda_batch_matmul_backward_i8_f16_device(
            grad_handle: u64,
            lhs_handle: u64,
            lhs_scale: f32,
            rhs_handle: u64,
            d_lhs_handle: u64,
            d_rhs_handle: u64,
            batch_count: usize,
            m: usize,
            k: usize,
            n: usize,
        ) -> c_int;
        fn lumen_cuda_batch_matmul_backward_i8_f32_device(
            grad_handle: u64,
            lhs_handle: u64,
            lhs_scale: f32,
            rhs_handle: u64,
            d_lhs_handle: u64,
            d_rhs_handle: u64,
            batch_count: usize,
            m: usize,
            k: usize,
            n: usize,
        ) -> c_int;
        fn lumen_cuda_batch_matmul_backward_i8_device(
            grad_handle: u64,
            lhs_handle: u64,
            lhs_scale: f32,
            rhs_handle: u64,
            rhs_scale: f32,
            d_lhs_handle: u64,
            d_rhs_handle: u64,
            batch_count: usize,
            m: usize,
            k: usize,
            n: usize,
        ) -> c_int;
        fn lumen_cuda_unary_f32_device(
            input_handle: u64,
            out_handle: u64,
            len: usize,
            op: c_int,
        ) -> c_int;
        fn lumen_cuda_unary_f16_device(
            input_handle: u64,
            out_handle: u64,
            len: usize,
            op: c_int,
        ) -> c_int;
        fn lumen_cuda_unary_f16_typed_out_device(
            input_handle: u64,
            out_handle: u64,
            len: usize,
            op: c_int,
        ) -> c_int;
        fn lumen_cuda_unary_bf16_device(
            input_handle: u64,
            out_handle: u64,
            len: usize,
            op: c_int,
        ) -> c_int;
        fn lumen_cuda_unary_bf16_typed_out_device(
            input_handle: u64,
            out_handle: u64,
            len: usize,
            op: c_int,
        ) -> c_int;
        fn lumen_cuda_unary_i8_device(
            input_handle: u64,
            scale: f32,
            out_handle: u64,
            len: usize,
            op: c_int,
        ) -> c_int;
        fn lumen_cuda_unary_i8_relu_typed_out_device(
            input_handle: u64,
            out_handle: u64,
            len: usize,
        ) -> c_int;
        fn lumen_cuda_unary_backward_f32_device(
            input_handle: u64,
            output_handle: u64,
            grad_handle: u64,
            out_handle: u64,
            len: usize,
            op: c_int,
        ) -> c_int;
        fn lumen_cuda_unary_backward_f16_device(
            input_handle: u64,
            output_handle: u64,
            grad_handle: u64,
            out_handle: u64,
            len: usize,
            op: c_int,
        ) -> c_int;
        fn lumen_cuda_unary_backward_bf16_device(
            input_handle: u64,
            output_handle: u64,
            grad_handle: u64,
            out_handle: u64,
            len: usize,
            op: c_int,
        ) -> c_int;
        fn lumen_cuda_unary_backward_i8_device(
            input_handle: u64,
            scale: f32,
            output_handle: u64,
            grad_handle: u64,
            out_handle: u64,
            len: usize,
            op: c_int,
        ) -> c_int;
        fn lumen_cuda_binary_f32_device(
            lhs_handle: u64,
            rhs_handle: u64,
            out_handle: u64,
            len: usize,
            op: c_int,
        ) -> c_int;
        fn lumen_cuda_binary_typed_device(
            lhs_handle: u64,
            lhs_dtype: c_int,
            lhs_scale: f32,
            rhs_handle: u64,
            rhs_dtype: c_int,
            rhs_scale: f32,
            out_handle: u64,
            len: usize,
            op: c_int,
        ) -> c_int;
        fn lumen_cuda_binary_lowp_typed_out_device(
            lhs_handle: u64,
            rhs_handle: u64,
            out_handle: u64,
            len: usize,
            dtype: c_int,
            op: c_int,
        ) -> c_int;
        fn lumen_cuda_binary_typed_lastdim_device(
            lhs_handle: u64,
            lhs_dtype: c_int,
            lhs_scale: f32,
            rhs_handle: u64,
            rhs_dtype: c_int,
            rhs_scale: f32,
            out_handle: u64,
            len: usize,
            last_dim: usize,
            vector_on_rhs: c_int,
            op: c_int,
        ) -> c_int;
        fn lumen_cuda_binary_lowp_typed_lastdim_out_device(
            lhs_handle: u64,
            rhs_handle: u64,
            out_handle: u64,
            len: usize,
            last_dim: usize,
            vector_on_rhs: c_int,
            dtype: c_int,
            op: c_int,
        ) -> c_int;
        fn lumen_cuda_binary_typed_row_scalar_device(
            lhs_handle: u64,
            lhs_dtype: c_int,
            lhs_scale: f32,
            rhs_handle: u64,
            rhs_dtype: c_int,
            rhs_scale: f32,
            out_handle: u64,
            rows: usize,
            last_dim: usize,
            scalar_on_rhs: c_int,
            op: c_int,
        ) -> c_int;
        fn lumen_cuda_binary_lowp_typed_row_scalar_out_device(
            lhs_handle: u64,
            rhs_handle: u64,
            out_handle: u64,
            rows: usize,
            last_dim: usize,
            scalar_on_rhs: c_int,
            dtype: c_int,
            op: c_int,
        ) -> c_int;
        fn lumen_cuda_binary_i8_typed_row_scalar_out_device(
            lhs_handle: u64,
            rhs_handle: u64,
            lhs_scale: f32,
            rhs_scale: f32,
            out_handle: u64,
            rows: usize,
            last_dim: usize,
            scalar_on_rhs: c_int,
            op: c_int,
            out_scale: *mut f32,
        ) -> c_int;
        fn lumen_cuda_binary_typed_broadcast_device(
            lhs_handle: u64,
            lhs_dtype: c_int,
            lhs_scale: f32,
            rhs_handle: u64,
            rhs_dtype: c_int,
            rhs_scale: f32,
            out_handle: u64,
            ndim: usize,
            out_strides: *const usize,
            lhs_shape: *const usize,
            lhs_strides: *const usize,
            rhs_shape: *const usize,
            rhs_strides: *const usize,
            len: usize,
            op: c_int,
        ) -> c_int;
        fn lumen_cuda_binary_lowp_typed_broadcast_out_device(
            lhs_handle: u64,
            rhs_handle: u64,
            out_handle: u64,
            ndim: usize,
            out_strides: *const usize,
            lhs_shape: *const usize,
            lhs_strides: *const usize,
            rhs_shape: *const usize,
            rhs_strides: *const usize,
            len: usize,
            dtype: c_int,
            op: c_int,
        ) -> c_int;
        fn lumen_cuda_binary_i8_typed_broadcast_out_device(
            lhs_handle: u64,
            rhs_handle: u64,
            lhs_scale: f32,
            rhs_scale: f32,
            out_handle: u64,
            ndim: usize,
            out_strides: *const usize,
            lhs_shape: *const usize,
            lhs_strides: *const usize,
            rhs_shape: *const usize,
            rhs_strides: *const usize,
            len: usize,
            op: c_int,
            out_scale: *mut f32,
        ) -> c_int;
        fn lumen_cuda_binary_typed_b1d_1h1_device(
            lhs_handle: u64,
            lhs_dtype: c_int,
            lhs_scale: f32,
            rhs_handle: u64,
            rhs_dtype: c_int,
            rhs_scale: f32,
            out_handle: u64,
            batch: usize,
            heads: usize,
            dim: usize,
            b1d_on_lhs: c_int,
            op: c_int,
        ) -> c_int;
        fn lumen_cuda_binary_lowp_typed_b1d_1h1_out_device(
            lhs_handle: u64,
            rhs_handle: u64,
            out_handle: u64,
            batch: usize,
            heads: usize,
            dim: usize,
            b1d_on_lhs: c_int,
            dtype: c_int,
            op: c_int,
        ) -> c_int;
        fn lumen_cuda_binary_i8_typed_b1d_1h1_out_device(
            lhs_handle: u64,
            rhs_handle: u64,
            lhs_scale: f32,
            rhs_scale: f32,
            out_handle: u64,
            batch: usize,
            heads: usize,
            dim: usize,
            b1d_on_lhs: c_int,
            op: c_int,
            out_scale: *mut f32,
        ) -> c_int;
        fn lumen_cuda_binary_typed_b1d_1hd_device(
            lhs_handle: u64,
            lhs_dtype: c_int,
            lhs_scale: f32,
            rhs_handle: u64,
            rhs_dtype: c_int,
            rhs_scale: f32,
            out_handle: u64,
            batch: usize,
            heads: usize,
            dim: usize,
            b1d_on_lhs: c_int,
            op: c_int,
        ) -> c_int;
        fn lumen_cuda_binary_lowp_typed_b1d_1hd_out_device(
            lhs_handle: u64,
            rhs_handle: u64,
            out_handle: u64,
            batch: usize,
            heads: usize,
            dim: usize,
            b1d_on_lhs: c_int,
            dtype: c_int,
            op: c_int,
        ) -> c_int;
        fn lumen_cuda_binary_i8_typed_b1d_1hd_out_device(
            lhs_handle: u64,
            rhs_handle: u64,
            lhs_scale: f32,
            rhs_scale: f32,
            out_handle: u64,
            batch: usize,
            heads: usize,
            dim: usize,
            b1d_on_lhs: c_int,
            op: c_int,
            out_scale: *mut f32,
        ) -> c_int;
        fn lumen_cuda_binary_f16_host_device(
            lhs_host: *const u16,
            rhs_host: *const u16,
            out_handle: u64,
            len: usize,
            op: c_int,
        ) -> c_int;
        fn lumen_cuda_binary_f16_device(
            lhs_handle: u64,
            rhs_handle: u64,
            out_handle: u64,
            len: usize,
            op: c_int,
        ) -> c_int;
        fn lumen_cuda_binary_f16_lastdim_device(
            lhs_handle: u64,
            rhs_handle: u64,
            out_handle: u64,
            len: usize,
            last_dim: usize,
            vector_on_rhs: c_int,
            op: c_int,
        ) -> c_int;
        fn lumen_cuda_binary_bf16_host_device(
            lhs_host: *const u16,
            rhs_host: *const u16,
            out_handle: u64,
            len: usize,
            op: c_int,
        ) -> c_int;
        fn lumen_cuda_binary_bf16_device(
            lhs_handle: u64,
            rhs_handle: u64,
            out_handle: u64,
            len: usize,
            op: c_int,
        ) -> c_int;
        fn lumen_cuda_binary_bf16_lastdim_device(
            lhs_handle: u64,
            rhs_handle: u64,
            out_handle: u64,
            len: usize,
            last_dim: usize,
            vector_on_rhs: c_int,
            op: c_int,
        ) -> c_int;
        fn lumen_cuda_binary_i8_host_device(
            lhs_host: *const i8,
            rhs_host: *const i8,
            lhs_scale: f32,
            rhs_scale: f32,
            out_handle: u64,
            len: usize,
            op: c_int,
        ) -> c_int;
        fn lumen_cuda_binary_i8_device(
            lhs_handle: u64,
            rhs_handle: u64,
            lhs_scale: f32,
            rhs_scale: f32,
            out_handle: u64,
            len: usize,
            op: c_int,
        ) -> c_int;
        fn lumen_cuda_binary_i8_typed_out_device(
            lhs_handle: u64,
            rhs_handle: u64,
            lhs_scale: f32,
            rhs_scale: f32,
            out_handle: u64,
            len: usize,
            op: c_int,
            out_scale: *mut f32,
        ) -> c_int;
        fn lumen_cuda_binary_i8_lastdim_device(
            lhs_handle: u64,
            rhs_handle: u64,
            lhs_scale: f32,
            rhs_scale: f32,
            out_handle: u64,
            len: usize,
            last_dim: usize,
            vector_on_rhs: c_int,
            op: c_int,
        ) -> c_int;
        fn lumen_cuda_binary_i8_typed_lastdim_out_device(
            lhs_handle: u64,
            rhs_handle: u64,
            lhs_scale: f32,
            rhs_scale: f32,
            out_handle: u64,
            len: usize,
            last_dim: usize,
            vector_on_rhs: c_int,
            op: c_int,
            out_scale: *mut f32,
        ) -> c_int;
        fn lumen_cuda_mul_grad_f16_host_device(
            grad_handle: u64,
            operand_host: *const u16,
            out_handle: u64,
            len: usize,
        ) -> c_int;
        fn lumen_cuda_mul_grad_f16_device(
            grad_handle: u64,
            operand_handle: u64,
            out_handle: u64,
            len: usize,
        ) -> c_int;
        fn lumen_cuda_mul_grad_f16_lastdim_device(
            grad_handle: u64,
            lhs_handle: u64,
            rhs_handle: u64,
            grad_lhs_handle: u64,
            grad_rhs_handle: u64,
            len: usize,
            last_dim: usize,
            vector_on_rhs: c_int,
        ) -> c_int;
        fn lumen_cuda_mul_grad_bf16_host_device(
            grad_handle: u64,
            operand_host: *const u16,
            out_handle: u64,
            len: usize,
        ) -> c_int;
        fn lumen_cuda_mul_grad_bf16_device(
            grad_handle: u64,
            operand_handle: u64,
            out_handle: u64,
            len: usize,
        ) -> c_int;
        fn lumen_cuda_mul_grad_bf16_lastdim_device(
            grad_handle: u64,
            lhs_handle: u64,
            rhs_handle: u64,
            grad_lhs_handle: u64,
            grad_rhs_handle: u64,
            len: usize,
            last_dim: usize,
            vector_on_rhs: c_int,
        ) -> c_int;
        fn lumen_cuda_mul_grad_i8_host_device(
            grad_handle: u64,
            operand_host: *const i8,
            scale: f32,
            out_handle: u64,
            len: usize,
        ) -> c_int;
        fn lumen_cuda_mul_grad_i8_device(
            grad_handle: u64,
            operand_handle: u64,
            scale: f32,
            out_handle: u64,
            len: usize,
        ) -> c_int;
        fn lumen_cuda_mul_grad_i8_lastdim_device(
            grad_handle: u64,
            lhs_handle: u64,
            rhs_handle: u64,
            lhs_scale: f32,
            rhs_scale: f32,
            grad_lhs_handle: u64,
            grad_rhs_handle: u64,
            len: usize,
            last_dim: usize,
            vector_on_rhs: c_int,
        ) -> c_int;
        fn lumen_cuda_mul_grad_typed_device(
            grad_handle: u64,
            lhs_handle: u64,
            lhs_dtype: c_int,
            lhs_scale: f32,
            rhs_handle: u64,
            rhs_dtype: c_int,
            rhs_scale: f32,
            grad_lhs_handle: u64,
            grad_rhs_handle: u64,
            len: usize,
        ) -> c_int;
        fn lumen_cuda_mul_grad_typed_lastdim_device(
            grad_handle: u64,
            lhs_handle: u64,
            lhs_dtype: c_int,
            lhs_scale: f32,
            rhs_handle: u64,
            rhs_dtype: c_int,
            rhs_scale: f32,
            grad_lhs_handle: u64,
            grad_rhs_handle: u64,
            len: usize,
            last_dim: usize,
            vector_on_rhs: c_int,
        ) -> c_int;
        fn lumen_cuda_mul_grad_typed_row_scalar_device(
            grad_handle: u64,
            lhs_handle: u64,
            lhs_dtype: c_int,
            lhs_scale: f32,
            rhs_handle: u64,
            rhs_dtype: c_int,
            rhs_scale: f32,
            grad_lhs_handle: u64,
            grad_rhs_handle: u64,
            rows: usize,
            last_dim: usize,
            scalar_on_rhs: c_int,
        ) -> c_int;
        fn lumen_cuda_mul_grad_typed_broadcast_device(
            grad_handle: u64,
            lhs_handle: u64,
            lhs_dtype: c_int,
            lhs_scale: f32,
            rhs_handle: u64,
            rhs_dtype: c_int,
            rhs_scale: f32,
            grad_lhs_handle: u64,
            grad_rhs_handle: u64,
            ndim: usize,
            out_strides: *const usize,
            lhs_shape: *const usize,
            lhs_strides: *const usize,
            rhs_shape: *const usize,
            rhs_strides: *const usize,
            out_len: usize,
            lhs_len: usize,
            rhs_len: usize,
        ) -> c_int;
        fn lumen_cuda_mul_grad_typed_b1d_1h1_device(
            grad_handle: u64,
            lhs_handle: u64,
            lhs_dtype: c_int,
            lhs_scale: f32,
            rhs_handle: u64,
            rhs_dtype: c_int,
            rhs_scale: f32,
            grad_lhs_handle: u64,
            grad_rhs_handle: u64,
            batch: usize,
            heads: usize,
            dim: usize,
            b1d_on_lhs: c_int,
        ) -> c_int;
        fn lumen_cuda_mul_grad_typed_b1d_1hd_device(
            grad_handle: u64,
            lhs_handle: u64,
            lhs_dtype: c_int,
            lhs_scale: f32,
            rhs_handle: u64,
            rhs_dtype: c_int,
            rhs_scale: f32,
            grad_lhs_handle: u64,
            grad_rhs_handle: u64,
            batch: usize,
            heads: usize,
            dim: usize,
            b1d_on_lhs: c_int,
        ) -> c_int;
        fn lumen_cuda_mul_grad_typed_scalar_device(
            grad_handle: u64,
            lhs_handle: u64,
            lhs_dtype: c_int,
            lhs_scale: f32,
            rhs_handle: u64,
            rhs_dtype: c_int,
            rhs_scale: f32,
            grad_lhs_handle: u64,
            grad_rhs_handle: u64,
            len: usize,
            scalar_on_rhs: c_int,
        ) -> c_int;
        fn lumen_cuda_binary_backward_f32_device(
            lhs_handle: u64,
            rhs_handle: u64,
            grad_handle: u64,
            grad_lhs_handle: u64,
            grad_rhs_handle: u64,
            len: usize,
            op: c_int,
        ) -> c_int;
        fn lumen_cuda_add_sub_backward_f32_device(
            grad_handle: u64,
            grad_lhs_handle: u64,
            grad_rhs_handle: u64,
            len: usize,
            op: c_int,
        ) -> c_int;
        fn lumen_cuda_add_sub_backward_lastdim_f32_device(
            grad_handle: u64,
            grad_lhs_handle: u64,
            grad_rhs_handle: u64,
            len: usize,
            last_dim: usize,
            vector_on_rhs: c_int,
            op: c_int,
        ) -> c_int;
        fn lumen_cuda_add_sub_backward_scalar_f32_device(
            grad_handle: u64,
            grad_lhs_handle: u64,
            grad_rhs_handle: u64,
            len: usize,
            scalar_on_rhs: c_int,
            op: c_int,
        ) -> c_int;
        fn lumen_cuda_add_sub_backward_row_scalar_f32_device(
            grad_handle: u64,
            grad_lhs_handle: u64,
            grad_rhs_handle: u64,
            rows: usize,
            last_dim: usize,
            scalar_on_rhs: c_int,
            op: c_int,
        ) -> c_int;
        fn lumen_cuda_add_sub_backward_b1d_1h1_f32_device(
            grad_handle: u64,
            grad_lhs_handle: u64,
            grad_rhs_handle: u64,
            batch: usize,
            heads: usize,
            dim: usize,
            b1d_on_lhs: c_int,
            op: c_int,
        ) -> c_int;
        fn lumen_cuda_add_sub_backward_b1d_1hd_f32_device(
            grad_handle: u64,
            grad_lhs_handle: u64,
            grad_rhs_handle: u64,
            batch: usize,
            heads: usize,
            dim: usize,
            b1d_on_lhs: c_int,
            op: c_int,
        ) -> c_int;
        fn lumen_cuda_add_sub_broadcast_backward_f32_device(
            grad_handle: u64,
            grad_lhs_handle: u64,
            grad_rhs_handle: u64,
            ndim: usize,
            out_strides: *const usize,
            lhs_shape: *const usize,
            lhs_strides: *const usize,
            rhs_shape: *const usize,
            rhs_strides: *const usize,
            out_len: usize,
            lhs_len: usize,
            rhs_len: usize,
            op: c_int,
        ) -> c_int;
        fn lumen_cuda_binary_broadcast_f32_device(
            lhs_handle: u64,
            rhs_handle: u64,
            out_handle: u64,
            ndim: usize,
            out_shape: *const usize,
            out_strides: *const usize,
            lhs_shape: *const usize,
            lhs_strides: *const usize,
            rhs_shape: *const usize,
            rhs_strides: *const usize,
            len: usize,
            op: c_int,
        ) -> c_int;
        fn lumen_cuda_binary_broadcast_backward_f32_device(
            lhs_handle: u64,
            rhs_handle: u64,
            grad_handle: u64,
            grad_lhs_handle: u64,
            grad_rhs_handle: u64,
            ndim: usize,
            out_shape: *const usize,
            out_strides: *const usize,
            lhs_shape: *const usize,
            lhs_strides: *const usize,
            rhs_shape: *const usize,
            rhs_strides: *const usize,
            out_len: usize,
            lhs_len: usize,
            rhs_len: usize,
            op: c_int,
        ) -> c_int;
        fn lumen_cuda_sum_f32_device(input_handle: u64, out_handle: u64, len: usize) -> c_int;
        fn lumen_cuda_sum_f16_device(input_handle: u64, out_handle: u64, len: usize) -> c_int;
        fn lumen_cuda_sum_bf16_device(input_handle: u64, out_handle: u64, len: usize) -> c_int;
        fn lumen_cuda_sum_i8_device(
            input_handle: u64,
            scale: f32,
            out_handle: u64,
            len: usize,
        ) -> c_int;
        fn lumen_cuda_fill_scalar_f32_device(out_handle: u64, len: usize, value: f32) -> c_int;
        fn lumen_cuda_add_inplace_f32_device(dst_handle: u64, src_handle: u64, len: usize)
        -> c_int;
        fn lumen_cuda_sum_lastdim_f32_device(
            input_handle: u64,
            out_handle: u64,
            rows: usize,
            last_dim: usize,
        ) -> c_int;
        fn lumen_cuda_bshd_to_bhsd_add_bias_f32_device(
            input_handle: u64,
            bias_handle: u64,
            out_handle: u64,
            batch: usize,
            seq: usize,
            heads: usize,
            dim: usize,
        ) -> c_int;
        fn lumen_cuda_mse_forward_typed_device(
            output_handle: u64,
            output_dtype: c_int,
            output_scale: f32,
            target_handle: u64,
            target_dtype: c_int,
            target_scale: f32,
            diff_handle: u64,
            loss_handle: u64,
            len: usize,
        ) -> c_int;
        fn lumen_cuda_mse_backward_f32_device(
            diff_handle: u64,
            grad_output_handle: u64,
            grad_target_handle: u64,
            len: usize,
            factor: f32,
        ) -> c_int;
        fn lumen_cuda_cross_entropy_backward_f32_device(
            softmax_handle: u64,
            target_handle: u64,
            out_handle: u64,
            len: usize,
            factor: f32,
        ) -> c_int;
        fn lumen_cuda_cross_entropy_loss_f32_device(
            softmax_handle: u64,
            target_handle: u64,
            out_handle: u64,
            len: usize,
            factor: f32,
        ) -> c_int;
        fn lumen_cuda_cross_entropy_backward_typed_target_device(
            softmax_handle: u64,
            target_handle: u64,
            target_dtype: c_int,
            target_scale: f32,
            out_handle: u64,
            len: usize,
            factor: f32,
        ) -> c_int;
        fn lumen_cuda_cross_entropy_loss_typed_target_device(
            softmax_handle: u64,
            target_handle: u64,
            target_dtype: c_int,
            target_scale: f32,
            out_handle: u64,
            len: usize,
            factor: f32,
        ) -> c_int;
        fn lumen_cuda_sgd_update_f32_device(
            param_handle: u64,
            grad_handle: u64,
            len: usize,
            lr: f32,
        ) -> c_int;
        fn lumen_cuda_sgd_update_f32_batched_device(
            param_handles: *const u64,
            grad_handles: *const u64,
            lens: *const usize,
            count: usize,
            lr: f32,
        ) -> c_int;
        fn lumen_cuda_quantize_f32_storage_device(
            param_handle: u64,
            len: usize,
            dtype: c_int,
            scale: f32,
        ) -> c_int;
        fn lumen_cuda_quantize_f32_to_i8_dynamic_device(
            input_handle: u64,
            out_handle: u64,
            len: usize,
            out_scale: *mut f32,
        ) -> c_int;
        fn lumen_cuda_f32_to_lowp_storage_device(
            input_handle: u64,
            out_handle: u64,
            len: usize,
            dtype: c_int,
        ) -> c_int;
        fn lumen_cuda_sgd_momentum_update_f32_device(
            param_handle: u64,
            grad_handle: u64,
            velocity_handle: u64,
            len: usize,
            lr: f32,
            momentum: f32,
        ) -> c_int;
        fn lumen_cuda_sgd_momentum_update_f32_batched_device(
            param_handles: *const u64,
            grad_handles: *const u64,
            velocity_handles: *const u64,
            lens: *const usize,
            count: usize,
            lr: f32,
            momentum: f32,
        ) -> c_int;
        fn lumen_cuda_adam_update_f32_device(
            param_handle: u64,
            grad_handle: u64,
            exp_avg_handle: u64,
            exp_avg_sq_handle: u64,
            len: usize,
            lr: f32,
            beta1: f32,
            beta2: f32,
            bias_correction1: f32,
            bias_correction2: f32,
            eps: f32,
        ) -> c_int;
        fn lumen_cuda_adam_update_f32_batched_device(
            param_handles: *const u64,
            grad_handles: *const u64,
            exp_avg_handles: *const u64,
            exp_avg_sq_handles: *const u64,
            lens: *const usize,
            count: usize,
            lr: f32,
            beta1: f32,
            beta2: f32,
            bias_correction1: f32,
            bias_correction2: f32,
            eps: f32,
        ) -> c_int;
        fn lumen_cuda_softmax_lastdim_f32_device(
            input_handle: u64,
            out_handle: u64,
            outer: usize,
            last_dim: usize,
        ) -> c_int;
        fn lumen_cuda_softmax_lastdim_typed_device(
            input_handle: u64,
            input_dtype: c_int,
            input_scale: f32,
            out_handle: u64,
            outer: usize,
            last_dim: usize,
        ) -> c_int;
        fn lumen_cuda_softmax_lastdim_backward_f32_device(
            output_handle: u64,
            grad_handle: u64,
            out_handle: u64,
            outer: usize,
            last_dim: usize,
        ) -> c_int;
        fn lumen_cuda_fused_softmax_f32_device(
            input_handle: u64,
            out_handle: u64,
            batch_heads: usize,
            q_len: usize,
            k_len: usize,
            scale: f32,
            is_causal: c_int,
        ) -> c_int;
        fn lumen_cuda_fused_softmax_backward_f32_device(
            output_handle: u64,
            grad_handle: u64,
            out_handle: u64,
            batch_heads: usize,
            q_len: usize,
            k_len: usize,
            scale: f32,
        ) -> c_int;
        fn lumen_cuda_fused_softmax_f32_with_past_device(
            input_handle: u64,
            out_handle: u64,
            batch_heads: usize,
            q_len: usize,
            k_len: usize,
            scale: f32,
            is_causal: c_int,
            past_len: usize,
        ) -> c_int;
        fn lumen_cuda_embedding_f32_device(
            indices_handle: u64,
            weight_handle: u64,
            out_handle: u64,
            num_indices: usize,
            vocab_size: usize,
            embed_dim: usize,
        ) -> c_int;
        fn lumen_cuda_embedding_typed_device(
            indices_handle: u64,
            weight_handle: u64,
            weight_dtype: c_int,
            weight_scale: f32,
            out_handle: u64,
            num_indices: usize,
            vocab_size: usize,
            embed_dim: usize,
        ) -> c_int;
        fn lumen_cuda_embedding_typed_same_dtype_device(
            indices_handle: u64,
            weight_handle: u64,
            weight_dtype: c_int,
            out_handle: u64,
            num_indices: usize,
            vocab_size: usize,
            embed_dim: usize,
        ) -> c_int;
        fn lumen_cuda_embedding_backward_f32_device(
            indices_handle: u64,
            grad_handle: u64,
            grad_weight_handle: u64,
            num_indices: usize,
            vocab_size: usize,
            embed_dim: usize,
        ) -> c_int;
        fn lumen_cuda_rms_norm_f32_device(
            input_handle: u64,
            weight_handle: u64,
            out_handle: u64,
            rows: usize,
            dim: usize,
            eps: f32,
        ) -> c_int;
        fn lumen_cuda_rms_norm_typed_device(
            input_handle: u64,
            input_dtype: c_int,
            input_scale: f32,
            weight_handle: u64,
            weight_dtype: c_int,
            weight_scale: f32,
            out_handle: u64,
            rows: usize,
            dim: usize,
            eps: f32,
        ) -> c_int;
        fn lumen_cuda_rms_norm_i8_typed_out_device(
            input_handle: u64,
            input_dtype: c_int,
            input_scale: f32,
            weight_handle: u64,
            weight_dtype: c_int,
            weight_scale: f32,
            out_handle: u64,
            rows: usize,
            dim: usize,
            eps: f32,
            out_scale: *mut f32,
        ) -> c_int;
        fn lumen_cuda_rms_norm_backward_f32_device(
            input_handle: u64,
            weight_handle: u64,
            grad_handle: u64,
            grad_input_handle: u64,
            grad_weight_handle: u64,
            rows: usize,
            dim: usize,
            eps: f32,
        ) -> c_int;
        fn lumen_cuda_rms_norm_backward_typed_device(
            input_handle: u64,
            input_dtype: c_int,
            input_scale: f32,
            weight_handle: u64,
            weight_dtype: c_int,
            weight_scale: f32,
            grad_handle: u64,
            grad_input_handle: u64,
            grad_weight_handle: u64,
            rows: usize,
            dim: usize,
            eps: f32,
        ) -> c_int;
        fn lumen_cuda_permute_f32_device(
            input_handle: u64,
            out_handle: u64,
            ndim: usize,
            out_shape: *const usize,
            out_strides: *const usize,
            mapped_input_strides: *const usize,
            len: usize,
        ) -> c_int;
        fn lumen_cuda_permute_typed_device(
            input_handle: u64,
            dtype: c_int,
            out_handle: u64,
            ndim: usize,
            out_shape: *const usize,
            out_strides: *const usize,
            mapped_input_strides: *const usize,
            len: usize,
        ) -> c_int;
        fn lumen_cuda_slice_lastdim_f32_device(
            input_handle: u64,
            out_handle: u64,
            outer: usize,
            input_last_dim: usize,
            start: usize,
            slice_len: usize,
        ) -> c_int;
        fn lumen_cuda_slice_lastdim_typed_device(
            input_handle: u64,
            dtype: c_int,
            out_handle: u64,
            outer: usize,
            input_last_dim: usize,
            start: usize,
            slice_len: usize,
        ) -> c_int;
        fn lumen_cuda_slice_lastdim_backward_f32_device(
            grad_handle: u64,
            out_handle: u64,
            outer: usize,
            input_last_dim: usize,
            start: usize,
            slice_len: usize,
        ) -> c_int;
        fn lumen_cuda_cat_f32_device(
            lhs_handle: u64,
            rhs_handle: u64,
            out_handle: u64,
            ndim: usize,
            out_shape: *const usize,
            out_strides: *const usize,
            lhs_strides: *const usize,
            rhs_strides: *const usize,
            axis: usize,
            lhs_axis_len: usize,
            len: usize,
        ) -> c_int;
        fn lumen_cuda_cat_typed_device(
            lhs_handle: u64,
            rhs_handle: u64,
            dtype: c_int,
            out_handle: u64,
            ndim: usize,
            out_shape: *const usize,
            out_strides: *const usize,
            lhs_strides: *const usize,
            rhs_strides: *const usize,
            axis: usize,
            lhs_axis_len: usize,
            len: usize,
        ) -> c_int;
        fn lumen_cuda_cat_backward_slice_f32_device(
            grad_handle: u64,
            out_handle: u64,
            ndim: usize,
            input_shape: *const usize,
            input_strides: *const usize,
            out_strides: *const usize,
            axis: usize,
            axis_start: usize,
            len: usize,
        ) -> c_int;
        fn lumen_cuda_repeat_kv_f32_device(
            input_handle: u64,
            out_handle: u64,
            batch_size: usize,
            num_kv_heads: usize,
            seq_len: usize,
            dim: usize,
            n_rep: usize,
        ) -> c_int;
        fn lumen_cuda_repeat_kv_typed_device(
            input_handle: u64,
            dtype: c_int,
            out_handle: u64,
            batch_size: usize,
            num_kv_heads: usize,
            seq_len: usize,
            dim: usize,
            n_rep: usize,
        ) -> c_int;
        fn lumen_cuda_repeat_kv_backward_f32_device(
            grad_handle: u64,
            out_handle: u64,
            batch_size: usize,
            num_kv_heads: usize,
            seq_len: usize,
            dim: usize,
            n_rep: usize,
        ) -> c_int;
        fn lumen_cuda_decode_attention_f32_device(
            q_handle: u64,
            k_handle: u64,
            v_handle: u64,
            out_handle: u64,
            batch_size: usize,
            num_heads: usize,
            num_kv_heads: usize,
            active_seq_len: usize,
            cache_seq_len: usize,
            dim: usize,
            n_rep: usize,
            scale: f32,
        ) -> c_int;
        fn lumen_cuda_prefill_attention_f32_device(
            q_handle: u64,
            k_handle: u64,
            v_handle: u64,
            out_handle: u64,
            batch_size: usize,
            num_heads: usize,
            num_kv_heads: usize,
            q_seq_len: usize,
            active_seq_len: usize,
            cache_seq_len: usize,
            dim: usize,
            n_rep: usize,
            past_len: usize,
            scale: f32,
            is_causal: c_int,
        ) -> c_int;
        fn lumen_cuda_fused_gate_up_silu_f32_device(
            input_handle: u64,
            gate_handle: u64,
            up_handle: u64,
            out_handle: u64,
            rows: usize,
            n_dim: usize,
            k_dim: usize,
        ) -> c_int;
        fn lumen_cuda_silu_mul_f32_device(
            gate_handle: u64,
            up_handle: u64,
            out_handle: u64,
            len: usize,
        ) -> c_int;
        fn lumen_cuda_fused_gate_up_silu_typed_device(
            input_handle: u64,
            input_dtype: c_int,
            input_scale: f32,
            gate_handle: u64,
            weight_dtype: c_int,
            gate_scale: f32,
            up_handle: u64,
            up_scale: f32,
            out_handle: u64,
            rows: usize,
            n_dim: usize,
            k_dim: usize,
        ) -> c_int;
        fn lumen_cuda_fused_gate_up_silu_typed_out_device(
            input_handle: u64,
            input_dtype: c_int,
            input_scale: f32,
            gate_handle: u64,
            weight_dtype: c_int,
            gate_scale: f32,
            up_handle: u64,
            up_scale: f32,
            out_handle: u64,
            output_dtype: c_int,
            rows: usize,
            n_dim: usize,
            k_dim: usize,
        ) -> c_int;
        fn lumen_cuda_fused_qkv_f32_device(
            input_handle: u64,
            q_handle: u64,
            k_handle: u64,
            v_handle: u64,
            q_out_handle: u64,
            k_out_handle: u64,
            v_out_handle: u64,
            rows: usize,
            q_n: usize,
            k_n: usize,
            k_dim: usize,
        ) -> c_int;
        fn lumen_cuda_fused_qkv_typed_device(
            input_handle: u64,
            input_dtype: c_int,
            input_scale: f32,
            q_handle: u64,
            k_handle: u64,
            v_handle: u64,
            weight_dtype: c_int,
            q_scale: f32,
            k_scale: f32,
            v_scale: f32,
            q_out_handle: u64,
            k_out_handle: u64,
            v_out_handle: u64,
            rows: usize,
            q_n: usize,
            k_n: usize,
            k_dim: usize,
        ) -> c_int;
        fn lumen_cuda_fused_qkv_typed_out_device(
            input_handle: u64,
            input_dtype: c_int,
            input_scale: f32,
            q_handle: u64,
            k_handle: u64,
            v_handle: u64,
            weight_dtype: c_int,
            q_scale: f32,
            k_scale: f32,
            v_scale: f32,
            q_out_handle: u64,
            k_out_handle: u64,
            v_out_handle: u64,
            output_dtype: c_int,
            rows: usize,
            q_n: usize,
            k_n: usize,
            k_dim: usize,
        ) -> c_int;
        fn lumen_cuda_rope_f32_device(
            input_handle: u64,
            cos_handle: u64,
            sin_handle: u64,
            out_handle: u64,
            batch_size: usize,
            num_heads: usize,
            seq_len: usize,
            dim: usize,
            offset: usize,
            cache_seq_len: usize,
        ) -> c_int;
        fn lumen_cuda_rope_typed_device(
            input_handle: u64,
            input_dtype: c_int,
            input_scale: f32,
            cos_handle: u64,
            sin_handle: u64,
            cache_dtype: c_int,
            cos_scale: f32,
            sin_scale: f32,
            out_handle: u64,
            batch_size: usize,
            num_heads: usize,
            seq_len: usize,
            dim: usize,
            offset: usize,
            cache_seq_len: usize,
        ) -> c_int;
        fn lumen_cuda_rope_typed_i8_dynamic_device(
            input_handle: u64,
            input_dtype: c_int,
            input_scale: f32,
            cos_handle: u64,
            sin_handle: u64,
            cache_dtype: c_int,
            cos_scale: f32,
            sin_scale: f32,
            out_handle: u64,
            out_scale: *mut f32,
            batch_size: usize,
            num_heads: usize,
            seq_len: usize,
            dim: usize,
            offset: usize,
            cache_seq_len: usize,
        ) -> c_int;
        fn lumen_cuda_rope_backward_f32_device(
            grad_handle: u64,
            cos_handle: u64,
            sin_handle: u64,
            out_handle: u64,
            batch_size: usize,
            num_heads: usize,
            seq_len: usize,
            dim: usize,
            offset: usize,
            cache_seq_len: usize,
        ) -> c_int;
        fn lumen_cuda_conv2d_f32_device(
            input_handle: u64,
            weight_handle: u64,
            bias_handle: u64,
            out_handle: u64,
            batch_size: usize,
            in_channels: usize,
            in_h: usize,
            in_w: usize,
            out_channels: usize,
            k_h: usize,
            k_w: usize,
            pad_h: usize,
            pad_w: usize,
            stride_h: usize,
            stride_w: usize,
            out_h: usize,
            out_w: usize,
        ) -> c_int;
        fn lumen_cuda_conv2d_typed_device(
            input_handle: u64,
            input_dtype: c_int,
            input_scale: f32,
            weight_handle: u64,
            weight_dtype: c_int,
            weight_scale: f32,
            bias_handle: u64,
            bias_dtype: c_int,
            bias_scale: f32,
            out_handle: u64,
            batch_size: usize,
            in_channels: usize,
            in_h: usize,
            in_w: usize,
            out_channels: usize,
            k_h: usize,
            k_w: usize,
            pad_h: usize,
            pad_w: usize,
            stride_h: usize,
            stride_w: usize,
            out_h: usize,
            out_w: usize,
        ) -> c_int;
        fn lumen_cuda_conv2d_backward_f32_device(
            input_handle: u64,
            weight_handle: u64,
            grad_output_handle: u64,
            grad_input_handle: u64,
            grad_weight_handle: u64,
            grad_bias_handle: u64,
            batch_size: usize,
            in_channels: usize,
            in_h: usize,
            in_w: usize,
            out_channels: usize,
            k_h: usize,
            k_w: usize,
            pad_h: usize,
            pad_w: usize,
            stride_h: usize,
            stride_w: usize,
            out_h: usize,
            out_w: usize,
        ) -> c_int;
        fn lumen_cuda_max_pool2d_f32_device(
            input_handle: u64,
            out_handle: u64,
            batch_size: usize,
            channels: usize,
            in_h: usize,
            in_w: usize,
            kernel_h: usize,
            kernel_w: usize,
            stride_h: usize,
            stride_w: usize,
            out_h: usize,
            out_w: usize,
        ) -> c_int;
        fn lumen_cuda_max_pool2d_typed_device(
            input_handle: u64,
            input_dtype: c_int,
            input_scale: f32,
            out_handle: u64,
            batch_size: usize,
            channels: usize,
            in_h: usize,
            in_w: usize,
            kernel_h: usize,
            kernel_w: usize,
            stride_h: usize,
            stride_w: usize,
            out_h: usize,
            out_w: usize,
        ) -> c_int;
        fn lumen_cuda_max_pool2d_backward_f32_device(
            input_handle: u64,
            grad_output_handle: u64,
            grad_input_handle: u64,
            batch_size: usize,
            channels: usize,
            in_h: usize,
            in_w: usize,
            kernel_h: usize,
            kernel_w: usize,
            stride_h: usize,
            stride_w: usize,
            out_h: usize,
            out_w: usize,
        ) -> c_int;
        fn lumen_cuda_max_pool2d_backward_typed_device(
            input_handle: u64,
            input_dtype: c_int,
            input_scale: f32,
            grad_output_handle: u64,
            grad_input_handle: u64,
            batch_size: usize,
            channels: usize,
            in_h: usize,
            in_w: usize,
            kernel_h: usize,
            kernel_w: usize,
            stride_h: usize,
            stride_w: usize,
            out_h: usize,
            out_w: usize,
        ) -> c_int;
        fn lumen_cuda_last_error_message() -> *const c_char;
    }

    fn last_error_message() -> String {
        unsafe {
            let ptr = lumen_cuda_last_error_message();
            if ptr.is_null() {
                return "unknown CUDA error".to_string();
            }
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    }

    fn row_major_strides(shape: &[usize], context: &str) -> Result<Vec<usize>, String> {
        let mut strides = vec![0usize; shape.len()];
        let mut stride = 1usize;
        for i in (0..shape.len()).rev() {
            strides[i] = stride;
            stride = stride
                .checked_mul(shape[i])
                .ok_or_else(|| format!("CUDA {context} stride overflow"))?;
        }
        Ok(strides)
    }

    fn aligned_broadcast_metadata(
        lhs_shape: &[usize],
        rhs_shape: &[usize],
        out_shape: &[usize],
    ) -> Result<BroadcastMetadata, String> {
        let ndim = out_shape.len();
        if ndim == 0 {
            return Err("CUDA broadcast expects at least 1 dimension".to_string());
        }
        if lhs_shape.len() > ndim || rhs_shape.len() > ndim {
            return Err(format!(
                "CUDA broadcast rank mismatch: lhs={:?}, rhs={:?}, out={:?}",
                lhs_shape, rhs_shape, out_shape
            ));
        }

        let mut lhs_aligned = vec![1usize; ndim];
        let mut rhs_aligned = vec![1usize; ndim];
        let lhs_offset = ndim - lhs_shape.len();
        let rhs_offset = ndim - rhs_shape.len();
        lhs_aligned[lhs_offset..].copy_from_slice(lhs_shape);
        rhs_aligned[rhs_offset..].copy_from_slice(rhs_shape);

        for i in 0..ndim {
            let lhs_dim = lhs_aligned[i];
            let rhs_dim = rhs_aligned[i];
            let out_dim = out_shape[i];
            let expected = lhs_dim.max(rhs_dim);
            if (lhs_dim != out_dim && lhs_dim != 1)
                || (rhs_dim != out_dim && rhs_dim != 1)
                || expected != out_dim
            {
                return Err(format!(
                    "CUDA broadcast shape mismatch: lhs={:?}, rhs={:?}, out={:?}",
                    lhs_shape, rhs_shape, out_shape
                ));
            }
        }

        let out_strides = row_major_strides(out_shape, "broadcast output")?;
        let lhs_raw_strides = row_major_strides(&lhs_aligned, "broadcast lhs")?;
        let rhs_raw_strides = row_major_strides(&rhs_aligned, "broadcast rhs")?;
        let lhs_strides = lhs_aligned
            .iter()
            .zip(lhs_raw_strides.iter())
            .map(|(&dim, &stride)| if dim == 1 { 0 } else { stride })
            .collect::<Vec<_>>();
        let rhs_strides = rhs_aligned
            .iter()
            .zip(rhs_raw_strides.iter())
            .map(|(&dim, &stride)| if dim == 1 { 0 } else { stride })
            .collect::<Vec<_>>();
        Ok((
            lhs_aligned,
            rhs_aligned,
            out_strides,
            lhs_strides,
            rhs_strides,
        ))
    }

    fn dtype_tag(dtype: DType) -> Result<c_int, String> {
        match dtype {
            DType::F32 => Ok(0),
            DType::F16 => Ok(1),
            DType::BF16 => Ok(2),
            DType::I8 => Ok(3),
        }
    }

    pub(super) fn checked_len(label: &str, factors: &[usize]) -> Result<usize, String> {
        factors.iter().try_fold(1usize, |product, &factor| {
            product
                .checked_mul(factor)
                .ok_or_else(|| format!("{label} overflow"))
        })
    }

    pub(super) fn range_fits(start: usize, len: usize, total: usize) -> bool {
        start <= total && len <= total - start
    }

    pub(super) fn conv_output_dim(
        input: usize,
        padding: usize,
        kernel: usize,
        stride: usize,
        label: &str,
    ) -> Result<usize, String> {
        if stride == 0 {
            return Err(format!("{label} stride must be greater than zero"));
        }
        let padded = padding
            .checked_mul(2)
            .and_then(|value| input.checked_add(value))
            .ok_or_else(|| format!("{label} padded input size overflow"))?;
        if padded < kernel {
            return Err(format!("{label} kernel is larger than the padded input"));
        }
        (padded - kernel)
            .checked_div(stride)
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| format!("{label} output size overflow"))
    }

    pub(super) fn ensure_finite(label: &str, value: f32) -> Result<(), String> {
        if value.is_finite() {
            Ok(())
        } else {
            Err(format!("{label} must be finite"))
        }
    }

    pub(super) fn ensure_positive_finite(label: &str, value: f32) -> Result<(), String> {
        if value.is_finite() && value > 0.0 {
            Ok(())
        } else {
            Err(format!("{label} must be finite and > 0"))
        }
    }

    pub fn is_available() -> bool {
        unsafe { lumen_cuda_is_available() == 1 }
    }

    pub fn synchronize() -> Result<(), String> {
        let status = unsafe { lumen_cuda_synchronize() };
        if status == 0 {
            Ok(())
        } else {
            Err(last_error_message())
        }
    }

    pub fn alloc_f32(len: usize) -> Result<CudaBuffer, String> {
        let mut handle = 0u64;
        let status = unsafe { lumen_cuda_alloc_f32(len, &mut handle as *mut u64) };
        if status == 0 {
            Ok(CudaBuffer::from_raw(handle, len))
        } else {
            Err(last_error_message())
        }
    }

    pub fn alloc_storage(len: usize) -> Result<CudaBuffer, String> {
        alloc_f32(len)
    }

    pub fn upload_f32(src: &[f32]) -> Result<CudaBuffer, String> {
        let buffer = alloc_f32(src.len())?;
        let status = unsafe { lumen_cuda_upload_f32(buffer.handle(), src.as_ptr(), src.len()) };
        if status == 0 {
            Ok(buffer)
        } else {
            Err(last_error_message())
        }
    }

    pub fn upload_u16_storage(src: &[u16]) -> Result<CudaBuffer, String> {
        let buffer = alloc_f32(src.len())?;
        let status = unsafe { lumen_cuda_upload_u16(buffer.handle(), src.as_ptr(), src.len()) };
        if status == 0 {
            Ok(buffer)
        } else {
            Err(last_error_message())
        }
    }

    pub fn upload_i8_storage(src: &[i8]) -> Result<CudaBuffer, String> {
        let buffer = alloc_f32(src.len())?;
        let status = unsafe { lumen_cuda_upload_i8(buffer.handle(), src.as_ptr(), src.len()) };
        if status == 0 {
            Ok(buffer)
        } else {
            Err(last_error_message())
        }
    }

    pub fn download_f32(buffer: &CudaBuffer) -> Result<Vec<f32>, String> {
        let mut out = vec![0.0f32; buffer.len()];
        let status =
            unsafe { lumen_cuda_download_f32(buffer.handle(), out.as_mut_ptr(), buffer.len()) };
        if status == 0 {
            Ok(out)
        } else {
            Err(last_error_message())
        }
    }

    pub fn download_u16_storage(buffer: &CudaBuffer) -> Result<Vec<u16>, String> {
        let mut out = vec![0u16; buffer.len()];
        let status =
            unsafe { lumen_cuda_download_u16(buffer.handle(), out.as_mut_ptr(), buffer.len()) };
        if status == 0 {
            Ok(out)
        } else {
            Err(last_error_message())
        }
    }

    pub fn download_i8_storage(buffer: &CudaBuffer) -> Result<Vec<i8>, String> {
        let mut out = vec![0i8; buffer.len()];
        let status =
            unsafe { lumen_cuda_download_i8(buffer.handle(), out.as_mut_ptr(), buffer.len()) };
        if status == 0 {
            Ok(out)
        } else {
            Err(last_error_message())
        }
    }

    pub fn download_f32_offset(
        buffer: &CudaBuffer,
        offset: usize,
        len: usize,
    ) -> Result<Vec<f32>, String> {
        if offset > buffer.len() || len > buffer.len().saturating_sub(offset) {
            return Err(format!(
                "CUDA download offset out of bounds: offset={}, len={}, buffer_len={}",
                offset,
                len,
                buffer.len()
            ));
        }
        let mut out = vec![0.0f32; len];
        let status = unsafe {
            lumen_cuda_download_f32_offset(buffer.handle(), out.as_mut_ptr(), offset, len)
        };
        if status == 0 {
            Ok(out)
        } else {
            Err(last_error_message())
        }
    }

    pub fn matvec_argmax_f32(
        input: &CudaBuffer,
        weight: &CudaBuffer,
        batch_size: usize,
        vocab_size: usize,
        hidden_size: usize,
    ) -> Result<Vec<usize>, String> {
        if batch_size == 0 || vocab_size == 0 || hidden_size == 0 {
            return Err("CUDA matvec argmax dimensions must be greater than zero".to_string());
        }
        let input_len = batch_size
            .checked_mul(hidden_size)
            .ok_or_else(|| "CUDA matvec argmax input length overflow".to_string())?;
        if input.len() != input_len {
            return Err(format!(
                "CUDA matvec argmax input length mismatch: expected {}, got {}",
                input_len,
                input.len()
            ));
        }
        let weight_len = vocab_size
            .checked_mul(hidden_size)
            .ok_or_else(|| "CUDA matvec argmax weight length overflow".to_string())?;
        if weight.len() != weight_len {
            return Err(format!(
                "CUDA matvec argmax weight length mismatch: expected {}, got {}",
                weight_len,
                weight.len()
            ));
        }

        let mut out = vec![0usize; batch_size];
        let status = unsafe {
            lumen_cuda_matvec_argmax_f32_device(
                input.handle(),
                weight.handle(),
                out.as_mut_ptr(),
                batch_size,
                vocab_size,
                hidden_size,
            )
        };
        if status == 0 {
            Ok(out)
        } else {
            Err(last_error_message())
        }
    }

    fn validate_matvec_argmax_dims(
        input: &CudaBuffer,
        weight: &CudaBuffer,
        batch_size: usize,
        vocab_size: usize,
        hidden_size: usize,
        label: &str,
    ) -> Result<(), String> {
        if batch_size == 0 || vocab_size == 0 || hidden_size == 0 {
            return Err(format!(
                "CUDA {label} argmax dimensions must be greater than zero"
            ));
        }
        let input_len = batch_size
            .checked_mul(hidden_size)
            .ok_or_else(|| format!("CUDA {label} argmax input length overflow"))?;
        if input.len() != input_len {
            return Err(format!(
                "CUDA {label} argmax input length mismatch: expected {}, got {}",
                input_len,
                input.len()
            ));
        }
        let weight_len = vocab_size
            .checked_mul(hidden_size)
            .ok_or_else(|| format!("CUDA {label} argmax weight length overflow"))?;
        if weight.len() != weight_len {
            return Err(format!(
                "CUDA {label} argmax weight length mismatch: expected {}, got {}",
                weight_len,
                weight.len()
            ));
        }
        Ok(())
    }

    fn validate_matvec_argmax_scale(scale: f32, label: &str) -> Result<(), String> {
        if !scale.is_finite() || scale <= 0.0 {
            return Err(format!("CUDA {label} argmax scale must be finite and > 0"));
        }
        Ok(())
    }

    pub fn matvec_argmax_bf16_i8(
        input: &CudaBuffer,
        weight: &CudaBuffer,
        weight_scale: f32,
        batch_size: usize,
        vocab_size: usize,
        hidden_size: usize,
    ) -> Result<Vec<usize>, String> {
        validate_matvec_argmax_dims(
            input,
            weight,
            batch_size,
            vocab_size,
            hidden_size,
            "BF16xI8",
        )?;
        validate_matvec_argmax_scale(weight_scale, "BF16xI8 weight")?;

        let mut out = vec![0usize; batch_size];
        let status = unsafe {
            lumen_cuda_matvec_argmax_bf16_i8_device(
                input.handle(),
                weight.handle(),
                weight_scale,
                out.as_mut_ptr(),
                batch_size,
                vocab_size,
                hidden_size,
            )
        };
        if status == 0 {
            Ok(out)
        } else {
            Err(last_error_message())
        }
    }

    pub fn matvec_argmax_f16_i8(
        input: &CudaBuffer,
        weight: &CudaBuffer,
        weight_scale: f32,
        batch_size: usize,
        vocab_size: usize,
        hidden_size: usize,
    ) -> Result<Vec<usize>, String> {
        validate_matvec_argmax_dims(input, weight, batch_size, vocab_size, hidden_size, "F16xI8")?;
        validate_matvec_argmax_scale(weight_scale, "F16xI8 weight")?;

        let mut out = vec![0usize; batch_size];
        let status = unsafe {
            lumen_cuda_matvec_argmax_f16_i8_device(
                input.handle(),
                weight.handle(),
                weight_scale,
                out.as_mut_ptr(),
                batch_size,
                vocab_size,
                hidden_size,
            )
        };
        if status == 0 {
            Ok(out)
        } else {
            Err(last_error_message())
        }
    }

    pub fn matvec_argmax_f32_i8(
        input: &CudaBuffer,
        weight: &CudaBuffer,
        weight_scale: f32,
        batch_size: usize,
        vocab_size: usize,
        hidden_size: usize,
    ) -> Result<Vec<usize>, String> {
        validate_matvec_argmax_dims(input, weight, batch_size, vocab_size, hidden_size, "F32xI8")?;
        validate_matvec_argmax_scale(weight_scale, "F32xI8 weight")?;

        let mut out = vec![0usize; batch_size];
        let status = unsafe {
            lumen_cuda_matvec_argmax_f32_i8_device(
                input.handle(),
                weight.handle(),
                weight_scale,
                out.as_mut_ptr(),
                batch_size,
                vocab_size,
                hidden_size,
            )
        };
        if status == 0 {
            Ok(out)
        } else {
            Err(last_error_message())
        }
    }

    pub fn matvec_argmax_i8_i8(
        input: &CudaBuffer,
        input_scale: f32,
        weight: &CudaBuffer,
        weight_scale: f32,
        batch_size: usize,
        vocab_size: usize,
        hidden_size: usize,
    ) -> Result<Vec<usize>, String> {
        validate_matvec_argmax_dims(input, weight, batch_size, vocab_size, hidden_size, "I8xI8")?;
        validate_matvec_argmax_scale(input_scale, "I8xI8 input")?;
        validate_matvec_argmax_scale(weight_scale, "I8xI8 weight")?;

        let mut out = vec![0usize; batch_size];
        let status = unsafe {
            lumen_cuda_matvec_argmax_i8_i8_device(
                input.handle(),
                input_scale,
                weight.handle(),
                weight_scale,
                out.as_mut_ptr(),
                batch_size,
                vocab_size,
                hidden_size,
            )
        };
        if status == 0 {
            Ok(out)
        } else {
            Err(last_error_message())
        }
    }

    pub fn upload_f32_offset(
        buffer: &CudaBuffer,
        offset: usize,
        src: &[f32],
    ) -> Result<(), String> {
        if offset > buffer.len() || src.len() > buffer.len() - offset {
            return Err(format!(
                "CUDA upload offset out of bounds: offset={}, len={}, buffer_len={}",
                offset,
                src.len(),
                buffer.len()
            ));
        }
        let status = unsafe {
            lumen_cuda_upload_f32_offset(buffer.handle(), src.as_ptr(), offset, src.len())
        };
        if status == 0 {
            Ok(())
        } else {
            Err(last_error_message())
        }
    }

    pub fn copy_f32_offset(
        dst: &CudaBuffer,
        dst_offset: usize,
        src: &CudaBuffer,
        src_offset: usize,
        len: usize,
    ) -> Result<(), String> {
        if dst_offset > dst.len() || len > dst.len().saturating_sub(dst_offset) {
            return Err(format!(
                "CUDA copy dst out of bounds: dst_offset={}, len={}, dst_len={}",
                dst_offset,
                len,
                dst.len()
            ));
        }
        if src_offset > src.len() || len > src.len().saturating_sub(src_offset) {
            return Err(format!(
                "CUDA copy src out of bounds: src_offset={}, len={}, src_len={}",
                src_offset,
                len,
                src.len()
            ));
        }
        let status = unsafe {
            lumen_cuda_copy_f32_offset(dst.handle(), dst_offset, src.handle(), src_offset, len)
        };
        if status == 0 {
            Ok(())
        } else {
            Err(last_error_message())
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn append_kv_cache_f32(
        dst: &CudaBuffer,
        src: &CudaBuffer,
        batch_size: usize,
        num_heads: usize,
        src_seq_len: usize,
        dst_seq_len: usize,
        dim: usize,
        dst_start: usize,
    ) -> Result<(), String> {
        let src_len = batch_size
            .checked_mul(num_heads)
            .and_then(|v| v.checked_mul(src_seq_len))
            .and_then(|v| v.checked_mul(dim))
            .ok_or_else(|| "CUDA KV cache source length overflow".to_string())?;
        if src.len() != src_len {
            return Err(format!(
                "CUDA KV cache source length mismatch: expected {}, got {}",
                src_len,
                src.len()
            ));
        }

        let dst_len = batch_size
            .checked_mul(num_heads)
            .and_then(|v| v.checked_mul(dst_seq_len))
            .and_then(|v| v.checked_mul(dim))
            .ok_or_else(|| "CUDA KV cache destination length overflow".to_string())?;
        if dst.len() != dst_len {
            return Err(format!(
                "CUDA KV cache destination length mismatch: expected {}, got {}",
                dst_len,
                dst.len()
            ));
        }
        if dst_start > dst_seq_len || src_seq_len > dst_seq_len.saturating_sub(dst_start) {
            return Err(format!(
                "CUDA KV cache append range out of bounds: start={}, src_seq_len={}, dst_seq_len={}",
                dst_start, src_seq_len, dst_seq_len
            ));
        }

        let status = unsafe {
            lumen_cuda_append_kv_cache_f32_device(
                dst.handle(),
                src.handle(),
                batch_size,
                num_heads,
                src_seq_len,
                dst_seq_len,
                dim,
                dst_start,
            )
        };
        if status == 0 {
            Ok(())
        } else {
            Err(last_error_message())
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn append_kv_cache_pair_f32(
        k_dst: &CudaBuffer,
        v_dst: &CudaBuffer,
        k_src: &CudaBuffer,
        v_src: &CudaBuffer,
        batch_size: usize,
        num_heads: usize,
        src_seq_len: usize,
        dst_seq_len: usize,
        dim: usize,
        dst_start: usize,
    ) -> Result<(), String> {
        let src_len = batch_size
            .checked_mul(num_heads)
            .and_then(|v| v.checked_mul(src_seq_len))
            .and_then(|v| v.checked_mul(dim))
            .ok_or_else(|| "CUDA KV cache pair source length overflow".to_string())?;
        if k_src.len() != src_len || v_src.len() != src_len {
            return Err(format!(
                "CUDA KV cache pair source length mismatch: expected {}, got k={}, v={}",
                src_len,
                k_src.len(),
                v_src.len()
            ));
        }

        let dst_len = batch_size
            .checked_mul(num_heads)
            .and_then(|v| v.checked_mul(dst_seq_len))
            .and_then(|v| v.checked_mul(dim))
            .ok_or_else(|| "CUDA KV cache pair destination length overflow".to_string())?;
        if k_dst.len() != dst_len || v_dst.len() != dst_len {
            return Err(format!(
                "CUDA KV cache pair destination length mismatch: expected {}, got k={}, v={}",
                dst_len,
                k_dst.len(),
                v_dst.len()
            ));
        }
        if dst_start > dst_seq_len || src_seq_len > dst_seq_len.saturating_sub(dst_start) {
            return Err(format!(
                "CUDA KV cache pair append range out of bounds: start={}, src_seq_len={}, dst_seq_len={}",
                dst_start, src_seq_len, dst_seq_len
            ));
        }

        let status = unsafe {
            lumen_cuda_append_kv_cache_pair_f32_device(
                k_dst.handle(),
                v_dst.handle(),
                k_src.handle(),
                v_src.handle(),
                batch_size,
                num_heads,
                src_seq_len,
                dst_seq_len,
                dim,
                dst_start,
            )
        };
        if status == 0 {
            Ok(())
        } else {
            Err(last_error_message())
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn decode_rope_q_append_kv_f32_buffer(
        q_src: &CudaBuffer,
        k_src: &CudaBuffer,
        v_src: &CudaBuffer,
        cos: &CudaBuffer,
        sin: &CudaBuffer,
        k_cache: &CudaBuffer,
        v_cache: &CudaBuffer,
        batch_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        dim: usize,
        dst_seq_len: usize,
        offset: usize,
        cache_seq_len: usize,
    ) -> Result<CudaBuffer, String> {
        let q_len = batch_size
            .checked_mul(num_heads)
            .and_then(|v| v.checked_mul(dim))
            .ok_or_else(|| "CUDA decode RoPE Q length overflow".to_string())?;
        let kv_step_len = batch_size
            .checked_mul(num_kv_heads)
            .and_then(|v| v.checked_mul(dim))
            .ok_or_else(|| "CUDA decode RoPE KV step length overflow".to_string())?;
        let cache_len = batch_size
            .checked_mul(num_kv_heads)
            .and_then(|v| v.checked_mul(dst_seq_len))
            .and_then(|v| v.checked_mul(dim))
            .ok_or_else(|| "CUDA decode RoPE KV cache length overflow".to_string())?;
        let rope_cache_len = cache_seq_len
            .checked_mul(dim)
            .ok_or_else(|| "CUDA decode RoPE cache length overflow".to_string())?;
        if q_src.len() != q_len {
            return Err(format!(
                "CUDA decode RoPE Q length mismatch: expected {}, got {}",
                q_len,
                q_src.len()
            ));
        }
        if k_src.len() != kv_step_len || v_src.len() != kv_step_len {
            return Err(format!(
                "CUDA decode RoPE KV step length mismatch: expected {}, got k={}, v={}",
                kv_step_len,
                k_src.len(),
                v_src.len()
            ));
        }
        if k_cache.len() != cache_len || v_cache.len() != cache_len {
            return Err(format!(
                "CUDA decode RoPE KV cache length mismatch: expected {}, got k={}, v={}",
                cache_len,
                k_cache.len(),
                v_cache.len()
            ));
        }
        if cos.len() != rope_cache_len || sin.len() != rope_cache_len {
            return Err(format!(
                "CUDA decode RoPE cache length mismatch: expected {}, got cos={}, sin={}",
                rope_cache_len,
                cos.len(),
                sin.len()
            ));
        }
        if batch_size == 0 || num_heads == 0 || num_kv_heads == 0 || dim == 0 || dst_seq_len == 0 {
            return Err("CUDA decode RoPE dimensions must be greater than zero".to_string());
        }
        if !dim.is_multiple_of(2) {
            return Err(format!(
                "CUDA decode RoPE expects a positive even dimension, got {}",
                dim
            ));
        }
        if offset >= dst_seq_len || offset >= cache_seq_len {
            return Err(format!(
                "CUDA decode RoPE offset out of bounds: offset={}, dst_seq_len={}, cache_seq_len={}",
                offset, dst_seq_len, cache_seq_len
            ));
        }

        let q_out = alloc_f32(q_len)?;
        let status = unsafe {
            lumen_cuda_decode_rope_q_append_kv_f32_device(
                q_src.handle(),
                k_src.handle(),
                v_src.handle(),
                cos.handle(),
                sin.handle(),
                q_out.handle(),
                k_cache.handle(),
                v_cache.handle(),
                batch_size,
                num_heads,
                num_kv_heads,
                dim,
                dst_seq_len,
                offset,
                cache_seq_len,
            )
        };
        if status == 0 {
            Ok(q_out)
        } else {
            Err(last_error_message())
        }
    }

    pub fn kv_cache_prefix_f32_buffer(
        src: &CudaBuffer,
        batch_size: usize,
        num_heads: usize,
        active_seq_len: usize,
        src_seq_len: usize,
        dim: usize,
    ) -> Result<CudaBuffer, String> {
        let src_len = batch_size
            .checked_mul(num_heads)
            .and_then(|v| v.checked_mul(src_seq_len))
            .and_then(|v| v.checked_mul(dim))
            .ok_or_else(|| "CUDA KV cache source length overflow".to_string())?;
        if src.len() != src_len {
            return Err(format!(
                "CUDA KV cache source length mismatch: expected {}, got {}",
                src_len,
                src.len()
            ));
        }
        if active_seq_len == 0 || active_seq_len > src_seq_len {
            return Err(format!(
                "CUDA KV cache prefix range out of bounds: active_seq_len={}, src_seq_len={}",
                active_seq_len, src_seq_len
            ));
        }
        let out_len = batch_size
            .checked_mul(num_heads)
            .and_then(|v| v.checked_mul(active_seq_len))
            .and_then(|v| v.checked_mul(dim))
            .ok_or_else(|| "CUDA KV cache prefix output length overflow".to_string())?;
        let out = alloc_f32(out_len)?;
        let status = unsafe {
            lumen_cuda_kv_cache_prefix_f32_device(
                src.handle(),
                out.handle(),
                batch_size,
                num_heads,
                active_seq_len,
                src_seq_len,
                dim,
            )
        };
        if status == 0 {
            Ok(out)
        } else {
            Err(last_error_message())
        }
    }

    pub fn kv_cache_prefix_typed_buffer(
        src: &CudaBuffer,
        dtype: DType,
        batch_size: usize,
        num_heads: usize,
        active_seq_len: usize,
        src_seq_len: usize,
        dim: usize,
    ) -> Result<CudaBuffer, String> {
        let src_len = batch_size
            .checked_mul(num_heads)
            .and_then(|v| v.checked_mul(src_seq_len))
            .and_then(|v| v.checked_mul(dim))
            .ok_or_else(|| "CUDA typed KV cache source length overflow".to_string())?;
        if src.len() != src_len {
            return Err(format!(
                "CUDA typed KV cache source length mismatch: expected {}, got {}",
                src_len,
                src.len()
            ));
        }
        if active_seq_len == 0 || active_seq_len > src_seq_len {
            return Err(format!(
                "CUDA typed KV cache prefix range out of bounds: active_seq_len={}, src_seq_len={}",
                active_seq_len, src_seq_len
            ));
        }
        let out_len = batch_size
            .checked_mul(num_heads)
            .and_then(|v| v.checked_mul(active_seq_len))
            .and_then(|v| v.checked_mul(dim))
            .ok_or_else(|| "CUDA typed KV cache prefix output length overflow".to_string())?;
        let out = alloc_storage(out_len)?;
        let status = unsafe {
            lumen_cuda_kv_cache_prefix_typed_device(
                src.handle(),
                dtype_tag(dtype)?,
                out.handle(),
                batch_size,
                num_heads,
                active_seq_len,
                src_seq_len,
                dim,
            )
        };
        if status == 0 {
            Ok(out)
        } else {
            Err(last_error_message())
        }
    }

    pub fn free_f32(handle: u64, len: usize) {
        unsafe { lumen_cuda_free_f32(handle, len) };
    }

    pub fn matmul_f32(
        a: &CudaBuffer,
        b: &CudaBuffer,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<(CudaBuffer, Vec<f32>), String> {
        let out = matmul_f32_no_host(a, b, m, n, k)?;
        let host = download_f32(&out)?;
        Ok((out, host))
    }

    pub fn matmul_f32_no_host(
        a: &CudaBuffer,
        b: &CudaBuffer,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<CudaBuffer, String> {
        if a.len() != m * k {
            return Err(format!(
                "CUDA matmul A length mismatch: expected {}, got {}",
                m * k,
                a.len()
            ));
        }
        if b.len() != n * k {
            return Err(format!(
                "CUDA matmul B length mismatch: expected {}, got {}",
                n * k,
                b.len()
            ));
        }
        let out = alloc_f32(m * n)?;
        let status =
            unsafe { lumen_cuda_matmul_f32_device(a.handle(), b.handle(), out.handle(), m, n, k) };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(out)
    }

    pub fn matmul_bf16_host_no_host(
        a: &[u16],
        b: &[u16],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<CudaBuffer, String> {
        if a.len() != m * k {
            return Err(format!(
                "CUDA BF16 matmul A length mismatch: expected {}, got {}",
                m * k,
                a.len()
            ));
        }
        if b.len() != n * k {
            return Err(format!(
                "CUDA BF16 matmul B length mismatch: expected {}, got {}",
                n * k,
                b.len()
            ));
        }
        let out = alloc_f32(m * n)?;
        let status = unsafe {
            lumen_cuda_matmul_bf16_host_device(a.as_ptr(), b.as_ptr(), out.handle(), m, n, k)
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(out)
    }

    pub fn matmul_bf16_buffer_no_host(
        a: &CudaBuffer,
        b: &CudaBuffer,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<CudaBuffer, String> {
        if a.len() != m * k {
            return Err(format!(
                "CUDA BF16 resident matmul A length mismatch: expected {}, got {}",
                m * k,
                a.len()
            ));
        }
        if b.len() != n * k {
            return Err(format!(
                "CUDA BF16 resident matmul B length mismatch: expected {}, got {}",
                n * k,
                b.len()
            ));
        }
        let out = alloc_f32(m * n)?;
        let status =
            unsafe { lumen_cuda_matmul_bf16_device(a.handle(), b.handle(), out.handle(), m, n, k) };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(out)
    }

    pub fn matmul_bf16_typed_output_buffer_no_host(
        a: &CudaBuffer,
        b: &CudaBuffer,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<CudaBuffer, String> {
        if a.len() != m * k {
            return Err(format!(
                "CUDA BF16 typed-output matmul A length mismatch: expected {}, got {}",
                m * k,
                a.len()
            ));
        }
        if b.len() != n * k {
            return Err(format!(
                "CUDA BF16 typed-output matmul B length mismatch: expected {}, got {}",
                n * k,
                b.len()
            ));
        }
        let out = alloc_storage(m * n)?;
        let status = unsafe {
            lumen_cuda_matmul_bf16_typed_out_device(a.handle(), b.handle(), out.handle(), m, n, k)
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(out)
    }

    pub fn matmul_f16_host_no_host(
        a: &[u16],
        b: &[u16],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<CudaBuffer, String> {
        if a.len() != m * k {
            return Err(format!(
                "CUDA F16 matmul A length mismatch: expected {}, got {}",
                m * k,
                a.len()
            ));
        }
        if b.len() != n * k {
            return Err(format!(
                "CUDA F16 matmul B length mismatch: expected {}, got {}",
                n * k,
                b.len()
            ));
        }
        let out = alloc_f32(m * n)?;
        let status = unsafe {
            lumen_cuda_matmul_f16_host_device(a.as_ptr(), b.as_ptr(), out.handle(), m, n, k)
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(out)
    }

    pub fn matmul_f16_buffer_no_host(
        a: &CudaBuffer,
        b: &CudaBuffer,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<CudaBuffer, String> {
        if a.len() != m * k {
            return Err(format!(
                "CUDA F16 resident matmul A length mismatch: expected {}, got {}",
                m * k,
                a.len()
            ));
        }
        if b.len() != n * k {
            return Err(format!(
                "CUDA F16 resident matmul B length mismatch: expected {}, got {}",
                n * k,
                b.len()
            ));
        }
        let out = alloc_f32(m * n)?;
        let status =
            unsafe { lumen_cuda_matmul_f16_device(a.handle(), b.handle(), out.handle(), m, n, k) };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(out)
    }

    pub fn matmul_f16_typed_output_buffer_no_host(
        a: &CudaBuffer,
        b: &CudaBuffer,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<CudaBuffer, String> {
        if a.len() != m * k {
            return Err(format!(
                "CUDA F16 typed-output matmul A length mismatch: expected {}, got {}",
                m * k,
                a.len()
            ));
        }
        if b.len() != n * k {
            return Err(format!(
                "CUDA F16 typed-output matmul B length mismatch: expected {}, got {}",
                n * k,
                b.len()
            ));
        }
        let out = alloc_storage(m * n)?;
        let status = unsafe {
            lumen_cuda_matmul_f16_typed_out_device(a.handle(), b.handle(), out.handle(), m, n, k)
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(out)
    }

    pub fn matmul_i8_host_no_host(
        a: &[i8],
        a_scale: f32,
        b: &[i8],
        b_scale: f32,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<CudaBuffer, String> {
        if a.len() != m * k {
            return Err(format!(
                "CUDA I8 matmul A length mismatch: expected {}, got {}",
                m * k,
                a.len()
            ));
        }
        if b.len() != n * k {
            return Err(format!(
                "CUDA I8 matmul B length mismatch: expected {}, got {}",
                n * k,
                b.len()
            ));
        }
        let out = alloc_f32(m * n)?;
        let status = unsafe {
            lumen_cuda_matmul_i8_host_device(
                a.as_ptr(),
                b.as_ptr(),
                a_scale,
                b_scale,
                out.handle(),
                m,
                n,
                k,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(out)
    }

    pub fn matmul_i8_buffer_no_host(
        a: &CudaBuffer,
        a_scale: f32,
        b: &CudaBuffer,
        b_scale: f32,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<CudaBuffer, String> {
        if a.len() != m * k {
            return Err(format!(
                "CUDA I8 resident matmul A length mismatch: expected {}, got {}",
                m * k,
                a.len()
            ));
        }
        if b.len() != n * k {
            return Err(format!(
                "CUDA I8 resident matmul B length mismatch: expected {}, got {}",
                n * k,
                b.len()
            ));
        }
        let out = alloc_f32(m * n)?;
        let status = unsafe {
            lumen_cuda_matmul_i8_device(
                a.handle(),
                a_scale,
                b.handle(),
                b_scale,
                out.handle(),
                m,
                n,
                k,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(out)
    }

    pub fn matmul_bf16_i8_buffer_no_host(
        a: &CudaBuffer,
        b: &CudaBuffer,
        b_scale: f32,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<CudaBuffer, String> {
        if a.len() != m * k {
            return Err(format!(
                "CUDA BF16xI8 matmul A length mismatch: expected {}, got {}",
                m * k,
                a.len()
            ));
        }
        if b.len() != n * k {
            return Err(format!(
                "CUDA BF16xI8 matmul B length mismatch: expected {}, got {}",
                n * k,
                b.len()
            ));
        }
        let out = alloc_f32(m * n)?;
        let status = unsafe {
            lumen_cuda_matmul_bf16_i8_device(a.handle(), b.handle(), b_scale, out.handle(), m, n, k)
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(out)
    }

    pub fn matmul_f16_i8_buffer_no_host(
        a: &CudaBuffer,
        b: &CudaBuffer,
        b_scale: f32,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<CudaBuffer, String> {
        if a.len() != m * k {
            return Err(format!(
                "CUDA F16xI8 matmul A length mismatch: expected {}, got {}",
                m * k,
                a.len()
            ));
        }
        if b.len() != n * k {
            return Err(format!(
                "CUDA F16xI8 matmul B length mismatch: expected {}, got {}",
                n * k,
                b.len()
            ));
        }
        let out = alloc_f32(m * n)?;
        let status = unsafe {
            lumen_cuda_matmul_f16_i8_device(a.handle(), b.handle(), b_scale, out.handle(), m, n, k)
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(out)
    }

    pub fn matmul_f32_i8_buffer_no_host(
        a: &CudaBuffer,
        b: &CudaBuffer,
        b_scale: f32,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<CudaBuffer, String> {
        if a.len() != m * k {
            return Err(format!(
                "CUDA F32xI8 matmul A length mismatch: expected {}, got {}",
                m * k,
                a.len()
            ));
        }
        if b.len() != n * k {
            return Err(format!(
                "CUDA F32xI8 matmul B length mismatch: expected {}, got {}",
                n * k,
                b.len()
            ));
        }
        let out = alloc_f32(m * n)?;
        let status = unsafe {
            lumen_cuda_matmul_f32_i8_device(a.handle(), b.handle(), b_scale, out.handle(), m, n, k)
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(out)
    }

    pub fn matmul_i8_bf16_buffer_no_host(
        a: &CudaBuffer,
        a_scale: f32,
        b: &CudaBuffer,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<CudaBuffer, String> {
        if a.len() != m * k {
            return Err(format!(
                "CUDA I8xBF16 matmul A length mismatch: expected {}, got {}",
                m * k,
                a.len()
            ));
        }
        if b.len() != n * k {
            return Err(format!(
                "CUDA I8xBF16 matmul B length mismatch: expected {}, got {}",
                n * k,
                b.len()
            ));
        }
        let out = alloc_f32(m * n)?;
        let status = unsafe {
            lumen_cuda_matmul_i8_bf16_device(a.handle(), a_scale, b.handle(), out.handle(), m, n, k)
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(out)
    }

    pub fn matmul_i8_f16_buffer_no_host(
        a: &CudaBuffer,
        a_scale: f32,
        b: &CudaBuffer,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<CudaBuffer, String> {
        if a.len() != m * k {
            return Err(format!(
                "CUDA I8xF16 matmul A length mismatch: expected {}, got {}",
                m * k,
                a.len()
            ));
        }
        if b.len() != n * k {
            return Err(format!(
                "CUDA I8xF16 matmul B length mismatch: expected {}, got {}",
                n * k,
                b.len()
            ));
        }
        let out = alloc_f32(m * n)?;
        let status = unsafe {
            lumen_cuda_matmul_i8_f16_device(a.handle(), a_scale, b.handle(), out.handle(), m, n, k)
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(out)
    }

    pub fn matmul_i8_f32_buffer_no_host(
        a: &CudaBuffer,
        a_scale: f32,
        b: &CudaBuffer,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<CudaBuffer, String> {
        if a.len() != m * k {
            return Err(format!(
                "CUDA I8xF32 matmul A length mismatch: expected {}, got {}",
                m * k,
                a.len()
            ));
        }
        if b.len() != n * k {
            return Err(format!(
                "CUDA I8xF32 matmul B length mismatch: expected {}, got {}",
                n * k,
                b.len()
            ));
        }
        let out = alloc_f32(m * n)?;
        let status = unsafe {
            lumen_cuda_matmul_i8_f32_device(a.handle(), a_scale, b.handle(), out.handle(), m, n, k)
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(out)
    }

    pub fn matmul_i8_typed_output_buffer_no_host(
        a: &CudaBuffer,
        a_scale: f32,
        b: &CudaBuffer,
        b_scale: f32,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<(CudaBuffer, f32), String> {
        if a.len() != m * k {
            return Err(format!(
                "CUDA I8 typed-output matmul A length mismatch: expected {}, got {}",
                m * k,
                a.len()
            ));
        }
        if b.len() != n * k {
            return Err(format!(
                "CUDA I8 typed-output matmul B length mismatch: expected {}, got {}",
                n * k,
                b.len()
            ));
        }
        let out = alloc_storage(m * n)?;
        let mut out_scale = 1.0f32;
        let status = unsafe {
            lumen_cuda_matmul_i8_typed_out_device(
                a.handle(),
                a_scale,
                b.handle(),
                b_scale,
                out.handle(),
                m,
                n,
                k,
                &mut out_scale as *mut f32,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok((out, out_scale))
    }

    pub fn batch_matmul_f32(
        lhs: &CudaBuffer,
        rhs: &CudaBuffer,
        batch_count: usize,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<(CudaBuffer, Vec<f32>), String> {
        let out = batch_matmul_f32_no_host(lhs, rhs, batch_count, m, n, k)?;
        let host = download_f32(&out)?;
        Ok((out, host))
    }

    pub fn batch_matmul_f32_no_host(
        lhs: &CudaBuffer,
        rhs: &CudaBuffer,
        batch_count: usize,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<CudaBuffer, String> {
        if lhs.len() != batch_count * m * k {
            return Err(format!(
                "CUDA batch_matmul lhs length mismatch: expected {}, got {}",
                batch_count * m * k,
                lhs.len()
            ));
        }
        if rhs.len() != batch_count * k * n {
            return Err(format!(
                "CUDA batch_matmul rhs length mismatch: expected {}, got {}",
                batch_count * k * n,
                rhs.len()
            ));
        }
        let out = alloc_f32(batch_count * m * n)?;
        let status = unsafe {
            lumen_cuda_batch_matmul_f32_device(
                lhs.handle(),
                rhs.handle(),
                out.handle(),
                batch_count,
                m,
                n,
                k,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(out)
    }

    pub fn batch_matmul_bf16_buffer_no_host(
        lhs: &CudaBuffer,
        rhs: &CudaBuffer,
        batch_count: usize,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<CudaBuffer, String> {
        if lhs.len() != batch_count * m * k {
            return Err(format!(
                "CUDA BF16 batch_matmul lhs length mismatch: expected {}, got {}",
                batch_count * m * k,
                lhs.len()
            ));
        }
        if rhs.len() != batch_count * k * n {
            return Err(format!(
                "CUDA BF16 batch_matmul rhs length mismatch: expected {}, got {}",
                batch_count * k * n,
                rhs.len()
            ));
        }
        let out = alloc_f32(batch_count * m * n)?;
        let status = unsafe {
            lumen_cuda_batch_matmul_bf16_device(
                lhs.handle(),
                rhs.handle(),
                out.handle(),
                batch_count,
                m,
                n,
                k,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(out)
    }

    pub fn batch_matmul_bf16_typed_output_buffer_no_host(
        lhs: &CudaBuffer,
        rhs: &CudaBuffer,
        batch_count: usize,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<CudaBuffer, String> {
        if lhs.len() != batch_count * m * k {
            return Err(format!(
                "CUDA BF16 typed-output batch_matmul lhs length mismatch: expected {}, got {}",
                batch_count * m * k,
                lhs.len()
            ));
        }
        if rhs.len() != batch_count * k * n {
            return Err(format!(
                "CUDA BF16 typed-output batch_matmul rhs length mismatch: expected {}, got {}",
                batch_count * k * n,
                rhs.len()
            ));
        }
        let out = alloc_storage(batch_count * m * n)?;
        let status = unsafe {
            lumen_cuda_batch_matmul_bf16_typed_out_device(
                lhs.handle(),
                rhs.handle(),
                out.handle(),
                batch_count,
                m,
                n,
                k,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(out)
    }

    pub fn batch_matmul_f16_buffer_no_host(
        lhs: &CudaBuffer,
        rhs: &CudaBuffer,
        batch_count: usize,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<CudaBuffer, String> {
        if lhs.len() != batch_count * m * k {
            return Err(format!(
                "CUDA F16 batch_matmul lhs length mismatch: expected {}, got {}",
                batch_count * m * k,
                lhs.len()
            ));
        }
        if rhs.len() != batch_count * k * n {
            return Err(format!(
                "CUDA F16 batch_matmul rhs length mismatch: expected {}, got {}",
                batch_count * k * n,
                rhs.len()
            ));
        }
        let out = alloc_f32(batch_count * m * n)?;
        let status = unsafe {
            lumen_cuda_batch_matmul_f16_device(
                lhs.handle(),
                rhs.handle(),
                out.handle(),
                batch_count,
                m,
                n,
                k,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(out)
    }

    pub fn batch_matmul_f16_typed_output_buffer_no_host(
        lhs: &CudaBuffer,
        rhs: &CudaBuffer,
        batch_count: usize,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<CudaBuffer, String> {
        if lhs.len() != batch_count * m * k {
            return Err(format!(
                "CUDA F16 typed-output batch_matmul lhs length mismatch: expected {}, got {}",
                batch_count * m * k,
                lhs.len()
            ));
        }
        if rhs.len() != batch_count * k * n {
            return Err(format!(
                "CUDA F16 typed-output batch_matmul rhs length mismatch: expected {}, got {}",
                batch_count * k * n,
                rhs.len()
            ));
        }
        let out = alloc_storage(batch_count * m * n)?;
        let status = unsafe {
            lumen_cuda_batch_matmul_f16_typed_out_device(
                lhs.handle(),
                rhs.handle(),
                out.handle(),
                batch_count,
                m,
                n,
                k,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(out)
    }

    pub fn batch_matmul_bf16_i8_buffer_no_host(
        lhs: &CudaBuffer,
        rhs: &CudaBuffer,
        rhs_scale: f32,
        batch_count: usize,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<CudaBuffer, String> {
        if lhs.len() != batch_count * m * k {
            return Err(format!(
                "CUDA BF16xI8 batch_matmul lhs length mismatch: expected {}, got {}",
                batch_count * m * k,
                lhs.len()
            ));
        }
        if rhs.len() != batch_count * k * n {
            return Err(format!(
                "CUDA BF16xI8 batch_matmul rhs length mismatch: expected {}, got {}",
                batch_count * k * n,
                rhs.len()
            ));
        }
        let out = alloc_f32(batch_count * m * n)?;
        let status = unsafe {
            lumen_cuda_batch_matmul_bf16_i8_device(
                lhs.handle(),
                rhs.handle(),
                rhs_scale,
                out.handle(),
                batch_count,
                m,
                n,
                k,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(out)
    }

    pub fn batch_matmul_f16_i8_buffer_no_host(
        lhs: &CudaBuffer,
        rhs: &CudaBuffer,
        rhs_scale: f32,
        batch_count: usize,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<CudaBuffer, String> {
        if lhs.len() != batch_count * m * k {
            return Err(format!(
                "CUDA F16xI8 batch_matmul lhs length mismatch: expected {}, got {}",
                batch_count * m * k,
                lhs.len()
            ));
        }
        if rhs.len() != batch_count * k * n {
            return Err(format!(
                "CUDA F16xI8 batch_matmul rhs length mismatch: expected {}, got {}",
                batch_count * k * n,
                rhs.len()
            ));
        }
        let out = alloc_f32(batch_count * m * n)?;
        let status = unsafe {
            lumen_cuda_batch_matmul_f16_i8_device(
                lhs.handle(),
                rhs.handle(),
                rhs_scale,
                out.handle(),
                batch_count,
                m,
                n,
                k,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(out)
    }

    pub fn batch_matmul_f32_i8_buffer_no_host(
        lhs: &CudaBuffer,
        rhs: &CudaBuffer,
        rhs_scale: f32,
        batch_count: usize,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<CudaBuffer, String> {
        if lhs.len() != batch_count * m * k {
            return Err(format!(
                "CUDA F32xI8 batch_matmul lhs length mismatch: expected {}, got {}",
                batch_count * m * k,
                lhs.len()
            ));
        }
        if rhs.len() != batch_count * k * n {
            return Err(format!(
                "CUDA F32xI8 batch_matmul rhs length mismatch: expected {}, got {}",
                batch_count * k * n,
                rhs.len()
            ));
        }
        let out = alloc_f32(batch_count * m * n)?;
        let status = unsafe {
            lumen_cuda_batch_matmul_f32_i8_device(
                lhs.handle(),
                rhs.handle(),
                rhs_scale,
                out.handle(),
                batch_count,
                m,
                n,
                k,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(out)
    }

    pub fn batch_matmul_i8_bf16_buffer_no_host(
        lhs: &CudaBuffer,
        lhs_scale: f32,
        rhs: &CudaBuffer,
        batch_count: usize,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<CudaBuffer, String> {
        if lhs.len() != batch_count * m * k {
            return Err(format!(
                "CUDA I8xBF16 batch_matmul lhs length mismatch: expected {}, got {}",
                batch_count * m * k,
                lhs.len()
            ));
        }
        if rhs.len() != batch_count * k * n {
            return Err(format!(
                "CUDA I8xBF16 batch_matmul rhs length mismatch: expected {}, got {}",
                batch_count * k * n,
                rhs.len()
            ));
        }
        let out = alloc_f32(batch_count * m * n)?;
        let status = unsafe {
            lumen_cuda_batch_matmul_i8_bf16_device(
                lhs.handle(),
                lhs_scale,
                rhs.handle(),
                out.handle(),
                batch_count,
                m,
                n,
                k,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(out)
    }

    pub fn batch_matmul_i8_f16_buffer_no_host(
        lhs: &CudaBuffer,
        lhs_scale: f32,
        rhs: &CudaBuffer,
        batch_count: usize,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<CudaBuffer, String> {
        if lhs.len() != batch_count * m * k {
            return Err(format!(
                "CUDA I8xF16 batch_matmul lhs length mismatch: expected {}, got {}",
                batch_count * m * k,
                lhs.len()
            ));
        }
        if rhs.len() != batch_count * k * n {
            return Err(format!(
                "CUDA I8xF16 batch_matmul rhs length mismatch: expected {}, got {}",
                batch_count * k * n,
                rhs.len()
            ));
        }
        let out = alloc_f32(batch_count * m * n)?;
        let status = unsafe {
            lumen_cuda_batch_matmul_i8_f16_device(
                lhs.handle(),
                lhs_scale,
                rhs.handle(),
                out.handle(),
                batch_count,
                m,
                n,
                k,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(out)
    }

    pub fn batch_matmul_i8_f32_buffer_no_host(
        lhs: &CudaBuffer,
        lhs_scale: f32,
        rhs: &CudaBuffer,
        batch_count: usize,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<CudaBuffer, String> {
        if lhs.len() != batch_count * m * k {
            return Err(format!(
                "CUDA I8xF32 batch_matmul lhs length mismatch: expected {}, got {}",
                batch_count * m * k,
                lhs.len()
            ));
        }
        if rhs.len() != batch_count * k * n {
            return Err(format!(
                "CUDA I8xF32 batch_matmul rhs length mismatch: expected {}, got {}",
                batch_count * k * n,
                rhs.len()
            ));
        }
        let out = alloc_f32(batch_count * m * n)?;
        let status = unsafe {
            lumen_cuda_batch_matmul_i8_f32_device(
                lhs.handle(),
                lhs_scale,
                rhs.handle(),
                out.handle(),
                batch_count,
                m,
                n,
                k,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn batch_matmul_i8_buffer_no_host(
        lhs: &CudaBuffer,
        lhs_scale: f32,
        rhs: &CudaBuffer,
        rhs_scale: f32,
        batch_count: usize,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<CudaBuffer, String> {
        if lhs.len() != batch_count * m * k {
            return Err(format!(
                "CUDA I8 batch_matmul lhs length mismatch: expected {}, got {}",
                batch_count * m * k,
                lhs.len()
            ));
        }
        if rhs.len() != batch_count * k * n {
            return Err(format!(
                "CUDA I8 batch_matmul rhs length mismatch: expected {}, got {}",
                batch_count * k * n,
                rhs.len()
            ));
        }
        let out = alloc_f32(batch_count * m * n)?;
        let status = unsafe {
            lumen_cuda_batch_matmul_i8_device(
                lhs.handle(),
                lhs_scale,
                rhs.handle(),
                rhs_scale,
                out.handle(),
                batch_count,
                m,
                n,
                k,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(out)
    }

    pub fn batch_matmul_i8_typed_output_buffer_no_host(
        lhs: &CudaBuffer,
        lhs_scale: f32,
        rhs: &CudaBuffer,
        rhs_scale: f32,
        batch_count: usize,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<(CudaBuffer, f32), String> {
        if lhs.len() != batch_count * m * k {
            return Err(format!(
                "CUDA I8 typed-output batch_matmul lhs length mismatch: expected {}, got {}",
                batch_count * m * k,
                lhs.len()
            ));
        }
        if rhs.len() != batch_count * k * n {
            return Err(format!(
                "CUDA I8 typed-output batch_matmul rhs length mismatch: expected {}, got {}",
                batch_count * k * n,
                rhs.len()
            ));
        }
        let out = alloc_storage(batch_count * m * n)?;
        let mut out_scale = 1.0f32;
        let status = unsafe {
            lumen_cuda_batch_matmul_i8_typed_out_device(
                lhs.handle(),
                lhs_scale,
                rhs.handle(),
                rhs_scale,
                out.handle(),
                batch_count,
                m,
                n,
                k,
                &mut out_scale as *mut f32,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok((out, out_scale))
    }

    pub fn matmul_backward_f32_no_host(
        grad: &CudaBuffer,
        a: &CudaBuffer,
        b: &CudaBuffer,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<(CudaBuffer, CudaBuffer), String> {
        if grad.len() != m * n {
            return Err(format!(
                "CUDA matmul backward grad length mismatch: expected {}, got {}",
                m * n,
                grad.len()
            ));
        }
        if a.len() != m * k {
            return Err(format!(
                "CUDA matmul backward A length mismatch: expected {}, got {}",
                m * k,
                a.len()
            ));
        }
        if b.len() != n * k {
            return Err(format!(
                "CUDA matmul backward B length mismatch: expected {}, got {}",
                n * k,
                b.len()
            ));
        }
        let da = alloc_f32(m * k)?;
        let db = alloc_f32(n * k)?;
        let status = unsafe {
            lumen_cuda_matmul_backward_f32_device(
                grad.handle(),
                a.handle(),
                b.handle(),
                da.handle(),
                db.handle(),
                m,
                k,
                n,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok((da, db))
    }

    pub fn matmul_backward_bf16_i8_no_host(
        grad: &CudaBuffer,
        lhs: &CudaBuffer,
        rhs: &CudaBuffer,
        rhs_scale: f32,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<(CudaBuffer, CudaBuffer), String> {
        validate_matmul_backward_lengths(grad, lhs, rhs, m, k, n, "BF16xI8")?;
        let d_lhs = alloc_f32(m * k)?;
        let d_rhs = alloc_f32(n * k)?;
        let status = unsafe {
            lumen_cuda_matmul_backward_bf16_i8_device(
                grad.handle(),
                lhs.handle(),
                rhs.handle(),
                rhs_scale,
                d_lhs.handle(),
                d_rhs.handle(),
                m,
                k,
                n,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok((d_lhs, d_rhs))
    }

    pub fn matmul_backward_f16_i8_no_host(
        grad: &CudaBuffer,
        lhs: &CudaBuffer,
        rhs: &CudaBuffer,
        rhs_scale: f32,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<(CudaBuffer, CudaBuffer), String> {
        validate_matmul_backward_lengths(grad, lhs, rhs, m, k, n, "F16xI8")?;
        let d_lhs = alloc_f32(m * k)?;
        let d_rhs = alloc_f32(n * k)?;
        let status = unsafe {
            lumen_cuda_matmul_backward_f16_i8_device(
                grad.handle(),
                lhs.handle(),
                rhs.handle(),
                rhs_scale,
                d_lhs.handle(),
                d_rhs.handle(),
                m,
                k,
                n,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok((d_lhs, d_rhs))
    }

    pub fn matmul_backward_f32_i8_no_host(
        grad: &CudaBuffer,
        lhs: &CudaBuffer,
        rhs: &CudaBuffer,
        rhs_scale: f32,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<(CudaBuffer, CudaBuffer), String> {
        validate_matmul_backward_lengths(grad, lhs, rhs, m, k, n, "F32xI8")?;
        let d_lhs = alloc_f32(m * k)?;
        let d_rhs = alloc_f32(n * k)?;
        let status = unsafe {
            lumen_cuda_matmul_backward_f32_i8_device(
                grad.handle(),
                lhs.handle(),
                rhs.handle(),
                rhs_scale,
                d_lhs.handle(),
                d_rhs.handle(),
                m,
                k,
                n,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok((d_lhs, d_rhs))
    }

    pub fn matmul_backward_i8_bf16_no_host(
        grad: &CudaBuffer,
        lhs: &CudaBuffer,
        lhs_scale: f32,
        rhs: &CudaBuffer,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<(CudaBuffer, CudaBuffer), String> {
        validate_matmul_backward_lengths(grad, lhs, rhs, m, k, n, "I8xBF16")?;
        let d_lhs = alloc_f32(m * k)?;
        let d_rhs = alloc_f32(n * k)?;
        let status = unsafe {
            lumen_cuda_matmul_backward_i8_bf16_device(
                grad.handle(),
                lhs.handle(),
                lhs_scale,
                rhs.handle(),
                d_lhs.handle(),
                d_rhs.handle(),
                m,
                k,
                n,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok((d_lhs, d_rhs))
    }

    pub fn matmul_backward_i8_f16_no_host(
        grad: &CudaBuffer,
        lhs: &CudaBuffer,
        lhs_scale: f32,
        rhs: &CudaBuffer,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<(CudaBuffer, CudaBuffer), String> {
        validate_matmul_backward_lengths(grad, lhs, rhs, m, k, n, "I8xF16")?;
        let d_lhs = alloc_f32(m * k)?;
        let d_rhs = alloc_f32(n * k)?;
        let status = unsafe {
            lumen_cuda_matmul_backward_i8_f16_device(
                grad.handle(),
                lhs.handle(),
                lhs_scale,
                rhs.handle(),
                d_lhs.handle(),
                d_rhs.handle(),
                m,
                k,
                n,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok((d_lhs, d_rhs))
    }

    pub fn matmul_backward_i8_f32_no_host(
        grad: &CudaBuffer,
        lhs: &CudaBuffer,
        lhs_scale: f32,
        rhs: &CudaBuffer,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<(CudaBuffer, CudaBuffer), String> {
        validate_matmul_backward_lengths(grad, lhs, rhs, m, k, n, "I8xF32")?;
        let d_lhs = alloc_f32(m * k)?;
        let d_rhs = alloc_f32(n * k)?;
        let status = unsafe {
            lumen_cuda_matmul_backward_i8_f32_device(
                grad.handle(),
                lhs.handle(),
                lhs_scale,
                rhs.handle(),
                d_lhs.handle(),
                d_rhs.handle(),
                m,
                k,
                n,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok((d_lhs, d_rhs))
    }

    fn validate_matmul_backward_lengths(
        grad: &CudaBuffer,
        lhs: &CudaBuffer,
        rhs: &CudaBuffer,
        m: usize,
        k: usize,
        n: usize,
        dtype: &str,
    ) -> Result<(), String> {
        if grad.len() != m * n {
            return Err(format!(
                "CUDA {dtype} matmul backward grad length mismatch: expected {}, got {}",
                m * n,
                grad.len()
            ));
        }
        if lhs.len() != m * k {
            return Err(format!(
                "CUDA {dtype} matmul backward lhs length mismatch: expected {}, got {}",
                m * k,
                lhs.len()
            ));
        }
        if rhs.len() != n * k {
            return Err(format!(
                "CUDA {dtype} matmul backward rhs length mismatch: expected {}, got {}",
                n * k,
                rhs.len()
            ));
        }
        Ok(())
    }

    pub fn batch_matmul_backward_f32_no_host(
        grad: &CudaBuffer,
        lhs: &CudaBuffer,
        rhs: &CudaBuffer,
        batch_count: usize,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<(CudaBuffer, CudaBuffer), String> {
        if grad.len() != batch_count * m * n {
            return Err(format!(
                "CUDA batch_matmul backward grad length mismatch: expected {}, got {}",
                batch_count * m * n,
                grad.len()
            ));
        }
        if lhs.len() != batch_count * m * k {
            return Err(format!(
                "CUDA batch_matmul backward lhs length mismatch: expected {}, got {}",
                batch_count * m * k,
                lhs.len()
            ));
        }
        if rhs.len() != batch_count * k * n {
            return Err(format!(
                "CUDA batch_matmul backward rhs length mismatch: expected {}, got {}",
                batch_count * k * n,
                rhs.len()
            ));
        }
        let d_lhs = alloc_f32(batch_count * m * k)?;
        let d_rhs = alloc_f32(batch_count * k * n)?;
        let status = unsafe {
            lumen_cuda_batch_matmul_backward_f32_device(
                grad.handle(),
                lhs.handle(),
                rhs.handle(),
                d_lhs.handle(),
                d_rhs.handle(),
                batch_count,
                m,
                k,
                n,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok((d_lhs, d_rhs))
    }

    pub fn batch_matmul_backward_bf16_no_host(
        grad: &CudaBuffer,
        lhs: &CudaBuffer,
        rhs: &CudaBuffer,
        batch_count: usize,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<(CudaBuffer, CudaBuffer), String> {
        validate_batch_matmul_backward_lengths(grad, lhs, rhs, batch_count, m, k, n, "BF16")?;
        let d_lhs = alloc_f32(batch_count * m * k)?;
        let d_rhs = alloc_f32(batch_count * k * n)?;
        let status = unsafe {
            lumen_cuda_batch_matmul_backward_bf16_device(
                grad.handle(),
                lhs.handle(),
                rhs.handle(),
                d_lhs.handle(),
                d_rhs.handle(),
                batch_count,
                m,
                k,
                n,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok((d_lhs, d_rhs))
    }

    pub fn batch_matmul_backward_f16_no_host(
        grad: &CudaBuffer,
        lhs: &CudaBuffer,
        rhs: &CudaBuffer,
        batch_count: usize,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<(CudaBuffer, CudaBuffer), String> {
        validate_batch_matmul_backward_lengths(grad, lhs, rhs, batch_count, m, k, n, "F16")?;
        let d_lhs = alloc_f32(batch_count * m * k)?;
        let d_rhs = alloc_f32(batch_count * k * n)?;
        let status = unsafe {
            lumen_cuda_batch_matmul_backward_f16_device(
                grad.handle(),
                lhs.handle(),
                rhs.handle(),
                d_lhs.handle(),
                d_rhs.handle(),
                batch_count,
                m,
                k,
                n,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok((d_lhs, d_rhs))
    }

    pub fn batch_matmul_backward_bf16_i8_no_host(
        grad: &CudaBuffer,
        lhs: &CudaBuffer,
        rhs: &CudaBuffer,
        rhs_scale: f32,
        batch_count: usize,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<(CudaBuffer, CudaBuffer), String> {
        validate_batch_matmul_backward_lengths(grad, lhs, rhs, batch_count, m, k, n, "BF16xI8")?;
        let d_lhs = alloc_f32(batch_count * m * k)?;
        let d_rhs = alloc_f32(batch_count * k * n)?;
        let status = unsafe {
            lumen_cuda_batch_matmul_backward_bf16_i8_device(
                grad.handle(),
                lhs.handle(),
                rhs.handle(),
                rhs_scale,
                d_lhs.handle(),
                d_rhs.handle(),
                batch_count,
                m,
                k,
                n,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok((d_lhs, d_rhs))
    }

    pub fn batch_matmul_backward_f16_i8_no_host(
        grad: &CudaBuffer,
        lhs: &CudaBuffer,
        rhs: &CudaBuffer,
        rhs_scale: f32,
        batch_count: usize,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<(CudaBuffer, CudaBuffer), String> {
        validate_batch_matmul_backward_lengths(grad, lhs, rhs, batch_count, m, k, n, "F16xI8")?;
        let d_lhs = alloc_f32(batch_count * m * k)?;
        let d_rhs = alloc_f32(batch_count * k * n)?;
        let status = unsafe {
            lumen_cuda_batch_matmul_backward_f16_i8_device(
                grad.handle(),
                lhs.handle(),
                rhs.handle(),
                rhs_scale,
                d_lhs.handle(),
                d_rhs.handle(),
                batch_count,
                m,
                k,
                n,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok((d_lhs, d_rhs))
    }

    pub fn batch_matmul_backward_f32_i8_no_host(
        grad: &CudaBuffer,
        lhs: &CudaBuffer,
        rhs: &CudaBuffer,
        rhs_scale: f32,
        batch_count: usize,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<(CudaBuffer, CudaBuffer), String> {
        validate_batch_matmul_backward_lengths(grad, lhs, rhs, batch_count, m, k, n, "F32xI8")?;
        let d_lhs = alloc_f32(batch_count * m * k)?;
        let d_rhs = alloc_f32(batch_count * k * n)?;
        let status = unsafe {
            lumen_cuda_batch_matmul_backward_f32_i8_device(
                grad.handle(),
                lhs.handle(),
                rhs.handle(),
                rhs_scale,
                d_lhs.handle(),
                d_rhs.handle(),
                batch_count,
                m,
                k,
                n,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok((d_lhs, d_rhs))
    }

    pub fn batch_matmul_backward_i8_bf16_no_host(
        grad: &CudaBuffer,
        lhs: &CudaBuffer,
        lhs_scale: f32,
        rhs: &CudaBuffer,
        batch_count: usize,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<(CudaBuffer, CudaBuffer), String> {
        validate_batch_matmul_backward_lengths(grad, lhs, rhs, batch_count, m, k, n, "I8xBF16")?;
        let d_lhs = alloc_f32(batch_count * m * k)?;
        let d_rhs = alloc_f32(batch_count * k * n)?;
        let status = unsafe {
            lumen_cuda_batch_matmul_backward_i8_bf16_device(
                grad.handle(),
                lhs.handle(),
                lhs_scale,
                rhs.handle(),
                d_lhs.handle(),
                d_rhs.handle(),
                batch_count,
                m,
                k,
                n,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok((d_lhs, d_rhs))
    }

    pub fn batch_matmul_backward_i8_f16_no_host(
        grad: &CudaBuffer,
        lhs: &CudaBuffer,
        lhs_scale: f32,
        rhs: &CudaBuffer,
        batch_count: usize,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<(CudaBuffer, CudaBuffer), String> {
        validate_batch_matmul_backward_lengths(grad, lhs, rhs, batch_count, m, k, n, "I8xF16")?;
        let d_lhs = alloc_f32(batch_count * m * k)?;
        let d_rhs = alloc_f32(batch_count * k * n)?;
        let status = unsafe {
            lumen_cuda_batch_matmul_backward_i8_f16_device(
                grad.handle(),
                lhs.handle(),
                lhs_scale,
                rhs.handle(),
                d_lhs.handle(),
                d_rhs.handle(),
                batch_count,
                m,
                k,
                n,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok((d_lhs, d_rhs))
    }

    pub fn batch_matmul_backward_i8_f32_no_host(
        grad: &CudaBuffer,
        lhs: &CudaBuffer,
        lhs_scale: f32,
        rhs: &CudaBuffer,
        batch_count: usize,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<(CudaBuffer, CudaBuffer), String> {
        validate_batch_matmul_backward_lengths(grad, lhs, rhs, batch_count, m, k, n, "I8xF32")?;
        let d_lhs = alloc_f32(batch_count * m * k)?;
        let d_rhs = alloc_f32(batch_count * k * n)?;
        let status = unsafe {
            lumen_cuda_batch_matmul_backward_i8_f32_device(
                grad.handle(),
                lhs.handle(),
                lhs_scale,
                rhs.handle(),
                d_lhs.handle(),
                d_rhs.handle(),
                batch_count,
                m,
                k,
                n,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok((d_lhs, d_rhs))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn batch_matmul_backward_i8_no_host(
        grad: &CudaBuffer,
        lhs: &CudaBuffer,
        lhs_scale: f32,
        rhs: &CudaBuffer,
        rhs_scale: f32,
        batch_count: usize,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<(CudaBuffer, CudaBuffer), String> {
        validate_batch_matmul_backward_lengths(grad, lhs, rhs, batch_count, m, k, n, "I8")?;
        let d_lhs = alloc_f32(batch_count * m * k)?;
        let d_rhs = alloc_f32(batch_count * k * n)?;
        let status = unsafe {
            lumen_cuda_batch_matmul_backward_i8_device(
                grad.handle(),
                lhs.handle(),
                lhs_scale,
                rhs.handle(),
                rhs_scale,
                d_lhs.handle(),
                d_rhs.handle(),
                batch_count,
                m,
                k,
                n,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok((d_lhs, d_rhs))
    }

    fn validate_batch_matmul_backward_lengths(
        grad: &CudaBuffer,
        lhs: &CudaBuffer,
        rhs: &CudaBuffer,
        batch_count: usize,
        m: usize,
        k: usize,
        n: usize,
        dtype: &str,
    ) -> Result<(), String> {
        if grad.len() != batch_count * m * n {
            return Err(format!(
                "CUDA {dtype} batch_matmul backward grad length mismatch: expected {}, got {}",
                batch_count * m * n,
                grad.len()
            ));
        }
        if lhs.len() != batch_count * m * k {
            return Err(format!(
                "CUDA {dtype} batch_matmul backward lhs length mismatch: expected {}, got {}",
                batch_count * m * k,
                lhs.len()
            ));
        }
        if rhs.len() != batch_count * k * n {
            return Err(format!(
                "CUDA {dtype} batch_matmul backward rhs length mismatch: expected {}, got {}",
                batch_count * k * n,
                rhs.len()
            ));
        }
        Ok(())
    }

    pub fn unary_f32(input: &CudaBuffer, op: UnaryOp) -> Result<(CudaBuffer, Vec<f32>), String> {
        let out = unary_f32_buffer(input, op)?;
        let host = download_f32(&out)?;
        Ok((out, host))
    }

    pub fn unary_f32_buffer(input: &CudaBuffer, op: UnaryOp) -> Result<CudaBuffer, String> {
        let out = alloc_f32(input.len())?;
        let status = unsafe {
            lumen_cuda_unary_f32_device(input.handle(), out.handle(), input.len(), op as c_int)
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(out)
    }

    pub fn unary_f16_buffer(input: &CudaBuffer, op: UnaryOp) -> Result<CudaBuffer, String> {
        let out = alloc_f32(input.len())?;
        let status = unsafe {
            lumen_cuda_unary_f16_device(input.handle(), out.handle(), input.len(), op as c_int)
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(out)
    }

    pub fn unary_f16_typed_output_buffer(
        input: &CudaBuffer,
        op: UnaryOp,
    ) -> Result<CudaBuffer, String> {
        let out = alloc_storage(input.len())?;
        let status = unsafe {
            lumen_cuda_unary_f16_typed_out_device(
                input.handle(),
                out.handle(),
                input.len(),
                op as c_int,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(out)
    }

    pub fn unary_bf16_buffer(input: &CudaBuffer, op: UnaryOp) -> Result<CudaBuffer, String> {
        let out = alloc_f32(input.len())?;
        let status = unsafe {
            lumen_cuda_unary_bf16_device(input.handle(), out.handle(), input.len(), op as c_int)
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(out)
    }

    pub fn unary_bf16_typed_output_buffer(
        input: &CudaBuffer,
        op: UnaryOp,
    ) -> Result<CudaBuffer, String> {
        let out = alloc_storage(input.len())?;
        let status = unsafe {
            lumen_cuda_unary_bf16_typed_out_device(
                input.handle(),
                out.handle(),
                input.len(),
                op as c_int,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(out)
    }

    pub fn unary_i8_buffer(
        input: &CudaBuffer,
        scale: f32,
        op: UnaryOp,
    ) -> Result<CudaBuffer, String> {
        let out = alloc_f32(input.len())?;
        let status = unsafe {
            lumen_cuda_unary_i8_device(
                input.handle(),
                scale,
                out.handle(),
                input.len(),
                op as c_int,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(out)
    }

    pub fn unary_i8_relu_typed_output_buffer(input: &CudaBuffer) -> Result<CudaBuffer, String> {
        let out = alloc_storage(input.len())?;
        let status = unsafe {
            lumen_cuda_unary_i8_relu_typed_out_device(input.handle(), out.handle(), input.len())
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(out)
    }

    pub fn unary_backward_f32(
        input: &CudaBuffer,
        output: &CudaBuffer,
        grad: &CudaBuffer,
        op: UnaryOp,
    ) -> Result<(CudaBuffer, Vec<f32>), String> {
        let out = unary_backward_f32_buffer(input, output, grad, op)?;
        let host = download_f32(&out)?;
        Ok((out, host))
    }

    pub fn unary_backward_f32_buffer(
        input: &CudaBuffer,
        output: &CudaBuffer,
        grad: &CudaBuffer,
        op: UnaryOp,
    ) -> Result<CudaBuffer, String> {
        if input.len() != output.len() || input.len() != grad.len() {
            return Err(format!(
                "CUDA unary backward length mismatch: input={}, output={}, grad={}",
                input.len(),
                output.len(),
                grad.len()
            ));
        }
        let out = alloc_f32(input.len())?;
        let status = unsafe {
            lumen_cuda_unary_backward_f32_device(
                input.handle(),
                output.handle(),
                grad.handle(),
                out.handle(),
                input.len(),
                op as c_int,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(out)
    }

    pub fn unary_backward_f16_buffer(
        input: &CudaBuffer,
        output: &CudaBuffer,
        grad: &CudaBuffer,
        op: UnaryOp,
    ) -> Result<CudaBuffer, String> {
        if input.len() != output.len() || input.len() != grad.len() {
            return Err(format!(
                "CUDA unary F16 backward length mismatch: input={}, output={}, grad={}",
                input.len(),
                output.len(),
                grad.len()
            ));
        }
        let out = alloc_f32(input.len())?;
        let status = unsafe {
            lumen_cuda_unary_backward_f16_device(
                input.handle(),
                output.handle(),
                grad.handle(),
                out.handle(),
                input.len(),
                op as c_int,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(out)
    }

    pub fn unary_backward_bf16_buffer(
        input: &CudaBuffer,
        output: &CudaBuffer,
        grad: &CudaBuffer,
        op: UnaryOp,
    ) -> Result<CudaBuffer, String> {
        if input.len() != output.len() || input.len() != grad.len() {
            return Err(format!(
                "CUDA unary BF16 backward length mismatch: input={}, output={}, grad={}",
                input.len(),
                output.len(),
                grad.len()
            ));
        }
        let out = alloc_f32(input.len())?;
        let status = unsafe {
            lumen_cuda_unary_backward_bf16_device(
                input.handle(),
                output.handle(),
                grad.handle(),
                out.handle(),
                input.len(),
                op as c_int,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(out)
    }

    pub fn unary_backward_i8_buffer(
        input: &CudaBuffer,
        scale: f32,
        output: &CudaBuffer,
        grad: &CudaBuffer,
        op: UnaryOp,
    ) -> Result<CudaBuffer, String> {
        if input.len() != output.len() || input.len() != grad.len() {
            return Err(format!(
                "CUDA unary I8 backward length mismatch: input={}, output={}, grad={}",
                input.len(),
                output.len(),
                grad.len()
            ));
        }
        let out = alloc_f32(input.len())?;
        let status = unsafe {
            lumen_cuda_unary_backward_i8_device(
                input.handle(),
                scale,
                output.handle(),
                grad.handle(),
                out.handle(),
                input.len(),
                op as c_int,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(out)
    }

    pub fn binary_f32(
        lhs: &CudaBuffer,
        rhs: &CudaBuffer,
        op: BinaryOp,
    ) -> Result<(CudaBuffer, Vec<f32>), String> {
        if lhs.len() != rhs.len() {
            return Err(format!(
                "CUDA binary op length mismatch: lhs={}, rhs={}",
                lhs.len(),
                rhs.len()
            ));
        }
        let out = alloc_f32(lhs.len())?;
        let status = unsafe {
            lumen_cuda_binary_f32_device(
                lhs.handle(),
                rhs.handle(),
                out.handle(),
                lhs.len(),
                op as c_int,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        let host = download_f32(&out)?;
        Ok((out, host))
    }

    pub fn binary_f32_buffer(
        lhs: &CudaBuffer,
        rhs: &CudaBuffer,
        op: BinaryOp,
    ) -> Result<CudaBuffer, String> {
        if lhs.len() != rhs.len() {
            return Err(format!(
                "CUDA binary op length mismatch: lhs={}, rhs={}",
                lhs.len(),
                rhs.len()
            ));
        }
        let out = alloc_f32(lhs.len())?;
        let status = unsafe {
            lumen_cuda_binary_f32_device(
                lhs.handle(),
                rhs.handle(),
                out.handle(),
                lhs.len(),
                op as c_int,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(out)
    }

    pub fn binary_typed_buffer_no_host(
        lhs: &CudaBuffer,
        lhs_dtype: DType,
        lhs_scale: Option<f32>,
        rhs: &CudaBuffer,
        rhs_dtype: DType,
        rhs_scale: Option<f32>,
        op: BinaryOp,
    ) -> Result<CudaBuffer, String> {
        if lhs.len() != rhs.len() {
            return Err(format!(
                "CUDA typed mixed binary length mismatch: lhs={}, rhs={}",
                lhs.len(),
                rhs.len()
            ));
        }
        let out = alloc_f32(lhs.len())?;
        let status = unsafe {
            lumen_cuda_binary_typed_device(
                lhs.handle(),
                dtype_tag(lhs_dtype)?,
                lhs_scale.unwrap_or(1.0),
                rhs.handle(),
                dtype_tag(rhs_dtype)?,
                rhs_scale.unwrap_or(1.0),
                out.handle(),
                lhs.len(),
                op as c_int,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(out)
    }

    pub fn binary_lowp_typed_output_buffer_no_host(
        lhs: &CudaBuffer,
        rhs: &CudaBuffer,
        dtype: DType,
        op: BinaryOp,
    ) -> Result<CudaBuffer, String> {
        if lhs.len() != rhs.len() {
            return Err(format!(
                "CUDA typed-out lowp binary length mismatch: lhs={}, rhs={}",
                lhs.len(),
                rhs.len()
            ));
        }
        if !matches!(dtype, DType::F16 | DType::BF16) {
            return Err("CUDA typed-out lowp binary supports only F16/BF16".to_string());
        }
        let out = alloc_storage(lhs.len())?;
        let status = unsafe {
            lumen_cuda_binary_lowp_typed_out_device(
                lhs.handle(),
                rhs.handle(),
                out.handle(),
                lhs.len(),
                dtype_tag(dtype)?,
                op as c_int,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn binary_typed_lastdim_buffer_no_host(
        lhs: &CudaBuffer,
        lhs_dtype: DType,
        lhs_scale: Option<f32>,
        rhs: &CudaBuffer,
        rhs_dtype: DType,
        rhs_scale: Option<f32>,
        out_len: usize,
        last_dim: usize,
        vector_on_rhs: bool,
        op: BinaryOp,
    ) -> Result<CudaBuffer, String> {
        if last_dim == 0 || out_len == 0 {
            return Err("CUDA typed mixed lastdim binary dimensions must be non-zero".to_string());
        }
        let vector_len = if vector_on_rhs { rhs.len() } else { lhs.len() };
        let full_len = if vector_on_rhs { lhs.len() } else { rhs.len() };
        if vector_len != last_dim || full_len != out_len {
            return Err(format!(
                "CUDA typed mixed lastdim binary length mismatch: lhs={}, rhs={}, out_len={}, last_dim={}, vector_on_rhs={}",
                lhs.len(),
                rhs.len(),
                out_len,
                last_dim,
                vector_on_rhs
            ));
        }
        let out = alloc_f32(out_len)?;
        let status = unsafe {
            lumen_cuda_binary_typed_lastdim_device(
                lhs.handle(),
                dtype_tag(lhs_dtype)?,
                lhs_scale.unwrap_or(1.0),
                rhs.handle(),
                dtype_tag(rhs_dtype)?,
                rhs_scale.unwrap_or(1.0),
                out.handle(),
                out_len,
                last_dim,
                if vector_on_rhs { 1 } else { 0 },
                op as c_int,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(out)
    }

    pub fn binary_lowp_typed_lastdim_output_buffer_no_host(
        lhs: &CudaBuffer,
        rhs: &CudaBuffer,
        out_len: usize,
        last_dim: usize,
        vector_on_rhs: bool,
        dtype: DType,
        op: BinaryOp,
    ) -> Result<CudaBuffer, String> {
        if last_dim == 0 || out_len == 0 {
            return Err(
                "CUDA typed-out lowp lastdim binary dimensions must be non-zero".to_string(),
            );
        }
        if !matches!(dtype, DType::F16 | DType::BF16) {
            return Err("CUDA typed-out lowp lastdim binary supports only F16/BF16".to_string());
        }
        let expected_lhs = if vector_on_rhs { out_len } else { last_dim };
        let expected_rhs = if vector_on_rhs { last_dim } else { out_len };
        if lhs.len() != expected_lhs || rhs.len() != expected_rhs {
            return Err(format!(
                "CUDA typed-out lowp lastdim binary length mismatch: expected lhs={}, rhs={}, got lhs={}, rhs={}",
                expected_lhs,
                expected_rhs,
                lhs.len(),
                rhs.len()
            ));
        }
        let out = alloc_storage(out_len)?;
        let status = unsafe {
            lumen_cuda_binary_lowp_typed_lastdim_out_device(
                lhs.handle(),
                rhs.handle(),
                out.handle(),
                out_len,
                last_dim,
                if vector_on_rhs { 1 } else { 0 },
                dtype_tag(dtype)?,
                op as c_int,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn binary_typed_row_scalar_buffer_no_host(
        lhs: &CudaBuffer,
        lhs_dtype: DType,
        lhs_scale: Option<f32>,
        rhs: &CudaBuffer,
        rhs_dtype: DType,
        rhs_scale: Option<f32>,
        rows: usize,
        last_dim: usize,
        scalar_on_rhs: bool,
        op: BinaryOp,
    ) -> Result<CudaBuffer, String> {
        let out_len = rows.checked_mul(last_dim).ok_or_else(|| {
            "CUDA typed mixed row-scalar binary output length overflow".to_string()
        })?;
        let expected_lhs = if scalar_on_rhs { out_len } else { rows };
        let expected_rhs = if scalar_on_rhs { rows } else { out_len };
        if rows == 0 || last_dim == 0 || lhs.len() != expected_lhs || rhs.len() != expected_rhs {
            return Err(format!(
                "CUDA typed mixed row-scalar binary length mismatch: expected lhs={}, rhs={}, got lhs={}, rhs={}",
                expected_lhs,
                expected_rhs,
                lhs.len(),
                rhs.len()
            ));
        }
        let out = alloc_f32(out_len)?;
        let status = unsafe {
            lumen_cuda_binary_typed_row_scalar_device(
                lhs.handle(),
                dtype_tag(lhs_dtype)?,
                lhs_scale.unwrap_or(1.0),
                rhs.handle(),
                dtype_tag(rhs_dtype)?,
                rhs_scale.unwrap_or(1.0),
                out.handle(),
                rows,
                last_dim,
                scalar_on_rhs as c_int,
                op as c_int,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn binary_lowp_typed_row_scalar_output_buffer_no_host(
        lhs: &CudaBuffer,
        rhs: &CudaBuffer,
        rows: usize,
        last_dim: usize,
        scalar_on_rhs: bool,
        dtype: DType,
        op: BinaryOp,
    ) -> Result<CudaBuffer, String> {
        let out_len = rows
            .checked_mul(last_dim)
            .ok_or_else(|| "CUDA typed-out lowp row-scalar output length overflow".to_string())?;
        let expected_lhs = if scalar_on_rhs { out_len } else { rows };
        let expected_rhs = if scalar_on_rhs { rows } else { out_len };
        if rows == 0 || last_dim == 0 || lhs.len() != expected_lhs || rhs.len() != expected_rhs {
            return Err(format!(
                "CUDA typed-out lowp row-scalar length mismatch: expected lhs={}, rhs={}, got lhs={}, rhs={}",
                expected_lhs,
                expected_rhs,
                lhs.len(),
                rhs.len()
            ));
        }
        if !matches!(dtype, DType::F16 | DType::BF16) {
            return Err("CUDA typed-out lowp row-scalar supports only F16/BF16".to_string());
        }
        let out = alloc_storage(out_len)?;
        let status = unsafe {
            lumen_cuda_binary_lowp_typed_row_scalar_out_device(
                lhs.handle(),
                rhs.handle(),
                out.handle(),
                rows,
                last_dim,
                if scalar_on_rhs { 1 } else { 0 },
                dtype_tag(dtype)?,
                op as c_int,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn binary_i8_typed_row_scalar_output_buffer_no_host(
        lhs: &CudaBuffer,
        lhs_scale: f32,
        rhs: &CudaBuffer,
        rhs_scale: f32,
        rows: usize,
        last_dim: usize,
        scalar_on_rhs: bool,
        op: BinaryOp,
    ) -> Result<(CudaBuffer, f32), String> {
        let out_len = rows
            .checked_mul(last_dim)
            .ok_or_else(|| "CUDA typed-out I8 row-scalar output length overflow".to_string())?;
        let expected_lhs = if scalar_on_rhs { out_len } else { rows };
        let expected_rhs = if scalar_on_rhs { rows } else { out_len };
        if rows == 0 || last_dim == 0 || lhs.len() != expected_lhs || rhs.len() != expected_rhs {
            return Err(format!(
                "CUDA typed-out I8 row-scalar length mismatch: expected lhs={}, rhs={}, got lhs={}, rhs={}",
                expected_lhs,
                expected_rhs,
                lhs.len(),
                rhs.len()
            ));
        }
        let out = alloc_storage(out_len)?;
        let mut out_scale = 1.0f32;
        let status = unsafe {
            lumen_cuda_binary_i8_typed_row_scalar_out_device(
                lhs.handle(),
                rhs.handle(),
                lhs_scale,
                rhs_scale,
                out.handle(),
                rows,
                last_dim,
                if scalar_on_rhs { 1 } else { 0 },
                op as c_int,
                &mut out_scale as *mut f32,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok((out, out_scale))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn binary_typed_broadcast_buffer_no_host(
        lhs: &CudaBuffer,
        lhs_dtype: DType,
        lhs_scale: Option<f32>,
        rhs: &CudaBuffer,
        rhs_dtype: DType,
        rhs_scale: Option<f32>,
        lhs_shape: &[usize],
        rhs_shape: &[usize],
        out_shape: &[usize],
        op: BinaryOp,
    ) -> Result<CudaBuffer, String> {
        let len = out_shape
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
            .ok_or_else(|| "CUDA typed mixed broadcast output length overflow".to_string())?;
        let lhs_len = lhs_shape
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
            .ok_or_else(|| "CUDA typed mixed broadcast lhs length overflow".to_string())?;
        let rhs_len = rhs_shape
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
            .ok_or_else(|| "CUDA typed mixed broadcast rhs length overflow".to_string())?;
        if lhs.len() != lhs_len || rhs.len() != rhs_len {
            return Err(format!(
                "CUDA typed mixed broadcast input length mismatch: expected lhs={}, rhs={}, got lhs={}, rhs={}",
                lhs_len,
                rhs_len,
                lhs.len(),
                rhs.len()
            ));
        }
        let (lhs_aligned, rhs_aligned, out_strides, lhs_strides, rhs_strides) =
            aligned_broadcast_metadata(lhs_shape, rhs_shape, out_shape)?;
        let out = alloc_f32(len)?;
        let status = unsafe {
            lumen_cuda_binary_typed_broadcast_device(
                lhs.handle(),
                dtype_tag(lhs_dtype)?,
                lhs_scale.unwrap_or(1.0),
                rhs.handle(),
                dtype_tag(rhs_dtype)?,
                rhs_scale.unwrap_or(1.0),
                out.handle(),
                out_shape.len(),
                out_strides.as_ptr(),
                lhs_aligned.as_ptr(),
                lhs_strides.as_ptr(),
                rhs_aligned.as_ptr(),
                rhs_strides.as_ptr(),
                len,
                op as c_int,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn binary_lowp_typed_broadcast_output_buffer_no_host(
        lhs: &CudaBuffer,
        rhs: &CudaBuffer,
        lhs_shape: &[usize],
        rhs_shape: &[usize],
        out_shape: &[usize],
        dtype: DType,
        op: BinaryOp,
    ) -> Result<CudaBuffer, String> {
        let len = out_shape
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
            .ok_or_else(|| "CUDA lowp typed-out broadcast output length overflow".to_string())?;
        let lhs_len = lhs_shape
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
            .ok_or_else(|| "CUDA lowp typed-out broadcast lhs length overflow".to_string())?;
        let rhs_len = rhs_shape
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
            .ok_or_else(|| "CUDA lowp typed-out broadcast rhs length overflow".to_string())?;
        if lhs.len() != lhs_len || rhs.len() != rhs_len {
            return Err(format!(
                "CUDA lowp typed-out broadcast input length mismatch: expected lhs={}, rhs={}, got lhs={}, rhs={}",
                lhs_len,
                rhs_len,
                lhs.len(),
                rhs.len()
            ));
        }
        if !matches!(dtype, DType::F16 | DType::BF16) {
            return Err("CUDA lowp typed-out broadcast supports only F16/BF16".to_string());
        }
        let (lhs_aligned, rhs_aligned, out_strides, lhs_strides, rhs_strides) =
            aligned_broadcast_metadata(lhs_shape, rhs_shape, out_shape)?;
        let out = alloc_storage(len)?;
        let status = unsafe {
            lumen_cuda_binary_lowp_typed_broadcast_out_device(
                lhs.handle(),
                rhs.handle(),
                out.handle(),
                out_shape.len(),
                out_strides.as_ptr(),
                lhs_aligned.as_ptr(),
                lhs_strides.as_ptr(),
                rhs_aligned.as_ptr(),
                rhs_strides.as_ptr(),
                len,
                dtype_tag(dtype)?,
                op as c_int,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn binary_i8_typed_broadcast_output_buffer_no_host(
        lhs: &CudaBuffer,
        lhs_scale: f32,
        rhs: &CudaBuffer,
        rhs_scale: f32,
        lhs_shape: &[usize],
        rhs_shape: &[usize],
        out_shape: &[usize],
        op: BinaryOp,
    ) -> Result<(CudaBuffer, f32), String> {
        let len = out_shape
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
            .ok_or_else(|| "CUDA I8 typed-out broadcast output length overflow".to_string())?;
        let lhs_len = lhs_shape
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
            .ok_or_else(|| "CUDA I8 typed-out broadcast lhs length overflow".to_string())?;
        let rhs_len = rhs_shape
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
            .ok_or_else(|| "CUDA I8 typed-out broadcast rhs length overflow".to_string())?;
        if lhs.len() != lhs_len || rhs.len() != rhs_len {
            return Err(format!(
                "CUDA I8 typed-out broadcast input length mismatch: expected lhs={}, rhs={}, got lhs={}, rhs={}",
                lhs_len,
                rhs_len,
                lhs.len(),
                rhs.len()
            ));
        }
        let (lhs_aligned, rhs_aligned, out_strides, lhs_strides, rhs_strides) =
            aligned_broadcast_metadata(lhs_shape, rhs_shape, out_shape)?;
        let out = alloc_storage(len)?;
        let mut out_scale = 1.0f32;
        let status = unsafe {
            lumen_cuda_binary_i8_typed_broadcast_out_device(
                lhs.handle(),
                rhs.handle(),
                lhs_scale,
                rhs_scale,
                out.handle(),
                out_shape.len(),
                out_strides.as_ptr(),
                lhs_aligned.as_ptr(),
                lhs_strides.as_ptr(),
                rhs_aligned.as_ptr(),
                rhs_strides.as_ptr(),
                len,
                op as c_int,
                &mut out_scale as *mut f32,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok((out, out_scale))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn binary_typed_b1d_1h1_buffer_no_host(
        lhs: &CudaBuffer,
        lhs_dtype: DType,
        lhs_scale: Option<f32>,
        rhs: &CudaBuffer,
        rhs_dtype: DType,
        rhs_scale: Option<f32>,
        batch: usize,
        heads: usize,
        dim: usize,
        b1d_on_lhs: bool,
        op: BinaryOp,
    ) -> Result<CudaBuffer, String> {
        let b1d_len = batch
            .checked_mul(dim)
            .ok_or_else(|| "CUDA typed mixed b1d/1h1 b1d length overflow".to_string())?;
        let out_len = b1d_len
            .checked_mul(heads)
            .ok_or_else(|| "CUDA typed mixed b1d/1h1 output length overflow".to_string())?;
        let expected_lhs = if b1d_on_lhs { b1d_len } else { heads };
        let expected_rhs = if b1d_on_lhs { heads } else { b1d_len };
        if lhs.len() != expected_lhs || rhs.len() != expected_rhs {
            return Err(format!(
                "CUDA typed mixed b1d/1h1 length mismatch: expected lhs={}, rhs={}, got lhs={}, rhs={}",
                expected_lhs,
                expected_rhs,
                lhs.len(),
                rhs.len()
            ));
        }
        let out = alloc_f32(out_len)?;
        let status = unsafe {
            lumen_cuda_binary_typed_b1d_1h1_device(
                lhs.handle(),
                dtype_tag(lhs_dtype)?,
                lhs_scale.unwrap_or(1.0),
                rhs.handle(),
                dtype_tag(rhs_dtype)?,
                rhs_scale.unwrap_or(1.0),
                out.handle(),
                batch,
                heads,
                dim,
                b1d_on_lhs as c_int,
                op as c_int,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn binary_lowp_typed_b1d_1h1_output_buffer_no_host(
        lhs: &CudaBuffer,
        rhs: &CudaBuffer,
        batch: usize,
        heads: usize,
        dim: usize,
        b1d_on_lhs: bool,
        dtype: DType,
        op: BinaryOp,
    ) -> Result<CudaBuffer, String> {
        let b1d_len = batch
            .checked_mul(dim)
            .ok_or_else(|| "CUDA typed-out lowp b1d/1h1 b1d length overflow".to_string())?;
        let out_len = b1d_len
            .checked_mul(heads)
            .ok_or_else(|| "CUDA typed-out lowp b1d/1h1 output length overflow".to_string())?;
        let expected_lhs = if b1d_on_lhs { b1d_len } else { heads };
        let expected_rhs = if b1d_on_lhs { heads } else { b1d_len };
        if batch == 0
            || heads == 0
            || dim == 0
            || lhs.len() != expected_lhs
            || rhs.len() != expected_rhs
        {
            return Err(format!(
                "CUDA typed-out lowp b1d/1h1 length mismatch: expected lhs={}, rhs={}, got lhs={}, rhs={}",
                expected_lhs,
                expected_rhs,
                lhs.len(),
                rhs.len()
            ));
        }
        if !matches!(dtype, DType::F16 | DType::BF16) {
            return Err("CUDA typed-out lowp b1d/1h1 supports only F16/BF16".to_string());
        }
        let out = alloc_storage(out_len)?;
        let status = unsafe {
            lumen_cuda_binary_lowp_typed_b1d_1h1_out_device(
                lhs.handle(),
                rhs.handle(),
                out.handle(),
                batch,
                heads,
                dim,
                if b1d_on_lhs { 1 } else { 0 },
                dtype_tag(dtype)?,
                op as c_int,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn binary_i8_typed_b1d_1h1_output_buffer_no_host(
        lhs: &CudaBuffer,
        lhs_scale: f32,
        rhs: &CudaBuffer,
        rhs_scale: f32,
        batch: usize,
        heads: usize,
        dim: usize,
        b1d_on_lhs: bool,
        op: BinaryOp,
    ) -> Result<(CudaBuffer, f32), String> {
        let b1d_len = batch
            .checked_mul(dim)
            .ok_or_else(|| "CUDA typed-out I8 b1d/1h1 b1d length overflow".to_string())?;
        let out_len = b1d_len
            .checked_mul(heads)
            .ok_or_else(|| "CUDA typed-out I8 b1d/1h1 output length overflow".to_string())?;
        let expected_lhs = if b1d_on_lhs { b1d_len } else { heads };
        let expected_rhs = if b1d_on_lhs { heads } else { b1d_len };
        if batch == 0
            || heads == 0
            || dim == 0
            || lhs.len() != expected_lhs
            || rhs.len() != expected_rhs
        {
            return Err(format!(
                "CUDA typed-out I8 b1d/1h1 length mismatch: expected lhs={}, rhs={}, got lhs={}, rhs={}",
                expected_lhs,
                expected_rhs,
                lhs.len(),
                rhs.len()
            ));
        }
        let out = alloc_storage(out_len)?;
        let mut out_scale = 1.0f32;
        let status = unsafe {
            lumen_cuda_binary_i8_typed_b1d_1h1_out_device(
                lhs.handle(),
                rhs.handle(),
                lhs_scale,
                rhs_scale,
                out.handle(),
                batch,
                heads,
                dim,
                if b1d_on_lhs { 1 } else { 0 },
                op as c_int,
                &mut out_scale as *mut f32,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok((out, out_scale))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn binary_typed_b1d_1hd_buffer_no_host(
        lhs: &CudaBuffer,
        lhs_dtype: DType,
        lhs_scale: Option<f32>,
        rhs: &CudaBuffer,
        rhs_dtype: DType,
        rhs_scale: Option<f32>,
        batch: usize,
        heads: usize,
        dim: usize,
        b1d_on_lhs: bool,
        op: BinaryOp,
    ) -> Result<CudaBuffer, String> {
        let b1d_len = batch
            .checked_mul(dim)
            .ok_or_else(|| "CUDA typed mixed b1d/1hd b1d length overflow".to_string())?;
        let hd_len = heads
            .checked_mul(dim)
            .ok_or_else(|| "CUDA typed mixed b1d/1hd hd length overflow".to_string())?;
        let out_len = b1d_len
            .checked_mul(heads)
            .ok_or_else(|| "CUDA typed mixed b1d/1hd output length overflow".to_string())?;
        let expected_lhs = if b1d_on_lhs { b1d_len } else { hd_len };
        let expected_rhs = if b1d_on_lhs { hd_len } else { b1d_len };
        if lhs.len() != expected_lhs || rhs.len() != expected_rhs {
            return Err(format!(
                "CUDA typed mixed b1d/1hd length mismatch: expected lhs={}, rhs={}, got lhs={}, rhs={}",
                expected_lhs,
                expected_rhs,
                lhs.len(),
                rhs.len()
            ));
        }
        let out = alloc_f32(out_len)?;
        let status = unsafe {
            lumen_cuda_binary_typed_b1d_1hd_device(
                lhs.handle(),
                dtype_tag(lhs_dtype)?,
                lhs_scale.unwrap_or(1.0),
                rhs.handle(),
                dtype_tag(rhs_dtype)?,
                rhs_scale.unwrap_or(1.0),
                out.handle(),
                batch,
                heads,
                dim,
                b1d_on_lhs as c_int,
                op as c_int,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn binary_lowp_typed_b1d_1hd_output_buffer_no_host(
        lhs: &CudaBuffer,
        rhs: &CudaBuffer,
        batch: usize,
        heads: usize,
        dim: usize,
        b1d_on_lhs: bool,
        dtype: DType,
        op: BinaryOp,
    ) -> Result<CudaBuffer, String> {
        let b1d_len = batch
            .checked_mul(dim)
            .ok_or_else(|| "CUDA typed-out lowp b1d/1hd b1d length overflow".to_string())?;
        let hd_len = heads
            .checked_mul(dim)
            .ok_or_else(|| "CUDA typed-out lowp b1d/1hd hd length overflow".to_string())?;
        let out_len = b1d_len
            .checked_mul(heads)
            .ok_or_else(|| "CUDA typed-out lowp b1d/1hd output length overflow".to_string())?;
        let expected_lhs = if b1d_on_lhs { b1d_len } else { hd_len };
        let expected_rhs = if b1d_on_lhs { hd_len } else { b1d_len };
        if batch == 0
            || heads == 0
            || dim == 0
            || lhs.len() != expected_lhs
            || rhs.len() != expected_rhs
        {
            return Err(format!(
                "CUDA typed-out lowp b1d/1hd length mismatch: expected lhs={}, rhs={}, got lhs={}, rhs={}",
                expected_lhs,
                expected_rhs,
                lhs.len(),
                rhs.len()
            ));
        }
        if !matches!(dtype, DType::F16 | DType::BF16) {
            return Err("CUDA typed-out lowp b1d/1hd supports only F16/BF16".to_string());
        }
        let out = alloc_storage(out_len)?;
        let status = unsafe {
            lumen_cuda_binary_lowp_typed_b1d_1hd_out_device(
                lhs.handle(),
                rhs.handle(),
                out.handle(),
                batch,
                heads,
                dim,
                if b1d_on_lhs { 1 } else { 0 },
                dtype_tag(dtype)?,
                op as c_int,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn binary_i8_typed_b1d_1hd_output_buffer_no_host(
        lhs: &CudaBuffer,
        lhs_scale: f32,
        rhs: &CudaBuffer,
        rhs_scale: f32,
        batch: usize,
        heads: usize,
        dim: usize,
        b1d_on_lhs: bool,
        op: BinaryOp,
    ) -> Result<(CudaBuffer, f32), String> {
        let b1d_len = batch
            .checked_mul(dim)
            .ok_or_else(|| "CUDA typed-out I8 b1d/1hd b1d length overflow".to_string())?;
        let hd_len = heads
            .checked_mul(dim)
            .ok_or_else(|| "CUDA typed-out I8 b1d/1hd hd length overflow".to_string())?;
        let out_len = b1d_len
            .checked_mul(heads)
            .ok_or_else(|| "CUDA typed-out I8 b1d/1hd output length overflow".to_string())?;
        let expected_lhs = if b1d_on_lhs { b1d_len } else { hd_len };
        let expected_rhs = if b1d_on_lhs { hd_len } else { b1d_len };
        if batch == 0
            || heads == 0
            || dim == 0
            || lhs.len() != expected_lhs
            || rhs.len() != expected_rhs
        {
            return Err(format!(
                "CUDA typed-out I8 b1d/1hd length mismatch: expected lhs={}, rhs={}, got lhs={}, rhs={}",
                expected_lhs,
                expected_rhs,
                lhs.len(),
                rhs.len()
            ));
        }
        let out = alloc_storage(out_len)?;
        let mut out_scale = 1.0f32;
        let status = unsafe {
            lumen_cuda_binary_i8_typed_b1d_1hd_out_device(
                lhs.handle(),
                rhs.handle(),
                lhs_scale,
                rhs_scale,
                out.handle(),
                batch,
                heads,
                dim,
                if b1d_on_lhs { 1 } else { 0 },
                op as c_int,
                &mut out_scale as *mut f32,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok((out, out_scale))
    }

    pub fn binary_f16_host_no_host(
        lhs: &[u16],
        rhs: &[u16],
        op: BinaryOp,
    ) -> Result<CudaBuffer, String> {
        if lhs.len() != rhs.len() {
            return Err(format!(
                "CUDA F16 binary length mismatch: lhs={}, rhs={}",
                lhs.len(),
                rhs.len()
            ));
        }
        let out = alloc_f32(lhs.len())?;
        let status = unsafe {
            lumen_cuda_binary_f16_host_device(
                lhs.as_ptr(),
                rhs.as_ptr(),
                out.handle(),
                lhs.len(),
                op as c_int,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(out)
    }

    pub fn binary_f16_buffer_no_host(
        lhs: &CudaBuffer,
        rhs: &CudaBuffer,
        op: BinaryOp,
    ) -> Result<CudaBuffer, String> {
        if lhs.len() != rhs.len() {
            return Err(format!(
                "CUDA resident F16 binary length mismatch: lhs={}, rhs={}",
                lhs.len(),
                rhs.len()
            ));
        }
        let out = alloc_f32(lhs.len())?;
        let status = unsafe {
            lumen_cuda_binary_f16_device(
                lhs.handle(),
                rhs.handle(),
                out.handle(),
                lhs.len(),
                op as c_int,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(out)
    }

    pub fn binary_f16_lastdim_buffer_no_host(
        lhs: &CudaBuffer,
        rhs: &CudaBuffer,
        out_len: usize,
        last_dim: usize,
        vector_on_rhs: bool,
        op: BinaryOp,
    ) -> Result<CudaBuffer, String> {
        if last_dim == 0 {
            return Err("CUDA F16 row-broadcast last_dim must be greater than zero".to_string());
        }
        let expected_lhs = if vector_on_rhs { out_len } else { last_dim };
        let expected_rhs = if vector_on_rhs { last_dim } else { out_len };
        if lhs.len() != expected_lhs || rhs.len() != expected_rhs {
            return Err(format!(
                "CUDA F16 row-broadcast length mismatch: expected lhs={}, rhs={}, got lhs={}, rhs={}",
                expected_lhs,
                expected_rhs,
                lhs.len(),
                rhs.len()
            ));
        }
        let out = alloc_f32(out_len)?;
        let status = unsafe {
            lumen_cuda_binary_f16_lastdim_device(
                lhs.handle(),
                rhs.handle(),
                out.handle(),
                out_len,
                last_dim,
                vector_on_rhs as c_int,
                op as c_int,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(out)
    }

    pub fn binary_bf16_host_no_host(
        lhs: &[u16],
        rhs: &[u16],
        op: BinaryOp,
    ) -> Result<CudaBuffer, String> {
        if lhs.len() != rhs.len() {
            return Err(format!(
                "CUDA BF16 binary length mismatch: lhs={}, rhs={}",
                lhs.len(),
                rhs.len()
            ));
        }
        let out = alloc_f32(lhs.len())?;
        let status = unsafe {
            lumen_cuda_binary_bf16_host_device(
                lhs.as_ptr(),
                rhs.as_ptr(),
                out.handle(),
                lhs.len(),
                op as c_int,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(out)
    }

    pub fn binary_bf16_buffer_no_host(
        lhs: &CudaBuffer,
        rhs: &CudaBuffer,
        op: BinaryOp,
    ) -> Result<CudaBuffer, String> {
        if lhs.len() != rhs.len() {
            return Err(format!(
                "CUDA resident BF16 binary length mismatch: lhs={}, rhs={}",
                lhs.len(),
                rhs.len()
            ));
        }
        let out = alloc_f32(lhs.len())?;
        let status = unsafe {
            lumen_cuda_binary_bf16_device(
                lhs.handle(),
                rhs.handle(),
                out.handle(),
                lhs.len(),
                op as c_int,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(out)
    }

    pub fn binary_bf16_lastdim_buffer_no_host(
        lhs: &CudaBuffer,
        rhs: &CudaBuffer,
        out_len: usize,
        last_dim: usize,
        vector_on_rhs: bool,
        op: BinaryOp,
    ) -> Result<CudaBuffer, String> {
        if last_dim == 0 {
            return Err("CUDA BF16 row-broadcast last_dim must be greater than zero".to_string());
        }
        let expected_lhs = if vector_on_rhs { out_len } else { last_dim };
        let expected_rhs = if vector_on_rhs { last_dim } else { out_len };
        if lhs.len() != expected_lhs || rhs.len() != expected_rhs {
            return Err(format!(
                "CUDA BF16 row-broadcast length mismatch: expected lhs={}, rhs={}, got lhs={}, rhs={}",
                expected_lhs,
                expected_rhs,
                lhs.len(),
                rhs.len()
            ));
        }
        let out = alloc_f32(out_len)?;
        let status = unsafe {
            lumen_cuda_binary_bf16_lastdim_device(
                lhs.handle(),
                rhs.handle(),
                out.handle(),
                out_len,
                last_dim,
                vector_on_rhs as c_int,
                op as c_int,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(out)
    }

    pub fn binary_i8_host_no_host(
        lhs: &[i8],
        lhs_scale: f32,
        rhs: &[i8],
        rhs_scale: f32,
        op: BinaryOp,
    ) -> Result<CudaBuffer, String> {
        if lhs.len() != rhs.len() {
            return Err(format!(
                "CUDA I8 binary length mismatch: lhs={}, rhs={}",
                lhs.len(),
                rhs.len()
            ));
        }
        let out = alloc_f32(lhs.len())?;
        let status = unsafe {
            lumen_cuda_binary_i8_host_device(
                lhs.as_ptr(),
                rhs.as_ptr(),
                lhs_scale,
                rhs_scale,
                out.handle(),
                lhs.len(),
                op as c_int,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(out)
    }

    pub fn binary_i8_buffer_no_host(
        lhs: &CudaBuffer,
        lhs_scale: f32,
        rhs: &CudaBuffer,
        rhs_scale: f32,
        op: BinaryOp,
    ) -> Result<CudaBuffer, String> {
        if lhs.len() != rhs.len() {
            return Err(format!(
                "CUDA resident I8 binary length mismatch: lhs={}, rhs={}",
                lhs.len(),
                rhs.len()
            ));
        }
        let out = alloc_f32(lhs.len())?;
        let status = unsafe {
            lumen_cuda_binary_i8_device(
                lhs.handle(),
                rhs.handle(),
                lhs_scale,
                rhs_scale,
                out.handle(),
                lhs.len(),
                op as c_int,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(out)
    }

    pub fn binary_i8_typed_output_buffer_no_host(
        lhs: &CudaBuffer,
        lhs_scale: f32,
        rhs: &CudaBuffer,
        rhs_scale: f32,
        op: BinaryOp,
    ) -> Result<(CudaBuffer, f32), String> {
        if lhs.len() != rhs.len() {
            return Err(format!(
                "CUDA resident I8 typed-out binary length mismatch: lhs={}, rhs={}",
                lhs.len(),
                rhs.len()
            ));
        }
        let out = alloc_storage(lhs.len())?;
        let mut out_scale = 1.0f32;
        let status = unsafe {
            lumen_cuda_binary_i8_typed_out_device(
                lhs.handle(),
                rhs.handle(),
                lhs_scale,
                rhs_scale,
                out.handle(),
                lhs.len(),
                op as c_int,
                &mut out_scale as *mut f32,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok((out, out_scale))
    }

    pub fn binary_i8_lastdim_buffer_no_host(
        lhs: &CudaBuffer,
        lhs_scale: f32,
        rhs: &CudaBuffer,
        rhs_scale: f32,
        out_len: usize,
        last_dim: usize,
        vector_on_rhs: bool,
        op: BinaryOp,
    ) -> Result<CudaBuffer, String> {
        if last_dim == 0 {
            return Err("CUDA I8 row-broadcast last_dim must be greater than zero".to_string());
        }
        let expected_lhs = if vector_on_rhs { out_len } else { last_dim };
        let expected_rhs = if vector_on_rhs { last_dim } else { out_len };
        if lhs.len() != expected_lhs || rhs.len() != expected_rhs {
            return Err(format!(
                "CUDA I8 row-broadcast length mismatch: expected lhs={}, rhs={}, got lhs={}, rhs={}",
                expected_lhs,
                expected_rhs,
                lhs.len(),
                rhs.len()
            ));
        }
        let out = alloc_f32(out_len)?;
        let status = unsafe {
            lumen_cuda_binary_i8_lastdim_device(
                lhs.handle(),
                rhs.handle(),
                lhs_scale,
                rhs_scale,
                out.handle(),
                out_len,
                last_dim,
                vector_on_rhs as c_int,
                op as c_int,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(out)
    }

    pub fn binary_i8_typed_lastdim_output_buffer_no_host(
        lhs: &CudaBuffer,
        lhs_scale: f32,
        rhs: &CudaBuffer,
        rhs_scale: f32,
        out_len: usize,
        last_dim: usize,
        vector_on_rhs: bool,
        op: BinaryOp,
    ) -> Result<(CudaBuffer, f32), String> {
        if last_dim == 0 {
            return Err(
                "CUDA I8 typed-out row-broadcast last_dim must be greater than zero".to_string(),
            );
        }
        let expected_lhs = if vector_on_rhs { out_len } else { last_dim };
        let expected_rhs = if vector_on_rhs { last_dim } else { out_len };
        if lhs.len() != expected_lhs || rhs.len() != expected_rhs {
            return Err(format!(
                "CUDA I8 typed-out row-broadcast length mismatch: expected lhs={}, rhs={}, got lhs={}, rhs={}",
                expected_lhs,
                expected_rhs,
                lhs.len(),
                rhs.len()
            ));
        }
        let out = alloc_storage(out_len)?;
        let mut out_scale = 1.0f32;
        let status = unsafe {
            lumen_cuda_binary_i8_typed_lastdim_out_device(
                lhs.handle(),
                rhs.handle(),
                lhs_scale,
                rhs_scale,
                out.handle(),
                out_len,
                last_dim,
                vector_on_rhs as c_int,
                op as c_int,
                &mut out_scale as *mut f32,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok((out, out_scale))
    }

    pub fn mul_grad_f16_host_no_host(
        grad: &CudaBuffer,
        operand: &[u16],
    ) -> Result<CudaBuffer, String> {
        if grad.len() != operand.len() {
            return Err(format!(
                "CUDA F16 mul grad length mismatch: grad={}, operand={}",
                grad.len(),
                operand.len()
            ));
        }
        let out = alloc_f32(operand.len())?;
        let status = unsafe {
            lumen_cuda_mul_grad_f16_host_device(
                grad.handle(),
                operand.as_ptr(),
                out.handle(),
                operand.len(),
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(out)
    }

    pub fn mul_grad_f16_buffer_no_host(
        grad: &CudaBuffer,
        operand: &CudaBuffer,
    ) -> Result<CudaBuffer, String> {
        if grad.len() != operand.len() {
            return Err(format!(
                "CUDA resident F16 mul grad length mismatch: grad={}, operand={}",
                grad.len(),
                operand.len()
            ));
        }
        let out = alloc_f32(operand.len())?;
        let status = unsafe {
            lumen_cuda_mul_grad_f16_device(
                grad.handle(),
                operand.handle(),
                out.handle(),
                operand.len(),
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(out)
    }

    pub fn mul_grad_f16_lastdim_buffer_no_host(
        grad: &CudaBuffer,
        lhs: &CudaBuffer,
        rhs: &CudaBuffer,
        out_len: usize,
        last_dim: usize,
        vector_on_rhs: bool,
    ) -> Result<(CudaBuffer, CudaBuffer), String> {
        let expected_lhs = if vector_on_rhs { out_len } else { last_dim };
        let expected_rhs = if vector_on_rhs { last_dim } else { out_len };
        if grad.len() != out_len || lhs.len() != expected_lhs || rhs.len() != expected_rhs {
            return Err(format!(
                "CUDA F16 row-broadcast mul grad length mismatch: grad={}, lhs={}, rhs={}, expected grad={}, lhs={}, rhs={}",
                grad.len(),
                lhs.len(),
                rhs.len(),
                out_len,
                expected_lhs,
                expected_rhs
            ));
        }
        let grad_lhs = alloc_f32(expected_lhs)?;
        let grad_rhs = alloc_f32(expected_rhs)?;
        let status = unsafe {
            lumen_cuda_mul_grad_f16_lastdim_device(
                grad.handle(),
                lhs.handle(),
                rhs.handle(),
                grad_lhs.handle(),
                grad_rhs.handle(),
                out_len,
                last_dim,
                vector_on_rhs as c_int,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok((grad_lhs, grad_rhs))
    }

    pub fn mul_grad_bf16_host_no_host(
        grad: &CudaBuffer,
        operand: &[u16],
    ) -> Result<CudaBuffer, String> {
        if grad.len() != operand.len() {
            return Err(format!(
                "CUDA BF16 mul grad length mismatch: grad={}, operand={}",
                grad.len(),
                operand.len()
            ));
        }
        let out = alloc_f32(operand.len())?;
        let status = unsafe {
            lumen_cuda_mul_grad_bf16_host_device(
                grad.handle(),
                operand.as_ptr(),
                out.handle(),
                operand.len(),
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(out)
    }

    pub fn mul_grad_bf16_buffer_no_host(
        grad: &CudaBuffer,
        operand: &CudaBuffer,
    ) -> Result<CudaBuffer, String> {
        if grad.len() != operand.len() {
            return Err(format!(
                "CUDA resident BF16 mul grad length mismatch: grad={}, operand={}",
                grad.len(),
                operand.len()
            ));
        }
        let out = alloc_f32(operand.len())?;
        let status = unsafe {
            lumen_cuda_mul_grad_bf16_device(
                grad.handle(),
                operand.handle(),
                out.handle(),
                operand.len(),
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(out)
    }

    pub fn mul_grad_bf16_lastdim_buffer_no_host(
        grad: &CudaBuffer,
        lhs: &CudaBuffer,
        rhs: &CudaBuffer,
        out_len: usize,
        last_dim: usize,
        vector_on_rhs: bool,
    ) -> Result<(CudaBuffer, CudaBuffer), String> {
        let expected_lhs = if vector_on_rhs { out_len } else { last_dim };
        let expected_rhs = if vector_on_rhs { last_dim } else { out_len };
        if grad.len() != out_len || lhs.len() != expected_lhs || rhs.len() != expected_rhs {
            return Err(format!(
                "CUDA BF16 row-broadcast mul grad length mismatch: grad={}, lhs={}, rhs={}, expected grad={}, lhs={}, rhs={}",
                grad.len(),
                lhs.len(),
                rhs.len(),
                out_len,
                expected_lhs,
                expected_rhs
            ));
        }
        let grad_lhs = alloc_f32(expected_lhs)?;
        let grad_rhs = alloc_f32(expected_rhs)?;
        let status = unsafe {
            lumen_cuda_mul_grad_bf16_lastdim_device(
                grad.handle(),
                lhs.handle(),
                rhs.handle(),
                grad_lhs.handle(),
                grad_rhs.handle(),
                out_len,
                last_dim,
                vector_on_rhs as c_int,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok((grad_lhs, grad_rhs))
    }

    pub fn mul_grad_i8_host_no_host(
        grad: &CudaBuffer,
        operand: &[i8],
        scale: f32,
    ) -> Result<CudaBuffer, String> {
        if grad.len() != operand.len() {
            return Err(format!(
                "CUDA I8 mul grad length mismatch: grad={}, operand={}",
                grad.len(),
                operand.len()
            ));
        }
        let out = alloc_f32(operand.len())?;
        let status = unsafe {
            lumen_cuda_mul_grad_i8_host_device(
                grad.handle(),
                operand.as_ptr(),
                scale,
                out.handle(),
                operand.len(),
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(out)
    }

    pub fn mul_grad_i8_buffer_no_host(
        grad: &CudaBuffer,
        operand: &CudaBuffer,
        scale: f32,
    ) -> Result<CudaBuffer, String> {
        if grad.len() != operand.len() {
            return Err(format!(
                "CUDA resident I8 mul grad length mismatch: grad={}, operand={}",
                grad.len(),
                operand.len()
            ));
        }
        let out = alloc_f32(operand.len())?;
        let status = unsafe {
            lumen_cuda_mul_grad_i8_device(
                grad.handle(),
                operand.handle(),
                scale,
                out.handle(),
                operand.len(),
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(out)
    }

    pub fn mul_grad_i8_lastdim_buffer_no_host(
        grad: &CudaBuffer,
        lhs: &CudaBuffer,
        lhs_scale: f32,
        rhs: &CudaBuffer,
        rhs_scale: f32,
        out_len: usize,
        last_dim: usize,
        vector_on_rhs: bool,
    ) -> Result<(CudaBuffer, CudaBuffer), String> {
        let expected_lhs = if vector_on_rhs { out_len } else { last_dim };
        let expected_rhs = if vector_on_rhs { last_dim } else { out_len };
        if grad.len() != out_len || lhs.len() != expected_lhs || rhs.len() != expected_rhs {
            return Err(format!(
                "CUDA I8 row-broadcast mul grad length mismatch: grad={}, lhs={}, rhs={}, expected grad={}, lhs={}, rhs={}",
                grad.len(),
                lhs.len(),
                rhs.len(),
                out_len,
                expected_lhs,
                expected_rhs
            ));
        }
        let grad_lhs = alloc_f32(expected_lhs)?;
        let grad_rhs = alloc_f32(expected_rhs)?;
        let status = unsafe {
            lumen_cuda_mul_grad_i8_lastdim_device(
                grad.handle(),
                lhs.handle(),
                rhs.handle(),
                lhs_scale,
                rhs_scale,
                grad_lhs.handle(),
                grad_rhs.handle(),
                out_len,
                last_dim,
                vector_on_rhs as c_int,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok((grad_lhs, grad_rhs))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn mul_grad_typed_lastdim_buffer_no_host(
        grad: &CudaBuffer,
        lhs: &CudaBuffer,
        lhs_dtype: DType,
        lhs_scale: Option<f32>,
        rhs: &CudaBuffer,
        rhs_dtype: DType,
        rhs_scale: Option<f32>,
        out_len: usize,
        last_dim: usize,
        vector_on_rhs: bool,
    ) -> Result<(CudaBuffer, CudaBuffer), String> {
        let expected_lhs = if vector_on_rhs { out_len } else { last_dim };
        let expected_rhs = if vector_on_rhs { last_dim } else { out_len };
        if grad.len() != out_len || lhs.len() != expected_lhs || rhs.len() != expected_rhs {
            return Err(format!(
                "CUDA typed mixed row-broadcast mul grad length mismatch: grad={}, lhs={}, rhs={}, expected grad={}, lhs={}, rhs={}",
                grad.len(),
                lhs.len(),
                rhs.len(),
                out_len,
                expected_lhs,
                expected_rhs
            ));
        }
        let grad_lhs = alloc_f32(expected_lhs)?;
        let grad_rhs = alloc_f32(expected_rhs)?;
        let status = unsafe {
            lumen_cuda_mul_grad_typed_lastdim_device(
                grad.handle(),
                lhs.handle(),
                dtype_tag(lhs_dtype)?,
                lhs_scale.unwrap_or(1.0),
                rhs.handle(),
                dtype_tag(rhs_dtype)?,
                rhs_scale.unwrap_or(1.0),
                grad_lhs.handle(),
                grad_rhs.handle(),
                out_len,
                last_dim,
                vector_on_rhs as c_int,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok((grad_lhs, grad_rhs))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn mul_grad_typed_buffer_no_host(
        grad: &CudaBuffer,
        lhs: &CudaBuffer,
        lhs_dtype: DType,
        lhs_scale: Option<f32>,
        rhs: &CudaBuffer,
        rhs_dtype: DType,
        rhs_scale: Option<f32>,
    ) -> Result<(CudaBuffer, CudaBuffer), String> {
        if grad.len() != lhs.len() || grad.len() != rhs.len() || grad.is_empty() {
            return Err(format!(
                "CUDA typed mixed same-shape mul grad length mismatch: grad={}, lhs={}, rhs={}",
                grad.len(),
                lhs.len(),
                rhs.len()
            ));
        }
        let grad_lhs = alloc_f32(grad.len())?;
        let grad_rhs = alloc_f32(grad.len())?;
        let status = unsafe {
            lumen_cuda_mul_grad_typed_device(
                grad.handle(),
                lhs.handle(),
                dtype_tag(lhs_dtype)?,
                lhs_scale.unwrap_or(1.0),
                rhs.handle(),
                dtype_tag(rhs_dtype)?,
                rhs_scale.unwrap_or(1.0),
                grad_lhs.handle(),
                grad_rhs.handle(),
                grad.len(),
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok((grad_lhs, grad_rhs))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn mul_grad_typed_row_scalar_buffer_no_host(
        grad: &CudaBuffer,
        lhs: &CudaBuffer,
        lhs_dtype: DType,
        lhs_scale: Option<f32>,
        rhs: &CudaBuffer,
        rhs_dtype: DType,
        rhs_scale: Option<f32>,
        rows: usize,
        last_dim: usize,
        scalar_on_rhs: bool,
    ) -> Result<(CudaBuffer, CudaBuffer), String> {
        let out_len = rows.checked_mul(last_dim).ok_or_else(|| {
            "CUDA typed mixed row-scalar mul grad output length overflow".to_string()
        })?;
        let expected_lhs = if scalar_on_rhs { out_len } else { rows };
        let expected_rhs = if scalar_on_rhs { rows } else { out_len };
        if grad.len() != out_len || lhs.len() != expected_lhs || rhs.len() != expected_rhs {
            return Err(format!(
                "CUDA typed mixed row-scalar mul grad length mismatch: expected grad={}, lhs={}, rhs={}, got grad={}, lhs={}, rhs={}",
                out_len,
                expected_lhs,
                expected_rhs,
                grad.len(),
                lhs.len(),
                rhs.len()
            ));
        }
        let grad_lhs = alloc_f32(expected_lhs)?;
        let grad_rhs = alloc_f32(expected_rhs)?;
        let status = unsafe {
            lumen_cuda_mul_grad_typed_row_scalar_device(
                grad.handle(),
                lhs.handle(),
                dtype_tag(lhs_dtype)?,
                lhs_scale.unwrap_or(1.0),
                rhs.handle(),
                dtype_tag(rhs_dtype)?,
                rhs_scale.unwrap_or(1.0),
                grad_lhs.handle(),
                grad_rhs.handle(),
                rows,
                last_dim,
                scalar_on_rhs as c_int,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok((grad_lhs, grad_rhs))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn mul_grad_typed_broadcast_buffer_no_host(
        grad: &CudaBuffer,
        lhs: &CudaBuffer,
        lhs_dtype: DType,
        lhs_scale: Option<f32>,
        rhs: &CudaBuffer,
        rhs_dtype: DType,
        rhs_scale: Option<f32>,
        lhs_shape: &[usize],
        rhs_shape: &[usize],
        out_shape: &[usize],
    ) -> Result<(CudaBuffer, CudaBuffer), String> {
        let out_len = out_shape
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
            .ok_or_else(|| {
                "CUDA typed mixed broadcast mul grad output length overflow".to_string()
            })?;
        let lhs_len = lhs_shape
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
            .ok_or_else(|| "CUDA typed mixed broadcast mul grad lhs length overflow".to_string())?;
        let rhs_len = rhs_shape
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
            .ok_or_else(|| "CUDA typed mixed broadcast mul grad rhs length overflow".to_string())?;
        if grad.len() != out_len || lhs.len() != lhs_len || rhs.len() != rhs_len {
            return Err(format!(
                "CUDA typed mixed broadcast mul grad length mismatch: expected grad={}, lhs={}, rhs={}, got grad={}, lhs={}, rhs={}",
                out_len,
                lhs_len,
                rhs_len,
                grad.len(),
                lhs.len(),
                rhs.len()
            ));
        }
        let (lhs_aligned, rhs_aligned, out_strides, lhs_strides, rhs_strides) =
            aligned_broadcast_metadata(lhs_shape, rhs_shape, out_shape)?;
        let grad_lhs = alloc_f32(lhs_len)?;
        let grad_rhs = alloc_f32(rhs_len)?;
        let status = unsafe {
            lumen_cuda_mul_grad_typed_broadcast_device(
                grad.handle(),
                lhs.handle(),
                dtype_tag(lhs_dtype)?,
                lhs_scale.unwrap_or(1.0),
                rhs.handle(),
                dtype_tag(rhs_dtype)?,
                rhs_scale.unwrap_or(1.0),
                grad_lhs.handle(),
                grad_rhs.handle(),
                out_shape.len(),
                out_strides.as_ptr(),
                lhs_aligned.as_ptr(),
                lhs_strides.as_ptr(),
                rhs_aligned.as_ptr(),
                rhs_strides.as_ptr(),
                out_len,
                lhs_len,
                rhs_len,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok((grad_lhs, grad_rhs))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn mul_grad_typed_b1d_1h1_buffer_no_host(
        grad: &CudaBuffer,
        lhs: &CudaBuffer,
        lhs_dtype: DType,
        lhs_scale: Option<f32>,
        rhs: &CudaBuffer,
        rhs_dtype: DType,
        rhs_scale: Option<f32>,
        batch: usize,
        heads: usize,
        dim: usize,
        b1d_on_lhs: bool,
    ) -> Result<(CudaBuffer, CudaBuffer), String> {
        let b1d_len = batch
            .checked_mul(dim)
            .ok_or_else(|| "CUDA typed mixed b1d/1h1 mul grad b1d length overflow".to_string())?;
        let out_len = b1d_len.checked_mul(heads).ok_or_else(|| {
            "CUDA typed mixed b1d/1h1 mul grad output length overflow".to_string()
        })?;
        let expected_lhs = if b1d_on_lhs { b1d_len } else { heads };
        let expected_rhs = if b1d_on_lhs { heads } else { b1d_len };
        if grad.len() != out_len || lhs.len() != expected_lhs || rhs.len() != expected_rhs {
            return Err(format!(
                "CUDA typed mixed b1d/1h1 mul grad length mismatch: expected grad={}, lhs={}, rhs={}, got grad={}, lhs={}, rhs={}",
                out_len,
                expected_lhs,
                expected_rhs,
                grad.len(),
                lhs.len(),
                rhs.len()
            ));
        }
        let grad_lhs = alloc_f32(expected_lhs)?;
        let grad_rhs = alloc_f32(expected_rhs)?;
        let status = unsafe {
            lumen_cuda_mul_grad_typed_b1d_1h1_device(
                grad.handle(),
                lhs.handle(),
                dtype_tag(lhs_dtype)?,
                lhs_scale.unwrap_or(1.0),
                rhs.handle(),
                dtype_tag(rhs_dtype)?,
                rhs_scale.unwrap_or(1.0),
                grad_lhs.handle(),
                grad_rhs.handle(),
                batch,
                heads,
                dim,
                b1d_on_lhs as c_int,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok((grad_lhs, grad_rhs))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn mul_grad_typed_b1d_1hd_buffer_no_host(
        grad: &CudaBuffer,
        lhs: &CudaBuffer,
        lhs_dtype: DType,
        lhs_scale: Option<f32>,
        rhs: &CudaBuffer,
        rhs_dtype: DType,
        rhs_scale: Option<f32>,
        batch: usize,
        heads: usize,
        dim: usize,
        b1d_on_lhs: bool,
    ) -> Result<(CudaBuffer, CudaBuffer), String> {
        let b1d_len = batch
            .checked_mul(dim)
            .ok_or_else(|| "CUDA typed mixed b1d/1hd mul grad b1d length overflow".to_string())?;
        let hd_len = heads
            .checked_mul(dim)
            .ok_or_else(|| "CUDA typed mixed b1d/1hd mul grad hd length overflow".to_string())?;
        let out_len = b1d_len.checked_mul(heads).ok_or_else(|| {
            "CUDA typed mixed b1d/1hd mul grad output length overflow".to_string()
        })?;
        let expected_lhs = if b1d_on_lhs { b1d_len } else { hd_len };
        let expected_rhs = if b1d_on_lhs { hd_len } else { b1d_len };
        if grad.len() != out_len || lhs.len() != expected_lhs || rhs.len() != expected_rhs {
            return Err(format!(
                "CUDA typed mixed b1d/1hd mul grad length mismatch: expected grad={}, lhs={}, rhs={}, got grad={}, lhs={}, rhs={}",
                out_len,
                expected_lhs,
                expected_rhs,
                grad.len(),
                lhs.len(),
                rhs.len()
            ));
        }
        let grad_lhs = alloc_f32(expected_lhs)?;
        let grad_rhs = alloc_f32(expected_rhs)?;
        let status = unsafe {
            lumen_cuda_mul_grad_typed_b1d_1hd_device(
                grad.handle(),
                lhs.handle(),
                dtype_tag(lhs_dtype)?,
                lhs_scale.unwrap_or(1.0),
                rhs.handle(),
                dtype_tag(rhs_dtype)?,
                rhs_scale.unwrap_or(1.0),
                grad_lhs.handle(),
                grad_rhs.handle(),
                batch,
                heads,
                dim,
                b1d_on_lhs as c_int,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok((grad_lhs, grad_rhs))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn mul_grad_typed_scalar_buffer_no_host(
        grad: &CudaBuffer,
        lhs: &CudaBuffer,
        lhs_dtype: DType,
        lhs_scale: Option<f32>,
        rhs: &CudaBuffer,
        rhs_dtype: DType,
        rhs_scale: Option<f32>,
        out_len: usize,
        scalar_on_rhs: bool,
    ) -> Result<(CudaBuffer, CudaBuffer), String> {
        let expected_lhs = if scalar_on_rhs { out_len } else { 1 };
        let expected_rhs = if scalar_on_rhs { 1 } else { out_len };
        if grad.len() != out_len || lhs.len() != expected_lhs || rhs.len() != expected_rhs {
            return Err(format!(
                "CUDA typed mixed scalar-broadcast mul grad length mismatch: expected grad={}, lhs={}, rhs={}, got grad={}, lhs={}, rhs={}",
                out_len,
                expected_lhs,
                expected_rhs,
                grad.len(),
                lhs.len(),
                rhs.len()
            ));
        }
        let grad_lhs = alloc_f32(expected_lhs)?;
        let grad_rhs = alloc_f32(expected_rhs)?;
        let status = unsafe {
            lumen_cuda_mul_grad_typed_scalar_device(
                grad.handle(),
                lhs.handle(),
                dtype_tag(lhs_dtype)?,
                lhs_scale.unwrap_or(1.0),
                rhs.handle(),
                dtype_tag(rhs_dtype)?,
                rhs_scale.unwrap_or(1.0),
                grad_lhs.handle(),
                grad_rhs.handle(),
                out_len,
                scalar_on_rhs as c_int,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok((grad_lhs, grad_rhs))
    }

    pub fn binary_backward_f32(
        lhs: &CudaBuffer,
        rhs: &CudaBuffer,
        grad: &CudaBuffer,
        op: BinaryOp,
    ) -> Result<CudaTwoHostBuffers, String> {
        let (grad_lhs, grad_rhs) = binary_backward_f32_buffers(lhs, rhs, grad, op)?;
        let grad_lhs_host = download_f32(&grad_lhs)?;
        let grad_rhs_host = download_f32(&grad_rhs)?;
        Ok(((grad_lhs, grad_lhs_host), (grad_rhs, grad_rhs_host)))
    }

    pub fn binary_backward_f32_buffers(
        lhs: &CudaBuffer,
        rhs: &CudaBuffer,
        grad: &CudaBuffer,
        op: BinaryOp,
    ) -> Result<(CudaBuffer, CudaBuffer), String> {
        if lhs.len() != rhs.len() || lhs.len() != grad.len() {
            return Err(format!(
                "CUDA binary backward length mismatch: lhs={}, rhs={}, grad={}",
                lhs.len(),
                rhs.len(),
                grad.len()
            ));
        }
        let grad_lhs = alloc_f32(lhs.len())?;
        let grad_rhs = alloc_f32(lhs.len())?;
        let status = unsafe {
            lumen_cuda_binary_backward_f32_device(
                lhs.handle(),
                rhs.handle(),
                grad.handle(),
                grad_lhs.handle(),
                grad_rhs.handle(),
                lhs.len(),
                op as c_int,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok((grad_lhs, grad_rhs))
    }

    pub fn add_sub_backward_f32_buffers(
        grad: &CudaBuffer,
        len: usize,
        op: BinaryOp,
    ) -> Result<(CudaBuffer, CudaBuffer), String> {
        if !matches!(op, BinaryOp::Add | BinaryOp::Sub) {
            return Err("CUDA add/sub backward only supports Add/Sub".to_string());
        }
        if grad.len() != len {
            return Err(format!(
                "CUDA add/sub backward length mismatch: grad={}, len={}",
                grad.len(),
                len
            ));
        }
        let grad_lhs = alloc_f32(len)?;
        let grad_rhs = alloc_f32(len)?;
        let status = unsafe {
            lumen_cuda_add_sub_backward_f32_device(
                grad.handle(),
                grad_lhs.handle(),
                grad_rhs.handle(),
                len,
                op as c_int,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok((grad_lhs, grad_rhs))
    }

    pub fn add_sub_backward_lastdim_f32_buffers(
        grad: &CudaBuffer,
        out_len: usize,
        last_dim: usize,
        vector_on_rhs: bool,
        op: BinaryOp,
    ) -> Result<(CudaBuffer, CudaBuffer), String> {
        if !matches!(op, BinaryOp::Add | BinaryOp::Sub) {
            return Err("CUDA add/sub row-broadcast backward only supports Add/Sub".to_string());
        }
        if last_dim == 0 {
            return Err(
                "CUDA add/sub row-broadcast backward last_dim must be greater than zero"
                    .to_string(),
            );
        }
        if grad.len() != out_len || !out_len.is_multiple_of(last_dim) {
            return Err(format!(
                "CUDA add/sub row-broadcast backward length mismatch: grad={}, out_len={}, last_dim={}",
                grad.len(),
                out_len,
                last_dim
            ));
        }
        let grad_lhs = alloc_f32(if vector_on_rhs { out_len } else { last_dim })?;
        let grad_rhs = alloc_f32(if vector_on_rhs { last_dim } else { out_len })?;
        let status = unsafe {
            lumen_cuda_add_sub_backward_lastdim_f32_device(
                grad.handle(),
                grad_lhs.handle(),
                grad_rhs.handle(),
                out_len,
                last_dim,
                vector_on_rhs as c_int,
                op as c_int,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok((grad_lhs, grad_rhs))
    }

    pub fn add_sub_backward_scalar_f32_buffers(
        grad: &CudaBuffer,
        out_len: usize,
        scalar_on_rhs: bool,
        op: BinaryOp,
    ) -> Result<(CudaBuffer, CudaBuffer), String> {
        if !matches!(op, BinaryOp::Add | BinaryOp::Sub) {
            return Err("CUDA add/sub scalar-broadcast backward only supports Add/Sub".to_string());
        }
        if grad.len() != out_len || out_len == 0 {
            return Err(format!(
                "CUDA add/sub scalar-broadcast backward length mismatch: grad={}, out_len={}",
                grad.len(),
                out_len
            ));
        }
        let grad_lhs = alloc_f32(if scalar_on_rhs { out_len } else { 1 })?;
        let grad_rhs = alloc_f32(if scalar_on_rhs { 1 } else { out_len })?;
        let status = unsafe {
            lumen_cuda_add_sub_backward_scalar_f32_device(
                grad.handle(),
                grad_lhs.handle(),
                grad_rhs.handle(),
                out_len,
                scalar_on_rhs as c_int,
                op as c_int,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok((grad_lhs, grad_rhs))
    }

    pub fn add_sub_backward_row_scalar_f32_buffers(
        grad: &CudaBuffer,
        rows: usize,
        last_dim: usize,
        scalar_on_rhs: bool,
        op: BinaryOp,
    ) -> Result<(CudaBuffer, CudaBuffer), String> {
        let out_len = rows
            .checked_mul(last_dim)
            .ok_or_else(|| "CUDA add/sub row-scalar backward output length overflow".to_string())?;
        if grad.len() != out_len {
            return Err(format!(
                "CUDA add/sub row-scalar backward length mismatch: expected grad={}, got {}",
                out_len,
                grad.len()
            ));
        }
        let grad_lhs = alloc_f32(if scalar_on_rhs { out_len } else { rows })?;
        let grad_rhs = alloc_f32(if scalar_on_rhs { rows } else { out_len })?;
        let status = unsafe {
            lumen_cuda_add_sub_backward_row_scalar_f32_device(
                grad.handle(),
                grad_lhs.handle(),
                grad_rhs.handle(),
                rows,
                last_dim,
                scalar_on_rhs as c_int,
                op as c_int,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok((grad_lhs, grad_rhs))
    }

    pub fn add_sub_backward_b1d_1h1_f32_buffers(
        grad: &CudaBuffer,
        batch: usize,
        heads: usize,
        dim: usize,
        b1d_on_lhs: bool,
        op: BinaryOp,
    ) -> Result<(CudaBuffer, CudaBuffer), String> {
        if !matches!(op, BinaryOp::Add | BinaryOp::Sub) {
            return Err("CUDA add/sub b1d/1h1 backward only supports Add/Sub".to_string());
        }
        let b1d_len = batch
            .checked_mul(dim)
            .ok_or_else(|| "CUDA add/sub b1d/1h1 backward b1d length overflow".to_string())?;
        let out_len = b1d_len
            .checked_mul(heads)
            .ok_or_else(|| "CUDA add/sub b1d/1h1 backward output length overflow".to_string())?;
        if grad.len() != out_len {
            return Err(format!(
                "CUDA add/sub b1d/1h1 backward length mismatch: grad={}, out_len={}",
                grad.len(),
                out_len
            ));
        }
        let grad_lhs = alloc_f32(if b1d_on_lhs { b1d_len } else { heads })?;
        let grad_rhs = alloc_f32(if b1d_on_lhs { heads } else { b1d_len })?;
        let status = unsafe {
            lumen_cuda_add_sub_backward_b1d_1h1_f32_device(
                grad.handle(),
                grad_lhs.handle(),
                grad_rhs.handle(),
                batch,
                heads,
                dim,
                b1d_on_lhs as c_int,
                op as c_int,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok((grad_lhs, grad_rhs))
    }

    pub fn add_sub_backward_b1d_1hd_f32_buffers(
        grad: &CudaBuffer,
        batch: usize,
        heads: usize,
        dim: usize,
        b1d_on_lhs: bool,
        op: BinaryOp,
    ) -> Result<(CudaBuffer, CudaBuffer), String> {
        if !matches!(op, BinaryOp::Add | BinaryOp::Sub) {
            return Err("CUDA add/sub b1d/1hd backward only supports Add/Sub".to_string());
        }
        let b1d_len = batch
            .checked_mul(dim)
            .ok_or_else(|| "CUDA add/sub b1d/1hd backward b1d length overflow".to_string())?;
        let hd_len = heads
            .checked_mul(dim)
            .ok_or_else(|| "CUDA add/sub b1d/1hd backward hd length overflow".to_string())?;
        let out_len = b1d_len
            .checked_mul(heads)
            .ok_or_else(|| "CUDA add/sub b1d/1hd backward output length overflow".to_string())?;
        if grad.len() != out_len {
            return Err(format!(
                "CUDA add/sub b1d/1hd backward length mismatch: grad={}, out_len={}",
                grad.len(),
                out_len
            ));
        }
        let grad_lhs = alloc_f32(if b1d_on_lhs { b1d_len } else { hd_len })?;
        let grad_rhs = alloc_f32(if b1d_on_lhs { hd_len } else { b1d_len })?;
        let status = unsafe {
            lumen_cuda_add_sub_backward_b1d_1hd_f32_device(
                grad.handle(),
                grad_lhs.handle(),
                grad_rhs.handle(),
                batch,
                heads,
                dim,
                b1d_on_lhs as c_int,
                op as c_int,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok((grad_lhs, grad_rhs))
    }

    pub fn add_sub_broadcast_backward_f32_buffers(
        grad: &CudaBuffer,
        lhs_shape: &[usize],
        rhs_shape: &[usize],
        out_shape: &[usize],
        op: BinaryOp,
    ) -> Result<(CudaBuffer, CudaBuffer), String> {
        if !matches!(op, BinaryOp::Add | BinaryOp::Sub) {
            return Err("CUDA add/sub broadcast backward only supports Add/Sub".to_string());
        }
        let out_len = out_shape
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
            .ok_or_else(|| "CUDA add/sub broadcast backward output length overflow".to_string())?;
        let lhs_len = lhs_shape
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
            .ok_or_else(|| "CUDA add/sub broadcast backward lhs length overflow".to_string())?;
        let rhs_len = rhs_shape
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
            .ok_or_else(|| "CUDA add/sub broadcast backward rhs length overflow".to_string())?;
        if grad.len() != out_len {
            return Err(format!(
                "CUDA add/sub broadcast backward length mismatch: grad={}, out_len={}",
                grad.len(),
                out_len
            ));
        }
        let (lhs_aligned, rhs_aligned, out_strides, lhs_strides, rhs_strides) =
            aligned_broadcast_metadata(lhs_shape, rhs_shape, out_shape)?;
        let grad_lhs = alloc_f32(lhs_len)?;
        let grad_rhs = alloc_f32(rhs_len)?;
        let status = unsafe {
            lumen_cuda_add_sub_broadcast_backward_f32_device(
                grad.handle(),
                grad_lhs.handle(),
                grad_rhs.handle(),
                out_shape.len(),
                out_strides.as_ptr(),
                lhs_aligned.as_ptr(),
                lhs_strides.as_ptr(),
                rhs_aligned.as_ptr(),
                rhs_strides.as_ptr(),
                out_len,
                lhs_len,
                rhs_len,
                op as c_int,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok((grad_lhs, grad_rhs))
    }

    pub fn binary_broadcast_f32(
        lhs: &CudaBuffer,
        rhs: &CudaBuffer,
        lhs_shape: &[usize],
        rhs_shape: &[usize],
        out_shape: &[usize],
        op: BinaryOp,
    ) -> Result<(CudaBuffer, Vec<f32>), String> {
        let out = binary_broadcast_f32_buffer(lhs, rhs, lhs_shape, rhs_shape, out_shape, op)?;
        let host = download_f32(&out)?;
        Ok((out, host))
    }

    pub fn binary_broadcast_f32_buffer(
        lhs: &CudaBuffer,
        rhs: &CudaBuffer,
        lhs_shape: &[usize],
        rhs_shape: &[usize],
        out_shape: &[usize],
        op: BinaryOp,
    ) -> Result<CudaBuffer, String> {
        let len = out_shape
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
            .ok_or_else(|| "CUDA binary broadcast output length overflow".to_string())?;
        let lhs_len = lhs_shape
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
            .ok_or_else(|| "CUDA binary broadcast lhs length overflow".to_string())?;
        let rhs_len = rhs_shape
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
            .ok_or_else(|| "CUDA binary broadcast rhs length overflow".to_string())?;
        if lhs.len() != lhs_len || rhs.len() != rhs_len {
            return Err(format!(
                "CUDA binary broadcast input length mismatch: expected lhs={}, rhs={}, got lhs={}, rhs={}",
                lhs_len,
                rhs_len,
                lhs.len(),
                rhs.len()
            ));
        }
        let (lhs_aligned, rhs_aligned, out_strides, lhs_strides, rhs_strides) =
            aligned_broadcast_metadata(lhs_shape, rhs_shape, out_shape)?;
        let out = alloc_f32(len)?;
        let status = unsafe {
            lumen_cuda_binary_broadcast_f32_device(
                lhs.handle(),
                rhs.handle(),
                out.handle(),
                out_shape.len(),
                out_shape.as_ptr(),
                out_strides.as_ptr(),
                lhs_aligned.as_ptr(),
                lhs_strides.as_ptr(),
                rhs_aligned.as_ptr(),
                rhs_strides.as_ptr(),
                len,
                op as c_int,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(out)
    }

    pub fn binary_broadcast_backward_f32(
        lhs: &CudaBuffer,
        rhs: &CudaBuffer,
        grad: &CudaBuffer,
        lhs_shape: &[usize],
        rhs_shape: &[usize],
        out_shape: &[usize],
        op: BinaryOp,
    ) -> Result<CudaTwoHostBuffers, String> {
        let (grad_lhs, grad_rhs) = binary_broadcast_backward_f32_buffers(
            lhs, rhs, grad, lhs_shape, rhs_shape, out_shape, op,
        )?;
        let grad_lhs_host = download_f32(&grad_lhs)?;
        let grad_rhs_host = download_f32(&grad_rhs)?;
        Ok(((grad_lhs, grad_lhs_host), (grad_rhs, grad_rhs_host)))
    }

    pub fn binary_broadcast_backward_f32_buffers(
        lhs: &CudaBuffer,
        rhs: &CudaBuffer,
        grad: &CudaBuffer,
        lhs_shape: &[usize],
        rhs_shape: &[usize],
        out_shape: &[usize],
        op: BinaryOp,
    ) -> Result<(CudaBuffer, CudaBuffer), String> {
        let out_len = out_shape
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
            .ok_or_else(|| "CUDA binary broadcast backward output length overflow".to_string())?;
        let lhs_len = lhs_shape
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
            .ok_or_else(|| "CUDA binary broadcast backward lhs length overflow".to_string())?;
        let rhs_len = rhs_shape
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
            .ok_or_else(|| "CUDA binary broadcast backward rhs length overflow".to_string())?;
        if lhs.len() != lhs_len || rhs.len() != rhs_len || grad.len() != out_len {
            return Err(format!(
                "CUDA binary broadcast backward length mismatch: expected lhs={}, rhs={}, grad={}, got lhs={}, rhs={}, grad={}",
                lhs_len,
                rhs_len,
                out_len,
                lhs.len(),
                rhs.len(),
                grad.len()
            ));
        }
        let (lhs_aligned, rhs_aligned, out_strides, lhs_strides, rhs_strides) =
            aligned_broadcast_metadata(lhs_shape, rhs_shape, out_shape)?;
        let grad_lhs = alloc_f32(lhs_len)?;
        let grad_rhs = alloc_f32(rhs_len)?;
        let status = unsafe {
            lumen_cuda_binary_broadcast_backward_f32_device(
                lhs.handle(),
                rhs.handle(),
                grad.handle(),
                grad_lhs.handle(),
                grad_rhs.handle(),
                out_shape.len(),
                out_shape.as_ptr(),
                out_strides.as_ptr(),
                lhs_aligned.as_ptr(),
                lhs_strides.as_ptr(),
                rhs_aligned.as_ptr(),
                rhs_strides.as_ptr(),
                out_len,
                lhs_len,
                rhs_len,
                op as c_int,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok((grad_lhs, grad_rhs))
    }

    pub fn sum_f32(input: &CudaBuffer) -> Result<(CudaBuffer, Vec<f32>), String> {
        let out = alloc_f32(1)?;
        let status =
            unsafe { lumen_cuda_sum_f32_device(input.handle(), out.handle(), input.len()) };
        if status != 0 {
            return Err(last_error_message());
        }
        let host = download_f32(&out)?;
        Ok((out, host))
    }

    pub fn sum_f16_buffer(input: &CudaBuffer) -> Result<(CudaBuffer, Vec<f32>), String> {
        let out = alloc_f32(1)?;
        let status =
            unsafe { lumen_cuda_sum_f16_device(input.handle(), out.handle(), input.len()) };
        if status != 0 {
            return Err(last_error_message());
        }
        let host = download_f32(&out)?;
        Ok((out, host))
    }

    pub fn sum_bf16_buffer(input: &CudaBuffer) -> Result<(CudaBuffer, Vec<f32>), String> {
        let out = alloc_f32(1)?;
        let status =
            unsafe { lumen_cuda_sum_bf16_device(input.handle(), out.handle(), input.len()) };
        if status != 0 {
            return Err(last_error_message());
        }
        let host = download_f32(&out)?;
        Ok((out, host))
    }

    pub fn sum_i8_buffer(input: &CudaBuffer, scale: f32) -> Result<(CudaBuffer, Vec<f32>), String> {
        let out = alloc_f32(1)?;
        let status =
            unsafe { lumen_cuda_sum_i8_device(input.handle(), scale, out.handle(), input.len()) };
        if status != 0 {
            return Err(last_error_message());
        }
        let host = download_f32(&out)?;
        Ok((out, host))
    }

    pub fn fill_scalar_f32(len: usize, value: f32) -> Result<(CudaBuffer, Vec<f32>), String> {
        let out = alloc_f32(len)?;
        let status = unsafe { lumen_cuda_fill_scalar_f32_device(out.handle(), len, value) };
        if status != 0 {
            return Err(last_error_message());
        }
        let host = download_f32(&out)?;
        Ok((out, host))
    }

    pub fn fill_scalar_f32_buffer(len: usize, value: f32) -> Result<CudaBuffer, String> {
        let out = alloc_f32(len)?;
        let status = unsafe { lumen_cuda_fill_scalar_f32_device(out.handle(), len, value) };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(out)
    }

    pub fn add_inplace_f32(dst: &CudaBuffer, src: &CudaBuffer) -> Result<(), String> {
        if dst.len() != src.len() {
            return Err(format!(
                "CUDA add_inplace length mismatch: dst={}, src={}",
                dst.len(),
                src.len()
            ));
        }
        let status =
            unsafe { lumen_cuda_add_inplace_f32_device(dst.handle(), src.handle(), dst.len()) };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(())
    }

    pub fn sum_lastdim_f32_buffer(
        input: &CudaBuffer,
        rows: usize,
        last_dim: usize,
    ) -> Result<CudaBuffer, String> {
        if rows == 0 || last_dim == 0 || input.len() != rows * last_dim {
            return Err(format!(
                "CUDA sum_lastdim length mismatch: input={}, rows={}, last_dim={}",
                input.len(),
                rows,
                last_dim
            ));
        }
        let out = alloc_f32(last_dim)?;
        let status = unsafe {
            lumen_cuda_sum_lastdim_f32_device(input.handle(), out.handle(), rows, last_dim)
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(out)
    }

    pub fn bshd_to_bhsd_add_bias_f32_buffer(
        input: &CudaBuffer,
        bias: &CudaBuffer,
        batch: usize,
        seq: usize,
        heads: usize,
        dim: usize,
    ) -> Result<CudaBuffer, String> {
        let len = batch
            .checked_mul(seq)
            .and_then(|v| v.checked_mul(heads))
            .and_then(|v| v.checked_mul(dim))
            .ok_or_else(|| "CUDA BSHD to BHSD add bias length overflow".to_string())?;
        let bias_len = heads
            .checked_mul(dim)
            .ok_or_else(|| "CUDA BSHD to BHSD add bias bias length overflow".to_string())?;
        if len == 0 || input.len() != len || bias.len() != bias_len {
            return Err(format!(
                "CUDA BSHD to BHSD add bias length mismatch: input={}, bias={}, batch={}, seq={}, heads={}, dim={}",
                input.len(),
                bias.len(),
                batch,
                seq,
                heads,
                dim
            ));
        }
        let out = alloc_f32(len)?;
        let status = unsafe {
            lumen_cuda_bshd_to_bhsd_add_bias_f32_device(
                input.handle(),
                bias.handle(),
                out.handle(),
                batch,
                seq,
                heads,
                dim,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(out)
    }

    pub fn mse_forward_typed(
        output: &CudaBuffer,
        output_dtype: DType,
        output_scale: Option<f32>,
        target: &CudaBuffer,
        target_dtype: DType,
        target_scale: Option<f32>,
    ) -> Result<(CudaBuffer, CudaBuffer, Vec<f32>), String> {
        if output.is_empty() {
            return Err("CUDA typed MSE expects at least one element".to_string());
        }
        if output.len() != target.len() {
            return Err(format!(
                "CUDA typed MSE length mismatch: output={}, target={}",
                output.len(),
                target.len()
            ));
        }
        let diff = alloc_f32(output.len())?;
        let loss = alloc_f32(1)?;
        let status = unsafe {
            lumen_cuda_mse_forward_typed_device(
                output.handle(),
                dtype_tag(output_dtype)?,
                output_scale.unwrap_or(1.0),
                target.handle(),
                dtype_tag(target_dtype)?,
                target_scale.unwrap_or(1.0),
                diff.handle(),
                loss.handle(),
                output.len(),
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        let host = download_f32(&loss)?;
        Ok((diff, loss, host))
    }

    pub fn mse_backward_f32(diff: &CudaBuffer, factor: f32) -> Result<CudaTwoHostBuffers, String> {
        let (grad_output, grad_target) = mse_backward_f32_buffers(diff, factor)?;
        let grad_output_host = download_f32(&grad_output)?;
        let grad_target_host = download_f32(&grad_target)?;
        Ok((
            (grad_output, grad_output_host),
            (grad_target, grad_target_host),
        ))
    }

    pub fn mse_backward_f32_buffers(
        diff: &CudaBuffer,
        factor: f32,
    ) -> Result<(CudaBuffer, CudaBuffer), String> {
        let grad_output = alloc_f32(diff.len())?;
        let grad_target = alloc_f32(diff.len())?;
        let status = unsafe {
            lumen_cuda_mse_backward_f32_device(
                diff.handle(),
                grad_output.handle(),
                grad_target.handle(),
                diff.len(),
                factor,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok((grad_output, grad_target))
    }

    pub fn cross_entropy_backward_f32(
        softmax: &CudaBuffer,
        target: &CudaBuffer,
        factor: f32,
    ) -> Result<(CudaBuffer, Vec<f32>), String> {
        let out = cross_entropy_backward_f32_buffer(softmax, target, factor)?;
        let host = download_f32(&out)?;
        Ok((out, host))
    }

    pub fn cross_entropy_backward_f32_buffer(
        softmax: &CudaBuffer,
        target: &CudaBuffer,
        factor: f32,
    ) -> Result<CudaBuffer, String> {
        if softmax.len() != target.len() {
            return Err(format!(
                "CUDA cross_entropy backward length mismatch: softmax={}, target={}",
                softmax.len(),
                target.len()
            ));
        }
        let out = alloc_f32(softmax.len())?;
        let status = unsafe {
            lumen_cuda_cross_entropy_backward_f32_device(
                softmax.handle(),
                target.handle(),
                out.handle(),
                softmax.len(),
                factor,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(out)
    }

    pub fn cross_entropy_loss_f32(
        softmax: &CudaBuffer,
        target: &CudaBuffer,
        batch_size: usize,
    ) -> Result<(CudaBuffer, Vec<f32>), String> {
        if batch_size == 0 {
            return Err("CUDA cross_entropy loss batch size must be greater than zero".to_string());
        }
        if softmax.len() != target.len() {
            return Err(format!(
                "CUDA cross_entropy loss length mismatch: softmax={}, target={}",
                softmax.len(),
                target.len()
            ));
        }
        let out = alloc_f32(1)?;
        let factor = 1.0 / batch_size as f32;
        let status = unsafe {
            lumen_cuda_cross_entropy_loss_f32_device(
                softmax.handle(),
                target.handle(),
                out.handle(),
                softmax.len(),
                factor,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        let host = download_f32(&out)?;
        Ok((out, host))
    }

    pub fn cross_entropy_backward_typed_target_buffer(
        softmax: &CudaBuffer,
        target: &CudaBuffer,
        target_dtype: DType,
        target_scale: Option<f32>,
        factor: f32,
    ) -> Result<CudaBuffer, String> {
        if softmax.len() != target.len() {
            return Err(format!(
                "CUDA typed target cross_entropy backward length mismatch: softmax={}, target={}",
                softmax.len(),
                target.len()
            ));
        }
        let out = alloc_f32(softmax.len())?;
        let status = unsafe {
            lumen_cuda_cross_entropy_backward_typed_target_device(
                softmax.handle(),
                target.handle(),
                dtype_tag(target_dtype)?,
                target_scale.unwrap_or(1.0),
                out.handle(),
                softmax.len(),
                factor,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(out)
    }

    pub fn cross_entropy_backward_typed_target(
        softmax: &CudaBuffer,
        target: &CudaBuffer,
        target_dtype: DType,
        target_scale: Option<f32>,
        factor: f32,
    ) -> Result<(CudaBuffer, Vec<f32>), String> {
        let out = cross_entropy_backward_typed_target_buffer(
            softmax,
            target,
            target_dtype,
            target_scale,
            factor,
        )?;
        let host = download_f32(&out)?;
        Ok((out, host))
    }

    pub fn cross_entropy_loss_typed_target(
        softmax: &CudaBuffer,
        target: &CudaBuffer,
        target_dtype: DType,
        target_scale: Option<f32>,
        batch_size: usize,
    ) -> Result<(CudaBuffer, Vec<f32>), String> {
        if batch_size == 0 {
            return Err(
                "CUDA typed target cross_entropy loss batch size must be greater than zero"
                    .to_string(),
            );
        }
        if softmax.len() != target.len() {
            return Err(format!(
                "CUDA typed target cross_entropy loss length mismatch: softmax={}, target={}",
                softmax.len(),
                target.len()
            ));
        }
        let out = alloc_f32(1)?;
        let factor = 1.0 / batch_size as f32;
        let status = unsafe {
            lumen_cuda_cross_entropy_loss_typed_target_device(
                softmax.handle(),
                target.handle(),
                dtype_tag(target_dtype)?,
                target_scale.unwrap_or(1.0),
                out.handle(),
                softmax.len(),
                factor,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        let host = download_f32(&out)?;
        Ok((out, host))
    }

    pub fn sgd_update_f32(
        param: &CudaBuffer,
        grad: &CudaBuffer,
        lr: f32,
    ) -> Result<Vec<f32>, String> {
        sgd_update_f32_no_host(param, grad, lr)?;
        download_f32(param)
    }

    pub fn sgd_update_f32_no_host(
        param: &CudaBuffer,
        grad: &CudaBuffer,
        lr: f32,
    ) -> Result<(), String> {
        if param.len() != grad.len() {
            return Err(format!(
                "CUDA SGD length mismatch: param={}, grad={}",
                param.len(),
                grad.len()
            ));
        }
        let status = unsafe {
            lumen_cuda_sgd_update_f32_device(param.handle(), grad.handle(), param.len(), lr)
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(())
    }

    pub fn sgd_update_f32_batched_no_host(
        params: &[CudaBuffer],
        grads: &[CudaBuffer],
        lr: f32,
    ) -> Result<(), String> {
        if params.len() != grads.len() {
            return Err(format!(
                "CUDA batched SGD count mismatch: params={}, grads={}",
                params.len(),
                grads.len()
            ));
        }
        if params.is_empty() {
            return Err("CUDA batched SGD needs at least one tensor".to_string());
        }
        let mut param_handles = Vec::with_capacity(params.len());
        let mut grad_handles = Vec::with_capacity(params.len());
        let mut lens = Vec::with_capacity(params.len());
        for (idx, (param, grad)) in params.iter().zip(grads.iter()).enumerate() {
            if param.len() != grad.len() {
                return Err(format!(
                    "CUDA batched SGD length mismatch at {idx}: param={}, grad={}",
                    param.len(),
                    grad.len()
                ));
            }
            param_handles.push(param.handle());
            grad_handles.push(grad.handle());
            lens.push(param.len());
        }
        let status = unsafe {
            lumen_cuda_sgd_update_f32_batched_device(
                param_handles.as_ptr(),
                grad_handles.as_ptr(),
                lens.as_ptr(),
                params.len(),
                lr,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(())
    }

    pub fn quantize_f32_storage_no_host(
        param: &CudaBuffer,
        dtype: crate::precision::DType,
        scale: Option<f32>,
    ) -> Result<(), String> {
        let dtype_id = match dtype {
            crate::precision::DType::F32 => return Ok(()),
            crate::precision::DType::F16 => 1,
            crate::precision::DType::BF16 => 2,
            crate::precision::DType::I8 => 3,
        };
        let scale = scale.unwrap_or(1.0);
        let status = unsafe {
            lumen_cuda_quantize_f32_storage_device(param.handle(), param.len(), dtype_id, scale)
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(())
    }

    pub fn quantize_f32_to_i8_dynamic_no_host(
        input: &CudaBuffer,
    ) -> Result<(CudaBuffer, f32), String> {
        let out = alloc_storage(input.len())?;
        let mut scale = 1.0f32;
        let status = unsafe {
            lumen_cuda_quantize_f32_to_i8_dynamic_device(
                input.handle(),
                out.handle(),
                input.len(),
                &mut scale as *mut f32,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok((out, scale))
    }

    pub fn f32_to_lowp_storage_no_host(
        input: &CudaBuffer,
        dtype: crate::precision::DType,
    ) -> Result<CudaBuffer, String> {
        if !matches!(
            dtype,
            crate::precision::DType::F16 | crate::precision::DType::BF16
        ) {
            return Err(format!(
                "CUDA f32 to lowp storage expects F16 or BF16 dtype, got {:?}",
                dtype
            ));
        }
        let out = alloc_storage(input.len())?;
        let status = unsafe {
            lumen_cuda_f32_to_lowp_storage_device(
                input.handle(),
                out.handle(),
                input.len(),
                dtype_tag(dtype)?,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(out)
    }

    pub fn sgd_momentum_update_f32(
        param: &CudaBuffer,
        grad: &CudaBuffer,
        velocity: &CudaBuffer,
        lr: f32,
        momentum: f32,
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        sgd_momentum_update_f32_no_host(param, grad, velocity, lr, momentum)?;
        Ok((download_f32(param)?, download_f32(velocity)?))
    }

    pub fn sgd_momentum_update_f32_no_host(
        param: &CudaBuffer,
        grad: &CudaBuffer,
        velocity: &CudaBuffer,
        lr: f32,
        momentum: f32,
    ) -> Result<(), String> {
        if param.len() != grad.len() || param.len() != velocity.len() {
            return Err(format!(
                "CUDA SGD momentum length mismatch: param={}, grad={}, velocity={}",
                param.len(),
                grad.len(),
                velocity.len()
            ));
        }
        let status = unsafe {
            lumen_cuda_sgd_momentum_update_f32_device(
                param.handle(),
                grad.handle(),
                velocity.handle(),
                param.len(),
                lr,
                momentum,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(())
    }

    pub fn sgd_momentum_update_f32_batched_no_host(
        params: &[CudaBuffer],
        grads: &[CudaBuffer],
        velocities: &[CudaBuffer],
        lr: f32,
        momentum: f32,
    ) -> Result<(), String> {
        if params.len() != grads.len() || params.len() != velocities.len() {
            return Err(format!(
                "CUDA batched SGD momentum count mismatch: params={}, grads={}, velocities={}",
                params.len(),
                grads.len(),
                velocities.len()
            ));
        }
        if params.is_empty() {
            return Err("CUDA batched SGD momentum needs at least one tensor".to_string());
        }
        let mut param_handles = Vec::with_capacity(params.len());
        let mut grad_handles = Vec::with_capacity(params.len());
        let mut velocity_handles = Vec::with_capacity(params.len());
        let mut lens = Vec::with_capacity(params.len());
        for (idx, ((param, grad), velocity)) in params
            .iter()
            .zip(grads.iter())
            .zip(velocities.iter())
            .enumerate()
        {
            if param.len() != grad.len() || param.len() != velocity.len() {
                return Err(format!(
                    "CUDA batched SGD momentum length mismatch at {idx}: param={}, grad={}, velocity={}",
                    param.len(),
                    grad.len(),
                    velocity.len()
                ));
            }
            param_handles.push(param.handle());
            grad_handles.push(grad.handle());
            velocity_handles.push(velocity.handle());
            lens.push(param.len());
        }
        let status = unsafe {
            lumen_cuda_sgd_momentum_update_f32_batched_device(
                param_handles.as_ptr(),
                grad_handles.as_ptr(),
                velocity_handles.as_ptr(),
                lens.as_ptr(),
                params.len(),
                lr,
                momentum,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn adam_update_f32(
        param: &CudaBuffer,
        grad: &CudaBuffer,
        exp_avg: &CudaBuffer,
        exp_avg_sq: &CudaBuffer,
        lr: f32,
        beta1: f32,
        beta2: f32,
        bias_correction1: f32,
        bias_correction2: f32,
        eps: f32,
    ) -> Result<CudaAdamHostState, String> {
        adam_update_f32_no_host(
            param,
            grad,
            exp_avg,
            exp_avg_sq,
            lr,
            beta1,
            beta2,
            bias_correction1,
            bias_correction2,
            eps,
        )?;
        Ok((
            download_f32(param)?,
            download_f32(exp_avg)?,
            download_f32(exp_avg_sq)?,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn adam_update_f32_no_host(
        param: &CudaBuffer,
        grad: &CudaBuffer,
        exp_avg: &CudaBuffer,
        exp_avg_sq: &CudaBuffer,
        lr: f32,
        beta1: f32,
        beta2: f32,
        bias_correction1: f32,
        bias_correction2: f32,
        eps: f32,
    ) -> Result<(), String> {
        if param.len() != grad.len()
            || param.len() != exp_avg.len()
            || param.len() != exp_avg_sq.len()
        {
            return Err(format!(
                "CUDA Adam length mismatch: param={}, grad={}, exp_avg={}, exp_avg_sq={}",
                param.len(),
                grad.len(),
                exp_avg.len(),
                exp_avg_sq.len()
            ));
        }
        let status = unsafe {
            lumen_cuda_adam_update_f32_device(
                param.handle(),
                grad.handle(),
                exp_avg.handle(),
                exp_avg_sq.handle(),
                param.len(),
                lr,
                beta1,
                beta2,
                bias_correction1,
                bias_correction2,
                eps,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn adam_update_f32_batched_no_host(
        params: &[CudaBuffer],
        grads: &[CudaBuffer],
        exp_avgs: &[CudaBuffer],
        exp_avg_sqs: &[CudaBuffer],
        lr: f32,
        beta1: f32,
        beta2: f32,
        bias_correction1: f32,
        bias_correction2: f32,
        eps: f32,
    ) -> Result<(), String> {
        if params.len() != grads.len()
            || params.len() != exp_avgs.len()
            || params.len() != exp_avg_sqs.len()
        {
            return Err(format!(
                "CUDA batched Adam count mismatch: params={}, grads={}, exp_avgs={}, exp_avg_sqs={}",
                params.len(),
                grads.len(),
                exp_avgs.len(),
                exp_avg_sqs.len()
            ));
        }
        if params.is_empty() {
            return Err("CUDA batched Adam needs at least one tensor".to_string());
        }
        let mut param_handles = Vec::with_capacity(params.len());
        let mut grad_handles = Vec::with_capacity(params.len());
        let mut exp_avg_handles = Vec::with_capacity(params.len());
        let mut exp_avg_sq_handles = Vec::with_capacity(params.len());
        let mut lens = Vec::with_capacity(params.len());
        for (idx, (((param, grad), exp_avg), exp_avg_sq)) in params
            .iter()
            .zip(grads.iter())
            .zip(exp_avgs.iter())
            .zip(exp_avg_sqs.iter())
            .enumerate()
        {
            if param.len() != grad.len()
                || param.len() != exp_avg.len()
                || param.len() != exp_avg_sq.len()
            {
                return Err(format!(
                    "CUDA batched Adam length mismatch at {idx}: param={}, grad={}, exp_avg={}, exp_avg_sq={}",
                    param.len(),
                    grad.len(),
                    exp_avg.len(),
                    exp_avg_sq.len()
                ));
            }
            param_handles.push(param.handle());
            grad_handles.push(grad.handle());
            exp_avg_handles.push(exp_avg.handle());
            exp_avg_sq_handles.push(exp_avg_sq.handle());
            lens.push(param.len());
        }
        let status = unsafe {
            lumen_cuda_adam_update_f32_batched_device(
                param_handles.as_ptr(),
                grad_handles.as_ptr(),
                exp_avg_handles.as_ptr(),
                exp_avg_sq_handles.as_ptr(),
                lens.as_ptr(),
                params.len(),
                lr,
                beta1,
                beta2,
                bias_correction1,
                bias_correction2,
                eps,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(())
    }

    pub fn softmax_lastdim_f32(
        input: &CudaBuffer,
        outer: usize,
        last_dim: usize,
    ) -> Result<(CudaBuffer, Vec<f32>), String> {
        if input.len() != outer * last_dim {
            return Err(format!(
                "CUDA softmax input length mismatch: expected {}, got {}",
                outer * last_dim,
                input.len()
            ));
        }
        let out = alloc_f32(input.len())?;
        let status = unsafe {
            lumen_cuda_softmax_lastdim_f32_device(input.handle(), out.handle(), outer, last_dim)
        };
        if status != 0 {
            return Err(last_error_message());
        }
        let host = download_f32(&out)?;
        Ok((out, host))
    }

    pub fn softmax_lastdim_f32_no_host(
        input: &CudaBuffer,
        outer: usize,
        last_dim: usize,
    ) -> Result<CudaBuffer, String> {
        if input.len() != outer * last_dim {
            return Err(format!(
                "CUDA softmax input length mismatch: expected {}, got {}",
                outer * last_dim,
                input.len()
            ));
        }
        let out = alloc_f32(input.len())?;
        let status = unsafe {
            lumen_cuda_softmax_lastdim_f32_device(input.handle(), out.handle(), outer, last_dim)
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(out)
    }

    pub fn softmax_lastdim_typed(
        input: &CudaBuffer,
        input_dtype: DType,
        input_scale: Option<f32>,
        outer: usize,
        last_dim: usize,
    ) -> Result<(CudaBuffer, Vec<f32>), String> {
        let out = softmax_lastdim_typed_no_host(input, input_dtype, input_scale, outer, last_dim)?;
        let host = download_f32(&out)?;
        Ok((out, host))
    }

    pub fn softmax_lastdim_typed_no_host(
        input: &CudaBuffer,
        input_dtype: DType,
        input_scale: Option<f32>,
        outer: usize,
        last_dim: usize,
    ) -> Result<CudaBuffer, String> {
        let expected_len = outer
            .checked_mul(last_dim)
            .ok_or_else(|| "CUDA typed softmax input length overflow".to_string())?;
        if input.len() != expected_len {
            return Err(format!(
                "CUDA typed softmax input length mismatch: expected {}, got {}",
                expected_len,
                input.len()
            ));
        }
        let out = alloc_f32(expected_len)?;
        let status = unsafe {
            lumen_cuda_softmax_lastdim_typed_device(
                input.handle(),
                dtype_tag(input_dtype)?,
                input_scale.unwrap_or(1.0),
                out.handle(),
                outer,
                last_dim,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(out)
    }

    pub fn softmax_lastdim_backward_f32(
        output: &CudaBuffer,
        grad: &CudaBuffer,
        outer: usize,
        last_dim: usize,
    ) -> Result<(CudaBuffer, Vec<f32>), String> {
        let out = softmax_lastdim_backward_f32_buffer(output, grad, outer, last_dim)?;
        let host = download_f32(&out)?;
        Ok((out, host))
    }

    pub fn softmax_lastdim_backward_f32_buffer(
        output: &CudaBuffer,
        grad: &CudaBuffer,
        outer: usize,
        last_dim: usize,
    ) -> Result<CudaBuffer, String> {
        let len = outer
            .checked_mul(last_dim)
            .ok_or_else(|| "CUDA softmax backward length overflow".to_string())?;
        if output.len() != len || grad.len() != len {
            return Err(format!(
                "CUDA softmax backward length mismatch: expected {}, output={}, grad={}",
                len,
                output.len(),
                grad.len()
            ));
        }
        let out = alloc_f32(len)?;
        let status = unsafe {
            lumen_cuda_softmax_lastdim_backward_f32_device(
                output.handle(),
                grad.handle(),
                out.handle(),
                outer,
                last_dim,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(out)
    }

    pub fn fused_softmax_f32(
        input: &CudaBuffer,
        batch_heads: usize,
        q_len: usize,
        k_len: usize,
        scale: f32,
        is_causal: bool,
    ) -> Result<(CudaBuffer, Vec<f32>), String> {
        let out = fused_softmax_f32_no_host(input, batch_heads, q_len, k_len, scale, is_causal)?;
        let host = download_f32(&out)?;
        Ok((out, host))
    }

    pub fn fused_softmax_f32_no_host(
        input: &CudaBuffer,
        batch_heads: usize,
        q_len: usize,
        k_len: usize,
        scale: f32,
        is_causal: bool,
    ) -> Result<CudaBuffer, String> {
        ensure_finite("CUDA fused_softmax scale", scale)?;
        let expected_len = batch_heads
            .checked_mul(q_len)
            .and_then(|value| value.checked_mul(k_len))
            .ok_or_else(|| "CUDA fused_softmax input length overflow".to_string())?;
        if input.len() != expected_len {
            return Err(format!(
                "CUDA fused_softmax input length mismatch: expected {}, got {}",
                expected_len,
                input.len()
            ));
        }
        let out = alloc_f32(expected_len)?;
        let status = unsafe {
            lumen_cuda_fused_softmax_f32_device(
                input.handle(),
                out.handle(),
                batch_heads,
                q_len,
                k_len,
                scale,
                if is_causal { 1 } else { 0 },
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(out)
    }

    pub fn fused_softmax_backward_f32(
        output: &CudaBuffer,
        grad: &CudaBuffer,
        batch_heads: usize,
        q_len: usize,
        k_len: usize,
        scale: f32,
    ) -> Result<(CudaBuffer, Vec<f32>), String> {
        let out =
            fused_softmax_backward_f32_buffer(output, grad, batch_heads, q_len, k_len, scale)?;
        let host = download_f32(&out)?;
        Ok((out, host))
    }

    pub fn fused_softmax_backward_f32_buffer(
        output: &CudaBuffer,
        grad: &CudaBuffer,
        batch_heads: usize,
        q_len: usize,
        k_len: usize,
        scale: f32,
    ) -> Result<CudaBuffer, String> {
        ensure_finite("CUDA fused_softmax backward scale", scale)?;
        let expected_len = batch_heads
            .checked_mul(q_len)
            .and_then(|value| value.checked_mul(k_len))
            .ok_or_else(|| "CUDA fused_softmax backward input length overflow".to_string())?;
        if output.len() != expected_len || grad.len() != expected_len {
            return Err(format!(
                "CUDA fused_softmax backward length mismatch: expected {}, output={}, grad={}",
                expected_len,
                output.len(),
                grad.len()
            ));
        }
        let out = alloc_f32(expected_len)?;
        let status = unsafe {
            lumen_cuda_fused_softmax_backward_f32_device(
                output.handle(),
                grad.handle(),
                out.handle(),
                batch_heads,
                q_len,
                k_len,
                scale,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(out)
    }

    pub fn fused_softmax_f32_with_past(
        input: &CudaBuffer,
        batch_heads: usize,
        q_len: usize,
        k_len: usize,
        scale: f32,
        is_causal: bool,
        past_len: usize,
    ) -> Result<(CudaBuffer, Vec<f32>), String> {
        let out = fused_softmax_f32_with_past_no_host(
            input,
            batch_heads,
            q_len,
            k_len,
            scale,
            is_causal,
            past_len,
        )?;
        let host = download_f32(&out)?;
        Ok((out, host))
    }

    pub fn fused_softmax_f32_with_past_no_host(
        input: &CudaBuffer,
        batch_heads: usize,
        q_len: usize,
        k_len: usize,
        scale: f32,
        is_causal: bool,
        past_len: usize,
    ) -> Result<CudaBuffer, String> {
        ensure_finite("CUDA fused_softmax_with_past scale", scale)?;
        let expected_len = batch_heads
            .checked_mul(q_len)
            .and_then(|value| value.checked_mul(k_len))
            .ok_or_else(|| "CUDA fused_softmax_with_past input length overflow".to_string())?;
        if input.len() != expected_len {
            return Err(format!(
                "CUDA fused_softmax_with_past input length mismatch: expected {}, got {}",
                expected_len,
                input.len()
            ));
        }
        if batch_heads == 0 || q_len == 0 || k_len == 0 {
            return Err(
                "CUDA fused_softmax_with_past dimensions must be greater than zero".to_string(),
            );
        }
        let causal_window_end = past_len
            .checked_add(q_len)
            .ok_or_else(|| "CUDA fused_softmax_with_past causal window overflow".to_string())?;
        if causal_window_end > k_len {
            return Err(format!(
                "CUDA fused_softmax_with_past causal window out of bounds: past_len({}) + q_len({}) > k_len({})",
                past_len, q_len, k_len
            ));
        }
        let out = alloc_f32(expected_len)?;
        let status = unsafe {
            lumen_cuda_fused_softmax_f32_with_past_device(
                input.handle(),
                out.handle(),
                batch_heads,
                q_len,
                k_len,
                scale,
                if is_causal { 1 } else { 0 },
                past_len,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(out)
    }

    pub fn embedding_f32(
        indices: &CudaBuffer,
        weight: &CudaBuffer,
        num_indices: usize,
        vocab_size: usize,
        embed_dim: usize,
    ) -> Result<(CudaBuffer, Vec<f32>), String> {
        let out = embedding_f32_buffer(indices, weight, num_indices, vocab_size, embed_dim)?;
        let host = download_f32(&out)?;
        Ok((out, host))
    }

    pub fn embedding_f32_buffer(
        indices: &CudaBuffer,
        weight: &CudaBuffer,
        num_indices: usize,
        vocab_size: usize,
        embed_dim: usize,
    ) -> Result<CudaBuffer, String> {
        if indices.len() != num_indices {
            return Err(format!(
                "CUDA embedding indices length mismatch: expected {}, got {}",
                num_indices,
                indices.len()
            ));
        }
        let weight_len = checked_len("CUDA embedding weight length", &[vocab_size, embed_dim])?;
        let output_len = checked_len("CUDA embedding output length", &[num_indices, embed_dim])?;
        if weight.len() != weight_len {
            return Err(format!(
                "CUDA embedding weight length mismatch: expected {}, got {}",
                weight_len,
                weight.len()
            ));
        }
        let out = alloc_f32(output_len)?;
        let status = unsafe {
            lumen_cuda_embedding_f32_device(
                indices.handle(),
                weight.handle(),
                out.handle(),
                num_indices,
                vocab_size,
                embed_dim,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(out)
    }

    pub fn embedding_typed(
        indices: &CudaBuffer,
        weight: &CudaBuffer,
        weight_dtype: DType,
        weight_scale: Option<f32>,
        num_indices: usize,
        vocab_size: usize,
        embed_dim: usize,
    ) -> Result<(CudaBuffer, Vec<f32>), String> {
        let out = embedding_typed_buffer(
            indices,
            weight,
            weight_dtype,
            weight_scale,
            num_indices,
            vocab_size,
            embed_dim,
        )?;
        let host = download_f32(&out)?;
        Ok((out, host))
    }

    pub fn embedding_typed_buffer(
        indices: &CudaBuffer,
        weight: &CudaBuffer,
        weight_dtype: DType,
        weight_scale: Option<f32>,
        num_indices: usize,
        vocab_size: usize,
        embed_dim: usize,
    ) -> Result<CudaBuffer, String> {
        if indices.len() != num_indices {
            return Err(format!(
                "CUDA typed embedding indices length mismatch: expected {}, got {}",
                num_indices,
                indices.len()
            ));
        }
        let weight_len = checked_len(
            "CUDA typed embedding weight length",
            &[vocab_size, embed_dim],
        )?;
        let output_len = checked_len(
            "CUDA typed embedding output length",
            &[num_indices, embed_dim],
        )?;
        if weight.len() != weight_len {
            return Err(format!(
                "CUDA typed embedding weight length mismatch: expected {}, got {}",
                weight_len,
                weight.len()
            ));
        }
        let out = alloc_f32(output_len)?;
        let status = unsafe {
            lumen_cuda_embedding_typed_device(
                indices.handle(),
                weight.handle(),
                dtype_tag(weight_dtype)?,
                weight_scale.unwrap_or(1.0),
                out.handle(),
                num_indices,
                vocab_size,
                embed_dim,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(out)
    }

    pub fn embedding_typed_same_dtype_buffer(
        indices: &CudaBuffer,
        weight: &CudaBuffer,
        weight_dtype: DType,
        num_indices: usize,
        vocab_size: usize,
        embed_dim: usize,
    ) -> Result<CudaBuffer, String> {
        if !matches!(weight_dtype, DType::F16 | DType::BF16 | DType::I8) {
            return Err(format!(
                "CUDA native embedding output only supports f16/bf16/i8, got {weight_dtype:?}"
            ));
        }
        if indices.len() != num_indices {
            return Err(format!(
                "CUDA native embedding indices length mismatch: expected {}, got {}",
                num_indices,
                indices.len()
            ));
        }
        let weight_len = checked_len(
            "CUDA native embedding weight length",
            &[vocab_size, embed_dim],
        )?;
        let output_len = checked_len(
            "CUDA native embedding output length",
            &[num_indices, embed_dim],
        )?;
        if weight.len() != weight_len {
            return Err(format!(
                "CUDA native embedding weight length mismatch: expected {}, got {}",
                weight_len,
                weight.len()
            ));
        }
        let out = alloc_storage(output_len)?;
        let status = unsafe {
            lumen_cuda_embedding_typed_same_dtype_device(
                indices.handle(),
                weight.handle(),
                dtype_tag(weight_dtype)?,
                out.handle(),
                num_indices,
                vocab_size,
                embed_dim,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(out)
    }

    pub fn embedding_backward_f32(
        indices: &CudaBuffer,
        grad: &CudaBuffer,
        num_indices: usize,
        vocab_size: usize,
        embed_dim: usize,
    ) -> Result<(CudaBuffer, Vec<f32>), String> {
        let grad_weight =
            embedding_backward_f32_buffer(indices, grad, num_indices, vocab_size, embed_dim)?;
        let host = download_f32(&grad_weight)?;
        Ok((grad_weight, host))
    }

    pub fn embedding_backward_f32_buffer(
        indices: &CudaBuffer,
        grad: &CudaBuffer,
        num_indices: usize,
        vocab_size: usize,
        embed_dim: usize,
    ) -> Result<CudaBuffer, String> {
        if indices.len() != num_indices {
            return Err(format!(
                "CUDA embedding backward indices length mismatch: expected {}, got {}",
                num_indices,
                indices.len()
            ));
        }
        let grad_len = checked_len(
            "CUDA embedding backward grad length",
            &[num_indices, embed_dim],
        )?;
        let grad_weight_len = checked_len(
            "CUDA embedding backward weight grad length",
            &[vocab_size, embed_dim],
        )?;
        if grad.len() != grad_len {
            return Err(format!(
                "CUDA embedding backward grad length mismatch: expected {}, got {}",
                grad_len,
                grad.len()
            ));
        }
        let grad_weight = alloc_f32(grad_weight_len)?;
        let status = unsafe {
            lumen_cuda_embedding_backward_f32_device(
                indices.handle(),
                grad.handle(),
                grad_weight.handle(),
                num_indices,
                vocab_size,
                embed_dim,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(grad_weight)
    }

    pub fn rms_norm_f32(
        input: &CudaBuffer,
        weight: &CudaBuffer,
        rows: usize,
        dim: usize,
        eps: f32,
    ) -> Result<(CudaBuffer, Vec<f32>), String> {
        let out = rms_norm_f32_buffer(input, weight, rows, dim, eps)?;
        let host = download_f32(&out)?;
        Ok((out, host))
    }

    pub fn rms_norm_f32_buffer(
        input: &CudaBuffer,
        weight: &CudaBuffer,
        rows: usize,
        dim: usize,
        eps: f32,
    ) -> Result<CudaBuffer, String> {
        ensure_positive_finite("CUDA RMSNorm epsilon", eps)?;
        let len = checked_len("CUDA RMSNorm length", &[rows, dim])?;
        if input.len() != len {
            return Err(format!(
                "CUDA RMSNorm input length mismatch: expected {}, got {}",
                len,
                input.len()
            ));
        }
        if weight.len() != dim {
            return Err(format!(
                "CUDA RMSNorm weight length mismatch: expected {}, got {}",
                dim,
                weight.len()
            ));
        }
        let out = alloc_f32(len)?;
        let status = unsafe {
            lumen_cuda_rms_norm_f32_device(
                input.handle(),
                weight.handle(),
                out.handle(),
                rows,
                dim,
                eps,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn rms_norm_typed(
        input: &CudaBuffer,
        input_dtype: DType,
        input_scale: Option<f32>,
        weight: &CudaBuffer,
        weight_dtype: DType,
        weight_scale: Option<f32>,
        rows: usize,
        dim: usize,
        eps: f32,
    ) -> Result<(CudaBuffer, Vec<f32>), String> {
        let out = rms_norm_typed_buffer(
            input,
            input_dtype,
            input_scale,
            weight,
            weight_dtype,
            weight_scale,
            rows,
            dim,
            eps,
        )?;
        let host = download_f32(&out)?;
        Ok((out, host))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn rms_norm_typed_buffer(
        input: &CudaBuffer,
        input_dtype: DType,
        input_scale: Option<f32>,
        weight: &CudaBuffer,
        weight_dtype: DType,
        weight_scale: Option<f32>,
        rows: usize,
        dim: usize,
        eps: f32,
    ) -> Result<CudaBuffer, String> {
        ensure_positive_finite("CUDA typed RMSNorm epsilon", eps)?;
        let len = rows
            .checked_mul(dim)
            .ok_or_else(|| "CUDA typed RMSNorm length overflow".to_string())?;
        if input.len() != len {
            return Err(format!(
                "CUDA typed RMSNorm input length mismatch: expected {}, got {}",
                len,
                input.len()
            ));
        }
        if weight.len() != dim {
            return Err(format!(
                "CUDA typed RMSNorm weight length mismatch: expected {}, got {}",
                dim,
                weight.len()
            ));
        }
        let out = alloc_f32(len)?;
        let status = unsafe {
            lumen_cuda_rms_norm_typed_device(
                input.handle(),
                dtype_tag(input_dtype)?,
                input_scale.unwrap_or(1.0),
                weight.handle(),
                dtype_tag(weight_dtype)?,
                weight_scale.unwrap_or(1.0),
                out.handle(),
                rows,
                dim,
                eps,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn rms_norm_i8_typed_output_buffer(
        input: &CudaBuffer,
        input_dtype: DType,
        input_scale: Option<f32>,
        weight: &CudaBuffer,
        weight_dtype: DType,
        weight_scale: Option<f32>,
        rows: usize,
        dim: usize,
        eps: f32,
    ) -> Result<(CudaBuffer, f32), String> {
        ensure_positive_finite("CUDA I8 typed-output RMSNorm epsilon", eps)?;
        let len = rows
            .checked_mul(dim)
            .ok_or_else(|| "CUDA I8 typed-output RMSNorm length overflow".to_string())?;
        if input.len() != len {
            return Err(format!(
                "CUDA I8 typed-output RMSNorm input length mismatch: expected {}, got {}",
                len,
                input.len()
            ));
        }
        if weight.len() != dim {
            return Err(format!(
                "CUDA I8 typed-output RMSNorm weight length mismatch: expected {}, got {}",
                dim,
                weight.len()
            ));
        }
        let out = alloc_storage(len)?;
        let mut out_scale = 1.0f32;
        let status = unsafe {
            lumen_cuda_rms_norm_i8_typed_out_device(
                input.handle(),
                dtype_tag(input_dtype)?,
                input_scale.unwrap_or(1.0),
                weight.handle(),
                dtype_tag(weight_dtype)?,
                weight_scale.unwrap_or(1.0),
                out.handle(),
                rows,
                dim,
                eps,
                &mut out_scale as *mut f32,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok((out, out_scale))
    }

    pub fn rms_norm_backward_f32(
        input: &CudaBuffer,
        weight: &CudaBuffer,
        grad: &CudaBuffer,
        rows: usize,
        dim: usize,
        eps: f32,
    ) -> Result<CudaTwoHostBuffers, String> {
        let (grad_input, grad_weight) =
            rms_norm_backward_f32_buffers(input, weight, grad, rows, dim, eps)?;
        let grad_input_host = download_f32(&grad_input)?;
        let grad_weight_host = download_f32(&grad_weight)?;
        Ok((
            (grad_input, grad_input_host),
            (grad_weight, grad_weight_host),
        ))
    }

    pub fn rms_norm_backward_f32_buffers(
        input: &CudaBuffer,
        weight: &CudaBuffer,
        grad: &CudaBuffer,
        rows: usize,
        dim: usize,
        eps: f32,
    ) -> Result<(CudaBuffer, CudaBuffer), String> {
        ensure_positive_finite("CUDA RMSNorm backward epsilon", eps)?;
        let len = rows
            .checked_mul(dim)
            .ok_or_else(|| "CUDA RMSNorm backward length overflow".to_string())?;
        if input.len() != len || grad.len() != len {
            return Err(format!(
                "CUDA RMSNorm backward input/grad length mismatch: expected {}, input={}, grad={}",
                len,
                input.len(),
                grad.len()
            ));
        }
        if weight.len() != dim {
            return Err(format!(
                "CUDA RMSNorm backward weight length mismatch: expected {}, got {}",
                dim,
                weight.len()
            ));
        }
        let grad_input = alloc_f32(len)?;
        let grad_weight = alloc_f32(dim)?;
        let status = unsafe {
            lumen_cuda_rms_norm_backward_f32_device(
                input.handle(),
                weight.handle(),
                grad.handle(),
                grad_input.handle(),
                grad_weight.handle(),
                rows,
                dim,
                eps,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok((grad_input, grad_weight))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn rms_norm_backward_typed(
        input: &CudaBuffer,
        input_dtype: DType,
        input_scale: Option<f32>,
        weight: &CudaBuffer,
        weight_dtype: DType,
        weight_scale: Option<f32>,
        grad: &CudaBuffer,
        rows: usize,
        dim: usize,
        eps: f32,
    ) -> Result<CudaTwoHostBuffers, String> {
        let (grad_input, grad_weight) = rms_norm_backward_typed_buffers(
            input,
            input_dtype,
            input_scale,
            weight,
            weight_dtype,
            weight_scale,
            grad,
            rows,
            dim,
            eps,
        )?;
        let grad_input_host = download_f32(&grad_input)?;
        let grad_weight_host = download_f32(&grad_weight)?;
        Ok((
            (grad_input, grad_input_host),
            (grad_weight, grad_weight_host),
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn rms_norm_backward_typed_buffers(
        input: &CudaBuffer,
        input_dtype: DType,
        input_scale: Option<f32>,
        weight: &CudaBuffer,
        weight_dtype: DType,
        weight_scale: Option<f32>,
        grad: &CudaBuffer,
        rows: usize,
        dim: usize,
        eps: f32,
    ) -> Result<(CudaBuffer, CudaBuffer), String> {
        ensure_positive_finite("CUDA typed RMSNorm backward epsilon", eps)?;
        let len = rows
            .checked_mul(dim)
            .ok_or_else(|| "CUDA typed RMSNorm backward length overflow".to_string())?;
        if input.len() != len || grad.len() != len {
            return Err(format!(
                "CUDA typed RMSNorm backward input/grad length mismatch: expected {}, input={}, grad={}",
                len,
                input.len(),
                grad.len()
            ));
        }
        if weight.len() != dim {
            return Err(format!(
                "CUDA typed RMSNorm backward weight length mismatch: expected {}, got {}",
                dim,
                weight.len()
            ));
        }
        let grad_input = alloc_f32(len)?;
        let grad_weight = alloc_f32(dim)?;
        let status = unsafe {
            lumen_cuda_rms_norm_backward_typed_device(
                input.handle(),
                dtype_tag(input_dtype)?,
                input_scale.unwrap_or(1.0),
                weight.handle(),
                dtype_tag(weight_dtype)?,
                weight_scale.unwrap_or(1.0),
                grad.handle(),
                grad_input.handle(),
                grad_weight.handle(),
                rows,
                dim,
                eps,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok((grad_input, grad_weight))
    }

    fn permute_metadata(
        input: &CudaBuffer,
        out_shape: &[usize],
        axes: &[usize],
    ) -> Result<(usize, Vec<usize>, Vec<usize>), String> {
        let ndim = out_shape.len();
        if axes.len() != ndim {
            return Err(format!(
                "CUDA permute axes length mismatch: axes={}, ndim={}",
                axes.len(),
                ndim
            ));
        }
        if ndim == 0 {
            return Err("CUDA permute expects at least 1 dimension".to_string());
        }

        let len = checked_len("CUDA permute output length", out_shape)?;
        if input.len() != len {
            return Err(format!(
                "CUDA permute input length mismatch: expected {}, got {}",
                len,
                input.len()
            ));
        }

        let mut seen = vec![false; ndim];
        for &axis in axes {
            if axis >= ndim || seen[axis] {
                return Err(format!("CUDA permute axes are invalid: {:?}", axes));
            }
            seen[axis] = true;
        }

        let mut input_shape = vec![0usize; ndim];
        for (out_dim, &input_axis) in axes.iter().enumerate() {
            input_shape[input_axis] = out_shape[out_dim];
        }

        let mut input_strides = vec![0usize; ndim];
        let mut stride = 1usize;
        for i in (0..ndim).rev() {
            input_strides[i] = stride;
            stride = stride
                .checked_mul(input_shape[i])
                .ok_or_else(|| "CUDA permute stride overflow".to_string())?;
        }

        let mut out_strides = vec![0usize; ndim];
        stride = 1usize;
        for i in (0..ndim).rev() {
            out_strides[i] = stride;
            stride = stride
                .checked_mul(out_shape[i])
                .ok_or_else(|| "CUDA permute output stride overflow".to_string())?;
        }

        let mapped_input_strides = axes
            .iter()
            .map(|&axis| input_strides[axis])
            .collect::<Vec<_>>();

        Ok((len, out_strides, mapped_input_strides))
    }

    pub fn permute_f32(
        input: &CudaBuffer,
        out_shape: &[usize],
        axes: &[usize],
    ) -> Result<(CudaBuffer, Vec<f32>), String> {
        let ndim = out_shape.len();
        if axes.len() != ndim {
            return Err(format!(
                "CUDA permute axes length mismatch: axes={}, ndim={}",
                axes.len(),
                ndim
            ));
        }
        if ndim == 0 {
            return Err("CUDA permute expects at least 1 dimension".to_string());
        }

        let len = checked_len("CUDA permute output length", out_shape)?;
        if input.len() != len {
            return Err(format!(
                "CUDA permute input length mismatch: expected {}, got {}",
                len,
                input.len()
            ));
        }

        let mut seen = vec![false; ndim];
        for &axis in axes {
            if axis >= ndim || seen[axis] {
                return Err(format!("CUDA permute axes are invalid: {:?}", axes));
            }
            seen[axis] = true;
        }

        let mut input_shape = vec![0usize; ndim];
        for (out_dim, &input_axis) in axes.iter().enumerate() {
            input_shape[input_axis] = out_shape[out_dim];
        }

        let mut input_strides = vec![0usize; ndim];
        let mut stride = 1usize;
        for i in (0..ndim).rev() {
            input_strides[i] = stride;
            stride = stride
                .checked_mul(input_shape[i])
                .ok_or_else(|| "CUDA permute stride overflow".to_string())?;
        }

        let mut out_strides = vec![0usize; ndim];
        stride = 1usize;
        for i in (0..ndim).rev() {
            out_strides[i] = stride;
            stride = stride
                .checked_mul(out_shape[i])
                .ok_or_else(|| "CUDA permute output stride overflow".to_string())?;
        }

        let mapped_input_strides = axes
            .iter()
            .map(|&axis| input_strides[axis])
            .collect::<Vec<_>>();
        let out = alloc_f32(len)?;
        let status = unsafe {
            lumen_cuda_permute_f32_device(
                input.handle(),
                out.handle(),
                ndim,
                out_shape.as_ptr(),
                out_strides.as_ptr(),
                mapped_input_strides.as_ptr(),
                len,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        let host = download_f32(&out)?;
        Ok((out, host))
    }

    pub fn permute_f32_buffer(
        input: &CudaBuffer,
        out_shape: &[usize],
        axes: &[usize],
    ) -> Result<CudaBuffer, String> {
        let ndim = out_shape.len();
        if axes.len() != ndim {
            return Err(format!(
                "CUDA permute axes length mismatch: axes={}, ndim={}",
                axes.len(),
                ndim
            ));
        }
        if ndim == 0 {
            return Err("CUDA permute expects at least 1 dimension".to_string());
        }

        let len = checked_len("CUDA permute output length", out_shape)?;
        if input.len() != len {
            return Err(format!(
                "CUDA permute input length mismatch: expected {}, got {}",
                len,
                input.len()
            ));
        }

        let mut seen = vec![false; ndim];
        for &axis in axes {
            if axis >= ndim || seen[axis] {
                return Err(format!("CUDA permute axes are invalid: {:?}", axes));
            }
            seen[axis] = true;
        }

        let mut input_shape = vec![0usize; ndim];
        for (out_dim, &input_axis) in axes.iter().enumerate() {
            input_shape[input_axis] = out_shape[out_dim];
        }

        let mut input_strides = vec![0usize; ndim];
        let mut stride = 1usize;
        for i in (0..ndim).rev() {
            input_strides[i] = stride;
            stride = stride
                .checked_mul(input_shape[i])
                .ok_or_else(|| "CUDA permute stride overflow".to_string())?;
        }

        let mut out_strides = vec![0usize; ndim];
        stride = 1usize;
        for i in (0..ndim).rev() {
            out_strides[i] = stride;
            stride = stride
                .checked_mul(out_shape[i])
                .ok_or_else(|| "CUDA permute output stride overflow".to_string())?;
        }

        let mapped_input_strides = axes
            .iter()
            .map(|&axis| input_strides[axis])
            .collect::<Vec<_>>();
        let out = alloc_f32(len)?;
        let status = unsafe {
            lumen_cuda_permute_f32_device(
                input.handle(),
                out.handle(),
                ndim,
                out_shape.as_ptr(),
                out_strides.as_ptr(),
                mapped_input_strides.as_ptr(),
                len,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(out)
    }

    pub fn permute_typed_buffer(
        input: &CudaBuffer,
        dtype: DType,
        out_shape: &[usize],
        axes: &[usize],
    ) -> Result<CudaBuffer, String> {
        let ndim = out_shape.len();
        let (len, out_strides, mapped_input_strides) = permute_metadata(input, out_shape, axes)?;
        let out = alloc_storage(len)?;
        let status = unsafe {
            lumen_cuda_permute_typed_device(
                input.handle(),
                dtype_tag(dtype)?,
                out.handle(),
                ndim,
                out_shape.as_ptr(),
                out_strides.as_ptr(),
                mapped_input_strides.as_ptr(),
                len,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(out)
    }

    pub fn slice_lastdim_f32(
        input: &CudaBuffer,
        outer: usize,
        input_last_dim: usize,
        start: usize,
        slice_len: usize,
    ) -> Result<(CudaBuffer, Vec<f32>), String> {
        let expected_len = outer
            .checked_mul(input_last_dim)
            .ok_or_else(|| "CUDA slice length overflow".to_string())?;
        if input.len() != expected_len {
            return Err(format!(
                "CUDA slice input length mismatch: expected {}, got {}",
                expected_len,
                input.len()
            ));
        }
        if !range_fits(start, slice_len, input_last_dim) {
            return Err(format!(
                "CUDA slice range out of bounds: start={}, len={}, input_last_dim={}",
                start, slice_len, input_last_dim
            ));
        }
        let out = alloc_f32(
            outer
                .checked_mul(slice_len)
                .ok_or_else(|| "CUDA slice output length overflow".to_string())?,
        )?;
        let status = unsafe {
            lumen_cuda_slice_lastdim_f32_device(
                input.handle(),
                out.handle(),
                outer,
                input_last_dim,
                start,
                slice_len,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        let host = download_f32(&out)?;
        Ok((out, host))
    }

    pub fn slice_lastdim_f32_buffer(
        input: &CudaBuffer,
        outer: usize,
        input_last_dim: usize,
        start: usize,
        slice_len: usize,
    ) -> Result<CudaBuffer, String> {
        let expected_len = outer
            .checked_mul(input_last_dim)
            .ok_or_else(|| "CUDA slice length overflow".to_string())?;
        if input.len() != expected_len {
            return Err(format!(
                "CUDA slice input length mismatch: expected {}, got {}",
                expected_len,
                input.len()
            ));
        }
        if !range_fits(start, slice_len, input_last_dim) {
            return Err(format!(
                "CUDA slice range out of bounds: start={}, len={}, input_last_dim={}",
                start, slice_len, input_last_dim
            ));
        }
        let out = alloc_f32(
            outer
                .checked_mul(slice_len)
                .ok_or_else(|| "CUDA slice output length overflow".to_string())?,
        )?;
        let status = unsafe {
            lumen_cuda_slice_lastdim_f32_device(
                input.handle(),
                out.handle(),
                outer,
                input_last_dim,
                start,
                slice_len,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(out)
    }

    pub fn slice_lastdim_typed_buffer(
        input: &CudaBuffer,
        dtype: DType,
        outer: usize,
        input_last_dim: usize,
        start: usize,
        slice_len: usize,
    ) -> Result<CudaBuffer, String> {
        let expected = outer
            .checked_mul(input_last_dim)
            .ok_or_else(|| "CUDA typed slice input length overflow".to_string())?;
        if input.len() != expected {
            return Err(format!(
                "CUDA typed slice input length mismatch: expected {}, got {}",
                expected,
                input.len()
            ));
        }
        if outer == 0 || input_last_dim == 0 || slice_len == 0 {
            return Err("CUDA typed slice dimensions must be greater than zero".to_string());
        }
        if !range_fits(start, slice_len, input_last_dim) {
            return Err(format!(
                "CUDA typed slice range out of bounds: start={}, slice_len={}, input_last_dim={}",
                start, slice_len, input_last_dim
            ));
        }
        let out_len = outer
            .checked_mul(slice_len)
            .ok_or_else(|| "CUDA typed slice output length overflow".to_string())?;
        let out = alloc_storage(out_len)?;
        let status = unsafe {
            lumen_cuda_slice_lastdim_typed_device(
                input.handle(),
                dtype_tag(dtype)?,
                out.handle(),
                outer,
                input_last_dim,
                start,
                slice_len,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(out)
    }

    pub fn slice_lastdim_backward_f32(
        grad: &CudaBuffer,
        outer: usize,
        input_last_dim: usize,
        start: usize,
        slice_len: usize,
    ) -> Result<(CudaBuffer, Vec<f32>), String> {
        let out = slice_lastdim_backward_f32_buffer(grad, outer, input_last_dim, start, slice_len)?;
        let host = download_f32(&out)?;
        Ok((out, host))
    }

    pub fn slice_lastdim_backward_f32_buffer(
        grad: &CudaBuffer,
        outer: usize,
        input_last_dim: usize,
        start: usize,
        slice_len: usize,
    ) -> Result<CudaBuffer, String> {
        let grad_len = outer
            .checked_mul(slice_len)
            .ok_or_else(|| "CUDA slice backward grad length overflow".to_string())?;
        if grad.len() != grad_len {
            return Err(format!(
                "CUDA slice backward grad length mismatch: expected {}, got {}",
                grad_len,
                grad.len()
            ));
        }
        if !range_fits(start, slice_len, input_last_dim) {
            return Err(format!(
                "CUDA slice backward range out of bounds: start={}, len={}, input_last_dim={}",
                start, slice_len, input_last_dim
            ));
        }
        let out_len = outer
            .checked_mul(input_last_dim)
            .ok_or_else(|| "CUDA slice backward output length overflow".to_string())?;
        let out = alloc_f32(out_len)?;
        let status = unsafe {
            lumen_cuda_slice_lastdim_backward_f32_device(
                grad.handle(),
                out.handle(),
                outer,
                input_last_dim,
                start,
                slice_len,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(out)
    }

    pub fn cat_f32(
        lhs: &CudaBuffer,
        rhs: &CudaBuffer,
        out_shape: &[usize],
        axis: usize,
        lhs_axis_len: usize,
    ) -> Result<(CudaBuffer, Vec<f32>), String> {
        let ndim = out_shape.len();
        if ndim == 0 {
            return Err("CUDA cat expects at least 1 dimension".to_string());
        }
        if axis >= ndim {
            return Err(format!(
                "CUDA cat axis out of bounds: axis={}, ndim={}",
                axis, ndim
            ));
        }
        let len = out_shape
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
            .ok_or_else(|| "CUDA cat output length overflow".to_string())?;
        if len == 0 {
            return Err("CUDA cat does not support empty outputs".to_string());
        }
        let out_axis_len = out_shape[axis];
        if lhs_axis_len > out_axis_len {
            return Err(format!(
                "CUDA cat lhs axis length out of bounds: lhs_axis_len={}, out_axis_len={}",
                lhs_axis_len, out_axis_len
            ));
        }
        let rhs_axis_len = out_axis_len - lhs_axis_len;

        let mut lhs_shape = out_shape.to_vec();
        lhs_shape[axis] = lhs_axis_len;
        let mut rhs_shape = out_shape.to_vec();
        rhs_shape[axis] = rhs_axis_len;

        let lhs_expected = lhs_shape
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
            .ok_or_else(|| "CUDA cat lhs length overflow".to_string())?;
        let rhs_expected = rhs_shape
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
            .ok_or_else(|| "CUDA cat rhs length overflow".to_string())?;
        if lhs.len() != lhs_expected || rhs.len() != rhs_expected {
            return Err(format!(
                "CUDA cat input length mismatch: expected lhs={}, rhs={}, got lhs={}, rhs={}",
                lhs_expected,
                rhs_expected,
                lhs.len(),
                rhs.len()
            ));
        }

        fn row_major_strides(shape: &[usize]) -> Result<Vec<usize>, String> {
            let mut strides = vec![0usize; shape.len()];
            let mut stride = 1usize;
            for i in (0..shape.len()).rev() {
                strides[i] = stride;
                stride = stride
                    .checked_mul(shape[i])
                    .ok_or_else(|| "CUDA cat stride overflow".to_string())?;
            }
            Ok(strides)
        }

        let out_strides = row_major_strides(out_shape)?;
        let lhs_strides = row_major_strides(&lhs_shape)?;
        let rhs_strides = row_major_strides(&rhs_shape)?;

        let out = alloc_f32(len)?;
        let status = unsafe {
            lumen_cuda_cat_f32_device(
                lhs.handle(),
                rhs.handle(),
                out.handle(),
                ndim,
                out_shape.as_ptr(),
                out_strides.as_ptr(),
                lhs_strides.as_ptr(),
                rhs_strides.as_ptr(),
                axis,
                lhs_axis_len,
                len,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        let host = download_f32(&out)?;
        Ok((out, host))
    }

    pub fn cat_f32_buffer(
        lhs: &CudaBuffer,
        rhs: &CudaBuffer,
        out_shape: &[usize],
        axis: usize,
        lhs_axis_len: usize,
    ) -> Result<CudaBuffer, String> {
        let ndim = out_shape.len();
        if ndim == 0 {
            return Err("CUDA cat expects at least 1 dimension".to_string());
        }
        if axis >= ndim {
            return Err(format!(
                "CUDA cat axis out of bounds: axis={}, ndim={}",
                axis, ndim
            ));
        }
        let len = out_shape
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
            .ok_or_else(|| "CUDA cat output length overflow".to_string())?;
        if len == 0 {
            return Err("CUDA cat does not support empty outputs".to_string());
        }
        let out_axis_len = out_shape[axis];
        if lhs_axis_len > out_axis_len {
            return Err(format!(
                "CUDA cat lhs axis length out of bounds: lhs_axis_len={}, out_axis_len={}",
                lhs_axis_len, out_axis_len
            ));
        }
        let rhs_axis_len = out_axis_len - lhs_axis_len;

        let lhs_len = out_shape
            .iter()
            .enumerate()
            .try_fold(1usize, |acc, (idx, &dim)| {
                let actual = if idx == axis { lhs_axis_len } else { dim };
                acc.checked_mul(actual)
            })
            .ok_or_else(|| "CUDA cat lhs length overflow".to_string())?;
        let rhs_len = out_shape
            .iter()
            .enumerate()
            .try_fold(1usize, |acc, (idx, &dim)| {
                let actual = if idx == axis { rhs_axis_len } else { dim };
                acc.checked_mul(actual)
            })
            .ok_or_else(|| "CUDA cat rhs length overflow".to_string())?;
        if lhs.len() != lhs_len || rhs.len() != rhs_len {
            return Err(format!(
                "CUDA cat input length mismatch: expected lhs={}, rhs={}, got lhs={}, rhs={}",
                lhs_len,
                rhs_len,
                lhs.len(),
                rhs.len()
            ));
        }

        let mut out_strides = vec![0usize; ndim];
        let mut stride = 1usize;
        for i in (0..ndim).rev() {
            out_strides[i] = stride;
            stride = stride
                .checked_mul(out_shape[i])
                .ok_or_else(|| "CUDA cat stride overflow".to_string())?;
        }

        let lhs_shape = out_shape
            .iter()
            .enumerate()
            .map(|(idx, &dim)| if idx == axis { lhs_axis_len } else { dim })
            .collect::<Vec<_>>();
        let rhs_shape = out_shape
            .iter()
            .enumerate()
            .map(|(idx, &dim)| if idx == axis { rhs_axis_len } else { dim })
            .collect::<Vec<_>>();

        let mut lhs_strides = vec![0usize; ndim];
        stride = 1usize;
        for i in (0..ndim).rev() {
            lhs_strides[i] = stride;
            stride = stride
                .checked_mul(lhs_shape[i])
                .ok_or_else(|| "CUDA cat lhs stride overflow".to_string())?;
        }

        let mut rhs_strides = vec![0usize; ndim];
        stride = 1usize;
        for i in (0..ndim).rev() {
            rhs_strides[i] = stride;
            stride = stride
                .checked_mul(rhs_shape[i])
                .ok_or_else(|| "CUDA cat rhs stride overflow".to_string())?;
        }

        let out = alloc_f32(len)?;
        let status = unsafe {
            lumen_cuda_cat_f32_device(
                lhs.handle(),
                rhs.handle(),
                out.handle(),
                ndim,
                out_shape.as_ptr(),
                out_strides.as_ptr(),
                lhs_strides.as_ptr(),
                rhs_strides.as_ptr(),
                axis,
                lhs_axis_len,
                len,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(out)
    }

    pub fn cat_typed_buffer(
        lhs: &CudaBuffer,
        rhs: &CudaBuffer,
        dtype: DType,
        out_shape: &[usize],
        axis: usize,
        lhs_axis_len: usize,
    ) -> Result<CudaBuffer, String> {
        let ndim = out_shape.len();
        if ndim == 0 {
            return Err("CUDA typed cat expects at least 1 dimension".to_string());
        }
        if axis >= ndim {
            return Err(format!(
                "CUDA typed cat axis out of bounds: axis={}, ndim={}",
                axis, ndim
            ));
        }
        let len = out_shape
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
            .ok_or_else(|| "CUDA typed cat output length overflow".to_string())?;
        if len == 0 {
            return Err("CUDA typed cat does not support empty outputs".to_string());
        }
        let out_axis_len = out_shape[axis];
        if lhs_axis_len > out_axis_len {
            return Err(format!(
                "CUDA typed cat lhs axis length out of bounds: lhs_axis_len={}, out_axis_len={}",
                lhs_axis_len, out_axis_len
            ));
        }
        let rhs_axis_len = out_axis_len - lhs_axis_len;

        let lhs_len = out_shape
            .iter()
            .enumerate()
            .try_fold(1usize, |acc, (idx, &dim)| {
                let actual = if idx == axis { lhs_axis_len } else { dim };
                acc.checked_mul(actual)
            })
            .ok_or_else(|| "CUDA typed cat lhs length overflow".to_string())?;
        let rhs_len = out_shape
            .iter()
            .enumerate()
            .try_fold(1usize, |acc, (idx, &dim)| {
                let actual = if idx == axis { rhs_axis_len } else { dim };
                acc.checked_mul(actual)
            })
            .ok_or_else(|| "CUDA typed cat rhs length overflow".to_string())?;
        if lhs.len() != lhs_len || rhs.len() != rhs_len {
            return Err(format!(
                "CUDA typed cat input length mismatch: expected lhs={}, rhs={}, got lhs={}, rhs={}",
                lhs_len,
                rhs_len,
                lhs.len(),
                rhs.len()
            ));
        }

        let mut out_strides = vec![0usize; ndim];
        let mut stride = 1usize;
        for i in (0..ndim).rev() {
            out_strides[i] = stride;
            stride = stride
                .checked_mul(out_shape[i])
                .ok_or_else(|| "CUDA typed cat stride overflow".to_string())?;
        }

        let lhs_shape = out_shape
            .iter()
            .enumerate()
            .map(|(idx, &dim)| if idx == axis { lhs_axis_len } else { dim })
            .collect::<Vec<_>>();
        let rhs_shape = out_shape
            .iter()
            .enumerate()
            .map(|(idx, &dim)| if idx == axis { rhs_axis_len } else { dim })
            .collect::<Vec<_>>();

        let mut lhs_strides = vec![0usize; ndim];
        stride = 1usize;
        for i in (0..ndim).rev() {
            lhs_strides[i] = stride;
            stride = stride
                .checked_mul(lhs_shape[i])
                .ok_or_else(|| "CUDA typed cat lhs stride overflow".to_string())?;
        }

        let mut rhs_strides = vec![0usize; ndim];
        stride = 1usize;
        for i in (0..ndim).rev() {
            rhs_strides[i] = stride;
            stride = stride
                .checked_mul(rhs_shape[i])
                .ok_or_else(|| "CUDA typed cat rhs stride overflow".to_string())?;
        }

        let out = alloc_storage(len)?;
        let status = unsafe {
            lumen_cuda_cat_typed_device(
                lhs.handle(),
                rhs.handle(),
                dtype_tag(dtype)?,
                out.handle(),
                ndim,
                out_shape.as_ptr(),
                out_strides.as_ptr(),
                lhs_strides.as_ptr(),
                rhs_strides.as_ptr(),
                axis,
                lhs_axis_len,
                len,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(out)
    }

    pub fn cat_backward_slice_f32(
        grad: &CudaBuffer,
        input_shape: &[usize],
        out_shape: &[usize],
        axis: usize,
        axis_start: usize,
    ) -> Result<(CudaBuffer, Vec<f32>), String> {
        let out = cat_backward_slice_f32_buffer(grad, input_shape, out_shape, axis, axis_start)?;
        let host = download_f32(&out)?;
        Ok((out, host))
    }

    pub fn cat_backward_slice_f32_buffer(
        grad: &CudaBuffer,
        input_shape: &[usize],
        out_shape: &[usize],
        axis: usize,
        axis_start: usize,
    ) -> Result<CudaBuffer, String> {
        let ndim = out_shape.len();
        if ndim == 0 || input_shape.len() != ndim {
            return Err(format!(
                "CUDA cat backward shape rank mismatch: input_ndim={}, out_ndim={}",
                input_shape.len(),
                ndim
            ));
        }
        if axis >= ndim {
            return Err(format!(
                "CUDA cat backward axis out of bounds: axis={}, ndim={}",
                axis, ndim
            ));
        }
        for (idx, (&in_dim, &out_dim)) in input_shape.iter().zip(out_shape.iter()).enumerate() {
            if idx == axis {
                if !range_fits(axis_start, in_dim, out_dim) {
                    return Err(format!(
                        "CUDA cat backward axis range out of bounds: start={}, len={}, out_dim={}",
                        axis_start, in_dim, out_dim
                    ));
                }
            } else if in_dim != out_dim {
                return Err(format!(
                    "CUDA cat backward non-axis dim mismatch at {}: input={}, output={}",
                    idx, in_dim, out_dim
                ));
            }
        }

        let out_len = out_shape
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
            .ok_or_else(|| "CUDA cat backward output grad length overflow".to_string())?;
        if grad.len() != out_len {
            return Err(format!(
                "CUDA cat backward grad length mismatch: expected {}, got {}",
                out_len,
                grad.len()
            ));
        }
        let input_len = input_shape
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
            .ok_or_else(|| "CUDA cat backward input grad length overflow".to_string())?;

        fn row_major_strides(shape: &[usize]) -> Result<Vec<usize>, String> {
            let mut strides = vec![0usize; shape.len()];
            let mut stride = 1usize;
            for i in (0..shape.len()).rev() {
                strides[i] = stride;
                stride = stride
                    .checked_mul(shape[i])
                    .ok_or_else(|| "CUDA cat backward stride overflow".to_string())?;
            }
            Ok(strides)
        }

        let input_strides = row_major_strides(input_shape)?;
        let out_strides = row_major_strides(out_shape)?;
        let out = alloc_f32(input_len)?;
        let status = unsafe {
            lumen_cuda_cat_backward_slice_f32_device(
                grad.handle(),
                out.handle(),
                ndim,
                input_shape.as_ptr(),
                input_strides.as_ptr(),
                out_strides.as_ptr(),
                axis,
                axis_start,
                input_len,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(out)
    }

    pub fn repeat_kv_f32(
        input: &CudaBuffer,
        batch_size: usize,
        num_kv_heads: usize,
        seq_len: usize,
        dim: usize,
        n_rep: usize,
    ) -> Result<(CudaBuffer, Vec<f32>), String> {
        let input_len = batch_size
            .checked_mul(num_kv_heads)
            .and_then(|value| value.checked_mul(seq_len))
            .and_then(|value| value.checked_mul(dim))
            .ok_or_else(|| "CUDA repeat_kv input length overflow".to_string())?;
        if input.len() != input_len {
            return Err(format!(
                "CUDA repeat_kv input length mismatch: expected {}, got {}",
                input_len,
                input.len()
            ));
        }
        if batch_size == 0 || num_kv_heads == 0 || seq_len == 0 || dim == 0 || n_rep == 0 {
            return Err("CUDA repeat_kv dimensions must be greater than zero".to_string());
        }
        let out_len = input_len
            .checked_mul(n_rep)
            .ok_or_else(|| "CUDA repeat_kv output length overflow".to_string())?;
        let out = alloc_f32(out_len)?;
        let status = unsafe {
            lumen_cuda_repeat_kv_f32_device(
                input.handle(),
                out.handle(),
                batch_size,
                num_kv_heads,
                seq_len,
                dim,
                n_rep,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        let host = download_f32(&out)?;
        Ok((out, host))
    }

    pub fn repeat_kv_f32_buffer(
        input: &CudaBuffer,
        batch_size: usize,
        num_kv_heads: usize,
        seq_len: usize,
        dim: usize,
        n_rep: usize,
    ) -> Result<CudaBuffer, String> {
        let input_len = batch_size
            .checked_mul(num_kv_heads)
            .and_then(|value| value.checked_mul(seq_len))
            .and_then(|value| value.checked_mul(dim))
            .ok_or_else(|| "CUDA repeat_kv input length overflow".to_string())?;
        if input.len() != input_len {
            return Err(format!(
                "CUDA repeat_kv input length mismatch: expected {}, got {}",
                input_len,
                input.len()
            ));
        }
        if batch_size == 0 || num_kv_heads == 0 || seq_len == 0 || dim == 0 || n_rep == 0 {
            return Err("CUDA repeat_kv dimensions must be greater than zero".to_string());
        }
        let out_len = input_len
            .checked_mul(n_rep)
            .ok_or_else(|| "CUDA repeat_kv output length overflow".to_string())?;
        let out = alloc_f32(out_len)?;
        let status = unsafe {
            lumen_cuda_repeat_kv_f32_device(
                input.handle(),
                out.handle(),
                batch_size,
                num_kv_heads,
                seq_len,
                dim,
                n_rep,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(out)
    }

    pub fn repeat_kv_typed_buffer(
        input: &CudaBuffer,
        dtype: DType,
        batch_size: usize,
        num_kv_heads: usize,
        seq_len: usize,
        dim: usize,
        n_rep: usize,
    ) -> Result<CudaBuffer, String> {
        let input_len = batch_size
            .checked_mul(num_kv_heads)
            .and_then(|value| value.checked_mul(seq_len))
            .and_then(|value| value.checked_mul(dim))
            .ok_or_else(|| "CUDA typed repeat_kv input length overflow".to_string())?;
        if input.len() != input_len {
            return Err(format!(
                "CUDA typed repeat_kv input length mismatch: expected {}, got {}",
                input_len,
                input.len()
            ));
        }
        if batch_size == 0 || num_kv_heads == 0 || seq_len == 0 || dim == 0 || n_rep == 0 {
            return Err("CUDA typed repeat_kv dimensions must be greater than zero".to_string());
        }
        let out_len = batch_size
            .checked_mul(num_kv_heads)
            .and_then(|value| value.checked_mul(n_rep))
            .and_then(|value| value.checked_mul(seq_len))
            .and_then(|value| value.checked_mul(dim))
            .ok_or_else(|| "CUDA typed repeat_kv output length overflow".to_string())?;
        let out = alloc_storage(out_len)?;
        let status = unsafe {
            lumen_cuda_repeat_kv_typed_device(
                input.handle(),
                dtype_tag(dtype)?,
                out.handle(),
                batch_size,
                num_kv_heads,
                seq_len,
                dim,
                n_rep,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(out)
    }

    pub fn repeat_kv_backward_f32(
        grad: &CudaBuffer,
        batch_size: usize,
        num_kv_heads: usize,
        seq_len: usize,
        dim: usize,
        n_rep: usize,
    ) -> Result<(CudaBuffer, Vec<f32>), String> {
        let out =
            repeat_kv_backward_f32_buffer(grad, batch_size, num_kv_heads, seq_len, dim, n_rep)?;
        let host = download_f32(&out)?;
        Ok((out, host))
    }

    pub fn repeat_kv_backward_f32_buffer(
        grad: &CudaBuffer,
        batch_size: usize,
        num_kv_heads: usize,
        seq_len: usize,
        dim: usize,
        n_rep: usize,
    ) -> Result<CudaBuffer, String> {
        let grad_len = batch_size
            .checked_mul(num_kv_heads)
            .and_then(|value| value.checked_mul(n_rep))
            .and_then(|value| value.checked_mul(seq_len))
            .and_then(|value| value.checked_mul(dim))
            .ok_or_else(|| "CUDA repeat_kv backward grad length overflow".to_string())?;
        if grad.len() != grad_len {
            return Err(format!(
                "CUDA repeat_kv backward grad length mismatch: expected {}, got {}",
                grad_len,
                grad.len()
            ));
        }
        let out_len = batch_size
            .checked_mul(num_kv_heads)
            .and_then(|value| value.checked_mul(seq_len))
            .and_then(|value| value.checked_mul(dim))
            .ok_or_else(|| "CUDA repeat_kv backward output length overflow".to_string())?;
        let out = alloc_f32(out_len)?;
        let status = unsafe {
            lumen_cuda_repeat_kv_backward_f32_device(
                grad.handle(),
                out.handle(),
                batch_size,
                num_kv_heads,
                seq_len,
                dim,
                n_rep,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn decode_attention_f32(
        q: &CudaBuffer,
        k: &CudaBuffer,
        v: &CudaBuffer,
        batch_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        active_seq_len: usize,
        cache_seq_len: usize,
        dim: usize,
        n_rep: usize,
        scale: f32,
    ) -> Result<(CudaBuffer, Vec<f32>), String> {
        ensure_finite("CUDA decode attention scale", scale)?;
        let q_len = batch_size
            .checked_mul(num_heads)
            .and_then(|value| value.checked_mul(dim))
            .ok_or_else(|| "CUDA decode attention q length overflow".to_string())?;
        let kv_len = batch_size
            .checked_mul(num_kv_heads)
            .and_then(|value| value.checked_mul(cache_seq_len))
            .and_then(|value| value.checked_mul(dim))
            .ok_or_else(|| "CUDA decode attention kv length overflow".to_string())?;
        if q.len() != q_len {
            return Err(format!(
                "CUDA decode attention q length mismatch: expected {}, got {}",
                q_len,
                q.len()
            ));
        }
        if k.len() != kv_len || v.len() != kv_len {
            return Err(format!(
                "CUDA decode attention kv length mismatch: expected {}, got k={}, v={}",
                kv_len,
                k.len(),
                v.len()
            ));
        }
        if batch_size == 0
            || num_heads == 0
            || num_kv_heads == 0
            || active_seq_len == 0
            || cache_seq_len == 0
            || dim == 0
            || n_rep == 0
        {
            return Err("CUDA decode attention dimensions must be greater than zero".to_string());
        }
        if active_seq_len > cache_seq_len {
            return Err(format!(
                "CUDA decode attention active_seq_len out of bounds: {} > {}",
                active_seq_len, cache_seq_len
            ));
        }
        let out_len = batch_size
            .checked_mul(num_heads)
            .and_then(|value| value.checked_mul(dim))
            .ok_or_else(|| "CUDA decode attention output length overflow".to_string())?;
        let out = alloc_f32(out_len)?;
        let status = unsafe {
            lumen_cuda_decode_attention_f32_device(
                q.handle(),
                k.handle(),
                v.handle(),
                out.handle(),
                batch_size,
                num_heads,
                num_kv_heads,
                active_seq_len,
                cache_seq_len,
                dim,
                n_rep,
                scale,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        let host = download_f32(&out)?;
        Ok((out, host))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn decode_attention_f32_buffer(
        q: &CudaBuffer,
        k: &CudaBuffer,
        v: &CudaBuffer,
        batch_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        active_seq_len: usize,
        cache_seq_len: usize,
        dim: usize,
        n_rep: usize,
        scale: f32,
    ) -> Result<CudaBuffer, String> {
        ensure_finite("CUDA decode attention scale", scale)?;
        let q_len = batch_size
            .checked_mul(num_heads)
            .and_then(|value| value.checked_mul(dim))
            .ok_or_else(|| "CUDA decode attention q length overflow".to_string())?;
        let kv_len = batch_size
            .checked_mul(num_kv_heads)
            .and_then(|value| value.checked_mul(cache_seq_len))
            .and_then(|value| value.checked_mul(dim))
            .ok_or_else(|| "CUDA decode attention kv length overflow".to_string())?;
        if q.len() != q_len {
            return Err(format!(
                "CUDA decode attention q length mismatch: expected {}, got {}",
                q_len,
                q.len()
            ));
        }
        if k.len() != kv_len || v.len() != kv_len {
            return Err(format!(
                "CUDA decode attention kv length mismatch: expected {}, got k={}, v={}",
                kv_len,
                k.len(),
                v.len()
            ));
        }
        if batch_size == 0
            || num_heads == 0
            || num_kv_heads == 0
            || active_seq_len == 0
            || cache_seq_len == 0
            || dim == 0
            || n_rep == 0
        {
            return Err("CUDA decode attention dimensions must be greater than zero".to_string());
        }
        if active_seq_len > cache_seq_len {
            return Err(format!(
                "CUDA decode attention active_seq_len out of bounds: {} > {}",
                active_seq_len, cache_seq_len
            ));
        }
        let out_len = batch_size
            .checked_mul(num_heads)
            .and_then(|value| value.checked_mul(dim))
            .ok_or_else(|| "CUDA decode attention output length overflow".to_string())?;
        let out = alloc_f32(out_len)?;
        let status = unsafe {
            lumen_cuda_decode_attention_f32_device(
                q.handle(),
                k.handle(),
                v.handle(),
                out.handle(),
                batch_size,
                num_heads,
                num_kv_heads,
                active_seq_len,
                cache_seq_len,
                dim,
                n_rep,
                scale,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn prefill_attention_f32_buffer(
        q: &CudaBuffer,
        k: &CudaBuffer,
        v: &CudaBuffer,
        batch_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        q_seq_len: usize,
        active_seq_len: usize,
        cache_seq_len: usize,
        dim: usize,
        n_rep: usize,
        past_len: usize,
        scale: f32,
        is_causal: bool,
    ) -> Result<CudaBuffer, String> {
        ensure_finite("CUDA prefill attention scale", scale)?;
        let q_len = batch_size
            .checked_mul(num_heads)
            .and_then(|value| value.checked_mul(q_seq_len))
            .and_then(|value| value.checked_mul(dim))
            .ok_or_else(|| "CUDA prefill attention q length overflow".to_string())?;
        let kv_len = batch_size
            .checked_mul(num_kv_heads)
            .and_then(|value| value.checked_mul(cache_seq_len))
            .and_then(|value| value.checked_mul(dim))
            .ok_or_else(|| "CUDA prefill attention kv length overflow".to_string())?;
        if q.len() != q_len {
            return Err(format!(
                "CUDA prefill attention q length mismatch: expected {}, got {}",
                q_len,
                q.len()
            ));
        }
        if k.len() != kv_len || v.len() != kv_len {
            return Err(format!(
                "CUDA prefill attention kv length mismatch: expected {}, got k={}, v={}",
                kv_len,
                k.len(),
                v.len()
            ));
        }
        if batch_size == 0
            || num_heads == 0
            || num_kv_heads == 0
            || q_seq_len == 0
            || active_seq_len == 0
            || cache_seq_len == 0
            || dim == 0
            || n_rep == 0
        {
            return Err("CUDA prefill attention dimensions must be greater than zero".to_string());
        }
        if active_seq_len > cache_seq_len
            || past_len > active_seq_len
            || q_seq_len > active_seq_len - past_len
        {
            return Err(format!(
                "CUDA prefill attention sequence range out of bounds: past_len={}, q_seq_len={}, active_seq_len={}, cache_seq_len={}",
                past_len, q_seq_len, active_seq_len, cache_seq_len
            ));
        }
        let out_len = batch_size
            .checked_mul(q_seq_len)
            .and_then(|value| value.checked_mul(num_heads))
            .and_then(|value| value.checked_mul(dim))
            .ok_or_else(|| "CUDA prefill attention output length overflow".to_string())?;
        let out = alloc_f32(out_len)?;
        let status = unsafe {
            lumen_cuda_prefill_attention_f32_device(
                q.handle(),
                k.handle(),
                v.handle(),
                out.handle(),
                batch_size,
                num_heads,
                num_kv_heads,
                q_seq_len,
                active_seq_len,
                cache_seq_len,
                dim,
                n_rep,
                past_len,
                scale,
                if is_causal { 1 } else { 0 },
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(out)
    }

    pub fn fused_gate_up_silu_f32(
        input: &CudaBuffer,
        gate: &CudaBuffer,
        up: &CudaBuffer,
        rows: usize,
        n_dim: usize,
        k_dim: usize,
    ) -> Result<(CudaBuffer, Vec<f32>), String> {
        let out = fused_gate_up_silu_f32_buffer(input, gate, up, rows, n_dim, k_dim)?;
        let host = download_f32(&out)?;
        Ok((out, host))
    }

    pub fn fused_gate_up_silu_f32_buffer(
        input: &CudaBuffer,
        gate: &CudaBuffer,
        up: &CudaBuffer,
        rows: usize,
        n_dim: usize,
        k_dim: usize,
    ) -> Result<CudaBuffer, String> {
        let input_len = rows
            .checked_mul(k_dim)
            .ok_or_else(|| "CUDA fused gate/up input length overflow".to_string())?;
        let weight_len = n_dim
            .checked_mul(k_dim)
            .ok_or_else(|| "CUDA fused gate/up weight length overflow".to_string())?;
        if input.len() != input_len {
            return Err(format!(
                "CUDA fused gate/up input length mismatch: expected {}, got {}",
                input_len,
                input.len()
            ));
        }
        if gate.len() != weight_len || up.len() != weight_len {
            return Err(format!(
                "CUDA fused gate/up weight length mismatch: expected {}, got gate={}, up={}",
                weight_len,
                gate.len(),
                up.len()
            ));
        }
        if rows == 0 || n_dim == 0 || k_dim == 0 {
            return Err("CUDA fused gate/up dimensions must be greater than zero".to_string());
        }
        let out_len = rows
            .checked_mul(n_dim)
            .ok_or_else(|| "CUDA fused gate/up output length overflow".to_string())?;
        let out = alloc_f32(out_len)?;
        let status = unsafe {
            lumen_cuda_fused_gate_up_silu_f32_device(
                input.handle(),
                gate.handle(),
                up.handle(),
                out.handle(),
                rows,
                n_dim,
                k_dim,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(out)
    }

    pub fn silu_mul_f32_buffer_no_host(
        gate: &CudaBuffer,
        up: &CudaBuffer,
    ) -> Result<CudaBuffer, String> {
        if gate.len() != up.len() {
            return Err(format!(
                "CUDA silu_mul length mismatch: gate={}, up={}",
                gate.len(),
                up.len()
            ));
        }
        if gate.is_empty() {
            return Err("CUDA silu_mul length must be greater than zero".to_string());
        }
        let out = alloc_f32(gate.len())?;
        let status = unsafe {
            lumen_cuda_silu_mul_f32_device(gate.handle(), up.handle(), out.handle(), gate.len())
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn fused_gate_up_silu_typed(
        input: &CudaBuffer,
        input_dtype: DType,
        input_scale: Option<f32>,
        gate: &CudaBuffer,
        weight_dtype: DType,
        gate_scale: Option<f32>,
        up: &CudaBuffer,
        up_scale: Option<f32>,
        rows: usize,
        n_dim: usize,
        k_dim: usize,
    ) -> Result<(CudaBuffer, Vec<f32>), String> {
        let out = fused_gate_up_silu_typed_buffer(
            input,
            input_dtype,
            input_scale,
            gate,
            weight_dtype,
            gate_scale,
            up,
            up_scale,
            rows,
            n_dim,
            k_dim,
        )?;
        let host = download_f32(&out)?;
        Ok((out, host))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn fused_gate_up_silu_typed_buffer(
        input: &CudaBuffer,
        input_dtype: DType,
        input_scale: Option<f32>,
        gate: &CudaBuffer,
        weight_dtype: DType,
        gate_scale: Option<f32>,
        up: &CudaBuffer,
        up_scale: Option<f32>,
        rows: usize,
        n_dim: usize,
        k_dim: usize,
    ) -> Result<CudaBuffer, String> {
        let input_len = rows
            .checked_mul(k_dim)
            .ok_or_else(|| "CUDA typed fused gate/up input length overflow".to_string())?;
        let weight_len = n_dim
            .checked_mul(k_dim)
            .ok_or_else(|| "CUDA typed fused gate/up weight length overflow".to_string())?;
        if input.len() != input_len {
            return Err(format!(
                "CUDA typed fused gate/up input length mismatch: expected {}, got {}",
                input_len,
                input.len()
            ));
        }
        if gate.len() != weight_len || up.len() != weight_len {
            return Err(format!(
                "CUDA typed fused gate/up weight length mismatch: expected {}, got gate={}, up={}",
                weight_len,
                gate.len(),
                up.len()
            ));
        }
        if rows == 0 || n_dim == 0 || k_dim == 0 {
            return Err(
                "CUDA typed fused gate/up dimensions must be greater than zero".to_string(),
            );
        }
        let out_len = rows
            .checked_mul(n_dim)
            .ok_or_else(|| "CUDA typed fused gate/up output length overflow".to_string())?;
        let out = alloc_f32(out_len)?;
        let status = unsafe {
            lumen_cuda_fused_gate_up_silu_typed_device(
                input.handle(),
                dtype_tag(input_dtype)?,
                input_scale.unwrap_or(1.0),
                gate.handle(),
                dtype_tag(weight_dtype)?,
                gate_scale.unwrap_or(1.0),
                up.handle(),
                up_scale.unwrap_or(1.0),
                out.handle(),
                rows,
                n_dim,
                k_dim,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn fused_gate_up_silu_typed_output_buffer(
        input: &CudaBuffer,
        input_dtype: DType,
        input_scale: Option<f32>,
        gate: &CudaBuffer,
        weight_dtype: DType,
        gate_scale: Option<f32>,
        up: &CudaBuffer,
        up_scale: Option<f32>,
        output_dtype: DType,
        rows: usize,
        n_dim: usize,
        k_dim: usize,
    ) -> Result<CudaBuffer, String> {
        if !matches!(output_dtype, DType::F16 | DType::BF16) {
            return Err(format!(
                "CUDA typed fused gate/up output only supports f16/bf16, got {output_dtype:?}"
            ));
        }
        let input_len = rows
            .checked_mul(k_dim)
            .ok_or_else(|| "CUDA typed fused gate/up output input length overflow".to_string())?;
        let weight_len = n_dim
            .checked_mul(k_dim)
            .ok_or_else(|| "CUDA typed fused gate/up output weight length overflow".to_string())?;
        if input.len() != input_len {
            return Err(format!(
                "CUDA typed fused gate/up output input length mismatch: expected {}, got {}",
                input_len,
                input.len()
            ));
        }
        if gate.len() != weight_len || up.len() != weight_len {
            return Err(format!(
                "CUDA typed fused gate/up output weight length mismatch: expected {}, got gate={}, up={}",
                weight_len,
                gate.len(),
                up.len()
            ));
        }
        if rows == 0 || n_dim == 0 || k_dim == 0 {
            return Err(
                "CUDA typed fused gate/up output dimensions must be greater than zero".to_string(),
            );
        }
        let out_len = rows
            .checked_mul(n_dim)
            .ok_or_else(|| "CUDA typed fused gate/up output length overflow".to_string())?;
        let out = alloc_storage(out_len)?;
        let status = unsafe {
            lumen_cuda_fused_gate_up_silu_typed_out_device(
                input.handle(),
                dtype_tag(input_dtype)?,
                input_scale.unwrap_or(1.0),
                gate.handle(),
                dtype_tag(weight_dtype)?,
                gate_scale.unwrap_or(1.0),
                up.handle(),
                up_scale.unwrap_or(1.0),
                out.handle(),
                dtype_tag(output_dtype)?,
                rows,
                n_dim,
                k_dim,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(out)
    }

    pub fn fused_qkv_f32(
        input: &CudaBuffer,
        q: &CudaBuffer,
        k: &CudaBuffer,
        v: &CudaBuffer,
        rows: usize,
        q_n: usize,
        k_n: usize,
        k_dim: usize,
    ) -> Result<CudaThreeHostBuffers, String> {
        let input_len = rows
            .checked_mul(k_dim)
            .ok_or_else(|| "CUDA fused qkv input length overflow".to_string())?;
        if input.len() != input_len {
            return Err(format!(
                "CUDA fused qkv input length mismatch: expected {}, got {}",
                input_len,
                input.len()
            ));
        }
        let q_weight_len = q_n
            .checked_mul(k_dim)
            .ok_or_else(|| "CUDA fused qkv q weight length overflow".to_string())?;
        let kv_weight_len = k_n
            .checked_mul(k_dim)
            .ok_or_else(|| "CUDA fused qkv kv weight length overflow".to_string())?;
        if q.len() != q_weight_len || k.len() != kv_weight_len || v.len() != kv_weight_len {
            return Err(format!(
                "CUDA fused qkv weight length mismatch: expected q={}, k/v={}, got q={}, k={}, v={}",
                q_weight_len,
                kv_weight_len,
                q.len(),
                k.len(),
                v.len()
            ));
        }
        if rows == 0 || q_n == 0 || k_n == 0 || k_dim == 0 {
            return Err("CUDA fused qkv dimensions must be greater than zero".to_string());
        }
        let q_out = alloc_f32(
            rows.checked_mul(q_n)
                .ok_or_else(|| "CUDA fused qkv q output length overflow".to_string())?,
        )?;
        let k_out = alloc_f32(
            rows.checked_mul(k_n)
                .ok_or_else(|| "CUDA fused qkv k output length overflow".to_string())?,
        )?;
        let v_out = alloc_f32(
            rows.checked_mul(k_n)
                .ok_or_else(|| "CUDA fused qkv v output length overflow".to_string())?,
        )?;
        let status = unsafe {
            lumen_cuda_fused_qkv_f32_device(
                input.handle(),
                q.handle(),
                k.handle(),
                v.handle(),
                q_out.handle(),
                k_out.handle(),
                v_out.handle(),
                rows,
                q_n,
                k_n,
                k_dim,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        let q_host = download_f32(&q_out)?;
        let k_host = download_f32(&k_out)?;
        let v_host = download_f32(&v_out)?;
        Ok(((q_out, q_host), (k_out, k_host), (v_out, v_host)))
    }

    pub fn fused_qkv_f32_buffer(
        input: &CudaBuffer,
        q: &CudaBuffer,
        k: &CudaBuffer,
        v: &CudaBuffer,
        rows: usize,
        q_n: usize,
        k_n: usize,
        k_dim: usize,
    ) -> Result<(CudaBuffer, CudaBuffer, CudaBuffer), String> {
        let input_len = rows
            .checked_mul(k_dim)
            .ok_or_else(|| "CUDA fused qkv input length overflow".to_string())?;
        if input.len() != input_len {
            return Err(format!(
                "CUDA fused qkv input length mismatch: expected {}, got {}",
                input_len,
                input.len()
            ));
        }
        let q_weight_len = q_n
            .checked_mul(k_dim)
            .ok_or_else(|| "CUDA fused qkv q weight length overflow".to_string())?;
        let kv_weight_len = k_n
            .checked_mul(k_dim)
            .ok_or_else(|| "CUDA fused qkv kv weight length overflow".to_string())?;
        if q.len() != q_weight_len || k.len() != kv_weight_len || v.len() != kv_weight_len {
            return Err(format!(
                "CUDA fused qkv weight length mismatch: expected q={}, k/v={}, got q={}, k={}, v={}",
                q_weight_len,
                kv_weight_len,
                q.len(),
                k.len(),
                v.len()
            ));
        }
        if rows == 0 || q_n == 0 || k_n == 0 || k_dim == 0 {
            return Err("CUDA fused qkv dimensions must be greater than zero".to_string());
        }
        let q_out = alloc_f32(
            rows.checked_mul(q_n)
                .ok_or_else(|| "CUDA fused qkv q output length overflow".to_string())?,
        )?;
        let k_out = alloc_f32(
            rows.checked_mul(k_n)
                .ok_or_else(|| "CUDA fused qkv k output length overflow".to_string())?,
        )?;
        let v_out = alloc_f32(
            rows.checked_mul(k_n)
                .ok_or_else(|| "CUDA fused qkv v output length overflow".to_string())?,
        )?;
        let status = unsafe {
            lumen_cuda_fused_qkv_f32_device(
                input.handle(),
                q.handle(),
                k.handle(),
                v.handle(),
                q_out.handle(),
                k_out.handle(),
                v_out.handle(),
                rows,
                q_n,
                k_n,
                k_dim,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok((q_out, k_out, v_out))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn fused_qkv_typed_buffer(
        input: &CudaBuffer,
        input_dtype: DType,
        input_scale: Option<f32>,
        q: &CudaBuffer,
        k: &CudaBuffer,
        v: &CudaBuffer,
        weight_dtype: DType,
        q_scale: Option<f32>,
        k_scale: Option<f32>,
        v_scale: Option<f32>,
        rows: usize,
        q_n: usize,
        k_n: usize,
        k_dim: usize,
    ) -> Result<(CudaBuffer, CudaBuffer, CudaBuffer), String> {
        let input_len = rows
            .checked_mul(k_dim)
            .ok_or_else(|| "CUDA typed fused qkv input length overflow".to_string())?;
        if input.len() != input_len {
            return Err(format!(
                "CUDA typed fused qkv input length mismatch: expected {}, got {}",
                input_len,
                input.len()
            ));
        }
        let q_weight_len = q_n
            .checked_mul(k_dim)
            .ok_or_else(|| "CUDA typed fused qkv q weight length overflow".to_string())?;
        let kv_weight_len = k_n
            .checked_mul(k_dim)
            .ok_or_else(|| "CUDA typed fused qkv kv weight length overflow".to_string())?;
        if q.len() != q_weight_len || k.len() != kv_weight_len || v.len() != kv_weight_len {
            return Err(format!(
                "CUDA typed fused qkv weight length mismatch: expected q={}, k/v={}, got q={}, k={}, v={}",
                q_weight_len,
                kv_weight_len,
                q.len(),
                k.len(),
                v.len()
            ));
        }
        if rows == 0 || q_n == 0 || k_n == 0 || k_dim == 0 {
            return Err("CUDA typed fused qkv dimensions must be greater than zero".to_string());
        }
        let q_out = alloc_f32(
            rows.checked_mul(q_n)
                .ok_or_else(|| "CUDA typed fused qkv q output length overflow".to_string())?,
        )?;
        let k_out = alloc_f32(
            rows.checked_mul(k_n)
                .ok_or_else(|| "CUDA typed fused qkv k output length overflow".to_string())?,
        )?;
        let v_out = alloc_f32(
            rows.checked_mul(k_n)
                .ok_or_else(|| "CUDA typed fused qkv v output length overflow".to_string())?,
        )?;
        let status = unsafe {
            lumen_cuda_fused_qkv_typed_device(
                input.handle(),
                dtype_tag(input_dtype)?,
                input_scale.unwrap_or(1.0),
                q.handle(),
                k.handle(),
                v.handle(),
                dtype_tag(weight_dtype)?,
                q_scale.unwrap_or(1.0),
                k_scale.unwrap_or(1.0),
                v_scale.unwrap_or(1.0),
                q_out.handle(),
                k_out.handle(),
                v_out.handle(),
                rows,
                q_n,
                k_n,
                k_dim,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok((q_out, k_out, v_out))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn fused_qkv_typed_output_buffer(
        input: &CudaBuffer,
        input_dtype: DType,
        input_scale: Option<f32>,
        q: &CudaBuffer,
        k: &CudaBuffer,
        v: &CudaBuffer,
        weight_dtype: DType,
        q_scale: Option<f32>,
        k_scale: Option<f32>,
        v_scale: Option<f32>,
        output_dtype: DType,
        rows: usize,
        q_n: usize,
        k_n: usize,
        k_dim: usize,
    ) -> Result<(CudaBuffer, CudaBuffer, CudaBuffer), String> {
        if !matches!(output_dtype, DType::F16 | DType::BF16) {
            return Err(format!(
                "CUDA typed fused qkv output only supports f16/bf16, got {output_dtype:?}"
            ));
        }
        let input_len = rows
            .checked_mul(k_dim)
            .ok_or_else(|| "CUDA typed fused qkv output input length overflow".to_string())?;
        if input.len() != input_len {
            return Err(format!(
                "CUDA typed fused qkv output input length mismatch: expected {}, got {}",
                input_len,
                input.len()
            ));
        }
        let q_weight_len = q_n
            .checked_mul(k_dim)
            .ok_or_else(|| "CUDA typed fused qkv output q weight length overflow".to_string())?;
        let kv_weight_len = k_n
            .checked_mul(k_dim)
            .ok_or_else(|| "CUDA typed fused qkv output kv weight length overflow".to_string())?;
        if q.len() != q_weight_len || k.len() != kv_weight_len || v.len() != kv_weight_len {
            return Err(format!(
                "CUDA typed fused qkv output weight length mismatch: expected q={}, k/v={}, got q={}, k={}, v={}",
                q_weight_len,
                kv_weight_len,
                q.len(),
                k.len(),
                v.len()
            ));
        }
        if rows == 0 || q_n == 0 || k_n == 0 || k_dim == 0 {
            return Err(
                "CUDA typed fused qkv output dimensions must be greater than zero".to_string(),
            );
        }
        let q_out = alloc_storage(
            rows.checked_mul(q_n)
                .ok_or_else(|| "CUDA typed fused qkv output q length overflow".to_string())?,
        )?;
        let k_out = alloc_storage(
            rows.checked_mul(k_n)
                .ok_or_else(|| "CUDA typed fused qkv output k length overflow".to_string())?,
        )?;
        let v_out = alloc_storage(
            rows.checked_mul(k_n)
                .ok_or_else(|| "CUDA typed fused qkv output v length overflow".to_string())?,
        )?;
        let status = unsafe {
            lumen_cuda_fused_qkv_typed_out_device(
                input.handle(),
                dtype_tag(input_dtype)?,
                input_scale.unwrap_or(1.0),
                q.handle(),
                k.handle(),
                v.handle(),
                dtype_tag(weight_dtype)?,
                q_scale.unwrap_or(1.0),
                k_scale.unwrap_or(1.0),
                v_scale.unwrap_or(1.0),
                q_out.handle(),
                k_out.handle(),
                v_out.handle(),
                dtype_tag(output_dtype)?,
                rows,
                q_n,
                k_n,
                k_dim,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok((q_out, k_out, v_out))
    }

    pub fn rope_f32(
        input: &CudaBuffer,
        cos: &CudaBuffer,
        sin: &CudaBuffer,
        batch_size: usize,
        num_heads: usize,
        seq_len: usize,
        dim: usize,
        offset: usize,
        cache_seq_len: usize,
    ) -> Result<(CudaBuffer, Vec<f32>), String> {
        let expected_len = batch_size
            .checked_mul(num_heads)
            .and_then(|value| value.checked_mul(seq_len))
            .and_then(|value| value.checked_mul(dim))
            .ok_or_else(|| "CUDA RoPE input length overflow".to_string())?;
        if input.len() != expected_len {
            return Err(format!(
                "CUDA RoPE input length mismatch: expected {}, got {}",
                expected_len,
                input.len()
            ));
        }
        if dim == 0 || !dim.is_multiple_of(2) {
            return Err(format!(
                "CUDA RoPE expects a positive even dimension, got {}",
                dim
            ));
        }
        if !range_fits(offset, seq_len, cache_seq_len) {
            return Err(format!(
                "CUDA RoPE offset out of bounds: offset {} + seq_len {} > cache_seq_len {}",
                offset, seq_len, cache_seq_len
            ));
        }
        let cache_expected_len = cache_seq_len
            .checked_mul(dim)
            .ok_or_else(|| "CUDA RoPE cache length overflow".to_string())?;
        if cos.len() != cache_expected_len || sin.len() != cache_expected_len {
            return Err(format!(
                "CUDA RoPE cache length mismatch: expected {}, got cos={}, sin={}",
                cache_expected_len,
                cos.len(),
                sin.len()
            ));
        }
        let out = alloc_f32(expected_len)?;
        let status = unsafe {
            lumen_cuda_rope_f32_device(
                input.handle(),
                cos.handle(),
                sin.handle(),
                out.handle(),
                batch_size,
                num_heads,
                seq_len,
                dim,
                offset,
                cache_seq_len,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        let host = download_f32(&out)?;
        Ok((out, host))
    }

    pub fn rope_f32_buffer(
        input: &CudaBuffer,
        cos: &CudaBuffer,
        sin: &CudaBuffer,
        batch_size: usize,
        num_heads: usize,
        seq_len: usize,
        dim: usize,
        offset: usize,
        cache_seq_len: usize,
    ) -> Result<CudaBuffer, String> {
        let expected_len = batch_size
            .checked_mul(num_heads)
            .and_then(|value| value.checked_mul(seq_len))
            .and_then(|value| value.checked_mul(dim))
            .ok_or_else(|| "CUDA RoPE input length overflow".to_string())?;
        if input.len() != expected_len {
            return Err(format!(
                "CUDA RoPE input length mismatch: expected {}, got {}",
                expected_len,
                input.len()
            ));
        }
        if dim == 0 || !dim.is_multiple_of(2) {
            return Err(format!(
                "CUDA RoPE expects a positive even dimension, got {}",
                dim
            ));
        }
        if !range_fits(offset, seq_len, cache_seq_len) {
            return Err(format!(
                "CUDA RoPE offset out of bounds: offset {} + seq_len {} > cache_seq_len {}",
                offset, seq_len, cache_seq_len
            ));
        }
        let cache_len = cache_seq_len
            .checked_mul(dim)
            .ok_or_else(|| "CUDA RoPE cache length overflow".to_string())?;
        if cos.len() != cache_len || sin.len() != cache_len {
            return Err(format!(
                "CUDA RoPE cache length mismatch: expected {}, got cos={}, sin={}",
                cache_len,
                cos.len(),
                sin.len()
            ));
        }
        let out = alloc_f32(expected_len)?;
        let status = unsafe {
            lumen_cuda_rope_f32_device(
                input.handle(),
                cos.handle(),
                sin.handle(),
                out.handle(),
                batch_size,
                num_heads,
                seq_len,
                dim,
                offset,
                cache_seq_len,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn rope_typed_buffer(
        input: &CudaBuffer,
        input_dtype: DType,
        input_scale: Option<f32>,
        cos: &CudaBuffer,
        sin: &CudaBuffer,
        cache_dtype: DType,
        cos_scale: Option<f32>,
        sin_scale: Option<f32>,
        batch_size: usize,
        num_heads: usize,
        seq_len: usize,
        dim: usize,
        offset: usize,
        cache_seq_len: usize,
    ) -> Result<CudaBuffer, String> {
        let expected_len = batch_size
            .checked_mul(num_heads)
            .and_then(|value| value.checked_mul(seq_len))
            .and_then(|value| value.checked_mul(dim))
            .ok_or_else(|| "CUDA typed RoPE input length overflow".to_string())?;
        if input.len() != expected_len {
            return Err(format!(
                "CUDA typed RoPE input length mismatch: expected {}, got {}",
                expected_len,
                input.len()
            ));
        }
        if dim == 0 || !dim.is_multiple_of(2) {
            return Err(format!(
                "CUDA typed RoPE expects a positive even dimension, got {}",
                dim
            ));
        }
        if !range_fits(offset, seq_len, cache_seq_len) {
            return Err(format!(
                "CUDA typed RoPE offset out of bounds: offset {} + seq_len {} > cache_seq_len {}",
                offset, seq_len, cache_seq_len
            ));
        }
        let cache_expected_len = cache_seq_len
            .checked_mul(dim)
            .ok_or_else(|| "CUDA typed RoPE cache length overflow".to_string())?;
        if cos.len() != cache_expected_len || sin.len() != cache_expected_len {
            return Err(format!(
                "CUDA typed RoPE cache length mismatch: expected {}, got cos={}, sin={}",
                cache_expected_len,
                cos.len(),
                sin.len()
            ));
        }
        let out = alloc_f32(expected_len)?;
        let status = unsafe {
            lumen_cuda_rope_typed_device(
                input.handle(),
                dtype_tag(input_dtype)?,
                input_scale.unwrap_or(1.0),
                cos.handle(),
                sin.handle(),
                dtype_tag(cache_dtype)?,
                cos_scale.unwrap_or(1.0),
                sin_scale.unwrap_or(1.0),
                out.handle(),
                batch_size,
                num_heads,
                seq_len,
                dim,
                offset,
                cache_seq_len,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn rope_typed_i8_dynamic_buffer(
        input: &CudaBuffer,
        input_dtype: DType,
        input_scale: Option<f32>,
        cos: &CudaBuffer,
        sin: &CudaBuffer,
        cache_dtype: DType,
        cos_scale: Option<f32>,
        sin_scale: Option<f32>,
        batch_size: usize,
        num_heads: usize,
        seq_len: usize,
        dim: usize,
        offset: usize,
        cache_seq_len: usize,
    ) -> Result<(CudaBuffer, f32), String> {
        let expected_len = batch_size
            .checked_mul(num_heads)
            .and_then(|value| value.checked_mul(seq_len))
            .and_then(|value| value.checked_mul(dim))
            .ok_or_else(|| "CUDA typed RoPE dynamic i8 input length overflow".to_string())?;
        if input.len() != expected_len {
            return Err(format!(
                "CUDA typed RoPE dynamic i8 input length mismatch: expected {}, got {}",
                expected_len,
                input.len()
            ));
        }
        if dim == 0 || !dim.is_multiple_of(2) {
            return Err(format!(
                "CUDA typed RoPE dynamic i8 expects a positive even dimension, got {}",
                dim
            ));
        }
        if !range_fits(offset, seq_len, cache_seq_len) {
            return Err(format!(
                "CUDA typed RoPE dynamic i8 offset out of bounds: offset {} + seq_len {} > cache_seq_len {}",
                offset, seq_len, cache_seq_len
            ));
        }
        let cache_expected_len = cache_seq_len
            .checked_mul(dim)
            .ok_or_else(|| "CUDA typed RoPE dynamic i8 cache length overflow".to_string())?;
        if cos.len() != cache_expected_len || sin.len() != cache_expected_len {
            return Err(format!(
                "CUDA typed RoPE dynamic i8 cache length mismatch: expected {}, got cos={}, sin={}",
                cache_expected_len,
                cos.len(),
                sin.len()
            ));
        }
        if input_dtype == DType::I8
            && !input_scale.is_some_and(|scale| scale.is_finite() && scale > 0.0)
        {
            return Err(
                "CUDA typed RoPE dynamic i8 input scale must be finite and > 0".to_string(),
            );
        }
        if cache_dtype == DType::I8
            && (!cos_scale.is_some_and(|scale| scale.is_finite() && scale > 0.0)
                || !sin_scale.is_some_and(|scale| scale.is_finite() && scale > 0.0))
        {
            return Err(
                "CUDA typed RoPE dynamic i8 cache scales must be finite and > 0".to_string(),
            );
        }

        let out = alloc_storage(expected_len)?;
        let mut scale = 1.0f32;
        let status = unsafe {
            lumen_cuda_rope_typed_i8_dynamic_device(
                input.handle(),
                dtype_tag(input_dtype)?,
                input_scale.unwrap_or(1.0),
                cos.handle(),
                sin.handle(),
                dtype_tag(cache_dtype)?,
                cos_scale.unwrap_or(1.0),
                sin_scale.unwrap_or(1.0),
                out.handle(),
                &mut scale as *mut f32,
                batch_size,
                num_heads,
                seq_len,
                dim,
                offset,
                cache_seq_len,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok((out, scale))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn rope_backward_f32(
        grad: &CudaBuffer,
        cos: &CudaBuffer,
        sin: &CudaBuffer,
        batch_size: usize,
        num_heads: usize,
        seq_len: usize,
        dim: usize,
        offset: usize,
        cache_seq_len: usize,
    ) -> Result<(CudaBuffer, Vec<f32>), String> {
        let out = rope_backward_f32_buffer(
            grad,
            cos,
            sin,
            batch_size,
            num_heads,
            seq_len,
            dim,
            offset,
            cache_seq_len,
        )?;
        let host = download_f32(&out)?;
        Ok((out, host))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn rope_backward_f32_buffer(
        grad: &CudaBuffer,
        cos: &CudaBuffer,
        sin: &CudaBuffer,
        batch_size: usize,
        num_heads: usize,
        seq_len: usize,
        dim: usize,
        offset: usize,
        cache_seq_len: usize,
    ) -> Result<CudaBuffer, String> {
        let expected_len = batch_size
            .checked_mul(num_heads)
            .and_then(|value| value.checked_mul(seq_len))
            .and_then(|value| value.checked_mul(dim))
            .ok_or_else(|| "CUDA RoPE backward grad length overflow".to_string())?;
        if grad.len() != expected_len {
            return Err(format!(
                "CUDA RoPE backward grad length mismatch: expected {}, got {}",
                expected_len,
                grad.len()
            ));
        }
        if dim == 0 || !dim.is_multiple_of(2) {
            return Err(format!(
                "CUDA RoPE backward expects a positive even dimension, got {}",
                dim
            ));
        }
        if !range_fits(offset, seq_len, cache_seq_len) {
            return Err(format!(
                "CUDA RoPE backward offset out of bounds: offset {} + seq_len {} > cache_seq_len {}",
                offset, seq_len, cache_seq_len
            ));
        }
        let cache_len = cache_seq_len
            .checked_mul(dim)
            .ok_or_else(|| "CUDA RoPE backward cache length overflow".to_string())?;
        if cos.len() != cache_len || sin.len() != cache_len {
            return Err(format!(
                "CUDA RoPE backward cache length mismatch: expected {}, got cos={}, sin={}",
                cache_len,
                cos.len(),
                sin.len()
            ));
        }
        let out = alloc_f32(expected_len)?;
        let status = unsafe {
            lumen_cuda_rope_backward_f32_device(
                grad.handle(),
                cos.handle(),
                sin.handle(),
                out.handle(),
                batch_size,
                num_heads,
                seq_len,
                dim,
                offset,
                cache_seq_len,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn conv2d_f32(
        input: &CudaBuffer,
        weight: &CudaBuffer,
        bias: Option<&CudaBuffer>,
        batch_size: usize,
        in_channels: usize,
        in_h: usize,
        in_w: usize,
        out_channels: usize,
        k_h: usize,
        k_w: usize,
        pad_h: usize,
        pad_w: usize,
        stride_h: usize,
        stride_w: usize,
    ) -> Result<(CudaBuffer, Vec<f32>, usize, usize), String> {
        let (out, out_h, out_w) = conv2d_f32_buffer(
            input,
            weight,
            bias,
            batch_size,
            in_channels,
            in_h,
            in_w,
            out_channels,
            k_h,
            k_w,
            pad_h,
            pad_w,
            stride_h,
            stride_w,
        )?;
        let host = download_f32(&out)?;
        Ok((out, host, out_h, out_w))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn conv2d_f32_buffer(
        input: &CudaBuffer,
        weight: &CudaBuffer,
        bias: Option<&CudaBuffer>,
        batch_size: usize,
        in_channels: usize,
        in_h: usize,
        in_w: usize,
        out_channels: usize,
        k_h: usize,
        k_w: usize,
        pad_h: usize,
        pad_w: usize,
        stride_h: usize,
        stride_w: usize,
    ) -> Result<(CudaBuffer, usize, usize), String> {
        let input_len = checked_len(
            "CUDA conv2d input length",
            &[batch_size, in_channels, in_h, in_w],
        )?;
        let weight_len = checked_len(
            "CUDA conv2d weight length",
            &[out_channels, in_channels, k_h, k_w],
        )?;
        if input.len() != input_len {
            return Err(format!(
                "CUDA conv2d input length mismatch: expected {}, got {}",
                input_len,
                input.len()
            ));
        }
        if weight.len() != weight_len {
            return Err(format!(
                "CUDA conv2d weight length mismatch: expected {}, got {}",
                weight_len,
                weight.len()
            ));
        }
        if let Some(bias) = bias
            && bias.len() != out_channels
        {
            return Err(format!(
                "CUDA conv2d bias length mismatch: expected {}, got {}",
                out_channels,
                bias.len()
            ));
        }
        let out_h = conv_output_dim(in_h, pad_h, k_h, stride_h, "CUDA conv2d height")?;
        let out_w = conv_output_dim(in_w, pad_w, k_w, stride_w, "CUDA conv2d width")?;
        let output_len = checked_len(
            "CUDA conv2d output length",
            &[batch_size, out_channels, out_h, out_w],
        )?;
        let out = alloc_f32(output_len)?;
        let bias_handle = bias.map(|buf| buf.handle()).unwrap_or(0);
        let status = unsafe {
            lumen_cuda_conv2d_f32_device(
                input.handle(),
                weight.handle(),
                bias_handle,
                out.handle(),
                batch_size,
                in_channels,
                in_h,
                in_w,
                out_channels,
                k_h,
                k_w,
                pad_h,
                pad_w,
                stride_h,
                stride_w,
                out_h,
                out_w,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok((out, out_h, out_w))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn conv2d_typed(
        input: &CudaBuffer,
        input_dtype: DType,
        input_scale: Option<f32>,
        weight: &CudaBuffer,
        weight_dtype: DType,
        weight_scale: Option<f32>,
        bias: Option<(&CudaBuffer, DType, Option<f32>)>,
        batch_size: usize,
        in_channels: usize,
        in_h: usize,
        in_w: usize,
        out_channels: usize,
        k_h: usize,
        k_w: usize,
        pad_h: usize,
        pad_w: usize,
        stride_h: usize,
        stride_w: usize,
    ) -> Result<(CudaBuffer, Vec<f32>, usize, usize), String> {
        let (out, out_h, out_w) = conv2d_typed_buffer(
            input,
            input_dtype,
            input_scale,
            weight,
            weight_dtype,
            weight_scale,
            bias,
            batch_size,
            in_channels,
            in_h,
            in_w,
            out_channels,
            k_h,
            k_w,
            pad_h,
            pad_w,
            stride_h,
            stride_w,
        )?;
        let host = download_f32(&out)?;
        Ok((out, host, out_h, out_w))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn conv2d_typed_buffer(
        input: &CudaBuffer,
        input_dtype: DType,
        input_scale: Option<f32>,
        weight: &CudaBuffer,
        weight_dtype: DType,
        weight_scale: Option<f32>,
        bias: Option<(&CudaBuffer, DType, Option<f32>)>,
        batch_size: usize,
        in_channels: usize,
        in_h: usize,
        in_w: usize,
        out_channels: usize,
        k_h: usize,
        k_w: usize,
        pad_h: usize,
        pad_w: usize,
        stride_h: usize,
        stride_w: usize,
    ) -> Result<(CudaBuffer, usize, usize), String> {
        if stride_h == 0 || stride_w == 0 {
            return Err("CUDA typed conv2d stride must be greater than zero".to_string());
        }
        let input_len = checked_len(
            "CUDA typed conv2d input length",
            &[batch_size, in_channels, in_h, in_w],
        )?;
        let weight_len = checked_len(
            "CUDA typed conv2d weight length",
            &[out_channels, in_channels, k_h, k_w],
        )?;
        if input.len() != input_len {
            return Err(format!(
                "CUDA typed conv2d input length mismatch: expected {}, got {}",
                input_len,
                input.len()
            ));
        }
        if weight.len() != weight_len {
            return Err(format!(
                "CUDA typed conv2d weight length mismatch: expected {}, got {}",
                weight_len,
                weight.len()
            ));
        }
        if let Some((bias, _, _)) = bias
            && bias.len() != out_channels
        {
            return Err(format!(
                "CUDA typed conv2d bias length mismatch: expected {}, got {}",
                out_channels,
                bias.len()
            ));
        }
        if input_dtype == DType::I8
            && !input_scale.is_some_and(|scale| scale.is_finite() && scale > 0.0)
        {
            return Err("CUDA typed conv2d input I8 scale must be finite and > 0".to_string());
        }
        if weight_dtype == DType::I8
            && !weight_scale.is_some_and(|scale| scale.is_finite() && scale > 0.0)
        {
            return Err("CUDA typed conv2d weight I8 scale must be finite and > 0".to_string());
        }
        if let Some((_, dtype, scale)) = bias
            && dtype == DType::I8
            && !scale.is_some_and(|value| value.is_finite() && value > 0.0)
        {
            return Err("CUDA typed conv2d bias I8 scale must be finite and > 0".to_string());
        }
        let out_h = conv_output_dim(in_h, pad_h, k_h, stride_h, "CUDA typed conv2d height")?;
        let out_w = conv_output_dim(in_w, pad_w, k_w, stride_w, "CUDA typed conv2d width")?;
        let output_len = checked_len(
            "CUDA typed conv2d output length",
            &[batch_size, out_channels, out_h, out_w],
        )?;
        let out = alloc_f32(output_len)?;
        let (bias_handle, bias_dtype, bias_scale) = match bias {
            Some((buffer, dtype, scale)) => {
                (buffer.handle(), dtype_tag(dtype)?, scale.unwrap_or(1.0))
            }
            None => (0, dtype_tag(DType::F32)?, 1.0),
        };
        let status = unsafe {
            lumen_cuda_conv2d_typed_device(
                input.handle(),
                dtype_tag(input_dtype)?,
                input_scale.unwrap_or(1.0),
                weight.handle(),
                dtype_tag(weight_dtype)?,
                weight_scale.unwrap_or(1.0),
                bias_handle,
                bias_dtype,
                bias_scale,
                out.handle(),
                batch_size,
                in_channels,
                in_h,
                in_w,
                out_channels,
                k_h,
                k_w,
                pad_h,
                pad_w,
                stride_h,
                stride_w,
                out_h,
                out_w,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok((out, out_h, out_w))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn conv2d_backward_f32(
        input: &CudaBuffer,
        weight: &CudaBuffer,
        grad_output: &CudaBuffer,
        compute_bias_grad: bool,
        batch_size: usize,
        in_channels: usize,
        in_h: usize,
        in_w: usize,
        out_channels: usize,
        k_h: usize,
        k_w: usize,
        pad_h: usize,
        pad_w: usize,
        stride_h: usize,
        stride_w: usize,
    ) -> Result<CudaConv2dBackwardHostBuffers, String> {
        let (grad_input, grad_weight, grad_bias) = conv2d_backward_f32_buffers(
            input,
            weight,
            grad_output,
            compute_bias_grad,
            batch_size,
            in_channels,
            in_h,
            in_w,
            out_channels,
            k_h,
            k_w,
            pad_h,
            pad_w,
            stride_h,
            stride_w,
        )?;
        let grad_input_host = download_f32(&grad_input)?;
        let grad_weight_host = download_f32(&grad_weight)?;
        let grad_bias_host = match grad_bias {
            Some(buffer) => Some((buffer.clone(), download_f32(&buffer)?)),
            None => None,
        };
        Ok((
            grad_input,
            grad_input_host,
            grad_weight,
            grad_weight_host,
            grad_bias_host,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn conv2d_backward_f32_buffers(
        input: &CudaBuffer,
        weight: &CudaBuffer,
        grad_output: &CudaBuffer,
        compute_bias_grad: bool,
        batch_size: usize,
        in_channels: usize,
        in_h: usize,
        in_w: usize,
        out_channels: usize,
        k_h: usize,
        k_w: usize,
        pad_h: usize,
        pad_w: usize,
        stride_h: usize,
        stride_w: usize,
    ) -> Result<(CudaBuffer, CudaBuffer, Option<CudaBuffer>), String> {
        let out_h = conv_output_dim(in_h, pad_h, k_h, stride_h, "CUDA conv2d backward height")?;
        let out_w = conv_output_dim(in_w, pad_w, k_w, stride_w, "CUDA conv2d backward width")?;
        let input_len = checked_len(
            "CUDA conv2d backward input length",
            &[batch_size, in_channels, in_h, in_w],
        )?;
        let weight_len = checked_len(
            "CUDA conv2d backward weight length",
            &[out_channels, in_channels, k_h, k_w],
        )?;
        let grad_output_len = checked_len(
            "CUDA conv2d backward grad output length",
            &[batch_size, out_channels, out_h, out_w],
        )?;
        if input.len() != input_len {
            return Err(format!(
                "CUDA conv2d backward input length mismatch: expected {}, got {}",
                input_len,
                input.len()
            ));
        }
        if weight.len() != weight_len {
            return Err(format!(
                "CUDA conv2d backward weight length mismatch: expected {}, got {}",
                weight_len,
                weight.len()
            ));
        }
        if grad_output.len() != grad_output_len {
            return Err(format!(
                "CUDA conv2d backward grad output length mismatch: expected {}, got {}",
                grad_output_len,
                grad_output.len()
            ));
        }
        let grad_input = alloc_f32(input_len)?;
        let grad_weight = alloc_f32(weight_len)?;
        let grad_bias = if compute_bias_grad {
            Some(alloc_f32(out_channels)?)
        } else {
            None
        };
        let status = unsafe {
            lumen_cuda_conv2d_backward_f32_device(
                input.handle(),
                weight.handle(),
                grad_output.handle(),
                grad_input.handle(),
                grad_weight.handle(),
                grad_bias.as_ref().map(|buf| buf.handle()).unwrap_or(0),
                batch_size,
                in_channels,
                in_h,
                in_w,
                out_channels,
                k_h,
                k_w,
                pad_h,
                pad_w,
                stride_h,
                stride_w,
                out_h,
                out_w,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok((grad_input, grad_weight, grad_bias))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn max_pool2d_f32(
        input: &CudaBuffer,
        batch_size: usize,
        channels: usize,
        in_h: usize,
        in_w: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
    ) -> Result<(CudaBuffer, Vec<f32>, usize, usize), String> {
        if kernel_h == 0 || kernel_w == 0 || stride_h == 0 || stride_w == 0 {
            return Err("CUDA max_pool2d kernel and stride must be greater than zero".to_string());
        }
        if in_h < kernel_h || in_w < kernel_w {
            return Err("CUDA max_pool2d kernel is larger than input".to_string());
        }
        let input_len = batch_size
            .checked_mul(channels)
            .and_then(|v| v.checked_mul(in_h))
            .and_then(|v| v.checked_mul(in_w))
            .ok_or_else(|| "CUDA max_pool2d input length overflow".to_string())?;
        if input.len() != input_len {
            return Err(format!(
                "CUDA max_pool2d input length mismatch: expected {}, got {}",
                input_len,
                input.len()
            ));
        }
        let out_h = (in_h - kernel_h) / stride_h + 1;
        let out_w = (in_w - kernel_w) / stride_w + 1;
        let out_len = batch_size
            .checked_mul(channels)
            .and_then(|v| v.checked_mul(out_h))
            .and_then(|v| v.checked_mul(out_w))
            .ok_or_else(|| "CUDA max_pool2d output length overflow".to_string())?;
        let out = alloc_f32(out_len)?;
        let status = unsafe {
            lumen_cuda_max_pool2d_f32_device(
                input.handle(),
                out.handle(),
                batch_size,
                channels,
                in_h,
                in_w,
                kernel_h,
                kernel_w,
                stride_h,
                stride_w,
                out_h,
                out_w,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        let host = download_f32(&out)?;
        Ok((out, host, out_h, out_w))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn max_pool2d_typed(
        input: &CudaBuffer,
        input_dtype: DType,
        input_scale: Option<f32>,
        batch_size: usize,
        channels: usize,
        in_h: usize,
        in_w: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
    ) -> Result<(CudaBuffer, Vec<f32>, usize, usize), String> {
        if kernel_h == 0 || kernel_w == 0 || stride_h == 0 || stride_w == 0 {
            return Err(
                "CUDA typed max_pool2d kernel and stride must be greater than zero".to_string(),
            );
        }
        if in_h < kernel_h || in_w < kernel_w {
            return Err("CUDA typed max_pool2d kernel is larger than input".to_string());
        }
        let input_len = batch_size
            .checked_mul(channels)
            .and_then(|v| v.checked_mul(in_h))
            .and_then(|v| v.checked_mul(in_w))
            .ok_or_else(|| "CUDA typed max_pool2d input length overflow".to_string())?;
        if input.len() != input_len {
            return Err(format!(
                "CUDA typed max_pool2d input length mismatch: expected {}, got {}",
                input_len,
                input.len()
            ));
        }
        if input_dtype == DType::I8
            && !input_scale.is_some_and(|scale| scale.is_finite() && scale > 0.0)
        {
            return Err("CUDA typed max_pool2d I8 scale must be finite and > 0".to_string());
        }
        let out_h = (in_h - kernel_h) / stride_h + 1;
        let out_w = (in_w - kernel_w) / stride_w + 1;
        let out_len = batch_size
            .checked_mul(channels)
            .and_then(|v| v.checked_mul(out_h))
            .and_then(|v| v.checked_mul(out_w))
            .ok_or_else(|| "CUDA typed max_pool2d output length overflow".to_string())?;
        let out = alloc_f32(out_len)?;
        let status = unsafe {
            lumen_cuda_max_pool2d_typed_device(
                input.handle(),
                dtype_tag(input_dtype)?,
                input_scale.unwrap_or(1.0),
                out.handle(),
                batch_size,
                channels,
                in_h,
                in_w,
                kernel_h,
                kernel_w,
                stride_h,
                stride_w,
                out_h,
                out_w,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        let host = download_f32(&out)?;
        Ok((out, host, out_h, out_w))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn max_pool2d_backward_f32(
        input: &CudaBuffer,
        grad_output: &CudaBuffer,
        batch_size: usize,
        channels: usize,
        in_h: usize,
        in_w: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
    ) -> Result<(CudaBuffer, Vec<f32>), String> {
        let grad_input = max_pool2d_backward_f32_buffer(
            input,
            grad_output,
            batch_size,
            channels,
            in_h,
            in_w,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
        )?;
        let host = download_f32(&grad_input)?;
        Ok((grad_input, host))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn max_pool2d_backward_f32_buffer(
        input: &CudaBuffer,
        grad_output: &CudaBuffer,
        batch_size: usize,
        channels: usize,
        in_h: usize,
        in_w: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
    ) -> Result<CudaBuffer, String> {
        if kernel_h == 0 || kernel_w == 0 || stride_h == 0 || stride_w == 0 {
            return Err(
                "CUDA max_pool2d backward kernel and stride must be greater than zero".to_string(),
            );
        }
        if in_h < kernel_h || in_w < kernel_w {
            return Err("CUDA max_pool2d backward kernel is larger than input".to_string());
        }
        let out_h = (in_h - kernel_h) / stride_h + 1;
        let out_w = (in_w - kernel_w) / stride_w + 1;
        let input_len = batch_size
            .checked_mul(channels)
            .and_then(|v| v.checked_mul(in_h))
            .and_then(|v| v.checked_mul(in_w))
            .ok_or_else(|| "CUDA max_pool2d backward input length overflow".to_string())?;
        let grad_output_len = batch_size
            .checked_mul(channels)
            .and_then(|v| v.checked_mul(out_h))
            .and_then(|v| v.checked_mul(out_w))
            .ok_or_else(|| "CUDA max_pool2d backward grad output length overflow".to_string())?;
        if input.len() != input_len {
            return Err(format!(
                "CUDA max_pool2d backward input length mismatch: expected {}, got {}",
                input_len,
                input.len()
            ));
        }
        if grad_output.len() != grad_output_len {
            return Err(format!(
                "CUDA max_pool2d backward grad output length mismatch: expected {}, got {}",
                grad_output_len,
                grad_output.len()
            ));
        }
        let grad_input = alloc_f32(input_len)?;
        let status = unsafe {
            lumen_cuda_max_pool2d_backward_f32_device(
                input.handle(),
                grad_output.handle(),
                grad_input.handle(),
                batch_size,
                channels,
                in_h,
                in_w,
                kernel_h,
                kernel_w,
                stride_h,
                stride_w,
                out_h,
                out_w,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(grad_input)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn max_pool2d_backward_typed_buffer(
        input: &CudaBuffer,
        input_dtype: DType,
        input_scale: Option<f32>,
        grad_output: &CudaBuffer,
        batch_size: usize,
        channels: usize,
        in_h: usize,
        in_w: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
    ) -> Result<CudaBuffer, String> {
        if kernel_h == 0 || kernel_w == 0 || stride_h == 0 || stride_w == 0 {
            return Err(
                "CUDA typed max_pool2d backward kernel and stride must be greater than zero"
                    .to_string(),
            );
        }
        if in_h < kernel_h || in_w < kernel_w {
            return Err("CUDA typed max_pool2d backward kernel is larger than input".to_string());
        }
        if input_dtype == DType::I8
            && !input_scale.is_some_and(|scale| scale.is_finite() && scale > 0.0)
        {
            return Err(
                "CUDA typed max_pool2d backward I8 scale must be finite and > 0".to_string(),
            );
        }
        let out_h = (in_h - kernel_h) / stride_h + 1;
        let out_w = (in_w - kernel_w) / stride_w + 1;
        let input_len = batch_size
            .checked_mul(channels)
            .and_then(|v| v.checked_mul(in_h))
            .and_then(|v| v.checked_mul(in_w))
            .ok_or_else(|| "CUDA typed max_pool2d backward input length overflow".to_string())?;
        let grad_output_len = batch_size
            .checked_mul(channels)
            .and_then(|v| v.checked_mul(out_h))
            .and_then(|v| v.checked_mul(out_w))
            .ok_or_else(|| {
                "CUDA typed max_pool2d backward grad output length overflow".to_string()
            })?;
        if input.len() != input_len {
            return Err(format!(
                "CUDA typed max_pool2d backward input length mismatch: expected {}, got {}",
                input_len,
                input.len()
            ));
        }
        if grad_output.len() != grad_output_len {
            return Err(format!(
                "CUDA typed max_pool2d backward grad output length mismatch: expected {}, got {}",
                grad_output_len,
                grad_output.len()
            ));
        }
        let grad_input = alloc_f32(input_len)?;
        let status = unsafe {
            lumen_cuda_max_pool2d_backward_typed_device(
                input.handle(),
                dtype_tag(input_dtype)?,
                input_scale.unwrap_or(1.0),
                grad_output.handle(),
                grad_input.handle(),
                batch_size,
                channels,
                in_h,
                in_w,
                kernel_h,
                kernel_w,
                stride_h,
                stride_w,
                out_h,
                out_w,
            )
        };
        if status != 0 {
            return Err(last_error_message());
        }
        Ok(grad_input)
    }
}

#[cfg(not(feature = "cuda"))]
mod imp {
    use super::{
        BinaryOp, BroadcastMetadata, CudaAdamHostState, CudaBuffer, CudaConv2dBackwardHostBuffers,
        CudaThreeHostBuffers, CudaTwoHostBuffers, UnaryOp,
    };

    pub fn is_available() -> bool {
        false
    }

    pub fn synchronize() -> Result<(), String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn alloc_f32(_len: usize) -> Result<CudaBuffer, String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn alloc_storage(_len: usize) -> Result<CudaBuffer, String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn upload_f32(_src: &[f32]) -> Result<CudaBuffer, String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn upload_u16_storage(_src: &[u16]) -> Result<CudaBuffer, String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn upload_i8_storage(_src: &[i8]) -> Result<CudaBuffer, String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn download_f32(_buffer: &CudaBuffer) -> Result<Vec<f32>, String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn download_u16_storage(_buffer: &CudaBuffer) -> Result<Vec<u16>, String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn download_i8_storage(_buffer: &CudaBuffer) -> Result<Vec<i8>, String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn download_f32_offset(
        _buffer: &CudaBuffer,
        _offset: usize,
        _len: usize,
    ) -> Result<Vec<f32>, String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn matvec_argmax_f32(
        _input: &CudaBuffer,
        _weight: &CudaBuffer,
        _batch_size: usize,
        _vocab_size: usize,
        _hidden_size: usize,
    ) -> Result<Vec<usize>, String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn matvec_argmax_bf16_i8(
        _input: &CudaBuffer,
        _weight: &CudaBuffer,
        _weight_scale: f32,
        _batch_size: usize,
        _vocab_size: usize,
        _hidden_size: usize,
    ) -> Result<Vec<usize>, String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn matvec_argmax_f32_i8(
        _input: &CudaBuffer,
        _weight: &CudaBuffer,
        _weight_scale: f32,
        _batch_size: usize,
        _vocab_size: usize,
        _hidden_size: usize,
    ) -> Result<Vec<usize>, String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn matvec_argmax_i8_i8(
        _input: &CudaBuffer,
        _input_scale: f32,
        _weight: &CudaBuffer,
        _weight_scale: f32,
        _batch_size: usize,
        _vocab_size: usize,
        _hidden_size: usize,
    ) -> Result<Vec<usize>, String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn upload_f32_offset(
        _buffer: &CudaBuffer,
        _offset: usize,
        _src: &[f32],
    ) -> Result<(), String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn copy_f32_offset(
        _dst: &CudaBuffer,
        _dst_offset: usize,
        _src: &CudaBuffer,
        _src_offset: usize,
        _len: usize,
    ) -> Result<(), String> {
        Err("CUDA feature is disabled".to_string())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn append_kv_cache_f32(
        _dst: &CudaBuffer,
        _src: &CudaBuffer,
        _batch_size: usize,
        _num_heads: usize,
        _src_seq_len: usize,
        _dst_seq_len: usize,
        _dim: usize,
        _dst_start: usize,
    ) -> Result<(), String> {
        Err("CUDA feature is disabled".to_string())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn append_kv_cache_pair_f32(
        _k_dst: &CudaBuffer,
        _v_dst: &CudaBuffer,
        _k_src: &CudaBuffer,
        _v_src: &CudaBuffer,
        _batch_size: usize,
        _num_heads: usize,
        _src_seq_len: usize,
        _dst_seq_len: usize,
        _dim: usize,
        _dst_start: usize,
    ) -> Result<(), String> {
        Err("CUDA feature is disabled".to_string())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn decode_rope_q_append_kv_f32_buffer(
        _q_src: &CudaBuffer,
        _k_src: &CudaBuffer,
        _v_src: &CudaBuffer,
        _cos: &CudaBuffer,
        _sin: &CudaBuffer,
        _k_cache: &CudaBuffer,
        _v_cache: &CudaBuffer,
        _batch_size: usize,
        _num_heads: usize,
        _num_kv_heads: usize,
        _dim: usize,
        _dst_seq_len: usize,
        _offset: usize,
        _cache_seq_len: usize,
    ) -> Result<CudaBuffer, String> {
        Err("CUDA feature is disabled".to_string())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn prefill_attention_f32_buffer(
        _q: &CudaBuffer,
        _k: &CudaBuffer,
        _v: &CudaBuffer,
        _batch_size: usize,
        _num_heads: usize,
        _num_kv_heads: usize,
        _q_seq_len: usize,
        _active_seq_len: usize,
        _cache_seq_len: usize,
        _dim: usize,
        _n_rep: usize,
        _past_len: usize,
        _scale: f32,
        _is_causal: bool,
    ) -> Result<CudaBuffer, String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn kv_cache_prefix_f32_buffer(
        _src: &CudaBuffer,
        _batch_size: usize,
        _num_heads: usize,
        _active_seq_len: usize,
        _src_seq_len: usize,
        _dim: usize,
    ) -> Result<CudaBuffer, String> {
        Err("CUDA feature is disabled".to_string())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn kv_cache_prefix_typed_buffer(
        _src: &CudaBuffer,
        _dtype: crate::precision::DType,
        _batch_size: usize,
        _num_heads: usize,
        _active_seq_len: usize,
        _src_seq_len: usize,
        _dim: usize,
    ) -> Result<CudaBuffer, String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn free_f32(_handle: u64, _len: usize) {}

    pub fn matmul_f32(
        _a: &CudaBuffer,
        _b: &CudaBuffer,
        _m: usize,
        _n: usize,
        _k: usize,
    ) -> Result<(CudaBuffer, Vec<f32>), String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn matmul_f32_no_host(
        _a: &CudaBuffer,
        _b: &CudaBuffer,
        _m: usize,
        _n: usize,
        _k: usize,
    ) -> Result<CudaBuffer, String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn matmul_bf16_host_no_host(
        _a: &[u16],
        _b: &[u16],
        _m: usize,
        _n: usize,
        _k: usize,
    ) -> Result<CudaBuffer, String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn matmul_bf16_buffer_no_host(
        _a: &CudaBuffer,
        _b: &CudaBuffer,
        _m: usize,
        _n: usize,
        _k: usize,
    ) -> Result<CudaBuffer, String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn matmul_f16_host_no_host(
        _a: &[u16],
        _b: &[u16],
        _m: usize,
        _n: usize,
        _k: usize,
    ) -> Result<CudaBuffer, String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn matmul_f16_buffer_no_host(
        _a: &CudaBuffer,
        _b: &CudaBuffer,
        _m: usize,
        _n: usize,
        _k: usize,
    ) -> Result<CudaBuffer, String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn matmul_i8_host_no_host(
        _a: &[i8],
        _a_scale: f32,
        _b: &[i8],
        _b_scale: f32,
        _m: usize,
        _n: usize,
        _k: usize,
    ) -> Result<CudaBuffer, String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn matmul_i8_buffer_no_host(
        _a: &CudaBuffer,
        _a_scale: f32,
        _b: &CudaBuffer,
        _b_scale: f32,
        _m: usize,
        _n: usize,
        _k: usize,
    ) -> Result<CudaBuffer, String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn matmul_i8_typed_output_buffer_no_host(
        _a: &CudaBuffer,
        _a_scale: f32,
        _b: &CudaBuffer,
        _b_scale: f32,
        _m: usize,
        _n: usize,
        _k: usize,
    ) -> Result<(CudaBuffer, f32), String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn batch_matmul_f32(
        _lhs: &CudaBuffer,
        _rhs: &CudaBuffer,
        _batch_count: usize,
        _m: usize,
        _n: usize,
        _k: usize,
    ) -> Result<(CudaBuffer, Vec<f32>), String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn batch_matmul_f32_no_host(
        _lhs: &CudaBuffer,
        _rhs: &CudaBuffer,
        _batch_count: usize,
        _m: usize,
        _n: usize,
        _k: usize,
    ) -> Result<CudaBuffer, String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn batch_matmul_bf16_buffer_no_host(
        _lhs: &CudaBuffer,
        _rhs: &CudaBuffer,
        _batch_count: usize,
        _m: usize,
        _n: usize,
        _k: usize,
    ) -> Result<CudaBuffer, String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn batch_matmul_f16_buffer_no_host(
        _lhs: &CudaBuffer,
        _rhs: &CudaBuffer,
        _batch_count: usize,
        _m: usize,
        _n: usize,
        _k: usize,
    ) -> Result<CudaBuffer, String> {
        Err("CUDA feature is disabled".to_string())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn batch_matmul_i8_buffer_no_host(
        _lhs: &CudaBuffer,
        _lhs_scale: f32,
        _rhs: &CudaBuffer,
        _rhs_scale: f32,
        _batch_count: usize,
        _m: usize,
        _n: usize,
        _k: usize,
    ) -> Result<CudaBuffer, String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn batch_matmul_i8_typed_output_buffer_no_host(
        _lhs: &CudaBuffer,
        _lhs_scale: f32,
        _rhs: &CudaBuffer,
        _rhs_scale: f32,
        _batch_count: usize,
        _m: usize,
        _n: usize,
        _k: usize,
    ) -> Result<(CudaBuffer, f32), String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn matmul_backward_f32_no_host(
        _grad: &CudaBuffer,
        _a: &CudaBuffer,
        _b: &CudaBuffer,
        _m: usize,
        _k: usize,
        _n: usize,
    ) -> Result<(CudaBuffer, CudaBuffer), String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn batch_matmul_backward_f32_no_host(
        _grad: &CudaBuffer,
        _lhs: &CudaBuffer,
        _rhs: &CudaBuffer,
        _batch_count: usize,
        _m: usize,
        _k: usize,
        _n: usize,
    ) -> Result<(CudaBuffer, CudaBuffer), String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn batch_matmul_backward_bf16_no_host(
        _grad: &CudaBuffer,
        _lhs: &CudaBuffer,
        _rhs: &CudaBuffer,
        _batch_count: usize,
        _m: usize,
        _k: usize,
        _n: usize,
    ) -> Result<(CudaBuffer, CudaBuffer), String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn batch_matmul_backward_f16_no_host(
        _grad: &CudaBuffer,
        _lhs: &CudaBuffer,
        _rhs: &CudaBuffer,
        _batch_count: usize,
        _m: usize,
        _k: usize,
        _n: usize,
    ) -> Result<(CudaBuffer, CudaBuffer), String> {
        Err("CUDA feature is disabled".to_string())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn batch_matmul_backward_i8_no_host(
        _grad: &CudaBuffer,
        _lhs: &CudaBuffer,
        _lhs_scale: f32,
        _rhs: &CudaBuffer,
        _rhs_scale: f32,
        _batch_count: usize,
        _m: usize,
        _k: usize,
        _n: usize,
    ) -> Result<(CudaBuffer, CudaBuffer), String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn unary_f32(_input: &CudaBuffer, _op: UnaryOp) -> Result<(CudaBuffer, Vec<f32>), String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn unary_f32_buffer(_input: &CudaBuffer, _op: UnaryOp) -> Result<CudaBuffer, String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn unary_f16_buffer(_input: &CudaBuffer, _op: UnaryOp) -> Result<CudaBuffer, String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn unary_bf16_buffer(_input: &CudaBuffer, _op: UnaryOp) -> Result<CudaBuffer, String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn unary_i8_buffer(
        _input: &CudaBuffer,
        _scale: f32,
        _op: UnaryOp,
    ) -> Result<CudaBuffer, String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn unary_i8_relu_typed_output_buffer(_input: &CudaBuffer) -> Result<CudaBuffer, String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn unary_backward_f32(
        _input: &CudaBuffer,
        _output: &CudaBuffer,
        _grad: &CudaBuffer,
        _op: UnaryOp,
    ) -> Result<(CudaBuffer, Vec<f32>), String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn unary_backward_f32_buffer(
        _input: &CudaBuffer,
        _output: &CudaBuffer,
        _grad: &CudaBuffer,
        _op: UnaryOp,
    ) -> Result<CudaBuffer, String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn unary_backward_f16_buffer(
        _input: &CudaBuffer,
        _output: &CudaBuffer,
        _grad: &CudaBuffer,
        _op: UnaryOp,
    ) -> Result<CudaBuffer, String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn unary_backward_bf16_buffer(
        _input: &CudaBuffer,
        _output: &CudaBuffer,
        _grad: &CudaBuffer,
        _op: UnaryOp,
    ) -> Result<CudaBuffer, String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn unary_backward_i8_buffer(
        _input: &CudaBuffer,
        _scale: f32,
        _output: &CudaBuffer,
        _grad: &CudaBuffer,
        _op: UnaryOp,
    ) -> Result<CudaBuffer, String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn binary_f32(
        _lhs: &CudaBuffer,
        _rhs: &CudaBuffer,
        _op: BinaryOp,
    ) -> Result<(CudaBuffer, Vec<f32>), String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn binary_f32_buffer(
        _lhs: &CudaBuffer,
        _rhs: &CudaBuffer,
        _op: BinaryOp,
    ) -> Result<CudaBuffer, String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn binary_typed_buffer_no_host(
        _lhs: &CudaBuffer,
        _lhs_dtype: crate::precision::DType,
        _lhs_scale: Option<f32>,
        _rhs: &CudaBuffer,
        _rhs_dtype: crate::precision::DType,
        _rhs_scale: Option<f32>,
        _op: BinaryOp,
    ) -> Result<CudaBuffer, String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn binary_lowp_typed_output_buffer_no_host(
        _lhs: &CudaBuffer,
        _rhs: &CudaBuffer,
        _dtype: crate::precision::DType,
        _op: BinaryOp,
    ) -> Result<CudaBuffer, String> {
        Err("CUDA feature is disabled".to_string())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn binary_typed_lastdim_buffer_no_host(
        _lhs: &CudaBuffer,
        _lhs_dtype: crate::precision::DType,
        _lhs_scale: Option<f32>,
        _rhs: &CudaBuffer,
        _rhs_dtype: crate::precision::DType,
        _rhs_scale: Option<f32>,
        _out_len: usize,
        _last_dim: usize,
        _vector_on_rhs: bool,
        _op: BinaryOp,
    ) -> Result<CudaBuffer, String> {
        Err("CUDA feature is disabled".to_string())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn binary_lowp_typed_lastdim_output_buffer_no_host(
        _lhs: &CudaBuffer,
        _rhs: &CudaBuffer,
        _out_len: usize,
        _last_dim: usize,
        _vector_on_rhs: bool,
        _dtype: crate::precision::DType,
        _op: BinaryOp,
    ) -> Result<CudaBuffer, String> {
        Err("CUDA feature is disabled".to_string())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn binary_typed_row_scalar_buffer_no_host(
        _lhs: &CudaBuffer,
        _lhs_dtype: crate::precision::DType,
        _lhs_scale: Option<f32>,
        _rhs: &CudaBuffer,
        _rhs_dtype: crate::precision::DType,
        _rhs_scale: Option<f32>,
        _rows: usize,
        _last_dim: usize,
        _scalar_on_rhs: bool,
        _op: BinaryOp,
    ) -> Result<CudaBuffer, String> {
        Err("CUDA feature is disabled".to_string())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn binary_lowp_typed_row_scalar_output_buffer_no_host(
        _lhs: &CudaBuffer,
        _rhs: &CudaBuffer,
        _rows: usize,
        _last_dim: usize,
        _scalar_on_rhs: bool,
        _dtype: crate::precision::DType,
        _op: BinaryOp,
    ) -> Result<CudaBuffer, String> {
        Err("CUDA feature is disabled".to_string())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn binary_i8_typed_row_scalar_output_buffer_no_host(
        _lhs: &CudaBuffer,
        _lhs_scale: f32,
        _rhs: &CudaBuffer,
        _rhs_scale: f32,
        _rows: usize,
        _last_dim: usize,
        _scalar_on_rhs: bool,
        _op: BinaryOp,
    ) -> Result<(CudaBuffer, f32), String> {
        Err("CUDA feature is disabled".to_string())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn binary_typed_broadcast_buffer_no_host(
        _lhs: &CudaBuffer,
        _lhs_dtype: crate::precision::DType,
        _lhs_scale: Option<f32>,
        _rhs: &CudaBuffer,
        _rhs_dtype: crate::precision::DType,
        _rhs_scale: Option<f32>,
        _lhs_shape: &[usize],
        _rhs_shape: &[usize],
        _out_shape: &[usize],
        _op: BinaryOp,
    ) -> Result<CudaBuffer, String> {
        Err("CUDA feature is disabled".to_string())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn binary_lowp_typed_broadcast_output_buffer_no_host(
        _lhs: &CudaBuffer,
        _rhs: &CudaBuffer,
        _lhs_shape: &[usize],
        _rhs_shape: &[usize],
        _out_shape: &[usize],
        _dtype: crate::precision::DType,
        _op: BinaryOp,
    ) -> Result<CudaBuffer, String> {
        Err("CUDA feature is disabled".to_string())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn binary_i8_typed_broadcast_output_buffer_no_host(
        _lhs: &CudaBuffer,
        _lhs_scale: f32,
        _rhs: &CudaBuffer,
        _rhs_scale: f32,
        _lhs_shape: &[usize],
        _rhs_shape: &[usize],
        _out_shape: &[usize],
        _op: BinaryOp,
    ) -> Result<(CudaBuffer, f32), String> {
        Err("CUDA feature is disabled".to_string())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn binary_typed_b1d_1h1_buffer_no_host(
        _lhs: &CudaBuffer,
        _lhs_dtype: crate::precision::DType,
        _lhs_scale: Option<f32>,
        _rhs: &CudaBuffer,
        _rhs_dtype: crate::precision::DType,
        _rhs_scale: Option<f32>,
        _batch: usize,
        _heads: usize,
        _dim: usize,
        _b1d_on_lhs: bool,
        _op: BinaryOp,
    ) -> Result<CudaBuffer, String> {
        Err("CUDA feature is disabled".to_string())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn binary_lowp_typed_b1d_1h1_output_buffer_no_host(
        _lhs: &CudaBuffer,
        _rhs: &CudaBuffer,
        _batch: usize,
        _heads: usize,
        _dim: usize,
        _b1d_on_lhs: bool,
        _dtype: crate::precision::DType,
        _op: BinaryOp,
    ) -> Result<CudaBuffer, String> {
        Err("CUDA feature is disabled".to_string())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn binary_i8_typed_b1d_1h1_output_buffer_no_host(
        _lhs: &CudaBuffer,
        _lhs_scale: f32,
        _rhs: &CudaBuffer,
        _rhs_scale: f32,
        _batch: usize,
        _heads: usize,
        _dim: usize,
        _b1d_on_lhs: bool,
        _op: BinaryOp,
    ) -> Result<(CudaBuffer, f32), String> {
        Err("CUDA feature is disabled".to_string())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn binary_typed_b1d_1hd_buffer_no_host(
        _lhs: &CudaBuffer,
        _lhs_dtype: crate::precision::DType,
        _lhs_scale: Option<f32>,
        _rhs: &CudaBuffer,
        _rhs_dtype: crate::precision::DType,
        _rhs_scale: Option<f32>,
        _batch: usize,
        _heads: usize,
        _dim: usize,
        _b1d_on_lhs: bool,
        _op: BinaryOp,
    ) -> Result<CudaBuffer, String> {
        Err("CUDA feature is disabled".to_string())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn binary_lowp_typed_b1d_1hd_output_buffer_no_host(
        _lhs: &CudaBuffer,
        _rhs: &CudaBuffer,
        _batch: usize,
        _heads: usize,
        _dim: usize,
        _b1d_on_lhs: bool,
        _dtype: crate::precision::DType,
        _op: BinaryOp,
    ) -> Result<CudaBuffer, String> {
        Err("CUDA feature is disabled".to_string())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn binary_i8_typed_b1d_1hd_output_buffer_no_host(
        _lhs: &CudaBuffer,
        _lhs_scale: f32,
        _rhs: &CudaBuffer,
        _rhs_scale: f32,
        _batch: usize,
        _heads: usize,
        _dim: usize,
        _b1d_on_lhs: bool,
        _op: BinaryOp,
    ) -> Result<(CudaBuffer, f32), String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn binary_f16_host_no_host(
        _lhs: &[u16],
        _rhs: &[u16],
        _op: BinaryOp,
    ) -> Result<CudaBuffer, String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn binary_f16_buffer_no_host(
        _lhs: &CudaBuffer,
        _rhs: &CudaBuffer,
        _op: BinaryOp,
    ) -> Result<CudaBuffer, String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn binary_f16_lastdim_buffer_no_host(
        _lhs: &CudaBuffer,
        _rhs: &CudaBuffer,
        _out_len: usize,
        _last_dim: usize,
        _vector_on_rhs: bool,
        _op: BinaryOp,
    ) -> Result<CudaBuffer, String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn binary_bf16_host_no_host(
        _lhs: &[u16],
        _rhs: &[u16],
        _op: BinaryOp,
    ) -> Result<CudaBuffer, String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn binary_bf16_buffer_no_host(
        _lhs: &CudaBuffer,
        _rhs: &CudaBuffer,
        _op: BinaryOp,
    ) -> Result<CudaBuffer, String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn binary_bf16_lastdim_buffer_no_host(
        _lhs: &CudaBuffer,
        _rhs: &CudaBuffer,
        _out_len: usize,
        _last_dim: usize,
        _vector_on_rhs: bool,
        _op: BinaryOp,
    ) -> Result<CudaBuffer, String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn binary_i8_host_no_host(
        _lhs: &[i8],
        _lhs_scale: f32,
        _rhs: &[i8],
        _rhs_scale: f32,
        _op: BinaryOp,
    ) -> Result<CudaBuffer, String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn binary_i8_buffer_no_host(
        _lhs: &CudaBuffer,
        _lhs_scale: f32,
        _rhs: &CudaBuffer,
        _rhs_scale: f32,
        _op: BinaryOp,
    ) -> Result<CudaBuffer, String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn binary_i8_typed_output_buffer_no_host(
        _lhs: &CudaBuffer,
        _lhs_scale: f32,
        _rhs: &CudaBuffer,
        _rhs_scale: f32,
        _op: BinaryOp,
    ) -> Result<(CudaBuffer, f32), String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn binary_i8_lastdim_buffer_no_host(
        _lhs: &CudaBuffer,
        _lhs_scale: f32,
        _rhs: &CudaBuffer,
        _rhs_scale: f32,
        _out_len: usize,
        _last_dim: usize,
        _vector_on_rhs: bool,
        _op: BinaryOp,
    ) -> Result<CudaBuffer, String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn binary_i8_typed_lastdim_output_buffer_no_host(
        _lhs: &CudaBuffer,
        _lhs_scale: f32,
        _rhs: &CudaBuffer,
        _rhs_scale: f32,
        _out_len: usize,
        _last_dim: usize,
        _vector_on_rhs: bool,
        _op: BinaryOp,
    ) -> Result<(CudaBuffer, f32), String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn mul_grad_f16_host_no_host(
        _grad: &CudaBuffer,
        _operand: &[u16],
    ) -> Result<CudaBuffer, String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn mul_grad_f16_buffer_no_host(
        _grad: &CudaBuffer,
        _operand: &CudaBuffer,
    ) -> Result<CudaBuffer, String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn mul_grad_f16_lastdim_buffer_no_host(
        _grad: &CudaBuffer,
        _lhs: &CudaBuffer,
        _rhs: &CudaBuffer,
        _out_len: usize,
        _last_dim: usize,
        _vector_on_rhs: bool,
    ) -> Result<(CudaBuffer, CudaBuffer), String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn mul_grad_bf16_host_no_host(
        _grad: &CudaBuffer,
        _operand: &[u16],
    ) -> Result<CudaBuffer, String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn mul_grad_bf16_buffer_no_host(
        _grad: &CudaBuffer,
        _operand: &CudaBuffer,
    ) -> Result<CudaBuffer, String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn mul_grad_bf16_lastdim_buffer_no_host(
        _grad: &CudaBuffer,
        _lhs: &CudaBuffer,
        _rhs: &CudaBuffer,
        _out_len: usize,
        _last_dim: usize,
        _vector_on_rhs: bool,
    ) -> Result<(CudaBuffer, CudaBuffer), String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn mul_grad_i8_host_no_host(
        _grad: &CudaBuffer,
        _operand: &[i8],
        _scale: f32,
    ) -> Result<CudaBuffer, String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn mul_grad_i8_buffer_no_host(
        _grad: &CudaBuffer,
        _operand: &CudaBuffer,
        _scale: f32,
    ) -> Result<CudaBuffer, String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn mul_grad_i8_lastdim_buffer_no_host(
        _grad: &CudaBuffer,
        _lhs: &CudaBuffer,
        _lhs_scale: f32,
        _rhs: &CudaBuffer,
        _rhs_scale: f32,
        _out_len: usize,
        _last_dim: usize,
        _vector_on_rhs: bool,
    ) -> Result<(CudaBuffer, CudaBuffer), String> {
        Err("CUDA feature is disabled".to_string())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn mul_grad_typed_lastdim_buffer_no_host(
        _grad: &CudaBuffer,
        _lhs: &CudaBuffer,
        _lhs_dtype: crate::precision::DType,
        _lhs_scale: Option<f32>,
        _rhs: &CudaBuffer,
        _rhs_dtype: crate::precision::DType,
        _rhs_scale: Option<f32>,
        _out_len: usize,
        _last_dim: usize,
        _vector_on_rhs: bool,
    ) -> Result<(CudaBuffer, CudaBuffer), String> {
        Err("CUDA feature is disabled".to_string())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn mul_grad_typed_buffer_no_host(
        _grad: &CudaBuffer,
        _lhs: &CudaBuffer,
        _lhs_dtype: crate::precision::DType,
        _lhs_scale: Option<f32>,
        _rhs: &CudaBuffer,
        _rhs_dtype: crate::precision::DType,
        _rhs_scale: Option<f32>,
    ) -> Result<(CudaBuffer, CudaBuffer), String> {
        Err("CUDA feature is disabled".to_string())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn mul_grad_typed_broadcast_buffer_no_host(
        _grad: &CudaBuffer,
        _lhs: &CudaBuffer,
        _lhs_dtype: crate::precision::DType,
        _lhs_scale: Option<f32>,
        _rhs: &CudaBuffer,
        _rhs_dtype: crate::precision::DType,
        _rhs_scale: Option<f32>,
        _lhs_shape: &[usize],
        _rhs_shape: &[usize],
        _out_shape: &[usize],
    ) -> Result<(CudaBuffer, CudaBuffer), String> {
        Err("CUDA feature is disabled".to_string())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn mul_grad_typed_b1d_1h1_buffer_no_host(
        _grad: &CudaBuffer,
        _lhs: &CudaBuffer,
        _lhs_dtype: crate::precision::DType,
        _lhs_scale: Option<f32>,
        _rhs: &CudaBuffer,
        _rhs_dtype: crate::precision::DType,
        _rhs_scale: Option<f32>,
        _batch: usize,
        _heads: usize,
        _dim: usize,
        _b1d_on_lhs: bool,
    ) -> Result<(CudaBuffer, CudaBuffer), String> {
        Err("CUDA feature is disabled".to_string())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn mul_grad_typed_b1d_1hd_buffer_no_host(
        _grad: &CudaBuffer,
        _lhs: &CudaBuffer,
        _lhs_dtype: crate::precision::DType,
        _lhs_scale: Option<f32>,
        _rhs: &CudaBuffer,
        _rhs_dtype: crate::precision::DType,
        _rhs_scale: Option<f32>,
        _batch: usize,
        _heads: usize,
        _dim: usize,
        _b1d_on_lhs: bool,
    ) -> Result<(CudaBuffer, CudaBuffer), String> {
        Err("CUDA feature is disabled".to_string())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn mul_grad_typed_row_scalar_buffer_no_host(
        _grad: &CudaBuffer,
        _lhs: &CudaBuffer,
        _lhs_dtype: crate::precision::DType,
        _lhs_scale: Option<f32>,
        _rhs: &CudaBuffer,
        _rhs_dtype: crate::precision::DType,
        _rhs_scale: Option<f32>,
        _rows: usize,
        _last_dim: usize,
        _scalar_on_rhs: bool,
    ) -> Result<(CudaBuffer, CudaBuffer), String> {
        Err("CUDA feature is disabled".to_string())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn mul_grad_typed_scalar_buffer_no_host(
        _grad: &CudaBuffer,
        _lhs: &CudaBuffer,
        _lhs_dtype: crate::precision::DType,
        _lhs_scale: Option<f32>,
        _rhs: &CudaBuffer,
        _rhs_dtype: crate::precision::DType,
        _rhs_scale: Option<f32>,
        _out_len: usize,
        _scalar_on_rhs: bool,
    ) -> Result<(CudaBuffer, CudaBuffer), String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn binary_backward_f32(
        _lhs: &CudaBuffer,
        _rhs: &CudaBuffer,
        _grad: &CudaBuffer,
        _op: BinaryOp,
    ) -> Result<CudaTwoHostBuffers, String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn binary_backward_f32_buffers(
        _lhs: &CudaBuffer,
        _rhs: &CudaBuffer,
        _grad: &CudaBuffer,
        _op: BinaryOp,
    ) -> Result<(CudaBuffer, CudaBuffer), String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn add_sub_backward_f32_buffers(
        _grad: &CudaBuffer,
        _len: usize,
        _op: BinaryOp,
    ) -> Result<(CudaBuffer, CudaBuffer), String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn add_sub_backward_lastdim_f32_buffers(
        _grad: &CudaBuffer,
        _out_len: usize,
        _last_dim: usize,
        _vector_on_rhs: bool,
        _op: BinaryOp,
    ) -> Result<(CudaBuffer, CudaBuffer), String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn add_sub_backward_scalar_f32_buffers(
        _grad: &CudaBuffer,
        _out_len: usize,
        _scalar_on_rhs: bool,
        _op: BinaryOp,
    ) -> Result<(CudaBuffer, CudaBuffer), String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn add_sub_backward_row_scalar_f32_buffers(
        _grad: &CudaBuffer,
        _rows: usize,
        _last_dim: usize,
        _scalar_on_rhs: bool,
        _op: BinaryOp,
    ) -> Result<(CudaBuffer, CudaBuffer), String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn add_sub_backward_b1d_1h1_f32_buffers(
        _grad: &CudaBuffer,
        _batch: usize,
        _heads: usize,
        _dim: usize,
        _b1d_on_lhs: bool,
        _op: BinaryOp,
    ) -> Result<(CudaBuffer, CudaBuffer), String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn add_sub_backward_b1d_1hd_f32_buffers(
        _grad: &CudaBuffer,
        _batch: usize,
        _heads: usize,
        _dim: usize,
        _b1d_on_lhs: bool,
        _op: BinaryOp,
    ) -> Result<(CudaBuffer, CudaBuffer), String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn add_sub_broadcast_backward_f32_buffers(
        _grad: &CudaBuffer,
        _lhs_shape: &[usize],
        _rhs_shape: &[usize],
        _out_shape: &[usize],
        _op: BinaryOp,
    ) -> Result<(CudaBuffer, CudaBuffer), String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn binary_broadcast_f32(
        _lhs: &CudaBuffer,
        _rhs: &CudaBuffer,
        _lhs_shape: &[usize],
        _rhs_shape: &[usize],
        _out_shape: &[usize],
        _op: BinaryOp,
    ) -> Result<(CudaBuffer, Vec<f32>), String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn binary_broadcast_f32_buffer(
        _lhs: &CudaBuffer,
        _rhs: &CudaBuffer,
        _lhs_shape: &[usize],
        _rhs_shape: &[usize],
        _out_shape: &[usize],
        _op: BinaryOp,
    ) -> Result<CudaBuffer, String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn binary_broadcast_backward_f32(
        _lhs: &CudaBuffer,
        _rhs: &CudaBuffer,
        _grad: &CudaBuffer,
        _lhs_shape: &[usize],
        _rhs_shape: &[usize],
        _out_shape: &[usize],
        _op: BinaryOp,
    ) -> Result<CudaTwoHostBuffers, String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn binary_broadcast_backward_f32_buffers(
        _lhs: &CudaBuffer,
        _rhs: &CudaBuffer,
        _grad: &CudaBuffer,
        _lhs_shape: &[usize],
        _rhs_shape: &[usize],
        _out_shape: &[usize],
        _op: BinaryOp,
    ) -> Result<(CudaBuffer, CudaBuffer), String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn sum_f32(_input: &CudaBuffer) -> Result<(CudaBuffer, Vec<f32>), String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn sum_f16_buffer(_input: &CudaBuffer) -> Result<(CudaBuffer, Vec<f32>), String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn sum_bf16_buffer(_input: &CudaBuffer) -> Result<(CudaBuffer, Vec<f32>), String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn sum_i8_buffer(
        _input: &CudaBuffer,
        _scale: f32,
    ) -> Result<(CudaBuffer, Vec<f32>), String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn fill_scalar_f32(_len: usize, _value: f32) -> Result<(CudaBuffer, Vec<f32>), String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn fill_scalar_f32_buffer(_len: usize, _value: f32) -> Result<CudaBuffer, String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn add_inplace_f32(_dst: &CudaBuffer, _src: &CudaBuffer) -> Result<(), String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn sum_lastdim_f32_buffer(
        _input: &CudaBuffer,
        _rows: usize,
        _last_dim: usize,
    ) -> Result<CudaBuffer, String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn bshd_to_bhsd_add_bias_f32_buffer(
        _input: &CudaBuffer,
        _bias: &CudaBuffer,
        _batch: usize,
        _seq: usize,
        _heads: usize,
        _dim: usize,
    ) -> Result<CudaBuffer, String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn mse_forward_typed(
        _output: &CudaBuffer,
        _output_dtype: crate::precision::DType,
        _output_scale: Option<f32>,
        _target: &CudaBuffer,
        _target_dtype: crate::precision::DType,
        _target_scale: Option<f32>,
    ) -> Result<(CudaBuffer, CudaBuffer, Vec<f32>), String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn mse_backward_f32(
        _diff: &CudaBuffer,
        _factor: f32,
    ) -> Result<CudaTwoHostBuffers, String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn mse_backward_f32_buffers(
        _diff: &CudaBuffer,
        _factor: f32,
    ) -> Result<(CudaBuffer, CudaBuffer), String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn cross_entropy_backward_f32(
        _softmax: &CudaBuffer,
        _target: &CudaBuffer,
        _factor: f32,
    ) -> Result<(CudaBuffer, Vec<f32>), String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn cross_entropy_backward_f32_buffer(
        _softmax: &CudaBuffer,
        _target: &CudaBuffer,
        _factor: f32,
    ) -> Result<CudaBuffer, String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn cross_entropy_loss_f32(
        _softmax: &CudaBuffer,
        _target: &CudaBuffer,
        _batch_size: usize,
    ) -> Result<(CudaBuffer, Vec<f32>), String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn cross_entropy_backward_typed_target_buffer(
        _softmax: &CudaBuffer,
        _target: &CudaBuffer,
        _target_dtype: crate::precision::DType,
        _target_scale: Option<f32>,
        _factor: f32,
    ) -> Result<CudaBuffer, String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn cross_entropy_backward_typed_target(
        _softmax: &CudaBuffer,
        _target: &CudaBuffer,
        _target_dtype: crate::precision::DType,
        _target_scale: Option<f32>,
        _factor: f32,
    ) -> Result<(CudaBuffer, Vec<f32>), String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn cross_entropy_loss_typed_target(
        _softmax: &CudaBuffer,
        _target: &CudaBuffer,
        _target_dtype: crate::precision::DType,
        _target_scale: Option<f32>,
        _batch_size: usize,
    ) -> Result<(CudaBuffer, Vec<f32>), String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn sgd_update_f32(
        _param: &CudaBuffer,
        _grad: &CudaBuffer,
        _lr: f32,
    ) -> Result<Vec<f32>, String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn sgd_update_f32_no_host(
        _param: &CudaBuffer,
        _grad: &CudaBuffer,
        _lr: f32,
    ) -> Result<(), String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn sgd_update_f32_batched_no_host(
        _params: &[CudaBuffer],
        _grads: &[CudaBuffer],
        _lr: f32,
    ) -> Result<(), String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn quantize_f32_storage_no_host(
        _param: &CudaBuffer,
        _dtype: crate::precision::DType,
        _scale: Option<f32>,
    ) -> Result<(), String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn quantize_f32_to_i8_dynamic_no_host(
        _input: &CudaBuffer,
    ) -> Result<(CudaBuffer, f32), String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn f32_to_lowp_storage_no_host(
        _input: &CudaBuffer,
        _dtype: crate::precision::DType,
    ) -> Result<CudaBuffer, String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn sgd_momentum_update_f32(
        _param: &CudaBuffer,
        _grad: &CudaBuffer,
        _velocity: &CudaBuffer,
        _lr: f32,
        _momentum: f32,
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn sgd_momentum_update_f32_no_host(
        _param: &CudaBuffer,
        _grad: &CudaBuffer,
        _velocity: &CudaBuffer,
        _lr: f32,
        _momentum: f32,
    ) -> Result<(), String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn sgd_momentum_update_f32_batched_no_host(
        _params: &[CudaBuffer],
        _grads: &[CudaBuffer],
        _velocities: &[CudaBuffer],
        _lr: f32,
        _momentum: f32,
    ) -> Result<(), String> {
        Err("CUDA feature is disabled".to_string())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn adam_update_f32(
        _param: &CudaBuffer,
        _grad: &CudaBuffer,
        _exp_avg: &CudaBuffer,
        _exp_avg_sq: &CudaBuffer,
        _lr: f32,
        _beta1: f32,
        _beta2: f32,
        _bias_correction1: f32,
        _bias_correction2: f32,
        _eps: f32,
    ) -> Result<CudaAdamHostState, String> {
        Err("CUDA feature is disabled".to_string())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn adam_update_f32_no_host(
        _param: &CudaBuffer,
        _grad: &CudaBuffer,
        _exp_avg: &CudaBuffer,
        _exp_avg_sq: &CudaBuffer,
        _lr: f32,
        _beta1: f32,
        _beta2: f32,
        _bias_correction1: f32,
        _bias_correction2: f32,
        _eps: f32,
    ) -> Result<(), String> {
        Err("CUDA feature is disabled".to_string())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn adam_update_f32_batched_no_host(
        _params: &[CudaBuffer],
        _grads: &[CudaBuffer],
        _exp_avgs: &[CudaBuffer],
        _exp_avg_sqs: &[CudaBuffer],
        _lr: f32,
        _beta1: f32,
        _beta2: f32,
        _bias_correction1: f32,
        _bias_correction2: f32,
        _eps: f32,
    ) -> Result<(), String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn softmax_lastdim_f32(
        _input: &CudaBuffer,
        _outer: usize,
        _last_dim: usize,
    ) -> Result<(CudaBuffer, Vec<f32>), String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn softmax_lastdim_f32_no_host(
        _input: &CudaBuffer,
        _outer: usize,
        _last_dim: usize,
    ) -> Result<CudaBuffer, String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn softmax_lastdim_typed(
        _input: &CudaBuffer,
        _input_dtype: crate::precision::DType,
        _input_scale: Option<f32>,
        _outer: usize,
        _last_dim: usize,
    ) -> Result<(CudaBuffer, Vec<f32>), String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn softmax_lastdim_typed_no_host(
        _input: &CudaBuffer,
        _input_dtype: crate::precision::DType,
        _input_scale: Option<f32>,
        _outer: usize,
        _last_dim: usize,
    ) -> Result<CudaBuffer, String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn softmax_lastdim_backward_f32(
        _output: &CudaBuffer,
        _grad: &CudaBuffer,
        _outer: usize,
        _last_dim: usize,
    ) -> Result<(CudaBuffer, Vec<f32>), String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn softmax_lastdim_backward_f32_buffer(
        _output: &CudaBuffer,
        _grad: &CudaBuffer,
        _outer: usize,
        _last_dim: usize,
    ) -> Result<CudaBuffer, String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn fused_softmax_f32(
        _input: &CudaBuffer,
        _batch_heads: usize,
        _q_len: usize,
        _k_len: usize,
        _scale: f32,
        _is_causal: bool,
    ) -> Result<(CudaBuffer, Vec<f32>), String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn fused_softmax_f32_no_host(
        _input: &CudaBuffer,
        _batch_heads: usize,
        _q_len: usize,
        _k_len: usize,
        _scale: f32,
        _is_causal: bool,
    ) -> Result<CudaBuffer, String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn fused_softmax_backward_f32(
        _output: &CudaBuffer,
        _grad: &CudaBuffer,
        _batch_heads: usize,
        _q_len: usize,
        _k_len: usize,
        _scale: f32,
    ) -> Result<(CudaBuffer, Vec<f32>), String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn fused_softmax_backward_f32_buffer(
        _output: &CudaBuffer,
        _grad: &CudaBuffer,
        _batch_heads: usize,
        _q_len: usize,
        _k_len: usize,
        _scale: f32,
    ) -> Result<CudaBuffer, String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn fused_softmax_f32_with_past(
        _input: &CudaBuffer,
        _batch_heads: usize,
        _q_len: usize,
        _k_len: usize,
        _scale: f32,
        _is_causal: bool,
        _past_len: usize,
    ) -> Result<(CudaBuffer, Vec<f32>), String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn fused_softmax_f32_with_past_no_host(
        _input: &CudaBuffer,
        _batch_heads: usize,
        _q_len: usize,
        _k_len: usize,
        _scale: f32,
        _is_causal: bool,
        _past_len: usize,
    ) -> Result<CudaBuffer, String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn embedding_f32(
        _indices: &CudaBuffer,
        _weight: &CudaBuffer,
        _num_indices: usize,
        _vocab_size: usize,
        _embed_dim: usize,
    ) -> Result<(CudaBuffer, Vec<f32>), String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn embedding_f32_buffer(
        _indices: &CudaBuffer,
        _weight: &CudaBuffer,
        _num_indices: usize,
        _vocab_size: usize,
        _embed_dim: usize,
    ) -> Result<CudaBuffer, String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn embedding_typed(
        _indices: &CudaBuffer,
        _weight: &CudaBuffer,
        _weight_dtype: crate::precision::DType,
        _weight_scale: Option<f32>,
        _num_indices: usize,
        _vocab_size: usize,
        _embed_dim: usize,
    ) -> Result<(CudaBuffer, Vec<f32>), String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn embedding_typed_buffer(
        _indices: &CudaBuffer,
        _weight: &CudaBuffer,
        _weight_dtype: crate::precision::DType,
        _weight_scale: Option<f32>,
        _num_indices: usize,
        _vocab_size: usize,
        _embed_dim: usize,
    ) -> Result<CudaBuffer, String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn embedding_typed_same_dtype_buffer(
        _indices: &CudaBuffer,
        _weight: &CudaBuffer,
        _weight_dtype: crate::precision::DType,
        _num_indices: usize,
        _vocab_size: usize,
        _embed_dim: usize,
    ) -> Result<CudaBuffer, String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn embedding_backward_f32(
        _indices: &CudaBuffer,
        _grad: &CudaBuffer,
        _num_indices: usize,
        _vocab_size: usize,
        _embed_dim: usize,
    ) -> Result<(CudaBuffer, Vec<f32>), String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn embedding_backward_f32_buffer(
        _indices: &CudaBuffer,
        _grad: &CudaBuffer,
        _num_indices: usize,
        _vocab_size: usize,
        _embed_dim: usize,
    ) -> Result<CudaBuffer, String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn rms_norm_f32(
        _input: &CudaBuffer,
        _weight: &CudaBuffer,
        _rows: usize,
        _dim: usize,
        _eps: f32,
    ) -> Result<(CudaBuffer, Vec<f32>), String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn rms_norm_f32_buffer(
        _input: &CudaBuffer,
        _weight: &CudaBuffer,
        _rows: usize,
        _dim: usize,
        _eps: f32,
    ) -> Result<CudaBuffer, String> {
        Err("CUDA feature is disabled".to_string())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn rms_norm_typed(
        _input: &CudaBuffer,
        _input_dtype: crate::precision::DType,
        _input_scale: Option<f32>,
        _weight: &CudaBuffer,
        _weight_dtype: crate::precision::DType,
        _weight_scale: Option<f32>,
        _rows: usize,
        _dim: usize,
        _eps: f32,
    ) -> Result<(CudaBuffer, Vec<f32>), String> {
        Err("CUDA feature is disabled".to_string())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn rms_norm_typed_buffer(
        _input: &CudaBuffer,
        _input_dtype: crate::precision::DType,
        _input_scale: Option<f32>,
        _weight: &CudaBuffer,
        _weight_dtype: crate::precision::DType,
        _weight_scale: Option<f32>,
        _rows: usize,
        _dim: usize,
        _eps: f32,
    ) -> Result<CudaBuffer, String> {
        Err("CUDA feature is disabled".to_string())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn rms_norm_i8_typed_output_buffer(
        _input: &CudaBuffer,
        _input_dtype: crate::precision::DType,
        _input_scale: Option<f32>,
        _weight: &CudaBuffer,
        _weight_dtype: crate::precision::DType,
        _weight_scale: Option<f32>,
        _rows: usize,
        _dim: usize,
        _eps: f32,
    ) -> Result<(CudaBuffer, f32), String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn rms_norm_backward_f32(
        _input: &CudaBuffer,
        _weight: &CudaBuffer,
        _grad: &CudaBuffer,
        _rows: usize,
        _dim: usize,
        _eps: f32,
    ) -> Result<CudaTwoHostBuffers, String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn rms_norm_backward_f32_buffers(
        _input: &CudaBuffer,
        _weight: &CudaBuffer,
        _grad: &CudaBuffer,
        _rows: usize,
        _dim: usize,
        _eps: f32,
    ) -> Result<(CudaBuffer, CudaBuffer), String> {
        Err("CUDA feature is disabled".to_string())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn rms_norm_backward_typed(
        _input: &CudaBuffer,
        _input_dtype: crate::precision::DType,
        _input_scale: Option<f32>,
        _weight: &CudaBuffer,
        _weight_dtype: crate::precision::DType,
        _weight_scale: Option<f32>,
        _grad: &CudaBuffer,
        _rows: usize,
        _dim: usize,
        _eps: f32,
    ) -> Result<CudaTwoHostBuffers, String> {
        Err("CUDA feature is disabled".to_string())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn rms_norm_backward_typed_buffers(
        _input: &CudaBuffer,
        _input_dtype: crate::precision::DType,
        _input_scale: Option<f32>,
        _weight: &CudaBuffer,
        _weight_dtype: crate::precision::DType,
        _weight_scale: Option<f32>,
        _grad: &CudaBuffer,
        _rows: usize,
        _dim: usize,
        _eps: f32,
    ) -> Result<(CudaBuffer, CudaBuffer), String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn permute_f32(
        _input: &CudaBuffer,
        _out_shape: &[usize],
        _axes: &[usize],
    ) -> Result<(CudaBuffer, Vec<f32>), String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn permute_f32_buffer(
        _input: &CudaBuffer,
        _out_shape: &[usize],
        _axes: &[usize],
    ) -> Result<CudaBuffer, String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn permute_typed_buffer(
        _input: &CudaBuffer,
        _dtype: crate::precision::DType,
        _out_shape: &[usize],
        _axes: &[usize],
    ) -> Result<CudaBuffer, String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn slice_lastdim_f32(
        _input: &CudaBuffer,
        _outer: usize,
        _input_last_dim: usize,
        _start: usize,
        _slice_len: usize,
    ) -> Result<(CudaBuffer, Vec<f32>), String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn slice_lastdim_f32_buffer(
        _input: &CudaBuffer,
        _outer: usize,
        _input_last_dim: usize,
        _start: usize,
        _slice_len: usize,
    ) -> Result<CudaBuffer, String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn slice_lastdim_typed_buffer(
        _input: &CudaBuffer,
        _dtype: crate::precision::DType,
        _outer: usize,
        _input_last_dim: usize,
        _start: usize,
        _slice_len: usize,
    ) -> Result<CudaBuffer, String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn slice_lastdim_backward_f32(
        _grad: &CudaBuffer,
        _outer: usize,
        _input_last_dim: usize,
        _start: usize,
        _slice_len: usize,
    ) -> Result<(CudaBuffer, Vec<f32>), String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn slice_lastdim_backward_f32_buffer(
        _grad: &CudaBuffer,
        _outer: usize,
        _input_last_dim: usize,
        _start: usize,
        _slice_len: usize,
    ) -> Result<CudaBuffer, String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn cat_f32(
        _lhs: &CudaBuffer,
        _rhs: &CudaBuffer,
        _out_shape: &[usize],
        _axis: usize,
        _lhs_axis_len: usize,
    ) -> Result<(CudaBuffer, Vec<f32>), String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn cat_f32_buffer(
        _lhs: &CudaBuffer,
        _rhs: &CudaBuffer,
        _out_shape: &[usize],
        _axis: usize,
        _lhs_axis_len: usize,
    ) -> Result<CudaBuffer, String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn cat_typed_buffer(
        _lhs: &CudaBuffer,
        _rhs: &CudaBuffer,
        _dtype: crate::precision::DType,
        _out_shape: &[usize],
        _axis: usize,
        _lhs_axis_len: usize,
    ) -> Result<CudaBuffer, String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn cat_backward_slice_f32(
        _grad: &CudaBuffer,
        _input_shape: &[usize],
        _out_shape: &[usize],
        _axis: usize,
        _axis_start: usize,
    ) -> Result<(CudaBuffer, Vec<f32>), String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn cat_backward_slice_f32_buffer(
        _grad: &CudaBuffer,
        _input_shape: &[usize],
        _out_shape: &[usize],
        _axis: usize,
        _axis_start: usize,
    ) -> Result<CudaBuffer, String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn repeat_kv_f32(
        _input: &CudaBuffer,
        _batch_size: usize,
        _num_kv_heads: usize,
        _seq_len: usize,
        _dim: usize,
        _n_rep: usize,
    ) -> Result<(CudaBuffer, Vec<f32>), String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn repeat_kv_f32_buffer(
        _input: &CudaBuffer,
        _batch_size: usize,
        _num_kv_heads: usize,
        _seq_len: usize,
        _dim: usize,
        _n_rep: usize,
    ) -> Result<CudaBuffer, String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn repeat_kv_typed_buffer(
        _input: &CudaBuffer,
        _dtype: crate::precision::DType,
        _batch_size: usize,
        _num_kv_heads: usize,
        _seq_len: usize,
        _dim: usize,
        _n_rep: usize,
    ) -> Result<CudaBuffer, String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn repeat_kv_backward_f32(
        _grad: &CudaBuffer,
        _batch_size: usize,
        _num_kv_heads: usize,
        _seq_len: usize,
        _dim: usize,
        _n_rep: usize,
    ) -> Result<(CudaBuffer, Vec<f32>), String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn repeat_kv_backward_f32_buffer(
        _grad: &CudaBuffer,
        _batch_size: usize,
        _num_kv_heads: usize,
        _seq_len: usize,
        _dim: usize,
        _n_rep: usize,
    ) -> Result<CudaBuffer, String> {
        Err("CUDA feature is disabled".to_string())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn decode_attention_f32_buffer(
        _q: &CudaBuffer,
        _k: &CudaBuffer,
        _v: &CudaBuffer,
        _batch_size: usize,
        _num_heads: usize,
        _num_kv_heads: usize,
        _active_seq_len: usize,
        _cache_seq_len: usize,
        _dim: usize,
        _n_rep: usize,
        _scale: f32,
    ) -> Result<CudaBuffer, String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn fused_gate_up_silu_f32(
        _input: &CudaBuffer,
        _gate: &CudaBuffer,
        _up: &CudaBuffer,
        _rows: usize,
        _n_dim: usize,
        _k_dim: usize,
    ) -> Result<(CudaBuffer, Vec<f32>), String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn fused_gate_up_silu_f32_buffer(
        _input: &CudaBuffer,
        _gate: &CudaBuffer,
        _up: &CudaBuffer,
        _rows: usize,
        _n_dim: usize,
        _k_dim: usize,
    ) -> Result<CudaBuffer, String> {
        Err("CUDA feature is disabled".to_string())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn fused_gate_up_silu_typed(
        _input: &CudaBuffer,
        _input_dtype: crate::precision::DType,
        _input_scale: Option<f32>,
        _gate: &CudaBuffer,
        _weight_dtype: crate::precision::DType,
        _gate_scale: Option<f32>,
        _up: &CudaBuffer,
        _up_scale: Option<f32>,
        _rows: usize,
        _n_dim: usize,
        _k_dim: usize,
    ) -> Result<(CudaBuffer, Vec<f32>), String> {
        Err("CUDA feature is disabled".to_string())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn fused_gate_up_silu_typed_buffer(
        _input: &CudaBuffer,
        _input_dtype: crate::precision::DType,
        _input_scale: Option<f32>,
        _gate: &CudaBuffer,
        _weight_dtype: crate::precision::DType,
        _gate_scale: Option<f32>,
        _up: &CudaBuffer,
        _up_scale: Option<f32>,
        _rows: usize,
        _n_dim: usize,
        _k_dim: usize,
    ) -> Result<CudaBuffer, String> {
        Err("CUDA feature is disabled".to_string())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn fused_gate_up_silu_typed_output_buffer(
        _input: &CudaBuffer,
        _input_dtype: crate::precision::DType,
        _input_scale: Option<f32>,
        _gate: &CudaBuffer,
        _weight_dtype: crate::precision::DType,
        _gate_scale: Option<f32>,
        _up: &CudaBuffer,
        _up_scale: Option<f32>,
        _output_dtype: crate::precision::DType,
        _rows: usize,
        _n_dim: usize,
        _k_dim: usize,
    ) -> Result<CudaBuffer, String> {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn fused_qkv_f32(
        _input: &CudaBuffer,
        _q: &CudaBuffer,
        _k: &CudaBuffer,
        _v: &CudaBuffer,
        _rows: usize,
        _q_n: usize,
        _k_n: usize,
        _k_dim: usize,
    ) -> Result<
        (
            (CudaBuffer, Vec<f32>),
            (CudaBuffer, Vec<f32>),
            (CudaBuffer, Vec<f32>),
        ),
        String,
    > {
        Err("CUDA feature is disabled".to_string())
    }

    pub fn fused_qkv_f32_buffer(
        _input: &CudaBuffer,
        _q: &CudaBuffer,
        _k: &CudaBuffer,
        _v: &CudaBuffer,
        _rows: usize,
        _q_n: usize,
        _k_n: usize,
        _k_dim: usize,
    ) -> Result<(CudaBuffer, CudaBuffer, CudaBuffer), String> {
        Err("CUDA feature is disabled".to_string())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn fused_qkv_typed_buffer(
        _input: &CudaBuffer,
        _input_dtype: crate::precision::DType,
        _input_scale: Option<f32>,
        _q: &CudaBuffer,
        _k: &CudaBuffer,
        _v: &CudaBuffer,
        _weight_dtype: crate::precision::DType,
        _q_scale: Option<f32>,
        _k_scale: Option<f32>,
        _v_scale: Option<f32>,
        _rows: usize,
        _q_n: usize,
        _k_n: usize,
        _k_dim: usize,
    ) -> Result<(CudaBuffer, CudaBuffer, CudaBuffer), String> {
        Err("CUDA feature is disabled".to_string())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn fused_qkv_typed_output_buffer(
        _input: &CudaBuffer,
        _input_dtype: crate::precision::DType,
        _input_scale: Option<f32>,
        _q: &CudaBuffer,
        _k: &CudaBuffer,
        _v: &CudaBuffer,
        _weight_dtype: crate::precision::DType,
        _q_scale: Option<f32>,
        _k_scale: Option<f32>,
        _v_scale: Option<f32>,
        _output_dtype: crate::precision::DType,
        _rows: usize,
        _q_n: usize,
        _k_n: usize,
        _k_dim: usize,
    ) -> Result<(CudaBuffer, CudaBuffer, CudaBuffer), String> {
        Err("CUDA feature is disabled".to_string())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn rope_f32(
        _input: &CudaBuffer,
        _cos: &CudaBuffer,
        _sin: &CudaBuffer,
        _batch_size: usize,
        _num_heads: usize,
        _seq_len: usize,
        _dim: usize,
        _offset: usize,
        _cache_seq_len: usize,
    ) -> Result<(CudaBuffer, Vec<f32>), String> {
        Err("CUDA feature is disabled".to_string())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn rope_f32_buffer(
        _input: &CudaBuffer,
        _cos: &CudaBuffer,
        _sin: &CudaBuffer,
        _batch_size: usize,
        _num_heads: usize,
        _seq_len: usize,
        _dim: usize,
        _offset: usize,
        _cache_seq_len: usize,
    ) -> Result<CudaBuffer, String> {
        Err("CUDA feature is disabled".to_string())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn rope_typed_buffer(
        _input: &CudaBuffer,
        _input_dtype: crate::precision::DType,
        _input_scale: Option<f32>,
        _cos: &CudaBuffer,
        _sin: &CudaBuffer,
        _cache_dtype: crate::precision::DType,
        _cos_scale: Option<f32>,
        _sin_scale: Option<f32>,
        _batch_size: usize,
        _num_heads: usize,
        _seq_len: usize,
        _dim: usize,
        _offset: usize,
        _cache_seq_len: usize,
    ) -> Result<CudaBuffer, String> {
        Err("CUDA feature is disabled".to_string())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn rope_typed_i8_dynamic_buffer(
        _input: &CudaBuffer,
        _input_dtype: crate::precision::DType,
        _input_scale: Option<f32>,
        _cos: &CudaBuffer,
        _sin: &CudaBuffer,
        _cache_dtype: crate::precision::DType,
        _cos_scale: Option<f32>,
        _sin_scale: Option<f32>,
        _batch_size: usize,
        _num_heads: usize,
        _seq_len: usize,
        _dim: usize,
        _offset: usize,
        _cache_seq_len: usize,
    ) -> Result<(CudaBuffer, f32), String> {
        Err("CUDA feature is disabled".to_string())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn rope_backward_f32(
        _grad: &CudaBuffer,
        _cos: &CudaBuffer,
        _sin: &CudaBuffer,
        _batch_size: usize,
        _num_heads: usize,
        _seq_len: usize,
        _dim: usize,
        _offset: usize,
        _cache_seq_len: usize,
    ) -> Result<(CudaBuffer, Vec<f32>), String> {
        Err("CUDA feature is disabled".to_string())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn rope_backward_f32_buffer(
        _grad: &CudaBuffer,
        _cos: &CudaBuffer,
        _sin: &CudaBuffer,
        _batch_size: usize,
        _num_heads: usize,
        _seq_len: usize,
        _dim: usize,
        _offset: usize,
        _cache_seq_len: usize,
    ) -> Result<CudaBuffer, String> {
        Err("CUDA feature is disabled".to_string())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn conv2d_f32(
        _input: &CudaBuffer,
        _weight: &CudaBuffer,
        _bias: Option<&CudaBuffer>,
        _batch_size: usize,
        _in_channels: usize,
        _in_h: usize,
        _in_w: usize,
        _out_channels: usize,
        _k_h: usize,
        _k_w: usize,
        _pad_h: usize,
        _pad_w: usize,
        _stride_h: usize,
        _stride_w: usize,
    ) -> Result<(CudaBuffer, Vec<f32>, usize, usize), String> {
        Err("CUDA feature is disabled".to_string())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn conv2d_f32_buffer(
        _input: &CudaBuffer,
        _weight: &CudaBuffer,
        _bias: Option<&CudaBuffer>,
        _batch_size: usize,
        _in_channels: usize,
        _in_h: usize,
        _in_w: usize,
        _out_channels: usize,
        _k_h: usize,
        _k_w: usize,
        _pad_h: usize,
        _pad_w: usize,
        _stride_h: usize,
        _stride_w: usize,
    ) -> Result<(CudaBuffer, usize, usize), String> {
        Err("CUDA feature is disabled".to_string())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn conv2d_typed(
        _input: &CudaBuffer,
        _input_dtype: crate::precision::DType,
        _input_scale: Option<f32>,
        _weight: &CudaBuffer,
        _weight_dtype: crate::precision::DType,
        _weight_scale: Option<f32>,
        _bias: Option<(&CudaBuffer, crate::precision::DType, Option<f32>)>,
        _batch_size: usize,
        _in_channels: usize,
        _in_h: usize,
        _in_w: usize,
        _out_channels: usize,
        _k_h: usize,
        _k_w: usize,
        _pad_h: usize,
        _pad_w: usize,
        _stride_h: usize,
        _stride_w: usize,
    ) -> Result<(CudaBuffer, Vec<f32>, usize, usize), String> {
        Err("CUDA feature is disabled".to_string())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn conv2d_typed_buffer(
        _input: &CudaBuffer,
        _input_dtype: crate::precision::DType,
        _input_scale: Option<f32>,
        _weight: &CudaBuffer,
        _weight_dtype: crate::precision::DType,
        _weight_scale: Option<f32>,
        _bias: Option<(&CudaBuffer, crate::precision::DType, Option<f32>)>,
        _batch_size: usize,
        _in_channels: usize,
        _in_h: usize,
        _in_w: usize,
        _out_channels: usize,
        _k_h: usize,
        _k_w: usize,
        _pad_h: usize,
        _pad_w: usize,
        _stride_h: usize,
        _stride_w: usize,
    ) -> Result<(CudaBuffer, usize, usize), String> {
        Err("CUDA feature is disabled".to_string())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn conv2d_backward_f32(
        _input: &CudaBuffer,
        _weight: &CudaBuffer,
        _grad_output: &CudaBuffer,
        _compute_bias_grad: bool,
        _batch_size: usize,
        _in_channels: usize,
        _in_h: usize,
        _in_w: usize,
        _out_channels: usize,
        _k_h: usize,
        _k_w: usize,
        _pad_h: usize,
        _pad_w: usize,
        _stride_h: usize,
        _stride_w: usize,
    ) -> Result<CudaConv2dBackwardHostBuffers, String> {
        Err("CUDA feature is disabled".to_string())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn conv2d_backward_f32_buffers(
        _input: &CudaBuffer,
        _weight: &CudaBuffer,
        _grad_output: &CudaBuffer,
        _compute_bias_grad: bool,
        _batch_size: usize,
        _in_channels: usize,
        _in_h: usize,
        _in_w: usize,
        _out_channels: usize,
        _k_h: usize,
        _k_w: usize,
        _pad_h: usize,
        _pad_w: usize,
        _stride_h: usize,
        _stride_w: usize,
    ) -> Result<(CudaBuffer, CudaBuffer, Option<CudaBuffer>), String> {
        Err("CUDA feature is disabled".to_string())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn max_pool2d_f32(
        _input: &CudaBuffer,
        _batch_size: usize,
        _channels: usize,
        _in_h: usize,
        _in_w: usize,
        _kernel_h: usize,
        _kernel_w: usize,
        _stride_h: usize,
        _stride_w: usize,
    ) -> Result<(CudaBuffer, Vec<f32>, usize, usize), String> {
        Err("CUDA feature is disabled".to_string())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn max_pool2d_typed(
        _input: &CudaBuffer,
        _input_dtype: crate::precision::DType,
        _input_scale: Option<f32>,
        _batch_size: usize,
        _channels: usize,
        _in_h: usize,
        _in_w: usize,
        _kernel_h: usize,
        _kernel_w: usize,
        _stride_h: usize,
        _stride_w: usize,
    ) -> Result<(CudaBuffer, Vec<f32>, usize, usize), String> {
        Err("CUDA feature is disabled".to_string())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn max_pool2d_backward_f32(
        _input: &CudaBuffer,
        _grad_output: &CudaBuffer,
        _batch_size: usize,
        _channels: usize,
        _in_h: usize,
        _in_w: usize,
        _kernel_h: usize,
        _kernel_w: usize,
        _stride_h: usize,
        _stride_w: usize,
    ) -> Result<(CudaBuffer, Vec<f32>), String> {
        Err("CUDA feature is disabled".to_string())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn max_pool2d_backward_f32_buffer(
        _input: &CudaBuffer,
        _grad_output: &CudaBuffer,
        _batch_size: usize,
        _channels: usize,
        _in_h: usize,
        _in_w: usize,
        _kernel_h: usize,
        _kernel_w: usize,
        _stride_h: usize,
        _stride_w: usize,
    ) -> Result<CudaBuffer, String> {
        Err("CUDA feature is disabled".to_string())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn max_pool2d_backward_typed_buffer(
        _input: &CudaBuffer,
        _input_dtype: crate::precision::DType,
        _input_scale: Option<f32>,
        _grad_output: &CudaBuffer,
        _batch_size: usize,
        _channels: usize,
        _in_h: usize,
        _in_w: usize,
        _kernel_h: usize,
        _kernel_w: usize,
        _stride_h: usize,
        _stride_w: usize,
    ) -> Result<CudaBuffer, String> {
        Err("CUDA feature is disabled".to_string())
    }
}

pub fn is_available() -> bool {
    imp::is_available()
}

pub fn synchronize() -> Result<(), String> {
    imp::synchronize()
}

pub fn alloc_f32(len: usize) -> Result<CudaBuffer, String> {
    imp::alloc_f32(len)
}

pub fn alloc_storage(len: usize) -> Result<CudaBuffer, String> {
    imp::alloc_storage(len)
}

pub fn upload_f32(src: &[f32]) -> Result<CudaBuffer, String> {
    imp::upload_f32(src)
}

pub fn upload_u16_storage(src: &[u16]) -> Result<CudaBuffer, String> {
    imp::upload_u16_storage(src)
}

pub fn upload_i8_storage(src: &[i8]) -> Result<CudaBuffer, String> {
    imp::upload_i8_storage(src)
}

pub fn download_f32(buffer: &CudaBuffer) -> Result<Vec<f32>, String> {
    imp::download_f32(buffer)
}

pub fn download_u16_storage(buffer: &CudaBuffer) -> Result<Vec<u16>, String> {
    imp::download_u16_storage(buffer)
}

pub fn download_i8_storage(buffer: &CudaBuffer) -> Result<Vec<i8>, String> {
    imp::download_i8_storage(buffer)
}

pub fn download_f32_offset(
    buffer: &CudaBuffer,
    offset: usize,
    len: usize,
) -> Result<Vec<f32>, String> {
    imp::download_f32_offset(buffer, offset, len)
}

pub fn matvec_argmax_f32(
    input: &CudaBuffer,
    weight: &CudaBuffer,
    batch_size: usize,
    vocab_size: usize,
    hidden_size: usize,
) -> Result<Vec<usize>, String> {
    imp::matvec_argmax_f32(input, weight, batch_size, vocab_size, hidden_size)
}

pub fn matvec_argmax_bf16_i8(
    input: &CudaBuffer,
    weight: &CudaBuffer,
    weight_scale: f32,
    batch_size: usize,
    vocab_size: usize,
    hidden_size: usize,
) -> Result<Vec<usize>, String> {
    imp::matvec_argmax_bf16_i8(
        input,
        weight,
        weight_scale,
        batch_size,
        vocab_size,
        hidden_size,
    )
}

pub fn matvec_argmax_f16_i8(
    input: &CudaBuffer,
    weight: &CudaBuffer,
    weight_scale: f32,
    batch_size: usize,
    vocab_size: usize,
    hidden_size: usize,
) -> Result<Vec<usize>, String> {
    imp::matvec_argmax_f16_i8(
        input,
        weight,
        weight_scale,
        batch_size,
        vocab_size,
        hidden_size,
    )
}

pub fn matvec_argmax_f32_i8(
    input: &CudaBuffer,
    weight: &CudaBuffer,
    weight_scale: f32,
    batch_size: usize,
    vocab_size: usize,
    hidden_size: usize,
) -> Result<Vec<usize>, String> {
    imp::matvec_argmax_f32_i8(
        input,
        weight,
        weight_scale,
        batch_size,
        vocab_size,
        hidden_size,
    )
}

pub fn matvec_argmax_i8_i8(
    input: &CudaBuffer,
    input_scale: f32,
    weight: &CudaBuffer,
    weight_scale: f32,
    batch_size: usize,
    vocab_size: usize,
    hidden_size: usize,
) -> Result<Vec<usize>, String> {
    imp::matvec_argmax_i8_i8(
        input,
        input_scale,
        weight,
        weight_scale,
        batch_size,
        vocab_size,
        hidden_size,
    )
}

pub fn upload_f32_offset(buffer: &CudaBuffer, offset: usize, src: &[f32]) -> Result<(), String> {
    imp::upload_f32_offset(buffer, offset, src)
}

pub fn copy_f32_offset(
    dst: &CudaBuffer,
    dst_offset: usize,
    src: &CudaBuffer,
    src_offset: usize,
    len: usize,
) -> Result<(), String> {
    imp::copy_f32_offset(dst, dst_offset, src, src_offset, len)
}

#[allow(clippy::too_many_arguments)]
pub fn append_kv_cache_f32(
    dst: &CudaBuffer,
    src: &CudaBuffer,
    batch_size: usize,
    num_heads: usize,
    src_seq_len: usize,
    dst_seq_len: usize,
    dim: usize,
    dst_start: usize,
) -> Result<(), String> {
    imp::append_kv_cache_f32(
        dst,
        src,
        batch_size,
        num_heads,
        src_seq_len,
        dst_seq_len,
        dim,
        dst_start,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn append_kv_cache_pair_f32(
    k_dst: &CudaBuffer,
    v_dst: &CudaBuffer,
    k_src: &CudaBuffer,
    v_src: &CudaBuffer,
    batch_size: usize,
    num_heads: usize,
    src_seq_len: usize,
    dst_seq_len: usize,
    dim: usize,
    dst_start: usize,
) -> Result<(), String> {
    imp::append_kv_cache_pair_f32(
        k_dst,
        v_dst,
        k_src,
        v_src,
        batch_size,
        num_heads,
        src_seq_len,
        dst_seq_len,
        dim,
        dst_start,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn decode_rope_q_append_kv_f32_buffer(
    q_src: &CudaBuffer,
    k_src: &CudaBuffer,
    v_src: &CudaBuffer,
    cos: &CudaBuffer,
    sin: &CudaBuffer,
    k_cache: &CudaBuffer,
    v_cache: &CudaBuffer,
    batch_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    dim: usize,
    dst_seq_len: usize,
    offset: usize,
    cache_seq_len: usize,
) -> Result<CudaBuffer, String> {
    imp::decode_rope_q_append_kv_f32_buffer(
        q_src,
        k_src,
        v_src,
        cos,
        sin,
        k_cache,
        v_cache,
        batch_size,
        num_heads,
        num_kv_heads,
        dim,
        dst_seq_len,
        offset,
        cache_seq_len,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn prefill_attention_f32_buffer(
    q: &CudaBuffer,
    k: &CudaBuffer,
    v: &CudaBuffer,
    batch_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    q_seq_len: usize,
    active_seq_len: usize,
    cache_seq_len: usize,
    dim: usize,
    n_rep: usize,
    past_len: usize,
    scale: f32,
    is_causal: bool,
) -> Result<CudaBuffer, String> {
    imp::prefill_attention_f32_buffer(
        q,
        k,
        v,
        batch_size,
        num_heads,
        num_kv_heads,
        q_seq_len,
        active_seq_len,
        cache_seq_len,
        dim,
        n_rep,
        past_len,
        scale,
        is_causal,
    )
}

pub fn kv_cache_prefix_f32_buffer(
    src: &CudaBuffer,
    batch_size: usize,
    num_heads: usize,
    active_seq_len: usize,
    src_seq_len: usize,
    dim: usize,
) -> Result<CudaBuffer, String> {
    imp::kv_cache_prefix_f32_buffer(src, batch_size, num_heads, active_seq_len, src_seq_len, dim)
}

#[allow(clippy::too_many_arguments)]
pub fn kv_cache_prefix_typed_buffer(
    src: &CudaBuffer,
    dtype: crate::precision::DType,
    batch_size: usize,
    num_heads: usize,
    active_seq_len: usize,
    src_seq_len: usize,
    dim: usize,
) -> Result<CudaBuffer, String> {
    imp::kv_cache_prefix_typed_buffer(
        src,
        dtype,
        batch_size,
        num_heads,
        active_seq_len,
        src_seq_len,
        dim,
    )
}

pub fn matmul_f32(
    a: &CudaBuffer,
    b: &CudaBuffer,
    m: usize,
    n: usize,
    k: usize,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    imp::matmul_f32(a, b, m, n, k)
}

pub fn matmul_f32_no_host(
    a: &CudaBuffer,
    b: &CudaBuffer,
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaBuffer, String> {
    imp::matmul_f32_no_host(a, b, m, n, k)
}

pub fn matmul_bf16_host_no_host(
    a: &[u16],
    b: &[u16],
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaBuffer, String> {
    imp::matmul_bf16_host_no_host(a, b, m, n, k)
}

pub fn matmul_bf16_buffer_no_host(
    a: &CudaBuffer,
    b: &CudaBuffer,
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaBuffer, String> {
    imp::matmul_bf16_buffer_no_host(a, b, m, n, k)
}

pub fn matmul_bf16_typed_output_buffer_no_host(
    a: &CudaBuffer,
    b: &CudaBuffer,
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaBuffer, String> {
    imp::matmul_bf16_typed_output_buffer_no_host(a, b, m, n, k)
}

pub fn matmul_f16_host_no_host(
    a: &[u16],
    b: &[u16],
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaBuffer, String> {
    imp::matmul_f16_host_no_host(a, b, m, n, k)
}

pub fn matmul_f16_buffer_no_host(
    a: &CudaBuffer,
    b: &CudaBuffer,
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaBuffer, String> {
    imp::matmul_f16_buffer_no_host(a, b, m, n, k)
}

pub fn matmul_f16_typed_output_buffer_no_host(
    a: &CudaBuffer,
    b: &CudaBuffer,
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaBuffer, String> {
    imp::matmul_f16_typed_output_buffer_no_host(a, b, m, n, k)
}

#[allow(clippy::too_many_arguments)]
pub fn matmul_i8_host_no_host(
    a: &[i8],
    a_scale: f32,
    b: &[i8],
    b_scale: f32,
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaBuffer, String> {
    imp::matmul_i8_host_no_host(a, a_scale, b, b_scale, m, n, k)
}

#[allow(clippy::too_many_arguments)]
pub fn matmul_i8_buffer_no_host(
    a: &CudaBuffer,
    a_scale: f32,
    b: &CudaBuffer,
    b_scale: f32,
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaBuffer, String> {
    imp::matmul_i8_buffer_no_host(a, a_scale, b, b_scale, m, n, k)
}

pub fn matmul_bf16_i8_buffer_no_host(
    a: &CudaBuffer,
    b: &CudaBuffer,
    b_scale: f32,
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaBuffer, String> {
    imp::matmul_bf16_i8_buffer_no_host(a, b, b_scale, m, n, k)
}

pub fn matmul_f16_i8_buffer_no_host(
    a: &CudaBuffer,
    b: &CudaBuffer,
    b_scale: f32,
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaBuffer, String> {
    imp::matmul_f16_i8_buffer_no_host(a, b, b_scale, m, n, k)
}

pub fn matmul_f32_i8_buffer_no_host(
    a: &CudaBuffer,
    b: &CudaBuffer,
    b_scale: f32,
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaBuffer, String> {
    imp::matmul_f32_i8_buffer_no_host(a, b, b_scale, m, n, k)
}

pub fn matmul_i8_bf16_buffer_no_host(
    a: &CudaBuffer,
    a_scale: f32,
    b: &CudaBuffer,
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaBuffer, String> {
    imp::matmul_i8_bf16_buffer_no_host(a, a_scale, b, m, n, k)
}

pub fn matmul_i8_f16_buffer_no_host(
    a: &CudaBuffer,
    a_scale: f32,
    b: &CudaBuffer,
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaBuffer, String> {
    imp::matmul_i8_f16_buffer_no_host(a, a_scale, b, m, n, k)
}

pub fn matmul_i8_f32_buffer_no_host(
    a: &CudaBuffer,
    a_scale: f32,
    b: &CudaBuffer,
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaBuffer, String> {
    imp::matmul_i8_f32_buffer_no_host(a, a_scale, b, m, n, k)
}

#[allow(clippy::too_many_arguments)]
pub fn matmul_i8_typed_output_buffer_no_host(
    a: &CudaBuffer,
    a_scale: f32,
    b: &CudaBuffer,
    b_scale: f32,
    m: usize,
    n: usize,
    k: usize,
) -> Result<(CudaBuffer, f32), String> {
    imp::matmul_i8_typed_output_buffer_no_host(a, a_scale, b, b_scale, m, n, k)
}

pub fn batch_matmul_f32(
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    batch_count: usize,
    m: usize,
    n: usize,
    k: usize,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    imp::batch_matmul_f32(lhs, rhs, batch_count, m, n, k)
}

pub fn batch_matmul_f32_no_host(
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    batch_count: usize,
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaBuffer, String> {
    imp::batch_matmul_f32_no_host(lhs, rhs, batch_count, m, n, k)
}

pub fn batch_matmul_bf16_buffer_no_host(
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    batch_count: usize,
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaBuffer, String> {
    imp::batch_matmul_bf16_buffer_no_host(lhs, rhs, batch_count, m, n, k)
}

pub fn batch_matmul_bf16_typed_output_buffer_no_host(
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    batch_count: usize,
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaBuffer, String> {
    imp::batch_matmul_bf16_typed_output_buffer_no_host(lhs, rhs, batch_count, m, n, k)
}

pub fn batch_matmul_f16_buffer_no_host(
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    batch_count: usize,
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaBuffer, String> {
    imp::batch_matmul_f16_buffer_no_host(lhs, rhs, batch_count, m, n, k)
}

pub fn batch_matmul_f16_typed_output_buffer_no_host(
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    batch_count: usize,
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaBuffer, String> {
    imp::batch_matmul_f16_typed_output_buffer_no_host(lhs, rhs, batch_count, m, n, k)
}

pub fn batch_matmul_bf16_i8_buffer_no_host(
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    rhs_scale: f32,
    batch_count: usize,
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaBuffer, String> {
    imp::batch_matmul_bf16_i8_buffer_no_host(lhs, rhs, rhs_scale, batch_count, m, n, k)
}

pub fn batch_matmul_f16_i8_buffer_no_host(
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    rhs_scale: f32,
    batch_count: usize,
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaBuffer, String> {
    imp::batch_matmul_f16_i8_buffer_no_host(lhs, rhs, rhs_scale, batch_count, m, n, k)
}

pub fn batch_matmul_f32_i8_buffer_no_host(
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    rhs_scale: f32,
    batch_count: usize,
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaBuffer, String> {
    imp::batch_matmul_f32_i8_buffer_no_host(lhs, rhs, rhs_scale, batch_count, m, n, k)
}

pub fn batch_matmul_i8_bf16_buffer_no_host(
    lhs: &CudaBuffer,
    lhs_scale: f32,
    rhs: &CudaBuffer,
    batch_count: usize,
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaBuffer, String> {
    imp::batch_matmul_i8_bf16_buffer_no_host(lhs, lhs_scale, rhs, batch_count, m, n, k)
}

pub fn batch_matmul_i8_f16_buffer_no_host(
    lhs: &CudaBuffer,
    lhs_scale: f32,
    rhs: &CudaBuffer,
    batch_count: usize,
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaBuffer, String> {
    imp::batch_matmul_i8_f16_buffer_no_host(lhs, lhs_scale, rhs, batch_count, m, n, k)
}

pub fn batch_matmul_i8_f32_buffer_no_host(
    lhs: &CudaBuffer,
    lhs_scale: f32,
    rhs: &CudaBuffer,
    batch_count: usize,
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaBuffer, String> {
    imp::batch_matmul_i8_f32_buffer_no_host(lhs, lhs_scale, rhs, batch_count, m, n, k)
}

#[allow(clippy::too_many_arguments)]
pub fn batch_matmul_i8_buffer_no_host(
    lhs: &CudaBuffer,
    lhs_scale: f32,
    rhs: &CudaBuffer,
    rhs_scale: f32,
    batch_count: usize,
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaBuffer, String> {
    imp::batch_matmul_i8_buffer_no_host(lhs, lhs_scale, rhs, rhs_scale, batch_count, m, n, k)
}

#[allow(clippy::too_many_arguments)]
pub fn batch_matmul_i8_typed_output_buffer_no_host(
    lhs: &CudaBuffer,
    lhs_scale: f32,
    rhs: &CudaBuffer,
    rhs_scale: f32,
    batch_count: usize,
    m: usize,
    n: usize,
    k: usize,
) -> Result<(CudaBuffer, f32), String> {
    imp::batch_matmul_i8_typed_output_buffer_no_host(
        lhs,
        lhs_scale,
        rhs,
        rhs_scale,
        batch_count,
        m,
        n,
        k,
    )
}

pub fn matmul_backward_f32_no_host(
    grad: &CudaBuffer,
    a: &CudaBuffer,
    b: &CudaBuffer,
    m: usize,
    k: usize,
    n: usize,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    imp::matmul_backward_f32_no_host(grad, a, b, m, k, n)
}

pub fn matmul_backward_bf16_i8_no_host(
    grad: &CudaBuffer,
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    rhs_scale: f32,
    m: usize,
    k: usize,
    n: usize,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    imp::matmul_backward_bf16_i8_no_host(grad, lhs, rhs, rhs_scale, m, k, n)
}

pub fn matmul_backward_f16_i8_no_host(
    grad: &CudaBuffer,
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    rhs_scale: f32,
    m: usize,
    k: usize,
    n: usize,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    imp::matmul_backward_f16_i8_no_host(grad, lhs, rhs, rhs_scale, m, k, n)
}

pub fn matmul_backward_f32_i8_no_host(
    grad: &CudaBuffer,
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    rhs_scale: f32,
    m: usize,
    k: usize,
    n: usize,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    imp::matmul_backward_f32_i8_no_host(grad, lhs, rhs, rhs_scale, m, k, n)
}

pub fn matmul_backward_i8_bf16_no_host(
    grad: &CudaBuffer,
    lhs: &CudaBuffer,
    lhs_scale: f32,
    rhs: &CudaBuffer,
    m: usize,
    k: usize,
    n: usize,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    imp::matmul_backward_i8_bf16_no_host(grad, lhs, lhs_scale, rhs, m, k, n)
}

pub fn matmul_backward_i8_f16_no_host(
    grad: &CudaBuffer,
    lhs: &CudaBuffer,
    lhs_scale: f32,
    rhs: &CudaBuffer,
    m: usize,
    k: usize,
    n: usize,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    imp::matmul_backward_i8_f16_no_host(grad, lhs, lhs_scale, rhs, m, k, n)
}

pub fn matmul_backward_i8_f32_no_host(
    grad: &CudaBuffer,
    lhs: &CudaBuffer,
    lhs_scale: f32,
    rhs: &CudaBuffer,
    m: usize,
    k: usize,
    n: usize,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    imp::matmul_backward_i8_f32_no_host(grad, lhs, lhs_scale, rhs, m, k, n)
}

pub fn batch_matmul_backward_f32_no_host(
    grad: &CudaBuffer,
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    batch_count: usize,
    m: usize,
    k: usize,
    n: usize,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    imp::batch_matmul_backward_f32_no_host(grad, lhs, rhs, batch_count, m, k, n)
}

pub fn batch_matmul_backward_bf16_no_host(
    grad: &CudaBuffer,
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    batch_count: usize,
    m: usize,
    k: usize,
    n: usize,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    imp::batch_matmul_backward_bf16_no_host(grad, lhs, rhs, batch_count, m, k, n)
}

pub fn batch_matmul_backward_f16_no_host(
    grad: &CudaBuffer,
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    batch_count: usize,
    m: usize,
    k: usize,
    n: usize,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    imp::batch_matmul_backward_f16_no_host(grad, lhs, rhs, batch_count, m, k, n)
}

pub fn batch_matmul_backward_bf16_i8_no_host(
    grad: &CudaBuffer,
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    rhs_scale: f32,
    batch_count: usize,
    m: usize,
    k: usize,
    n: usize,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    imp::batch_matmul_backward_bf16_i8_no_host(grad, lhs, rhs, rhs_scale, batch_count, m, k, n)
}

pub fn batch_matmul_backward_f16_i8_no_host(
    grad: &CudaBuffer,
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    rhs_scale: f32,
    batch_count: usize,
    m: usize,
    k: usize,
    n: usize,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    imp::batch_matmul_backward_f16_i8_no_host(grad, lhs, rhs, rhs_scale, batch_count, m, k, n)
}

pub fn batch_matmul_backward_f32_i8_no_host(
    grad: &CudaBuffer,
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    rhs_scale: f32,
    batch_count: usize,
    m: usize,
    k: usize,
    n: usize,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    imp::batch_matmul_backward_f32_i8_no_host(grad, lhs, rhs, rhs_scale, batch_count, m, k, n)
}

pub fn batch_matmul_backward_i8_bf16_no_host(
    grad: &CudaBuffer,
    lhs: &CudaBuffer,
    lhs_scale: f32,
    rhs: &CudaBuffer,
    batch_count: usize,
    m: usize,
    k: usize,
    n: usize,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    imp::batch_matmul_backward_i8_bf16_no_host(grad, lhs, lhs_scale, rhs, batch_count, m, k, n)
}

pub fn batch_matmul_backward_i8_f16_no_host(
    grad: &CudaBuffer,
    lhs: &CudaBuffer,
    lhs_scale: f32,
    rhs: &CudaBuffer,
    batch_count: usize,
    m: usize,
    k: usize,
    n: usize,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    imp::batch_matmul_backward_i8_f16_no_host(grad, lhs, lhs_scale, rhs, batch_count, m, k, n)
}

pub fn batch_matmul_backward_i8_f32_no_host(
    grad: &CudaBuffer,
    lhs: &CudaBuffer,
    lhs_scale: f32,
    rhs: &CudaBuffer,
    batch_count: usize,
    m: usize,
    k: usize,
    n: usize,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    imp::batch_matmul_backward_i8_f32_no_host(grad, lhs, lhs_scale, rhs, batch_count, m, k, n)
}

#[allow(clippy::too_many_arguments)]
pub fn batch_matmul_backward_i8_no_host(
    grad: &CudaBuffer,
    lhs: &CudaBuffer,
    lhs_scale: f32,
    rhs: &CudaBuffer,
    rhs_scale: f32,
    batch_count: usize,
    m: usize,
    k: usize,
    n: usize,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    imp::batch_matmul_backward_i8_no_host(
        grad,
        lhs,
        lhs_scale,
        rhs,
        rhs_scale,
        batch_count,
        m,
        k,
        n,
    )
}

pub fn unary_f32(input: &CudaBuffer, op: UnaryOp) -> Result<(CudaBuffer, Vec<f32>), String> {
    imp::unary_f32(input, op)
}

pub fn unary_f32_buffer(input: &CudaBuffer, op: UnaryOp) -> Result<CudaBuffer, String> {
    imp::unary_f32_buffer(input, op)
}

pub fn unary_f16_buffer(input: &CudaBuffer, op: UnaryOp) -> Result<CudaBuffer, String> {
    imp::unary_f16_buffer(input, op)
}

pub fn unary_f16_typed_output_buffer(
    input: &CudaBuffer,
    op: UnaryOp,
) -> Result<CudaBuffer, String> {
    imp::unary_f16_typed_output_buffer(input, op)
}

pub fn unary_bf16_buffer(input: &CudaBuffer, op: UnaryOp) -> Result<CudaBuffer, String> {
    imp::unary_bf16_buffer(input, op)
}

pub fn unary_bf16_typed_output_buffer(
    input: &CudaBuffer,
    op: UnaryOp,
) -> Result<CudaBuffer, String> {
    imp::unary_bf16_typed_output_buffer(input, op)
}

pub fn unary_i8_buffer(input: &CudaBuffer, scale: f32, op: UnaryOp) -> Result<CudaBuffer, String> {
    imp::unary_i8_buffer(input, scale, op)
}

pub fn unary_i8_relu_typed_output_buffer(input: &CudaBuffer) -> Result<CudaBuffer, String> {
    imp::unary_i8_relu_typed_output_buffer(input)
}

pub fn unary_backward_f32(
    input: &CudaBuffer,
    output: &CudaBuffer,
    grad: &CudaBuffer,
    op: UnaryOp,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    imp::unary_backward_f32(input, output, grad, op)
}

pub fn unary_backward_f32_buffer(
    input: &CudaBuffer,
    output: &CudaBuffer,
    grad: &CudaBuffer,
    op: UnaryOp,
) -> Result<CudaBuffer, String> {
    imp::unary_backward_f32_buffer(input, output, grad, op)
}

pub fn unary_backward_f16_buffer(
    input: &CudaBuffer,
    output: &CudaBuffer,
    grad: &CudaBuffer,
    op: UnaryOp,
) -> Result<CudaBuffer, String> {
    imp::unary_backward_f16_buffer(input, output, grad, op)
}

pub fn unary_backward_bf16_buffer(
    input: &CudaBuffer,
    output: &CudaBuffer,
    grad: &CudaBuffer,
    op: UnaryOp,
) -> Result<CudaBuffer, String> {
    imp::unary_backward_bf16_buffer(input, output, grad, op)
}

pub fn unary_backward_i8_buffer(
    input: &CudaBuffer,
    scale: f32,
    output: &CudaBuffer,
    grad: &CudaBuffer,
    op: UnaryOp,
) -> Result<CudaBuffer, String> {
    imp::unary_backward_i8_buffer(input, scale, output, grad, op)
}

pub fn binary_f32(
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    op: BinaryOp,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    imp::binary_f32(lhs, rhs, op)
}

pub fn binary_f32_buffer(
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    op: BinaryOp,
) -> Result<CudaBuffer, String> {
    imp::binary_f32_buffer(lhs, rhs, op)
}

pub fn binary_typed_buffer_no_host(
    lhs: &CudaBuffer,
    lhs_dtype: crate::precision::DType,
    lhs_scale: Option<f32>,
    rhs: &CudaBuffer,
    rhs_dtype: crate::precision::DType,
    rhs_scale: Option<f32>,
    op: BinaryOp,
) -> Result<CudaBuffer, String> {
    imp::binary_typed_buffer_no_host(lhs, lhs_dtype, lhs_scale, rhs, rhs_dtype, rhs_scale, op)
}

pub fn binary_lowp_typed_output_buffer_no_host(
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    dtype: crate::precision::DType,
    op: BinaryOp,
) -> Result<CudaBuffer, String> {
    imp::binary_lowp_typed_output_buffer_no_host(lhs, rhs, dtype, op)
}

#[allow(clippy::too_many_arguments)]
pub fn binary_typed_lastdim_buffer_no_host(
    lhs: &CudaBuffer,
    lhs_dtype: crate::precision::DType,
    lhs_scale: Option<f32>,
    rhs: &CudaBuffer,
    rhs_dtype: crate::precision::DType,
    rhs_scale: Option<f32>,
    out_len: usize,
    last_dim: usize,
    vector_on_rhs: bool,
    op: BinaryOp,
) -> Result<CudaBuffer, String> {
    imp::binary_typed_lastdim_buffer_no_host(
        lhs,
        lhs_dtype,
        lhs_scale,
        rhs,
        rhs_dtype,
        rhs_scale,
        out_len,
        last_dim,
        vector_on_rhs,
        op,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn binary_lowp_typed_lastdim_output_buffer_no_host(
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    out_len: usize,
    last_dim: usize,
    vector_on_rhs: bool,
    dtype: crate::precision::DType,
    op: BinaryOp,
) -> Result<CudaBuffer, String> {
    imp::binary_lowp_typed_lastdim_output_buffer_no_host(
        lhs,
        rhs,
        out_len,
        last_dim,
        vector_on_rhs,
        dtype,
        op,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn binary_typed_row_scalar_buffer_no_host(
    lhs: &CudaBuffer,
    lhs_dtype: crate::precision::DType,
    lhs_scale: Option<f32>,
    rhs: &CudaBuffer,
    rhs_dtype: crate::precision::DType,
    rhs_scale: Option<f32>,
    rows: usize,
    last_dim: usize,
    scalar_on_rhs: bool,
    op: BinaryOp,
) -> Result<CudaBuffer, String> {
    imp::binary_typed_row_scalar_buffer_no_host(
        lhs,
        lhs_dtype,
        lhs_scale,
        rhs,
        rhs_dtype,
        rhs_scale,
        rows,
        last_dim,
        scalar_on_rhs,
        op,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn binary_lowp_typed_row_scalar_output_buffer_no_host(
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    rows: usize,
    last_dim: usize,
    scalar_on_rhs: bool,
    dtype: crate::precision::DType,
    op: BinaryOp,
) -> Result<CudaBuffer, String> {
    imp::binary_lowp_typed_row_scalar_output_buffer_no_host(
        lhs,
        rhs,
        rows,
        last_dim,
        scalar_on_rhs,
        dtype,
        op,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn binary_i8_typed_row_scalar_output_buffer_no_host(
    lhs: &CudaBuffer,
    lhs_scale: f32,
    rhs: &CudaBuffer,
    rhs_scale: f32,
    rows: usize,
    last_dim: usize,
    scalar_on_rhs: bool,
    op: BinaryOp,
) -> Result<(CudaBuffer, f32), String> {
    imp::binary_i8_typed_row_scalar_output_buffer_no_host(
        lhs,
        lhs_scale,
        rhs,
        rhs_scale,
        rows,
        last_dim,
        scalar_on_rhs,
        op,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn binary_typed_broadcast_buffer_no_host(
    lhs: &CudaBuffer,
    lhs_dtype: crate::precision::DType,
    lhs_scale: Option<f32>,
    rhs: &CudaBuffer,
    rhs_dtype: crate::precision::DType,
    rhs_scale: Option<f32>,
    lhs_shape: &[usize],
    rhs_shape: &[usize],
    out_shape: &[usize],
    op: BinaryOp,
) -> Result<CudaBuffer, String> {
    imp::binary_typed_broadcast_buffer_no_host(
        lhs, lhs_dtype, lhs_scale, rhs, rhs_dtype, rhs_scale, lhs_shape, rhs_shape, out_shape, op,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn binary_lowp_typed_broadcast_output_buffer_no_host(
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    lhs_shape: &[usize],
    rhs_shape: &[usize],
    out_shape: &[usize],
    dtype: crate::precision::DType,
    op: BinaryOp,
) -> Result<CudaBuffer, String> {
    imp::binary_lowp_typed_broadcast_output_buffer_no_host(
        lhs, rhs, lhs_shape, rhs_shape, out_shape, dtype, op,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn binary_i8_typed_broadcast_output_buffer_no_host(
    lhs: &CudaBuffer,
    lhs_scale: f32,
    rhs: &CudaBuffer,
    rhs_scale: f32,
    lhs_shape: &[usize],
    rhs_shape: &[usize],
    out_shape: &[usize],
    op: BinaryOp,
) -> Result<(CudaBuffer, f32), String> {
    imp::binary_i8_typed_broadcast_output_buffer_no_host(
        lhs, lhs_scale, rhs, rhs_scale, lhs_shape, rhs_shape, out_shape, op,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn binary_typed_b1d_1h1_buffer_no_host(
    lhs: &CudaBuffer,
    lhs_dtype: crate::precision::DType,
    lhs_scale: Option<f32>,
    rhs: &CudaBuffer,
    rhs_dtype: crate::precision::DType,
    rhs_scale: Option<f32>,
    batch: usize,
    heads: usize,
    dim: usize,
    b1d_on_lhs: bool,
    op: BinaryOp,
) -> Result<CudaBuffer, String> {
    imp::binary_typed_b1d_1h1_buffer_no_host(
        lhs, lhs_dtype, lhs_scale, rhs, rhs_dtype, rhs_scale, batch, heads, dim, b1d_on_lhs, op,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn binary_lowp_typed_b1d_1h1_output_buffer_no_host(
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    batch: usize,
    heads: usize,
    dim: usize,
    b1d_on_lhs: bool,
    dtype: crate::precision::DType,
    op: BinaryOp,
) -> Result<CudaBuffer, String> {
    imp::binary_lowp_typed_b1d_1h1_output_buffer_no_host(
        lhs, rhs, batch, heads, dim, b1d_on_lhs, dtype, op,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn binary_i8_typed_b1d_1h1_output_buffer_no_host(
    lhs: &CudaBuffer,
    lhs_scale: f32,
    rhs: &CudaBuffer,
    rhs_scale: f32,
    batch: usize,
    heads: usize,
    dim: usize,
    b1d_on_lhs: bool,
    op: BinaryOp,
) -> Result<(CudaBuffer, f32), String> {
    imp::binary_i8_typed_b1d_1h1_output_buffer_no_host(
        lhs, lhs_scale, rhs, rhs_scale, batch, heads, dim, b1d_on_lhs, op,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn binary_typed_b1d_1hd_buffer_no_host(
    lhs: &CudaBuffer,
    lhs_dtype: crate::precision::DType,
    lhs_scale: Option<f32>,
    rhs: &CudaBuffer,
    rhs_dtype: crate::precision::DType,
    rhs_scale: Option<f32>,
    batch: usize,
    heads: usize,
    dim: usize,
    b1d_on_lhs: bool,
    op: BinaryOp,
) -> Result<CudaBuffer, String> {
    imp::binary_typed_b1d_1hd_buffer_no_host(
        lhs, lhs_dtype, lhs_scale, rhs, rhs_dtype, rhs_scale, batch, heads, dim, b1d_on_lhs, op,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn binary_lowp_typed_b1d_1hd_output_buffer_no_host(
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    batch: usize,
    heads: usize,
    dim: usize,
    b1d_on_lhs: bool,
    dtype: crate::precision::DType,
    op: BinaryOp,
) -> Result<CudaBuffer, String> {
    imp::binary_lowp_typed_b1d_1hd_output_buffer_no_host(
        lhs, rhs, batch, heads, dim, b1d_on_lhs, dtype, op,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn binary_i8_typed_b1d_1hd_output_buffer_no_host(
    lhs: &CudaBuffer,
    lhs_scale: f32,
    rhs: &CudaBuffer,
    rhs_scale: f32,
    batch: usize,
    heads: usize,
    dim: usize,
    b1d_on_lhs: bool,
    op: BinaryOp,
) -> Result<(CudaBuffer, f32), String> {
    imp::binary_i8_typed_b1d_1hd_output_buffer_no_host(
        lhs, lhs_scale, rhs, rhs_scale, batch, heads, dim, b1d_on_lhs, op,
    )
}

pub fn binary_f16_host_no_host(
    lhs: &[u16],
    rhs: &[u16],
    op: BinaryOp,
) -> Result<CudaBuffer, String> {
    imp::binary_f16_host_no_host(lhs, rhs, op)
}

pub fn binary_f16_buffer_no_host(
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    op: BinaryOp,
) -> Result<CudaBuffer, String> {
    imp::binary_f16_buffer_no_host(lhs, rhs, op)
}

pub fn binary_f16_lastdim_buffer_no_host(
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    out_len: usize,
    last_dim: usize,
    vector_on_rhs: bool,
    op: BinaryOp,
) -> Result<CudaBuffer, String> {
    imp::binary_f16_lastdim_buffer_no_host(lhs, rhs, out_len, last_dim, vector_on_rhs, op)
}

pub fn binary_bf16_host_no_host(
    lhs: &[u16],
    rhs: &[u16],
    op: BinaryOp,
) -> Result<CudaBuffer, String> {
    imp::binary_bf16_host_no_host(lhs, rhs, op)
}

pub fn binary_bf16_buffer_no_host(
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    op: BinaryOp,
) -> Result<CudaBuffer, String> {
    imp::binary_bf16_buffer_no_host(lhs, rhs, op)
}

pub fn binary_bf16_lastdim_buffer_no_host(
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    out_len: usize,
    last_dim: usize,
    vector_on_rhs: bool,
    op: BinaryOp,
) -> Result<CudaBuffer, String> {
    imp::binary_bf16_lastdim_buffer_no_host(lhs, rhs, out_len, last_dim, vector_on_rhs, op)
}

pub fn binary_i8_host_no_host(
    lhs: &[i8],
    lhs_scale: f32,
    rhs: &[i8],
    rhs_scale: f32,
    op: BinaryOp,
) -> Result<CudaBuffer, String> {
    imp::binary_i8_host_no_host(lhs, lhs_scale, rhs, rhs_scale, op)
}

pub fn binary_i8_buffer_no_host(
    lhs: &CudaBuffer,
    lhs_scale: f32,
    rhs: &CudaBuffer,
    rhs_scale: f32,
    op: BinaryOp,
) -> Result<CudaBuffer, String> {
    imp::binary_i8_buffer_no_host(lhs, lhs_scale, rhs, rhs_scale, op)
}

pub fn binary_i8_typed_output_buffer_no_host(
    lhs: &CudaBuffer,
    lhs_scale: f32,
    rhs: &CudaBuffer,
    rhs_scale: f32,
    op: BinaryOp,
) -> Result<(CudaBuffer, f32), String> {
    imp::binary_i8_typed_output_buffer_no_host(lhs, lhs_scale, rhs, rhs_scale, op)
}

pub fn binary_i8_lastdim_buffer_no_host(
    lhs: &CudaBuffer,
    lhs_scale: f32,
    rhs: &CudaBuffer,
    rhs_scale: f32,
    out_len: usize,
    last_dim: usize,
    vector_on_rhs: bool,
    op: BinaryOp,
) -> Result<CudaBuffer, String> {
    imp::binary_i8_lastdim_buffer_no_host(
        lhs,
        lhs_scale,
        rhs,
        rhs_scale,
        out_len,
        last_dim,
        vector_on_rhs,
        op,
    )
}

pub fn binary_i8_typed_lastdim_output_buffer_no_host(
    lhs: &CudaBuffer,
    lhs_scale: f32,
    rhs: &CudaBuffer,
    rhs_scale: f32,
    out_len: usize,
    last_dim: usize,
    vector_on_rhs: bool,
    op: BinaryOp,
) -> Result<(CudaBuffer, f32), String> {
    imp::binary_i8_typed_lastdim_output_buffer_no_host(
        lhs,
        lhs_scale,
        rhs,
        rhs_scale,
        out_len,
        last_dim,
        vector_on_rhs,
        op,
    )
}

pub fn mul_grad_f16_host_no_host(grad: &CudaBuffer, operand: &[u16]) -> Result<CudaBuffer, String> {
    imp::mul_grad_f16_host_no_host(grad, operand)
}

pub fn mul_grad_f16_buffer_no_host(
    grad: &CudaBuffer,
    operand: &CudaBuffer,
) -> Result<CudaBuffer, String> {
    imp::mul_grad_f16_buffer_no_host(grad, operand)
}

pub fn mul_grad_f16_lastdim_buffer_no_host(
    grad: &CudaBuffer,
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    out_len: usize,
    last_dim: usize,
    vector_on_rhs: bool,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    imp::mul_grad_f16_lastdim_buffer_no_host(grad, lhs, rhs, out_len, last_dim, vector_on_rhs)
}

pub fn mul_grad_bf16_host_no_host(
    grad: &CudaBuffer,
    operand: &[u16],
) -> Result<CudaBuffer, String> {
    imp::mul_grad_bf16_host_no_host(grad, operand)
}

pub fn mul_grad_bf16_buffer_no_host(
    grad: &CudaBuffer,
    operand: &CudaBuffer,
) -> Result<CudaBuffer, String> {
    imp::mul_grad_bf16_buffer_no_host(grad, operand)
}

pub fn mul_grad_bf16_lastdim_buffer_no_host(
    grad: &CudaBuffer,
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    out_len: usize,
    last_dim: usize,
    vector_on_rhs: bool,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    imp::mul_grad_bf16_lastdim_buffer_no_host(grad, lhs, rhs, out_len, last_dim, vector_on_rhs)
}

pub fn mul_grad_i8_host_no_host(
    grad: &CudaBuffer,
    operand: &[i8],
    scale: f32,
) -> Result<CudaBuffer, String> {
    imp::mul_grad_i8_host_no_host(grad, operand, scale)
}

pub fn mul_grad_i8_buffer_no_host(
    grad: &CudaBuffer,
    operand: &CudaBuffer,
    scale: f32,
) -> Result<CudaBuffer, String> {
    imp::mul_grad_i8_buffer_no_host(grad, operand, scale)
}

pub fn mul_grad_i8_lastdim_buffer_no_host(
    grad: &CudaBuffer,
    lhs: &CudaBuffer,
    lhs_scale: f32,
    rhs: &CudaBuffer,
    rhs_scale: f32,
    out_len: usize,
    last_dim: usize,
    vector_on_rhs: bool,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    imp::mul_grad_i8_lastdim_buffer_no_host(
        grad,
        lhs,
        lhs_scale,
        rhs,
        rhs_scale,
        out_len,
        last_dim,
        vector_on_rhs,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn mul_grad_typed_lastdim_buffer_no_host(
    grad: &CudaBuffer,
    lhs: &CudaBuffer,
    lhs_dtype: crate::precision::DType,
    lhs_scale: Option<f32>,
    rhs: &CudaBuffer,
    rhs_dtype: crate::precision::DType,
    rhs_scale: Option<f32>,
    out_len: usize,
    last_dim: usize,
    vector_on_rhs: bool,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    imp::mul_grad_typed_lastdim_buffer_no_host(
        grad,
        lhs,
        lhs_dtype,
        lhs_scale,
        rhs,
        rhs_dtype,
        rhs_scale,
        out_len,
        last_dim,
        vector_on_rhs,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn mul_grad_typed_row_scalar_buffer_no_host(
    grad: &CudaBuffer,
    lhs: &CudaBuffer,
    lhs_dtype: crate::precision::DType,
    lhs_scale: Option<f32>,
    rhs: &CudaBuffer,
    rhs_dtype: crate::precision::DType,
    rhs_scale: Option<f32>,
    rows: usize,
    last_dim: usize,
    scalar_on_rhs: bool,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    imp::mul_grad_typed_row_scalar_buffer_no_host(
        grad,
        lhs,
        lhs_dtype,
        lhs_scale,
        rhs,
        rhs_dtype,
        rhs_scale,
        rows,
        last_dim,
        scalar_on_rhs,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn mul_grad_typed_buffer_no_host(
    grad: &CudaBuffer,
    lhs: &CudaBuffer,
    lhs_dtype: crate::precision::DType,
    lhs_scale: Option<f32>,
    rhs: &CudaBuffer,
    rhs_dtype: crate::precision::DType,
    rhs_scale: Option<f32>,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    imp::mul_grad_typed_buffer_no_host(grad, lhs, lhs_dtype, lhs_scale, rhs, rhs_dtype, rhs_scale)
}

#[allow(clippy::too_many_arguments)]
pub fn mul_grad_typed_broadcast_buffer_no_host(
    grad: &CudaBuffer,
    lhs: &CudaBuffer,
    lhs_dtype: crate::precision::DType,
    lhs_scale: Option<f32>,
    rhs: &CudaBuffer,
    rhs_dtype: crate::precision::DType,
    rhs_scale: Option<f32>,
    lhs_shape: &[usize],
    rhs_shape: &[usize],
    out_shape: &[usize],
) -> Result<(CudaBuffer, CudaBuffer), String> {
    imp::mul_grad_typed_broadcast_buffer_no_host(
        grad, lhs, lhs_dtype, lhs_scale, rhs, rhs_dtype, rhs_scale, lhs_shape, rhs_shape, out_shape,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn mul_grad_typed_b1d_1h1_buffer_no_host(
    grad: &CudaBuffer,
    lhs: &CudaBuffer,
    lhs_dtype: crate::precision::DType,
    lhs_scale: Option<f32>,
    rhs: &CudaBuffer,
    rhs_dtype: crate::precision::DType,
    rhs_scale: Option<f32>,
    batch: usize,
    heads: usize,
    dim: usize,
    b1d_on_lhs: bool,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    imp::mul_grad_typed_b1d_1h1_buffer_no_host(
        grad, lhs, lhs_dtype, lhs_scale, rhs, rhs_dtype, rhs_scale, batch, heads, dim, b1d_on_lhs,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn mul_grad_typed_b1d_1hd_buffer_no_host(
    grad: &CudaBuffer,
    lhs: &CudaBuffer,
    lhs_dtype: crate::precision::DType,
    lhs_scale: Option<f32>,
    rhs: &CudaBuffer,
    rhs_dtype: crate::precision::DType,
    rhs_scale: Option<f32>,
    batch: usize,
    heads: usize,
    dim: usize,
    b1d_on_lhs: bool,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    imp::mul_grad_typed_b1d_1hd_buffer_no_host(
        grad, lhs, lhs_dtype, lhs_scale, rhs, rhs_dtype, rhs_scale, batch, heads, dim, b1d_on_lhs,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn mul_grad_typed_scalar_buffer_no_host(
    grad: &CudaBuffer,
    lhs: &CudaBuffer,
    lhs_dtype: crate::precision::DType,
    lhs_scale: Option<f32>,
    rhs: &CudaBuffer,
    rhs_dtype: crate::precision::DType,
    rhs_scale: Option<f32>,
    out_len: usize,
    scalar_on_rhs: bool,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    imp::mul_grad_typed_scalar_buffer_no_host(
        grad,
        lhs,
        lhs_dtype,
        lhs_scale,
        rhs,
        rhs_dtype,
        rhs_scale,
        out_len,
        scalar_on_rhs,
    )
}

pub fn binary_backward_f32(
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    grad: &CudaBuffer,
    op: BinaryOp,
) -> Result<CudaTwoHostBuffers, String> {
    imp::binary_backward_f32(lhs, rhs, grad, op)
}

pub fn binary_backward_f32_buffers(
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    grad: &CudaBuffer,
    op: BinaryOp,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    imp::binary_backward_f32_buffers(lhs, rhs, grad, op)
}

pub fn add_sub_backward_f32_buffers(
    grad: &CudaBuffer,
    len: usize,
    op: BinaryOp,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    imp::add_sub_backward_f32_buffers(grad, len, op)
}

pub fn add_sub_backward_lastdim_f32_buffers(
    grad: &CudaBuffer,
    out_len: usize,
    last_dim: usize,
    vector_on_rhs: bool,
    op: BinaryOp,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    imp::add_sub_backward_lastdim_f32_buffers(grad, out_len, last_dim, vector_on_rhs, op)
}

pub fn add_sub_backward_scalar_f32_buffers(
    grad: &CudaBuffer,
    out_len: usize,
    scalar_on_rhs: bool,
    op: BinaryOp,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    imp::add_sub_backward_scalar_f32_buffers(grad, out_len, scalar_on_rhs, op)
}

pub fn add_sub_backward_row_scalar_f32_buffers(
    grad: &CudaBuffer,
    rows: usize,
    last_dim: usize,
    scalar_on_rhs: bool,
    op: BinaryOp,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    imp::add_sub_backward_row_scalar_f32_buffers(grad, rows, last_dim, scalar_on_rhs, op)
}

pub fn add_sub_backward_b1d_1h1_f32_buffers(
    grad: &CudaBuffer,
    batch: usize,
    heads: usize,
    dim: usize,
    b1d_on_lhs: bool,
    op: BinaryOp,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    imp::add_sub_backward_b1d_1h1_f32_buffers(grad, batch, heads, dim, b1d_on_lhs, op)
}

pub fn add_sub_backward_b1d_1hd_f32_buffers(
    grad: &CudaBuffer,
    batch: usize,
    heads: usize,
    dim: usize,
    b1d_on_lhs: bool,
    op: BinaryOp,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    imp::add_sub_backward_b1d_1hd_f32_buffers(grad, batch, heads, dim, b1d_on_lhs, op)
}

pub fn add_sub_broadcast_backward_f32_buffers(
    grad: &CudaBuffer,
    lhs_shape: &[usize],
    rhs_shape: &[usize],
    out_shape: &[usize],
    op: BinaryOp,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    imp::add_sub_broadcast_backward_f32_buffers(grad, lhs_shape, rhs_shape, out_shape, op)
}

pub fn binary_broadcast_f32(
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    lhs_shape: &[usize],
    rhs_shape: &[usize],
    out_shape: &[usize],
    op: BinaryOp,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    imp::binary_broadcast_f32(lhs, rhs, lhs_shape, rhs_shape, out_shape, op)
}

pub fn binary_broadcast_f32_buffer(
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    lhs_shape: &[usize],
    rhs_shape: &[usize],
    out_shape: &[usize],
    op: BinaryOp,
) -> Result<CudaBuffer, String> {
    imp::binary_broadcast_f32_buffer(lhs, rhs, lhs_shape, rhs_shape, out_shape, op)
}

pub fn binary_broadcast_backward_f32(
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    grad: &CudaBuffer,
    lhs_shape: &[usize],
    rhs_shape: &[usize],
    out_shape: &[usize],
    op: BinaryOp,
) -> Result<CudaTwoHostBuffers, String> {
    imp::binary_broadcast_backward_f32(lhs, rhs, grad, lhs_shape, rhs_shape, out_shape, op)
}

pub fn binary_broadcast_backward_f32_buffers(
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    grad: &CudaBuffer,
    lhs_shape: &[usize],
    rhs_shape: &[usize],
    out_shape: &[usize],
    op: BinaryOp,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    imp::binary_broadcast_backward_f32_buffers(lhs, rhs, grad, lhs_shape, rhs_shape, out_shape, op)
}

pub fn sum_f32(input: &CudaBuffer) -> Result<(CudaBuffer, Vec<f32>), String> {
    imp::sum_f32(input)
}

pub fn sum_f16_buffer(input: &CudaBuffer) -> Result<(CudaBuffer, Vec<f32>), String> {
    imp::sum_f16_buffer(input)
}

pub fn sum_bf16_buffer(input: &CudaBuffer) -> Result<(CudaBuffer, Vec<f32>), String> {
    imp::sum_bf16_buffer(input)
}

pub fn sum_i8_buffer(input: &CudaBuffer, scale: f32) -> Result<(CudaBuffer, Vec<f32>), String> {
    imp::sum_i8_buffer(input, scale)
}

pub fn fill_scalar_f32(len: usize, value: f32) -> Result<(CudaBuffer, Vec<f32>), String> {
    imp::fill_scalar_f32(len, value)
}

pub fn fill_scalar_f32_buffer(len: usize, value: f32) -> Result<CudaBuffer, String> {
    imp::fill_scalar_f32_buffer(len, value)
}

pub fn add_inplace_f32(dst: &CudaBuffer, src: &CudaBuffer) -> Result<(), String> {
    imp::add_inplace_f32(dst, src)
}

pub fn sum_lastdim_f32_buffer(
    input: &CudaBuffer,
    rows: usize,
    last_dim: usize,
) -> Result<CudaBuffer, String> {
    imp::sum_lastdim_f32_buffer(input, rows, last_dim)
}

pub fn bshd_to_bhsd_add_bias_f32_buffer(
    input: &CudaBuffer,
    bias: &CudaBuffer,
    batch: usize,
    seq: usize,
    heads: usize,
    dim: usize,
) -> Result<CudaBuffer, String> {
    imp::bshd_to_bhsd_add_bias_f32_buffer(input, bias, batch, seq, heads, dim)
}

pub fn mse_forward_typed(
    output: &CudaBuffer,
    output_dtype: crate::precision::DType,
    output_scale: Option<f32>,
    target: &CudaBuffer,
    target_dtype: crate::precision::DType,
    target_scale: Option<f32>,
) -> Result<(CudaBuffer, CudaBuffer, Vec<f32>), String> {
    imp::mse_forward_typed(
        output,
        output_dtype,
        output_scale,
        target,
        target_dtype,
        target_scale,
    )
}

pub fn mse_backward_f32(diff: &CudaBuffer, factor: f32) -> Result<CudaTwoHostBuffers, String> {
    imp::mse_backward_f32(diff, factor)
}

pub fn mse_backward_f32_buffers(
    diff: &CudaBuffer,
    factor: f32,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    imp::mse_backward_f32_buffers(diff, factor)
}

pub fn cross_entropy_backward_f32(
    softmax: &CudaBuffer,
    target: &CudaBuffer,
    factor: f32,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    imp::cross_entropy_backward_f32(softmax, target, factor)
}

pub fn cross_entropy_backward_f32_buffer(
    softmax: &CudaBuffer,
    target: &CudaBuffer,
    factor: f32,
) -> Result<CudaBuffer, String> {
    imp::cross_entropy_backward_f32_buffer(softmax, target, factor)
}

pub fn cross_entropy_loss_f32(
    softmax: &CudaBuffer,
    target: &CudaBuffer,
    batch_size: usize,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    imp::cross_entropy_loss_f32(softmax, target, batch_size)
}

pub fn cross_entropy_backward_typed_target_buffer(
    softmax: &CudaBuffer,
    target: &CudaBuffer,
    target_dtype: crate::precision::DType,
    target_scale: Option<f32>,
    factor: f32,
) -> Result<CudaBuffer, String> {
    imp::cross_entropy_backward_typed_target_buffer(
        softmax,
        target,
        target_dtype,
        target_scale,
        factor,
    )
}

pub fn cross_entropy_backward_typed_target(
    softmax: &CudaBuffer,
    target: &CudaBuffer,
    target_dtype: crate::precision::DType,
    target_scale: Option<f32>,
    factor: f32,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    imp::cross_entropy_backward_typed_target(softmax, target, target_dtype, target_scale, factor)
}

pub fn cross_entropy_loss_typed_target(
    softmax: &CudaBuffer,
    target: &CudaBuffer,
    target_dtype: crate::precision::DType,
    target_scale: Option<f32>,
    batch_size: usize,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    imp::cross_entropy_loss_typed_target(softmax, target, target_dtype, target_scale, batch_size)
}

pub fn sgd_update_f32(param: &CudaBuffer, grad: &CudaBuffer, lr: f32) -> Result<Vec<f32>, String> {
    imp::sgd_update_f32(param, grad, lr)
}

pub fn sgd_update_f32_no_host(
    param: &CudaBuffer,
    grad: &CudaBuffer,
    lr: f32,
) -> Result<(), String> {
    imp::sgd_update_f32_no_host(param, grad, lr)
}

pub fn sgd_update_f32_batched_no_host(
    params: &[CudaBuffer],
    grads: &[CudaBuffer],
    lr: f32,
) -> Result<(), String> {
    imp::sgd_update_f32_batched_no_host(params, grads, lr)
}

pub fn quantize_f32_storage_no_host(
    param: &CudaBuffer,
    dtype: crate::precision::DType,
    scale: Option<f32>,
) -> Result<(), String> {
    imp::quantize_f32_storage_no_host(param, dtype, scale)
}

pub fn quantize_f32_to_i8_dynamic_no_host(input: &CudaBuffer) -> Result<(CudaBuffer, f32), String> {
    imp::quantize_f32_to_i8_dynamic_no_host(input)
}

pub fn f32_to_lowp_storage_no_host(
    input: &CudaBuffer,
    dtype: crate::precision::DType,
) -> Result<CudaBuffer, String> {
    imp::f32_to_lowp_storage_no_host(input, dtype)
}

pub fn sgd_momentum_update_f32(
    param: &CudaBuffer,
    grad: &CudaBuffer,
    velocity: &CudaBuffer,
    lr: f32,
    momentum: f32,
) -> Result<(Vec<f32>, Vec<f32>), String> {
    imp::sgd_momentum_update_f32(param, grad, velocity, lr, momentum)
}

pub fn sgd_momentum_update_f32_no_host(
    param: &CudaBuffer,
    grad: &CudaBuffer,
    velocity: &CudaBuffer,
    lr: f32,
    momentum: f32,
) -> Result<(), String> {
    imp::sgd_momentum_update_f32_no_host(param, grad, velocity, lr, momentum)
}

pub fn sgd_momentum_update_f32_batched_no_host(
    params: &[CudaBuffer],
    grads: &[CudaBuffer],
    velocities: &[CudaBuffer],
    lr: f32,
    momentum: f32,
) -> Result<(), String> {
    imp::sgd_momentum_update_f32_batched_no_host(params, grads, velocities, lr, momentum)
}

#[allow(clippy::too_many_arguments)]
pub fn adam_update_f32(
    param: &CudaBuffer,
    grad: &CudaBuffer,
    exp_avg: &CudaBuffer,
    exp_avg_sq: &CudaBuffer,
    lr: f32,
    beta1: f32,
    beta2: f32,
    bias_correction1: f32,
    bias_correction2: f32,
    eps: f32,
) -> Result<CudaAdamHostState, String> {
    imp::adam_update_f32(
        param,
        grad,
        exp_avg,
        exp_avg_sq,
        lr,
        beta1,
        beta2,
        bias_correction1,
        bias_correction2,
        eps,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn adam_update_f32_no_host(
    param: &CudaBuffer,
    grad: &CudaBuffer,
    exp_avg: &CudaBuffer,
    exp_avg_sq: &CudaBuffer,
    lr: f32,
    beta1: f32,
    beta2: f32,
    bias_correction1: f32,
    bias_correction2: f32,
    eps: f32,
) -> Result<(), String> {
    imp::adam_update_f32_no_host(
        param,
        grad,
        exp_avg,
        exp_avg_sq,
        lr,
        beta1,
        beta2,
        bias_correction1,
        bias_correction2,
        eps,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn adam_update_f32_batched_no_host(
    params: &[CudaBuffer],
    grads: &[CudaBuffer],
    exp_avgs: &[CudaBuffer],
    exp_avg_sqs: &[CudaBuffer],
    lr: f32,
    beta1: f32,
    beta2: f32,
    bias_correction1: f32,
    bias_correction2: f32,
    eps: f32,
) -> Result<(), String> {
    imp::adam_update_f32_batched_no_host(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        lr,
        beta1,
        beta2,
        bias_correction1,
        bias_correction2,
        eps,
    )
}

pub fn softmax_lastdim_f32(
    input: &CudaBuffer,
    outer: usize,
    last_dim: usize,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    imp::softmax_lastdim_f32(input, outer, last_dim)
}

pub fn softmax_lastdim_f32_no_host(
    input: &CudaBuffer,
    outer: usize,
    last_dim: usize,
) -> Result<CudaBuffer, String> {
    imp::softmax_lastdim_f32_no_host(input, outer, last_dim)
}

pub fn softmax_lastdim_typed(
    input: &CudaBuffer,
    input_dtype: crate::precision::DType,
    input_scale: Option<f32>,
    outer: usize,
    last_dim: usize,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    imp::softmax_lastdim_typed(input, input_dtype, input_scale, outer, last_dim)
}

pub fn softmax_lastdim_typed_no_host(
    input: &CudaBuffer,
    input_dtype: crate::precision::DType,
    input_scale: Option<f32>,
    outer: usize,
    last_dim: usize,
) -> Result<CudaBuffer, String> {
    imp::softmax_lastdim_typed_no_host(input, input_dtype, input_scale, outer, last_dim)
}

pub fn softmax_lastdim_backward_f32(
    output: &CudaBuffer,
    grad: &CudaBuffer,
    outer: usize,
    last_dim: usize,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    imp::softmax_lastdim_backward_f32(output, grad, outer, last_dim)
}

pub fn softmax_lastdim_backward_f32_buffer(
    output: &CudaBuffer,
    grad: &CudaBuffer,
    outer: usize,
    last_dim: usize,
) -> Result<CudaBuffer, String> {
    imp::softmax_lastdim_backward_f32_buffer(output, grad, outer, last_dim)
}

pub fn fused_softmax_f32(
    input: &CudaBuffer,
    batch_heads: usize,
    q_len: usize,
    k_len: usize,
    scale: f32,
    is_causal: bool,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    imp::fused_softmax_f32(input, batch_heads, q_len, k_len, scale, is_causal)
}

pub fn fused_softmax_f32_no_host(
    input: &CudaBuffer,
    batch_heads: usize,
    q_len: usize,
    k_len: usize,
    scale: f32,
    is_causal: bool,
) -> Result<CudaBuffer, String> {
    imp::fused_softmax_f32_no_host(input, batch_heads, q_len, k_len, scale, is_causal)
}

pub fn fused_softmax_backward_f32(
    output: &CudaBuffer,
    grad: &CudaBuffer,
    batch_heads: usize,
    q_len: usize,
    k_len: usize,
    scale: f32,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    imp::fused_softmax_backward_f32(output, grad, batch_heads, q_len, k_len, scale)
}

pub fn fused_softmax_backward_f32_buffer(
    output: &CudaBuffer,
    grad: &CudaBuffer,
    batch_heads: usize,
    q_len: usize,
    k_len: usize,
    scale: f32,
) -> Result<CudaBuffer, String> {
    imp::fused_softmax_backward_f32_buffer(output, grad, batch_heads, q_len, k_len, scale)
}

pub fn fused_softmax_f32_with_past(
    input: &CudaBuffer,
    batch_heads: usize,
    q_len: usize,
    k_len: usize,
    scale: f32,
    is_causal: bool,
    past_len: usize,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    imp::fused_softmax_f32_with_past(input, batch_heads, q_len, k_len, scale, is_causal, past_len)
}

pub fn fused_softmax_f32_with_past_no_host(
    input: &CudaBuffer,
    batch_heads: usize,
    q_len: usize,
    k_len: usize,
    scale: f32,
    is_causal: bool,
    past_len: usize,
) -> Result<CudaBuffer, String> {
    imp::fused_softmax_f32_with_past_no_host(
        input,
        batch_heads,
        q_len,
        k_len,
        scale,
        is_causal,
        past_len,
    )
}

pub fn embedding_f32(
    indices: &CudaBuffer,
    weight: &CudaBuffer,
    num_indices: usize,
    vocab_size: usize,
    embed_dim: usize,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    imp::embedding_f32(indices, weight, num_indices, vocab_size, embed_dim)
}

pub fn embedding_f32_buffer(
    indices: &CudaBuffer,
    weight: &CudaBuffer,
    num_indices: usize,
    vocab_size: usize,
    embed_dim: usize,
) -> Result<CudaBuffer, String> {
    imp::embedding_f32_buffer(indices, weight, num_indices, vocab_size, embed_dim)
}

pub fn embedding_typed(
    indices: &CudaBuffer,
    weight: &CudaBuffer,
    weight_dtype: crate::precision::DType,
    weight_scale: Option<f32>,
    num_indices: usize,
    vocab_size: usize,
    embed_dim: usize,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    imp::embedding_typed(
        indices,
        weight,
        weight_dtype,
        weight_scale,
        num_indices,
        vocab_size,
        embed_dim,
    )
}

pub fn embedding_typed_buffer(
    indices: &CudaBuffer,
    weight: &CudaBuffer,
    weight_dtype: crate::precision::DType,
    weight_scale: Option<f32>,
    num_indices: usize,
    vocab_size: usize,
    embed_dim: usize,
) -> Result<CudaBuffer, String> {
    imp::embedding_typed_buffer(
        indices,
        weight,
        weight_dtype,
        weight_scale,
        num_indices,
        vocab_size,
        embed_dim,
    )
}

pub fn embedding_typed_same_dtype_buffer(
    indices: &CudaBuffer,
    weight: &CudaBuffer,
    weight_dtype: crate::precision::DType,
    num_indices: usize,
    vocab_size: usize,
    embed_dim: usize,
) -> Result<CudaBuffer, String> {
    imp::embedding_typed_same_dtype_buffer(
        indices,
        weight,
        weight_dtype,
        num_indices,
        vocab_size,
        embed_dim,
    )
}

pub fn embedding_backward_f32(
    indices: &CudaBuffer,
    grad: &CudaBuffer,
    num_indices: usize,
    vocab_size: usize,
    embed_dim: usize,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    imp::embedding_backward_f32(indices, grad, num_indices, vocab_size, embed_dim)
}

pub fn embedding_backward_f32_buffer(
    indices: &CudaBuffer,
    grad: &CudaBuffer,
    num_indices: usize,
    vocab_size: usize,
    embed_dim: usize,
) -> Result<CudaBuffer, String> {
    imp::embedding_backward_f32_buffer(indices, grad, num_indices, vocab_size, embed_dim)
}

pub fn rms_norm_f32(
    input: &CudaBuffer,
    weight: &CudaBuffer,
    rows: usize,
    dim: usize,
    eps: f32,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    imp::rms_norm_f32(input, weight, rows, dim, eps)
}

pub fn rms_norm_f32_buffer(
    input: &CudaBuffer,
    weight: &CudaBuffer,
    rows: usize,
    dim: usize,
    eps: f32,
) -> Result<CudaBuffer, String> {
    imp::rms_norm_f32_buffer(input, weight, rows, dim, eps)
}

#[allow(clippy::too_many_arguments)]
pub fn rms_norm_typed(
    input: &CudaBuffer,
    input_dtype: crate::precision::DType,
    input_scale: Option<f32>,
    weight: &CudaBuffer,
    weight_dtype: crate::precision::DType,
    weight_scale: Option<f32>,
    rows: usize,
    dim: usize,
    eps: f32,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    imp::rms_norm_typed(
        input,
        input_dtype,
        input_scale,
        weight,
        weight_dtype,
        weight_scale,
        rows,
        dim,
        eps,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn rms_norm_typed_buffer(
    input: &CudaBuffer,
    input_dtype: crate::precision::DType,
    input_scale: Option<f32>,
    weight: &CudaBuffer,
    weight_dtype: crate::precision::DType,
    weight_scale: Option<f32>,
    rows: usize,
    dim: usize,
    eps: f32,
) -> Result<CudaBuffer, String> {
    imp::rms_norm_typed_buffer(
        input,
        input_dtype,
        input_scale,
        weight,
        weight_dtype,
        weight_scale,
        rows,
        dim,
        eps,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn rms_norm_i8_typed_output_buffer(
    input: &CudaBuffer,
    input_dtype: crate::precision::DType,
    input_scale: Option<f32>,
    weight: &CudaBuffer,
    weight_dtype: crate::precision::DType,
    weight_scale: Option<f32>,
    rows: usize,
    dim: usize,
    eps: f32,
) -> Result<(CudaBuffer, f32), String> {
    imp::rms_norm_i8_typed_output_buffer(
        input,
        input_dtype,
        input_scale,
        weight,
        weight_dtype,
        weight_scale,
        rows,
        dim,
        eps,
    )
}

pub fn rms_norm_backward_f32(
    input: &CudaBuffer,
    weight: &CudaBuffer,
    grad: &CudaBuffer,
    rows: usize,
    dim: usize,
    eps: f32,
) -> Result<CudaTwoHostBuffers, String> {
    imp::rms_norm_backward_f32(input, weight, grad, rows, dim, eps)
}

pub fn rms_norm_backward_f32_buffers(
    input: &CudaBuffer,
    weight: &CudaBuffer,
    grad: &CudaBuffer,
    rows: usize,
    dim: usize,
    eps: f32,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    imp::rms_norm_backward_f32_buffers(input, weight, grad, rows, dim, eps)
}

#[allow(clippy::too_many_arguments)]
pub fn rms_norm_backward_typed(
    input: &CudaBuffer,
    input_dtype: crate::precision::DType,
    input_scale: Option<f32>,
    weight: &CudaBuffer,
    weight_dtype: crate::precision::DType,
    weight_scale: Option<f32>,
    grad: &CudaBuffer,
    rows: usize,
    dim: usize,
    eps: f32,
) -> Result<CudaTwoHostBuffers, String> {
    imp::rms_norm_backward_typed(
        input,
        input_dtype,
        input_scale,
        weight,
        weight_dtype,
        weight_scale,
        grad,
        rows,
        dim,
        eps,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn rms_norm_backward_typed_buffers(
    input: &CudaBuffer,
    input_dtype: crate::precision::DType,
    input_scale: Option<f32>,
    weight: &CudaBuffer,
    weight_dtype: crate::precision::DType,
    weight_scale: Option<f32>,
    grad: &CudaBuffer,
    rows: usize,
    dim: usize,
    eps: f32,
) -> Result<(CudaBuffer, CudaBuffer), String> {
    imp::rms_norm_backward_typed_buffers(
        input,
        input_dtype,
        input_scale,
        weight,
        weight_dtype,
        weight_scale,
        grad,
        rows,
        dim,
        eps,
    )
}

pub fn permute_f32(
    input: &CudaBuffer,
    out_shape: &[usize],
    axes: &[usize],
) -> Result<(CudaBuffer, Vec<f32>), String> {
    imp::permute_f32(input, out_shape, axes)
}

pub fn permute_f32_buffer(
    input: &CudaBuffer,
    out_shape: &[usize],
    axes: &[usize],
) -> Result<CudaBuffer, String> {
    imp::permute_f32_buffer(input, out_shape, axes)
}

pub fn permute_typed_buffer(
    input: &CudaBuffer,
    dtype: crate::precision::DType,
    out_shape: &[usize],
    axes: &[usize],
) -> Result<CudaBuffer, String> {
    imp::permute_typed_buffer(input, dtype, out_shape, axes)
}

pub fn slice_lastdim_f32(
    input: &CudaBuffer,
    outer: usize,
    input_last_dim: usize,
    start: usize,
    slice_len: usize,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    imp::slice_lastdim_f32(input, outer, input_last_dim, start, slice_len)
}

pub fn slice_lastdim_f32_buffer(
    input: &CudaBuffer,
    outer: usize,
    input_last_dim: usize,
    start: usize,
    slice_len: usize,
) -> Result<CudaBuffer, String> {
    imp::slice_lastdim_f32_buffer(input, outer, input_last_dim, start, slice_len)
}

pub fn slice_lastdim_typed_buffer(
    input: &CudaBuffer,
    dtype: crate::precision::DType,
    outer: usize,
    input_last_dim: usize,
    start: usize,
    slice_len: usize,
) -> Result<CudaBuffer, String> {
    imp::slice_lastdim_typed_buffer(input, dtype, outer, input_last_dim, start, slice_len)
}

pub fn slice_lastdim_backward_f32(
    grad: &CudaBuffer,
    outer: usize,
    input_last_dim: usize,
    start: usize,
    slice_len: usize,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    imp::slice_lastdim_backward_f32(grad, outer, input_last_dim, start, slice_len)
}

pub fn slice_lastdim_backward_f32_buffer(
    grad: &CudaBuffer,
    outer: usize,
    input_last_dim: usize,
    start: usize,
    slice_len: usize,
) -> Result<CudaBuffer, String> {
    imp::slice_lastdim_backward_f32_buffer(grad, outer, input_last_dim, start, slice_len)
}

pub fn cat_f32(
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    out_shape: &[usize],
    axis: usize,
    lhs_axis_len: usize,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    imp::cat_f32(lhs, rhs, out_shape, axis, lhs_axis_len)
}

pub fn cat_f32_buffer(
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    out_shape: &[usize],
    axis: usize,
    lhs_axis_len: usize,
) -> Result<CudaBuffer, String> {
    imp::cat_f32_buffer(lhs, rhs, out_shape, axis, lhs_axis_len)
}

pub fn cat_typed_buffer(
    lhs: &CudaBuffer,
    rhs: &CudaBuffer,
    dtype: crate::precision::DType,
    out_shape: &[usize],
    axis: usize,
    lhs_axis_len: usize,
) -> Result<CudaBuffer, String> {
    imp::cat_typed_buffer(lhs, rhs, dtype, out_shape, axis, lhs_axis_len)
}

pub fn cat_backward_slice_f32(
    grad: &CudaBuffer,
    input_shape: &[usize],
    out_shape: &[usize],
    axis: usize,
    axis_start: usize,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    imp::cat_backward_slice_f32(grad, input_shape, out_shape, axis, axis_start)
}

pub fn cat_backward_slice_f32_buffer(
    grad: &CudaBuffer,
    input_shape: &[usize],
    out_shape: &[usize],
    axis: usize,
    axis_start: usize,
) -> Result<CudaBuffer, String> {
    imp::cat_backward_slice_f32_buffer(grad, input_shape, out_shape, axis, axis_start)
}

pub fn repeat_kv_f32(
    input: &CudaBuffer,
    batch_size: usize,
    num_kv_heads: usize,
    seq_len: usize,
    dim: usize,
    n_rep: usize,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    imp::repeat_kv_f32(input, batch_size, num_kv_heads, seq_len, dim, n_rep)
}

pub fn repeat_kv_f32_buffer(
    input: &CudaBuffer,
    batch_size: usize,
    num_kv_heads: usize,
    seq_len: usize,
    dim: usize,
    n_rep: usize,
) -> Result<CudaBuffer, String> {
    imp::repeat_kv_f32_buffer(input, batch_size, num_kv_heads, seq_len, dim, n_rep)
}

pub fn repeat_kv_typed_buffer(
    input: &CudaBuffer,
    dtype: crate::precision::DType,
    batch_size: usize,
    num_kv_heads: usize,
    seq_len: usize,
    dim: usize,
    n_rep: usize,
) -> Result<CudaBuffer, String> {
    imp::repeat_kv_typed_buffer(input, dtype, batch_size, num_kv_heads, seq_len, dim, n_rep)
}

pub fn repeat_kv_backward_f32(
    grad: &CudaBuffer,
    batch_size: usize,
    num_kv_heads: usize,
    seq_len: usize,
    dim: usize,
    n_rep: usize,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    imp::repeat_kv_backward_f32(grad, batch_size, num_kv_heads, seq_len, dim, n_rep)
}

pub fn repeat_kv_backward_f32_buffer(
    grad: &CudaBuffer,
    batch_size: usize,
    num_kv_heads: usize,
    seq_len: usize,
    dim: usize,
    n_rep: usize,
) -> Result<CudaBuffer, String> {
    imp::repeat_kv_backward_f32_buffer(grad, batch_size, num_kv_heads, seq_len, dim, n_rep)
}

#[allow(clippy::too_many_arguments)]
pub fn decode_attention_f32(
    q: &CudaBuffer,
    k: &CudaBuffer,
    v: &CudaBuffer,
    batch_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    active_seq_len: usize,
    cache_seq_len: usize,
    dim: usize,
    n_rep: usize,
    scale: f32,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    imp::decode_attention_f32(
        q,
        k,
        v,
        batch_size,
        num_heads,
        num_kv_heads,
        active_seq_len,
        cache_seq_len,
        dim,
        n_rep,
        scale,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn decode_attention_f32_buffer(
    q: &CudaBuffer,
    k: &CudaBuffer,
    v: &CudaBuffer,
    batch_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    active_seq_len: usize,
    cache_seq_len: usize,
    dim: usize,
    n_rep: usize,
    scale: f32,
) -> Result<CudaBuffer, String> {
    imp::decode_attention_f32_buffer(
        q,
        k,
        v,
        batch_size,
        num_heads,
        num_kv_heads,
        active_seq_len,
        cache_seq_len,
        dim,
        n_rep,
        scale,
    )
}

pub fn fused_gate_up_silu_f32(
    input: &CudaBuffer,
    gate: &CudaBuffer,
    up: &CudaBuffer,
    rows: usize,
    n_dim: usize,
    k_dim: usize,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    imp::fused_gate_up_silu_f32(input, gate, up, rows, n_dim, k_dim)
}

pub fn fused_gate_up_silu_f32_buffer(
    input: &CudaBuffer,
    gate: &CudaBuffer,
    up: &CudaBuffer,
    rows: usize,
    n_dim: usize,
    k_dim: usize,
) -> Result<CudaBuffer, String> {
    imp::fused_gate_up_silu_f32_buffer(input, gate, up, rows, n_dim, k_dim)
}

pub fn silu_mul_f32_buffer_no_host(
    gate: &CudaBuffer,
    up: &CudaBuffer,
) -> Result<CudaBuffer, String> {
    imp::silu_mul_f32_buffer_no_host(gate, up)
}

#[allow(clippy::too_many_arguments)]
pub fn fused_gate_up_silu_typed(
    input: &CudaBuffer,
    input_dtype: crate::precision::DType,
    input_scale: Option<f32>,
    gate: &CudaBuffer,
    weight_dtype: crate::precision::DType,
    gate_scale: Option<f32>,
    up: &CudaBuffer,
    up_scale: Option<f32>,
    rows: usize,
    n_dim: usize,
    k_dim: usize,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    imp::fused_gate_up_silu_typed(
        input,
        input_dtype,
        input_scale,
        gate,
        weight_dtype,
        gate_scale,
        up,
        up_scale,
        rows,
        n_dim,
        k_dim,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn fused_gate_up_silu_typed_buffer(
    input: &CudaBuffer,
    input_dtype: crate::precision::DType,
    input_scale: Option<f32>,
    gate: &CudaBuffer,
    weight_dtype: crate::precision::DType,
    gate_scale: Option<f32>,
    up: &CudaBuffer,
    up_scale: Option<f32>,
    rows: usize,
    n_dim: usize,
    k_dim: usize,
) -> Result<CudaBuffer, String> {
    imp::fused_gate_up_silu_typed_buffer(
        input,
        input_dtype,
        input_scale,
        gate,
        weight_dtype,
        gate_scale,
        up,
        up_scale,
        rows,
        n_dim,
        k_dim,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn fused_gate_up_silu_typed_output_buffer(
    input: &CudaBuffer,
    input_dtype: crate::precision::DType,
    input_scale: Option<f32>,
    gate: &CudaBuffer,
    weight_dtype: crate::precision::DType,
    gate_scale: Option<f32>,
    up: &CudaBuffer,
    up_scale: Option<f32>,
    output_dtype: crate::precision::DType,
    rows: usize,
    n_dim: usize,
    k_dim: usize,
) -> Result<CudaBuffer, String> {
    imp::fused_gate_up_silu_typed_output_buffer(
        input,
        input_dtype,
        input_scale,
        gate,
        weight_dtype,
        gate_scale,
        up,
        up_scale,
        output_dtype,
        rows,
        n_dim,
        k_dim,
    )
}

pub fn fused_qkv_f32(
    input: &CudaBuffer,
    q: &CudaBuffer,
    k: &CudaBuffer,
    v: &CudaBuffer,
    rows: usize,
    q_n: usize,
    k_n: usize,
    k_dim: usize,
) -> Result<CudaThreeHostBuffers, String> {
    imp::fused_qkv_f32(input, q, k, v, rows, q_n, k_n, k_dim)
}

pub fn fused_qkv_f32_buffer(
    input: &CudaBuffer,
    q: &CudaBuffer,
    k: &CudaBuffer,
    v: &CudaBuffer,
    rows: usize,
    q_n: usize,
    k_n: usize,
    k_dim: usize,
) -> Result<(CudaBuffer, CudaBuffer, CudaBuffer), String> {
    imp::fused_qkv_f32_buffer(input, q, k, v, rows, q_n, k_n, k_dim)
}

#[allow(clippy::too_many_arguments)]
pub fn fused_qkv_typed_buffer(
    input: &CudaBuffer,
    input_dtype: crate::precision::DType,
    input_scale: Option<f32>,
    q: &CudaBuffer,
    k: &CudaBuffer,
    v: &CudaBuffer,
    weight_dtype: crate::precision::DType,
    q_scale: Option<f32>,
    k_scale: Option<f32>,
    v_scale: Option<f32>,
    rows: usize,
    q_n: usize,
    k_n: usize,
    k_dim: usize,
) -> Result<(CudaBuffer, CudaBuffer, CudaBuffer), String> {
    imp::fused_qkv_typed_buffer(
        input,
        input_dtype,
        input_scale,
        q,
        k,
        v,
        weight_dtype,
        q_scale,
        k_scale,
        v_scale,
        rows,
        q_n,
        k_n,
        k_dim,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn fused_qkv_typed_output_buffer(
    input: &CudaBuffer,
    input_dtype: crate::precision::DType,
    input_scale: Option<f32>,
    q: &CudaBuffer,
    k: &CudaBuffer,
    v: &CudaBuffer,
    weight_dtype: crate::precision::DType,
    q_scale: Option<f32>,
    k_scale: Option<f32>,
    v_scale: Option<f32>,
    output_dtype: crate::precision::DType,
    rows: usize,
    q_n: usize,
    k_n: usize,
    k_dim: usize,
) -> Result<(CudaBuffer, CudaBuffer, CudaBuffer), String> {
    imp::fused_qkv_typed_output_buffer(
        input,
        input_dtype,
        input_scale,
        q,
        k,
        v,
        weight_dtype,
        q_scale,
        k_scale,
        v_scale,
        output_dtype,
        rows,
        q_n,
        k_n,
        k_dim,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn rope_f32(
    input: &CudaBuffer,
    cos: &CudaBuffer,
    sin: &CudaBuffer,
    batch_size: usize,
    num_heads: usize,
    seq_len: usize,
    dim: usize,
    offset: usize,
    cache_seq_len: usize,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    imp::rope_f32(
        input,
        cos,
        sin,
        batch_size,
        num_heads,
        seq_len,
        dim,
        offset,
        cache_seq_len,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn rope_f32_buffer(
    input: &CudaBuffer,
    cos: &CudaBuffer,
    sin: &CudaBuffer,
    batch_size: usize,
    num_heads: usize,
    seq_len: usize,
    dim: usize,
    offset: usize,
    cache_seq_len: usize,
) -> Result<CudaBuffer, String> {
    imp::rope_f32_buffer(
        input,
        cos,
        sin,
        batch_size,
        num_heads,
        seq_len,
        dim,
        offset,
        cache_seq_len,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn rope_typed_buffer(
    input: &CudaBuffer,
    input_dtype: crate::precision::DType,
    input_scale: Option<f32>,
    cos: &CudaBuffer,
    sin: &CudaBuffer,
    cache_dtype: crate::precision::DType,
    cos_scale: Option<f32>,
    sin_scale: Option<f32>,
    batch_size: usize,
    num_heads: usize,
    seq_len: usize,
    dim: usize,
    offset: usize,
    cache_seq_len: usize,
) -> Result<CudaBuffer, String> {
    imp::rope_typed_buffer(
        input,
        input_dtype,
        input_scale,
        cos,
        sin,
        cache_dtype,
        cos_scale,
        sin_scale,
        batch_size,
        num_heads,
        seq_len,
        dim,
        offset,
        cache_seq_len,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn rope_typed_i8_dynamic_buffer(
    input: &CudaBuffer,
    input_dtype: crate::precision::DType,
    input_scale: Option<f32>,
    cos: &CudaBuffer,
    sin: &CudaBuffer,
    cache_dtype: crate::precision::DType,
    cos_scale: Option<f32>,
    sin_scale: Option<f32>,
    batch_size: usize,
    num_heads: usize,
    seq_len: usize,
    dim: usize,
    offset: usize,
    cache_seq_len: usize,
) -> Result<(CudaBuffer, f32), String> {
    imp::rope_typed_i8_dynamic_buffer(
        input,
        input_dtype,
        input_scale,
        cos,
        sin,
        cache_dtype,
        cos_scale,
        sin_scale,
        batch_size,
        num_heads,
        seq_len,
        dim,
        offset,
        cache_seq_len,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn rope_backward_f32(
    grad: &CudaBuffer,
    cos: &CudaBuffer,
    sin: &CudaBuffer,
    batch_size: usize,
    num_heads: usize,
    seq_len: usize,
    dim: usize,
    offset: usize,
    cache_seq_len: usize,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    imp::rope_backward_f32(
        grad,
        cos,
        sin,
        batch_size,
        num_heads,
        seq_len,
        dim,
        offset,
        cache_seq_len,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn rope_backward_f32_buffer(
    grad: &CudaBuffer,
    cos: &CudaBuffer,
    sin: &CudaBuffer,
    batch_size: usize,
    num_heads: usize,
    seq_len: usize,
    dim: usize,
    offset: usize,
    cache_seq_len: usize,
) -> Result<CudaBuffer, String> {
    imp::rope_backward_f32_buffer(
        grad,
        cos,
        sin,
        batch_size,
        num_heads,
        seq_len,
        dim,
        offset,
        cache_seq_len,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn conv2d_f32(
    input: &CudaBuffer,
    weight: &CudaBuffer,
    bias: Option<&CudaBuffer>,
    batch_size: usize,
    in_channels: usize,
    in_h: usize,
    in_w: usize,
    out_channels: usize,
    k_h: usize,
    k_w: usize,
    pad_h: usize,
    pad_w: usize,
    stride_h: usize,
    stride_w: usize,
) -> Result<(CudaBuffer, Vec<f32>, usize, usize), String> {
    imp::conv2d_f32(
        input,
        weight,
        bias,
        batch_size,
        in_channels,
        in_h,
        in_w,
        out_channels,
        k_h,
        k_w,
        pad_h,
        pad_w,
        stride_h,
        stride_w,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn conv2d_f32_buffer(
    input: &CudaBuffer,
    weight: &CudaBuffer,
    bias: Option<&CudaBuffer>,
    batch_size: usize,
    in_channels: usize,
    in_h: usize,
    in_w: usize,
    out_channels: usize,
    k_h: usize,
    k_w: usize,
    pad_h: usize,
    pad_w: usize,
    stride_h: usize,
    stride_w: usize,
) -> Result<(CudaBuffer, usize, usize), String> {
    imp::conv2d_f32_buffer(
        input,
        weight,
        bias,
        batch_size,
        in_channels,
        in_h,
        in_w,
        out_channels,
        k_h,
        k_w,
        pad_h,
        pad_w,
        stride_h,
        stride_w,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn conv2d_typed(
    input: &CudaBuffer,
    input_dtype: crate::precision::DType,
    input_scale: Option<f32>,
    weight: &CudaBuffer,
    weight_dtype: crate::precision::DType,
    weight_scale: Option<f32>,
    bias: Option<(&CudaBuffer, crate::precision::DType, Option<f32>)>,
    batch_size: usize,
    in_channels: usize,
    in_h: usize,
    in_w: usize,
    out_channels: usize,
    k_h: usize,
    k_w: usize,
    pad_h: usize,
    pad_w: usize,
    stride_h: usize,
    stride_w: usize,
) -> Result<(CudaBuffer, Vec<f32>, usize, usize), String> {
    imp::conv2d_typed(
        input,
        input_dtype,
        input_scale,
        weight,
        weight_dtype,
        weight_scale,
        bias,
        batch_size,
        in_channels,
        in_h,
        in_w,
        out_channels,
        k_h,
        k_w,
        pad_h,
        pad_w,
        stride_h,
        stride_w,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn conv2d_typed_buffer(
    input: &CudaBuffer,
    input_dtype: crate::precision::DType,
    input_scale: Option<f32>,
    weight: &CudaBuffer,
    weight_dtype: crate::precision::DType,
    weight_scale: Option<f32>,
    bias: Option<(&CudaBuffer, crate::precision::DType, Option<f32>)>,
    batch_size: usize,
    in_channels: usize,
    in_h: usize,
    in_w: usize,
    out_channels: usize,
    k_h: usize,
    k_w: usize,
    pad_h: usize,
    pad_w: usize,
    stride_h: usize,
    stride_w: usize,
) -> Result<(CudaBuffer, usize, usize), String> {
    imp::conv2d_typed_buffer(
        input,
        input_dtype,
        input_scale,
        weight,
        weight_dtype,
        weight_scale,
        bias,
        batch_size,
        in_channels,
        in_h,
        in_w,
        out_channels,
        k_h,
        k_w,
        pad_h,
        pad_w,
        stride_h,
        stride_w,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn conv2d_backward_f32(
    input: &CudaBuffer,
    weight: &CudaBuffer,
    grad_output: &CudaBuffer,
    compute_bias_grad: bool,
    batch_size: usize,
    in_channels: usize,
    in_h: usize,
    in_w: usize,
    out_channels: usize,
    k_h: usize,
    k_w: usize,
    pad_h: usize,
    pad_w: usize,
    stride_h: usize,
    stride_w: usize,
) -> Result<CudaConv2dBackwardHostBuffers, String> {
    imp::conv2d_backward_f32(
        input,
        weight,
        grad_output,
        compute_bias_grad,
        batch_size,
        in_channels,
        in_h,
        in_w,
        out_channels,
        k_h,
        k_w,
        pad_h,
        pad_w,
        stride_h,
        stride_w,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn conv2d_backward_f32_buffers(
    input: &CudaBuffer,
    weight: &CudaBuffer,
    grad_output: &CudaBuffer,
    compute_bias_grad: bool,
    batch_size: usize,
    in_channels: usize,
    in_h: usize,
    in_w: usize,
    out_channels: usize,
    k_h: usize,
    k_w: usize,
    pad_h: usize,
    pad_w: usize,
    stride_h: usize,
    stride_w: usize,
) -> Result<(CudaBuffer, CudaBuffer, Option<CudaBuffer>), String> {
    imp::conv2d_backward_f32_buffers(
        input,
        weight,
        grad_output,
        compute_bias_grad,
        batch_size,
        in_channels,
        in_h,
        in_w,
        out_channels,
        k_h,
        k_w,
        pad_h,
        pad_w,
        stride_h,
        stride_w,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn max_pool2d_f32(
    input: &CudaBuffer,
    batch_size: usize,
    channels: usize,
    in_h: usize,
    in_w: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
) -> Result<(CudaBuffer, Vec<f32>, usize, usize), String> {
    imp::max_pool2d_f32(
        input, batch_size, channels, in_h, in_w, kernel_h, kernel_w, stride_h, stride_w,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn max_pool2d_typed(
    input: &CudaBuffer,
    input_dtype: crate::precision::DType,
    input_scale: Option<f32>,
    batch_size: usize,
    channels: usize,
    in_h: usize,
    in_w: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
) -> Result<(CudaBuffer, Vec<f32>, usize, usize), String> {
    imp::max_pool2d_typed(
        input,
        input_dtype,
        input_scale,
        batch_size,
        channels,
        in_h,
        in_w,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn max_pool2d_backward_f32(
    input: &CudaBuffer,
    grad_output: &CudaBuffer,
    batch_size: usize,
    channels: usize,
    in_h: usize,
    in_w: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
) -> Result<(CudaBuffer, Vec<f32>), String> {
    imp::max_pool2d_backward_f32(
        input,
        grad_output,
        batch_size,
        channels,
        in_h,
        in_w,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn max_pool2d_backward_typed_buffer(
    input: &CudaBuffer,
    input_dtype: crate::precision::DType,
    input_scale: Option<f32>,
    grad_output: &CudaBuffer,
    batch_size: usize,
    channels: usize,
    in_h: usize,
    in_w: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
) -> Result<CudaBuffer, String> {
    imp::max_pool2d_backward_typed_buffer(
        input,
        input_dtype,
        input_scale,
        grad_output,
        batch_size,
        channels,
        in_h,
        in_w,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn max_pool2d_backward_f32_buffer(
    input: &CudaBuffer,
    grad_output: &CudaBuffer,
    batch_size: usize,
    channels: usize,
    in_h: usize,
    in_w: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
) -> Result<CudaBuffer, String> {
    imp::max_pool2d_backward_f32_buffer(
        input,
        grad_output,
        batch_size,
        channels,
        in_h,
        in_w,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
    )
}
