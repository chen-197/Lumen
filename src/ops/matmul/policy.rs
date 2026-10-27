use super::*;

#[inline]
pub(super) fn should_use_mixed_matvec_block_kernel(n_rows: usize) -> bool {
    #[cfg(all(feature = "arm64-fp-kernels", target_arch = "aarch64"))]
    {
        let _ = n_rows;
        false
    }

    #[cfg(not(all(feature = "arm64-fp-kernels", target_arch = "aarch64")))]
    {
        let _ = n_rows;
        false
    }
}

pub(super) fn should_use_mixed_dual_block_kernel(n_rows: usize) -> bool {
    #[cfg(all(feature = "arm64-fp-kernels", target_arch = "aarch64"))]
    {
        let _ = n_rows;
        false
    }

    #[cfg(not(all(feature = "arm64-fp-kernels", target_arch = "aarch64")))]
    {
        let _ = n_rows;
        false
    }
}

#[inline]
pub(super) fn should_use_argmax_block_kernel(n_rows: usize) -> bool {
    #[cfg(all(feature = "arm64-fp-kernels", target_arch = "aarch64"))]
    {
        let _ = n_rows;
        false
    }

    #[cfg(not(all(feature = "arm64-fp-kernels", target_arch = "aarch64")))]
    {
        n_rows >= MATVEC_BLOCK_THRESHOLD
    }
}

#[inline]
pub(super) fn should_use_i8_block_kernel(n_rows: usize, k_dim: usize) -> bool {
    #[cfg(all(feature = "arm64-int8-kernels", target_arch = "aarch64"))]
    {
        let _ = k_dim;
        n_rows >= MATVEC_BLOCK_THRESHOLD
    }

    #[cfg(not(all(feature = "arm64-int8-kernels", target_arch = "aarch64")))]
    {
        let _ = (n_rows, k_dim);
        false
    }
}

#[inline]
pub(super) fn should_use_i8_silu_block_kernel(n_rows: usize, k_dim: usize) -> bool {
    #[cfg(all(feature = "arm64-int8-kernels", target_arch = "aarch64"))]
    {
        let _ = (n_rows, k_dim);
        false
    }

    #[cfg(not(all(feature = "arm64-int8-kernels", target_arch = "aarch64")))]
    {
        let _ = (n_rows, k_dim);
        false
    }
}

#[inline]
pub(super) fn should_use_i8_matmul_block_kernel(n_rows: usize, k_dim: usize) -> bool {
    #[cfg(all(feature = "arm64-int8-kernels", target_arch = "aarch64"))]
    {
        let _ = (n_rows, k_dim);
        false
    }

    #[cfg(not(all(feature = "arm64-int8-kernels", target_arch = "aarch64")))]
    {
        let _ = (n_rows, k_dim);
        false
    }
}

#[inline]
pub(super) fn should_use_i8_qkv_row4_kernel() -> bool {
    #[cfg(all(feature = "arm64-int8-kernels", target_arch = "aarch64"))]
    {
        false
    }

    #[cfg(not(all(feature = "arm64-int8-kernels", target_arch = "aarch64")))]
    {
        false
    }
}
