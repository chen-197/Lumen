#[cfg(all(feature = "arm64-int8-kernels", not(target_arch = "aarch64")))]
compile_error!("feature `arm64-int8-kernels` only supports the `aarch64` target architecture");

#[cfg(all(feature = "arm64-fp-kernels", not(target_arch = "aarch64")))]
compile_error!("feature `arm64-fp-kernels` only supports the `aarch64` target architecture");

#[cfg(all(
    feature = "x86-fp-kernels",
    not(any(target_arch = "x86_64", target_arch = "x86"))
))]
compile_error!("feature `x86-fp-kernels` only supports the `x86_64` or `x86` target architectures");

#[cfg(all(
    feature = "x86-int8-kernels",
    not(any(target_arch = "x86_64", target_arch = "x86"))
))]
compile_error!(
    "feature `x86-int8-kernels` only supports the `x86_64` or `x86` target architectures"
);

/// Initialize CPU CLI workers before any parallel work starts. Embedders keep
/// control of Rayon unless they explicitly call this function.
pub fn init_cpu_inference_threads(requested: Option<usize>) -> Result<(), String> {
    if requested == Some(0) {
        return Err("CPU worker count must be greater than zero".to_string());
    }
    if requested.is_none() {
        return Ok(());
    }
    let threads = requested.unwrap();
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .map_err(|err| format!("Failed to initialize CPU inference workers: {err}"))
}

#[inline]
pub const fn arm64_int8_kernels_compiled() -> bool {
    cfg!(all(feature = "arm64-int8-kernels", target_arch = "aarch64"))
}

#[inline]
pub const fn arm64_fp_kernels_compiled() -> bool {
    cfg!(all(feature = "arm64-fp-kernels", target_arch = "aarch64"))
}

#[inline]
pub const fn x86_int8_kernels_compiled() -> bool {
    cfg!(all(
        feature = "x86-int8-kernels",
        any(target_arch = "x86_64", target_arch = "x86")
    ))
}

#[inline]
pub const fn x86_fp_kernels_compiled() -> bool {
    cfg!(all(
        feature = "x86-fp-kernels",
        any(target_arch = "x86_64", target_arch = "x86")
    ))
}

#[inline]
pub const fn any_arch_int8_kernels_compiled() -> bool {
    arm64_int8_kernels_compiled() || x86_int8_kernels_compiled()
}

#[inline]
pub fn arm64_i8_kernel_runtime_available() -> bool {
    arm64_int8_kernels_compiled()
}

#[inline]
pub fn arm64_fp_kernel_runtime_available() -> bool {
    arm64_fp_kernels_compiled()
}

#[inline]
pub fn arm64_fp16_kernel_runtime_available() -> bool {
    #[cfg(all(feature = "arm64-fp-kernels", target_arch = "aarch64"))]
    {
        std::arch::is_aarch64_feature_detected!("fp16")
    }
    #[cfg(not(all(feature = "arm64-fp-kernels", target_arch = "aarch64")))]
    {
        false
    }
}

#[inline]
pub fn x86_avx512_fp_kernel_runtime_available() -> bool {
    #[cfg(all(
        feature = "x86-fp-kernels",
        any(target_arch = "x86_64", target_arch = "x86")
    ))]
    {
        std::arch::is_x86_feature_detected!("avx512f")
    }
    #[cfg(not(all(
        feature = "x86-fp-kernels",
        any(target_arch = "x86_64", target_arch = "x86")
    )))]
    {
        false
    }
}

#[inline]
pub fn x86_avx512_bf16_kernel_runtime_available() -> bool {
    #[cfg(all(
        feature = "x86-fp-kernels",
        any(target_arch = "x86_64", target_arch = "x86")
    ))]
    {
        x86_avx512_fp_kernel_runtime_available()
            && std::arch::is_x86_feature_detected!("avx512vl")
            && std::arch::is_x86_feature_detected!("avx512bf16")
    }
    #[cfg(not(all(
        feature = "x86-fp-kernels",
        any(target_arch = "x86_64", target_arch = "x86")
    )))]
    {
        false
    }
}

#[inline]
pub fn x86_avx512_fp16_kernel_runtime_available() -> bool {
    #[cfg(all(
        feature = "x86-fp-kernels-nightly",
        any(target_arch = "x86_64", target_arch = "x86")
    ))]
    {
        x86_avx512_fp_kernel_runtime_available()
            && std::arch::is_x86_feature_detected!("avx512fp16")
    }
    #[cfg(not(all(
        feature = "x86-fp-kernels-nightly",
        any(target_arch = "x86_64", target_arch = "x86")
    )))]
    {
        false
    }
}

#[inline]
pub fn x86_fp_kernel_runtime_available() -> bool {
    #[cfg(all(
        feature = "x86-fp-kernels",
        any(target_arch = "x86_64", target_arch = "x86")
    ))]
    {
        std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
    }
    #[cfg(not(all(
        feature = "x86-fp-kernels",
        any(target_arch = "x86_64", target_arch = "x86")
    )))]
    {
        false
    }
}

#[inline]
pub fn x86_f16c_kernel_runtime_available() -> bool {
    #[cfg(all(
        feature = "x86-fp-kernels",
        any(target_arch = "x86_64", target_arch = "x86")
    ))]
    {
        x86_fp_kernel_runtime_available() && std::arch::is_x86_feature_detected!("f16c")
    }
    #[cfg(not(all(
        feature = "x86-fp-kernels",
        any(target_arch = "x86_64", target_arch = "x86")
    )))]
    {
        false
    }
}

#[inline]
pub fn x86_fp16_kernel_runtime_available() -> bool {
    #[cfg(all(
        feature = "x86-fp-kernels",
        any(target_arch = "x86_64", target_arch = "x86")
    ))]
    {
        x86_avx512_fp16_kernel_runtime_available() || x86_f16c_kernel_runtime_available()
    }
    #[cfg(not(all(
        feature = "x86-fp-kernels",
        any(target_arch = "x86_64", target_arch = "x86")
    )))]
    {
        false
    }
}

#[inline]
pub fn x86_i8_kernel_runtime_available() -> bool {
    #[cfg(all(
        feature = "x86-int8-kernels",
        any(target_arch = "x86_64", target_arch = "x86")
    ))]
    {
        std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
    }
    #[cfg(not(all(
        feature = "x86-int8-kernels",
        any(target_arch = "x86_64", target_arch = "x86")
    )))]
    {
        false
    }
}

#[inline]
pub fn x86_avx512_i8_i8_kernel_runtime_available() -> bool {
    #[cfg(all(
        feature = "x86-int8-kernels",
        any(target_arch = "x86_64", target_arch = "x86")
    ))]
    {
        std::arch::is_x86_feature_detected!("avx")
            && std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw")
    }
    #[cfg(not(all(
        feature = "x86-int8-kernels",
        any(target_arch = "x86_64", target_arch = "x86")
    )))]
    {
        false
    }
}

#[inline]
pub fn x86_avx512_i8_kernel_runtime_available() -> bool {
    #[cfg(all(
        feature = "x86-int8-kernels",
        any(target_arch = "x86_64", target_arch = "x86")
    ))]
    {
        x86_avx512_i8_i8_kernel_runtime_available() && std::arch::is_x86_feature_detected!("fma")
    }
    #[cfg(not(all(
        feature = "x86-int8-kernels",
        any(target_arch = "x86_64", target_arch = "x86")
    )))]
    {
        false
    }
}

#[inline]
pub fn preferred_i8_kernel_backend() -> &'static str {
    if arm64_i8_kernel_runtime_available() {
        "arm64-neon"
    } else if x86_avx512_i8_kernel_runtime_available() {
        "x86-avx512bw"
    } else if x86_i8_kernel_runtime_available() {
        "x86-avx2"
    } else {
        "portable"
    }
}

#[inline]
pub fn preferred_fp_kernel_backend() -> &'static str {
    if arm64_fp_kernel_runtime_available() {
        "arm64-neon"
    } else if x86_avx512_fp_kernel_runtime_available() {
        "x86-avx512"
    } else if x86_fp_kernel_runtime_available() {
        "x86-avx2"
    } else {
        "portable"
    }
}

#[cfg(test)]
#[path = "../tests/unit/arch/tests.rs"]
mod tests;
