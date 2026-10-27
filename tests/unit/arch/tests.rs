use super::*;

#[test]
fn architecture_specific_i8_kernels_are_disabled_by_default() {
    assert_eq!(
        any_arch_int8_kernels_compiled(),
        arm64_int8_kernels_compiled() || x86_int8_kernels_compiled()
    );
}

#[test]
fn preferred_backend_is_consistent_with_runtime_probes() {
    let backend = preferred_i8_kernel_backend();
    if arm64_i8_kernel_runtime_available() {
        assert_eq!(backend, "arm64-neon");
    } else if x86_avx512_i8_kernel_runtime_available() {
        assert_eq!(backend, "x86-avx512bw");
    } else if x86_i8_kernel_runtime_available() {
        assert_eq!(backend, "x86-avx2");
    } else {
        assert_eq!(backend, "portable");
    }
}

#[test]
fn preferred_fp_backend_is_consistent_with_runtime_probes() {
    let backend = preferred_fp_kernel_backend();
    if arm64_fp_kernel_runtime_available() {
        assert_eq!(backend, "arm64-neon");
    } else if x86_avx512_fp_kernel_runtime_available() {
        assert_eq!(backend, "x86-avx512");
    } else if x86_fp_kernel_runtime_available() {
        assert_eq!(backend, "x86-avx2");
    } else {
        assert_eq!(backend, "portable");
    }
}

#[test]
fn arm64_fp16_probe_is_never_broader_than_fp_backend_probe() {
    assert!(!arm64_fp16_kernel_runtime_available() || arm64_fp_kernel_runtime_available());
}

#[test]
fn x86_fp16_probe_is_never_broader_than_fp_backend_probe() {
    assert!(!x86_fp16_kernel_runtime_available() || x86_fp_kernel_runtime_available());
}

#[test]
fn x86_f16c_probe_is_never_broader_than_fp_backend_probe() {
    assert!(!x86_f16c_kernel_runtime_available() || x86_fp_kernel_runtime_available());
}
