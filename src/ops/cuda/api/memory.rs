use super::*;

pub fn is_available() -> bool {
    imp::is_available()
}

pub fn synchronize() -> Result<(), String> {
    imp::synchronize()
}

/// Releases CUDA caches owned by the calling thread and all currently idle
/// buffers in the shared allocation pool.
///
/// Call this only at an application-controlled idle boundary. It does not
/// release thread-local caches owned by other live threads, and concurrent CUDA
/// work may repopulate the shared pool before this function returns.
pub fn release_cached_memory() -> Result<(), String> {
    imp::release_cached_memory()
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

#[cfg(test)]
pub(crate) fn test_host_downloads() -> usize {
    TEST_HOST_DOWNLOADS.with(Cell::get)
}

pub fn download_f32(buffer: &CudaBuffer) -> Result<Vec<f32>, String> {
    #[cfg(test)]
    TEST_HOST_DOWNLOADS.with(|count| count.set(count.get() + 1));
    imp::download_f32(buffer)
}

pub fn download_u16_storage(buffer: &CudaBuffer) -> Result<Vec<u16>, String> {
    #[cfg(test)]
    TEST_HOST_DOWNLOADS.with(|count| count.set(count.get() + 1));
    imp::download_u16_storage(buffer)
}

pub fn download_i8_storage(buffer: &CudaBuffer) -> Result<Vec<i8>, String> {
    #[cfg(test)]
    TEST_HOST_DOWNLOADS.with(|count| count.set(count.get() + 1));
    imp::download_i8_storage(buffer)
}

pub fn download_f32_offset(
    buffer: &CudaBuffer,
    offset: usize,
    len: usize,
) -> Result<Vec<f32>, String> {
    #[cfg(test)]
    TEST_HOST_DOWNLOADS.with(|count| count.set(count.get() + 1));
    imp::download_f32_offset(buffer, offset, len)
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

pub fn fill_scalar_f32(len: usize, value: f32) -> Result<(CudaBuffer, Vec<f32>), String> {
    imp::fill_scalar_f32(len, value)
}

pub fn fill_scalar_f32_buffer(len: usize, value: f32) -> Result<CudaBuffer, String> {
    imp::fill_scalar_f32_buffer(len, value)
}

pub fn add_inplace_f32(dst: &CudaBuffer, src: &CudaBuffer) -> Result<(), String> {
    imp::add_inplace_f32(dst, src)
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

pub fn f32_to_i8_storage_no_host(input: &CudaBuffer, scale: f32) -> Result<CudaBuffer, String> {
    imp::f32_to_i8_storage_no_host(input, scale)
}
