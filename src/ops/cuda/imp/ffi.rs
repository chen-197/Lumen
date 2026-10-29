use super::*;

unsafe extern "C" {
    pub(super) fn lumen_cuda_round_kv_cache_rows_f32_device(
        handle: u64,
        batch_size: usize,
        num_heads: usize,
        rows: usize,
        capacity: usize,
        dim: usize,
        start: usize,
        dtype: i32,
    ) -> i32;
    pub(super) fn lumen_cuda_is_available() -> c_int;
    pub(super) fn lumen_cuda_alloc_f32(len: usize, out_handle: *mut u64) -> c_int;
    pub(super) fn lumen_cuda_upload_f32(handle: u64, src: *const f32, len: usize) -> c_int;
    pub(super) fn lumen_cuda_upload_u16(handle: u64, src: *const u16, len: usize) -> c_int;
    pub(super) fn lumen_cuda_upload_i8(handle: u64, src: *const i8, len: usize) -> c_int;
    pub(super) fn lumen_cuda_upload_f32_offset(
        handle: u64,
        src: *const f32,
        offset: usize,
        len: usize,
    ) -> c_int;
    pub(super) fn lumen_cuda_copy_f32_offset(
        dst_handle: u64,
        dst_offset: usize,
        src_handle: u64,
        src_offset: usize,
        len: usize,
    ) -> c_int;
    pub(super) fn lumen_cuda_append_kv_cache_f32_device(
        dst_handle: u64,
        src_handle: u64,
        batch_size: usize,
        num_heads: usize,
        src_seq_len: usize,
        dst_seq_len: usize,
        dim: usize,
        dst_start: usize,
    ) -> c_int;
    pub(super) fn lumen_cuda_append_kv_cache_pair_f32_device(
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
    pub(super) fn lumen_cuda_decode_rope_q_append_kv_f32_device(
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
    pub(super) fn lumen_cuda_kv_cache_prefix_f32_device(
        src_handle: u64,
        out_handle: u64,
        batch_size: usize,
        num_heads: usize,
        active_seq_len: usize,
        src_seq_len: usize,
        dim: usize,
    ) -> c_int;
    pub(super) fn lumen_cuda_kv_cache_prefix_typed_device(
        src_handle: u64,
        dtype: c_int,
        out_handle: u64,
        batch_size: usize,
        num_heads: usize,
        active_seq_len: usize,
        src_seq_len: usize,
        dim: usize,
    ) -> c_int;
    pub(super) fn lumen_cuda_download_f32(handle: u64, dst: *mut f32, len: usize) -> c_int;
    pub(super) fn lumen_cuda_download_u16(handle: u64, dst: *mut u16, len: usize) -> c_int;
    pub(super) fn lumen_cuda_download_i8(handle: u64, dst: *mut i8, len: usize) -> c_int;
    pub(super) fn lumen_cuda_download_f32_offset(
        handle: u64,
        dst: *mut f32,
        offset: usize,
        len: usize,
    ) -> c_int;
    pub(super) fn lumen_cuda_free_f32(handle: u64, len: usize);
    pub(super) fn lumen_cuda_synchronize() -> c_int;
    pub(super) fn lumen_cuda_release_cached_memory() -> c_int;
    pub(super) fn lumen_cuda_matvec_argmax_f32_device(
        input_handle: u64,
        weight_handle: u64,
        out_indices: *mut usize,
        batch_size: usize,
        vocab_size: usize,
        hidden_size: usize,
    ) -> c_int;
    pub(super) fn lumen_cuda_matvec_argmax_bf16_i8_device(
        input_handle: u64,
        weight_handle: u64,
        weight_scale: f32,
        out_indices: *mut usize,
        batch_size: usize,
        vocab_size: usize,
        hidden_size: usize,
    ) -> c_int;
    pub(super) fn lumen_cuda_matvec_argmax_f16_i8_device(
        input_handle: u64,
        weight_handle: u64,
        weight_scale: f32,
        out_indices: *mut usize,
        batch_size: usize,
        vocab_size: usize,
        hidden_size: usize,
    ) -> c_int;
    pub(super) fn lumen_cuda_matvec_argmax_f32_i8_device(
        input_handle: u64,
        weight_handle: u64,
        weight_scale: f32,
        out_indices: *mut usize,
        batch_size: usize,
        vocab_size: usize,
        hidden_size: usize,
    ) -> c_int;
    pub(super) fn lumen_cuda_matvec_argmax_i8_i8_device(
        input_handle: u64,
        input_scale: f32,
        weight_handle: u64,
        weight_scale: f32,
        out_indices: *mut usize,
        batch_size: usize,
        vocab_size: usize,
        hidden_size: usize,
    ) -> c_int;
    pub(super) fn lumen_cuda_matmul_f32_device(
        a_handle: u64,
        b_handle: u64,
        out_handle: u64,
        m: usize,
        n: usize,
        k: usize,
    ) -> c_int;
    pub(super) fn lumen_cuda_matmul_bf16_host_device(
        a_host: *const u16,
        b_host: *const u16,
        out_handle: u64,
        m: usize,
        n: usize,
        k: usize,
    ) -> c_int;
    pub(super) fn lumen_cuda_matmul_bf16_device(
        a_handle: u64,
        b_handle: u64,
        out_handle: u64,
        m: usize,
        n: usize,
        k: usize,
    ) -> c_int;
    pub(super) fn lumen_cuda_matmul_bf16_typed_out_device(
        a_handle: u64,
        b_handle: u64,
        out_handle: u64,
        m: usize,
        n: usize,
        k: usize,
    ) -> c_int;
    pub(super) fn lumen_cuda_matmul_f16_host_device(
        a_host: *const u16,
        b_host: *const u16,
        out_handle: u64,
        m: usize,
        n: usize,
        k: usize,
    ) -> c_int;
    pub(super) fn lumen_cuda_matmul_f16_typed_out_device(
        a_handle: u64,
        b_handle: u64,
        out_handle: u64,
        m: usize,
        n: usize,
        k: usize,
    ) -> c_int;
    pub(super) fn lumen_cuda_matmul_f16_device(
        a_handle: u64,
        b_handle: u64,
        out_handle: u64,
        m: usize,
        n: usize,
        k: usize,
    ) -> c_int;
    pub(super) fn lumen_cuda_matmul_i8_host_device(
        a_host: *const i8,
        b_host: *const i8,
        a_scale: f32,
        b_scale: f32,
        out_handle: u64,
        m: usize,
        n: usize,
        k: usize,
    ) -> c_int;
    pub(super) fn lumen_cuda_matmul_i8_device(
        a_handle: u64,
        a_scale: f32,
        b_handle: u64,
        b_scale: f32,
        out_handle: u64,
        m: usize,
        n: usize,
        k: usize,
    ) -> c_int;
    pub(super) fn lumen_cuda_matmul_bf16_i8_device(
        a_handle: u64,
        b_handle: u64,
        b_scale: f32,
        out_handle: u64,
        m: usize,
        n: usize,
        k: usize,
    ) -> c_int;
    pub(super) fn lumen_cuda_matmul_f16_i8_device(
        a_handle: u64,
        b_handle: u64,
        b_scale: f32,
        out_handle: u64,
        m: usize,
        n: usize,
        k: usize,
    ) -> c_int;
    pub(super) fn lumen_cuda_matmul_bf16_i8_dynamic_device(
        a_handle: u64,
        b_handle: u64,
        b_scale: f32,
        out_handle: u64,
        m: usize,
        n: usize,
        k: usize,
    ) -> c_int;
    pub(super) fn lumen_cuda_matmul_f16_i8_dynamic_device(
        a_handle: u64,
        b_handle: u64,
        b_scale: f32,
        out_handle: u64,
        m: usize,
        n: usize,
        k: usize,
    ) -> c_int;
    pub(super) fn lumen_cuda_matmul_f32_i8_device(
        a_handle: u64,
        b_handle: u64,
        b_scale: f32,
        out_handle: u64,
        m: usize,
        n: usize,
        k: usize,
    ) -> c_int;
    pub(super) fn lumen_cuda_matmul_i8_bf16_device(
        a_handle: u64,
        a_scale: f32,
        b_handle: u64,
        out_handle: u64,
        m: usize,
        n: usize,
        k: usize,
    ) -> c_int;
    pub(super) fn lumen_cuda_matmul_i8_f16_device(
        a_handle: u64,
        a_scale: f32,
        b_handle: u64,
        out_handle: u64,
        m: usize,
        n: usize,
        k: usize,
    ) -> c_int;
    pub(super) fn lumen_cuda_matmul_i8_f32_device(
        a_handle: u64,
        a_scale: f32,
        b_handle: u64,
        out_handle: u64,
        m: usize,
        n: usize,
        k: usize,
    ) -> c_int;
    pub(super) fn lumen_cuda_matmul_i8_typed_out_device(
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
    pub(super) fn lumen_cuda_batch_matmul_f32_device(
        lhs_handle: u64,
        rhs_handle: u64,
        out_handle: u64,
        batch_count: usize,
        m: usize,
        n: usize,
        k: usize,
    ) -> c_int;
    pub(super) fn lumen_cuda_batch_matmul_bf16_device(
        lhs_handle: u64,
        rhs_handle: u64,
        out_handle: u64,
        batch_count: usize,
        m: usize,
        n: usize,
        k: usize,
    ) -> c_int;
    pub(super) fn lumen_cuda_batch_matmul_bf16_typed_out_device(
        lhs_handle: u64,
        rhs_handle: u64,
        out_handle: u64,
        batch_count: usize,
        m: usize,
        n: usize,
        k: usize,
    ) -> c_int;
    pub(super) fn lumen_cuda_batch_matmul_f16_device(
        lhs_handle: u64,
        rhs_handle: u64,
        out_handle: u64,
        batch_count: usize,
        m: usize,
        n: usize,
        k: usize,
    ) -> c_int;
    pub(super) fn lumen_cuda_batch_matmul_f16_typed_out_device(
        lhs_handle: u64,
        rhs_handle: u64,
        out_handle: u64,
        batch_count: usize,
        m: usize,
        n: usize,
        k: usize,
    ) -> c_int;
    pub(super) fn lumen_cuda_batch_matmul_bf16_i8_device(
        lhs_handle: u64,
        rhs_handle: u64,
        rhs_scale: f32,
        out_handle: u64,
        batch_count: usize,
        m: usize,
        n: usize,
        k: usize,
    ) -> c_int;
    pub(super) fn lumen_cuda_batch_matmul_f16_i8_device(
        lhs_handle: u64,
        rhs_handle: u64,
        rhs_scale: f32,
        out_handle: u64,
        batch_count: usize,
        m: usize,
        n: usize,
        k: usize,
    ) -> c_int;
    pub(super) fn lumen_cuda_batch_matmul_f32_i8_device(
        lhs_handle: u64,
        rhs_handle: u64,
        rhs_scale: f32,
        out_handle: u64,
        batch_count: usize,
        m: usize,
        n: usize,
        k: usize,
    ) -> c_int;
    pub(super) fn lumen_cuda_batch_matmul_i8_bf16_device(
        lhs_handle: u64,
        lhs_scale: f32,
        rhs_handle: u64,
        out_handle: u64,
        batch_count: usize,
        m: usize,
        n: usize,
        k: usize,
    ) -> c_int;
    pub(super) fn lumen_cuda_batch_matmul_i8_f16_device(
        lhs_handle: u64,
        lhs_scale: f32,
        rhs_handle: u64,
        out_handle: u64,
        batch_count: usize,
        m: usize,
        n: usize,
        k: usize,
    ) -> c_int;
    pub(super) fn lumen_cuda_batch_matmul_i8_f32_device(
        lhs_handle: u64,
        lhs_scale: f32,
        rhs_handle: u64,
        out_handle: u64,
        batch_count: usize,
        m: usize,
        n: usize,
        k: usize,
    ) -> c_int;
    pub(super) fn lumen_cuda_batch_matmul_i8_device(
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
    pub(super) fn lumen_cuda_batch_matmul_i8_typed_out_device(
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
    pub(super) fn lumen_cuda_matmul_backward_f32_device(
        grad_handle: u64,
        a_handle: u64,
        b_handle: u64,
        da_handle: u64,
        db_handle: u64,
        m: usize,
        k: usize,
        n: usize,
    ) -> c_int;
    pub(super) fn lumen_cuda_matmul_backward_bf16_i8_device(
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
    pub(super) fn lumen_cuda_matmul_backward_f16_i8_device(
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
    pub(super) fn lumen_cuda_matmul_backward_f32_i8_device(
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
    pub(super) fn lumen_cuda_matmul_backward_i8_bf16_device(
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
    pub(super) fn lumen_cuda_matmul_backward_i8_f16_device(
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
    pub(super) fn lumen_cuda_matmul_backward_i8_f32_device(
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
    pub(super) fn lumen_cuda_batch_matmul_backward_f32_device(
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
    pub(super) fn lumen_cuda_batch_matmul_backward_bf16_device(
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
    pub(super) fn lumen_cuda_batch_matmul_backward_f16_device(
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
    pub(super) fn lumen_cuda_batch_matmul_backward_bf16_i8_device(
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
    pub(super) fn lumen_cuda_batch_matmul_backward_f16_i8_device(
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
    pub(super) fn lumen_cuda_batch_matmul_backward_f32_i8_device(
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
    pub(super) fn lumen_cuda_batch_matmul_backward_i8_bf16_device(
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
    pub(super) fn lumen_cuda_batch_matmul_backward_i8_f16_device(
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
    pub(super) fn lumen_cuda_batch_matmul_backward_i8_f32_device(
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
    pub(super) fn lumen_cuda_batch_matmul_backward_i8_device(
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
    pub(super) fn lumen_cuda_unary_f32_device(
        input_handle: u64,
        out_handle: u64,
        len: usize,
        op: c_int,
    ) -> c_int;
    pub(super) fn lumen_cuda_unary_f16_device(
        input_handle: u64,
        out_handle: u64,
        len: usize,
        op: c_int,
    ) -> c_int;
    pub(super) fn lumen_cuda_unary_f16_typed_out_device(
        input_handle: u64,
        out_handle: u64,
        len: usize,
        op: c_int,
    ) -> c_int;
    pub(super) fn lumen_cuda_unary_bf16_device(
        input_handle: u64,
        out_handle: u64,
        len: usize,
        op: c_int,
    ) -> c_int;
    pub(super) fn lumen_cuda_unary_bf16_typed_out_device(
        input_handle: u64,
        out_handle: u64,
        len: usize,
        op: c_int,
    ) -> c_int;
    pub(super) fn lumen_cuda_unary_i8_device(
        input_handle: u64,
        scale: f32,
        out_handle: u64,
        len: usize,
        op: c_int,
    ) -> c_int;
    pub(super) fn lumen_cuda_unary_i8_relu_typed_out_device(
        input_handle: u64,
        out_handle: u64,
        len: usize,
    ) -> c_int;
    pub(super) fn lumen_cuda_unary_backward_f32_device(
        input_handle: u64,
        output_handle: u64,
        grad_handle: u64,
        out_handle: u64,
        len: usize,
        op: c_int,
    ) -> c_int;
    pub(super) fn lumen_cuda_unary_backward_f16_device(
        input_handle: u64,
        output_handle: u64,
        grad_handle: u64,
        out_handle: u64,
        len: usize,
        op: c_int,
    ) -> c_int;
    pub(super) fn lumen_cuda_unary_backward_bf16_device(
        input_handle: u64,
        output_handle: u64,
        grad_handle: u64,
        out_handle: u64,
        len: usize,
        op: c_int,
    ) -> c_int;
    pub(super) fn lumen_cuda_unary_backward_i8_device(
        input_handle: u64,
        scale: f32,
        output_handle: u64,
        grad_handle: u64,
        out_handle: u64,
        len: usize,
        op: c_int,
    ) -> c_int;
    pub(super) fn lumen_cuda_binary_f32_device(
        lhs_handle: u64,
        rhs_handle: u64,
        out_handle: u64,
        len: usize,
        op: c_int,
    ) -> c_int;
    pub(super) fn lumen_cuda_binary_typed_device(
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
    pub(super) fn lumen_cuda_binary_lowp_typed_out_device(
        lhs_handle: u64,
        rhs_handle: u64,
        out_handle: u64,
        len: usize,
        dtype: c_int,
        op: c_int,
    ) -> c_int;
    pub(super) fn lumen_cuda_binary_typed_lastdim_device(
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
    pub(super) fn lumen_cuda_binary_lowp_typed_lastdim_out_device(
        lhs_handle: u64,
        rhs_handle: u64,
        out_handle: u64,
        len: usize,
        last_dim: usize,
        vector_on_rhs: c_int,
        dtype: c_int,
        op: c_int,
    ) -> c_int;
    pub(super) fn lumen_cuda_binary_typed_row_scalar_device(
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
    pub(super) fn lumen_cuda_binary_lowp_typed_row_scalar_out_device(
        lhs_handle: u64,
        rhs_handle: u64,
        out_handle: u64,
        rows: usize,
        last_dim: usize,
        scalar_on_rhs: c_int,
        dtype: c_int,
        op: c_int,
    ) -> c_int;
    pub(super) fn lumen_cuda_binary_i8_typed_row_scalar_out_device(
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
    pub(super) fn lumen_cuda_binary_typed_broadcast_device(
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
    pub(super) fn lumen_cuda_binary_lowp_typed_broadcast_out_device(
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
    pub(super) fn lumen_cuda_binary_i8_typed_broadcast_out_device(
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
    pub(super) fn lumen_cuda_binary_typed_b1d_1h1_device(
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
    pub(super) fn lumen_cuda_binary_lowp_typed_b1d_1h1_out_device(
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
    pub(super) fn lumen_cuda_binary_i8_typed_b1d_1h1_out_device(
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
    pub(super) fn lumen_cuda_binary_typed_b1d_1hd_device(
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
    pub(super) fn lumen_cuda_binary_lowp_typed_b1d_1hd_out_device(
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
    pub(super) fn lumen_cuda_binary_i8_typed_b1d_1hd_out_device(
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
    pub(super) fn lumen_cuda_binary_f16_host_device(
        lhs_host: *const u16,
        rhs_host: *const u16,
        out_handle: u64,
        len: usize,
        op: c_int,
    ) -> c_int;
    pub(super) fn lumen_cuda_binary_f16_device(
        lhs_handle: u64,
        rhs_handle: u64,
        out_handle: u64,
        len: usize,
        op: c_int,
    ) -> c_int;
    pub(super) fn lumen_cuda_binary_f16_lastdim_device(
        lhs_handle: u64,
        rhs_handle: u64,
        out_handle: u64,
        len: usize,
        last_dim: usize,
        vector_on_rhs: c_int,
        op: c_int,
    ) -> c_int;
    pub(super) fn lumen_cuda_binary_bf16_host_device(
        lhs_host: *const u16,
        rhs_host: *const u16,
        out_handle: u64,
        len: usize,
        op: c_int,
    ) -> c_int;
    pub(super) fn lumen_cuda_binary_bf16_device(
        lhs_handle: u64,
        rhs_handle: u64,
        out_handle: u64,
        len: usize,
        op: c_int,
    ) -> c_int;
    pub(super) fn lumen_cuda_binary_bf16_lastdim_device(
        lhs_handle: u64,
        rhs_handle: u64,
        out_handle: u64,
        len: usize,
        last_dim: usize,
        vector_on_rhs: c_int,
        op: c_int,
    ) -> c_int;
    pub(super) fn lumen_cuda_binary_i8_host_device(
        lhs_host: *const i8,
        rhs_host: *const i8,
        lhs_scale: f32,
        rhs_scale: f32,
        out_handle: u64,
        len: usize,
        op: c_int,
    ) -> c_int;
    pub(super) fn lumen_cuda_binary_i8_device(
        lhs_handle: u64,
        rhs_handle: u64,
        lhs_scale: f32,
        rhs_scale: f32,
        out_handle: u64,
        len: usize,
        op: c_int,
    ) -> c_int;
    pub(super) fn lumen_cuda_binary_i8_typed_out_device(
        lhs_handle: u64,
        rhs_handle: u64,
        lhs_scale: f32,
        rhs_scale: f32,
        out_handle: u64,
        len: usize,
        op: c_int,
        out_scale: *mut f32,
    ) -> c_int;
    pub(super) fn lumen_cuda_binary_i8_lastdim_device(
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
    pub(super) fn lumen_cuda_binary_i8_typed_lastdim_out_device(
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
    pub(super) fn lumen_cuda_mul_grad_f16_host_device(
        grad_handle: u64,
        operand_host: *const u16,
        out_handle: u64,
        len: usize,
    ) -> c_int;
    pub(super) fn lumen_cuda_mul_grad_f16_device(
        grad_handle: u64,
        operand_handle: u64,
        out_handle: u64,
        len: usize,
    ) -> c_int;
    pub(super) fn lumen_cuda_mul_grad_f16_lastdim_device(
        grad_handle: u64,
        lhs_handle: u64,
        rhs_handle: u64,
        grad_lhs_handle: u64,
        grad_rhs_handle: u64,
        len: usize,
        last_dim: usize,
        vector_on_rhs: c_int,
    ) -> c_int;
    pub(super) fn lumen_cuda_mul_grad_bf16_host_device(
        grad_handle: u64,
        operand_host: *const u16,
        out_handle: u64,
        len: usize,
    ) -> c_int;
    pub(super) fn lumen_cuda_mul_grad_bf16_device(
        grad_handle: u64,
        operand_handle: u64,
        out_handle: u64,
        len: usize,
    ) -> c_int;
    pub(super) fn lumen_cuda_mul_grad_bf16_lastdim_device(
        grad_handle: u64,
        lhs_handle: u64,
        rhs_handle: u64,
        grad_lhs_handle: u64,
        grad_rhs_handle: u64,
        len: usize,
        last_dim: usize,
        vector_on_rhs: c_int,
    ) -> c_int;
    pub(super) fn lumen_cuda_mul_grad_i8_host_device(
        grad_handle: u64,
        operand_host: *const i8,
        scale: f32,
        out_handle: u64,
        len: usize,
    ) -> c_int;
    pub(super) fn lumen_cuda_mul_grad_i8_device(
        grad_handle: u64,
        operand_handle: u64,
        scale: f32,
        out_handle: u64,
        len: usize,
    ) -> c_int;
    pub(super) fn lumen_cuda_mul_grad_i8_lastdim_device(
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
    pub(super) fn lumen_cuda_mul_grad_typed_device(
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
    pub(super) fn lumen_cuda_mul_grad_typed_lastdim_device(
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
    pub(super) fn lumen_cuda_mul_grad_typed_row_scalar_device(
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
    pub(super) fn lumen_cuda_mul_grad_typed_broadcast_device(
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
    pub(super) fn lumen_cuda_mul_grad_typed_b1d_1h1_device(
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
    pub(super) fn lumen_cuda_mul_grad_typed_b1d_1hd_device(
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
    pub(super) fn lumen_cuda_mul_grad_typed_scalar_device(
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
    pub(super) fn lumen_cuda_binary_backward_f32_device(
        lhs_handle: u64,
        rhs_handle: u64,
        grad_handle: u64,
        grad_lhs_handle: u64,
        grad_rhs_handle: u64,
        len: usize,
        op: c_int,
    ) -> c_int;
    pub(super) fn lumen_cuda_add_sub_backward_f32_device(
        grad_handle: u64,
        grad_lhs_handle: u64,
        grad_rhs_handle: u64,
        len: usize,
        op: c_int,
    ) -> c_int;
    pub(super) fn lumen_cuda_add_sub_backward_lastdim_f32_device(
        grad_handle: u64,
        grad_lhs_handle: u64,
        grad_rhs_handle: u64,
        len: usize,
        last_dim: usize,
        vector_on_rhs: c_int,
        op: c_int,
    ) -> c_int;
    pub(super) fn lumen_cuda_add_sub_backward_scalar_f32_device(
        grad_handle: u64,
        grad_lhs_handle: u64,
        grad_rhs_handle: u64,
        len: usize,
        scalar_on_rhs: c_int,
        op: c_int,
    ) -> c_int;
    pub(super) fn lumen_cuda_add_sub_backward_row_scalar_f32_device(
        grad_handle: u64,
        grad_lhs_handle: u64,
        grad_rhs_handle: u64,
        rows: usize,
        last_dim: usize,
        scalar_on_rhs: c_int,
        op: c_int,
    ) -> c_int;
    pub(super) fn lumen_cuda_add_sub_backward_b1d_1h1_f32_device(
        grad_handle: u64,
        grad_lhs_handle: u64,
        grad_rhs_handle: u64,
        batch: usize,
        heads: usize,
        dim: usize,
        b1d_on_lhs: c_int,
        op: c_int,
    ) -> c_int;
    pub(super) fn lumen_cuda_add_sub_backward_b1d_1hd_f32_device(
        grad_handle: u64,
        grad_lhs_handle: u64,
        grad_rhs_handle: u64,
        batch: usize,
        heads: usize,
        dim: usize,
        b1d_on_lhs: c_int,
        op: c_int,
    ) -> c_int;
    pub(super) fn lumen_cuda_add_sub_broadcast_backward_f32_device(
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
    pub(super) fn lumen_cuda_binary_broadcast_f32_device(
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
    pub(super) fn lumen_cuda_binary_broadcast_backward_f32_device(
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
    pub(super) fn lumen_cuda_sum_f32_device(
        input_handle: u64,
        out_handle: u64,
        len: usize,
    ) -> c_int;
    pub(super) fn lumen_cuda_sum_f16_device(
        input_handle: u64,
        out_handle: u64,
        len: usize,
    ) -> c_int;
    pub(super) fn lumen_cuda_sum_bf16_device(
        input_handle: u64,
        out_handle: u64,
        len: usize,
    ) -> c_int;
    pub(super) fn lumen_cuda_sum_i8_device(
        input_handle: u64,
        scale: f32,
        out_handle: u64,
        len: usize,
    ) -> c_int;
    pub(super) fn lumen_cuda_fill_scalar_f32_device(
        out_handle: u64,
        len: usize,
        value: f32,
    ) -> c_int;
    pub(super) fn lumen_cuda_add_inplace_f32_device(
        dst_handle: u64,
        src_handle: u64,
        len: usize,
    ) -> c_int;
    pub(super) fn lumen_cuda_sum_lastdim_f32_device(
        input_handle: u64,
        out_handle: u64,
        rows: usize,
        last_dim: usize,
    ) -> c_int;
    pub(super) fn lumen_cuda_bshd_to_bhsd_add_bias_f32_device(
        input_handle: u64,
        bias_handle: u64,
        out_handle: u64,
        batch: usize,
        seq: usize,
        heads: usize,
        dim: usize,
    ) -> c_int;
    pub(super) fn lumen_cuda_mse_forward_typed_device(
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
    pub(super) fn lumen_cuda_mse_backward_f32_device(
        diff_handle: u64,
        grad_output_handle: u64,
        grad_target_handle: u64,
        len: usize,
        factor: f32,
    ) -> c_int;
    pub(super) fn lumen_cuda_cross_entropy_backward_f32_device(
        softmax_handle: u64,
        target_handle: u64,
        out_handle: u64,
        len: usize,
        factor: f32,
    ) -> c_int;
    pub(super) fn lumen_cuda_cross_entropy_loss_f32_device(
        softmax_handle: u64,
        target_handle: u64,
        out_handle: u64,
        len: usize,
        factor: f32,
    ) -> c_int;
    pub(super) fn lumen_cuda_cross_entropy_backward_typed_target_device(
        softmax_handle: u64,
        target_handle: u64,
        target_dtype: c_int,
        target_scale: f32,
        out_handle: u64,
        len: usize,
        factor: f32,
    ) -> c_int;
    pub(super) fn lumen_cuda_cross_entropy_loss_typed_target_device(
        softmax_handle: u64,
        target_handle: u64,
        target_dtype: c_int,
        target_scale: f32,
        out_handle: u64,
        len: usize,
        factor: f32,
    ) -> c_int;
    pub(super) fn lumen_cuda_cross_entropy_logits_device(
        logits: u64,
        logits_dtype: c_int,
        logits_scale: f32,
        target: u64,
        target_dtype: c_int,
        target_scale: f32,
        probabilities: u64,
        loss: u64,
        rows: usize,
        cols: usize,
    ) -> c_int;
    pub(super) fn lumen_cuda_cross_entropy_logits_backward_device(
        logits: u64,
        logits_dtype: c_int,
        logits_scale: f32,
        target: u64,
        target_dtype: c_int,
        target_scale: f32,
        probabilities: u64,
        grad_logits: u64,
        grad_targets: u64,
        rows: usize,
        cols: usize,
        factor: f32,
    ) -> c_int;
    pub(super) fn lumen_cuda_sgd_update_f32_device(
        param_handle: u64,
        grad_handle: u64,
        len: usize,
        lr: f32,
    ) -> c_int;
    pub(super) fn lumen_cuda_sgd_update_f32_batched_device(
        param_handles: *const u64,
        grad_handles: *const u64,
        lens: *const usize,
        count: usize,
        lr: f32,
    ) -> c_int;
    pub(super) fn lumen_cuda_quantize_f32_storage_device(
        param_handle: u64,
        len: usize,
        dtype: c_int,
        scale: f32,
    ) -> c_int;
    pub(super) fn lumen_cuda_quantize_f32_to_i8_dynamic_device(
        input_handle: u64,
        out_handle: u64,
        len: usize,
        out_scale: *mut f32,
    ) -> c_int;
    pub(super) fn lumen_cuda_f32_to_lowp_storage_device(
        input_handle: u64,
        out_handle: u64,
        len: usize,
        dtype: c_int,
    ) -> c_int;
    pub(super) fn lumen_cuda_f32_to_i8_storage_device(
        input_handle: u64,
        out_handle: u64,
        len: usize,
        scale: f32,
    ) -> c_int;
    pub(super) fn lumen_cuda_optimizer_lowp_state_device(
        params: *const u64,
        grads: *const u64,
        first: *const u64,
        second: *const u64,
        lens: *const usize,
        count: usize,
        state_dtype: c_int,
        lr: f32,
        beta1: f32,
        beta2: f32,
        correction1: f32,
        correction2: f32,
        eps: f32,
    ) -> c_int;
    pub(super) fn lumen_cuda_sgd_momentum_update_f32_device(
        param_handle: u64,
        grad_handle: u64,
        velocity_handle: u64,
        len: usize,
        lr: f32,
        momentum: f32,
    ) -> c_int;
    pub(super) fn lumen_cuda_sgd_momentum_update_f32_batched_device(
        param_handles: *const u64,
        grad_handles: *const u64,
        velocity_handles: *const u64,
        lens: *const usize,
        count: usize,
        lr: f32,
        momentum: f32,
    ) -> c_int;
    pub(super) fn lumen_cuda_adam_update_f32_device(
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
    pub(super) fn lumen_cuda_adam_update_f32_batched_device(
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
    pub(super) fn lumen_cuda_softmax_lastdim_f32_device(
        input_handle: u64,
        out_handle: u64,
        outer: usize,
        last_dim: usize,
    ) -> c_int;
    pub(super) fn lumen_cuda_softmax_lastdim_typed_device(
        input_handle: u64,
        input_dtype: c_int,
        input_scale: f32,
        out_handle: u64,
        outer: usize,
        last_dim: usize,
    ) -> c_int;
    pub(super) fn lumen_cuda_softmax_lastdim_backward_f32_device(
        output_handle: u64,
        grad_handle: u64,
        out_handle: u64,
        outer: usize,
        last_dim: usize,
    ) -> c_int;
    pub(super) fn lumen_cuda_fused_softmax_f32_device(
        input_handle: u64,
        out_handle: u64,
        batch_heads: usize,
        q_len: usize,
        k_len: usize,
        scale: f32,
        is_causal: c_int,
    ) -> c_int;
    pub(super) fn lumen_cuda_fused_softmax_backward_f32_device(
        output_handle: u64,
        grad_handle: u64,
        out_handle: u64,
        batch_heads: usize,
        q_len: usize,
        k_len: usize,
        scale: f32,
    ) -> c_int;
    pub(super) fn lumen_cuda_fused_softmax_f32_with_past_device(
        input_handle: u64,
        out_handle: u64,
        batch_heads: usize,
        q_len: usize,
        k_len: usize,
        scale: f32,
        is_causal: c_int,
        past_len: usize,
    ) -> c_int;
    pub(super) fn lumen_cuda_embedding_f32_device(
        indices_handle: u64,
        weight_handle: u64,
        out_handle: u64,
        num_indices: usize,
        vocab_size: usize,
        embed_dim: usize,
    ) -> c_int;
    pub(super) fn lumen_cuda_embedding_typed_device(
        indices_handle: u64,
        weight_handle: u64,
        weight_dtype: c_int,
        weight_scale: f32,
        out_handle: u64,
        num_indices: usize,
        vocab_size: usize,
        embed_dim: usize,
    ) -> c_int;
    pub(super) fn lumen_cuda_embedding_typed_same_dtype_device(
        indices_handle: u64,
        weight_handle: u64,
        weight_dtype: c_int,
        out_handle: u64,
        num_indices: usize,
        vocab_size: usize,
        embed_dim: usize,
    ) -> c_int;
    pub(super) fn lumen_cuda_embedding_backward_f32_device(
        indices_handle: u64,
        grad_handle: u64,
        grad_weight_handle: u64,
        num_indices: usize,
        vocab_size: usize,
        embed_dim: usize,
    ) -> c_int;
    pub(super) fn lumen_cuda_rms_norm_f32_device(
        input_handle: u64,
        weight_handle: u64,
        out_handle: u64,
        rows: usize,
        dim: usize,
        eps: f32,
    ) -> c_int;
    pub(super) fn lumen_cuda_rms_norm_typed_device(
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
    pub(super) fn lumen_cuda_rms_norm_i8_typed_out_device(
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
    pub(super) fn lumen_cuda_rms_norm_backward_f32_device(
        input_handle: u64,
        weight_handle: u64,
        grad_handle: u64,
        grad_input_handle: u64,
        grad_weight_handle: u64,
        rows: usize,
        dim: usize,
        eps: f32,
    ) -> c_int;
    pub(super) fn lumen_cuda_rms_norm_backward_typed_device(
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
    pub(super) fn lumen_cuda_permute_f32_device(
        input_handle: u64,
        out_handle: u64,
        ndim: usize,
        out_shape: *const usize,
        out_strides: *const usize,
        mapped_input_strides: *const usize,
        len: usize,
    ) -> c_int;
    pub(super) fn lumen_cuda_permute_typed_device(
        input_handle: u64,
        dtype: c_int,
        out_handle: u64,
        ndim: usize,
        out_shape: *const usize,
        out_strides: *const usize,
        mapped_input_strides: *const usize,
        len: usize,
    ) -> c_int;
    pub(super) fn lumen_cuda_slice_lastdim_f32_device(
        input_handle: u64,
        out_handle: u64,
        outer: usize,
        input_last_dim: usize,
        start: usize,
        slice_len: usize,
    ) -> c_int;
    pub(super) fn lumen_cuda_slice_lastdim_typed_device(
        input_handle: u64,
        dtype: c_int,
        out_handle: u64,
        outer: usize,
        input_last_dim: usize,
        start: usize,
        slice_len: usize,
    ) -> c_int;
    pub(super) fn lumen_cuda_slice_lastdim_backward_f32_device(
        grad_handle: u64,
        out_handle: u64,
        outer: usize,
        input_last_dim: usize,
        start: usize,
        slice_len: usize,
    ) -> c_int;
    pub(super) fn lumen_cuda_cat_f32_device(
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
    pub(super) fn lumen_cuda_cat_typed_device(
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
    pub(super) fn lumen_cuda_cat_backward_slice_f32_device(
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
    pub(super) fn lumen_cuda_repeat_kv_f32_device(
        input_handle: u64,
        out_handle: u64,
        batch_size: usize,
        num_kv_heads: usize,
        seq_len: usize,
        dim: usize,
        n_rep: usize,
    ) -> c_int;
    pub(super) fn lumen_cuda_repeat_kv_typed_device(
        input_handle: u64,
        dtype: c_int,
        out_handle: u64,
        batch_size: usize,
        num_kv_heads: usize,
        seq_len: usize,
        dim: usize,
        n_rep: usize,
    ) -> c_int;
    pub(super) fn lumen_cuda_repeat_kv_backward_f32_device(
        grad_handle: u64,
        out_handle: u64,
        batch_size: usize,
        num_kv_heads: usize,
        seq_len: usize,
        dim: usize,
        n_rep: usize,
    ) -> c_int;
    pub(super) fn lumen_cuda_decode_attention_f32_device(
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
    pub(super) fn lumen_cuda_prefill_attention_f32_device(
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
    pub(super) fn lumen_cuda_fused_gate_up_silu_f32_device(
        input_handle: u64,
        gate_handle: u64,
        up_handle: u64,
        out_handle: u64,
        rows: usize,
        n_dim: usize,
        k_dim: usize,
    ) -> c_int;
    pub(super) fn lumen_cuda_silu_mul_f32_device(
        gate_handle: u64,
        up_handle: u64,
        out_handle: u64,
        len: usize,
    ) -> c_int;
    pub(super) fn lumen_cuda_fused_gate_up_silu_typed_device(
        input_handle: u64,
        input_dtype: c_int,
        input_scale: f32,
        gate_handle: u64,
        weight_dtype: c_int,
        gate_scale: f32,
        up_handle: u64,
        up_scale: f32,
        out_handle: u64,
        allow_dynamic_i8: c_int,
        rows: usize,
        n_dim: usize,
        k_dim: usize,
    ) -> c_int;
    pub(super) fn lumen_cuda_fused_gate_up_silu_typed_out_device(
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
    pub(super) fn lumen_cuda_fused_qkv_f32_device(
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
    pub(super) fn lumen_cuda_fused_qkv_typed_device(
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
        allow_dynamic_i8: c_int,
        rows: usize,
        q_n: usize,
        k_n: usize,
        k_dim: usize,
    ) -> c_int;
    pub(super) fn lumen_cuda_fused_qkv_typed_out_device(
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
    pub(super) fn lumen_cuda_rope_f32_device(
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
    pub(super) fn lumen_cuda_rope_typed_device(
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
    pub(super) fn lumen_cuda_rope_typed_i8_dynamic_device(
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
    pub(super) fn lumen_cuda_rope_backward_f32_device(
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
    pub(super) fn lumen_cuda_conv2d_f32_device(
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
    pub(super) fn lumen_cuda_conv2d_typed_device(
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
    pub(super) fn lumen_cuda_conv2d_backward_f32_device(
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
    pub(super) fn lumen_cuda_max_pool2d_f32_device(
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
    pub(super) fn lumen_cuda_max_pool2d_typed_device(
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
    pub(super) fn lumen_cuda_max_pool2d_backward_f32_device(
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
    pub(super) fn lumen_cuda_max_pool2d_backward_typed_device(
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
    pub(super) fn lumen_cuda_last_error_message() -> *const c_char;
}
