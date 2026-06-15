use crate::autograd::{
    Device, StoragePreference, Tensor, TensorData, TensorStorageOwned, TensorStorageView,
    assert_native_device_support, is_no_grad, is_strict_device_execution,
};
use crate::ops::cuda;
use crate::precision::{DType, default_activation_dtype};
use half::{bf16, f16};
use ndarray::{Array, ArrayD, Ix2, Zip, s};
use std::cell::RefCell;
use std::rc::Rc;

pub struct RotaryEmbedding {
    dim: usize,
    max_seq_len: usize,
    // 缓存预计算的 cos/sin
    // Shape: [1, 1, Max_Seq, Dim]
    cos_cache: Tensor,
    sin_cache: Tensor,
}

fn cuda_rope_storage_buffer(tensor: &Tensor) -> Option<(DType, cuda::CudaBuffer, Option<f32>)> {
    tensor.cloned_cuda_native_lowp_buffer().or_else(|| {
        if tensor.dtype() == DType::F32 && tensor.is_cuda() {
            tensor
                .cloned_cuda_f32_buffer()
                .map(|buffer| (DType::F32, buffer, None))
        } else {
            None
        }
    })
}

impl RotaryEmbedding {
    pub fn new(dim: usize, max_seq_len: usize, theta: f32) -> Self {
        Self::new_with_dtype(dim, max_seq_len, theta, default_activation_dtype())
    }

    pub fn new_with_dtype(dim: usize, max_seq_len: usize, theta: f32, dtype: DType) -> Self {
        let cache_dtype = if dtype == DType::I8 {
            DType::F32
        } else {
            dtype
        };
        assert!(
            cache_dtype.is_float(),
            "RotaryEmbedding cache currently only supports floating runtime dtypes, got {:?}",
            dtype
        );
        let (cos, sin) = Self::precompute_freqs_cis(dim, max_seq_len, theta);
        let cos_cache = Tensor::from_array_no_grad(cos);
        let sin_cache = Tensor::from_array_no_grad(sin);
        cos_cache.cast_inplace(cache_dtype);
        sin_cache.cast_inplace(cache_dtype);

        Self {
            dim,
            max_seq_len,
            cos_cache,
            sin_cache,
        }
    }

    #[inline]
    pub fn cache_dtype(&self) -> DType {
        self.cos_cache.dtype()
    }

    fn precompute_freqs_cis(
        dim: usize,
        max_seq_len: usize,
        theta: f32,
    ) -> (ArrayD<f32>, ArrayD<f32>) {
        let half_d = dim / 2;
        let mut cos_arr = Array::zeros((1, 1, max_seq_len, dim));
        let mut sin_arr = Array::zeros((1, 1, max_seq_len, dim));

        for i in 0..max_seq_len {
            let pos = i as f32;
            for j in 0..half_d {
                let freq = 1.0 / theta.powf((j as f32 * 2.0) / dim as f32);
                let val = pos * freq;

                let c = val.cos();
                let s = val.sin();

                cos_arr[[0, 0, i, j]] = c;
                cos_arr[[0, 0, i, j + half_d]] = c;
                sin_arr[[0, 0, i, j]] = s;
                sin_arr[[0, 0, i, j + half_d]] = s;
            }
        }
        (cos_arr.into_dyn(), sin_arr.into_dyn())
    }

    pub fn to_device(&self, device: Device) {
        self.cos_cache.to_device_inplace(device);
        self.sin_cache.to_device_inplace(device);
    }

    pub fn to_cpu(&self) {
        self.to_device(Device::Cpu);
    }

    pub fn to_cuda(&self) {
        self.to_device(Device::Cuda);
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn decode_rope_q_append_kv_cuda(
        &self,
        q_src: &cuda::CudaBuffer,
        k_src: &cuda::CudaBuffer,
        v_src: &cuda::CudaBuffer,
        k_cache: &cuda::CudaBuffer,
        v_cache: &cuda::CudaBuffer,
        batch_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        dst_seq_len: usize,
        offset: usize,
    ) -> Result<cuda::CudaBuffer, String> {
        self.to_device(Device::Cuda);
        if offset >= self.max_seq_len {
            return Err(format!(
                "CUDA decode RoPE offset out of bounds: offset {} >= max_seq_len {}",
                offset, self.max_seq_len
            ));
        }
        self.cos_cache.with_cuda_f32_buffer(|cos_buf| {
            self.sin_cache.with_cuda_f32_buffer(|sin_buf| {
                cuda::decode_rope_q_append_kv_f32_buffer(
                    q_src,
                    k_src,
                    v_src,
                    cos_buf,
                    sin_buf,
                    k_cache,
                    v_cache,
                    batch_size,
                    num_heads,
                    num_kv_heads,
                    self.dim,
                    dst_seq_len,
                    offset,
                    self.max_seq_len,
                )
            })
        })
    }

    pub fn forward(&self, x: &Tensor, offset: usize) -> Tensor {
        let output_device = x.device();
        let build_graph = !is_no_grad() && x.requires_grad();
        let cuda_native_supported = output_device == Device::Cuda;
        assert_native_device_support(output_device, "rope", cuda_native_supported);
        self.to_device(output_device);

        if !build_graph {
            let shape = x.shape_vec();
            assert_eq!(shape.len(), 4, "RoPE expects input [B,H,S,D]");
            let (b, h, seq_len, d) = (shape[0], shape[1], shape[2], shape[3]);
            assert_eq!(d, self.dim, "RoPE dimension mismatch");
            let end = offset + seq_len;
            if end > self.max_seq_len {
                panic!(
                    "RoPE index out of range: offset {} + len {} > max {}",
                    offset, seq_len, self.max_seq_len
                );
            }

            if output_device == Device::Cuda && !x.is_empty() {
                if let Some((dtype, x_buf, scale)) = x.cloned_cuda_native_lowp_buffer()
                    && let (
                        Some((cos_dtype, cos_buf, cos_scale)),
                        Some((sin_dtype, sin_buf, sin_scale)),
                    ) = (
                        cuda_rope_storage_buffer(&self.cos_cache),
                        cuda_rope_storage_buffer(&self.sin_cache),
                    )
                    && cos_dtype == sin_dtype
                {
                    if x.dtype() == DType::I8 {
                        match cuda::rope_typed_i8_dynamic_buffer(
                            &x_buf,
                            dtype,
                            scale,
                            &cos_buf,
                            &sin_buf,
                            cos_dtype,
                            cos_scale,
                            sin_scale,
                            b,
                            h,
                            seq_len,
                            d,
                            offset,
                            self.max_seq_len,
                        ) {
                            Ok((i8_buffer, scale)) => {
                                return Tensor::from_cuda_native_lowp_buffer_no_host_with_dtype(
                                    &shape,
                                    i8_buffer,
                                    output_device,
                                    DType::I8,
                                    Some(scale),
                                );
                            }
                            Err(err) => {
                                assert!(
                                    !is_strict_device_execution(),
                                    "CUDA RoPE fused I8 output quantization failed while strict device execution is enabled: {err}"
                                );
                            }
                        }
                    }
                    let cuda_out = cuda::rope_typed_buffer(
                        &x_buf,
                        dtype,
                        scale,
                        &cos_buf,
                        &sin_buf,
                        cos_dtype,
                        cos_scale,
                        sin_scale,
                        b,
                        h,
                        seq_len,
                        d,
                        offset,
                        self.max_seq_len,
                    );
                    if let Ok(buffer) = cuda_out {
                        if matches!(x.dtype(), DType::F16 | DType::BF16) {
                            match cuda::f32_to_lowp_storage_no_host(&buffer, x.dtype()) {
                                Ok(lowp_buffer) => {
                                    return Tensor::from_cuda_native_lowp_buffer_no_host_with_dtype(
                                        &shape,
                                        lowp_buffer,
                                        output_device,
                                        x.dtype(),
                                        None,
                                    );
                                }
                                Err(err) => {
                                    assert!(
                                        !is_strict_device_execution(),
                                        "CUDA RoPE {:?} output conversion failed while strict device execution is enabled: {err}",
                                        x.dtype()
                                    );
                                }
                            }
                        }
                        return Tensor::from_cuda_f32_buffer_no_host_with_dtype(
                            &shape,
                            buffer,
                            output_device,
                            x.dtype(),
                        );
                    }
                }

                if x.dtype().is_float() {
                    let cuda_out = x.with_cuda_f32_buffer(|x_buf| {
                        self.cos_cache.with_cuda_f32_buffer(|cos_buf| {
                            self.sin_cache.with_cuda_f32_buffer(|sin_buf| {
                                cuda::rope_f32_buffer(
                                    x_buf,
                                    cos_buf,
                                    sin_buf,
                                    b,
                                    h,
                                    seq_len,
                                    d,
                                    offset,
                                    self.max_seq_len,
                                )
                            })
                        })
                    });
                    if let Ok(buffer) = cuda_out {
                        if matches!(x.dtype(), DType::F16 | DType::BF16) {
                            match cuda::f32_to_lowp_storage_no_host(&buffer, x.dtype()) {
                                Ok(lowp_buffer) => {
                                    return Tensor::from_cuda_native_lowp_buffer_no_host_with_dtype(
                                        &shape,
                                        lowp_buffer,
                                        output_device,
                                        x.dtype(),
                                        None,
                                    );
                                }
                                Err(err) => {
                                    assert!(
                                        !is_strict_device_execution(),
                                        "CUDA RoPE {:?} output conversion failed while strict device execution is enabled: {err}",
                                        x.dtype()
                                    );
                                }
                            }
                        }
                        return Tensor::from_cuda_f32_buffer_no_host_with_dtype(
                            &shape,
                            buffer,
                            output_device,
                            x.dtype(),
                        );
                    }
                } else {
                    let cuda_i8_out = if x.dtype() == DType::I8 {
                        x.with_cuda_f32_buffer(|x_buf| {
                            self.cos_cache.with_cuda_f32_buffer(|cos_buf| {
                                self.sin_cache.with_cuda_f32_buffer(|sin_buf| {
                                    cuda::rope_typed_i8_dynamic_buffer(
                                        x_buf,
                                        DType::F32,
                                        None,
                                        cos_buf,
                                        sin_buf,
                                        DType::F32,
                                        None,
                                        None,
                                        b,
                                        h,
                                        seq_len,
                                        d,
                                        offset,
                                        self.max_seq_len,
                                    )
                                })
                            })
                        })
                    } else {
                        Err("CUDA RoPE fused I8 fallback is only valid for I8 output".to_string())
                    };
                    if let Ok((i8_buffer, scale)) = cuda_i8_out {
                        return Tensor::from_cuda_native_lowp_buffer_no_host_with_dtype(
                            &shape,
                            i8_buffer,
                            output_device,
                            DType::I8,
                            Some(scale),
                        );
                    }

                    let cuda_out = x.with_cuda_f32_buffer(|x_buf| {
                        self.cos_cache.with_cuda_f32_buffer(|cos_buf| {
                            self.sin_cache.with_cuda_f32_buffer(|sin_buf| {
                                cuda::rope_f32_buffer(
                                    x_buf,
                                    cos_buf,
                                    sin_buf,
                                    b,
                                    h,
                                    seq_len,
                                    d,
                                    offset,
                                    self.max_seq_len,
                                )
                            })
                        })
                    });
                    if let Ok(buffer) = cuda_out {
                        if x.dtype() == DType::I8 {
                            match cuda::quantize_f32_to_i8_dynamic_no_host(&buffer) {
                                Ok((i8_buffer, scale)) => {
                                    return Tensor::from_cuda_native_lowp_buffer_no_host_with_dtype(
                                        &shape,
                                        i8_buffer,
                                        output_device,
                                        DType::I8,
                                        Some(scale),
                                    );
                                }
                                Err(err) => {
                                    assert!(
                                        !is_strict_device_execution(),
                                        "CUDA RoPE I8 output quantization failed while strict device execution is enabled: {err}"
                                    );
                                }
                            }
                        }
                        if matches!(x.dtype(), DType::F16 | DType::BF16) {
                            match cuda::f32_to_lowp_storage_no_host(&buffer, x.dtype()) {
                                Ok(lowp_buffer) => {
                                    return Tensor::from_cuda_native_lowp_buffer_no_host_with_dtype(
                                        &shape,
                                        lowp_buffer,
                                        output_device,
                                        x.dtype(),
                                        None,
                                    );
                                }
                                Err(err) => {
                                    assert!(
                                        !is_strict_device_execution(),
                                        "CUDA RoPE {:?} output conversion failed while strict device execution is enabled: {err}",
                                        x.dtype()
                                    );
                                }
                            }
                        }
                        return Tensor::from_cuda_f32_buffer_no_host_with_dtype(
                            &shape,
                            buffer,
                            output_device,
                            x.dtype(),
                        );
                    }
                }
            }

            if x.dtype() == DType::I8 {
                return match x.native_storage_owned() {
                    TensorStorageOwned::I8(x_data, x_scale) => self
                        .cos_cache
                        .with_storage_view_preferring(StoragePreference::F32Compute, |cos_view| {
                            self.sin_cache.with_storage_view_preferring(
                                StoragePreference::F32Compute,
                                |sin_view| {
                                    let cos_4d = match cos_view {
                                        TensorStorageView::F32(view) => view
                                            .into_dimensionality::<ndarray::Ix4>()
                                            .expect("RoPE Cache dimensionality mismatch"),
                                        TensorStorageView::F16(_) => unreachable!(
                                            "f32 compute preference should expose f32 view"
                                        ),
                                        TensorStorageView::BF16(_) => unreachable!(
                                            "f32 compute preference should expose f32 view"
                                        ),
                                    };
                                    let sin_4d = match sin_view {
                                        TensorStorageView::F32(view) => view
                                            .into_dimensionality::<ndarray::Ix4>()
                                            .expect("RoPE Cache dimensionality mismatch"),
                                        TensorStorageView::F16(_) => unreachable!(
                                            "f32 compute preference should expose f32 view"
                                        ),
                                        TensorStorageView::BF16(_) => unreachable!(
                                            "f32 compute preference should expose f32 view"
                                        ),
                                    };
                                    let cos_slice_2d = cos_4d
                                        .slice(s![0, 0, offset..end, ..])
                                        .into_dimensionality::<Ix2>()
                                        .expect("RoPE Cache dimensionality mismatch");
                                    let sin_slice_2d = sin_4d
                                        .slice(s![0, 0, offset..end, ..])
                                        .into_dimensionality::<Ix2>()
                                        .expect("RoPE Cache dimensionality mismatch");

                                    let x_view = x_data
                                        .view()
                                        .into_dimensionality::<ndarray::Ix4>()
                                        .unwrap();
                                    let mut out = Array::zeros(x_data.raw_dim());
                                    let mut out_view = out
                                        .view_mut()
                                        .into_dimensionality::<ndarray::Ix4>()
                                        .unwrap();

                                    Zip::from(out_view.outer_iter_mut())
                                        .and(x_view.outer_iter())
                                        .par_for_each(|mut out_b, x_b| {
                                            Zip::from(out_b.outer_iter_mut())
                                                .and(x_b.outer_iter())
                                                .for_each(|mut out_h, x_h| {
                                                    let half = d / 2;
                                                    for ss in 0..seq_len {
                                                        for j in 0..half {
                                                            let x1 = x_h[[ss, j]] as f32 * x_scale;
                                                            let x2 = x_h[[ss, j + half]] as f32
                                                                * x_scale;
                                                            let c = cos_slice_2d[[ss, j]];
                                                            let s_val = sin_slice_2d[[ss, j]];
                                                            out_h[[ss, j]] = x1 * c - x2 * s_val;
                                                            out_h[[ss, j + half]] =
                                                                x2 * c + x1 * s_val;
                                                        }
                                                    }
                                                });
                                        });

                                    let out = Tensor::from_f32_data_no_grad_with_device_dtype(
                                        out.into_dyn(),
                                        DType::F32,
                                        output_device,
                                    );
                                    out.cast_inplace(DType::I8);
                                    out
                                },
                            )
                        }),
                    TensorStorageOwned::F32(_)
                    | TensorStorageOwned::F16(_)
                    | TensorStorageOwned::BF16(_) => unreachable!("checked i8 input above"),
                };
            }

            return x.with_storage_view_preferring(StoragePreference::Native, |x_view| {
                let shape = match &x_view {
                    TensorStorageView::F32(x_view) => x_view.shape().to_vec(),
                    TensorStorageView::F16(x_view) => x_view.shape().to_vec(),
                    TensorStorageView::BF16(x_view) => x_view.shape().to_vec(),
                };
                self.cos_cache.with_storage_view_preferring(
                    StoragePreference::F32Compute,
                    |cos_view| {
                        self.sin_cache.with_storage_view_preferring(
                            StoragePreference::F32Compute,
                            |sin_view| {
                                let cos_4d = match cos_view {
                                    TensorStorageView::F32(view) => view
                                        .into_dimensionality::<ndarray::Ix4>()
                                        .expect("RoPE Cache dimensionality mismatch"),
                                    TensorStorageView::F16(_) => {
                                        unreachable!(
                                            "f32 compute preference should expose f32 view"
                                        )
                                    }
                                    TensorStorageView::BF16(_) => {
                                        unreachable!(
                                            "f32 compute preference should expose f32 view"
                                        )
                                    }
                                };
                                let sin_4d = match sin_view {
                                    TensorStorageView::F32(view) => view
                                        .into_dimensionality::<ndarray::Ix4>()
                                        .expect("RoPE Cache dimensionality mismatch"),
                                    TensorStorageView::F16(_) => {
                                        unreachable!(
                                            "f32 compute preference should expose f32 view"
                                        )
                                    }
                                    TensorStorageView::BF16(_) => {
                                        unreachable!(
                                            "f32 compute preference should expose f32 view"
                                        )
                                    }
                                };
                                let cos_slice_2d = cos_4d
                                    .slice(s![0, 0, offset..end, ..])
                                    .into_dimensionality::<Ix2>()
                                    .expect("RoPE Cache dimensionality mismatch");
                                let sin_slice_2d = sin_4d
                                    .slice(s![0, 0, offset..end, ..])
                                    .into_dimensionality::<Ix2>()
                                    .expect("RoPE Cache dimensionality mismatch");

                                match x_view {
                                    TensorStorageView::F32(x_view) => {
                                        let mut out = Array::zeros(x_view.raw_dim());
                                        let x_view =
                                            x_view.into_dimensionality::<ndarray::Ix4>().unwrap();
                                        let mut out_view = out
                                            .view_mut()
                                            .into_dimensionality::<ndarray::Ix4>()
                                            .unwrap();

                                        Zip::from(out_view.outer_iter_mut())
                                            .and(x_view.outer_iter())
                                            .par_for_each(|mut out_b, x_b| {
                                                Zip::from(out_b.outer_iter_mut())
                                                    .and(x_b.outer_iter())
                                                    .for_each(|mut out_h, x_h| {
                                                        let half = d / 2;
                                                        for ss in 0..seq_len {
                                                            for j in 0..half {
                                                                let x1 = x_h[[ss, j]];
                                                                let x2 = x_h[[ss, j + half]];
                                                                let c = cos_slice_2d[[ss, j]];
                                                                let s_val = sin_slice_2d[[ss, j]];
                                                                out_h[[ss, j]] =
                                                                    x1 * c - x2 * s_val;
                                                                out_h[[ss, j + half]] =
                                                                    x2 * c + x1 * s_val;
                                                            }
                                                        }
                                                    });
                                            });

                                        Tensor::from_f32_data_no_grad_with_device_dtype(
                                            out.into_dyn(),
                                            DType::F32,
                                            output_device,
                                        )
                                    }
                                    TensorStorageView::F16(x_view) => {
                                        let mut out = ndarray::ArrayD::<f16>::from_elem(
                                            ndarray::IxDyn(&shape),
                                            f16::from_bits(0),
                                        );
                                        let x_view =
                                            x_view.into_dimensionality::<ndarray::Ix4>().unwrap();
                                        let mut out_view = out
                                            .view_mut()
                                            .into_dimensionality::<ndarray::Ix4>()
                                            .unwrap();

                                        Zip::from(out_view.outer_iter_mut())
                                            .and(x_view.outer_iter())
                                            .par_for_each(|mut out_b, x_b| {
                                                Zip::from(out_b.outer_iter_mut())
                                                    .and(x_b.outer_iter())
                                                    .for_each(|mut out_h, x_h| {
                                                        let half = d / 2;
                                                        for ss in 0..seq_len {
                                                            for j in 0..half {
                                                                let x1 = x_h[[ss, j]].to_f32();
                                                                let x2 =
                                                                    x_h[[ss, j + half]].to_f32();
                                                                let c = cos_slice_2d[[ss, j]];
                                                                let s_val = sin_slice_2d[[ss, j]];
                                                                out_h[[ss, j]] = f16::from_f32(
                                                                    x1 * c - x2 * s_val,
                                                                );
                                                                out_h[[ss, j + half]] =
                                                                    f16::from_f32(
                                                                        x2 * c + x1 * s_val,
                                                                    );
                                                            }
                                                        }
                                                    });
                                            });

                                        Tensor::from_shared_f16_no_grad_with_device(
                                            out.into_shared(),
                                            output_device,
                                        )
                                    }
                                    TensorStorageView::BF16(x_view) => {
                                        let mut out = ndarray::ArrayD::<bf16>::from_elem(
                                            ndarray::IxDyn(&shape),
                                            bf16::from_bits(0),
                                        );
                                        let x_view =
                                            x_view.into_dimensionality::<ndarray::Ix4>().unwrap();
                                        let mut out_view = out
                                            .view_mut()
                                            .into_dimensionality::<ndarray::Ix4>()
                                            .unwrap();

                                        Zip::from(out_view.outer_iter_mut())
                                            .and(x_view.outer_iter())
                                            .par_for_each(|mut out_b, x_b| {
                                                Zip::from(out_b.outer_iter_mut())
                                                    .and(x_b.outer_iter())
                                                    .for_each(|mut out_h, x_h| {
                                                        let half = d / 2;
                                                        for ss in 0..seq_len {
                                                            for j in 0..half {
                                                                let x1 = x_h[[ss, j]].to_f32();
                                                                let x2 =
                                                                    x_h[[ss, j + half]].to_f32();
                                                                let c = cos_slice_2d[[ss, j]];
                                                                let s_val = sin_slice_2d[[ss, j]];
                                                                out_h[[ss, j]] = bf16::from_f32(
                                                                    x1 * c - x2 * s_val,
                                                                );
                                                                out_h[[ss, j + half]] =
                                                                    bf16::from_f32(
                                                                        x2 * c + x1 * s_val,
                                                                    );
                                                            }
                                                        }
                                                    });
                                            });

                                        Tensor::from_shared_bf16_no_grad_with_device(
                                            out.into_shared(),
                                            output_device,
                                        )
                                    }
                                }
                            },
                        )
                    },
                )
            });
        }

        if output_device == Device::Cuda && !x.is_empty() {
            let shape = x.shape_vec();
            assert_eq!(shape.len(), 4, "RoPE expects input [B,H,S,D]");
            let (b, h, seq_len, d) = (shape[0], shape[1], shape[2], shape[3]);
            assert_eq!(d, self.dim, "RoPE dimension mismatch");
            let end = offset + seq_len;
            if end > self.max_seq_len {
                panic!(
                    "RoPE index out of range: offset {} + len {} > max {}",
                    offset, seq_len, self.max_seq_len
                );
            }

            let cuda_out = if let (
                Some((x_dtype, x_buf, x_scale)),
                Some((cos_dtype, cos_buf, cos_scale)),
                Some((sin_dtype, sin_buf, sin_scale)),
            ) = (
                x.cloned_cuda_native_lowp_buffer(),
                cuda_rope_storage_buffer(&self.cos_cache),
                cuda_rope_storage_buffer(&self.sin_cache),
            ) {
                if cos_dtype == sin_dtype {
                    cuda::rope_typed_buffer(
                        &x_buf,
                        x_dtype,
                        x_scale,
                        &cos_buf,
                        &sin_buf,
                        cos_dtype,
                        cos_scale,
                        sin_scale,
                        b,
                        h,
                        seq_len,
                        d,
                        offset,
                        self.max_seq_len,
                    )
                } else {
                    x.with_cuda_f32_buffer(|x_buf| {
                        self.cos_cache.with_cuda_f32_buffer(|cos_buf| {
                            self.sin_cache.with_cuda_f32_buffer(|sin_buf| {
                                cuda::rope_f32_buffer(
                                    x_buf,
                                    cos_buf,
                                    sin_buf,
                                    b,
                                    h,
                                    seq_len,
                                    d,
                                    offset,
                                    self.max_seq_len,
                                )
                            })
                        })
                    })
                }
            } else {
                x.with_cuda_f32_buffer(|x_buf| {
                    self.cos_cache.with_cuda_f32_buffer(|cos_buf| {
                        self.sin_cache.with_cuda_f32_buffer(|sin_buf| {
                            cuda::rope_f32_buffer(
                                x_buf,
                                cos_buf,
                                sin_buf,
                                b,
                                h,
                                seq_len,
                                d,
                                offset,
                                self.max_seq_len,
                            )
                        })
                    })
                })
            };
            if let Ok(buffer) = cuda_out {
                let x_clone = x.clone();
                let cos_cache = self.cos_cache.clone();
                let sin_cache = self.sin_cache.clone();
                let max_seq_len = self.max_seq_len;
                let output_self = Rc::new(RefCell::new(None::<Tensor>));
                let output_self_for_backward = output_self.clone();

                let tensor = Tensor(Rc::new(RefCell::new(TensorData {
                    data: Array::zeros(ndarray::IxDyn(&shape)).into_shared(),
                    f16_data: None,
                    bf16_data: None,
                    i8_data: None,
                    cuda_f32_data: Some(buffer),
                    cuda_f16_data: None,
                    cuda_bf16_data: None,
                    cuda_i8_data: None,
                    i8_scale: None,
                    has_f32_data: false,
                    storage_dtype: crate::precision::DType::F32,
                    cache_dirty: false,
                    is_parameter: false,
                    grad: None,
                    cuda_f32_grad: None,
                    parents: vec![x.clone()],
                    backward_op: Some(std::rc::Rc::new(move |grad| {
                        let cuda_grad = if let Some(grad_buf) = output_self_for_backward
                            .borrow()
                            .as_ref()
                            .and_then(|output| output.cloned_cuda_f32_grad())
                            .filter(|grad_buf| grad_buf.len() == grad.len())
                        {
                            if is_strict_device_execution() {
                                match cos_cache.with_cuda_f32_buffer(|cos_buf| {
                                    sin_cache.with_cuda_f32_buffer(|sin_buf| {
                                        cuda::rope_backward_f32_buffer(
                                            &grad_buf,
                                            cos_buf,
                                            sin_buf,
                                            b,
                                            h,
                                            seq_len,
                                            d,
                                            offset,
                                            max_seq_len,
                                        )
                                    })
                                }) {
                                    Ok(grad_buffer) => {
                                        x_clone.add_cuda_grad_buffer_only(grad_buffer);
                                        return;
                                    }
                                    Err(err) => {
                                        panic!("CUDA RoPE backward failed: {err}");
                                    }
                                }
                            }
                            cos_cache.with_cuda_f32_buffer(|cos_buf| {
                                sin_cache.with_cuda_f32_buffer(|sin_buf| {
                                    cuda::rope_backward_f32(
                                        &grad_buf,
                                        cos_buf,
                                        sin_buf,
                                        b,
                                        h,
                                        seq_len,
                                        d,
                                        offset,
                                        max_seq_len,
                                    )
                                })
                            })
                        } else {
                            let grad_host = grad.iter().copied().collect::<Vec<_>>();
                            if is_strict_device_execution() {
                                match cuda::upload_f32(&grad_host).and_then(|grad_buf| {
                                    cos_cache.with_cuda_f32_buffer(|cos_buf| {
                                        sin_cache.with_cuda_f32_buffer(|sin_buf| {
                                            cuda::rope_backward_f32_buffer(
                                                &grad_buf,
                                                cos_buf,
                                                sin_buf,
                                                b,
                                                h,
                                                seq_len,
                                                d,
                                                offset,
                                                max_seq_len,
                                            )
                                        })
                                    })
                                }) {
                                    Ok(grad_buffer) => {
                                        x_clone.add_cuda_grad_buffer_only(grad_buffer);
                                        return;
                                    }
                                    Err(err) => {
                                        panic!("CUDA RoPE backward failed: {err}");
                                    }
                                }
                            }
                            cuda::upload_f32(&grad_host).and_then(|grad_buf| {
                                cos_cache.with_cuda_f32_buffer(|cos_buf| {
                                    sin_cache.with_cuda_f32_buffer(|sin_buf| {
                                        cuda::rope_backward_f32(
                                            &grad_buf,
                                            cos_buf,
                                            sin_buf,
                                            b,
                                            h,
                                            seq_len,
                                            d,
                                            offset,
                                            max_seq_len,
                                        )
                                    })
                                })
                            })
                        };
                        match cuda_grad {
                            Ok((grad_buffer, grad_host)) => {
                                let d_x = Array::from_shape_vec(ndarray::IxDyn(&shape), grad_host)
                                    .expect("CUDA RoPE grad shape build failed")
                                    .into_dyn();
                                x_clone.add_grad_with_cuda_buffer(d_x, Some(grad_buffer));
                            }
                            Err(err) => {
                                panic!("CUDA RoPE backward failed: {err}");
                            }
                        }
                    })),
                    requires_grad: true,
                    device: output_device,
                })));
                *output_self.borrow_mut() = Some(tensor.clone());
                return tensor;
            }
        }

        let x_data = x.data_ref();
        let shape = x_data.shape();
        assert_eq!(shape.len(), 4, "RoPE expects input [B,H,S,D]");
        let (b, h, seq_len, d) = (shape[0], shape[1], shape[2], shape[3]);
        assert_eq!(d, self.dim, "RoPE dimension mismatch");

        let end = offset + seq_len;
        if end > self.max_seq_len {
            panic!(
                "RoPE index out of range: offset {} + len {} > max {}",
                offset, seq_len, self.max_seq_len
            );
        }

        self.cos_cache
            .with_storage_view_preferring(StoragePreference::F32Compute, |cos_view| {
                self.sin_cache.with_storage_view_preferring(
                    StoragePreference::F32Compute,
                    |sin_view| {
                        let cos_4d = match cos_view {
                            TensorStorageView::F32(view) => view
                                .into_dimensionality::<ndarray::Ix4>()
                                .expect("K cache update expects [B,H,S,D]"),
                            TensorStorageView::F16(_) => {
                                unreachable!("f32 compute preference should expose f32 view")
                            }
                            TensorStorageView::BF16(_) => {
                                unreachable!("f32 compute preference should expose f32 view")
                            }
                        };

                        let sin_4d = match sin_view {
                            TensorStorageView::F32(view) => view
                                .into_dimensionality::<ndarray::Ix4>()
                                .expect("K cache update expects [B,H,S,D]"),
                            TensorStorageView::F16(_) => {
                                unreachable!("f32 compute preference should expose f32 view")
                            }
                            TensorStorageView::BF16(_) => {
                                unreachable!("f32 compute preference should expose f32 view")
                            }
                        };
                        let cos_slice_2d = cos_4d
                            .slice(s![0, 0, offset..end, ..])
                            .into_dimensionality::<Ix2>()
                            .expect("RoPE Cache dimensionality mismatch");
                        let sin_slice_2d = sin_4d
                            .slice(s![0, 0, offset..end, ..])
                            .into_dimensionality::<Ix2>()
                            .expect("RoPE Cache dimensionality mismatch");

                        let mut out = Array::zeros(x_data.dim());

                        let x_view = x_data.view().into_dimensionality::<ndarray::Ix4>().unwrap();
                        let mut out_view = out
                            .view_mut()
                            .into_dimensionality::<ndarray::Ix4>()
                            .unwrap();

                        Zip::from(out_view.outer_iter_mut())
                            .and(x_view.outer_iter())
                            .par_for_each(|mut out_b, x_b| {
                                Zip::from(out_b.outer_iter_mut())
                                    .and(x_b.outer_iter())
                                    .for_each(|mut out_h, x_h| {
                                        let half = d / 2;
                                        for ss in 0..seq_len {
                                            for j in 0..half {
                                                let x1 = x_h[[ss, j]];
                                                let x2 = x_h[[ss, j + half]];
                                                let c = cos_slice_2d[[ss, j]];
                                                let s_val = sin_slice_2d[[ss, j]];
                                                out_h[[ss, j]] = x1 * c - x2 * s_val;
                                                out_h[[ss, j + half]] = x2 * c + x1 * s_val;
                                            }
                                        }
                                    });
                            });

                        let x_clone = x.clone();
                        let cos_backward = cos_slice_2d.to_owned();
                        let sin_backward = sin_slice_2d.to_owned();

                        Tensor(Rc::new(RefCell::new(TensorData {
                            data: out.into_dyn().into_shared(),
                            f16_data: None,
                            bf16_data: None,
                            i8_data: None,
                            cuda_f32_data: None,
                            cuda_f16_data: None,
                            cuda_bf16_data: None,
                            cuda_i8_data: None,
                            i8_scale: None,
                            has_f32_data: true,
                            storage_dtype: crate::precision::DType::F32,
                            cache_dirty: false,
                            is_parameter: false,
                            grad: None,
                            cuda_f32_grad: None,
                            parents: vec![x.clone()],
                            backward_op: Some(std::rc::Rc::new(move |grad| {
                                let grad_view =
                                    grad.view().into_dimensionality::<ndarray::Ix4>().unwrap();
                                let mut d_x = Array::zeros((b, h, seq_len, d));

                                Zip::from(d_x.outer_iter_mut())
                                    .and(grad_view.outer_iter())
                                    .par_for_each(|mut dx_b, g_b| {
                                        Zip::from(dx_b.outer_iter_mut())
                                            .and(g_b.outer_iter())
                                            .for_each(|mut dx_h, g_h| {
                                                let half = d / 2;
                                                for ss in 0..seq_len {
                                                    for j in 0..half {
                                                        let g1 = g_h[[ss, j]];
                                                        let g2 = g_h[[ss, j + half]];

                                                        let c = cos_backward[[ss, j]];
                                                        let s_val = sin_backward[[ss, j]];

                                                        dx_h[[ss, j]] = g1 * c + g2 * s_val;
                                                        dx_h[[ss, j + half]] = g2 * c - g1 * s_val;
                                                    }
                                                }
                                            });
                                    });

                                x_clone.add_grad(d_x.into_dyn());
                            })),
                            requires_grad: true,
                            device: output_device,
                        })))
                    },
                )
            })
    }

    // Apply RoPE for a single token at absolute position `pos`.
    //
    // Decode (S=1) hot-path helper to avoid allocating intermediate q_rot/k_rot tensors.
    // `src` and `dst` must both have length == `self.dim`.
    #[inline]
    pub fn rope_1token_copy(&self, src: &[f32], dst: &mut [f32], pos: usize) {
        assert_eq!(src.len(), self.dim, "RoPE src len mismatch");
        assert_eq!(dst.len(), self.dim, "RoPE dst len mismatch");
        if pos >= self.max_seq_len {
            panic!(
                "RoPE index out of range: pos {} >= max {}",
                pos, self.max_seq_len
            );
        }

        self.cos_cache
            .with_storage_view_preferring(StoragePreference::F32Compute, |cos_view| {
                self.sin_cache.with_storage_view_preferring(
                    StoragePreference::F32Compute,
                    |sin_view| {
                        let cos_row = match cos_view {
                            TensorStorageView::F32(view) => {
                                let cache4 = view
                                    .into_dimensionality::<ndarray::Ix4>()
                                    .expect("RoPE Cache dimensionality mismatch");
                                cache4.slice(s![0, 0, pos, ..]).to_owned().into_raw_vec()
                            }
                            TensorStorageView::F16(_) => {
                                unreachable!("f32 compute preference should expose f32 view")
                            }
                            TensorStorageView::BF16(_) => {
                                unreachable!("f32 compute preference should expose f32 view")
                            }
                        };
                        let sin_row = match sin_view {
                            TensorStorageView::F32(view) => {
                                let cache4 = view
                                    .into_dimensionality::<ndarray::Ix4>()
                                    .expect("RoPE Cache dimensionality mismatch");
                                cache4.slice(s![0, 0, pos, ..]).to_owned().into_raw_vec()
                            }
                            TensorStorageView::F16(_) => {
                                unreachable!("f32 compute preference should expose f32 view")
                            }
                            TensorStorageView::BF16(_) => {
                                unreachable!("f32 compute preference should expose f32 view")
                            }
                        };

                        let half = self.dim / 2;
                        for j in 0..half {
                            let x1 = src[j];
                            let x2 = src[j + half];
                            let c = cos_row[j];
                            let s_val = sin_row[j];
                            dst[j] = x1 * c - x2 * s_val;
                            dst[j + half] = x2 * c + x1 * s_val;
                        }
                    },
                )
            });
    }

    // Get (cos, sin) row at position `pos` as owned Vecs.
    // This is useful to pass into rayon-parallel decode kernels without capturing Tensor/Rc.
    pub fn cos_sin_row_vec(&self, pos: usize) -> (Vec<f32>, Vec<f32>) {
        if pos >= self.max_seq_len {
            panic!(
                "RoPE index out of range: pos {} >= max {}",
                pos, self.max_seq_len
            );
        }
        self.cos_cache
            .with_storage_view_preferring(StoragePreference::F32Compute, |cos_view| {
                self.sin_cache.with_storage_view_preferring(
                    StoragePreference::F32Compute,
                    |sin_view| {
                        let cos_row = match cos_view {
                            TensorStorageView::F32(view) => {
                                let cache4 = view
                                    .into_dimensionality::<ndarray::Ix4>()
                                    .expect("RoPE Cache dimensionality mismatch");
                                cache4.slice(s![0, 0, pos, ..]).to_owned().into_raw_vec()
                            }
                            TensorStorageView::F16(_) => {
                                unreachable!("f32 compute preference should expose f32 view")
                            }
                            TensorStorageView::BF16(_) => {
                                unreachable!("f32 compute preference should expose f32 view")
                            }
                        };
                        let sin_row = match sin_view {
                            TensorStorageView::F32(view) => {
                                let cache4 = view
                                    .into_dimensionality::<ndarray::Ix4>()
                                    .expect("RoPE Cache dimensionality mismatch");
                                cache4.slice(s![0, 0, pos, ..]).to_owned().into_raw_vec()
                            }
                            TensorStorageView::F16(_) => {
                                unreachable!("f32 compute preference should expose f32 view")
                            }
                            TensorStorageView::BF16(_) => {
                                unreachable!("f32 compute preference should expose f32 view")
                            }
                        };

                        (cos_row, sin_row)
                    },
                )
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::no_grad;
    #[cfg(feature = "cuda")]
    use crate::autograd::set_strict_device_execution_scoped;
    use crate::precision::{PrecisionConfig, with_precision_config};
    use ndarray::{Array, IxDyn};
    #[cfg(feature = "cuda")]
    use std::sync::{Mutex, MutexGuard, OnceLock};

    #[cfg(feature = "cuda")]
    fn cuda_strict_test_guard() -> MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
            .lock()
            .expect("cuda strict test mutex poisoned")
    }

    fn make_tensor(shape: &[usize], data: Vec<f32>, dtype: DType) -> Tensor {
        let t = Tensor::from_array_no_grad(
            Array::from_shape_vec(IxDyn(shape), data)
                .expect("test tensor shape mismatch")
                .into_dyn(),
        );
        t.cast_inplace(dtype);
        t
    }

    #[test]
    fn rope_no_grad_preserves_bf16_input_dtype() {
        let rope = RotaryEmbedding::new(4, 8, 10000.0);
        let input_f32 = make_tensor(
            &[1, 1, 2, 4],
            vec![1.0, 2.0, 3.0, 4.0, -1.0, 0.5, 2.5, -3.0],
            DType::F32,
        );
        let input_bf16 = make_tensor(
            &[1, 1, 2, 4],
            vec![1.0, 2.0, 3.0, 4.0, -1.0, 0.5, 2.5, -3.0],
            DType::BF16,
        );

        let ref_out = no_grad(|| rope.forward(&input_f32, 0));
        let bf16_out = no_grad(|| rope.forward(&input_bf16, 0));

        assert_eq!(input_bf16.dtype(), DType::BF16);
        assert_eq!(bf16_out.dtype(), DType::BF16);

        let ref_vals = ref_out
            .data_ref()
            .iter()
            .map(|&v| bf16::from_f32(v).to_f32())
            .collect::<Vec<_>>();
        bf16_out.with_storage_view(|view| match view {
            TensorStorageView::BF16(view) => {
                let vals = view.iter().map(|v| v.to_f32()).collect::<Vec<_>>();
                assert_eq!(vals, ref_vals);
            }
            TensorStorageView::F16(_) => panic!("bf16 RoPE output should stay bf16 in no-grad"),
            TensorStorageView::F32(_) => panic!("bf16 RoPE output should stay bf16 in no-grad"),
        });
    }

    #[test]
    fn rope_cache_creation_follows_runtime_dtype() {
        with_precision_config(
            PrecisionConfig {
                parameter_dtype: DType::F32,
                runtime_dtype: DType::BF16,
                allow_parameter_dtype_copies: false,
            },
            || {
                let rope = RotaryEmbedding::new(4, 8, 10000.0);
                assert_eq!(rope.cos_cache.dtype(), DType::BF16);
                assert_eq!(rope.sin_cache.dtype(), DType::BF16);
            },
        );
    }

    #[test]
    fn rope_explicit_dtype_overrides_global_default() {
        with_precision_config(
            PrecisionConfig {
                parameter_dtype: DType::F32,
                runtime_dtype: DType::BF16,
                allow_parameter_dtype_copies: false,
            },
            || {
                let rope = RotaryEmbedding::new_with_dtype(4, 8, 10000.0, DType::F32);
                assert_eq!(rope.cos_cache.dtype(), DType::F32);
                assert_eq!(rope.sin_cache.dtype(), DType::F32);
            },
        );
    }

    #[test]
    fn rope_no_grad_preserves_i8_input_dtype() {
        let rope = RotaryEmbedding::new(4, 8, 10000.0);
        let input_i8 = make_tensor(
            &[1, 1, 2, 4],
            vec![1.0, 2.0, 3.0, 4.0, -1.0, 0.5, 2.5, -3.0],
            DType::I8,
        );

        let out = no_grad(|| rope.forward(&input_i8, 0));
        assert_eq!(out.dtype(), DType::I8);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_rope_matches_cpu_reference_in_strict_mode() {
        if !crate::ops::cuda::is_available() {
            return;
        }
        let _test_guard = cuda_strict_test_guard();

        let rope = RotaryEmbedding::new_with_dtype(4, 8, 10000.0, DType::BF16);
        let rope_ref = RotaryEmbedding::new_with_dtype(4, 8, 10000.0, DType::BF16);
        rope.to_cuda();
        let input = make_tensor(
            &[1, 2, 2, 4],
            vec![
                1.0, 2.0, 3.0, 4.0, -1.0, 0.5, 2.5, -3.0, 0.25, -0.75, 1.5, -2.0, 3.0, -1.5, 0.5,
                2.25,
            ],
            DType::BF16,
        )
        .to_cuda();
        assert!(input.0.borrow().cuda_f32_data.is_none());
        assert!(rope.cos_cache.0.borrow().cuda_f32_data.is_none());
        assert!(rope.sin_cache.0.borrow().cuda_f32_data.is_none());

        let cuda_out = {
            let _cuda_guard = crate::ops::cuda::set_enabled_scoped(true);
            let _strict_guard = set_strict_device_execution_scoped(true);
            let cuda_out = no_grad(|| rope.forward(&input, 1));
            assert!(input.0.borrow().cuda_f32_data.is_none());
            assert!(rope.cos_cache.0.borrow().cuda_f32_data.is_none());
            assert!(rope.sin_cache.0.borrow().cuda_f32_data.is_none());
            cuda_out
        };

        let cpu_out = no_grad(|| rope_ref.forward(&input.to_cpu(), 1));
        assert!(cuda_out.is_cuda());
        assert_eq!(cuda_out.dtype(), DType::BF16);
        {
            let inner = cuda_out.0.borrow();
            assert!(
                !inner.has_f32_data,
                "CUDA RoPE BF16 no-grad output should not eagerly materialize host f32 data"
            );
            assert!(
                inner.cuda_f32_data.is_none(),
                "CUDA RoPE BF16 no-grad output should not retain only an f32 compute buffer"
            );
            assert!(
                inner.cuda_bf16_data.is_some(),
                "CUDA RoPE BF16 no-grad output should keep resident bf16 storage"
            );
        }

        for (got, expect) in cuda_out.data_ref().iter().zip(cpu_out.data_ref().iter()) {
            assert!((got - expect).abs() < 2e-2, "got {got}, expect {expect}");
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_empty_rope_no_grad_stays_on_cuda_in_strict_mode() {
        if !crate::ops::cuda::is_available() {
            return;
        }
        let _test_guard = cuda_strict_test_guard();

        let rope = RotaryEmbedding::new_with_dtype(4, 8, 10000.0, DType::BF16);
        rope.to_cuda();
        let input = make_tensor(&[1, 2, 0, 4], vec![], DType::BF16).to_cuda();

        let out = {
            let _cuda_guard = crate::ops::cuda::set_enabled_scoped(true);
            let _strict_guard = set_strict_device_execution_scoped(true);
            no_grad(|| rope.forward(&input, 3))
        };

        assert!(out.is_cuda());
        assert_eq!(out.dtype(), DType::BF16);
        assert_eq!(out.shape_vec(), vec![1, 2, 0, 4]);
        assert_eq!(out.len(), 0);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_rope_backward_matches_cpu_reference_in_strict_mode() {
        if !crate::ops::cuda::is_available() {
            return;
        }
        let _test_guard = cuda_strict_test_guard();

        let rope = RotaryEmbedding::new_with_dtype(4, 8, 10000.0, DType::F32);
        let rope_ref = RotaryEmbedding::new_with_dtype(4, 8, 10000.0, DType::F32);
        rope.to_cuda();

        let input_values = vec![
            1.0, 2.0, 3.0, 4.0, -1.0, 0.5, 2.5, -3.0, 0.25, -0.75, 1.5, -2.0, 3.0, -1.5, 0.5, 2.25,
        ];
        let coeff_values = vec![
            0.5, -1.0, 2.0, 0.25, -0.75, 1.5, -0.5, 0.75, 0.1, 0.2, -0.3, 0.4, 1.25, -1.5, 0.6,
            -0.9,
        ];
        let input_cpu = Tensor::from_data_with_grad_flag(
            Array::from_shape_vec(IxDyn(&[1, 2, 2, 4]), input_values)
                .expect("input shape mismatch")
                .into_dyn(),
            true,
        );
        let coeff_cpu = Tensor::from_data_with_grad_flag(
            Array::from_shape_vec(IxDyn(&[1, 2, 2, 4]), coeff_values)
                .expect("coeff shape mismatch")
                .into_dyn(),
            false,
        );
        let input_cuda = input_cpu.to_cuda();
        let coeff_cuda = coeff_cpu.to_cuda();

        {
            let _cuda_guard = crate::ops::cuda::set_enabled_scoped(true);
            let _strict_guard = set_strict_device_execution_scoped(true);
            let cuda_out = rope.forward(&input_cuda, 1);
            let cuda_loss = crate::ops::arithmetic::sum(&(&cuda_out * &coeff_cuda));
            cuda_loss.backward();
            assert!(input_cuda.cloned_cuda_f32_grad().is_some());
            assert!(!input_cuda.has_host_grad());
        }

        let cpu_out = rope_ref.forward(&input_cpu, 1);
        let cpu_loss = crate::ops::arithmetic::sum(&(&cpu_out * &coeff_cpu));
        cpu_loss.backward();

        let cuda_grad = input_cuda.grad().expect("cuda RoPE input grad");
        let cpu_grad = input_cpu.grad().expect("cpu RoPE input grad");
        for (got, expect) in cuda_grad.iter().zip(cpu_grad.iter()) {
            assert!((got - expect).abs() < 1e-5, "got {got}, expect {expect}");
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_rope_preserves_i8_dtype_in_strict_mode() {
        if !crate::ops::cuda::is_available() {
            return;
        }
        let _test_guard = cuda_strict_test_guard();

        let rope = RotaryEmbedding::new_with_dtype(4, 8, 10000.0, DType::BF16);
        let rope_ref = RotaryEmbedding::new_with_dtype(4, 8, 10000.0, DType::BF16);
        rope.to_cuda();
        let input_values = vec![1.0, 2.0, 3.0, 4.0, -1.0, 0.5, 2.5, -3.0];
        let input = make_tensor(&[1, 1, 2, 4], input_values.clone(), DType::I8).to_cuda();
        assert!(input.0.borrow().cuda_f32_data.is_none());

        let out = {
            let _cuda_guard = crate::ops::cuda::set_enabled_scoped(true);
            let _strict_guard = set_strict_device_execution_scoped(true);
            no_grad(|| rope.forward(&input, 0))
        };

        assert!(out.is_cuda());
        assert_eq!(out.dtype(), DType::I8);
        {
            let inner = out.0.borrow();
            assert!(
                !inner.has_f32_data,
                "CUDA RoPE I8 no-grad output should not eagerly materialize host f32 data"
            );
            assert!(
                inner.cuda_f32_data.is_none(),
                "CUDA RoPE I8 no-grad output should not retain only an f32 compute buffer"
            );
            assert!(
                inner.cuda_i8_data.is_some(),
                "CUDA RoPE I8 no-grad output should keep resident i8 storage"
            );
            assert!(inner.i8_scale.is_some());
        }

        let cpu_input = make_tensor(&[1, 1, 2, 4], input_values, DType::I8);
        let cpu_out = no_grad(|| rope_ref.forward(&cpu_input, 0));
        for (got, expect) in out.data_ref().iter().zip(cpu_out.data_ref().iter()) {
            assert!((got - expect).abs() < 1e-5, "got {got}, expect {expect}");
        }
    }
}
