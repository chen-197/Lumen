use crate::autograd::{ArcArray, IxDyn, Tensor, TensorData, TensorStorageOwned, is_no_grad};
use crate::device::{assert_native_device_support, is_strict_device_execution};
use crate::ops::cuda;
use crate::precision::DType;
use ndarray::Array;
use ndarray::{Axis, Slice};
use std::cell::RefCell;
use std::rc::Rc;

// 说明：Tensor 的 data 现在是 ArcArray（共享底层 + stride）。
// 因此 reshape/permute 在 stride 兼容时可以做到零拷贝（只改元数据）。

fn reshape_shared<T: Clone>(data: ArcArray<T, IxDyn>, new_shape: &[usize]) -> ArcArray<T, IxDyn> {
    if data.shape() == new_shape {
        return data;
    }
    // ndarray can reshape column-major storage in physical order. Tensor
    // reshape instead preserves the row-major sequence of logical elements.
    if data.is_standard_layout() {
        return data
            .into_shape(new_shape.to_vec())
            .expect("Reshape failed: Total element count mismatch")
            .into_dyn();
    }
    data.as_standard_layout()
        .into_owned()
        .into_shape(new_shape.to_vec())
        .expect("Reshape failed: Total element count mismatch")
        .into_dyn()
        .into_shared()
}

fn validate_permute_axes(ndim: usize, axes: &[usize]) {
    assert_eq!(
        axes.len(),
        ndim,
        "Permute axes length mismatch: got {}, expected {}",
        axes.len(),
        ndim
    );
    let mut seen = vec![false; ndim];
    for (i, &axis) in axes.iter().enumerate() {
        assert!(
            axis < ndim,
            "Permute axis {} out of bounds for ndim {}",
            axis,
            ndim
        );
        assert!(
            !seen[axis],
            "Permute axes must be unique, duplicate axis {} at position {}",
            axis, i
        );
        seen[axis] = true;
    }
}

fn validate_cat_inputs(tensors: &[Tensor], axis: usize) -> Vec<Vec<usize>> {
    let shapes = tensors.iter().map(Tensor::shape_vec).collect::<Vec<_>>();
    let ndim = shapes[0].len();
    assert!(
        axis < ndim,
        "Concat axis {} out of bounds for ndim {}",
        axis,
        ndim
    );
    for shape in shapes.iter().skip(1) {
        assert_eq!(
            shape.len(),
            ndim,
            "Concat ndim mismatch: expected {}, got {}",
            ndim,
            shape.len()
        );
    }
    for dim in 0..ndim {
        if dim == axis {
            continue;
        }
        let expected = shapes[0][dim];
        for shape in shapes.iter().skip(1) {
            assert_eq!(
                shape[dim], expected,
                "Concat shape mismatch on dim {}: expected {}, got {}",
                dim, expected, shape[dim]
            );
        }
    }
    shapes
}

fn cat_output_shape(shapes: &[Vec<usize>], axis: usize) -> Vec<usize> {
    let mut out_shape = shapes[0].clone();
    out_shape[axis] = shapes
        .iter()
        .try_fold(0usize, |sum, shape| sum.checked_add(shape[axis]))
        .expect("Concat axis length overflow");
    out_shape
}

fn preserve_device_after_view(
    tensor: Tensor,
    output_device: crate::device::Device,
    source: &Tensor,
) -> Tensor {
    match output_device {
        crate::device::Device::Cpu => tensor,
        crate::device::Device::Cuda => {
            if let Some((dtype, buffer, scale)) = source.cloned_cuda_native_lowp_buffer() {
                let shape = tensor.shape_vec();
                return Tensor::from_cuda_native_lowp_buffer_no_host_with_dtype(
                    &shape,
                    buffer,
                    output_device,
                    dtype,
                    scale,
                );
            } else if let Some(buffer) = source.cloned_cuda_f32_buffer() {
                tensor.set_cuda_f32_buffer_inplace(buffer);
            } else {
                tensor.to_cuda_inplace();
            }
            tensor
        }
    }
}

fn reverse_permute_axes(axes: &[usize]) -> Vec<usize> {
    let mut rev_axes = vec![0; axes.len()];
    for (i, &ax) in axes.iter().enumerate() {
        rev_axes[ax] = i;
    }
    rev_axes
}

fn try_cuda_permute_graph(
    input: &Tensor,
    axes: &[usize],
    input_shape: &[usize],
    out_shape: &[usize],
) -> Option<Tensor> {
    let cuda_buffer = input
        .cloned_cuda_f32_buffer()
        .and_then(|input_buf| cuda::permute_f32_buffer(&input_buf, out_shape, axes).ok())?;
    let output_self = Rc::new(RefCell::new(std::rc::Weak::<RefCell<TensorData>>::new()));
    let output_self_for_backward = output_self.clone();
    let input_clone = input.clone();
    let input_shape_for_backward = input_shape.to_vec();
    let rev_axes_for_backward = reverse_permute_axes(axes);
    let tensor = Tensor(Rc::new(RefCell::new(TensorData {
        data: Array::zeros(IxDyn(out_shape)).into_shared(),
        f16_data: None,
        bf16_data: None,
        i8_data: None,
        cuda_f32_data: Some(cuda_buffer),
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
        parents: vec![input.clone()],
        backward_op: Some(std::rc::Rc::new(move |grad| {
            if input_clone.is_cuda() {
                let cuda_result = if let Some(grad_buffer) = output_self_for_backward
                    .borrow()
                    .upgrade()
                    .map(Tensor)
                    .and_then(|output| output.cloned_cuda_f32_grad())
                    .filter(|grad_buffer| grad_buffer.len() == grad.len())
                {
                    if is_strict_device_execution() {
                        match cuda::permute_f32_buffer(
                            &grad_buffer,
                            &input_shape_for_backward,
                            &rev_axes_for_backward,
                        ) {
                            Ok(grad_buffer) => {
                                input_clone.add_backward_cuda_grad_buffer_only(grad_buffer);
                                return;
                            }
                            Err(err) => {
                                panic!(
                                    "permute CUDA backward failed while strict device execution is enabled: {err}"
                                );
                            }
                        }
                    }
                    cuda::permute_f32(
                        &grad_buffer,
                        &input_shape_for_backward,
                        &rev_axes_for_backward,
                    )
                } else {
                    let grad_host = grad.iter().copied().collect::<Vec<_>>();
                    cuda::upload_f32(&grad_host).and_then(|grad_buf| {
                        cuda::permute_f32(
                            &grad_buf,
                            &input_shape_for_backward,
                            &rev_axes_for_backward,
                        )
                    })
                };
                match cuda_result {
                    Ok((grad_buffer, grad_restored_host)) => {
                        let grad_restored = Array::from_shape_vec(
                            IxDyn(&input_shape_for_backward),
                            grad_restored_host,
                        )
                        .expect("CUDA permute grad shape build failed")
                        .into_dyn();
                        input_clone
                            .add_backward_grad_with_cuda_buffer(grad_restored, Some(grad_buffer));
                        return;
                    }
                    Err(err) => {
                        assert!(
                            !is_strict_device_execution(),
                            "permute CUDA backward failed while strict device execution is enabled: {err}"
                        );
                    }
                }
            }
            let grad_restored = grad
                .view()
                .permuted_axes(rev_axes_for_backward.clone())
                .to_owned();
            input_clone.add_backward_grad(grad_restored);
        })),
        requires_grad: true,
        device: crate::device::Device::Cuda,
    })));
    *output_self.borrow_mut() = Rc::downgrade(&tensor.0);
    Some(tensor)
}

pub fn reshape(input: &Tensor, shape: Vec<i32>) -> Tensor {
    let output_device = input.device();
    assert!(!shape.is_empty(), "Reshape expects at least one dimension");

    let input_len = input.len();
    let mut infer_axis = None;
    let mut known_product = 1usize;
    let mut new_shape = Vec::with_capacity(shape.len());

    for (axis, &dim) in shape.iter().enumerate() {
        match dim {
            -1 => {
                assert!(
                    infer_axis.is_none(),
                    "Reshape only supports one inferred dimension (-1)"
                );
                infer_axis = Some(axis);
                new_shape.push(0);
            }
            d if d >= 0 => {
                let dim_usize = d as usize;
                known_product = known_product
                    .checked_mul(dim_usize)
                    .expect("Reshape dimension product overflow");
                new_shape.push(dim_usize);
            }
            _ => {
                panic!(
                    "Reshape dimension at axis {} must be >= -1, got {}",
                    axis, dim
                );
            }
        }
    }

    if let Some(axis) = infer_axis {
        assert!(
            known_product > 0,
            "Reshape cannot infer dimension when known product is zero"
        );
        assert!(
            input_len.is_multiple_of(known_product),
            "Reshape inferred dimension mismatch: input elements {} not divisible by known product {}",
            input_len,
            known_product
        );
        new_shape[axis] = input_len / known_product;
    } else {
        let new_len = new_shape
            .iter()
            .try_fold(1usize, |product, &dim| product.checked_mul(dim))
            .expect("Reshape dimension product overflow");
        assert_eq!(
            new_len, input_len,
            "Reshape failed: total element count mismatch (input {}, target {})",
            input_len, new_len
        );
    }

    if is_no_grad() || !input.requires_grad() {
        if output_device == crate::device::Device::Cuda {
            if let Some((dtype, buffer, scale)) = input.cloned_cuda_native_lowp_buffer() {
                return Tensor::from_cuda_native_lowp_buffer_no_host_with_dtype(
                    &new_shape,
                    buffer,
                    output_device,
                    dtype,
                    scale,
                );
            }
            if input.dtype() == DType::F32
                && let Some(buffer) = input.cloned_cuda_f32_buffer()
            {
                return Tensor::from_cuda_f32_buffer_no_host(&new_shape, buffer, output_device);
            }
        }
        return match input.native_storage_owned() {
            TensorStorageOwned::F32(data) => preserve_device_after_view(
                Tensor::from_data_no_grad(reshape_shared(data, &new_shape)),
                output_device,
                input,
            ),
            TensorStorageOwned::F16(data) => preserve_device_after_view(
                Tensor::from_f16_data_no_grad(reshape_shared(data, &new_shape)),
                output_device,
                input,
            ),
            TensorStorageOwned::BF16(data) => preserve_device_after_view(
                Tensor::from_bf16_data_no_grad(reshape_shared(data, &new_shape)),
                output_device,
                input,
            ),
            TensorStorageOwned::I8(data, scale) => preserve_device_after_view(
                Tensor::from_i8_data_no_grad(reshape_shared(data, &new_shape), scale),
                output_device,
                input,
            ),
        };
    }

    if output_device == crate::device::Device::Cuda
        && let Some(cuda_buffer) = input.cloned_cuda_f32_buffer()
    {
        let input_clone = input.clone();
        let input_shape = input.shape_vec();
        let output_self = Rc::new(RefCell::new(std::rc::Weak::<RefCell<TensorData>>::new()));
        let output_self_for_backward = output_self.clone();
        let tensor = Tensor(Rc::new(RefCell::new(TensorData {
            data: Array::zeros(IxDyn(&new_shape)).into_shared(),
            f16_data: None,
            bf16_data: None,
            i8_data: None,
            cuda_f32_data: Some(cuda_buffer),
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
            parents: vec![input.clone()],
            backward_op: Some(std::rc::Rc::new(move |grad| {
                let upstream_cuda_grad = output_self_for_backward
                    .borrow()
                    .upgrade()
                    .map(Tensor)
                    .and_then(|output| output.cloned_cuda_f32_grad())
                    .filter(|buffer| buffer.len() == grad.len());
                if is_strict_device_execution()
                    && let Some(buffer) = upstream_cuda_grad.clone()
                {
                    input_clone.add_backward_cuda_grad_buffer_only(buffer);
                    return;
                }

                let grad_contig = grad.as_standard_layout().into_owned();
                let cuda_grad = upstream_cuda_grad.or_else(|| {
                            let grad_host = grad_contig.iter().copied().collect::<Vec<_>>();
                            match cuda::upload_f32(&grad_host) {
                                Ok(buffer) => Some(buffer),
                                Err(err) => {
                                    assert!(
                                        !is_strict_device_execution(),
                                        "reshape CUDA backward failed while strict device execution is enabled: {err}"
                                    );
                                    None
                                }
                            }
                        });
                let grad_reshaped = grad_contig
                    .into_shape(input_shape.clone())
                    .expect("Backward Reshape failed")
                    .into_dyn();
                input_clone.add_backward_grad_with_cuda_buffer(grad_reshaped, cuda_grad);
            })),
            requires_grad: true,
            device: output_device,
        })));
        *output_self.borrow_mut() = Rc::downgrade(&tensor.0);
        return tensor;
    }

    assert_native_device_support(output_device, "reshape", true);

    // clone 仅增加 refcount
    let data: ArcArray<f32, IxDyn> = input.data_arc();
    let reshaped = reshape_shared(data, &new_shape);

    let input_clone = input.clone();
    let cuda_buffer = if output_device == crate::device::Device::Cuda {
        input.cloned_cuda_f32_buffer()
    } else {
        None
    };
    Tensor(Rc::new(RefCell::new(TensorData {
        data: reshaped,
        f16_data: None,
        bf16_data: None,
        i8_data: None,
        cuda_f32_data: cuda_buffer,
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
        parents: vec![input.clone()],
        backward_op: Some(std::rc::Rc::new(move |grad| {
            let old_shape = input_clone.data_ref().shape().to_vec();
            let grad_contig = grad.as_standard_layout().into_owned();
            let cuda_grad = if input_clone.is_cuda() && !grad_contig.is_empty() {
                let grad_host = grad_contig.iter().copied().collect::<Vec<_>>();
                match cuda::upload_f32(&grad_host) {
                    Ok(buffer) => Some(buffer),
                    Err(err) => {
                        assert!(
                            !is_strict_device_execution(),
                            "reshape CUDA backward failed while strict device execution is enabled: {err}"
                        );
                        None
                    }
                }
            } else {
                None
            };
            let grad_reshaped = grad_contig
                .into_shape(old_shape)
                .expect("Backward Reshape failed")
                .into_dyn();
            input_clone.add_backward_grad_with_cuda_buffer(grad_reshaped, cuda_grad);
        })),
        requires_grad: true,
        device: output_device,
    })))
}

pub fn permute(input: &Tensor, axes: Vec<usize>) -> Tensor {
    let output_device = input.device();
    let ndim = input.ndim();
    validate_permute_axes(ndim, &axes);

    let build_graph = !is_no_grad() && input.requires_grad();
    assert_native_device_support(
        output_device,
        "permute",
        output_device == crate::device::Device::Cuda,
    );

    if !build_graph && output_device == crate::device::Device::Cuda && input.is_empty() {
        return match input.native_storage_owned() {
            TensorStorageOwned::F32(data) => Tensor::from_shared_f32_no_grad_with_device(
                data.permuted_axes(axes.clone()).into_dyn(),
                output_device,
            ),
            TensorStorageOwned::F16(data) => Tensor::from_shared_f16_no_grad_with_device(
                data.permuted_axes(axes.clone()).into_dyn(),
                output_device,
            ),
            TensorStorageOwned::BF16(data) => Tensor::from_shared_bf16_no_grad_with_device(
                data.permuted_axes(axes.clone()).into_dyn(),
                output_device,
            ),
            TensorStorageOwned::I8(data, scale) => Tensor::from_shared_i8_no_grad_with_device(
                data.permuted_axes(axes.clone()).into_dyn(),
                scale,
                output_device,
            ),
        };
    }

    if !build_graph && output_device == crate::device::Device::Cuda {
        let out_shape = axes
            .iter()
            .map(|&axis| input.shape_vec()[axis])
            .collect::<Vec<_>>();
        if let Some((dtype, input_buf, i8_scale)) = input.cloned_cuda_native_lowp_buffer() {
            match cuda::permute_typed_buffer(&input_buf, dtype, &out_shape, &axes) {
                Ok(buffer) => {
                    return Tensor::from_cuda_native_lowp_buffer_no_host_with_dtype(
                        &out_shape,
                        buffer,
                        output_device,
                        dtype,
                        i8_scale,
                    );
                }
                Err(err) => {
                    assert!(
                        !is_strict_device_execution(),
                        "CUDA typed permute forward failed while strict device execution is enabled: {err}"
                    );
                }
            }
        }
        if input.dtype() == DType::F32 {
            let cuda_out = input.with_cuda_f32_buffer(|input_buf| {
                cuda::permute_f32_buffer(input_buf, &out_shape, &axes)
            });
            if let Ok(buffer) = cuda_out {
                return Tensor::from_cuda_f32_buffer_no_host(&out_shape, buffer, output_device);
            }
        }
        let cuda_out =
            input.with_cuda_f32_buffer(|input_buf| cuda::permute_f32(input_buf, &out_shape, &axes));
        if let Ok((buffer, out)) = cuda_out {
            let out = ndarray::Array::from_shape_vec(ndarray::IxDyn(&out_shape), out)
                .expect("CUDA permute output shape build failed")
                .into_dyn();
            return Tensor::from_f32_data_no_grad_with_device_dtype_and_cuda_buffer(
                out,
                input.dtype(),
                output_device,
                Some(buffer),
            );
        }
    }

    if !build_graph {
        return match input.native_storage_owned() {
            TensorStorageOwned::F32(data) => Tensor::from_shared_f32_no_grad_with_device(
                data.permuted_axes(axes.clone()).into_dyn(),
                output_device,
            ),
            TensorStorageOwned::F16(data) => Tensor::from_shared_f16_no_grad_with_device(
                data.permuted_axes(axes.clone()).into_dyn(),
                output_device,
            ),
            TensorStorageOwned::BF16(data) => Tensor::from_shared_bf16_no_grad_with_device(
                data.permuted_axes(axes.clone()).into_dyn(),
                output_device,
            ),
            TensorStorageOwned::I8(data, scale) => Tensor::from_shared_i8_no_grad_with_device(
                data.permuted_axes(axes.clone()).into_dyn(),
                scale,
                output_device,
            ),
        };
    }

    let input_shape = input.shape_vec();
    let out_shape = axes
        .iter()
        .map(|&axis| input_shape[axis])
        .collect::<Vec<_>>();
    if output_device == crate::device::Device::Cuda
        && let Some(tensor) = try_cuda_permute_graph(input, &axes, &input_shape, &out_shape)
    {
        return tensor;
    }

    let data: ArcArray<f32, IxDyn> = input.data_arc();
    let permuted: ArcArray<f32, IxDyn> = data.permuted_axes(axes.clone()).into_dyn();

    let input_clone = input.clone();
    let cuda_buffer = if output_device == crate::device::Device::Cuda {
        input
            .cloned_cuda_f32_buffer()
            .and_then(|input_buf| cuda::permute_f32_buffer(&input_buf, &out_shape, &axes).ok())
    } else {
        None
    };
    let rev_axes = reverse_permute_axes(&axes);

    Tensor(Rc::new(RefCell::new(TensorData {
        data: permuted,
        f16_data: None,
        bf16_data: None,
        i8_data: None,
        cuda_f32_data: cuda_buffer,
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
        parents: vec![input.clone()],
        backward_op: Some(std::rc::Rc::new(move |grad| {
            if input_clone.is_cuda() && !grad.is_empty() {
                let grad_host = grad.iter().copied().collect::<Vec<_>>();
                let cuda_grad = cuda::upload_f32(&grad_host)
                    .and_then(|grad_buf| cuda::permute_f32(&grad_buf, &input_shape, &rev_axes));
                match cuda_grad {
                    Ok((grad_buffer, grad_restored_host)) => {
                        let grad_restored =
                            Array::from_shape_vec(IxDyn(&input_shape), grad_restored_host)
                                .expect("CUDA permute grad shape build failed")
                                .into_dyn();
                        input_clone
                            .add_backward_grad_with_cuda_buffer(grad_restored, Some(grad_buffer));
                        return;
                    }
                    Err(err) => {
                        assert!(
                            !is_strict_device_execution(),
                            "permute CUDA backward failed while strict device execution is enabled: {err}"
                        );
                    }
                }
            }
            let grad_restored = grad.view().permuted_axes(rev_axes.clone()).to_owned();
            input_clone.add_backward_grad(grad_restored);
        })),
        requires_grad: true,
        device: output_device,
    })))
}

fn cuda_cat_nonempty_f32(
    tensors: &[Tensor],
    shapes: &[Vec<usize>],
    axis: usize,
) -> Result<cuda::CudaBuffer, String> {
    let mut inputs = tensors
        .iter()
        .zip(shapes)
        .filter(|(_, shape)| !shape.contains(&0));
    let (first, first_shape) = inputs
        .next()
        .ok_or_else(|| "CUDA cat has no nonempty input".to_string())?;
    let mut acc_shape = first_shape.clone();
    let mut acc_buffer = first
        .cloned_cuda_f32_buffer()
        .ok_or_else(|| "CUDA cat expected first resident buffer".to_string())?;
    for (rhs, rhs_shape) in inputs {
        let rhs_buffer = rhs
            .cloned_cuda_f32_buffer()
            .ok_or_else(|| "CUDA cat expected next resident buffer".to_string())?;
        let mut next_shape = acc_shape.clone();
        next_shape[axis] = next_shape[axis]
            .checked_add(rhs_shape[axis])
            .expect("Concat axis length overflow");
        acc_buffer =
            cuda::cat_f32_buffer(&acc_buffer, &rhs_buffer, &next_shape, axis, acc_shape[axis])?;
        acc_shape = next_shape;
    }
    Ok(acc_buffer)
}

pub fn cat(tensors: &[Tensor], axis: usize) -> Tensor {
    assert!(!tensors.is_empty(), "Concat expects at least one tensor");
    let output_device = tensors[0].device();
    let build_graph = !(is_no_grad() || tensors.iter().all(|t| !t.requires_grad()));
    assert_native_device_support(
        output_device,
        "cat",
        output_device == crate::device::Device::Cuda,
    );
    for tensor in tensors.iter().skip(1) {
        assert_eq!(
            tensor.device(),
            output_device,
            "cat expects all tensors on the same device"
        );
    }
    let shapes = validate_cat_inputs(tensors, axis);
    let out_shape = cat_output_shape(&shapes, axis);

    if tensors.len() == 1 && !tensors[0].requires_grad() {
        return tensors[0].clone();
    }

    if !build_graph {
        if output_device == crate::device::Device::Cuda {
            let same_dtype = tensors
                .iter()
                .all(|tensor| tensor.dtype() == tensors[0].dtype());
            let i8_scale = if same_dtype && tensors[0].dtype() == DType::I8 {
                let scales = tensors
                    .iter()
                    .filter_map(Tensor::quantization_scale)
                    .collect::<Vec<_>>();
                if scales.len() == tensors.len() && scales.windows(2).all(|pair| pair[0] == pair[1])
                {
                    Some(scales[0])
                } else {
                    None
                }
            } else {
                None
            };
            let output_dtype = if same_dtype {
                match tensors[0].dtype() {
                    DType::F32 => DType::F32,
                    DType::F16 => DType::F16,
                    DType::BF16 => DType::BF16,
                    DType::I8 if i8_scale.is_some() => DType::I8,
                    DType::I8 => DType::F32,
                }
            } else {
                DType::F32
            };

            if output_dtype != DType::F32 {
                let cuda_out = (|| -> Result<_, String> {
                    let mut inputs = tensors
                        .iter()
                        .zip(&shapes)
                        .filter(|(_, shape)| !shape.contains(&0));
                    let (first, first_shape) = inputs
                        .next()
                        .ok_or_else(|| "CUDA typed cat has no nonempty input".to_string())?;
                    let mut acc_shape = first_shape.clone();
                    let (dtype, mut acc_buffer, scale) = first
                        .cloned_cuda_native_lowp_buffer()
                        .ok_or_else(|| "CUDA typed cat expected lhs native buffer".to_string())?;
                    if dtype != output_dtype {
                        return Err(format!(
                            "CUDA typed cat dtype mismatch: expected {:?}, got {:?}",
                            output_dtype, dtype
                        ));
                    }
                    let mut out_scale = scale;
                    for (rhs, rhs_shape) in inputs {
                        let (rhs_dtype, rhs_buffer, rhs_scale) =
                            rhs.cloned_cuda_native_lowp_buffer().ok_or_else(|| {
                                "CUDA typed cat expected rhs native buffer".to_string()
                            })?;
                        if rhs_dtype != output_dtype {
                            return Err(format!(
                                "CUDA typed cat dtype mismatch: expected {:?}, got {:?}",
                                output_dtype, rhs_dtype
                            ));
                        }
                        if output_dtype == DType::I8 && rhs_scale != out_scale {
                            return Err("CUDA typed cat i8 scale mismatch".to_string());
                        }
                        let mut next_shape = acc_shape.clone();
                        next_shape[axis] = next_shape[axis]
                            .checked_add(rhs_shape[axis])
                            .expect("Concat axis length overflow");
                        acc_buffer = cuda::cat_typed_buffer(
                            &acc_buffer,
                            &rhs_buffer,
                            output_dtype,
                            &next_shape,
                            axis,
                            acc_shape[axis],
                        )?;
                        acc_shape = next_shape;
                        out_scale = rhs_scale;
                    }
                    Ok((acc_buffer, out_scale))
                })();
                if let Ok((buffer, scale)) = cuda_out {
                    return Tensor::from_cuda_native_lowp_buffer_no_host_with_dtype(
                        &out_shape,
                        buffer,
                        output_device,
                        output_dtype,
                        scale,
                    );
                }
            }

            if output_dtype == DType::F32 {
                let cuda_out = cuda_cat_nonempty_f32(tensors, &shapes, axis);
                if let Ok(buffer) = cuda_out {
                    return Tensor::from_cuda_f32_buffer_no_host(&out_shape, buffer, output_device);
                }
            }

            let cuda_out = (|| -> Result<_, String> {
                let buffer = cuda_cat_nonempty_f32(tensors, &shapes, axis)?;
                let host = cuda::download_f32(&buffer)?;
                Ok((buffer, host))
            })();
            if let Ok((buffer, out)) = cuda_out {
                if output_dtype == DType::I8 {
                    let scale = i8_scale.expect("checked i8 scale above");
                    let raw = out
                        .iter()
                        .map(|&v| (v / scale).round().clamp(-127.0, 127.0) as i8)
                        .collect::<Vec<_>>();
                    let data = Array::from_shape_vec(IxDyn(&out_shape), raw)
                        .expect("CUDA cat i8 output shape build failed")
                        .into_dyn()
                        .into_shared();
                    let tensor =
                        Tensor::from_shared_i8_no_grad_with_device(data, scale, output_device);
                    tensor.set_cuda_f32_buffer_inplace(buffer);
                    return tensor;
                }

                let out = Array::from_shape_vec(IxDyn(&out_shape), out)
                    .expect("CUDA cat output shape build failed")
                    .into_dyn();
                return Tensor::from_f32_data_no_grad_with_device_dtype_and_cuda_buffer(
                    out,
                    output_dtype,
                    output_device,
                    Some(buffer),
                );
            }
        }

        let storages = tensors
            .iter()
            .map(Tensor::native_storage_owned)
            .collect::<Vec<_>>();

        if storages
            .iter()
            .all(|s| matches!(s, TensorStorageOwned::F32(_)))
        {
            let arrays = storages
                .into_iter()
                .map(|storage| match storage {
                    TensorStorageOwned::F32(array) => array,
                    TensorStorageOwned::F16(_)
                    | TensorStorageOwned::BF16(_)
                    | TensorStorageOwned::I8(_, _) => unreachable!("checked above"),
                })
                .collect::<Vec<_>>();
            let views = arrays.iter().map(|a| a.view()).collect::<Vec<_>>();
            let result = ndarray::concatenate(Axis(axis), &views)
                .expect("Concat failed: shape mismatch or invalid axis")
                .into_dyn()
                .into_shared();
            return Tensor::from_shared_f32_no_grad_with_device(result, output_device);
        }

        if storages
            .iter()
            .all(|s| matches!(s, TensorStorageOwned::F16(_)))
        {
            let arrays = storages
                .into_iter()
                .map(|storage| match storage {
                    TensorStorageOwned::F16(array) => array,
                    TensorStorageOwned::F32(_)
                    | TensorStorageOwned::BF16(_)
                    | TensorStorageOwned::I8(_, _) => unreachable!("checked above"),
                })
                .collect::<Vec<_>>();
            let views = arrays.iter().map(|a| a.view()).collect::<Vec<_>>();
            let result = ndarray::concatenate(Axis(axis), &views)
                .expect("Concat failed: shape mismatch or invalid axis")
                .into_dyn()
                .into_shared();
            return Tensor::from_shared_f16_no_grad_with_device(result, output_device);
        }

        if storages
            .iter()
            .all(|s| matches!(s, TensorStorageOwned::BF16(_)))
        {
            let arrays = storages
                .into_iter()
                .map(|storage| match storage {
                    TensorStorageOwned::BF16(array) => array,
                    TensorStorageOwned::F32(_)
                    | TensorStorageOwned::F16(_)
                    | TensorStorageOwned::I8(_, _) => unreachable!("checked above"),
                })
                .collect::<Vec<_>>();
            let views = arrays.iter().map(|a| a.view()).collect::<Vec<_>>();
            let result = ndarray::concatenate(Axis(axis), &views)
                .expect("Concat failed: shape mismatch or invalid axis")
                .into_dyn()
                .into_shared();
            return Tensor::from_shared_bf16_no_grad_with_device(result, output_device);
        }

        if storages
            .iter()
            .all(|s| matches!(s, TensorStorageOwned::I8(_, _)))
        {
            let scales = storages
                .iter()
                .map(|storage| match storage {
                    TensorStorageOwned::I8(_, scale) => *scale,
                    TensorStorageOwned::F32(_)
                    | TensorStorageOwned::F16(_)
                    | TensorStorageOwned::BF16(_) => unreachable!("checked above"),
                })
                .collect::<Vec<_>>();
            if scales.windows(2).all(|pair| pair[0] == pair[1]) {
                let scale = scales[0];
                let arrays = storages
                    .into_iter()
                    .map(|storage| match storage {
                        TensorStorageOwned::I8(array, _) => array,
                        TensorStorageOwned::F32(_)
                        | TensorStorageOwned::F16(_)
                        | TensorStorageOwned::BF16(_) => unreachable!("checked above"),
                    })
                    .collect::<Vec<_>>();
                let views = arrays.iter().map(|a| a.view()).collect::<Vec<_>>();
                let result = ndarray::concatenate(Axis(axis), &views)
                    .expect("Concat failed: shape mismatch or invalid axis")
                    .into_dyn()
                    .into_shared();
                return Tensor::from_shared_i8_no_grad_with_device(result, scale, output_device);
            }
        }

        // Mixed storage and differing I8 scales promote to F32, while retaining
        // the no-graph contract even outside an explicit no_grad() scope.
        let arrays = storages
            .into_iter()
            .map(|storage| match storage {
                TensorStorageOwned::F32(a) => a,
                TensorStorageOwned::F16(a) => a.mapv(|v| v.to_f32()).into_shared(),
                TensorStorageOwned::BF16(a) => a.mapv(|v| v.to_f32()).into_shared(),
                TensorStorageOwned::I8(a, scale) => a.mapv(|v| v as f32 * scale).into_shared(),
            })
            .collect::<Vec<_>>();
        let views = arrays.iter().map(|a| a.view()).collect::<Vec<_>>();
        let result = ndarray::concatenate(Axis(axis), &views)
            .expect("Concat failed: shape mismatch or invalid axis")
            .into_dyn()
            .into_shared();
        return Tensor::from_shared_f32_no_grad_with_device(result, output_device);
    }

    if output_device == crate::device::Device::Cuda && !out_shape.contains(&0) {
        let cuda_out = cuda_cat_nonempty_f32(tensors, &shapes, axis);
        match cuda_out {
            Ok(buffer) => {
                let lengths: Vec<usize> = shapes.iter().map(|shape| shape[axis]).collect();
                let tensors_clone: Vec<Tensor> = tensors.to_vec();
                let shapes_for_backward = shapes.clone();
                let out_shape_for_backward = out_shape.clone();
                let output_self =
                    Rc::new(RefCell::new(std::rc::Weak::<RefCell<TensorData>>::new()));
                let output_self_for_backward = output_self.clone();
                let tensor = Tensor(Rc::new(RefCell::new(TensorData {
                    data: Array::zeros(IxDyn(&out_shape)).into_shared(),
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
                    parents: tensors.to_vec(),
                    backward_op: Some(std::rc::Rc::new(move |grad| {
                        let cuda_result = if let Some(grad_buffer) = output_self_for_backward
                            .borrow()
                            .upgrade()
                            .map(Tensor)
                            .and_then(|output| output.cloned_cuda_f32_grad())
                            .filter(|grad_buffer| grad_buffer.len() == grad.len())
                        {
                            if is_strict_device_execution() {
                                match (|| -> Result<Vec<Option<cuda::CudaBuffer>>, String> {
                                    let mut axis_start = 0usize;
                                    let mut pieces = Vec::with_capacity(shapes_for_backward.len());
                                    for shape in &shapes_for_backward {
                                        if shape.contains(&0) {
                                            pieces.push(None);
                                            continue;
                                        }
                                        let piece = cuda::cat_backward_slice_f32_buffer(
                                            &grad_buffer,
                                            shape,
                                            &out_shape_for_backward,
                                            axis,
                                            axis_start,
                                        )?;
                                        pieces.push(Some(piece));
                                        axis_start += shape[axis];
                                    }
                                    Ok(pieces)
                                })() {
                                    Ok(pieces) => {
                                        for (tensor, grad_buffer) in
                                            tensors_clone.iter().zip(pieces)
                                        {
                                            if let Some(buffer) = grad_buffer {
                                                tensor.add_backward_cuda_grad_buffer_only(buffer);
                                            } else {
                                                tensor.add_backward_grad(Array::zeros(IxDyn(
                                                    &tensor.shape_vec(),
                                                )));
                                            }
                                        }
                                        return;
                                    }
                                    Err(err) => {
                                        panic!(
                                            "cat CUDA backward failed while strict device execution is enabled: {err}"
                                        );
                                    }
                                }
                            }
                            (|| -> Result<_, String> {
                                let mut axis_start = 0usize;
                                let mut pieces = Vec::with_capacity(shapes_for_backward.len());
                                for shape in &shapes_for_backward {
                                    if shape.contains(&0) {
                                        pieces.push((None, Vec::new()));
                                        continue;
                                    }
                                    let (buffer, host) = cuda::cat_backward_slice_f32(
                                        &grad_buffer,
                                        shape,
                                        &out_shape_for_backward,
                                        axis,
                                        axis_start,
                                    )?;
                                    pieces.push((Some(buffer), host));
                                    axis_start += shape[axis];
                                }
                                Ok(pieces)
                            })()
                        } else {
                            let grad_host = grad.iter().copied().collect::<Vec<_>>();
                            cuda::upload_f32(&grad_host).and_then(|grad_buf| {
                                let mut axis_start = 0usize;
                                let mut pieces = Vec::with_capacity(shapes_for_backward.len());
                                for shape in &shapes_for_backward {
                                    if shape.contains(&0) {
                                        pieces.push((None, Vec::new()));
                                        continue;
                                    }
                                    let (buffer, host) = cuda::cat_backward_slice_f32(
                                        &grad_buf,
                                        shape,
                                        &out_shape_for_backward,
                                        axis,
                                        axis_start,
                                    )?;
                                    pieces.push((Some(buffer), host));
                                    axis_start += shape[axis];
                                }
                                Ok::<_, String>(pieces)
                            })
                        };

                        match cuda_result {
                            Ok(pieces) => {
                                for (i, (grad_buffer, grad_host)) in pieces.into_iter().enumerate()
                                {
                                    let sub_grad = Array::from_shape_vec(
                                        IxDyn(&shapes_for_backward[i]),
                                        grad_host,
                                    )
                                    .expect("CUDA cat grad shape build failed")
                                    .into_dyn();
                                    tensors_clone[i]
                                        .add_backward_grad_with_cuda_buffer(sub_grad, grad_buffer);
                                }
                                return;
                            }
                            Err(err) => {
                                assert!(
                                    !is_strict_device_execution(),
                                    "cat CUDA backward failed while strict device execution is enabled: {err}"
                                );
                            }
                        }

                        let axis_obj = Axis(axis);
                        let mut start_idx = 0;
                        for (i, &len) in lengths.iter().enumerate() {
                            let slice_info = Slice::from(start_idx..start_idx + len);
                            let sub_grad =
                                grad.slice_axis(axis_obj, slice_info).to_owned().into_dyn();
                            tensors_clone[i].add_backward_grad(sub_grad);
                            start_idx += len;
                        }
                    })),
                    requires_grad: true,
                    device: output_device,
                })));
                *output_self.borrow_mut() = Rc::downgrade(&tensor.0);
                return tensor;
            }
            Err(err) => {
                assert!(
                    !is_strict_device_execution(),
                    "cat CUDA forward failed while strict device execution is enabled: {err}"
                );
            }
        }
    }

    // concatenate 本身会 materialize 结果；输入侧尽量保持零拷贝 view
    let arrays: Vec<_> = tensors.iter().map(|t| t.data_arc()).collect();
    let views: Vec<_> = arrays.iter().map(|a| a.view()).collect();

    let axis_obj = Axis(axis);
    let result = ndarray::concatenate(axis_obj, &views)
        .expect("Concat failed: shape mismatch or invalid axis")
        .into_dyn()
        .into_shared();

    let lengths: Vec<usize> = shapes.iter().map(|shape| shape[axis]).collect();
    let tensors_clone: Vec<Tensor> = tensors.to_vec();
    let out_shape_for_cuda = out_shape;
    let has_output = !out_shape_for_cuda.contains(&0);
    let cuda_buffer = if output_device == crate::device::Device::Cuda && has_output {
        let cuda_out = cuda_cat_nonempty_f32(tensors, &shapes, axis);
        match cuda_out {
            Ok(buffer) => Some(buffer),
            Err(err) => {
                assert!(
                    !is_strict_device_execution(),
                    "cat CUDA forward failed while strict device execution is enabled: {err}"
                );
                None
            }
        }
    } else {
        None
    };
    let shapes_for_backward = shapes.clone();
    let out_shape_for_backward = out_shape_for_cuda.clone();

    Tensor(Rc::new(RefCell::new(TensorData {
        data: result,
        f16_data: None,
        bf16_data: None,
        i8_data: None,
        cuda_f32_data: cuda_buffer,
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
        parents: tensors.to_vec(),
        backward_op: Some(std::rc::Rc::new(move |grad| {
            if output_device == crate::device::Device::Cuda && !grad.is_empty() {
                let grad_host = grad.iter().copied().collect::<Vec<_>>();
                let cuda_result = cuda::upload_f32(&grad_host).and_then(|grad_buf| {
                    let mut axis_start = 0usize;
                    let mut pieces = Vec::with_capacity(shapes_for_backward.len());
                    for shape in &shapes_for_backward {
                        if shape.contains(&0) {
                            pieces.push((None, Vec::new()));
                            continue;
                        }

                        let (buffer, host) = cuda::cat_backward_slice_f32(
                            &grad_buf,
                            shape,
                            &out_shape_for_backward,
                            axis,
                            axis_start,
                        )?;

                        pieces.push((Some(buffer), host));
                        axis_start += shape[axis];
                    }
                    Ok::<_, String>(pieces)
                });

                match cuda_result {
                    Ok(pieces) => {
                        for (i, (grad_buffer, grad_host)) in pieces.into_iter().enumerate() {
                            let sub_grad =
                                Array::from_shape_vec(IxDyn(&shapes_for_backward[i]), grad_host)
                                    .expect("CUDA cat grad shape build failed")
                                    .into_dyn();
                            tensors_clone[i]
                                .add_backward_grad_with_cuda_buffer(sub_grad, grad_buffer);
                        }
                        return;
                    }
                    Err(err) => {
                        assert!(
                            !is_strict_device_execution(),
                            "cat CUDA backward failed while strict device execution is enabled: {err}"
                        );
                    }
                }
            }

            let mut start_idx = 0;
            for (i, &len) in lengths.iter().enumerate() {
                let slice_info = Slice::from(start_idx..start_idx + len);
                let sub_grad = grad.slice_axis(axis_obj, slice_info).to_owned().into_dyn();
                tensors_clone[i].add_backward_grad(sub_grad);
                start_idx += len;
            }
        })),
        requires_grad: true,
        device: output_device,
    })))
}

pub fn slice_last_dim(input: &Tensor, start: usize, end: usize) -> Tensor {
    let output_device = input.device();
    let input_shape = input.shape_vec();
    assert!(
        !input_shape.is_empty(),
        "slice_last_dim expects at least 1D input"
    );
    let last_dim = input_shape.len() - 1;
    let last_len = input_shape[last_dim];
    assert!(start <= end, "slice_last_dim expects start <= end");
    assert!(
        end <= last_len,
        "slice_last_dim end out of bounds: {} > {}",
        end,
        last_len
    );
    let axis = ndarray::Axis(last_dim);
    let build_graph = !is_no_grad() && input.requires_grad();
    assert_native_device_support(
        output_device,
        "slice_last_dim",
        output_device == crate::device::Device::Cuda,
    );

    if !build_graph && !input.requires_grad() && start == 0 && end == last_len {
        return input.clone();
    }

    if !build_graph && (start == end || input.is_empty()) {
        // Empty results need only metadata, not a snapshot of the whole input.
        let mut out_shape = input_shape.clone();
        out_shape[last_dim] = end - start;
        let tensor = if input.dtype() == DType::I8 {
            Tensor::from_i8_data_no_grad(
                Array::zeros(IxDyn(&out_shape)).into_shared(),
                input
                    .quantization_scale()
                    .expect("I8 slice expected a quantization scale"),
            )
        } else {
            Tensor::from_f32_data_no_grad_with_dtype(Array::zeros(IxDyn(&out_shape)), input.dtype())
        };
        tensor.set_device_inplace(output_device);
        return tensor;
    }

    if !build_graph && output_device == crate::device::Device::Cuda && !input.is_empty() {
        let slice_len = end - start;
        let outer = input.len() / last_len;
        let mut out_shape = input_shape.clone();
        out_shape[last_dim] = slice_len;
        if let Some((dtype, input_buf, i8_scale)) = input.cloned_cuda_native_lowp_buffer() {
            match cuda::slice_lastdim_typed_buffer(
                &input_buf, dtype, outer, last_len, start, slice_len,
            ) {
                Ok(buffer) => {
                    return Tensor::from_cuda_native_lowp_buffer_no_host_with_dtype(
                        &out_shape,
                        buffer,
                        output_device,
                        dtype,
                        i8_scale,
                    );
                }
                Err(err) => {
                    assert!(
                        !is_strict_device_execution(),
                        "CUDA typed slice_last_dim forward failed while strict device execution is enabled: {err}"
                    );
                }
            }
        }
        if input.dtype() == DType::F32 {
            let cuda_out = input.with_cuda_f32_buffer(|input_buf| {
                cuda::slice_lastdim_f32_buffer(input_buf, outer, last_len, start, slice_len)
            });
            if let Ok(buffer) = cuda_out {
                return Tensor::from_cuda_f32_buffer_no_host(&out_shape, buffer, output_device);
            }
        }
        let cuda_out = input.with_cuda_f32_buffer(|input_buf| {
            cuda::slice_lastdim_f32(input_buf, outer, last_len, start, slice_len)
        });
        if let Ok((buffer, out)) = cuda_out {
            let out = ndarray::Array::from_shape_vec(ndarray::IxDyn(&out_shape), out)
                .expect("CUDA slice_last_dim output shape build failed")
                .into_dyn();
            return Tensor::from_f32_data_no_grad_with_device_dtype_and_cuda_buffer(
                out,
                input.dtype(),
                output_device,
                Some(buffer),
            );
        }
    }

    if !build_graph {
        return match input.native_storage_owned() {
            TensorStorageOwned::F32(mut sliced) => {
                sliced.slice_axis_inplace(axis, ndarray::Slice::from(start..end));
                Tensor::from_shared_f32_no_grad_with_device(sliced.into_dyn(), output_device)
            }
            TensorStorageOwned::F16(mut sliced) => {
                sliced.slice_axis_inplace(axis, ndarray::Slice::from(start..end));
                Tensor::from_shared_f16_no_grad_with_device(sliced.into_dyn(), output_device)
            }
            TensorStorageOwned::BF16(mut sliced) => {
                sliced.slice_axis_inplace(axis, ndarray::Slice::from(start..end));
                Tensor::from_shared_bf16_no_grad_with_device(sliced.into_dyn(), output_device)
            }
            TensorStorageOwned::I8(mut sliced, scale) => {
                sliced.slice_axis_inplace(axis, ndarray::Slice::from(start..end));
                Tensor::from_shared_i8_no_grad_with_device(sliced.into_dyn(), scale, output_device)
            }
        };
    }

    if output_device == crate::device::Device::Cuda && !input.is_empty() && start < end {
        let slice_len = end - start;
        let outer = input.len() / last_len;
        let mut out_shape = input_shape.clone();
        out_shape[last_dim] = slice_len;
        let cuda_out = input.with_cuda_f32_buffer(|input_buf| {
            cuda::slice_lastdim_f32_buffer(input_buf, outer, last_len, start, slice_len)
        });
        match cuda_out {
            Ok(buffer) => {
                let input_clone = input.clone();
                let full_shape = input_shape.clone();
                let output_self =
                    Rc::new(RefCell::new(std::rc::Weak::<RefCell<TensorData>>::new()));
                let output_self_for_backward = output_self.clone();
                let tensor = Tensor(Rc::new(RefCell::new(TensorData {
                    data: Array::zeros(IxDyn(&out_shape)).into_shared(),
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
                    parents: vec![input.clone()],
                    backward_op: Some(std::rc::Rc::new(move |grad| {
                        let cuda_grad = if let Some(grad_buffer) = output_self_for_backward
                            .borrow()
                            .upgrade()
                            .map(Tensor)
                            .and_then(|output| output.cloned_cuda_f32_grad())
                            .filter(|grad_buffer| grad_buffer.len() == grad.len())
                        {
                            if is_strict_device_execution() {
                                match cuda::slice_lastdim_backward_f32_buffer(
                                    &grad_buffer,
                                    outer,
                                    last_len,
                                    start,
                                    slice_len,
                                ) {
                                    Ok(grad_buffer) => {
                                        input_clone.add_backward_cuda_grad_buffer_only(grad_buffer);
                                        return;
                                    }
                                    Err(err) => {
                                        panic!(
                                            "slice_last_dim CUDA backward failed while strict device execution is enabled: {err}"
                                        );
                                    }
                                }
                            }
                            cuda::slice_lastdim_backward_f32(
                                &grad_buffer,
                                outer,
                                last_len,
                                start,
                                slice_len,
                            )
                        } else {
                            let grad_host = grad.iter().copied().collect::<Vec<_>>();
                            cuda::upload_f32(&grad_host).and_then(|grad_buf| {
                                cuda::slice_lastdim_backward_f32(
                                    &grad_buf, outer, last_len, start, slice_len,
                                )
                            })
                        };
                        match cuda_grad {
                            Ok((grad_buffer, full_grad_host)) => {
                                let full_grad =
                                    Array::from_shape_vec(IxDyn(&full_shape), full_grad_host)
                                        .expect("CUDA slice_last_dim grad shape build failed")
                                        .into_dyn();
                                input_clone.add_backward_grad_with_cuda_buffer(
                                    full_grad,
                                    Some(grad_buffer),
                                );
                            }
                            Err(err) => {
                                assert!(
                                    !is_strict_device_execution(),
                                    "slice_last_dim CUDA backward failed while strict device execution is enabled: {err}"
                                );
                                let mut full_grad = ndarray::Array::zeros(full_shape.clone());
                                full_grad
                                    .slice_axis_mut(axis, ndarray::Slice::from(start..end))
                                    .assign(grad);
                                input_clone.add_backward_grad(full_grad.into_dyn());
                            }
                        }
                    })),
                    requires_grad: true,
                    device: output_device,
                })));
                *output_self.borrow_mut() = Rc::downgrade(&tensor.0);
                return tensor;
            }
            Err(err) => {
                assert!(
                    !is_strict_device_execution(),
                    "slice_last_dim CUDA forward failed while strict device execution is enabled: {err}"
                );
            }
        }
    }

    let sliced = if start == end || input.is_empty() {
        let mut out_shape = input_shape.clone();
        out_shape[last_dim] = end - start;
        Array::zeros(IxDyn(&out_shape)).into_shared()
    } else {
        input
            .data_ref()
            .slice_axis(axis, ndarray::Slice::from(start..end))
            .to_owned()
            .into_dyn()
            .into_shared()
    };

    let input_clone = input.clone();
    let full_shape = input_shape;

    Tensor(Rc::new(RefCell::new(TensorData {
        data: sliced,
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
        parents: vec![input.clone()],
        backward_op: Some(std::rc::Rc::new(move |grad| {
            if input_clone.is_cuda() && grad.is_empty() && !full_shape.contains(&0) {
                // A zero VJP must not download a previously resident gradient.
                match cuda::fill_scalar_f32_buffer(input_clone.len(), 0.0) {
                    Ok(buffer) => {
                        input_clone.add_backward_cuda_grad_buffer_only(buffer);
                        return;
                    }
                    Err(err) => assert!(
                        !is_strict_device_execution(),
                        "slice_last_dim CUDA zero backward failed while strict device execution is enabled: {err}"
                    ),
                }
            }
            let mut full_grad = ndarray::Array::zeros(full_shape.clone());
            full_grad
                .slice_axis_mut(axis, ndarray::Slice::from(start..end))
                .assign(grad);
            input_clone.add_backward_grad(full_grad.into_dyn());
        })),
        requires_grad: true,
        device: output_device,
    })))
}

#[cfg(test)]
#[path = "../../tests/unit/ops/shape/tests.rs"]
mod tests;
