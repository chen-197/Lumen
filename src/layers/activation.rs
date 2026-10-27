use crate::autograd::{
    Device, StoragePreference, Tensor, TensorData, TensorStorageView, assert_native_device_support,
    is_no_grad, is_strict_device_execution,
};
use crate::module::Module;
use crate::ops::cuda;
use crate::precision::DType;
use ndarray::{Array2, IxDyn, Zip};
use std::cell::RefCell;
use std::rc::Rc;

const CPU_UNARY_SERIAL_FAST_LEN: usize = 1 << 16;

fn try_cuda_unary_native_lowp_buffer(
    input: &Tensor,
    cuda_op: cuda::UnaryOp,
) -> Option<Result<cuda::CudaBuffer, String>> {
    if input.device() != Device::Cuda || input.is_empty() {
        return None;
    }

    let (dtype, buffer, scale) = input.cloned_cuda_native_lowp_buffer()?;
    Some(match dtype {
        DType::F16 => cuda::unary_f16_buffer(&buffer, cuda_op),
        DType::BF16 => cuda::unary_bf16_buffer(&buffer, cuda_op),
        DType::I8 => cuda::unary_i8_buffer(&buffer, scale?, cuda_op),
        DType::F32 => return None,
    })
}

fn try_cuda_unary_backward_native_lowp_buffer(
    input: &Tensor,
    output: &cuda::CudaBuffer,
    grad: &cuda::CudaBuffer,
    cuda_op: cuda::UnaryOp,
) -> Option<Result<cuda::CudaBuffer, String>> {
    if input.device() != Device::Cuda || input.is_empty() {
        return None;
    }

    let (dtype, buffer, scale) = input.cloned_cuda_native_lowp_buffer()?;
    Some(match dtype {
        DType::F16 => cuda::unary_backward_f16_buffer(&buffer, output, grad, cuda_op),
        DType::BF16 => cuda::unary_backward_bf16_buffer(&buffer, output, grad, cuda_op),
        DType::I8 => cuda::unary_backward_i8_buffer(&buffer, scale?, output, grad, cuda_op),
        DType::F32 => return None,
    })
}

fn cuda_unary_backward_buffer(
    input: &Tensor,
    output: &cuda::CudaBuffer,
    grad: &cuda::CudaBuffer,
    cuda_op: cuda::UnaryOp,
) -> Result<cuda::CudaBuffer, String> {
    try_cuda_unary_backward_native_lowp_buffer(input, output, grad, cuda_op).unwrap_or_else(|| {
        input.with_cuda_f32_buffer(|input_buf| {
            cuda::unary_backward_f32_buffer(input_buf, output, grad, cuda_op)
        })
    })
}

fn unary_no_grad(
    input: &Tensor,
    cuda_op: cuda::UnaryOp,
    op: impl Fn(f32) -> f32 + Copy + Send + Sync,
) -> Tensor {
    let output_dtype = input.dtype();
    let output_device = input.device();
    if input.is_empty() {
        return Tensor::from_f32_data_no_grad_with_device_dtype(
            ndarray::ArrayD::<f32>::zeros(IxDyn(&input.shape_vec())),
            output_dtype,
            output_device,
        );
    }
    if output_device == Device::Cuda
        && (cuda::should_accelerate_elementwise(input.len()) || is_strict_device_execution())
    {
        if let Some((dtype, buffer, scale)) = input.cloned_cuda_native_lowp_buffer()
            && dtype == output_dtype
        {
            let typed_out = match dtype {
                DType::F16 => cuda::unary_f16_typed_output_buffer(&buffer, cuda_op).ok(),
                DType::BF16 => cuda::unary_bf16_typed_output_buffer(&buffer, cuda_op).ok(),
                DType::I8 if cuda_op == cuda::UnaryOp::Relu => {
                    cuda::unary_i8_relu_typed_output_buffer(&buffer).ok()
                }
                DType::F32 | DType::I8 => None,
            };
            if let Some(buffer) = typed_out {
                return Tensor::from_cuda_native_lowp_buffer_no_host_with_dtype(
                    &input.shape_vec(),
                    buffer,
                    output_device,
                    output_dtype,
                    if dtype == DType::I8 { scale } else { None },
                );
            }
        }
        if let Some(Ok(buffer)) = try_cuda_unary_native_lowp_buffer(input, cuda_op) {
            return Tensor::from_cuda_f32_buffer_no_host_with_dtype(
                &input.shape_vec(),
                buffer,
                output_device,
                output_dtype,
            );
        }
        if output_dtype == DType::F32
            && let Ok(buffer) =
                input.with_cuda_f32_buffer(|input_buf| cuda::unary_f32_buffer(input_buf, cuda_op))
        {
            return Tensor::from_cuda_f32_buffer_no_host(&input.shape_vec(), buffer, output_device);
        }
    }

    input.with_storage_view_preferring(StoragePreference::F32Compute, |input_view| {
        let input_f32 = match input_view {
            TensorStorageView::F32(view) => view,
            TensorStorageView::F16(_) => {
                unreachable!("f32 compute preference should expose f32 view")
            }
            TensorStorageView::BF16(_) => {
                unreachable!("f32 compute preference should expose f32 view")
            }
        };
        if output_device == crate::autograd::Device::Cuda
            && (cuda::should_accelerate_elementwise(input_f32.len())
                || is_strict_device_execution())
        {
            if output_dtype == DType::F32 {
                let cuda_out = input
                    .with_cuda_f32_buffer(|input_buf| cuda::unary_f32_buffer(input_buf, cuda_op));
                if let Ok(buffer) = cuda_out {
                    return Tensor::from_cuda_f32_buffer_no_host(
                        input_f32.shape(),
                        buffer,
                        output_device,
                    );
                }
            } else {
                let cuda_out =
                    input.with_cuda_f32_buffer(|input_buf| cuda::unary_f32(input_buf, cuda_op));
                if let Ok((buffer, out)) = cuda_out {
                    let data = ndarray::Array::from_shape_vec(input_f32.raw_dim(), out)
                        .expect("CUDA unary output shape build failed")
                        .into_dyn();
                    return Tensor::from_f32_data_no_grad_with_device_dtype_and_cuda_buffer(
                        data,
                        output_dtype,
                        output_device,
                        Some(buffer),
                    );
                }
            }
        }
        if input_f32.len() <= CPU_UNARY_SERIAL_FAST_LEN
            && let Some(input_slice) = input_f32.as_slice()
        {
            let mut out = Vec::with_capacity(input_slice.len());
            out.extend(input_slice.iter().map(|&x| op(x)));
            let data = ndarray::Array::from_shape_vec(input_f32.raw_dim(), out)
                .expect("CPU unary output shape build failed")
                .into_dyn();
            return Tensor::from_f32_data_no_grad_with_device_dtype(
                data,
                output_dtype,
                output_device,
            );
        }
        let data = Zip::from(&input_f32).par_map_collect(|&x| op(x)).into_dyn();
        Tensor::from_f32_data_no_grad_with_device_dtype(data, output_dtype, output_device)
    })
}

fn softmax_no_grad(input: &Tensor, axis: usize) -> Tensor {
    let output_dtype = if input.dtype().is_float() {
        input.dtype()
    } else {
        DType::F32
    };
    let output_device = input.device();

    if output_device == crate::autograd::Device::Cuda && !input.is_empty() {
        let shape = input.shape_vec();
        assert!(!shape.is_empty(), "Softmax expects at least 1D input");
        assert_eq!(
            axis,
            shape.len() - 1,
            "Softmax currently only supports the last dimension in this implementation"
        );
        let last_dim = shape[axis];
        assert!(
            last_dim > 0,
            "Softmax last dimension must be greater than zero"
        );
        let outer_dim = input.len() / last_dim;
        if cuda::should_accelerate_softmax(outer_dim, last_dim)
            && let Some((dtype, input_buf, scale)) = input.cloned_cuda_native_lowp_buffer()
        {
            if output_dtype == DType::F32 {
                if let Ok(buffer) = cuda::softmax_lastdim_typed_no_host(
                    &input_buf, dtype, scale, outer_dim, last_dim,
                ) {
                    return Tensor::from_cuda_f32_buffer_no_host(&shape, buffer, output_device);
                }
            } else if let Ok(buffer) =
                cuda::softmax_lastdim_typed_no_host(&input_buf, dtype, scale, outer_dim, last_dim)
            {
                match cuda::f32_to_lowp_storage_no_host(&buffer, output_dtype) {
                    Ok(lowp_buffer) => {
                        return Tensor::from_cuda_native_lowp_buffer_no_host_with_dtype(
                            &shape,
                            lowp_buffer,
                            output_device,
                            output_dtype,
                            None,
                        );
                    }
                    Err(err) => {
                        assert!(
                            !is_strict_device_execution(),
                            "softmax CUDA low precision output conversion failed while strict device execution is enabled: {err}"
                        );
                    }
                }
            }
        }
    }

    input.with_storage_view_preferring(StoragePreference::F32Compute, |input_view| {
        let input_view = match input_view {
            TensorStorageView::F32(view) => view,
            TensorStorageView::F16(_) => {
                unreachable!("f32 compute preference should expose f32 view")
            }
            TensorStorageView::BF16(_) => {
                unreachable!("f32 compute preference should expose f32 view")
            }
        };
        let shape = input_view.shape().to_vec();
        assert!(!shape.is_empty(), "Softmax expects at least 1D input");
        assert_eq!(
            axis,
            shape.len() - 1,
            "Softmax currently only supports the last dimension in this implementation"
        );
        let last_dim = shape[axis];
        assert!(
            last_dim > 0,
            "Softmax last dimension must be greater than zero"
        );
        let outer_dim = shape.iter().product::<usize>() / last_dim;

        if output_device == crate::autograd::Device::Cuda
            && cuda::should_accelerate_softmax(outer_dim, last_dim)
        {
            if output_dtype == DType::F32 {
                let cuda_out = input.with_cuda_f32_buffer(|input_buf| {
                    cuda::softmax_lastdim_f32_no_host(input_buf, outer_dim, last_dim)
                });
                if let Ok(buffer) = cuda_out {
                    return Tensor::from_cuda_f32_buffer_no_host(&shape, buffer, output_device);
                }
            } else {
                let cuda_out = input.with_cuda_f32_buffer(|input_buf| {
                    cuda::softmax_lastdim_f32_no_host(input_buf, outer_dim, last_dim)
                });
                if let Ok(buffer) = cuda_out {
                    match cuda::f32_to_lowp_storage_no_host(&buffer, output_dtype) {
                        Ok(lowp_buffer) => {
                            return Tensor::from_cuda_native_lowp_buffer_no_host_with_dtype(
                                &shape,
                                lowp_buffer,
                                output_device,
                                output_dtype,
                                None,
                            );
                        }
                        Err(err) => {
                            assert!(
                                !is_strict_device_execution(),
                                "softmax CUDA low precision output conversion failed while strict device execution is enabled: {err}"
                            );
                        }
                    }
                }
            }
        }

        let x_cow = input_view.as_standard_layout();
        let x_2d = x_cow.view().into_shape((outer_dim, last_dim)).unwrap();
        let mut y_flat = Array2::<f32>::zeros((outer_dim, last_dim));
        Zip::from(y_flat.outer_iter_mut())
            .and(x_2d.outer_iter())
            .par_for_each(|mut y_row, x_row| {
                let max_val = x_row.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let mut sum = 0.0f32;

                for (y_val, &x_val) in y_row.iter_mut().zip(x_row.iter()) {
                    let e = (x_val - max_val).exp();
                    *y_val = e;
                    sum += e;
                }

                let inv_sum = 1.0 / sum;
                for y_val in y_row.iter_mut() {
                    *y_val *= inv_sum;
                }
            });

        Tensor::from_f32_data_no_grad_with_device_dtype(
            y_flat.into_shape(shape).unwrap().into_dyn(),
            output_dtype,
            output_device,
        )
    })
}

fn relu_f32(x: f32) -> f32 {
    x.max(0.0)
}

fn relu_backward_f32(x: f32, _y: f32, grad: f32) -> f32 {
    if x > 0.0 { grad } else { 0.0 }
}

fn sigmoid_f32(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn sigmoid_backward_f32(_x: f32, y: f32, grad: f32) -> f32 {
    grad * y * (1.0 - y)
}

fn tanh_f32(x: f32) -> f32 {
    x.tanh()
}

fn tanh_backward_f32(_x: f32, y: f32, grad: f32) -> f32 {
    grad * (1.0 - y * y)
}

fn silu_f32(x: f32) -> f32 {
    x * sigmoid_f32(x)
}

fn silu_backward_f32(x: f32, _y: f32, grad: f32) -> f32 {
    let sig = sigmoid_f32(x);
    grad * (sig + x * sig * (1.0 - sig))
}

fn gelu_f32(x: f32) -> f32 {
    const C: f32 = 0.797_884_6;
    const K: f32 = 0.044715;
    let x3 = x * x * x;
    0.5 * x * (1.0 + (C * (x + K * x3)).tanh())
}

fn gelu_backward_f32(x: f32, _y: f32, grad: f32) -> f32 {
    const C: f32 = 0.797_884_6;
    const K: f32 = 0.044715;
    let x3 = x * x * x;
    let inner = C * (x + K * x3);
    let tanh_i = inner.tanh();
    let sech2 = 1.0 - tanh_i * tanh_i;
    grad * (0.5 * (1.0 + tanh_i) + 0.5 * x * sech2 * C * (1.0 + 3.0 * K * x * x))
}

fn cpu_unary_activation_data(input: &Tensor, forward: fn(f32) -> f32) -> ndarray::ArrayD<f32> {
    let input_ref = input.data_ref();
    if input_ref.len() <= CPU_UNARY_SERIAL_FAST_LEN
        && let Some(input_slice) = input_ref.as_slice()
    {
        let mut out = Vec::with_capacity(input_slice.len());
        out.extend(input_slice.iter().map(|&x| forward(x)));
        return ndarray::Array::from_shape_vec(input_ref.raw_dim(), out)
            .expect("CPU unary activation output shape build failed")
            .into_dyn();
    }
    Zip::from(&*input_ref)
        .par_map_collect(|&x| forward(x))
        .into_dyn()
}

fn cpu_unary_activation_grad(
    input: &Tensor,
    output_data: &ndarray::ArrayD<f32>,
    grad: &ndarray::ArrayViewD<'_, f32>,
    backward: fn(f32, f32, f32) -> f32,
) -> ndarray::ArrayD<f32> {
    let input_ref = input.data_ref();
    if grad.len() <= CPU_UNARY_SERIAL_FAST_LEN
        && let (Some(grad_slice), Some(input_slice), Some(output_slice)) = (
            grad.as_slice(),
            input_ref.as_slice(),
            output_data.as_slice(),
        )
    {
        let mut out = Vec::with_capacity(grad_slice.len());
        out.extend(
            grad_slice
                .iter()
                .zip(input_slice)
                .zip(output_slice)
                .map(|((&g, &x), &y)| backward(x, y, g)),
        );
        return ndarray::Array::from_shape_vec(grad.raw_dim(), out)
            .expect("CPU unary activation grad shape build failed")
            .into_dyn();
    }
    let mut grad_input = grad.to_owned().into_dyn();
    Zip::from(grad_input.view_mut())
        .and(&*input_ref)
        .and(output_data)
        .par_for_each(|g, &x, &y| {
            *g = backward(x, y, *g);
        });
    grad_input
}

fn unary_activation_with_backward(
    input: Tensor,
    op_name: &'static str,
    cuda_op: cuda::UnaryOp,
    forward: fn(f32) -> f32,
    backward: fn(f32, f32, f32) -> f32,
) -> Tensor {
    let build_graph = !is_no_grad() && input.requires_grad();
    let output_device = input.device();

    if !build_graph {
        return unary_no_grad(&input, cuda_op, forward);
    }

    let cuda_native_supported = output_device == Device::Cuda;
    assert_native_device_support(output_device, op_name, cuda_native_supported);

    if output_device == Device::Cuda && !input.is_empty() {
        let shape = input.shape_vec();
        let cuda_out = try_cuda_unary_native_lowp_buffer(&input, cuda_op).unwrap_or_else(|| {
            input.with_cuda_f32_buffer(|input_buf| cuda::unary_f32_buffer(input_buf, cuda_op))
        });
        match cuda_out {
            Ok(buffer) => {
                let output_buffer = buffer.clone();
                let input_clone = input.clone();
                let output_self =
                    Rc::new(RefCell::new(std::rc::Weak::<RefCell<TensorData>>::new()));
                let output_self_for_backward = output_self.clone();

                let tensor = Tensor(Rc::new(RefCell::new(TensorData {
                    data: ndarray::ArrayD::<f32>::zeros(ndarray::IxDyn(&shape)).into_shared(),
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
                        let grad_shape = grad.shape().to_vec();
                        let upstream_cuda_grad = output_self_for_backward
                            .borrow()
                            .upgrade()
                            .map(Tensor)
                            .and_then(|output| output.cloned_cuda_f32_grad())
                            .filter(|grad_buf| grad_buf.len() == grad.len());
                        let cuda_grad = if let Some(grad_buf) = upstream_cuda_grad {
                            if is_strict_device_execution() {
                                match cuda_unary_backward_buffer(
                                    &input_clone,
                                    &output_buffer,
                                    &grad_buf,
                                    cuda_op,
                                ) {
                                    Ok(grad_buffer) => {
                                        input_clone.add_backward_cuda_grad_buffer_only(grad_buffer);
                                        return;
                                    }
                                    Err(err) => {
                                        panic!(
                                            "{op_name} CUDA backward failed while strict device execution is enabled: {err}"
                                        );
                                    }
                                }
                            }
                            cuda_unary_backward_buffer(
                                &input_clone,
                                &output_buffer,
                                &grad_buf,
                                cuda_op,
                            )
                            .and_then(|grad_buffer| {
                                cuda::download_f32(&grad_buffer)
                                    .map(|grad_host| (grad_buffer, grad_host))
                            })
                        } else {
                            let grad_host = grad.iter().copied().collect::<Vec<_>>();
                            cuda::upload_f32(&grad_host).and_then(|grad_buf| {
                                cuda_unary_backward_buffer(
                                    &input_clone,
                                    &output_buffer,
                                    &grad_buf,
                                    cuda_op,
                                )
                                .and_then(|grad_buffer| {
                                    cuda::download_f32(&grad_buffer)
                                        .map(|grad_host| (grad_buffer, grad_host))
                                })
                            })
                        };

                        match cuda_grad {
                            Ok((grad_buffer, grad_input_host)) => {
                                let grad_input = ndarray::Array::from_shape_vec(
                                    ndarray::IxDyn(&grad_shape),
                                    grad_input_host,
                                )
                                .expect("CUDA unary activation grad shape build failed")
                                .into_dyn();
                                input_clone.add_backward_grad_with_cuda_buffer(
                                    grad_input,
                                    Some(grad_buffer),
                                );
                            }
                            Err(err) => {
                                assert!(
                                    !is_strict_device_execution(),
                                    "{op_name} CUDA backward failed while strict device execution is enabled: {err}"
                                );
                                if let Ok(output_host) = cuda::download_f32(&output_buffer) {
                                    let output_data = ndarray::Array::from_shape_vec(
                                        ndarray::IxDyn(&grad_shape),
                                        output_host,
                                    )
                                    .expect("CUDA unary fallback output shape build failed")
                                    .into_dyn();
                                    let grad_input = cpu_unary_activation_grad(
                                        &input_clone,
                                        &output_data,
                                        grad,
                                        backward,
                                    );
                                    input_clone.add_backward_grad(grad_input);
                                }
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
                    "{op_name} CUDA forward failed while strict device execution is enabled: {err}"
                );
            }
        }
    }

    let data = cpu_unary_activation_data(&input, forward);
    let output_data = data.clone();
    let input_clone = input.clone();
    Tensor(Rc::new(RefCell::new(TensorData {
        data: data.into_shared(),
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
            let grad_input = cpu_unary_activation_grad(&input_clone, &output_data, grad, backward);
            input_clone.add_backward_grad(grad_input);
        })),
        requires_grad: true,
        device: output_device,
    })))
}

pub struct ReLU;
impl ReLU {
    pub fn new() -> Self {
        ReLU
    }
}

impl Default for ReLU {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for ReLU {
    fn forward(&self, input: Tensor) -> Tensor {
        unary_activation_with_backward(
            input,
            "relu",
            cuda::UnaryOp::Relu,
            relu_f32,
            relu_backward_f32,
        )
    }
    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }
}

pub struct Sigmoid;
impl Sigmoid {
    pub fn new() -> Self {
        Sigmoid
    }
}

impl Default for Sigmoid {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for Sigmoid {
    fn forward(&self, input: Tensor) -> Tensor {
        unary_activation_with_backward(
            input,
            "sigmoid",
            cuda::UnaryOp::Sigmoid,
            sigmoid_f32,
            sigmoid_backward_f32,
        )
    }
    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }
}

pub struct Tanh;
impl Tanh {
    pub fn new() -> Self {
        Tanh
    }
}

impl Default for Tanh {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for Tanh {
    fn forward(&self, input: Tensor) -> Tensor {
        unary_activation_with_backward(
            input,
            "tanh",
            cuda::UnaryOp::Tanh,
            tanh_f32,
            tanh_backward_f32,
        )
    }
    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }
}

pub struct SiLU;
impl SiLU {
    pub fn new() -> Self {
        SiLU
    }
}

impl Default for SiLU {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for SiLU {
    fn forward(&self, input: Tensor) -> Tensor {
        unary_activation_with_backward(
            input,
            "silu",
            cuda::UnaryOp::Silu,
            silu_f32,
            silu_backward_f32,
        )
    }
    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }
}

pub struct Softmax {
    axis: usize,
}

impl Softmax {
    pub fn new(axis: usize) -> Self {
        Softmax { axis }
    }
}

impl Module for Softmax {
    fn forward(&self, input: Tensor) -> Tensor {
        let build_graph = !is_no_grad() && input.requires_grad();
        let output_device = input.device();

        if !build_graph {
            return softmax_no_grad(&input, self.axis);
        }

        let shape = input.shape_vec();
        assert!(!shape.is_empty(), "Softmax expects at least 1D input");
        assert_eq!(
            self.axis,
            shape.len() - 1,
            "Softmax currently only supports the last dimension in this implementation"
        );
        let axis_idx = self.axis;
        let last_dim = shape[axis_idx];
        assert!(
            last_dim > 0,
            "Softmax last dimension must be greater than zero"
        );
        let outer_dim = input.len() / last_dim;
        let cuda_native_supported = output_device == Device::Cuda;
        assert_native_device_support(output_device, "softmax", cuda_native_supported);

        if output_device == Device::Cuda && !input.is_empty() {
            let cuda_out = if let Some((dtype, input_buf, scale)) =
                input.cloned_cuda_native_lowp_buffer()
            {
                cuda::softmax_lastdim_typed_no_host(&input_buf, dtype, scale, outer_dim, last_dim)
            } else {
                input.with_cuda_f32_buffer(|input_buf| {
                    cuda::softmax_lastdim_f32_no_host(input_buf, outer_dim, last_dim)
                })
            };
            match cuda_out {
                Ok(buffer) => {
                    let output_buffer = buffer.clone();
                    let input_clone = input.clone();
                    let output_self =
                        Rc::new(RefCell::new(std::rc::Weak::<RefCell<TensorData>>::new()));
                    let output_self_for_backward = output_self.clone();

                    let tensor = Tensor(Rc::new(RefCell::new(TensorData {
                        data: ndarray::ArrayD::<f32>::zeros(ndarray::IxDyn(&shape)).into_shared(),
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
                        backward_op: Some(std::rc::Rc::new(move |grad_output| {
                            let grad_shape = grad_output.shape().to_vec();
                            let dim = grad_shape[axis_idx];
                            let outer = grad_output.len() / dim;
                            let upstream_cuda_grad = output_self_for_backward
                                .borrow()
                                .upgrade()
                                .map(Tensor)
                                .and_then(|output| output.cloned_cuda_f32_grad())
                                .filter(|grad_buf| grad_buf.len() == grad_output.len());
                            let cuda_grad = if let Some(grad_buf) = upstream_cuda_grad {
                                if is_strict_device_execution() {
                                    match cuda::softmax_lastdim_backward_f32_buffer(
                                        &output_buffer,
                                        &grad_buf,
                                        outer,
                                        dim,
                                    ) {
                                        Ok(grad_buffer) => {
                                            input_clone
                                                .add_backward_cuda_grad_buffer_only(grad_buffer);
                                            return;
                                        }
                                        Err(err) => {
                                            panic!(
                                                "softmax CUDA backward failed while strict device execution is enabled: {err}"
                                            );
                                        }
                                    }
                                }
                                cuda::softmax_lastdim_backward_f32(
                                    &output_buffer,
                                    &grad_buf,
                                    outer,
                                    dim,
                                )
                            } else {
                                let grad_host = grad_output.iter().copied().collect::<Vec<_>>();
                                cuda::upload_f32(&grad_host).and_then(|grad_buf| {
                                    cuda::softmax_lastdim_backward_f32(
                                        &output_buffer,
                                        &grad_buf,
                                        outer,
                                        dim,
                                    )
                                })
                            };

                            match cuda_grad {
                                Ok((grad_buffer, grad_input_host)) => {
                                    let grad_input = ndarray::Array::from_shape_vec(
                                        ndarray::IxDyn(&grad_shape),
                                        grad_input_host,
                                    )
                                    .expect("CUDA softmax grad shape build failed")
                                    .into_dyn();
                                    input_clone.add_backward_grad_with_cuda_buffer(
                                        grad_input,
                                        Some(grad_buffer),
                                    );
                                }
                                Err(err) => {
                                    assert!(
                                        !is_strict_device_execution(),
                                        "softmax CUDA backward failed while strict device execution is enabled: {err}"
                                    );
                                    if let Ok(output_host) = cuda::download_f32(&output_buffer) {
                                        let output_data = ndarray::Array::from_shape_vec(
                                            ndarray::IxDyn(&grad_shape),
                                            output_host,
                                        )
                                        .expect("CUDA softmax fallback output shape build failed")
                                        .into_dyn();
                                        let g_cow = grad_output.as_standard_layout();
                                        let g_2d = g_cow.view().into_shape((outer, dim)).unwrap();
                                        let y_cow = output_data.as_standard_layout();
                                        let y_2d = y_cow.view().into_shape((outer, dim)).unwrap();
                                        let mut d_input_flat = Array2::<f32>::zeros((outer, dim));

                                        Zip::from(d_input_flat.outer_iter_mut())
                                            .and(y_2d.outer_iter())
                                            .and(g_2d.outer_iter())
                                            .par_for_each(|mut di_row, y_row, g_row| {
                                                let dot: f32 = y_row
                                                    .iter()
                                                    .zip(g_row.iter())
                                                    .map(|(&y, &g)| y * g)
                                                    .sum();

                                                for (di, (&y, &g)) in di_row
                                                    .iter_mut()
                                                    .zip(y_row.iter().zip(g_row.iter()))
                                                {
                                                    *di = y * (g - dot);
                                                }
                                            });

                                        let d_input =
                                            d_input_flat.into_shape(grad_shape).unwrap().into_dyn();
                                        input_clone.add_backward_grad(d_input);
                                    }
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
                        "softmax CUDA forward failed while strict device execution is enabled: {err}"
                    );
                }
            }
        }

        let (y, output_data) = {
            let input_ref = input.data_ref();
            let x = &*input_ref;
            let shape = x.shape();
            assert!(!shape.is_empty(), "Softmax expects at least 1D input");
            assert_eq!(
                self.axis,
                shape.len() - 1,
                "Softmax currently only supports the last dimension in this implementation"
            );

            let axis = self.axis;

            let last_dim = shape[axis];
            let outer_dim = x.len() / last_dim;
            let x_cow = x.as_standard_layout();
            let x_2d = x_cow.view().into_shape((outer_dim, last_dim)).unwrap();

            let mut y_flat = Array2::<f32>::zeros((outer_dim, last_dim));
            Zip::from(y_flat.outer_iter_mut())
                .and(x_2d.outer_iter())
                .par_for_each(|mut y_row, x_row| {
                    let max_val = x_row.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                    let mut sum = 0.0f32;

                    for (y_val, &x_val) in y_row.iter_mut().zip(x_row.iter()) {
                        let e = (x_val - max_val).exp();
                        *y_val = e;
                        sum += e;
                    }

                    let inv_sum = 1.0 / sum;
                    for y_val in y_row.iter_mut() {
                        *y_val *= inv_sum;
                    }
                });

            let y = y_flat.into_shape(shape).unwrap();
            (y.clone(), y)
        };

        let input_clone = input.clone();
        let axis_idx = self.axis;

        Tensor(Rc::new(RefCell::new(TensorData {
            data: y.into_shared(),
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
            backward_op: Some(std::rc::Rc::new(move |grad_output| {
                let grad_shape = grad_output.shape();
                let dim = grad_shape[axis_idx];
                let outer = grad_output.len() / dim;
                let g_cow = grad_output.as_standard_layout();
                let g_2d = g_cow.view().into_shape((outer, dim)).unwrap();

                let y_cow = output_data.as_standard_layout();
                let y_2d = y_cow.view().into_shape((outer, dim)).unwrap();

                let mut d_input_flat = Array2::<f32>::zeros((outer, dim));

                Zip::from(d_input_flat.outer_iter_mut())
                    .and(y_2d.outer_iter())
                    .and(g_2d.outer_iter())
                    .par_for_each(|mut di_row, y_row, g_row| {
                        let dot: f32 = y_row.iter().zip(g_row.iter()).map(|(&y, &g)| y * g).sum();

                        for (di, (&y, &g)) in di_row.iter_mut().zip(y_row.iter().zip(g_row.iter()))
                        {
                            *di = y * (g - dot);
                        }
                    });

                let d_input = d_input_flat.into_shape(grad_shape).unwrap();
                input_clone.add_backward_grad(d_input);
            })),
            requires_grad: true,
            device: output_device,
        })))
    }
    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }
}

pub struct Gelu;
impl Gelu {
    pub fn new() -> Self {
        Gelu
    }
}

impl Default for Gelu {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for Gelu {
    fn forward(&self, input: Tensor) -> Tensor {
        unary_activation_with_backward(
            input,
            "gelu",
            cuda::UnaryOp::Gelu,
            gelu_f32,
            gelu_backward_f32,
        )
    }
    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }
}

#[cfg(test)]
#[path = "../../tests/unit/layers/activation/tests.rs"]
mod tests;
