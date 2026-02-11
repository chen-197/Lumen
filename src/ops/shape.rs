use crate::autograd::{is_no_grad, ArcArray, IxDyn, Tensor, TensorData};
use ndarray::{Axis, Slice};
use std::cell::RefCell;
use std::rc::Rc;

// 说明：Tensor 的 data 现在是 ArcArray（共享底层 + stride）。
// 因此 reshape/permute 在 stride 兼容时可以做到零拷贝（只改元数据）。

pub fn reshape(input: &Tensor, shape: Vec<i32>) -> Tensor {
    let new_shape: Vec<usize> = shape.iter().map(|&x| x as usize).collect();

    // clone 仅增加 refcount
    let data: ArcArray<f32, IxDyn> = input.data_arc();
    // into_shape 会 move data，所以用 clone（仅增 refcount）来尝试零拷贝 reshape
    let reshaped: ArcArray<f32, IxDyn> = match data.clone().into_shape(new_shape.clone()) {
        Ok(a) => a.into_dyn(),
        Err(_) => {
            // stride 不兼容时才退化为 copy
            data.to_owned()
                .as_standard_layout()
                .into_owned()
                .into_shape(new_shape)
                .expect("Reshape failed: Total element count mismatch")
                .into_dyn()
                .into_shared()
        }
    };

    if is_no_grad() || !input.requires_grad() {
        return Tensor::from_data_no_grad(reshaped);
    }

    let input_clone = input.clone();
    Tensor(Rc::new(RefCell::new(TensorData {
        data: reshaped,
        grad: None,
        parents: vec![input.clone()],
        backward_op: Some(Box::new(move |grad| {
            let old_shape = input_clone.data_ref().shape().to_vec();
            let grad_contig = grad.as_standard_layout().into_owned();
            let grad_reshaped = grad_contig
                .into_shape(old_shape)
                .expect("Backward Reshape failed")
                .into_dyn();
            input_clone.add_grad(grad_reshaped);
        })),
        requires_grad: true,
    })))
}

pub fn permute(input: &Tensor, axes: Vec<usize>) -> Tensor {
    let data: ArcArray<f32, IxDyn> = input.data_arc();
    let permuted: ArcArray<f32, IxDyn> = data.permuted_axes(axes.clone()).into_dyn();

    if is_no_grad() || !input.requires_grad() {
        return Tensor::from_data_no_grad(permuted);
    }

    let input_clone = input.clone();
    let mut rev_axes = vec![0; axes.len()];
    for (i, &ax) in axes.iter().enumerate() {
        rev_axes[ax] = i;
    }

    Tensor(Rc::new(RefCell::new(TensorData {
        data: permuted,
        grad: None,
        parents: vec![input.clone()],
        backward_op: Some(Box::new(move |grad| {
            let grad_restored = grad.view().permuted_axes(rev_axes.clone()).to_owned();
            input_clone.add_grad(grad_restored);
        })),
        requires_grad: true,
    })))
}

pub fn cat(tensors: &[Tensor], axis: usize) -> Tensor {
    assert!(!tensors.is_empty(), "Concat expects at least one tensor");

    // cat 必然 materialize
    let arrays: Vec<_> = tensors.iter().map(|t| t.data_ref().to_owned()).collect();
    let views: Vec<_> = arrays.iter().map(|a| a.view()).collect();

    let axis_obj = Axis(axis);
    let result = ndarray::concatenate(axis_obj, &views)
        .expect("Concat failed: shape mismatch or invalid axis")
        .into_dyn()
        .into_shared();

    if is_no_grad() || tensors.iter().all(|t| !t.requires_grad()) {
        return Tensor::from_data_no_grad(result);
    }

    let lengths: Vec<usize> = tensors.iter().map(|t| t.data_ref().shape()[axis]).collect();
    let tensors_clone: Vec<Tensor> = tensors.to_vec();

    Tensor(Rc::new(RefCell::new(TensorData {
        data: result,
        grad: None,
        parents: tensors.to_vec(),
        backward_op: Some(Box::new(move |grad| {
            let mut start_idx = 0;
            for (i, &len) in lengths.iter().enumerate() {
                let slice_info = Slice::from(start_idx..start_idx + len);
                let sub_grad = grad.slice_axis(axis_obj, slice_info).to_owned().into_dyn();
                tensors_clone[i].add_grad(sub_grad);
                start_idx += len;
            }
        })),
        requires_grad: true,
    })))
}

pub fn slice_last_dim(input: &Tensor, start: usize, end: usize) -> Tensor {
    let input_data = input.data_ref();
    let last_dim = input_data.ndim() - 1;
    let axis = ndarray::Axis(last_dim);

    // 这里先保持 materialize（便于 backward）
    let sliced = input_data
        .slice_axis(axis, ndarray::Slice::from(start..end))
        .to_owned()
        .into_dyn()
        .into_shared();

    if is_no_grad() || !input.requires_grad() {
        return Tensor::from_data_no_grad(sliced);
    }

    let input_clone = input.clone();
    let full_shape = input_data.shape().to_vec();

    Tensor(Rc::new(RefCell::new(TensorData {
        data: sliced,
        grad: None,
        parents: vec![input.clone()],
        backward_op: Some(Box::new(move |grad| {
            let mut full_grad = ndarray::Array::zeros(full_shape.clone());
            full_grad
                .slice_axis_mut(axis, ndarray::Slice::from(start..end))
                .assign(&grad);
            input_clone.add_grad(full_grad.into_dyn());
        })),
        requires_grad: true,
    })))
}
