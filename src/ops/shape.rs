use crate::autograd::{Tensor, TensorData};
use std::rc::Rc;
use std::cell::RefCell;
use ndarray::{Axis, Slice}; // 引入 Slice 用于 cat 的反向传播

pub fn reshape(input: &Tensor, shape: Vec<i32>) -> Tensor {
    let input_data = input.data();
    let new_shape: Vec<usize> = shape.iter().map(|&x| x as usize).collect();

    let contiguous_data = input_data.as_standard_layout().into_owned();

    let reshaped_data = contiguous_data.into_shape(new_shape)
        .expect("Reshape failed: Total element count mismatch")
        .into_dyn();

    let input_clone = input.clone();
    
    Tensor(Rc::new(RefCell::new(TensorData {
        data: reshaped_data,
        grad: None,
        parents: vec![input.clone()],
        backward_op: Some(Box::new(move |grad| {
            let old_shape = input_clone.data().shape().to_vec();
            // 反向传播也需要保证连续
            let grad_contiguous = grad.as_standard_layout().into_owned();
            let grad_reshaped = grad_contiguous.into_shape(old_shape)
                .expect("Backward Reshape failed")
                .into_dyn();
            input_clone.add_grad(grad_reshaped);
        })),
        requires_grad: true
    })))
}

pub fn permute(input: &Tensor, axes: Vec<usize>) -> Tensor {
    let input_data = input.data();
    let data = input_data.view().permuted_axes(axes.clone()).to_owned(); 
    
    let input_clone = input.clone();
    let mut rev_axes = vec![0; axes.len()];
    for (i, &ax) in axes.iter().enumerate() {
        rev_axes[ax] = i;
    }

    Tensor(Rc::new(RefCell::new(TensorData {
        data,
        grad: None,
        parents: vec![input.clone()],
        backward_op: Some(Box::new(move |grad| {
            let grad_restored = grad.view().permuted_axes(rev_axes.clone()).to_owned();
            input_clone.add_grad(grad_restored);
        })),
        requires_grad: true
    })))
}

pub fn cat(tensors: &[Tensor], axis: usize) -> Tensor {
    assert!(!tensors.is_empty(), "Concat expects at least one tensor");
    
    let arrays: Vec<_> = tensors.iter().map(|t| t.data()).collect();
    let views: Vec<_> = arrays.iter().map(|a| a.view()).collect();
    
    let axis_obj = Axis(axis);
    let result = ndarray::concatenate(axis_obj, &views)
        .expect("Concat failed: shape mismatch or invalid axis")
        .into_dyn();
    
    let lengths: Vec<usize> = tensors.iter().map(|t| t.data().shape()[axis]).collect();
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
        requires_grad: true
    })))
}

pub fn slice_last_dim(input: &Tensor, start: usize, end: usize) -> Tensor {
    let input_data = input.data();
    let last_dim = input_data.ndim() - 1;
    let axis = ndarray::Axis(last_dim);
    
    // Forward: Slice
    // slice_axis 返回 View，to_owned 变独立数据
    let sliced = input_data.slice_axis(axis, ndarray::Slice::from(start..end)).to_owned().into_dyn();
    
    let input_clone = input.clone();
    let full_shape = input_data.shape().to_vec();

    Tensor(Rc::new(RefCell::new(TensorData {
        data: sliced,
        grad: None,
        parents: vec![input.clone()],
        backward_op: Some(Box::new(move |grad| {
            // Backward: 创建一个全 0 的大梯度，把当前梯度填回对应的位置
            let mut full_grad = ndarray::Array::zeros(full_shape.clone());
            
            // 将小梯度写入大梯度的对应切片位置
            full_grad.slice_axis_mut(axis, ndarray::Slice::from(start..end))
                .assign(&grad);
            
            input_clone.add_grad(full_grad.into_dyn());
        })),
        requires_grad: true
    })))
}